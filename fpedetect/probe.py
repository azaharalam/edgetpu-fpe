#!/usr/bin/env python3
"""
fpe_probe.py -- runtime FP-exception instrumentation for Edge TPU inference.
v2: corrected saturation semantics.

Mechanisms (all non-intrusive: no compiler, libedgetpu, or model changes):

  M1  IEEE flag polling      feclearexcept/fetestexcept around Invoke().
  M2  Tensor scanning        NaN / Inf / subnormal, per tensor and element.
  M3  Saturation counting    quantized activations pinned at a clamp bound
                             that is NOT the tensor's zero point.

SATURATION SEMANTICS (v2 fix). A quantized value sitting at a clamp bound is
only evidence of range overflow if that bound is not the tensor's own zero
point. For a uint8 ReLU output with zero_point=0, the value 0 means exactly
0.0 -- the activation was clipped by ReLU, which is the layer's intended
behaviour, not a numeric fault. Counting it as saturation inflates the rate
to ~90% on healthy models. v2 therefore:
    - reports upper- and lower-clamp hits separately,
    - excludes a clamp bound that coincides with the zero point,
    - scans only activations, never weights/biases/constants.

LIMITATIONS (for the poster):
  * Python cannot intercept kernel invocations, so M1 is inference-granular;
    M2 supplies localization. Per-op M1 attribution needs the C++ wrapper.
  * TFLite_Detection_PostProcess is one opaque kernel (exp, division, NMS);
    exceptions localize to the kernel, not within it.
  * Under delegation the TPU partition is a single custom op; no intermediate
    activation inside it is observable.
"""

import argparse
import ctypes
import json
import os
import platform
import sys
import time

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


# ---------------------------------------------------------------------------
# M1: IEEE 754 status flags
# ---------------------------------------------------------------------------

_MACHINE = platform.machine()

if _MACHINE in ("x86_64", "i386", "i686"):
    FE = {"invalid": 0x01, "denormal": 0x02, "divbyzero": 0x04,
          "overflow": 0x08, "underflow": 0x10, "inexact": 0x20}
    FE_ALL = 0x3F
elif _MACHINE in ("aarch64", "armv7l", "armv8l"):
    FE = {"invalid": 0x01, "divbyzero": 0x02, "overflow": 0x04,
          "underflow": 0x08, "inexact": 0x10}
    FE_ALL = 0x1F
else:
    FE = {"invalid": 0x01, "divbyzero": 0x04, "overflow": 0x08,
          "underflow": 0x10, "inexact": 0x20}
    FE_ALL = 0x3F

_libm = None
for _cand in ("libm.so.6", "libm.so", "libc.so.6"):
    try:
        _libm = ctypes.CDLL(_cand)
        _libm.feclearexcept, _libm.fetestexcept
        break
    except (OSError, AttributeError):
        _libm = None
if _libm is None:
    sys.exit("could not load libm; flag polling unavailable")


def clear_flags():
    _libm.feclearexcept(ctypes.c_int(FE_ALL))


def read_flags():
    raised = _libm.fetestexcept(ctypes.c_int(FE_ALL))
    return {name: bool(raised & bit) for name, bit in FE.items()}


# ---------------------------------------------------------------------------
# M2 / M3: tensor analysis
# ---------------------------------------------------------------------------

FLOAT_TYPES = ("float32", "float16", "float64")
F32_TINY = np.finfo(np.float32).tiny

QUANT_BOUNDS = {"int8": (-128, 127), "uint8": (0, 255),
                "int16": (-32768, 32767)}


def scan_float_tensor(arr):
    nan = int(np.isnan(arr).sum())
    inf = int(np.isinf(arr).sum())
    finite = np.isfinite(arr)
    nz = arr[finite & (arr != 0)]
    sub = int((np.abs(nz) < F32_TINY).sum()) if nz.size else 0
    return {
        "elements": int(arr.size),
        "nan": nan,
        "inf": inf,
        "posinf": int((arr == np.inf).sum()),
        "neginf": int((arr == -np.inf).sum()),
        "subnormal": sub,
        "anomalous": bool(nan or inf or sub),
    }


def scan_quant_tensor(arr, dtype_name, zero_point):
    """
    Clamp-bound saturation, excluding bounds that coincide with the zero
    point. Hitting the zero point is a representable, intended value (e.g.
    a ReLU output of exactly 0.0); hitting the far bound means the value
    exceeded the representable range -- the fixed-point analog of overflow.
    """
    if dtype_name not in QUANT_BOUNDS:
        return None
    lo, hi = QUANT_BOUNDS[dtype_name]
    n = int(arr.size)
    if n == 0:
        return None

    at_lo = int((arr == lo).sum())
    at_hi = int((arr == hi).sum())

    lo_is_zp = (zero_point is not None and int(zero_point) == lo)
    hi_is_zp = (zero_point is not None and int(zero_point) == hi)

    sat = (0 if lo_is_zp else at_lo) + (0 if hi_is_zp else at_hi)

    return {
        "elements": n,
        "zero_point": None if zero_point is None else int(zero_point),
        "at_lower_bound": at_lo,
        "at_upper_bound": at_hi,
        "lower_bound_is_zero_point": lo_is_zp,
        "saturated": sat,
        "saturation_rate": sat / n,
        "upper_saturation_rate": 0.0 if hi_is_zp else at_hi / n,
    }


def quant_zero_point(detail):
    qp = detail.get("quantization_parameters", {})
    zps = qp.get("zero_points", [])
    if len(zps):
        return zps[0]
    q = detail.get("quantization", None)
    if isinstance(q, (tuple, list)) and len(q) == 2:
        return q[1]
    return None


def activation_indices(interp):
    """
    Indices of tensors produced by an op (true activations). Weights, biases
    and other constants are never op outputs and must not be scanned -- they
    are static parameters, not runtime values.
    """
    idxs = set()
    try:
        for op in interp._get_ops_details():
            for o in op.get("outputs", []):
                idxs.add(int(o))
    except Exception:
        return None  # caller falls back to scanning everything
    return idxs


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

EDGETPU_LIB = {"Linux": "libedgetpu.so.1",
               "Darwin": "libedgetpu.1.dylib",
               "Windows": "edgetpu.dll"}[platform.system()]


def build_interpreter(model_path, device, preserve):
    delegates = [load_delegate(EDGETPU_LIB)] if device == "tpu" else []
    meta = {"preserve_all_tensors": False, "preserve_error": None}

    if preserve:
        try:
            interp = Interpreter(model_path=model_path,
                                 experimental_delegates=delegates,
                                 experimental_preserve_all_tensors=True)
            interp.allocate_tensors()
            meta["preserve_all_tensors"] = True
            return interp, meta
        except Exception as exc:
            meta["preserve_error"] = str(exc)[:200]
            if device == "tpu":
                delegates = [load_delegate(EDGETPU_LIB)]

    interp = Interpreter(model_path=model_path,
                         experimental_delegates=delegates)
    interp.allocate_tensors()
    return interp, meta


def make_input(interp, input_path, seed):
    """Real input if supplied, else random. Random input is NOT valid for
    saturation statistics -- it lacks natural spatial structure and drives
    activations toward the clamp bounds."""
    rng = np.random.default_rng(seed)
    feeds = []
    for d in interp.get_input_details():
        dt = np.dtype(d["dtype"])
        if input_path:
            arr = np.load(input_path).astype(dt).reshape(d["shape"])
        elif dt in (np.uint8, np.int8):
            info = np.iinfo(dt)
            arr = rng.integers(info.min, info.max + 1, size=d["shape"], dtype=dt)
        else:
            arr = rng.random(size=d["shape"]).astype(dt)
        feeds.append((d["index"], arr))
    return feeds


# ---------------------------------------------------------------------------
# One instrumented inference
# ---------------------------------------------------------------------------

def run_once(interp, feeds, output_indices, act_indices, sat_threshold):
    for idx, arr in feeds:
        interp.set_tensor(idx, arr)

    # Clear immediately before, read immediately after. The FPU status word is
    # process-global and sticky; any numpy call in between contaminates it.
    clear_flags()
    t0 = time.perf_counter()
    interp.invoke()
    elapsed = time.perf_counter() - t0
    flags = read_flags()

    float_report, sat_report = {}, {}
    unreachable = 0

    for t in interp.get_tensor_details():
        idx, name = t["index"], t["name"]
        dt = np.dtype(t["dtype"]).name
        if act_indices is not None and idx not in act_indices:
            continue  # constant / weight / bias
        try:
            arr = interp.get_tensor(idx)
        except Exception:
            unreachable += 1
            continue

        if dt in FLOAT_TYPES:
            r = scan_float_tensor(arr)
            if r["anomalous"]:
                r["is_model_output"] = idx in output_indices
                float_report[f"{idx}:{name[:60]}"] = r
        else:
            r = scan_quant_tensor(arr, dt, quant_zero_point(t))
            if r and r["saturation_rate"] > sat_threshold:
                sat_report[f"{idx}:{name[:60]}"] = r

    output_anomaly = any(v.get("is_model_output") for v in float_report.values())

    return {
        "latency_s": elapsed,
        "flags": flags,
        "flags_raised": [k for k, v in flags.items() if v],
        "float_anomalies": float_report,
        "n_float_anomalous_tensors": len(float_report),
        "saturated_tensors": sat_report,
        "n_saturated_tensors": len(sat_report),
        "unreachable_tensors": unreachable,
        "detected_by_full_scan": bool(float_report),
        "detected_by_output_only": output_anomaly,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", choices=["cpu", "tpu"], default="cpu")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--input", help=".npy input; random if omitted")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sat-threshold", type=float, default=0.01,
                    help="report tensors above this saturation rate")
    ap.add_argument("--scan-constants", action="store_true",
                    help="also scan weights/biases (not recommended)")
    ap.add_argument("--no-preserve", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()

    interp, meta = build_interpreter(args.model, args.device,
                                     preserve=not args.no_preserve)
    output_indices = {d["index"] for d in interp.get_output_details()}
    act = None if args.scan_constants else activation_indices(interp)
    n_all = len(interp.get_tensor_details())

    runs = []
    for i in range(args.runs):
        feeds = make_input(interp, args.input, args.seed + i)
        runs.append(run_once(interp, feeds, output_indices, act,
                             args.sat_threshold))

    lat = [r["latency_s"] for r in runs]
    flag_totals = {k: sum(r["flags"][k] for r in runs) for k in FE}
    full = sum(r["detected_by_full_scan"] for r in runs)
    out_only = sum(r["detected_by_output_only"] for r in runs)

    report = {
        "model": os.path.basename(args.model),
        "device": args.device,
        "arch": _MACHINE,
        "runs": args.runs,
        "input": args.input or "random (NOT valid for saturation stats)",
        "preserve_all_tensors": meta["preserve_all_tensors"],
        "preserve_error": meta["preserve_error"],
        "tensors_total": n_all,
        "tensors_scanned": n_all if act is None else len(act),
        "latency_median_ms": float(np.median(lat) * 1e3),
        "latency_iqr_ms": float((np.percentile(lat, 75) -
                                 np.percentile(lat, 25)) * 1e3),
        "flag_counts": flag_totals,
        "runs_detected_full_scan": full,
        "runs_detected_output_only": out_only,
        "masking_undercount": full - out_only,
        "per_run": runs,
    }

    print(f"model            {report['model']}")
    print(f"device / arch    {args.device} / {_MACHINE}")
    print(f"input            {report['input']}")
    print(f"preserve_all     {meta['preserve_all_tensors']}"
          + (f"  ({meta['preserve_error']})" if meta["preserve_error"] else ""))
    print(f"tensors          {report['tensors_scanned']} scanned "
          f"of {n_all} ({runs[0]['unreachable_tensors']} unreachable)")
    print(f"latency          {report['latency_median_ms']:.3f} ms "
          f"(IQR {report['latency_iqr_ms']:.3f})")
    print(f"flags            {flag_totals}")
    print(f"detected f/o     {full}/{out_only}  "
          f"(masking undercount {full - out_only})")

    last = runs[-1]
    if last["float_anomalies"]:
        print("\nfloat anomalies (last run):")
        for k, v in last["float_anomalies"].items():
            print(f"  {k}\n      nan={v['nan']} inf={v['inf']} "
                  f"subnormal={v['subnormal']} of {v['elements']}")
    else:
        print("\nfloat anomalies: none")

    if last["saturated_tensors"]:
        top = sorted(last["saturated_tensors"].items(),
                     key=lambda kv: -kv[1]["saturation_rate"])[:8]
        print(f"\ntop saturated activations (>{args.sat_threshold:.1%}, "
              f"zero-point excluded):")
        for k, v in top:
            print(f"  {k}\n      {v['saturation_rate']*100:.2f}% "
                  f"({v['saturated']}/{v['elements']}) "
                  f"zp={v['zero_point']} hi={v['at_upper_bound']} "
                  f"lo={v['at_lower_bound']}")
    else:
        print(f"\nsaturated activations: none above {args.sat_threshold:.1%}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
