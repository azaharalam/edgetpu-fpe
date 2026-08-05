#!/usr/bin/env python3
"""
fault_inject.py -- size-preserving fault injection into TFLite models.

Every model in the Coral zoo takes uint8/int8 input, so NaN/Inf cannot be
introduced through the input tensor, and the Python interpreter cannot write
intermediate tensors mid-inference. Faults must therefore be injected into
the model file itself.

This tool patches constant buffers and quantization parameters in place,
never changing any field's byte width, so the flatbuffer layout is untouched
and no re-serialization is needed. That makes injection fast enough to run
thousands of trials.

FAULT CLASSES

  invalid       write NaN into a float constant, or set a quantization scale
                to NaN so DEQUANTIZE propagates it
  divzero       zero a float constant used as a divisor (MoveNet's DIV) so
                the kernel divides by zero
  overflow      multiply a quantization scale by a large factor so
                DEQUANTIZE produces values beyond float32 range
  underflow     multiply a quantization scale by a tiny factor so DEQUANTIZE
                produces subnormals or flushes to zero
  saturate      drive int8/uint8 weights to a clamp bound -- the fixed-point
                analog, and the only fault class reachable inside the TPU
                partition

ACTIVATION CHECK

An injected fault that never influences the output is not a detection
opportunity and must not count in the denominator. `--verify` re-runs
inference and compares against the clean baseline; a fault is ACTIVATED only
if the output actually changes.

Usage
    python fault_inject.py list  --model M.tflite
    python fault_inject.py list  --model M.tflite --kind scale
    python fault_inject.py inject --model M.tflite --out F.tflite \\
        --target 19 --fault divzero
    python fault_inject.py inject --model M.tflite --out F.tflite \\
        --target 19 --fault overflow --magnitude 1e30 \\
        --verify --input img.npy
"""

import argparse
import os
import shutil
import struct
import sys

import numpy as np

try:
    import flatbuffers  # noqa: F401
    import tflite
except ImportError:
    sys.exit("needs the schema bindings:  pip install tflite flatbuffers")

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate

EDGETPU_LIB = {"Linux": "libedgetpu.so.1",
               "Darwin": "libedgetpu.1.dylib",
               "Windows": "edgetpu.dll"}[__import__("platform").system()]


def make_interpreter(model_path, device="cpu"):
    """
    A compiled model contains edgetpu-custom-op, which only the delegate can
    resolve; building a plain Interpreter over it fails at prepare time. The
    device must therefore follow the model through every inference path,
    including the isolated child processes.
    """
    delegates = [load_delegate(EDGETPU_LIB)] if device == "tpu" else []
    interp = Interpreter(model_path=model_path,
                         experimental_delegates=delegates)
    interp.allocate_tensors()
    return interp


# flatbuffers vtable slots
BUFFER_DATA_SLOT = 4          # Buffer.data
QUANT_SCALE_SLOT = 8          # QuantizationParameters.scale
QUANT_ZP_SLOT = 10            # QuantizationParameters.zero_point

QUANT_BOUNDS = {"int8": (-128, 127), "uint8": (0, 255),
                "int16": (-32768, 32767)}

# --- IEEE flag polling (mirrors fpe_probe.py) ------------------------------
import ctypes as _ct
import platform as _pl

_M = _pl.machine()
if _M in ("aarch64", "armv7l", "armv8l"):
    _FE = {"invalid": 0x01, "divbyzero": 0x02, "overflow": 0x04,
           "underflow": 0x08, "inexact": 0x10}
    _FE_ALL = 0x1F
else:
    _FE = {"invalid": 0x01, "denormal": 0x02, "divbyzero": 0x04,
           "overflow": 0x08, "underflow": 0x10, "inexact": 0x20}
    _FE_ALL = 0x3F

_LIBM = None
for _c in ("libm.so.6", "libm.so", "libc.so.6"):
    try:
        _LIBM = _ct.CDLL(_c)
        _LIBM.feclearexcept, _LIBM.fetestexcept
        break
    except (OSError, AttributeError):
        _LIBM = None


def _clear_flags():
    if _LIBM:
        _LIBM.feclearexcept(_ct.c_int(_FE_ALL))


def _read_flags():
    if not _LIBM:
        return {}
    r = _LIBM.fetestexcept(_ct.c_int(_FE_ALL))
    return {k: bool(r & v) for k, v in _FE.items()}


# Exceptions that indicate a real numeric event. `inexact` fires on almost
# every float operation and carries no diagnostic value.
MEANINGFUL = ("invalid", "divbyzero", "overflow", "underflow")

DTYPE_OF = {    tflite.TensorType.FLOAT32: np.float32,
    tflite.TensorType.FLOAT16: np.float16,
    tflite.TensorType.INT32: np.int32,
    tflite.TensorType.UINT8: np.uint8,
    tflite.TensorType.INT8: np.int8,
    tflite.TensorType.INT16: np.int16,
    tflite.TensorType.INT64: np.int64,
}


def load(path):
    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    model = tflite.Model.GetRootAsModel(bytes(data), 0)
    return model, data


def vector_span(table, slot):
    """Absolute (offset, length) of a flatbuffer vector, or None."""
    o = table._tab.Offset(slot)
    if o == 0:
        return None
    return table._tab.Vector(o), table._tab.VectorLen(o)


def enumerate_targets(model, subgraph=0):
    """All patchable sites: constant buffers and quantization scales."""
    sg = model.Subgraphs(subgraph)
    targets = []

    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        name = t.Name().decode("utf-8", "replace") if t.Name() else f"t{i}"
        np_dtype = DTYPE_OF.get(t.Type())
        buf_idx = t.Buffer()
        buf = model.Buffers(buf_idx)
        span = vector_span(buf, BUFFER_DATA_SLOT)

        entry = {
            "tensor": i,
            "name": name,
            "dtype": None if np_dtype is None else np.dtype(np_dtype).name,
            "shape": [int(x) for x in t.ShapeAsNumpy()] if t.ShapeLength() else [],
            "buffer": buf_idx,
            "const_offset": None,
            "const_bytes": 0,
            "scale_offset": None,
            "scale_count": 0,
            "zp_offset": None,
            "zp_count": 0,
        }

        if span and span[1] > 0:
            entry["const_offset"], entry["const_bytes"] = span

        q = t.Quantization()
        if q is not None:
            s = vector_span(q, QUANT_SCALE_SLOT)
            if s:
                entry["scale_offset"], entry["scale_count"] = s[0], s[1]
            z = vector_span(q, QUANT_ZP_SLOT)
            if z:
                entry["zp_offset"], entry["zp_count"] = z[0], z[1]

        targets.append(entry)

    return targets


def read_scales(data, entry):
    off, n = entry["scale_offset"], entry["scale_count"]
    return list(struct.unpack_from(f"<{n}f", data, off))


def write_scales(data, entry, values):
    off = entry["scale_offset"]
    struct.pack_into(f"<{len(values)}f", data, off, *values)


def read_zero_points(data, entry):
    off, n = entry["zp_offset"], entry["zp_count"]
    return list(struct.unpack_from(f"<{n}q", data, off))


def write_zero_points(data, entry, values):
    struct.pack_into(f"<{len(values)}q", data, entry["zp_offset"], *values)


def patch(data, entry, fault, magnitude, rng, element=None):
    """Apply one fault in place. Returns a record of what changed."""
    rec = {"fault": fault, "tensor": entry["tensor"], "name": entry["name"],
           "site": None, "before": None, "after": None}

    # Zero-point corruption. Dequantization computes (q - zero_point) * scale,
    # so moving the zero point into the tensor's observed value range makes
    # the dequantized result exactly 0.0 at those elements. Unlike scale, the
    # zero point feeds no multiplier arithmetic, so TFLite's QuantizeMultiplier
    # range checks -- which reject extreme scales -- do not apply here.
    if fault == "zp_set":
        if entry["zp_offset"] is None:
            raise ValueError(f"tensor {entry['tensor']} has no zero point")
        zps = read_zero_points(data, entry)
        j = 0 if element is None else element
        rec["site"] = f"zero_point[{j}]"
        rec["before"] = zps[j]
        if magnitude is None:
            raise ValueError("zp_set needs --magnitude (the new zero point)")
        zps[j] = int(magnitude)
        rec["after"] = zps[j]
        write_zero_points(data, entry, zps)
        return rec

    scale_faults = ("overflow", "underflow", "scale_nan",
                    "scale_zero", "scale_negative")
    if fault in scale_faults:
        if entry["scale_offset"] is None:
            raise ValueError(f"tensor {entry['tensor']} has no quantization scale")
        scales = read_scales(data, entry)
        j = rng.integers(len(scales)) if element is None else element
        rec["site"] = f"scale[{j}]"
        rec["before"] = scales[j]
        if fault == "overflow":
            scales[j] = float(magnitude if magnitude else 1e30)
        elif fault == "underflow":
            scales[j] = float(magnitude if magnitude else 1e-38)
        elif fault == "scale_zero":
            scales[j] = 0.0
        elif fault == "scale_negative":
            scales[j] = -abs(scales[j]) if scales[j] else -1.0
        else:
            scales[j] = float("nan")
        rec["after"] = scales[j]
        write_scales(data, entry, scales)
        return rec

    # constant-buffer faults
    if entry["const_offset"] is None:
        raise ValueError(f"tensor {entry['tensor']} has no constant buffer")

    dt = np.dtype(entry["dtype"])
    off, nbytes = entry["const_offset"], entry["const_bytes"]
    n = nbytes // dt.itemsize
    arr = np.frombuffer(bytes(data[off:off + nbytes]), dtype=dt).copy()
    j = int(rng.integers(n)) if element is None else element
    rec["site"] = f"const[{j}]/{n}"
    rec["before"] = float(arr[j])

    if fault == "invalid":
        if dt.kind != "f":
            raise ValueError("NaN needs a float constant; use 'saturate' for ints")
        arr[j] = np.float32("nan")
    elif fault == "divzero":
        if dt.kind != "f":
            raise ValueError("divzero needs a float constant")
        arr[j] = 0.0
    elif fault == "inf":
        if dt.kind != "f":
            raise ValueError("inf needs a float constant")
        arr[j] = np.float32(np.inf)
    elif fault == "saturate":
        if dt.kind not in "iu":
            raise ValueError("saturate applies to integer constants")
        info = np.iinfo(dt)
        arr[j] = info.max if rng.random() < 0.5 else info.min
    else:
        raise ValueError(f"unknown fault {fault}")

    rec["after"] = float(arr[j])
    data[off:off + nbytes] = arr.tobytes()
    return rec


def infer(model_path, input_path, seed=0, device="cpu"):
    interp = make_interpreter(model_path, device)
    rng = np.random.default_rng(seed)
    for d in interp.get_input_details():
        dt = np.dtype(d["dtype"])
        if input_path:
            arr = np.load(input_path).astype(dt).reshape(d["shape"])
        elif dt in (np.uint8, np.int8):
            info = np.iinfo(dt)
            arr = rng.integers(info.min, info.max + 1, size=d["shape"], dtype=dt)
        else:
            arr = rng.random(size=d["shape"]).astype(dt)
        interp.set_tensor(d["index"], arr)
    interp.invoke()
    return [interp.get_tensor(d["index"]).copy()
            for d in interp.get_output_details()]


def infer_isolated(model_path, input_path, seed, outdir, device="cpu"):
    """
    Run inference in a child process. Malformed quantization metadata can
    make TFLite's native code abort (SIGABRT), which no Python try/except can
    catch; without isolation a single bad injection kills an entire campaign.
    Returns (status, outputs_or_detail).
    """
    import subprocess, tempfile, glob as _glob
    tmp = tempfile.mkdtemp(dir=outdir)
    cmd = [sys.executable, os.path.abspath(__file__), "_infer",
           "--model", model_path, "--seed", str(seed), "--outdir", tmp,
           "--device", device]
    if input_path:
        cmd += ["--input", input_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if proc.returncode < 0:
        return "CRASHED", f"child killed by signal {-proc.returncode}"
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return "REJECTED", (tail[-1][:150] if tail else "non-zero exit")

    outs = []
    for f in sorted(_glob.glob(os.path.join(tmp, "out_*.npy"))):
        outs.append(np.load(f))
    flags = {}
    fp = os.path.join(tmp, "flags.json")
    if os.path.exists(fp):
        import json as _json
        with open(fp) as fh:
            flags = _json.load(fh)
    return "OK", (outs, flags)


def predict_exception(model_path, tensor_index, fault, magnitude):
    """
    Analytic oracle. For DEQUANTIZE-reachable scale corruption the resulting
    float magnitude is bounded by max|q - zero_point| * scale, so the IEEE
    exception class is predictable independently of the detector -- which is
    what makes the detection rate a real measurement rather than a tautology.
    """
    model, _ = load(model_path)
    for e in enumerate_targets(model):
        if e["tensor"] != tensor_index:
            continue
        dt = e["dtype"]
        if dt not in QUANT_BOUNDS:
            return None
        lo, hi = QUANT_BOUNDS[dt]
        span = float(max(abs(lo), abs(hi)))
        f32_max = float(np.finfo(np.float32).max)
        f32_tiny = float(np.finfo(np.float32).tiny)
        if fault == "overflow":
            s = magnitude if magnitude else 1e30
            return "overflow" if span * s > f32_max else "none"
        if fault == "underflow":
            s = magnitude if magnitude else 1e-38
            return "underflow" if span * s < f32_tiny else "none"
        if fault == "scale_nan":
            return "invalid"
        if fault == "scale_zero":
            return "divbyzero_or_none"
        if fault == "scale_negative":
            return "invalid_if_sqrt"
        if fault == "zp_set":
            return "divbyzero/invalid if the tensor reaches this value"
    return None


def outcome(model_clean, model_faulty, input_path, seed, device="cpu"):
    """
    Four-way result. TFLite kernels validate quantization parameters to
    differing degrees (LOGISTIC hard-checks its output scale at 1/256;
    CONCATENATION requires input and output scales to match; DEQUANTIZE
    checks nothing), so a fault may be REJECTED at prepare time or CRASH the
    runtime. Neither is a missed detection, and both must be reported apart
    from genuine misses.
    """
    outdir = os.path.dirname(os.path.abspath(model_faulty)) or "."
    st_c, res_c = infer_isolated(model_clean, input_path, seed, outdir, device)
    if st_c != "OK":
        return "BASELINE_FAILED", str(res_c)[:150]
    st_f, res_f = infer_isolated(model_faulty, input_path, seed, outdir, device)
    if st_f != "OK":
        return st_f, str(res_f)[:150]

    clean, f_clean = res_c
    faulty, f_fault = res_f

    # Two independent activation criteria. A fault can raise a genuine IEEE
    # exception and still leave the output untouched, because post-processing
    # (NMS filtering, score thresholding, clamping) absorbs it. Judging
    # activation by output alone discards real detections.
    out_changed = activated(clean, faulty)
    new_exc = [k for k in MEANINGFUL
               if f_fault.get(k) and not f_clean.get(k)]

    if new_exc and out_changed:
        return "ACTIVATED_BOTH", f"exceptions: {', '.join(new_exc)}"
    if new_exc:
        return "ACTIVATED_EXCEPTION", (
            f"exceptions: {', '.join(new_exc)}; output unchanged "
            f"(masked downstream)")
    if out_changed:
        return "ACTIVATED_OUTPUT", "output changed, no IEEE exception raised"
    return "NOT_ACTIVATED", None


def activated(clean, faulty):
    """A fault counts only if it actually reached the output."""
    if len(clean) != len(faulty):
        return True
    for a, b in zip(clean, faulty):
        if a.shape != b.shape:
            return True
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            af, bf = a.astype(np.float64), b.astype(np.float64)
            if not np.array_equal(np.isnan(af), np.isnan(bf)):
                return True
            m = ~(np.isnan(af) | np.isnan(bf))
            if m.any() and not np.array_equal(af[m], bf[m]):
                return True
        elif not np.array_equal(a, b):
            return True
    return False


def cmd_infer_worker(args):
    """Child-process entry point for isolated inference, with flag polling."""
    import json as _json
    interp = make_interpreter(args.model, getattr(args, "device", "cpu"))
    rng = np.random.default_rng(args.seed)
    for d in interp.get_input_details():
        dt = np.dtype(d["dtype"])
        if args.input:
            a = np.load(args.input).astype(dt).reshape(d["shape"])
        elif dt in (np.uint8, np.int8):
            info = np.iinfo(dt)
            a = rng.integers(info.min, info.max + 1, size=d["shape"], dtype=dt)
        else:
            a = rng.random(size=d["shape"]).astype(dt)
        interp.set_tensor(d["index"], a)

    _clear_flags()
    interp.invoke()
    flags = _read_flags()

    for i, d in enumerate(interp.get_output_details()):
        np.save(os.path.join(args.outdir, f"out_{i:03d}.npy"),
                interp.get_tensor(d["index"]))
    with open(os.path.join(args.outdir, "flags.json"), "w") as fh:
        _json.dump(flags, fh)


def cmd_list(args):
    model, data = load(args.model)
    targets = enumerate_targets(model)

    print(f"{'idx':>4} {'dtype':>8} {'const_B':>9} {'scales':>7}  name")
    print("-" * 92)
    shown = 0
    for e in targets:
        is_const = e["const_offset"] is not None
        has_scale = e["scale_offset"] is not None
        if args.kind == "const" and not is_const:
            continue
        if args.kind == "scale" and (not has_scale or e["scale_count"] == 0):
            continue
        if args.kind == "float" and (e["dtype"] not in ("float32", "float16")
                                     or not is_const):
            continue
        print(f"{e['tensor']:>4} {str(e['dtype']):>8} "
              f"{e['const_bytes'] if is_const else 0:>9} "
              f"{e['scale_count']:>7}  {e['name'][:58]}")
        shown += 1
    print(f"\n{shown} target(s). "
          f"float constants -> invalid/divzero/inf; "
          f"int constants -> saturate; scales -> overflow/underflow/scale_nan")


def cmd_inject(args):
    model, data = load(args.model)
    targets = {e["tensor"]: e for e in enumerate_targets(model)}
    if args.target not in targets:
        sys.exit(f"tensor {args.target} not found")

    rng = np.random.default_rng(args.seed)
    entry = targets[args.target]

    try:
        rec = patch(data, entry, args.fault, args.magnitude, rng, args.element)
    except ValueError as exc:
        sys.exit(f"injection failed: {exc}")

    with open(args.out, "wb") as fh:
        fh.write(bytes(data))

    print(f"injected {rec['fault']} into tensor {rec['tensor']} "
          f"({rec['name'][:50]}) at {rec['site']}")
    print(f"  {rec['before']!r} -> {rec['after']!r}")
    print(f"  wrote {args.out} "
          f"({os.path.getsize(args.out)} B, "
          f"{os.path.getsize(args.model)} B original)")

    if args.verify:
        pred = predict_exception(args.model, args.target, args.fault,
                                 args.magnitude)
        if pred is not None:
            print(f"  ORACLE:  predicted exception = {pred}")
        status, detail = outcome(args.model, args.out, args.input, args.seed,
                                 args.device)
        note = {"ACTIVATED_BOTH": "exception raised and output changed",
                "ACTIVATED_EXCEPTION": "exception raised but fully masked "
                                       "downstream -- output inspection "
                                       "cannot see this",
                "ACTIVATED_OUTPUT": "numerically wrong but no IEEE exception",
                "ACTIVATED": "counts as a detection opportunity",
                "NOT_ACTIVATED": "excluded from denominator",
                "REJECTED": "runtime refused the model at prepare time",
                "CRASHED": "runtime aborted; robustness finding, not a miss",
                "BASELINE_FAILED": "clean model failed; check input"}[status]
        print(f"  OUTCOME: {status}  ({note})")
        if detail:
            print(f"    {detail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="enumerate injectable sites")
    p.add_argument("--model", required=True)
    p.add_argument("--kind", choices=["all", "const", "scale", "float"],
                   default="all")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("inject", help="apply one fault")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", type=int, required=True, help="tensor index")
    p.add_argument("--fault", required=True,
                   choices=["invalid", "divzero", "inf", "saturate",
                            "overflow", "underflow", "scale_nan",
                            "scale_zero", "scale_negative", "zp_set"])
    p.add_argument("--magnitude", type=float, default=None)
    p.add_argument("--element", type=int, default=None,
                   help="specific element; random if omitted")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", action="store_true",
                   help="check the fault actually reaches the output")
    p.add_argument("--input", help=".npy input for verification")
    p.add_argument("--device", choices=["cpu", "tpu"], default="cpu",
                   help="use tpu for compiled *_edgetpu.tflite models")
    p.set_defaults(func=cmd_inject)

    p = sub.add_parser("_infer", help=argparse.SUPPRESS)
    p.add_argument("--model", required=True)
    p.add_argument("--input")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", choices=["cpu", "tpu"], default="cpu")
    p.set_defaults(func=cmd_infer_worker)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()