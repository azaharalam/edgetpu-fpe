#!/usr/bin/env python3
"""
run_campaign.py -- fault-injection campaign for FPE detection on Edge TPU.

Sweeps (fault class x injection site x magnitude x trial), classifies every
trial, and reports detection rate per exception class with Wilson confidence
intervals.

WHY THE CLASSIFICATION MATTERS

A fault-injection study is only as good as its denominator. Four things can
happen that are NOT missed detections, and folding any of them into the
denominator understates the tool:

  REJECTED   TFLite refuses the model at prepare time. Kernels validate
             quantization metadata where their semantics demand it --
             LOGISTIC pins its output scale at 1/256, CONCATENATION requires
             input and output scales to match. The runtime prevented the
             fault; nothing occurred to detect.
  CRASHED    Native code aborts (SIGABRT). Extreme scales blow the shift
             bounds in QuantizeMultiplier. A robustness finding, not a miss.
  NOT_ACT    The fault never influenced anything measurable.
  ACT_OUTPUT Numerically wrong output, but no IEEE exception was raised --
             so there is genuinely nothing for an exception detector to see.

Only trials that actually raise an IEEE exception are detection
opportunities. Among those, a fault whose exception is absorbed before the
output (NMS filtering, score thresholding, clamping) is still a real
detection -- and is precisely the case output inspection cannot handle.

Every trial runs in a child process, because a single malformed model can
abort the interpreter and would otherwise kill the whole campaign.

Usage
    python run_campaign.py --model M.tflite --input img.npy --trials 30
    python run_campaign.py --model M.tflite --input img.npy \\
        --trials 30 --json campaign.json --csv campaign.csv
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fault_inject as FI  # noqa: E402

MEANINGFUL = ("invalid", "divbyzero", "overflow", "underflow")


# ---------------------------------------------------------------------------
# child-process detector
# ---------------------------------------------------------------------------

def detect_worker(model_path, input_path, seed, outfile, device="cpu"):
    """Run one instrumented inference; write flags + scan summary as JSON."""
    import ctypes, platform
    from tflite_runtime.interpreter import Interpreter, load_delegate

    m = platform.machine()
    if m in ("aarch64", "armv7l", "armv8l"):
        fe = {"invalid": 0x01, "divbyzero": 0x02, "overflow": 0x04,
              "underflow": 0x08, "inexact": 0x10}
        fe_all = 0x1F
    else:
        fe = {"invalid": 0x01, "divbyzero": 0x04, "overflow": 0x08,
              "underflow": 0x10, "inexact": 0x20}
        fe_all = 0x3F
    libm = ctypes.CDLL("libm.so.6")

    # A compiled model needs the delegate to resolve edgetpu-custom-op.
    # preserve_all_tensors works under delegation, but the fused partition
    # collapses to one op, so far fewer tensors remain observable.
    delegates = [load_delegate("libedgetpu.so.1")] if device == "tpu" else []
    interp = Interpreter(model_path=model_path,
                         experimental_delegates=delegates,
                         experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    rng = np.random.default_rng(seed)
    for d in interp.get_input_details():
        dt = np.dtype(d["dtype"])
        if input_path:
            a = np.load(input_path).astype(dt).reshape(d["shape"])
        elif dt in (np.uint8, np.int8):
            info = np.iinfo(dt)
            a = rng.integers(info.min, info.max + 1, size=d["shape"], dtype=dt)
        else:
            a = rng.random(size=d["shape"]).astype(dt)
        interp.set_tensor(d["index"], a)

    libm.feclearexcept(ctypes.c_int(fe_all))
    interp.invoke()
    raised = libm.fetestexcept(ctypes.c_int(fe_all))
    flags = {k: bool(raised & v) for k, v in fe.items()}

    out_idx = {d["index"] for d in interp.get_output_details()}
    tiny = np.finfo(np.float32).tiny
    scan_hits, out_hits, sat_max = 0, 0, 0.0
    outputs = []

    for t in interp.get_tensor_details():
        try:
            a = interp.get_tensor(t["index"])
        except Exception:
            continue
        if np.dtype(t["dtype"]).kind == "f":
            bad = bool(np.isnan(a).any() or np.isinf(a).any())
            nz = a[np.isfinite(a) & (a != 0)]
            bad = bad or (nz.size and bool((np.abs(nz) < tiny).any()))
            if bad:
                scan_hits += 1
                if t["index"] in out_idx:
                    out_hits += 1
        if t["index"] in out_idx:
            outputs.append(np.asarray(a).ravel()[:4096].tolist())

    with open(outfile, "w") as fh:
        json.dump({"flags": flags, "scan_hits": scan_hits,
                   "output_hits": out_hits, "sat_max": sat_max,
                   "outputs": outputs}, fh)


def run_detect(model_path, input_path, seed, workdir, device="cpu"):
    """Invoke the detector in a child process; classify failures."""
    outfile = os.path.join(workdir, "det.json")
    cmd = [sys.executable, os.path.abspath(__file__), "--_worker",
           "--model", model_path, "--seed", str(seed), "--outfile", outfile,
           "--device", device]
    if input_path:
        cmd += ["--input", input_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None
    if proc.returncode < 0:
        return "CRASHED", None
    if proc.returncode != 0:
        return "REJECTED", (proc.stderr or "").strip().splitlines()[-1:] or None
    with open(outfile) as fh:
        return "OK", json.load(fh)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """Wilson score interval -- correct for small n and proportions near 0/1,
    where the normal approximation is not."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------

def build_plan(model_path, trials, rng, only=None):
    """
    Choose injection sites appropriate to each fault class. `only` restricts
    scale/zero-point sites to a set of tensor indices -- useful because
    exception reachability is concentrated at the few tensors that feed an
    amplifying float op, and uniform sampling over all scales spends most
    trials on sites where no exception is possible.
    """
    model, _ = FI.load(model_path)
    targets = FI.enumerate_targets(model)

    float_consts = [e for e in targets
                    if e["const_offset"] and e["dtype"] == "float32"]
    int_consts = [e for e in targets
                  if e["const_offset"] and e["dtype"] in ("int8", "uint8")]
    scaled = [e for e in targets if e["scale_offset"] and e["scale_count"]]
    zped = [e for e in targets if e["zp_offset"] and e["zp_count"]]
    if only:
        scaled = [e for e in scaled if e["tensor"] in only] or scaled
        zped = [e for e in zped if e["tensor"] in only] or zped

    plan = []

    def add(fault, pool, magnitudes):
        if not pool:
            return
        for _ in range(trials):
            e = pool[int(rng.integers(len(pool)))]
            mag = (magnitudes[int(rng.integers(len(magnitudes)))]
                   if magnitudes else None)
            plan.append({"fault": fault, "target": e["tensor"],
                         "name": e["name"], "magnitude": mag})

    add("invalid", float_consts, None)
    add("divzero", float_consts, None)
    add("inf", float_consts, None)
    add("saturate", int_consts, None)
    # magnitudes span the runtime-accepted range and beyond, so the campaign
    # also characterises where TFLite starts rejecting metadata
    # Magnitudes concentrated in the band TFLite accepts. Extreme values are
    # rejected at prepare time and produce no detection opportunity, so a
    # coarse ladder wastes trials and leaves the overflow/underflow rows of
    # the coverage table empty. Scale 10 on a box-encoding tensor is already
    # enough to overflow float32 once exp() amplifies it.
    add("overflow", scaled,
        [5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0])
    add("underflow", scaled,
        [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12])
    add("scale_nan", scaled, None)
    add("zp_set", zped, [0, -64, 64, 127, -128])
    return plan


def classify(base, trial):
    """Five-way outcome plus which mechanisms fired."""
    if trial[0] != "OK":
        return trial[0], {}, {}
    det = trial[1]
    new_exc = [k for k in MEANINGFUL
               if det["flags"].get(k) and not base["flags"].get(k)]
    out_changed = det["outputs"] != base["outputs"]

    if new_exc and out_changed:
        status = "ACTIVATED_BOTH"
    elif new_exc:
        status = "ACTIVATED_EXCEPTION"   # masked before the output
    elif out_changed:
        status = "ACTIVATED_OUTPUT"      # wrong, but no exception to detect
    else:
        status = "NOT_ACTIVATED"

    mech = {
        "flags": bool(new_exc),
        "scan": det["scan_hits"] > base["scan_hits"],
        "output_only": det["output_hits"] > base["output_hits"],
    }
    return status, mech, {"exceptions": new_exc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--input")
    ap.add_argument("--trials", type=int, default=30,
                    help="trials per fault class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", help="comma-separated tensor indices to target "
                                   "for scale/zero-point faults")
    ap.add_argument("--device", choices=["cpu", "tpu"], default="cpu",
                    help="tpu requires a compiled *_edgetpu.tflite model")
    ap.add_argument("--json")
    ap.add_argument("--csv")
    # hidden worker mode
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--outfile", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker:
        detect_worker(args.model, args.input, args.seed, args.outfile,
                      args.device)
        return

    if not args.model:
        sys.exit("--model is required")

    rng = np.random.default_rng(args.seed)
    work = tempfile.mkdtemp(prefix="campaign_")

    st, base = run_detect(args.model, args.input, args.seed, work,
                          args.device)
    if st != "OK":
        sys.exit(f"clean baseline failed ({st}); cannot run campaign")
    print(f"baseline: flags={[k for k,v in base['flags'].items() if v]} "
          f"scan_hits={base['scan_hits']}\n")

    only = set(int(x) for x in args.only.split(",")) if args.only else None
    plan = build_plan(args.model, args.trials, rng, only)
    print(f"{len(plan)} trials planned\n")

    rows = []
    for n, item in enumerate(plan, 1):
        model, data = FI.load(args.model)
        entry = {e["tensor"]: e for e in FI.enumerate_targets(model)}[item["target"]]
        faulty = os.path.join(work, "faulty.tflite")
        try:
            rec = FI.patch(data, entry, item["fault"], item["magnitude"],
                           rng, None)
        except ValueError:
            continue
        with open(faulty, "wb") as fh:
            fh.write(bytes(data))

        trial = run_detect(faulty, args.input, args.seed, work, args.device)
        status, mech, extra = classify(base, trial)

        rows.append({
            "trial": n,
            "fault": item["fault"],
            "target": item["target"],
            "tensor_name": item["name"][:40],
            "magnitude": item["magnitude"],
            "status": status,
            "exceptions": ";".join(extra.get("exceptions", [])),
            "det_flags": mech.get("flags", False),
            "det_scan": mech.get("scan", False),
            "det_output_only": mech.get("output_only", False),
        })
        if n % 10 == 0 or n == len(plan):
            print(f"  {n}/{len(plan)} trials", end="\r", flush=True)
    print()

    # ---- outcome breakdown ------------------------------------------------
    print("\nOUTCOME BREAKDOWN")
    print(f"{'fault':<12} {'n':>4} {'rej':>4} {'crash':>6} {'inert':>6} "
          f"{'out':>5} {'exc':>5} {'both':>5}")
    print("-" * 56)
    faults = sorted({r["fault"] for r in rows})
    for f in faults:
        sub = [r for r in rows if r["fault"] == f]
        c = lambda s: sum(r["status"] == s for r in sub)  # noqa: E731
        print(f"{f:<12} {len(sub):>4} {c('REJECTED'):>4} {c('CRASHED'):>6} "
              f"{c('NOT_ACTIVATED'):>6} {c('ACTIVATED_OUTPUT'):>5} "
              f"{c('ACTIVATED_EXCEPTION'):>5} {c('ACTIVATED_BOTH'):>5}")

    # ---- detection rate over genuine opportunities -------------------------
    opp = [r for r in rows
           if r["status"] in ("ACTIVATED_EXCEPTION", "ACTIVATED_BOTH")]
    print(f"\nDETECTION RATE  (denominator = {len(opp)} trials that raised "
          f"an IEEE exception)")
    print(f"{'mechanism':<16} {'k/n':>9} {'rate':>7}  95% CI")
    print("-" * 48)
    for label, key in (("flag polling", "det_flags"),
                       ("tensor scan", "det_scan"),
                       ("output only", "det_output_only")):
        k = sum(r[key] for r in opp)
        p, lo, hi = wilson(k, len(opp))
        print(f"{label:<16} {k:>4}/{len(opp):<4} {p*100:>6.1f}%  "
              f"[{lo*100:.1f}, {hi*100:.1f}]")

    # ---- per exception class ---------------------------------------------
    print("\nBY EXCEPTION CLASS")
    print(f"{'class':<12} {'n':>4} {'flags':>7} {'scan':>7} {'out':>7}")
    print("-" * 40)
    for cls in MEANINGFUL:
        sub = [r for r in opp if cls in r["exceptions"].split(";")]
        if not sub:
            continue
        print(f"{cls:<12} {len(sub):>4} "
              f"{sum(r['det_flags'] for r in sub):>7} "
              f"{sum(r['det_scan'] for r in sub):>7} "
              f"{sum(r['det_output_only'] for r in sub):>7}")

    masked = [r for r in opp if r["status"] == "ACTIVATED_EXCEPTION"]
    print(f"\nfully masked before output: {len(masked)}/{len(opp)} "
          f"-- invisible to output inspection")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"model": os.path.basename(args.model),
                       "device": args.device,
                       "baseline": base["flags"], "rows": rows}, fh, indent=2)
        print(f"wrote {args.json}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()