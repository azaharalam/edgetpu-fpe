#!/usr/bin/env python3
"""
characterize_fp_surface.py

Characterizes the floating-point exception surface of TFLite models for
Edge TPU instrumentation. For each model it reports:

  - partition split (Edge TPU ops vs CPU ops), from edgetpu_compiler -s
  - float32 tensor count and total float32 element count
  - which ops produce float32 outputs  (= the FPE-susceptible surface)
  - int8/uint8 tensor count (= the saturation-analog surface)

The key point: CPU fallback does NOT imply floating point. An op can fall
back to the CPU and still run entirely in the quantized integer domain, in
which case no IEEE exception can occur there. This script measures the
distinction instead of assuming it.

Usage:
    python characterize_fp_surface.py test_data/
    python characterize_fp_surface.py test_data/ --json surface.json
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


FLOAT_TYPES = {"float32", "float16", "float64"}
INT_TYPES = {"int8", "uint8", "int16", "int32", "int64"}


def compiler_partition(model_path, timeout=240):
    """Run edgetpu_compiler -s and parse the TPU/CPU op split."""
    result = {"tpu_ops": None, "cpu_ops": None, "compiles": False, "error": None}
    try:
        proc = subprocess.run(
            ["edgetpu_compiler", "-s", "-o", "/tmp", model_path],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        result["error"] = "edgetpu_compiler not installed"
        return result
    except subprocess.TimeoutExpired:
        result["error"] = "compiler timeout"
        return result

    out = proc.stdout + proc.stderr
    if "Compilation succeeded" in out:
        result["compiles"] = True
    m = re.search(r"run on Edge TPU:\s*(\d+)", out)
    if m:
        result["tpu_ops"] = int(m.group(1))
    m = re.search(r"run on CPU:\s*(\d+)", out)
    result["cpu_ops"] = int(m.group(1)) if m else (0 if result["compiles"] else None)
    if not result["compiles"]:
        m = re.search(r"Compilation failed:\s*(.+)", out)
        if m:
            result["error"] = m.group(1).strip()
    return result


def tensor_profile(model_path):
    """Inspect tensors and ops to locate the floating-point surface."""
    prof = {
        "total_tensors": 0,
        "float_tensors": 0,
        "float_elements": 0,
        "int_tensors": 0,
        "int_elements": 0,
        "float_producing_ops": {},
        "op_histogram": {},
        "error": None,
    }

    try:
        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
    except Exception as exc:
        prof["error"] = str(exc)[:120]
        return prof

    details = interp.get_tensor_details()
    prof["total_tensors"] = len(details)

    by_index = {}
    for t in details:
        dtype = np.dtype(t["dtype"]).name
        n = int(np.prod(t["shape"])) if len(t["shape"]) else 0
        by_index[t["index"]] = dtype
        if dtype in FLOAT_TYPES:
            prof["float_tensors"] += 1
            prof["float_elements"] += n
        elif dtype in INT_TYPES:
            prof["int_tensors"] += 1
            prof["int_elements"] += n

    # Map ops -> output dtypes so we know which kernels can raise an FPE.
    try:
        ops = interp._get_ops_details()
    except Exception:
        ops = []
        prof["float_producing_ops"] = {"<op details unavailable>": prof["float_tensors"]}

    for op in ops:
        name = op.get("op_name", "UNKNOWN")
        prof["op_histogram"][name] = prof["op_histogram"].get(name, 0) + 1
        for out_idx in op.get("outputs", []):
            if by_index.get(int(out_idx)) in FLOAT_TYPES:
                prof["float_producing_ops"][name] = (
                    prof["float_producing_ops"].get(name, 0) + 1
                )
                break

    return prof


def classify_role(part, prof):
    """Assign a benchmark role based on measured surface, not op count."""
    has_fp = prof["float_tensors"] > 0
    cpu = part["cpu_ops"]
    if not part["compiles"]:
        return "EXCLUDED (does not compile)"
    if cpu == 0:
        return "null control (no CPU partition)"
    if not has_fp:
        return "negative control (CPU fallback, integer only)"
    if cpu is not None and cpu >= 30:
        return "large FP surface (localization)"
    return "FP surface present"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="directory containing .tflite models")
    ap.add_argument("--json", help="write full results to this JSON file")
    ap.add_argument("--skip-compiler", action="store_true",
                    help="skip edgetpu_compiler (tensor profile only)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.model_dir, "*.tflite")))
    paths = [p for p in paths if not p.endswith("_edgetpu.tflite")]
    if not paths:
        sys.exit(f"no uncompiled .tflite models found in {args.model_dir}")

    results = {}
    header = (f"{'model':<52} {'TPU':>5} {'CPU':>5} {'f32_t':>6} "
              f"{'f32_elem':>10} {'role':<40}")
    print(header)
    print("-" * len(header))

    for path in paths:
        name = os.path.basename(path)
        part = ({"tpu_ops": None, "cpu_ops": None, "compiles": True, "error": None}
                if args.skip_compiler else compiler_partition(path))
        prof = tensor_profile(path)
        role = classify_role(part, prof)

        results[name] = {"partition": part, "profile": prof, "role": role}
        print(f"{name:<52} {str(part['tpu_ops']):>5} {str(part['cpu_ops']):>5} "
              f"{prof['float_tensors']:>6} {prof['float_elements']:>10} {role:<40}")

    # Detail for models that actually have an FP surface.
    print("\n\nFloat-producing ops (the FPE-susceptible kernels):")
    print("-" * 70)
    for name, r in results.items():
        fp_ops = r["profile"]["float_producing_ops"]
        if fp_ops and r["profile"]["float_tensors"] > 0:
            listed = ", ".join(f"{k}({v})" for k, v in sorted(fp_ops.items()))
            print(f"\n{name}\n    {listed}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
