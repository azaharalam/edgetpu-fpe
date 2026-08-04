#!/usr/bin/env python3
"""Verify the environment before anything else runs.

The Coral stack is frozen at old versions: pycoral ships wheels only up to
CPython 3.9, and its C extension is built against NumPy 1.x. Running from the
wrong interpreter produces confusing import errors rather than a clear one.
"""
import sys

ok = True

def check(label, cond, hint):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        print(f"        {hint}")
        ok = False

print("environment check\n")
v = sys.version_info
check(f"python {v.major}.{v.minor}.{v.micro}",
      (v.major, v.minor) == (3, 9),
      "pycoral has no wheels above 3.9 -- run: conda activate coral")

try:
    import numpy
    check(f"numpy {numpy.__version__}", numpy.__version__.startswith("1."),
          "pycoral's C extension is built against NumPy 1.x -- "
          "run: pip install 'numpy<2'")
except ImportError:
    check("numpy", False, "pip install 'numpy<2'")

for mod, hint in [
    ("tflite_runtime", "pip install --extra-index-url "
                       "https://google-coral.github.io/py-repo/ "
                       "tflite-runtime==2.5.0.post1"),
    ("pycoral", "pip install --extra-index-url "
                "https://google-coral.github.io/py-repo/ pycoral==2.0.0"),
    ("tflite", "pip install tflite flatbuffers"),
    ("flatbuffers", "pip install tflite flatbuffers"),
    ("PIL", "pip install pillow"),
]:
    try:
        __import__(mod)
        check(mod, True, "")
    except ImportError:
        check(mod, False, hint)

try:
    from pycoral.utils import edgetpu
    n = len(edgetpu.list_edge_tpus())
    print(f"\n  edge tpu devices: {n}"
          + ("  (CPU arm only -- this is fine)" if n == 0 else ""))
except Exception as e:
    print(f"\n  edge tpu enumeration failed: {str(e)[:80]}")

print("\n" + ("environment ready" if ok else "environment incomplete"))
sys.exit(0 if ok else 1)
