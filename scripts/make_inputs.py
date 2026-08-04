#!/usr/bin/env python3
"""Regenerate the .npy inputs. Each model has its own expected resolution;
using the wrong one fails at reshape and is reported as a rejected model."""
import numpy as np, os
from PIL import Image

SPECS = [
    ("test_data/kite_and_cold.jpg", 192, "inputs/movenet_person.npy"),
    ("test_data/pets.jpg",          300, "inputs/ssd_input.npy"),
    ("test_data/pets.jpg",          320, "inputs/effdet_input.npy"),
    ("test_data/parrot.jpg",        224, "inputs/mobilenet_input.npy"),
    ("test_data/pets.jpg",          513, "inputs/deeplab_input.npy"),
]

os.makedirs("inputs", exist_ok=True)
for src, size, dst in SPECS:
    img = Image.open(src).convert("RGB").resize((size, size))
    np.save(dst, np.asarray(img, dtype=np.uint8)[None, ...])
    print(f"{dst}  {size}x{size}")
