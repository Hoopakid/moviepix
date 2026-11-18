"""Contact sheet of style augmentations: each row = one character, walk (down) frames.

    uv run python scripts/show_augment.py data/built/<name> out.png [--characters 6]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from moviepix.data import augment as AUG
from moviepix.data.atlas import load_gray, load_png, unpack, unpack_parts
from moviepix.utils.io import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--characters", type=int, default=6)
a = ap.parse_args()
entries = read_jsonl(a.root / "manifest.jsonl")[: a.characters]
rows = []
for ci, e in enumerate(entries):
    S = e["size"]
    sheets = unpack(load_png(a.root / e["atlas"]), e["layout"], S)
    parts = unpack_parts(load_gray(a.root / e["parts"]), e["layout"], S)
    variants = [sheets]
    for t in AUG.TRANSFORMS:
        out, _ = AUG.augment(sheets, parts, np.random.default_rng([ci, len(variants)]), 1.0, transforms=(t,))
        variants.append(out)
    out, names = AUG.augment(sheets, parts, np.random.default_rng([ci, 99]), 1.0)
    variants.append(out)
    # per variant: reference + 4 walk frames (down row)
    cells = [np.concatenate([v["walk"][2, f] for f in (0, 1, 3, 5, 7)], 1) for v in variants]
    sep = np.zeros((S, 4, 4), np.uint8)
    rows.append(np.concatenate(sum([[c, sep] for c in cells], [])[:-1], 1))
img = np.concatenate(rows, 0)
bg = np.zeros_like(img); bg[..., :3] = 200; bg[..., 3] = 255
comp = np.where(img[..., 3:] > 0, img, bg)
k = 3 if S == 64 else 5
Image.fromarray(comp).resize((comp.shape[1] * k, comp.shape[0] * k), Image.NEAREST).save(a.out)
print("columns: original,", ", ".join(AUG.TRANSFORMS), ", random mix")
