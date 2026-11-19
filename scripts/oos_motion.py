"""How much do models MOVE out-of-style sprites? Share of pixels that differ from the
still reference (alpha or colour), in the truth walk vs. in generated frames. A model
that scores near the "repeat" baseline with little motion is just copying the still.

    uv run python scripts/oos_motion.py RUN [RUN ...] [--device mps]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from oos_walk import DIRS, SCALE, place, to_t, walks  # noqa: E402

from moviepix import actions as A  # noqa: E402
from moviepix.train.checkpoint import load_model  # noqa: E402


def changed(a, b):
    ao, bo = a[..., 3] > 0, b[..., 3] > 0
    diff = (ao != bo) | (ao & bo & np.any(a[..., :3] != b[..., :3], -1))
    return diff.sum() / max((ao | bo).sum(), 1)


ap = argparse.ArgumentParser()
ap.add_argument("runs", type=Path, nargs="+")
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
walk = A.get("walk")
for run in a.runs:
    model, _ = load_model(run / "best.pt", a.device)
    S = model.size
    per = {}
    for pack, char, d, ref, cells in walks():
        placed = place([ref] + cells, ref, S, SCALE[S][pack])
        r, truth = placed[0], placed[1:]
        with torch.no_grad():
            pred = model.generate(to_t(r).to(a.device), torch.tensor([walk.id], device=a.device),
                                  torch.tensor([DIRS.index(d)], device=a.device),
                                  torch.tensor([walk.frames], device=a.device), steps=1)[0]
        pf = (pred.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        pf[..., 3] = np.where(pf[..., 3] > 127, 255, 0)
        pf[pf[..., 3] == 0] = 0
        t = np.mean([changed(f, r) for f in truth])
        p = np.mean([changed(f, r) for f in pf[1:]])  # skip frame 0 (standing)
        per.setdefault(pack, []).append((t, p))
    print(f"\n{run.name}\n{'pack':12s} {'truth moves':>12s} {'model moves':>12s}")
    for k, v in per.items():
        v = np.array(v)
        print(f"{k:12s} {v[:, 0].mean():12.3f} {v[:, 1].mean():12.3f}")
