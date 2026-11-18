"""Compare decoding orders and round counts for a copy_animator checkpoint.

    uv run python scripts/decode_experiments.py data/runs/<run> [--characters 64]

Same examples for every variant. Prints a table and writes <run>/decode-experiments.json.
"""
import argparse
import json
import time
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from moviepix.data.dataset import EvalSet, collate
from moviepix.eval.metrics import add, frame_metrics, summarize
from moviepix.train.checkpoint import load_model
from moviepix.utils.device import best_device

VARIANTS = [(1, "confidence"),
            (2, "confidence"), (2, "balanced"), (2, "silhouette"),
            (4, "confidence"), (4, "balanced"), (4, "random"), (4, "silhouette"),
            (8, "confidence"), (8, "balanced"), (8, "random"), (8, "silhouette")]

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--split", default="validation")
ap.add_argument("--characters", type=int, default=64)
ap.add_argument("--per-character", type=int, default=4)
a = ap.parse_args()
cfg = tomllib.loads((a.run / "config.toml").read_text())
dev = best_device()
model, _ = load_model(a.run / "best.pt", dev)
ds = EvalSet(Path("data/built") / cfg["dataset"], a.split, characters=a.characters, per_character=a.per_character, seed=123)
batches = list(DataLoader(ds, batch_size=32, collate_fn=collate))
rows = []
print(f"{len(ds)} examples, {cfg['dataset']}, {a.split}")
print(f"{'variant':16s} {'exact':>7s} {'iou':>7s} {'stray+':>7s} {'speck+':>7s} {'sec':>6s}")
for steps, order in VARIANTS:
    totals = {}
    t0 = time.time()
    for b in batches:
        ref, act, d, n, tgt = (x.to(dev) for x in (b.reference, b.action, b.direction, b.n_frames, b.target))
        pred = model.generate(ref, act, d, n, steps=steps, order=order,
                              generator=torch.Generator(device=dev).manual_seed(0) if order == "random" else None)
        add(totals, frame_metrics(pred, tgt, ref, n))
    if dev == "mps":
        torch.mps.empty_cache()
    r = {"steps": steps, "order": order, **summarize(totals), "seconds": round(time.time() - t0, 1)}
    rows.append(r)
    print(f"{steps} {order:14s} {r['pixel_exact']:7.4f} {r['alpha_iou']:7.4f} {r['stray_excess']:7.2f} "
          f"{r['speckle_excess']:7.2f} {r['seconds']:6.1f}", flush=True)
(a.run / "decode-experiments.json").write_text(json.dumps({"split": a.split, "examples": len(ds), "rows": rows}, indent=2))
