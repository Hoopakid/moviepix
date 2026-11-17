"""Compare several runs on the same examples.

    moviepix-compare data/runs/A data/runs/B data/runs/C --split validation
    moviepix-compare data/runs/C --steps 1 8        # one model, several decoding modes

Writes data/compare/<date>/compare-<split>.json, compare-<split>.png (one block per example:
reference | truth | each model) and prints a table.
"""

from __future__ import annotations

import argparse
import inspect
import json
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from moviepix.data.dataset import EvalSet, collate
from moviepix.eval.metrics import add, frame_metrics, summarize
from moviepix.train.checkpoint import load_model
from moviepix.utils.device import best_device
from moviepix.utils.io import write_json
from moviepix.viz.preview import _to_uint8
from moviepix.viz.sheets import on_background, save_png, upscale


def _variants(runs: list[Path], steps: list[int] | None):
    for run in runs:
        model_steps = [None]
        if steps:
            model_steps = steps
        for s in model_steps:
            yield run, s


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--split", default="validation", choices=["validation", "test", "unseen"])
    ap.add_argument("--characters", type=int, default=256)
    ap.add_argument("--per-character", type=int, default=4)
    ap.add_argument("--steps", type=int, nargs="*", help="decoding rounds to try for iterative models")
    ap.add_argument("--examples", type=int, default=10, help="examples in the image")
    ap.add_argument("--built", type=Path, default=Path("data/built"))
    ap.add_argument("--out", type=Path, default=Path("data/compare"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = best_device(args.device)
    datasets = {tomllib.loads((r / "config.toml").read_text())["dataset"] for r in args.runs}
    if len(datasets) != 1:
        raise SystemExit(f"runs use different datasets: {datasets}")
    ds = EvalSet(args.built / datasets.pop(), args.split, characters=args.characters,
                 per_character=args.per_character, seed=123)
    loader = DataLoader(ds, batch_size=64, collate_fn=collate)

    rows, previews, first = [], [], None
    for run, steps in _variants(args.runs, args.steps):
        model, state = load_model(run / "best.pt", device)
        takes_steps = "steps" in inspect.signature(model.generate).parameters
        if steps is not None and not takes_steps:
            if steps != (args.steps or [None])[0]:
                continue  # non-iterative model: evaluate once
            steps = None
        kw = {"steps": steps} if steps is not None else {}
        totals, shown = {}, None
        for batch in loader:
            ref, act, d, n, tgt = (x.to(device) for x in (batch.reference, batch.action, batch.direction,
                                                          batch.n_frames, batch.target))
            with torch.no_grad():
                pred = model.generate(ref, act, d, n, **kw)
            add(totals, frame_metrics(pred, tgt, ref, n))
            if shown is None:
                shown = pred[: args.examples].cpu()
                if first is None:
                    first = (ref[: args.examples].cpu(), tgt[: args.examples].cpu(), n[: args.examples].cpu())
        label = f"{state['model_name']} ({run.name[:13]})" + (f" steps={steps}" if steps is not None else "")
        rows.append({"label": label, "run": str(run), "step": state.get("step"),
                     "parameters": sum(p.numel() for p in model.parameters()), **summarize(totals)})
        previews.append(shown)
        print(f"{label:44s} exact {rows[-1]['pixel_exact']:.4f}  iou {rows[-1]['alpha_iou']:.4f}  "
              f"palette {rows[-1]['palette_fidelity']:.4f}", flush=True)

    out = args.out / f"{datetime.now():%Y%m%d-%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"compare-{args.split}.json", {"split": args.split, "examples": len(ds), "rows": rows})

    ref, tgt, n = first
    blocks = []
    for i in range(len(n)):
        k = int(n[i])
        s = ref.shape[-1]
        lines = [np.concatenate([_to_uint8(ref[i]), np.zeros((s, s * (k - 1), 4), np.uint8)], 1),
                 np.concatenate(list(_to_uint8(tgt[i, :k])), 1)]
        lines += [np.concatenate(list(_to_uint8(p[i, :k])), 1) for p in previews]
        block = np.concatenate([np.pad(l, ((0, 0), (0, s * (13 - k)), (0, 0))) for l in lines], 0)
        blocks.append(np.pad(block, ((0, s // 2), (0, 0), (0, 0))))
    save_png(upscale(on_background(np.concatenate(blocks, 0), (120, 120, 120)), 2), out / f"compare-{args.split}.png")
    labels = ["reference", "truth"] + [r["label"] for r in rows]
    (out / "README.txt").write_text("Each block, top to bottom: " + " | ".join(labels) + "\n")

    print(f"\n{"model":44s} {'params':>9s} {'exact':>7s} {'iou':>7s} {'rgb_l1':>7s} {'palette':>8s}")
    for r in rows:
        print(f"{r["label"]:44s} {r['parameters']:9d} {r['pixel_exact']:7.4f} {r['alpha_iou']:7.4f} "
              f"{r['rgb_l1']:7.4f} {r['palette_fidelity']:8.4f}")
    print(out)


if __name__ == "__main__":
    main()
