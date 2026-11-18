"""Evaluate a checkpoint: quality on a split, plus the local-first numbers.

    moviepix-eval data/runs/<run>                   # best.pt on validation
    moviepix-eval data/runs/<run> --split unseen    # held-out items
    moviepix-eval data/runs/<run> --split test --characters all

Writes <run>/eval-<split>.json and <run>/eval-<split>.png, and prints a summary.
Local-first numbers (docs/local-first.md): parameters, fp32/fp16 size, and
CPU latency for one 8-frame animation using 4 threads.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from moviepix import actions as A
from moviepix.data.dataset import EvalSet, collate
from moviepix.eval.metrics import add, frame_metrics, summarize
from moviepix.train.checkpoint import load_model
from moviepix.utils.device import best_device
from moviepix.utils.io import write_json
from moviepix.viz.preview import comparison_grid
from moviepix.viz.sheets import save_png


def _gen_kwargs(model, steps):
    import inspect

    return {"steps": steps} if steps and "steps" in inspect.signature(model.generate).parameters else {}


@torch.no_grad()
def cpu_latency_ms(model, threads: int = 4, frames: int = 8, repeats: int = 10, steps: int | None = None) -> float:
    model = model.to("cpu").float().eval()
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    s = model.size
    args = (torch.rand(1, 4, s, s), torch.tensor([A.get("walk").id]), torch.tensor([2]), torch.tensor([frames]))
    kw = _gen_kwargs(model, steps)
    for _ in range(2):
        model.generate(*args, **kw)
    t = time.perf_counter()
    for _ in range(repeats):
        model.generate(*args, **kw)
    torch.set_num_threads(old)
    return (time.perf_counter() - t) / repeats * 1000


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--split", default="validation", choices=["validation", "test", "unseen"])
    ap.add_argument("--characters", default="256", help="number of characters, or 'all'")
    ap.add_argument("--per-character", type=int, default=4)
    ap.add_argument("--built", type=Path, default=Path("data/built"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, help="decoding rounds for iterative models (default: the model's)")
    args = ap.parse_args(argv)

    import tomllib

    cfg = tomllib.loads((args.run / "config.toml").read_text())
    device = best_device(args.device)
    model, state = load_model(args.run / args.checkpoint, device)
    chars = None if args.characters == "all" else int(args.characters)
    ds = EvalSet(args.built / cfg["dataset"], args.split, cfg.get("train", {}).get("actions"),
                 characters=chars, per_character=args.per_character, seed=123)
    totals, first = {}, None
    per_action: dict[str, dict] = {}
    for batch in DataLoader(ds, batch_size=16 if ds.size >= 64 else 64, collate_fn=collate):
        ref, act, d, n, tgt = (x.to(device) for x in (batch.reference, batch.action, batch.direction, batch.n_frames, batch.target))
        pred = model.generate(ref, act, d, n, **_gen_kwargs(model, args.steps))
        add(totals, frame_metrics(pred, tgt, ref, n))
        for a in act.unique().tolist():
            m = act == a
            add(per_action.setdefault(A.ACTIONS[a].name, {}), frame_metrics(pred[m], tgt[m], ref[m], n[m]))
        if first is None:
            first = (ref, tgt, pred, n)

    params = sum(p.numel() for p in model.parameters())
    result = {
        "run": str(args.run), "checkpoint": args.checkpoint, "step": state.get("step"), "split": args.split,
        "decode_steps": args.steps or getattr(model, "steps", None),
        "examples": len(ds), **summarize(totals),
        "per_action": {k: summarize(v) for k, v in sorted(per_action.items())},
        "local_first": {
            "parameters": params,
            "size_mb_fp32": round(params * 4 / 1e6, 2),
            "size_mb_fp16": round(params * 2 / 1e6, 2),
            "cpu_ms_8_frames_4_threads": round(cpu_latency_ms(model, steps=args.steps), 1),
        },
    }
    tag = args.split + (f"-steps{args.steps}" if args.steps else "")
    write_json(args.run / f"eval-{tag}.json", result)
    save_png(comparison_grid(*first, max_rows=12), args.run / f"eval-{tag}.png")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items() if k != "per_action"}, indent=1))


if __name__ == "__main__":
    main()
