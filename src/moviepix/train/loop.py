"""Train a model from a config.

    moviepix-train configs/model/unet_baseline.toml
    moviepix-train --resume data/runs/<run>          # continue after a stop
    moviepix-train configs/model/unet_baseline.toml --steps 200 --name smoke   # quick check

A run folder (data/runs/<date>-<name>/) holds:
    config.toml   the exact config used
    recipe.json   git commit, dataset id, seed, parameter count, device
    log.jsonl     one line per log interval (loss, speed) and per validation
    last.pt       latest checkpoint (model + optimizer + step), for --resume
    best.pt       best validation checkpoint (model only), for eval and export
    samples/      reference | truth | prediction images at each validation
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import time
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import moviepix.models  # noqa: F401  (registers models)
from moviepix.data.dataset import EvalSet, TrainSet, collate, collate_groups
from moviepix.eval.metrics import add, frame_metrics, summarize
from moviepix.models.interface import Batch
from moviepix.models.registry import build
from moviepix.train import checkpoint
from moviepix.utils.device import best_device
from moviepix.utils.io import file_sha256, git_commit, write_json
from moviepix.utils.seed import seed_everything
from moviepix.viz.preview import comparison_grid
from moviepix.viz.sheets import save_png

DEFAULTS = {
    "device": "auto", "amp": "bf16", "batch_characters": 8, "per_character": 4,
    "steps": 20000, "lr": 3e-4, "warmup": 500, "weight_decay": 0.01, "grad_clip": 1.0,
    "seed": 1, "workers": 4, "log_every": 50, "val_every": 1000,
    "val_characters": 128, "val_per_character": 4, "actions": None,
    # Apple GPUs share memory with the whole Mac. Cap what training may take
    # (fraction of the GPU's recommended working set) so the Mac stays usable;
    # over the cap, training stops with an out-of-memory error instead of swapping.
    "mps_memory_fraction": 0.6,
}


def to_device(batch: Batch, device: str) -> Batch:
    return Batch(*(x.to(device, non_blocking=True) if torch.is_tensor(x) else x for x in
                   (batch.reference, batch.action, batch.direction, batch.n_frames, batch.target, batch.text)))


def lr_at(step: int, cfg: dict) -> float:
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    progress = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def validate(model, loader, device, amp_dtype, sample_path: Path | None = None) -> dict:
    model.eval()
    totals, losses, first = {}, [], None
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, _ = model.loss(batch)
            pred = model.generate(batch.reference, batch.action, batch.direction, batch.n_frames)
        losses.append(loss.item())
        add(totals, frame_metrics(pred, batch.target, batch.reference, batch.n_frames))
        if first is None:
            first = (batch, pred)
    if device == "mps":
        torch.mps.empty_cache()  # hand cached decoding memory back to the Mac
    if sample_path is not None and first is not None:
        b, p = first
        save_png(comparison_grid(b.reference, b.target, p, b.n_frames), sample_path)
    model.train()
    return {"val_loss": float(np.mean(losses)), **summarize(totals)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, nargs="?")
    ap.add_argument("--resume", type=Path, help="a run folder to continue")
    ap.add_argument("--runs", type=Path, default=Path("data/runs"))
    ap.add_argument("--built", type=Path, default=Path("data/built"))
    ap.add_argument("--steps", type=int, help="override train.steps")
    ap.add_argument("--name", help="run name (default: config file name)")
    args = ap.parse_args(argv)

    if args.resume:
        run = args.resume
        cfg = tomllib.loads((run / "config.toml").read_text())
    elif args.config:
        cfg = tomllib.loads(args.config.read_text())
        name = args.name or args.config.stem
        run = args.runs / f"{datetime.now():%Y%m%d-%H%M}-{name}"
        run.mkdir(parents=True)
        shutil.copy(args.config, run / "config.toml")
    else:
        ap.error("give a config or --resume")
    tc = DEFAULTS | cfg.get("train", {})
    if args.steps:
        tc["steps"] = args.steps

    device = best_device(tc["device"])
    if device == "mps" and tc["mps_memory_fraction"]:
        torch.mps.set_per_process_memory_fraction(tc["mps_memory_fraction"])
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(tc["amp"]) if device != "cpu" else None
    seed_everything(tc["seed"])
    data_root = args.built / cfg["dataset"]
    info = json.loads((data_root / "dataset.json").read_text())

    model = build(cfg["model"], size=info["config"]["size"], **cfg.get("model_options", {})).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    step, best = 0, float("inf")

    if args.resume and (run / "last.pt").exists():
        state = torch.load(run / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step, best = state["step"], state.get("best", best)
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed {run} at step {step}")
    else:
        write_json(run / "recipe.json", {
            "model": cfg["model"], "dataset": cfg["dataset"], "dataset_manifest_sha256": file_sha256(data_root / "manifest.jsonl"),
            "git": git_commit(), "parameters": params, "device": device, "amp": tc["amp"],
            "machine": platform.platform(), "torch": torch.__version__, "train": tc,
        })

    train_set = TrainSet(data_root, "train", tc["actions"], per_character=tc["per_character"], seed=tc["seed"])
    val_set = EvalSet(data_root, "validation", tc["actions"], characters=tc["val_characters"],
                      per_character=tc["val_per_character"], seed=tc["seed"])
    val_loader = DataLoader(val_set, batch_size=tc["batch_characters"] * tc["per_character"], collate_fn=collate)
    (run / "samples").mkdir(exist_ok=True)

    print(json.dumps({"run": str(run), "model": cfg["model"], "parameters": params, "device": device,
                      "amp": tc["amp"], "train_characters": len(train_set), "val_examples": len(val_set)}), flush=True)

    model.train()
    log = open(run / "log.jsonl", "a")
    t_last, seen, running = time.time(), 0, []
    t_start, step_start = time.time(), step
    while step < tc["steps"]:
        train_set.epoch = step  # new shuffle and example picks after every resume or epoch
        loader = DataLoader(train_set, batch_size=tc["batch_characters"], shuffle=True, drop_last=True,
                            num_workers=tc["workers"], collate_fn=collate_groups, persistent_workers=False,
                            generator=torch.Generator().manual_seed(tc["seed"] + step))
        for batch in loader:
            if step >= tc["steps"]:
                break
            batch = to_device(batch, device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, tc)
            with torch.autocast(device, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss, logs = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            step += 1
            seen += len(batch.action)
            running.append(logs)

            if step % tc["log_every"] == 0:
                now = time.time()
                rate = (step - step_start) / (now - t_start)
                rec = {"step": step, "lr": round(lr_at(step, tc), 7),
                       **{k: round(float(np.mean([r[k] for r in running])), 5) for k in running[0]},
                       "examples_per_s": round(seen / (now - t_last), 1),
                       "eta_min": round((tc["steps"] - step) / max(rate, 1e-9) / 60, 1)}
                if device == "mps":
                    rec["gpu_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                t_last, seen, running = now, 0, []

            if step % tc["val_every"] == 0 or step == tc["steps"]:
                v = validate(model, val_loader, device, amp_dtype, run / "samples" / f"step-{step:07d}.png")
                rec = {"step": step, **{k: round(x, 5) if isinstance(x, float) else x for k, x in v.items()}}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                if v["val_loss"] < best:
                    best = v["val_loss"]
                    checkpoint.save(run / "best.pt", model, cfg["model"], cfg.get("model_options", {}), step,
                                    {"val": v})
                checkpoint.save(run / "last.pt", model, cfg["model"], cfg.get("model_options", {}), step,
                                {"optimizer": opt.state_dict(), "best": best, "torch_rng": torch.get_rng_state()})
    log.close()
    print(f"done: {run}")


if __name__ == "__main__":
    main()
