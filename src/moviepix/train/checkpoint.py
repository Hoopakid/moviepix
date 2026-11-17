"""Checkpoints that describe themselves: weights + model name + options + size.

A checkpoint alone is enough to rebuild the model for inference or export.
Training checkpoints also carry optimizer, scheduler and RNG state for resume.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from moviepix.models.registry import build


def save(path: Path, model, model_name: str, model_options: dict, step: int, extra: dict | None = None) -> None:
    state = {"model": model.state_dict(), "model_name": model_name, "model_options": model_options,
             "size": model.size, "step": step, **(extra or {})}
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_model(path: Path | str, device: str = "cpu"):
    import moviepix.models  # noqa: F401  (registers models)

    state = torch.load(path, map_location=device, weights_only=False)
    model = build(state["model_name"], size=state["size"], **state["model_options"])
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state
