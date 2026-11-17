"""Prediction previews: for each example a row of reference | truth | prediction."""

from __future__ import annotations

import numpy as np
import torch

from .sheets import on_background, upscale


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    """[..., 4, S, S] float 0..1 -> uint8 [..., S, S, 4] with binary alpha."""
    x = x.detach().float().cpu().clamp(0, 1).movedim(-3, -1).numpy()
    out = (x * 255).round().astype(np.uint8)
    out[..., 3] = np.where(x[..., 3] > 0.5, 255, 0)
    return out


def comparison_grid(reference, target, pred, n_frames, max_rows: int = 8, scale: int = 2) -> np.ndarray:
    rows = []
    for i in range(min(max_rows, len(n_frames))):
        n = int(n_frames[i])
        ref = _to_uint8(reference[i])
        truth = _to_uint8(target[i, :n])
        guess = _to_uint8(pred[i, :n])
        s = ref.shape[0]
        gap = np.zeros((s, s // 4, 4), np.uint8)
        pad = np.zeros((s, s * (13 - n), 4), np.uint8)
        row = np.concatenate([ref, gap, *truth, pad, gap, *guess, pad], 1)
        rows.append(row)
    img = np.concatenate(rows, 0)
    return upscale(on_background(img, (120, 120, 120)), scale)
