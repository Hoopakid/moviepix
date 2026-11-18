"""Pixel-art-aware downscaling by an integer factor (64 -> 32).

Each output pixel stands for a k x k block. It is opaque when at least half
of the block is opaque, and takes the block's most common opaque color. Ties
(common in 2x2 blocks) go to the color closest to the average of the
surrounding 2k x 2k area, so fills win over stray outline pixels inside the
sprite while the silhouette still comes from alpha. No colors are blended:
the result uses only colors from the original sprite.
"""

from __future__ import annotations

import numpy as np


def downscale(img: np.ndarray, factor: int = 2, labels: np.ndarray | None = None):
    """uint8 [..., H, W, 4] -> uint8 [..., H/factor, W/factor, 4].

    With labels (uint8 [..., H, W], e.g. body parts) also returns the labels of the
    picked pixels, so every output pixel keeps the label of the colour it took."""
    *lead, h, w, _ = img.shape
    if h % factor or w % factor:
        raise ValueError(f"size {h}x{w} is not divisible by {factor}")
    k = factor * factor
    blocks = img.reshape(-1, h // factor, factor, w // factor, factor, 4).transpose(0, 1, 3, 2, 4, 5)
    blocks = blocks.reshape(-1, k, 4)  # [N, k, 4]

    opaque = blocks[..., 3] > 0
    rgb = blocks[..., :3].astype(np.uint32)
    key = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    # votes[n, i] = how many opaque pixels in block n share pixel i's color
    same = (key[:, :, None] == key[:, None, :]) & opaque[:, None, :] & opaque[:, :, None]
    votes = same.sum(-1)
    # tie-break: closeness to the mean opaque color of the surrounding area
    flat = img.reshape(-1, h, w, 4).astype(np.float64)
    a = (flat[..., 3:] > 0).astype(np.float64)
    pad = factor // 2
    def box(x):
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        c = x.cumsum(1).cumsum(2)
        c = np.pad(c, ((0, 0), (1, 0), (1, 0), (0, 0)))
        n = 2 * factor
        hs, ws = (h + 2 * pad - n) // factor + 1, (w + 2 * pad - n) // factor + 1
        ys, xs = np.arange(hs) * factor, np.arange(ws) * factor
        y0, x0 = ys[:, None], xs[None, :]
        return c[:, y0 + n, x0 + n] - c[:, y0, x0 + n] - c[:, y0 + n, x0] + c[:, y0, x0]
    hb, wb = h // factor, w // factor
    mean = (box(flat[..., :3] * a) / np.maximum(box(a), 1))[:, :hb, :wb].reshape(-1, 1, 3)
    dist = np.abs(rgb.astype(np.float64) - mean).sum(-1) / (3 * 255 * 2)  # < 0.5
    score = np.where(opaque, votes - dist, -1.0)
    pick = score.argmax(1)

    out = blocks[np.arange(len(blocks)), pick].copy()
    keep = opaque.sum(1) * 2 >= k
    out[~keep] = 0
    out[keep, 3] = 255
    out = out.reshape(*lead, h // factor, w // factor, 4)
    if labels is None:
        return out
    lb = labels.reshape(-1, h // factor, factor, w // factor, factor).transpose(0, 1, 3, 2, 4).reshape(-1, k)
    lab = lb[np.arange(len(lb)), pick].copy()
    lab[~keep] = 0
    return out, lab.reshape(*lead, h // factor, w // factor)
