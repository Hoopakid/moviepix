"""Sprite-sheet helpers for previews."""

from __future__ import annotations

import numpy as np
from PIL import Image


def on_background(img: np.ndarray, color=(46, 52, 64)) -> np.ndarray:
    """Composite RGBA pixel art onto a flat color (alpha is 0 or 255)."""
    out = np.empty(img.shape[:-1] + (3,), np.uint8)
    out[:] = color
    m = img[..., 3] > 0
    out[m] = img[..., :3][m]
    return out


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    return img.repeat(factor, -3).repeat(factor, -2)


def frames_strip(frames: np.ndarray) -> np.ndarray:
    """[T, S, S, C] -> [S, T*S, C]."""
    t, s, _, c = frames.shape
    return frames.transpose(1, 0, 2, 3).reshape(s, t * s, c)


def save_png(img: np.ndarray, path) -> None:
    Image.fromarray(img).save(path, optimize=True)
