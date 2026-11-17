"""Write animations as GIFs for previews and reports."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .sheets import on_background, upscale


def save_gif(frames: np.ndarray, path, scale: int = 3, fps: int = 8, background=(46, 52, 64)) -> None:
    """frames uint8 [T, S, S, 4] -> looping GIF."""
    imgs = [Image.fromarray(upscale(on_background(f, background), scale)) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0, optimize=False)
