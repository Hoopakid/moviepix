"""One PNG per character holding every action, like an LPC universal sheet.

Actions are stacked top to bottom in moviepix.actions order; each action takes
one row per direction and one column per frame. The layout dict records where
each action is, so the image can be cut back into sheets.
"""

from __future__ import annotations

import numpy as np

from moviepix.actions import ACTIONS, MAX_FRAMES


def pack(sheets: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    """sheets[action] uint8 [dirs, frames, S, S, 4] -> (atlas [H, W, 4], layout)."""
    size = next(iter(sheets.values())).shape[2]
    layout, blocks, row = {}, [], 0
    for action in ACTIONS:
        sheet = sheets.get(action.name)
        if sheet is None:
            continue
        dirs, frames = sheet.shape[:2]
        block = np.zeros((dirs, MAX_FRAMES, size, size, 4), np.uint8)
        block[:, :frames] = sheet
        blocks.append(block.transpose(0, 2, 1, 3, 4).reshape(dirs * size, MAX_FRAMES * size, 4))
        layout[action.name] = {"row": row, "dirs": dirs, "frames": frames}
        row += dirs
    return np.concatenate(blocks, 0), layout


def unpack(atlas: np.ndarray, layout: dict, size: int, actions=None) -> dict[str, np.ndarray]:
    out = {}
    for name, spec in layout.items():
        if actions is not None and name not in actions:
            continue
        r, d, f = spec["row"], spec["dirs"], spec["frames"]
        block = atlas[r * size:(r + d) * size, : f * size]
        out[name] = block.reshape(d, size, f, size, 4).transpose(0, 2, 1, 3, 4)
    return out


TRANSPARENT = 0xFFFFFFFF


def encode_png(img: np.ndarray, compress_level: int = 6) -> bytes:
    """Lossless indexed-color PNG (pixel art has few colors). Falls back to RGBA
    if an image ever has more than 255 colors."""
    import io

    from PIL import Image

    rgb = img[..., :3].astype(np.uint32)
    key = np.where(img[..., 3] > 0, (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2], TRANSPARENT)
    colors, index = np.unique(key, return_inverse=True)
    buf = io.BytesIO()
    if len(colors) > 256:
        Image.fromarray(img).save(buf, format="PNG", compress_level=compress_level)
        return buf.getvalue()
    pal = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1).astype(np.uint8)
    pal[colors == TRANSPARENT] = 0
    p = Image.fromarray(index.reshape(key.shape).astype(np.uint8), "P")
    p.putpalette(pal.flatten().tolist())
    hole = np.nonzero(colors == TRANSPARENT)[0]
    kwargs = {"transparency": int(hole[0])} if len(hole) else {}
    p.save(buf, format="PNG", compress_level=compress_level, **kwargs)
    return buf.getvalue()


def load_png(path) -> np.ndarray:
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGBA")).copy()
    img[img[..., 3] == 0] = 0
    return img
