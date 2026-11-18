"""Style augmentation for training (Phase 2, docs/research/phase2-design.md).

Every transform maps a whole character (all action sheets, plus its body-part maps)
deterministically, frame by frame, with parameters drawn once per character. The
reference (walk frame 0) and the target frames therefore go through exactly the same
function, so the target is still an exact pixel-art animation of the new reference.

    sheets[action]  uint8 [dirs, frames, S, S, 4]
    parts[action]   uint8 [dirs, frames, S, S]   ids from sources/lpc/compose.PARTS
"""

from __future__ import annotations

import colorsys

import numpy as np

from moviepix import actions as A

HEADS = (2, 3, 4)  # head, hair, headwear
LEGS = 7
TRANSFORMS = ("chibi", "short_legs", "size", "posterise", "outline", "recolour", "mirror")


def _frames(sheets):
    """Every frame of a character as one array [N, S, S, 4] plus a function to put it back."""
    keys = list(sheets)
    shapes = [sheets[k].shape for k in keys]
    flat = np.concatenate([sheets[k].reshape(-1, *sheets[k].shape[2:]) for k in keys])

    def back(arr):
        out, i = {}, 0
        for k, sh in zip(keys, shapes):
            n = sh[0] * sh[1]
            out[k] = arr[i:i + n].reshape(*sh[:2], *arr.shape[1:])
            i += n
        return out
    return flat, back


def _colours(img):
    """Unique opaque colours of [..., 4] and the packed key of every pixel (-1 = transparent)."""
    rgb = img[..., :3].astype(np.int64)
    key = np.where(img[..., 3] > 0, (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2], -1)
    return np.unique(key[key >= 0]), key


def _remap(img, mapping):
    """Replace colours: mapping = (keys sorted [C], new rgb [C, 3])."""
    keys, new = mapping
    _, key = _colours(img)
    out = img.copy()
    m = key >= 0
    out[..., :3][m] = new[np.searchsorted(keys, key[m])]
    return out


def _luma(keys):
    rgb = np.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], -1).astype(np.float64)
    return rgb @ np.array([0.299, 0.587, 0.114]), rgb


# ---- transforms ------------------------------------------------------------

def chibi(frames, parts, rng, S):
    """Scale the head group (head, hair, headwear) above the neck by k, about the
    bottom-centre of the face, per frame. Long hair below the neck stays as it is."""
    k0 = rng.choice([1.25, 1.5, 1.75])
    out, pout = frames.copy(), parts.copy()
    yy, xx = np.mgrid[0:S, 0:S]
    for i in range(len(frames)):
        face = (parts[i] == 2) & (frames[i, ..., 3] > 0)
        if face.sum() < 4:
            continue
        fy, fx = np.nonzero(face)
        ay, ax = fy.max() + 1, (fx.min() + fx.max() + 1) / 2  # neck line, face centre
        m = np.isin(parts[i], HEADS) & (frames[i, ..., 3] > 0) & (yy < ay)
        top = np.nonzero(m)[0].min()
        k = min(k0, (ay - 1) / max(ay - top, 1))  # keep the head inside the canvas
        if k <= 1.05:
            continue
        head, hpart = np.where(m[..., None], frames[i], 0), np.where(m, parts[i], 0)
        out[i][m], pout[i][m] = 0, 0
        sy = np.floor(ay + (yy + 0.5 - ay) / k).astype(int)  # nearest source pixel
        sx = np.floor(ax + (xx + 0.5 - ax) / k).astype(int)
        ok = (sy >= 0) & (sy < S) & (sx >= 0) & (sx < S)
        src = np.zeros((S, S, 4), np.uint8)
        sp = np.zeros((S, S), np.uint8)
        src[ok] = head[sy[ok], sx[ok]]
        sp[ok] = hpart[sy[ok], sx[ok]]
        d = src[..., 3] > 0
        out[i][d], pout[i][d] = src[d], sp[d]
    return out, pout


def short_legs(frames, parts, rng, S, ref):
    """Remove n rows from the leg band (the same canvas rows in every frame); shift the
    upper body down so the feet stay where they were."""
    legs = (parts[ref] == LEGS) & (frames[ref, ..., 3] > 0)
    if legs.sum() < 4:
        return frames, parts
    ys = np.nonzero(legs)[0]
    top, bottom = ys.min(), ys.max()
    span = bottom - top + 1
    n = int(rng.integers(max(1, span // 5), max(2, span // 2) + 1))
    start = top + (span - n) // 2
    keep = np.r_[0:start, start + n:S]
    out = np.zeros_like(frames)
    pout = np.zeros_like(parts)
    out[:, n:] = frames[:, keep]
    pout[:, n:] = parts[:, keep]
    return out, pout


def size(frames, parts, rng, S, ref):
    """Shrink the whole character (nearest) about its feet: a smaller sprite in the same canvas."""
    s = rng.choice([0.5, 0.625, 0.75])
    a = frames[ref, ..., 3] > 0
    ys, xs = np.nonzero(a)
    if len(ys) == 0:
        return frames, parts
    fy, cx = ys.max() + 1, (xs.min() + xs.max() + 1) / 2
    yy, xx = np.mgrid[0:S, 0:S]
    sy = np.floor(fy + (yy + 0.5 - fy) / s).astype(int)
    sx = np.floor(cx + (xx + 0.5 - cx) / s).astype(int)
    ok = (sy >= 0) & (sy < S) & (sx >= 0) & (sx < S)
    out = np.zeros_like(frames)
    pout = np.zeros_like(parts)
    out[:, ok] = frames[:, sy[ok], sx[ok]]
    pout[:, ok] = parts[:, sy[ok], sx[ok]]
    return out, pout


def posterise(frames, rng):
    """Merge the palette into 3-6 luminance groups (flat / retro shading)."""
    keys, _ = _colours(frames)
    n = int(rng.integers(3, 7))
    if len(keys) <= n:
        return frames
    luma, rgb = _luma(keys)
    order = np.argsort(luma)
    groups = np.array_split(order, n)
    _, key = _colours(frames)
    counts = np.bincount(np.searchsorted(keys, key[key >= 0]), minlength=len(keys))
    new = rgb.copy()
    for g in groups:
        new[g] = rgb[g[np.argmax(counts[g])]]  # each group takes its most used colour
    return _remap(frames, (keys, new.astype(np.uint8)))


def outline(frames, rng, ref):
    """Add a 1 px outline in the reference's darkest colour, or remove a dark outer ring."""
    keys, _ = _colours(frames[ref])
    if len(keys) == 0:
        return frames
    luma, rgb = _luma(keys)
    dark = rgb[np.argmin(luma)].astype(np.uint8)
    a = frames[..., 3] > 0
    pad = np.pad(a, ((0, 0), (1, 1), (1, 1)))
    nb_on = pad[:, :-2, 1:-1] | pad[:, 2:, 1:-1] | pad[:, 1:-1, :-2] | pad[:, 1:-1, 2:]
    nb_off = ~(pad[:, :-2, 1:-1] & pad[:, 2:, 1:-1] & pad[:, 1:-1, :-2] & pad[:, 1:-1, 2:])
    out = frames.copy()
    if rng.random() < 0.7:
        ring = ~a & nb_on
        out[ring, :3] = dark
        out[ring, 3] = 255
    else:
        lum = frames[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        ring = a & nb_off & (lum <= luma.min() + 30)
        out[ring] = 0
    return out


def recolour(frames, rng):
    """Rotate the hue and scale the value of every colour (one mapping for all frames)."""
    keys, _ = _colours(frames)
    if len(keys) == 0:
        return frames
    _, rgb = _luma(keys)
    dh, sv = rng.random(), rng.uniform(0.8, 1.2)
    new = np.array([colorsys.hsv_to_rgb((h + dh) % 1, s, min(1.0, v * sv))
                    for h, s, v in (colorsys.rgb_to_hsv(*(c / 255)) for c in rgb)])
    return _remap(frames, (keys, np.round(new * 255).astype(np.uint8)))


def mirror(sheets, parts):
    """Flip horizontally; left and right swap direction rows."""
    li, ri = A.DIRECTIONS.index("left"), A.DIRECTIONS.index("right")
    out, pout = {}, {}
    for k, v in sheets.items():
        f, p = v[:, :, :, ::-1].copy(), parts[k][:, :, :, ::-1].copy()
        if v.shape[0] == len(A.DIRECTIONS):
            idx = list(range(4))
            idx[li], idx[ri] = ri, li
            f, p = f[idx], p[idx]
        out[k], pout[k] = f, p
    return out, pout


# ---- entry point -----------------------------------------------------------

def augment(sheets: dict, parts: dict | None, rng: np.random.Generator, prob: float = 0.5,
            transforms=TRANSFORMS) -> tuple[dict, list[str]]:
    """With probability `prob`, apply 1-3 random transforms to a whole character.
    Returns (sheets, names of the transforms applied)."""
    if rng.random() >= prob:
        return sheets, []
    allowed = [t for t in transforms if parts is not None or t not in ("chibi", "short_legs")]
    names = list(rng.choice(allowed, size=min(len(allowed), int(rng.integers(1, 4))), replace=False))
    if parts is None:
        parts = {k: np.zeros(v.shape[:4], np.uint8) for k, v in sheets.items()}
    S = next(iter(sheets.values())).shape[2]
    if "mirror" in names:
        sheets, parts = mirror(sheets, parts)
    flat, back = _frames(sheets)
    pflat, pback = _frames({k: v[..., None] for k, v in parts.items()})
    pflat = pflat[..., 0]
    # index of the reference (walk, first frame of the first row) in the flat array
    ref = 0
    for k in sheets:
        if k == "walk":
            break
        ref += sheets[k].shape[0] * sheets[k].shape[1]
    # fixed order: shapes first (they use the part maps), then colours
    for name in ("chibi", "short_legs", "size", "posterise", "outline", "recolour"):
        if name not in names:
            continue
        if name == "chibi":
            flat, pflat = chibi(flat, pflat, rng, S)
        elif name == "short_legs":
            flat, pflat = short_legs(flat, pflat, rng, S, ref)
        elif name == "size":
            flat, pflat = size(flat, pflat, rng, S, ref)
        elif name == "posterise":
            flat = posterise(flat, rng)
        elif name == "outline":
            flat = outline(flat, rng, ref)
        elif name == "recolour":
            flat = recolour(flat, rng)
    return back(flat), sorted(names)
