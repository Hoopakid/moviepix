"""Scoring predicted animations against a truth sequence, where standing still earns nothing.

A frame is a [S, S] int array of packed colours, -1 = transparent (see `keys`).
"Changed" means different from the still reference, in colour or transparency.

motion F1   2*hits / (changed in truth + changed in prediction), where a hit is a changed
            pixel whose new value also appears, as a changed pixel, within +-1 pixel in the
            other image. Repeating the reference changes nothing, so it scores exactly 0.
moved       share of the union of opaque pixels that changed (too little or too much motion).
exact, iou  strict appearance scores, frame by frame in order.

Truth and prediction are matched IN ORDER: one cyclic offset per sequence for looping
actions (chosen by motion F1), a straight time-resampling for one-shot actions. There is no
best frame per frame, unlike scripts/oos_walk.py. Truth frames identical to the reference
(standing frames of short cycles) keep their place in the order but are not scored: an LPC
walk cycle has none, and repeating the reference would get them for free.
"""

from __future__ import annotations

import numpy as np


def keys(img: np.ndarray, palette: np.ndarray | None = None) -> np.ndarray:
    """[..., 4] uint8 -> packed colours, -1 where transparent. With a palette [C, 3],
    each opaque colour is first snapped to its nearest palette colour (L1)."""
    rgb = img[..., :3].astype(np.int32)
    if palette is not None and len(palette):
        pal = palette.astype(np.int32)
        rgb = pal[np.abs(rgb[..., None, :] - pal).sum(-1).argmin(-1)]
    key = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    return np.where(img[..., 3] > 127, key, -1)


def _near(src: np.ndarray, src_mask: np.ndarray, other: np.ndarray, other_mask: np.ndarray) -> np.ndarray:
    """For each masked src pixel: is the same value present, as a masked pixel, within +-1 px?"""
    S0, S1 = src.shape
    pad = np.pad(np.where(other_mask, other, -2), 1, constant_values=-2)
    hit = np.zeros_like(src_mask)
    for dy in range(3):
        for dx in range(3):
            hit |= pad[dy:dy + S0, dx:dx + S1] == src
    return hit & src_mask


def counts(pred: np.ndarray, truth: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """[truth-changed, pred-changed, truth hits, pred hits, strict hits] for one frame pair."""
    dt, dp = truth != ref, pred != ref
    return np.array([dt.sum(), dp.sum(), _near(truth, dt, pred, dp).sum(), _near(pred, dp, truth, dt).sum(),
                     (dt & (pred == truth)).sum()], dtype=np.float64)


def f1(c: np.ndarray) -> float:
    """Motion F1 from summed counts (0 when either side never moves)."""
    return float((c[2] + c[3]) / (c[0] + c[1])) if c[0] + c[1] else 0.0


def align(n_truth: int, n_pred: int, loops: bool) -> list[list[int]]:
    """Candidate index lists into the predicted frames, one per allowed time alignment."""
    if not loops:
        return [list(np.round(np.linspace(0, n_pred - 1, n_truth)).astype(int))]
    base = np.round(np.arange(n_truth) * n_pred / n_truth).astype(int)
    return [list((base + o) % n_pred) for o in range(n_pred)]


def score_sequence(pred: list, truth: list, ref: np.ndarray, loops: bool = True) -> dict:
    """Score a predicted sequence (its frames in order, WITHOUT the standing frame if it
    has one) against the truth. All frames are key arrays."""
    moving = [j for j, t in enumerate(truth) if (t != ref).any()]
    if not moving:
        raise ValueError("the truth never differs from the reference")
    table = np.stack([[counts(p, t, ref) for t in truth] for p in pred])  # [P, T, 5]
    best = max(align(len(truth), len(pred), loops), key=lambda idx: f1(sum(table[idx[j], j] for j in moving)))
    tot = sum(table[best[j], j] for j in moving)
    exact = iou = moved_t = moved_p = 0.0
    for j in moving:
        i = best[j]
        p, t = pred[i], truth[j]
        to, po = t >= 0, p >= 0
        exact += (to & (p == t)).sum() / max(to.sum(), 1)
        iou += (to & po).sum() / max((to | po).sum(), 1)
        moved_t += (t != ref).sum() / max(((ref >= 0) | to).sum(), 1)
        moved_p += (p != ref).sum() / max(((ref >= 0) | po).sum(), 1)
    n = len(moving)
    return {"motion_f1": f1(tot), "motion_strict": float(2 * tot[4] / (tot[0] + tot[1])) if tot[0] + tot[1] else 0.0,
            "exact": exact / n, "iou": iou / n, "moved_truth": moved_t / n, "moved_pred": moved_p / n}


def shift(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.full_like(frame, -1)
    S0, S1 = frame.shape
    out[max(dy, 0):S0 + min(dy, 0), max(dx, 0):S1 + min(dx, 0)] = \
        frame[max(-dy, 0):S0 + min(-dy, 0), max(-dx, 0):S1 + min(-dx, 0)]
    return out


def still_baseline(truth: list, ref: np.ndarray) -> dict:
    return score_sequence([ref] * len(truth), truth, ref, loops=False)


def nudge_baseline(truth: list, ref: np.ndarray, oracle: bool = False) -> dict:
    """The dumbest animation: the whole still sprite moved 1 pixel, in every frame. Fixed
    rule: always up. With oracle=True, the best of the 4 directions is picked by looking at
    the truth (an upper bound for "no limb motion at all")."""
    ways = ((-1, 0), (1, 0), (0, -1), (0, 1)) if oracle else ((-1, 0),)
    return max((score_sequence([shift(ref, dy, dx)] * len(truth), truth, ref, loops=False) for dy, dx in ways),
               key=lambda s: s["motion_f1"])
