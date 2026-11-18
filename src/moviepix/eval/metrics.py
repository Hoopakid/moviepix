"""Metrics on predicted frames, computed only over real (non-padding) frames.

alpha_iou        overlap of predicted and true silhouettes (alpha > 0.5)
pixel_exact      share of true opaque pixels the model gets exactly right, after
                 snapping each predicted color to the nearest color of the
                 reference sprite (what a pixel artist would do)
rgb_l1           mean color error on true opaque pixels, 0..1
palette_fidelity share of predicted opaque pixels whose raw color is already
                 within 8/255 of a reference color: low = blur / invented colors
stray_excess     extra stray pixels per frame versus the truth (an opaque pixel with
                 no opaque pixel among its 8 neighbours): 0 is ideal
speckle_excess   extra speckles per frame versus the truth (an opaque pixel whose
                 colour matches none of its 4 opaque neighbours: the "torn" look): 0 is ideal
detail_recall    share of the truth's speckles (1-pixel details: eye highlights, buttons)
                 that the prediction reproduces in the same colour within +-1 pixel
detail_precision share of the prediction's speckles that match a same-coloured truth
                 pixel within +-1 pixel (invented details lower it)
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from moviepix.models.blocks import masked_select_frames


def _keys_and_speckles(frames: torch.Tensor):
    a = frames[:, 3] > 0.5
    key = (frames[:, :3] * 255).round()
    key = torch.where(a, key[:, 0] * 65536 + key[:, 1] * 256 + key[:, 2], torch.full_like(key[:, 0], -1.0))
    same_any = torch.zeros_like(a)
    has_nb = torch.zeros_like(a)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        k2 = torch.roll(key, (dy, dx), (1, 2))
        o2 = torch.roll(a, (dy, dx), (1, 2))
        has_nb |= o2
        same_any |= o2 & (k2 == key)
    return key, a & has_nb & ~same_any


def _near_same_colour(src_key, src_mask, other_key):
    """For each masked pixel of src: does other have the same colour within the 3x3 neighbourhood?"""
    hit = torch.zeros_like(src_mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            hit |= torch.roll(other_key, (dy, dx), (1, 2)) == src_key
    return hit & src_mask


def detail_counts(pred: torch.Tensor, truth: torch.Tensor) -> tuple[int, int, int, int]:
    """(truth speckles, recovered, predicted speckles, correct predicted)."""
    pk, ps = _keys_and_speckles(pred)
    tk, ts = _keys_and_speckles(truth)
    return int(ts.sum()), int(_near_same_colour(tk, ts, pk).sum()), int(ps.sum()), int(_near_same_colour(pk, ps, tk).sum())


def coherence_counts(frames: torch.Tensor) -> tuple[int, int]:
    """(stray pixels, speckles) summed over frames [N, 4, S, S] in 0..1."""
    a = (frames[:, 3:4] > 0.5).float()
    n8 = F.conv2d(F.pad(a, (1, 1, 1, 1)), torch.ones(1, 1, 3, 3, device=a.device)) - a
    stray = int(((a > 0) & (n8 == 0)).sum())
    key = (frames[:, :3] * 255).round()
    key = key[:, 0] * 65536 + key[:, 1] * 256 + key[:, 2]  # [N, S, S]
    op = a[:, 0] > 0
    same_any = torch.zeros_like(op)
    has_nb = torch.zeros_like(op)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        k2 = torch.roll(key, (dy, dx), (1, 2))
        o2 = torch.roll(op, (dy, dx), (1, 2))
        has_nb |= o2
        same_any |= o2 & (k2 == key)
    speckle = int((op & has_nb & ~same_any).sum())
    return stray, speckle


def _palette(reference: torch.Tensor) -> list[torch.Tensor]:
    """Per sample, the unique opaque colors of the reference, [K, 3]."""
    out = []
    for ref in reference:
        rgb = ref[:3].flatten(1).T
        a = ref[3].flatten() > 0.5
        out.append(torch.unique((rgb[a] * 255).round(), dim=0) / 255 if a.any() else rgb[:1] * 0)
    return out


@torch.no_grad()
def frame_metrics(pred: torch.Tensor, target: torch.Tensor, reference: torch.Tensor, n_frames: torch.Tensor) -> dict:
    """pred, target: [B, T, 4, S, S] in 0..1. Returns sums and counts to aggregate."""
    t_max = min(pred.shape[1], target.shape[1])
    b, t, _ = masked_select_frames(n_frames, t_max)
    p, g = pred[b, t].float(), target[b, t].float()
    pa, ga = p[:, 3] > 0.5, g[:, 3] > 0.5
    inter = (pa & ga).sum().item()
    union = (pa | ga).sum().item()

    palettes = _palette(reference.float())
    exact = 0
    near = 0
    for i in range(len(b)):
        pal = palettes[b[i]]
        prgb = p[i, :3].flatten(1).T  # [S*S, 3]
        d = torch.cdist(prgb, pal)  # [S*S, K]
        dmin, idx = d.min(1)
        snapped = pal[idx]
        grgb = g[i, :3].flatten(1).T
        gm, pm = ga[i].flatten(), pa[i].flatten()
        ok = (snapped - grgb).abs().max(1).values < 0.5 / 255
        exact += (ok & gm & pm).sum().item()
        near += ((dmin < 8 / 255 * 3**0.5) & pm).sum().item()

    rgb_err = ((p[:, :3] - g[:, :3]).abs().mean(1) * ga).sum().item()
    ps, pk = coherence_counts(p)
    gs, gk = coherence_counts(g)
    d_true, d_rec, d_pred, d_ok = detail_counts(p, g)
    return {"inter": inter, "union": union, "exact": exact, "near": near, "rgb_err": rgb_err,
            "stray_pred": ps, "stray_true": gs, "speckle_pred": pk, "speckle_true": gk,
            "detail_true": d_true, "detail_rec": d_rec, "detail_pred": d_pred, "detail_ok": d_ok,
            "true_opaque": ga.sum().item(), "pred_opaque": pa.sum().item(), "frames": len(b)}


def summarize(totals: dict) -> dict:
    return {
        "alpha_iou": totals["inter"] / max(totals["union"], 1),
        "pixel_exact": totals["exact"] / max(totals["true_opaque"], 1),
        "rgb_l1": totals["rgb_err"] / max(totals["true_opaque"], 1),
        "palette_fidelity": totals["near"] / max(totals["pred_opaque"], 1),
        "stray_excess": (totals.get("stray_pred", 0) - totals.get("stray_true", 0)) / max(totals["frames"], 1),
        "speckle_excess": (totals.get("speckle_pred", 0) - totals.get("speckle_true", 0)) / max(totals["frames"], 1),
        "detail_recall": totals.get("detail_rec", 0) / max(totals.get("detail_true", 0), 1),
        "detail_precision": totals.get("detail_ok", 0) / max(totals.get("detail_pred", 0), 1),
        "frames": totals["frames"],
    }


def add(totals: dict, part: dict) -> dict:
    for k, v in part.items():
        totals[k] = totals.get(k, 0) + v
    return totals
