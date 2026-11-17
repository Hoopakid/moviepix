"""Metrics on predicted frames, computed only over real (non-padding) frames.

alpha_iou        overlap of predicted and true silhouettes (alpha > 0.5)
pixel_exact      share of true opaque pixels the model gets exactly right, after
                 snapping each predicted color to the nearest color of the
                 reference sprite (what a pixel artist would do)
rgb_l1           mean color error on true opaque pixels, 0..1
palette_fidelity share of predicted opaque pixels whose raw color is already
                 within 8/255 of a reference color: low = blur / invented colors
"""

from __future__ import annotations

import torch

from moviepix.models.blocks import masked_select_frames


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
    return {"inter": inter, "union": union, "exact": exact, "near": near, "rgb_err": rgb_err,
            "true_opaque": ga.sum().item(), "pred_opaque": pa.sum().item(), "frames": len(b)}


def summarize(totals: dict) -> dict:
    return {
        "alpha_iou": totals["inter"] / max(totals["union"], 1),
        "pixel_exact": totals["exact"] / max(totals["true_opaque"], 1),
        "rgb_l1": totals["rgb_err"] / max(totals["true_opaque"], 1),
        "palette_fidelity": totals["near"] / max(totals["pred_opaque"], 1),
        "frames": totals["frames"],
    }


def add(totals: dict, part: dict) -> dict:
    for k, v in part.items():
        totals[k] = totals.get(k, 0) + v
    return totals
