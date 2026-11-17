"""Baseline A: a U-Net that predicts each frame directly from the reference.

Every output frame is one U-Net pass over the reference sprite, conditioned
on (action, direction, frame index, frame count). It is trained with plain
reconstruction losses, so where the motion is uncertain it averages the
possibilities and blurs. That is expected: it is the reference point the
real models must beat, and it shows why a generative or copy-based model is
needed.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


from .blocks import Conditioning, UNetMixin, masked_select_frames
from .interface import Animator, Batch
from .registry import register


@register("unet_baseline")
class UNetBaseline(Animator, UNetMixin):
    def __init__(self, size: int = 64, channels=(32, 64, 128, 128), cond_dim: int = 128):
        super().__init__()
        self.size = size
        self.cond = Conditioning(cond_dim)
        out_ch = self.build_unet(channels, cond_dim)
        self.head = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv2d(out_ch, 4, 3, padding=1))

    def _frames(self, reference, action, direction, n_frames):
        """Predict only the real frames. Returns (rgb, alpha_logit, b_idx, t_idx)."""
        t_max = int(n_frames.max())
        b, t, _ = masked_select_frames(n_frames, t_max)
        c = self.cond(action[b], direction[b], t, n_frames[b])
        out = self.head(self.unet(reference[b], c))
        return torch.sigmoid(out[:, :3]), out[:, 3:], b, t

    def forward(self, reference, action, direction, n_frames, text=None):
        rgb, alpha_logit, b, t = self._frames(reference, action, direction, n_frames)
        B, S = reference.shape[0], reference.shape[-1]
        out = reference.new_zeros(B, int(n_frames.max()), 4, S, S)
        out[b, t] = torch.cat([rgb, torch.sigmoid(alpha_logit)], 1).to(out.dtype)
        return out

    def loss(self, batch: Batch):
        rgb, alpha_logit, b, t = self._frames(batch.reference, batch.action, batch.direction, batch.n_frames)
        target = batch.target[b, t]
        alpha = target[:, 3:]
        alpha_loss = F.binary_cross_entropy_with_logits(alpha_logit, alpha)
        rgb_loss = ((rgb - target[:, :3]).abs() * alpha).sum() / (alpha.sum() * 3).clamp(min=1)
        loss = alpha_loss + rgb_loss
        return loss, {"loss": loss.item(), "alpha_bce": alpha_loss.item(), "rgb_l1": rgb_loss.item()}


