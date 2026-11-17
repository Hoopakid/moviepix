"""Baseline B: copy by color.

For each output pixel the model chooses one of:
    transparent
    one of the reference sprite's colors (up to max_colors)
    "new color": a color not in the reference (a part hidden in the still
                 pose, e.g. the back of a cape); then a small RGB head paints it

Choosing among the reference's own colors means the output is crisp and uses
the sprite's exact palette, whatever the style. Palette colors are scored by
a dot product between the pixel's features and an embedding of the color
value, so any palette works, not a fixed one.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import Conditioning, UNetMixin, masked_select_frames
from .interface import Animator, Batch
from .registry import register


def _pack(rgb: torch.Tensor) -> torch.Tensor:
    """float 0..1 [..., 3] -> int key r<<16 | g<<8 | b."""
    q = (rgb * 255).round().long()
    return (q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2]


def reference_palettes(reference: torch.Tensor, max_colors: int):
    """Unique opaque colors per sprite: (colors [B, K, 3], valid [B, K])."""
    B = reference.shape[0]
    colors = reference.new_zeros(B, max_colors, 3)
    valid = torch.zeros(B, max_colors, dtype=torch.bool, device=reference.device)
    for i in range(B):
        rgb = reference[i, :3].flatten(1).T
        a = reference[i, 3].flatten() > 0.5
        keys, counts = torch.unique(_pack(rgb[a]), return_counts=True)
        keys = keys[counts.argsort(descending=True)][:max_colors]  # keep the most used colors
        k = len(keys)
        colors[i, :k] = torch.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], -1).float() / 255
        valid[i, :k] = True
    return colors, valid


@register("palette_tokens")
class PaletteTokens(Animator, UNetMixin):
    def __init__(self, size: int = 64, channels=(32, 64, 128, 128), cond_dim: int = 128,
                 dim: int = 64, max_colors: int = 64):
        super().__init__()
        self.size = size
        self.max_colors = max_colors
        self.cond = Conditioning(cond_dim)
        out_ch = self.build_unet(channels, cond_dim)
        self.to_query = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv2d(out_ch, dim, 1))
        self.special = nn.Conv2d(dim, 2, 1)  # logits: transparent, new color
        self.new_rgb = nn.Conv2d(dim, 3, 1)
        self.color_embed = nn.Sequential(nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.scale = dim ** -0.5

    def _logits(self, reference, action, direction, n_frames):
        t_max = int(n_frames.max())
        b, t, _ = masked_select_frames(n_frames, t_max)
        c = self.cond(action[b], direction[b], t, n_frames[b])
        q = self.to_query(self.unet(reference[b], c))  # [N, D, S, S]
        colors, valid = reference_palettes(reference, self.max_colors)
        keys = self.color_embed(colors.to(q.dtype))[b]  # [N, K, D]
        pal = torch.einsum("nds,nkd->nks", q.flatten(2), keys) * self.scale  # [N, K, S*S]
        pal = pal.masked_fill(~valid[b][:, :, None], float("-inf"))
        logits = torch.cat([self.special(q).flatten(2), pal], 1)  # [N, 2+K, S*S]
        return logits, torch.sigmoid(self.new_rgb(q)), colors, b, t

    def forward(self, reference, action, direction, n_frames, text=None):
        logits, new_rgb, colors, b, t = self._logits(reference, action, direction, n_frames)
        N, S = logits.shape[0], reference.shape[-1]
        choice = logits.float().argmax(1)  # [N, S*S]
        pal_idx = (choice - 2).clamp(min=0)
        rgb = torch.gather(colors[b].float(), 1, pal_idx[..., None].expand(-1, -1, 3))  # [N, S*S, 3]
        rgb = rgb.transpose(1, 2).reshape(N, 3, S, S)
        is_new = (choice == 1).reshape(N, 1, S, S)
        rgb = torch.where(is_new, new_rgb.float(), rgb)
        alpha = (choice != 0).reshape(N, 1, S, S).float()
        out = reference.new_zeros(reference.shape[0], int(n_frames.max()), 4, S, S)
        out[b, t] = torch.cat([rgb, alpha], 1).to(out.dtype)
        return out

    def loss(self, batch: Batch):
        logits, new_rgb, colors, b, t = self._logits(batch.reference, batch.action, batch.direction, batch.n_frames)
        target = batch.target[b, t]  # [N, 4, S, S]
        N = target.shape[0]
        tgt_rgb = target[:, :3].flatten(2).transpose(1, 2)  # [N, S*S, 3]
        opaque = target[:, 3].flatten(1) > 0.5
        # label: 0 transparent, 1 new color, 2+k reference color k
        match = _pack(tgt_rgb)[:, :, None] == _pack(colors[b])[:, None, :]  # [N, S*S, K]
        _, valid = reference_palettes(batch.reference, self.max_colors)
        match &= valid[b][:, None, :]
        found = match.any(-1)
        label = torch.where(found, match.float().argmax(-1) + 2, torch.ones_like(found, dtype=torch.long))
        label = torch.where(opaque, label, torch.zeros_like(label))
        ce = F.cross_entropy(logits.float(), label)
        new_mask = (opaque & ~found).reshape(N, 1, *target.shape[-2:]).float()
        rgb_l1 = ((new_rgb.float() - target[:, :3]).abs() * new_mask).sum() / (new_mask.sum() * 3).clamp(min=1)
        loss = ce + rgb_l1
        return loss, {"loss": loss.item(), "ce": ce.item(), "new_rgb_l1": rgb_l1.item(),
                      "new_share": (new_mask.sum() / opaque.sum().clamp(min=1)).item()}
