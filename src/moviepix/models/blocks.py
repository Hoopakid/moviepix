"""Shared layers: conditioning embeddings and FiLM residual conv blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from moviepix import actions as A


class Conditioning(nn.Module):
    """(action, direction, frame index, frame count) -> one vector per output frame."""

    def __init__(self, dim: int):
        super().__init__()
        self.action = nn.Embedding(len(A.ACTIONS), dim)
        self.direction = nn.Embedding(len(A.DIRECTIONS), dim)
        self.frame = nn.Embedding(A.MAX_FRAMES, dim)
        self.count = nn.Embedding(A.MAX_FRAMES + 1, dim)
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, action, direction, frame, n_frames) -> torch.Tensor:
        e = self.action(action) + self.direction(direction) + self.frame(frame) + self.count(n_frames)
        return self.mlp(e)


def groups(ch: int) -> int:
    return 8 if ch % 8 == 0 else 1


class ResBlock(nn.Module):
    """Conv residual block whose normalization is scaled and shifted by the condition (FiLM)."""

    def __init__(self, cin: int, cout: int, cond: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups(cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups(cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(cond, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, c):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(c)[:, :, None, None].chunk(2, 1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


def masked_select_frames(n_frames: torch.Tensor, t_max: int):
    """Indices (b, t) of the real (non-padding) frames, flattened."""
    t = torch.arange(t_max, device=n_frames.device)
    valid = t[None, :] < n_frames[:, None]  # [B, T]
    b_idx, t_idx = valid.nonzero(as_tuple=True)
    return b_idx, t_idx, valid
