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


class UNetMixin:
    """A conditioned U-Net trunk added to a model as attributes stem/down/pool/mid/up
    (names kept stable so saved checkpoints keep loading)."""

    def build_unet(self, channels, cond_dim: int, in_ch: int = 4) -> int:
        self.stem = nn.Conv2d(in_ch, channels[0], 3, padding=1)
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.down.append(ResBlock(prev, ch, cond_dim))
            self.pool.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            prev = ch
        self.mid = ResBlock(prev, prev, cond_dim)
        self.up = nn.ModuleList()
        for ch in reversed(channels):
            self.up.append(ResBlock(prev + ch, ch, cond_dim))
            prev = ch
        return prev  # channels of the output features

    def unet(self, x, c, mid_hook=None):
        x = self.stem(x)
        skips = []
        for block, pool in zip(self.down, self.pool):
            x = block(x, c)
            skips.append(x)
            x = pool(x)
        x = self.mid(x, c)
        if mid_hook is not None:
            x = mid_hook(x)
        for block in self.up:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = block(torch.cat([x, skip], 1), c)
        return x


class CondUNet(UNetMixin, nn.Module):
    """A standalone conditioned U-Net (for models that need more than one)."""

    def __init__(self, channels, cond_dim: int, in_ch: int = 4):
        super().__init__()
        self.out_channels = self.build_unet(channels, cond_dim, in_ch)

    def forward(self, x, c, mid_hook=None):
        return self.unet(x, c, mid_hook)


class FrameAttention(nn.Module):
    """Frames of the same animation attend to each other, per spatial position.

    Input is the flat list of real frames [N, C, h, w] with their sample (b) and
    frame (t) indices; padding frames are masked out.
    """

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x, b_idx, t_idx, batch: int, t_max: int):
        N, C, h, w = x.shape
        padded = x.new_zeros(batch, t_max, C, h, w)
        padded[b_idx, t_idx] = x
        seq = padded.permute(0, 3, 4, 1, 2).reshape(batch * h * w, t_max, C)
        present = torch.zeros(batch, t_max, dtype=torch.bool, device=x.device)
        present[b_idx, t_idx] = True
        pad_mask = (~present)[:, None, :].expand(batch, h * w, t_max).reshape(batch * h * w, t_max)
        y = self.norm(seq)
        y, _ = self.attn(y, y, y, key_padding_mask=pad_mask, need_weights=False)
        y = y.reshape(batch, h, w, t_max, C).permute(0, 3, 4, 1, 2)
        return x + y[b_idx, t_idx]
