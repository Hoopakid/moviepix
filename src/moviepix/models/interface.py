"""The contract every moviepix model follows.

Training, evaluation, inference and ONNX export only use this interface, so a
new model is one file plus a registry entry.

Inputs (batched, B = batch):
    reference  float [B, 4, S, S]   the still sprite, RGBA scaled to 0..1
    action     long  [B]            moviepix.actions id
    direction  long  [B]            index into moviepix.actions.DIRECTIONS
    n_frames   long  [B]            frames to produce (<= MAX_FRAMES)
    text       optional             reserved for the later text -> sprite stage

Output:
    frames     float [B, T, 4, S, S] with T = max(n_frames); frames past
               n_frames[i] are padding and ignored by losses and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Batch:
    reference: torch.Tensor
    action: torch.Tensor
    direction: torch.Tensor
    n_frames: torch.Tensor
    target: torch.Tensor | None = None  # [B, T, 4, S, S], training only
    text: list[str] | None = None


class Animator(nn.Module):
    """Base class. Subclasses implement forward() and loss()."""

    size: int  # resolution the model was built for (64 or 32)

    def forward(self, reference, action, direction, n_frames, text=None) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, batch: Batch) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the training loss and a dict of values to log."""
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, reference, action, direction, n_frames, text=None) -> torch.Tensor:
        """Inference entry point; models that sample (diffusion) override this."""
        return self.forward(reference, action, direction, n_frames, text)
