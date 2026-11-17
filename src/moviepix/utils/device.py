"""Pick the best available device: Apple MPS, then CUDA, then CPU."""

from __future__ import annotations


def best_device(preferred: str | None = None) -> str:
    import torch

    if preferred and preferred != "auto":
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
