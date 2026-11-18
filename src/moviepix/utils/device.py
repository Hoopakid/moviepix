"""Pick the best available device: Apple MPS, then CUDA, then CPU."""

from __future__ import annotations

# Apple GPUs share memory with the whole Mac. Every moviepix command caps what it
# may take, so a training or evaluation run can never push the Mac into heavy swap
# (an uncapped 64 px evaluation once took ~17 GB). Over the cap a command stops
# with an out-of-memory error instead.
DEFAULT_MPS_FRACTION = 0.6


def best_device(preferred: str | None = None, mps_memory_fraction: float | None = DEFAULT_MPS_FRACTION) -> str:
    import torch

    if preferred and preferred != "auto":
        device = preferred
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if device == "mps" and mps_memory_fraction:
        torch.mps.set_per_process_memory_fraction(mps_memory_fraction)
    return device
