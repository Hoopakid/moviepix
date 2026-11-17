"""Measure training step speed of a model on a device with random data."""
import argparse
import time

import torch

import moviepix.models.unet_baseline  # noqa: F401
from moviepix.models.interface import Batch
from moviepix.models.registry import build

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unet_baseline")
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--device", default="mps")
ap.add_argument("--amp", choices=["none", "fp16", "bf16"], default="none")
ap.add_argument("--channels-last", action="store_true")
a = ap.parse_args()
dev, S, B = a.device, a.size, a.batch
m = build(a.model, size=S).to(dev)
if a.channels_last:
    m = m.to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(m.parameters(), 1e-3)
g = torch.Generator().manual_seed(0)
batch = Batch(torch.rand(B, 4, S, S, generator=g).to(dev), torch.randint(0, 15, (B,), generator=g).to(dev),
              torch.randint(0, 4, (B,), generator=g).to(dev), torch.randint(2, 14, (B,), generator=g).to(dev),
              torch.rand(B, 13, 4, S, S, generator=g).round().to(dev))
dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(a.amp)
sync = torch.mps.synchronize if dev == "mps" else (torch.cuda.synchronize if dev == "cuda" else (lambda: None))
for i in range(13):
    if i == 3:
        sync(); t = time.time()
    with torch.autocast(dev, dtype=dtype, enabled=dtype is not None):
        loss, _ = m.loss(batch)
    opt.zero_grad(); loss.backward(); opt.step()
sync()
dt = (time.time() - t) / 10
print(f"{a.model} {S}px batch {B} amp={a.amp} cl={a.channels_last}: {1/dt:.2f} step/s, {int(batch.n_frames.sum())/dt:.0f} frames/s")
