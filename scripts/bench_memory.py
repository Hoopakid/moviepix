"""Peak GPU memory and speed of training steps on real data (MPS)."""
import argparse
import time

import torch

import moviepix.models  # noqa: F401
from moviepix.data.dataset import EvalSet, collate
from moviepix.models.registry import build

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="copy_animator")
ap.add_argument("--data", default="data/built/lpc32")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--steps", type=int, default=8)
a = ap.parse_args()
ds = EvalSet(a.data, "train", characters=a.batch * 3, per_character=1, seed=0)
batches = [collate([ds[i] for i in range(j, j + a.batch)]) for j in range(0, a.batch * 3, a.batch)]
size = ds.size
m = build(a.model, size=size).to("mps")
opt = torch.optim.AdamW(m.parameters(), 3e-4)
peak = 0
for i in range(a.steps):
    b = batches[i % len(batches)]
    b = type(b)(*(x.to("mps") if torch.is_tensor(x) else x for x in (b.reference, b.action, b.direction, b.n_frames, b.target, b.text)))
    if i == 2:
        torch.mps.synchronize(); t = time.time()
    with torch.autocast("mps", dtype=torch.bfloat16):
        loss, _ = m.loss(b)
    opt.zero_grad(); loss.backward(); opt.step()
    torch.mps.synchronize()
    peak = max(peak, torch.mps.driver_allocated_memory())
dt = (time.time() - t) / (a.steps - 2)
print(f"{a.model} {size}px batch {a.batch}: peak GPU memory {peak/1e9:.2f} GB, {1/dt:.2f} step/s, {a.batch/dt:.0f} examples/s")
