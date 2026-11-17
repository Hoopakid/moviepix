"""Where does training memory go? Peak MPS memory for one training step."""
import argparse
import tomllib

import torch

import moviepix.models  # noqa: F401
from moviepix.data.dataset import TrainSet, collate_groups
from moviepix.models.registry import build

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--chars", type=int, default=2)
ap.add_argument("--per", type=int, default=4)
ap.add_argument("--loss-chunk", type=int)
ap.add_argument("--no-backward", action="store_true")
a = ap.parse_args()
cfg = tomllib.loads(open(a.config).read())
opts = dict(cfg.get("model_options", {}))
if a.loss_chunk:
    opts["loss_chunk"] = a.loss_chunk
ds = TrainSet(f"data/built/{cfg['dataset']}", per_character=a.per, seed=0)
# worst case: characters with the longest actions
groups = [ds[i] for i in range(a.chars)]
b = collate_groups(groups)
dev = "mps"
b = type(b)(*(x.to(dev) if torch.is_tensor(x) else x for x in (b.reference, b.action, b.direction, b.n_frames, b.target, b.text)))
m = build(cfg["model"], size=ds.size, **opts).to(dev)
torch.mps.empty_cache()
base = torch.mps.current_allocated_memory()
with torch.autocast(dev, dtype=torch.bfloat16):
    loss, _ = m.loss(b)
fwd = torch.mps.current_allocated_memory()
if not a.no_backward:
    loss.backward()
torch.mps.synchronize()
peak = torch.mps.driver_allocated_memory()
print(f"frames {int(b.n_frames.sum())}  K~{int((b.reference[:,3]>0.5).flatten(1).sum(1).max())}  "
      f"after forward {(fwd-base)/1e9:.2f} GB  driver peak {peak/1e9:.2f} GB")
