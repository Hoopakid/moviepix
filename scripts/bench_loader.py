"""Measure data loading speed (examples per second)."""
import sys
import time

from torch.utils.data import DataLoader

from moviepix.data.dataset import EvalSet, TrainSet, collate, collate_groups

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/built/lpc64"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ds = TrainSet(root, per_character=8)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=workers, collate_fn=collate_groups)
    it = iter(dl)
    b = next(it)
    print(b.reference.shape, b.target.shape, b.action[:8].tolist(), b.direction[:8].tolist(), b.n_frames[:8].tolist())
    t, n = time.time(), 0
    for i, b in enumerate(it):
        n += len(b.action)
        if i == 40:
            break
    print(f"train examples/s: {n / (time.time() - t):.0f} with {workers} workers")
    ev = EvalSet(root, "validation", characters=128, per_character=4)
    t = time.time()
    k = sum(len(b.action) for b in DataLoader(ev, batch_size=64, collate_fn=collate))
    print(f"eval examples/s: {k / (time.time() - t):.0f}")
