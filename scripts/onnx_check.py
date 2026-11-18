"""Export a run to ONNX and check it matches PyTorch one-shot decoding on real sprites."""
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from moviepix import actions as A
from moviepix.data.dataset import EvalSet
from moviepix.export.onnx import export
from moviepix.train.checkpoint import load_model

torch.set_num_threads(2)
run = Path(sys.argv[1])
out = Path("data/export") / f"{run.name}.onnx"
t0 = time.time()
export(run, out)
print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
model, _ = load_model(run / "best.pt", "cpu")
cfg = tomllib.loads((run / "config.toml").read_text())
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
sess = ort.InferenceSession(str(out), opts, providers=["CPUExecutionProvider"])
ds = EvalSet(Path("data/built") / cfg["dataset"], "validation", characters=12, per_character=3, seed=5)
agree, total, worst, times = 0, 0, 1.0, []
for i in range(len(ds)):
    ex = ds[i]
    ref = torch.from_numpy(ex["reference"]).permute(2, 0, 1)[None].float() / 255
    n = ex["n_frames"]
    with torch.no_grad():
        pt = model.generate(ref, torch.tensor([ex["action"]]), torch.tensor([ex["direction"]]), torch.tensor([n]), steps=1)[0, :n]
    t1 = time.time()
    ox = sess.run(None, {"reference": ref.numpy(), "action": np.array([ex["action"]]), "direction": np.array([ex["direction"]]),
                         "frame_ids": np.arange(n)})[0]
    times.append((time.time() - t1) * 1000 * 8 / n)
    same = (np.abs(ox - pt.numpy()).max(1) < 1e-3)  # per pixel
    agree += int(same.sum())
    total += same.size
    worst = min(worst, float(same.mean()))
print(f"pixels identical to PyTorch: {agree / total:.4%} (worst example {worst:.4%})")
print(f"onnxruntime CPU, 4 threads, per 8 frames: median {np.median(times):.1f} ms")
