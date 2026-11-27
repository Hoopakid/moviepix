"""Smaller copies of model.onnx (Phase 3 step 2), each kept only if it decides like fp32.

    python -m moviepix.export.quantize data/runs/<run> --dir web/model [--examples 256]

Writes model-fp16.onnx (weights and maths in float16, inputs/outputs float32) and model-int8.onnx
(weights int8, dynamic quantization of MatMul and Conv), then scores fp32, fp16 and int8 on the
same validation examples. A copy ships only if its exact is within 0.005 of fp32 (phase3-design.md).
Prints a table and writes compare.json next to the models.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from moviepix.export.onnx import _validation, onnx_inputs, to_uint8, write_selftest

TOLERANCE = 0.005


def make_fp16(src: Path, dst: Path) -> Path:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    onnx.save(model, str(dst))
    return dst


def make_int8(src: Path, dst: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Conv"])
    return dst


def compare(run: Path, d: Path, names: list[str], n: int, threads: int = 4) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sessions = {k: ort.InferenceSession(str(d / f"{k}.onnx"), opts, providers=["CPUExecutionProvider"]) for k in names}
    colours = json.loads((d / "actions.json").read_text())["max_colours"]
    exact = {k: [] for k in names}
    same = {k: [] for k in names}
    ms = {k: [] for k in names}
    for ref, action, direction, truth in _validation(run, n):
        feeds = onnx_inputs(ref, action, direction, colours)
        drawn = truth[..., 3] > 0
        base = None
        for k, s in sessions.items():
            t0 = time.perf_counter()
            o = to_uint8(s.run(None, feeds)[0])
            ms[k].append(1000 * (time.perf_counter() - t0))
            base = o if base is None else base
            exact[k].append(((o == truth).all(-1) & drawn).sum() / max(drawn.sum(), 1))
            same[k].append((o == base).all(-1).mean())
    ref_exact = float(np.mean(exact[names[0]]))
    return {k: {"MB": round((d / f"{k}.onnx").stat().st_size / 1e6, 1), "exact": round(float(np.mean(exact[k])), 4),
                "pixels_same_as_fp32": round(float(np.mean(same[k])), 5),
                "cpu_ms_median": round(float(np.median(ms[k][1:])), 1),
                "ships": bool(ref_exact - float(np.mean(exact[k])) <= TOLERANCE)} for k in names}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--dir", type=Path, default=Path("web/model"))
    ap.add_argument("--examples", type=int, default=256)
    a = ap.parse_args(argv)
    make_fp16(a.dir / "model.onnx", a.dir / "model-fp16.onnx")
    make_int8(a.dir / "model.onnx", a.dir / "model-int8.onnx")
    res = compare(a.run, a.dir, ["model", "model-fp16", "model-int8"], a.examples)
    (a.dir / "compare.json").write_text(json.dumps(res, indent=1))
    print(f"{'file':12s} {'MB':>5s} {'exact':>7s} {'same':>8s} {'ms':>6s}  ships")
    for k, r in res.items():
        print(f"{k:12s} {r['MB']:5.1f} {r['exact']:7.4f} {r['pixels_same_as_fp32']:8.5f} {r['cpu_ms_median']:6.1f}  {r['ships']}")
        if not r["ships"]:
            (a.dir / f"{k}.onnx").unlink()  # the page only offers files that decide like fp32
    print(write_selftest(a.run, a.dir))


if __name__ == "__main__":
    main()
