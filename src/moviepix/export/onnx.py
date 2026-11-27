"""Export copy_animator to ONNX for the browser demo (Phase 3, docs/research/phase3-design.md).

    moviepix-export data/runs/<run> --out-dir web/model [--colours 64] [--verify 256]

Writes into --out-dir:
    model.onnx     one-shot decoding for ONE sprite
    actions.json   size, placement pivot, colour limit, and every action (frames, directions, loops)
    selftest.json  one reference and the frames this model.onnx gives for it, for the page's self-test

The graph:
    inputs   reference  float32 [1, 4, S, S]  RGBA in 0..1, alpha 0 or 1
             action     int64   [1]
             direction  int64   [1]           index into actions.DIRECTIONS
             frame_ids  int64   [T]           0..T-1, T = the action's frame count (dynamic)
             colour_id  int64   [S*S]         colour number per reference pixel, -1 where transparent
                                              (colour_ids() below; the page computes the same)
    output   frames     float32 [T, 4, S, S]

It decides by COLOUR, like CopyAnimator.generate(steps=1, aggregate=True): the copy probabilities of
all reference pixels with the same colour are summed, then the best colour (or "new colour") wins.
Export-friendly differences from the PyTorch path (same results):
- copy attention runs over all S*S reference pixels with transparent ones masked,
  instead of compacting them (data-dependent shapes do not export well);
- colours are numbered by the caller (colour_id) and summed with a one-hot matmul, so the
  graph has no data-dependent unique(); at most `colours` colours per sprite;
- frame attention is written for a single sprite, so no padding/scatter is needed.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from moviepix import actions as A
from moviepix.models.copy_animator import NEW, TRANSPARENT


def colour_ids(reference: np.ndarray, limit: int) -> np.ndarray:
    """uint8 [S, S, 4] -> int64 [S*S]: colours numbered by first appearance (row-major), -1 transparent."""
    px = reference.reshape(-1, 4)
    out = np.full(len(px), -1, np.int64)
    seen: dict[tuple, int] = {}
    for i, p in enumerate(px):
        if p[3] >= 128:
            out[i] = seen.setdefault(tuple(int(c) for c in p[:3]), len(seen))
    if len(seen) > limit:
        raise ValueError(f"sprite has {len(seen)} colours; the model takes at most {limit}")
    return out


class OneShotExport(nn.Module):
    def __init__(self, model, colours: int = 64):
        super().__init__()
        self.m = model
        self.colours = colours

    def _frame_attention(self, h):
        """nn.MultiheadAttention written out by hand: the built-in module bakes the
        traced frame count into its reshapes, which breaks a dynamic frame axis."""
        fa = self.m.frame_attn
        attn = fa.attn
        _, C, hh, ww = h.shape
        H = attn.num_heads
        seq = h.permute(2, 3, 0, 1).flatten(0, 1)  # [L, T, C], L = hh*ww
        y = fa.norm(seq)
        qkv = torch.nn.functional.linear(y, attn.in_proj_weight, attn.in_proj_bias)  # [L, T, 3C]
        q, k, v = qkv.chunk(3, -1)
        split = lambda z: z.unflatten(-1, (H, C // H)).transpose(1, 2)  # noqa: E731  [L, H, T, Dh]
        q, k, v = split(q), split(k), split(v)
        w = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (C // H) ** -0.5, -1)
        o = torch.matmul(w, v).transpose(1, 2).flatten(2)  # [L, T, C]
        o = attn.out_proj(o)
        return h + o.unflatten(0, (hh, ww)).permute(2, 3, 0, 1)

    def forward(self, reference, action, direction, frame_ids, colour_id):
        m = self.m
        T = frame_ids.shape[0]
        S = reference.shape[-1]
        zeros_t = torch.zeros_like(frame_ids)
        n_frames = zeros_t + T
        cond = m.cond(action[zeros_t], direction[zeros_t], frame_ids, n_frames)

        # reference keys over all pixels; transparent ones masked out
        feat = m.ref_unet(reference, m.ref_cond[None])
        keys = (m.to_key(feat) + m.pos).flatten(2).transpose(1, 2)[0]  # [P, D]
        valid = reference[0, 3].flatten() > 0.5  # [P]
        colors = reference[0, :3].flatten(1).transpose(0, 1)  # [P, 3]

        ref_t = reference.expand(T, -1, -1, -1)
        x = torch.cat([ref_t, torch.zeros_like(ref_t), torch.zeros_like(ref_t[:, :1])], 1)
        dec = m.dec_norm(m.dec_unet(x, cond, mid_hook=self._frame_attention))
        q = (m.to_query(dec) + m.pos).flatten(2).transpose(1, 2)  # [T, P, D]
        copy = torch.matmul(q, keys.transpose(0, 1)) * m.scale  # [T, P, P]
        # -1e4, not -inf: WebGPU shaders do not handle infinities, and -1e4 fits in fp16; after the
        # softmax both give exactly 0 in float32, so the result is the same as PyTorch's -inf.
        copy = copy.masked_fill(~valid[None, None, :], -1e4)
        special = m.special(dec).flatten(2).transpose(1, 2)  # [T, P, 2]
        new_rgb = torch.sigmoid(m.new_rgb(dec)).flatten(2).transpose(1, 2)  # [T, P, 3]

        # probabilities as in CopyAnimator._choose, then summed per colour
        if m.alpha is not None:  # transparent is decided by the alpha head, not by this choice
            probs = torch.softmax(torch.cat([special[..., NEW:NEW + 1], copy], -1), -1)
            p_special = torch.cat([torch.zeros_like(probs[..., :1]), probs[..., :1]], -1)
            p_copy = probs[..., 1:]
        else:
            probs = torch.softmax(torch.cat([special, copy], -1), -1)
            p_special, p_copy = probs[..., :2], probs[..., 2:]
        onehot = ((colour_id[:, None] == torch.arange(self.colours)[None, :]) & valid[:, None]).float()  # [P, K]
        classes = torch.cat([p_special, torch.matmul(p_copy, onehot)], -1)  # [T, P, 2 + K]
        pick = classes.argmax(-1)  # [T, P]
        first = onehot.argmax(0)  # [K] first reference pixel of each colour
        rgb = colors[first][(pick - 2).clamp(min=0)]  # [T, P, 3]
        rgb = torch.where((pick == NEW)[..., None], new_rgb, rgb)
        if m.alpha is not None:
            drawn = m.alpha(dec).flatten(1) > 0  # sigmoid > 0.5
        else:
            drawn = pick != TRANSPARENT
        alpha = drawn.float()[..., None]
        return torch.cat([rgb * alpha, alpha], -1).transpose(1, 2).reshape(T, 4, S, S)


def onnx_inputs(reference: np.ndarray, action: str, direction: str, colours: int) -> dict:
    """uint8 [S, S, 4] reference + names -> the graph's inputs (numpy)."""
    act = A.get(action)
    return {"reference": (reference.astype(np.float32) / 255).transpose(2, 0, 1)[None],
            "action": np.array([act.id], np.int64),
            "direction": np.array([A.DIRECTIONS.index(direction)], np.int64),
            "frame_ids": np.arange(act.frames, dtype=np.int64),
            "colour_id": colour_ids(reference, colours)}


def to_uint8(x: np.ndarray) -> np.ndarray:
    """float [T, 4, S, S] in 0..1 -> uint8 [T, S, S, 4]."""
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8).transpose(0, 2, 3, 1)


def actions_info(size: int, colours: int) -> dict:
    return {
        "size": size,
        "pivot": [size // 2, size - 2],  # feet of the reference go here (data/packs.py)
        "max_colours": colours,
        "directions": list(A.DIRECTIONS),
        "actions": [{"id": a.id, "name": a.name, "frames": a.frames, "directions": list(a.directions),
                     "loops": a.loops,
                     # LPC walks start with the standing pose, which is the reference itself
                     "drop_first": a.name == "walk"} for a in A.ACTIONS],
    }


def export(run: Path, out_dir: Path, colours: int = 64, frames: int = 8, opset: int = 17) -> Path:
    from moviepix.train.checkpoint import load_model

    model, _ = load_model(run / "best.pt", "cpu")
    wrapper = OneShotExport(model, colours).eval()
    S = model.size
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "model.onnx"
    ref = torch.rand(1, 4, S, S).round()
    ids = torch.where(ref[0, 3].flatten() > 0.5, torch.randint(0, colours, (S * S,)), -1)
    args = (ref, torch.tensor([0]), torch.tensor([2]), torch.arange(frames), ids)
    torch.onnx.export(wrapper, args, str(out), input_names=["reference", "action", "direction", "frame_ids", "colour_id"],
                      output_names=["frames"], dynamic_axes={"frame_ids": {0: "T"}, "frames": {0: "T"}},
                      opset_version=opset, dynamo=False)
    (out_dir / "actions.json").write_text(json.dumps(actions_info(S, colours), indent=1))
    return out


def _validation(run: Path, n: int):
    """(reference uint8 [S,S,4], action name, direction, truth uint8 [T,S,S,4]) from the run's validation split."""
    import tomllib

    from moviepix.data.dataset import EvalSet

    cfg = tomllib.loads((run / "config.toml").read_text())
    ds = EvalSet(Path("data/built") / cfg["dataset"], "validation", characters=max(1, n // 4), per_character=4, seed=0)
    for i in range(min(n, len(ds))):
        e = ds[i]
        name = A.ACTIONS[e["action"]].name
        yield e["reference"], name, A.DIRECTIONS[e["direction"]], e["target"][:e["n_frames"]]


MODEL_FILES = ("model.onnx", "model-fp16.onnx", "model-int8.onnx")


def write_selftest(run: Path, out_dir: Path) -> Path:
    """One validation reference and the frames each model file in out_dir makes for it, by native
    onnxruntime on the CPU; the page re-runs it with the file it loaded and compares."""
    import onnxruntime as ort

    info = json.loads((out_dir / "actions.json").read_text())
    ref, _, _, _ = next(_validation(run, 1))
    feeds = onnx_inputs(ref, "walk", "down", info["max_colours"])
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()  # noqa: E731
    expected = {}
    for name in MODEL_FILES:
        if (out_dir / name).exists():
            sess = ort.InferenceSession(str(out_dir / name), providers=["CPUExecutionProvider"])
            expected[name] = b64(to_uint8(sess.run(None, feeds)[0]))
    out = out_dir / "selftest.json"
    out.write_text(json.dumps({"action": "walk", "direction": "down", "size": info["size"],
                               "frames": int(len(feeds["frame_ids"])), "reference_rgba": b64(ref),
                               "expected_rgba": expected}))
    return out


def verify(run: Path, out_dir: Path, n: int = 256, threads: int = 4) -> dict:
    """model.onnx against PyTorch (one-shot, colour decoding) on n validation examples."""
    import onnxruntime as ort

    from moviepix.train.checkpoint import load_model

    torch.set_num_threads(threads)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(out_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    colours = json.loads((out_dir / "actions.json").read_text())["max_colours"]
    model, _ = load_model(run / "best.pt", "cpu")
    same, whole, ex_onnx, ex_torch, count = [], [], [], [], 0
    for ref, action, d, truth in _validation(run, n):
        feeds = onnx_inputs(ref, action, d, colours)
        o = to_uint8(sess.run(None, feeds)[0])
        with torch.no_grad():
            t = model.generate(torch.from_numpy(feeds["reference"]), torch.from_numpy(feeds["action"]),
                               torch.from_numpy(feeds["direction"]), torch.tensor([len(truth)]), steps=1,
                               aggregate=True)[0]
        t = to_uint8(t.numpy())
        eq = (o == t).all(-1)
        same.append(eq.mean())
        whole.append(eq.all())
        drawn = truth[..., 3] > 0  # exact on the character's pixels, as eval.metrics
        ex_onnx.append(((o == truth).all(-1) & drawn).sum() / max(drawn.sum(), 1))
        ex_torch.append(((t == truth).all(-1) & drawn).sum() / max(drawn.sum(), 1))
        count += 1
    return {"examples": count, "pixels_same": float(np.mean(same)), "examples_identical": float(np.mean(whole)),
            "exact_onnx": float(np.mean(ex_onnx)), "exact_torch": float(np.mean(ex_torch))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--colours", type=int, default=64, help="most colours a sprite may have")
    ap.add_argument("--verify", type=int, default=0, help="compare with PyTorch on this many validation examples")
    args = ap.parse_args(argv)
    out = export(args.run, args.out_dir, args.colours)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")
    print(write_selftest(args.run, args.out_dir))
    if args.verify:
        print(json.dumps(verify(args.run, args.out_dir, args.verify), indent=1))


if __name__ == "__main__":
    main()
