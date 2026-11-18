"""Export copy_animator to ONNX (prototype, milestone 5).

    moviepix-export data/runs/<run> --out model.onnx [--frames 8]

The exported graph does one-shot decoding for ONE sprite:
    inputs   reference  float32 [1, 4, S, S]  RGBA in 0..1
             action     int64   [1]
             direction  int64   [1]
             frame_ids  int64   [T]           0..T-1 (T = the action's frame count; dynamic)
    output   frames     float32 [T, 4, S, S]

Export-friendly differences from the PyTorch path (same results):
- copy attention runs over all S*S reference pixels with transparent ones masked,
  instead of compacting them (data-dependent shapes do not export well);
- frame attention is written for a single sprite, so no padding/scatter is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from moviepix.models.copy_animator import NEW, TRANSPARENT


class OneShotExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

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

    def forward(self, reference, action, direction, frame_ids):
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
        copy = copy.masked_fill(~valid[None, None, :], -1e9)
        special = m.special(dec).flatten(2).transpose(1, 2)  # [T, P, 2]
        new_rgb = torch.sigmoid(m.new_rgb(dec)).flatten(2).transpose(1, 2)  # [T, P, 3]

        best_copy, idx = copy.max(-1)  # [T, P]
        rgb = colors[idx]  # [T, P, 3]
        if m.alpha is not None:
            drawn = m.alpha(dec).flatten(1) > 0  # [T, P]
            use_new = special[..., NEW] > best_copy
        else:
            best_special, which = special.max(-1)
            drawn = ~((best_special > best_copy) & (which == TRANSPARENT))
            use_new = (best_special > best_copy) & (which == NEW)
        rgb = torch.where(use_new[..., None], new_rgb, rgb)
        alpha = drawn.float()[..., None]
        return torch.cat([rgb * alpha, alpha], -1).transpose(1, 2).reshape(T, 4, S, S)


def export(run: Path, out: Path, frames: int = 8, opset: int = 17) -> Path:
    from moviepix.train.checkpoint import load_model

    model, _ = load_model(run / "best.pt", "cpu")
    wrapper = OneShotExport(model).eval()
    S = model.size
    args = (torch.rand(1, 4, S, S).round(), torch.tensor([0]), torch.tensor([2]), torch.arange(frames))
    torch.onnx.export(wrapper, args, str(out), input_names=["reference", "action", "direction", "frame_ids"],
                      output_names=["frames"], dynamic_axes={"frame_ids": {0: "T"}, "frames": {0: "T"}},
                      opset_version=opset, dynamo=False)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args(argv)
    print(export(args.run, args.out, args.frames))


if __name__ == "__main__":
    main()
