"""Out-of-style walk test: how do models do on pixel art they never saw?

    uv run python scripts/oos_walk.py RUN [RUN ...] [--device cpu]

Packs (CC0 / CC-BY, see docs/research/wave2-oos-packs.md) are placed into the model's
canvas by integer nearest-neighbour scaling with the feet on LPC's pivot. Frame counts
differ from LPC (4-step walks vs 9 frames), so scoring is frame-count free: for every
truth walk frame that differs from the still reference, take the best-matching
predicted frame and measure exact pixels (truth-opaque pixels whose predicted colour,
snapped to the reference palette, matches) and silhouette IoU. The baseline repeats the
still reference: a model must beat it to show real motion.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from moviepix import actions as A
from moviepix.data.atlas import load_png, unpack
from moviepix.train.checkpoint import load_model
from moviepix.utils.io import read_jsonl

OOS = Path("data/raw/ooS")
DIRS = ("up", "left", "down", "right")  # moviepix direction order
PIVOT = {64: (32, 61), 32: (16, 30)}  # (feet centre x, feet row y) of LPC bodies
SCALE = {64: {"puny": 2, "ninja": 2, "knight": 1, "ranger": 1, "antifarea": 2},
         32: {"puny": 1, "ninja": 1, "knight": 1, "ranger": 1, "antifarea": 1}}


def rgba(path):
    im = Image.open(path)
    return np.asarray(im.convert("RGBA")).copy()


def crop(img, x, y, w, h):
    return img[y:y + h, x:x + w]


def colour_key(img, key):
    out = img.copy()
    out[np.all(out[..., :3] == key, -1)] = 0
    return out


def walks():
    """Yield (pack, character, direction, reference cell, [walk cells])."""
    root = OOS / "puny-characters/x/Puny-Characters"
    for sheet in ["Archer-Green", "Archer-Purple", "Character-Base", "Mage-Cyan", "Mage-Red", "Soldier-Blue",
                  "Soldier-Red", "Soldier-Yellow", "Warrior-Blue", "Warrior-Red"]:
        img = rgba(root / f"{sheet}.png")
        for d, row in {"down": 0, "right": 2, "up": 4, "left": 6}.items():
            cell = lambda c: crop(img, 32 * c, 32 * row, 32, 32)  # noqa: E731
            yield "puny", sheet, d, cell(0), [cell(c) for c in (0, 2, 0, 3)]
    for f in sorted((OOS / "ninja-adventure-superpowers").glob("*.png"), key=lambda p: p.name):
        img = rgba(f)
        if img.shape != (112, 64, 4):
            continue
        for d, col in {"down": 0, "up": 1, "left": 2, "right": 3}.items():
            cell = lambda r: crop(img, 16 * col, 16 * r, 16, 16)  # noqa: E731
            yield "ninja", f.stem, d, cell(0), [cell(r) for r in range(4)]
    D = {"up": "NORTH", "left": "WEST", "down": "SOUTH", "right": "EAST"}
    kn = OOS / "chasersgaming-knight-nes/x/NES Knight Free Version"
    ra = OOS / "chasersgaming-ranger-nes/x/NES Ranger Files FREE/NES Ranger Free Set PNG/NES Ranger Single Sheets"
    for d, name in D.items():
        idle = rgba(kn / f"Idle PNG/NES PNG Knight_Idle_{name}_strip4.png")
        walk = rgba(kn / f"Walk PNG/NES PNG Knight_Walk_{name}_strip4.png")
        yield "knight", "knight", d, crop(idle, 0, 0, 16, 24), [crop(walk, 16 * i, 0, 16, 24) for i in range(4)]
        idle = rgba(ra / f"Idle/NES_Ranger_Idle_{name}.png")
        walk = rgba(ra / f"Walk/NES_Ranger_Walk_{name}.png")
        yield "ranger", "ranger", d, crop(idle, 0, 0, 16, 24), [crop(walk, 16 * i, 0, 16, 24) for i in range(4)]
    img = rgba(OOS / "antifarea-twelve/charsets_12_m-f_complete_by_antifarea_0.png")
    classes = ["warrior", "magician", "healer", "ninja", "ranger", "townfolk"]
    for blk, (x0, y0, key) in {"m": (16, 180, (123, 213, 254)), "f": (16, 306, (246, 139, 205))}.items():
        for ci, cls in enumerate(classes):
            for d, row in {"up": 0, "right": 1, "down": 2, "left": 3}.items():
                cell = lambda c: colour_key(crop(img, x0 + 16 * (3 * ci + c), y0 + 18 * row, 16, 18), key)  # noqa: E731
                yield "antifarea", f"{cls}-{blk}", d, cell(1), [cell(c) for c in (1, 0, 1, 2)]


def place(cells, ref, S, k):
    """Scale by k and place all cells with one offset that puts the reference's feet on the pivot."""
    a = ref[..., 3] > 0
    ys, xs = np.nonzero(a)
    feet_y, cx = ys.max(), (xs.min() + xs.max() + 1) / 2
    px, py = PIVOT[S]
    ox, oy = int(round(px - cx * k)), int(py - (feet_y * k + k - 1))
    out = []
    for c in cells:
        big = c.repeat(k, 0).repeat(k, 1)
        canvas = np.zeros((S, S, 4), np.uint8)
        h, w = big.shape[:2]
        y0, x0 = max(oy, 0), max(ox, 0)
        sy, sx = y0 - oy, x0 - ox
        hh, ww = min(h - sy, S - y0), min(w - sx, S - x0)
        if hh > 0 and ww > 0:
            canvas[y0:y0 + hh, x0:x0 + ww] = big[sy:sy + hh, sx:sx + ww]
        canvas[canvas[..., 3] < 128] = 0
        canvas[..., 3] = np.where(canvas[..., 3] > 0, 255, 0)
        out.append(canvas)
    return out


def score(pred, truth, ref):
    """exact (snapped to reference palette) and IoU for one predicted vs truth frame."""
    pal = np.unique(ref[ref[..., 3] > 0][:, :3], axis=0).astype(np.int32)
    po, to = pred[..., 3] > 0, truth[..., 3] > 0
    p = pred[..., :3].astype(np.int32)
    d = np.abs(p[:, :, None, :] - pal[None, None]).sum(-1)
    snapped = pal[d.argmin(-1)]
    ok = to & po & np.all(snapped == truth[..., :3], -1)
    exact = ok.sum() / max(to.sum(), 1)
    iou = (po & to).sum() / max((po | to).sum(), 1)
    return exact, iou


def to_t(x):
    return torch.from_numpy(x).permute(2, 0, 1)[None].float() / 255


def evaluate(run, device, samples_by_size, lpc_samples):
    model, state = load_model(run / "best.pt", device)
    S = model.size
    walk = A.get("walk")
    res = {}
    for name, samples in list(samples_by_size[S].items()) + [("LPC (in-style)", lpc_samples[S])]:
        m_ex, b_ex, m_iou, b_iou, n = 0.0, 0.0, 0.0, 0.0, 0
        for ref, truth_frames, d in samples:
            with torch.no_grad():
                pred = model.generate(to_t(ref).to(device), torch.tensor([walk.id], device=device),
                                      torch.tensor([DIRS.index(d)], device=device), torch.tensor([walk.frames], device=device),
                                      steps=1)[0]
            pf = (pred.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
            pf[..., 3] = np.where(pf[..., 3] > 127, 255, 0)
            for tf in truth_frames:
                if np.array_equal(tf, ref):
                    continue  # standing frames: copying the reference is trivially right
                best = max((score(p, tf, ref) for p in pf), key=lambda s: s[0])
                base = score(ref, tf, ref)
                m_ex += best[0]; m_iou += best[1]; b_ex += base[0]; b_iou += base[1]; n += 1
        if n:
            res[name] = {"frames": n, "model_exact": m_ex / n, "baseline_exact": b_ex / n,
                         "model_iou": m_iou / n, "baseline_iou": b_iou / n}
    return state, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lpc-characters", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("docs/research/oos-walk-results.json"))
    a = ap.parse_args()
    torch.set_num_threads(4)
    samples = {}
    for S in (32, 64):
        samples[S] = {}
        for pack, char, d, ref, cells in walks():
            k = SCALE[S][pack]
            placed = place([ref] + cells, ref, S, k)
            samples[S].setdefault(pack, []).append((placed[0], placed[1:], d))
    lpc = {}
    for S, ds in ((32, "lpc32v2"), (64, "lpc64v2")):
        entries = [e for e in read_jsonl(Path("data/built") / ds / "manifest.jsonl") if e["split"] == "validation"]
        lpc[S] = []
        for e in entries[: a.lpc_characters]:
            sh = unpack(load_png(Path("data/built") / ds / e["atlas"]), e["layout"], S)["walk"]
            for r, d in enumerate(DIRS):
                lpc[S].append((sh[r, 0], list(sh[r, 1:]), d))
    out = {}
    for run in a.runs:
        state, res = evaluate(run, a.device, samples, lpc)
        out[run.name] = res
        print(f"\n{run.name}  (model trained on {json.loads((run / 'recipe.json').read_text())['dataset']})")
        print(f"{'pack':18s} {'frames':>6s} {'exact':>7s} {'base':>7s} {'gain':>7s} {'iou':>6s} {'base':>6s}")
        for k, v in res.items():
            print(f"{k:18s} {v['frames']:6d} {v['model_exact']:7.3f} {v['baseline_exact']:7.3f} "
                  f"{v['model_exact'] - v['baseline_exact']:+7.3f} {v['model_iou']:6.3f} {v['baseline_iou']:6.3f}", flush=True)
    a.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    sys.exit(main())
