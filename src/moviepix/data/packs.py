"""Shared code for building datasets from hand-downloaded sprite packs (scripts/build_packs32.py,
scripts/build_final32.py): reading sheets, placing characters on the 32 px canvas, writing the
LPC-style atlas dataset."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from moviepix import actions as A
from moviepix.data.atlas import encode_png, pack
from moviepix.data.resize import downscale
from moviepix.utils.io import git_commit, write_json

S, PIVOT = 32, (16, 30)
NUDGE = 2  # px a character may move off the pivot to fit at 1x
DIRS = A.DIRECTIONS  # up, left, down, right


def rgba(path) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("RGBA")).copy()
    a[a[..., 3] < 128] = 0
    a[..., 3] = np.where(a[..., 3] > 0, 255, 0)
    return a


def cells(img: np.ndarray, w: int, h: int, row: int = 0, n: int | None = None, col0: int = 0) -> np.ndarray:
    """Frames of one sheet row -> [n, h, w, 4]."""
    n = n if n is not None else img.shape[1] // w - col0
    return np.stack([img[row * h:(row + 1) * h, (col0 + i) * w:(col0 + i + 1) * w] for i in range(n)])


def over(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    out = base.copy()
    m = top[..., 3] > 0
    out[m] = top[m]
    return out


# ---------------------------------------------------------------- placement
def place(raw: dict, ref_dir: str = "down") -> tuple[dict, int, list] | None:
    """raw[action] = [dirs, frames, h, w, 4] native cells -> (sheets on the 32 canvas, scale divisor,
    dropped actions). 1x is kept when the reference and the locomotion (walk or run) fit; other
    actions that do not fit (a jump, a wide hurt) are dropped rather than shrinking the character."""
    moves = [a for a in ("walk", "run") if a in raw and raw[a].shape[1] > 1]
    for k in (1, 2):
        sh = raw if k == 1 else {a: downscale(v, 2) for a, v in raw.items()}
        ref = sh["walk"][DIRS.index(ref_dir), 0]
        ys, xs = np.nonzero(ref[..., 3])
        oy = PIVOT[1] - ys.max()
        ox = int(round(PIVOT[0] - (xs.min() + xs.max() + 1) / 2))
        # A walk that dips a pixel or two below the standing feet (or reaches past a side) is nudged
        # by at most NUDGE px so it fits, rather than shrinking the whole character.
        core = [sh[a] for a in ["walk", *moves]]
        yy = np.concatenate([np.nonzero(v[..., 3].reshape(-1, *v.shape[2:4]).any(0))[0] for v in core]) + oy
        xx = np.concatenate([np.nonzero(v[..., 3].reshape(-1, *v.shape[2:4]).any(0))[1] for v in core]) + ox
        for lo, hi, axis in ((yy.min(), yy.max(), "y"), (xx.min(), xx.max(), "x")):
            shift = (S - 1 - hi) if hi > S - 1 else (-lo if lo < 0 else 0)
            if 0 < abs(shift) <= NUDGE and lo + shift >= 0 and hi + shift <= S - 1:
                if axis == "y":
                    oy += shift
                else:
                    ox += shift
        out = {}
        for a, v in sh.items():
            h, w = v.shape[2:4]
            occ = v[..., 3].reshape(-1, h, w).any(0)
            if a == "walk" and v.shape[1] == 1:
                occ = v[DIRS.index(ref_dir), 0, ..., 3] > 0 if not moves else occ
            yy, xx = np.nonzero(occ)
            if yy.min() + oy < 0 or yy.max() + oy >= S or xx.min() + ox < 0 or xx.max() + ox >= S:
                continue
            canvas = np.zeros((*v.shape[:2], S, S, 4), np.uint8)
            y0, x0 = max(oy, 0), max(ox, 0)
            sy, sx = y0 - oy, x0 - ox
            hh, ww = min(h - sy, S - y0), min(w - sx, S - x0)
            canvas[:, :, y0:y0 + hh, x0:x0 + ww] = v[:, :, sy:sy + hh, sx:sx + ww]
            out[a] = canvas
        if moves and "walk" in out and all(m in out for m in moves):
            return out, k, sorted(set(raw) - set(out))
    return None


def with_stand(stand: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    """[dirs, ...] standing frames + [dirs, n, ...] cycle -> walk [dirs, 1 + n, ...]."""
    return np.concatenate([stand[:, None], cycle], 1)


def preview(sheets: dict) -> np.ndarray:
    """One row per direction: the reference, then the walk cycle (or the run when walk is reference only)."""
    move = sheets["walk"][:, 1:] if sheets["walk"].shape[1] > 1 else sheets["run"]
    seq = np.concatenate([sheets["walk"][:, :1], move], 1)
    rows = [np.concatenate(list(seq[r]) + [np.zeros((S, S, 4), np.uint8)] * (A.MAX_FRAMES + 1 - seq.shape[1]), 1)
            for r in range(4)]
    img = np.concatenate(rows, 0)
    bg = np.full_like(img, 255)
    bg[..., :3] = (200, 200, 210)
    img = np.where(img[..., 3:] > 0, img, bg)
    return np.asarray(Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST))


def write_dataset(chars, out: Path, split: str, check, config: dict, previews: bool = False) -> dict:
    """Place each character, write atlas PNGs, manifest.jsonl, dataset.json and CREDITS.csv.
    `check(manifest)` must raise for any artist that does not belong in this dataset."""
    if out.exists():
        shutil.rmtree(out)
    (out / "atlas").mkdir(parents=True)
    if previews:
        (out / "preview").mkdir()
    manifest, skipped = [], []
    for c in chars:
        placed = place(c.pop("raw"))
        if placed is None:
            skipped.append(c["id"])
            continue
        sheets, k, dropped = placed
        atlas, layout = pack(sheets)
        path = f"atlas/{c['id']}.png"
        (out / path).write_bytes(encode_png(atlas))
        manifest.append({
            "id": c["id"], "split": split, "atlas": path, "size": S, "layout": layout,
            "leading_stand": ["walk"] if layout["walk"]["frames"] > 1 else [],
            "attributes": {"artist": c["artist"], "pack": c["pack"], "url": c["url"], "licence": c["licence"],
                           "design": c["design"], "scale": f"1/{k}", "dropped_too_big": dropped,
                           "synthetic_dirs": c["synthetic_dirs"],
                           "reference_from_cycle": c.get("reference_from_cycle", False),
                           **({"look": c["look"]} if "look" in c else {})}})
        if previews:
            Image.fromarray(preview(sheets)).save(out / "preview" / f"{c['id']}.png")
    check(manifest)
    with open(out / "manifest.jsonl", "w") as f:
        for e in manifest:
            f.write(json.dumps(e) + "\n")
    by = {}
    for e in manifest:
        by.setdefault(e["attributes"]["artist"], []).append(e["id"])
    info = {"characters": len(manifest), "designs": len({e["attributes"]["design"] for e in manifest}),
            "by_artist": {k: len(v) for k, v in by.items()},
            "scale": {e["id"]: e["attributes"]["scale"] for e in manifest if e["attributes"]["scale"] != "1/1"},
            "skipped_too_big": skipped}
    write_json(out / "dataset.json", {"name": out.name, "config": {"name": out.name, "size": S, **config},
                                      "git": git_commit(), "counts": {split: len(manifest)}, **info})
    with open(out / "CREDITS.csv", "w") as f:
        f.write("id,artist,pack,url,licence\n")
        for e in manifest:
            t = e["attributes"]
            f.write(f"{e['id']},{t['artist']},\"{t['pack']}\",{t['url']},{t['licence']}\n")
    return info
