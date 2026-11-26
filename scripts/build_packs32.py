"""Build data/built/packs32: real CC0 characters by the E3 training artists, in the LPC atlas format.

    uv run python -I scripts/build_packs32.py            # build
    uv run python -I scripts/build_packs32.py --preview  # also write one preview PNG per character

Reads the owner-downloaded packs in data/raw/e3/train/<pack>/x (see SOURCE.txt in each).
Only artists on data/heldout.TRAIN_ARTISTS are read; the final-test packs are never touched.

Placement. Every character uses one integer offset for all its frames (the art is already
aligned in its cells): the standing reference facing down gets its feet on LPC's pivot
(16, 30). A character that does not fit the 32 px canvas at 1x is first shrunk 2x with the
same pixel-art downscaler that made lpc32 from lpc64 (data/resize.py); nothing is cropped.

Walk. Frame 0 of `walk` is the standing reference (as in LPC) and `leading_stand` tells the
loader that frame 0 is not part of the cycle. A pack with no standing pose (MetroCity is run
only) stores its reference as a 1-frame walk, which the loader never uses as a target, and is
flagged `reference_from_cycle`. Sides drawn only once are mirrored and flagged in
`synthetic_dirs`.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from moviepix import actions as A
from moviepix.data.atlas import encode_png, pack
from moviepix.data.heldout import TRAIN_ARTISTS, check_manifest
from moviepix.data.resize import downscale
from moviepix.utils.io import git_commit, write_json

RAW = Path("data/raw/e3/train")
S, PIVOT = 32, (16, 30)
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
def place(raw: dict, ref_dir: str = "down") -> tuple[dict, int] | None:
    """raw[action] = [dirs, frames, h, w, 4] native cells -> (sheets on the 32 canvas, scale divisor)."""
    for k in (1, 2):
        sh = raw if k == 1 else {a: downscale(v, 2) for a, v in raw.items()}
        ref = sh["walk"][DIRS.index(ref_dir), 0]
        ys, xs = np.nonzero(ref[..., 3])
        oy = PIVOT[1] - ys.max()
        ox = int(round(PIVOT[0] - (xs.min() + xs.max() + 1) / 2))
        out, fits = {}, True
        for a, v in sh.items():
            h, w = v.shape[2:4]
            occ = v[..., 3].reshape(-1, h, w).any(0)
            yy, xx = np.nonzero(occ)
            if yy.min() + oy < 0 or yy.max() + oy >= S or xx.min() + ox < 0 or xx.max() + ox >= S:
                fits = False
                break
            canvas = np.zeros((*v.shape[:2], S, S, 4), np.uint8)
            y0, x0 = max(oy, 0), max(ox, 0)
            sy, sx = y0 - oy, x0 - ox
            hh, ww = min(h - sy, S - y0), min(w - sx, S - x0)
            canvas[:, :, y0:y0 + hh, x0:x0 + ww] = v[:, :, sy:sy + hh, sx:sx + ww]
            out[a] = canvas
        if fits:
            return out, k
    return None


def with_stand(stand: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    """[dirs, ...] standing frames + [dirs, n, ...] cycle -> walk [dirs, 1 + n, ...]."""
    return np.concatenate([stand[:, None], cycle], 1)


# ---------------------------------------------------------------- packs
def bumblebee_custom(rng, n_looks: int):
    """Ordinary Bumblebee, 32x32 Customizable Character Pack: layered, one variant per sheet row."""
    root = RAW / "ordinary-bumblebee-customizable/x/32x32 Customizable Character Pack"
    names = {"up": "Back", "left": "Left", "down": "Front", "right": "Right"}
    layers = ["Character/Character", "Eyes/Eyes", "Clothing/Clothing_Shoes", "Clothing/Clothing_Bottoms",
              "Clothing/Clothing_Top*", "Hair/Hair"]

    def sheet(layer, act, d):
        folder, stem = layer.split("/")
        hits = sorted((root / act / folder).glob(f"{stem}_{act}_{names[d]}-Sheet.png"))
        return rgba(hits[0]) if hits else None

    def rows(layer):
        return sheet(layer, "Walk", "down").shape[0] // 32

    counts = {ly: rows(ly) for ly in layers}
    acts = {"walk": "Walk", "idle": "Idle", "run": "Run", "jump": "Jump", "hurt": "Hurt"}
    for i in range(n_looks):
        pick = {ly: int(rng.integers(counts[ly])) for ly in layers}

        def frames(act, d):
            out = None
            for ly in layers:
                s = sheet(ly, act, d)
                if s is None:
                    if ly.startswith("Eyes") and d == "up":
                        continue  # no eyes from behind
                    return None
                if s.shape[0] // 32 != counts[ly]:
                    return None  # rows do not line up with the walk sheet
                c = cells(s, 32, 32, pick[ly])
                out = c if out is None else np.stack([over(o, t) for o, t in zip(out, c)])
            return out

        raw = {}
        stand = np.stack([frames("Idle", d)[0] for d in DIRS])
        for name, act in acts.items():
            dirs = A.get(name).directions
            fr = [frames(act, d) for d in dirs]
            if any(f is None for f in fr):
                continue
            arr = np.stack(fr)
            raw[name] = with_stand(stand, arr) if name == "walk" else arr
        yield {"id": f"bumblebee-custom-{i}", "artist": "ordinary bumblebee", "pack": "32x32 Customizable Character Pack",
               "url": "https://ordinary-bumblebee.itch.io/customizable-character-pack", "licence": "CC0-1.0",
               "design": "bumblebee-custom-base", "look": pick, "raw": raw, "synthetic_dirs": []}


def bumblebee_8dir(colours=("Blue", "Green")):
    """Ordinary Bumblebee, 8 Direction Walk Cycle (32x48 cells). Uses the 4 straight directions."""
    root = RAW / "ordinary-bumblebee-8dir-walk/x/8 Direction Walk Sheets (10 Colors)"
    names = {"up": "NORTH", "left": "WEST", "down": "SOUTH", "right": "EAST"}
    for col in colours:
        d = root / col
        walk = {k: cells(rgba(next(d.glob(f"*_walk_{v}-Sheet.png"))), 32, 48) for k, v in names.items()}
        standing = cells(rgba(next(d.glob("*_8direction_standing-Sheet.png"))), 32, 48)
        # The standing sheet's direction order is not documented: give each direction the standing
        # frame that best matches its own walk frames (silhouette IoU).
        stand = []
        for k in DIRS:
            w = walk[k][..., 3] > 0

            def score(s):
                m = s[..., 3] > 0
                return max((m & f).sum() / max((m | f).sum(), 1) for f in w)
            stand.append(max(standing, key=score))
        raw = {"walk": with_stand(np.stack(stand), np.stack([walk[k] for k in DIRS]))}
        yield {"id": f"bumblebee-8dir-{col.lower()}", "artist": "ordinary bumblebee", "pack": "8 Direction Walk Cycle",
               "url": "https://ordinary-bumblebee.itch.io/8-direction-walk-cycle", "licence": "CC0-1.0",
               "design": "bumblebee-8dir", "raw": raw, "synthetic_dirs": []}


def lexlom():
    """Lexlom, 32 Free animated characters: 32x64 cells; rows front, back, side (faces right);
    3 frames [step, stand, step]. The left side is the mirrored right side."""
    img = rgba(RAW / "lexlom-32-characters/dl/VillagersSheetPNG.png")
    n = 0
    for block in (0, 1, 2):  # three blocks of three rows
        for c in range(0, img.shape[1] // 32 - 2, 3):
            rows = [cells(img, 32, 64, 4 * block + r, 3, c) for r in range(3)]  # blocks start at y 0, 256, 512
            if not all(f[..., 3].any() for r in rows for f in r):
                continue
            down, up, right = rows
            left = right[:, :, ::-1]
            per = {"up": up, "left": left, "down": down, "right": right}
            stand = np.stack([per[d][1] for d in DIRS])
            cyc = np.stack([per[d][[0, 1, 2, 1]] for d in DIRS])  # step, stand, step, stand
            yield {"id": f"lexlom-{n}", "artist": "lexlom", "pack": "32 Free animated characters pack",
                   "url": "https://lexlom.itch.io/32-character-pack-pixel-art-4-direction-idle-walking",
                   "licence": "public domain (author comment)", "design": f"lexlom-{n}",
                   "raw": {"walk": with_stand(stand, cyc)}, "synthetic_dirs": ["left"]}
            n += 1


def foozle_lucifer():
    """Foozle (commissioned from Baldur), Lucifer Warrior: 48x48 cells, 4 drawn directions."""
    root = RAW / "foozle-lucifer-warrior/x/Foozle_2DC0009_Lucifer_Warrior_Pixel_Art"
    names = {"up": "Up", "left": "Left", "down": "Down", "right": "Right"}

    def strip(d, act):
        return cells(rgba(root / names[d] / "Png" / f"Warrior{names[d]}{act}.png"), 48, 48)
    stand = np.stack([strip(d, "Idle")[0] for d in DIRS])
    raw = {"walk": with_stand(stand, np.stack([strip(d, "Walk") for d in DIRS])),
           "idle": np.stack([strip(d, "Idle") for d in DIRS]),
           "jump": np.stack([strip(d, "Jump") for d in DIRS]),
           "hurt": strip("down", "Hurt")[None],
           "climb": strip("up", "Climb")[None]}
    yield {"id": "foozle-lucifer-warrior", "artist": "foozle", "pack": "Lucifer Warrior (drawn by Baldur)",
           "url": "https://foozlecc.itch.io/lucifer-warrior", "licence": "CC0-1.0", "design": "foozle-lucifer-warrior",
           "raw": raw, "synthetic_dirs": []}


def metrocity(rng, n_looks: int):
    """JIK-A-4, MetroCity: 32x32 cells, 24 columns = run x 6 frames for down, right, up, left;
    6 body rows; outfits and hair as layers. No standing pose: the reference is run frame 0."""
    root = RAW / "jik-a-4-metrocity/x"
    bodies = rgba(root / "MetroCity/CharacterModel/Character Model.png")
    outfits = [rgba(p) for p in sorted((root / "MetroCity/Outfits").glob("Outfit*.png"))]
    outfits += [r for p in ("Suit.png", "Suit1.png") for r in np.split(rgba(root / "MetroCity 2.0" / p),
                                                                       rgba(root / "MetroCity 2.0" / p).shape[0] // 32)]
    hairs = np.split(rgba(root / "MetroCity/Hair/Hairs.png"), 8) + np.split(rgba(root / "MetroCity 2.0/Hair.png"), 5)
    col0 = {"down": 0, "right": 6, "up": 12, "left": 18}
    for i in range(n_looks):
        b, o, h = int(rng.integers(bodies.shape[0] // 32)), int(rng.integers(len(outfits))), int(rng.integers(len(hairs)))
        per = {}
        for d in DIRS:
            fr = cells(bodies, 32, 32, b, 6, col0[d])
            fr = np.stack([over(f, t) for f, t in zip(fr, cells(outfits[o], 32, 32, 0, 6, col0[d]))])
            fr = np.stack([over(f, t) for f, t in zip(fr, cells(hairs[h], 32, 32, 0, 6, col0[d]))])
            per[d] = fr
        run = np.stack([per[d] for d in DIRS])
        yield {"id": f"metrocity-{i}", "artist": "jik-a-4", "pack": "MetroCity (1.0 and 2.0)",
               "url": "https://jik-a-4.itch.io/metrocity-free-topdown-character-pack", "licence": "CC0-1.0",
               "design": "metrocity-base", "look": {"body": b, "outfit": o, "hair": h},
               "raw": {"walk": run[:, :1], "run": run}, "synthetic_dirs": [], "reference_from_cycle": True}


# ---------------------------------------------------------------- build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/built/packs32"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bumblebee-looks", type=int, default=12)
    ap.add_argument("--metro-looks", type=int, default=8)
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    if a.out.exists():
        shutil.rmtree(a.out)
    (a.out / "atlas").mkdir(parents=True)
    if a.preview:
        (a.out / "preview").mkdir()

    chars = [*bumblebee_custom(rng, a.bumblebee_looks), *bumblebee_8dir(), *lexlom(), *foozle_lucifer(),
             *metrocity(rng, a.metro_looks)]
    manifest, skipped = [], []
    for c in chars:
        placed = place(c.pop("raw"))
        if placed is None:
            skipped.append(c["id"])
            continue
        sheets, k = placed
        atlas, layout = pack(sheets)
        path = f"atlas/{c['id']}.png"
        (a.out / path).write_bytes(encode_png(atlas))
        entry = {"id": c["id"], "split": "train", "atlas": path, "size": S, "layout": layout,
                 "leading_stand": ["walk"] if layout["walk"]["frames"] > 1 else [],
                 "attributes": {"artist": c["artist"], "pack": c["pack"], "url": c["url"], "licence": c["licence"],
                                "design": c["design"], "scale": f"1/{k}", "synthetic_dirs": c["synthetic_dirs"],
                                "reference_from_cycle": c.get("reference_from_cycle", False),
                                **({"look": c["look"]} if "look" in c else {})}}
        manifest.append(entry)
        if a.preview:
            rows = [np.concatenate(list(sheets["walk"][r]) + [np.zeros((S, S, 4), np.uint8)] *
                                   (A.MAX_FRAMES - sheets["walk"].shape[1]), 1) for r in range(4)]
            img = np.concatenate(rows, 0)
            bg = np.full_like(img, 255); bg[..., :3] = (200, 200, 210)
            img = np.where(img[..., 3:] > 0, img, bg)
            Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(
                a.out / "preview" / f"{c['id']}.png")

    check_manifest(manifest, TRAIN_ARTISTS)
    with open(a.out / "manifest.jsonl", "w") as f:
        for e in manifest:
            f.write(json.dumps(e) + "\n")
    by = {}
    for e in manifest:
        by.setdefault(e["attributes"]["artist"], []).append(e["id"])
    write_json(a.out / "dataset.json", {
        "name": a.out.name, "config": {"name": a.out.name, "size": S, "seed": a.seed,
                                       "bumblebee_looks": a.bumblebee_looks, "metro_looks": a.metro_looks},
        "git": git_commit(), "counts": {"train": len(manifest)},
        "designs": len({e["attributes"]["design"] for e in manifest}),
        "by_artist": {k: len(v) for k, v in by.items()}, "skipped_too_big": skipped})
    with open(a.out / "CREDITS.csv", "w") as f:
        f.write("id,artist,pack,url,licence\n")
        for e in manifest:
            t = e["attributes"]
            f.write(f"{e['id']},{t['artist']},\"{t['pack']}\",{t['url']},{t['licence']}\n")
    print(json.dumps({"characters": len(manifest), "by_artist": {k: len(v) for k, v in by.items()},
                      "scale": {e["id"]: e["attributes"]["scale"] for e in manifest if e["attributes"]["scale"] != "1/1"},
                      "skipped": skipped}, indent=1))


if __name__ == "__main__":
    main()
