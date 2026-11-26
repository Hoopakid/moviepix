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
from pathlib import Path

import numpy as np

from moviepix import actions as A
from moviepix.data.heldout import TRAIN_ARTISTS, check_manifest
from moviepix.data.packs import DIRS, cells, over, rgba, with_stand, write_dataset

RAW = Path("data/raw/e3/train")
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


FOOZLE = {  # folder suffix -> (itch.io slug, file prefix); all drawn by Baldur, distributed by Foozle, CC0
    "lucifer-warrior": ("lucifer-warrior", "Warrior"),
    "necromancer": ("lucifer-necromancer", "Necromancer"),
    "sorceress": ("lucifer-sorceress", "Sorceress"),
    "cultist": ("lucifer-cultist-enemy", "Cultist"),
    "goblin-berserker": ("lucifer-goblin-berserker-enemy", "Goblin"),
    "goblin-slinger": ("lucifer-goblin-slinger-enemy", "GoblinSlinger"),
    "goblin-beast": ("lucifer-goblin-beast-boss", "GoblinBeast"),
    "skeleton-grunt": ("lucifer-skeleton-grunt-enemy", "SkeletonWithSword"),
    "skeleton-hunter": ("lucifer-skeleton-hunter-enemy", "SkeletonWithBow"),
    "skeleton-king": ("lucifer-skeleton-king-boss", "SkeletonKing"),
    "skeleton-ancient": ("lucifer-skeleton-ancient-boss", "AncientSkeleton"),
    # Goblin Rider is left out: a goblin on a mount, "Move" sheets rather than a walk.
}


def foozle():
    """Foozle "Lucifer" series: one strip per direction and action, square cells (48 or 80 px),
    4 drawn directions. Reference = Idle frame 0. Packs drawn with Run and no Walk keep a
    1-frame walk (the standing reference only) and train on run."""
    names = {"up": "Up", "left": "Left", "down": "Down", "right": "Right"}
    acts = {"idle": "Idle", "jump": "Jump", "hurt": "Hurt", "climb": "Climb"}
    for folder, (slug, prefix) in FOOZLE.items():
        hits = list((RAW / f"foozle-{folder}/x").glob("*/Down/Png"))
        if not hits:
            continue
        root = hits[0].parent.parent

        def strip(d, act):
            hits = [f for f in (root / names[d] / "Png").glob("*.png")
                    if f.stem.lower().endswith(act.lower()) and "attack" not in f.stem.lower()]
            if not hits:
                return None
            img = rgba(sorted(hits)[0])
            return cells(img, img.shape[0], img.shape[0])

        stand = [strip(d, "Idle") for d in DIRS]
        if any(x is None for x in stand):
            continue
        cell = stand[0].shape[1]
        stand = np.stack([x[0] for x in stand])
        raw = {}
        # The series names one locomotion cycle "Walk" in some directions and "Run" in others
        # (and has typos like "Leftt"): take Walk else Run per direction, and call the cycle a walk
        # when most directions say Walk.
        loco = [(strip(d, "Walk"), "walk") if strip(d, "Walk") is not None else (strip(d, "Run"), "run") for d in DIRS]
        if all(x is not None for x, _ in loco) and len({x.shape[0] for x, _ in loco}) == 1 \
                and all(x.shape[1] == cell for x, _ in loco):
            # the cycle must stand on the same ground line as the idle reference (Skeleton King's
            # walk sits 4 px lower than its idle, so it is left out)
            feet = lambda x: np.nonzero(x[..., 3].any(-1).any(0))[0].max()  # noqa: E731
            if any(abs(feet(x) - feet(st[None])) > 2 for (x, _), st in zip(loco, stand)):
                continue
            name = "walk" if sum(n == "walk" for _, n in loco) > 2 else "run"
            arr = np.stack([x for x, _ in loco])
            raw[name] = with_stand(stand, arr) if name == "walk" else arr
        for name, act in acts.items():
            if name in ("walk", "run"):
                continue
            fr = [strip(d, act) for d in A.get(name).directions]
            if any(x is None or x.shape[1] != cell for x in fr) or len({x.shape[0] for x in fr}) != 1:
                continue
            raw[name] = np.stack(fr)
        if "walk" not in raw and "run" not in raw:
            continue  # no usable locomotion (Skeleton Ancient: side walks use a different cell size)
        raw.setdefault("walk", stand[:, None])  # no drawn walk: reference only
        yield {"id": f"foozle-{folder}", "artist": "foozle", "pack": f"Lucifer {folder.replace('-', ' ').title()} (drawn by Baldur)",
               "url": f"https://foozlecc.itch.io/{slug}", "licence": "CC0-1.0", "design": f"foozle-{folder}",
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
    chars = [*bumblebee_custom(rng, a.bumblebee_looks), *bumblebee_8dir(), *lexlom(), *foozle(),
             *metrocity(rng, a.metro_looks)]
    info = write_dataset(chars, a.out, "train", lambda m: check_manifest(m, TRAIN_ARTISTS),
                         {"seed": a.seed, "bumblebee_looks": a.bumblebee_looks, "metro_looks": a.metro_looks}, a.preview)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
