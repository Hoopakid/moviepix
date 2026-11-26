"""Build data/built/final32: the FINAL TEST, characters by artists that are never trained on.

    uv run python -I scripts/build_final32.py [--preview]

Reads the owner-downloaded packs in data/raw/e3/final/<pack> (see SOURCE.txt in each). Every entry is
split "final", so no training code can load it (TrainSet reads split "train" only), and the build
refuses any artist that is not on data/heldout.FINAL_ARTISTS. Scored once, after E3 is decided
(docs/research/phase2-design.md). Building it means looking at the art, never at model outputs.

Same placement as packs32 (data/packs.py): feet of the down-facing reference on (16, 30), 2x
pixel-art shrink only when a character does not fit. Sides drawn once are mirrored and listed in
`synthetic_dirs`; the scorer should score drawn directions only. Packs with no standing pose use a
frame of the cycle as the reference and are flagged `reference_from_cycle`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from moviepix import actions as A
from moviepix.data.heldout import FINAL_ARTISTS, _norm
from moviepix.data.packs import DIRS, cells, rgba, with_stand, write_dataset

RAW = Path("data/raw/e3/final")


def mirror_left(per: dict) -> dict:
    """per has up, down, right: add left as the mirrored right."""
    return {**per, "left": per["right"][:, :, ::-1]}


def stack(per: dict) -> np.ndarray:
    return np.stack([per[d] for d in DIRS])


def keyed(path, key) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGBA")).copy()
    img[np.all(img[..., :3] == key, -1)] = 0
    return img


def char(cid, artist, pack, url, licence, raw, synthetic=(), from_cycle=False, design=None):
    return {"id": cid, "artist": artist, "pack": pack, "url": url, "licence": licence, "design": design or cid,
            "raw": raw, "synthetic_dirs": list(synthetic), "reference_from_cycle": from_cycle}


# ---------------------------------------------------------------- ansimuz
def faune():
    """Legend of Faune: one PNG per frame, walk and run for down, up, side (faces right). No idle:
    the reference is walk frame 1."""
    root = next((RAW / "ansimuz-legend-of-faune/x").glob("*/Assets/Sprites/Player/sprites"))
    names = {"up": "up", "down": "down", "right": "side"}

    def seq(act, d):
        fs = sorted((root / f"{act}-{names[d]}").glob(f"{act}-{names[d]}-*.png"),
                    key=lambda f: int(re.search(r"-(\d+)\.png$", f.name).group(1)))
        return np.stack([rgba(f) for f in fs])
    walk = stack(mirror_left({d: seq("walk", d) for d in names}))
    run = stack(mirror_left({d: seq("run", d) for d in names}))
    yield char("ansimuz-faune", "ansimuz", "Legend of Faune", "https://ansimuz.itch.io/legend-of-faune", "CC0-1.0",
               {"walk": with_stand(walk[:, 0], walk), "run": run}, synthetic=["left"], from_cycle=True)


def tiny_rpg():
    """Tiny RPG Forest: hero, mole, treant; idle (one frame) and walk strips for front, back, side
    (faces right)."""
    root = next((RAW / "ansimuz-tiny-rpg/x").glob("*/Assets/PNG/spritesheets"))
    names = {"up": "back", "down": "front", "right": "side"}
    for who in ("hero", "mole", "treant"):
        def strip(act, d):
            f = next(f for f in (root / who / act).glob("*.png") if names[d] in f.stem and "weapon" not in f.stem)
            img = rgba(f)
            h = img.shape[0]
            w = {"hero": 32, "mole": 24, "treant": 31}[who]
            return cells(img, w, h)
        stand = stack(mirror_left({d: strip("idle", d)[:1] for d in names}))[:, 0]
        walk = stack(mirror_left({d: strip("walk", d) for d in names}))
        yield char(f"ansimuz-tinyrpg-{who}", "ansimuz", "Tiny RPG Forest", "https://ansimuz.itch.io/tiny-rpg-forest",
                   "CC0-1.0", {"walk": with_stand(stand, walk)}, synthetic=["left"])


# ---------------------------------------------------------------- 4-frame [stand, step, stand, step] walks
def stand_cycle(frames: np.ndarray) -> np.ndarray:
    """[dirs, 4 = stand, A, stand, B] -> walk [dirs, 5 = stand | A, stand, B, stand]."""
    return with_stand(frames[:, 0], frames[:, [1, 2, 3, 0]])


def gamekrazzy():
    root = RAW / "gamekrazzy-8dir/x/TopDownCharacter/Character"
    names = {"up": "Up", "left": "Left", "down": "Down", "right": "Right"}
    fr = np.stack([cells(rgba(root / f"Character_{names[d]}.png"), 32, 32) for d in DIRS])
    yield char("gamekrazzy-hero", "gamekrazzy", "8-direction Top Down Character",
               "https://gamekrazzy.itch.io/8-direction-top-down-character", "public domain (author comment)",
               {"walk": stand_cycle(fr)})


def gibbongl():
    """Game Boy template: 16x16 cells, 8 direction rows (matched to the pack's own named GIFs:
    down 0, left 2, up 4, right 6), 4 frames; GB Studio export keys (101, 255, 0) as transparent."""
    img = keyed(RAW / "gibbongl-gameboy/x/8-Directional Gameboy Character Template/loose sprites gbstudio.png",
                (101, 255, 0))
    rows = {"down": 0, "left": 2, "up": 4, "right": 6}
    fr = np.stack([cells(img, 16, 16, rows[d], 4) for d in DIRS])
    yield char("gibbongl-monkey", "gibbongl", "8-Directional Game Boy Character Template",
               "https://gibbongl.itch.io/8-directional-gameboy-character-template", "CC0-1.0", {"walk": stand_cycle(fr)})


def hormelz():
    """LowRez: 16x16, 2 frames per strip. Idle and Run in 4 straight directions; walk is the idle
    reference only."""
    root = RAW / "hormelz-lowrez/x"
    idle = {"up": "IdleBack", "left": "IdleLeft", "down": "IdleFront", "right": "IdleRight"}
    run = {"up": "Run_Up", "left": "Run_Left", "down": "Run_Down", "right": "Run_Right"}
    idle_fr = np.stack([cells(rgba(root / f"{idle[d]}.png"), 16, 16) for d in DIRS])
    run_fr = np.stack([cells(rgba(root / f"{run[d]}.png"), 16, 16) for d in DIRS])
    yield char("hormelz-lowrez", "hormelz", "LowRez 8-Directional 2D Sprite", "https://hormelz.itch.io/8-directional-2d-sprite",
               "CC0-1.0", {"walk": idle_fr[:, :1], "idle": idle_fr, "run": run_fr})


# ---------------------------------------------------------------- CC-BY (final test only, credit given)
def robocelot_template():
    """smooth-sprite.0.2.png: 48x60 cells; rows 0-3 walk (8) and rows 4-7 idle (6), each in the order
    south, east, north, west (checked by eye, same order as Gumdrop)."""
    img = rgba(RAW / "robocelot-template/dl/smooth-sprite.0.2.png")
    order = {"down": 0, "right": 1, "up": 2, "left": 3}
    walk = np.stack([cells(img, 48, 60, order[d], 8) for d in DIRS])
    idle = np.stack([cells(img, 48, 60, 4 + order[d], 6) for d in DIRS])
    yield char("robocelot-template", "robocelot", "Pixel Art Character Template",
               "https://robocelot.itch.io/pixelart-character-template", "CC-BY-4.0 (credit Robocelot)",
               {"walk": with_stand(idle[:, 0], walk), "idle": idle})


def robocelot_gumdrop():
    """Gumdrop: 64x64 cells, 10 columns; rows from SpriteSheets/flash.json: idle 0-3, walk 4-7, run 8-11,
    hurt 24-27, each south, east, north, west. One character per *_P1 sheet (other P numbers are palettes)."""
    root = RAW / "robocelot-gumdrop/x/Gumdrop_Character_Assets/SpriteSheets"
    order = {"down": 0, "right": 1, "up": 2, "left": 3}
    sheets = sorted(f for f in root.iterdir() if re.search(r"_P1\.png$", f.name, re.I))
    for f in sheets:
        img = rgba(f)
        if img.shape[:2] != (3072, 640):
            continue
        name = re.sub(r"_P1$", "", f.stem, flags=re.I).lower().replace("_", "-")
        idle = np.stack([cells(img, 64, 64, order[d], 8) for d in DIRS])
        walk = np.stack([cells(img, 64, 64, 4 + order[d], 8) for d in DIRS])
        run = np.stack([cells(img, 64, 64, 8 + order[d], 8) for d in DIRS])
        hurt = cells(img, 64, 64, 24, 3)[None]
        yield char(f"robocelot-gumdrop-{name}", "robocelot", "Gumdrop Galaxy Characters",
                   "https://robocelot.itch.io/gumdrop-source-and-sprites", "CC-BY-4.0 (credit Robocelot)",
                   {"walk": with_stand(idle[:, 0], walk), "idle": idle, "run": run, "hurt": hurt})


def axulart():
    """Small 8-direction Characters: 16x24 cells; three blocks (template, girl, boy) of an arrow row and
    3 frame rows; columns follow the arrows (up 0, right 2, down 4, left 6). No idle: the middle frame
    (feet together) is the reference and the cycle is [step, middle, step, middle]."""
    img = rgba(next((RAW / "axulart-8dir/x").glob("*/Small-8-Direction-Characters_by_AxulArt.png")))
    cols = {"up": 0, "left": 6, "down": 4, "right": 2}
    for blk, who in enumerate(("template", "girl", "boy")):
        fr = np.stack([np.stack([img[(4 * blk + 1 + r) * 24:(4 * blk + 2 + r) * 24, cols[d] * 16:cols[d] * 16 + 16]
                                 for r in range(3)]) for d in DIRS])
        yield char(f"axulart-{who}", "axulart", "Small 8-direction Characters",
                   "https://axulart.itch.io/small-8-direction-characters", "CC-BY-4.0 (credit AxulArt)",
                   {"walk": with_stand(fr[:, 1], fr[:, [0, 1, 2, 1]])}, from_cycle=True)


def check_final(manifest):
    ok = {_norm(a) for a in FINAL_ARTISTS}
    for e in manifest:
        if _norm(e["attributes"]["artist"]) not in ok:
            raise ValueError(f"{e['id']}: {e['attributes']['artist']!r} is not a final-test artist")
        if e["split"] != "final":
            raise ValueError(f"{e['id']}: final-test entries must have split 'final'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/built/final32"))
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    chars = [*faune(), *tiny_rpg(), *gamekrazzy(), *gibbongl(), *hormelz(), *robocelot_template(),
             *robocelot_gumdrop(), *axulart()]
    for c in chars:  # actions must match moviepix.actions directions
        for name, v in c["raw"].items():
            assert v.shape[0] == len(A.get(name).directions), (c["id"], name, v.shape)
    info = write_dataset(chars, a.out, "final", check_final, {"note": "FINAL TEST: never train, score once"},
                         a.preview)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
