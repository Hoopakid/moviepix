"""Build README visuals from a built dataset (training data, not model output).

    uv run python scripts/make_readme_assets.py data/built/lpc64

Writes assets/hero.gif and assets/CREDITS.md (art credits for the characters shown).
"""
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepix.data.atlas import load_png, unpack
from moviepix.utils.io import read_jsonl

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/built/lpc64")
ACTIONS = ["walk", "run", "slash", "spellcast", "shoot", "jump"]
PICK = 4
BG, FG, MUTED = (24, 24, 27), (236, 236, 234), (140, 140, 134)
SCALE, LABEL_H, GAP = 2, 18, 10

NATURAL_SKIN = {"light", "amber", "olive", "taupe", "bronze", "brown", "black"}
entries = [e for e in read_jsonl(ROOT / "manifest.jsonl") if e["split"] == "train"
           and all(a in e["layout"] for a in ACTIONS) and e["attributes"]["skin"] in NATURAL_SKIN
           and any(it["slot"] == "hair" for it in e["attributes"]["items"])
           and not any(it["slot"] in ("hat", "neck", "eyes") for it in e["attributes"]["items"])]
rng = np.random.default_rng(11)
chosen = [entries[i] for i in rng.choice(len(entries), PICK, replace=False)]

sheets = [unpack(load_png(ROOT / e["atlas"]), e["layout"], 64) for e in chosen]
cell = 64 * SCALE
cols = 1 + len(ACTIONS)
W = cols * cell + (cols - 1) * GAP + 2 * GAP + GAP
H = LABEL_H + PICK * cell + (PICK + 1) * GAP
font = ImageFont.load_default()
frames = []
for f in range(24):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for c, name in enumerate(["input"] + ACTIONS):
        x = GAP + c * (cell + GAP) + (GAP if c else 0)
        d.text((x + 4, 4), name, fill=MUTED if c == 0 else FG, font=font)
    for r, sh in enumerate(sheets):
        y = LABEL_H + GAP + r * (cell + GAP)
        tiles = [sh["walk"][2, 0]] + [sh[a][2, f % sh[a].shape[1]] for a in ACTIONS]
        for c, tile in enumerate(tiles):
            x = GAP + c * (cell + GAP) + (GAP if c else 0)
            t = Image.fromarray(tile).resize((cell, cell), Image.NEAREST)
            img.paste(t, (x, y), t)
        d.line([(GAP + cell + GAP // 2 + GAP // 2, y), (GAP + cell + GAP // 2 + GAP // 2, y + cell)], fill=(60, 60, 64))
    frames.append(img)
Path("assets").mkdir(exist_ok=True)
frames[0].save("assets/hero.gif", save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)

items = {it["item"] for e in chosen for it in e["attributes"]["items"]}
rows = [r for r in csv.DictReader(open(ROOT / "CREDITS.csv")) if r["item"] in items]
with open("assets/CREDITS.md", "w") as f:
    f.write("# Credits for the art in these images\n\n")
    f.write("The characters in `hero.gif` are composed from [Liberated Pixel Cup](https://lpc.opengameart.org/) art "
            "via the [Universal LPC Spritesheet Character Generator](https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator). "
            "They are training data, not model output. Each piece is used under the first permissive license listed.\n\n")
    f.write("| Item | Authors | Licenses | Source |\n|---|---|---|---|\n")
    seen = set()
    for r in sorted(rows, key=lambda r: r["item"]):
        key = (r["item"], r["authors"])
        if key in seen:
            continue
        seen.add(key)
        urls = " ".join(f"[link]({u})" for u in r["urls"].split())
        f.write(f"| `{r['item']}` | {r['authors']} | {r['licenses']} | {urls} |\n")
print("assets/hero.gif", Path("assets/hero.gif").stat().st_size // 1024, "KB;", len(seen), "credit rows")
