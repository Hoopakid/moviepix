"""Write tests/fixtures/placement.json: odd-sized sprites and where data/packs.place_reference puts them.

    uv run python -I scripts/make_placement_fixture.py

The web demo's place.js must give the same canvases (web/place.test.mjs; tests/test_web.py runs it).
Sprites are random blobs, not art, so the fixture can live in git.
"""
import base64
import json
from pathlib import Path

import numpy as np

from moviepix.data.packs import place_reference

OUT = Path("tests/fixtures/placement.json")


def blob(rng, h, w, n_colours, fill=0.7):
    img = np.zeros((h, w, 4), np.uint8)
    pal = rng.integers(0, 256, (n_colours, 3))
    m = rng.random((h, w)) < fill
    m[-1, rng.integers(0, w)] = True  # feet on the last row
    img[m, :3] = pal[rng.integers(0, n_colours, m.sum())]
    img[m, 3] = 255
    return img


def main():
    rng = np.random.default_rng(7)
    shapes = [(16, 16), (24, 16), (17, 13), (30, 9), (31, 31), (32, 20), (33, 18), (40, 24), (45, 32), (48, 48),
              (61, 30), (64, 64), (70, 20), (9, 30), (12, 33), (20, 34), (8, 8), (15, 22), (27, 5), (36, 63)]
    cases = []
    for i, (h, w) in enumerate(shapes):
        img = blob(rng, h, w, int(rng.integers(2, 12)), fill=float(rng.uniform(0.3, 0.95)))
        if i % 3 == 0:  # a margin of empty pixels around the character
            img = np.pad(img, ((int(rng.integers(0, 5)), int(rng.integers(0, 3))), (int(rng.integers(0, 4)),) * 2, (0, 0)))
        res = place_reference(img)
        b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()  # noqa: E731
        cases.append({"h": img.shape[0], "w": img.shape[1], "rgba": b64(img),
                      "scale": None if res is None else res[1], "canvas": None if res is None else b64(res[0])})
    OUT.write_text(json.dumps({"size": 32, "cases": cases}))
    print(OUT, len(cases), "cases;", sum(c["scale"] == 1 for c in cases), "at 1x,", sum(c["scale"] == 2 for c in cases),
          "at 2x,", sum(c["scale"] is None for c in cases), "refused")


if __name__ == "__main__":
    main()
