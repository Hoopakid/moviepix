"""The web demo places sprites exactly like the training data (Phase 3)."""
import base64
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from moviepix.data.packs import place_reference

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests/fixtures/placement.json"


def test_placement_fixture_is_current():
    """Regenerate with scripts/make_placement_fixture.py if data/packs.place_reference changes."""
    fx = json.loads(FIXTURE.read_text())
    for c in fx["cases"]:
        img = np.frombuffer(base64.b64decode(c["rgba"]), np.uint8).reshape(c["h"], c["w"], 4)
        got = place_reference(img, fx["size"])
        assert (got is None) == (c["scale"] is None)
        if got is not None:
            assert got[1] == c["scale"]
            assert got[0].tobytes() == base64.b64decode(c["canvas"])


def test_place_reference_matches_pack_placement():
    """A sprite placed alone lands where data/packs.place puts the same standing frame."""
    from moviepix.data.packs import place

    rng = np.random.default_rng(3)
    ref = np.zeros((40, 24, 4), np.uint8)
    ref[6:38, 5:19, :3] = rng.integers(0, 255, (32, 14, 3))
    ref[6:38, 5:19, 3] = 255
    walk = np.stack([np.stack([ref, ref])] * 4)  # [dirs, 2 frames, h, w, 4]
    sheets, k, _ = place({"walk": walk})
    alone, k2 = place_reference(ref)
    assert k == k2 and (sheets["walk"][2, 0] == alone).all()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_place_js_matches_python():
    r = subprocess.run(["node", str(ROOT / "web/place.test.mjs")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_gif_writer_round_trips(tmp_path):
    """web/gif.js frames decode (Pillow) to exactly the frames given, zoomed, with transparency."""
    from PIL import Image

    out = tmp_path / "t.gif"
    r = subprocess.run(["node", str(ROOT / "web/gif.test.mjs"), str(out)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    ref = json.loads(Path(str(out) + ".json").read_text())
    w, h, k = ref["w"], ref["h"], ref["scale"]
    im = Image.open(out)
    assert im.n_frames == len(ref["frames"]) and im.info.get("loop") == 0
    for t, b64 in enumerate(ref["frames"]):
        im.seek(t)
        got = np.asarray(im.convert("RGBA"))
        want = np.frombuffer(base64.b64decode(b64), np.uint8).reshape(h, w, 4).repeat(k, 0).repeat(k, 1)
        drawn = want[..., 3] > 0
        assert (got[..., 3] > 0).tolist() == drawn.tolist()
        assert (got[drawn][:, :3] == want[drawn][:, :3]).all()
