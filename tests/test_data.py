import numpy as np
import pytest

from moviepix.data import atlas
from moviepix.data.resize import downscale
from moviepix.data.splits import assign, held_out_items


def _sheets(size=64, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name, dirs, frames in (("walk", 4, 9), ("hurt", 1, 6), ("idle", 4, 2)):
        s = rng.integers(0, 4, (dirs, frames, size, size, 4), dtype=np.uint8) * 60
        s[..., 3] = np.where(s[..., 3] > 60, 255, 0)
        s[s[..., 3] == 0] = 0
        out[name] = s
    return out


def test_atlas_roundtrip_and_png_is_lossless(tmp_path):
    sheets = _sheets()
    img, layout = atlas.pack(sheets)
    path = tmp_path / "a.png"
    path.write_bytes(atlas.encode_png(img))
    back = atlas.unpack(atlas.load_png(path), layout, 64)
    assert set(back) == set(sheets)
    for k in sheets:
        assert np.array_equal(back[k], sheets[k])


def test_downscale_keeps_only_original_colors_and_binary_alpha():
    img = np.zeros((4, 4, 4), np.uint8)
    img[0:2, 0:2] = (200, 10, 10, 255)
    img[0, 0] = (10, 10, 10, 255)       # one dark pixel loses the vote 3-1
    img[2:4, 2:4] = (0, 0, 0, 0)
    img[2, 2] = (5, 5, 5, 255)          # 1 of 4 opaque -> transparent
    img[0:2, 2:4] = [(9, 9, 9, 255), (190, 20, 20, 255)]  # 2-2 tie -> closer to the area's mean wins
    small = downscale(img)
    assert small.shape == (2, 2, 4)
    assert small[0, 0].tolist() == [200, 10, 10, 255]
    assert small[1, 1, 3] == 0
    assert small[0, 1].tolist() == [190, 20, 20, 255]
    assert set(np.unique(small[..., 3])) <= {0, 255}


def test_downscale_rejects_odd_sizes():
    with pytest.raises(ValueError):
        downscale(np.zeros((3, 4, 4), np.uint8))


def test_splits_are_stable_and_held_out_items_deterministic():
    assert assign("lpc-1-5", 0.03, 0.03) == assign("lpc-1-5", 0.03, 0.03)
    splits = [assign(f"lpc-1-{i}", 0.1, 0.1) for i in range(5000)]
    assert 0.07 < splits.count("validation") / 5000 < 0.13
    ids = [f"item{i}" for i in range(100)]
    assert held_out_items(ids, 0.05, 1) == held_out_items(ids, 0.05, 1)
    assert len(held_out_items(ids, 0.05, 1)) == 5
