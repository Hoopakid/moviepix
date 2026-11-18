import numpy as np

from moviepix.data import augment as AUG


def _character(S=32, seed=0):
    rng = np.random.default_rng(seed)
    sheets, parts = {}, {}
    for name, dirs, frames in (("walk", 4, 9), ("slash", 4, 6), ("hurt", 1, 6)):
        img = np.zeros((dirs, frames, S, S, 4), np.uint8)
        part = np.zeros((dirs, frames, S, S), np.uint8)
        # head (part 2) on top, torso (5), legs (7), with a few colours
        for y0, y1, pid in ((4, 12, 2), (12, 20, 5), (20, 28, 7)):
            img[..., y0:y1, 10:22, :3] = rng.integers(0, 255, 3)
            img[..., y0:y1, 10:22, 3] = 255
            part[..., y0:y1, 10:22] = pid
        img[..., 6, 12, :3] = (1, 2, 3)  # an eye
        sheets[name], parts[name] = img, part
    return sheets, parts


def test_every_transform_keeps_shapes_and_binary_alpha():
    sheets, parts = _character()
    for t in AUG.TRANSFORMS:
        out, names = AUG.augment(sheets, parts, np.random.default_rng(1), 1.0, transforms=(t,))
        assert names == [t]
        for k, v in sheets.items():
            assert out[k].shape == v.shape and out[k].dtype == np.uint8
            assert np.isin(out[k][..., 3], (0, 255)).all()
            assert (out[k][..., 3] > 0).any()


def test_same_frame_gets_same_result():
    """Reference and target frames go through one function: equal inputs stay equal."""
    sheets, parts = _character()
    out, _ = AUG.augment(sheets, parts, np.random.default_rng(3), 1.0)
    assert np.array_equal(out["walk"][2, 0], out["walk"][2, 1])  # all frames identical in the toy data


def test_colour_transforms_only_use_existing_or_mapped_colours():
    sheets, parts = _character()
    out, _ = AUG.augment(sheets, parts, np.random.default_rng(0), 1.0, transforms=("posterise",))
    before = np.unique(sheets["walk"][..., :3][sheets["walk"][..., 3] > 0], axis=0)
    after = np.unique(out["walk"][..., :3][out["walk"][..., 3] > 0], axis=0)
    assert len(after) <= len(before)
    assert all((before == c).all(1).any() for c in after)


def test_probability_zero_is_identity():
    sheets, parts = _character()
    out, names = AUG.augment(sheets, parts, np.random.default_rng(0), 0.0)
    assert names == [] and out is sheets


def test_mirror_swaps_left_and_right():
    sheets, parts = _character()
    sheets["walk"][1, :, 0, 0] = (9, 9, 9, 255)  # mark the left row
    out, _ = AUG.augment(sheets, parts, np.random.default_rng(0), 1.0, transforms=("mirror",))
    assert (out["walk"][3, :, 0, -1] == (9, 9, 9, 255)).all()
