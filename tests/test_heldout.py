import json

import numpy as np
import pytest

from moviepix.data import atlas
from moviepix.data.dataset import TrainSet, collate_groups
from moviepix.data.heldout import DEV_ARTISTS, FINAL_ARTISTS, TRAIN_ARTISTS, check_manifest


def entry(artist):
    return {"id": "x", "attributes": {"artist": artist} if artist else {}}


def test_lists_never_overlap():
    norm = lambda xs: {"".join(c for c in x if c.isalnum()) for x in xs}  # noqa: E731
    assert not norm(TRAIN_ARTISTS) & (norm(DEV_ARTISTS) | norm(FINAL_ARTISTS))
    assert not norm(DEV_ARTISTS) & norm(FINAL_ARTISTS)


@pytest.mark.parametrize("artist", [*DEV_ARTISTS, *FINAL_ARTISTS, "Ansimuz", "Some Stranger", None])
def test_held_out_or_unknown_artists_are_refused(artist):
    with pytest.raises(ValueError):
        check_manifest([entry("Lexlom"), entry(artist)])


def test_training_artists_pass():
    check_manifest([entry("Lexlom"), entry("Ordinary Bumblebee"), entry("JIK-A-4"), entry("Foozle")])


def write(root, cid, frames, **extra):
    rng = np.random.default_rng(1)
    sheet = np.zeros((4, frames, 32, 32, 4), np.uint8)
    sheet[:, :, 10:20, 10:20, :3] = rng.integers(0, 255, (4, frames, 10, 10, 3))
    sheet[:, :, 10:20, 10:20, 3] = 255
    img, layout = atlas.pack({"walk": sheet})
    (root / "atlas").mkdir(parents=True, exist_ok=True)
    (root / "atlas" / f"{cid}.png").write_bytes(atlas.encode_png(img))
    with open(root / "manifest.jsonl", "a") as f:
        f.write(json.dumps({"id": cid, "split": "train", "atlas": f"atlas/{cid}.png", "size": 32,
                            "layout": layout, "attributes": {}, **extra}) + "\n")


def test_extra_characters_are_mixed_in_and_drop_their_standing_frame(tmp_path):
    main, packs = tmp_path / "main", tmp_path / "packs"
    for i in range(4):
        write(main, f"m{i}", 9)
    write(packs, "p0", 5, attributes={"artist": "Lexlom"}, leading_stand=["walk"])
    ts = TrainSet(main, per_character=2, extra=packs, extra_share=1.0)
    g = ts[0]
    assert [e["n_frames"] for e in g] == [4, 4]
    assert collate_groups([ts[0], ts[1]]).target.shape[0] == 4
    ts0 = TrainSet(main, per_character=2, extra=packs, extra_share=0.0)
    assert [e["n_frames"] for e in ts0[0]] == [9, 9]


def test_extra_dataset_with_a_held_out_artist_cannot_load(tmp_path):
    main, packs = tmp_path / "main", tmp_path / "packs"
    write(main, "m0", 9)
    write(packs, "p0", 5, attributes={"artist": "ansimuz"})
    with pytest.raises(ValueError, match="held-out"):
        TrainSet(main, extra=packs, extra_share=0.1)
