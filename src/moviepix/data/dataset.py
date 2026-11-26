"""PyTorch datasets over a built dataset (data/built/<name>).

A training example is (character, action, direction):
    reference  the character standing still, facing that direction
               (frame 0 of its walk cycle, which is the standing pose in LPC)
    target     the action's frames in that direction, padded to MAX_FRAMES

Decoding an atlas PNG is the slow part, so the training set decodes a
character once and draws several examples from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from moviepix import actions as A
from moviepix.data.atlas import load_gray, load_png, unpack, unpack_parts
from moviepix.data.augment import augment
from moviepix.data.heldout import check_manifest
from moviepix.models.interface import Batch
from moviepix.utils.io import read_jsonl


def _examples_of(entry: dict) -> list[tuple[str, int]]:
    """(action, row) pairs available for a character; row indexes the action's directions.

    A 1-frame action is a reference only (a pack walk with no drawn cycle), never a target."""
    out = []
    for name, spec in entry["layout"].items():
        if spec["frames"] < 2:
            continue
        for row in range(spec["dirs"]):
            out.append((name, row))
    return out


def _direction(action: str, row: int) -> str:
    return A.get(action).directions[row]


def make_example(sheets: dict, action: str, row: int, size: int, keep: np.ndarray | None = None) -> dict:
    """keep: optional frame indices of the action to use (fewer-frame animations)."""
    direction = _direction(action, row)
    walk_row = A.DIRECTIONS.index(direction)
    reference = sheets["walk"][walk_row, 0]
    frames = sheets[action][row]
    if keep is not None:
        frames = frames[keep]
    t = frames.shape[0]
    target = np.zeros((A.MAX_FRAMES, size, size, 4), np.uint8)
    target[:t] = frames
    return {
        "reference": reference,
        "target": target,
        "action": A.get(action).id,
        "direction": walk_row,
        "n_frames": t,
    }


def collate(examples: list[dict]) -> Batch:
    def img(x):  # uint8 [..., S, S, 4] -> float [..., 4, S, S] in 0..1
        return torch.from_numpy(np.stack(x)).movedim(-1, -3).float() / 255

    return Batch(
        reference=img([e["reference"] for e in examples]),
        action=torch.tensor([e["action"] for e in examples]),
        direction=torch.tensor([e["direction"] for e in examples]),
        n_frames=torch.tensor([e["n_frames"] for e in examples]),
        target=img([e["target"] for e in examples]),
    )


def collate_groups(groups: list[list[dict]]) -> Batch:
    return collate([e for g in groups for e in g])


class _Base(Dataset):
    def __init__(self, root: Path | str, split: str, actions: list[str] | None = None):
        self.root = Path(root)
        entries = [e for e in read_jsonl(self.root / "manifest.jsonl") if e["split"] == split]
        if actions:
            keep = set(actions) | {"walk"}
            for e in entries:
                e["layout"] = {k: v for k, v in e["layout"].items() if k in keep}
        self.entries = [e for e in entries if "walk" in e["layout"]]
        self.actions = set(actions) if actions else None
        if not self.entries:
            raise ValueError(f"no characters in split {split!r} of {self.root}")
        self.size = self.entries[0]["size"]

    def _sheets(self, entry: dict) -> dict:
        return unpack(load_png(self.root / entry["atlas"]), entry["layout"], self.size)

    def _parts(self, entry: dict) -> dict | None:
        if "parts" not in entry:
            return None
        return unpack_parts(load_gray(self.root / entry["parts"]), entry["layout"], self.size)

    def _wanted(self, entry: dict) -> list[tuple[str, int]]:
        ex = _examples_of(entry)
        return [x for x in ex if self.actions is None or x[0] in self.actions]


class TrainSet(_Base):
    """Item i = `per_character` random examples from character i (use collate_groups)."""

    def __init__(self, root, split="train", actions=None, per_character: int = 4, seed: int = 0,
                 augment: float = 0.0, subsample: float = 0.0, augment_gentle: bool = False,
                 extra=None, extra_share: float = 0.0):
        super().__init__(root, split, actions)
        self.per_character = per_character
        self.seed = seed
        self.epoch = 0
        self.augment = augment  # share of characters that get style transforms (data/augment.py)
        self.subsample = subsample  # share of walk/run examples cut to 3-4 evenly spaced frames
        self.augment_gentle = augment_gentle  # no extreme transform settings (E1b)
        # Real characters in other styles (E3): a second built dataset that supplies this
        # share of the training characters. Its artists are checked against data/heldout.py.
        self.extra = _Base(extra, "train", actions) if extra is not None and extra_share > 0 else None
        self.extra_share = extra_share
        if self.extra is not None:
            check_manifest(self.extra.entries)
            if self.extra.size != self.size:
                raise ValueError(f"extra dataset is {self.extra.size} px, main is {self.size} px")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> list[dict]:
        rng = np.random.default_rng([self.seed, self.epoch, i])
        base, entry = self, self.entries[i]
        if self.extra is not None and rng.random() < self.extra_share:
            base, entry = self.extra, self.extra.entries[int(rng.integers(len(self.extra.entries)))]
        options = base._wanted(entry)
        picks = rng.choice(len(options), size=min(self.per_character, len(options)), replace=False)
        if len(picks) < self.per_character:  # a pack character may have few sequences: repeat some
            picks = rng.choice(len(options), size=self.per_character, replace=True)
        sheets = base._sheets(entry)
        if self.augment > 0:
            sheets, _ = augment(sheets, base._parts(entry), rng, self.augment, gentle=self.augment_gentle)
        out = []
        for p in picks:
            action, row = options[p]
            keep = None
            n = sheets[action].shape[1]
            if action in entry.get("leading_stand", ()):  # frame 0 is only the reference, not part of the cycle
                keep = np.arange(1, n)
            elif action in ("walk", "run") and n >= 6 and rng.random() < self.subsample:
                m = int(rng.integers(3, 5))
                keep = np.round(np.linspace(1, n, m, endpoint=False)).astype(int)  # skip the standing frame 0
            out.append(make_example(sheets, action, row, self.size, keep))
        return out


class EvalSet(_Base):
    """A fixed list of examples, the same every time (for validation and test).

    `characters` limits how many characters are used and `per_character` how
    many (action, direction) examples each gives; None means all.
    """

    def __init__(self, root, split="validation", actions=None, characters: int | None = None,
                 per_character: int | None = None, seed: int = 0):
        super().__init__(root, split, actions)
        rng = np.random.default_rng(seed)
        chars = range(len(self.entries))
        if characters is not None and characters < len(self.entries):
            chars = sorted(rng.choice(len(self.entries), size=characters, replace=False))
        items = []
        for ci in chars:
            options = self._wanted(self.entries[ci])
            if per_character is not None and per_character < len(options):
                options = [options[j] for j in sorted(rng.choice(len(options), size=per_character, replace=False))]
            items += [(ci, a, r) for a, r in options]
        self.items = items
        self._cache: tuple[int, dict] | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        ci, action, row = self.items[i]
        if self._cache is None or self._cache[0] != ci:  # items are sorted by character
            self._cache = (ci, self._sheets(self.entries[ci]))
        ex = make_example(self._cache[1], action, row, self.size)
        ex["character"] = self.entries[ci]["id"]
        return ex
