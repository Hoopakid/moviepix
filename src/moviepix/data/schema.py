"""The one sample format every stage uses.

A sample is one character doing one action in one direction:

    reference  uint8 [S, S, 4]      the still sprite the model is given
    frames     uint8 [T, S, S, 4]   the animation it should produce
    action     str                  a name from moviepix.actions
    direction  str                  up / left / down / right

S is the dataset resolution (64 or 32). Alpha is either 0 or 255: pixel art
has no soft edges. T varies by action, so batching pads and masks.

Attributes (body type, items, colors) are stored from the start so captions
for a later text -> sprite stage can be made without rebuilding the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Sample:
    character_id: str
    action: str
    direction: str
    reference: np.ndarray
    frames: np.ndarray
    attributes: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.reference.shape[0]

    def validate(self) -> None:
        s = self.size
        assert self.reference.shape == (s, s, 4) and self.reference.dtype == np.uint8
        assert self.frames.ndim == 4 and self.frames.shape[1:] == (s, s, 4) and self.frames.dtype == np.uint8
        for a in (self.reference[..., 3], self.frames[..., 3]):
            assert np.isin(a, (0, 255)).all(), "alpha must be 0 or 255"


@dataclass
class Character:
    """What a source yields: every action sheet of one character, plus credits.

    sheets[action] is uint8 [directions, frames, 64, 64, 4], rows in
    moviepix.actions.DIRECTIONS order (or the action's own direction list).
    """

    id: str
    sheets: dict[str, np.ndarray]
    attributes: dict
    credits: list[dict]
