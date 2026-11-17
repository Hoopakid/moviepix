"""Splits by character, never by frame, plus an "unseen items" split.

- train / validation / test: characters made only from non-held-out items,
  assigned by a stable hash of the character id.
- unseen: characters that each wear at least one held-out item, which no
  train character ever wears. This is the honest generalization test.
"""

from __future__ import annotations

import hashlib

import numpy as np


def held_out_items(item_ids: list[str], fraction: float, seed: int) -> frozenset[str]:
    rng = np.random.default_rng([seed, 7])
    n = round(len(item_ids) * fraction)
    return frozenset(rng.choice(sorted(item_ids), size=n, replace=False).tolist()) if n else frozenset()


def assign(character_id: str, validation: float, test: float) -> str:
    h = int(hashlib.sha256(character_id.encode()).hexdigest()[:12], 16) / 16**12
    if h < validation:
        return "validation"
    if h < validation + test:
        return "test"
    return "train"
