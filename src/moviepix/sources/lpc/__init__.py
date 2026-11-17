"""The LPC source: random characters composed from Liberated Pixel Cup art."""

from __future__ import annotations

import numpy as np

from moviepix.data.schema import Character
from moviepix.sources.base import register

from .catalog import Catalog
from .compose import SLOTS, Composer

# Slots whose items can be held out. Bodies and heads are too few to hold out.
HOLDOUT_SLOTS = {"hair", "beard", "torso", "torso_over", "waist", "cape", "backpack", "legs",
                 "feet", "arms", "hat", "neck", "weapon", "shield"}


@register("lpc")
class LpcSource:
    name = "lpc"

    def __init__(self, root: str = "data/raw/lpc", licenses: str = "permissive"):
        self.catalog = Catalog(root)
        self.composer = Composer(self.catalog, permissive=licenses == "permissive")

    def item_ids(self) -> list[str]:
        ids = set()
        for slot, folders, _ in SLOTS:
            if slot in HOLDOUT_SLOTS:
                ids |= {it.id for it in self.catalog.under(*folders, permissive=self.composer.permissive)}
        return sorted(ids)

    def character(self, index: int, seed: int, exclude=frozenset(), require_any=frozenset()) -> Character | None:
        rng = np.random.default_rng([seed, index])
        composed = self.composer.sample(rng, exclude=exclude)
        used = {c.item.id for c in composed.choices}
        if require_any and not used & require_any:
            return None
        if not composed.sheets:
            return None
        return Character(
            id=f"lpc-{seed}-{index}",
            sheets=composed.sheets,
            attributes=composed.attributes(),
            credits=composed.credits(),
        )
