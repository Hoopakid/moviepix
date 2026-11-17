"""The action vocabulary: every animation moviepix knows, and its shape.

This is the single place to add or change an action. Datasets, models and the
web demo read it, so a new action needs no other code change. Ids are stable:
append new actions, never renumber, or old checkpoints decode the wrong action.
"""

from __future__ import annotations

from dataclasses import dataclass

# Row order of a sprite sheet, as in the LPC art.
DIRECTIONS = ("up", "left", "down", "right")


@dataclass(frozen=True)
class Action:
    id: int
    name: str
    frames: int  # frames in one cycle, as drawn in the source art
    directions: tuple[str, ...] = DIRECTIONS
    loops: bool = True  # walk loops; hurt plays once


ACTIONS: tuple[Action, ...] = (
    Action(0, "walk", 9),
    Action(1, "idle", 2),
    Action(2, "run", 8),
    Action(3, "slash", 6, loops=False),
    Action(4, "thrust", 8, loops=False),
    Action(5, "spellcast", 7, loops=False),
    Action(6, "shoot", 13, loops=False),
    Action(7, "hurt", 6, directions=("down",), loops=False),
    Action(8, "climb", 6, directions=("up",)),
    Action(9, "jump", 5, loops=False),
    Action(10, "sit", 3, loops=False),
    Action(11, "emote", 3, loops=False),
    Action(12, "combat_idle", 2),
    Action(13, "backslash", 13, loops=False),
    Action(14, "halfslash", 6, loops=False),
)

BY_NAME = {a.name: a for a in ACTIONS}
MAX_FRAMES = max(a.frames for a in ACTIONS)


def get(name: str) -> Action:
    try:
        return BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown action {name!r}; known: {', '.join(BY_NAME)}") from None
