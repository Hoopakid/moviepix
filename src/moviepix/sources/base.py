"""Source interface: anything that can produce animated characters.

LPC is the first source. Later ones (other CC0 sprite sets, distilled video
models) go in their own folder under sources/ and register here; the dataset
builder only talks to this interface.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from moviepix.data.schema import Character

SOURCES: dict[str, type] = {}


def register(name: str):
    def wrap(cls):
        SOURCES[name] = cls
        return cls
    return wrap


class Source(Protocol):
    name: str

    def item_ids(self) -> list[str]:
        """Parts characters are built from that may be held out for the
        "unseen items" test split (hair styles, shirts, weapons ...)."""
        ...

    def character(self, index: int, seed: int, exclude: frozenset[str] = frozenset(),
                  require_any: frozenset[str] = frozenset()) -> Character | None:
        """Make character number `index`, reproducibly for a given seed.

        Items in `exclude` are never used. With `require_any`, return a
        character that uses at least one of those items, or None if this
        index does not produce one (the builder then tries the next index).
        """
        ...


def get(name: str, **options) -> Source:
    if name not in SOURCES:
        raise KeyError(f"unknown source {name!r}; known: {', '.join(SOURCES)}")
    return SOURCES[name](**options)


def load_all() -> None:
    """Import the built-in sources so they register themselves."""
    from moviepix.sources import lpc  # noqa: F401
