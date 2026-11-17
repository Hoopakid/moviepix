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

    def characters(self, count: int, seed: int) -> Iterator[Character]:
        """Yield `count` characters, reproducibly for a given seed."""
        ...


def get(name: str, **options) -> Source:
    if name not in SOURCES:
        raise KeyError(f"unknown source {name!r}; known: {', '.join(SOURCES)}")
    return SOURCES[name](**options)
