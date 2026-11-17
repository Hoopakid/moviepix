"""Model registry: configs name a model, this turns the name into a class."""

from __future__ import annotations

MODELS: dict[str, type] = {}


def register(name: str):
    def wrap(cls):
        MODELS[name] = cls
        return cls
    return wrap


def build(name: str, **config):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(MODELS) or 'none yet'}")
    return MODELS[name](**config)
