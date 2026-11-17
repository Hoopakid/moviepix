import importlib
import pkgutil

import numpy as np
import pytest

import moviepix
from moviepix import actions
from moviepix.data.schema import Sample


def test_all_modules_import():
    skip = {"moviepix.models.interface", "moviepix.utils.device"}  # need torch
    for mod in pkgutil.walk_packages(moviepix.__path__, "moviepix."):
        if mod.name not in skip:
            importlib.import_module(mod.name)


def test_action_ids_are_stable_and_unique():
    assert [a.id for a in actions.ACTIONS] == list(range(len(actions.ACTIONS)))
    assert len(actions.BY_NAME) == len(actions.ACTIONS)
    assert actions.get("walk").frames == 9
    with pytest.raises(KeyError):
        actions.get("moonwalk")


@pytest.mark.parametrize("size", [64, 32])
def test_sample_validates(size):
    ref = np.zeros((size, size, 4), np.uint8)
    frames = np.zeros((9, size, size, 4), np.uint8)
    frames[0, 0, 0] = (10, 20, 30, 255)
    Sample("c0", "walk", "down", ref, frames).validate()
    frames[0, 0, 0, 3] = 128
    with pytest.raises(AssertionError):
        Sample("c0", "walk", "down", ref, frames).validate()
