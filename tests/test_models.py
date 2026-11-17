import pytest
import torch

import moviepix.models  # noqa: F401
from moviepix.eval.metrics import frame_metrics, summarize
from moviepix.models.interface import Batch
from moviepix.models.registry import MODELS, build


def _batch(size, b=3):
    g = torch.Generator().manual_seed(0)
    ref = torch.rand(b, 4, size, size, generator=g).round()
    return Batch(ref, torch.tensor([0, 7, 6][:b]), torch.tensor([2, 2, 1][:b]), torch.tensor([9, 6, 13][:b]),
                 torch.rand(b, 13, 4, size, size, generator=g).round())


@pytest.mark.parametrize("name", sorted(MODELS))
@pytest.mark.parametrize("size", [32, 64])
def test_model_contract(name, size):
    model = build(name, size=size)
    batch = _batch(size)
    loss, logs = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss) and "loss" in logs
    out = model.generate(batch.reference, batch.action, batch.direction, batch.n_frames)
    assert out.shape == (3, 13, 4, size, size)
    assert out.min() >= 0 and out.max() <= 1
    assert (out[1, 6:] == 0).all()  # padding frames stay empty


def test_metrics_perfect_prediction():
    b = _batch(32)
    # make every target color one of the reference's colors
    tgt = b.reference[:, None].repeat(1, 13, 1, 1, 1)
    m = summarize(frame_metrics(tgt, tgt, b.reference, b.n_frames))
    assert m["alpha_iou"] == 1 and m["pixel_exact"] == 1 and m["rgb_l1"] == 0 and m["palette_fidelity"] == 1
