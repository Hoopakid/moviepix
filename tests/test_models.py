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


def test_copy_animator_decoding_modes():
    torch.manual_seed(0)
    model = build("copy_animator", size=32, steps=4).eval()
    b = _batch(32)
    one = model.generate(b.reference, b.action, b.direction, b.n_frames, steps=1)
    many = model.generate(b.reference, b.action, b.direction, b.n_frames, steps=4)
    assert one.shape == many.shape == (3, 13, 4, 32, 32)
    # every opaque output pixel is a reference color or a "new" color; alpha is binary
    assert set(many[..., 3, :, :].unique().tolist()) <= {0.0, 1.0}

    # editing: pixels given as known come back unchanged
    canvas = b.target.clone()
    known = torch.zeros(3, 13, 1, 32, 32)
    known[:, :, :, :16] = 1
    edited = model.generate(b.reference, b.action, b.direction, b.n_frames, steps=4, canvas=canvas, known=known)
    for i, n in enumerate(b.n_frames.tolist()):
        assert torch.equal(edited[i, :n, :, :16], canvas[i, :n, :, :16])

    # variations: sampling with different seeds differs, same seed repeats
    g = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
    v1 = model.generate(b.reference, b.action, b.direction, b.n_frames, steps=4, temperature=1.0, generator=g(1))
    v2 = model.generate(b.reference, b.action, b.direction, b.n_frames, steps=4, temperature=1.0, generator=g(2))
    v1b = model.generate(b.reference, b.action, b.direction, b.n_frames, steps=4, temperature=1.0, generator=g(1))
    assert not torch.equal(v1, v2) and torch.equal(v1, v1b)


def test_coherence_counts():
    from moviepix.eval.metrics import coherence_counts

    f = torch.zeros(1, 4, 8, 8)
    f[0, 3, 1:4, 1:4] = 1          # a 3x3 block, one colour
    f[0, :3, 1:4, 1:4] = 0.5
    f[0, :3, 2, 2] = 0.9           # a speckle in the middle
    f[0, 3, 6, 6] = 1              # a stray pixel
    stray, speckle = coherence_counts(f)
    assert stray == 1 and speckle == 1


def test_copy_animator_decoding_orders():
    torch.manual_seed(0)
    model = build("copy_animator", size=32, steps=4).eval()
    b = _batch(32)
    for order in model.ORDERS:
        out = model.generate(b.reference, b.action, b.direction, b.n_frames, steps=4, order=order)
        assert out.shape == (3, 13, 4, 32, 32)
        assert set(out[:, :, 3].unique().tolist()) <= {0.0, 1.0}


def test_copy_animator_alpha_head_and_self_conditioning():
    torch.manual_seed(0)
    model = build("copy_animator", size=32, steps=2, alpha_head=True, self_cond_prob=0.6)
    b = _batch(32)
    loss, logs = model.loss(b)
    loss.backward()
    assert torch.isfinite(loss) and logs["alpha_bce"] > 0 and logs["self_cond_share"] > 0
    assert model.alpha.weight.grad is not None
    model.eval()
    out = model.generate(b.reference, b.action, b.direction, b.n_frames, steps=2)
    assert out.shape == (3, 13, 4, 32, 32) and set(out[:, :, 3].unique().tolist()) <= {0.0, 1.0}


def test_color_aggregation_prefers_a_spread_colour_over_transparent():
    # one colour spread over many reference pixels must beat a single larger slot
    model = build("copy_animator", size=32).eval()
    ref = torch.zeros(1, 4, 32, 32)
    ref[0, :3, :10, :10] = 0.5
    ref[0, 3, :10, :10] = 1  # 100 pixels, one colour
    keys, colors, valid = model.encode_reference(ref)
    ids, rep, cvalid = model.color_groups((keys, colors, valid))
    assert int(cvalid.sum()) == 1 and int(ids[valid].max()) == 0
    probs = torch.full((1, 1, 2 + valid.shape[1]), 0.0)
    probs[..., 0] = 0.3                      # transparent
    probs[..., 2:] = 0.7 / valid.shape[1]    # the colour, spread
    agg = probs.new_zeros(1, 1, 2).scatter_add_(-1, ids[:, None, :], probs[..., 2:])
    assert float(agg[..., 0]) > 0.3          # aggregated colour wins
    assert int(probs.argmax(-1)) == 0        # per-pixel argmax would pick transparent


def test_detail_metrics():
    from moviepix.eval.metrics import detail_counts

    t = torch.zeros(1, 4, 8, 8)
    t[0, 3, 1:6, 1:6] = 1
    t[0, :3, 1:6, 1:6] = 0.5
    t[0, :3, 3, 3] = 0.9                  # a 1-pixel detail
    p = t.clone()
    p[0, :3, 3, 3] = 0.5                  # erased
    p[0, :3, 3, 4] = 0.9                  # ...but drawn 1 px to the right: still counts
    true_n, rec, pred_n, ok = detail_counts(p, t)
    assert true_n == 1 and rec == 1 and pred_n == 1 and ok == 1
    p2 = t.clone(); p2[0, :3, 3, 3] = 0.5  # erased, nowhere else
    assert detail_counts(p2, t)[1] == 0
