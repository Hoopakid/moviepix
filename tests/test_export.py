"""The ONNX export decides exactly like PyTorch one-shot colour decoding (Phase 3).

Needs the export extra:  uv run --extra export pytest tests/test_export.py
"""
import json

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from moviepix import actions as A  # noqa: E402
from moviepix.export.onnx import OneShotExport, actions_info, colour_ids, onnx_inputs, to_uint8  # noqa: E402
from moviepix.models.registry import build  # noqa: E402


def sprite(rng, size=16, n_colours=5):
    """A blob of a few flat colours on a transparent background."""
    img = np.zeros((size, size, 4), np.uint8)
    palette = rng.integers(0, 256, (n_colours, 3))
    y0, x0 = size // 4, size // 3
    img[y0:size - 2, x0:size - x0, :3] = palette[rng.integers(0, n_colours, (size - 2 - y0, size - 2 * x0))]
    img[y0:size - 2, x0:size - x0, 3] = 255
    return img


@pytest.mark.parametrize("alpha_head", [True, False])
def test_onnx_matches_pytorch_colour_decoding(tmp_path, alpha_head):
    torch.manual_seed(0)
    model = build("copy_animator", size=16, alpha_head=alpha_head).eval()
    path = tmp_path / "m.onnx"
    S, K = 16, 8
    ref = torch.zeros(1, 4, S, S)
    ref[0, 3, 4:14, 5:11] = 1
    ids = torch.where(ref[0, 3].flatten() > 0.5, torch.randint(0, K, (S * S,)), -1)
    torch.onnx.export(OneShotExport(model, K).eval(), (ref, torch.tensor([0]), torch.tensor([2]), torch.arange(4), ids),
                      str(path), input_names=["reference", "action", "direction", "frame_ids", "colour_id"],
                      output_names=["frames"], dynamic_axes={"frame_ids": {0: "T"}, "frames": {0: "T"}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    same = []
    for action, direction in [("walk", "down"), ("slash", "left"), ("hurt", "down"), ("climb", "up")]:
        img = sprite(rng)
        feeds = onnx_inputs(img, action, direction, K)
        o = to_uint8(sess.run(None, feeds)[0])
        with torch.no_grad():
            t = model.generate(torch.from_numpy(feeds["reference"]), torch.from_numpy(feeds["action"]),
                               torch.from_numpy(feeds["direction"]), torch.tensor([A.get(action).frames]),
                               steps=1, aggregate=True)[0]
        assert o.shape == (A.get(action).frames, S, S, 4)  # the frame axis is dynamic
        same.append((o == to_uint8(t.numpy())).all(-1).mean())
    assert np.mean(same) >= 0.999


def test_colour_ids_number_by_first_appearance():
    img = np.zeros((2, 3, 4), np.uint8)
    img[0, 0] = [9, 9, 9, 255]
    img[0, 2] = [1, 2, 3, 255]
    img[1, 1] = [9, 9, 9, 255]
    img[1, 2] = [1, 2, 3, 100]  # alpha < 128 counts as transparent, as data/packs.rgba
    assert colour_ids(img, 4).tolist() == [0, -1, 1, -1, 0, -1]
    with pytest.raises(ValueError, match="at most 1"):
        colour_ids(img, 1)


def test_actions_json_lists_every_action():
    info = actions_info(32, 64)
    assert info["pivot"] == [16, 30]
    assert [a["name"] for a in info["actions"]] == [a.name for a in A.ACTIONS]
    assert next(a for a in info["actions"] if a["name"] == "hurt")["directions"] == ["down"]
    json.dumps(info)
