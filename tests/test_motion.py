import numpy as np

from moviepix.eval import motion as M


def seq():
    rng = np.random.default_rng(0)
    ref = np.full((12, 12), -1)
    ref[3:9, 4:8] = rng.integers(0, 5, (6, 4))
    truth = []
    for i in range(4):
        f = ref.copy()
        f[7:9, 4:8] = -1
        f[7:9, 3 + i:5 + i] = 9
        truth.append(f)
    return ref, truth


def test_still_scores_zero_and_truth_scores_one():
    ref, truth = seq()
    assert M.still_baseline(truth, ref)["motion_f1"] == 0.0
    s = M.score_sequence(truth, truth, ref)
    assert s["motion_f1"] == 1.0 and s["exact"] == 1.0 and s["iou"] == 1.0


def test_loop_offset_is_found_but_order_matters():
    ref, truth = seq()
    rotated = truth[2:] + truth[:2]
    assert M.score_sequence(rotated, truth, ref)["exact"] == 1.0
    shuffled = [truth[i] for i in (0, 2, 1, 3)]
    assert M.score_sequence(shuffled, truth, ref)["exact"] < 1.0
    assert M.score_sequence(rotated, truth, ref, loops=False)["exact"] < 1.0


def test_resampling_longer_prediction():
    ref, truth = seq()
    longer = [truth[i // 2] for i in range(8)]
    assert M.score_sequence(longer, truth, ref)["exact"] == 1.0


def test_tolerance_is_one_pixel():
    ref = np.full((9, 9), -1)
    ref[4, 4] = 1
    t = ref.copy(); t[2, 2] = 5
    near = ref.copy(); near[2, 3] = 5
    far = ref.copy(); far[2, 4] = 5
    assert M.f1(M.counts(near, t, ref)) == 1.0
    assert M.f1(M.counts(far, t, ref)) == 0.0


def test_extra_motion_is_punished():
    ref, truth = seq()
    noisy = [f.copy() for f in truth]
    for f in noisy:
        f[0:2] = 7
    assert M.score_sequence(noisy, truth, ref)["motion_f1"] < 0.8


def test_standing_frames_are_not_scored():
    ref, truth = seq()
    cycle = [ref, truth[0], ref, truth[2]]
    s = M.score_sequence(cycle, cycle, ref)
    assert s["exact"] == 1.0
    # a still prediction gets no credit for the standing frames
    assert M.still_baseline(cycle, ref)["exact"] == M.still_baseline([truth[0], truth[2]], ref)["exact"]


def test_nudge_wins_on_a_pure_bob():
    ref, _ = seq()
    bob = [ref, M.shift(ref, -1, 0), ref, M.shift(ref, -1, 0)]
    assert M.nudge_baseline(bob, ref)["motion_f1"] == 1.0
    assert M.nudge_baseline([M.shift(ref, 1, 0)] * 2, ref)["exact"] < M.nudge_baseline([M.shift(ref, 1, 0)] * 2, ref, oracle=True)["exact"]
