"""Out-of-style test, version 2: in-order motion scoring (replaces oos_walk.py + oos_motion.py).

    uv run python scripts/oos_score.py RUN [RUN ...] [--device mps] [--against RUN_NAME]

Sequences: walks of all 5 development packs, plus Puny sword/bow/hurt and Chasersgaming
run/attack/climb. Scores: src/moviepix/eval/motion.py. Baselines: the still sprite;
"nudge" (still sprite moved up 1 px) and "nudge*" (best of 4 directions, peeking at the truth). Only walks carry the pass marks; the other actions are
reported, because weapons appear that are not in the reference, the Chasersgaming attack
cell offset is estimated, and the Puny hurt/death boundary is ambiguous.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from oos_walk import DIRS, OOS, SCALE, crop, place, rgba, to_t, walks  # noqa: E402

from moviepix import actions as A  # noqa: E402
from moviepix.data.atlas import load_png, unpack  # noqa: E402
from moviepix.eval import motion as M  # noqa: E402
from moviepix.train.checkpoint import load_model  # noqa: E402
from moviepix.utils.io import read_jsonl  # noqa: E402

PACKS = ("puny", "ninja", "knight", "ranger", "antifarea")
METRICS = ("motion_f1", "motion_strict", "exact", "iou", "moved_truth", "moved_pred")


def widen(ref, cell0):
    """Chasersgaming attack cells are 24 wide, the reference 16: find where the body sits
    (largest silhouette overlap with attack frame 0) and pad the reference to 24."""
    a, r = cell0[..., 3] > 0, ref[..., 3] > 0
    dx = max(range(9), key=lambda d: (a[:, d:d + 16] & r).sum())
    out = np.zeros((24, 24, 4), np.uint8)
    out[:, dx:dx + 16] = ref
    return out


def sequences():
    """Yield (pack, unit, sequence name, action, direction, reference cell, truth cells).
    A unit is one independent character design (Puny's recolours share a unit)."""
    for pack, char, d, ref, cells in walks():
        unit = char.split("-")[0] if pack == "puny" else char
        yield pack, unit, f"{char}/walk/{d}", "walk", d, ref, cells
    root = OOS / "puny-characters/x/Puny-Characters"
    for sheet in ["Archer-Green", "Archer-Purple", "Character-Base", "Mage-Cyan", "Mage-Red", "Soldier-Blue",
                  "Soldier-Red", "Soldier-Yellow", "Warrior-Blue", "Warrior-Red"]:
        img = rgba(root / f"{sheet}.png")
        unit = sheet.split("-")[0]
        for action, cols, dirs in (("slash", range(4, 8), DIRS), ("shoot", range(8, 12), DIRS),
                                   ("hurt", range(19, 24), ("down",))):
            for d in dirs:
                row = {"down": 0, "right": 2, "up": 4, "left": 6}[d]
                yield "puny", unit, f"{sheet}/{action}/{d}", action, d, crop(img, 0, 32 * row, 32, 32), \
                    [crop(img, 32 * c, 32 * row, 32, 32) for c in cols]
    D = {"up": "NORTH", "left": "WEST", "down": "SOUTH", "right": "EAST"}
    kn = OOS / "chasersgaming-knight-nes/x/NES Knight Free Version"
    ra = OOS / "chasersgaming-ranger-nes/x/NES Ranger Files FREE/NES Ranger Free Set PNG/NES Ranger Single Sheets"
    for pack, path in (("knight", lambda a, n: kn / f"{a} PNG/NES PNG Knight_{a}_{n}_strip4.png"),
                       ("ranger", lambda a, n: ra / f"{a}/NES_Ranger_{a}_{n}.png")):
        for action, folder, dirs in (("run", "Run", DIRS), ("slash", "Attack", DIRS), ("climb", "Climb", ("up",))):
            for d in dirs:
                ref = crop(rgba(path("Idle", D[d])), 0, 0, 16, 24)
                w = 24 if folder == "Attack" else 16
                cells = [crop(rgba(path(folder, D[d])), w * i, 0, w, 24) for i in range(4)]
                yield pack, pack, f"{pack}/{action}/{d}", action, d, (widen(ref, cells[0]) if w == 24 else ref), cells


def lpc_sequences(S, n):
    ds = {32: "lpc32v3", 64: "lpc64v3"}[S]
    entries = [e for e in read_jsonl(Path("data/built") / ds / "manifest.jsonl") if e["split"] == "validation"]
    for e in entries[:n]:
        sh = unpack(load_png(Path("data/built") / ds / e["atlas"]), e["layout"], S)["walk"]
        for r, d in enumerate(DIRS):
            yield "LPC", e["id"], f"{e['id']}/walk/{d}", "walk", d, sh[r, 0], list(sh[r, 1:])


def predict(model, device, ref, action, direction, n_frames=None):
    act = A.get(action)
    n = n_frames or act.frames
    with torch.no_grad():
        out = model.generate(to_t(ref).to(device), torch.tensor([act.id], device=device),
                             torch.tensor([DIRS.index(direction)], device=device),
                             torch.tensor([n], device=device), steps=1)[0]
    f = (out.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
    # a full LPC walk starts with the standing frame; a shortened one does not (dataset.py)
    return list(f[1:]) if action == "walk" and n_frames is None else list(f)


def evaluate(run, device, lpc_n, n_frames):
    model, _ = load_model(run / "best.pt", device)
    S = model.size
    rows = []
    seqs = [(p, u, n, a, d, *(lambda q: (q[0], q[1:]))(place([ref] + list(cells), ref, S, SCALE[S][p])))
            for p, u, n, a, d, ref, cells in sequences()]
    if not n_frames:
        seqs += list(lpc_sequences(S, lpc_n))
    for pack, unit, name, action, d, ref, truth in seqs:
        pal = np.unique(ref[ref[..., 3] > 0][:, :3], axis=0)
        rk, tk = M.keys(ref), [M.keys(t) for t in truth]
        if all((t == rk).all() for t in tk):
            continue
        pk = [M.keys(p, pal) for p in predict(model, device, ref, action, d, n_frames)]
        loops = A.get(action).loops
        row = {"pack": pack, "unit": unit, "name": name, "action": action}
        for who, s in (("model", M.score_sequence(pk, tk, rk, loops)), ("still", M.still_baseline(tk, rk)),
                       ("nudge", M.nudge_baseline(tk, rk)), ("nudge_oracle", M.nudge_baseline(tk, rk, oracle=True))):
            row[who] = s
        rows.append(row)
    return rows


def unit_means(rows, who, metric):
    """Mean per unit first, so recolours of one design do not count as independent characters."""
    per = {}
    for r in rows:
        per.setdefault(r["unit"], []).append(r[who][metric])
    return {u: float(np.mean(v)) for u, v in per.items()}


def table(rows, title):
    print(f"\n{title}\n{'pack':10s} {'action':7s} {'seqs':>4s} {'units':>5s} | {'mF1':>6s} {'nudge':>6s} {'nudge*':>6s} | "
          f"{'exact':>6s} {'still':>6s} | {'iou':>6s} {'still':>6s} | {'moved':>6s} {'truth':>6s}")
    groups = sorted({(r["pack"], r["action"]) for r in rows}, key=lambda g: ((*PACKS, "LPC").index(g[0]), g[1] != "walk", g[1]))
    for g in groups:
        rr = [r for r in rows if (r["pack"], r["action"]) == g]
        m = lambda who, k: np.mean(list(unit_means(rr, who, k).values()))  # noqa: E731
        print(f"{g[0]:10s} {g[1]:7s} {len(rr):4d} {len({r['unit'] for r in rr}):5d} | {m('model', 'motion_f1'):6.3f} "
              f"{m('nudge', 'motion_f1'):6.3f} {m('nudge_oracle', 'motion_f1'):6.3f} | {m('model', 'exact'):6.3f} {m('still', 'exact'):6.3f} | "
              f"{m('model', 'iou'):6.3f} {m('still', 'iou'):6.3f} | {m('model', 'moved_pred'):6.3f} {m('model', 'moved_truth'):6.3f}", flush=True)


def compare(a_rows, b_rows, pack, metric, rng, n=10000):
    """b - a on walks, paired by unit; 95% interval by bootstrap over units (needs >= 5 units)."""
    sel = lambda rows: unit_means([r for r in rows if r["pack"] == pack and r["action"] == "walk"], "model", metric)  # noqa: E731
    a, b = sel(a_rows), sel(b_rows)
    d = np.array([b[u] - a[u] for u in a])
    if len(d) < 5:
        return d.mean(), None, len(d)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, [2.5, 97.5]), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lpc-characters", type=int, default=40)
    ap.add_argument("--n-frames", type=int, default=None, help="ask for this many frames instead of the LPC count (walk only)")
    ap.add_argument("--against", help="run name to compare the others with (paired bootstrap on walk motion F1)")
    ap.add_argument("--out", type=Path, default=Path("docs/research/oos-score-results.json"))
    a = ap.parse_args()
    torch.set_num_threads(2)
    res = {}
    for run in a.runs:
        res[run.name] = evaluate(run, a.device, a.lpc_characters, a.n_frames)
        rows = [r for r in res[run.name] if r["action"] == "walk"] if a.n_frames else res[run.name]
        table(rows, run.name + (f"  (asked for {a.n_frames} frames)" if a.n_frames else ""))
    if a.against:
        rng = np.random.default_rng(0)
        for metric in ("motion_f1", "exact"):
            print(f"\nwalk {metric}: difference from {a.against}, paired by character design, 95% interval")
            for name, rows in res.items():
                if name == a.against:
                    continue
                print(f"  {name}")
                for p in (*PACKS, *([] if a.n_frames else ["LPC"])):
                    m, ci, k = compare(res[a.against], rows, p, metric, rng)
                    print(f"    {p:10s} {m:+.3f}  " + (f"[{ci[0]:+.3f}, {ci[1]:+.3f}]  ({k} designs)" if ci is not None
                                                       else f"({k} design(s): too few for an interval)"))
    a.out.write_text(json.dumps(res, indent=0))


if __name__ == "__main__":
    sys.exit(main())
