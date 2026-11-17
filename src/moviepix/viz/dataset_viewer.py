"""Make an HTML page of a built dataset for eyeballing.

    moviepix-view data/built/lpc64 [--compare data/built/lpc32] [--per-split 6]

Writes <dataset>/viewer/index.html with every action of a few characters per
split, animated, plus counts. Open it in a browser.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np

from moviepix.actions import ACTIONS
from moviepix.data.atlas import load_png, unpack
from moviepix.utils.io import read_jsonl
from moviepix.viz.gif import save_gif

STYLE = """
:root { color-scheme: light dark; --bg:#f4f4f2; --fg:#1d1d1b; --muted:#6b6b66; --card:#ffffff; --line:#deded8; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ececea; --muted:#9a9a94; --card:#20201e; --line:#33332f; } }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui, sans-serif; }
main { max-width:1200px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; }
.muted { color:var(--muted); }
table.counts td, table.counts th { padding:2px 12px 2px 0; text-align:left; }
.char { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin:12px 0; }
.char h3 { font-size:14px; margin:0 0 4px; } .items { font-size:12px; color:var(--muted); margin-bottom:8px; }
.grid { display:flex; flex-wrap:wrap; gap:10px; }
figure { margin:0; text-align:center; font-size:11px; color:var(--muted); }
img { image-rendering:pixelated; display:block; }
"""


def _direction_row(spec: dict) -> int:
    return 2 if spec["dirs"] == 4 else 0  # "down" when there are four directions


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--compare", type=Path, help="a derived dataset (e.g. lpc32) to show beside it")
    ap.add_argument("--per-split", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    entries = read_jsonl(args.dataset / "manifest.jsonl")
    info = json.loads((args.dataset / "dataset.json").read_text())
    other = {e["id"]: e for e in read_jsonl(args.compare / "manifest.jsonl")} if args.compare else {}
    out = args.dataset / "viewer"
    (out / "gif").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    action_counts = Counter(a for e in entries for a in e["layout"])
    body_counts = Counter(e["attributes"]["body_type"] for e in entries)
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{html.escape(info['name'])} dataset</title><style>{STYLE}</style></head><body><main>",
             f"<h1>{html.escape(info['name'])}</h1>",
             f"<p class='muted'>{len(entries)} characters · built from commit {html.escape(str(info.get('git')))} · "
             f"{len(info.get('held_out_items', []))} held-out items</p>",
             "<table class='counts'><tr><th>Split</th><th>Characters</th></tr>",
             *[f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(info["counts"].items())],
             "</table><table class='counts'><tr><th>Action</th><th>Characters with it</th></tr>",
             *[f"<tr><td>{a.name}</td><td>{action_counts[a.name]}</td></tr>" for a in ACTIONS],
             "</table><p class='muted'>Bodies: " + ", ".join(f"{k} {v}" for k, v in body_counts.most_common()) + "</p>"]

    by_split: dict[str, list] = {}
    for e in entries:
        by_split.setdefault(e["split"], []).append(e)
    for split in ("train", "validation", "test", "unseen"):
        pool = by_split.get(split, [])
        if not pool:
            continue
        parts.append(f"<h2>{split}</h2>")
        for i in rng.choice(len(pool), size=min(args.per_split, len(pool)), replace=False):
            e = pool[i]
            datasets = [(e, args.dataset)] + ([(other[e["id"]], args.compare)] if e["id"] in other else [])
            attrs = e["attributes"]
            items = ", ".join(f"{it['slot']}: {it['name']}" + (f" ({', '.join(it['colors'])})" if it["colors"] else "")
                              for it in attrs["items"] if it["slot"] != "body")
            parts.append(f"<div class='char'><h3>{html.escape(e['id'])} · {attrs['body_type']} · skin {attrs['skin']}</h3>"
                         f"<div class='items'>{html.escape(items)}</div>")
            for entry, root in datasets:
                size = entry["size"]
                sheets = unpack(load_png(root / entry["atlas"]), entry["layout"], size)
                scale = 128 // size
                parts.append(f"<div class='grid'>")
                for a in ACTIONS:
                    if a.name not in sheets:
                        continue
                    spec = entry["layout"][a.name]
                    frames = sheets[a.name][_direction_row(spec)]
                    name = f"{entry['id']}-{size}-{a.name}.gif"
                    save_gif(frames, out / "gif" / name, scale=scale)
                    parts.append(f"<figure><img src='gif/{name}' width='{size * scale}' height='{size * scale}' alt=''>"
                                 f"{a.name} {size}px</figure>")
                parts.append("</div>")
            parts.append("</div>")
    parts.append("</main></body></html>")
    (out / "index.html").write_text("\n".join(parts))
    print(out / "index.html")


if __name__ == "__main__":
    main()
