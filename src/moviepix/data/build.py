"""Build a dataset from a data config.

    moviepix-build configs/data/lpc64.toml
    moviepix-build configs/data/lpc32.toml     # derived from the lpc64 build

Output, in data/built/<name>/:
    atlas/NNN/<id>.png   one image per character with every action (data/atlas.py)
    manifest.jsonl       one line per character: id, split, atlas, layout, attributes
    CREDITS.csv          every piece of art used, with authors, licenses and links
    dataset.json         config, counts, held-out items, git commit
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import tomllib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from moviepix.data import atlas as atlas_mod
from moviepix.data.resize import downscale
from moviepix.data.splits import assign, held_out_items
from moviepix.sources import base as sources
from moviepix.utils.io import git_commit, read_jsonl, write_json

UNSEEN_OFFSET = 10_000_000  # unseen characters use their own index range

_source = None


def _init_worker(source_name: str, options: dict) -> None:
    global _source
    sources.load_all()
    _source = sources.get(source_name, **options)


def _make(task):
    index, seed, exclude, require_any = task
    ch = _source.character(index, seed, exclude=exclude, require_any=require_any)
    if ch is None:
        return None
    img, layout = atlas_mod.pack(ch.sheets)
    parts = atlas_mod.encode_gray(atlas_mod.pack_parts(ch.parts)) if ch.parts else None
    return ch.id, atlas_mod.encode_png(img), layout, ch.attributes, ch.credits, parts


def _write(out: Path, n: int, result, split: str, size: int) -> tuple[dict, list]:
    cid, png, layout, attributes, credits, parts = result
    rel = Path("atlas") / f"{n // 1000:03d}" / f"{cid}.png"
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    (out / rel).write_bytes(png)
    entry = {"id": cid, "split": split, "atlas": str(rel), "size": size, "layout": layout, "attributes": attributes}
    if parts is not None:
        prel = Path("parts") / rel.relative_to("atlas")
        (out / prel).parent.mkdir(parents=True, exist_ok=True)
        (out / prel).write_bytes(parts)
        entry["parts"] = str(prel)
    return entry, credits


def build_from_source(cfg: dict, out: Path, workers: int) -> dict:
    sources.load_all()
    options = cfg.get("source_options", {}) | {"licenses": cfg.get("licenses", "permissive")}
    probe = sources.get(cfg["source"], **options)
    seed, count = cfg["seed"], cfg["characters"]
    sp = cfg["splits"]
    held = held_out_items(probe.item_ids(), sp["unseen_items"], seed)

    manifest, credit_rows, counts = [], {}, Counter()
    t0 = time.time()

    def run(kind: str, wanted: int, start: int, exclude, require_any):
        got, nxt = 0, start
        with Pool(workers, _init_worker, (cfg["source"], options)) as pool:
            while got < wanted:
                batch = max(256, (wanted - got) * (4 if require_any else 1) + 64)
                tasks = [(i, seed, exclude, require_any) for i in range(nxt, nxt + batch)]
                nxt += batch
                for result in pool.imap(_make, tasks, chunksize=8):
                    if result is None or got >= wanted:
                        continue
                    split = "unseen" if kind == "unseen" else assign(result[0], sp["validation"], sp["test"])
                    entry, credits = _write(out, len(manifest), result, split, 64)
                    manifest.append(entry)
                    counts[split] += 1
                    for c in credits:
                        key = (c["item"], c.get("file", ""))
                        credit_rows.setdefault(key, {**c, "characters": 0})["characters"] += 1
                    got += 1
                    if got % 1000 == 0:
                        print(f"{kind}: {got}/{wanted} ({time.time() - t0:.0f}s)", flush=True)

    run("main", count, 0, held, frozenset())
    if held and sp.get("unseen_characters", 0):
        run("unseen", sp["unseen_characters"], UNSEEN_OFFSET, frozenset(), held)

    with open(out / "manifest.jsonl", "w") as f:
        for e in manifest:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    _write_credits(out / "CREDITS.csv", credit_rows.values())
    return {"counts": dict(counts), "held_out_items": sorted(held), "seconds": round(time.time() - t0)}


def _write_credits(path: Path, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "file", "authors", "licenses", "urls", "notes", "characters"])
        for r in sorted(rows, key=lambda r: (r["item"], r.get("file", ""))):
            w.writerow([r["item"], r.get("file", ""), "; ".join(r.get("authors", [])),
                        "; ".join(r.get("licenses", [])), " ".join(r.get("urls", [])),
                        r.get("notes", ""), r["characters"]])


def _derive_one(task):
    src, dst, factor, psrc, pdst = task
    img = atlas_mod.load_png(src)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if psrc is None:
        Path(dst).write_bytes(atlas_mod.encode_png(downscale(img, factor)))
        return
    small, parts = downscale(img, factor, labels=atlas_mod.load_gray(psrc))
    Path(dst).write_bytes(atlas_mod.encode_png(small))
    Path(pdst).parent.mkdir(parents=True, exist_ok=True)
    Path(pdst).write_bytes(atlas_mod.encode_gray(parts))


def build_derived(cfg: dict, out: Path, workers: int, built: Path) -> dict:
    parent = built / cfg["derive_from"]
    parent_info = json.loads((parent / "dataset.json").read_text())
    factor = parent_info["config"]["size"] // cfg["size"]
    entries = read_jsonl(parent / "manifest.jsonl")
    t0 = time.time()
    tasks = [(str(parent / e["atlas"]), str(out / e["atlas"]), factor,
              str(parent / e["parts"]) if "parts" in e else None, str(out / e["parts"]) if "parts" in e else None)
             for e in entries]
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_derive_one, tasks, chunksize=16), 1):
            if i % 2000 == 0:
                print(f"downscaled {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    with open(out / "manifest.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e | {"size": cfg["size"]}, ensure_ascii=False) + "\n")
    shutil.copy(parent / "CREDITS.csv", out / "CREDITS.csv")
    return {"counts": parent_info["counts"], "held_out_items": parent_info["held_out_items"],
            "derived_from": cfg["derive_from"], "seconds": round(time.time() - t0)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--built", type=Path, default=Path("data/built"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--characters", type=int, help="override the config (for quick test builds)")
    ap.add_argument("--unseen", type=int, help="override splits.unseen_characters")
    ap.add_argument("--name", help="override the output name")
    args = ap.parse_args(argv)

    cfg = tomllib.loads(args.config.read_text())
    if args.characters is not None:
        cfg["characters"] = args.characters
    if args.unseen is not None:
        cfg.setdefault("splits", {})["unseen_characters"] = args.unseen
    if args.name:
        cfg["name"] = args.name
    out = args.built / cfg["name"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    info = build_derived(cfg, out, args.workers, args.built) if "derive_from" in cfg \
        else build_from_source(cfg, out, args.workers)
    write_json(out / "dataset.json", {"name": cfg["name"], "config": cfg, "git": git_commit(), **info})
    print(json.dumps({"name": cfg["name"], **{k: v for k, v in info.items() if k != "held_out_items"}}))


if __name__ == "__main__":
    main()
