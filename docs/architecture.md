# Architecture

moviepix turns one pixel-art sprite into an animation: `sprite + action + direction -> frames`.

## Pipeline

```
sources ──► data build ──► train ──► eval ──► export ──► web
(LPC; later   (64 and 32,     (any model   (metrics,  (ONNX)   (browser demo)
 other sets)   shards+credits) by name)     reports)
```

Stages talk only through files on disk, so each can be rerun or replaced alone.

| Stage | Code | Reads | Writes |
|---|---|---|---|
| Sources | `sources/` | raw art in `data/raw/` | `Character` objects (in memory) |
| Build | `data/build.py` | a data config | `data/built/<name>/`: shards, `manifest.jsonl`, `CREDITS.csv`, `dataset.json` |
| Train | `train/loop.py` | a model config + built dataset | `data/runs/<date>-<name>/`: config, git hash, dataset id, seed, log, checkpoints |
| Eval | `eval/report.py` | checkpoint + split | metrics JSON + example GIFs |
| Export | `export/onnx.py` | checkpoint | `model.onnx` + `actions.json` |
| Web | `web/` | ONNX + actions.json | static demo |

## Contracts

- **Actions** (`actions.py`): the only list of actions, frame counts and directions. Ids are append-only.
- **Sample** (`data/schema.py`): `reference [S,S,4]`, `frames [T,S,S,4]`, `action`, `direction`, `attributes`. Alpha is 0 or 255.
- **Source** (`sources/base.py`): `characters(count, seed) -> Iterator[Character]`. New art sets are new sources.
- **Animator** (`models/interface.py`): `forward(reference, action, direction, n_frames, text=None)`, `loss(batch)`, `generate(...)`. New models are one file plus `@register`.

## Extension points

| Future change | Where it goes |
|---|---|
| New resolution | a data config; models read `size` from the dataset |
| New action | one line in `actions.py` |
| New art source | a folder in `sources/` |
| New model | a file in `models/` + a config in `configs/model/` |
| Text -> sprite stage | captions from stored `attributes`; the `text=` input is reserved |
| GPU training | `device = "auto"` picks MPS, CUDA or CPU |

## Milestones

1. **Data**: LPC source, build at 64 and 32, splits, credits, viewer.
2. **Baselines**: dataset loader, training loop, U-Net and palette-token baselines, metrics.
3. **Main model**: copy-based animator, inference CLI.
4. **Generalization**: augmentations, out-of-style tests, tuning.
5. **Release**: ONNX export, web demo, model and dataset cards, write-up.

## Rules

- The LPC generator repository is read as **data only** (JSON + PNG). None of its GPL code is copied or imported.
- Every built sample keeps the ids of the art it came from; every build writes `CREDITS.csv`.
- Splits are by character, never by frame, so no character appears in two splits.
