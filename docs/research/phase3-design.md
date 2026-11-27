# Phase 3 design: ONNX export and browser demo (draft 2026-11-27, to agree before coding)

Model: **E1b** (`data/runs/20261124-1816-c32v3_gentle`, 32 px). E3 did not pass (phase2-design.md), so E1b stays
the reference. 64 px is Phase 4; the export and the page take the size from the model, so nothing here is 32-only.

## What exists (probe 2026-11-27, CPU only)

`moviepix-export` (src/moviepix/export/onnx.py, milestone-5 prototype) still exports E1b:

| | |
|---|---|
| Parameters | 4.41M |
| model.onnx (fp32) | 17.4 MB |
| ONNX vs PyTorch, one-shot per-pixel decoding | identical on 64 validation examples |
| CPU latency, one 8-frame walk, onnxruntime 4 threads | ~60 ms median (target: < 200 ms) |

Gap: the exporter decides **per pixel**. E1b's best decoding decides **by colour** (`_choose(aggregate=True)`: probabilities of
reference pixels with the same colour are summed), worth about +0.009 exact on LPC validation (0.782 vs ~0.773).

## Decisions to agree

1. **Colour decoding inside the graph.** The page computes a colour id per reference pixel (`color_id [S*S]`, -1 for
   transparent; at most K = 256 colours, refused above) and passes it in. The graph turns it into a one-hot
   [P, K] and sums copy probabilities per colour with one matmul, as `_choose(aggregate=True)` does. Static shapes,
   no data-dependent `unique` in the graph. Extra cost ~T*P*P*K multiply-adds (~0.5 GFLOP at 32 px, K = 64).
   *Alternative:* keep per-pixel decoding (simpler, -0.009 exact).
2. **Parity marks** (tests/test_export.py, run in CI when onnxruntime is installed):
   - fp32 ONNX vs PyTorch colour decoding: identical on >= 99.9% of pixels over 256 validation examples.
   - fp16 and int8 (dynamic, weights only): LPC exact within 0.005 of fp32, else that precision is not shipped.
   - Report size and CPU latency for each (local-first.md).
3. **Runtime:** ONNX Runtime Web, WASM by default, WebGPU when the browser has it. A pinned version, **vendored**
   into `web/vendor/` (no CDN at run time: "no network after download").
4. **Page:** plain HTML + JS, no framework, no build step (`web/`). One page:
   - drop or pick a PNG sprite (one standing character facing down; 8-64 px);
   - placement exactly as `data/packs.py`: feet on (16, 30) at 32 px, 2x pixel-art shrink if too tall, refused if
     still too big; the placement is shown so the user sees what the model got;
   - pick action and direction (from `actions.json`, written by the exporter: names, frame counts, directions,
     which actions are down-only/up-only);
   - result: looping preview at 1-8x zoom, frame strip, download as a PNG sprite sheet (one row) and GIF;
   - a built-in self-test: one stored reference and its expected frames (exported by Python) are run on load in
     a debug mode; a mismatch shows a warning.
   Nothing leaves the browser: no upload, analytics, fonts or CDN.
5. **Not published in Phase 3.** The page runs locally (`python -m http.server -d web`). Hosting the weights (GitHub
   Pages, Hugging Face, itch.io) is a release, so it waits for the Phase 5 licence and memorisation checks
   (including open blocker decision 0019: LPC licences verified upstream). `*.onnx` is already git-ignored.

## Steps (all CPU or short GPU, no long runs)

1. Exporter: colour ids input, `actions.json`, self-test fixture; test_export.py parity (fp32).
2. fp16 / int8 variants; size, latency, parity table appended here.
3. `web/`: page, placement in JS mirrored from data/packs.py, with a Python-generated placement fixture so both
   agree on a set of odd-sized sprites.
4. Check in Safari and Chrome on this Mac (owner opens the page; I give the steps); record WASM and WebGPU latency.

Out of scope: editing tools (fixed pixels via canvas/known), text prompts, Aseprite/Godot plugins (after release).

## Agreed 2026-11-27

The owner reviewed the prototype canvas (page mock, pipeline, files and steps) and agreed to the design as drafted.

## Step 1 result (2026-11-27): exporter with colour decoding

`moviepix-export data/runs/20261124-1816-c32v3_gentle --out-dir web/model --verify 256` (CPU, 4 threads, ~1 min).
Writes `model.onnx`, `actions.json` (all 15 actions, pivot, colour limit) and `selftest.json`; `web/model/` is git-ignored.

| Check | Result | Mark |
|---|---|---|
| Pixels identical to PyTorch `generate(steps=1, aggregate=True)`, 256 LPC validation examples | 99.99997% (255 of 256 animations identical; one differs by a pixel) | >= 99.9%: pass |
| Exact on the truth's drawn pixels, ONNX vs PyTorch | 0.7765 vs 0.7765 | same |
| model.onnx size (fp32) | 17.4 MB | |
| CPU latency, onnxruntime 4 threads: walk (9 frames) / shoot (13) | 109 ms / 123 ms | < 200 ms: pass |
| Colours per reference sprite (limit 64): LPC / packs32 / final32 | median 29 / 12 / 11, max 50 / 31 / 20 | all fit |

Colour decoding costs ~45 ms over the per-pixel graph (the [T, P, P] softmax and the [P, K] matmul); still inside the
target. `tests/test_export.py` checks the same parity on a random small model (with and without the alpha head),
the colour numbering, and actions.json, so it runs without the checkpoint.
