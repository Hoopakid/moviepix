# moviepix: Animating Pixel-Art Sprites by Copying Pixels

*Draft technical report — Hoopakid — work in progress, not submitted anywhere.*
*Sections marked **[TODO]** wait for running experiments or the literature review.*

## Abstract

We present moviepix, a 4.4–4.6M-parameter model that turns a single still pixel-art
sprite into a multi-frame animation (walk, run, slash, spellcast and eleven other
actions, four directions) and runs on a laptop CPU in tens of milliseconds. The key
observation is that animation frames of pixel art are almost entirely made of moved
copies of the input's pixels: in our data only about 2% of target pixels carry a colour
absent from the still sprite. moviepix therefore does not paint colours; for every
output pixel it points, through attention, to the reference pixel to copy, chooses
transparency, or (rarely) a new colour. Trained from scratch on 21,000 characters
composed from openly licensed Liberated Pixel Cup art, it reproduces 77–80% of target
pixels exactly, against 51% for a U-Net that regresses colours and 45% for a model that
classifies over the sprite's palette, while always keeping the sprite's exact palette.
We also report what did not work: confidence-ordered MaskGIT decoding made results
worse until the model was trained on its own partial predictions (self-conditioning),
and per-pixel argmax decoding systematically erases one-pixel details. All code, data
recipes, training logs and costs are public. **[TODO: final numbers after phase 1–4.]**

## 1. Introduction

- Drawing every frame of a character's animations is the slowest part of pixel-art
  game production: one character with 15 actions in 4 directions is ~340 frames.
- Existing automated tools are mostly closed, cloud-based and paid; general video
  generators are large and break pixel-art constraints (exact palette, crisp edges).
- Goal: an open model small enough for a laptop or a browser, that keeps the artist's
  own pixels, trained only on permissively licensed art with full credits.
- Contributions:
  1. A copy-based formulation of sprite animation (per-pixel pointer to the reference)
     trained with a marginal-likelihood loss over all same-coloured source pixels.
  2. A controlled comparison against colour regression and palette classification at
     equal data and compute (Section 5).
  3. An analysis of iterative masked decoding for this task: why confidence ordering
     fails, and how self-conditioning repairs it (Section 6).
  4. A fully reproducible pipeline on one consumer laptop. **[TODO: out-of-style
     results; browser deployment.]**

## 2. Related work

**[TODO — from the literature review (docs/research/).]** Planned subsections: pixel-art
generation; single-image animation and motion transfer (warping, appearance flow);
cartoon in-betweening; copy/pointer mechanisms; masked generative models (MaskGIT and
successors) and self-conditioning; commercial sprite tools.

## 3. Data

- **Source.** Universal LPC Spritesheet Character Generator assets, read as data only
  (no GPL code used). An item is used only if every credit entry offers a permissive
  licence (CC0, CC-BY, OGA-BY); multi-licensed art qualifies through its permissive
  option. 517 of 646 items pass; the muscular and pregnant bodies are excluded.
- **Composition.** A character = body type (male/female/teen) + skin palette + one
  item per slot (head, hair, torso, legs, feet, optional hats, weapons, shields, …),
  each recoloured with LPC's palette definitions; layers are stacked by z-order for
  each of 15 actions (2–13 frames; 4 directions except hurt and climb).
- **Scale.** 21,000 characters (18.8k train, ~600 validation, ~600 test, 1,000
  "unseen items": characters wearing items held out of training), ≈1M examples
  (character × action × direction); 64×64 native, 32×32 by a palette-preserving
  majority downscale. Building takes ~25 minutes on a laptop CPU.
- **A data bug we found.** Dataset v1 sampled LPC's `eyes` folder as a regular slot;
  it contains only cyclops eyes, so 70% of characters were cyclopes, and ~11% carried
  heater-shield paint layers without a shield. v2 fixes both (2.2% cyclopes). The
  32 px baselines in Section 5 use v1 for all models; later runs use v2.

## 4. Models

Input: reference sprite R (RGBA, S×S), action a, direction d, frame index t, frame
count n. Output: n frames, S×S RGBA. Alpha is binary.

- **A — U-Net regression** (2.6M). Per frame, a FiLM-conditioned U-Net maps R to RGB
  (L1 on opaque pixels) and alpha (BCE).
- **B — palette classification** (2.7M). Per pixel, a categorical over {transparent,
  new colour, colour k of R's palette}; colour k is scored by a dot product with an
  embedding of its RGB value.
- **C — copy_animator** (4.4M at 32 px, 4.6M at 64 px). A reference encoder (U-Net, run
  once) gives a key per opaque reference pixel (plus a learned position embedding). A
  frame decoder (FiLM U-Net over [R, partial canvas, known mask]; frames of one
  animation attend to each other at the bottleneck) gives a query per output pixel.
  Per pixel: logits over {transparent, new colour, copy pixel j} with copy logits
  q·k_j/√D. Training loss: −log Σ_{j: colour(j)=target} p(j) (all same-coloured source
  pixels count as correct), plus BCE/L1 for the rare new colours.
  - *Masked training:* a cosine-distributed share of pixels per frame is hidden (25% of
    frames fully hidden); visible pixels are given as ground truth.
  - *Self-conditioning (C+sc):* 40% of frames instead receive the model's own most
    confident one-shot predictions as the visible pixels.
  - *Alpha head (C+α):* a separate drawn/not-drawn output; the categorical then covers
    only {new, copy j}.
- **Memory.** The [frames × pixels × reference pixels] logit table dominates memory; it
  is computed in chunks of frames with activation checkpointing, so 64 px training fits
  in ~5 GB on a 16 GB laptop.

## 5. Results: copying beats painting

32×32, dataset v1, 640k training examples each, one MacBook Air (Apple M5, 16 GB).
1,024 validation examples; "unseen" = characters wearing held-out items.

| Model | Exact pixels | Exact, unseen | Silhouette IoU | Palette fidelity | Params | CPU, 8 frames |
|---|---:|---:|---:|---:|---:|---:|
| A U-Net regression | 0.506 | 0.489 | **0.989** | 0.565 | 2.6M | 11 ms |
| B palette classification | 0.451 | 0.436 | 0.968 | 0.978 | 2.7M | 23 ms |
| C copy, one-shot | **0.774** | **0.757** | 0.980 | **0.986** | 4.4M | 16 ms |
| C copy, 8 rounds | 0.765 | 0.749 | 0.973 | **0.986** | 4.4M | 119 ms |

*Exact pixels:* share of target-opaque pixels whose predicted colour (snapped to the
nearest reference colour) and alpha match. *Palette fidelity:* share of predicted opaque
pixels whose raw colour is already a reference colour.

Observations. B keeps colours exact but swaps whole regions to a colour of a similar
role (e.g., a red hood becomes blond): choosing colours *by value* does not identify
*which part* a pixel belongs to. Copying *by position* does. Generalisation to unseen
items costs C under 2 points.

**[TODO: phase 1 — c32v2_new vs c32v2_control at equal budget on dataset v2.]**

## 6. Iterative decoding and over-smoothing

Decoding orders evaluated without retraining (256 validation examples).

| Model | Rounds / order | Exact | IoU | Speckle excess / frame |
|---|---|---:|---:|---:|
| C 64 px (v2, 144k ex.) | 1 | 0.754 | 0.955 | −19.6 |
| | 8, confidence | 0.701 | 0.917 | −17.1 |
| | 8, random | 0.749 | 0.952 | −16.2 |
| | 8, balanced | 0.740 | 0.933 | −5.7 |
| C+sc+α 64 px (warm start, +64k ex.) | 1 | **0.798** | **0.964** | −12.0 |
| | 4, confidence | **0.798** | 0.963 | −11.6 |
| | 4, random | 0.793 | 0.963 | −9.9 |

- **Confidence ordering fails** because transparent background is predicted with
  near-certainty: rounds 1–7 commit background and the whole character is decided last,
  in a context (background known, character unknown) never seen in training.
- **Self-conditioning repairs it**: confidence-ordered rounds go from −5.3 points below
  one-shot to equal. They do not yet beat one-shot.
- **Over-smoothing.** "Speckle excess" counts pixels whose colour matches none of their
  four opaque neighbours, relative to ground truth. All models produce fewer such
  one-pixel details than the truth (−6 to −20 per frame): independent per-pixel argmax
  favours the majority colour of a neighbourhood. **[TODO: fixes from the literature
  review.]**
- Caveat: the C+sc+α run is warm-started from C 64 px, so its gain mixes the new recipe
  with extra training; Section 5's phase-1 runs separate the two.

## 7. Deployment

- One-shot decoding exports to ONNX (opset 17) with a dynamic frame axis; outputs are
  99.9996% pixel-identical to PyTorch. nn.MultiheadAttention had to be written out by
  hand because the exporter bakes the traced frame count into its reshapes.
- **[TODO: in-graph compaction of opaque reference pixels; int8; browser latency.]**

## 8. Limitations

LPC style only (top-down humanoids); 64×64 maximum; some source weapons lack frames
for some actions; the evaluation is automatic (no human study yet). **[TODO: out-of-
style test set results.]**

## 9. Reproducibility and cost

Everything ran on one MacBook Air (Apple M5, 16 GB, fanless): dataset 25 min CPU;
32 px models 2–5 h each; 64 px 7.5 h (from scratch) and 4.5 h (warm start). No rented
compute. Code, configs, logs and decisions: https://github.com/Hoopakid/moviepix.

## Acknowledgements

The Liberated Pixel Cup artists (full credits in each dataset build). Inspired by
tiny-voca (https://github.com/n4ze3m/tiny-voca).
