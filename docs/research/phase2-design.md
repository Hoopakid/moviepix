# Phase 2 design — other art styles (draft for discussion)

Status: proposal, not implemented. It follows wave 3's out-of-style walk test: the models
beat "repeat the still sprite" by 16–22 points on LPC but lose to it on Puny (chibi), and
the 64 px models' silhouettes collapse on 16×24 NES sprites. The models learned *LPC
proportions*, not *motion*.

## Goal and pass criteria

The 32 px model trained on the new data, at the same budget as `c32v2_control`, must:
1. stay within 1 point of the control on LPC validation exact pixels;
2. beat the repeat-reference baseline on **every** out-of-style pack (exact and IoU);
3. not collapse silhouettes: out-of-style IoU ≥ baseline − 0.02.

The worst pack is reported, not just the average.

## 1. Part labels (data, free from the composer)

`Composer._render` already stacks the layers in z order. It also records which slot drew
the top pixel. The result is a `parts` array of the same shape as the sheet, uint8 class
ids:

| id | part | LPC slots |
|---|---|---|
| 0 | none | — |
| 1 | skin/body | body, nose, ears |
| 2 | head features | head, eyes, facial hair |
| 3 | hair | hair |
| 4 | headwear | hats, helmets |
| 5 | torso | torso, armour, capes (front) |
| 6 | arms/hands | gloves, bracers |
| 7 | legs | legs |
| 8 | feet | shoes |
| 9 | held item | weapons, shields, tools |
| 10 | other | accessories, neck, belts |

- **Storage:** a second indexed PNG per character (`parts.png`, same layout as the atlas,
  ≈ +10% disk).
- **32 px:** use the part of the colour chosen by the block majority. Ties go to the
  colour-mean rule already used, so parts stay consistent with the downscaled pixels.
- **Uses:**
  - (a) Augmentation, section 2.
  - (b) An optional auxiliary part head on decoder queries (CE, weight 0.1). It is not an
    input at inference, since users do not have part maps.
  - (c) A "region-swap" metric: the share of predicted pixels copied from a reference
    pixel of a different part than the truth.

## 2. Style augmentation (exact ground truth kept)

Every transform is applied **identically to the reference and to every frame** of the
example. Each is a deterministic function of the sheet plus the part map, with
parameters drawn once per character, so the target stays an exact pixel-art animation.

| Transform | How | Targets |
|---|---|---|
| Chibi head | Upscale parts 2–4 by ×1.5 or ×2 (nearest) about the head's bottom-centre, found **per frame** so the head still follows the body; paste over the body | Puny-like proportions |
| Short legs | Drop k ∈ {2..6} rows of the leg band (rows between the reference's hip and ankle lines, the same canvas rows in every frame) and shift everything above down | chibi, NES |
| Size | Block-majority downscale of the whole frame to 48/40/32 (in 64) or 24/16 (in 32), feet on the pivot | small sprites in a big canvas (NES collapse) |
| Posterise | Merge the palette to 3–6 colours (merge nearest colours, the same map for all frames) | NES/GB, flat shading |
| Outline | Add a 1 px dark outline around the silhouette, or remove an existing dark outer ring | outlined packs (Ninja, Puny) |
| Recolour | Global hue/value shift of the palette | colour variety |
| Mirror | Flip horizontally and swap the left/right direction rows | asymmetric items |
| Frame subsample | Keep 3–4 evenly spaced frames of a walk/run; `n_frames` is already an input | 3–4-frame walks in other packs |

- **Mixing:** about 50% of training examples stay unaugmented, and each augmented example
  gets 1–3 transforms.
- **Where it runs:** in the **training loader**, not baked into the dataset, so the
  probabilities can be tuned without rebuilding. Only the part map is new on disk.
- **Cost:** numpy on ≤64×64×13 frames, well under 1 ms per example, so it fits the 2
  loader workers.

## 3. Real packs (after augmentation is measured)

- Add CC0 packs that are **not** in the evaluation set (Bit Era Extended and other CC0
  extras from track 3) as a small second training source, weighted at about 10%.
- Hold out by pack **and** artist.
- Evaluation packs (Puny, Ninja, NES Knight/Ranger, Antifarea) never enter training. A CI
  check enforces this.
- Every new pack's licence is re-verified at its source before use (same rule as decision
  0019).

## 4. Evaluation upgrades (before the Phase 2 run)

- **OOS walk scoring:** the current best-match per frame is lenient for the model. Add
  ±1 px tolerant exact and a phase-aligned score: align contact poses (legs apart) in
  truth and prediction.
- **More actions:** extend the test to attacks and hurt where the packs have them (Puny,
  Ninja), not only walk.
- **Region-swap rate:** on LPC validation, using the part maps.

## 5. Experiment plan (one GPU job at a time)

| Run | Data | Budget | Question |
|---|---|---|---|
| E1 | lpc32v2 + augmentation (p = 0.5) | = control (40k × 16) | Does augmentation alone fix out-of-style? |
| E2 | E1 + aux part head | same | Do part labels reduce region swaps and help out-of-style? |
| E3 | E2 + CC0 packs | same | Does real style variety add to synthetic? |

- Every run is from scratch. Each 32 px run takes about 4 h on the laptop.
- E1 is the decisive one. If it fails criterion 2, the next step is real data (E3) before
  anything else.
- Only after Phase 2 passes: the final 64 px run (phase 4).

## Open questions

- Chibi head scaling over occluding items (weapons in front of the face): paste the head
  group under items with a higher z? That needs per-layer images, which the composer has
  at render time. So augmentation may need the per-slot layers, not only the flattened
  part map. Alternative: do head scaling inside the composer at build time, as an
  augmented dataset variant.
- Can 64 px models learn small sprites from size augmentation alone, or do they need a
  size/scale input?

## Update 2026-11-25 — pass marks for E3 (FROZEN 2026-11-26, agreed before E3 data was collected)

Scored with scripts/oos_score.py (dev packs) and the LPC decode evaluation. Reference run: E1b.
1. **LPC:** validation exact (256 examples, 1-shot, colour) >= 0.772, i.e. at most 1 point below E1b (0.782).
2. **Motion:** walk motion F1 above E1b on Puny, Ninja and Antifarea, with a 95% paired interval above 0 on at
   least two of the three, and above the fixed nudge on all three. Knight and Ranger (one character each) are
   reported and must not fall by more than 0.05.
3. **Appearance:** walk exact >= the still sprite on at least 4 of 5 packs; IoU >= still - 0.02 on all 5.
4. **Seeds:** the result must hold for two training seeds.
5. **Final test:** the held-out artists are scored once, after 1-4 are decided.

The nudge* bound and the attack/hurt actions are reported, not required. E2 (aux part head) is optional, after E3.

Final-test artists (never trained on, scored once): ansimuz, GibbonGL, Gamekrazzy, Hormelz (CC0); Robocelot, AxulArt
(CC-BY, with credit, final test only). Training artists: Ordinary Bumblebee, Lexlom, Foozle, JIK-A-4.

## Update 2026-11-26 — packs32 built (E3 training data)

`scripts/build_packs32.py` builds `data/built/packs32` from the owner-downloaded packs in `data/raw/e3/train`
(not in git). 63 characters (44 designs), all split `train`, artists checked by `data/heldout.py`:

| Artist | Pack | Characters | Actions | Notes |
|---|---|---|---|---|
| Ordinary Bumblebee | 32x32 Customizable Character Pack | 12 composed looks (one base design) | walk 4, idle 4, run 4, jump 3, hurt 3 | 1x; reference = idle frame 0 |
| Ordinary Bumblebee | 8 Direction Walk Cycle | 2 colours (one design) | walk 8 | 45 px tall, shrunk 2x; standing frame per direction matched by silhouette |
| Lexlom | 32 Free animated characters | 32 designs | walk [step, stand, step, stand] | 25 of 32 shrunk 2x (36-40 px tall); left side is the mirrored right side |
| Foozle (drawn by Baldur) | Lucifer series, 9 of 12 packs | 9 designs | walk or run 6-8, idle, jump, hurt, climb | 6 shrunk 2x; 3 at 1x (too-big jump/hurt dropped where needed); Walk/Run file names are mixed per direction, so the cycle is named by majority |
| JIK-A-4 | MetroCity 1.0 + 2.0 | 8 composed looks (one base) | run 6 | no standing pose: reference = run frame 0 (`reference_from_cycle`) |

Placement (`data/packs.py`, shared with the final test): one integer offset per character, feet of the down-facing
reference on (16, 30); a walk that reaches 1-2 px past the canvas is nudged by up to 2 px. Characters that still do not
fit at 1x are shrunk 2x with `data/resize.downscale` (the lpc64 -> lpc32 method); nothing is cropped. A 1-frame walk
is a reference only and never a target (`dataset._examples_of`). Composed looks and recolours share a design.
Foozle not used: Goblin Rider (a mount, no walk), Skeleton King (walk 4 px below its idle), Skeleton Ancient
(mixed cell sizes); Possessed was not downloaded. A character is kept at 1x when its reference and walk/run fit;
only actions that do not fit are dropped (`dropped_too_big`).

Not used for training: Bumblebee ModelSheet (a single bald model with mixed rows), Lexlom ModelSheet (same).
Final-test packs: see the next section.

## Update 2026-11-26 — final32 built (FINAL TEST, never trained on, scored once)

`scripts/build_final32.py` builds `data/built/final32` from `data/raw/e3/final` (not in git). Every entry is split
`final` (training reads `train` only, tested) and the build refuses any artist not on `FINAL_ARTISTS`. Built by
looking at the art only; no model has been run on it.

| Artist | Characters | Actions | Notes |
|---|---|---|---|
| ansimuz | Faune; Tiny RPG hero, mole, treant (4) | walk (Faune also run) | 1x; side drawn facing right, left mirrored (synthetic, not to be scored); Faune has no idle, reference = walk frame 1 |
| Gamekrazzy | 1 hero | walk [stand, step, stand, step] | 1x after a 2 px nudge; left and right drawn separately |
| GibbonGL | 1 Game Boy monkey | walk [stand, step, stand, step] | 1x, 16 px; direction rows matched to the pack's named GIFs |
| Hormelz | 1 | idle 2, run 2 | 1x, 16 px; walk is the idle reference only |
| Robocelot (CC-BY) | Gumdrop x10 (one base body, different outfits), Character Template x1 | walk 8, idle, run 8, hurt 3 | about 34 px tall, shrunk 2x; row order from SpriteSheets/flash.json |
| AxulArt (CC-BY) | template, girl, boy (3) | walk [step, middle, step, middle] | 1x, 16x24; no idle, middle frame is the reference |

21 characters, 6 artists. Robocelot has 11 of the 21, so the final score should be averaged per artist. Not used:
Gamekrazzy roll and slash (no matching action), Gumdrop attacks/draw/sheath, Tiny RPG attacks, Hormelz attack/block,
Tiny RPG Mountain (tiles only). A scorer for final32 (in-order motion F1, drawn directions only) is still to write.

## Update 2026-11-27 — E3 seed 1 result: does not pass

Run `data/runs/20261126-2047-c32v3_packs_s1` (40000 steps, commit dd5e109). Log: `docs/research/e3-s1-eval.log`.

| Mark | Result | |
|---|---|---|
| 1. LPC exact >= 0.772 | 0.776 (E1b 0.782) | pass, narrowly |
| 2. Walk motion F1 vs E1b | Ninja +0.042 [+0.030, +0.053]; Antifarea +0.014 [-0.003, +0.030]; **Puny -0.101 [-0.127, -0.071]** | fail: Puny below E1b, only 1 of 3 intervals above 0 |
| 2. above the nudge | Puny 0.291 > 0.202, Ninja 0.518 > 0.480, Antifarea 0.580 > 0.431 | pass |
| 2. Knight / Ranger not down > 0.05 | Knight +0.051, **Ranger -0.066** | fail |
| 3. exact >= still on 4 of 5 | 2 of 5 (Ninja, Antifarea); E1b is also 2 of 5 | fail (as E1b) |
| 3. IoU >= still - 0.02 on all 5 | all 5 | pass |

Walk exact rose on every dev pack (Ninja +0.020 with interval above 0). LPC walk motion F1 fell 0.009. The Puny walk moves
less than the truth (moved 0.133 vs 0.236; E1b 0.171). As frozen, E3 fails mark 2 on seed 1, so mark 4 cannot hold;
seed 2 was not run. E1b stays the reference model. The final test was not scored.

### Why Puny fell (diagnosis 2026-11-27, no training)

- Where: front and back views (down 0.480 -> 0.308, up 0.513 -> 0.368). In those views E3 mostly copies the standing
  pose; Puny's one-leg step is missing. Ranger's drop is also mostly the front view.
- Not the 2x shrink: training pack walks move as much as LPC walks (0.31-0.43 of the character's pixels change per
  frame vs 0.34-0.46 for LPC; shrunk Lexlom moves as much as full-size Lexlom).
- Not the frame count: asked for 4 frames (the packs' own length), E3 is still below E1b on Puny (-0.049).
- Not checkpoint noise: step 38000 (best) and 40000 (last) differ by at most 0.004 on every walk.
- Open: training-seed noise. Puny walk F1 across earlier runs spans 0.255-0.393 (E1b is the highest), and Knight and
  Ranger span 0.37-0.53, but those runs also differ in config. A second seed of E1b would measure it.
- Pictures (rows: truth, E1b, E3; first cell is the reference): `docs/research/figures/e3-puny/`.
