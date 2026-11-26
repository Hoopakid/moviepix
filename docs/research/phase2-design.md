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
(not in git). 55 characters, all split `train`, artists checked by `data/heldout.py`:

| Artist | Pack | Characters | Actions | Notes |
|---|---|---|---|---|
| Ordinary Bumblebee | 32x32 Customizable Character Pack | 12 composed looks (one base design) | walk 4, idle 4, run 4, jump 3, hurt 3 | 1x; reference = idle frame 0 |
| Ordinary Bumblebee | 8 Direction Walk Cycle | 2 colours (one design) | walk 8 | 45 px tall, shrunk 2x; standing frame per direction matched by silhouette |
| Lexlom | 32 Free animated characters | 32 designs | walk [step, stand, step, stand] | 29 of 32 shrunk 2x (36-40 px tall); left side is the mirrored right side |
| Foozle (drawn by Baldur) | Lucifer Warrior | 1 | walk 8, idle 5, jump 5, hurt 4, climb 6 | about 34 px, shrunk 2x |
| JIK-A-4 | MetroCity 1.0 + 2.0 | 8 composed looks (one base) | run 6 | no standing pose: reference = run frame 0 (`reference_from_cycle`) |

Placement: one integer offset per character, feet of the down-facing reference on (16, 30). Characters that do not
fit at 1x are shrunk 2x with `data/resize.downscale` (the lpc64 -> lpc32 method); nothing is cropped. A 1-frame walk
is a reference only and never a target (`dataset._examples_of`). Distinct designs: about 36; the composed looks and
recolours share a design. Only one of the 13 Foozle packs was downloaded, so Foozle is one character.

Not used for training: Bumblebee ModelSheet (a single bald model with mixed rows), Lexlom ModelSheet (same).
Final-test packs are copied and extracted but not built yet; the Robocelot template download could not be identified.
