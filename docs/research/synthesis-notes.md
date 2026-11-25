# Research synthesis — running notes (main researcher)

## Track 5 — users (received)
- Strongest user need matches our design: consistency + "keep MY pixels/palette";
  competitors break palettes, jitter, mix pixel sizes. Our palette-exact copying is the
  differentiator. No competitor is local/offline-on-CPU or publishes data provenance.
- Wanted next: 4-direction sets from one view, Aseprite integration (extensions can call
  local processes; WebSocket client exists), artist-in-the-loop editing (we have the
  `canvas/known` editing path), engine-ready exports (Aseprite JSON, strips, GIF).
- Audience is the most AI-skeptical group (GDC 2026: visual artists 64% negative). Position
  factually: "a machine-learning tool", verifiable provenance, ship a disclosure sentence
  and CREDITS; users must tag output on itch.io/Steam. Launch first in r/gamedev, r/godot,
  HN, itch devlogs — not pixel-art spaces that ban AI.
- **Correction to the agent's licensing concern:** it assumed our data is CC-BY-SA/GPL.
  We only use items whose credits ALL offer a permissive option (CC0/CC-BY/OGA-BY), and
  choose that option (decision 0002). Remaining real obligations: attribution (BY/OGA-BY)
  for the dataset and, conservatively, for weights trained on it. Still worth a careful
  licence review before release (wave 2 question).
- Unverified (blocked sites): r/PixelArt rules, OpenGameArt AI policy text.

## Track 1 — prior work (received)
- Sprite-specific academic work is small and regression-based: Coutinho & Chaimowicz
  (sprite direction translation, LPC-derived data, Pix2Pix→CollaGAN 105M; 1→3 input
  views FID 8.39→1.51; outputs NOT palette-exact), Alonso & Patow 2026 (in-betweening,
  ~10M U-Net per direction, 280 private triplets — numbers to treat with care),
  Sprite Sheet Diffusion (Animate Anyone fine-tune, overfits). Adjacent fields agree
  copying beats regressing (appearance flow 1605.03557, voxel flow 1702.02463,
  AnimeInterp, BasicPBC region-level colour copying).
- **No prior paper found that frames sprite animation as per-pixel pointer selection
  over the reference** → our formulation looks novel (verify in wave 2).
- No open, local, palette-exact animator; no credited benchmark → a benchmark with
  character-level + unseen-item + out-of-style splits would itself be a contribution.
- Best ideas (my ranking for us):
  1. **Body-part labels for free**: our composer knows which LPC layer each pixel came
     from → auxiliary part head and/or part-constrained copying (should cut region
     swaps and help other styles). Needs a label channel in the dataset (rebuild).
  2. **Edge/detail-weighted copy loss** (+ maybe tiny 2×2 PatchGAN fine-tune) vs over-smoothing.
  3. **In-betweening mode** via whole-frame masking (artist fixes keyframes) — near-free.
  4. **Multiple reference views** (1–4, random dropout in training).
  5. **Copy from already generated frames** (temporal consistency; could make rounds pay off).
  6. Metrics: FLIP, detail-pixel accuracy, region-swap rate, per-action breakdown.
- To verify in wave 2: existence/details of 2610.16423, 2610.27253, 2611.20628 (PixelBench).

## Track 2 — techniques (received)
- **Key insight about our own code:** the loss optimises probability per COLOUR (sum over
  same-coloured reference pixels) but decoding chose per REFERENCE PIXEL. A colour spread
  over many pixels then loses to the single "transparent"/"new" slots → likely causes weak
  silhouettes, lost details, and background-first confidence ordering; it also biased which
  pixels self-conditioning treated as known. Fix implemented (colour-aggregated decoding,
  default on); to be measured on all checkpoints.
- Theory: per-pixel argmax is the optimal decoder for a per-pixel error count, so rounds
  cannot beat one-shot on pixel_exact by construction; their value is joint coherence /
  details → need detail metrics (speckle precision/recall with ±1 px tolerance, colour-
  histogram L1).
- Decoding-only ideas to test next: count-matched decoding (match expected colour counts
  → restores rare 1-px details), offset-consensus (pool copy offsets over neighbours),
  Halton / margin orders, ReMDM revision passes. Training-side: detail-weighted loss, soft
  self-conditioning + loss on known pixels, a tiny Token-Critic head, relative-position
  bias on copy logits. Autoguidance speculative.

## Track 4 — deployment (received)
- Matches my prototype (ONNX export works, 99.9996% parity). Plan: 3 graphs (reference
  encoder once; decoder trunk; copy head run 1–2 frames at a time with max−logsumexp
  confidence), host-side compaction/argsort/loop in JS; onnxruntime-web WASM first,
  WebGPU optional; static int8 QDQ on convs only, copy path fp32; HF static Space with
  COOP/COEP headers; Aseprite via Lua extension calling the CLI.
- Risks: copy-logit memory (split head), GroupNorm op coverage on WebGPU/Unity, MHA
  fast path (already hand-written in my export), int8 flipping near-tie argmaxes.
- 32 px browser safe; 64 px < 200 ms in browser uncertain → one-shot default, maybe D=32.

## Track 3 — other styles (received)
- Verified-permissive animated packs for an out-of-style test set (licences quoted in the
  track notes): Puny Characters (CC0, chibi 32 px, 8 dirs, idle/walk/attacks/hurt/death),
  Ninja Adventure (CC0, 50+ chars, 16 px), Chasersgaming RPG Ranger/Knight NES/GB (CC0,
  2–4 colour palettes), Antifarea 16×18 (CC-BY 3.0), Bit Era Extended (CC0, 20 colour
  variants), Calciumtrice (CC-BY 3.0, side), LuizMelo fighters (CC0, large side view),
  0x72, Pixel Frog, Sunny Land, Ars Notoria (CC0 extras).
- Excluded: Tiny RPG (Zerie), Mystic Woods, Sunnyside, Pipoya, CraftPix, Mixamo (all forbid
  AI training and/or redistribution).
- Strategy ranking: (1) layer-aware LPC augmentation (chibi head scaling, short legs, size,
  outline add/remove, flat/posterised shading, hue, mirrored directions) — exact GT kept;
  (2) add real CC0 packs, hold out by pack AND artist; (3) few-shot fine-tune on user
  frames; (4) CC0 3D renders (Quaternius + CMU mocap) later; (5) skeleton/part conditioning deferred.
- Evaluation: separate splits (LPC held-out parts / near out-of-style / far out-of-style),
  report worst pack, align by pivot and animation phase, ±1 px tolerant metrics, no FID.

## Wave 2 — verification (received)
- 11/13 references fully verified, 2 with nuances; none fabricated. Corrections recorded in
  wave2-verification.md (PixelBench DOES include animation tasks; Alonso & Patow 280 =
  whole dataset, "directional" = temporal; Tessera-small is a still-image generator;
  GDC 64% = visual + technical artists; Token-Critic worsens IS at 256 px).
- **Novelty = partially novel.** Our colour-marginal loss equals Vondrick et al. 2018
  ("Tracking Emerges by Colorizing Videos") cross-entropy over attention-weighted
  reference colours — but there the target frame is observed (grayscale queries). New:
  pointer over reference pixels as the GENERATIVE output space for an unobserved frame
  from one still + action/direction; {transparent, new, pointer} vocabulary (palette-exact
  by construction); combined with masked training/self-conditioning/iterative decoding;
  application to sprite animation. Paper must cite Vondrick 2018 for the loss and not
  claim novelty for copying, the loss, or masked decoding.

## Wave 2 — licensing (received; not legal advice)
- **Release blocker (data provenance):** our permissive filter trusts the LPC generator's
  per-file credit metadata. Upstream OGA pages for 26 of 81 linked packs show NO
  permissive option on the page (often per-file options live in a CREDITS file inside the
  zip, e.g. lpc-hair). 90 of 370 credit rows link only to such pages, incl. heavily used
  heads/bases (BenCreating: page says CC-BY-SA/GPL; he commented 2026-07-08 he has "no
  problem licensing these ... under OGA-BY", except some animal heads) and the legion torso.
  → Before ANY release of weights/dataset: script-verify those rows against each pack's
  internal credits file; save evidence; add license_used/license_url/title/modifications
  columns to CREDITS.csv; fix assets/CREDITS.md rows (missing URLs, version, licence links).
- Choosing a permissive option of multi-licensed work is valid (OGA + CC FAQs).
- Weights: courts split (UK Getty v Stability: weights not copies; DE GEMA v OpenAI:
  memorised works reproduced). Practice: ship credits with the weights + run a
  memorisation check (tiny model on repetitive parts).
- Outputs from users' own sprites: low risk; README must say users need rights to inputs.
- CC-BY-SA art for evaluation only: fine; keep separate, CI check it never enters training.
- GPL: reading data files triggers nothing; don't commit generator code/JSON.
- HF: weights `apache-2.0` + NOTICE + credits; dataset `license: other` + LICENSE.md.

## Wave 2 — out-of-style packs (received)
- Measured 7 packs (downloads in data/raw/ooS/, licences re-read 2026-11-18): Puny (CC0,
  32×32, 8 dirs), Ninja Adventure (CC0 repo pinned e8674a0, 16×16, 4 dirs), Chasersgaming
  Knight/Ranger NES + Ranger GB (CC0, 16×24, 3–4 colours), Antifarea (CC-BY 3.0, 16×18,
  colour-key bg), Bit Era (CC0, 3 colours, 20 recolours), Calciumtrice (CC-BY 3.0, 32×32
  side view). Slicing specs + placement rules (feet pivot, integer scaling) in
  wave2-oos-packs.md. LuizMelo not downloaded (the agent's itch.io download was blocked
  by the permission system — left for the user to decide).
- Problems: frame counts never match LPC (3–4-frame walks vs 9) → phase-aligned scoring
  + frame-count-free metrics; single-frame actions are pose checks only; mirrored or
  missing directions reported separately.

## Wave 3 — own experiments, batch 1 (docs/research/wave3-eval-batch-1.log)
### Colour-aggregated decoding (decide by colour) — decoding-only, all checkpoints
| model | exact pixel→colour | IoU | detail recall | detail precision |
|---|---|---|---|---|
| C32 (v1) | 0.792 → **0.802** | 0.981 → 0.983 | 0.809 → 0.802 | 0.820 → 0.850 |
| C32 v2 new (α+sc) | 0.790 → **0.798** | 0.974 = | 0.813 → 0.807 | 0.793 → 0.824 |
| C64 (v2) | 0.754 → **0.771** | 0.955 → 0.963 | 0.830 → 0.808 | 0.836 → 0.876 |
| C64 sc (warm) | 0.798 → **0.806** | 0.964 = | 0.861 → 0.842 | 0.835 → 0.876 |
- +0.8 to +1.7 points exact everywhere, silhouettes same or better, details more precise
  but slightly fewer (recall −0.6 to −2 pts). Default stays ON. Slower at 64 px (25 s vs
  10 s for 256 ex.: per-call colour grouping loop) → optimise (cache groups per reference).
- Rounds still ≈ one-shot (consistent with theory: argmax optimal for per-pixel metrics).

### First out-of-style walk test (188 samples, 5 packs; best-match per truth step frame)
- **Generalisation is weak — the key Phase 2 finding.** In-style (LPC) the models beat the
  "repeat the still sprite" baseline by +16 to +22 points exact. Out of style:
  Puny (chibi 32 px): all models WORSE than the baseline (−5.5 to −14.7 pts);
  Ninja/Antifarea: modest gains (+4 to +13 pts) but silhouette IoU below baseline;
  Knight/Ranger (16×24, ×1 in a 64 canvas): 64 px models' silhouettes collapse
  (IoU 0.26–0.62 vs 0.86–0.89 baseline) — they likely draw LPC-sized bodies.
- 32 px models generalise better than 64 px ones on small sprites (sprites at ×1 in
  32 px are closer in size to LPC-32 characters).
- Caveat: tiny-sprite walks move few pixels, so the baseline is strong; packs have only
  3–4-frame walks. Still, the direction is clear: Phase 2 (augmentation + varied data) is essential.

## Phase 1 — clean comparison (docs/research/phase1-eval.log)
c32v2_new (alpha head + self-conditioning 0.4) vs c32v2_control (neither); same data
(lpc32v2), same budget (40k steps × 16), both from scratch. 256 examples per split.

| one-shot, colour decoding | control | new | Δ |
|---|---:|---:|---:|
| exact, validation | 0.768 | **0.798** | **+3.0** |
| exact, unseen items | 0.759 | **0.786** | **+2.7** |
| IoU, validation | **0.978** | 0.974 | −0.4 |
| speckle excess (truth = 0) | −11.5 | **−7.6** | closer to truth |
| detail recall / precision | 0.783 / 0.821 | **0.807 / 0.824** | +2.4 / +0.3 |
| exact, 8 confidence rounds vs one-shot | −0.6 | −0.1 | rounds no longer hurt |

- **Verdict: the new recipe wins in-style**, cleanly attributed (no warm start). It
  becomes the default for later runs. Its small IoU cost comes from the alpha head (it
  is slightly more conservative at edges).
- **Out of style it does not help**: Puny is still below baseline for both (−5.8 / −6.9);
  Antifarea gain drops +13.6 → +8.8; Ninja/NES similar. Out-of-style generalisation is a
  data problem, not a decoding/recipe problem → Phase 2 (docs/research/phase2-design.md).
- Colour-decoding speed fix (one-hot matmul instead of scatter_add, 36× faster on MPS
  in a microbenchmark) landed mid-evaluation: the control-validation timings (9.4 s for 4
  colour rounds) use the old code, the rest the new (≈ same as per-pixel). Results are
  numerically identical.

## Phase 2 — E1: style augmentation (docs/research/e1-eval.log, e1-motion.log)
c32v3_aug = c32v2_new + augment 0.5 + subsample 0.25 (same budget, same characters).

| | c32v2_new | E1 (aug) |
|---|---:|---:|
| LPC validation exact (1-shot, colour) | 0.798 | 0.764 (−3.4) |
| LPC unseen exact | 0.786 | 0.758 (−2.8) |
| LPC IoU | 0.974 | 0.965 |
| OOS gain vs repeat: Puny / Ninja / Knight / Ranger / Antifarea | −6.9 / +6.3 / +5.8 / +5.2 / +8.8 | **+0.3 / +5.5 / +1.0 / +3.0 / +11.9** |
| OOS IoU vs baseline (Puny / Ninja / Knight / Ranger / Antifarea) | −4.2 / −16.2 / −5.9 / −7.2 / −14.2 pts | **+0.3 / +2.4 / −0.1 / +0.6 / +4.0** |
| Share of pixels moved vs truth (Puny .11 / Ninja .28 / Knight .33 / Ranger .22 / Anti .35) | .48 / .47 / .37 / .28 / .59 | .13 / .15 / .09 / .09 / .17 |

- **Silhouette collapse fixed**: out-of-style IoU now ≥ baseline on every pack (criterion 3 PASS).
- **Criterion 2 (beat baseline on every pack) passes only barely** (Puny +0.3).
- **Criterion 1 FAILS**: −3.4 pts on LPC. Likely cause: half the budget went to augmented
  characters (fewer plain LPC examples at equal steps) and/or overly hard transforms.
- **New motion metric explains it**: the v2 model over-moves foreign sprites (repaints 37–59%
  of pixels where truth changes 11–35%: it redraws them as LPC bodies). E1 under-moves
  small sprites (9–17%): it learned to be safe, close to copying the still. Puny motion
  (.13 vs .11) is now right.
- Interpretation: synthetic augmentation teaches *robustness to shape* but not *how small,
  differently drawn characters move*. That needs real motion examples in other styles
  (E3) — and/or more budget so LPC accuracy recovers.

## Phase 2 — E1b: gentler augmentation (docs/research/e1b-eval.log, oos-walk-e1b.json)
c32v3_gentle = E1 with augment 0.25 and no chibi ×1.75 / size 0.5. One seed, same held-out characters.

| | control | c32v2_new | E1 | E1b |
|---|---:|---:|---:|---:|
| LPC validation exact (1-shot, colour, 256 examples) | 0.768 | 0.798 | 0.764 | 0.782 |
| LPC unseen exact | 0.759 | 0.786 | 0.758 | 0.773 |
| LPC validation exact, training log's own sample | 0.775 | 0.797 | 0.769 | 0.789 |
| OOS exact gain: Puny / Ninja / Knight / Ranger / Antifarea | −5.8 / +7.1 / +4.4 / +4.8 / +13.6 | −6.9 / +6.3 / +5.8 / +5.2 / +8.8 | +0.3 / +5.5 / +1.0 / +3.0 / +11.9 | +0.0 / +6.0 / +0.8 / +2.5 / +13.3 |
| OOS IoU minus baseline (pts) | −8.0 / −11.0 / −8.3 / −5.7 / −8.3 | −4.1 / −16.2 / −5.9 / −7.2 / −14.2 | +0.4 / +2.4 / −0.2 / +0.6 / +4.0 | +0.4 / +2.0 / +0.4 / −0.2 / +3.5 |
| OOS share of pixels moved (truth .11/.28/.33/.22/.35) | – | .48/.47/.37/.28/.59 | .13/.15/.09/.09/.17 | .15/.16/.11/.09/.16 |

- Criterion 1, as written (within 1 pt of **the control**, 0.768): E1b passes (+1.4); E1 passes too (−0.4).
  Against the stricter bar used in the E1 notes and the E1b config (within 1 pt of c32v2_new, i.e. ≥ 0.788):
  E1b misses by 0.6 pt. **Correction:** the E1 section's "criterion 1 FAILS" used c32v2_new, not the control.
- Criterion 2 (exact AND IoU above baseline on every pack): **not met by E1 or E1b**. E1b: Ranger IoU −0.2,
  Puny exact tied (+0.002 pt). **Correction:** the E1 section said IoU ≥ baseline everywhere; Knight was −0.2.
- Criterion 3 (IoU ≥ baseline − 0.02): met by both.
- Exact gain is not better than the un-augmented control on Ninja, Knight, Ranger or Antifarea. The augmentation
  win is Puny plus the silhouettes. Both E1 and E1b under-move every pack except Puny.
- Caveats: the score takes the best of all predicted frames, so a model that copies the still scores at or near
  baseline; Knight and Ranger are one character each (14 frames); the same 5 packs were used to pick
  between runs. Design section 4 (phase-aligned scoring) was planned before these runs and is not done yet.
