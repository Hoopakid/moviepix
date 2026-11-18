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
