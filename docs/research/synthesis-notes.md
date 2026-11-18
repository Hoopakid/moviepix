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
