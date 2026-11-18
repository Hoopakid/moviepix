# Wave 1, track 2: model techniques for moviepix's open problems

Date: 2026-11-18. Scope: over-smoothing (lost 1-px details / "speckles"), making iterative masked decoding beat one-shot, copy vs. warping, temporal consistency, and quality-per-parameter for the ~4.5M-param `copy_animator` (src/moviepix/models/copy_animator.py).

Conventions: every external claim has a URL. **[unverified]** marks numbers I took from abstracts, search snippets, third-party summaries or model cards without reading the paper's tables. **[analysis]** marks my own reasoning about moviepix (not a published result). Nothing here was run; all "expected effects" are hypotheses to test.

---

## 0. Two observations from reading our own code (before any literature)

These matter more than any paper below, and both are decoding-only.

### 0.1 The decoder takes argmax over *reference pixels*, but the loss is a marginal over *colors* [analysis]

`_chunk_nll` trains `-log Σ_{j: color(j)=target} p(j)`: the model is only rewarded for the total mass on the right *color*, not on any particular reference pixel. Nothing pushes it to put that mass on one key, so it may spread mass over all K_c reference pixels of a color c.

`_choose` then does `probs.argmax(-1)` over the `2+K` classes and uses `conf = p(chosen class)`. Consequences:

- **Choice bias.** A color whose mass is spread across many keys (a big fill, e.g. skin with 200 matching pixels) shows a small per-key maximum. Classes that are a single slot (`transparent`, `new`) or a color with one or two keys (an eye highlight) look much larger per-slot than they are per-color. The decision rule does not match the trained quantity. Whether this hurts or helps speckles depends on how concentrated the learned attention is, and that has not been measured.
- **Confidence bias.** `conf` is per key, so character pixels (mass split over many same-colored keys) get *systematically low* confidence. Transparent pixels (one class, alpha head or not) get high confidence. This alone predicts finding (a): "confidence order commits background first and the whole character last" (decision 0017). The same `conf` also picks the "own most confident" pixels in self-conditioned training (`own_known`), so self-conditioning has been trained on this biased order too.
- **Fix (zero parameters).** Aggregate before deciding: `P(c|p) = Σ_{j: color(j)=c} p(j|p)` (a `scatter_add` of the copy probabilities into the ≤65 unique palette colors per sprite; decision 0008 says at most 65 colors per character). Then take argmax over {transparent, new, c₁…c_M} and use `P(chosen color)` as the confidence. To render, pick any key of that color (the one with the highest p, so `new_rgb`-free colors stay exact). This is also the Bayes-correct decision for the "is the color right" metric (see 0.2).
- Test: rerun `scripts/decode_experiments.py` at 32 and 64 px with (i) one-shot color-argmax, (ii) confidence order with color-aggregated confidence. Report pixel_exact, silhouette IoU, speckle deficit. Cost: one `scatter_add` per frame.

### 0.2 Per-pixel argmax is *optimal* for pixel_exact. That is why rounds cannot beat one-shot on that metric [analysis + citation]

Taking each variable's marginal-most-probable value (MPM decoding) minimizes the expected number of per-variable errors (bit error rate, i.e. Hamming loss). Taking the most probable *joint* configuration (MAP) minimizes whole-block error. This is standard decoding theory; see the MAP vs. MPM discussion in Kabashima/Saad-style analyses of Gallager codes: https://arxiv.org/pdf/cond-mat/0110023 ("the MPM procedure is known to minimize the bit error probability").

Our `pixel_exact` is a Hamming-type metric. If the model's marginals are calibrated, one-shot argmax of marginals is already the best decoder for it in expectation. Iterative decoding helps only to the extent that conditioning on earlier commits makes later marginals *better informed*, i.e. when the network cannot represent the joint well in one pass. Iterative decoding mostly buys **joint coherence**: one consistent eye position instead of a blurred-away eye. Pixel_exact barely rewards that and sometimes penalizes it: a sharp eye one pixel off costs 2 pixels, while a missing eye costs 1.

**Implication.** "Rounds ≈ one-shot on 79.8% exact" is the expected outcome, not a failure. Iterative and sampling-based decoders should be judged on coherence and detail metrics, alongside pixel_exact:
- speckle count deficit (already exists),
- **speckle precision/recall with ±1 px tolerance** (did we draw the eye highlight, approximately where it belongs?),
- per-frame color-histogram distance to ground truth (e.g. L1 over palette counts),
- optionally a learned/perceptual score later.

MaskGIT itself is evaluated with FID (a distribution metric), and it *samples* tokens with temperature instead of taking argmax (https://arxiv.org/abs/2202.04200, Sec. 3.2). None of the MaskGIT-family gains below are claims about per-token accuracy.

---

## 1. Over-smoothing / mode-averaging

### 1.1 Why argmax erases rare details (mechanism) [analysis]

Suppose an eye highlight is 1 px. Given the reference and the action, the model is unsure whether it lands at (x, y) or (x+1, y), say 45%/45%, while the skin color holds 55% at each of those pixels. Per-pixel argmax picks skin at both, and the highlight vanishes. Yet the *expected count* of highlight pixels, Σ_p P(highlight|p) = 0.9, says there should be about one. Per-pixel factorized prediction cannot express "exactly one of these two", which is the "independent denoising" limitation that Discrete Copula Diffusion identifies as the cause of poor few-step quality in discrete diffusion (https://arxiv.org/abs/2410.01949). Muse states the same assumption explicitly ("many tokens are conditionally independent given other tokens"; https://arxiv.org/abs/2301.00704, Sec. 2.8).

Three families of fixes follow:
(A) make the *decision rule* aware of joint structure without retraining (counts, offset consensus, sequential commits);
(B) change *training* so the marginals are sharper on details (reweighting);
(C) add a *joint* model (critic, copula, adversarial).

### 1.2 Decision-rule fixes (no retraining)

| Technique | Mechanism | Evidence | Cost | Expected effect for us |
|---|---|---|---|---|
| Color-aggregated argmax (0.1) | Decide over colors, not keys | [analysis] | ~0 | Removes the slot-count bias; fixes confidence ordering |
| **Expected-count (histogram) matched decoding** | Per frame, target count for color c: n_c = round(Σ_p P(c\|p)). Solve the assignment of pixels to colors that maximizes Σ log P subject to those counts (greedy by margin, min-cost flow, or a few Sinkhorn iterations then rounding) | OT/Sinkhorn for label assignment under class-proportion constraints: Sinkhorn Label Allocation, https://arxiv.org/abs/2102.08622; Sinkhorn distances, https://arxiv.org/abs/1306.0895 | M ≤ 65 colors × P pixels; Sinkhorn with ~20 iterations is negligible next to the U-Net | Directly targets the speckle deficit: a color expected ~1× gets ~1 pixel, placed where it is most probable. Risk: a misplaced pixel costs a pixel_exact hit (see 0.2). Use a soft version that only *adds* rare colors where P(c\|p) > τ |
| Reference-count prior / "coverage" | Animation frames of LPC sprites largely re-show the same parts, so color counts in a frame ≈ counts in the reference, except for occlusion | Coverage mechanism for pointer-generator copy attention (penalizes repeatedly attending to the same source tokens; "remarkably effective for eliminating repetition"): https://arxiv.org/abs/1704.04368 | As above | Gives a second target-count estimate (reference histogram) that is independent of the model. Blend it with the model's expected counts |
| **Offset-consensus decoding** (see §3.4) | Convert each pixel's copy distribution into a distribution over displacements δ = pos(j) − p. Pool those over a 3×3 or segment neighbourhood and copy with the consensus δ. A rigidly moving head then carries its eye highlight along | Segment-level matching for flat-colored animation (AnimeInterp's Segment-Guided Matching treats color pieces as units to avoid mismatches on texture-less regions): https://arxiv.org/abs/2104.02495; inclusion matching for paint-bucket colorization: https://arxiv.org/abs/2403.18342 | Top-M offsets per pixel, local pooling: small | Should preserve details that move with their part, which is most LPC motion. Decoding-only |
| Temperature < 1 / top-k on colors | Sharpens but keeps argmax identity at T→0 | Vondrick et al. use temperature 0.5 for their pointer copy at inference: https://arxiv.org/abs/1806.09594 | 0 | Alone it does *not* bring back an argmax-losing color. Only helps combined with sampling + sequential commits |
| Sampling + sequential commits | Sample one highlight location, commit it, re-predict neighbours | MaskGIT decoding: https://arxiv.org/abs/2202.04200 | k forward passes | Restores the details as a joint sample. Pixel_exact may drop slightly; judge on speckle P/R |

### 1.3 Training-side fixes (retrain at 32 px)

| Technique | Mechanism / equation | Evidence | Cost | Expected effect |
|---|---|---|---|---|
| **Detail-weighted NLL** | w(p) = 1 + λ·[p is a speckle or edge in the target], or the U-Net paper's weight map w(x) = w_c(x) + w₀·exp(−(d₁+d₂)²/2σ²) emphasizing thin separations | Original U-Net border weight map: https://arxiv.org/abs/1505.04597 | 0 params, a 3×3 neighbour comparison on the target | Raises P(rare color) at detail pixels so argmax keeps them more often. Watch calibration: it biases the counts of 0.1/1.2. Cheapest training change |
| Focal loss | FL = −(1−p_t)^γ log p_t: down-weights easy pixels (flat fills, background) | https://arxiv.org/abs/1708.02002 | 0 | Similar to the above but untargeted. Under our marginal-over-matches NLL, use p_t = Σ_matches p. Known to *hurt calibration toward under-confidence*, which matters for confidence-ordered decoding |
| Class-balanced weighting by effective number | w_c ∝ (1−β)/(1−β^{n_c}) | https://arxiv.org/abs/1901.05555 | 0 | "Classes" here are per-sprite colors, so weight by the rarity of the target color *within that sprite*: an eye color with 2 pixels gets a high weight |
| Count / size constraint loss | Penalize (Σ_p P(c\|p) − n_c^target)² per frame and color | Constrained-CNN losses (differentiable size constraints): https://arxiv.org/abs/1805.04628 | 0 | Keeps *expected* counts calibrated, which makes the count-matched decoder of 1.2 work better |
| Thin-structure loss | soft-clDice on skeletons, for 1-px lines/outlines | https://arxiv.org/abs/2003.07311 | Small (soft skeleton via pooling) | Outlines in pixel art are 1 px; relevant for silhouette/outline breaks. Lower priority than speckles |
| "Perceptual label smoothing" | Smooth targets toward perceptually close palette colors | Used by Tessera-small (47.6M-param masked-pixel sprite model; reports 78.8% masked-pixel accuracy): https://huggingface.co/thejagstudio/tessera-small **[unverified, model card only]** | 0 | Probably the *opposite* of what we want for speckles. Listed for completeness |

### 1.4 Joint / adversarial models

- **Token-Critic** (https://arxiv.org/abs/2209.04439): a second transformer (20 layers vs. the generator's 24) is trained with per-token BCE to tell which tokens of a masked-then-regenerated image came from the generator. At sampling time, the R lowest-scoring tokens are remasked on a cosine schedule. ImageNet 256: MaskGIT FID 6.56 → 4.69 (IS 203.6 → 174.5); 512: 8.48 → 6.80 (IS 167.1 → 182.1), at 18(×2) steps (numbers from the paper's Table 1, read via https://ar5iv.labs.arxiv.org/html/2209.04439). For us, the critic could be a **second 1×1 head** on the existing decoder (a "was this canvas pixel real or generated?" logit), trained on the self-conditioned frames we already produce. That costs ~65 params, not a second network, but it changes the training data. Expected effect: better choice of *which* pixels to revise. It is the cleanest route to iterative > one-shot on coherence.
- **Discrete Copula Diffusion** (https://arxiv.org/abs/2410.01949): combines per-token marginals with a separately trained copula (they use an autoregressive LM) at inference. Reports comparable or better quality with 8–32× fewer steps [unverified, abstract]. Too heavy as stated for a 4.5M local model. The *idea* (marginals + a cheap dependency model) motivates the count and offset-consensus decoders above, which are hand-built "copulas" for our domain.
- **Adversarial loss on discrete outputs**: pix2pix-style PatchGAN (https://arxiv.org/abs/1611.07004) needs a differentiable output. Options are Gumbel-softmax / straight-through (https://arxiv.org/abs/1611.01144) or feeding the discriminator expected colors. A pixel-art sprite GAN based on pix2pix exists (https://arxiv.org/abs/2208.06413), as does an imputation GAN for missing sprite poses (https://arxiv.org/abs/2409.10721 [unverified, abstract]). **Assessment [analysis]:** high cost and instability for a tiny model trained on a laptop; it would fight the palette-exact copy formulation. Try last, if at all.
- **Classifier-free guidance / autoguidance**: Muse uses ℓ_g = (1+t)ℓ_c − t·ℓ_u with 10% condition dropout and a guidance scale that increases linearly over decoding steps (https://arxiv.org/abs/2301.00704, Sec. 2.7). Discrete-state guidance theory: https://arxiv.org/abs/2406.01572. **Autoguidance** replaces the unconditional model with a *weaker version of the same model* (less trained or smaller) and reports large FID gains (ImageNet 512 FID 1.25 [authors' claim]): https://arxiv.org/abs/2406.02507. For us: ℓ = ℓ_final + w(ℓ_final − ℓ_early_checkpoint) on the copy/new/alpha logits. This is decoding-only if an early checkpoint exists, and costs 2× inference. Mechanism for speckles [analysis]: an under-trained model typically also lacks details, so extrapolating away from it should push probability *toward* rare details. Guidance on action/direction conditions (drop the action id 10% of the time) is a retraining option and might sharpen action-specific poses.

### 1.5 Discrete diffusion framing (background)

- D3PM (structured transition matrices, including absorbing/mask): https://arxiv.org/abs/2107.03006
- MDLM (simple masked diffusion objective = weighted MLM losses): https://arxiv.org/abs/2406.07524
- Discrete Flow Matching: https://arxiv.org/abs/2407.15595

Our training is already an absorbing-state discrete diffusion with a cosine schedule. These frameworks matter here mainly through their samplers (ReMDM, P2, informed correctors, §2). Their training objectives do not obviously help a 4.5M model with a tiny vocabulary per pixel.

---

## 2. Making iterative masked decoding beat one-shot

### 2.1 What the literature says about order/schedule

| Method | Key point | Numbers | Retrain? |
|---|---|---|---|
| MaskGIT (https://arxiv.org/abs/2202.04200) | Sample tokens; keep most confident; cosine schedule best; **more steps not monotonically better**, sweet spot 8–12 for cosine | ImageNet 256 FID: cosine 6.06 (T=10), square 6.35, linear 7.51 (T=16), log 29.17 | – |
| **Halton scheduler** (https://arxiv.org/abs/2503.17076, ICLR 2025, code https://github.com/valeoai/Halton-MaskGIT) | Confidence picks *clustered* tokens (high confidence next to known tokens). Neighbours decided in the same step share information, so parallel sampling makes them inconsistent; and the uncertain regions pile up into a final-step entropy spike. Halton: a fixed quasi-random 2-D low-discrepancy order that spreads commits evenly | 705M model, ImageNet 256, 32 steps: confidence 5.93, random 6.33, Halton **3.74**; 12 steps: 6.80 / 7.90 / 6.19. Confidence degrades as steps grow; confidence images described as "overly simplistic and smooth" | No |
| Spatial NMS in confidence order | Do not commit adjacent pixels in the same step | Used by Tessera-small (pixel-level masked sprite model): https://huggingface.co/thejagstudio/tessera-small **[unverified, model card]** | No |
| Top-k **margin** confidence | Rank positions by p(top1) − p(top2) instead of p(top1) | "Train for the worst, plan for the best", ICML 2025: adaptive order lifts Sudoku solving from <7% to ≈90% (abstract) https://arxiv.org/abs/2502.06768 | No |
| P2 path planning | Planner chooses both what to unmask *and what to remask*; self-planning uses the denoiser's own probabilities | https://arxiv.org/abs/2502.03540 (22% relative gain in protein foldability [unverified]) | No (self-planning) |

**Reading for moviepix [analysis].** Our "random" order (≈ one-shot) matches Halton's random baseline. Halton beat random at ≥12 steps in a 32×32-token setting with ~1000 tokens per image, about our 32-px pixel count. The margin rule plus *color-aggregated* probabilities (0.1) is the natural confidence for us: a pixel torn between two colors is genuinely uncertain, while a pixel whose mass is spread over 200 same-colored keys is not.

### 2.2 Revising committed pixels

| Method | Mechanism | Evidence | Fit |
|---|---|---|---|
| Mask-Predict (https://arxiv.org/abs/1904.09324) | Each iteration re-predicts *all* tokens and remasks the lowest-probability ones (linear decay); committed tokens can change | +4 BLEU over prior non-autoregressive models with a constant number of iterations [abstract] | Direct: needs a model that predicts known pixels too. Our loss currently only covers hidden pixels; add a small loss on known pixels in self-conditioned frames |
| **ReMDM** (https://arxiv.org/abs/2503.00307) | Unmasked token is remasked with prob σ_t; variants "cap", "rescale", **"conf"** (per-token σ ∝ softmax of −confidence); "loop" phase holds α fixed and repeatedly remasks/re-predicts | ImageNet FID T=64: MaskGIT 4.85 → ReMDM 4.45 (IS 196 → 209); at T=16 MaskGIT is *better* (6.74 vs 7.40). No retraining needed | Decoding-only; most useful as a "loop" of 2–4 revise passes after a one-shot draft |
| Draft-and-Revise (https://arxiv.org/abs/2206.04452) | Explicit draft phase then revise phase that keeps global structure | SOTA at the time on ImageNet/CC-3M [abstract] | Same recipe as "one-shot draft + ReMDM loop" |
| Token-Critic (§1.4) | Learned "real vs generated" per token decides remasking | FID 6.56 → 4.69 | Needs a critic head; the best-evidenced option |
| Informed correctors (https://arxiv.org/abs/2407.21243) | Corrector steps use the model's own information to fix errors; hollow transformer gives all corrector distributions in one pass | Better FID with fewer steps on tokenized ImageNet [abstract] | Conceptually like Token-Critic; heavier |
| SUNDAE (https://arxiv.org/abs/2112.06749) | **Step-unrolled training**: corrupt, let the model predict, then train it to denoise its *own* prediction; ablations say this is crucial | WMT14 En-De SOTA among non-AR at the time [abstract] | Our self-conditioning is a 1-step version. Next step: also give the model its own *wrong* commits and train it to fix them (requires loss on known pixels) |

### 2.3 Self-conditioning

Analog Bits (https://arxiv.org/abs/2208.04202): with 50% probability, first compute x̃₀ = f(x_t, 0, t) without gradient, then train f(x_t, x̃₀, t). At inference, feed the previous step's estimate. Training cost is less than 25% extra, sampling cost negligible. Gains: ImageNet 64 FID 6.10 → 4.24 (ε-pred, 20 NFE) and 12.13 → 6.94 (x₀-pred, 20 NFE) (Appendix E, Table 8 of https://arxiv.org/html/2208.04202v2). RIN uses the same idea for latent tokens: https://arxiv.org/abs/2212.11972.

**Our variant differs [analysis].** Our self-conditioning feeds the model's top-confidence predictions *as known pixels* (hard commits, in the canvas channel). The Analog-Bits form feeds the *full soft previous prediction* in an extra input channel, for *all* pixels, separately from the known mask. The soft form tells the decoder "here is what I thought everywhere, with confidence," which is exactly what a revise step needs. Cost: input channels 9 → 9 + 4 (previous RGBA estimate) + 1 (previous confidence). That is a few hundred parameters in the first conv. It requires a retrain, but warm-starting is easy (zero-init the new input weights).

### 2.4 Recommended decode-experiment grid (all decoding-only, on the self-cond + alpha model)

1. one-shot, key-argmax (current) vs color-argmax;
2. 4/8 rounds × order ∈ {random, Halton, confidence(color-agg), margin(color-agg), margin + 3×3 NMS};
3. same with temperature sampling T ∈ {0.5, 0.8} for *values* (argmax-free), annealed to 0 by the final round;
4. one-shot draft + ReMDM-conf "loop" of 2–4 passes remasking 10–20% least confident.

Report pixel_exact, silhouette IoU, speckle deficit, speckle P/R (±1 px), color-histogram L1, ms/frame. Expected [analysis]: pixel_exact flat or −0.5 pt; speckle deficit shrinks with Halton/NMS + sampling + revision.

---

## 3. Copy / pointer mechanisms vs. warping

### 3.1 Related copy mechanisms

- Pointer networks (output = attention over inputs): https://arxiv.org/abs/1506.03134
- Pointer-generator with p_gen mixing copy vs vocabulary, plus coverage: https://arxiv.org/abs/1704.04368. Our {new, copy j} with a "new color" head is exactly this pattern.
- **Tracking Emerges by Colorizing Videos** (https://arxiv.org/abs/1806.09594): predicts target pixel color as Σ_i A_ij c_i with A = softmax(f_iᵀ f_j) over reference pixels. Colors quantized to 16 k-means clusters, categorical loss, 64-d embeddings on a 32×32 grid, inference temperature 0.5, 3 reference frames with a sliding window of its own predictions for long videos. This is the closest published analogue of our copy attention.
- MAST (https://arxiv.org/abs/2002.07793): coarse-to-fine **ROI-restricted attention** (dilated local windows located by soft-argmax), plus short- and long-term memory frames. Both memories help (𝒥: 44.6 long only, 57.3 short only, 61.2 both). Note: for natural video they found regression in Lab beat classification (61.2 vs 47.1 𝒥). That does not transfer: our targets are palette-exact and categorical is correct.
- Contrastive random walk correspondence (space-time): https://arxiv.org/abs/2006.14613

### 3.2 Warping / displacement alternatives

| Approach | Output | Palette-exact? | Cost per pixel | Notes |
|---|---|---|---|---|
| Global copy attention (ours) | categorical over K ref pixels | yes | K·D (K ≈ opaque px; ~10³ at 64 px [estimate]) | Handles big moves (decision 0011) |
| Appearance flow (https://arxiv.org/abs/1605.03557) | continuous 2-D flow + bilinear sampling | **no** (bilinear blends colors); nearest sampling is non-differentiable | O(1) + conv | Needs straight-through or Gumbel to stay discrete |
| Deformable conv / STN (https://arxiv.org/abs/1703.06211, https://arxiv.org/abs/1506.02025) | learned offsets for feature sampling | n/a (features) | small | Useful *inside* the decoder to align reference features to the target pose before attention |
| Adaptive (separable) kernels (https://arxiv.org/abs/1708.01692; dynamic filters https://arxiv.org/abs/1605.09673) | per-pixel local kernel over a window | only if kernel is one-hot | (2r+1)² or 2(2r+1) | SepConv: separable 41×41 kernels need 1.27 GB vs 26 GB at 1080p |
| **Local categorical copy** (= discrete displacement field) | categorical over (2r+1)² offsets | yes | (2r+1)²·D | 9×9 window ≈ 81 keys vs ~10³: ~10–20× fewer copy MACs [analysis]. Fails on large motions unless combined with global or coarse-to-fine (MAST-style) |
| Splatting (forward warp; https://arxiv.org/abs/2003.05534) | push reference pixels to targets | yes if one-hot | small | Natural for "rigid parts move" but needs conflict resolution |
| Keypoint/motion-model warps (FOMM, https://arxiv.org/abs/2003.00196) | sparse keypoints + local affine | no (bilinear) | small | Too smooth for 1-px art; motion priors are interesting for actions |

**Conclusion [analysis].** A displacement field with categorical (one-hot) sampling *is* local copy attention with a relative-position parameterization. It is cheaper but loses global reach. The best quality-per-parameter move is to **keep global copy and add a relative-position prior**, not replace it:

### 3.3 Relative-position bias on copy logits (retrain; tiny)

logit(p, j) = q_pᵀ k_j / √D + B_{a,d}[Δ(p, j)], where Δ = pos(j) − pos(p), clipped to ±r. B is a learned table per action/direction (or a small MLP of Δ and the condition embedding), in the style of Swin's relative position bias (https://arxiv.org/abs/2103.14030). Cost: (2r+1)² × (#actions·#directions) scalars, e.g. 17²×15×4 ≈ 17k params, or an MLP of ~2k params. It gives the model an explicit "most pixels move by a small, action-typical offset" prior. The learned absolute `pos` embedding has to discover that implicitly. Expected: sharper copy distributions, so fewer argmax losses on details, and better generalization to unseen items. Test at 32 px with an equal step budget.

### 3.4 Offset-consensus decoding (decoding-only) [analysis]

1. For each output pixel p, take the top-M copy keys (M ≈ 8 after color aggregation is *not* applied; this needs the per-key distribution) and record offsets δ = pos(j) − pos(p) with weights p(j|p).
2. For each p, score each candidate δ by Σ_{q∈N(p)} p_q(key at q+δ). N is a 3×3 window, or better the connected region of the *predicted* color / alpha mask.
3. If the best consensus offset δ* has support above a threshold and the reference color at p+δ* has P(c|p) above a small floor, render that color. Otherwise keep the color-argmax.

This is the per-pixel analogue of AnimeInterp's segment matching (https://arxiv.org/abs/2104.02495) and the inclusion-matching paint-bucket colorization (https://arxiv.org/abs/2403.18342). Details travel with their part. The marginal-over-matches loss means flat regions have ambiguous offsets, but textured neighbours (outline, eye) disambiguate them. Cost: P·9·M lookups. Risk: over-commits to rigid motion at joints.

---

## 4. Temporal consistency across frames (small models)

Current: frames attend to each other at the U-Net bottleneck.

| Idea | Mechanism | Evidence | Cost |
|---|---|---|---|
| **Copy from other frames' decided pixels** | In rounds ≥2, add keys from the already-committed pixels of neighbouring frames (t±1) to the copy set; the query can copy "what I drew in the previous frame" | Multi-reference copying with a sliding window of the model's own predictions: https://arxiv.org/abs/1806.09594; short+long memory both help: https://arxiv.org/abs/2002.07793 | Keys from 2 extra frames; retrain needed (keys must be produced by the decoder) |
| Shared draft, then joint revise | Decode all frames one-shot, then revise passes that see all frames' soft predictions (2.3) | Self-conditioning: https://arxiv.org/abs/2208.04202 | As 2.3 |
| Frame-order commits | Commit the frames nearest the reference pose first (frame 0 or the "contact" poses), then the rest | [analysis]; analogous to MaskGIT ordering, but on the time axis | Decoding-only |
| Flow-warp consistency loss | Penalize differences between frame t and frame t−1 warped by the (copy-derived) flow | vid2vid temporal loss: https://arxiv.org/abs/1808.06601 | Training only; risky for pixel-exact art |
| Video tokens / masked video transformer | Joint space-time masking | MAGVIT: https://arxiv.org/abs/2212.05199 | Architecture change; too heavy |

Expected [analysis]: for LPC cycles (walk/slash with 6–9 frames), the biggest coherence error is likely small details flickering (present in some frames, absent in others). That is the same over-smoothing problem as §1, and the count- and offset-consensus decoders address it per frame. Copying from neighbour frames addresses it across frames.

---

## 5. Other quality-per-parameter levers

| Lever | Notes | Cost |
|---|---|---|
| EMA of weights for eval | Standard for generative models; usually cheap gains [general practice, not separately cited] | 1× params memory |
| Horizontal-flip TTA with left/right direction swap | If LPC left/right animations are mirror images (**unverified for LPC; check the data**), average logits of the flipped run. Caution: averaging marginals increases over-smoothing. Use it for alpha/silhouette, not colors | 2× inference |
| Data augmentation: palette permutation | Copy keys depend on colors via the reference encoder. Randomly permuting the palette (same permutation for reference and targets) forces copy-by-position. Decision 0013's color-copy failure suggests the model should not lean on color identity | 0 |
| Auxiliary loss on known pixels | Needed for any revise method (Mask-Predict/ReMDM/Token-Critic) | 0 params |
| Soft self-conditioning channel (2.3) | | ~hundreds of params |
| Relative-position copy bias (3.3) | | ~2k–17k params |

---

## 6. Ranked recommendations (cost → expected gain → test)

1. **Color-aggregated decision + confidence** (0.1). Cost 0. Gain: fixes the background-first/character-last ordering and the self-cond `own_known` selection bias, and makes confidence/margin orders meaningful. Test: decode_experiments at 32/64 px, existing checkpoints.
2. **Add detail metrics** (speckle P/R ±1 px, histogram L1) and stop judging decoders on pixel_exact alone (0.2). Cost: eval code.
3. **Expected-count-matched decoding** (soft: only add rare colors where P > τ). Cost ≈ 0. Directly targets the 12–25 missing speckles/frame. Test: decoding-only.
4. **Offset-consensus decoding** (3.4). Small cost. Targets speckles and part coherence. Decoding-only.
5. **Halton / margin + NMS orders, value sampling with annealed temperature, then ReMDM-conf loop** (2.1, 2.2). Decoding-only. Gain on coherence; evidence from Halton (FID 5.93 → 3.74) and ReMDM (4.85 → 4.45 at 64 steps), both on large models.
6. **Detail-weighted NLL** (+ optional count-constraint loss) (1.3). Retrain at 32 px, 0 params.
7. **Soft self-conditioning channel + loss on known pixels** (2.3, SUNDAE-style). Retrain, ~hundreds of params. Enables genuine revision.
8. **Critic head (Token-Critic as 1×1 head)** (1.4). Retrain. Strongest published evidence for revise-based gains (FID 6.56 → 4.69).
9. **Relative-position copy bias** (3.3). Retrain, ≤17k params.
10. **Autoguidance from an early checkpoint** (1.4). Decoding-only, 2× compute; speculative for discrete copy logits.
11. Neighbour-frame copy keys (§4); palette-permutation augmentation (§5).
12. Adversarial / GAN losses: last resort.

## 7. Open questions discovered

- How concentrated is the learned copy attention? Measure the entropy of p(j|p) *within* the correct color class. If mass is spread, 0.1 matters a lot. If it is peaked on the positionally-correct key, offset-consensus (3.4) is easy.
- Are the model's marginals calibrated (reliability diagram on P(color))? Count-matched decoding and confidence orders both assume calibration.
- What share of missing speckles are *misplaced by ≤1 px* vs *absent entirely*? The first calls for joint decoding/consensus; the second calls for training reweighting.
- Are LPC left/right directions exact mirrors (enables flip TTA and free augmentation)?
- How often do target pixels need global (>8 px) displacement (weapons, slash arcs)? This decides whether local/MAST-style copy is viable.
- Does self-conditioning with color-aggregated confidence change the 79.8% result (current self-cond used the biased per-key confidence)?
- Would a pixel_exact-optimal one-shot plus a *separate* small "detail restoration" pass (critic-guided) be simpler than full iterative decoding?
