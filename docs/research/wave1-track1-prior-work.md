# Wave 1 / Track 1: Prior work on sprite and 2D character animation

Research date: 2026-11-18. Scope: academic papers, open-source repos, commercial tools and datasets related to moviepix (one still pixel-art sprite + action + direction -> frames, ~4.5M params, CPU, copy-based).

Conventions:
- Every claim has a URL. "unverified" means only third-party or marketing sources were found, or the number could not be checked in the primary source.
- Numbers are copied from the cited source. Papers do not share a metric, so numbers in different rows cannot be compared directly.

---

## 1. Closest prior work (sprite / pixel-art specific)

### 1.1 Coutinho & Chaimowicz: pose-to-pose pixel-art sprite GANs (2022, 2024)

**Generating Pixel Art Character Sprites using GANs** (SBGames 2022). https://arxiv.org/abs/2208.06413
- Task: one sprite facing one direction -> the same character facing another direction (e.g. front -> right).
- Method: Pix2Pix. U-Net generator on 64x64x4 RGBA, PatchGAN discriminator, L1 (lambda 100) + adversarial loss. One model per source/target pair.
- Datasets: Tiny Hero (64x64, 912 images, 1 per pose), RPG Maker 2000 (32x24, 216), RPG Maker XP (48x32, 294), RPG Maker VX Ace (32x32, 408). Trained on a GTX 1050 Ti, about 1h30m per model.
- Results (FID on the test set): Tiny Hero 0.115, RM2K 2.306, RMXP 9.493, RMVX 5.495. Good results on Tiny Hero come from its modular construction ("it learned how to translate each body part it sees during training"), not from real generalization.
- Findings useful for us:
  - A **2x2 PatchGAN** (very local discriminator) gave the best results. 64x64 (single patch) gave "dangling pixels" outside the sprite.
  - **Keeping the alpha channel** in input and output gave better shapes and fewer stray pixels than RGB only.
  - When it fails, the output has "high-frequency color noise and very faint inner edges". Colors are not taken from the input palette.

**A Missing Data Imputation GAN for Character Sprite Generation** (SBGames 2024). https://arxiv.org/abs/2409.10721 . Code: https://github.com/fegemo/mdigan-characters (no license file found; TensorFlow 2.10; datasets included in the repo).
- Task: given the other 3 directions, produce the missing 4th (CollaGAN-style imputation).
- Model: generator 104.9M params, discriminator 44.7M. About 110 ms per batch on a GTX 1050.
- Data: Tiny Hero (912 images, **taken from LPC**, GPL 3.0 / CC-BY-SA 3.0) plus a scraped and modularly composed set, 14,202 four-direction examples in total (12,074 train / 2,128 test). Licenses of the scraped sources are not stated. Augmentation: hue rotation.
- Results (average over target poses): CollaGAN-3 inputs FID 1.508 / L1 0.04078; with 2 inputs FID 4.277 / L1 0.05035; with **1 input FID 8.393 / L1 0.06449**. Pix2Pix FID 4.091, StarGAN FID 2.288.
- "Conservative" input dropout (keep 3/2/1 inputs with probability 60/30/10%) was the best training scheme: FID 4.726 vs 18.847 without dropout.
- Weaknesses they report: colors use "many tone variations rather than the input palette", and they suggest palette quantization afterwards. Left/right is easy (close to a mirror). The back view shows faint faces.
- Related: journal version, "Pixel art character generation as an image-to-image translation problem using GANs", Graphical Models vol. 132 (2024), https://doaj.org/article/9cd3cc501bea4dc4820edffc09b93a3b . AIIDE 2022 "On the Challenges of Generating Pixel Art Character Sprites Using GANs", https://ojs.aaai.org/index.php/AIIDE/article/view/21951 (not read in full).

**Relevance to moviepix:** this is the closest academic line of work (LPC-derived data, RGBA, 64x64). It covers direction change only, not action frames. It regresses colors, so it is off-palette, and it is 20-30x larger than moviepix. Its numbers suggest that **more reference views help a lot** (FID 8.4 -> 1.5 going from 1 to 3 inputs).

### 1.2 Alonso & Patow: sprite in-betweening and directional frame generation (Oct 2026)

**Low-Cost Deep Learning for Automated Keyframe Interpolation in Sprite Animation**, https://arxiv.org/abs/2610.16423
**Neural Controlled Directional Frame Generation for Sprite Animation**, https://arxiv.org/abs/2610.27253
Code (MIT, 3 stars, no released weights found): https://github.com/NethamGH/deepinterframes_neural_sprite_inbetweening
- Task: two RGBA frames -> the frame in between (or previous/next frame in the second paper: three "expert" models).
- Model: U-Net, about 10M params per model, 206 MB file per model (the paper says this includes optimizer state). Adds 15 sprite-specific tricks: premultiplied alpha, 3-stage coarse-to-fine flow (+-5/+-2/+-1 px), alpha-gated blending, SE attention, separate RGB/alpha warping and others.
- Loss: L1 0.25 + SSIM 0.15 + Sobel edge 0.25 + color stats 0.10 + FFT 0.15 + Laplacian 0.10 (+ mask binarization 0.05).
- Data: **only 280 triplets** from 18 games, normalized to 128x128, split is **not source-disjoint**, **not released** (IP constraints).
- Results (FLIP, lower is better, 42 test triplets): their model 0.258, RIFE 0.313, Lucas-Kanade 0.449, CyclicGen 0.603. By category: walk 0.198 (best), idle 0.214, attack 0.265, effects 0.289-0.334. Ablation: removing skip connections raises FLIP to 0.398.
- Speed: "approximately 8-10 seconds per frame" on a Tesla T4 (as written in the paper, surprisingly slow for a 10M U-Net).
- Pilot user study with 5 artists: Spearman rho 0.89 between FLIP and artist ranking (pilot only, not a formal study).
- Several inconsistencies inside the papers (learning rate, epochs, percentages, sample counts). Treat numbers with care.

**Relevance:** confirms that small models can work for sprites, and lists losses aimed at sharp edges. But it needs two input frames, regresses RGBA (not palette-exact), uses tiny private data, and baselines were not retrained. It also suggests **FLIP** as a metric that tracks artist judgement.

### 1.3 Sprite Sheet Diffusion (Hsieh, Zhang, Yan; CMU; Dec 2024 / Mar 2025)

https://arxiv.org/abs/2412.03685 . Project: https://chenganhsieh.github.io/spritesheet-diffusion/
- Task: reference image + pose sequence -> sprite sheet frames.
- Method: fine-tuned Animate Anyone (https://arxiv.org/abs/2311.17117): ReferenceNet + pose guider + motion module, on Stable Diffusion 1.5 (built from Moore-AnimateAnyone, https://github.com/MooreThreads/Moore-AnimateAnyone). Param count not reported (SD 1.5-based models are around 1B params; not stated in the paper).
- Data: GameArt2D + SpriteDatabase, 152 action sequences / 916 frame sequences (another summary gives 619 triplets from 16 characters). Licenses not stated. Poses partly labeled by hand because "OpenPose generalize poorly to sprite illustrations".
- Results (in-sample): SSIM 0.659, PSNR 18.4, LPIPS 0.125 vs vanilla Animate Anyone SSIM 0.330.
- Limitations they state: overfitting, loses small details ("a gun disappearing"), weak on hair and props. Not pixel-art specific (no palette or grid handling). No code release found.

### 1.4 SketchBetween (Loftsdóttir & Guzdial, FDG 2022)

https://arxiv.org/abs/2209.00185 . Code: https://github.com/ribombee/SketchBetween
- Task: 2 rendered keyframes + sketched in-betweens -> rendered in-betweens.
- Method: VQ-VAE (Keras) on 5x128x128x3 stacks, SSIM loss. Data: MGIF (1,000 cartoon animal GIFs, 128x128), with sketches made by Canny edges.
- Results: SSIM 0.943 / PSNR 27.48 vs an adapted FOMM 0.799 / 19.46. Without sketches: 0.883 / 20.19.
- Limitations: blurry output and lost details. Fails on a **pixel-art donkey jump** (large motion at low frame rate).
- Lesson: FOMM-style keypoint/affine warping "fails to capture changes in the shapes of objects that are not affine". Sprite motion is stylized.

### 1.5 Other early sprite generation work (cited inside 1.1)

- Serpa & Rodrigues, "Towards Machine-Learning Assisted Asset Generation for Games: A Study on Pixel Art Sprite Sheets", SBGames 2019. Line art -> shaded and segmented sprite. https://ieeexplore.ieee.org/document/8924853 (described in https://arxiv.org/abs/2208.06413)
- Hong, Kim, Kang, "Game sprite generator using a multi discriminator GAN", KSII TIIS 2019: bone graph (target pose) + appearance sprite -> posed sprite. Needs pose data, which pixel art does not have. http://itiis.org/digital-library/manuscript/2473 (described in https://arxiv.org/abs/2208.06413)
- Gonzalez, Guzdial, Ramos, "Generating Gameplay-Relevant Art Assets with Transfer Learning" (Pokémon type swap, VAE): blurry, noisy colors. https://arxiv.org/abs/2010.01681
- Saravanan & Guzdial, "Pixel VQ-VAEs for Improved Pixel Art Representation" (EXAG 2022): VQ-VAE built for single-pixel detail. https://arxiv.org/abs/2203.12130

### 1.6 Pixel-art generation (still images) with palette control

- **SD-πXL** (Binninger & Sorkine-Hornung, SIGGRAPH Asia 2024), https://arxiv.org/abs/2410.06236 . Score distillation from a pretrained diffusion model. The output image is an HxWxn tensor of palette logits with **Gumbel-softmax** sampling, so every pixel is exactly a palette color, and outputs are crisp. The paper says earlier methods did not strictly follow a palette. Per-image optimization (slow, not a feed-forward model).
- **PixelBench** (Nov 2026), https://arxiv.org/abs/2611.20628 : benchmark for LLM agents drawing and editing pixel art with code tools, with an MLLM judge validated against humans. Abstract does not mention animation (unverified whether editing tasks include frames; one search snippet said so).
- What goes wrong in AI "pixel art" from generic image models (vendor blog, one example image, treat as anecdote): pixel blocks 26-32 px wide instead of uniform, 44,725 unique colors in a sprite asked to have ~12, ~4.5% anti-aliased transition pixels, and per-frame palette drift causing flicker. https://rangy.ai/blog/ai-pixel-art

---

## 2. Adjacent academic work (cartoon animation, image animation, correspondence)

### 2.1 "Copy pixels instead of generating them"

| Work | What is copied | Why it matters to moviepix | URL |
|---|---|---|---|
| Appearance Flow (Zhou et al. 2016) | Each target pixel predicts 2D coordinates in the input image to copy from (novel view synthesis) | The direct ancestor of moviepix's copy idea, with continuous coordinates and bilinear sampling (so it blurs) | https://arxiv.org/abs/1605.03557 |
| CDNA / pixel-motion video prediction (Finn, Goodfellow, Levine, NIPS 2016) | Action-conditioned prediction of a distribution over how pixels move from the previous frame | Action-conditioned + copy; they found modeling motion rather than appearance generalizes better to unseen objects | https://arxiv.org/abs/1605.07157 |
| Deep Voxel Flow (Liu et al., ICCV 2017) | Interpolated frame made by moving pixels from neighbours | Argues direct pixel regression is blurry; copying is sharp | https://arxiv.org/abs/1702.02463 |
| GFLA (Ren et al. 2020) | Global flow + local attention to sample source features for pose transfer | "Attention picks where to copy from" for pose transfer, at feature level | https://arxiv.org/abs/2003.00696 |
| CoCosNet (Zhang et al., CVPR 2020) | Dense correlation matrix warps an exemplar into the target layout | The same soft-attention warp; v2 works at full resolution | https://arxiv.org/abs/2004.05571 , https://github.com/microsoft/CoCosNet |
| AnimeInterp (Li Siyao et al., CVPR 2021) | Segment-Guided Matching: matches flat color **segments**, not pixels, then flow refinement | Cartoons have flat regions with no texture; matching at segment level fixes that | https://arxiv.org/abs/2104.02495 |
| BasicPBC / Inclusion Matching (Dai et al., CVPR 2024) | Copies a **color per line-art segment** from a reference frame; learns "inclusion" between segments to handle occlusion | Closest analogue to moviepix's palette-exact copy at region level. Code S-Lab License 1.0 (non-commercial), 308 stars | https://arxiv.org/abs/2403.18342 , https://github.com/ykdai/BasicPBC |
| Paint bucket with color design sheets (Dai et al. 2024) | Region colors copied from a character design sheet with semantic color labels | Shows design-sheet (multi-reference) conditioning works for exact colors | https://arxiv.org/abs/2410.19424 |
| EbSynth / Stylizing Video by Example (Jamriška et al., SIGGRAPH 2019) | Patch-based copying from painted keyframes | Copy keeps exact texture; no training needed | https://dcgi.fel.cvut.cz/home/sykorad/ebsynth.html |
| Few-shot patch-based training (Texler et al. 2020) | Small network trained from scratch on a few keyframes, real-time | Shows a tiny per-sequence network can be enough | https://arxiv.org/abs/2004.14489 |
| MaskGIT (Chang et al. 2022) | Masked token prediction, parallel decoding, up to 64x faster than autoregressive | moviepix's training scheme; no pixel-art or sprite use of it was found | https://arxiv.org/abs/2202.04200 |

### 2.2 Image animation / motion transfer from a single image

| Work | Method | Notes for sprites | URL |
|---|---|---|---|
| FOMM (Siarohin et al., NeurIPS 2019) | Learned keypoints + local affine transforms + occlusion-aware generator | SketchBetween found it fails on non-affine stylized motion (manes, tails) | https://arxiv.org/abs/2003.00196 |
| Thin-Plate Spline Motion Model (Zhao & Zhang, CVPR 2022) | TPS warps instead of affine; MIT code | Smooth warps blur 1-px details unless followed by a copy/quantize step | https://arxiv.org/abs/2203.14367 , https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model |
| Animate Anyone (Hu et al. 2023) | Diffusion + ReferenceNet + pose guider | Base of Sprite Sheet Diffusion | https://arxiv.org/abs/2311.17117 |
| Animate-X (ICLR 2025) | Diffusion, generalizes to anthropomorphic characters; A2Bench (500 characters) | Big video model; non-human bodies are a known hard case | https://arxiv.org/abs/2410.10306 |
| MikuDance (ICCV 2025) | Diffusion for stylized character art with mixed motion | Big video model | https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_MikuDance_Animating_Character_Art_with_Mixed_Motion_Dynamics_ICCV_2025_paper.html |
| Animated Drawings (Meta; Smith et al., TOG 2023) | Detect -> segment -> 2D pose -> ARAP rig -> retarget BVH mocap | Non-learned deformation at run time; MIT code and weights; 12.8k stars; **archived Sept 2025** | https://github.com/facebookresearch/AnimatedDrawings , https://doi.org/10.1145/3592788 |
| Neural Puppet (Poursaeed et al., WACV 2020) | Layered 2.5D template mesh fitted per frame + image refinement network | Part-based puppet from one segmented frame | https://arxiv.org/abs/1910.02060 |
| CoNR (Lin et al., IJCAI 2023) | Character sheet (several references) + "Ultra-Dense Pose" -> new pose; cross-view feature warping; 700k-image dataset | More references -> clearly better results | https://arxiv.org/abs/2207.05378 , https://github.com/megvii-research/CoNR |
| 2D motion retargeting (Aberman et al., SIGGRAPH 2019) | Splits 2D pose sequences into motion, skeleton and view codes, then recombines | Idea for transferable "action codes" | https://arxiv.org/abs/1905.01680 |

### 2.3 Cartoon in-betweening and line-art video

- **ToonCrafter** (Xing et al., SIGGRAPH Asia 2024), https://arxiv.org/abs/2405.17933 , https://github.com/ToonCrafter/ToonCrafter : adapts a live-action video diffusion prior to cartoons ("toon rectification learning"), dual-reference 3D decoder, sketch encoder. Released model works at 512x320, ~2 s at 8 FPS (per the Hugging Face model card summary). License reported as Apache-2.0 on GitHub, but sources disagree (unverified). Alonso & Patow list it at 2.3 GB and 8-12 s inference (https://arxiv.org/abs/2610.27253).
- **AniDoc** (CVPR 2025), https://arxiv.org/abs/2412.14173 : video-diffusion line-art colorization using explicit reference-to-sketch correspondence matching. Code https://github.com/yihao-meng/AniDoc .
- **Sprite-from-Sprite** (Zhang, Wong, Liu; SIGGRAPH Asia 2022), https://lllyasviel.github.io/GitPageToonDecompose , code https://github.com/lllyasviel/ToonDecompose : splits cartoon videos into sprite layers with pixel MLPs.
- **MarioNette** (Smirnov et al., NeurIPS 2021), https://arxiv.org/abs/2104.14553 : learns a dictionary of small transparent sprite patches + placements that rebuild game frames, without supervision.
- "Deep Sprite-based Image Models: An Analysis" (2026), https://arxiv.org/abs/2604.19480 : about image decomposition into recurring patterns, **not** game sprites. Not relevant.

---

## 3. Open-source repos

| Repo | What it is | License | Stars (seen 2026-11) | Model / size | Local? | URL |
|---|---|---|---|---|---|---|
| Universal LPC Spritesheet Character Generator | Layered LPC art + web generator. Animations: spellcast, slash, thrust, walk, shoot, hurt; expanded: bow, climb, run, jump (many assets not yet drawn for these). 64x64 frames, oversize 128/192 for some | Art: CC0 / CC-BY / CC-BY-SA / OGA-BY / GPL per file; `CREDITS.csv` lists each file | ~2.0k | none (art only) | yes | https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator |
| fegemo/mdigan-characters | Code + datasets for 1.1 | none found | 2 | 105M generator | yes (GPU) | https://github.com/fegemo/mdigan-characters |
| deepinterframes | Code for 1.2 | MIT | 3 | ~10M per model, no weights found | yes | https://github.com/NethamGH/deepinterframes_neural_sprite_inbetweening |
| SketchBetween | Code for 1.4 | not checked | not checked | VQ-VAE | yes | https://github.com/ribombee/SketchBetween |
| BasicPBC | Segment color copying for cel animation | S-Lab License 1.0 (non-commercial) | 308 | not stated | yes | https://github.com/ykdai/BasicPBC |
| Thin-Plate Spline Motion Model | Image animation | MIT (per index) | not checked | not checked | yes (GPU) | https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model |
| AnimatedDrawings | Rig + retarget drawings | MIT (code, weights, data) | 12.8k, archived | detector + pose model | yes | https://github.com/facebookresearch/AnimatedDrawings |
| ToonCrafter | Cartoon in-betweening | Apache-2.0 per GitHub (sources conflict) | not checked | video diffusion, ~2.3 GB | GPU only | https://github.com/ToonCrafter/ToonCrafter |
| Sprite Sage | GPL-3.0 sprite editor + AI frames. Local model is Qwen Image 2.1 INT8 (research-only license, Windows + NVIDIA); cloud via OpenAI/Google. No dedicated animation model; "From Template" (experimental) uses a 3D motion template | GPL-3.0 | 30 | general image model | partly | https://github.com/keystoneintelligence/spritesage |
| PixelHand, pixelmaker-godot-plugin, perfectpixel-studio, pixel-plugin | Agent/LLM tools or wrappers around cloud image APIs; no trained sprite animator (from search snippets, unverified in detail) | various | small | n/a | mostly cloud | https://github.com/SirLichDevCraft/PixelHand , https://github.com/kbraiden/pixelmaker-godot-plugin/ , https://github.com/gykim80/perfectpixel-studio , https://github.com/willibrandon/pixel-plugin |
| spraix (SDXL fine-tune) | Text -> 9-frame sprite sheet image, trained on 62 sheets | GPL-3.0 dataset | small | SDXL | GPU | https://huggingface.co/datasets/pawkanarek/spraix_1024_9frames |
| PIXEL-T2I (undergrad project) | Linked from the LPC Hugging Face datasets in section 5 | not checked | not checked | not checked | n/a | linked from https://huggingface.co/datasets/carlosuperb/lpc-action-pixel-art-diffusion |

**Finding:** no open-source, trained, **small** sprite-animation model with released weights was found. Open repos are either research code with no weights, general cartoon/video models of GB size, or wrappers around cloud APIs.

---

## 4. Commercial tools

| Tool | What it does | Input -> output | Pricing | Local? | Known issues | Sources |
|---|---|---|---|---|---|---|
| **PixelLab** | The broadest pixel-art tool set. Animate with text, **animate with skeleton** (template skeleton, "animation to animation", or manual edits; freeze 1 -> generate 2 frames, then inpaint and iterate), 4/8-direction characters, rotation, inpainting, interpolation, outfit transfer, tilesets | Skeleton tool canvases 16-256 px; `animate-with-text` 64x64, 4 frames; v3 up to 256 px and 16 frames; skeleton v3 3/8/15 frames | API per call, e.g. animate-with-skeleton 32x32 ≈ $0.0136; animate-character v3 (4 frames) 64 px ≈ $0.0129 per direction; Pro endpoints $0.08-0.15. Subscriptions from ~$12/mo (third-party figure); skeleton tool needs Tier 1+ | No ("run on cloud GPUs"). Aseprite extension, web editor, API, MCP | Workflow assumes manual skeleton fixes and repeated regeneration; a "fixed head" option exists for face consistency. Reviews say frames still need cleanup (secondary sources) | https://www.pixellab.ai/ , https://www.pixellab.ai/pixellab-api , https://www.pixellab.ai/docs/tools/animate-with-skeleton , https://ludo.ai/compare/ludo-vs-pixellab |
| **Retro Diffusion** | Custom pixel-art image models (RD Fast/Mini/Core/Pro) + **Advanced Animations**: one still -> 4-16 looping frames with presets (walk, idle, attack, jump, crouch, destroy, rotate, subtle motion, custom); strict palette input; palette tools | rd-animation styles have fixed sizes (four-angle walking 48x48 only; small sprites 32) | Advanced Animations from $0.10, most moves $0.14, custom $0.25 | Website/API: cloud. **Aseprite extension runs locally** ($65 one-time; $20 lite), but the developer says it "can't handle any kind of animation"; 64x64 takes ~5 s on an RTX 3060 and ~10 min on an i5-8300H CPU | Image references "are not used as direct initial images" in rd-animation, so identity can drift from the input | https://www.retrodiffusion.ai/ , https://replicate.com/retro-diffusion/rd-animation/readme , https://replicate.com/retro-diffusion/rd-animation/versions/2e4eabbc836578423e151ba27352a1d8a566c6f1186c0da3ddf66c54deacf28b/api , https://astropulse.itch.io/retrodiffusion |
| **Scenario** | Hosts third-party models including Retro Diffusion Animation; spritesheet apps (run cycle, action sprite studio) | text/image -> sheet | not checked | cloud | A competitor says Scenario sheets come from a video model "where pixels can drift" (biased source) | https://www.scenario.com/models/retro-diffusion-animation , https://help.scenario.com/articles/9088582240-create-spritesheets-with-scenario , https://www.spritecook.ai/compare/scenario |
| **Sorceress** | Quick Sprites (prompt -> retro sprite sheets: 4/8-dir walk, idle, attack, cast, hit, death) runs **Retro Diffusion rd-animation via Replicate**, 9 credits each; Auto-Sprite v2 = character image -> AI video -> frame extraction | text -> PNG grid | credits (100 starter credits) | cloud | vendor-reported only | https://sorceress.games/tutorials/quick-sprites , https://sorceress.games/blog/cast-an-ai-game-sprite-generator-walk-cycle-2026 , https://sorceress.games/pages/auto-sprite |
| **Ludo.ai** | General game-asset suite; pixel styles on a known grid; animation presets on sprites; sprite sheet export | image/text -> sheet | ~$12-50/mo (third-party figures, unverified) | cloud | not checked | https://ludo.ai/features/sprite-generator , https://ludo.ai/tools/pixel-art-sprite-generator |
| **God Mode AI** | Upload a character image, pick from ~100 actions/views (side-scroller, 8-dir isometric), pixel-art conversion step | image -> sheet | conflicting ($10-19/mo), unverified | cloud | only third-party descriptions found (unverified) | https://www.producthunt.com/posts/god-mode-ai |
| **AutoSprite** | Character image -> idle/walk/run/jump/attack sheets; Unity/Godot/GameMaker/Phaser export | image -> sheet | free 15 credits/mo; $12/$29/$99 (third-party) | cloud; MCP + REST | unverified | https://autosprite.itch.io/autosprite , https://alternativeto.net/software/autosprite/about |

Patterns across commercial tools:
1. Everything that animates runs in the cloud and charges per call. The only local product (Retro Diffusion's Aseprite extension) does not animate and needs a GPU to be practical.
2. Two technical families: (a) diffusion trained on pixel art with fixed sheet layouts (Retro Diffusion), and (b) skeleton/pose conditioning plus inpainting with a human in the loop (PixelLab). Some tools also run a general video model and then extract and pixelate frames (Sorceress Auto-Sprite, per vendor).
3. Recurring complaints: identity and palette drift between frames, foot sliding, extra colors and anti-aliasing, and cleanup still needed for final assets. https://rangy.ai/blog/ai-pixel-art , https://dev.to/makko_ai/ai-game-art-generator-characters-backgrounds-animations-and-why-consistency-is-the-hard-part-4apm (secondary sources)
4. No vendor publishes quantitative accuracy against ground-truth frames.

---

## 5. Datasets

| Dataset | Content | Size | License | URL |
|---|---|---|---|---|
| LPC / Universal LPC spritesheets | Layered humanoid parts, 4 directions, many actions, 64x64 | thousands of layer files | CC0 / CC-BY / CC-BY-SA / OGA-BY / GPL per file, `CREDITS.csv` | https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator , https://opengameart.org/comment/112623 |
| Tiny Hero (in Coutinho repo) | LPC characters, 4 directions, 64x64 | 912 images | GPL 3.0 / CC-BY-SA 3.0 | https://arxiv.org/abs/2409.10721 |
| Coutinho RPG Maker sets + misc | 4-direction sprites | 216 / 294 / 408 / 12,372 | not stated (RPG Maker assets are usually not freely redistributable; unverified) | https://github.com/fegemo/mdigan-characters |
| carlosuperb/lpc-action-pixel-art-diffusion | LPC action sheets (walk, thrust, slash), pairs with a 4-view set | 50,000 rows, 2.74 GB | CC-BY-SA-3.0 | https://huggingface.co/datasets/carlosuperb/lpc-action-pixel-art-diffusion |
| carlosuperb/lpc-4view-pixel-art-diffusion | LPC 4-view stills with text captions | not checked | not checked | https://huggingface.co/datasets/carlosuperb/lpc-4view-pixel-art-diffusion |
| Sprite Sheet Diffusion data | GameArt2D + SpriteDatabase | 152 action sequences | not stated; not released | https://arxiv.org/abs/2412.03685 |
| Alonso & Patow triplets | 18 games | 280 triplets | not released (IP) | https://arxiv.org/abs/2610.16423 |
| spraix | 9-frame sheets with captions | 62 | GPL-3.0 | https://huggingface.co/datasets/pawkanarek/spraix_1024_9frames |
| alucard-sprites | ~128x128 RGBA sprites with CLIP captions | not checked | custom "fair-1.0.0" | https://huggingface.co/datasets/evilsocket/alucard-sprites |
| MGIF | 1,000 cartoon animal GIFs, 128x128 | 1,000 | not checked | used in https://arxiv.org/abs/2209.00185 |
| ATD-12K | Anime triplets for interpolation | 12,000 triplets | not checked | https://arxiv.org/abs/2104.02495 |
| PaintBucket-Character | 3D-rendered cel frames + design sheets | 11,345 train / 3,000 test | S-Lab (non-commercial) | https://github.com/ykdai/BasicPBC |
| CoNR data | Hand-drawn + synthesized anime | >700,000 images | not checked | https://arxiv.org/abs/2207.05378 |
| Amateur Drawings / ChildlikeSHAPES | Children's drawings with annotations | not checked | MIT / CC-BY 4.0 | https://github.com/facebookresearch/AnimatedDrawings |

**Finding:** besides moviepix's own ~21k-character build, the largest open **animated** pixel-art set found is the carlosuperb LPC action set (3 actions, CC-BY-SA). Every non-LPC sprite animation dataset found is tiny (62-280 items) or unreleased. No standard benchmark for sprite animation exists (https://arxiv.org/abs/2412.03685 and searches; no benchmark found).

---

## 6. Comparison: where moviepix sits

| | Input | Output exactness | Params | Runs on | Data | Open weights |
|---|---|---|---|---|---|---|
| moviepix copy_animator | 1 sprite + action + direction | palette-exact by construction (copy) + ~2% new colors | ~4.5M | laptop CPU, 16-120 ms | ~21k LPC characters, credited | planned |
| Coutinho MDI-GAN | 3 (or 1) directions | regressed RGBA, off-palette | 105M G | GPU | 14k (partly unclear licenses) | code only |
| Alonso & Patow | 2 frames | regressed RGBA | ~10M x3 | GPU, 8-10 s/frame (as reported) | 280 private triplets | code only, no weights found |
| Sprite Sheet Diffusion | 1 image + poses | diffusion RGB | ~1B class (unverified) | GPU | 152 sequences | no |
| ToonCrafter | 2 frames (+ sketch) | diffusion RGB, 512x320 | ~2.3 GB | GPU | large anime video | yes |
| PixelLab skeleton | 1 image + skeleton | palette option | unknown | cloud | unknown | no |
| Retro Diffusion animation | 1 image or text + preset | strict palette option | unknown | cloud | licensed art (vendor claim) | no |

---

## 7. Synthesis

### 7.1 What works for small, palette-exact pixel art

1. **Copy, don't regress.** Every regression-based sprite model reports off-palette tones, color noise or blur: Coutinho 2022/2024 (https://arxiv.org/abs/2208.06413 , https://arxiv.org/abs/2409.10721), SketchBetween (https://arxiv.org/abs/2209.00185). Copy- or flow-based methods are sharper in general video (Deep Voxel Flow, https://arxiv.org/abs/1702.02463), and in cel animation the best colorization methods copy region colors (BasicPBC, https://arxiv.org/abs/2403.18342). moviepix's 77-80% vs 51% (U-Net) fits this pattern.
2. **Hard discrete outputs.** SD-πXL's per-pixel palette logits + Gumbel-softmax is the main published way to get exact palettes and crisp pixels (https://arxiv.org/abs/2410.06236). moviepix's copy-pointer is the same idea, with the "palette" being the reference's own pixels.
3. **Very local adversarial or edge signals.** 2x2 PatchGAN was best in Coutinho 2022. Edge, Laplacian and FFT losses carry ~50% of the loss weight in Alonso & Patow.
4. **Alpha as a first-class channel.** RGBA beats RGB (Coutinho 2022). Premultiplied alpha, separate alpha warping and binarized masks help (Alonso & Patow).
5. **More reference views.** 1 -> 3 inputs: FID 8.39 -> 1.51 (https://arxiv.org/abs/2409.10721). CoNR also gets clearly better with more character-sheet views (https://arxiv.org/abs/2207.05378).
6. **Segment-level matching for flat-color art.** AnimeInterp and BasicPBC match regions, not pixels, because flat regions have no texture to match on (https://arxiv.org/abs/2104.02495 , https://arxiv.org/abs/2403.18342).

### 7.2 What fails

- Generic video or image diffusion: drifting identity and palette, anti-aliasing, extra colors, off-grid pixels, large compute (https://rangy.ai/blog/ai-pixel-art ; Sprite Sheet Diffusion's own limitations).
- Affine or keypoint warping (FOMM) on stylized, non-rigid sprite motion (SketchBetween, https://arxiv.org/abs/2209.00185). Smooth warps (TPS) also interpolate colors unless followed by quantization.
- Off-the-shelf pose estimators on sprites (https://arxiv.org/abs/2412.03685). Any pose-conditioned approach must make its own skeletons. LPC gives these almost for free, because frames follow fixed templates.
- Tiny or leaky datasets: 280 triplets with no source-disjoint split; early stopping on the test set in Coutinho 2024. moviepix's character-level splits are a methodological advantage.

### 7.3 Ideas to borrow (ranked by expected value per cost for a ~4.5M CPU model)

1. **Edge/outline-weighted copy loss + a tiny 2x2 PatchGAN fine-tune** to fight over-smoothing.
   - Source: https://arxiv.org/abs/2208.06413 (2x2 patch), https://arxiv.org/abs/2610.16423 (Sobel/Laplacian weights).
   - In moviepix terms: up-weight the cross-entropy on pixels whose target differs from any 4-neighbour (outlines, single-pixel highlights). Optionally add a 2x2-patch discriminator on the one-hot or copied output, used for a short fine-tune only, so inference cost does not change.
   - Expected gain: more single-pixel details kept. Overall accuracy may not change much, so measure a "detail-pixel accuracy" metric too.
   - Cost: hours of training; no inference cost.
2. **Part/layer auxiliary head from LPC layer ids.**
   - Inspired by segment matching: https://arxiv.org/abs/2104.02495 , https://arxiv.org/abs/2403.18342 .
   - moviepix composes characters from layers, so every target pixel has a known source layer (hair, torso, legs, weapon...). Add a small head that predicts the layer of each output pixel, and/or restrict copy attention to reference pixels of the same predicted layer.
   - Expected gain: fewer "region swapped to wrong color" errors (the failure seen in the palette-classification baseline) and more coherent copying.
   - Cost: very low (labels are free; one extra 1x1 conv head). This is unique to synthetic layered data.
3. **Optional multi-reference input with conservative input dropout.**
   - Source: https://arxiv.org/abs/2409.10721 , https://arxiv.org/abs/2207.05378 .
   - Allow 1-4 reference directions (keys = all reference pixels). Train with keep-1/2/3 probabilities.
   - Expected gain: large for back/side views that the single reference does not show (Coutinho: FID 8.4 -> 1.5). Inference cost grows linearly with the number of keys.
   - Product win: "give me your 4-direction idle, get every action".
4. **Keyframe / in-betweening mode for free.**
   - Source: MaskGIT masking (https://arxiv.org/abs/2202.04200); in-betweening tools in PixelLab (interpolation endpoint) and Alonso & Patow.
   - moviepix already trains with random masking. Expose "artist fixes some frames, model fills the rest", and train with whole-frame masks plus partially known frames as extra copy sources.
   - Expected gain: a feature the cloud tools charge for (PixelLab interpolation-v2 $0.08-0.15 per call). The 2026 papers need a separate 10M model for each direction of this.
   - Cost: training-data change only.
5. **Copy from previous output frames (temporal copy), not only from the reference.**
   - Source: CDNA (https://arxiv.org/abs/1605.07157), DVF (https://arxiv.org/abs/1702.02463).
   - Make keys = reference pixels + pixels of already-decoded frames.
   - Expected gain: better frame-to-frame consistency, and a fix for "new color" pixels (motion blur, slash trails) that appear in one frame and persist.
   - Cost: moderate; needs ordered or iterative decoding, which may also make iterative decoding finally useful.
6. **Local offset prior on attention.**
   - Source: appearance flow (https://arxiv.org/abs/1605.03557), GFLA global flow + local attention (https://arxiv.org/abs/2003.00696).
   - Bias attention logits toward the location predicted by a coarse per-part offset (or the same pixel shifted by the expected body motion for this action/frame).
   - Expected gain: sample efficiency and out-of-style robustness, since copying becomes "mostly short moves".
   - Cost: small.
7. **Better metrics.**
   - Add FLIP (pilot rho 0.89 with artists; https://arxiv.org/abs/2610.16423), palette-adherence %, outline/detail-pixel accuracy and identity-swap rate.
   - Report per-action numbers, as Alonso & Patow do by category.
   - Cost: low. Makes moviepix comparable with the only other sprite in-betweening work.
8. **Hue/palette-permutation augmentation + out-of-style test sets.**
   - Source: Coutinho hue rotation (https://arxiv.org/abs/2409.10721).
   - A copy model should be close to palette-invariant. Permuting reference colors consistently in reference and target tests and enforces this.
   - Out-of-style evaluation could use CC0/CC-BY sprites (OpenGameArt), not RPG Maker sets whose licenses are unclear.
9. **Skeleton/motion-code conditioning for custom actions (later).**
   - Source: PixelLab animate-with-skeleton and "animation to animation" (https://www.pixellab.ai/docs/tools/animate-with-skeleton); 2D retargeting (https://arxiv.org/abs/1905.01680).
   - Replace or augment the action id with a sequence of 2D joint positions (derivable from LPC templates) or a motion code taken from an example animation of another character.
   - Expected gain: actions outside the fixed list, and non-LPC body proportions.
   - Cost: high (needs skeleton labels, new conditioning). Do after the core is solid.
10. **Gumbel/straight-through hard copy during training.**
    - Source: SD-πXL (https://arxiv.org/abs/2410.06236).
    - Use it only if over-smoothing comes from soft attention averaging in training. If decoding is already argmax over a categorical, this is low priority.

### 7.4 The gap moviepix can own

- **Open + local + CPU-fast + palette-exact + licensed data.**
  - All animating products found are cloud-only and charge per call.
  - The one local pixel-art model product (Retro Diffusion extension) does not animate and needs minutes per 64x64 image on CPU (https://astropulse.itch.io/retrodiffusion).
  - Academic sprite models are 10M-105M+ params, regress colors, and ship no weights or data. moviepix at 4.5M params and 16-120 ms on CPU is one to two orders of magnitude smaller than every alternative found, and it is exact by construction.
- **A clean, credited, character-split benchmark for sprite animation.** None exists. Existing sets are tiny, unreleased, or not source-disjoint. A released LPC-derived benchmark with CREDITS.csv, per-action metrics and an out-of-style split would be a citable contribution.
- **A "reference-pixel pointer" formulation for exact-palette generation.** No paper was found that frames sprite animation as per-pixel pointer selection over the reference (+ transparent + new color). The nearest are appearance flow (continuous, blurry), BasicPBC (segment colors for line art) and SD-πXL (fixed palette, per-image optimization).
- **Runs in the browser** (ONNX), so it can live inside Aseprite or Godot plugins without servers. Nothing comparable was found.

### 7.5 Open questions for follow-up

1. How does moviepix do on out-of-style sprites (RPG Maker-style 32x32 side views, CC0 OpenGameArt packs, non-humanoids)? Which licensed sources can serve as test sets?
2. Is the over-smoothing caused by the loss (majority color wins), by the attention resolution, or by the decoder? A per-pixel-type error breakdown (outline, highlight, interior, new color) would tell which of ideas 1, 2 or 6 to try first.
3. Does restricting copy attention to the same LPC layer or body part reduce region-swap errors, and does it hurt when layers overlap (weapons, capes)?
4. How much do extra reference directions help moviepix (repeat Coutinho's 1/2/3-input study)?
5. Can whole-frame masking turn moviepix into a competitive in-betweener? Compare with RIFE and with the Alonso & Patow FLIP numbers on a shared, licensed test set.
6. What is the right decoding order for iterative mode (by frame, by confidence, by layer)? Does temporal copying make iterative decoding beat one-shot?
7. Can a skeleton sequence or motion code extracted from LPC frame templates replace action ids without losing accuracy, opening custom actions?
8. Licensing: does training on CC-BY-SA / GPL LPC layers impose terms on released weights or outputs? Needs a dedicated legal/licensing track. The Hugging Face LPC datasets use CC-BY-SA-3.0 for the data.
9. Do PixelLab and Retro Diffusion outputs keep exact identity and palette with the input sprite? A small, careful side-by-side using a few paid calls would need the maintainer's approval before spending.
10. Is FLIP (or another perceptual metric) well correlated with pixel artists' judgement at 32x32 and 64x64? The only evidence is a 5-artist pilot.
11. Check the full PixelBench paper for animation tasks and whether its judge protocol could rate moviepix outputs (https://arxiv.org/abs/2611.20628).
