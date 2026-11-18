# Wave 2: Verification of references and novelty check

Date: 2026-11-18. Method: every entry below was checked against the primary source (arXiv abs/HTML pages, the paper PDF, the Hugging Face model card, the GDC article). Where only a secondary source was reachable, this is stated. Nothing was taken from the wave-1 notes without re-checking.

Status labels: **VERIFIED** (the source exists and the stated facts match), **PARTLY** (the source exists, but some stated facts are wrong, imprecise or only in the body), **NOT FOUND**, **CONTRADICTED**.

---

## 1. Verification table

| # | Claim checked | Status | What the primary source says | Evidence |
|---|---|---|---|---|
| 1 | arXiv 2610.16423, Alonso & Patow, sprite in-betweening, ~10M U-Net, 280 triplets, FLIP 0.258 vs RIFE 0.313 | **VERIFIED** (one nuance) | "Low-Cost Deep Learning for Automated Keyframe Interpolation in Sprite Animation", Kenneth Alonso, Gustavo Patow, v1 12 Oct 2026, no venue. Body: "approximately 10M trainable parameters"; 280 triplets total, split 196 train / 42 val / 42 test, from 18 games, **not source-disjoint**; FLIP 0.258 ± 0.156 vs RIFE 0.313 ± 0.214 (also LK 0.449, CyclicGen 0.603); 8-10 s/frame on a T4. Nuance: the abstract gives neither the param count nor 280 (it says "a few carefully curated" triplets); 280 is the **total**, not the training count | https://arxiv.org/abs/2610.16423 , https://arxiv.org/html/2610.16423 |
| 2 | arXiv 2610.27253, Alonso & Patow | **VERIFIED** (one nuance) | "Neural Controlled Directional Frame Generation for Sprite Animation", same authors, v1 19 Oct 2026, cs.GR, no venue. Three ~10M U-Net "experts" (previous / middle / next frame from two input frames); 206 MB per checkpoint "including optimizer state"; FLIP 0.265 / 0.258 / 0.271; RIFE 0.313 (middle task only). "Directional" means **temporal** direction (previous/next), not facing direction. Its abstract says "280 sprite triplets used for training", which conflicts with its own 196/42/42 split | https://arxiv.org/abs/2610.27253 , https://arxiv.org/html/2610.27253v1 |
| 3 | arXiv 2208.06413, Coutinho & Chaimowicz | **VERIFIED** | "Generating Pixel Art Character Sprites using GANs", Flávio Coutinho, Luiz Chaimowicz, 15 Aug 2022, "submitted to SBGames 2022". Pix2Pix, front -> other direction, evaluated with FID | https://arxiv.org/abs/2208.06413 |
| 4 | arXiv 2409.10721: FID 8.39 -> 1.51 with 1 -> 3 views; 105M generator | **VERIFIED** | "A Missing Data Imputation GAN for Character Sprite Generation", same authors, 16 Sep 2024, "Published in SBGames 2024". Generator "104,887,616" params, discriminator 44,726,272. Table 3, CollaGAN averaged over targets: 1 input 8.393, 2 inputs 4.277, 3 inputs 1.508. 14,202 examples (12,074 / 2,128). Conservative input dropout 4.726 vs 18.847 without. The abstract has no numbers; all are from the body | https://arxiv.org/abs/2409.10721 , https://arxiv.org/html/2409.10721 |
| 5 | arXiv 2412.03685 Sprite Sheet Diffusion | **VERIFIED** (with corrections) | "Sprite Sheet Diffusion: Generate Game Character for Animation", Cheng-An Hsieh, Jing Zhang, Ava Yan; v1 4 Dec 2024, v2 16 Mar 2025; **no venue** (comments give only the project page). Fine-tuned Animate Anyone. Data: 152 action sequences / 916 frame sequences, **40 characters** (28 seen, 12 unseen), GameArt2D + SpriteDatabase. In-sample SSIM 0.659, PSNR 18.405, LPIPS 0.125; vanilla Animate Anyone 0.330. **No parameter count** in the paper. No code release stated. The "619 triplets from 16 characters" figure in wave-1 notes (from a third-party summary) was **not found** in the paper | https://arxiv.org/abs/2412.03685 , https://arxiv.org/html/2412.03685 |
| 6 | arXiv 2209.00185 SketchBetween | **VERIFIED** | "SketchBetween: Video-to-Video Synthesis for Sprite Animation via Sketches", Dagmar Lukka Loftsdóttir, Matthew Guzdial, 1 Sep 2022, ACM FDG 2022, DOI 10.1145/3555858.3555928. PDF tables: Ours SSIM 0.943 / PSNR 27.48; FOMM 0.799 / 19.46; no sketch 0.883 / 20.19; no final image 0.938 / 25.4. MGIF, 1000 videos | https://arxiv.org/abs/2209.00185 , https://arxiv.org/pdf/2209.00185 |
| 7 | arXiv 2611.20628 "PixelBench": exists? animation tasks? | **VERIFIED, and wave-1 was wrong on animation** | "PixelBench: Evaluating Language Models for Pixel-Art Creation", Mutian Hong, Jianxuan Yao, Mingkun Yue, Changkai Li, Hower Yang, Haibin Ling; v1 16 Nov 2026; 38 pp. LLM agents draw/edit pixel art via code tools; MLLM judge validated against humans. The abstract does not mention animation, but the body **does include animation**: 10 "animation" editing tasks (continue an action from the given first frame, 2-3 frames, e.g. Bat Flapping Wings, Bunny Hopping, Flag Waving) plus 4 multi-frame text-to-image tasks (e.g. Swordsman Walk Animation, 2 frames). Judge adds "inter-frame continuity and pose plausibility" for animation | https://arxiv.org/abs/2611.20628 , https://arxiv.org/html/2611.20628 |
| 8 | arXiv 2503.17076 Halton scheduler: FID 5.93 / 6.33 / 3.74 | **VERIFIED** | "Halton Scheduler For Masked Generative Image Transformer", Victor Besnier, Mickael Chen, David Hurych, Eduardo Valle, Matthieu Cord; ICLR 2025 (proceedings + poster page; arXiv page lists no venue). Table 1, ImageNet 256, 705M model, 32 steps: Confidence 5.93, Random 6.33, Halton 3.74; 12 steps: 6.80 / 7.90 / 6.19. Also a "484M" block (7.5 / 9.7 / 5.3) whose param count conflicts with the paper's own appendix | https://arxiv.org/html/2503.17076 , https://iclr.cc/virtual/2025/poster/29658 , https://proceedings.iclr.cc/paper_files/paper/2025/file/ce29be756db03b0562204265256db075-Paper-Conference.pdf |
| 9 | arXiv 2503.00307 ReMDM | **VERIFIED** | "Remasking Discrete Diffusion Models with Inference-Time Scaling", Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, Volodymyr Kuleshov; NeurIPS 2025 (arXiv comment). Table 2 ImageNet FID: T=16 MaskGIT 6.74 vs ReMDM 7.40; T=32 4.92 vs 4.92; T=64 4.85 vs 4.45 (IS 196.38 vs 209.45). Wave-1 numbers match | https://arxiv.org/abs/2503.00307 , https://arxiv.org/html/2503.00307 |
| 10 | arXiv 2209.04439 Token-Critic, FID 6.56 -> 4.69 | **VERIFIED** | "Improved Masked Image Generation with Token-Critic", José Lezama, Huiwen Chang, Lu Jiang, Irfan Essa; ECCV 2022. Table 1: ImageNet 256 FID 6.56 -> 4.69 (IS 203.6 -> 174.5, i.e. IS gets worse); 512: 8.48 -> 6.80 (IS 167.1 -> 182.1); 18 steps, "×2" forward passes | https://arxiv.org/abs/2209.04439 , https://arxiv.org/html/2209.04439 |
| 11 | arXiv 2208.04202 Analog Bits self-conditioning | **VERIFIED** (existence/venue) | "Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning", Ting Chen, Ruixiang Zhang, Geoffrey Hinton; ICLR 2023. Introduces Self-Conditioning and Asymmetric Time Intervals. Wave-1's appendix FID numbers (6.10 -> 4.24 etc.) were not re-checked in this pass | https://arxiv.org/abs/2208.04202 |
| 12 | HF thejagstudio/tessera-small: 47.6M sprite generator, 78.8% masked-pixel acc. | **VERIFIED** (model card only) | Card: "47.6M (Tessera-small)"; EMA validation accuracy "78.8% of masked pixels"; joint text+pixel bidirectional transformer (hourglass 2+8+2, width 512), frozen CLIP text encoder, no VAE; sampler uses "spatial NMS"; training uses "perceptual label smoothing"; trained on "Pixel Art Atlas 1.4M"; license CC BY-NC 4.0. It is a **text-to-sprite** generator (still images), not an animator. Self-reported, no paper | https://huggingface.co/thejagstudio/tessera-small |
| 13 | GDC 2026 State of the Industry: 52% say gen-AI hurts the industry; visual/technical artists 64% | **VERIFIED** (wording nuance) | GDC article: "over half (52%) of game industry professionals think generative AI is having a negative impact"; ~7% positive; "Workers in visual and technical art (64%), game design and narrative (63%), and game programming (59%)" most unfavorable. 64% is for the **combined** "visual and technical art" group. >2,300 respondents; published ~29-30 Jan 2026 (80.lv dated 30 Jan 2026). Note: a newer "State of the Game Industry: November 2026" edition (2,600+ respondents, dated 12 Nov 2026) exists; its gen-AI figures were not checked here | https://gdconf.com/article/gdc-2026-state-of-the-game-industry-reveals-impact-of-layoffs-generative-ai-and-more/ , https://80.lv/articles/gdc-survey-over-50-of-game-devs-say-generative-ai-harms-industry , https://www.gamedeveloper.com/business/layoffs-ai-and-next-gen-what-2-600-game-professionals-told-gdc-this-year |

Not reachable: the Business Wire press release returned HTTP 403, so the GDC figures rest on GDC's own article and press coverage.

### Corrections to earlier notes (wave-1)

1. **PixelBench does include animation** (10 animation editing tasks + 4 multi-frame generation tasks). Wave-1 track 1 §1.6 said only that the abstract does not mention it.
2. **Alonso & Patow, "280 triplets"** is the total dataset (196 train). The 10M params and 280 come from the paper bodies, not the abstracts. "Directional" in 2610.27253 means temporal direction (previous/next frame), not facing direction.
3. **Sprite Sheet Diffusion** has 40 characters, 152 sequences; the "619 triplets from 16 characters" figure is not in the paper. It has no venue (arXiv only) and reports no param count.
4. **GDC 64%** is the combined "visual and technical art" group, not visual artists alone.
5. **Halton** venue is ICLR 2025 (verified via proceedings); the arXiv page lists no venue. Its Table 1 also has a 484M block with a param count that conflicts with the appendix.
6. **Token-Critic** improves FID but lowers IS at 256 px (203.6 -> 174.5). Do not cite it as improving both.
7. **Tessera-small** is text-to-image, not animation, and its numbers are only self-reported on the model card.

---

## 2. Novelty analysis

### 2.1 What moviepix does (for reference)

For every output pixel: a categorical over {transparent, new colour, opaque reference pixel 1..K}; trained with the marginal likelihood -log Σ_{j: colour(j)=target} p(j); MaskGIT-style masking with self-conditioning; one-shot or multi-round decoding; conditioned on a discrete action + direction; input is one still RGBA sprite, output is new (unobserved) frames.

### 2.2 Closest prior work found

| Work | What it shares with moviepix | Key differences |
|---|---|---|
| **Vondrick et al., "Tracking Emerges by Colorizing Videos", ECCV 2018** (https://arxiv.org/abs/1806.09594) | **Closest in the copy-and-loss formulation.** Target colour y_j = Σ_i A_ij c_i, A_ij = softmax_i(f_iᵀ f_j) over reference-frame pixels, c_i one-hot quantized colour (16 k-means clusters), categorical cross-entropy loss. Since y_j is a categorical over colours, this CE equals -log Σ_{i: colour(i)=target} A_ij, **the same marginal-over-same-coloured-reference-pixels objective moviepix uses**. Inference temperature 0.5 | The **target frame is observed** (grayscale) and supplies the query features; the task is colour propagation / tracking, not generation. No transparent or new-colour classes, no action conditioning, no masked iterative decoding; colours are quantized clusters, not exact palette values |
| Label-propagation successors: CorrFlow / MAST (https://arxiv.org/abs/2002.07793), contrastive random walk (https://arxiv.org/abs/2006.14613) | Same attention-copy of reference labels/colours per pixel | Same difference: target frame observed; correspondence learning, not synthesis |
| **Appearance Flow** (Zhou, Tulsiani, Sun, Malik, Efros, ECCV 2016; https://arxiv.org/abs/1605.03557) | Each target pixel chooses where in the source image to copy from; novel view synthesis from one image + a transformation code (a close analogue of action + direction) | Continuous 2-D coordinates with bilinear sampling (blends colours, not a categorical); no "new"/transparent class; regression loss on pixels |
| **CDNA / pixel-motion prediction** (Finn, Goodfellow, Levine, NIPS 2016; https://arxiv.org/abs/1605.07157) | Action-conditioned frame prediction by transforming pixels of the previous frame | Local convolutional kernels / affine transforms combined with soft masks; outputs are blends; video, not sprites |
| **Pointer networks** (Vinyals et al., NIPS 2015; https://arxiv.org/abs/1506.03134) and **pointer-generator** (See, Liu, Manning, ACL 2017; https://arxiv.org/abs/1704.04368) | Output distribution = attention over input positions; See et al. mix copy with a generate distribution (our {copy j} + "new colour"). See et al. also sum attention over all source positions holding the same word, the text analogue of our same-colour marginal | Text sequences, autoregressive; no image or masked decoding |
| Copy-and-Paste Networks for video inpainting (Lee, Oh, Won, Kim, ICCV 2019; https://arxiv.org/abs/1908.11587) | Fills target pixels by attending to and copying from other frames | Soft attention over features, then a decoder (paste network) regresses pixels; inpainting with observed context |
| PixelNN (Bansal, Sheikh, Ramanan, ICLR 2018; https://arxiv.org/abs/1708.05349) | Sharp outputs by copying actual pixels (per-pixel nearest neighbour) instead of regressing | Non-parametric post-step over a training set after a blurry CNN; no learned pointer distribution |
| Grid Partitioned Attention (Jetchev et al., 2021; https://arxiv.org/abs/2107.03742) | Copying-based architecture for pose morphing; frames appearance flow as the hard limit of attention | Soft attention over features, then regression; human photos |
| Reference-based cel / line-art colourization: BasicPBC inclusion matching (https://arxiv.org/abs/2403.18342), Shi et al. few-reference line-art video colorization (IEEE TVCG; https://orca.cardiff.ac.uk/id/eprint/146848), AnimeInterp (https://arxiv.org/abs/2104.02495) | Exact colours copied from a reference by correspondence (BasicPBC copies per-segment colours, closest to "palette-exact") | Target **line art is given**; the task is filling colours into a known shape, not generating the shape/pose |
| SD-πXL (https://arxiv.org/abs/2410.06236) | Per-pixel categorical over a palette (Gumbel-softmax), so output is palette-exact | Palette is a fixed list, not pointers into a reference; per-image optimization; still images |
| Relative Pixel Prediction (Ling et al., 2019; https://astro.paperswithcode.com/paper/relative-pixel-prediction-for-autoregressive) | Predicts pixels relative to previously generated or conditioning pixels | Could not access the full text (OpenReview blocked automated access); appears to predict intensity offsets, not a categorical pointer. Treat as unverified |
| Sprite-specific work: Coutinho & Chaimowicz 2022/2024, Alonso & Patow 2026, Sprite Sheet Diffusion, SketchBetween (table above) | Same domain | All regress RGB(A) or use diffusion; none uses a copy/pointer output; all report off-palette colours or blur |

Searches that returned nothing closer: "pixel art sprite animation copy pixels from reference pointer", "video prediction per-pixel categorical over source pixels pointer", "masked generative transformer copy mechanism pointer to reference image tokens", "image generation as permutation of source pixels" (only Gumbel-Sinkhorn jigsaw / pixel-rearrangement illusion work, https://arxiv.org/abs/1802.08665 , https://arxiv.org/abs/2411.15925), "cel colorization categorical copy". Search engines cannot prove absence; this is a strong but not exhaustive check.

### 2.3 Verdict: **partially novel**

Not novel on their own:
- Per-pixel attention-copy of colours from a reference image (Vondrick 2018 and the label-propagation line).
- The training objective. Cross-entropy on attention-pooled one-hot colours (Vondrick 2018) **is** the marginal likelihood over same-coloured reference pixels. Pointer-generator also sums copy probability over repeated source tokens.
- Copy-vs-generate mixing (pointer-generator, See et al. 2017).
- Masked iterative decoding with confidence scheduling (MaskGIT) and self-conditioning (Analog Bits).
- Copy-instead-of-regress for sharpness (appearance flow, Deep Voxel Flow, PixelNN).

Not found in prior work (likely new):
1. Using the categorical pointer over reference pixels as the **generative output space** for an **unobserved** target frame. In Vondrick-style work the target image is given and defines the queries. In moviepix the target is synthesized from one still + discrete action/direction.
2. Combining that pointer output space with **MaskGIT-style masked training and iterative decoding**, so committed pixels are fed back in as conditioning.
3. The output vocabulary {transparent, new colour, pointer-to-opaque-reference-pixel} for RGBA pixel art, which makes the output **palette-exact by construction** apart from the rare "new" class.
4. Application: one-still sprite animation conditioned on action and direction, at ~4.5M params on CPU. No sprite paper found uses copying.

**Suggested contribution statement (precise and defensible):**
> "We cast single-image sprite animation as per-pixel *pointer* prediction: each output pixel is a categorical over the reference sprite's opaque pixels plus `transparent` and a rare `new colour` class, so outputs are palette-exact by construction. Following attention-based colour propagation (Vondrick et al., 2018), we train with the likelihood of the target colour marginalized over all reference pixels of that colour. Unlike that work, the target frame is not observed: we generate it from a single still and a discrete action/direction code, using MaskGIT-style masked training with self-conditioning and one-shot or iterative decoding. To our knowledge this is the first use of a reference-pixel pointer output space for generative frame synthesis, and the first copy-based model for pixel-art sprite animation."

Avoid: claiming novelty for "pointer/copy for images", for the marginal-likelihood loss itself, or for masked decoding.

---

## 3. BibTeX

Fields were checked against arXiv abs pages (titles, authors, IDs) and the venues noted in section 1. Venues marked `% venue from proceedings/common record` were not on the arXiv page itself: Appearance Flow ECCV 2016, MaskGIT CVPR 2022, See et al. ACL 2017, Pointer Networks NIPS 2015, Copy-and-Paste ICCV 2019 (CVF page), PixelNN ICLR 2018 (mlanthology). Page numbers are omitted where not checked.

```bibtex
% ---- Copy / pointer formulation ----
@inproceedings{vondrick2018tracking,
  title     = {Tracking Emerges by Colorizing Videos},
  author    = {Vondrick, Carl and Shrivastava, Abhinav and Fathi, Alireza and Guadarrama, Sergio and Murphy, Kevin},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2018},
  eprint    = {1806.09594},
  archivePrefix = {arXiv}
}

@inproceedings{zhou2016appearanceflow,
  title     = {View Synthesis by Appearance Flow},
  author    = {Zhou, Tinghui and Tulsiani, Shubham and Sun, Weilun and Malik, Jitendra and Efros, Alexei A.},
  booktitle = {European Conference on Computer Vision (ECCV)}, % venue from common record
  year      = {2016},
  eprint    = {1605.03557},
  archivePrefix = {arXiv}
}

@inproceedings{finn2016unsupervised,
  title     = {Unsupervised Learning for Physical Interaction through Video Prediction},
  author    = {Finn, Chelsea and Goodfellow, Ian and Levine, Sergey},
  booktitle = {Advances in Neural Information Processing Systems (NIPS)},
  year      = {2016},
  eprint    = {1605.07157},
  archivePrefix = {arXiv}
}

@inproceedings{vinyals2015pointer,
  title     = {Pointer Networks},
  author    = {Vinyals, Oriol and Fortunato, Meire and Jaitly, Navdeep},
  booktitle = {Advances in Neural Information Processing Systems (NIPS)}, % venue from common record
  year      = {2015},
  eprint    = {1506.03134},
  archivePrefix = {arXiv}
}

@inproceedings{see2017get,
  title     = {Get To The Point: Summarization with Pointer-Generator Networks},
  author    = {See, Abigail and Liu, Peter J. and Manning, Christopher D.},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL)}, % venue from common record
  year      = {2017},
  eprint    = {1704.04368},
  archivePrefix = {arXiv}
}

@inproceedings{lee2019copypaste,
  title     = {Copy-and-Paste Networks for Deep Video Inpainting},
  author    = {Lee, Sungho and Oh, Seoung Wug and Won, DaeYeun and Kim, Seon Joo},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {4413--4421},
  year      = {2019},
  doi       = {10.1109/ICCV.2019.00451},
  eprint    = {1908.11587},
  archivePrefix = {arXiv}
}

@inproceedings{bansal2018pixelnn,
  title     = {{PixelNN}: Example-based Image Synthesis},
  author    = {Bansal, Aayush and Sheikh, Yaser and Ramanan, Deva},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018},
  eprint    = {1708.05349},
  archivePrefix = {arXiv}
}

% ---- Masked decoding ----
@inproceedings{chang2022maskgit,
  title     = {{MaskGIT}: Masked Generative Image Transformer},
  author    = {Chang, Huiwen and Zhang, Han and Jiang, Lu and Liu, Ce and Freeman, William T.},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, % venue from common record
  year      = {2022},
  eprint    = {2202.04200},
  archivePrefix = {arXiv}
}

@inproceedings{besnier2025halton,
  title     = {Halton Scheduler for Masked Generative Image Transformer},
  author    = {Besnier, Victor and Chen, Mickael and Hurych, David and Valle, Eduardo and Cord, Matthieu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2503.17076},
  archivePrefix = {arXiv}
}

@inproceedings{lezama2022tokencritic,
  title     = {Improved Masked Image Generation with Token-Critic},
  author    = {Lezama, Jos{\'e} and Chang, Huiwen and Jiang, Lu and Essa, Irfan},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022},
  eprint    = {2209.04439},
  archivePrefix = {arXiv}
}

@inproceedings{wang2025remdm,
  title     = {Remasking Discrete Diffusion Models with Inference-Time Scaling},
  author    = {Wang, Guanghan and Schiff, Yair and Sahoo, Subham Sekhar and Kuleshov, Volodymyr},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  eprint    = {2503.00307},
  archivePrefix = {arXiv}
}

@inproceedings{chen2023analogbits,
  title     = {Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning},
  author    = {Chen, Ting and Zhang, Ruixiang and Hinton, Geoffrey},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  eprint    = {2208.04202},
  archivePrefix = {arXiv}
}

% ---- Sprite / pixel-art animation ----
@inproceedings{coutinho2022generating,
  title     = {Generating Pixel Art Character Sprites using {GANs}},
  author    = {Coutinho, Fl{\'a}vio and Chaimowicz, Luiz},
  booktitle = {Brazilian Symposium on Computer Games and Digital Entertainment (SBGames)},
  year      = {2022},
  eprint    = {2208.06413},
  archivePrefix = {arXiv},
  note      = {arXiv comment: submitted to SBGames 2022}
}

@inproceedings{coutinho2024mdigan,
  title     = {A Missing Data Imputation {GAN} for Character Sprite Generation},
  author    = {Coutinho, Fl{\'a}vio and Chaimowicz, Luiz},
  booktitle = {Brazilian Symposium on Computer Games and Digital Entertainment (SBGames)},
  year      = {2024},
  eprint    = {2409.10721},
  archivePrefix = {arXiv}
}

@misc{alonso2026lowcost,
  title     = {Low-Cost Deep Learning for Automated Keyframe Interpolation in Sprite Animation},
  author    = {Alonso, Kenneth and Patow, Gustavo},
  year      = {2026},
  eprint    = {2610.16423},
  archivePrefix = {arXiv},
  primaryClass = {cs.GR}
}

@misc{alonso2026directional,
  title     = {Neural Controlled Directional Frame Generation for Sprite Animation},
  author    = {Alonso, Kenneth and Patow, Gustavo},
  year      = {2026},
  eprint    = {2610.27253},
  archivePrefix = {arXiv},
  primaryClass = {cs.GR}
}

@misc{hsieh2024spritesheet,
  title     = {Sprite Sheet Diffusion: Generate Game Character for Animation},
  author    = {Hsieh, Cheng-An and Zhang, Jing and Yan, Ava},
  year      = {2024},
  eprint    = {2412.03685},
  archivePrefix = {arXiv},
  primaryClass = {cs.GR}
}

@inproceedings{loftsdottir2022sketchbetween,
  title     = {{SketchBetween}: Video-to-Video Synthesis for Sprite Animation via Sketches},
  author    = {Loftsd{\'o}ttir, Dagmar Lukka and Guzdial, Matthew},
  booktitle = {Proceedings of the 17th International Conference on the Foundations of Digital Games (FDG)},
  year      = {2022},
  doi       = {10.1145/3555858.3555928},
  eprint    = {2209.00185},
  archivePrefix = {arXiv}
}

@misc{hong2026pixelbench,
  title     = {{PixelBench}: Evaluating Language Models for Pixel-Art Creation},
  author    = {Hong, Mutian and Yao, Jianxuan and Yue, Mingkun and Li, Changkai and Yang, Hower and Ling, Haibin},
  year      = {2026},
  eprint    = {2611.20628},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```

Note on `loftsdottir2022sketchbetween`: "17th" is FDG 2022's edition number from the common record; the arXiv page says only "ACM Conference on the Foundations of Digital Games". Drop it if unsure.
