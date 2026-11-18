# Wave 1 / Track 5 — What users need, how they feel about AI sprite tools

Research date: 2026-11-18. Author of the project is referred to as Hoopakid.

## Method and limits (read first)

- Sources were web search plus page fetches. **Reddit (reddit.com, old.reddit.com) could not be fetched** from this environment, and Reddit mirrors were blocked. Reddit content below comes only from search-engine snippets or third-party reporting, and is marked **[snippet]** or **[secondhand]**. Treat it as unverified until someone reads the threads directly.
- OpenGameArt.org returned HTTP 403 to the fetcher. OGA policy wording is from search-result snippets and is marked **[snippet]**.
- Vendor comparison pages (Ludo, Sorceress, Sprite AI, SpriteCook, Cinevva) were used for competitor prices but they are biased. Prices change often. Every price here is a snapshot and should be re-checked before it is quoted publicly.
- Quotes are reproduced as returned by the page fetcher. Usernames are the ones shown on the page. Nothing was posted or uploaded.

---

## 1. What pixel artists and indie devs say

### 1.1 Recurring complaints about AI sprite / animation tools

| Theme | Evidence | Source |
|---|---|---|
| Frame-to-frame inconsistency | "The background jitters, details go missing from frame to frame" (thrance, HN, Jun 2025) | https://news.ycombinator.com/item?id=44204181 |
| Low quality, recognisably AI | "you can tell the sprites are low quality and AI generated." (9283409232, HN) | https://news.ycombinator.com/item?id=44204181 |
| Tool ruins the user's own art | "I uploaded a photo of a pixel art electrician and it just completely ruins it" (Madmallard, HN) | https://hn.nuxt.dev/item/44204181 (mirror of the same HN item) |
| Glitchiness | "The "glitchiness" of the AI sprites can be quite unsettling" (bandoti, HN) | https://hn.nuxt.dev/item/44204181 |
| Pixel-size inconsistency across frames | An itch.io user said sprite sheets are "not consistent pixel sizes between frames making it worthless for animations" **[snippet]** | https://crown21.itch.io/warrior-sprite-64-x-64/comments |
| "Mixels", soft edges, palette bloat | Blog/vendor analyses describe mixed pixel sizes inside one sprite, anti-aliased outlines, and too many near-identical colours as the tell-tale signs of AI pixel art (vendor-written, but describes a widespread critique) | https://www.spritecook.ai/blog/why-ai-cant-match-pixel-art-resolution , https://rangy.ai/blog/ai-pixel-art , https://dev.to/handsupmin/why-ai-pixel-art-looks-fake-and-how-i-turned-it-into-real-pixel-art-5b1b |
| Grid / palette / noise problems | "which can have issues with grid alignment, palette choices, and high-frequency noise." (vunderba, HN) | https://news.ycombinator.com/item?id=50172003 |
| Changing the look kills the time saving | A jam team using Stable Diffusion for animations said changing the look means it "falls back on your art skills again and it doesn't save much time" **[snippet]** | https://bromberry.itch.io/dueling-dynasties/devlog/522129/we-made-animations-with-ai-stable-diffusion |
| Prompt adherence | Retro Diffusion buyer: "they all do weird stuff and absolutely ignore half of what I mention." (loneroamer) | https://astropulse.itch.io/retrodiffusion/comments |
| Local hardware pain | "Radeon graphics cards aren't supported on Windows" (IsaacTakuma); "Any update on eta for AMD support on windows?" (yuttgyh) | https://astropulse.itch.io/retrodiffusion/comments |
| Install failures | "I'm receiving the following error when I try to generate anything" (gillianyue, Mac) | https://astropulse.itch.io/retrodiffusion/comments |
| Abandonment worry | "judging by the development log the last update to this product was 14 months ago." (deganderson) | https://astropulse.itch.io/retrodiffusion/comments |
| Research view of the gap | Sprite Sheet Diffusion paper: single-frame conditional models "lack mechanisms to ensure consistency across sequential outputs" | https://arxiv.org/html/2412.03685v2 |

### 1.2 What people actually want automated

| Want | Evidence | Source |
|---|---|---|
| Keep the keyframes human, automate the in-betweens | "I think most animators would rather just do the keyframes and leave the tweens in the hands of the machine." (bitwize, HN) | https://news.ycombinator.com/item?id=44204181 |
| Directional views | "As a start, there should be projections for all four cardinal directions." (palmfacehn, HN) | https://news.ycombinator.com/item?id=44204181 |
| Top-down / isometric | "Can it do isometric and top down views?" (imatworkyo, HN) | https://news.ycombinator.com/item?id=44204181 |
| Aseprite integration | "Does this have plans or support for Aseprite?" (alstonite, HN) | https://news.ycombinator.com/item?id=44204181 |
| Pipeline automation | "honestly it's very tedious and I'm looking to automate as much of the pipeline as I can" (archerx, HN) | https://news.ycombinator.com/item?id=44204181 |
| Even imperfect output saves time | "Still, a huge time boost." (nurettin, HN) | https://news.ycombinator.com/item?id=44204181 |
| Animate *my* sprite while keeping its style | "Can this also be used to animate sprites? And mantain their quality and art style" (vinicat, Retro Diffusion page) | https://astropulse.itch.io/retrodiffusion/comments |
| Frame suggestions from the artist's own drawing | Isitpossible (2021): "if I draw a stick figure, the AI would be able to suggest next frames for it (based on the original drawing)." | https://community.aseprite.org/t/ai-aided-drawing-is-it-possible/9161 |
| Generate remaining frames from one image | J11 (2023): "Hi, is it possible to ai generate frames to make it easy?" | https://community.aseprite.org/t/features-ai-generator/18950 |
| Deterministic, not random | "turn something nondeterministic into something more deterministic" (satvikpendem, HN) | https://news.ycombinator.com/item?id=50172003 |
| Budget / time constraint | itch.io dev with no budget to hire an artist; another says sprite creation "takes forever with full time jobs and families" **[snippet]** | https://itch.io/post/11923759 , https://itch.io/post/12566545 |

Takeaway: the demand is for *consistency with the user's own art* and for removing tedium (in-betweens, directions, sheet assembly), not for "generate a character from a prompt".

### 1.3 What people refuse

- **Training on art without consent.** "Unless you created all the training material yourself, any model like this is highly unethical, in my opinion." (nkrisc, HN) — https://hn.nuxt.dev/item/44204181
- **Replacing artists.** "Generating art solves no real problem and only suppresses a struggling profession" (okkdev, HN) — https://hn.nuxt.dev/item/44204181
- **Loss of human authorship.** Godot forum, Mar 2026: "Art is, by definition, human expression." and "its not art because no one is responsible for it." (gobbel); "LLMs are trained on stolen copyrighted content." and "There's a homongenous effect that takes over much AI art." (dragonforge-dev) — https://forum.godotengine.org/t/off-topic-why-is-a-i-generated-content-like-music-images-looked-down-upon-in-video-games/135788
- **But nuance exists in the same thread:** "using ai to get over a tiny bump in the road isnt necessarily a big deal." (gobbel); "I personally don't know that it means art cannot be made with LLMs as tools." (dragonforge-dev) — same URL.
- **Pixel artist on AI generators:** Pixel Parmesan (pixel artist, Apr 2024) accepts AI for reference/mockups but holds that "human decision-making and self-expression are still very necessary in the creation of high quality pixel art" and does not use pixel-generation tools — https://pixelparmesan.com/ai-assisted-reference-and-photobashing-for-pixel-artists/
- **"Probably gets hatred because stolen art."** (Plop, Aseprite forum, 2024) — https://community.aseprite.org/t/features-ai-generator/18950
- **Concerns about energy and jobs** (dragonforge-dev, Godot forum, above).

### 1.4 Industry-wide sentiment numbers

| Survey | Finding | Source |
|---|---|---|
| GDC State of the Game Industry 2026 (Mar 2026, >2,300 respondents) | 52% say generative AI is having a negative impact (up from 30% and 18% in the two prior years); 7% positive; 36% use genAI at work | https://www.implicator.ai/game-developers-anti-ai-sentiment-hits-record-high-in-gdc-2026-survey/ , https://hitmarker.net/news/over-50-of-game-developers-say-generative-ai-is-harming-the-industry-report-reveals-1635057 |
| Same survey, by role | Visual and technical artists 64% unfavourable — the most negative group | https://www.implicator.ai/game-developers-anti-ai-sentiment-hits-record-high-in-gdc-2026-survey/ |
| GDC follow-up reported Nov 13, 2026 | 55% negative, 40% use genAI (reported by 80.lv; I did not see the primary GDC document) **[secondhand]** | https://80.lv/articles/gdc-2026-survey-55-of-developers-consider-ai-s-impact-on-games-negative/ |

Implication: moviepix's core audience (pixel artists) is the most hostile segment. Indie programmers without art budgets are the most receptive. Messaging must work for both.

---

## 2. Ethics, community rules, disclosure

### 2.1 Rules and policies

| Venue | Rule | Source / status |
|---|---|---|
| r/PixelArt (~2.7M members) | Listed as "No AI Art" by a third-party rules aggregator (read 2026-09-26) | https://threadfox.vip/rules/ai-content **[secondhand — could not open Reddit to confirm wording]** |
| r/Art | "No 'AI' art, ever" — and banned a human artist whose work "looked AI" | https://www.pcgamer.com/artist-banned-from-art-subreddit-because-their-work-looked-ai-generated , https://hyperallergic.com/791951/artist-banned-from-subreddit-over-work-resembling-ai-art/ |
| r/gameassets (run by Kenney) | Feb 2024: no longer accepts submissions made with generative AI; Kenney's objection is to datasets built without consent | Search snippets of https://lr.eu.psf.lt/r/gameassets/comments/1ase12i/ai_generated_game_assets_will_no_longer_be/ **[snippet]**; background: https://gamesradar.com/game-dev-legend-hailed-as-asset-jesus-says-they-were-banned-from-development-events-a-day-after-speaking-out-against-ai-sponsors |
| itch.io (since Nov 20, 2024) | Asset creators **must** fill a "Generative AI disclosure" field (Graphics / Sound / Text & Dialog / Code). "Assets comprised of generative AI (even if modified afterwards) that are not tagged will no longer be eligible for indexing." leafo: "If the output of Gen AI is something you put into your project, then tag it." Projects answering No get a "No AI" tag. | https://itch.io/t/4309690/generative-ai-disclosure-tagging , https://www.creativebloq.com/3d/video-game-design/itch-ios-no-ai-tag-is-a-positive-step-in-the-battle-against-ai-slop |
| Steam (rewritten mid-Jan 2026) | Disclosure focuses on AI-generated content players see/hear (art, sound, text, marketing) and on AI generated live during gameplay; background efficiency tools (e.g. code assistants) are not the focus | https://www.notebookcheck.net/Steam-updates-AI-disclosure-form-requiring-developers-to-report-visible-and-in-game-AI-but-not-background-tools.1206103.0.html , https://www.pcgamer.com/software/ai/steam-updates-ai-disclosure-form-to-specify-that-its-focused-on-ai-generated-content-that-is-consumed-by-players-not-efficiency-tools-used-behind-the-scenes/ (exact form text not retrieved) |
| OpenGameArt.org | AI-assisted art is not banned but must name the AI tech, version and ToU link; AI-assisted assets are marked as a potential licensing issue / download-disabled for now. **Exemption:** art generated with technology whose "training dataset, in its entirety, is demonstrably openly licensed". | https://opengameart.org/content/artificial-intelligence-assisted-artwork , https://opengameart.org/content/art-submission-guidelines **[snippet — OGA returned 403]** |
| Game jams | Vary widely: some ban AI art/music but allow code autocomplete; One Page RPG Jam bans AI text too; OpenGameArt Spring Jam 2025 allowed genAI with disclosure + credits (model/website/license) | https://itch.io/jam/one-page-rpg-jam-2025/topic/5041892/generative-ai-policy , https://itch.io/jam/opengameartorg-spring-game-jam-2025/topic/4762321/a-clarification-on-the-use-of-ai-generated-art . GMTK / Ludum Dare rules **not verified**. |
| Aseprite (the editor) | Lead developer dacap (May 2024): "No generative AI feature is planned." Forum user claim (unofficial): "The team developing Aseprite are strictly anti-AI" | https://community.aseprite.org/t/features-ai-generator/18950 , https://community.aseprite.org/t/aseprite-ia-integration/25587 |
| Lospec | No published AI policy found. Dropped its /r/Lospec feed citing "forced AI training" among reasons (Oct 2024). | https://lospec.com/ |
| Pixel Joint | No AI policy found in search. **Unverified.** | — |

### 2.2 The LPC license question (important for moviepix specifically)

- LPC base art is dual-licensed **CC-BY-SA 3.0 and GPL 3.0**; the Universal LPC generator also includes CC0, CC-BY and OGA-BY art, and requires crediting all non-CC0 authors when you use generated sprites. https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator
- Creative Commons' own guidance: "If AI models or outputs are based on ShareAlike content and they will be shared publicly, following the ShareAlike condition would require AI developers to use the same CC license as the original works." For attribution, "a simple link to the source of the dataset" may be enough for training. https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/
- CC also says this only matters where copyright permission is needed at all (exceptions such as fair use / TDM may apply), and that its guidance is conservative and not legal advice (same URL). IViR's report argues the SA question for ML is genuinely unsettled: https://www.ivir.nl/publicaties/download/Share-Alike-and-ML.pdf
- OGA forum posters (snippets) note that if models are found to be derivative works, attribution licenses other than CC0 still bite. **[snippet]** https://lpc.opengameart.org/forumtopic/ai-use

Consequence: "trained only on openly licensed art" is a real differentiator (it qualifies for OGA's exemption as described in snippets), **but** it does not make outputs license-free. moviepix should (a) publish full credits, (b) state plainly what licenses the weights carry, and (c) tell users what (if anything) they owe for output frames. This needs a legal read, not a guess — see follow-ups.

### 2.3 What would make a tool acceptable (synthesised from the evidence)

1. **Consent / open data with credits.** The main objection (Kenney, nkrisc, Plop, dragonforge-dev) is the dataset, not the tech. Retro Diffusion markets itself as "trained on licensed assets from Astropulse and other pixel artists... with their consent" (https://astropulse.itch.io/retrodiffusion). moviepix can go further: a public, per-artist credits list and dataset manifest.
2. **Works on the artist's own pixels.** Complaints centre on tools that "ruin" the input. A tool that only rearranges the user's pixels/palette maps onto the "assistive" side of the line that commenters draw (keyframes by hand, machine does tweens).
3. **Local, no upload.** Lospec's statement shows sensitivity to platforms using art for training. Every listed competitor except Retro Diffusion's extension is cloud-only, and PixelLab's ToS restricts training on outputs (https://www.pixellab.ai/termsofservice).
4. **Honest disclosure.** itch.io's rule covers generated output "even if modified afterwards". Frames drawn by moviepix will almost certainly count as "AI Generated Graphics" on itch.io and as player-facing content on Steam. Claiming otherwise would backfire. GhostFrames, an Aseprite RIFE interpolation extension, leads with "Not generative AI." (https://community.aseprite.org/t/extension-ghostframes-estimates-in-between-frames-from-two-images-it-is-not-an-image-generator/28559). moviepix should **not** copy that framing, because it does synthesise new frames.
5. **Human stays in charge.** Edit/accept/reject every frame. Output is a starting point inside Aseprite, not a finished asset.

### 2.4 Recommended labelling for moviepix

- One-line: "A small, open, offline model that animates *your* pixel-art sprite using only its own pixels — trained solely on credited, openly licensed LPC art."
- Say plainly: "moviepix is a machine-learning (generative) tool. If you ship its frames, platforms like itch.io and Steam expect you to disclose AI-generated graphics." Link to the policies.
- Avoid "not AI", "ethical AI" as slogans (unprovable, and they attract scrutiny). Prefer verifiable facts: dataset manifest, licenses, artist list, no network calls, model size, reproducible training.
- Provide a ready-made disclosure snippet and a CREDITS file users can paste.
- Do not advertise on r/PixelArt or the Aseprite forum as AI art; communities there are hostile and Aseprite devs have said no genAI is planned. Lead with r/gamedev / r/godot / itch.io devlogs aimed at programmers, and with transparent technical writing (HN-style).

---

## 3. Competitor analysis

Prices are snapshots from the cited pages; re-check before quoting.

| Tool | Model / where it runs | Price (snapshot) | Animation & directions | Notes / gaps | Sources |
|---|---|---|---|---|---|
| **PixelLab** | Cloud (REST API v2, hosted MCP, Python/JS SDKs). No local option documented. | Ludo's page: $12/mo for 2,000 images ($8.25 yearly) up to $50 "Architect" (10,000); free trial 40 fast gens then 5/day. Third-party prices disagree. | Text→animation, skeleton animation, animate-automatic (≤64×64, Tier 1+), 4/8-direction rotation, 8-rotation endpoint; Aseprite extension + Pixelorama-based web editor; Aseprite export | No training-data disclosure found. ToS forbids training other models on outputs without permission. Metered generations. | https://www.pixellab.ai/llms.txt , https://www.pixellab.ai/docs/tools/animate-automatic , https://pixellab.ai/docs/tools/rotate , https://ludo.ai/compare/ludo-vs-pixellab , https://www.pixellab.ai/termsofservice |
| **Retro Diffusion** | Local Aseprite extension (Python + ~6 GB venv, GPU strongly preferred) **and** a separate cloud website/API with different models | Extension $65 one-time ("No subscriptions and no credits"), Lite ~$20; web credits from $5 | Local extension: no animation (per dev comments); cloud rd-animation model has styles incl. `four_angle_walking` (4 dir × 4 frames, 48 px only) | Ethics claim: trained on licensed assets with consent. Hardware/AMD issues, long gaps between updates. 64×64 gen: ~5 s on RTX 3060, ~2.5 min on GTX 1050 Ti, 4–10 min CPU. | https://astropulse.itch.io/retrodiffusion , https://astropulse.gitbook.io/retro-diffusion/aseprite-extension/retro-diffusion-for-aseprite/installation-and-setup , https://help.scenario.com/en/articles/retro-diffusion-models-the-essentials/ , https://replicate.com/retro-diffusion/rd-animation |
| **Ludo.ai** | Cloud (app, API, MCP, CLI) | Indie $20/mo ($15 yearly), Pro $50/mo; 30 free credits; unpaid accounts get watermarked sheets | Sprite animation from prompt/keyframes/reference video, 641 motion presets, up to 8 directions, "Change View" | General game-AI suite; vendor-written comparisons | https://ludo.ai/compare/ludo-vs-pixellab , https://ludo.ai/compare/best-ai-sprite-generators , https://app.cinevva.com/guides/ai-pixel-art-generators |
| **Sorceress** | Browser; mixes own tools with Retro Diffusion model | $49 one-time "lifetime" for non-AI tools; credits (~1 credit = 1¢); Quick Sprites 9 credits/sheet | Quick Sprites (4-dir walk via rd-animation), Auto-Sprite v2 | Self-reported pricing, inconsistent pages | https://sorceress.games/blog/cast-an-ai-game-sprite-generator-walk-cycle-2026 , https://sorceress.games/pages/true-pixel |
| **Scenario** | Cloud | Starter $15/mo, Pro $45/mo (custom training) | Hosts Retro Diffusion animation model | General asset platform | https://app.cinevva.com/guides/ai-pixel-art-generators , https://www.scenario.com/models/retro-diffusion-animation |
| **Layer.ai** | Cloud | Usage-based ("Creative Units"); Ludo lists from $10/mo | Sprite-sheet tools; no animation presets/directions found | Studio-oriented | https://app.cinevva.com/guides/ai-pixel-art-generators , https://ludo.ai/compare/best-ai-sprite-generators |
| **God Mode AI** | Cloud | Inconsistent: mirror of plans shows $12/20 credits, $100/250 credits; others say $29/mo | One character image → sprite animations, Spine animations, 3D; 360° rotation | Show HN thread was heavily criticised for quality and ethics | https://backiee.wasmer.app/https_godmodeai_co/plans , https://news.ycombinator.com/item?id=44204181 |
| **Sprite AI** (sprite-ai.art) | Cloud; free Aseprite extension that calls its API via curl | Token-based; Animate 14–20 tokens; Rotate 12 | Animate: 4–16 frames under one tag, "can't turn the character"; separate Rotate | Shows the Aseprite-extension-calls-external-service pattern | https://www.sprite-ai.art/docs/aseprite |
| **SpriteLab, SpriteCook, AutoSprite, pixie.haus** | Cloud | ~$8–12/mo | Various; SpriteLab claims 4/8-direction rotation | Long tail of similar SaaS | https://ludo.ai/compare/best-ai-sprite-generators |

Non-AI tools (the "no-AI" alternative many artists prefer):

| Tool | What it does | Price | Source |
|---|---|---|---|
| **Aseprite** | Frame-by-frame editor, onion skin (default 2 frames each way), tags, per-frame ms timing; Lua scripting/extensions | (paid editor; price not re-checked) | https://www.aseprite.org/docs/animation/ , https://www.aseprite.org/api/ |
| **Tweencel, Easy Interpolations, CelTween** | Aseprite extensions that generate in-betweens by interpolating transforms | free/cheap | https://community.aseprite.org/t/extension-tweencel-tweening-and-animation-control-for-aseprite/24229 , https://community.aseprite.org/t/extension-free-easy-interpolations-for-aseprite/28716 , https://violetpixel13.itch.io/celtween |
| **GhostFrames** | Aseprite extension using RIFE optical-flow interpolation between keyframes (local; Windows build bundles deps, Linux needs python3) | — | https://community.aseprite.org/t/extension-ghostframes-estimates-in-between-frames-from-two-images-it-is-not-an-image-generator/28559 |
| **PixelOver** | Bones, IK, mesh deformation on pixel art with pixel-accurate transforms, dithering/indexation; exports sheets | $10.49, Early Access, 85% of 187 reviews positive | https://store.steampowered.com/app/1762550 |
| **Smack Studio** | Fighting-game sandbox that maps 2D pixel art onto a 3D bone rig and can export sprites | $8.49, 85% of 528 positive | https://store.steampowered.com/app/1739300/Smack_Studio/ |
| **Pixel Composer** | Node-based pixel VFX/animation | ~$10 | https://makham.itch.io/pixel-composer/purchase , https://www.steamspy.com/app/2299510 |
| **Spine** | Skeletal 2D animation | Essential $69, Professional $369 (sale prices shown on official page; list $99 / $439) | https://esotericsoftware.com/spine-purchase |
| **framelet** | Free browser tool, procedural effects (tween, colour cycling) on existing pixel art | free | https://itch.io/blog/965802/framelet-procedural-2d-sprite-animation-tool |
| DragonBones | Not researched (status unverified) | — | — |

### Gaps moviepix can fill

1. **No offline, CPU-only, fast option for animation.** Every AI animation competitor above is cloud. Retro Diffusion's local extension needs Python, a 6 GB env and a GPU, and doesn't animate.
2. **Exact palette/pixel preservation.** Users' top complaints (mixels, palette bloat, "ruins it", details vanish between frames) are what a pixel-copying model avoids by design.
3. **Transparent, credited training data.** No competitor publishes a dataset manifest; PixelLab says nothing on data. Only Retro Diffusion makes a consent claim.
4. **Open weights + no metering + no output restrictions** (vs PixelLab ToS, credits everywhere, watermarks on Ludo's free tier).
5. **Fits the "keyframes human, tweens machine" ideal** better than prompt-to-sprite tools; complements non-AI tools (PixelOver, Tweencel) rather than replacing artists.

Gaps where competitors are ahead (be honest): arbitrary prompt-defined actions, sizes beyond 64×64, non-humanoid bodies, isometric views, tilesets/VFX, polished web UIs, MCP/agent integrations.

---

## 4. Workflows and formats to support

### 4.1 Aseprite

- Lua scripting since v1.2.10; extensions add menu commands via the `Plugin` API and dialogs via `Dialog`. https://www.aseprite.org/api/
- **Calling a local program:** the API docs say `os.execute` and `io.open` "will ask for permissions to the user" (a security prompt). https://www.aseprite.org/api/ Forum reports say `io.popen` read-mode / `os.execute` can be unreliable across platforms; one user recommends WebSockets for talking to other languages. https://community.aseprite.org/t/os-execute-function-not-working/21132 , https://community.aseprite.org/t/calling-a-c-function-from-lua-inside-aseprite/17862 **[forum anecdotes]**
- **WebSocket class** exists (connect to e.g. `127.0.0.1`, `sendBinary`), which allows a local moviepix server. https://github.com/aseprite/api/blob/main/api/websocket.md
- Precedents: Retro Diffusion's extension drives a local Python environment; GhostFrames ships a local RIFE binary on Windows; Sprite AI's extension shells out to `curl`. So "Aseprite extension → local executable" is proven in practice.
- Recommended design: write output as new frames under a tag named after the action (Sprite AI does the same — "4 to 16 timed frames under one tag"), preserving the active palette and using per-frame durations.
- CLI export used by pipelines: `aseprite -b sprite.ase --format json-array --data out.json --sheet out.png --list-tags`, `--split-tags` with `{tag}` filenames. https://community.aseprite.org/t/how-to-process-the-generated-json-sprite-sheet/34 , https://mintlify.com/aseprite/aseprite/features/sprite-sheets (third-party mirror; verify against https://www.aseprite.org/docs/cli/)

### 4.2 Engines

| Engine | Expected input | Source |
|---|---|---|
| Godot 4 | Aseprite Wizard imports .aseprite → SpriteFrames / AnimatedSprite2D / AnimationPlayer; one animation per tag; ms durations → FPS. Alternative: sheet + JSON with tags. | https://github.com/viniciusgerevini/godot-aseprite-wizard , https://github.com/lavabird-games/godot-animated-aseprite |
| Unity | Official Aseprite Importer (com.unity.2d.aseprite): one Animation Clip per tag, loop from tag repeat (needs Aseprite 1.3), only Forward direction supported | https://docs.unity3d.com/Packages/com.unity.2d.aseprite@1.1/manual/AsepriteFeatures.html |
| GameMaker | Strip PNG named `name_stripN.png` auto-splits into N frames; else Import Strip Image | https://gamemaker.io/ru/blog/gms2-features-you-might-not-know-about , https://manual.yoyogames.com/The_Asset_Editors/Sprites.htm |
| Phaser / web | Aseprite JSON via `createFromAseprite` | https://newdocs.phaser.io/docs/3.55.1/focus/Phaser.Animations.AnimationManager-createFromAseprite |
| Generic | Aseprite JSON (hash/array), TexturePacker JSON | https://www.codeandweb.com/texturepacker/tutorials/how-to-use-sprite-sheets-with-gamemaker |

Minimum export set: `.aseprite` (tags), PNG sheet + Aseprite-style JSON (array, with `frameTags`), GIF preview, `_stripN` PNGs, individual frame PNGs. Godot `.tres` SpriteFrames is a nice-to-have (Aseprite Wizard already covers it).

### 4.3 Sizes and frame counts

- LPC: 64×64 base grid; original animations spellcast, thrust, walk, slash, shoot, hurt; LPC Expanded adds bow, climb, run, jump; oversize 128/192 px weapon animations exist. https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator
- Exact LPC frame counts per animation were **not verified** in this pass (commonly cited as walk 9, slash 6, spellcast 7, thrust 8, shoot 13, hurt 6 — confirm from the sheets).
- Rules of thumb (vendor guides, treat as soft): idle 2–4 frames, walk 4–8 (4 for small sprites, 6–8 at 32×32+), attack 3–6; 16×16 classic, 32×32 common, 64×64 large characters. https://www.sprite-ai.art/blog/sprite-animation-frames
- Aseprite docs: ~12 FPS ("on twos") common, 10 FPS retro. https://www.aseprite.org/docs/animation/

---

## 5. Feature wishlist ranking

Ranking basis: frequency/intensity of the requests above, competitor gaps, and fit with moviepix's strengths. It is a judgement, not survey data.

| Rank | Feature | Why | Evidence |
|---|---|---|---|
| 1 | **Pixel/palette lock (guaranteed)** — output uses only the input's colours, no new pixels outside palette, consistent pixel grid | Directly answers the top complaints; core differentiator | §1.1 rows on mixels, palette, "ruins it" |
| 2 | **Consistent 4-direction set (8 later) from one view** | Most-requested by devs; every paid competitor sells it | palmfacehn HN; PixelLab/Ludo/SpriteLab features |
| 3 | **Aseprite extension (local, no upload)** writing frames into a tag | Where pixel artists work; asked for explicitly | alstonite HN; Retro Diffusion/Sprite AI precedents |
| 4 | **Edit-and-regenerate loop**: lock/accept frames, fix one frame, regenerate the rest; keyframe-guided in-betweens | "keyframes human, tweens machine"; artist-in-the-loop is the ethical line | bitwize HN; Isitpossible Aseprite forum |
| 5 | **Engine-ready export**: Aseprite JSON + PNG sheet, `_stripN`, GIF, Godot/Unity via Aseprite importers | Kills tedium; easy win | §4.2 |
| 6 | **Onion-skin / side-by-side preview with FPS control** (browser demo + CLI `--preview`) | Lets users judge motion before committing | Aseprite onion-skin workflow |
| 7 | **Batch mode** (folder of sprites × actions × directions) in CLI | Indie programmers automating pipelines | archerx HN |
| 8 | **Credits/disclosure generator**: emits CREDITS.txt for LPC artists and an itch.io/Steam disclosure sentence | Makes compliance painless; signals good faith | §2 |
| 9 | Godot plugin | Large indie audience; but Aseprite Wizard already bridges | §4.2 |
| 10 | Sizes beyond 64×64, isometric, non-humanoids | Requested but outside training data; later | imatworkyo HN |

---

## 6. Follow-up questions

1. **Licensing of outputs and weights:** given CC-BY-SA/GPL training data and CC's ShareAlike guidance, what license must the weights carry, and what do users owe for output frames? Needs a qualified legal opinion. Apache-2.0 for code is fine; weights may need CC-BY-SA.
2. Confirm r/PixelArt, r/gamedev, r/godot, r/IndieDev rules by reading Reddit directly (blocked here).
3. Confirm OGA's openly-licensed-dataset exemption wording (403 here) and whether moviepix output would qualify.
4. Does moviepix output trigger itch.io "AI Generated Graphics"? Based on leafo's wording, almost certainly yes — decide whether to say so up front (recommended).
5. Verify LPC per-animation frame counts and direction row order from the actual sheets.
6. Run a small user test (5–10 indie devs, 3–5 pixel artists) on the browser demo: what failure modes do they notice first?
7. Check DragonBones status, Aseprite's current price, and GMTK / Ludum Dare AI rules.
8. Test the Aseprite permission prompt behaviour for `os.execute`/`io.popen` on macOS/Windows/Linux vs a WebSocket bridge.
