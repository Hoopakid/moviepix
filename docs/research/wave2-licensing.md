# Wave 2: Licensing for training data, weights, dataset and outputs

Status: research notes, 2026-11-18. Author of the project: Hoopakid.

> **Not legal advice.** These are research notes by a non-lawyer. They summarise
> primary sources (licence texts, Creative Commons and OpenGameArt FAQs, platform
> rules) and published commentary. Copyright law on AI training differs by country
> and is still being decided in court. Before a large or commercial release, have
> a lawyer review the points under "Open questions for a lawyer".

Conventions:
- **VERIFIED**: read on the primary page during this research. The quote is copied from that page or file.
- **SECONDARY**: from press coverage, law-firm summaries or forum posts. Treat with care.
- **INFERENCE**: my own reasoning from the sources. It needs a lawyer's check before anyone relies on it.
- Risk levels: **LOW** (well supported, standard practice), **MEDIUM** (reasonable, but there is a known gap or the law is unsettled), **HIGH** (likely problem; fix before release).
- OpenGameArt (OGA) pages block the WebFetch tool (HTTP 403). They were read with a plain HTTP GET, and the text was pulled from the HTML.

---

## 0. Summary

| # | Question | Short answer | Risk |
|---|---|---|---|
| 1 | Choosing the permissive option of a multi-licensed work | Valid. OGA's FAQ says "You must follow only one of the licenses." The pitfall is upstream: the generator's licence list can claim an option that an upstream page does not show. This happened for the **body and head bases** (see 1.4). 90 of 370 credit rows rely only on pack pages whose licence field is SA/GPL, with permissive options stated per file inside the pack (section 9) | LOW for the principle, **MEDIUM** for data accuracy |
| 2 | Attribution and model weights | CC's conservative guidance says credit can be "a simple link to the source of the dataset". Whether weights are a copy or adaptation is unsettled: the UK *Getty v Stability* ruling says no, the German *GEMA v OpenAI* ruling says yes where a work is memorised. A tiny model trained on a few hundred layers may memorise, so ship credits with the weights | MEDIUM (low if credits ship with weights) |
| 3 | Publishing the composed dataset | Per-file licences stay attached. Hoopakid can license the compilation and metadata separately. Give TASL credits plus a "modified" notice and licence links | LOW if done as in section 3 |
| 4 | Licence of generated frames | No source supports the training data's licence carrying over to frames made from a user's own sprite. Risk rises only if outputs reproduce LPC art | LOW |
| 5 | CC-BY-SA art for evaluation only | Using it is fine. Redistributing it inside a benchmark is fine if those files stay CC-BY-SA with credits. The benchmark as a whole need not be SA | LOW |
| 6 | GPL and reading generator data files | Reading PNG/JSON data does not trigger the GPL on moviepix's code. The art carries its own licences. Do not copy the generator's code or its JSON schema wholesale without a check | LOW (JSON copying: LOW to MEDIUM) |
| 7 | AI disclosure on itch.io and Steam | Both platforms ask for disclosure of player-facing content made with generative AI. Frames made with moviepix are very likely in scope | (policy, not legal) |
| 8 | Release checklist | See section 8 | - |

---

## 1. Choosing the permissive option of a multi-licensed work

### 1.1 Sources

- **OpenGameArt FAQ** (VERIFIED, https://opengameart.org/content/faq), question "Some artists have multiple licenses listed. Does that mean you need apply the rules of all licenses or can we pick the one license we prefer?":
  > "You must follow only one of the licenses. However, when you re-distribute/edit, you are encouraged to include/use all of the licenses, so the license spectrum (and thus sum of people/projects who can use the art) doesn't shrink."
- **Same FAQ**, on artists changing licences:
  > "We encourage you to respect the wishes of the artist if they decide to change the license on their work, however you are under absolutely no obligation to do so, because the license you obtained it under is irrevocable."
- **Creative Commons FAQ** (VERIFIED, https://creativecommons.org/faq/), "Can I enter into separate or supplemental agreements with users of my work?":
  > "You may offer the licensed material under other licenses in addition to the CC license" ... "A reuser may then choose which set of terms to comply with."
- **Universal LPC Spritesheet Character Generator README** (VERIFIED, https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator):
  > "Each piece of artwork distributed from this project" is licensed under "one or more of the following supported open license(s)" ... "you must at least credit all the authors (except for CC0 licensed artwork)" ... "Importantly, the individual licenses may impose additional restrictions. It's your responsibility to conform to the licenses imposed by the artwork in use."
  The README does **not** say in so many words that you may pick one licence. That rule comes from the OGA FAQ and general practice.

### 1.2 The permissive options themselves

- **OGA-BY 3.0** (VERIFIED, https://static.opengameart.org/OGA-BY-3.0.txt): this is CC BY 3.0 with the technical-measures (DRM) restriction removed. Section 8(g) says: "You may distribute this work under the terms of the Creative Commons Attribution 3.0 license (CC-BY 3.0)". Section 4(b) requires "the name of the Original Author (or pseudonym, if applicable) if supplied", "the title of the Work if supplied", a URI "where practicable", and, for adaptations, "a credit identifying the use of the Work in the Adaptation". Section 3(b) requires adaptations to take "reasonable steps to clearly label, demarcate or otherwise identify that changes were made to the original Work".
- **OGA-BY 4.0** (VERIFIED, https://opengameart.org/sites/default/files/archive/oga-by-40.txt and https://opengameart.org/content/oga-by-40-faq): the FAQ says "OGA-BY 4.0 is a license based on CC-BY 4.0 that removes that license's restriction on technical measures that prevent redistribution of a work" and "OGA-BY 4.0 explicitly allows content to be relicensed under CC-BY 4.0." The licence text says (section 7(c)): "You may distribute this work under the terms of the Creative Commons Attribution 4.0 license (CC-BY 4.0)".
- **OGA FAQ on the difference** (VERIFIED): the OGA-BY licences "are derivatives of the CC-BY licenses ... which removes the restriction against technical measures that prevent redistribution of a work. (eg. DRM)".
- **CC BY 3.0, section 4(a)** (VERIFIED, https://creativecommons.org/licenses/by/3.0/legalcode): "You may not impose any effective technological measures on the Work that restrict the ability of a recipient ..." This is the clause OGA-BY drops. It does not matter for Hugging Face or GitHub, which use no DRM.

### 1.3 Pitfalls

1. **The licence list is per credit row (per file group), not per item.** The generator's `CREDITS.csv` has one row per file path. One item, for example `legs/pants/legs_cuffed`, can have rows with different licence lists ("OGA-BY 3.0" and "OGA-BY 3.0; GPL 3.0"). moviepix's rule 0002 already checks every credit row for the files actually used. Keep it that way. Never decide at item level. (VERIFIED in `data/built/lpc64v2/CREDITS.csv`.)
2. **Derivative files and upstream sources.** Many LPC files are edits of edits: "original by wulax, recolors by bigbeargames; adapted to v3 bases ... by bluecarrot16". A derived file can be offered under a licence only if **every** upstream contributor offered it (or something that converts to it). The generator's maintainers did that analysis when they wrote each row. moviepix relies on it. If a row is wrong, moviepix inherits the error. Risk: MEDIUM. Mitigation: the audit in 1.4 and 1.5.
3. **Version strings.** Rows contain "OGA-BY 3.0+", "CC-BY 3.0+", "OGA-BY-3.0" and bare "CC-BY" with no version. In credits, write the exact version you rely on (for example OGA-BY 3.0) and link that licence text. For bare "CC-BY", check the source page for the version before use. In the README credits, `head/heads/human/heads_human_elderly_small` has "OGA-BY 3.0; CC-BY" and **no URL**, and `head/heads/human/heads_human_female` links only to `https://opengameart.org/content/` (a broken link). Both need fixing.
4. **Which option you take matters for redistribution, not for training.** Taking OGA-BY 3.0 means the attribution duties of OGA-BY 3.0 apply wherever you share the files or adaptations of them. Section 3 covers this.
5. **Recording the choice.** `assets/CREDITS.md` says "Each piece is used under the first permissive license listed." That is a good, explicit rule. Better still, add a `license_used` column to `CREDITS.csv`, so the chosen licence is machine-readable.

### 1.4 Finding: the LPC body and head bases (MEDIUM, act on it)

The `body/body` rows and four `head/heads/human/*` rows credit **Benjamin K. Smith (BenCreating)** and claim "OGA-BY 3.0; CC-BY-SA 3.0; GPL 3.0". Every moviepix character uses a body.

- BenCreating's page "[LPC] Character Bases" (https://opengameart.org/content/lpc-character-bases, VERIFIED 2026-11-18) lists **"License(s): CC-BY-SA 3.0 GPL 3.0"**, with no OGA-BY.
- The 2012 LPC base assets page (https://opengameart.org/content/liberated-pixel-cup-lpc-base-assets-sprites-map-tiles, VERIFIED) also lists "CC-BY-SA 3.0 GPL 3.0", but adds: "NOTE: The assets by Leana "Sharm" Zimmerman and Stephen "Redshrike" Challener are also available under the OGA-BY 3.0 License."
- On the Character Bases comment thread (page 2, VERIFIED), a user asked on 2026-06-30 why the generator lists OGA-BY when the page does not. An experienced OGA user (MedicineStorm) answered on 2026-07-01: "if BenCreating can be contacted and agrees to also license these derivatives as OGA-BY as well, then all downstream uses can also inlcude OGA-BY ... Never-the-less, we need his permission for these derivatives to be licensed OGA-BY."
- **BenCreating replied on 2026-07-08**: "I have no problem licensing these (or any of my other submissions) under OGA-BY. I haven't had time to check the licenses recently, but I seem to remember some of the animal heads were not compatible yet."

INFERENCE: the author's own public comment is a reasonable basis for using his base contributions under OGA-BY 3.0. But the page's licence field has not been updated, and the comment is informal. Steps:
1. Save a copy of that comment (URL, date, text) in the repo, for example `docs/licensing/evidence/` (outside this note's scope; Hoopakid's call).
2. Ask BenCreating, politely, to add OGA-BY 3.0 to the page's licence field. Ask the generator project to cite the comment in its credits.
3. Do not use animal heads credited to BenCreating until checked ("some of the animal heads were not compatible yet").
4. Mention this in the dataset card's known-issues section.

### 1.5 Audit of every upstream OGA page in the v2 credits

See section 9. Short version: 26 of 81 upstream OGA pages show no permissive licence in their licence field, and 90 of 370 credit rows link only to such pages. Most packs state per-file permissive options in an internal credits file, so these rows are probably fine but **unverified** by moviepix.

---

## 2. Training on CC-BY / OGA-BY images: does attribution reach the weights?

### 2.1 Creative Commons' position

- **"Understanding CC Licenses and Generative AI"** (VERIFIED, https://creativecommons.org/2023/08/18/understanding-cc-licenses-and-generative-ai/): CC licences "do not supersede existing limitations and exceptions". In CC's view, in the US "using copyrighted works to train generative AI models would be fair use", and in the EU "such training can be protected by the text and data mining exception". It adds that "whether these limitations apply may depend on the particular use case", and: "To the extent you are relying on CC licenses to train AI, you will need to follow the relevant requirements under the licenses."
- **"Using CC-Licensed Works for AI Training"** (VERIFIED, https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/):
  - "CC licenses apply only when copyright permission is required." "If exceptions or limitations apply, then the CC license terms don't apply."
  - Attribution, ShareAlike and NoDerivatives "are triggered only when works or adaptations of works are publicly shared."
  - "For AI model training, attribution could be a simple link to the source of the dataset used to train the model."
  - "If AI models or outputs are based on ShareAlike content and they will be shared publicly, ... following the ShareAlike condition would require AI developers to use the same CC license as the original works." (This is why decision 0002 excludes SA-only art.)
  - Caveat from CC itself: "following this guidance is likely to lead to overcompliance with both copyright law and CC license terms. It assumes the most restrictive legal interpretation for those who wish to take a conservative approach and minimize risk."
  - On memorisation: mitigations "can mitigate, but may not eliminate, the risk of copyright infringement."
- **CC FAQ** (VERIFIED, https://creativecommons.org/faq/): "If someone uses a CC-licensed work with any new or developing technology, and if copyright permission is required, then the CC license allows that use without the need to seek permission from the copyright owner." On datasets: "Where a CC-licensed work is distributed as part of a database or dataset ... the license conditions must be respected", by "providing the required attribution information in a way that is reasonable under the circumstances", and "in some cases that may be as simple as providing a link to the website".
- **CC licences' own flexibility** (VERIFIED, CC BY 4.0 s. 3(a)(2), https://creativecommons.org/licenses/by/4.0/legalcode.en): conditions may be met "in any reasonable manner based on the medium, means, and context in which You Share the Licensed Material".

### 2.2 Preference signals (CC signals)

- Announced 2025-06-25 (VERIFIED, https://creativecommons.org/2025/06/25/introducing-cc-signals-a-new-social-contract-for-the-age-of-ai/) as "a new preference signals framework designed to increase reciprocity and sustain a creative commons in the age of AI", letting dataset holders "signal their preferences for how their content can be reused by machines". The signals "may range in enforceability".
- Update 2026-04-23 (VERIFIED, https://creativecommons.org/2026/04/23/update-on-cc-signals-what-changed-and-why/): "CC signals is no longer limited to signaling preferences"; feedback said "preference signals without enforcement do not meaningfully shift power". CC decided not to change the licences: "CC licenses were designed not to enable control beyond copyright." A public alpha is planned for 2027 (SECONDARY: headline of a related CC post).
- Relevance to moviepix: no CC signal is attached to LPC art today. The direction of CC signals, credit and reciprocity, matches the plan to publish credits with weights and dataset. Re-check before each release.

### 2.3 Courts on whether weights hold copies (SECONDARY: law-firm summaries)

- **Getty Images v Stability AI**, [2025] EWHC 2863 (Ch), 4 Nov 2025 (UK High Court; e.g. https://www.mayerbrown.com/insights/publications/2025/11/getty-images-v-stability-ai-what-the-high-courts-decision-means-for-rights-holders-and-ai-developers, https://cm.twobirds.com/en/insights/2025/uk/stability-ai-defeats-getty-images-copyright-claims-in-first-of-its-kind-dispute-before-the-high-cour): the model can be an "article", but Stable Diffusion's weights did not store or contain copies of the works, so it was not an "infringing copy".
- **GEMA v OpenAI**, LG München I, 42 O 14139/24, 11 Nov 2025 (e.g. https://www.taylorwessing.com/de/insights-and-events/insights/2025/11/gema-v-openai, https://cm.twobirds.com/en/insights/2025/landmark-ruling-of-the-munich-regional-court-(gema-v-openai)-on-copyright-and-ai-training): lyrics **memorised** in the model count as reproduction in the weights, and the TDM exception does not cover that memorisation. First instance, appeal announced.
- INFERENCE for moviepix: the model is tiny, and the training set is a few hundred LPC layer files recombined. A small model trained on a small, repetitive set can memorise parts. Wave 1 notes say models trained on modular LPC data "mostly memorize parts". Under the GEMA reasoning, released weights could then contain reproductions of OGA-BY / CC-BY art, and the attribution conditions would apply to the weights. Fix: always ship the full credits **inside** the weights repo, not only as a link. That meets the conservative reading at almost no cost.

### 2.4 Academic commentary

- Szkalej and Senftleben (IViR, University of Amsterdam), "Mapping the Impact of Share Alike/Copyleft Licensing on Machine Learning and Generative AI", 12 June 2024 (VERIFIED, https://ivir.nl/publicaties/download/Share-Alike-and-ML.pdf). It finds "significant problems with several concepts in CC licensing agreements ('adapted material' and 'technical modification') that could serve as a basis for applying ShareAlike and copyleft conditions to trained models, curated datasets and AI output". Where TDM exceptions prevail, AI developers are "free to use CC material for AI training purposes without seeking permission – and without accepting SA obligations". It supports the view that licence conditions reaching models is uncertain both ways. moviepix's choice to comply anyway is the safe one.

### 2.5 Recommended practice (INFERENCE from the sources above)

- Model card: a "Training data" section naming LPC, the generator, the rule (permissive options only), a link to the dataset or dataset card, and a link to the **exact** `CREDITS.csv` used for that checkpoint.
- Put `CREDITS.csv` (and a human-readable `CREDITS.md`) in the weights repo. Add a sentence: "These weights were trained on the artworks listed in CREDITS.csv, used under the licences named there. If the weights are considered to contain any of these works, the credits and licence links in CREDITS.csv apply to them."
- Keep "modified" language: "Training images were composed from these layers, recoloured and downscaled; this does not imply endorsement by the artists." CC BY and OGA-BY both forbid implying endorsement.
- Pin each checkpoint to a dataset build hash, so the credits match exactly.

Risk: **MEDIUM** in law (unsettled), **LOW** in practice if the steps above are followed.

---

## 3. Publishing the composed dataset

### 3.1 Sources

- CC FAQ (VERIFIED, https://creativecommons.org/faq/):
  - "All Creative Commons licenses (including the version 4.0 licenses) allow licensed material to be included in collections" ... "You may choose a license for the collection, however this does not change the license applicable to the original material."
  - Combining: "The first question to ask is whether doing so constitutes an adaptation." "If the combination does not create an adaptation, then you may combine any CC-licensed content so long as you provide attribution".
  - Adaptations: "the original CC license always applies to the material you are adapting even once adapted." "Recipients of the adaptation must comply with both the CC license on the original and your adapter's license."
  - Modifications: "You must also indicate if you have modified the work" ... "It is not necessary to note trivial alterations, such as correcting a typo or changing a font size." ... "you must retain an indication of previous modifications to the work."
  - "CC licenses have a flexible attribution requirement, so there is not necessarily one correct way to provide attribution."
- CC BY 4.0, s. 3(a)(1) (VERIFIED): keep "identification of the creator(s)", "a copyright notice", "a notice that refers to this Public License", "a notice that refers to the disclaimer of warranties", "a URI or hyperlink to the Licensed Material to the extent reasonably practicable"; "indicate if You modified the Licensed Material and retain an indication of any previous modifications"; "indicate the Licensed Material is licensed under this Public License, and include the text of, or the URI or hyperlink to, this Public License."
- CC BY 4.0, s. 2(a)(5)(B): "You may not offer or impose any additional or different terms or conditions on ... the Licensed Material if doing so restricts exercise of the Licensed Rights by any recipient". So no extra terms (for example "no AI use") on the art files themselves.
- OGA-BY 3.0, s. 4(a): "You must include a copy of, or the Uniform Resource Identifier (URI) for, this License with every copy" you distribute.
- CC "Best practices for attribution" (VERIFIED, https://wiki.creativecommons.org/wiki/Best_practices_for_attribution): "we recommend that the attribution include the Title, Author, Source, and License" (TASL). Its example for an adaptation begins "This work, "90fied", is adapted from ..." and ends "licensed under CC BY 4.0 by [Your name here]". For many works: "Specific works are named", "Different authors noted for the different works", and "it is clear which attribution belongs to which work."
- LPC generator README (VERIFIED): to meet attribution you may "distribute the entire CREDITS.csv file with your project or distribute a composed list of credits for the assets you use."

### 3.2 What the dataset is, legally (INFERENCE)

Each character image layers several pieces into one picture, recolours some and may downscale to 32 px. That is more like an **adaptation** of each piece than a plain collection. CC 3.0 counts a Collection as works assembled "in unmodified form"; a layered, recoloured sprite is not that. So:
- Each composed PNG is an adaptation of several works, each under its chosen licence (CC0 / CC-BY 3.0 / CC-BY 4.0 / OGA-BY 3.0 / OGA-BY 4.0).
- Combining these is allowed: none is ShareAlike or NoDerivatives. BY licences let you license your own adaptation under any terms as long as the originals' conditions are still met (CC BY 4.0 s. 3(a)(4) has similar wording; check before relying on it).
- The dataset as a whole (selection, arrangement, metadata, splits, captions, code) is Hoopakid's compilation. It can carry its own licence.

### 3.3 Recommended licence and notice

- **Simplest honest option:** a `LICENSE` / `LICENSE.md` for the dataset saying:
  1. "The artwork in this dataset is not under a single licence. Each image is an adaptation of the works listed for it in CREDITS.csv, each used under the licence named in the `license_used` column (CC0 1.0, CC BY 3.0, CC BY 4.0, OGA-BY 3.0 or OGA-BY 4.0). You must follow those licences when you reuse the images, including attribution."
  2. "Hoopakid's own contribution (the selection, composition, recolouring, metadata, CSV files and splits) is released under CC BY 4.0" (or CC0, if Hoopakid wants the fewest strings).
  3. Licence links for each licence used, and the warranty disclaimers.
- On Hugging Face, the `license` field takes one identifier (VERIFIED: the dataset card spec's `license` field is a single value, with `license: other`, `license_name`, `license_link` for anything else). Use `license: other`, `license_name: moviepix-lpc-mixed` (any id) and `license_link: LICENSE.md`. Alternative: since OGA-BY 3.0 may be distributed as CC BY 3.0 (s. 8(g)) and CC0 imposes nothing, nearly the whole set *could* be offered as CC BY 3.0. But mixed 3.0/4.0 items make a single tag misleading. Prefer `other` with a clear explanation.

### 3.4 Credits format (TASL plus modifications)

Add these columns to `CREDITS.csv`: `title` (OGA page title or generator item name), `authors`, `source_urls`, `licenses_offered`, `license_used`, `license_url`, `modifications`, `files`, `notes` (keep upstream notes such as "original by wulax, recolors by ...").

Example human-readable line (CREDITS.md):

> "LPC Medieval Fantasy Character Sprites" (plate armour, male) by Johannes Sjölund (wulax), Matthew Krohn (makrohn), Michael Whitlock (bigbeargames), bluecarrot16, JaidynReiman. Source: https://opengameart.org/content/lpc-medieval-fantasy-character-sprites. Used under OGA-BY 3.0 (https://static.opengameart.org/OGA-BY-3.0.txt). Modified by Hoopakid: composited with other layers, recoloured, cropped/downscaled.

Dataset-level notice (top of CREDITS.md and the dataset card):

> "Images in this dataset are adaptations of Liberated Pixel Cup artwork. Layers were composited, some were recoloured, and 32 px versions were downscaled. These changes are by Hoopakid and are not endorsed by the original artists. The original works are provided without warranties; see each licence."

Risk: **LOW** if every image row maps to credit rows, and licence URLs and modification notes are present.

---

## 4. Outputs made from users' own sprites

- No primary source says a model's training-data licence attaches to outputs that do not reproduce the training works. CC's 2023 post: a CC licence "will apply to the creative work that you contribute to the final product", while "the portion produced by the generative AI system itself may be uncopyrightable". CC's training guide treats outputs as subject to licence conditions only where they are "based on" the licensed content and shared (and then mainly for ShareAlike, which moviepix avoids).
- INFERENCE: moviepix copies pixels from the **user's** sprite by position. The frames are adaptations of the user's own input. The only realistic route for LPC terms to reach an output is if the model reproduces recognisable LPC pieces, for example by filling occluded areas with memorised LPC armour or hair. Even then the relevant licences are attribution-only, so the duty would be to credit, not to share-alike.
- Separate issue (INFERENCE, MEDIUM for users): if a user feeds in a sprite they have no rights to, the output is still an adaptation of that sprite. Say so plainly.
- Suggested README / model card text:
  > "Outputs: moviepix animates the sprite you give it. Hoopakid claims no rights in your outputs. moviepix's code and weights are Apache-2.0, and nothing in those licences restricts what you do with your outputs. The model was trained only on artwork offered under permissive licences (CC0, CC BY, OGA-BY). We know of no licence term from that artwork that applies to frames generated from your own sprite. If an output closely resembles a specific LPC artwork, treat it as that artwork and credit it (see CREDITS.csv). You are responsible for having the rights to the sprites you animate. Copyright in AI-assisted output depends on your jurisdiction and on how much of it is your own creative work. This is not legal advice."
- Risk: **LOW**.

---

## 5. CC-BY-SA art only for evaluation

- Using SA art privately to compute metrics is not "sharing". CC: conditions "are triggered only when works or adaptations of works are publicly shared" (VERIFIED, training guide). INFERENCE: weights trained without SA data are not adaptations of SA test images. Evaluating on them does not change that. Keep a hard guard so SA files can never enter a training split (test in CI).
- Publishing eval **numbers** is fine. Publishing SA test **images** (originals or model outputs on them) is sharing. Outputs made from an SA input are adaptations of that input, so publish them under CC BY-SA with credits.
- Benchmark redistribution (VERIFIED, CC FAQ): "CC licenses do not require the collection or the compilation itself to be made available under an SA license, even though each individual work is still licensed individually under an SA license." A benchmark can therefore include SA test images if (a) each SA file stays under its licence, with TASL credits and a "modified" note if cropped or converted, (b) model outputs on those inputs are released CC BY-SA, (c) the benchmark's own code and metadata can stay Apache-2.0 / CC BY. Keep SA files in a separate folder (for example `eval/sharealike/`) with their own `LICENSE` and `CREDITS`.
- GPL-only art: the same approach works (GPL 3.0 for those files and their adaptations), but GPL art in a benchmark is awkward ("source" for an image is unclear). Prefer CC BY-SA or permissive art for public test sets.
- Risk: **LOW** (training-split leakage is the main practical risk).

---

## 6. GPL and reading the generator's data files

- The generator repo's licence (GPL-3.0, per the GitHub sidebar) covers its code. Its README says the art in `spritesheets/` carries its own per-file licences (VERIFIED quote in 1.1).
- GNU GPL FAQ (could not be fetched during this research because of connection resets on gnu.org; from memory, so **UNVERIFIED**): the FAQ entry "In what cases is the output of a GPL program covered by the GPL too?" (anchor `#GPLOutput`) says output is covered only when the program copies part of itself into it. moviepix does not even run the generator's code. It reads files. Check the exact wording at https://www.gnu.org/licenses/gpl-faq.html#GPLOutput.
- INFERENCE:
  - PNG art: governed by the licence in CREDITS.csv, not by the repo's GPL. Art offered under GPL **only** is excluded by rule 0002.
  - JSON sheet definitions: these are written by the generator's contributors and are in the GPL-3.0 repo. Many are mostly factual (file paths, layer order, credit lists, palette names). Factual data has thin or no copyright, but a JSON file as a whole could be treated as GPL-covered. Reading them at build time to compose images is a use, not a distribution, and the GPL does not restrict running or private use. Do **not** commit copies of the generator's JSON (or code ported from its JS) into the Apache-2.0 repo or the dataset. Store only derived facts (paths, author names, licences), which moviepix's CREDITS.csv already does. Keep the generator as a fetched dependency with a pinned commit.
  - `CREDITS.csv` from the generator is factual metadata (names, licences, URLs). Reproducing the entries needed for attribution is what the README asks downstream users to do.
- Risk: **LOW** (LOW to MEDIUM only if whole JSON files or JS code get copied into the repo).

---

## 7. AI disclosure on itch.io and Steam

- **Steam** (SECONDARY; Valve's Steamworks post from January 2024 could not be read with the tool):
  - Since January 2024 the Content Survey has an AI section with **Pre-Generated** content ("created with the help of AI tools during development") and **Live-Generated** content (made while the game runs). Developers must state that pre-generated content is not illegal or infringing. For live generation they must describe their guardrails. Disclosures appear on the store page. (PC Gamer, 2024-01-10, https://www.pcgamer.com/valve-updates-its-steam-policy-on-ai-so-it-can-release-the-vast-majority-of-games-that-use-it/)
  - January 2026 change to the form, as quoted by GamingOnLinux (2026-01-17, https://www.gamingonlinux.com/2026/01/valve-tweak-steam-ai-disclosure-form-for-developers-to-clarify-its-for-content-consumed-by-players): "Efficiency gains through the use of these tools is not the focus of this section." "Instead, it is concerned with the use of AI in creating content that ships with your game, and is consumed by players." "This includes content such as artwork, sound, narrative, localization, etc."
  - INFERENCE: sprite frames generated by moviepix and shipped in a game are "artwork ... consumed by players". A developer should tick **Pre-Generated** and describe it. A game that runs moviepix at runtime is **Live-Generated**.
- **itch.io** (SECONDARY): since November 2024 asset creators must fill in an AI generation disclosure. Projects that answer yes get an "AI Generated" tag (with sub-tags for graphics, sound, text, code), others "No AI". Undisclosed AI projects can lose browse indexing (Engadget, 2024-11-21, https://www.engadget.com/ai/itchio-marketplace-now-requires-asset-creators-to-disclose-their-use-of-generative-ai-130031999.html; GamesMarket, 2024-11-21, which says games also get the tag). Engadget reports even hand-edited AI output must be tagged. A forum quote of itch.io's wording: "We ask that you accurately tag your project if it contains materials produced by generative AI" (user post, https://itch.io/post/14465369, UNVERIFIED as official text). Read the current wording on the project edit page before relying on it.
- Suggested disclosure text for users (put it in the README "Using outputs in games" section):
  > "Some character animation frames in this game were generated with moviepix, an open-source AI model (Apache-2.0) that animates sprites from a single still image. The input sprites are [our own art / from <pack> by <artist>, <licence>], and the generated frames were [reviewed and hand-edited]. moviepix was trained only on openly licensed Liberated Pixel Cup art (CC0, CC BY, OGA-BY); credits: <link to CREDITS>."
  Steam short form: "Pre-generated: some sprite animation frames were created with the moviepix AI tool from our own hand-drawn sprites, then reviewed and edited by our artists."
- Hoopakid's own posture: the model card and README should say clearly that moviepix is generative AI, so users know they must disclose. On itch.io, any demo or asset pack Hoopakid publishes with model outputs should be tagged AI Generated (graphics).

---

## 8. Release checklist

### Before release (data)
- [ ] Rule 0002 check runs per credit row (done). Add a test that fails the build if any used file's row lacks a permissive option.
- [ ] Add `license_used`, `license_url`, `title`, `modifications` columns to `CREDITS.csv`.
- [ ] Fix rows with no URL or a broken URL (`heads_human_elderly_small`, `heads_human_female`), and bare "CC-BY" without a version.
- [ ] Resolve the **BenCreating base / head** licence gap (1.4): archive the 2026-07-08 comment, ask for the OGA licence field to be updated, and skip BenCreating animal heads until checked.
- [ ] Review section 9 flags (90 rows rely only on flagged pages; check them against each pack's internal CREDITS file, starting with `torso_armour_legion` and the human heads). For each flagged OGA page, confirm the permissive option comes from another upstream page or an author statement, or drop the file.
- [ ] Pin the generator commit used, and record it in each build's metadata.
- [ ] Keep SA/GPL-only art out of training splits (CI test). Keep eval-only SA art in its own folder with its own LICENSE/CREDITS.

### Weights repo (Hugging Face model)
- [ ] YAML: `license: apache-2.0`, `datasets: [<hf dataset id>]` (if published), `library_name`, `pipeline_tag` (e.g. image-to-image), `tags: [pixel-art, sprite-animation, lpc]`.
- [ ] Files: `LICENSE` (Apache-2.0), `CREDITS.csv` + `CREDITS.md` for the exact training build, `NOTICE` (short: "Trained on Liberated Pixel Cup artwork used under CC0/CC BY/OGA-BY; see CREDITS.md").
- [ ] Model card sections: Model description; Intended use and limitations; **Training data** (source, filter rule, build hash, link to credits); **Attribution** (sentence from 2.5); **Outputs** (text from section 4); **AI disclosure guidance** (section 7); Evaluation; Bias/limitations (LPC style only, memorisation of LPC parts); How to cite; "Not legal advice" line.
- [ ] Run a memorisation check: compare outputs on held-out inputs with training layers, and report the nearest-neighbour similarity. This supports the "no copies in weights" position (Getty) and shows good faith under GEMA-style reasoning.

### Dataset repo (Hugging Face dataset), if published
- [ ] YAML: `license: other`, `license_name: moviepix-lpc-mixed`, `license_link: LICENSE.md`, `pretty_name`, `task_categories`, `tags`.
- [ ] `LICENSE.md` with the three-part text in 3.3, plus links to CC0 1.0, CC BY 3.0, CC BY 4.0, OGA-BY 3.0, OGA-BY 4.0 texts.
- [ ] `CREDITS.csv` (machine) + `CREDITS.md` (human, TASL), with a per-image mapping (character id -> credit rows).
- [ ] Dataset card sections: Summary; Source and licences; How images were made (modifications); Known issues (1.4); Personal info (none, only artist names as published); Citation; Not legal advice.

### GitHub repo
- [ ] README "Data and licenses": add "modified" wording, the Outputs paragraph, and a link to this note.
- [ ] `assets/CREDITS.md`: add licence links (OGA-BY 3.0 URL, etc.) and a modification line. These are required by OGA-BY 3.0 s. 4(a) and CC BY s. 3(a)(1)(C).
- [ ] OSAID 1.0 (VERIFIED, https://opensource.org/ai/open-source-ai-definition) asks for "Sufficiently detailed information about the data used to train the system", "The complete source code used to train and run the system" and "The model parameters". moviepix can meet all three (data is shareable, code is Apache-2.0, weights Apache-2.0). Say so in the model card only after the dataset or a full build recipe is public.

---

## 9. Automatic audit of upstream OGA licence fields

(Filled in below. The method: for every unique opengameart.org URL in `data/built/lpc64v2/CREDITS.csv`, fetch the page and read its "License(s)" field. A page is flagged **XX** if it shows none of CC0, OGA-BY, CC-BY 3.0 or CC-BY 4.0. A flag does not by itself mean the file is unusable. A credit row often lists several source pages, and the permissive option may come from an author statement elsewhere, as in 1.4. Each flag needs a manual look.)

**Results (2026-11-18, VERIFIED by fetching each page):**

- 81 unique OGA pages in the v2 credits. **55** show a permissive option in the page licence field. **26** do not: 23 list only CC-BY-SA and/or GPL, and 3 (`lpc-jewelry`, `lpc-expanded-armor`, `lpc-expanded-armour`) could not be parsed and need a manual check.
- Of 370 credit rows, **90 rows link only to flagged pages**, and 168 more link to at least one flagged page.
- **Main explanation (per-pack vs per-file licensing):** many LPC packs give the pack page a "lowest common" licence (CC-BY-SA) and list permissive options per file in an internal credits file. Quotes from the page descriptions:
  - `lpc-hair`: "License for the whole submission is CC-BY-SA 3.0, but most assets herein are also available under GPL 3.0, OGA-BY 3.0, or CC0. See CREDITS-hair.txt within."
  - `lpc-expanded-socks-shoes`: "All assets except Pennomi and Laetissima's Ghillies and Luke Mehl's Slippers are also available in OGA-BY 3.0." ... "Shoes (ALL OGA-BY 3.0)"
  - `lpc-pirates`: "All original items drawn by me-self be licensed OGA-BY 3.0. Items based on the work of others be available under the original license, usually CC-BY-SA 3.0 and GPL v3.0."
  - `lpc-extended-weapon-animations`: "If you use this entire submission, the license is CC-BY-SA 3.0. However, most items are also available under OGA-BY 3.0+ and/or GPL v3.0 license(s). See README-weapons.txt for details."
  - `lpc-helmets-mega-pack`: "most sheets are available under additional licenses, including CC-BY 3.0+ and OGA-BY 3.0. See CREDITS-helmet.csv"
  - `lpc-folk`: "many assets allow OGA-BY 3.0 license as well. See CREDITS-heads.csv within"
  - `lpc-more-backpacks`: "everything except `basket_contents/wood/*` was created entirely by bluecarrot16 and can be used under CC0 license."
  - `lpc-expanded-hats-facial-helmets` marks items one by one, for example "Pirate Eyepatch (also OGA-BY 3.0 ...)" versus "Small Eyepatch (CC-BY-SA 3.0 and GPL 3.0 only ...)".
  - `liberated-pixel-cup-lpc-base-assets-sprites-map-tiles`: OGA-BY 3.0 only for "The assets by Leana "Sharm" Zimmerman and Stephen "Redshrike" Challener".
  So a flag usually means the permissive claim lives in a file inside the download, not that it is wrong. This is exactly the "per-file vs per-item" pitfall: **never read the pack page's licence field alone, and never assume the whole pack is permissive.**
- **Flagged pages whose description has no visible permissive statement** (need a manual check of the in-pack credits file or an author statement): `lpc-character-bases` (see 1.4; resolved by BenCreating's comment, pending archive), `lpc-roman-armor` (legion torso/shoulders/skirt/helmet/sandals; `torso_armour_legion` is used by 1,394 characters), `lpc-clothing-updates` (aprons, straight skirt), `lpc-7-womens-shirts` (blouses), `lpc-santa` (winter beard CC0, christmas hat), `lpc-gentleman` (santa jacket), `lpc-2-characters`, `lpc-glow-sword`, `lpc-pregnancy-bases-maternity-wear`, `lpc-teen-*` pages, `lpc-walk-animations-for-bows`, `lpc-expanded-hair`, plus the three unparsed pages. Rows depending only on these include `head/heads/human/heads_human_female` (2,599 characters) and `heads_human_male` (2,573).
- Suggested process: write a small script that downloads each pack zip once, reads its internal CREDITS file, and checks that each file moviepix uses is listed with the permissive option claimed in the generator's CREDITS.csv. Store the pack file, the date and the matching line as evidence. Until that check passes, mark these rows "unverified upstream" in the dataset card. Risk: **MEDIUM** (the generator maintainers are careful and the claims are probably right, but moviepix currently trusts them without checking).

---

## 10. Open questions for a lawyer

1. Is an informal public comment by an author (BenCreating, 2026-07-08) enough to rely on OGA-BY for his derivative bases, or is a written grant or an updated licence field needed?
2. If a small model memorises LPC parts, are the released weights a "copy" or "adaptation" under (a) EU/German law after *GEMA v OpenAI* and (b) US law? Does shipping CREDITS with the weights fully meet CC BY / OGA-BY attribution in that case?
3. Are composed, recoloured characters "Adaptations" (3.0) / "Adapted Material" (4.0), and does licensing Hoopakid's contribution under CC BY 4.0 (or CC0) work with mixed 3.0 and 4.0 inputs?
4. Do the generator's JSON sheet definitions carry enough originality to be covered by GPL-3.0, and does storing derived facts (paths, credits) in an Apache-2.0 repo raise any issue?
5. For eval-only CC BY-SA images: does publishing model outputs on them create SA obligations only for those outputs (expected) and not for the weights?
6. Jurisdiction: which country's law applies to Hoopakid's publication, and does a TDM exception (EU) or fair use (US) apply to the training itself?

---

## Sources

- OpenGameArt FAQ: https://opengameart.org/content/faq
- OGA-BY 3.0 text: https://static.opengameart.org/OGA-BY-3.0.txt
- OGA-BY 4.0 text and FAQ: https://opengameart.org/sites/default/files/archive/oga-by-40.txt, https://opengameart.org/content/oga-by-40-faq
- LPC Character Bases page and comments: https://opengameart.org/content/lpc-character-bases (and `?page=1`)
- LPC base assets page: https://opengameart.org/content/liberated-pixel-cup-lpc-base-assets-sprites-map-tiles
- Universal LPC generator README: https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator
- CC BY 4.0 legal code: https://creativecommons.org/licenses/by/4.0/legalcode.en
- CC BY 3.0 legal code: https://creativecommons.org/licenses/by/3.0/legalcode
- CC FAQ: https://creativecommons.org/faq/
- CC, Understanding CC Licenses and Generative AI (2023-08-18): https://creativecommons.org/2023/08/18/understanding-cc-licenses-and-generative-ai/
- CC, Using CC-Licensed Works for AI Training: https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/
- CC, Best practices for attribution: https://wiki.creativecommons.org/wiki/Best_practices_for_attribution
- CC signals: https://creativecommons.org/2025/06/25/introducing-cc-signals-a-new-social-contract-for-the-age-of-ai/ and https://creativecommons.org/2026/04/23/update-on-cc-signals-what-changed-and-why/
- Szkalej & Senftleben, IViR (2024): https://ivir.nl/publicaties/download/Share-Alike-and-ML.pdf
- Getty v Stability (summaries): https://www.mayerbrown.com/insights/publications/2025/11/getty-images-v-stability-ai-what-the-high-courts-decision-means-for-rights-holders-and-ai-developers
- GEMA v OpenAI (summaries): https://www.taylorwessing.com/de/insights-and-events/insights/2025/11/gema-v-openai
- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
- Hugging Face licences: https://huggingface.co/docs/hub/repositories-licenses ; model cards: https://huggingface.co/docs/hub/model-cards ; dataset card spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md
- OSAID 1.0: https://opensource.org/ai/open-source-ai-definition
- Steam: https://www.pcgamer.com/valve-updates-its-steam-policy-on-ai-so-it-can-release-the-vast-majority-of-games-that-use-it/ ; https://www.gamingonlinux.com/2026/01/valve-tweak-steam-ai-disclosure-form-for-developers-to-clarify-its-for-content-consumed-by-players
- itch.io: https://www.engadget.com/ai/itchio-marketplace-now-requires-asset-creators-to-disclose-their-use-of-generative-ai-130031999.html
