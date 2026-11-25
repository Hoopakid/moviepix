# E3: candidate character packs for training and final test

Status: shortlist only. Date of all page reads: **2026-11-25**. Nothing was downloaded.

## 0. How to read this file

- **Licence read** means I read the page on 2026-11-25 with a web-fetch tool. The tool uses a small model to pull quotes from the page. I did not see raw HTML. So a quote may differ by small details. **A person should re-read the licence line on the page before any download.**
- **UNVERIFIED** means the page did not state it. I did not guess.
- **OpenGameArt was not used.** Its robots.txt names Claude-User, ClaudeBot and anthropic-ai with `Disallow: /`, and it returned HTTP 403. I did not try another route. This removes many well-known CC0 packs (section 6). Note: the wave 1 and wave 2 notes say OGA was read by plain HTTP GET after the same 403. That was a way around the block, and it should not be repeated.
- itch.io browse pages (`/game-assets/...`) also returned 403, so the search was by web search, itch "games like" pages, artist profiles and user collections. It is not exhaustive. Collections and "games like" pages were used only to find packs, never as a licence source.
- Frame counts, cell sizes and layouts must be **measured** after download, as in `wave2-oos-packs.md`. Most pages do not state them.
- `Puny` / `Ninja Adventure` / `Chasersgaming` / `Antifarea` and LPC work are excluded. Packs already in `data/raw/ooS` (Bit Era, Calciumtrice) are not listed again, because the project has already looked at them.
- No AI-training clause was found on any page in section 1 unless it says so. No one wrote "AI training is allowed" either; CC0 has no such limit.

Flag letters used below:
**C** = the CC0 / public-domain words are only in the author's own comment or profile, not in a licence field.
**M** = the artist forbids machine learning on *other* packs (a sign of their wishes).
**R** = another page by the same artist has a redistribution limit.
**A** = an AI disclosure appears on this page or on another pack by the artist.

**Spot check (2026-11-25):** the licence line was re-read in the raw page HTML for 9 pages (Ordinary Bumblebee x2,
Lexlom, Foozle x2, JIK-A-4 MetroCity, ansimuz x2, Robocelot template) and matched what is written below. All other
pages were read only through the fetch tool.

## 1. Summary table, ranked by usefulness (CC0 / public domain, free)

| # | Pack | Artist | Licence | Size stated | Directions stated | Actions stated | Characters | Flag | Split |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Customizable Character Pack | Ordinary Bumblebee | CC0 field + text | 32x32 | 4 | walk, run, idle, jump, interact, punch, sword, hurt | 1 layered system (count UNVERIFIED) | | TRAIN |
| 2 | 32 Free animated characters pack | Lexlom | Public domain, author comment | 32x32 grid | 4 (one profile; mirror it) | idle, walk (+ model sprite: run) | 32 | C | TRAIN |
| 3 | Lucifer heroes and enemies (13 packs) | Foozle, art commissioned from "David" (12) and "Baldur" (1) | CC0 field + text | UNVERIFIED | 4 | 10 / 7 / 9 "in each direction"; **names not given, walk UNVERIFIED** | 13 | | TRAIN |
| 4 | 8 Direction Walk Cycle | Ordinary Bumblebee | CC0 field + text | 32x48 character | 8 | walk, standstill | 1 in 10 colours | | TRAIN |
| 5 | MetroCity - Character Pack | JIK-A-4 | CC0 field | UNVERIFIED | "4 Axes" | run only | 3 models + hair/outfits | | TRAIN |
| 6 | Free RPG Tiles and Running Animation | JIK-A-4 | CC0 field | 32x32 (tag) | 4 ("Forward, Rearward, and Left and Right") | run (24 sprites total) | 1 | | TRAIN |
| 7 | Tiny Farmer / TinyPlayerTemplate | Oul Studio | CC0 field | UNVERIFIED | 4 | walk, idle, tool, pick up, sprinkler / + punch, sword, hurt | 1 + 1 (may share a base) | R | TRAIN (pending) |
| 8 | 16bit Super Chicken Pack | Ordinary Bumblebee | CC0 field + text | 16x16 | 4 | attack, talk, walk, more | 1 creature | | TRAIN (minority) |
| 9 | Legend of Faune | ansimuz | CC0 field | about 32x32 | up, side, down | walk, walk+lift, run, push, faint | 1 | | **FINAL TEST** |
| 10 | Tiny RPG Adventure (tiny-rpg-forest) | ansimuz | CC0 field | 16x16 tiles; character UNVERIFIED | 4 | not named | 1 hero (+2 enemies, directions UNVERIFIED) | | **FINAL TEST** |
| 11 | 8-direction Top Down Character | Gamekrazzy | Public domain, author comment | UNVERIFIED | 8 | walk, roll, attack | 1 | C | **FINAL TEST** |
| 12 | 8-Directional Game Boy Character Template | GibbonGL | CC0 field | 16x16 | 8 | walk UNVERIFIED | 1 (monkey) | | **FINAL TEST** |
| 13 | LowRez 8-Directional 2D Sprite | Hormelz | CC0 field | 16x16 canvas | 8 (4 for attack/block) | run or walk (page conflicts), idle, attack, block | 1 | R | **FINAL TEST** |
| 14 | Free 16x16 Pixel Character Template | PixelFight | CC0, author comment | 16x16 tile sets | 4 | walk, action, attack, blink, faint, idle/crouch, fall | 1 blank template | C, A | RESERVE |
| 15 | Bird + Cat + Fox + Raccoon | PixelFight | CC0, author comment | 32x32 grid | 4 | run (8 frames), idle | 4 creatures | C, A | RESERVE |
| 16 | Top-Down RPG Character Sprite | Tnypixels | CC0 field | UNVERIFIED | 4 | walk, idle | 1 | A | RESERVE |
| 17 | 16x16 Playable Character Template | 5yvalia | CC0 field + text | 16x16 grid | 4 | walk, push, pull, pickup/throw, punch, slash, swim, climb | 1 | M | HOLD |
| 18 | Top Down NPCs | 5yvalia | CC0 field + text | 12x16 grid | 4 | walk | parts + premade, count UNVERIFIED | M | HOLD |
| 19 | 16x16 Character Template - 8 Direction | 5yvalia | CC0 field + text | 16x16 | 8 | walk, idle | 2 (male, female) | M | HOLD |

**Totals.** Section 1 holds 32 page-verified free packs: Foozle 13 (David 12, Baldur 1), Ordinary Bumblebee 3, 5yvalia 3, JIK-A-4 2, Oul Studio 2, ansimuz 2, PixelFight 2, and one each from Lexlom, Gamekrazzy, GibbonGL, Hormelz and Tnypixels. That is 13 artists, counting David and Baldur apart.
- 28 packs show a CC0 licence field on the page. The other 4 (Lexlom, Gamekrazzy, PixelFight x2) rely on the author's own CC0 / public-domain comment.
- "Clean" artists (no M, R or A flag; C is allowed): David, Baldur, Ordinary Bumblebee, JIK-A-4, ansimuz, GibbonGL, Lexlom (C), Gamekrazzy (C). That is 8 artists.
- Section 2 adds 5 CC-BY packs by 4 artists.

## 2. CC-BY, needs credit (kept separate)

Use only with a credits entry. I suggest final test only, not training (see section 8).

| Pack | Artist | URL | Licence words on page | Size / directions / actions as stated | Characters |
|---|---|---|---|---|---|
| Gumdrop Galaxy Characters | Robocelot | https://robocelot.itch.io/gumdrop-source-and-sprites | "The license is Creative Commons 4.0 International" (text) and "Creative Commons Attribution v4.0 International" (licence field) | about 24x30 in 64x64 cels (author comment). "each animation is provided in four directions" (tool's words, not a checked quote). Idle, walk, run, draw, sheath, hurt, drawn idle/walk/hurt, 3 attacks. Frame counts per action not stated | 8 designs, 3+ colours |
| Pixelart Character Template | Robocelot | https://robocelot.itch.io/pixelart-character-template | "Creative Commons Attribution v4.0 International" (field). Author comment: "even modified versions shouldn't be used for NFT content or promotion" | "8 frames walk x 4 directions", "6 frames idle x 4 directions"; search text says 48x48 | 1 |
| Small 8-direction Characters | AxulArt | https://axulart.itch.io/small-8-direction-characters | "This work is licensed under a Creative Commons Attribution 4.0 International License." (text) | Author comment: "in reality they are 16x17px in size". 8 directions. Action and frame count not stated | Girl, Boy, Template |
| Skull Knight/Hero | Burrito | https://immortal-burrito.itch.io/skull-knighthero | "Creative Commons Attribution v4.0 International" (field); author: "as long as you credit me (Immortal_burrito or just as Burrito)" | "A fantasy 16x16 Hero". Idle, walk, attack, parry in 4 directions; frames not stated | 1 |
| Free Pets Pack | Frakassets | https://frakassets.itch.io/free-pets-pack | "License: https://creativecommons.org/licenses/by/4.0/ (credit "Frakassets")" (text) | 16, 32, 48, 64 px. 4 sides. Idle, walk, happy, sleep. "Play them at 12 frames per second." | 10 animals (creatures) |

All five list "No generative AI was used". No AI-training clause was found on any of the five. Other CC-BY pages seen but weaker: **Gumdrop Basic Character Template** (Robocelot, CC-BY field, 24x32 in 64 cels, "3 Directions", 5 animations, https://robocelot.itch.io/gg-basic-template-2021); **o_lobster TOP-DOWN Adventure Pack** (CC-BY field, main character 4 sides, no frame counts, https://o-lobster.itch.io/adventure-pack); **marceles 8px assets** (CC-BY field, 8 px, directions not stated, too small); **iPixl animated character** (CC-BY field, directions not stated). **Gumdrop Character Template V 2** shows no licence, so it is excluded.

## 3. Per-pack details (section 1)

For every pack: date read 2026-11-25. Price is "Name your own price" unless stated. "AI row" is the itch Content field. "Clauses" means AI-training, redistribution or resale limits on that page.

### 3.1 Ordinary Bumblebee (TRAIN)
- **Customizable Character Pack**, https://ordinary-bumblebee.itch.io/customizable-character-pack. Licence: "License CC0" (end of description) and field "Creative Commons Zero v1.0 Universal". AI row: "No generative AI was used". Clauses: none found. Size: "Customize your own 32x32 characters!". Directions: "all of them 4 directions:". Actions: walk, run, idle, jump, interact, punch, sword, hurt. Frames: not stated. Characters: 1 layered body; "More than 2.5 million character variations!". Style: cute 32 px, clothing layers. Why: the only free CC0 pack here with 8 actions and 4 stated directions, and it can be composed into many looks. Note: composed looks are not independent characters. Do not count them as 2.5 million.
- **8 Direction Walk Cycle**, https://ordinary-bumblebee.itch.io/8-direction-walk-cycle. Licence: "License CC0" (description) and the CC0 field; "Credit is not necessary but always appreciated :)". AI row: no AI. Clauses: none found. "Character is 32x48". "8 direction walk cycle and 8 direction standstill." Frames: not stated. "Same character in 10 different colors." View: not stated. Why: taller proportions (48 px).
- **16bit Super Chicken Pack**, https://ordinary-bumblebee.itch.io/16bit-super-chicken-pack. Licence: "License CC0" (description) and the CC0 field. AI row: no AI. "16x16 chicken character." "4 directions and a total of 19 different movements and reactions." "Simple 4 frame movements." View: not stated. Why: creature minority.

### 3.2 Lexlom (TRAIN)
- **32 Free animated characters pack**, https://lexlom.itch.io/32-character-pack-pixel-art-4-direction-idle-walking. Licence field: absent. Author comment (Lexlom): "Yes, it is Public Domain, absolutely free. Enjoy it!" Also "It is totally free. No payments, no rigths, no licenses, etc!" AI row: no AI. Clauses: none found. "32x32px grid". 4 directions; one profile direction, mirror for the other. Actions: idle, walk; a model sprite also has walk and run. Frames: not stated. 32 characters. Why: the widest character variety of any free pack found. Weak point: the dedication is a comment, so save a screenshot of it before use. Lexlom's other free pack (15 characters, 64x64) has **no licence text**, so it is not listed.

### 3.3 Foozle "Lucifer" series (TRAIN, as one group)
Pages read: `lucifer-warrior`, `lucifer-necromancer`, `lucifer-sorceress`, `lucifer-cultist-enemy`, `lucifer-goblin-berserker-enemy`, `lucifer-goblin-slinger-enemy`, `lucifer-skeleton-grunt-enemy`, `lucifer-skeleton-hunter-enemy`, `lucifer-possessed-enemy`, `lucifer-skeleton-king-boss`, `lucifer-skeleton-ancient-boss`, `lucifer-goblin-rider-boss`, `lucifer-goblin-beast-boss`, all at https://foozlecc.itch.io/<slug>.
- Licence (Licence section): "License: (Creative Commons Zero, CC0)"; "This content is free to use and modify for all projects, including commercial projects."; "Attribution not required." Field: "Creative Commons Zero v1.0 Universal". (The first line was not returned for Goblin Rider and Skeleton Hunter; the field and "free to use" sentence were.)
- AI row: "No generative AI was used". Clauses: none found.
- Artist: "Commissioned from: David" on 12 packs, "Commissioned from: Baldur" on Possessed. Distributed by Foozle. Treat the series as one style group.
- Directions and actions: heroes "4 direction warrior/necromancer/sorceress", "10 animations in each direction"; enemies "7 animations in each direction"; Skeleton King, Ancient and Goblin Beast "9 animations in each direction"; Goblin Rider 7.
- **Action names, frame counts, cell size, character height: not stated.** Walk is very likely but UNVERIFIED. A commenter only guessed 32x32. Files differ a lot in size (0.7 to 6.5 MB), and bosses may exceed 64 px. Measure first.
- Characters: 13 (3 heroes, 6 regular enemies plus 4 bosses; Goblin Rider is a mounted character and may be non-humanoid).
- Why: the largest clean 4-direction set by a small number of artists, with several actions each.

### 3.4 JIK-A-4 (TRAIN)
- **MetroCity - Free Top Down Character Pack**, https://jik-a-4.itch.io/metrocity-free-topdown-character-pack. Licence: field "Creative Commons Zero v1.0 Universal"; description "You can use it as you wish."; author comment "Yes you can also use it in a commercial project. ;-)". AI row: no AI. Clauses: none found. "4 Axes Running Animation". "3 Basic character models." Plus hair, outfits, shirts. Size and frames: not stated. Style: larger heads (author comment, summarised). Note: a commenter says hair is animated only facing forward. Why: modern-city chibi style.
- **Free RPG Tiles and Running Animation**, https://jik-a-4.itch.io/rpg. Licence: field "Creative Commons Zero v1.0 Universal" (code field "Unlicense"). AI row: no AI. 32x32 (tag). "Forward, Rearward, and Left and Right" (the page says it has "a total of 24 sprites including 4 animations"). Frames per direction: not stated. 1 character. Why: one more 32 px body. Both packs are run, not walk: map to `run` unless the art shows otherwise.

### 3.5 Oul Studio (TRAIN, pending a decision)
- **Tiny Farmer**, https://oul-studio.itch.io/tiny-farmer. Licence: field "Creative Commons Zero v1.0 Universal"; author comment "It's CC0 ;)". AI row: no AI. Clauses: none found. 4 directions. Walk, idle, tool, pick up, sprinkler. Size, frames: not stated. 1 character.
- **TinyPlayerTemplate**, https://oul-studio.itch.io/tinyplayertemplate. Licence: field "Creative Commons Zero v1.0 Universal". AI row: none on the page. "Player Sprites (4 directions):". Walk, idle, tool, punch, pick up, sword, sprinkler, hurt. The page says "create your own characters like this tiny farmer", so it may share a base with Tiny Farmer.
- Doubt (flag R): the same artist's **GB Adventure Characters** page has the CC0 field but also says "Don't resell or redistribute the raw files." That is self-contradictory, so that pack is excluded. Ask whether the artist intends this limit elsewhere before using these two.

### 3.6 ansimuz (FINAL TEST)
- **Legend of Faune**, https://ansimuz.itch.io/legend-of-faune. Licence: field "Creative Commons Zero v1.0 Universal". AI row: no AI. Clauses: none found. "Character ~32x32 px". "Faune Anims (11): Walk (Up, Side, Down), Walk with Lift (Up, Side, Front), Run (Up, Side, Down), Push, and Faint." Frames: not stated. "1 fully animated female protagonist." Why: cleanest final-test character: 3 drawn + mirrored side, walk, run, faint.
- **Tiny RPG Adventure**, https://ansimuz.itch.io/tiny-rpg-forest. Licence: field "Creative Commons Zero v1.0 Universal". AI row: no AI. Clauses: none found. "Fully animated character (4 directions)". Action names, size, frames: not stated. "2 enemies: Treant & Mole" (directions not stated). Why: 16-bit outlined style.
- Excluded, same artist: *Top Down Adventure Assets* says "You can NOT re-distribute the file", so it is not usable. Read each ansimuz page on its own.

### 3.7 Gamekrazzy (FINAL TEST)
- **8-direction Top Down Character**, https://gamekrazzy.itch.io/8-direction-top-down-character. Licence field: absent. Author comment: "so it's Public Domain, and you are free to use it for whatever you want." Description: "this character graphic is completely free". AI row: no AI. Clauses: none found. 8 directions, left and right drawn separately. Walk, roll, attack; sword on a separate layer. Size, frames: not stated. 1 character. Why: asymmetric character, so mirrored-side assumptions are tested.

### 3.8 GibbonGL (FINAL TEST)
- **8-Directional Game Boy Character Template**, https://gibbongl.itch.io/8-directional-gameboy-character-template. Licence: field "Creative Commons Zero v1.0 Universal"; description "Feel free to use it in any way you want but i encourage you to make your own adaptations. Credits are appreciated!". AI row: no AI. Clauses: none found. "8-directional 16x16 character template". Animation names, frames: not stated. 1 monkey. Why: Game Boy low colour. Not a humanoid in the strict sense.

### 3.9 Hormelz (FINAL TEST)
- **LowRez 8-Directional 2D Sprite**, https://hormelz.itch.io/8-directional-2d-sprite. Licence: field "Creative Commons Zero v1.0 Universal". AI row: no AI. Clauses on this page: none found. "Canvas size 16x16". Idle and run are 8-directional; "Attack and Block animations are only available in 4 directions." The list says Run but the text says "2 animations, idle and walk." "2-frame 8-direction." 1 character. Why: very small 16 px style; only 2 frames.
- Doubt (flag R): on another Hormelz page the author wrote "These assets are CC0, meaning you can basically use them however you like, short of reposting them." Use privately; do not republish crops without asking.

### 3.10 PixelFight (RESERVE)
- **Free 16x16 Pixel Character Template**, https://pixelfight.itch.io/16x16-retro-pixel-character-template. Licence field: absent. Author comment: "CC0 no credit, no license, no nothing." AI row: no AI. Walk, action, attack in 4 directions; blink, faint, idle/crouch, fall. Frames: not stated. 1 blank template.
- **Bird + Cat + Fox + Raccoon**, https://pixelfight.itch.io/birdcat. Licence field: absent. Author comment: "Licensing is CC0 so you have full control to do as you wish!" AI row: no AI. "Each spritesheet is built for a 32x32 grid." "a standard 8 frame run animation in four directions". 4 animals. Alt text lists Run 1 to 4, which conflicts with 8.
- Doubt (flag A): the same artist's *104 Free Dogs* page is tagged "AI Assisted, Graphics". Excluded. *Bear* and *Half Sized Hero* have no licence text. Excluded.

### 3.11 Tnypixels (RESERVE)
- https://tnypixels.itch.io/top-down-rpg-character-sprite-walk-idle-animation (agusstt.itch.io redirects here; same person, not a reupload). Licence: field "Creative Commons Zero v1.0 Universal"; "Ready for commercial and personal use". **AI disclosure: "AI Assisted" with the sub-tag "Text".** Graphics are not tagged as AI. Clauses: none found. "4-direction walk cycles (up, down, left, right)", idle. Size, frames: not stated. 1 character. The disclosure is limited to text, but keep it out until checked.

### 3.12 5yvalia (HOLD, needs the project owner's decision)
- **16x16 Top Down Playable Character Template**, https://5yvalia.itch.io/rpg-type-retro-top-down-playable-character-template. "**LICENSING**: Creative Commons Zero, CC0"; "This content is free to use and modify for all projects, including commercial projects."; CC0 field. AI row: no AI. Clauses on this page: none found. "3-color palette". Walk, push, pull, pickup/throw, use/punch, side slash, swim, climb. "Most animations have 3 unique frames". 1 template.
- **Top Down NPCs**, https://5yvalia.itch.io/rpg-type-retro-top-down-npc-sprites. CC0 field and the same "free to use" sentence. AI row: no AI. "12x16" grid. "Includes simplified walk animations in cardinal directions (north, south, east, west)". Count not stated.
- **16x16 Character Template - 8 Direction**, https://5yvalia.itch.io/16x16-character-template-8-direction. CC0 field and the same sentence. AI row: no AI. "a male and female base". "8-direction idle" and "8-direction walk". Frames not stated.
- **Why HOLD:** the artist's newer *urban characters* page and *character creator* page say "You may not use, scrape, or feed these assets or their derivative works into any machine learning models". Those two packs are excluded. The three CC0 packs do not carry this clause on their own pages, so CC0 covers them. But the artist's wishes are clear. I suggest leaving them out unless the owner decides otherwise.

## 4. Maybe

Not 4-direction, too large, paid, or thin page text.

| Pack | Why only "maybe" |
|---|---|
| Kevin's Mom's House *camelot_*, https://kevins-moms-house.itch.io/camelot, and *dungeonSprites_*, https://kevins-moms-house.itch.io/dungeonsprites | Text "[CC-0]" linked to CC0. AI row: no AI. 10 and 15 characters; idle, run, walk/run, hurt, death. Tag Top-Down, but directions not stated, and a "turn" animation suggests one side view. Size not stated. |
| Vaca Roxa *Generic RPG Pack*, https://bakudas.itch.io/generic-rpg-pack | CC0 field, AI row: no AI. "two little animated characters", "six mobs". Directions, frames not stated. |
| Eduardo Scarpato *Top-Down Character Animations*, https://eduardscarpato.itch.io/top-down-character | "CC0. Free for commercial/personal use." (License section). AI row: no AI. "10+" animations. Directions, size, frames not stated. Says "crafted by Dagon Dev" (tool summary), so the artist name is unclear. |
| Kenney *RPG Urban Pack*, https://kenney.nl/assets/rpg-urban-pack | Label "Creative Commons CC0"; support page: "all game assets on the asset pages are public domain licensed (CC0)". 16x16, tag "character". No text on characters, directions or frames. |
| o_lobster *Simple Dungeon Crawler*, https://o-lobster.itch.io/simple-dungeon-crawler-16x16-pixel-pack | CC0 field, AI row: no AI. Directions and actions not stated. |
| Gnash Cypher *Pixel Character Base*, https://gnash-cypher.itch.io/pixel-character-base | CC0 field, AI row: no AI. "Small five frame 48x48 pixel walking character." Directions not stated. |
| GibbonGL *Mini Game Boy Asset Pack 1*, https://gibbongl.itch.io/free-mini-game-boy-asset-pack-1 | CC0 field, AI row: no AI. "An elf/gnome character with a bunch of animations". Directions not stated. Same artist as #12, so FINAL TEST side only. |
| KabGames *GB Zelda-Style*, https://kaboff.itch.io/gb-zelda-style-tileset-sprites | CC0 field, AI row: no AI. Characters, directions, animations not described. |
| Stealthix *Animated Monsters*, https://stealthix.itch.io/animated-monsters, and *Mr Placeholder Jr.* | CC0 field, AI row: no AI. View and directions not stated. |
| Foozle *Spire - Enemy Pack 1 and 2*, *Scallywag - Pirates* | CC0, AI row: no AI. Directions not stated. Pirates are by Pixel Carvel, pack 1 and 2 by David. |
| Hormelz 8-Directional Knight, Girl, 2DHD, https://hormelz.itch.io/ | CC0 field, but 256x256 canvas. The Girl page says it is a 2D version of Styloo's Mari. Too large. |
| RGS_Dev *Hand-Drawn Square Characters*, https://rgsdev.itch.io/hand-drawn-square-characters-animated-8-directions-top-down-free-cc0 | "The license is CC0" and CC0 field; no AI. 8 directions, 4 characters, but 128x128 and pixel art not stated. |
| styloo *2D Village*, https://styloo.itch.io/2d-village | CC0 field; AI row: no AI; 8 directions; 107 MB zip. Blender is in "Made with", and whether it is pixel art is not stated. |
| Panhandler *Animated Walking People*, https://panhandler.itch.io/animated-walking-people | "License: no restrictions, it's yours". Not a named CC0 text. 64x64, 8 frames; directions not stated. Wording is strong but informal. |
| Dajeki *8x8 Fantasy Character Sprites* | CC0 field, but 8x8 and directions not stated. |
| Paid CC0 packs (not free) | Riclmbr *Pixel Enemies* ("CC0 license – use in commercial projects, no credit required", $5.19, 48x48, 50 enemies) and *Modular Pixel Characters* ("CC0 1.0 - Public Domain", $4.21, 48x48, 4 directions, walk/run/idle, 24 + 14 characters); PixelFight *12 Pack* (CC0 field, $3); GibbonGL *GB Oracles* ($5; licence read only from search text); Oul *Top-Down Player Template* (CC0 field, $1, 22x28, walk 8 / run 8 / idle 5 frames, 4 directions). Buying is the owner's decision. |

Side-view-only or not usable view (kept for reference only): Zeenaz *Pixel Boy 32x64* ("CC0 licence, free to use"; directions not stated); LuizMelo, Pixel Frog, 0x72, Sunny Land and GrafxKid sprite packs (side-view in wave 1 notes; GrafxKid pages not checked for licence here).

## 5. Excluded (seen and rejected)

| Pack / artist | Reason on page |
|---|---|
| Oul *GB Adventure Characters* | CC0 field but "Don't resell or redistribute the raw files." |
| Atari Boy *Water Knight* | CC0 field but "You can't redistribute or resell the assets, modified or not!" and "(NFT not allowed!)" |
| 5yvalia urban, 4-direction template, creator | No-redistribution; ML ban on urban and creator |
| freezepops walk packs | $2, and *Animated Walk Characters* shows "AI Assisted, Graphics" |
| PixelFight *104 Free Dogs* | "AI Assisted" "Graphics" |
| u_master *Ox Warrior* | "Character graphics were created with OpenAI image generation" |
| lucasguido wizard, GegX walk pack | AI graphics disclosure (GegX from search text only) |
| bobddadoo free pack | "Do not use these assets to train AI / machine-learning models" |
| furkan90x 8-Direction Hero | "Not for AI / machine-learning training datasets." |
| Sleeping Robot *Memao* | "Use the assets in AI training or generative AI tools." is prohibited |
| azwaryoga | "Use the assets to train AI image generators" prohibited |
| rustydbolt | "Ask first if you are going to use it to train an AI model." |
| pancakemonster117, ertylerex (Super) | No AI training / no machine learning |
| Larchrow, playbiscuitgames, RunninBlood, LyniaDesign, Nectanebo, MikAnimus, Vryell, Pixel_Poem, Arcade Island, comedylife, tonguesurgery, kettoman, deepdive, snoblin, 10KStudios, yazumazero, Mattz Art, Sscary, gzh_d | No-redistribution clause |
| kva55, spectrecz, naruka0, bossnelnel, pixivan, szadiart | No-resale clause or custom terms |
| RGS_Dev pixel base template, Toadzillart, KingKelpo | CC BY-ND field |
| route1rodent 16x16 RPG | "CC BY-SA 4.0"; also adapted from OGA NES sprites |
| TheLazyStone | Non-commercial unless paid; no redistribution |
| Game Between The Lines, Immunity, andreecy, Maytch, Azibela, tokka, Maranza, Tyst, Superdark, ankousse26, piepzeru, TotusLotus, savvycow, Warren Clark (2), Drasnus, aeynit, bannerscalled, i-am-44, Chroma Dave pack | Free-use words but no clear licence (custom or none) |
| saltandpixel NES characters | $3, "No downloading/re-uploading" |

## 6. Could not verify (pages not read)

- **OpenGameArt (all pages).** 403 and robots.txt block. These appeared in web-search text only and are **not** verified: *Character 4 directional walking* (sodri), *Simple animated character*, *Pixel Art Character 64x64* (acasas), *Girl RPG sprite sheet*, *Walking character set*, *8x8 character pack*, *Top down player sprite sheet Julia*, *Zelda-like* sprites. A person can read them in a browser.
- itch.io browse pages (403): used only as search-engine text. *Cute Pixel Animals* (Riclmbr), *Pixel Chibi Heroes*, *Tiny Heroes* (DRStudios, price shown as $2.99 or $3.99), *CHROME DISTRICT* (booliebuilds), *50 Free Chibi Characters Kit* (24thForce): licence and AI status not read.
- 404 or redirects not followed: *tmpelay top-down player template* (404; the Oul page covers it), `samvieten` asset pack (404), `lyaseek` minifolks (404), `pixelwrightstudio` top-down-heroes (redirect not followed), `vexed` (redirect not followed), `schwarnhild` basic character (404).
- Never opened: Pipoya, Anokolisa, Clockwork Raven, shubibubi (already excluded or unclear in wave 1).

## 7. Doubts to resolve before any download

1. Foozle: **action names, frame counts and cell sizes are not on the pages**. Walk is UNVERIFIED for all 13. If the art has no walk, rank 3 falls.
2. JIK-A-4 packs are `run`, not `walk`, by the page text.
3. Lexlom, Gamekrazzy and PixelFight: the CC0 words are in comments, not the licence field. Archive them.
4. Derivatives: no page in section 1 says it derives from LPC or the excluded artists. That is silence, not proof. Check by eye, especially Lexlom (32 px villager proportions) and Oul.
5. Quotes come through a fetch tool, not raw HTML. Re-read before use.
6. The "AI row" is a self-declaration by the artist. TinyPlayerTemplate has no AI row.
7. Fewer than 15 clean, distinct artists were found, because OGA was off limits. If the owner reads OGA by hand, the list will grow.

## 8. Proposed split of artists

**Rule: the split is by artist, not by pack. Every artist in FINAL TEST must never be trained on, not fine-tuned on, not used for hyperparameter or checkpoint choice, not used for augmentation or pixelisation, and not shown in prompt examples. This includes all of that artist's other packs, recolours and any pack derived from them. Open and measure the final-test files only once the model and settings are frozen, if possible.**

| Split | Artists | Distinct characters (as stated) |
|---|---|---|
| TRAIN | Foozle/David + Baldur (one group), Ordinary Bumblebee, Lexlom, JIK-A-4, and Oul Studio if section 3.5 is cleared | Foozle 13 + Bumblebee 3 (1 layered body, 1 walk-cycle body, 1 chicken) + Lexlom 32 + JIK-A-4 4 = **52**, plus Oul 1 to 2 = **53 to 54** |
| FINAL TEST (CC0) | ansimuz, GibbonGL, Gamekrazzy, Hormelz | Faune 1 + Tiny RPG hero 1 + monkey 1 + Gamekrazzy 1 + LowRez 1 = **5** (Treant and Mole excluded, directions unknown) |
| FINAL TEST (CC-BY add-on) | Robocelot, AxulArt | 8 + 1 + 3 = **12**, with credit |
| RESERVE | Tnypixels, PixelFight | 1 + 5 |
| HOLD | 5yvalia | pending a decision |

Reasoning
- TRAIN gets the artists with the most characters or actions, because diversity is what the model lacks. Lexlom and the Foozle group are too big to spend on a test.
- FINAL TEST gets artists who are small, clean or have a style that TRAIN lacks: 16 px Game Boy (GibbonGL), 16 px low-res 2-frame (Hormelz), 16-bit about 32 px (ansimuz), an asymmetric 8-direction hero (Gamekrazzy).
- Five CC0 characters are too few for a tight score. The CC-BY add-on gives **17 characters across 6 artists**, and Robocelot brings 24x30 characters in 64 px cels with 4 directions and attack and hurt. The CC-BY files are used for evaluation only, so no credit question reaches the weights; a `CREDITS` entry is still needed for any published crops. If the owner wants CC0 only, use the 5 and report wide error bars.
- David and Baldur stay together: they were commissioned for one series with one brief, so splitting them would leak style.
- The GibbonGL and Hormelz other packs also move to the FINAL TEST side, and must not be trained on later (for example Hormelz Knight or GibbonGL Mini pack). The same holds for all ansimuz packs, including *Sunny Land*, which wave 1 lists.
- Checked: none of these artists except ansimuz (Sunny Land, listed in wave 1 but not downloaded) appears in the repo text. `data/raw/ooS` has none of them.
- Style gap: no verified CC0 free pack is clearly "large 48 to 64 px". That axis rests on Ordinary Bumblebee (32x48), Robocelot (64 cels, CC-BY) and the paid Riclmbr packs.
