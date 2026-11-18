# Wave 2: Out-of-style (OOS) test set from real sprite packs

Status: research notes, 2026-11-18. Project author: Hoopakid.
Input: `wave1-track3-styles-data.md` §2.1. This wave re-checked the licences, downloaded the free files, measured them, and wrote slicing specs.

Conventions
- **MEASURED** means read from the downloaded PNGs with Pillow or numpy. The scripts are in `data/raw/ooS/_scripts/` (`measure.py`, `cells.py`, `compare.py`, `strips.py`, `blocks.py`, `locate.py`, `gifmatch.py`, `preview.py`). Previews are in `data/raw/ooS/_previews/`.
- **VISUAL** means I worked it out by looking at upscaled previews. It is not stated by the author or proven by pixel equality.
- All rectangles are `[x, y, w, h]` in source-image pixels. Frame lists are in playback order.
- Directions use moviepix names (`up, left, down, right`, the order of `moviepix.actions.DIRECTIONS`).
- Every licence was read on the original page on **2026-11-18**. OGA pages were fetched with plain `curl`, and the "License(s)" field and body text were extracted from the HTML.
- Nothing from the downloads was executed. Only `.png` members were extracted from the zips (with `unzip '*.png'`). `.bmp` and `Thumbs.db` were left inside the archives.

---

## 0. Summary table (measured)

| Pack | Licence | View | Native cell | Chars used | Dirs (independent) | Usable actions → moviepix | Colours / sprite | Background |
|---|---|---|---|---|---|---|---|---|
| Puny Characters | CC0 | top-down chibi | 32×32 (content ≈ 13–15 px tall, feet at y=22) | 5 designs, 10 sheets (colour variants) | 8 (4 used), all drawn, almost never mirrored | idle 2, walk 4-cycle, sword 4, bow 4, stave 3, throw 4, hurt/death 5 | 32–44 per sheet | alpha 0/255 |
| Ninja Adventure (Superpowers copy) | CC0 | top-down chibi big-head | 16×16 (content fills 15–16 px) | 23 full + 2 small + dog | 4; right is an exact mirror of left for 10/23 chars | walk 4, attack 1, jump 1 | 4–12 | alpha 0/255 |
| Chasersgaming Knight NES | CC0 | top-down retro | 16×24 (attack 24×24) | 1 | 4, all drawn (no mirrors) | idle 4, walk 4, run 4, attack 4, push 4, climb 4 | 3 (+transparent) | alpha 0/255 |
| Chasersgaming Ranger NES | CC0 | top-down retro | 16×24 (attack 24×24) | 1 | 4, all drawn | idle 4, walk 4, run 4, attack 4, push, climb | 3 per strip, 5 with weapon | alpha 0/255 |
| Chasersgaming Ranger GB | CC0 | top-down retro, 4-shade GB | 16×24 in a composite sheet | 1 | 4 | idle 4, walk 4 (others in the composite are not mapped) | 4 (GB greens) | alpha 0/255 |
| Antifarea twelve 16×18 | CC-BY 3.0 | top-down JRPG | 16×18 | 12 (+1 template) | 4; left middle frame = mirror of right | walk 3 | 9–17 | **opaque**; colour-key blocks (blue/pink/grey) |
| Bit Era Extended | CC0 | top-down/side chibi | 16×16 | 1 design × 20 recolours | 3 (down, right, up); left = mirror | walk 4, idle(side) 4, attack 3 (right), jump/fall/death 1 each | 3 | alpha 0/255 |
| Calciumtrice Rogue/Warrior/Wizard | CC-BY 3.0 | 3/4 side, facing right | 32×32 (content up to 32 px tall) | 6 (3 classes × m/f) | 1 (right only) | idle 10, gesture 10, walk 10, attack 10, death 10 | 15–22 per sheet | alpha 0/255 |
| LuizMelo Martial Hero (not downloaded) | CC0 | side, large | not measured | 1 | 1 | page: Idle 8, Run 8, Jump 4, Fall 4, Attack1 6, Attack2 6, Take Hit 4, Death 6 | not measured | not measured |

LPC reference for placement (MEASURED on `data/raw/lpc/spritesheets/body/bodies/male/walk.png`): the body bbox in a 64×64 cell is x 17–47, y 32–62 (exclusive). So the body is about 30 px tall, the feet sit on y = 61 (the last opaque row), and it is centred at x ≈ 32. The child body is y 37–62.

---

## 1. Licences (re-verified 2026-11-18)

| Pack | URL | Quote (copied from the page/file today) |
|---|---|---|
| Puny Characters (free) | https://opengameart.org/content/puny-characters | "License(s): CC0". Body: "This is a free asset pack! Feel free to use this for your game (commercially or not). Feel free to creatively modify these sprites however you like. No need to give me credit." Also "All of the sprites are made in 32x32 canvas but they can fit in a 16x16 space." and "Animations (8 Directional) : Idle … Walk … Sword Attack … Bow Attack … Stave Attack … Throw … Hurt … Death". |
| Ninja Adventure (Superpowers repo) | https://github.com/sparklinlabs/superpowers-asset-packs (commit `e8674a03ab4456802f71f848c4df79eccca23f7a`, 2019-09-21) | README: "They are released under the Creative Commons Zero (CC0) license." "You can use the assets found in this repository in your own games, even commercial ones. Attribution is not required but appreciated." `LICENSE.txt` begins "CC0 1.0 Universal" (saved as `ninja-adventure-superpowers/LICENSE.txt`). |
| RPG Character Knight NES | https://opengameart.org/content/rpg-character-knight-nes | "License(s): CC0". "4 Directional character sprite sheet, imitating the colour pallet of a NES system(PAL) and limitations. 4 Animations frames: Idle, Walk, Run, Push, Attack, Hit/Hurt (16x24 pixels)". |
| RPG Character Ranger NES | https://opengameart.org/content/rpg-character-ranger-nes | "License(s): CC0". Same text, plus "UPDATED: Better animations, individual sprite sheets, BMP files added that are NESmaker compatible." |
| RPG Character Ranger Gameboy | https://opengameart.org/content/rpg-character-ranger-gameboy | "License(s): CC0". "4 Directional character sprite sheet, Gameboy Colour Palette." |
| (Knight Gameboy, not downloaded) | https://opengameart.org/content/rpg-character-knight-gameboy | "License(s): CC0" |
| Twelve 16x18 RPG sprites, plus base | https://opengameart.org/content/twelve-16x18-rpg-sprites-plus-base | "License(s): CC-BY 3.0". "If you use this art, you must include a link back to OpenGameArt.org in your attribution (as well as attributing the author, Antifarea)." |
| Bit Era Game Character - Extended | https://opengameart.org/content/bit-era-game-character-extended | "License(s): CC0". "All spritesheets have a transparent background." "Cell size: 16px H=6, V = 4". "I'm releasing this under a CC0 licence, so enjoy!" (The original by Pav Creations is CC0 per wave 1.) |
| Calciumtrice Animated Rogue / Warrior / Wizard | https://opengameart.org/content/animated-rogue, …/animated-warrior, …/animated-wizard | "License(s): CC-BY 3.0". "Animated Rogue by Calciumtrice, usable under Creative Commons Attribution 3.0 license." "Credit should be given to Calciumtrice." Each lists "idle gesture walk attack death". |
| LuizMelo Martial Hero | https://luizmelo.itch.io/martial-hero | "This package can be used freely and commercially - CC0(creative commons zero). Credits are not required but I would appreciate it." Asset license field: "Creative Commons Zero v1.0 Universal". The page also says "Content: No generative AI was used". That describes the art; it is not a restriction on use. Frame list: "Idle - 8 frames Run - 8 frames Jump - 4 frames Fall - 4 frames Attack1 - 6 frames Attack2 - 6 frames Take Hit - 4 frames Death - 6 frames". |

None of these pages has a no-AI or no-redistribution clause. The CC-BY packs (Antifarea, Calciumtrice) need a `CREDITS` entry wherever crops are released. Antifarea's entry must include a link to OpenGameArt.org.

---

## 2. Downloaded files

All files are under `data/raw/ooS/` (4.2 MB in total).

| Folder | File (sha256, first 16 hex digits) | Used content |
|---|---|---|
| `puny-characters/` | `puny-characters.zip` (971d7b07eea18fb2) → `x/Puny-Characters/*.png` | 10 sheets of 768×256: Archer-Green/Purple, Character-Base, Mage-Cyan/Red, Soldier-Blue/Red/Yellow, Warrior-Blue/Red. Also Slime.png (480×32, 14 frames, not directional) and `Environment/`. `page-gifs/` holds the OGA preview GIFs 2–9 (used only to confirm the walk cycle). |
| `ninja-adventure-superpowers/` | `1.png`…`25.png`, `dog.png`, `LICENSE.txt`, `preview-part-1.png` (pinned commit) | 23 sheets are 64×112. `4.png` and `8.png` are 64×32. `dog.png` is 32×16. |
| `chasersgaming-knight-nes/` | `NES_Knight_Free_Version.zip` (d6acb6b143250d31), `Knight_Sprite_Sheet_1_NES.zip` (older 250×232 sheet) | `x/NES Knight Free Version/<Action> PNG/NES PNG Knight_<Action>_<DIR>_strip4.png` |
| `chasersgaming-ranger-nes/` | `NES_Ranger_Files_FREE.zip` (e2c34f1399678d1a), `NES_Ranger_SpriteSheet_Free.png` | `x/NES Ranger Files FREE/NES Ranger Free Set PNG/NES Ranger Single Sheets/<Action>/NES_Ranger_<Action>_<DIR>.png` |
| `chasersgaming-ranger-gb/` | `GB_Ranger_SpriteSheet_Free_0.png` (c18b8fa543fe21f1), 352×280 | The GB zip download stalled and was deleted. Only the composite sheet is used. |
| `antifarea-twelve/` | `charsets_12_m-f_complete_by_antifarea_0.png` (5d05afffe5484523), 330×400, mode P | The two OGA files are byte-identical, so one was removed. |
| `bit-era-extended/` | `biteracutecharacter20variations_byimogiagames.zip` (cf574691c760b830) → 20 × 64×128 PNG | All 20 have identical alpha masks (MEASURED). Only the colours differ. |
| `calciumtrice/` | `rogue_spritesheet.png` (0522889a…), `warrior_spritesheet.png` (d316475a…), `wizard_spritesheet.png` (217ae8e3…), each 320×320 | |

**Not downloaded:** LuizMelo. itch.io serves the zip only through a scripted "download_url" step, and the agent's request for it was blocked by the permission system. Hoopakid can download `Martial Hero.zip` (32 kB) by hand into `data/raw/ooS/luizmelo-martial-hero/`, then measure it with `measure.py`.

---

## 3. Per-pack facts, slicing spec and mapping

Common rule for all packs: **reference = the standing pose in the same direction**. This matches `dataset.make_example`, which uses walk frame 0 of that direction. Where a pack has an explicit idle, I use idle frame 0 instead. Where it does not, I use the walk frame that is pixel-identical to the idle or neutral pose.

### 3.1 Puny Characters (CC0): the main "near OOS" set

MEASURED
- Each sheet is 768×256 = 24 columns × 8 rows of 32×32. Alpha is 0/255 only, with no partial alpha.
- Colours per sheet: Base 32, Mage 40, Soldier 42, Warrior 43, Archer 44.
- Idle frame 0 bbox (Base, down) is x 9–23, y 10–23. Bottom opaque row is y = 22 in every row and sheet. Content is about 13–15 px tall.
- Rows are 8 directions. The page GIF (screenshot-4, 4 frames, 100 ms) was template-matched against Mage-Cyan. Its on-screen layout confirms the rows as: r0 down, r1 down-right, r2 right, r3 up-right, r4 up, r5 up-left, r6 left, r7 down-left.
- Left and right are separately drawn. Only 0–3 of 24 frames per row pair are exact mirrors.
- Walk cycle, from the same GIF: frames = [c0 (stand), c2, c0, c3]. So c2 and c3 are the two step poses, and c0/c1 are idle frames that the walk shares.

VISUAL (column groups; not stated by the author)
- c0–1 idle, c2–3 walk steps, c4–7 sword, c8–11 bow, c12–14 stave, c15–18 throw.
- c19–23 hurt + death: c19 is a white flash, c20 a hurt face, c21 standing again, c22–23 lying down.
- The hurt/death boundary (c19–20 vs c19–21) is ambiguous. Treat c19–23 as one "hurt→death" sequence if it is used at all.

```json
{"pack":"puny","cell":[32,32],
 "sheets":["Archer-Green","Archer-Purple","Character-Base","Mage-Cyan","Mage-Red","Soldier-Blue","Soldier-Red","Soldier-Yellow","Warrior-Blue","Warrior-Red"],
 "dir_rows":{"down":0,"right":2,"up":4,"left":6},
 "dir_rows_diagonal_unused":{"down_right":1,"up_right":3,"up_left":5,"down_left":7},
 "frame_rect":"[32*col, 32*row, 32, 32]",
 "reference":{"col":0},
 "actions":{
   "idle":  {"cols":[0,1]},
   "walk":  {"cols":[0,2,0,3]},
   "slash": {"cols":[4,5,6,7],   "source":"Sword Attack"},
   "shoot": {"cols":[8,9,10,11], "source":"Bow Attack"},
   "stave": {"cols":[12,13,14],  "source":"Stave Attack", "moviepix":null},
   "throw": {"cols":[15,16,17,18],"source":"Throw", "moviepix":null},
   "hurt":  {"cols":[19,20,21,22,23], "source":"Hurt+Death (boundary unverified)", "dirs":["down"]}
 }}
```

Mapping table

| Puny | moviepix | Frames (Puny vs LPC) | Note |
|---|---|---|---|
| idle c0–1 | idle | 2 vs 2 | exact count |
| walk [0,2,0,3] | walk | 4 vs 9 (LPC 8 + stand) | big timing mismatch; compare per phase (see §5) |
| sword c4–7 | slash | 4 vs 6 | |
| bow c8–11 | shoot | 4 vs 13 | |
| stave c12–14 | (none; closest halfslash/slash) | 3 | overhead swing; evaluate as "unseen action" only |
| throw c15–18 | (none) | 4 | exclude |
| hurt+death c19–23 | hurt (down only) | 5 vs 6 | LPC hurt is a collapse, so it matches the death end |

### 3.2 Ninja Adventure, Superpowers copy (CC0): the "far OOS, tiny big-head" set

MEASURED
- 23 sheets of 64×112 = 4 columns × 7 rows of 16×16. Two sheets (`4.png`, `8.png`) are 64×32 with only 2 rows. `dog.png` is 32×16 (not humanoid). Alpha is 0/255.
- Colours: 4–12 per character.
- Content fills the cell: the bottom opaque row is y = 15 in every walk frame, and top rows start at y = 0–2. There is no margin, so sprites can touch the neighbouring cell.
- Columns are directions: c0 down, c1 up, c2 left, c3 right (VISUAL, from the eye positions in `_previews/ninja1_top.png`).
- c3 is an exact mirror of c2 in all 4 walk rows for 10 of 23 characters (1, 2, 3, 5, 6, 7, 13, 14, 17, 19). The others are partly or fully hand-drawn.
- Walk = rows 0–3. Row 0 equals row 2 exactly (a repeated neutral pose) in all directions for characters 1, 5 and 7, and partly for others. Down rows 1 and 3 are mirror images (left step / right step).

VISUAL
- Row 0 is the standing/neutral pose. Row 4 is attack (one frame, arm/weapon forward). Row 5 is jump (one frame).
- Row 6 holds miscellaneous one-off poses (lying, item-lift, etc.), not per direction. Ignore it.

```json
{"pack":"ninja_superpowers","cell":[16,16],
 "sheets":"1..25 except 4,8 (64x112 only); dog excluded",
 "dir_cols":{"down":0,"up":1,"left":2,"right":3},
 "frame_rect":"[16*col, 16*row, 16, 16]",
 "reference":{"row":0},
 "actions":{
   "walk":{"rows":[0,1,2,3]},
   "slash_pose":{"rows":[4], "note":"single frame; pose-only check"},
   "jump_pose":{"rows":[5], "note":"single frame"}
 },
 "mirror_flag":"right==mirror(left) for chars 1,2,3,5,6,7,13,14,17,19 -> count right as non-independent"}
```

| Ninja | moviepix | Frames | Note |
|---|---|---|---|
| rows 0–3 | walk | 4 vs 9 | the only real sequence |
| row 4 | slash (pose only) | 1 vs 6 | compare with the LPC slash peak frame, or skip |
| row 5 | jump (pose only) | 1 vs 5 | |

Note: the current itch.io Ninja Adventure (CC0) is a larger, newer version. This older pinned copy is enough for testing and is fully reproducible.

### 3.3 Chasersgaming Knight NES, Ranger NES, Ranger GB (CC0): the "far OOS, 3–4 colours, 16×24" set

MEASURED (NES strips)
- Every `<Action>_<DIR>` strip is 64×24 = 4 frames of 16×24. Attack strips are 96×24 = 4 frames of **24×24**.
- Hit, Jump and SingleAttack files are single 16×24 frames per direction.
- Each strip has 3 opaque colours (attack has 5, including the weapon). Alpha is 0/255.
- WEST is never a pixel mirror of EAST: all four directions are drawn.
- Content is x 2–14 and y 0–24. The feet reach y = 22–23.
- Neutral-frame facts:
  - Knight: Idle frames 1 and 3 are identical. Walk SOUTH frame 0 = frame 2 = Idle frame 0.
  - Ranger: Walk SOUTH frame 0 = Idle SOUTH frame 0; Walk WEST frame 1 = Idle WEST frame 1; Walk EAST frame 0 = Idle EAST frame 2.
- So the walk "standing" frame index differs per direction. Use **Idle frame 0** as the reference.
- Packaging: `Hit PNG/` also contains a stray `Knight_Jump_SpriteSheet_4way.png`. The Knight `Walk BMP/` folder holds PNGs despite its name. Use the `* PNG/` folders only.

Ranger GB composite (352×280)
- The NES single-sheet silhouettes were found in the GB composite by alpha-mask match (`locate.py`).
- Walk block: x = 16 + 16·f, y = 120 (up), 144 (left), 168 (down), 192 (right), f = 0..3. Same frame order as the NES strips.
- Idle block: x = 16 + 16·f, y = 24 (up), 48 (left), 72 (down), 96 (right). The left/right idle sprites sit 1 px off the 16-px grid (matched at x = 15/31/47/63 and 17/33/49/65). Cropping on the grid is still safe because content is ≤ 12 px wide.
- 4 colours: (15,56,15), (48,98,48), (139,172,15), (155,188,15).
- The page-level NES composite does **not** byte-match the zip strips (an older revision), so never mix the two.

```json
{"pack":"chasers_knight_nes","root":"chasersgaming-knight-nes/x/NES Knight Free Version",
 "file":"{A} PNG/NES PNG Knight_{A}_{D}_strip4.png", "D":{"up":"NORTH","left":"WEST","down":"SOUTH","right":"EAST"},
 "frame_rect":"[fw*i, 0, fw, 24]", "fw":{"Idle":16,"Walk":16,"Run":16,"Push":16,"Climb":16,"Attack":24},
 "reference":{"A":"Idle","i":0},
 "actions":{"idle":{"A":"Idle","i":[0,1,2,3]},"walk":{"A":"Walk","i":[0,1,2,3]},
            "run":{"A":"Run","i":[0,1,2,3]},"slash":{"A":"Attack","i":[0,1,2,3]},
            "climb":{"A":"Climb","i":[0,1,2,3],"dirs":["up"]}}}
{"pack":"chasers_ranger_nes","root":"chasersgaming-ranger-nes/x/NES Ranger Files FREE/NES Ranger Free Set PNG/NES Ranger Single Sheets",
 "file":"{A}/NES_Ranger_{A}_{D}.png", "same D/fw/reference/actions as knight (no Push/Climb needed)"}
{"pack":"chasers_ranger_gb","file":"chasersgaming-ranger-gb/GB_Ranger_SpriteSheet_Free_0.png",
 "frame_rect":"[16+16*f, Y[dir], 16, 24]",
 "actions":{"walk":{"Y":{"up":120,"left":144,"down":168,"right":192},"f":[0,1,2,3]},
            "idle":{"Y":{"up":24,"left":48,"down":72,"right":96},"f":[0,1,2,3]}},
 "reference":{"action":"idle","f":0}}
```

| Chasers | moviepix | Frames | Note |
|---|---|---|---|
| Idle 4 | idle | 4 vs 2 | breathing loop |
| Walk 4 | walk | 4 vs 9 | |
| Run 4 | run | 4 vs 8 | |
| Attack 4 (24 px wide) | slash | 4 vs 6 | weapon extends outside 16 px |
| Climb 4 | climb (up only) | 4 vs 6 | |
| Push 4, Hit 1, Jump 1 | none / hurt / jump | | skip (Hit and Jump 4-way order not verified) |

### 3.4 Antifarea twelve 16×18 (CC-BY 3.0): walk-only, opaque background

MEASURED
- The image is 330×400, palette mode, and **fully opaque** (background colour (239,208,116)).
- Characters sit on solid colour-key blocks:
  - male block (123,213,254) at x 16–304, y 180–252;
  - female block (246,139,205) at x 16–304, y 306–378;
  - template block (127,127,127) at x 32–80, y 72–144.
- Each block is 6 characters × 3 columns × 4 rows of 16×18. Class order: warrior, magician, healer, ninja, ranger, townfolk (from the sheet labels).
- Rows: r0 up, r1 right, r2 down, r3 left (VISUAL).
- Middle frame c1 of left is an exact mirror of right c1 in all 12 characters. Right c0 = mirror(left c2) in 11 of 12. So left is effectively mirrored right.
- 9–17 colours per character (template 6). Content spans y 0–18 (touches both cell edges) and x 1–15.
- Colour-key conversion: within each block, pixels equal to the block colour become alpha 0. There are no anti-aliased edges.

```json
{"pack":"antifarea","file":"antifarea-twelve/charsets_12_m-f_complete_by_antifarea_0.png",
 "blocks":{"m":{"x0":16,"y0":180,"key":[123,213,254]},"f":{"x0":16,"y0":306,"key":[246,139,205]}},
 "classes":["warrior","magician","healer","ninja","ranger","townfolk"],
 "frame_rect":"[x0+16*(3*cls+col), y0+18*row, 16, 18]",
 "dir_rows":{"up":0,"right":1,"down":2,"left":3},
 "reference":{"col":1},
 "actions":{"walk":{"cols":[1,0,1,2]}},
 "mirror_flag":"left ~= mirror(right)"}
```

| Antifarea | moviepix | Frames | Note |
|---|---|---|---|
| 3-frame walk, played [1,0,1,2] | walk | 4-cycle vs 9 | VISUAL: c1 is the stand frame (RPG-Maker convention) |

### 3.5 Bit Era Extended (CC0): palette-invariance probe

MEASURED
- 20 files of 64×128 = 4 columns × 8 rows of 16×16. Rows 6–7 are empty, so in effect it is 6 rows × 4 columns. The page's "H=6, V=4" appears to name rows and columns the other way round.
- All 20 alpha masks are identical. Each file has exactly 3 colours.
- Rows (VISUAL, consistent with the page list):
  - r0 idle-side (right, 4 frames; c1 = c3);
  - r1 walk down (c0 = c2);
  - r2 walk right (c0 = c2 = r0c0 = r5c0, the neutral side pose);
  - r3 walk up (c0 = c2);
  - r4 jump c0, fall c1, death c2;
  - r5 attack right, 3 frames (c0 is the neutral pose).
- Content fills the 16 px height. There is no left facing; mirror right if needed (flag).

```json
{"pack":"bitera","files":"bit-era-extended/x/BitEraCuteCharacter-Extended-byImogiaGames-{00..19}.png",
 "frame_rect":"[16*col, 16*row, 16, 16]",
 "actions":{"walk":{"down":{"row":1,"cols":[0,1,2,3]},"right":{"row":2,"cols":[0,1,2,3]},"up":{"row":3,"cols":[0,1,2,3]}},
            "idle":{"right":{"row":0,"cols":[0,1,2,3]}},
            "slash":{"right":{"row":5,"cols":[0,1,2]}}},
 "reference":{"down":[1,0],"right":[2,0],"up":[3,0]},
 "left":"none (mirror of right only - exclude or mark synthetic)"}
```

| Bit Era | moviepix | Frames |
|---|---|---|
| walk r1–r3 | walk | 4 vs 9 |
| idle-side r0 | idle (right) | 4 vs 2 |
| attack r5 | slash (right) | 3 vs 6 |
| jump/fall | jump (pose only) | 1+1 vs 5 |

### 3.6 Calciumtrice (CC-BY 3.0): the side-view set

MEASURED
- Three sheets of 320×320 = 10×10 cells of 32×32. Every cell is non-empty. Alpha is 0/255.
- 15 (rogue), 22 (warrior) and 16 (wizard) colours per sheet.
- Content touches the cell bottom (y = 32) in almost every frame. The top reaches y = 0 in some frames, and x reaches 0 or 32 in attack frames, so content can touch the cell edge.
- Warrior idle repeats frames (c0 = c1 = c8 = c9, c4 = c5 = c6), so its idle is a 10-step timeline with holds.
- Rows (VISUAL, matches the page list in order): r0 idle, r1 gesture, r2 walk, r3 attack, r4 death (male); r5–r9 the same for the female version. Single facing: **right**, in a 3/4 view.

```json
{"pack":"calciumtrice","files":["calciumtrice/rogue_spritesheet.png","calciumtrice/warrior_spritesheet.png","calciumtrice/wizard_spritesheet.png"],
 "frame_rect":"[32*col, 32*row, 32, 32]",
 "variants":{"m":0,"f":5},
 "actions":{"idle":{"row":0,"cols":"0..9"},"walk":{"row":2,"cols":"0..9"},"slash":{"row":3,"cols":"0..9","note":"wizard row 3 = cast/projectile -> spellcast"},
            "gesture":{"row":1,"moviepix":"emote"},"death":{"row":4,"moviepix":"hurt (right, not down)"}},
 "rows_add_variant_offset":true,
 "direction":"right only",
 "reference":{"row":0,"col":0}}
```

| Calciumtrice | moviepix | Frames | Note |
|---|---|---|---|
| walk r2 | walk (right) | 10 vs 9 | closest count of all packs |
| idle r0 | idle (right) | 10 vs 2 | |
| attack r3 | slash (rogue, warrior) / spellcast (wizard) | 10 vs 6 / 7 | VISUAL: wizard attack shows a projectile |
| gesture r1 | emote | 10 vs 3 | |
| death r4 | hurt | 10 vs 6 | LPC hurt is down-facing only |

### 3.7 LuizMelo Martial Hero (CC0): not measured

The licence was verified today, and the page lists its frame counts. No files have been inspected. Do not write a slicing spec until the zip has been downloaded by hand and measured.

---

## 4. Placement into moviepix canvases (32 or 64)

Rule: never resample except by an integer nearest-neighbour factor k. Pivot = feet. Place the sprite so its source cell's bottom-centre lands on the LPC feet point:
- **64 canvas:** feet row y = 61, centre x = 32 (MEASURED from the LPC body).
- **32 canvas:** feet row y = 30, centre x = 16 (LPC halved; VERIFY against the D=32 pipeline).

Use **one offset per character, computed from its reference frame**: align the reference's bottom opaque row and its source-cell centre. Apply that same offset to every frame of every action. Never recentre per frame, because that destroys bob, lunge and recoil motion.

| Pack | Native content height | k for 64 (height vs LPC ≈30) | k for 32 | Fits? |
|---|---|---|---|---|
| Puny | ~14 px in a 32 cell | ×2 → 28 px; whole 32-cell ×2 = 64, so the cell maps 1:1 onto the canvas. The last feet row becomes y = 2·23 − 1 = 45, so shift +16 to reach y = 61. The head then sits about y 36; nothing clips because the top content is at y ≥ 2·8. | ×1 | yes |
| Ninja | 16 | ×2 → 32 | ×1 (fills 16 of 32) | yes |
| Chasers | 23–24 | ×1 → 24 (small) or ×2 → 48 (large). Recommend ×1 for "size shift" and ×2 for "scale-matched" | ×1 | yes; attack 24 px wide ×2 = 48 fits |
| Antifarea | 18 | ×2 → 36 | ×1 | yes |
| Bit Era | 16 | ×2 → 32 | ×1 | yes |
| Calciumtrice | up to 32 | ×1 → ~31 | does **not** fit: content touches the top, bottom and sides of the 32 cell, and the feet must sit at y = 30. Use 64 only. | 64 only |
| LuizMelo | unknown (wave 1: about 160 px canvas) | would need downscaling, which is not integer-safe | — | flag |

Store `{pack, char, k, offset_xy}` in the manifest, so predictions can be mapped back to source pixels for comparison at native resolution as well.

---

## 5. Problems and flags

1. **Opaque background:** Antifarea (colour-key blocks; also the 330×400 sheet carries text and a template). Every other pack has a clean 0/255 alpha.
2. **Mirrored-only directions:** Bit Era has no left; Calciumtrice has right only; Antifarea's left is about a mirror of right; Ninja has left = mirror for 10 of 23 characters. Count mirrored directions as non-independent: report them separately or drop them.
3. **Frame-count mismatch everywhere.** moviepix walk is 9 frames (frame 0 = stand); these packs have 3–4-frame cycles (Calciumtrice has 10). Recommendation: evaluate walk by **phase-aligned matching**. Resample the 8 moving LPC-length predictions to the pack's cycle (pick predicted frames at phases i/n). Alternatively, score the best cyclic alignment and report frame-agnostic metrics too: silhouette IoU against the nearest GT frame, palette fidelity, and identity preservation vs the reference.
4. **Single-frame actions** (Ninja attack/jump, Chasers Hit/Jump): pose-only probes, not sequence tests.
5. **No explicit standing pose:** Ninja and Antifarea have no idle row; the reference is the neutral walk frame (Ninja row 0, Antifarea c1; VISUAL). For Chasers, the walk-neutral index differs per direction, so use Idle frame 0.
6. **Layout inconsistencies:**
   - Chasers' page composite differs from the zip strips.
   - Knight's `Hit PNG/` holds a Jump sheet; the `Walk BMP/` folder holds PNGs.
   - Ranger GB idle side frames sit 1 px off the grid.
   - Ninja 4.png and 8.png have only 2 rows.
   - Bit Era's page says H=6/V=4, but the file is 4 columns × 8 rows (2 rows empty).
7. **Edge contact / clipping:** Ninja, Antifarea, Bit Era and Calciumtrice content touches the cell edges. Crop exactly on the grid; never crop by bbox.
8. **Unverified semantics:** Puny hurt/death boundary and stave/throw meaning; Ninja rows 4–6; Antifarea and Calciumtrice direction and row labels are VISUAL. The column/row geometry is measured.
9. **Puny 8 directions:** only the 4 cardinal rows map to moviepix. The diagonals (r1, r3, r5, r7) can serve a future 8-direction test.
10. **Not fetched:** LuizMelo (manual download needed), Chasers Knight GB, Puny orc pack.

---

## 6. Recommended test-set composition

Hold all of these out of training entirely, including the CC0 packs, which are tempting as training data. If any pack is later used for training, move it out of this list.

**Near OOS** (top-down, 4 directions, walk plus attacks, chibi but LPC-like semantics). Primary metric set.

| Pack | Characters | Dirs | Actions | Approx. sequences |
|---|---|---|---|---|
| Puny (one colour variant per design: Base, Archer-Green, Mage-Cyan, Soldier-Blue, Warrior-Blue) | 5 | 4 | idle, walk, slash, shoot (+ hurt down) | 5 × 4 × 4 + 5 = 85 |
| Antifarea | 12 | 4 (left mirrored) | walk | 48 |
| Chasers Knight NES + Ranger NES | 2 | 4 | idle, walk, run, slash, (climb up) | 2 × (16 + 1) = 34 |

**Far OOS** (stronger style, palette or proportion shift).

| Pack | Characters | Dirs | Actions | Why |
|---|---|---|---|---|
| Ninja Adventure | 23 | 4 (mirror-flagged) | walk (+ pose probes) | 16 px big-head, 4–12 colours |
| Chasers Ranger GB | 1 | 4 | idle, walk | 4-shade GB palette; pairs with the NES Ranger for a palette-only comparison |
| Bit Era | 1 design × 20 palettes (report as 1 character, 20-palette spread) | down, right, up | walk, idle-right, slash-right | 3 colours; palette-invariance spread |
| Calciumtrice | 6 | right only | walk, idle, slash/spellcast, emote, hurt | side/3/4 view at 32 px: tests the view shift (64 canvas only) |
| LuizMelo Martial Hero (after manual download and measurement) | 1 | right | run, slash, hurt | large side-view; scale stress |

Extra controls
- **Palette-only pairs:** NES vs GB Ranger, Puny colour variants, the 20 Bit Era recolours. Prediction quality should not change across these.
- **Scale control:** run Chasers at k = 1 and k = 2 to separate the effects of "style" and "size".
- Report every number per pack and per direction, flagging mirrored directions. Do not pool near and far OOS.

Next step: implement `scripts/build_oos_manifest.py`. It should read these specs, apply the colour key and placement, write `{reference, frames[], action, direction, pack, char, k, offset, mirrored}` crops plus a `CREDITS.csv` (Antifarea with an OGA link; Calciumtrice), and compare predictions at native pixel scale.
