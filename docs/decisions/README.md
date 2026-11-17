# Decisions

One short note per decision that shapes the project. Add new ones with the next number.

| # | Decision | Why |
|---|---|---|
| 0001 | Input is **sprite -> animation**; text -> sprite comes later as a separate first stage | Artists keep their own art; a copy-based model can stay tiny; no open tiny model does this yet |
| 0002 | Train only on **permissively licensed** art (CC0, CC-BY, OGA-BY). Multi-licensed art qualifies through its permissive option | Weights can be Apache-2.0 without the share-alike question. Excludes the muscular and pregnant LPC bodies |
| 0003 | **Both 64x64 and 32x32**; 32 is derived from 64 renders | One source of truth; LPC is drawn at 64 |
| 0004 | **All LPC actions** (15), variable frame counts | Adding actions later only costs data |
| 0005 | **PyTorch with MPS**, export to ONNX | Simple path to a browser demo; same stack as tiny-voca |
| 0006 | Name: **moviepix** | Chosen by the maintainer |
| 0007 | Child body left out for now | It lacks 9 of 15 actions and has almost no permissively licensed clothing |
| 0008 | Datasets stored as one indexed-color PNG per character (all actions) | Lossless, ~100 KB per character instead of ~250 KB; max 65 colors seen |
| 0009 | 32x32 = majority color per 2x2 block, ties to the local average color | "Darkest wins" turned faces dark; this keeps fills and only uses original colors |
| 0010 | **Local-first** is a hard requirement; see [local-first.md](../local-first.md) | The project's point is AI that runs on ordinary computers |
| 0011 | Main model copies **by position** with **global** attention over the sprite's opaque pixels | Patterns move together; handles big moves (weapon swings); keys only over opaque pixels keeps it cheap |
| 0012 | **Iterative (MaskGIT-style) decoding**, trained with random masking so one model also does one-shot (`steps=1`) | Coherent results when motion is uncertain; enables variations, editing and the later text -> sprite stage |
