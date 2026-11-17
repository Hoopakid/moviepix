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
