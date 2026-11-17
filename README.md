# moviepix

A tiny open model that animates pixel-art sprites. Give it one sprite and an
action (walk, run, slash, spellcast and more), and it draws the frames.

> Status: early. The project structure is in place; the dataset is being built.

## Plan

- Trained from scratch on composed [LPC](https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator) characters, using only permissively licensed art.
- 64x64 and 32x32 sprites, 15 actions, 4 directions.
- Small enough to run in the browser.

See [docs/architecture.md](docs/architecture.md) for how it fits together and
[docs/decisions](docs/decisions/README.md) for why.

## Setup

```bash
uv sync
# LPC art, used as data only
git clone --depth 1 https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator.git data/raw/lpc
```

## Tests

```bash
uv run pytest
```

## License

Apache-2.0 for code and weights. Datasets built from LPC art carry the art's
own licenses and credits (`CREDITS.csv` in each build).
