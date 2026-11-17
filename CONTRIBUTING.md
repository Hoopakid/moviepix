# Contributing to moviepix

Thanks for helping. moviepix is early, so small, focused contributions are the
most useful.

## Good first contributions

- **Sprite sets with open licenses** (CC0, CC-BY, OGA-BY) in styles other than
  LPC, for training and for the out-of-style test set (milestone 4).
- **Test sprites in other styles**: your own art, shared under an open license.
- **CPU and browser speed**: profiling and optimizing `copy_animator` inference.
- **Bug reports** with the exact command and machine.

## Setup

```bash
uv sync --extra train
uv run pytest
```

Building data and training are described in the [README](README.md#quick-start).

## Ground rules

- **Licenses.** Art must be CC0, CC-BY or OGA-BY (multi-licensed art qualifies
  through its permissive option). Include credits. Do not add art taken from
  commercial games, even for testing.
- **One idea per pull request**, with tests where it makes sense.
- **Model or data changes come with numbers**: `moviepix-eval` or
  `moviepix-compare` output on the same split, before and after.
- **Big design changes**: open an issue first, and add a line to
  [docs/decisions](docs/decisions/README.md) with the reason.
- **Local-first**: changes should not push the model past the targets in
  [docs/local-first.md](docs/local-first.md).

## Code layout

See [docs/architecture.md](docs/architecture.md). New models are one file in
`src/moviepix/models/` plus a config in `configs/model/`; new art sources are a
folder in `src/moviepix/sources/`.

## Conduct

Be kind and assume good faith. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
