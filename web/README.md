# moviepix web demo (Phase 3, local only)

A static page: drop in one standing sprite, pick an action and direction, get the animation as a
looping preview, a PNG sheet and a GIF. The model runs in the browser with ONNX Runtime Web; nothing
is uploaded. Design and results: `docs/research/phase3-design.md`.

## Run it on this Mac

```sh
# 1. model files into web/model/ (git-ignored): model.onnx, model-fp16.onnx, model-int8.onnx,
#    actions.json, selftest.json. CPU only, about 3 minutes.
uv run --extra export moviepix-export data/runs/20261124-1816-c32v3_gentle --out-dir web/model --verify 256
uv run --extra export python -m moviepix.export.quantize data/runs/20261124-1816-c32v3_gentle --dir web/model

# 2. the pinned ONNX Runtime Web files into web/vendor/ (git-ignored, ~44 MB, sha512-checked)
uv run python -I scripts/fetch_ort_web.py

# 3. serve on 127.0.0.1 only (adds the headers that allow several CPU threads)
python3 web/serve.py          # then open http://127.0.0.1:8000
```

## Files

| File | What it does |
|---|---|
| `index.html` | the page and its styles |
| `app.js` | loads the model, runs the self-test, animates, downloads |
| `place.js` | puts the sprite on the 32 px canvas exactly like `data/packs.place_reference` |
| `gif.js` | a small GIF writer (one palette, LZW, loops) |
| `serve.py` | local server, 127.0.0.1 only |
| `place.test.mjs`, `gif.test.mjs` | checks run by `tests/test_web.py` (needs `node`) |

On load the page runs a self-test: one reference whose frames native onnxruntime computed for each model
file (`selftest.json`). The page uses `model-int8.onnx` on WASM and `model-fp16.onnx` on WebGPU, falls back
to `model.onnx` if a file fails, and turns WebGPU off if no file passes there.

`http://127.0.0.1:8000/#autotest` runs every action on the example and writes the results as JSON into the
page (used for the headless Chrome check).

Not published yet: hosting the model is a release (Phase 5 licence and memorisation checks). The example
character and the self-test reference are LPC art from the validation split.
