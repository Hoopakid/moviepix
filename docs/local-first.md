# Local-first targets

moviepix exists to show that useful generative AI can run on ordinary
computers. These targets are requirements, checked by `moviepix-eval` and the
release checklist, not wishes.

## Targets for the released model

| | Target |
|---|---|
| Download size | under 20 MB (fp16), under 10 MB (int8) |
| Parameters | under 15M |
| Speed, 8-frame animation, 64x64 | under 200 ms on a 4-core laptop CPU, no GPU |
| Runs in a browser | ONNX Runtime Web (WASM), WebGPU when available |
| Memory at inference | under 300 MB |
| Network | none after download; no account, no API key |
| Training | reproducible on one consumer machine (Apple Silicon laptop or one gaming GPU), under $10 of rented GPU if any |
| Openness | code, weights, dataset, training recipe and costs all published |

## How each stage serves it

- **Data**: built on a laptop's CPU in under an hour.
- **Models**: architectures are chosen for size and CPU speed first (the copy-based
  head, palette indices). Every model reports parameters, size and CPU latency
  next to quality.
- **Export**: ONNX fp32, fp16 and int8; latency measured on CPU for each.
- **Distribution**: a static web page, a CLI, and plugins for tools artists
  already use (Aseprite, Godot) after the first release.

## Benchmark

Every release reports quality *and* cost (size, latency on CPU, memory) against
the open alternatives that exist at that time, measured on the same machine.
