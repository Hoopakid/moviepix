# Wave 1, track 4: running moviepix locally (export, browser, speed, size)

Research notes, 2026-11-18. Scope: how to ship the `copy_animator` model (about 4.5M params,
32/64 px, 2–13 frames) as an offline browser demo, a CLI and editor plugins within the targets
in [`docs/local-first.md`](../local-first.md).

Labels: **[verified]** means checked against the cited page, or against the locally installed
package where marked "local". **[unverified]** means it comes from a secondary source, a search
snippet or my own reasoning and still needs a test. **[estimate]** marks arithmetic of mine;
none of those numbers were measured.

Local environment (checked in `.venv`): torch 2.15.0, onnx 1.23.2, onnxruntime 1.32.0.
**onnxscript is not installed**, and the default exporter needs it (see §1.1). Latest
onnxruntime-web on npm: 1.31.1 (local check of `registry.npmjs.org/onnxruntime-web/latest`).

---

## 0. What in the current code matters for export

From `src/moviepix/models/copy_animator.py` and `blocks.py`:

| Code | Why export cares |
|---|---|
| `encode_reference`: `k = max(int(opaque.sum(1).max()), 1)` then `argsort(..., stable=True)[:, :k]` and `gather` | `int(...)` on a tensor value is a data-dependent size. torch.export raises `GuardOnDataDependentSymNode` unless it gets `torch._check` hints ([PyTorch guide](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/dynamic_shapes_troubleshooting_guardon_errors.html)). The legacy tracer would bake K in as a constant. |
| `copy_logits`: `einsum("npd,nkd->npk")` + `masked_fill(-inf)` | This is a batched MatMul. Einsum is in ORT-web's WebGPU list ([op table](https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/docs/webgpu-operators.md)). The output size is the memory risk (§5.1). |
| `decode`/`_choose`: Python loops over chunks of 32 or `loss_chunk` frames | A tracer unrolls these for the example N, so they must be rewritten for export. |
| `FrameAttention`: `padded[b_idx, t_idx] = x` (index_put → ScatterND), `nn.MultiheadAttention` with `key_padding_mask` | With batch=1 and all frames present at inference, padding and masking are no-ops. An export-only path can drop them. |
| `nn.GroupNorm` everywhere | Op coverage of GroupNormalization is patchy (§2.3). |
| `_choose`: `softmax`, `argmax`, `multinomial`, `gather`; `generate`: `argsort().argsort()` ranks, a Python loop over rounds | Keep sampling and the round loop **outside** the graph, in JS or Rust. |
| `self.pos` is `[dim, size, size]` | Each resolution is its own export (32 and 64 are separate checkpoints anyway). |

Config: `dim=64`, `channels=(32,64,128,128)`, `ref_channels=(32,64,128)`, `cond_dim=128`.

---

## 1. Exporting to ONNX

### 1.1 Which exporter

- Since PyTorch 2.9, `torch.onnx.export(..., dynamo=True)` is the default. `dynamo=False` keeps the old TorchScript exporter, and `torch.onnx.dynamo_export` was removed ([2.9 release notes](https://newreleases.io/project/pypi/torch/release/2.9.0), [2.9 docs](https://docs.pytorch.org/docs/2.9/onnx_export.html)). [verified]
- On `main`, `dynamic_shapes` is the preferred way to declare dynamic dimensions, and `dynamic_axes` is deprecated for the dynamo path. `fallback=` was removed in 2.11. `verify=True` checks the export with ORT. `report=True` writes a markdown report on failure. `optimize` defaults to True. **`external_data` defaults to True**, which writes the weights to a separate file ([export docs](https://docs.pytorch.org/docs/main/onnx_export.html)). [verified]
  - For the browser, pass `external_data=False` so the weights stay in one `.onnx` file. At about 18 MB fp32 the 2 GB protobuf limit is far away.
- Local torch 2.15: default opset is **20** (`ONNX_DEFAULT_OPSET = 20`, max 23). The TorchScript exporter still ships, capped at opset 20 (`torch/onnx/_constants.py`). The dynamo exporter does `import onnxscript` (`_internal/exporter/_core.py`). [verified, local]
  → **Add `onnxscript` to the `export` extra in `pyproject.toml`**. It is missing today.
- Recommendation: use the dynamo exporter with an explicit `opset_version` (18–20), `dynamic_shapes`, `external_data=False` and `verify=True`. If one op is stubborn, try `dynamo=False` as a cross-check. Both still exist in 2.15.

### 1.2 Data-dependent shapes: move them out of the graph

The simplest robust design keeps every data-dependent decision in the host language (JS or Rust). The ONNX graphs then contain only *symbolic* dimensions, never *data-dependent* ones:

1. **Opaque-pixel selection (K).** Export the reference encoder as `ref_unet + to_key + pos` with output `keys_full [1, S*S, 64]`. The host computes `opaque = alpha > 0.5` and the index list, then gathers `keys [K, 64]` and `colors [K, 3]`. That is about 4096 comparisons in JS, which is trivial. The decoder graph then takes `keys` with a symbolic dimension `K`. This avoids `int(tensor)`, `argsort(stable=True)`, the slice by a data value and every `torch._check`.
   - Alternative that stays fully in-graph and static: keep all S*S keys and mask the transparent ones with `-inf`. This needs no gather, but the copy head does about 4096/1100 ≈ 3.7× more work [estimate].
   - Do not rely on `argsort(stable=True)` surviving export. PyTorch `argsort` historically lacked an ONNX mapping, and the usual workaround is `sort` → TopK ([pytorch#33412](https://github.com/pytorch/pytorch/issues/33412)). [verified issue exists; current status unverified] ONNX TopK breaks ties by index, so the lower index comes first ([TopK spec](https://onnx.ai/onnx/operators/onnx__TopK.html)), which acts like a stable sort. [verified] TopK is **not** in the ORT-web WebGPU op table (§2.3).
2. **Frame count N.** Use a symbolic dimension `frames` (2–13) for the decoder inputs (canvas, known, conditioning ids). For a batch of one sprite, FrameAttention becomes self-attention over the `frames` axis per spatial position: `[h*w, frames, C]`. No `index_put`, no `key_padding_mask`.
3. **Iterative decoding.** Export **one decoding step**: `(ref image, keys, canvas, known, cond ids) → logits or choices`. Run the round loop, the ranking (`argsort().argsort()` → "top n_known per frame") and any sampling in the host. A typed-array sort over 4096 values per frame is cheap in JS [unverified, not measured]. An ONNX `Loop` is not in the WebGPU op table and is not supported by Unity's runtime (§4.4), so it buys nothing.
4. **Conditioning embeddings.** `nn.Embedding` exports as Gather, which is supported everywhere. Alternatively, precompute FiLM vectors per (action, direction, frame) on the host from a small JSON table.

### 1.3 nn.MultiheadAttention

- In eval/no_grad mode with `batch_first=True`, PyTorch can take the "fast path" `aten::_native_multi_head_attention`, which older exporters could not convert ([forum](https://discuss.pytorch.org/t/multiheadattention-export-to-onnx-fails-when-using-torch-no-grad/198843)). PyTorch PR #106824 disabled the fast path while tracing ([commit mirror](https://app.semanticdiff.com/gh/strisunshinewentingwang/pytorch/commit/0b57581dec068d468f90470e5739ab35d6c8753a)). [unverified for 2.15; test]
- Safest option: an export-only `FrameAttention` written as explicit `q/k/v` Linear + `softmax(q kᵀ/√d) v` (or `F.scaled_dot_product_attention`), loading the same `in_proj_weight/out_proj` tensors. The sequence length is only the frame count (≤13), so this costs nothing.

### 1.4 GroupNorm

- The legacy exporter historically decomposes GroupNorm into Reshape → InstanceNormalization → Reshape → Mul/Add. ORT's WebGPU work notes "each InstanceNormalization is surrounded with two reshape ops" in the SD VAE decoder ([ORT commit](https://app.semanticdiff.com/gh/microsoft/onnxruntime/commit/8f7b89bd5bbfce6983dbd1935e7073bad7701921)). [verified the comment; that it comes from the GroupNorm export is my inference]
- ONNX `GroupNormalization` exists (opset 18, deprecated; opset 21 changed scale/bias to shape [C]) ([spec](https://onnx.ai/onnx/operators/onnx__GroupNormalization.html), [18 vs 21](https://onnx.ai/onnx/operators/text_diff_GroupNormalization_18_21.html)). ORT reverted the deprecation of opset 18 ([ORT PR #24449](https://app.semanticdiff.com/gh/microsoft/onnxruntime/pull/24449/overview)). [verified via mirror]
- I could not see what onnxscript's torchlib emits for `aten.native_group_norm` (the GitHub fetch failed from this environment). [unverified]
- **Action:** after export, count the ops (`collections.Counter(n.op_type for n in model.graph.node)`). If `GroupNormalization` appears and WebGPU, Unity or tract are targets, either use an export-only GroupNorm (reshape + `mean`/`var` or InstanceNorm) or rewrite the graph.

### 1.5 Known pitfalls checklist

- Call `model.eval()` and export under `torch.no_grad()`, but watch the MHA fast path (§1.3).
- Python loops over N (`for i in range(0, len(b), 32)`) unroll at trace time. Write a vectorized export path.
- `float("-inf")` in `masked_fill` is fine for fp32. For fp16 conversion, `-inf` and very large constants get clipped (`max_finite_val=1e4` by default in the converter) ([ORT fp16 docs](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)). [verified] Keep the copy head in fp32.
- Verify numerically: compare PyTorch against ORT CPU on the validation set, using **pixel_exact agreement of the argmax**, not only `allclose` on logits.

---

## 2. ONNX Runtime Web

### 2.1 Backends

From the [onnxruntime-web 1.31.1 README](https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/README.md) [verified]:
- **WASM (CPU):** supports all `ai.onnx` and `ai.onnx.ml` operators. This is the safe target.
- **WebGPU:** marked "still an experimental feature". The README's table still says not on Safari, Firefox or iOS. That is out of date for browsers: web.dev lists WebGPU in Chrome/Edge 144, Firefox 141 (Windows first) and Safari 26 (macOS/iOS/iPadOS 26) ([web.dev](https://web.dev/blog/webgpu-supported-major-browsers)). I found no source confirming that ORT-web's WebGPU EP works well on Safari 26. [unverified]
- **WebGL:** maintenance mode. v1.29.0 announced the deprecation of WebGL **and JSEP**. The native C++ WebGPU EP (Dawn-based) is the recommended path ([v1.29.0 notes](https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.29.0)). [verified]
- **WebNN:** experimental and behind a flag in ORT-web. Chrome's WebNN origin trial started in 146, was paused for release-blocking bugs, and had no new target milestone in the latest thread I found ([blink-dev](https://groups.google.com/a/chromium.org/g/blink-dev/c/5CWKSChYo98)). [verified snippet] Not a target for now.

### 2.2 Threads, SIMD and headers

- `env.wasm.numThreads` defaults to 0 (auto). Multi-threading works "only when the browser supports WebAssembly multi-threading and `crossOriginIsolated` mode is enabled". The proxy worker moves inference off the main thread (better for UI, not faster) and **cannot be used with WebGPU** ([env flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html), [perf diagnosis](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html)). [verified]
- Cross-origin isolation needs `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` (or `credentialless`) ([web.dev guide](https://web.dev/articles/cross-origin-isolation-guide)). Without them ORT falls back to a single thread, so latency is roughly ×(threads) worse on a 4-core machine [unverified, estimate].
- You can run ORT yourself inside a Web Worker instead of using the proxy option, which works with any EP. [unverified, standard practice]

### 2.3 Operator coverage (WebGPU EP)

From [webgpu-operators.md @1.31.1](https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/docs/webgpu-operators.md) [verified]:
- **Present:** Conv (notes "need perf optimization"), ConvTranspose, InstanceNormalization, LayerNormalization, Einsum, MatMul, Gemm, Gather, GatherElements, GatherND, ScatterND, ArgMax, Softmax, Where, Expand, Resize, Sigmoid, CumSum, Range, If, Cast, Equal/Greater/Less, Not, Tile, Attention / MultiHeadAttention (com.microsoft; "need implementing mask"), ReduceLogSumExp, ReduceMax.
- **Absent:** **GroupNormalization, TopK, NonZero, Loop, ScatterElements, And, Mod, OneHot**.
- It is unclear whether this table describes JSEP or the new native WebGPU EP (the README path); the native EP's coverage may differ. [unverified]
- Unsupported nodes are assigned to the CPU EP with the warning "Some nodes were not assigned to the preferred execution providers" ([HF discussion](https://huggingface.co/Xenova/stablelm-2-zephyr-1_6b/discussions/1)). Each such node costs a GPU↔CPU copy. Check with `ort.env.logLevel='verbose'` and `ort.env.debug=true` ([perf diagnosis](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html)).
- With the design in §1.2 (no TopK, no NonZero, GroupNorm decomposed), every op in the step graph appears on the list.

### 2.4 WebGPU vs WASM for a model this small

- ORT's own guidance: WASM is for "very small models or environments where GPU is not available" ([perf diagnosis](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html)). [verified]
- A Mozilla bug thread reports WebGPU overtakes WASM only as inputs grow, that multi-threaded WASM is comparable, and that WebGPU was slow with int8 there ([bugzilla 1939943](https://bugzilla.mozilla.org/show_bug.cgi?id=1939943)). [snippet, unverified in detail]
- Our step graph is about 10 GFLOP per call at 64 px / 8 frames [estimate, §5]. That is big enough that WebGPU may win, so measure. Graph capture needs static shapes and all kernels on WebGPU ([env flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)). It could suit a fixed (frames, K-bucket) shape. Keep outputs on GPU (`preferredOutputLocation: 'gpu-buffer'`) across rounds ([WebGPU tutorial](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)). [verified]
- fp16 on WebGPU needs Chrome 121+/Edge 122+ ([README](https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/README.md)). One report shows a model producing bad output with WebGPU+fp16 but correct with fp32 ([onnxruntime#26732](https://github.com/microsoft/onnxruntime/issues/26732)). [snippet]

### 2.5 Download size: the runtime is bigger than the model

Measured locally (jsDelivr file listing and `Content-Length` with `Accept-Encoding: br`, onnxruntime-web 1.31.1) [verified, local]:

| File | Raw | Brotli over the wire |
|---|---|---|
| `ort-wasm-simd-threaded.wasm` (CPU) | 14.8 MB | **3.2 MB** |
| `ort-wasm-simd-threaded.jspi.wasm` | 17.9 MB | 3.3 MB |
| `ort-wasm-simd-threaded.jsep.wasm` (JSEP WebGPU) | 29.5 MB | 4.8 MB |
| `ort-wasm-simd-threaded.asyncify.wasm` (likely native WebGPU EP build, [unverified]) | 28.7 MB | 6.0 MB |
| `ort.wasm.min.mjs` / `ort.webgpu.min.mjs` JS | 54 KB / 71 KB | – |

- Total for a WASM demo ≈ 3.2 MB runtime plus the model (≈9 MB fp16 weights or ≈4.5 MB int8; ONNX files compress somewhat). This fits the 20 MB budget. The WebGPU path adds about 3 MB.
- Custom "reduced operator" builds can shrink the wasm (`--include_ops_by_config`, `--minimal_build`) ([reduced op config](https://onnxruntime.ai/docs/reference/operators/reduced-operator-config-file.html), [custom build](https://onnxruntime.ai/docs/build/custom.html); web section "under construction" per [deploy docs](https://onnxruntime.ai/docs/tutorials/web/deploy.html)). Not worth it before v1.
- Cache the model in IndexedDB or the Cache API for offline use ([deploy docs](https://onnxruntime.ai/docs/tutorials/web/deploy.html)). Add a Service Worker for full offline/PWA use.
- **fp16 in WASM is slow**: "Avoid float16 models if possible, as float16 is not natively supported by CPU". ORT recommends uint8-quantized models for WASM ([perf diagnosis](https://onnxruntime.ai/docs/tutorials/web/performance-diagnosis.html)). [verified] So the ~9 MB fp16 file is a *download* format for WebGPU. For WASM, ship fp32 (~18 MB, over budget once the runtime is added) or int8 QDQ (~5 MB). Another option is fp16 storage with a cast to fp32 at load time (`convert_float_to_float16` puts Casts in the graph; whether ORT folds them back to fp32 at session init is [unverified]).

### 2.6 Alternatives

- **Transformers.js** runs onnxruntime-web underneath and passes `session_options` through ([docs](https://huggingface.co/docs/transformers.js/api/backends/onnx)). It adds Hub conventions but no value for a custom non-transformer model, so use ORT-web directly.
- **TensorFlow.js / LiteRT.js:** Google announced LiteRT.js (Aug 2026), widely described as TF.js's successor ([betterstack](https://betterstack.com/community/guides/ai/litert-js-browser-ai.md), [ai.google.dev](https://ai.google.dev/edge/litert/web)). Reaching it would mean PyTorch → TFLite conversion. Not recommended.
- **ExecuTorch WebGPU backend:** "experimental and under active development", no automatic fallback for unsupported ops, dynamic shapes via max-size buffers ([docs](https://docs.pytorch.org/executorch/main/backends/webgpu/webgpu-overview.html)). [verified] Worth watching. It would avoid ONNX entirely.
- **tinygrad WebGPU export:** examples are outdated ([sd-webgpu](https://github.com/softwiredtech/stable-diffusion-webgpu)). Not recommended.
- **Hand-written WebGPU/WGSL:** feasible for a model this size, but it is a maintenance burden. Not recommended for v1.

---

## 3. Quantization

- ORT recommends **static quantization for CNNs** and dynamic for RNN/transformers. QDQ S8S8 is the default and balances speed and accuracy. Per-channel helps when weight ranges are wide. `reduce_range` is for AVX2/AVX512 without VNNI ([ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)). [verified]
- `quantize_static` / `quantize_dynamic` accept `op_types_to_quantize`, `nodes_to_exclude` and `per_channel` (local `onnxruntime/quantization/quantize.py`, ORT 1.32.0). [verified, local] Run `quant_pre_process()` first, and use `qdq_loss_debug` to compare activations ([docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)).
- **Do not dynamic-quantize Conv:** it produces `ConvInteger`, which reportedly has no ORT CPU kernel (["Could not find an implementation for ConvInteger(10)"](https://huggingface.co/Xenova/tiny-random-WhisperForConditionalGeneration/discussions/1); [zenn write-up](https://zenn.dev/ux_xu/articles/ort-int8-ptq-cnn)). [snippet]
- Int8 is not reliably faster in WASM. Xenova's MiniLM benchmarks show mixed results ([HF discussion 50](https://huggingface.co/spaces/Xenova/webgpu-embedding-benchmark/discussions/50)). [snippet] For WebGPU, one report says int8 is "extremely slow" ([bugzilla](https://bugzilla.mozilla.org/show_bug.cgi?id=1939943)). Int8 is a **size** win first. Speed has to be measured.
- **Pointer/argmax risk (our specific case):** the output is an argmax over `q·k` scores, where many reference pixels are near-ties (same color, neighbouring positions). Quantization noise flips argmaxes. A flip to a same-color key is harmless, but a flip to a different-color key is a visible pixel error. Mitigations:
  1. Quantize only the Conv weights and activations of the U-Nets. Keep `to_query`, `to_key`, `pos`, the einsum, `special`/`alpha` heads and FrameAttention in fp32 (`nodes_to_exclude`).
  2. Calibrate on real sprites (a few hundred validation references and frames).
  3. Accept or reject on **pixel_exact / silhouette IoU deltas** from `moviepix-eval`, not on logit MSE. Suggested gate: ≤0.5 pt drop in pixel_exact [my suggestion].
- **fp16** via `onnxconverter-common`: use `keep_io_types=True` and `node_block_list` for the copy head ([ORT fp16](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)). The mixed-precision auto tool needs a GPU to validate. Useful for WebGPU only.

---

## 4. Native options

### 4.1 ONNX Runtime desktop (Python/C/C#/Rust)
- Same `.onnx` everywhere. The Python CLI can already use `onnxruntime` (in the `export` extra).
- Rust `ort` crate 2.0.0-rc.x: the `load-dynamic` feature loads `libonnxruntime` at runtime. A strict runtime minor-version check can panic, and on Windows the DLL must sit next to the binary because of old System32 copies ([ort docs.rs source](https://docs.rs/crate/ort/latest/source/src/lib.rs), [search summary](https://www.skillsdirectory.com/skills/bobmatnyc-ort)). [snippet] A "single binary" in practice means binary + dylib, or a static build.
- **tract** (Sonos, pure Rust, Apache-2.0/MIT): loads ONNX and NNEF and compiles to wasm32 ([repo](https://github.com/sonos/tract)). Its README historically reported ~85% of the ONNX backend tests passing, up to opset 18 [snippet; per-op coverage for Einsum/GroupNorm unverified]. A true single static binary with no dylib makes it attractive for the CLI if our ops are covered (the §1.2 graph only needs Conv, InstanceNorm/Reduce, MatMul/Einsum, Softmax, Gather, Where, Sigmoid).
- **candle / burn-onnx** are alternatives (burn has `burn-onnx` codegen with a GroupNorm node ([docs.rs](https://docs.rs/crate/burn-onnx/latest/source/src/burn/node/group_norm.rs))). [unverified quality]

### 4.2 Core ML (Apple)
- coremltools converts `torch.export` programs (beta since coremltools 8.0). A `torch.export.Dim` becomes a `RangeDim`. **The Neural Engine does not support true dynamic shapes**: with RangeDim, Core ML schedules on CPU/GPU. EnumeratedShapes are the ANE-friendly route ([coremltools model export](https://apple.github.io/coremltools/docs-guides/source/model-exporting.html), [ExecuTorch Core ML partitioner](https://docs.pytorch.org/executorch/main/backends/coreml/coreml-partitioner.html)). [snippet]
- For us: enumerate frames ∈ {2..13} (or pad to 4/8/13) and K buckets (pad K to 256/512/1024/2048/4096 with masked keys). This is Mac-only. Defer until after the web demo. Apple also publishes `coreai-torch` docs with a `group_norm` composite op ([page](https://apple.github.io/coreai-torch/main/api/composite-ops/group-norm.html)). Its relation to coremltools is unclear. [unverified]

### 4.3 Godot
- No mature ONNX Runtime GDExtension for Godot 4 found. There are ncnn-based RL plugins and the alpha iree.gd ([asset 5358](https://www.godotengine.org/asset-library/asset/5358), [iree.gd](https://github.com/RechieKho/iree.gd)). [snippet]
- Practical routes: (a) Godot .NET + the `Microsoft.ML.OnnxRuntime` NuGet package [unverified for Godot 4 export targets]; (b) GDScript `OS.execute` calling the moviepix CLI [unverified, standard Godot API]; (c) an editor-time plugin that writes sprite sheets, so the game ships no model at all. (c) is the most useful for artists.

### 4.4 Unity
- The Unity Inference Engine (formerly Sentis), `com.unity.ai.inference` 2.6: **GroupNormalization unsupported, Loop unsupported, Attention unsupported**. NonZero is CPU only. Einsum on GPUCompute is limited to 1–2 inputs. TopK, GatherElements, ScatterND, InstanceNorm, LayerNorm, ArgMax and Where are supported ([op list](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/supported-operators.html)). [verified]
- So the §1.2 graph with decomposed GroupNorm and explicit MatMul attention should load. That is one more reason to avoid `GroupNormalization` and `MultiHeadAttention` contrib ops.

### 4.5 Aseprite
- The Lua API has `os.execute` and `io.open`, which **ask the user for permission** the first time ([api-draft](https://github.com/aseprite/api-draft)). Reports on `io.popen` conflict, so the common pattern is to run the command, write output to a temp file and read it back ([forum](https://community.aseprite.org/t/is-io-popen-available-in-lua/21572)). [snippet]
- Aseprite ≥1.2.30/1.3-beta7 has a **WebSocket API** (client) ([release notes](https://www.aseprite.org/release-notes), [forum](https://community.aseprite.org/t/11466/1)). A local `moviepix serve` process could talk to a Lua extension over WebSocket. Simpler first version: `os.execute("moviepix animate in.png --action walk --out out.png")` and then load the sheet as new frames.

---

## 5. Model-side levers

### 5.1 Where time and memory go (rough arithmetic, not measured)

At 64 px, 8 frames, K≈1100 opaque pixels, D=64:
- Copy logits: 8 × 4096 × 1100 × 64 × 2 ≈ **4.6 GFLOP** per decoding round [estimate].
- Logits tensor: 8 × 4096 × 1100 × 4 B ≈ **144 MB** in fp32, and softmax creates another of the same size [estimate]. Against the 300 MB memory target this is the biggest risk in a one-graph export. In PyTorch, `_choose` chunks frames, which is why it is fine today.
- The U-Net cost was not estimated precisely here. Profile it (ORT `enableProfiling`, or `torch.profiler`) before optimizing anything.

### 5.2 Concrete, realistic levers (ordered by expected value / effort)

1. **Don't build the full softmax in the export.** One-shot output only needs `argmax(logits)`. Confidence for iterative rounds is `max_prob = exp(max - logsumexp)`, using `ReduceMax` + `ReduceLogSumExp`, both on the WebGPU list. In ONNX, Einsum → ArgMax still materializes the logits, so **split the graph**: (A) a "trunk" producing `q [N, P, 64]`, `special`, `alpha`, `new_rgb`; (B) a tiny "copy head" `(q_chunk, keys) → choice, conf` run per 1–2 frames by the host. Memory per frame ≈ 18 MB [estimate].
2. **Encode the reference once** per sprite, and reuse it across actions, directions and rounds (the code already does this per call; keep it a separate ONNX graph).
3. **Fewer rounds:** decisions 0016/0017 show one-shot is as good or better today. Ship 1 round by default and 2–4 as an option. 8 rounds at 64 px is ~890 ms native, so it is out of budget in WASM.
4. **Smaller D (64 → 32)** halves the copy-head FLOPs. Needs retraining; check pixel_exact.
5. **Reduce K:** drop interior keys? Risky. Copying by position (decision 0011) needs every source. A better option: **K-bucket padding** (256/512/1024/2048) for static-shape runtimes (Core ML ANE, WebGPU graph capture).
6. **Top-k sparse copy attention** does not save compute at inference, because you still compute all q·k to find the top-k. It only helps training memory. Not worth it.
7. **Depthwise-separable convs / narrower decoder** (e.g. channels (24,48,96,96)): standard MobileNet-style savings ([MobileNets](https://arxiv.org/abs/1704.04861)). Needs retraining. Depthwise convs are often memory-bound on CPU/WASM, so speedups are smaller than the FLOP savings [unverified].
8. **Distillation** from the current model into a narrower student ([Hinton et al.](https://arxiv.org/abs/1503.02531)): train on the teacher's logits over the copy distribution. Realistic but costs a training run.
9. **Pruning:** at 4.5M params, unstructured pruning gives no speedup on ORT CPU/WASM without sparse kernels [unverified]. Structured channel pruning ≈ "narrower decoder" anyway.

---

## 6. Free static hosting with cross-origin isolation

| Host | COOP/COEP | Notes |
|---|---|---|
| **Hugging Face Spaces (sdk: static)** | Yes: `custom_headers` in README YAML; only COEP/COOP/CORP allowed, lowercase ([config reference](https://huggingface.co/docs/hub/spaces-config-reference)) [verified] | Model can live in an HF model repo. Optional `app_build_command` for Vite builds. |
| **GitHub Pages** | No custom headers ([community #13309](https://github.com/orgs/community/discussions/13309)) | Use `coi-serviceworker` (injects headers via a Service Worker; first load reloads; reports of reload loops) ([guide](https://dev.to/stefnotch/enabling-coop-coep-without-touching-the-server-2d3n)). [snippet] Or ship single-threaded WASM. |
| **itch.io HTML5** | "SharedArrayBuffer support" checkbox under Embed options. Sets COOP same-origin and COEP require-corp on files, and `credentialless` on the embedding page ([itch.io changelog thread](https://itch.io/t/2025776/experimental-sharedarraybuffer-support)) [snippet] | Reports of browser-specific failures (older threads). Good for reaching game devs. |
| **Cloudflare Pages** | `_headers` file, up to 100 rules ([docs](https://developers.cloudflare.com/pages/configuration/headers/)) [verified] | Free tier limits not checked. |

With COEP `require-corp`, cross-origin assets (e.g. the ORT wasm on jsDelivr) must send CORP/CORS headers. Simplest: **self-host the ORT `.wasm` files** next to the page (`env.wasm.wasmPaths`) ([deploy docs](https://onnxruntime.ai/docs/tutorials/web/deploy.html)).

---

## 7. Latency expectations (to verify by measurement)

- Native PyTorch today (M5, 4 threads): 16 ms (32 px) and 117 ms (64 px) one-shot for 8 frames (decision 0016).
- WASM vs native: the "Not So Fast" study measured WASM 45–55% slower on average and up to ~2.5× slower at peak on SPEC ([USENIX ATC'19](https://www.usenix.org/conference/atc19/presentation/jangda)). Mozilla reports its native backend 2–10× faster than ORT-web WASM for its models (3.5 s → 350 ms on one image-to-text model) because WASM SIMD can't use NEON/AVX-512-specific paths ([Mozilla, Aug 2025](https://blog.mozilla.org/en/firefox/firefox-ai/speeding-up-firefox-local-ai-runtime/)). [verified]
- **My rough expectation [estimate]:** browser WASM with 4 threads ≈ 1.5–4× native ORT, so roughly 25–70 ms at 32 px and 180–500 ms at 64 px one-shot on an M-series Mac. Slower on an older 4-core x86 laptop. Single-threaded WASM (no COOP/COEP) is several times worse. **64 px within 200 ms in the browser is uncertain**. 32 px is safe. WebGPU might bring 64 px under 200 ms on decent GPUs. Measure.

---

## 8. Open questions

- What does onnxscript (current) emit for `native_group_norm`, `_native_multi_head_attention` and `argsort(stable)` under torch 2.15? (Answer: export a toy model and print the op census.)
- Is the asyncify wasm build the native WebGPU EP, and what op coverage does it have compared with the JSEP table?
- Does ORT-web WebGPU work on Safari 26 / iOS 26 for our graph?
- What is the U-Net vs copy-head time split (ORT profiling)?
- How large is the int8 pixel_exact drop with the copy head kept in fp32?
- tract coverage for our exact op set (would allow a pure-Rust, single-file CLI and an alternative wasm runtime).
