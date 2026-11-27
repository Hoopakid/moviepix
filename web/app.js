// moviepix demo page: place a sprite, run model.onnx with ONNX Runtime Web, preview and download.
// Everything happens in this tab; the only files fetched are the page's own (model/, vendor/).
import { colourIds, placeReference, roundHalfEven, threshold } from "./place.js";
import { encodeGif } from "./gif.js";

const $ = (id) => document.getElementById(id);
const VENDOR = new URL("./vendor/onnxruntime-web/", location.href).href;
const MODEL_FOR = { wasm: "model-int8.onnx", webgpu: "model-fp16.onnx" };  // phase3-design.md, step 2
const FALLBACK = "model.onnx";
const SELFTEST_PASS = 0.995;  // share of pixels equal to native onnxruntime's frames
const FPS = 8;

let info, selftest;
const ortFor = {}, sessions = {};
const st = { action: "walk", dir: "down", backend: "wasm", zoom: 8, playing: true, frame: 0,
             frames: [], sprite: null, name: "sprite", busy: false };

// ------------------------------------------------------------ runtime
async function loadOrt(backend) {
  if (!ortFor[backend]) {
    const ort = await import(VENDOR + (backend === "webgpu" ? "ort.webgpu.min.mjs" : "ort.wasm.min.mjs"));
    ort.env.wasm.wasmPaths = VENDOR;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    ortFor[backend] = ort;
  }
  return ortFor[backend];
}

const bytes = (b64) => Uint8ClampedArray.from(atob(b64), (c) => c.charCodeAt(0));

// float [T, 4, S, S] in 0..1 -> T frames of RGBA bytes, rounded like export/onnx.py to_uint8 (float32, half to even)
function toFrames(data, T, S) {
  const P = S * S, out = [];
  for (let t = 0; t < T; t++) {
    const f = new Uint8ClampedArray(P * 4);
    for (let c = 0; c < 4; c++) for (let p = 0; p < P; p++) {
      const v = Math.min(Math.max(data[(t * 4 + c) * P + p], 0), 1);
      f[p * 4 + c] = roundHalfEven(Math.fround(Math.fround(v) * 255));
    }
    out.push(f);
  }
  return out;
}

async function infer(entry, canvas, actionName, dir) {
  const { ort, session } = entry, S = info.size, P = S * S;
  const act = info.actions.find((a) => a.name === actionName);
  const ref = new Float32Array(4 * P);
  for (let p = 0; p < P; p++) for (let c = 0; c < 4; c++) ref[c * P + p] = canvas[p * 4 + c] / 255;
  const { ids, count } = colourIds(canvas);
  if (count > info.max_colours) throw new Error(`This sprite has ${count} colours; the model takes at most ${info.max_colours}.`);
  const i64 = (xs) => BigInt64Array.from(xs, (x) => BigInt(x));
  const feeds = {
    reference: new ort.Tensor("float32", ref, [1, 4, S, S]),
    action: new ort.Tensor("int64", i64([act.id]), [1]),
    direction: new ort.Tensor("int64", i64([info.directions.indexOf(dir)]), [1]),
    frame_ids: new ort.Tensor("int64", i64(Array.from({ length: act.frames }, (_, i) => i)), [act.frames]),
    colour_id: new ort.Tensor("int64", ids, [P]),
  };
  const t0 = performance.now();
  const out = await session.run(feeds);
  const ms = performance.now() - t0;
  return { frames: toFrames(await out.frames.getData(), act.frames, S), ms, act };
}

async function runSelfTest(entry) {
  const want = bytes(selftest.expected_rgba[entry.file]);
  const { frames } = await infer(entry, bytes(selftest.reference_rgba), selftest.action, selftest.direction);
  let same = 0, n = 0, alphaDiff = 0, colourDiff = 0;
  frames.forEach((f, t) => {
    for (let p = 0; p < f.length; p += 4, n++) {
      const o = t * f.length + p;
      const eq = f[p] === want[o] && f[p + 1] === want[o + 1] && f[p + 2] === want[o + 2] && f[p + 3] === want[o + 3];
      same += eq;
      if (!eq) { if (f[p + 3] !== want[o + 3]) alphaDiff++; else colourDiff++; }
    }
  });
  entry.diff = { alpha: alphaDiff / n, colour: colourDiff / n };
  return same / n;
}

// The smaller file for this backend if it loads and passes the self-test, else the fp32 file.
async function getSession(backend) {
  if (sessions[backend]) return sessions[backend];
  const ort = await loadOrt(backend);
  const tries = [...new Set([MODEL_FOR[backend], FALLBACK])].filter((f) => selftest.expected_rgba[f]);
  let lastError = null;
  const attempts = [];
  for (const file of tries) {
    try {
      const session = await ort.InferenceSession.create(new URL(`./model/${file}`, location.href).href,
                                                        { executionProviders: [backend] });
      const entry = { ort, session, file, backend, attempts };
      entry.selftest = await runSelfTest(entry);
      attempts.push({ file, selftest: entry.selftest, diff: entry.diff });
      if (entry.selftest >= SELFTEST_PASS) return (sessions[backend] = entry);
      lastError = new Error(`${file} failed the self-test (${(100 * entry.selftest).toFixed(2)}% of pixels equal)`);
    } catch (e) { lastError = e; attempts.push({ file, error: String(e.message || e) }); }
  }
  console.warn("moviepix:", backend, JSON.stringify(attempts));
  throw lastError || new Error("no model file found in web/model/");
}

// ------------------------------------------------------------ drawing
function draw(canvas, rgba, w, h) {
  canvas.width = w; canvas.height = h;
  canvas.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
}

function showSelfTest(entry) {
  const chip = $("selftest");
  const pct = (100 * entry.selftest).toFixed(2);
  chip.className = "chip " + (entry.selftest >= SELFTEST_PASS ? "ok" : "bad");
  chip.textContent = entry.selftest >= SELFTEST_PASS ? `Self-test passed (${pct}%)` : `Self-test: only ${pct}% match`;
  $("modelinfo").textContent = `E1b · ${entry.file} · ${entry.backend === "webgpu" ? "WebGPU" : "WASM"}` +
    (entry.backend === "wasm" ? ` · ${self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1} thread(s)` : "");
}

function renderControls() {
  const act = info.actions.find((a) => a.name === st.action);
  $("actions").replaceChildren(...info.actions.map((a) => {
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-pressed", a.name === st.action);
    const shown = a.frames - (a.drop_first ? 1 : 0);
    b.innerHTML = `${a.name.replace("_", " ")}<small>${shown} frames${a.directions.length < 4 ? " · " + a.directions[0] + " only" : ""}</small>`;
    b.onclick = () => { st.action = a.name; if (!a.directions.includes(st.dir)) st.dir = a.directions[0]; renderControls(); };
    return b;
  }));
  $("dirs").replaceChildren(...info.directions.map((d) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = d; b.style.textAlign = "center";
    b.disabled = !act.directions.includes(d);
    b.setAttribute("aria-pressed", d === st.dir);
    b.onclick = () => { st.dir = d; renderControls(); };
    return b;
  }));
  $("backends").replaceChildren(...[["wasm", "WASM (CPU)"], ["webgpu", "WebGPU"]].map(([k, label]) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = label; b.style.textAlign = "center";
    b.disabled = k === "webgpu" && (!navigator.gpu || !!webgpuProblem);
    if (b.disabled) b.title = !navigator.gpu ? "This browser has no WebGPU" : `WebGPU is off: ${webgpuProblem}`;
    b.setAttribute("aria-pressed", k === st.backend);
    b.onclick = async () => { st.backend = k; renderControls(); await prepare(); };
    return b;
  }));
  $("zooms").replaceChildren(...[4, 6, 8].map((z) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = `${z}×`; b.setAttribute("aria-pressed", z === st.zoom);
    b.onclick = () => { st.zoom = z; renderControls(); renderResult(); };
    return b;
  }));
  $("go").disabled = !st.sprite || st.busy;
}

function renderResult() {
  const S = info.size, pv = $("preview");
  pv.style.width = pv.style.height = `${S * st.zoom}px`;
  if (!st.frames.length) return;
  const i = st.frame % st.frames.length;
  draw(pv, st.frames[i], S, S);
  $("framelabel").textContent = `Frame ${i + 1} of ${st.frames.length}`;
  [...$("strip").children].forEach((c, j) => c.classList.toggle("on", j === i));
}

function renderStrip() {
  const S = info.size;
  $("strip").replaceChildren(...st.frames.map((f, j) => {
    const c = document.createElement("canvas");
    c.className = "pix"; c.setAttribute("aria-label", `frame ${j + 1}`);
    draw(c, f, S, S);
    c.onclick = () => { st.playing = false; st.frame = j; $("play").textContent = "Play"; renderResult(); };
    return c;
  }));
}

// ------------------------------------------------------------ input
async function useImage(blob, name) {
  const bmp = await createImageBitmap(blob, { premultiplyAlpha: "none", colorSpaceConversion: "none" });
  if (bmp.width > 256 || bmp.height > 256) throw new Error(`That image is ${bmp.width} × ${bmp.height}; give one character, up to about 64 px.`);
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  setSprite(threshold(ctx.getImageData(0, 0, bmp.width, bmp.height).data), bmp.height, bmp.width, name);
}

function setSprite(rgba, h, w, name) {
  const S = info.size;
  draw($("orig"), rgba, w, h);
  const k = Math.max(1, Math.floor(128 / Math.max(w, h)));
  $("orig").style.width = `${w * k}px`; $("orig").style.height = `${h * k}px`;
  const placed = placeReference(rgba, h, w, S);
  if (!placed) {
    st.sprite = null;
    $("placeinfo").textContent = `${w} × ${h}: too big to fit the ${S} × ${S} canvas even at half size.`;
    renderControls();
    return;
  }
  st.sprite = placed.canvas; st.name = name.replace(/\.[a-z]+$/i, "") || "sprite";
  draw($("placed"), placed.canvas, S, S);
  const px = 128 / S;
  $("pivot").style.left = `${(info.pivot[0] + 0.5) * px}px`; $("pivot").style.top = `${(info.pivot[1] + 0.5) * px}px`;
  const n = colourIds(placed.canvas).count;
  $("placeinfo").textContent = `${w} × ${h}. Feet on (${info.pivot.join(", ")}), ` +
    (placed.scale === 1 ? "full size. " : "shrunk 2× (pixel-art shrink) to fit. ") +
    `${n} colours` + (n > info.max_colours ? ` — too many, the model takes at most ${info.max_colours}.` : ".");
  st.frames = []; $("strip").replaceChildren(); $("dlpng").disabled = $("dlgif").disabled = true;
  renderControls();
}

// ------------------------------------------------------------ actions
let webgpuProblem = null;  // set when WebGPU loads but fails the self-test in this browser

async function prepare() {
  $("selftest").className = "chip"; $("selftest").textContent = "Loading the model…";
  try {
    showSelfTest(await getSession(st.backend));
    $("error").textContent = "";
  } catch (e) {
    if (st.backend === "webgpu") {  // WebGPU is optional: fall back to WASM and say why
      webgpuProblem = e.message;
      st.backend = "wasm"; renderControls();
      await prepare();
      $("error").textContent = `WebGPU is off in this browser: ${e.message}. Using WASM instead.`;
      return;
    }
    $("selftest").className = "chip bad"; $("selftest").textContent = "Model not loaded";
    $("error").textContent = `Could not load the model: ${e.message}`;
  }
}

async function animate() {
  if (!st.sprite) return;
  st.busy = true; renderControls();
  $("status").textContent = "Working…";
  try {
    const entry = await getSession(st.backend);
    const { frames, ms, act } = await infer(entry, st.sprite, st.action, st.dir);
    st.frames = act.drop_first ? frames.slice(1) : frames;
    st.frame = 0; st.playing = true; $("play").textContent = "Pause";
    renderStrip(); renderResult();
    $("status").textContent = `${st.frames.length} frames of ${st.action.replace("_", " ")} (${st.dir}) in ${ms.toFixed(0)} ms on ${entry.backend === "webgpu" ? "WebGPU" : "WASM"}.`;
    $("dlpng").disabled = $("dlgif").disabled = false;
    $("error").textContent = "";
  } catch (e) {
    $("error").textContent = e.message;
    $("status").textContent = "";
  }
  st.busy = false; renderControls();
}

function save(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function downloadPng() {
  const S = info.size, c = new OffscreenCanvas(S * st.frames.length, S), ctx = c.getContext("2d");
  st.frames.forEach((f, i) => ctx.putImageData(new ImageData(new Uint8ClampedArray(f), S, S), i * S, 0));
  save(await c.convertToBlob({ type: "image/png" }), `${st.name}-${st.action}-${st.dir}.png`);
}

function downloadGif() {
  const S = info.size;
  save(new Blob([encodeGif(st.frames, S, S, 4, Math.round(100 / FPS))], { type: "image/gif" }),
       `${st.name}-${st.action}-${st.dir}.gif`);
}

// ------------------------------------------------------------ start
async function main() {
  try {
    [info, selftest] = await Promise.all(["actions.json", "selftest.json"].map(async (f) => {
      const r = await fetch(`./model/${f}`);
      if (!r.ok) throw new Error(`web/model/${f} is missing (see web/README.md)`);
      return r.json();
    }));
  } catch (e) {
    $("error").textContent = e.message; $("selftest").textContent = "No model";
    return;
  }
  renderControls();
  $("file").onchange = (e) => e.target.files[0] && useImage(e.target.files[0], e.target.files[0].name)
    .catch((err) => { $("error").textContent = err.message; });
  const drop = $("drop");
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => {
    e.preventDefault(); drop.classList.remove("over");
    const f = e.dataTransfer.files[0];
    if (f) useImage(f, f.name).catch((err) => { $("error").textContent = err.message; });
  };
  $("example").onclick = () => setSprite(bytes(selftest.reference_rgba), selftest.size, selftest.size, "example");
  $("go").onclick = animate;
  $("play").onclick = () => { st.playing = !st.playing; $("play").textContent = st.playing ? "Pause" : "Play"; };
  $("dlpng").onclick = downloadPng;
  $("dlgif").onclick = downloadGif;
  setInterval(() => { if (st.playing && st.frames.length) { st.frame++; renderResult(); } }, 1000 / FPS);
  renderResult();
  await prepare();
  if (location.hash === "#autotest") await autotest();
}

// Headless check (web/README.md): self-test on each backend this browser has, then the example
// through every action. Results go to <pre id="autotest"> as JSON.
async function autotest() {
  const out = { crossOriginIsolated: self.crossOriginIsolated, webgpu: !!navigator.gpu, backends: {}, actions: {} };
  for (const b of ["wasm", ...(navigator.gpu ? ["webgpu"] : [])]) {
    try {
      const e = await getSession(b);
      out.backends[b] = { file: e.file, selftest: e.selftest, attempts: e.attempts };
    } catch (err) { out.backends[b] = { error: String(err.message || err) }; }
  }
  $("example").click();
  for (const a of info.actions) {
    st.action = a.name; st.dir = a.directions.includes("down") ? "down" : a.directions[0];
    const t0 = performance.now();
    await animate();
    out.actions[a.name] = { frames: st.frames.length, ms: Math.round(performance.now() - t0), error: $("error").textContent || undefined };
  }
  const pre = document.createElement("pre");
  pre.id = "autotest"; pre.textContent = JSON.stringify(out);
  document.body.append(pre);
}

main();
