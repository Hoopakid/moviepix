// node web/gif.test.mjs OUT.gif — writes a test GIF (random frames, many colours, transparency, 2x)
// and the frames it was made from (OUT.gif.json); tests/test_web.py decodes it with Pillow.
import { writeFileSync } from "node:fs";
import { encodeGif } from "./gif.js";

const [w, h, n] = [32, 32, 5];
let seed = 1;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const frames = [];
for (let t = 0; t < n; t++) {
  const f = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    if (rand() < 0.3) continue;  // transparent
    const c = Math.floor(rand() * 200);  // ~200 colours: a 255-entry palette keeps them exact
    f.set([c, (c * 7) % 256, (c * 13) % 256, 255], i * 4);
  }
  frames.push(f);
}
const out = process.argv[2];
writeFileSync(out, encodeGif(frames, w, h, 2, 10));
writeFileSync(out + ".json", JSON.stringify({ w, h, scale: 2, frames: frames.map((f) => Buffer.from(f).toString("base64")) }));
