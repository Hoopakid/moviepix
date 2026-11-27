// node web/place.test.mjs  — place.js must place every fixture sprite exactly like data/packs.place_reference.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { colourIds, placeReference, roundHalfEven } from "./place.js";

const root = fileURLToPath(new URL("..", import.meta.url));
const fx = JSON.parse(readFileSync(root + "tests/fixtures/placement.json", "utf8"));
const bytes = (b64) => new Uint8ClampedArray(Buffer.from(b64, "base64"));
let failed = 0;

for (const [x, want] of [[10.5, 10], [11.5, 12], [-0.5, -0], [2.4, 2], [2.6, 3]]) {
  if (roundHalfEven(x) !== want) { console.log(`roundHalfEven(${x}) = ${roundHalfEven(x)}, want ${want}`); failed++; }
}

fx.cases.forEach((c, i) => {
  const got = placeReference(bytes(c.rgba), c.h, c.w, fx.size);
  if (c.scale === null || got === null) {
    if ((c.scale === null) !== (got === null)) { console.log(`case ${i} (${c.h}x${c.w}): refused mismatch`); failed++; }
    return;
  }
  const want = bytes(c.canvas);
  const diff = want.reduce((n, v, j) => n + (v !== got.canvas[j]), 0);
  if (got.scale !== c.scale || diff) {
    console.log(`case ${i} (${c.h}x${c.w}): scale ${got.scale} want ${c.scale}, ${diff} bytes differ`);
    failed++;
  }
});

const ids = colourIds(new Uint8ClampedArray([9, 9, 9, 255, 0, 0, 0, 0, 1, 2, 3, 255, 9, 9, 9, 255]));
if (ids.count !== 2 || [...ids.ids].join() !== "0,-1,1,0") { console.log("colourIds", [...ids.ids]); failed++; }

console.log(failed ? `FAILED ${failed}` : `ok: ${fx.cases.length} placements match Python`);
process.exit(failed ? 1 : 0);
