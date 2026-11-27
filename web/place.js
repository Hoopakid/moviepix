// Placement of one standing sprite on the model's canvas. Mirrors data/packs.py place_reference()
// and data/resize.py downscale() exactly; tests/fixtures/placement.json checks it (place.test.mjs).

export const NUDGE = 2;

// Python's round(): halves go to the even neighbour.
export function roundHalfEven(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// Alpha < 128 becomes fully transparent (all channels 0), the rest fully opaque (data/packs.rgba).
export function threshold(rgba) {
  const out = new Uint8ClampedArray(rgba);
  for (let i = 0; i < out.length; i += 4) {
    if (out[i + 3] < 128) { out[i] = out[i + 1] = out[i + 2] = out[i + 3] = 0; } else out[i + 3] = 255;
  }
  return out;
}

function padEven(img, h, w) {
  const H = h + (h % 2), W = w + (w % 2);
  if (H === h && W === w) return { img, h, w };
  const out = new Uint8ClampedArray(H * W * 4);
  for (let y = 0; y < h; y++) out.set(img.subarray(y * w * 4, (y + 1) * w * 4), y * W * 4);
  return { img: out, h: H, w: W };
}

// Pixel-art 2x shrink (data/resize.py downscale, factor 2): each 2x2 block keeps its most common
// opaque colour, ties broken by closeness to the mean colour of the 4x4 area around it; a block
// with fewer than 2 opaque pixels becomes transparent.
export function downscale2(img, h, w) {
  const hb = h / 2, wb = w / 2, out = new Uint8ClampedArray(hb * wb * 4);
  const at = (y, x, c) => img[(y * w + x) * 4 + c];
  for (let by = 0; by < hb; by++) {
    for (let bx = 0; bx < wb; bx++) {
      // mean opaque colour over rows 2by-1 .. 2by+2, cols 2bx-1 .. 2bx+2 (zero padding outside)
      let sr = 0, sg = 0, sb = 0, n = 0;
      for (let y = 2 * by - 1; y <= 2 * by + 2; y++) {
        for (let x = 2 * bx - 1; x <= 2 * bx + 2; x++) {
          if (y < 0 || x < 0 || y >= h || x >= w || at(y, x, 3) === 0) continue;
          sr += at(y, x, 0); sg += at(y, x, 1); sb += at(y, x, 2); n += 1;
        }
      }
      const d = Math.max(n, 1), mean = [sr / d, sg / d, sb / d];
      const px = [];
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const y = 2 * by + dy, x = 2 * bx + dx;
        px.push([at(y, x, 0), at(y, x, 1), at(y, x, 2), at(y, x, 3)]);
      }
      let best = -Infinity, pick = 0, opaque = 0;
      for (let i = 0; i < 4; i++) {
        const p = px[i];
        if (p[3] > 0) opaque += 1;
        let score = -1;
        if (p[3] > 0) {
          let votes = 0;
          for (const q of px) if (q[3] > 0 && q[0] === p[0] && q[1] === p[1] && q[2] === p[2]) votes += 1;
          const dist = (Math.abs(p[0] - mean[0]) + Math.abs(p[1] - mean[1]) + Math.abs(p[2] - mean[2])) / (3 * 255 * 2);
          score = votes - dist;
        }
        if (score > best) { best = score; pick = i; }
      }
      const o = (by * wb + bx) * 4;
      if (opaque * 2 >= 4) { out[o] = px[pick][0]; out[o + 1] = px[pick][1]; out[o + 2] = px[pick][2]; out[o + 3] = 255; }
    }
  }
  return out;
}

// img: thresholded RGBA [h * w * 4]. Returns { canvas: Uint8ClampedArray [size * size * 4], scale } or null.
export function placeReference(img, h, w, size = 32) {
  const pivot = [Math.floor(size / 2), size - 2];
  for (const k of [1, 2]) {
    let r = img, rh = h, rw = w;
    if (k === 2) {
      const p = padEven(img, h, w);
      r = downscale2(p.img, p.h, p.w); rh = p.h / 2; rw = p.w / 2;
    }
    let ymin = Infinity, ymax = -Infinity, xmin = Infinity, xmax = -Infinity;
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
      if (r[(y * rw + x) * 4 + 3] === 0) continue;
      ymin = Math.min(ymin, y); ymax = Math.max(ymax, y); xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
    }
    if (ymax < 0) return null;
    let oy = pivot[1] - ymax;
    let ox = roundHalfEven(pivot[0] - (xmin + xmax + 1) / 2);
    for (const [axis, lo0, hi0] of [["y", ymin, ymax], ["x", xmin, xmax]]) {
      const lo = lo0 + (axis === "y" ? oy : ox), hi = hi0 + (axis === "y" ? oy : ox);
      const shift = hi > size - 1 ? size - 1 - hi : (lo < 0 ? -lo : 0);
      if (Math.abs(shift) > 0 && Math.abs(shift) <= NUDGE && lo + shift >= 0 && hi + shift <= size - 1) {
        if (axis === "y") oy += shift; else ox += shift;
      }
    }
    if (ymin + oy < 0 || ymax + oy >= size || xmin + ox < 0 || xmax + ox >= size) continue;
    const canvas = new Uint8ClampedArray(size * size * 4);
    for (let y = 0; y < rh; y++) for (let x = 0; x < rw; x++) {
      const s = (y * rw + x) * 4;
      if (r[s + 3] === 0) continue;
      canvas.set(r.subarray(s, s + 4), ((y + oy) * size + x + ox) * 4);
    }
    return { canvas, scale: k };
  }
  return null;
}

// Colour number per pixel, by first appearance in row-major order; -1 where transparent
// (export/onnx.py colour_ids). Returns { ids: BigInt64Array, count }.
export function colourIds(canvas) {
  const n = canvas.length / 4, ids = new BigInt64Array(n), seen = new Map();
  for (let i = 0; i < n; i++) {
    if (canvas[i * 4 + 3] < 128) { ids[i] = -1n; continue; }
    const key = (canvas[i * 4] << 16) | (canvas[i * 4 + 1] << 8) | canvas[i * 4 + 2];
    if (!seen.has(key)) seen.set(key, seen.size);
    ids[i] = BigInt(seen.get(key));
  }
  return { ids, count: seen.size };
}
