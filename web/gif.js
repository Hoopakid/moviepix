// A small animated-GIF writer for pixel art: one global palette (transparent + up to 255 colours),
// LZW-compressed frames, looping forever. No dependencies.

// frames: array of Uint8ClampedArray RGBA [w * h * 4] at 1x; scale: integer zoom; delay: 1/100 s.
export function encodeGif(frames, w, h, scale = 1, delay = 12) {
  // palette: index 0 is transparent; the most frequent colours keep their exact value
  const counts = new Map();
  for (const f of frames) for (let i = 0; i < f.length; i += 4) {
    if (f[i + 3] < 128) continue;
    const key = (f[i] << 16) | (f[i + 1] << 8) | f[i + 2];
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  const kept = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 255).map(([k]) => k);
  const index = new Map(kept.map((k, i) => [k, i + 1]));
  const nearest = (key) => {  // a colour beyond 255 (rare: "new" pixels) maps to the closest kept one
    let best = 1, bd = Infinity;
    kept.forEach((k, i) => {
      const d = Math.abs((k >> 16) - (key >> 16)) + Math.abs(((k >> 8) & 255) - ((key >> 8) & 255)) + Math.abs((k & 255) - (key & 255));
      if (d < bd) { bd = d; best = i + 1; }
    });
    index.set(key, best);
    return best;
  };
  let bits = 2;
  while ((1 << bits) < kept.length + 1) bits++;
  const W = w * scale, H = h * scale, out = [];
  const u16 = (v) => out.push(v & 255, (v >> 8) & 255);
  const str = (s) => { for (const c of s) out.push(c.charCodeAt(0)); };

  str("GIF89a"); u16(W); u16(H);
  out.push(0x80 | (bits - 1), 0, 0);  // global colour table of 2^bits entries
  for (let i = 0; i < (1 << bits); i++) {
    const k = i === 0 || i > kept.length ? 0 : kept[i - 1];
    out.push((k >> 16) & 255, (k >> 8) & 255, k & 255);
  }
  out.push(0x21, 0xff, 11); str("NETSCAPE2.0"); out.push(3, 1, 0, 0, 0);  // loop forever

  for (const f of frames) {
    out.push(0x21, 0xf9, 4, (2 << 2) | 1); u16(delay); out.push(0, 0);  // dispose to background, index 0 transparent
    out.push(0x2c); u16(0); u16(0); u16(W); u16(H); out.push(0);
    const px = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = ((Math.floor(y / scale) * w) + Math.floor(x / scale)) * 4;
      if (f[i + 3] < 128) continue;
      const key = (f[i] << 16) | (f[i + 1] << 8) | f[i + 2];
      px[y * W + x] = index.get(key) ?? nearest(key);
    }
    out.push(bits);
    const data = lzw(px, bits);
    for (let i = 0; i < data.length; i += 255) {
      const chunk = data.slice(i, i + 255);
      out.push(chunk.length, ...chunk);
    }
    out.push(0);
  }
  out.push(0x3b);
  return new Uint8Array(out);
}

// GIF LZW (variable code size up to 12 bits, clear when the table is full).
function lzw(indices, minCode) {
  const clear = 1 << minCode, eoi = clear + 1, out = [];
  let size = minCode + 1, next = eoi + 1, table = new Map(), cur = 0, shift = 0;
  const emit = (code) => {
    cur |= code << shift; shift += size;
    while (shift >= 8) { out.push(cur & 255); cur >>>= 8; shift -= 8; }
  };
  emit(clear);
  let prefix = indices[0];
  for (let i = 1; i < indices.length; i++) {
    const k = indices[i], key = (prefix << 8) | k, hit = table.get(key);
    if (hit !== undefined) { prefix = hit; continue; }
    emit(prefix);
    if (next === 4096) {
      emit(clear); table = new Map(); next = eoi + 1; size = minCode + 1;
    } else {
      if (next >= (1 << size)) size++;
      table.set(key, next++);
    }
    prefix = k;
  }
  emit(prefix); emit(eoi);
  if (shift > 0) out.push(cur & 255);
  return out;
}
