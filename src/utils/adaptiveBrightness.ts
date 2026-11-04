/** Image-derived illumination model. Coordinates are normalized for preview/export parity. */
export interface IlluminationRegion {
  x: number; y: number; width: number; height: number;
  light: number; confidence: number;
}
export interface IlluminationModel {
  regions: IlluminationRegion[];
  target: number;
}
export interface ProtectedRegion { x: number; y: number; width: number; height: number }
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const smooth = (v: number) => { const t = clamp(v, 0, 1); return t * t * (3 - 2 * t); };
const quantile = (a: number[], q: number) => a[Math.min(a.length - 1, Math.floor((a.length - 1) * q))] ?? 0;

/** Robust multi-scale samples; subdivision follows tonal variation, never the UI grid. */
export function analyzeIllumination(source: ImageData): IlluminationModel {
  const scale = Math.min(1, 256 / Math.max(source.width, source.height));
  const width = Math.max(1, Math.round(source.width * scale));
  const height = Math.max(1, Math.round(source.height * scale));
  const lights = new Float32Array(width * height);
  const chroma = new Float32Array(width * height);
  const valid = new Uint8Array(width * height);
  // Area sampling suppresses sensor texture and makes analysis stable at different sizes.
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let r = 0, g = 0, b = 0, n = 0;
    const x0 = Math.floor(x * source.width / width), x1 = Math.floor((x + 1) * source.width / width);
    const y0 = Math.floor(y * source.height / height), y1 = Math.floor((y + 1) * source.height / height);
    const stride = Math.max(1, Math.floor(Math.sqrt((x1 - x0) * (y1 - y0) / 16)));
    for (let py = y0; py < y1; py += stride) for (let px = x0; px < x1; px += stride) {
      const i = (py * source.width + px) * 4;
      if (source.data[i + 3] < 250) continue;
      r += source.data[i]; g += source.data[i + 1]; b += source.data[i + 2]; n++;
    }
    if (!n) continue;
    r /= n; g /= n; b /= n;
    const l = r * .2126 + g * .7152 + b * .0722;
    const i = y * width + x;
    lights[i] = Math.log2(Math.max(8, l));
    chroma[i] = (Math.max(r, g, b) - Math.min(r, g, b)) / Math.max(16, l);
    valid[i] = l > 12 && l < 245 ? 1 : 0;
  }
  const regions: IlluminationRegion[] = [];
  function split(x: number, y: number, w: number, h: number, depth: number) {
    const values: number[] = []; let color = 0;
    for (let py = y; py < y + h; py++) for (let px = x; px < x + w; px++) {
      const i = py * width + px;
      if (valid[i]) { values.push(lights[i]); color += chroma[i]; }
    }
    if (!values.length) return;
    values.sort((a, b) => a - b);
    const spread = quantile(values, .85) - quantile(values, .15);
    // Always establish a coarse scale, then refine only where evidence changes.
    if (depth < 5 && w >= 12 && h >= 12 && (depth < 2 || spread > .18)) {
      const hw = Math.floor(w / 2), hh = Math.floor(h / 2);
      split(x, y, hw, hh, depth + 1); split(x + hw, y, w - hw, hh, depth + 1);
      split(x, y + hh, hw, h - hh, depth + 1); split(x + hw, y + hh, w - hw, h - hh, depth + 1);
      return;
    }
    regions.push({ x: x / width, y: y / height, width: w / width, height: h / height,
      light: quantile(values, .6),
      // Texture, saturated objects, clipped pixels and sparse support are weak illumination evidence.
      confidence: values.length / (w * h) / (1 + 4 * spread * spread + 3 * (color / values.length) ** 2),
    });
  }
  split(0, 0, width, height, 0);
  let sum = 0, weight = 0;
  for (const r of regions) { const w = r.width * r.height * r.confidence; sum += r.light * w; weight += w; }
  return { regions, target: weight ? sum / weight : 0 };
}

/** Continuous overlapping kernels: quadtree boundaries never become rendering boundaries. */
export function applyIllumination(source: ImageData, model: IlluminationModel, strength = .5, protectedRegions: ProtectedRegion[] = []): ImageData {
  strength = clamp(strength, 0, 1);
  if (!strength || !model.regions.length) return source;
  const size = 64;
  const field = new Float32Array(size * size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const nx = x / (size - 1), ny = y / (size - 1);
    let sum = 0, weights = 0;
    for (const r of model.regions) {
      const dx = (nx - r.x - r.width / 2) / Math.max(.055, r.width * .8);
      const dy = (ny - r.y - r.height / 2) / Math.max(.055, r.height * .8);
      const w = Math.exp(-.5 * (dx * dx + dy * dy)) * r.confidence * r.width * r.height;
      sum += r.light * w; weights += w;
    }
    field[y * size + x] = weights ? clamp(model.target - sum / weights, -.5, .5) * strength : 0;
  }
  const out = new ImageData(new Uint8ClampedArray(source.data), source.width, source.height);
  for (let y = 0; y < source.height; y++) for (let x = 0; x < source.width; x++) {
    const nx = (x + .5) / source.width, ny = (y + .5) / source.height;
    const fx = nx * (size - 1), fy = ny * (size - 1);
    const ix = Math.floor(fx), iy = Math.floor(fy), jx = Math.min(size - 1, ix + 1), jy = Math.min(size - 1, iy + 1);
    const tx = fx - ix, ty = fy - iy;
    let stops = (field[iy * size + ix] * (1 - tx) + field[iy * size + jx] * tx) * (1 - ty)
      + (field[jy * size + ix] * (1 - tx) + field[jy * size + jx] * tx) * ty;
    let protection = 1;
    for (const r of protectedRegions) {
      const dx = Math.max(r.x - nx, nx - r.x - r.width, 0);
      const dy = Math.max(r.y - ny, ny - r.y - r.height, 0);
      protection = Math.min(protection, smooth(Math.hypot(dx, dy) / .04));
    }
    const i = (y * source.width + x) * 4;
    const peak = Math.max(source.data[i], source.data[i + 1], source.data[i + 2]);
    stops *= protection * smooth((255 - peak) / 24) * smooth(peak / 24);
    const gain = 2 ** stops;
    // One RGB gain preserves channel ratios; cap it jointly to avoid hue shifts at clipping.
    const safeGain = Math.min(gain, 255 / Math.max(1, peak));
    if (!source.data[i + 3]) continue;
    out.data[i] = source.data[i] * safeGain;
    out.data[i + 1] = source.data[i + 1] * safeGain;
    out.data[i + 2] = source.data[i + 2] * safeGain;
  }
  return out;
}
