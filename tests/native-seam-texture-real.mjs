// Texture-only paired evaluation at labelled positions, not automatic seam recall.
// node tests/native-seam-texture-real.mjs requests.json output.json [baseline.ts]
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';
globalThis.ImageData = class {
  constructor(data, width, height) {
    Object.assign(this, { data, width, height });
  }
};
async function load(entry) {
  const b = await build({ entryPoints: [entry], bundle: true, format: 'esm', write: false });
  return import('data:text/javascript;base64,' + Buffer.from(b.outputFiles[0].contents).toString('base64'));
}
const [requestPath, outputPath, baselinePath] = process.argv.slice(2);
if (!requestPath || !outputPath) throw new Error('Provide requests.json output.json [baseline.ts]');
const current = await load('src/utils/nativeSeamTexture.ts');
const baseline = baselinePath ? await load(baselinePath) : null;
const requests = JSON.parse(fs.readFileSync(requestPath));
const offsets = [-14, -10, -3, 5, 11, 14, 11, 4];
function blurHalf(raw, size, axis, boundary) {
  const out = new Uint8ClampedArray(raw),
    stride = size + 1,
    radius = 4;
  const sums = new Float64Array(stride * stride);
  for (let channel = 0; channel < 3; channel++) {
    sums.fill(0);
    for (let y = 0; y < size; y++) {
      let sum = 0;
      for (let x = 0; x < size; x++) {
        sum += raw[(y * size + x) * 4 + channel];
        sums[(y + 1) * stride + x + 1] = sums[y * stride + x + 1] + sum;
      }
    }
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++) {
        if ((axis === 0 ? x : y) < boundary) continue;
        // Do not sample across the synthetic boundary when constructing the blur.
        const x0 = Math.max(axis === 0 ? boundary : 0, x - radius),
          x1 = Math.min(size, x + radius + 1);
        const y0 = Math.max(axis === 1 ? boundary : 0, y - radius),
          y1 = Math.min(size, y + radius + 1);
        out[(y * size + x) * 4 + channel] =
          (sums[y1 * stride + x1] - sums[y0 * stride + x1] - sums[y1 * stride + x0] + sums[y0 * stride + x0]) /
          ((x1 - x0) * (y1 - y0));
      }
  }
  return out;
}
const result = [];
for (const image of new Set(requests.map((r) => r.image)))
  for (const kind of ['vertical', 'horizontal']) {
    const rows = requests.filter((r) => r.image === image && r.kind === kind).sort((a, b) => a.sample - b.sample);
    if (rows.length !== 8) throw new Error('Expected eight crops per path');
    const cases = Object.fromEntries(['original', 'gain8', 'loss8', 'blur'].map((name) => [name, []]));
    for (const [k, r] of rows.entries()) {
      const rawPath = r.output + '.rgba',
        size = r.size ?? 640,
        center = size / 2,
        axis = kind === 'vertical' ? 0 : 1,
        normal = axis === 0 ? [1, 0] : [0, 1];
      const raw = fs.existsSync(rawPath)
        ? fs.readFileSync(rawPath)
        : execFileSync('ffmpeg', ['-v', 'error', '-i', r.output, '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'], {
            maxBuffer: 16 * 1024 * 1024,
          });
      if (raw.length !== size * size * 4) throw new Error('Wrong crop size');
      if (!fs.existsSync(rawPath)) fs.writeFileSync(rawPath, raw);
      for (const name of Object.keys(cases)) {
        const boundary = center + offsets[k];
        const data = name === 'blur' ? blurHalf(raw, size, axis, boundary) : new Uint8ClampedArray(raw);
        const gain = name === 'gain8' ? 1.08 : name === 'loss8' ? 0.92 : 1;
        if (gain !== 1)
          for (let y = 0; y < size; y++)
            for (let x = 0; x < size; x++)
              if ((axis === 0 ? x : y) >= boundary)
                for (let ch = 0; ch < 3; ch++) data[(y * size + x) * 4 + ch] *= gain;
        const args = [
          new ImageData(data, size, size),
          center + offsets[k] * normal[0],
          center + offsets[k] * normal[1],
          normal,
        ];
        cases[name].push({
          current: current.examineNativeTexture(...args),
          baseline: baseline?.examineNativeTexture(...args),
        });
      }
    }
    const summary = Object.fromEntries(
      Object.entries(cases).map(([name, evidence]) => [
        name,
        {
          current: current.summarizeNativeTexture(evidence.map((e) => e.current)),
          baseline: baseline?.summarizeNativeTexture(evidence.map((e) => e.baseline)),
          evidence,
        },
      ]),
    );
    result.push({ image, kind, crops: rows, cases: summary });
    console.log(
      image,
      kind,
      Object.entries(summary)
        .map(([name, v]) => `${name}: ${v.baseline?.status ?? '-'} -> ${v.current.status}`)
        .join('; '),
    );
  }
fs.writeFileSync(
  outputPath,
  JSON.stringify(
    { scope: 'texture-only at labelled native positions; not automatic path recall', groups: result },
    null,
    2,
  ),
);
