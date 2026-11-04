// Original-resolution real-texture path benchmark. Does not overwrite originals.
// node tests/native-seam-path-real.mjs crop-requests.json output.json
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';
globalThis.ImageData = class {
  constructor(data, width, height) {
    Object.assign(this, { data, width, height });
  }
};
const [requestPath, outputPath] = process.argv.slice(2);
if (!requestPath || !outputPath) throw new Error('Provide crop request manifest and output JSON');
async function load(entry) {
  const b = await build({ entryPoints: [entry], bundle: true, format: 'esm', write: false });
  return import('data:text/javascript;base64,' + Buffer.from(b.outputFiles[0].contents).toString('base64'));
}
const { scanNativeSeam, resolveNativeSeamSequence, resolveNativeSeamSequences } =
  await load('src/utils/nativeSeamPath.ts');
const { examineNativeTexture, summarizeNativeTexture } = await load('src/utils/nativeSeamTexture.ts');
const requests = JSON.parse(fs.readFileSync(requestPath, 'utf8')).filter((r) => r.kind !== 'candidate');
const groups = [];
for (const image of new Set(requests.map((r) => r.image)))
  for (const kind of ['vertical', 'horizontal']) {
    const rows = requests.filter((r) => r.image === image && r.kind === kind).sort((a, b) => a.sample - b.sample);
    const size = rows[0].size ?? 640,
      center = size / 2,
      span = rows[0].span ?? 480;
    if (rows.length !== 8) throw new Error('Expected 8 distinct native crops per path');
    const originals = rows.map((r) => {
      const cache = r.output + '.rgba';
      const data = fs.existsSync(cache)
        ? fs.readFileSync(cache)
        : execFileSync('ffmpeg', ['-v', 'error', '-i', r.output, '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'], {
            maxBuffer: 16 * 1024 * 1024,
          });
      if (data.length !== size * size * 4) throw new Error('Invalid native crop');
      if (!fs.existsSync(cache)) fs.writeFileSync(cache, data);
      return data;
    });
    const cases = [];
    for (const gain of [1, 1.04, 1.08, 0.92, 1.2]) {
      const offsets = [-14, -10, -3, 5, 11, 14, 11, 4];
      const patches = originals.map((raw, k) => {
        const data = new Uint8ClampedArray(raw);
        if (gain !== 1)
          for (let y = 0; y < size; y++)
            for (let x = 0; x < size; x++) {
              if ((kind === 'vertical' ? x : y) < center + offsets[k]) continue;
              const i = (y * size + x) * 4;
              for (let c = 0; c < 3; c++) data[i + c] *= gain;
            }
        return new ImageData(data, size, size);
      });
      const normal = kind === 'vertical' ? [1, 0] : [0, 1];
      const scans = patches.map((p) => scanNativeSeam(p, center, center, normal, span));
      const result = resolveNativeSeamSequence(scans);
      const alternatives = resolveNativeSeamSequences(scans);
      const matchedAlternatives = alternatives.filter(
        (a) => a.offsets.filter((v, k) => Math.abs(v - offsets[k]) <= 4).length >= 6,
      );
      const textures = patches.map((p, k) => {
        const offset = result.status === 'stable' ? result.offsets[k] : 0;
        return examineNativeTexture(p, center + offset * normal[0], center + offset * normal[1], normal);
      });
      const errors = result.offsets.map((v, k) => Math.abs(v - offsets[k]));
      cases.push({
        gain,
        expectedOffsets: offsets,
        ...result,
        alternatives,
        matchedAlternatives: matchedAlternatives.length,
        matchedLocations: errors.filter((v) => v <= 4).length,
        maxError: errors.length ? Math.max(...errors) : null,
        texture: summarizeNativeTexture(textures),
        textureCounts: {
          supported: textures.filter((t) => t.status === 'supported').length,
          inconsistent: textures.filter((t) => t.status === 'inconsistent').length,
          insufficient: textures.filter((t) => t.status === 'insufficient').length,
        },
      });
    }
    groups.push({ image, kind, source: rows[0].source, crops: rows, cases });
    console.log(
      image,
      kind,
      cases
        .map(
          (c) =>
            `${c.gain}: ${c.status}, matched paths ${c.matchedAlternatives}, located ${c.matchedLocations}/8, texture ${c.texture.status}`,
        )
        .join('; '),
    );
  }
fs.writeFileSync(outputPath, JSON.stringify(groups, null, 2));
