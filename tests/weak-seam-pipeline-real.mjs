import fs from 'node:fs';
import { build } from 'esbuild';
globalThis.ImageData = class {
  constructor(data, width, height) {
    Object.assign(this, { data, width, height });
  }
};
async function load(entry) {
  const b = await build({ entryPoints: ['src/utils/' + entry], bundle: true, format: 'esm', write: false });
  return import('data:text/javascript;base64,' + Buffer.from(b.outputFiles[0].contents).toString('base64'));
}
const { detectWeakSeamCandidates } = await load('weakSeamCandidates.ts');
const { scanNativeSeam, resolveNativeSeamSequences } = await load('nativeSeamPath.ts');
const { examineNativeTexture, summarizeNativeTexture } = await load('nativeSeamTexture.ts');
const [manifestPath, outputPath] = process.argv.slice(2);
if (!manifestPath || !outputPath) throw new Error('Provide dataset manifest JSON and output JSON');
const meta = JSON.parse(fs.readFileSync(manifestPath));
const results = [];
for (const dataset of meta)
  for (const kind of ['original', 'vertical', 'horizontal']) {
    const { id, width, height } = dataset;
    const data = new Uint8ClampedArray(fs.readFileSync(dataset.rgba));
    if (kind !== 'original')
      for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++)
          if (kind === 'vertical' ? x >= width * 0.5 : y >= height * 0.5) {
            const i = (y * width + x) * 4;
            for (let c = 0; c < 3; c++) data[i + c] *= 1.08;
          }
    const start = performance.now(),
      proposals = [...detectWeakSeamCandidates(new ImageData(data, width, height))].filter(Boolean),
      seams = [];
    for (const c of proposals) {
      const normal = c.normals[0],
        span = Math.max(
          32,
          Math.min(
            960,
            Math.min(
              ...c.points
                .slice(1)
                .map((p, i) =>
                  Math.abs(
                    Math.round(p[normal[0] ? 1 : 0] * (normal[0] ? height : width)) -
                      Math.round(c.points[i][normal[0] ? 1 : 0] * (normal[0] ? height : width)),
                  ),
                ),
            ) * 0.8,
          ),
        ),
        patches = [];
      for (const p of c.points) {
        const px = Math.round(p[0] * width),
          py = Math.round(p[1] * height),
          copy = new Uint8ClampedArray(1280 * 1280 * 4);
        for (let y = 0; y < 1280; y++) {
          const sy = py - 640 + y;
          if (sy < 0 || sy >= height) continue;
          const left = Math.max(0, px - 640),
            right = Math.min(width, px + 640);
          copy.set(
            data.subarray((sy * width + left) * 4, (sy * width + right) * 4),
            (y * 1280 + left - (px - 640)) * 4,
          );
        }
        const pixels = new ImageData(copy, 1280, 1280);
        patches.push({ px, py, pixels, textureCache: new Map(), scan: scanNativeSeam(pixels, 640, 640, normal, span) });
      }
      const sequences = resolveNativeSeamSequences(
        patches.map((p) => p.scan),
        patches.map((p) => p.px * normal[0] + p.py * normal[1]),
      );
      for (const sequence of sequences) {
        const refined = patches.map((p, i) => [
          (p.px + sequence.offsets[i] * normal[0]) / width,
          (p.py + sequence.offsets[i] * normal[1]) / height,
        ]);
        const texture = summarizeNativeTexture(
          patches.map((p, i) => {
            const offset = sequence.offsets[i];
            if (!p.textureCache.has(offset))
              p.textureCache.set(
                offset,
                examineNativeTexture(p.pixels, 640 + offset * normal[0], 640 + offset * normal[1], normal),
              );
            return p.textureCache.get(offset);
          }),
        );
        seams.push({ ...c, sequence, refined, texture });
      }
    }
    const axis = kind === 'vertical' ? 0 : 1;
    const hits =
      kind === 'original'
        ? []
        : seams.filter(
            (s) =>
              s.normals[0][axis] === 1 &&
              s.refined.filter((p) => Math.abs(p[axis] - 0.5) * (axis === 0 ? width : height) <= 4).length >= 6,
          );
    const row = {
      id,
      kind,
      width,
      height,
      ms: performance.now() - start,
      proposals: proposals.length,
      hits: hits.length,
      textureHits: hits.filter((s) => s.texture.status === 'supported').length,
      seams,
    };
    results.push(row);
    console.log(
      id,
      kind,
      'proposals',
      row.proposals,
      'verified',
      seams.length,
      'hits',
      row.hits,
      'texture',
      row.textureHits,
      'ms',
      Math.round(row.ms),
    );
  }
fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
