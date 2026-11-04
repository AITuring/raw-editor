// Reuses native 640px crops and provenance from the original-resolution audit.
// Usage: node tests/native-seam-profile-real.mjs requests.json output.json
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';
import path from 'node:path';
globalThis.ImageData = class {
  constructor(data, width, height) {
    Object.assign(this, { data, width, height });
  }
};
const [requestsPath, outputPath] = process.argv.slice(2);
if (!requestsPath || !outputPath) throw new Error('Provide native crop requests JSON and output JSON');
async function load(entry) {
  const b = await build({ entryPoints: [entry], bundle: true, format: 'esm', write: false });
  return import('data:text/javascript;base64,' + Buffer.from(b.outputFiles[0].contents).toString('base64'));
}
const { examineNativeSeamProfile, summarizeNativeSeam } = await load('src/utils/nativeSeamProfile.ts');
const { examineNativeTexture } = await load('src/utils/nativeSeamTexture.ts');
const requests = JSON.parse(fs.readFileSync(requestsPath, 'utf8'));
const samples = [];
const analyze = (data, normal) => {
  const image = new ImageData(data, 640, 640);
  const profile = examineNativeSeamProfile(image, 320, 320, normal);
  const length = Math.hypot(...normal),
    offset = profile.status === 'stable' ? profile.offset : 0;
  const texture = examineNativeTexture(
    image,
    320 + (offset * normal[0]) / length,
    320 + (offset * normal[1]) / length,
    normal,
  );
  return { profile, texture, textureSupported: texture.status === 'supported' };
};
for (const request of requests) {
  const cache = request.output + '.rgba';
  const raw = fs.existsSync(cache)
    ? fs.readFileSync(cache)
    : execFileSync('ffmpeg', ['-v', 'error', '-i', request.output, '-f', 'rawvideo', '-pix_fmt', 'rgba', 'pipe:1'], {
        maxBuffer: 2 * 1024 * 1024,
      });
  if (raw.length !== 640 * 640 * 4) throw new Error('Expected native 640px crop: ' + request.output);
  if (!fs.existsSync(cache)) fs.writeFileSync(cache, raw);
  const data = new Uint8ClampedArray(raw),
    control = analyze(data, request.normal),
    injected = [];
  if (request.kind !== 'candidate') {
    for (const [gain, offset] of [
      [1.08, -23],
      [1.2, 27],
    ]) {
      const copy = new Uint8ClampedArray(data);
      for (let y = 0; y < 640; y++)
        for (let x = 0; x < 640; x++) {
          if ((request.kind === 'vertical' ? x : y) < 320 + offset) continue;
          const i = (y * 640 + x) * 4;
          for (let c = 0; c < 3; c++) copy[i + c] *= gain;
        }
      const result = analyze(copy, request.normal);
      injected.push({ gain, offset, ...result, locationError: Math.abs(result.profile.offset - offset) });
      // A 1:1 visual pair for the first local sample of each image/direction.
      if (request.sample === 0 && gain === 1.2) {
        const base = path.join(path.dirname(outputPath), `profile-${request.image}-${request.kind}`);
        fs.copyFileSync(request.output, base + '-original.png');
        execFileSync(
          'ffmpeg',
          [
            '-v',
            'error',
            '-y',
            '-f',
            'rawvideo',
            '-pixel_format',
            'rgba',
            '-video_size',
            '640x640',
            '-i',
            'pipe:0',
            '-frames:v',
            '1',
            base + '-injected.png',
          ],
          { input: copy },
        );
      }
    }
  }
  samples.push({ ...request, control, injected });
}
const candidates = [];
for (let image = 1; image <= 3; image++) {
  const rows = samples.filter((s) => s.image === image && s.kind === 'candidate');
  for (const candidate of new Set(rows.map((r) => r.candidate))) {
    candidates.push({
      image,
      candidate,
      evidence: summarizeNativeSeam(rows.filter((r) => r.candidate === candidate).map((r) => r.control)),
    });
  }
}
const summary = [1.08, 1.2].map((gain) => {
  const rows = samples.flatMap((s) => s.injected.filter((i) => i.gain === gain));
  const located = rows.filter((r) => r.profile.status === 'stable');
  const correct = located.filter((r) => r.locationError <= 4);
  return {
    gain,
    total: rows.length,
    localized: located.length,
    correctlyLocalized: correct.length,
    textureCompatible: correct.filter((r) => r.textureSupported).length,
    maxAcceptedError: Math.max(0, ...located.map((r) => r.locationError)),
    meanDeltaError: correct.length
      ? correct.reduce((s, r) => s + Math.abs(r.profile.delta - Math.log2(gain)), 0) / correct.length
      : null,
  };
});
fs.writeFileSync(outputPath, JSON.stringify({ summary, candidates, samples }, null, 2));
console.log(JSON.stringify({ summary, candidates }, null, 2));
