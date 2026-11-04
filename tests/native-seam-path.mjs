import assert from 'node:assert/strict';
import { build } from 'esbuild';
globalThis.ImageData = class {
  constructor(data, width, height) {
    Object.assign(this, { data, width, height });
  }
};
const buildResult = await build({
  entryPoints: ['src/utils/nativeSeamPath.ts'],
  bundle: true,
  format: 'esm',
  write: false,
});
const { scanNativeSeam, resolveNativeSeamSequence, resolveNativeSeamSequences } = await import(
  'data:text/javascript;base64,' + Buffer.from(buildResult.outputFiles[0].contents).toString('base64')
);
const noise = (x, y, k) => {
  let z = Math.imul(x + 1 + k * 29, 374761393) ^ Math.imul(y + 1, 668265263);
  z = Math.imul(z ^ (z >>> 13), 1274126177);
  return ((z ^ (z >>> 16)) >>> 0) / 4294967295 - 0.5;
};
function patch(k, kind, normal) {
  const data = new Uint8ClampedArray(640 * 640 * 4);
  const offset = -12 + k * 3;
  for (let y = 0; y < 640; y++)
    for (let x = 0; x < 640; x++) {
      const n = (x - 320) * normal[0] + (y - 320) * normal[1];
      const texture =
        110 +
        35 * noise(x, y, k) +
        3 * Math.sin(x * 0.04 + k) +
        (kind.startsWith('weave:') ? 20 * Math.sin((n * 2 * Math.PI) / Number(kind.split(':')[1]) + k * 0.2) : 0);
      const gain =
        kind === 'step'
          ? n >= offset
            ? 1.08
            : 1
          : kind === 'line'
            ? Math.abs(n - offset) < 2
              ? 0.85
              : 1
            : kind === 'ramp'
              ? 2 ** (n / 1280)
              : kind === 'partial'
                ? k < 2 && n >= offset
                  ? 1.08
                  : 1
                : kind === 'opposite'
                  ? n >= offset
                    ? k % 2
                      ? 1.08
                      : 1 / 1.08
                    : 1
                  : 1;
      const v = texture * gain;
      data.set([v, v, v, 255], (y * 640 + x) * 4);
    }
  return new ImageData(data, 640, 640);
}
for (const normal of [
  [1, 0],
  [0, 1],
  [Math.SQRT1_2, Math.SQRT1_2],
]) {
  const scans = Array.from({ length: 8 }, (_, k) => scanNativeSeam(patch(k, 'step', normal), 320, 320, normal));
  const result = resolveNativeSeamSequence(scans);
  assert.equal(result.status, 'stable', JSON.stringify(result));
  assert.ok(
    result.offsets.every((offset, k) => Math.abs(offset - (-12 + k * 3)) <= 2),
    JSON.stringify(result),
  );
}
for (const kind of [
  'noise',
  'line',
  'ramp',
  'partial',
  'opposite',
  ...[4, 7, 11, 17, 23].map((period) => `weave:${period}`),
]) {
  const scans = Array.from({ length: 8 }, (_, k) => scanNativeSeam(patch(k, kind, [1, 0]), 320, 320, [1, 0]));
  const result = resolveNativeSeamSequence(scans);
  assert.equal(result.status, 'insufficient', kind + JSON.stringify(result));
}
const invalid = patch(0, 'noise', [1, 0]);
invalid.data.fill(0);
assert.equal(
  resolveNativeSeamSequence(Array.from({ length: 8 }, () => scanNativeSeam(invalid, 320, 320, [1, 0]))).status,
  'insufficient',
);
assert.equal(resolveNativeSeamSequence([]).status, 'insufficient');
const occluded = Array.from({ length: 8 }, (_, k) =>
  scanNativeSeam(k === 2 || k === 5 ? invalid : patch(k, 'step', [1, 0]), 320, 320, [1, 0]),
);
assert.equal(
  resolveNativeSeamSequence(occluded).status,
  'stable',
  'two missing samples must not force the path onto a different edge',
);
// A nearby stronger contour differs only in the middle. Mean-distance merging
// previously removed the true straight seam despite its independent evidence.
const adjacentContour = [0, 0, 11, 14, 11, -1, 0, 0].map((offset) => {
  const scan = {
    training: Array(225).fill(0),
    validation: Array(225).fill(0),
    delta: Array(225).fill(0),
    plateau: Array(225).fill(0),
  };
  for (const position of [0, offset]) {
    scan.training[position + 112] = position === 0 ? 1.8 : 2.1;
    scan.validation[position + 112] = 2;
    scan.delta[position + 112] = 0.1;
    scan.plateau[position + 112] = 0.1;
  }
  return scan;
});
assert.ok(
  resolveNativeSeamSequences(adjacentContour).some((path) => path.offsets.every((o) => Math.abs(o) <= 2)),
  'a stronger curved neighbor must not erase the independently supported straight path',
);
console.log(
  'seam path: weak drifting/diagonal paths, silk, line/ramp, partial/sign-conflict and invalid controls passed',
);
