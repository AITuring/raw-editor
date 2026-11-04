import assert from 'node:assert/strict';
import { build } from 'esbuild';
globalThis.ImageData = class {
  constructor(data, width, height) {
    Object.assign(this, { data, width, height });
  }
};
const bundle = await build({
  entryPoints: ['src/utils/nativeSeamProfile.ts'],
  bundle: true,
  format: 'esm',
  write: false,
});
const { examineNativeSeamProfile, summarizeNativeSeam } = await import(
  'data:text/javascript;base64,' + Buffer.from(bundle.outputFiles[0].contents).toString('base64')
);
const noise = (x, y) => {
  let z = Math.imul(x + 1, 374761393) ^ Math.imul(y + 1, 668265263);
  z = Math.imul(z ^ (z >>> 13), 1274126177);
  return ((z ^ (z >>> 16)) >>> 0) / 4294967295 - 0.5;
};
const make = (pixel) => {
  const data = new Uint8ClampedArray(640 * 640 * 4);
  for (let y = 0; y < 640; y++)
    for (let x = 0; x < 640; x++) {
      const v = pixel(x, y);
      data.set([v, v, v, 255], (y * 640 + x) * 4);
    }
  return new ImageData(data, 640, 640);
};
const texture = (x, y) => 110 + 8 * noise(x, y) + 2 * Math.sin(y * 0.8);
let supported;
for (const normal of [
  [1, 0],
  [0, 1],
  [Math.SQRT1_2, Math.SQRT1_2],
]) {
  for (const offset of [-31, 0, 29]) {
    const result = examineNativeSeamProfile(
      make((x, y) => texture(x, y) * ((x - 320) * normal[0] + (y - 320) * normal[1] >= offset ? 1.2 : 1)),
      320,
      320,
      normal,
    );
    assert.equal(result.status, 'stable', JSON.stringify({ normal, offset, result }));
    assert.ok(Math.abs(result.offset - offset) <= 3, 'native location must recover thumbnail offset');
    assert.ok(Math.abs(result.delta - Math.log2(1.2)) < 0.025, JSON.stringify({normal,offset,result}));
    supported = result;
  }
}
for (const [name, pixel] of [
  ['random silk', texture],
  ['smooth illumination', (x, y) => texture(x, y) * 2 ** ((x - 320) / 640)],
  ['thin line', (x, y) => texture(x, y) * (Math.abs(x - 320) < 2 ? 0.8 : 1)],
  ['single strip accident', (x, y) => texture(x, y) * (x >= 320 && y < 255 ? 1.2 : 1)],
])
  assert.equal(examineNativeSeamProfile(make(pixel), 320, 320, [1, 0]).status, 'insufficient', name);
assert.equal(examineNativeSeamProfile(make(texture), 5, 5, [1, 0]).status, 'insufficient');
assert.equal(examineNativeSeamProfile(make(texture), 320, 320, [0, 0]).status, 'insufficient');
const untouched = make((x, y) => texture(x, y) * (x >= 320 ? 1.2 : 1));
const originalPixels = new Uint8ClampedArray(untouched.data);
examineNativeSeamProfile(untouched, 320, 320, [1, 0]);
assert.deepEqual(untouched.data, originalPixels, 'diagnostics must never change source pixels');
assert.equal(summarizeNativeSeam(Array(8).fill({ profile: supported, textureSupported: true })).status, 'supported');
assert.equal(
  summarizeNativeSeam(Array(8).fill({ profile: supported, textureSupported: false })).status,
  'insufficient',
);
assert.equal(
  summarizeNativeSeam(
    Array.from({ length: 8 }, (_, i) => ({
      profile: { ...supported, delta: i % 2 ? -0.26 : 0.26 },
      textureSupported: true,
    })),
  ).status,
  'insufficient',
);
console.log('native seam profile: shifted/diagonal steps, held-out strips, controls, ramps, lines, aggregation passed');
