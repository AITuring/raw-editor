import assert from 'node:assert/strict';
import { build } from 'esbuild';
globalThis.ImageData = class {
  constructor(data, width, height) {
    Object.assign(this, { data, width, height });
  }
};
const b = await build({ entryPoints: ['src/utils/nativeSeamTexture.ts'], bundle: true, format: 'esm', write: false });
const { examineNativeTexture, summarizeNativeTexture } = await import(
  'data:text/javascript;base64,' + Buffer.from(b.outputFiles[0].contents).toString('base64')
);
function make(pixel) {
  const data = new Uint8ClampedArray(320 * 320 * 4);
  for (let y = 0; y < 320; y++)
    for (let x = 0; x < 320; x++) {
      const v = pixel(x, y);
      data.set([v, v, v, 255], (y * 320 + x) * 4);
    }
  return new ImageData(data, 320, 320);
}
const noise = (x, y) => {
  let z = Math.imul(x + 1, 374761393) ^ Math.imul(y + 1, 668265263);
  z = Math.imul(z ^ (z >>> 13), 1274126177);
  return ((z ^ (z >>> 16)) >>> 0) / 4294967295 - 0.5;
};
const texture = (x, y) => 110 + 12 * noise(x, y) + 4 * Math.sin(y * 0.8);
const check = (fn) => examineNativeTexture(make(fn), 160, 160, [1, 0]);
const same = check(texture);
assert.equal(same.status, 'supported', 'random fibres need not have matching phase');
for (const gain of [0.8, 1, 1.2])
  assert.equal(
    check((x, y) => texture(x, y) * (x < 160 ? 1 : gain)).status,
    'supported',
    'stationary texture across exposure step',
  );
assert.equal(
  check((x, y) => 110 + (x < 160 ? 15 * Math.sin(y * 0.8) : 15 * Math.sin(x * 0.8))).status,
  'inconsistent',
  'changed texture orientation',
);
assert.equal(check(() => 110).status, 'insufficient', 'blank material is not evidence');
assert.equal(
  check((x, y) => (x < 160 ? texture(x, y) : 110)).status,
  'insufficient',
  'missing texture evidence is not confirmation',
);
const image = make(texture);
assert.equal(examineNativeTexture(image, 5, 5, [1, 0]).status, 'insufficient');
assert.equal(summarizeNativeTexture([same, same, same, same]).status, 'supported');
for (const offset of [-4, 4]) {
  const stepped = make((x, y) => texture(x, y) * (x < 160 + offset ? 1 : 1.2));
  assert.equal(
    examineNativeTexture(stepped, 160, 160, [1, 0]).status,
    'supported',
    'position uncertainty must not sample the exposure edge',
  );
  assert.equal(
    examineNativeTexture(stepped, 160, 160, [-2, 0]).status,
    'supported',
    'normal length and sign do not change the texture decision',
  );
}
assert.equal(
  check((x, y) => 100 * Math.exp(x * 0.003 + y * 0.002)).status,
  'insufficient',
  'a smooth local slope is not texture',
);
assert.notEqual(
  check((x, y) => 110 + (texture(x, y) - 110) * (x < 160 ? 1 : 0.2)).status,
  'supported',
  'grain contrast collapse must not pass',
);
assert.equal(examineNativeTexture(image, 160, 160, [NaN, 1]).status, 'insufficient');
const transparent = make(texture);
transparent.data.fill(0);
assert.equal(examineNativeTexture(transparent, 160, 160, [1, 0]).status, 'insufficient');
console.log('native texture controls: stochastic fibres, gain invariance, orientation, missing support passed');
