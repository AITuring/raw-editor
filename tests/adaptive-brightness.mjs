import assert from 'node:assert/strict';
import { build } from 'esbuild';
globalThis.ImageData = class { constructor(data, width, height) { Object.assign(this, { data, width, height }); } };
const bundle = await build({ entryPoints: ['src/utils/adaptiveBrightness.ts'], bundle: true, format: 'esm', write: false });
const { analyzeIllumination, applyIllumination } = await import('data:text/javascript;base64,' + Buffer.from(bundle.outputFiles[0].contents).toString('base64'));
function image(w, h, pixel) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set(pixel(x, y, w, h), (y * w + x) * 4);
  return new ImageData(data, w, h);
}
const flat = image(200, 100, () => [100, 100, 100, 255]);
assert.deepEqual(applyIllumination(flat, analyzeIllumination(flat)).data, flat.data);
const gradient = image(240, 160, (x) => { const l = 70 + x / 3; return [l, l, l, 255]; });
const model = analyzeIllumination(gradient);
assert.ok(model.regions.length > analyzeIllumination(flat).regions.length, 'subdivision must follow image variation');
assert.equal(applyIllumination(gradient, model, 0), gradient);
const corrected = applyIllumination(gradient, model, 1);
const range = (im) => im.data[(80 * 240 + 220) * 4] - im.data[(80 * 240 + 20) * 4];
assert.ok(range(corrected) < range(gradient) * .7, 'broad uneven illumination should decrease');
for (let x = 1; x < 240; x++) assert.ok(Math.abs(corrected.data[(80 * 240 + x) * 4] - corrected.data[(80 * 240 + x - 1) * 4]) <= 2, 'no partition boundaries');
const protectedOutput = applyIllumination(gradient, model, 1, [{ x: 0, y: 0, width: .25, height: 1 }]);
for (let y = 0; y < 160; y++) for (let x = 0; x < 60; x++) {
  const i = (y * 240 + x) * 4;
  assert.equal(protectedOutput.data[i], gradient.data[i], 'protected pixels remain original');
}
const transparent = image(20, 20, () => [200, 20, 40, 0]);
assert.deepEqual(applyIllumination(transparent, analyzeIllumination(transparent)).data, transparent.data);
const color = image(200, 100, (x) => x < 100 ? [160, 40, 20, 255] : [20, 40, 160, 255]);
const colorOut = applyIllumination(color, analyzeIllumination(color));
for (let i = 0; i < color.data.length; i += 4) {
  assert.equal(colorOut.data[i + 3], 255);
  assert.ok(Math.abs(colorOut.data[i] * color.data[i + 2] - colorOut.data[i + 2] * color.data[i]) <= 180, 'preserve hue ratios');
}
const large = image(480, 320, (x) => { const l = 70 + Math.floor(x / 2) / 3; return [l, l, l, 255]; });
const exported = applyIllumination(large, model, 1);
for (let y = 0; y < 160; y += 8) for (let x = 0; x < 240; x += 8) {
  assert.ok(Math.abs(exported.data[(y * 2 * 480 + x * 2) * 4] - corrected.data[(y * 240 + x) * 4]) <= 1, 'same normalized field for preview and export');
}
for (const [w, h] of [[1, 1], [1, 50], [50, 1]]) {
  const tiny = image(w, h, () => [80, 80, 80, 120]);
  assert.deepEqual(applyIllumination(tiny, analyzeIllumination(tiny)).data, tiny.data);
}
console.log('adaptive brightness: subdivision, continuity, protection, hue, alpha and export parity passed');
