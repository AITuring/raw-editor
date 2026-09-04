import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const componentSource = fs.readFileSync(path.join(repoRoot, 'src/components/modals/StyleTransferModal.tsx'), 'utf8');
const styles = fs.readFileSync(path.join(repoRoot, 'src/styles.css'), 'utf8');

const bundled = await build({
  entryPoints: [path.join(repoRoot, 'src/utils/compareSlider.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'node20',
  write: false,
});
const moduleSource = Buffer.from(bundled.outputFiles[0].contents).toString('base64');
const { comparePositionFromClientX, comparePositionFromKey } = await import(
  `data:text/javascript;base64,${moduleSource}`
);

const styleTransferBundle = await build({
  entryPoints: [path.join(repoRoot, 'src/utils/styleTransfer.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'node20',
  write: false,
});
const styleTransferModuleSource = Buffer.from(styleTransferBundle.outputFiles[0].contents).toString('base64');
const { analyzeStyleTransfer, applyStyleTransfer, cloneImageData } = await import(
  `data:text/javascript;base64,${styleTransferModuleSource}`
);

assert.equal(comparePositionFromClientX(300, 100, 400), 0.5);
assert.equal(comparePositionFromClientX(0, 100, 400), 0, 'pointer positions before the preview must clamp left');
assert.equal(comparePositionFromClientX(800, 100, 400), 1, 'pointer positions after the preview must clamp right');
assert.equal(comparePositionFromClientX(200, 100, 0), null, 'zero-width previews must be ignored');

assert.equal(comparePositionFromKey(0.5, 'ArrowRight'), 0.51);
assert.equal(comparePositionFromKey(0.5, 'ArrowLeft', true), 0.4);
assert.equal(comparePositionFromKey(0.96, 'PageUp'), 1);
assert.equal(comparePositionFromKey(0.04, 'PageDown'), 0);
assert.equal(comparePositionFromKey(0.5, 'Home'), 0);
assert.equal(comparePositionFromKey(0.5, 'End'), 1);
assert.equal(comparePositionFromKey(0.5, 'Enter'), null);

const originalPixels = new Uint8ClampedArray([24, 48, 72, 255, 96, 120, 144, 255]);
const originalImage = { data: originalPixels, height: 1, width: 2 };
const originalSnapshot = new Uint8ClampedArray(originalPixels);
const styledImage = cloneImageData(originalImage);

assert.notStrictEqual(styledImage, originalImage, 'styled preview must use a different image object');
assert.notStrictEqual(styledImage.data, originalImage.data, 'styled preview must use a different pixel buffer');
applyStyleTransfer(
  styledImage,
  {
    channelOffset: [0.2, 0.1, -0.1],
    channelScale: [1.4, 1.2, 0.8],
    hueShift: 0,
    mode: 'distribution',
    referenceMean: [0.7, 0.6, 0.4],
    saturationScale: 1,
    targetMean: [0.3, 0.4, 0.5],
    targetValueMean: 0.5,
    valueContrast: 1,
    valueOffset: 0,
    valueScale: 1,
  },
  1,
);
assert.deepEqual(originalImage.data, originalSnapshot, 'rendering the styled preview must not mutate original pixels');
assert.notDeepEqual(styledImage.data, originalSnapshot, 'the styled preview fixture must actually change');

const makeGrayTexture = (low, high) => {
  const data = new Uint8ClampedArray(64 * 64 * 4);
  for (let y = 0; y < 64; y += 1) {
    for (let x = 0; x < 64; x += 1) {
      const offset = (y * 64 + x) * 4;
      const value = (x + y) % 2 === 0 ? low : high;
      data[offset] = value;
      data[offset + 1] = value;
      data[offset + 2] = value;
      data[offset + 3] = 255;
    }
  }
  return { data, height: 64, width: 64 };
};
const meanHorizontalGradient = (image) => {
  let total = 0;
  let count = 0;
  for (let y = 0; y < image.height; y += 1) {
    for (let x = 1; x < image.width; x += 1) {
      const offset = (y * image.width + x) * 4;
      total += Math.abs(image.data[offset] - image.data[offset - 4]);
      count += 1;
    }
  }
  return total / count;
};
const texturedTarget = makeGrayTexture(96, 112);
const highContrastReference = makeGrayTexture(32, 224);

for (const mode of ['mood', 'distribution']) {
  const transform = analyzeStyleTransfer(highContrastReference, texturedTarget, mode);
  const naturallyStyled = cloneImageData(texturedTarget);
  applyStyleTransfer(naturallyStyled, transform, 1);
  const gradientRatio = meanHorizontalGradient(naturallyStyled) / meanHorizontalGradient(texturedTarget);
  assert.ok(gradientRatio <= 1.25, `${mode} preview amplified detail contrast by ${gradientRatio.toFixed(3)}x`);
}

for (const pointerContract of [
  'onPointerDown={handleComparePointerDown}',
  'onPointerMove={handleComparePointerMove}',
  'onPointerUp={handleComparePointerEnd}',
  'setPointerCapture(event.pointerId)',
  'role="slider"',
  'tabIndex={0}',
]) {
  assert.ok(componentSource.includes(pointerContract), `comparison divider must keep ${pointerContract}`);
}

assert.ok(
  componentSource.includes('onChange={(event) => setComparePosition(Number(event.target.value) / 100)}'),
  'the fallback range control must update the shared comparison position',
);
assert.ok(
  componentSource.includes('value={Math.round(comparePosition * 100)}'),
  'the fallback range control must stay synchronized with direct dragging',
);
assert.ok(
  componentSource.includes('const resultData = cloneImageData(targetData);'),
  'the generated canvas must render from a cloned target buffer',
);

const dividerRule = styles.match(/\.style-transfer-compare-line\s*\{([^}]*)\}/s)?.[1] ?? '';
assert.match(dividerRule, /width:\s*44px/, 'comparison divider must keep a usable pointer hit target');
assert.match(dividerRule, /touch-action:\s*none/, 'comparison divider must support touch dragging');
assert.doesNotMatch(dividerRule, /pointer-events:\s*none/, 'comparison divider must remain interactive');

console.log('Style-transfer comparison interaction contract passed.');
