import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const bundled = await build({
  entryPoints: [path.join(repoRoot, 'src/utils/previewResolution.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'node20',
  write: false,
});
const moduleSource = Buffer.from(bundled.outputFiles[0].contents).toString('base64');
const { calculatePreviewTargetResolution, snapImageTranslationToDevicePixels } = await import(
  `data:text/javascript;base64,${moduleSource}`
);

const source = { width: 5727, height: 7637 };
const fit = { width: 768, height: 1024 };

const at100Percent = calculatePreviewTargetResolution({
  displaySize: { width: source.width / 2, height: source.height / 2 },
  baseRenderSize: fit,
  originalSize: source,
  editorPreviewResolution: 1920,
  enableZoomHifi: true,
  useFullDpiRendering: false,
  highResZoomMultiplier: 1,
  devicePixelRatio: 2,
});
assert.equal(at100Percent, source.height, 'Retina 100% zoom must request the full source resolution');

const at112Percent = calculatePreviewTargetResolution({
  displaySize: { width: (source.width * 1.12) / 2, height: (source.height * 1.12) / 2 },
  baseRenderSize: fit,
  originalSize: source,
  editorPreviewResolution: 1920,
  enableZoomHifi: true,
  useFullDpiRendering: false,
  highResZoomMultiplier: 0.75,
  devicePixelRatio: 2,
});
assert.equal(
  at112Percent,
  source.height,
  'quality multipliers must not make a settled zoomed preview undersample device pixels',
);

const fitResolution = calculatePreviewTargetResolution({
  displaySize: fit,
  baseRenderSize: fit,
  originalSize: source,
  editorPreviewResolution: 1920,
  enableZoomHifi: true,
  useFullDpiRendering: false,
  highResZoomMultiplier: 1,
  devicePixelRatio: 2,
});
assert.ok(fitResolution >= 1920, 'fit-to-window rendering must honor the configured preview baseline');
assert.ok(fitResolution < source.height, 'fit-to-window rendering should not eagerly decode at full display size');

const disabledResolution = calculatePreviewTargetResolution({
  displaySize: { width: source.width / 2, height: source.height / 2 },
  baseRenderSize: fit,
  originalSize: source,
  editorPreviewResolution: 1920,
  enableZoomHifi: false,
  devicePixelRatio: 2,
});
assert.equal(disabledResolution, 1920, 'disabling zoom HiFi must retain the configured fixed preview size');

const smallSourceResolution = calculatePreviewTargetResolution({
  displaySize: { width: 1600, height: 1200 },
  baseRenderSize: { width: 1600, height: 1200 },
  originalSize: { width: 1200, height: 900 },
  editorPreviewResolution: 1920,
  enableZoomHifi: true,
  devicePixelRatio: 2,
});
assert.equal(smallSourceResolution, 1200, 'preview requests must never exceed the available source detail');

const snapped = snapImageTranslationToDevicePixels({
  positionX: 13.17,
  positionY: -8.36,
  scale: 2.25,
  imageOffsetX: 101.3,
  imageOffsetY: 47.7,
  devicePixelRatio: 2,
});
const snappedPhysicalX = (snapped.positionX + 101.3 * 2.25) * 2;
const snappedPhysicalY = (snapped.positionY + 47.7 * 2.25) * 2;
assert.ok(Math.abs(snappedPhysicalX - Math.round(snappedPhysicalX)) < 1e-9);
assert.ok(Math.abs(snappedPhysicalY - Math.round(snappedPhysicalY)) < 1e-9);
assert.ok(Math.abs(snapped.positionX - 13.17) <= 0.25);
assert.ok(Math.abs(snapped.positionY + 8.36) <= 0.25);

console.log('Validated physical-pixel preview resolution at fit, Retina 100%, and zoomed states.');
