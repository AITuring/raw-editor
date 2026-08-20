import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const previewSource = fs.readFileSync(path.join(repoRoot, 'src/components/modals/ImageStackResultPreview.tsx'), 'utf8');

const cleanupStart = previewSource.indexOf('useEffect(\n    () => () => {');
const cleanupEnd = previewSource.indexOf('\n    },\n    [],\n  );', cleanupStart);

assert.ok(cleanupStart >= 0 && cleanupEnd > cleanupStart, 'image-stack preview must keep an explicit unmount cleanup');

const cleanup = previewSource.slice(cleanupStart, cleanupEnd);

assert.match(cleanup, /cancelAnimationFrame\(renderFrameRef\.current\)/);
assert.match(
  cleanup,
  /cancelAnimationFrame\(renderFrameRef\.current\);\s*renderFrameRef\.current = null;/,
  'cancelled animation frames must release their scheduling guard so React StrictMode can schedule the remount frame',
);

for (const timerRef of ['settleTimerRef', 'transitionTimerRef']) {
  assert.match(cleanup, new RegExp(`window\\.clearTimeout\\(${timerRef}\\.current\\)`));
  assert.match(
    cleanup,
    new RegExp(`window\\.clearTimeout\\(${timerRef}\\.current\\);\\s*${timerRef}\\.current = null;`),
    `${timerRef} must not retain a cancelled timer handle after effect cleanup`,
  );
}

assert.match(previewSource, /data-image-stack-preview-toolbar/);
assert.match(previewSource, /closest\('\[data-image-stack-preview-toolbar\]'\)/);
assert.match(previewSource, /onClick=\{\(\) => stepZoom\(-1\)\}/);
assert.match(previewSource, /onClick=\{\(\) => stepZoom\(1\)\}/);

const bundled = await build({
  entryPoints: [path.join(repoRoot, 'src/utils/imageStackZoom.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'node20',
  write: false,
});
const moduleSource = Buffer.from(bundled.outputFiles[0].contents).toString('base64');
const { calculateFitPixelZoom, calculateMaxTransformScale, calculatePixelZoom, resolveZoomStep } = await import(
  `data:text/javascript;base64,${moduleSource}`
);

const sourceWidth = 10013;
const sourceHeight = 6281;
const displayWidth = 1000;
const displayHeight = (displayWidth * sourceHeight) / sourceWidth;
const fitPixelZoom = calculateFitPixelZoom({
  devicePixelRatio: 2,
  displayHeight,
  displayWidth,
  sourceHeight,
  sourceWidth,
});
const oneHundredPercentScale = 1 / fitPixelZoom;
const maximumScale = calculateMaxTransformScale(fitPixelZoom);

assert.equal(Math.round(fitPixelZoom * 100), 20, 'fit view must report its real output-pixel zoom');
assert.equal(
  Math.round(calculatePixelZoom(oneHundredPercentScale, fitPixelZoom) * 100),
  100,
  'one output pixel per physical display pixel must report 100%, not the internal fit-relative transform',
);
assert.equal(
  Math.round(calculatePixelZoom(maximumScale, fitPixelZoom) * 100),
  800,
  'the preview must allow deliberate inspection beyond 100%',
);
assert.equal(
  Math.round(calculatePixelZoom(resolveZoomStep(1, 1, fitPixelZoom, maximumScale), fitPixelZoom) * 100),
  25,
  'zoom-in from a 20% fit view must visibly advance to 25%',
);
assert.equal(
  Math.round(
    calculatePixelZoom(resolveZoomStep(oneHundredPercentScale, 1, fitPixelZoom, maximumScale), fitPixelZoom) * 100,
  ),
  125,
  'zoom-in must advance from 100% to the next visible stop',
);
assert.equal(
  Math.round(
    calculatePixelZoom(resolveZoomStep(oneHundredPercentScale, -1, fitPixelZoom, maximumScale), fitPixelZoom) * 100,
  ),
  75,
  'zoom-out must retreat from 100% to the prior visible stop',
);

console.log('Validated image-stack preview cleanup, toolbar click routing, and output-pixel zoom semantics.');
