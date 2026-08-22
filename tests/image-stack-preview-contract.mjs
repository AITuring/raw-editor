import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const previewSource = fs.readFileSync(path.join(repoRoot, 'src/components/modals/ImageStackResultPreview.tsx'), 'utf8');
const reusablePreviewSource = fs.readFileSync(
  path.join(repoRoot, 'src/components/preview/ZoomableImagePreview.tsx'),
  'utf8',
);
const previewModuleSource = fs.readFileSync(path.join(repoRoot, 'src/components/preview/index.ts'), 'utf8');
const reusablePreviewStyleSource = fs.readFileSync(
  path.join(repoRoot, 'src/components/preview/ZoomableImagePreview.css'),
  'utf8',
);
const editorSource = fs.readFileSync(path.join(repoRoot, 'src/components/panel/Editor.tsx'), 'utf8');
const screenPreviewSource = fs.readFileSync(
  path.join(repoRoot, 'src/components/preview/ScreenSpacePreview.tsx'),
  'utf8',
);
const screenTransformSource = fs.readFileSync(
  path.join(repoRoot, 'src/hooks/useScreenSpacePreviewTransform.ts'),
  'utf8',
);
const pipelineSource = fs.readFileSync(path.join(repoRoot, 'src/utils/imageStackPipeline.ts'), 'utf8');
const rustStackSource = fs.readFileSync(path.join(repoRoot, 'src-tauri/src/image_stack.rs'), 'utf8');
const productivityActionsSource = fs.readFileSync(path.join(repoRoot, 'src/hooks/useProductivityActions.ts'), 'utf8');
const listenerSource = fs.readFileSync(path.join(repoRoot, 'src/hooks/useTauriListeners.ts'), 'utf8');

const frontendPipelineVersion = pipelineSource.match(/IMAGE_STACK_PIPELINE_VERSION\s*=\s*'([^']+)'/)?.[1];
const backendPipelineVersion = rustStackSource.match(/IMAGE_STACK_PIPELINE_VERSION:\s*&str\s*=\s*"([^"]+)"/)?.[1];

assert.ok(frontendPipelineVersion, 'frontend image-stack pipeline version must be declared');
assert.equal(
  frontendPipelineVersion,
  backendPipelineVersion,
  'frontend and backend image-stack pipeline versions must change together',
);
assert.match(productivityActionsSource, /pipelineVersion:\s*IMAGE_STACK_PIPELINE_VERSION/);
assert.match(listenerSource, /pipelineVersion\s*!==\s*IMAGE_STACK_PIPELINE_VERSION/);

const cleanupMatch = reusablePreviewSource.match(/useEffect\(\s*\(\) => \(\) => \{([\s\S]*?)\}\s*,\s*\[\]\s*,?\s*\);/);

assert.ok(cleanupMatch, 'reusable preview must keep an explicit unmount cleanup');

const cleanup = cleanupMatch[1];

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

assert.match(reusablePreviewSource, /data-zoomable-image-preview-toolbar/);
assert.match(reusablePreviewSource, /closest\('\[data-zoomable-image-preview-toolbar\]'\)/);
assert.match(reusablePreviewSource, /onClick=\{\(\) => stepZoom\(-1\)\}/);
assert.match(reusablePreviewSource, /onClick=\{\(\) => stepZoom\(1\)\}/);

const bundled = await build({
  entryPoints: [path.join(repoRoot, 'src/components/preview/imagePreviewZoom.ts')],
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

const previewGeometryBundle = await build({
  entryPoints: [path.join(repoRoot, 'src/utils/previewResolution.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'node20',
  write: false,
});
const previewGeometryModuleSource = Buffer.from(previewGeometryBundle.outputFiles[0].contents).toString('base64');
const { calculateScreenSpacePreviewGeometry } = await import(
  `data:text/javascript;base64,${previewGeometryModuleSource}`
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

const realResultWidth = 6333;
const realResultHeight = 9501;
const realFitWidth = 640;
const realFitHeight = (realFitWidth * realResultHeight) / realResultWidth;
const realDevicePixelRatio = 2;
const realFitPixelZoom = calculateFitPixelZoom({
  devicePixelRatio: realDevicePixelRatio,
  displayHeight: realFitHeight,
  displayWidth: realFitWidth,
  sourceHeight: realResultHeight,
  sourceWidth: realResultWidth,
});
const fiftySixPercentTransformScale = 0.56 / realFitPixelZoom;
const fiftySixPercentGeometry = calculateScreenSpacePreviewGeometry({
  imageHeight: realFitHeight,
  imageOffsetX: 0,
  imageOffsetY: 0,
  imageWidth: realFitWidth,
  positionX: 0,
  positionY: 0,
  transformScale: fiftySixPercentTransformScale,
});

assert.equal(
  Math.round(fiftySixPercentGeometry.width * realDevicePixelRatio),
  Math.round(realResultWidth * 0.56),
  'the shared settled renderer must allocate the physical screen width represented by 56% output-pixel zoom',
);
assert.equal(
  Math.round(fiftySixPercentGeometry.height * realDevicePixelRatio),
  Math.round(realResultHeight * 0.56),
  'the shared settled renderer must allocate the physical screen height represented by 56% output-pixel zoom',
);

assert.match(previewSource, /import \{ ZoomableImagePreview \} from '\.\.\/preview'/);
assert.match(previewSource, /<ZoomableImagePreview/);
assert.doesNotMatch(previewSource, /requestAnimationFrame|transformRef|stepZoom/);
assert.match(previewModuleSource, /export \{ default as ZoomableImagePreview \}/);
assert.match(previewModuleSource, /ZoomableImagePreviewHandle/);
assert.match(reusablePreviewSource, /import ScreenSpacePreview from '\.\/ScreenSpacePreview'/);
assert.match(reusablePreviewSource, /useScreenSpacePreviewTransform\(\{/);
assert.match(reusablePreviewSource, /children\?: ReactNode/);
assert.match(reusablePreviewSource, /toolbarEnd\?: ReactNode/);
assert.match(reusablePreviewSource, /import '\.\/ZoomableImagePreview\.css'/);
assert.match(reusablePreviewStyleSource, /\.zoomable-image-preview\[data-detail-ready='true'\]/);
assert.doesNotMatch(reusablePreviewSource, /ImageStack|modals\.imageStack/);
assert.match(editorSource, /useScreenSpacePreviewTransform\(\{/);
assert.match(screenPreviewSource, /settled preview has no CSS\s+ \* scale transform at all/);
assert.match(screenTransformSource, /element\.style\.width = `\$\{geometry\.width\}px`/);
assert.match(screenTransformSource, /element\.style\.transform = 'none'/);
assert.match(screenTransformSource, /element\.style\.transform = `matrix\(/);

console.log(
  'Validated image-stack pipeline handshake, reusable preview boundaries, cleanup, toolbar routing, output-pixel zoom semantics, and the shared editor renderer.',
);
