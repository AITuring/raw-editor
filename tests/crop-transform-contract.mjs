import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { transform } from 'esbuild';

async function loadTypeScriptModule(relativePath) {
  const source = await readFile(resolve(relativePath), 'utf8');
  const { code } = await transform(source, { format: 'esm', loader: 'ts', target: 'es2022' });
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
}

const upright = await loadTypeScriptModule('src/utils/upright.ts');
const crop = await loadTypeScriptModule('src/utils/cropUtils.ts');
const latestQueue = await loadTypeScriptModule('src/utils/latestOnlyAsyncQueue.ts');
const [
  tauriCommands,
  imageProcessingSource,
  cropPanelSource,
  editorSource,
  canvasSource,
  invokeSource,
  processingHookSource,
  adjustmentsSource,
] = await Promise.all([
  readFile(resolve('src-tauri/src/lib.rs'), 'utf8'),
  readFile(resolve('src-tauri/src/image_processing.rs'), 'utf8'),
  readFile(resolve('src/components/panel/right/CropPanel.tsx'), 'utf8'),
  readFile(resolve('src/components/panel/Editor.tsx'), 'utf8'),
  readFile(resolve('src/components/panel/editor/ImageCanvas.tsx'), 'utf8'),
  readFile(resolve('src/components/ui/AppProperties.tsx'), 'utf8'),
  readFile(resolve('src/hooks/useImageProcessing.ts'), 'utf8'),
  readFile(resolve('src/utils/adjustments.ts'), 'utf8'),
]);

assert.match(tauriCommands, /async fn analyze_crop_upright\(/, 'Upright must execute through a native image command');
assert.match(tauriCommands, /generate_handler!\[[\s\S]*analyze_crop_upright,/, 'Upright command must be registered');
assert.match(invokeSource, /AnalyzeCropUpright\s*=\s*'analyze_crop_upright'/, 'frontend invoke name must match Rust');
assert.match(
  cropPanelSource,
  /invoke<UprightAnalysisResult>\(Invokes\.AnalyzeCropUpright/,
  'automatic Upright modes must call native analysis',
);
assert.match(cropPanelSource, /transformVertical:\s*result\.vertical/, 'native Upright output must update geometry');
assert.match(canvasSource, /onUprightGuideAdd\(\{/, 'Guided Upright must send canvas lines into the solver');
assert.match(imageProcessingSource, /pub projection:\s*f32/, 'Projection must be a native geometry parameter');
assert.match(imageProcessingSource, /unproject_geometry_point\(/, 'Projection must alter native inverse sampling');
assert.match(
  processingHookSource,
  /createLatestOnlyAsyncQueue<UncroppedPreviewRequest, ArrayBuffer>/,
  'crop preview requests must be coalesced instead of queueing every slider event',
);
assert.match(
  processingHookSource,
  /useEffect\(\(\) => \(\) => uncroppedPreviewQueue\.cancel\(\)/,
  'Strict Mode cleanup must cancel rather than permanently dispose the crop preview queue',
);
assert.doesNotMatch(
  processingHookSource,
  /\(\) => uncroppedPreviewQueue\.dispose\(\)/,
  'Strict Mode must not leave the memoized crop preview queue disposed',
);
assert.match(
  tauriCommands,
  /let checker = if \(\(x \/ 12\) \+ \(y \/ 12\)\)\.is_multiple_of\(2\)/,
  'transparent crop-preview edges must be composited onto a visible neutral canvas',
);
assert.doesNotMatch(
  imageProcessingSource,
  /compute_lens_auto_crop_scale|auto_crop_scale/,
  'geometry must not silently zoom to hide blank edges',
);
assert.match(
  editorSource,
  /if \(isSliderDragging \|\| \(liveRotation !== null && liveRotation !== undefined\)\) \{\s*return;/,
  'geometry gestures must keep the crop frame stable until release',
);
assert.match(adjustmentsSource, /constrainCrop:\s*false,/, 'blank transform edges must be allowed by default');

const cropResetKeysStart = cropPanelSource.indexOf('const CROP_RESET_KEYS = [');
const cropResetKeysEnd = cropPanelSource.indexOf(
  '] as const satisfies ReadonlyArray<keyof Adjustments>;',
  cropResetKeysStart,
);
assert.ok(cropResetKeysStart >= 0 && cropResetKeysEnd > cropResetKeysStart, 'crop reset keys must be explicit');
const cropResetKeysSource = cropPanelSource.slice(cropResetKeysStart, cropResetKeysEnd);
for (const key of [
  'aspectRatio',
  'constrainCrop',
  'crop',
  'flipHorizontal',
  'flipVertical',
  'orientationSteps',
  'rotation',
  'transformProjection',
  'transformVertical',
  'transformHorizontal',
  'transformRotate',
  'transformAspect',
  'transformScale',
  'transformXOffset',
  'transformYOffset',
]) {
  assert.match(cropResetKeysSource, new RegExp(`'${key}'`), `crop reset must restore ${key}`);
}
assert.doesNotMatch(
  cropResetKeysSource,
  /'transformDistortion'/,
  'crop reset must preserve the independent Optics distortion correction',
);

const cropResetHandlerStart = cropPanelSource.indexOf('const handleResetCrop = useCallback');
const cropResetHandlerEnd = cropPanelSource.indexOf('const isOrientationToggleDisabled', cropResetHandlerStart);
assert.ok(cropResetHandlerStart >= 0 && cropResetHandlerEnd > cropResetHandlerStart, 'crop reset handler must exist');
const cropResetHandlerSource = cropPanelSource.slice(cropResetHandlerStart, cropResetHandlerEnd);
assert.match(
  cropResetHandlerSource,
  /const resetSnapshot = createCropResetSnapshot\(originalAspectRatio\)/,
  'crop reset must restore the complete crop and geometry reset snapshot',
);
assert.match(cropResetHandlerSource, /\.\.\.resetSnapshot,/, 'crop reset snapshot must be applied atomically');
assert.match(cropResetHandlerSource, /isGuidedUprightActive:\s*false/, 'crop reset must exit Guided Upright');
assert.match(cropResetHandlerSource, /isSliderDragging:\s*false/, 'crop reset must end geometry drag state');
assert.match(cropResetHandlerSource, /uprightMode:\s*'off'/, 'crop reset must clear the active Upright mode');
assert.match(
  cropResetHandlerSource,
  /sectionVisibility:\s*\{\s*\.\.\.prev\.sectionVisibility,\s*geometry:\s*true\s*\}/,
  'crop reset must leave reset geometry enabled',
);
assert.ok(
  tauriCommands.indexOf('downscale_f32_image(\n                    patched_image.as_ref()') <
    tauriCommands.indexOf('let warped_image = apply_geometry_warp(preview_source'),
  'crop preview must downscale before the expensive geometry warp',
);

function projectVerticalLine(x, perspective) {
  const project = (y) => {
    const centeredX = x - 0.5;
    const centeredY = y - 0.5;
    const denominator = 1 + (perspective / 50) * centeredY;
    return { x: 0.5 + centeredX / denominator, y: 0.5 + centeredY / denominator };
  };
  return { start: project(0.08), end: project(0.92) };
}

const sourcePerspective = 26;
const left = projectVerticalLine(0.28, sourcePerspective);
const right = projectVerticalLine(0.72, sourcePerspective);
const guided = upright.solveGuidedUpright([
  { id: 'left', axis: 'vertical', ...left },
  { id: 'right', axis: 'vertical', ...right },
]);
assert.ok(Math.abs(guided.vertical + sourcePerspective) < 0.6, JSON.stringify(guided));
assert.ok(Math.abs(guided.rotation) < 0.2, JSON.stringify(guided));

function rotateNormalizedPoint(point, degrees, aspectRatio) {
  const radians = (degrees * Math.PI) / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  const x = (point.x - 0.5) * aspectRatio;
  const y = point.y - 0.5;
  return {
    x: 0.5 + (x * cosine - y * sine) / aspectRatio,
    y: 0.5 + x * sine + y * cosine,
  };
}

const portraitAspect = 0.8;
const portraitRotation = 6;
const portraitGuides = [0.3, 0.7].map((x, index) => ({
  id: `portrait-${index}`,
  axis: 'vertical',
  start: rotateNormalizedPoint({ x, y: 0.1 }, portraitRotation, portraitAspect),
  end: rotateNormalizedPoint({ x, y: 0.9 }, portraitRotation, portraitAspect),
}));
const portraitCorrection = upright.solveGuidedUpright(portraitGuides, portraitAspect);
assert.ok(Math.abs(portraitCorrection.rotation + portraitRotation) < 0.2, JSON.stringify(portraitCorrection));
assert.equal(upright.classifyUprightGuide({ x: 0.1, y: 0.1 }, { x: 0.8, y: 0.7 }, portraitAspect), 'vertical');

const quarterTurn = upright.mapOrientedUprightCorrection(
  { rotation: 3, vertical: 20, horizontal: -12, confidence: 1 },
  1,
  false,
  false,
);
assert.equal(quarterTurn.vertical, -12);
assert.equal(quarterTurn.horizontal, -20);
assert.equal(quarterTurn.rotation, 3);

const identityTransform = {
  orientationSteps: 0,
  flipHorizontal: false,
  flipVertical: false,
  transformDistortion: 0,
  transformProjection: 0,
  transformVertical: 0,
  transformHorizontal: 0,
  transformRotate: 0,
  transformAspect: 0,
  transformScale: 100,
  transformXOffset: 0,
  transformYOffset: 0,
  lensDistortionAmount: 100,
  lensDistortionEnabled: true,
  lensDistortionParams: null,
  sectionVisibility: { geometry: true, optics: true },
};
const fullCrop = { unit: 'px', x: 0, y: 0, width: 1000, height: 800 };
assert.equal(crop.isCropWithinBounds(fullCrop, 1000, 800, 0, true, identityTransform), true);

const scaledTransform = { ...identityTransform, transformScale: 75 };
assert.equal(crop.isCropWithinBounds(fullCrop, 1000, 800, 0, true, scaledTransform), false);
assert.equal(crop.isCropWithinBounds(fullCrop, 1000, 800, 0, false, scaledTransform), true);
const scaledCrop = crop.calculateCenteredCrop(1000, 800, 0, 1.25, 0, true, scaledTransform);
assert.ok(scaledCrop, 'scaled geometry should produce a safe centered crop');
assert.ok(scaledCrop.width >= 740 && scaledCrop.width <= 755, JSON.stringify(scaledCrop));
assert.ok(scaledCrop.height >= 590 && scaledCrop.height <= 605, JSON.stringify(scaledCrop));
assert.equal(crop.isCropWithinBounds(scaledCrop, 1000, 800, 0, true, scaledTransform), true);

const perspectiveTransform = { ...identityTransform, transformVertical: 38, transformHorizontal: -18 };
assert.equal(crop.isCropWithinBounds(fullCrop, 1000, 800, 0, true, perspectiveTransform), false);
const perspectiveCrop = crop.calculateCenteredCrop(1000, 800, 0, 1.25, 0, true, perspectiveTransform);
assert.ok(perspectiveCrop, 'perspective geometry should produce a safe centered crop');
assert.ok(perspectiveCrop.width < 1000 && perspectiveCrop.height < 800, JSON.stringify(perspectiveCrop));
assert.equal(crop.isCropWithinBounds(perspectiveCrop, 1000, 800, 0, true, perspectiveTransform), true);
const constrainedFromFull = crop.resolveCropForConstraintChange(
  1000,
  800,
  0,
  1.25,
  0,
  fullCrop,
  true,
  perspectiveTransform,
);
assert.ok(crop.areCropsApproximatelyEqual(constrainedFromFull, perspectiveCrop, 3));
const restoredBlankCanvas = crop.resolveCropForConstraintChange(
  1000,
  800,
  0,
  1.25,
  0,
  perspectiveCrop,
  false,
  perspectiveTransform,
);
assert.ok(crop.areCropsApproximatelyEqual(restoredBlankCanvas, fullCrop, 1));
const customCrop = { unit: 'px', x: 220, y: 160, width: 500, height: 400 };
assert.deepEqual(
  crop.resolveCropForConstraintChange(1000, 800, 0, 1.25, 0, customCrop, false, perspectiveTransform),
  customCrop,
  'disabling the constraint must preserve a hand-authored crop',
);

const projectionTransform = { ...identityTransform, transformProjection: 100 };
assert.equal(crop.isCropWithinBounds(fullCrop, 1000, 800, 0, true, projectionTransform), false);
const projectionCrop = crop.calculateCenteredCrop(1000, 800, 0, 1.25, 0, true, projectionTransform);
assert.ok(projectionCrop, 'projection correction should produce a safe centered crop');
assert.ok(projectionCrop.width < 900 && projectionCrop.height < 720, JSON.stringify(projectionCrop));
assert.equal(crop.isCropWithinBounds(projectionCrop, 1000, 800, 0, true, projectionTransform), true);

const pendingExecutions = [];
const queueResults = [];
const busyTransitions = [];
const queue = latestQueue.createLatestOnlyAsyncQueue({
  execute(input) {
    return new Promise((resolvePromise) => pendingExecutions.push({ input, resolve: resolvePromise }));
  },
  getKey: (input) => input,
  onResult: (output) => queueResults.push(output),
  onBusyChange: (busy) => busyTransitions.push(busy),
});

queue.submit('first');
queue.submit('obsolete');
queue.submit('latest');
assert.equal(pendingExecutions.length, 1, 'only one native-style request may run at once');
pendingExecutions[0].resolve('first-result');
await new Promise((resolvePromise) => setImmediate(resolvePromise));
assert.equal(pendingExecutions.length, 2, 'the queue should start one follow-up request');
assert.equal(pendingExecutions[1].input, 'latest', 'intermediate slider requests must be discarded');
pendingExecutions[1].resolve('latest-result');
await new Promise((resolvePromise) => setImmediate(resolvePromise));
assert.deepEqual(queueResults, ['first-result', 'latest-result']);
assert.deepEqual(busyTransitions, [true, false]);

queue.submit('cancelled');
assert.equal(pendingExecutions.length, 3);
queue.cancel();
pendingExecutions[2].resolve('stale-result');
await new Promise((resolvePromise) => setImmediate(resolvePromise));
assert.deepEqual(queueResults, ['first-result', 'latest-result'], 'cancelled results must never replace the preview');
queue.submit('resumed');
assert.equal(pendingExecutions.length, 4, 'a Strict Mode cleanup cancellation must leave the queue reusable');
pendingExecutions[3].resolve('resumed-result');
await new Promise((resolvePromise) => setImmediate(resolvePromise));
assert.deepEqual(queueResults, ['first-result', 'latest-result', 'resumed-result']);
queue.dispose();

console.log('Validated crop reset, Upright, projection constraints, and latest-only crop preview scheduling.');
