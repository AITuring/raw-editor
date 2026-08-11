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
const [tauriCommands, cropPanelSource, canvasSource, invokeSource] = await Promise.all([
  readFile(resolve('src-tauri/src/lib.rs'), 'utf8'),
  readFile(resolve('src/components/panel/right/CropPanel.tsx'), 'utf8'),
  readFile(resolve('src/components/panel/editor/ImageCanvas.tsx'), 'utf8'),
  readFile(resolve('src/components/ui/AppProperties.tsx'), 'utf8'),
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

console.log('Validated Guided Upright solving and geometry-aware crop constraints.');
