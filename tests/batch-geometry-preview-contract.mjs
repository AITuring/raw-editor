import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const modalSource = fs.readFileSync(path.join(repoRoot, 'src/components/modals/BatchGeometryModal.tsx'), 'utf8');
const styles = fs.readFileSync(path.join(repoRoot, 'src/styles.css'), 'utf8');
const rustSource = fs.readFileSync(path.join(repoRoot, 'src-tauri/src/export_processing.rs'), 'utf8');

assert.match(modalSource, /comparePositionFromClientX/);
assert.match(modalSource, /comparePositionFromKey/);
assert.match(modalSource, /role="slider"/);
assert.match(modalSource, /onPointerDown=\{handleComparePointerDown\}/);
assert.match(modalSource, /onPointerMove=\{handleComparePointerMove\}/);
assert.match(modalSource, /type="range"/);
assert.match(modalSource, /onWheel=\{handlePreviewWheel\}/);
assert.match(modalSource, /handlePanPointerMove/);
assert.match(modalSource, /PreviewBatchGeometryCorrection/);
assert.match(modalSource, /setPreviewsStale\(true\)/);
assert.match(modalSource, /setProcessingPhase\('analysis'\)/);
assert.match(modalSource, /profileDisabled/);
assert.match(modalSource, /showGuides/);
assert.match(modalSource, /onToggleGuides/);

const previewEdge = Number(rustSource.match(/BATCH_GEOMETRY_PREVIEW_EDGE:\s*u32\s*=\s*(\d+)/)?.[1]);
const analysisEdge = Number(rustSource.match(/CONTENT_ORIENTATION_ANALYSIS_EDGE:\s*u32\s*=\s*(\d+)/)?.[1]);
assert.ok(previewEdge > analysisEdge, 'display previews must have more detail than model analysis previews');
assert.match(rustSource, /overlay_batch_geometry_grid/);
assert.match(rustSource, /render_batch_geometry_preview_images\(.*show_guides/s);
assert.match(rustSource, /show_guides: Option<bool>/);

for (const selector of [
  '.batch-geometry-preview-stage',
  '.batch-geometry-compare-line',
  '.batch-geometry-compare-control',
  '.batch-geometry-preview-facts',
]) {
  assert.ok(styles.includes(selector), `${selector} must keep the visual review surface`);
}

const dividerRule = styles.match(/\.batch-geometry-compare-line\s*\{([^}]*)\}/s)?.[1] ?? '';
assert.match(dividerRule, /width:\s*44px/);
assert.match(dividerRule, /touch-action:\s*none/);
assert.doesNotMatch(dividerRule, /pointer-events:\s*none/);

console.log('Batch geometry live preview contract passed.');
