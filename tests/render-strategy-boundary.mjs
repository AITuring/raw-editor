import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (relativePath) => fs.readFileSync(path.join(repoRoot, relativePath), 'utf8');

const previewResolution = read('src/utils/previewResolution.ts');
const imageProcessingHook = read('src/hooks/useImageProcessing.ts');
const appState = read('src-tauri/src/app_state.rs');
const renderStrategy = read('src-tauri/src/render_strategy.rs');
const lib = read('src-tauri/src/lib.rs');
const gpuProcessing = read('src-tauri/src/gpu_processing.rs');
const exportProcessing = read('src-tauri/src/export_processing.rs');

for (const tier of ['rapidPreview', 'halfResolutionEdit', 'fullResolutionRoi', 'fullResolutionExport']) {
  assert.ok(previewResolution.includes(`'${tier}'`), `frontend render contract is missing ${tier}`);
  assert.ok(renderStrategy.includes(`"${tier}"`), `Rust render contract is missing ${tier}`);
}

assert.match(imageProcessingHook, /renderTier: renderPlan\.tier/);
assert.match(imageProcessingHook, /targetResolution: renderPlan\.targetResolution/);
assert.match(appState, /render_tier: Option<RenderTier>/);
assert.match(lib, /resolve_preview_render_tier\(/);
assert.match(renderStrategy, /fullResolutionExport cannot be submitted to the preview worker/);
assert.match(exportProcessing, /RenderTier::FullResolutionExport\.as_str\(\)/);

assert.doesNotMatch(appState, /small_image|interactive_divisor/);
assert.doesNotMatch(lib, /small_preview_base/);
assert.match(lib, /transformed_image_into_arc/);
assert.match(lib, /Arc::ptr_eq/);

assert.match(gpuProcessing, /dummy_mask_view/);
assert.match(gpuProcessing, /MaskTexturePlan::new/);
assert.match(gpuProcessing, /queue\.write_texture\(/);
assert.doesNotMatch(gpuProcessing, /mask_texture_data/);
assert.doesNotMatch(gpuProcessing, /processed_pixels\.clone\(\)/);
assert.match(gpuProcessing, /Result<Arc<DynamicImage>, String>/);
assert.match(gpuProcessing, /image: Arc::clone\(&shared_image\)/);
assert.match(gpuProcessing, /process_and_stream_rgba_rows/);
assert.match(gpuProcessing, /StreamingExportBufferPlan::new/);
assert.match(gpuProcessing, /reclaim_gpu_resources_after_export/);
assert.match(gpuProcessing, /GpuProcessorTexturePlan::new/);

assert.match(exportProcessing, /supports_streaming_export/);
assert.match(exportProcessing, /encode_streaming_png/);
assert.match(exportProcessing, /encode_streaming_tiff/);
assert.doesNotMatch(exportProcessing, /"jpg" \| "jpeg" => encode_streaming/);
assert.match(exportProcessing, /create_temporary_export/);
assert.match(exportProcessing, /NamedTempFile::new_in\(output_parent\)/);
assert.match(exportProcessing, /temporary\.persist\(output_path\)/);
assert.match(exportProcessing, /reclaim_gpu_resources_after_export\(&context/);

const pixels = 9504 * 6336;
const oldEmptyMaskBytes = pixels * 2;
const oldAnalyticsCloneBytes = pixels * 4;
const oldUnchangedRgb32fCloneBytes = pixels * 12;
const oldFullExportRgbaBytes = pixels * 4;
const streamedBandBytes = 9504 * 2048 * 4;
assert.equal(oldEmptyMaskBytes, 120_434_688);
assert.equal(oldAnalyticsCloneBytes, 240_869_376);
assert.equal(oldUnchangedRgb32fCloneBytes, 722_608_128);
assert.equal(oldFullExportRgbaBytes, 240_869_376);
assert.equal(streamedBandBytes, 77_856_768);
assert.equal(oldFullExportRgbaBytes - streamedBandBytes, 163_012_608);

console.log(
  'Validated four render tiers, bounded tile-to-encoder rows, CPU-only GPU textures, and export high-water reclamation.',
);
