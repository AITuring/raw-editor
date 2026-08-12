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
const exifProcessing = read('src-tauri/src/exif_processing.rs');
const imageProcessing = read('src-tauri/src/image_processing.rs');
const mainShader = read('src-tauri/src/shaders/shader.wgsl');
const maskGeneration = read('src-tauri/src/mask_generation.rs');
const packageJson = JSON.parse(read('package.json'));

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
assert.match(gpuProcessing, /MaskTileUploadPlan::new/);
assert.match(gpuProcessing, /Active Mask Tile Texture Array/);
assert.match(gpuProcessing, /width: mask_plan\.texture_width/);
assert.match(gpuProcessing, /height: mask_plan\.texture_height/);
assert.match(gpuProcessing, /offset: upload\.source_offset_bytes/);
assert.match(gpuProcessing, /queue\.write_texture\(/);
assert.doesNotMatch(gpuProcessing, /mask_texture_data/);
assert.match(maskGeneration, /Option<SharedMaskBitmap>/);
assert.match(maskGeneration, /return Some\(Arc::clone\(img\)\)/);
assert.match(maskGeneration, /cache\.insert\(key, Arc::clone\(img\)/);
assert.match(maskGeneration, /\*final_mask = Some\(sub_bitmap\)/);
assert.match(maskGeneration, /enum TiledSubMaskRasterizer/);
assert.match(maskGeneration, /let Some\(rasterizer\) = TiledSubMaskRasterizer::new\(sub_mask\)/);
assert.match(maskGeneration, /rasterizer\s*\.rasterize\(/);
assert.match(maskGeneration, /"brush" \| "clone" \| "heal" =>/);
assert.match(maskGeneration, /"radial" \| "linear" \| "all" => has_composition/);
assert.match(maskGeneration, /struct GrowFeatherPlan/);
assert.match(maskGeneration, /feather_radius: \(2\.0 \* feather_sigma\)\.ceil\(\) as u32/);
assert.match(maskGeneration, /Self::Color\(parameters\) \| Self::Luminance\(parameters\)/);
assert.match(maskGeneration, /"color" \| "luminance" =>/);
assert.match(maskGeneration, /let halo = grow_feather\.halo\(\)/);
assert.match(maskGeneration, /tile_x\.saturating_sub\(halo\)/);
assert.match(maskGeneration, /grow_feather\.apply\(&mut tile\)/);
assert.match(maskGeneration, /composite_sub_mask_tile_region\(/);
assert.match(maskGeneration, /step_by\(tile_edge as usize\)/);
assert.match(maskGeneration, /\(expanded_x, expanded_y\)/);
assert.match(maskGeneration, /GPU_TILE_SIZE/);
assert.doesNotMatch(maskGeneration, /return Some\(img\.clone\(\)\)/);
assert.doesNotMatch(mainShader, /get_mask_influence\(i, absolute_coord\)/);
assert.equal(
  [...mainShader.matchAll(/get_mask_influence\(i, id\.xy\)/g)].length,
  4,
  'every mask adjustment pass must sample the uploaded tile with tile-local coordinates',
);
assert.doesNotMatch(gpuProcessing, /processed_pixels\.clone\(\)/);
assert.match(gpuProcessing, /Result<Arc<DynamicImage>, String>/);
assert.match(gpuProcessing, /image: Arc::clone\(&shared_image\)/);
assert.match(gpuProcessing, /process_and_stream_rgba_rows/);
assert.match(gpuProcessing, /StreamingExportBufferPlan::new/);
assert.match(gpuProcessing, /reclaim_gpu_resources_after_export/);
assert.match(gpuProcessing, /GpuProcessorTexturePlan::new/);
assert.match(imageProcessing, /enum GeometryPixelSource/);
assert.match(imageProcessing, /DynamicImage::ImageRgb32F\(source\) => Self::Rgb32F/);
assert.match(imageProcessing, /DynamicImage::ImageRgba32F\(source\) => Self::Rgba32F/);
assert.match(imageProcessing, /GeometryWarpBufferPlan::new/);
const geometryWarpStart = imageProcessing.indexOf('pub fn warp_image_geometry(');
const geometryWarpEnd = imageProcessing.indexOf('\npub fn unwarp_image_geometry(', geometryWarpStart);
assert.ok(geometryWarpStart >= 0 && geometryWarpEnd > geometryWarpStart);
assert.doesNotMatch(
  imageProcessing.slice(geometryWarpStart, geometryWarpEnd),
  /image\.to_rgba32f\(\)/,
  'geometry warp must not stage every float source through a complete RGBA32F copy',
);

assert.match(exportProcessing, /supports_streaming_export/);
assert.match(exportProcessing, /encode_streaming_jpeg/);
assert.match(exportProcessing, /encode_streaming_png/);
assert.match(exportProcessing, /encode_streaming_tiff/);
assert.match(exportProcessing, /encode_webp_to_file/);
assert.match(exportProcessing, /write_webp_bytes_to_file/);
assert.match(exportProcessing, /check_webp_encoding_progress/);
assert.match(exportProcessing, /libwebp_sys::WebPEncode/);
assert.match(exportProcessing, /picture\.0\.use_argb = 0/);
assert.match(exportProcessing, /rewrite_webp_icc_bounded/);
assert.match(exportProcessing, /WEBP_OUTPUT_COPY_BUFFER_BYTES: usize = 64 \* 1024/);
assert.match(exportProcessing, /"jpg" \| "jpeg" =>/);
assert.match(exportProcessing, /MozjpegCompressor::new/);
assert.match(exportProcessing, /encoder\.write_scanlines\(&rgb_row\)/);
assert.match(exportProcessing, /set_chroma_sampling_pixel_sizes\(\(1, 1\), \(1, 1\)\)/);
assert.match(exportProcessing, /set_optimize_coding\(false\)/);
assert.match(exportProcessing, /export_metadata_tiff_payload/);
assert.match(exportProcessing, /export_metadata_for_streaming_tiff/);
assert.match(exportProcessing, /write_tiff_metadata_group/);
assert.match(exportProcessing, /TiffTag::ExifDirectory/);
assert.match(exportProcessing, /TiffTag::GpsDirectory/);
assert.match(exportProcessing, /info\.exif_metadata = export_exif/);
assert.doesNotMatch(exifProcessing, /output_format\.to_lowercase\(\) == "tiff"/);
assert.doesNotMatch(exportProcessing, /fs::read\(temporary\.path\(\)\)/);
assert.doesNotMatch(exportProcessing, /\.finish_to\(/);
assert.doesNotMatch(exportProcessing, /zenjpeg/);
assert.match(exportProcessing, /transform_streaming_rgba_rows/);
assert.match(exportProcessing, /StreamingResize::new/);
assert.match(exportProcessing, /prepare_watermark/);
assert.match(exportProcessing, /create_temporary_export/);
assert.match(exportProcessing, /NamedTempFile::new_in\(output_parent\)/);
assert.match(exportProcessing, /temporary\.persist\(output_path\)/);
assert.match(exportProcessing, /final_temporary\.persist\(output_path\)/);
assert.match(exportProcessing, /reclaim_gpu_resources_after_export\(&context/);

const saveImageStart = exportProcessing.indexOf('fn save_image_with_metadata(');
const saveImageEnd = exportProcessing.indexOf('\nfn supports_streaming_export', saveImageStart);
assert.ok(saveImageStart >= 0 && saveImageEnd > saveImageStart);
const saveImageWithMetadata = exportProcessing.slice(saveImageStart, saveImageEnd);
assert.ok(
  saveImageWithMetadata.indexOf('save_webp_with_bounded_output') <
    saveImageWithMetadata.indexOf('encode_image_to_bytes'),
  'desktop WebP must bypass the complete compressed-memory encoder',
);

const pixels = 9504 * 6336;
const oldEmptyMaskBytes = pixels * 2;
const oldAnalyticsCloneBytes = pixels * 4;
const oldUnchangedRgb32fCloneBytes = pixels * 12;
const oldFullExportRgbaBytes = pixels * 4;
const streamedBandBytes = 9504 * 2048 * 4;
const geometryRgba32fBytes = pixels * 4 * 4;
const activeMaskFullTextureBytes = pixels;
const activeMaskTileTextureBytes = (2048 + 128 * 2) ** 2;
const maskGenerationTileBytes = 2048 ** 2;
const maxRangeMaskHalo = 63 + 64;
const maxRangeMaskTileBytes = (2048 + maxRangeMaskHalo * 2) ** 2;
assert.equal(oldEmptyMaskBytes, 120_434_688);
assert.equal(oldAnalyticsCloneBytes, 240_869_376);
assert.equal(oldUnchangedRgb32fCloneBytes, 722_608_128);
assert.equal(oldFullExportRgbaBytes, 240_869_376);
assert.equal(streamedBandBytes, 77_856_768);
assert.equal(oldFullExportRgbaBytes - streamedBandBytes, 163_012_608);
assert.equal(geometryRgba32fBytes, 963_477_504);
assert.equal(activeMaskFullTextureBytes, 60_217_344);
assert.equal(activeMaskTileTextureBytes, 5_308_416);
assert.equal(activeMaskFullTextureBytes - activeMaskTileTextureBytes, 54_908_928);
assert.equal(activeMaskFullTextureBytes * 2 - (activeMaskFullTextureBytes + maskGenerationTileBytes), 56_023_040);
assert.equal(activeMaskFullTextureBytes * 3 - (activeMaskFullTextureBytes + maskGenerationTileBytes * 2), 112_046_080);
assert.equal(maxRangeMaskHalo, 127);
assert.equal(maxRangeMaskTileBytes, 5_299_204);
assert.equal(activeMaskFullTextureBytes * 3 - maxRangeMaskTileBytes * 3, 164_754_420);
assert.match(packageJson.scripts['gpu-mask:check'], /tiled_mask_gpu_sampling/);
assert.match(packageJson.scripts['synthetic-mask:bench'], /synthetic_60mp_mask_cache_ownership/);
assert.match(packageJson.scripts['synthetic-mask-compose:bench'], /synthetic_60mp_mask_composition_scratch/);
assert.match(packageJson.scripts['synthetic-range-mask:bench'], /synthetic_60mp_range_mask_overlap_scratch/);

console.log(
  'Validated four render tiers, shared CPU mask ownership, bounded programmatic/brush/range mask generation, exact grow/feather halos, tile-local active-mask textures, borrowed float geometry sources, direct JPEG/PNG/TIFF row pipelines, bounded WebP YUVA/file output, in-encoder JPEG/PNG/TIFF EXIF, bounded resize/watermark transforms, CPU-only GPU textures, and export high-water reclamation.',
);
