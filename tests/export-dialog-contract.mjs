import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (relativePath) => fs.readFileSync(path.join(repoRoot, relativePath), 'utf8');

const dialogSource = read('src/features/export/ExportImageDialog.tsx');
const appModalsSource = read('src/components/modals/AppModals.tsx');
const stackModalSource = read('src/components/modals/ImageStackModal.tsx');
const productivitySource = read('src/hooks/useProductivityActions.ts');
const exportProcessingSource = read('src-tauri/src/export_processing.rs');
const exifProcessingSource = read('src-tauri/src/exif_processing.rs');
const stackProcessingSource = read('src-tauri/src/image_stack.rs');

assert.match(dialogSource, /import ZoomableImagePreview from/);
assert.match(dialogSource, /role="dialog"/);
assert.match(dialogSource, /metadataModeOptions/);
assert.match(dialogSource, /aria-expanded=\{isMetadataExpanded\}/);
assert.match(dialogSource, /estimatedFileSizes/);
assert.match(dialogSource, /embedColorProfile/);
assert.match(dialogSource, /sourceSize=\{\{ width: settings\.resizeWidth, height: settings\.resizeHeight \}\}/);
assert.match(appModalsSource, /<ExportImageDialog/);
assert.match(appModalsSource, /onEstimateSize=\{handleEstimateEditorExportSize\}/);
assert.match(stackModalSource, /<ExportImageDialog/);
assert.match(appModalsSource, /buildBackendExportSettings\(settings/);
assert.match(appModalsSource, /waitForCompletion:\s*true/);
assert.match(productivitySource, /buildBackendExportSettings\(settings/);
assert.match(exportProcessingSource, /metadata_overrides: Option<exif_processing::ExportMetadataOverrides>/);
assert.match(exportProcessingSource, /embed_color_profile: bool/);
assert.match(exportProcessingSource, /wait_for_completion: Option<bool>/);
assert.match(exifProcessingSource, /ExifTag::Artist/);
assert.match(exifProcessingSource, /ExifTag::Copyright/);
assert.match(exifProcessingSource, /ExifTag::UserComment/);
assert.match(stackProcessingSource, /apply_export_resize_and_watermark/);
assert.match(stackProcessingSource, /write_image_stack_output_with_settings/);

const bundled = await build({
  entryPoints: [path.join(repoRoot, 'src/features/export/exportDialog.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'node20',
  write: false,
});
const moduleSource = Buffer.from(bundled.outputFiles[0].contents).toString('base64');
const {
  buildBackendExportSettings,
  buildExportMetadataEntries,
  buildSuggestedExportPath,
  createInitialExportDialogSettings,
  dimensionsFromPercent,
  ensureExportPathExtension,
  estimateExportFileSize,
} = await import(`data:text/javascript;base64,${moduleSource}`);

const initial = createInitialExportDialogSettings({ width: 6480, height: 9664 }, 'jpeg', {
  Artist: 'Museum Team',
  Copyright: 'Copyright 2026',
});
assert.equal(initial.resizePercent, 100);
assert.equal(initial.sourceWidth, 6480);
assert.equal(initial.artist, 'Museum Team');
assert.equal(buildBackendExportSettings(initial, null).resize, null);
assert.equal(buildBackendExportSettings(initial, null).metadataOverrides, null);

const resized = { ...initial, resizeWidth: 6479, resizeHeight: 9663, resizePercent: 100 };
assert.deepEqual(buildBackendExportSettings(resized, null).resize, {
  mode: 'width',
  value: 6479,
  dontEnlarge: false,
});
assert.deepEqual(dimensionsFromPercent(6480, 9664, 50), { width: 3240, height: 4832 });

const jpegEstimate = estimateExportFileSize('jpeg', 6480, 9664, 95);
const pngEstimate = estimateExportFileSize('png', 6480, 9664, 95);
const tiffEstimate = estimateExportFileSize('tiff', 6480, 9664, 95);
assert.ok(jpegEstimate < pngEstimate);
assert.ok(pngEstimate < tiffEstimate);
assert.ok(estimateExportFileSize('jpeg', 6480, 9664, 100) > estimateExportFileSize('jpeg', 6480, 9664, 50));

const copyrightOnly = buildBackendExportSettings(
  {
    ...initial,
    contact: 'archive@example.test',
    description: 'Must not leak into copyright-only metadata',
    metadataMode: 'copyright',
  },
  null,
);
assert.equal(copyrightOnly.keepMetadata, false);
assert.equal(copyrightOnly.metadataOverrides.artist, 'Museum Team');
assert.equal(copyrightOnly.metadataOverrides.contact, 'archive@example.test');
assert.equal(copyrightOnly.metadataOverrides.description, null);

const clearedAllMetadata = buildBackendExportSettings(
  {
    ...initial,
    artist: '',
    metadataEditedFields: { ...initial.metadataEditedFields, artist: true },
    metadataMode: 'all',
  },
  null,
);
assert.equal(clearedAllMetadata.metadataOverrides.artist, '');
assert.equal(clearedAllMetadata.metadataOverrides.description, null);

const visibleMetadata = buildExportMetadataEntries(
  {
    Artist: 'Museum Team',
    GPSLatitude: '31 deg 14 min',
    LensModel: 'Archive Lens',
  },
  initial,
);
assert.deepEqual(
  visibleMetadata.map(({ key }) => key),
  ['Artist', 'LensModel'],
);
assert.deepEqual(
  buildExportMetadataEntries(null, { ...initial, contact: 'archive@example.test', metadataMode: 'copyright' }),
  [
    { key: 'Artist', value: 'Museum Team' },
    { key: 'Copyright', value: 'Copyright 2026' },
    { key: 'Contact', value: 'archive@example.test' },
  ],
);
assert.deepEqual(buildExportMetadataEntries({ Artist: 'Museum Team' }, { ...initial, metadataMode: 'none' }), []);

assert.equal(buildSuggestedExportPath('/photos/source.jpg?vc=3', '_edited', 'tiff'), '/photos/source_edited.tif');
assert.equal(ensureExportPathExtension('/photos/export.jpeg', 'jpeg'), '/photos/export.jpeg');
assert.equal(ensureExportPathExtension('/photos/export.png', 'tiff'), '/photos/export.tif');
assert.equal(ensureExportPathExtension('/photos/export', 'png'), '/photos/export.png');

console.log(
  'Validated the shared editor/stack export dialog, exact resize settings, format path handling, metadata modes, EXIF overrides, and ICC backend contract.',
);
