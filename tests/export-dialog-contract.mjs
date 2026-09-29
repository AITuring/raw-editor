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
const stackSaveSource = stackProcessingSource.slice(stackProcessingSource.indexOf('pub async fn save_image_stack'));

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
assert.match(stackModalSource, /if \(isSaving \|\| !finalImageBase64\) return null;/);
assert.match(stackModalSource, /disabled=\{isSaving \|\| isProcessing\}/);
assert.doesNotMatch(stackModalSource, /disabled=\{isSaving \|\| isProcessing \|\| Boolean\(savedPath\)\}/);
assert.match(appModalsSource, /buildBackendExportSettings\(settings/);
assert.match(appModalsSource, /waitForCompletion:\s*true/);
assert.match(productivitySource, /buildBackendExportSettings\(settings/);
assert.match(exportProcessingSource, /metadata_overrides: Option<exif_processing::ExportMetadataOverrides>/);
assert.match(exportProcessingSource, /embed_color_profile: bool/);
assert.match(exportProcessingSource, /bit_depth: u8/);
assert.match(dialogSource, /supports16BitExport/);
assert.match(dialogSource, /settings\.bitDepth/);
assert.match(exportProcessingSource, /wait_for_completion: Option<bool>/);
assert.match(exifProcessingSource, /ExifTag::Artist/);
assert.match(exifProcessingSource, /ExifTag::Copyright/);
assert.match(exifProcessingSource, /ExifTag::UserComment/);
assert.match(stackProcessingSource, /apply_export_resize_and_watermark/);
assert.match(stackProcessingSource, /write_image_stack_output_with_settings/);
assert.match(stackSaveSource, /let \(stored_result_id, image, degradation_ledger\) = result\s*\.as_ref\(\)/);
assert.doesNotMatch(stackSaveSource, /\*result\s*=\s*None/);

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
assert.equal(initial.bitDepth, 8);
assert.equal(buildBackendExportSettings(initial, null).resize, null);
assert.equal(buildBackendExportSettings(initial, null).metadataOverrides, null);

const stackInitial = createInitialExportDialogSettings({ width: 6480, height: 9664 }, 'tiff');
assert.equal(stackInitial.bitDepth, 16);
assert.equal(buildBackendExportSettings(stackInitial, null).bitDepth, 16);

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
assert.ok(estimateExportFileSize('png', 6480, 9664, 95, 16) > estimateExportFileSize('png', 6480, 9664, 95, 8));
assert.ok(estimateExportFileSize('tiff', 6480, 9664, 95, 16) > estimateExportFileSize('tiff', 6480, 9664, 95, 8));
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

const stackSaveMocks = {
  react: 'export const useCallback = (callback) => callback;',
  '@tauri-apps/api/core': 'export const invoke = (...args) => globalThis.__stackSaveContract.invoke(...args);',
  '@tauri-apps/plugin-dialog': `
    export const save = (...args) => globalThis.__stackSaveContract.save(...args);
    export const confirm = (...args) => globalThis.__stackSaveContract.confirm(...args);
  `,
  '../store/useUIStore': `
    export const useUIStore = Object.assign(
      (selector) => selector(globalThis.__stackSaveContract.ui),
      { getState: () => globalThis.__stackSaveContract.ui },
    );
  `,
  '../store/useSettingsStore': `
    export const useSettingsStore = { getState: () => ({ osPlatform: 'macos' }) };
  `,
  '../i18n': 'export default { t: (key) => key };',
};
const stackActionsBundle = await build({
  entryPoints: [path.join(repoRoot, 'src/hooks/useProductivityActions.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'node20',
  write: false,
  plugins: [
    {
      name: 'stack-save-contract',
      setup(builder) {
        builder.onResolve({ filter: /.*/ }, (args) =>
          Object.hasOwn(stackSaveMocks, args.path) ? { path: args.path, namespace: 'stack-save-mock' } : null,
        );
        builder.onLoad({ filter: /.*/, namespace: 'stack-save-mock' }, (args) => ({
          contents: stackSaveMocks[args.path],
          loader: 'js',
        }));
      },
    },
  ],
});
const { useProductivityActions } = await import(
  `data:text/javascript;base64,${Buffer.from(stackActionsBundle.outputFiles[0].contents).toString('base64')}`
);
const exerciseStackSave = async (notices, confirmed) => {
  const calls = [];
  const ui = {
    imageStackModalState: { sourcePaths: ['/photos/source.nef'], resultId: 'current-stack-result' },
    setUI(update) {
      Object.assign(ui, update(ui));
    },
  };
  globalThis.__stackSaveContract = {
    ui,
    async invoke(command, args) {
      calls.push([command, args]);
      if (command === 'image_stack_output_notices') return notices;
      assert.equal(command, 'save_image_stack');
      return '/photos/export.tif';
    },
    async confirm(message, options) {
      calls.push(['confirm', message, options]);
      return confirmed;
    },
    async save() {
      calls.push(['choose-path']);
      return '/photos/export.tif';
    },
  };
  try {
    const actions = useProductivityActions(
      async () => calls.push(['refresh']),
      () => {},
    );
    const result = await actions.handleSaveImageStack('focus', stackInitial);
    return { calls, result };
  } finally {
    delete globalThis.__stackSaveContract;
  }
};
const declined = await exerciseStackSave(['Transparency will be lost.'], false);
assert.equal(declined.result, null);
assert.deepEqual(
  declined.calls.map(([name]) => name),
  ['image_stack_output_notices', 'confirm'],
);
assert.deepEqual(declined.calls[0][1], { outputFormat: 'tiff', bitDepth: 16, resultId: 'current-stack-result' });
assert.equal(declined.calls[1][1], 'Transparency will be lost.');
assert.equal(declined.calls[1][2].kind, 'warning');
const confirmed = await exerciseStackSave(['Bit depth will be reduced.', 'Transparency will be lost.'], true);
assert.equal(confirmed.result, '/photos/export.tif');
assert.deepEqual(
  confirmed.calls.map(([name]) => name),
  ['image_stack_output_notices', 'confirm', 'choose-path', 'save_image_stack', 'refresh'],
);
assert.equal(confirmed.calls[1][1], 'Bit depth will be reduced.\n\nTransparency will be lost.');
assert.equal(confirmed.calls[3][1].resultId, 'current-stack-result');
const lossless = await exerciseStackSave([], false);
assert.equal(lossless.result, '/photos/export.tif');
assert.deepEqual(
  lossless.calls.map(([name]) => name),
  ['image_stack_output_notices', 'choose-path', 'save_image_stack', 'refresh'],
);

console.log(
  'Validated the shared editor/stack export dialog, exact resize settings, format path handling, metadata modes, EXIF overrides, and ICC backend contract.',
);
