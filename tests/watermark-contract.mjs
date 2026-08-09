import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (relativePath) => fs.readFileSync(path.join(repoRoot, relativePath), 'utf8');

const asset = fs.readFileSync(path.join(repoRoot, 'src/assets/default-watermark.png'));
const watermarkContract = read('src/features/export/watermark.ts');
const exportSettings = read('src/hooks/useExportSettings.ts');
const exportPanel = read('src/components/panel/right/ExportPanel.tsx');
const imagePicker = read('src/components/ui/ImagePicker.tsx');
const appProperties = read('src/components/ui/AppProperties.tsx');
const rustExport = read('src-tauri/src/export_processing.rs');
const rustLib = read('src-tauri/src/lib.rs');
const tauriConfig = read('src-tauri/tauri.conf.json');
const assetProtocolScope = JSON.parse(tauriConfig).app.security.assetProtocol.scope;

assert.equal(
  crypto.createHash('sha256').update(asset).digest('hex'),
  '4db6f91d61c973c4a580bc0710c9115a9b84da9c46e52f1db1e2de129bf7f8c0',
  'the built-in watermark must remain byte-identical to the referenced my-watermark logo',
);
assert.equal(asset.readUInt32BE(16), 1908);
assert.equal(asset.readUInt32BE(20), 1462);

assert.match(watermarkContract, /DEFAULT_WATERMARK_PATH = 'builtin:\/\/default-watermark'/);
assert.match(watermarkContract, /default-watermark\.png/);
assert.match(watermarkContract, /convertFileSrc/);
assert.match(exportSettings, /useState\(DEFAULT_WATERMARK_PATH\)/);
assert.match(exportSettings, /WatermarkAnchor\.Center/);
assert.match(exportSettings, /useState\(80\)/);

assert.match(exportPanel, /watermarkImageSrc/);
assert.match(exportPanel, /baseImageSrc=\{watermarkBaseImageSrc\}/);
assert.match(exportPanel, /<img\s+alt=""/);
assert.match(exportPanel, /onUseDefault=\{\(\) => setWatermarkPath\(DEFAULT_WATERMARK_PATH\)\}/);
assert.match(imagePicker, /\['png', 'jpg', 'jpeg', 'webp', 'tif', 'tiff', 'gif'\]/);
assert.match(imagePicker, /replaceWatermark/);
assert.match(imagePicker, /useDefaultWatermark/);
assert.match(appProperties, /ImportWatermarkImage = 'import_watermark_image'/);

assert.match(rustExport, /include_bytes!\("\.\.\/\.\.\/src\/assets\/default-watermark\.png"\)/);
assert.match(rustExport, /load_from_memory_with_format\(DEFAULT_WATERMARK_BYTES, ImageFormat::Png\)/);
assert.match(rustExport, /asset_protocol_scope\(\)\.is_allowed/);
assert.match(rustExport, /import_watermark_image_impl/);
assert.match(rustExport, /persist_noclobber/);
assert.match(rustExport, /bundled_default_watermark_matches_the_reference_asset/);
assert.match(rustLib, /export_processing::import_watermark_image/);
assert.deepEqual(assetProtocolScope, ['$APPCACHE/thumbnails/*', '$APPDATA/watermarks/*']);

console.log('Validated the referenced default watermark asset, picker flow, preview, and Rust export contract.');
