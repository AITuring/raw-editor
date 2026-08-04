import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (relativePath) => fs.readFileSync(path.join(repoRoot, relativePath), 'utf8');

const failures = [];
const expect = (condition, message) => {
  if (!condition) failures.push(message);
};

const sectionBetween = (source, startMarker, endMarker, label) => {
  const start = source.indexOf(startMarker);
  const end = source.indexOf(endMarker, start + startMarker.length);
  expect(start >= 0, `${label}: missing start marker ${startMarker}`);
  expect(end > start, `${label}: missing end marker ${endMarker}`);
  return start >= 0 && end > start ? source.slice(start, end) : '';
};

const rustLib = read('src-tauri/src/lib.rs');
const maskGeneration = read('src-tauri/src/mask_generation.rs');
const rustSections = [
  sectionBetween(
    rustLib,
    'async fn generate_uncropped_preview(',
    'fn generate_original_transformed_preview(',
    'uncropped preview',
  ),
  sectionBetween(
    rustLib,
    'fn generate_original_transformed_preview(',
    'async fn preview_geometry_transform(',
    'original preview',
  ),
  sectionBetween(rustLib, 'async fn preview_geometry_transform(', 'pub fn get_original_image(', 'geometry preview'),
  sectionBetween(
    maskGeneration,
    'pub fn generate_mask_overlay(',
    'pub fn resolve_warped_image_for_masks(',
    'mask overlay',
  ),
];

for (const [index, section] of rustSections.entries()) {
  expect(section.includes('Result<Response, String>'), `Rust preview section ${index + 1} must return an IPC Response`);
  expect(section.includes('Response::new('), `Rust preview section ${index + 1} must return raw bytes`);
  expect(!/base64|data:image|general_purpose/i.test(section), `Rust preview section ${index + 1} reintroduced Base64`);
}

const listeners = read('src/hooks/useTauriListeners.ts');
expect(!listeners.includes('preview-update-uncropped'), 'uncropped previews must not use a serialized Tauri event');

const frontendExpectations = [
  ['src/hooks/useImageProcessing.ts', 'Invokes.GenerateUncroppedPreview'],
  ['src/hooks/useImageProcessing.ts', 'Invokes.GenerateOriginalTransformedPreview'],
  ['src/components/modals/TransformModal.tsx', 'Invokes.PreviewGeometryTransform'],
  ['src/components/modals/LensCorrectionModal.tsx', 'Invokes.PreviewGeometryTransform'],
  ['src/components/panel/Editor.tsx', 'Invokes.GenerateMaskOverlay'],
];

for (const [relativePath, invokeName] of frontendExpectations) {
  const source = read(relativePath);
  const invokeIndex = source.indexOf(invokeName);
  expect(invokeIndex >= 0, `${relativePath}: missing ${invokeName}`);
  const nearbySource = invokeIndex >= 0 ? source.slice(Math.max(0, invokeIndex - 80), invokeIndex + 320) : '';
  expect(/ArrayBuffer/.test(nearbySource), `${relativePath}: ${invokeName} must be received as ArrayBuffer`);
}

const objectUrlUtility = read('src/utils/imageObjectUrl.ts');
expect(objectUrlUtility.includes('URL.createObjectURL'), 'binary image responses must be exposed through Blob URLs');
expect(objectUrlUtility.includes('URL.revokeObjectURL'), 'Blob URLs must have an explicit release path');

if (failures.length > 0) {
  console.error('Binary preview transport boundary violated:');
  for (const failure of [...new Set(failures)].sort()) console.error(`- ${failure}`);
  process.exitCode = 1;
} else {
  console.log('Validated binary IPC transport and Blob URL ownership for core editing previews.');
}
