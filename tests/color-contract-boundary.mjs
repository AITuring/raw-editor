import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (relativePath) => fs.readFileSync(path.join(repoRoot, relativePath), 'utf8');

const failures = [];
const expect = (condition, message) => {
  if (!condition) failures.push(message);
};
const expectIncludes = (source, value, message) => expect(source.includes(value), message);

function functionBody(source, name, label) {
  const match = new RegExp(`\\bfn\\s+${name}\\s*\\(`).exec(source);
  expect(match !== null, `${label}: missing ${name}`);
  if (!match) return '';

  const openingBrace = source.indexOf('{', match.index);
  expect(openingBrace >= 0, `${label}: ${name} has no body`);
  if (openingBrace < 0) return '';

  let depth = 0;
  for (let index = openingBrace; index < source.length; index += 1) {
    if (source[index] === '{') depth += 1;
    if (source[index] === '}') depth -= 1;
    if (depth === 0) return source.slice(openingBrace, index + 1);
  }

  failures.push(`${label}: ${name} has an unterminated body`);
  return '';
}

function collectRustFiles(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const absolutePath = path.join(directory, entry.name);
    if (entry.isDirectory()) return collectRustFiles(absolutePath);
    return entry.name.endsWith('.rs') ? [absolutePath] : [];
  });
}

const colorManagement = read('src-tauri/src/color_management.rs');
const rawProcessing = read('src-tauri/src/raw_processing.rs');
const imageLoader = read('src-tauri/src/image_loader.rs');
const imageProcessing = read('src-tauri/src/image_processing.rs');
const gpuProcessing = read('src-tauri/src/gpu_processing.rs');
const exportProcessing = read('src-tauri/src/export_processing.rs');
const mainShader = read('src-tauri/src/shaders/shader.wgsl');
const flareShader = read('src-tauri/src/shaders/flare.wgsl');
const contractDocument = read('docs/color-pipeline.md');

const rustEotf = functionBody(colorManagement, 'srgb_to_linear_channel', 'canonical CPU transfer');
for (const token of ['max(0.0)', '0.04045', '/ 12.92', 'powf(2.4)']) {
  expectIncludes(rustEotf, token, `canonical CPU EOTF must contain ${token}`);
}

const rustOetf = functionBody(colorManagement, 'linear_to_srgb_channel', 'canonical CPU transfer');
for (const token of ['max(0.0)', '0.003_130_8', '* 12.92', 'powf(1.0 / 2.4)']) {
  expectIncludes(rustOetf, token, `canonical CPU OETF must contain ${token}`);
}

const canonicalPath = path.join(repoRoot, 'src-tauri', 'src', 'color_management.rs');
const transferDefinition = /\bfn\s+(?:srgb_to_linear(?:_channel)?|linear_to_srgb(?:_channel)?)\s*\(/g;
for (const file of collectRustFiles(path.join(repoRoot, 'src-tauri', 'src'))) {
  if (file === canonicalPath) continue;
  const source = fs.readFileSync(file, 'utf8');
  if (transferDefinition.test(source)) {
    failures.push(`${path.relative(repoRoot, file)}: duplicate CPU sRGB transfer definition`);
  }
  transferDefinition.lastIndex = 0;
}

expectIncludes(
  rawProcessing,
  'use crate::color_management::srgb_to_linear_channel;',
  'RAW normalization must use the canonical CPU EOTF',
);
expectIncludes(
  rawProcessing,
  'srgb_to_linear_channel(rescaled.clamp(0.0, 1.0))',
  'LinearRaw gamma mode must decode through the canonical EOTF',
);
expect(!rawProcessing.includes('powf(3.0)'), 'LinearRaw gamma mode must not restore the old exponent 3.0');
expectIncludes(
  rawProcessing,
  'step != ProcessingStep::SRgb',
  'RAW development must stay linear by omitting rawler sRGB encoding',
);

expectIncludes(
  imageLoader,
  'use crate::color_management::srgb_to_linear_channel;',
  'image patch LUT must use the canonical CPU EOTF',
);
expectIncludes(
  imageLoader,
  '*v = srgb_to_linear_channel(i as f32 / 255.0);',
  'image patch LUT must be built from the canonical CPU EOTF',
);
expectIncludes(imageProcessing, 'linear_to_srgb_channel(p[0])', 'RGBA CPU encoding must use the canonical OETF');
expectIncludes(imageProcessing, 'srgb_to_linear_channel(p[0])', 'RGBA CPU decoding must use the canonical EOTF');

for (const [source, label] of [
  [mainShader, 'main shader'],
  [flareShader, 'flare shader'],
]) {
  const eotf = functionBody(source, 'srgb_to_linear', label);
  for (const token of ['0.04045', '0.055', '2.4', '12.92']) {
    expectIncludes(eotf, token, `${label} EOTF must contain ${token}`);
  }

  const oetf = functionBody(source, 'linear_to_srgb', label);
  for (const token of ['0.0031308', '0.055', '1.0 / 2.4', '12.92']) {
    expectIncludes(oetf, token, `${label} OETF must contain ${token}`);
  }
}

expect(
  /if \(is_raw == 0u\) \{\s*initial_linear_rgb = srgb_to_linear\(color_from_texture\);\s*\} else \{\s*initial_linear_rgb = color_from_texture;/m.test(
    mainShader,
  ),
  'main shader must decode non-RAW sRGB and keep RAW input linear',
);
expectIncludes(
  mainShader,
  'var output_texture: texture_storage_2d<rgba8unorm, write>;',
  'main shader output must remain 8-bit unorm display data',
);
expectIncludes(
  mainShader,
  'clamp(final_rgb, vec3<f32>(0.0), vec3<f32>(1.0))',
  'main shader must clamp only at the display/output boundary',
);
expectIncludes(
  gpuProcessing,
  'format: wgpu::TextureFormat::Rgba16Float',
  'GPU upload/intermediate contract must retain float texture storage',
);
expectIncludes(
  gpuProcessing,
  'format: wgpu::TextureFormat::Rgba8Unorm',
  'GPU readback/display contract must retain encoded 8-bit output',
);
expectIncludes(
  gpuProcessing,
  'color_space: wgpu::SurfaceColorSpace::Srgb',
  'native display surface must explicitly request sRGB',
);
expectIncludes(
  exportProcessing,
  'set_icc_profile(srgb_v4_profile().to_vec())',
  'supported exports must embed the bundled sRGB profile',
);

for (const limitation of ['嵌入输入 ICC', '显示器 ICC', 'Sony α7R V']) {
  expectIncludes(contractDocument, limitation, `color pipeline document must retain limitation: ${limitation}`);
}

if (failures.length > 0) {
  console.error('Color pipeline contract violated:');
  for (const failure of [...new Set(failures)].sort()) console.error(`- ${failure}`);
  process.exitCode = 1;
} else {
  console.log('Validated the CPU/WGSL sRGB transfer, RAW routing, display boundary, and export profile contract.');
}
