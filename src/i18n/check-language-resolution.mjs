import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { build } from 'esbuild';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const entryPoint = path.join(scriptDir, 'languages.ts');
const result = await build({
  bundle: true,
  entryPoints: [entryPoint],
  format: 'esm',
  logLevel: 'silent',
  platform: 'node',
  target: 'node24',
  write: false,
});

const bundledSource = result.outputFiles[0]?.text;
if (!bundledSource) {
  throw new Error('Failed to bundle the language resolver for validation.');
}

const moduleUrl = `data:text/javascript;base64,${Buffer.from(bundledSource).toString('base64')}`;
const { resolveSupportedLanguage } = await import(moduleUrl);

const cases = new Map([
  ['en-US', 'en'],
  ['zh', 'zh-CN'],
  ['zh-Hans-CN', 'zh-CN'],
  ['zh-SG', 'zh-CN'],
  ['zh-Hant-HK', 'zh-TW'],
  ['zh-HK', 'zh-TW'],
  ['pt-BR', 'pt'],
  ['unknown', 'en'],
]);

for (const [input, expected] of cases) {
  const actual = resolveSupportedLanguage(input);
  if (actual !== expected) {
    throw new Error(`${input}: expected ${expected}, received ${actual}`);
  }
}

console.log(`Validated ${cases.size} locale resolution cases.`);
