#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

// Cross-platform, repository-self-contained contract and unit checks.
// Real-material runners (focus-stack-datasets:check, *-real.mjs) are intentionally
// excluded here because they depend on local RAW datasets or macOS-only tools.
const checks = [
  { name: 'local-only:check', args: ['tests/local-only-boundary.mjs'] },
  { name: 'color-contract:check', args: ['tests/color-contract-boundary.mjs'] },
  { name: 'batch-geometry-preview:check', args: ['tests/batch-geometry-preview-contract.mjs'] },
  { name: 'crop-transform:check', args: ['tests/crop-transform-contract.mjs'] },
  { name: 'export-dialog:check', args: ['tests/export-dialog-contract.mjs'] },
  { name: 'image-stack-preview:check', args: ['tests/image-stack-preview-contract.mjs'] },
  { name: 'stack-report-inspection:check', args: ['tests/stack-report-inspection.mjs'] },
  { name: 'source-lock-sidecar:check', args: ['tests/source-lock-sidecar.mjs'] },
  { name: 'focus-stack-quality:check', args: ['tests/focus-stack-quality-contract.mjs'] },
  { name: 'focus-stack-regression:check', args: ['--test', 'tests/focus-stack-regression-verdict.mjs'] },
  { name: 'focus-stack-lanting:guard:check', args: ['--test', 'tests/focus-stack-lanting-guard.test.mjs'] },
  { name: 'focus-stack-scale-audit:check', args: ['--test', 'tests/focus-stack-scale-audit.test.mjs'] },
  { name: 'preview-transport:check', args: ['tests/preview-transport-boundary.mjs'] },
  { name: 'preview-resolution:check', args: ['tests/preview-resolution.mjs'] },
  { name: 'render-strategy:check', args: ['tests/render-strategy-boundary.mjs'] },
  { name: 'style-transfer-compare:check', args: ['tests/style-transfer-compare-contract.mjs'] },
  { name: 'ui-token:check', args: ['tests/ui-token-contract.mjs'] },
  { name: 'watermark-contract:check', args: ['tests/watermark-contract.mjs'] },
  { name: 'brightness-map:check', args: ['tests/brightness-map.mjs'] },
  { name: 'adaptive-brightness:check', args: ['tests/adaptive-brightness.mjs'] },
  { name: 'seam-candidates:check', args: ['tests/seam-candidates.mjs'] },
  { name: 'seam-texture:check', args: ['tests/seam-texture.mjs'] },
  { name: 'native-seam-texture:check', args: ['tests/native-seam-texture.mjs'] },
  { name: 'native-seam-profile:check', args: ['tests/native-seam-profile.mjs'] },
  { name: 'native-seam-path:check', args: ['tests/native-seam-path.mjs'] },
];

const startedAt = Date.now();
const failures = [];

for (let index = 0; index < checks.length; index += 1) {
  const check = checks[index];
  const label = `[${index + 1}/${checks.length}] ${check.name}`;
  console.log(`\n==> ${label}`);
  const result = spawnSync(process.execPath, check.args, {
    cwd: rootDir,
    stdio: 'inherit',
  });
  if (result.error) {
    console.error(`Failed to launch ${check.name}: ${result.error.message}`);
    failures.push(check.name);
  } else if (result.status !== 0) {
    failures.push(`${check.name} (exit ${result.status ?? 'signal'})`);
  }
}

const elapsedSeconds = ((Date.now() - startedAt) / 1000).toFixed(2);

if (failures.length > 0) {
  console.error(`\nContract suite failed (${failures.length}/${checks.length} failed in ${elapsedSeconds}s):`);
  for (const failure of failures) {
    console.error(`  - ${failure}`);
  }
  process.exit(1);
}

console.log(`\nAll ${checks.length} contract and unit checks passed in ${elapsedSeconds}s.`);
