import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const previewSource = fs.readFileSync(path.join(repoRoot, 'src/components/modals/ImageStackResultPreview.tsx'), 'utf8');

const cleanupStart = previewSource.indexOf('useEffect(\n    () => () => {');
const cleanupEnd = previewSource.indexOf('\n    },\n    [],\n  );', cleanupStart);

assert.ok(cleanupStart >= 0 && cleanupEnd > cleanupStart, 'image-stack preview must keep an explicit unmount cleanup');

const cleanup = previewSource.slice(cleanupStart, cleanupEnd);

assert.match(cleanup, /cancelAnimationFrame\(renderFrameRef\.current\)/);
assert.match(
  cleanup,
  /cancelAnimationFrame\(renderFrameRef\.current\);\s*renderFrameRef\.current = null;/,
  'cancelled animation frames must release their scheduling guard so React StrictMode can schedule the remount frame',
);

for (const timerRef of ['settleTimerRef', 'transitionTimerRef']) {
  assert.match(cleanup, new RegExp(`window\\.clearTimeout\\(${timerRef}\\.current\\)`));
  assert.match(
    cleanup,
    new RegExp(`window\\.clearTimeout\\(${timerRef}\\.current\\);\\s*${timerRef}\\.current = null;`),
    `${timerRef} must not retain a cancelled timer handle after effect cleanup`,
  );
}

console.log('Validated image-stack preview scheduling cleanup across React StrictMode remounts.');
