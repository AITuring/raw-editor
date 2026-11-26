import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const bundled = await build({
  entryPoints: [path.join(repoRoot, 'src/utils/sourceLockSidecar.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'node20',
  write: false,
});
const moduleSource = Buffer.from(bundled.outputFiles[0].contents).toString('base64');
const sidecar = await import(`data:text/javascript;base64,${moduleSource}`);

// 0. Pure-TS SHA-256 must match Node's OpenSSL SHA-256 bit-for-bit.
for (const sample of ['', 'raw-editor', '书画多机位翻拍-source-lock-v1', 'a'.repeat(1000)]) {
  const expected = crypto.createHash('sha256').update(sample, 'utf8').digest('hex');
  assert.equal(sidecar.sha256Hex(sample), expected, `SHA-256 mismatch for input length ${sample.length}`);
}

const baseFrame = {
  id: 'world-v1',
  canvas: [31090, 11008],
  origin: [2607, 875],
  units: 'native_output_pixels',
};

const baseBindings = [
  {
    source_id: 42,
    path: '/artwork/DSC_3687.NEF',
    sha256: '11'.repeat(32),
    station_id: 7,
    capture_group_id: 3,
    source_to_world_sha256: 'aa'.repeat(32),
  },
  {
    source_id: 43,
    path: '/artwork/DSC_3688.NEF',
    sha256: '22'.repeat(32),
    station_id: 7,
    capture_group_id: 3,
    source_to_world_sha256: 'bb'.repeat(32),
  },
];

const validEvidence = {
  coordinate_frame_id: 'world-v1',
  input_manifest_sha256: 'mm'.repeat(32),
  eligible_tiles: 17,
  covered_tiles: 17,
  has_no_owner_or_transparent: false,
  footprint_and_inverse_valid: true,
  integrity_errors: [],
  coverage_owner_tile_digest: 'cc'.repeat(32),
};

const poly = [
  [120, 80],
  [640, 80],
  [640, 500],
  [120, 500],
];

const doc0 = sidecar.createSourceLockDocument({
  runId: 'native-fixture-001',
  inputManifestSha256: 'mm'.repeat(32),
  coordinateFrame: baseFrame,
  sourceBindings: baseBindings,
});

// 1. Task 9 AC1 & AC2: Reject unregistered owner IDs and missing coverage/footprint/alpha/digest.
const unregistered = sidecar.validateLockEligibility(doc0, [999], 'single_source', validEvidence);
assert.equal(unregistered.eligible, false);
assert.ok(unregistered.reasons.includes('unregistered_source_id:999'));

const badEvidence = sidecar.validateLockEligibility(doc0, [42], 'single_source', {
  ...validEvidence,
  has_no_owner_or_transparent: true,
  footprint_and_inverse_valid: false,
});
assert.equal(badEvidence.eligible, false);
assert.ok(badEvidence.reasons.includes('uncovered_or_no_owner_pixels'));
assert.ok(badEvidence.reasons.includes('invalid_source_footprint_or_inverse'));

// 2. Deterministic selection_digest independent of source ordering and free of RGB.
const region = sidecar.computeWorldRegion(poly);
const d1 = sidecar.computeSelectionDigest({
  requestedSources: [43, 42],
  region,
  coverageOwnerTileDigest: validEvidence.coverage_owner_tile_digest,
  sourceBindings: baseBindings,
  policy: 'allowed_sources',
  coordinateFrameId: 'world-v1',
});
const d2 = sidecar.computeSelectionDigest({
  requestedSources: [42, 43],
  region,
  coverageOwnerTileDigest: validEvidence.coverage_owner_tile_digest,
  sourceBindings: baseBindings,
  policy: 'allowed_sources',
  coordinateFrameId: 'world-v1',
});
assert.equal(d1, d2);

// 3. Task 9 AC3: Monotonic event log, undo appends inverse without erasing history, concurrent head rejected.
const doc1 = sidecar.commitAddSourceLock(
  doc0,
  { base_run_id: 'native-fixture-001', coordinate_frame_id: 'world-v1', expected_head: 0 },
  { polygonWorld: poly, requestedSources: [42], policy: 'single_source', evidence: validEvidence },
);
assert.equal(doc1.locks.length, 1);
assert.equal(doc1.history.head, 1);
assert.equal(doc1.locks[0].status, 'active');

assert.throws(
  () =>
    sidecar.commitAddManualReviewRegion(
      doc1,
      { base_run_id: 'native-fixture-001', coordinate_frame_id: 'world-v1', expected_head: 0 },
      { polygonWorld: poly, reason: 'paper_seam_review' },
    ),
  /Concurrent session conflict/,
);

const doc2 = sidecar.commitAddManualReviewRegion(
  doc1,
  { base_run_id: 'native-fixture-001', coordinate_frame_id: 'world-v1', expected_head: 1 },
  { polygonWorld: poly, reason: 'paper_seam_review' },
);
assert.equal(doc2.manual_review_regions.length, 1);
assert.equal(doc2.history.head, 2);

const doc3 = sidecar.commitUndo(doc2, {
  base_run_id: 'native-fixture-001',
  coordinate_frame_id: 'world-v1',
  expected_head: 2,
});
assert.equal(doc3.manual_review_regions.length, 0);
assert.equal(doc3.locks.length, 1);
assert.equal(doc3.history.head, 3, 'undo must advance monotonic event_id without deleting prior events');
assert.equal(doc3.history.events.length, 3);
assert.equal(doc3.history.events[2].kind, 'undo');
assert.equal(doc3.history.events[2].reverted_event_id, 2);

// 4. Task 9 AC4: Re-crop/origin/manifest/pose changes mark existing locks stale & top priority.
const staleDoc = sidecar.reconcileDocumentStaleness(doc1, {
  inputManifestSha256: 'mm'.repeat(32),
  coordinateFrame: { ...baseFrame, origin: [2608, 875] },
  sourceBindings: baseBindings,
});
assert.equal(staleDoc.locks[0].status, 'stale');
assert.equal(staleDoc.locks[0].stale_reason, 'coordinate_frame_or_crop_changed');
assert.equal(sidecar.resolveLockPriority(staleDoc.locks), 'stale_or_unresolved');
assert.equal(sidecar.resolveLockPriority(doc1.locks), 'single_source');

// 5. Task 9 AC5: Manual review QG accounting keeps failures & fails closed on insufficient samples.
const qgInsufficient = sidecar.computeManualReviewQualityGateAccounting({
  allTiles: [
    { tile_id: 't1', failed: true, manual_region_id: doc2.manual_review_regions[0].region_id },
    { tile_id: 't2', failed: false, manual_region_id: null },
  ],
  doc: doc2,
  minimumAutoSamples: 4,
});
assert.equal(qgInsufficient.all_tiles, 2);
assert.equal(qgInsufficient.auto_eligible_tiles, 1);
assert.equal(qgInsufficient.failed_all_tiles, 1);
assert.equal(qgInsufficient.verdict, 'insufficient_evidence');

// 6. Task 9 AC6: Export binding detects stale_export and binding_mismatch.
const recordedSidecarSha = sidecar.computeSidecarDigest(doc1);
assert.equal(
  sidecar.verifyExportBinding({
    doc: doc1,
    currentTiffSha256: 'tt'.repeat(32),
    recordedTiffSha256: 'tt'.repeat(32),
    recordedSidecarSha256: recordedSidecarSha,
  }).status,
  'valid',
);
assert.equal(
  sidecar.verifyExportBinding({
    doc: doc2,
    currentTiffSha256: 'tt'.repeat(32),
    recordedTiffSha256: 'tt'.repeat(32),
    recordedSidecarSha256: recordedSidecarSha,
  }).status,
  'stale_export',
);
assert.equal(
  sidecar.verifyExportBinding({
    doc: doc1,
    currentTiffSha256: 'ff'.repeat(32),
    recordedTiffSha256: 'tt'.repeat(32),
    recordedSidecarSha256: recordedSidecarSha,
  }).status,
  'binding_mismatch',
);
assert.equal(
  sidecar.verifyExportBinding({
    doc: staleDoc,
    currentTiffSha256: 'tt'.repeat(32),
    recordedTiffSha256: 'tt'.repeat(32),
    recordedSidecarSha256: sidecar.computeSidecarDigest(staleDoc),
  }).status,
  'has_stale_or_unresolved_locks',
);

const serialized = sidecar.serializeSidecarFiles(doc3);
assert.equal(JSON.parse(serialized.sidecarJson).history.head, 3);
assert.equal(serialized.eventsJsonl.trim().split('\n').length, 3);

console.log(
  'Validated Task 9 / Task 20 source-lock sidecar contract, SHA-256 digest, undo log, staleness, QG accounting, and export binding.',
);
