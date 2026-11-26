import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const bundled = await build({
  entryPoints: [path.join(repoRoot, 'src/utils/stackReportInspection.ts')],
  bundle: true,
  format: 'esm',
  platform: 'node',
  target: 'node20',
  write: false,
});
const moduleSource = Buffer.from(bundled.outputFiles[0].contents).toString('base64');
const { parseStackReportInspection } = await import(`data:text/javascript;base64,${moduleSource}`);

// 1. Real sleeping-45 scale=1 shaped report: QG fail + local-scale ROIs + tone violations + density bypass.
const sleepingReport = {
  schema: 8,
  run_id: 'run-sleeping-45',
  pipeline_version: 'stack-2026.10.28.1',
  selected_path: 'layered_virtual_tile',
  degradation: {
    result: 'rejected',
    entries: [
      { reason: 'quality_gate_criterion_failed', severity: 'rejected', detail: {} },
      { reason: 'tone_boundary_delta_e_exceeded', severity: 'degraded', detail: {} },
      { reason: 'tone_boundary_delta_e_exceeded', severity: 'degraded', detail: {} },
    ],
  },
  grouping: { station_count: 15 },
  closure: {
    status: 'converged',
    max_direct_pair_p95_px: 2.6943,
    residual_median_px: 0.3313,
    residual_p95_px: 1.3743,
    fallback_reason: null,
  },
  composition: {
    canvas: { width: 12113, height: 10280 },
    output_density: {
      status: 'bypassed',
      factor: 1.0,
      invalid_probe_count: 30,
      bypass_reason: 'nonidentity_world_residual',
    },
  },
  tone: {
    status: 'degraded',
    boundary_delta_e: {
      max: 3.3363,
      threshold: 1.5,
      violations: [
        { left: 10, right: 13, delta_e00: 3.3363, world: { left: 1280, top: 8448, width: 256, height: 256 } },
      ],
    },
  },
  quality_gate: {
    verdict: 'fail',
    roi_count: 8,
    owner_reverse_lookup_failures: 0,
    criteria: [
      {
        name: 'local_scale_median',
        verdict: 'fail',
        threshold: 0.98,
        threshold_min: null,
        threshold_max: null,
        measurable_count: 8,
        unmeasurable_count: 0,
        measured: [
          { world: { x: 8772, y: 84 }, measured: 1.0017, owner_path: '/data/DSC_6498.NEF' },
          { world: { x: 6724, y: 3284 }, measured: 0.8989, owner_path: '/data/DSC_6469.NEF' },
        ],
      },
      {
        name: 'effective_pixel_count',
        verdict: 'pass',
        threshold: 0.98,
        measurable_count: 1,
        unmeasurable_count: 0,
        measured: [{ world: { x: 0, y: 0 }, measured: 0.9999, owner_path: '' }],
      },
    ],
  },
  input: {
    sources: [
      { path: '/data/DSC_6469.NEF', sha256_before: 'aaa', sha256_after: 'aaa', decoded: true },
      { path: '/data/DSC_6499.NEF', sha256_before: '', sha256_after: '', decoded: false },
    ],
    excluded_sources: [{ path: '/data/DSC_6499.NEF', reason: 'zero_bytes' }],
  },
  resources: {
    peak_rss_bytes: 11054006272,
    memory_threshold_bytes: 25769803776,
    memory_threshold_exceeded: false,
    worker_threads: 8,
  },
  performance: { stage_seconds: { total_seconds: 6755.4 } },
  output: {
    written: true,
    format: 'TIFF',
    bit_depth: 16,
    alpha_preserved: true,
    artifact_validation: 'pass',
    path: '/out/sleeping.tiff',
    result_id: 'res-1',
  },
};

const parsed = parseStackReportInspection(JSON.stringify(sleepingReport));
assert.ok(parsed, 'valid Stack_Report JSON must parse');
assert.equal(parsed.verdict, 'rejected');
assert.equal(parsed.qualityGate.verdict, 'fail');
assert.equal(parsed.stationCount, 15);
assert.equal(parsed.excludedCount, 1);
assert.equal(parsed.sources.length, 2);
assert.equal(parsed.sources[0].digestIntact, true);
assert.equal(parsed.sources[1].excluded, true);
assert.equal(parsed.sources[1].exclusionReason, 'zero_bytes');
assert.equal(parsed.issues[0].code, 'quality_gate_criterion_failed');
assert.equal(parsed.issues[0].severity, 'error');
assert.equal(parsed.issues[1].code, 'tone_boundary_delta_e_exceeded');
assert.equal(parsed.issues[1].count, 2);

// Only the failing ROI (0.8989 < 0.98) and the tone violation should generate markers.
assert.equal(parsed.markers.length, 2);
const qgMarker = parsed.markers.find((m) => m.kind === 'qg_fail');
const toneMarker = parsed.markers.find((m) => m.kind === 'tone_violation');
assert.ok(qgMarker, 'failing local_scale_median ROI must create a qg_fail marker');
assert.equal(qgMarker.ownerPath, '/data/DSC_6469.NEF');
assert.ok(qgMarker.normalizedBox.x > 0.5 && qgMarker.normalizedBox.x < 0.6);
assert.ok(toneMarker, 'tone violation must create a tone_violation marker');
assert.deepEqual(toneMarker.stations, [10, 13]);
assert.ok(toneMarker.normalizedBox.y > 0.8 && toneMarker.normalizedBox.y < 0.9);

// 2. Malformed payloads fail cleanly to null.
assert.equal(parseStackReportInspection('{broken'), null);
assert.equal(parseStackReportInspection(null), null);
assert.equal(parseStackReportInspection([]), null);

console.log('Validated Stack_Report inspection parsing, failure explanations, source ledger, and marker coordinates.');
