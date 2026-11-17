import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { evaluateLantingGuard } from './focus-stack-lanting-guard.mjs';
const requiredCriteria = Array.from({ length: 8 }, (_, i) => `criterion-${i}`);
const contract = {
  sourceCount: 45,
  inputsSha: 'input-manifest',
  outputSha: 'pixels',
  referenceSha: 'reference',
  ownerSha: 'owner',
  coverageSha: 'coverage',
  canvas: [31085, 11011],
  covered: 421256167,
  owned: 421256167,
  paper: { minimumPoints: 1000, maximumBoundaryP95: 3, maximumBoundaryStep: 7 },
  requiredCriteria,
};
const valid = () => ({
  ...contract,
  inputsUnchanged: true,
  stationCount: 44,
  sourcePaths: Array.from({ length: 45 }, (_, i) => `source-${i}`),
  written: true,
  artifactValidation: 'pass',
  alphaPreserved: true,
  bitDepth: 16,
  paper: { paper_points: 1008, source_boundary: { p95: 3, max: 7 } },
  qualityGate: {
    legacy_native_evidence: {
      method: 'streaming_native_owner_direct_sampler_five_probe_v1',
      focus_population: 'actual_capture_station_members_only',
      capture_station_count: 44,
      sources: Array.from({ length: 45 }, (_, i) => ({
        owner: i + 1,
        path: `source-${i}`,
        source_tone_replayed: true,
        source_to_world: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      })),
      sampled_cell_count: 100,
      candidate_measurement_count: 200,
      delivered_pixels: 31085 * 11011,
      unresolved_pixel_count: 0,
    },
    verdict: 'pass',
    roi_count: 8,
    criteria: requiredCriteria.map((name) => ({ name, verdict: 'pass' })),
  },
});
test('insufficient evidence cannot pass the strict gate despite identical TIFF', () => {
  const a = valid();
  a.qualityGate = { verdict: 'insufficient_evidence', roi_count: 0, criteria: [] };
  assert.deepEqual(evaluateLantingGuard(contract, a).failures, ['legacy_quality_gate_not_passed']);
  assert.equal(evaluateLantingGuard(contract, a, { regressionOnly: true }).status, 'regression_only_quality_pending');
});
test('reported pass without ROI/criteria evidence fails', () => {
  const a = valid();
  a.qualityGate = { verdict: 'pass', roi_count: 0, criteria: [] };
  assert.equal(evaluateLantingGuard(contract, a).pass, false);
});
test('pixel, owner or paper changes block a pure performance candidate', () => {
  for (const mutate of [
    (a) => (a.outputSha = 'other'),
    (a) => (a.ownerSha = 'other'),
    (a) => (a.inputsSha = 'other'),
    (a) => (a.paper.source_boundary.p95 = 4),
  ]) {
    const a = valid();
    mutate(a);
    assert.equal(evaluateLantingGuard(contract, a).pass, false);
  }
});
test('missing paper evidence never silently passes', () => {
  const a = valid();
  delete a.paper;
  assert.equal(evaluateLantingGuard(contract, a).pass, false);
});
test('existing overall gate semantics permit diagnostic content limitations', () => {
  const a = valid();
  a.qualityGate.criteria[0].verdict = 'unmeasurable';
  a.qualityGate.criteria[0].diagnostic = true;
  assert.equal(evaluateLantingGuard(contract, a).pass, true);
});

test('reported quality pass cannot conceal missing or unresolved native source evidence', () => {
  for (const mutate of [
    (a) => delete a.qualityGate.legacy_native_evidence,
    (a) => (a.qualityGate.legacy_native_evidence.unresolved_pixel_count = 1),
    (a) => a.qualityGate.legacy_native_evidence.sources.pop(),
    (a) => (a.qualityGate.legacy_native_evidence.sources[0].path = 'unrelated-source'),
    (a) => (a.qualityGate.legacy_native_evidence.focus_population = 'all_camera_sources'),
    (a) => (a.qualityGate.legacy_native_evidence.sources[0].source_tone_replayed = false),
    (a) => (a.qualityGate.legacy_native_evidence.sources[0].source_to_world[0] = NaN),
  ]) {
    const actual = valid();
    mutate(actual);
    assert.equal(evaluateLantingGuard(contract, actual).pass, false);
  }
});

test('task 6 large-step ceiling fails closed on missing evidence or an increase', () => {
  const c = structuredClone(contract);
  c.paper.maximumBoundaryAbove6 = 2;
  const a = valid();
  a.paper.source_boundary.above_6 = 2;
  assert.equal(evaluateLantingGuard(c, a).pass, true);
  a.paper.source_boundary.above_6 = 3;
  assert.deepEqual(evaluateLantingGuard(c, a).failures, ['paper_large_steps_regression']);
  delete a.paper.source_boundary.above_6;
  assert.equal(evaluateLantingGuard(c, a).pass, false);
});

test('task 6 audit binds the reference conversion and the measured image artifacts', () => {
  const c = structuredClone(contract);
  c.paper.referencePreviewSha = 'expected-reference-preview';
  const a = valid();
  a.paperReferencePreviewSha = c.paper.referencePreviewSha;
  a.paperCurrentPreviewSha = 'current-preview';
  a.paperProvenance = { reference_preview_sha256: a.paperReferencePreviewSha,
    current_preview_sha256: a.paperCurrentPreviewSha, output_sha256: a.outputSha,
    reference_sha256: a.referenceSha, owner_sha256: a.ownerSha };
  assert.equal(evaluateLantingGuard(c, a).pass, true);
  for (const field of Object.keys(a.paperProvenance)) {
    const broken = structuredClone(a);
    broken.paperProvenance[field] = 'different';
    assert.deepEqual(evaluateLantingGuard(c, broken).failures, ['paper_audit_substrate_changed']);
  }
  a.paperReferencePreviewSha = 'sandbox-conversion';
  assert.equal(evaluateLantingGuard(c, a).pass, false);
});

test('legacy CLI contracts do not require the newly bound preview artifacts', async () => {
  // Synthetic guard plumbing only: native image/QG production is tested separately.
  const dir = await mkdtemp(join(tmpdir(), 'lanting-guard-legacy-'));
  try {
    const base = join(dir, 'lanting');
    const reportDir = join(base, 'run', 'fixture-stack-reports');
    await mkdir(reportDir, { recursive: true });
    await mkdir(join(base, 'diagnostics'));
    const hash = (s) => createHash('sha256').update(s).digest('hex');
    const actual = valid();
    const inputs = actual.sourcePaths.map((path) => ({ path }));
    const c = { ...contract, outputSha: hash('pixels'), referenceSha: hash('reference'),
      ownerSha: hash('owner'), coverageSha: hash('coverage'), inputsSha: hash(JSON.stringify(inputs)),
      referencePath: join(dir, 'reference.jpg') };
    const report = { input: { source_count: 45 }, grouping: { station_count: 44 },
      composition: { source_covered_pixels: c.covered, source_owned_pixels: c.owned },
      output: { path: join(dir, 'fixture.tiff'), width: c.canvas[0], height: c.canvas[1],
        written: true, artifact_validation: 'pass', alpha_preserved: true, bit_depth: 16 },
      quality_gate: actual.qualityGate };
    const json = (p, value) => writeFile(p, JSON.stringify(value));
    await Promise.all([
      writeFile(report.output.path, 'pixels'), writeFile(c.referencePath, 'reference'),
      writeFile(join(base, 'diagnostics/owner.png'), 'owner'),
      writeFile(join(base, 'diagnostics/coverage.png'), 'coverage'),
      json(join(reportDir, 'stack-report-fixture.json'), report),
      json(join(base, 'analysis.json'), { output_sha256: c.outputSha, checks: { inputs_unchanged: true } }),
      json(join(base, 'execution.json'), { inputs }),
      json(join(dir, 'paper-tone-audit.json'), { summary: actual.paper }),
      json(join(dir, 'contract.json'), c),
    ]);
    const result = spawnSync(process.execPath, [fileURLToPath(new URL('./focus-stack-lanting-guard.mjs', import.meta.url)),
      join(dir, 'contract.json'), base], { encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(JSON.parse(result.stdout).pass, true);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
