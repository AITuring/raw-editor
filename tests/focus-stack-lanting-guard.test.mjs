import test from 'node:test';
import assert from 'node:assert/strict';
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
