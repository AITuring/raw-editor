import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export function evaluateLantingGuard(contract, actual, { regressionOnly = false } = {}) {
  const failures = [];
  const require = (ok, reason) => {
    if (!ok) failures.push(reason);
  };
  require(actual.sourceCount === contract.sourceCount, 'source_count_changed');
  require(typeof contract.inputsSha === 'string' &&
    actual.inputsSha === contract.inputsSha &&
    actual.inputsUnchanged === true, 'input_manifest_changed');
  require(actual.outputSha === contract.outputSha, 'output_pixels_changed');
  require(actual.referenceSha === contract.referenceSha, 'reference_changed');
  require(actual.ownerSha === contract.ownerSha, 'owner_changed');
  require(actual.coverageSha === contract.coverageSha, 'coverage_changed');
  require(JSON.stringify(actual.canvas) === JSON.stringify(contract.canvas), 'canvas_changed');
  require(actual.covered === contract.covered && actual.owned === contract.owned, 'source_evidence_changed');
  require(actual.written === true && actual.artifactValidation === 'pass', 'output_not_validated');
  require(actual.alphaPreserved === true && actual.bitDepth === 16, 'native_alpha_or_depth_changed');
  const paper = actual.paper;
  require(paper?.paper_points >= contract.paper.minimumPoints, 'paper_evidence_missing');
  require(Number.isFinite(paper?.source_boundary?.p95) &&
    paper.source_boundary.p95 <= contract.paper.maximumBoundaryP95, 'paper_boundary_regression');
  require(Number.isFinite(paper?.source_boundary?.max) &&
    paper.source_boundary.max <= contract.paper.maximumBoundaryStep, 'paper_block_regression');
  const gate = actual.qualityGate;
  const names = new Set(gate?.criteria?.map((c) => c.name) ?? []);
  const native = gate?.legacy_native_evidence;
  const nativeEvidencePass = Boolean(
    native?.method === 'streaming_native_owner_direct_sampler_five_probe_v1' &&
    native.focus_population === 'actual_capture_station_members_only' &&
    native.capture_station_count === actual.stationCount &&
    native.sources?.length === contract.sourceCount &&
    actual.sourcePaths?.length === contract.sourceCount &&
    native.sources.every((source) => actual.sourcePaths.includes(source.path)) &&
    new Set(native.sources.map((source) => source.owner)).size === contract.sourceCount &&
    native.sources.every(
      (source) =>
        Number.isInteger(source.owner) &&
        source.owner > 0 &&
        typeof source.path === 'string' &&
        source.path.length > 0 &&
        source.source_tone_replayed === true &&
        source.source_to_world?.length === 9 &&
        source.source_to_world.every(Number.isFinite),
    ) &&
    native.sampled_cell_count > 0 &&
    native.candidate_measurement_count >= native.sampled_cell_count &&
    native.delivered_pixels === contract.canvas[0] * contract.canvas[1] &&
    native.unresolved_pixel_count === 0,
  );
  const qualityPass =
    nativeEvidencePass &&
    gate?.verdict === 'pass' &&
    gate.roi_count >= 8 &&
    contract.requiredCriteria.length >= 8 &&
    contract.requiredCriteria.every((name) => names.has(name));
  if (!regressionOnly) require(qualityPass, 'legacy_quality_gate_not_passed');
  return {
    pass: failures.length === 0,
    failures,
    status: failures.length
      ? 'rejected'
      : qualityPass
        ? 'quality_and_regression_pass'
        : 'regression_only_quality_pending',
    qualityPass,
    nativeEvidencePass,
    regressionOnly,
  };
}

async function sha(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}
const read = async (path) => JSON.parse(await readFile(path, 'utf8'));
async function main() {
  const args = process.argv.slice(2);
  const positional = args.filter((arg) => !arg.startsWith('--'));
  if (positional.length !== 2)
    throw new Error(
      'Usage: node tests/focus-stack-lanting-guard.mjs <contract.json> <lanting-run-directory> [--regression-only]',
    );
  const contractPath = resolve(positional[0]);
  const contract = await read(contractPath);
  const base = resolve(positional[1]);
  const analysis = await read(resolve(base, 'analysis.json'));
  const folders = (await readdir(resolve(base, 'run'))).filter((name) => name.endsWith('stack-reports'));
  if (folders.length !== 1) throw new Error('Expected exactly one Stack_Report directory');
  const folder = folders[0];
  const reports = (await readdir(resolve(base, 'run', folder))).filter(
    (name) => name.startsWith('stack-report-') && name.endsWith('.json'),
  );
  if (reports.length !== 1) throw new Error('Expected exactly one Stack_Report');
  const reportPath = resolve(base, 'run', folder, reports[0]);
  const report = await read(reportPath);
  const execution = await read(resolve(base, 'execution.json'));
  const paper = await read(resolve(base, '..', 'paper-tone-audit.json'));
  const output = report.output;
  const actual = {
    sourceCount: report.input.source_count,
    stationCount: report.grouping.station_count,
    sourcePaths: execution.inputs.map((source) => source.path),
    inputsSha: createHash('sha256').update(JSON.stringify(execution.inputs)).digest('hex'),
    inputsUnchanged: analysis.checks.inputs_unchanged,
    outputSha: await sha(output.path),
    referenceSha: await sha(contract.referencePath),
    ownerSha: await sha(resolve(base, 'diagnostics', 'owner.png')),
    coverageSha: await sha(resolve(base, 'diagnostics', 'coverage.png')),
    canvas: [output.width, output.height],
    covered: report.composition.source_covered_pixels,
    owned: report.composition.source_owned_pixels,
    written: output.written,
    artifactValidation: output.artifact_validation,
    alphaPreserved: output.alpha_preserved,
    bitDepth: output.bit_depth,
    paper: paper.summary,
    qualityGate: report.quality_gate,
  };
  if (analysis.output_sha256 !== actual.outputSha) throw new Error('TIFF hash differs from recorded analysis');
  const result = evaluateLantingGuard(contract, actual, { regressionOnly: args.includes('--regression-only') });
  result.contractSha = await sha(contractPath);
  result.reportSha = await sha(reportPath);
  result.outputSha = actual.outputSha;
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  if (!result.pass) process.exitCode = 1;
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
