// Exercise the real CLI's verdict, report and baseline handling without RAW work.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const runner = fileURLToPath(new URL('./focus-stack-regression.mjs', import.meta.url));

function fixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'stack-verdict-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const repo = path.join(directory, 'repo');
  const data = path.join(directory, 'data');
  const out = path.join(directory, 'output');
  fs.mkdirSync(path.join(repo, 'tests', 'acceptance'), { recursive: true });
  fs.mkdirSync(path.join(repo, 'src-tauri'));
  fs.mkdirSync(path.join(data, 'fixture'), { recursive: true });
  fs.copyFileSync(runner, path.join(repo, 'tests', 'focus-stack-regression.mjs'));
  fs.writeFileSync(path.join(repo, 'tests', 'acceptance', 'focus-stack-datasets.json'), JSON.stringify({
    rootEnvironment: 'RAW_EDITOR_FOCUS_STACK_DATA_ROOT', datasets: [{
      id: 'fixture', directory: 'fixture', expectedSourceCount: 2,
      sourceExtensions: ['.nef'], excludePatterns: [],
    }],
  }));
  for (const name of ['one.nef', 'two.nef']) fs.writeFileSync(path.join(data, 'fixture', name), 'fixture');
  const baseline = path.join(repo, 'tests', 'acceptance', 'focus-stack-regression-baselines.json');
  const initialBaseline = JSON.stringify({ schemaVersion: 1, baselines: [] });
  fs.writeFileSync(baseline, initialBaseline);
  const binary = path.join(directory, 'fake-harness.mjs');
  fs.writeFileSync(binary, `#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
const env = process.env;
const output = env.RAW_EDITOR_STACK_ACCEPTANCE_OUTPUT;
const previewOnly = Boolean(env.RAW_EDITOR_STACK_ACCEPTANCE_PREVIEW_ONLY);
const scenario = env.STACK_VERDICT_SCENARIO;
fs.mkdirSync(env.RAW_EDITOR_STACK_REPORT_DIR, {recursive:true});
const reportPath = path.join(env.RAW_EDITOR_STACK_REPORT_DIR, 'report.json');
const report = {quality_gate:{verdict:'pass'}, output:{written:!previewOnly,
  artifact_validation:previewOnly?'not_run':'pass',
  format:'TIFF', path:output, width:2, height:3, bit_depth:16, icc:'sRGB',
  alpha_preserved:true, tiff_extra_samples:[2]}};
if (scenario === 'framed') report.composition = {publication_crop:{x:1,y:1,width:2,height:3}};
if (scenario === 'source-owned' || scenario === 'source-unresolved') {
  report.composition = {source_covered_pixels:6,source_owned_pixels:6};
  report.quality_gate.legacy_native_evidence = {unresolved_pixel_count:scenario === 'source-unresolved' ? 1 : 0,
    sources:env.RAW_EDITOR_FOCUS_STACK_PATHS.split(path.delimiter).map((path,i)=>({owner:i+1,path}))};
}
if (scenario === 'missing-report-export') report.output.written = false;
if (scenario === 'unverified-report-export') delete report.output.artifact_validation;
if (scenario === 'missing-alpha-tag') report.output.tiff_extra_samples = null;
if (!previewOnly) fs.writeFileSync(output, 'fake TIFF: CLI contract test only');
fs.writeFileSync(reportPath, JSON.stringify(report));
fs.writeFileSync(env.RAW_EDITOR_STACK_ACCEPTANCE_METRICS,
  JSON.stringify({inputs:{ordered:2,missing_from_order:[]},self:{}}));
console.log('Stack report written to '+reportPath);
console.log('focus-stack result: 2x3');
if (scenario === 'native-rejected') {
  console.log("thread 'fixture' panicked at fixture.rs:1:1:\\nnative artifact alpha declaration failed; report: "+reportPath);
  process.exitCode=101;
} else console.log('test result: ok. 1 passed; 0 failed');
`, { mode: 0o755 });
  const invoke = (scenario, extra = []) => spawnSync(process.execPath, [
    path.join(repo, 'tests', 'focus-stack-regression.mjs'), '--dataset', 'fixture',
    '--scale', '1', '--label', 'test', '--out-dir', out, '--binary', binary, ...extra,
  ], { encoding: 'utf8', env: { ...process.env, RAW_EDITOR_FOCUS_STACK_DATA_ROOT: data,
    RAW_EDITOR_STACK_ACCEPTANCE_COVERAGE_ALPHA: '1', STACK_VERDICT_SCENARIO: scenario } });
  return { directory, repo, out, baseline, initialBaseline, invoke,
    result: () => JSON.parse(fs.readFileSync(path.join(out, 'fixture-s1-test.summary.json'), 'utf8')) };
}

test('verified native export succeeds without a baseline', (t) => {
  const f = fixture(t);
  const run = f.invoke('valid');
  assert.equal(run.status, 0, run.stderr + run.stdout);
  assert.equal(f.result().summary.artifactContract, 'pass');
  assert.equal(f.result().verdict, 'no-baseline');
});

for (const scenario of ['missing-report-export', 'unverified-report-export', 'missing-alpha-tag']) {
  test(`QG pass cannot conceal ${scenario}`, (t) => {
    const f = fixture(t);
    assert.equal(f.invoke(scenario).status, 1);
    assert.equal(f.result().summary.qualityGate, 'pass');
    assert.equal(f.result().summary.artifactContract, 'fail');
    assert.equal(f.result().verdict, 'failed');
  });
}

test('native rejection remains a hard failure even with --rebaseline', (t) => {
  const f = fixture(t);
  assert.equal(f.invoke('native-rejected', ['--rebaseline']).status, 1);
  assert.match(f.result().failures.join('\n'), /native artifact alpha declaration failed/);
  assert.equal(f.result().verdict, 'failed');
  assert.equal(fs.readFileSync(f.baseline, 'utf8'), f.initialBaseline);
});

test('preview-only does not claim a verified TIFF export', (t) => {
  const f = fixture(t);
  assert.equal(f.invoke('valid', ['--preview-only']).status, 0);
  assert.equal(f.result().summary.artifactContract, 'not-run-preview');
  assert.equal(fs.existsSync(path.join(f.out, 'fixture-s1-test.tiff')), false);
});

for (const nativeAcceptance of ['rejected: native tone boundary failed', 'passed-or-not-applicable']) {
  test(`legacy unverified summary cannot become a baseline (${nativeAcceptance})`, (t) => {
    const f = fixture(t);
    const summaryPath = path.join(f.directory, 'legacy.json');
    fs.writeFileSync(summaryPath, JSON.stringify({ failures: [], summary: { nativeAcceptance, previewOnly: false } }));
    const run = spawnSync(process.execPath, [path.join(f.repo, 'tests', 'focus-stack-regression.mjs'),
      '--record-summary', summaryPath], { encoding: 'utf8' });
    assert.equal(run.status, 1);
    assert.equal(fs.readFileSync(f.baseline, 'utf8'), f.initialBaseline);
  });
}


test('framed artwork baseline does not replace historical source-union baseline or its floors', (t) => {
  const f = fixture(t);
  let run = f.invoke('valid', ['--write-output', '--update-baseline']);
  assert.equal(run.status, 0, run.stderr + run.stdout);
  const old = JSON.parse(fs.readFileSync(f.baseline, 'utf8'));
  delete old.baselines[0].publicationCropMode;
  fs.writeFileSync(f.baseline, JSON.stringify(old));
  const historical = structuredClone(old.baselines[0]);
  run = f.invoke('framed', ['--write-output', '--update-baseline']);
  assert.equal(run.status, 0, run.stderr + run.stdout);
  assert.equal(f.result().summary.publicationCropMode, 'framed-artwork');
  const saved = JSON.parse(fs.readFileSync(f.baseline, 'utf8'));
  assert.equal(saved.baselines.length, 2);
  assert.deepEqual(saved.baselines.find((entry) => !entry.publicationCropMode), historical);
  run = f.invoke('valid', ['--write-output']);
  assert.equal(run.status, 0, run.stderr + run.stdout);
  assert.equal(f.result().verdict, 'identical');
});

test('legacy ownership observation uses native source evidence and fails closed for unresolved pixels', (t) => {
  const f = fixture(t);
  let run = f.invoke('source-owned', ['--write-output']);
  assert.equal(run.status, 0, run.stderr + run.stdout);
  assert.equal(f.result().summary.ownershipVerified, true);
  run = f.invoke('source-unresolved', ['--write-output']);
  assert.equal(run.status, 0, run.stderr + run.stdout);
  assert.equal(f.result().summary.ownershipVerified, false);
});
