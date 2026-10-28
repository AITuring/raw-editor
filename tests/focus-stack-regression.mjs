// Real-data focus-stack regression runner.
//
// Runs the ignored Rust harness `real_focus_stack_fixture_from_env` on one
// catalogued dataset, writes every artifact under an output directory outside
// the dataset (default /private/tmp/raw-editor-regression), and checks:
//   - every input entered the stitching order (and, when the streaming
//     compositor ran, every input was streamed);
//   - the per-pixel source ownership verification line, when the baseline has it;
//   - the reference rendering is byte-identical before and after the run;
//   - output dimensions and digests against the recorded baseline;
//   - when the output changed, the registered tile-sharpness floors.
//
// Usage:
//   RAW_EDITOR_FOCUS_STACK_DATA_ROOT=/Users/dp/Downloads/书画拼图 \
//     npm run focus-stack-regression -- --dataset lanting [--scale 0.1] [--label n17]
//   Options: --subset <id> | --limit <n>, --write-output, --preview-only,
//            --out-dir <dir>, --binary <cargo test --no-run executable>,
//            --update-baseline, --record-summary <run.summary.json>,
//            --reuse-artifacts (re-evaluate the last run), --dry-run
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..');
const manifestPath = path.join(here, 'acceptance', 'focus-stack-datasets.json');
const baselinePath = path.join(here, 'acceptance', 'focus-stack-regression-baselines.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const CROPS = ['top-left', 'top-center', 'top-right', 'bottom-left', 'bottom-center', 'bottom-right'];
const PREVIEW_EDGE = 1200;

// Floors applied only when the output differs from the baseline digest. They
// are relative to the recorded baseline, which was visually verified as sharp
// everywhere; an identical digest needs no floor.
const FLOORS = {
  referenceRatioMedian: 0.9,
  referenceRatioP05: 0.85,
  referenceThirdMedian: 0.85,
  softTileFractionSlack: 0.01,
  lowContrastFractionSlack: 0.01,
  selfSharpnessMedian: 0.9,
  selfSharpnessP10: 0.85,
};

function parseArgs(argv) {
  const options = { scale: 0.1, label: 'run', outDir: '/private/tmp/raw-editor-regression' };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    const value = () => {
      const next = argv[index + 1];
      if (next === undefined) throw new Error(`${argument} needs a value`);
      index += 1;
      return next;
    };
    switch (argument) {
      case '--dataset':
        options.dataset = value();
        break;
      case '--subset':
        options.subset = value();
        break;
      case '--limit':
        options.limit = Number(value());
        break;
      case '--scale':
        options.scale = Number(value());
        break;
      case '--label':
        options.label = value();
        break;
      case '--out-dir':
        options.outDir = path.resolve(value());
        break;
      case '--write-output':
        options.writeOutput = true;
        break;
      case '--preview-only':
        options.previewOnly = true;
        break;
      case '--update-baseline':
        options.updateBaseline = true;
        break;
      case '--record-summary':
        options.recordSummary = path.resolve(value());
        break;
      case '--binary':
        options.binary = path.resolve(value());
        break;
      case '--reuse-artifacts':
        options.reuseArtifacts = true;
        break;
      case '--dry-run':
        options.dryRun = true;
        break;
      default:
        throw new Error(`unknown argument ${argument}`);
    }
  }
  if (options.recordSummary) return options;
  if (!options.dataset) throw new Error('--dataset is required');
  if (!(options.scale > 0 && options.scale <= 1)) throw new Error('--scale must be in (0, 1]');
  if (options.limit !== undefined && !(Number.isInteger(options.limit) && options.limit >= 2)) {
    throw new Error('--limit must be an integer >= 2');
  }
  return options;
}

function patternToRegExp(pattern) {
  const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`^${escaped.replaceAll('*', '.*')}$`, 'i');
}

function datasetInputs(dataset, root) {
  const directory = path.join(root, dataset.directory);
  const excluded = (name) => dataset.excludePatterns.some((pattern) => patternToRegExp(pattern).test(name));
  const files = fs
    .readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name)
    .filter((name) => dataset.sourceExtensions.includes(path.extname(name).toLowerCase()))
    .filter((name) => !excluded(name))
    .sort((left, right) => (left < right ? -1 : left > right ? 1 : 0));
  return { directory, files };
}

function sha256File(filePath) {
  const hash = crypto.createHash('sha256');
  const descriptor = fs.openSync(filePath, 'r');
  const buffer = Buffer.allocUnsafe(8 * 1024 * 1024);
  try {
    let read;
    while ((read = fs.readSync(descriptor, buffer, 0, buffer.length, null)) > 0) {
      hash.update(buffer.subarray(0, read));
    }
  } finally {
    fs.closeSync(descriptor);
  }
  return hash.digest('hex');
}

function referenceState(dataset, directory) {
  if (!dataset.referenceOutput) return null;
  const filePath = path.join(directory, dataset.referenceOutput);
  if (!fs.existsSync(filePath)) return { filePath, missing: true };
  return { filePath, bytes: fs.statSync(filePath).size, sha256: sha256File(filePath) };
}

function parseLog(text) {
  const streamed = new Set();
  for (const match of text.matchAll(/Streaming '(.+?)': projected region/g)) streamed.add(match[1]);
  const result = text.match(/focus-stack result: (\d+)x(\d+)/);
  const ownership = text.match(/Verified covered output: (\d+)x(\d+); every pixel has source ownership/);
  const report = text.match(/Stack report written to (\S+\.json)/);
  const seconds = text.match(/Stitching completed in ([\d.]+)s/);
  // The harness checks a layered native report after every artifact is
  // written; such a rejection is a tracked verdict, not a crashed run.
  const nativeRejection = text.match(/panicked at [^\n]*\n((?:native |missing or duplicate quality criterion)[^\n;]*)/);
  return {
    streamed,
    result: result ? { width: Number(result[1]), height: Number(result[2]) } : null,
    ownershipVerified: Boolean(ownership),
    reportPath: report?.[1] ?? null,
    stitchSeconds: seconds ? Number(seconds[1]) : null,
    passed: /test result: ok\. 1 passed/.test(text),
    nativeRejection: nativeRejection?.[1] ?? null,
  };
}

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function metricSummary(metrics) {
  if (!metrics) return null;
  const reference = metrics.reference;
  return {
    self: {
      contentTiles: metrics.self?.content_tiles ?? 0,
      sharpnessMedian: metrics.self?.sharpness?.median ?? null,
      sharpnessP10: metrics.self?.sharpness?.p10 ?? null,
      thirds: (metrics.self?.thirds ?? []).map((band) => ({
        band: band.band,
        sharpnessMedian: band.sharpness_median,
      })),
    },
    reference: reference
      ? {
          file: reference.file,
          inliers: reference.registration?.inliers ?? 0,
          localScale: reference.registration?.local_scale ?? null,
          contentTiles: reference.tiles?.content ?? 0,
          ratioMedian: reference.ratio?.median ?? null,
          ratioP05: reference.ratio?.p05 ?? null,
          ratioP10: reference.ratio?.p10 ?? null,
          softTileFraction: reference.normalized_below_0_5 ?? null,
          lowContrastFraction: reference.contrast_below_0_5 ?? null,
          thirds: (reference.thirds ?? []).map((band) => ({
            band: band.band,
            ratioMedian: band.ratio_median,
            ratioP10: band.ratio_p10,
          })),
        }
      : null,
  };
}

function compareFloors(current, baseline, failures) {
  if (!current || !baseline) return;
  const atLeast = (name, value, floor) => {
    if (value === null || value === undefined || floor === null || floor === undefined) return;
    if (value < floor) failures.push(`${name} ${value.toFixed(4)} < floor ${floor.toFixed(4)}`);
  };
  if (baseline.reference && current.reference) {
    const base = baseline.reference;
    const now = current.reference;
    atLeast('reference ratio median', now.ratioMedian, base.ratioMedian * FLOORS.referenceRatioMedian);
    atLeast('reference ratio p05', now.ratioP05, base.ratioP05 * FLOORS.referenceRatioP05);
    for (const band of base.thirds) {
      const match = now.thirds.find((candidate) => candidate.band === band.band);
      atLeast(
        `reference ${band.band} ratio median`,
        match?.ratioMedian,
        band.ratioMedian * FLOORS.referenceThirdMedian,
      );
    }
    if (now.softTileFraction > base.softTileFraction + FLOORS.softTileFractionSlack) {
      failures.push(
        `soft tile fraction ${now.softTileFraction.toFixed(4)} > ${base.softTileFraction.toFixed(4)} + slack`,
      );
    }
    if (now.lowContrastFraction > base.lowContrastFraction + FLOORS.lowContrastFractionSlack) {
      failures.push(
        `low-contrast tile fraction ${now.lowContrastFraction.toFixed(4)} > ${base.lowContrastFraction.toFixed(4)} + slack`,
      );
    }
  } else if (baseline.reference && !current.reference) {
    failures.push('reference metrics are missing although the baseline has them');
  }
  atLeast(
    'self sharpness median',
    current.self.sharpnessMedian,
    baseline.self.sharpnessMedian * FLOORS.selfSharpnessMedian,
  );
  atLeast('self sharpness p10', current.self.sharpnessP10, baseline.self.sharpnessP10 * FLOORS.selfSharpnessP10);
}

function baselineKey(entry) {
  return `${entry.dataset}|${entry.subset ?? ''}|${entry.scale}|${entry.previewOnly}`;
}

function readBaselines() {
  return readJson(baselinePath) ?? { schemaVersion: 1, baselines: [] };
}

// Only sanitized counts, sizes, digests and metric summaries are recorded;
// logs, reports, previews and absolute temporary paths stay out of the tree.
function recordBaseline(summary) {
  const baselines = readBaselines();
  const commit = spawnSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: repoRoot, encoding: 'utf8' });
  const entry = {
    ...summary,
    recordedAt: new Date().toISOString().slice(0, 10),
    commit: commit.stdout.trim() || null,
  };
  delete entry.elapsedSeconds;
  delete entry.stitchSeconds;
  baselines.baselines = baselines.baselines.filter((candidate) => baselineKey(candidate) !== baselineKey(entry));
  baselines.baselines.push(entry);
  baselines.baselines.sort((left, right) => baselineKey(left).localeCompare(baselineKey(right)));
  fs.writeFileSync(baselinePath, `${JSON.stringify(baselines, null, 2)}\n`);
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.recordSummary) {
    const recorded = readJson(options.recordSummary);
    if (!recorded?.summary) throw new Error(`${options.recordSummary} is not a regression summary`);
    if (recorded.failures.length > 0) throw new Error('a failed run cannot become a baseline');
    recordBaseline(recorded.summary);
    console.log(`focus-stack-regression: recorded ${recorded.runId} in ${path.relative(repoRoot, baselinePath)}`);
    return;
  }
  const dataset = manifest.datasets.find((candidate) => candidate.id === options.dataset);
  if (!dataset) throw new Error(`unknown dataset ${options.dataset}`);
  const root = path.resolve(process.env[manifest.rootEnvironment] ?? manifest.defaultRoot);
  const { directory, files } = datasetInputs(dataset, root);
  if (files.length !== dataset.expectedSourceCount) {
    throw new Error(`${dataset.id}: expected ${dataset.expectedSourceCount} inputs, found ${files.length}`);
  }
  let limit = options.limit;
  let subsetId = null;
  if (options.subset) {
    const subset = (dataset.regressionSubsets ?? []).find((candidate) => candidate.id === options.subset);
    if (!subset) throw new Error(`${dataset.id}: unknown subset ${options.subset}`);
    subsetId = subset.id;
    limit = subset.first;
  } else if (limit !== undefined) {
    subsetId = `first-${limit}`;
  }
  const inputs = (limit ? files.slice(0, limit) : files).map((name) => path.join(directory, name));
  const scaleTag = `s${String(options.scale).replace('.', 'p')}`;
  const runPrefix = subsetId?.startsWith(dataset.id) ? [subsetId] : [dataset.id, subsetId];
  const runId = [...runPrefix, scaleTag, options.label].filter(Boolean).join('-');
  fs.mkdirSync(options.outDir, { recursive: true });
  if (path.resolve(options.outDir).startsWith(path.resolve(root))) {
    throw new Error('the output directory must be outside the dataset root');
  }
  const outputPath = path.join(options.outDir, `${runId}.tiff`);
  const reportDir = path.join(options.outDir, `${runId}-stack-reports`);
  const metricsPath = path.join(options.outDir, `${runId}.metrics.json`);
  const logPath = path.join(options.outDir, `${runId}.log`);
  const previewOnly = options.previewOnly || (options.scale < 1 && !options.writeOutput);
  const env = {
    ...process.env,
    CARGO_NET_OFFLINE: 'true',
    RAW_EDITOR_FOCUS_STACK_PATHS: inputs.join(path.delimiter),
    RAW_EDITOR_STACK_ACCEPTANCE_OUTPUT: outputPath,
    RAW_EDITOR_STACK_REPORT_DIR: reportDir,
    RAW_EDITOR_STACK_ACCEPTANCE_CROPS: '1',
    RAW_EDITOR_STACK_ACCEPTANCE_PREVIEW_EDGE: String(PREVIEW_EDGE),
    RAW_EDITOR_STACK_ACCEPTANCE_METRICS: metricsPath,
  };
  delete env.RAW_EDITOR_FOCUS_STACK_REFERENCE_JPEG;
  if (options.scale < 1) env.RAW_EDITOR_STACK_ACCEPTANCE_RENDER_SCALE = String(options.scale);
  else delete env.RAW_EDITOR_STACK_ACCEPTANCE_RENDER_SCALE;
  if (previewOnly) env.RAW_EDITOR_STACK_ACCEPTANCE_PREVIEW_ONLY = '1';
  else delete env.RAW_EDITOR_STACK_ACCEPTANCE_PREVIEW_ONLY;
  if (dataset.referenceOutput) {
    env.RAW_EDITOR_STACK_ACCEPTANCE_METRICS_REFERENCE = path.join(directory, dataset.referenceOutput);
  } else {
    delete env.RAW_EDITOR_STACK_ACCEPTANCE_METRICS_REFERENCE;
  }
  const command = [
    'test',
    '--release',
    '--manifest-path',
    'src-tauri/Cargo.toml',
    '--lib',
    'real_focus_stack_fixture_from_env',
    '--',
    '--ignored',
    '--nocapture',
    '--test-threads=1',
  ];
  console.log(`focus-stack-regression: ${runId}: ${inputs.length} inputs, scale ${options.scale}`);
  console.log(`  log: ${logPath}`);
  if (options.dryRun) {
    console.log(JSON.stringify({ runId, inputs: inputs.length, previewOnly, command }, null, 2));
    return;
  }

  const referenceBefore = referenceState(dataset, directory);
  if (!options.reuseArtifacts) {
    fs.rmSync(reportDir, { recursive: true, force: true });
    fs.rmSync(metricsPath, { force: true });
  }
  const logDescriptor = options.reuseArtifacts ? null : fs.openSync(logPath, 'w');
  const started = Date.now();
  // `--binary` runs a test executable produced by `cargo test --release --no-run`
  // directly, so source edits made while a run is in flight cannot leak into it.
  // `--reuse-artifacts` re-evaluates the log, report and images of the last run.
  const run = options.reuseArtifacts
    ? { status: /test result: ok\. 1 passed/.test(fs.readFileSync(logPath, 'utf8')) ? 0 : 101 }
    : options.binary
      ? spawnSync(
          options.binary,
          command.slice(command.indexOf('real_focus_stack_fixture_from_env')).filter((arg) => arg !== '--'),
          {
            cwd: path.join(repoRoot, 'src-tauri'),
            env,
            stdio: ['ignore', logDescriptor, logDescriptor],
          },
        )
      : spawnSync('cargo', command, {
          cwd: repoRoot,
          env,
          stdio: ['ignore', logDescriptor, logDescriptor],
        });
  if (logDescriptor !== null) fs.closeSync(logDescriptor);
  const elapsedSeconds = (Date.now() - started) / 1000;
  const referenceAfter = referenceState(dataset, directory);
  const log = parseLog(fs.readFileSync(logPath, 'utf8'));
  const metrics = readJson(metricsPath);
  const report = log.reportPath ? readJson(log.reportPath) : null;
  const failures = [];
  const notices = [];

  // A layered native rejection raised after every artifact was written is
  // compared with the baseline below; any other non-zero exit is a crash.
  const nativeRejectedAfterOutput = run.status !== 0 && log.nativeRejection && log.result && metrics;
  if ((run.status !== 0 || !log.passed) && !nativeRejectedAfterOutput) {
    const cause = run.error ? `, ${run.error.message}` : run.signal ? `, signal ${run.signal}` : '';
    failures.push(`harness failed (exit ${run.status}${cause}); see ${logPath}`);
  }
  if (!log.result) failures.push('no focus-stack result line in the log');
  if (!metrics) failures.push('acceptance metrics were not written');
  const missing = metrics?.inputs?.missing_from_order ?? [];
  if (missing.length > 0) failures.push(`inputs missing from the stitching order: ${missing.join(', ')}`);
  if (log.streamed.size > 0 && log.streamed.size !== inputs.length) {
    const absent = inputs.filter((input) => !log.streamed.has(input)).map((input) => path.basename(input));
    failures.push(
      `streaming compositor used ${log.streamed.size}/${inputs.length} inputs; absent: ${absent.join(', ')}`,
    );
  }
  if (referenceBefore && dataset.referenceSha256) {
    for (const state of [referenceBefore, referenceAfter]) {
      if (state.missing || state.sha256 !== dataset.referenceSha256 || state.bytes !== dataset.referenceBytes) {
        failures.push(`reference ${dataset.referenceOutput} differs from the frozen manifest digest`);
        break;
      }
    }
  } else if (referenceBefore && referenceAfter && referenceBefore.sha256 !== referenceAfter.sha256) {
    failures.push(`reference ${dataset.referenceOutput} changed during the run`);
  }

  const artifactDigest = (filePath) => (fs.existsSync(filePath) ? sha256File(filePath) : null);
  const stem = outputPath.replace(/\.tiff$/, '');
  const digests = {
    preview: artifactDigest(`${stem}.preview.jpg`),
    crops: Object.fromEntries(CROPS.map((name) => [name, artifactDigest(`${stem}.${name}.jpg`)])),
    appJpeg: previewOnly ? null : artifactDigest(`${stem}.app.jpg`),
  };
  const summary = {
    dataset: dataset.id,
    subset: subsetId,
    scale: options.scale,
    previewEdge: PREVIEW_EDGE,
    previewOnly,
    output: log.result,
    inputs: inputs.length,
    orderedInputs: metrics?.inputs?.ordered ?? null,
    streamedSources: log.streamed.size || null,
    ownershipVerified: log.ownershipVerified,
    selectedPath: report?.selected_path ?? null,
    stationCount: report?.grouping?.station_count ?? null,
    qualityGate: report?.quality_gate?.verdict ?? null,
    toneStatus: report?.tone?.status ?? null,
    nativeAcceptance: log.nativeRejection ? `rejected: ${log.nativeRejection}` : 'passed-or-not-applicable',
    stitchSeconds: log.stitchSeconds,
    elapsedSeconds,
    digests,
    metrics: metricSummary(metrics),
  };

  const baselines = readBaselines();
  const baseline = baselines.baselines.find((entry) => baselineKey(entry) === baselineKey(summary));
  let verdict = 'no-baseline';
  if (log.nativeRejection) {
    notices.push(`layered native acceptance rejected the report: ${log.nativeRejection}`);
  }
  if (baseline) {
    if (baseline.ownershipVerified && !summary.ownershipVerified) {
      failures.push('per-pixel source ownership verification disappeared');
    }
    if (baseline.nativeAcceptance === 'passed-or-not-applicable' && log.nativeRejection) {
      failures.push('layered native acceptance regressed from the baseline');
    }
    if (baseline.streamedSources && summary.streamedSources !== baseline.streamedSources) {
      failures.push(`streamed sources ${summary.streamedSources} != baseline ${baseline.streamedSources}`);
    }
    const identical =
      digests.preview === baseline.digests.preview &&
      CROPS.every((name) => digests.crops[name] === baseline.digests.crops[name]) &&
      (summary.previewOnly || digests.appJpeg === baseline.digests.appJpeg);
    if (identical) {
      verdict = 'identical';
    } else {
      verdict = 'changed';
      if (summary.output?.width !== baseline.output?.width || summary.output?.height !== baseline.output?.height) {
        notices.push(
          `output size changed: ${summary.output?.width}x${summary.output?.height} (baseline ${baseline.output.width}x${baseline.output.height})`,
        );
      }
      compareFloors(summary.metrics, baseline.metrics, failures);
      notices.push('output changed; sharpness floors were applied — inspect the preview and crops visually');
    }
  }
  if (options.updateBaseline && failures.length === 0) {
    recordBaseline(summary);
    notices.push(`baseline recorded in ${path.relative(repoRoot, baselinePath)}`);
  }

  const result = { runId, verdict, failures, notices, summary, logPath };
  fs.writeFileSync(path.join(options.outDir, `${runId}.summary.json`), `${JSON.stringify(result, null, 2)}\n`);
  console.log(
    JSON.stringify(
      {
        runId,
        verdict,
        failures,
        notices,
        output: summary.output,
        inputs: summary.inputs,
        streamedSources: summary.streamedSources,
        metrics: summary.metrics,
      },
      null,
      2,
    ),
  );
  if (failures.length > 0) process.exitCode = 1;
}

try {
  main();
} catch (error) {
  console.error(`focus-stack-regression: ${error.message}`);
  process.exitCode = 1;
}
