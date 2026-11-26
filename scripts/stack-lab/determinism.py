#!/usr/bin/env python3
"""Run a bounded determinism & resilience matrix for one frozen material scope.

This tool deliberately measures only repeatability and fail-closed resilience
(thread counts, renamed/permuted input order, concurrent load, cancellation,
and OOM memory-gate enforcement).  It never changes quality thresholds or
converts a rejected run into an accepted one.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time

from common import (
    ACCEPTANCE,
    DATA,
    dataset_entry,
    dataset_files,
    read,
    repo_root,
    sha,
    valid_label,
    verify_binary,
    write,
)

HERE = Path(__file__).resolve().parent
MAX_THREADS = 8  # exclusive slot contract; higher requests would be capped.
ALL_OPTIONAL_SCENARIOS = (
    'order_permutation',
    'concurrent_load',
    'cancellation',
    'oom',
    'provider_fallback',
)
SUPPORTED_SCENARIOS = (
    'threads',
    'order_permutation',
    'concurrent_load',
    'cancellation',
    'oom',
)


def stable_signature(analysis):
    """Return verdict/output/source fields that must agree across runs.

    Timing, RSS and report paths are intentionally excluded: they are resource
    observations, not deterministic output.  Rejected runs remain rejected and
    are compared by the same contract as accepted runs.
    """
    qg = analysis.get('quality_gate') or {}
    sources = analysis.get('sources') or {}
    output = analysis.get('output') or {}
    checks = analysis.get('checks') or {}
    return {
        'status': analysis.get('status'),
        'acceptance_state': analysis.get('acceptance_state'),
        'checks': {key: checks.get(key) for key in sorted(checks)},
        'quality_gate': {
            'verdict': qg.get('verdict'),
            'roi_count': qg.get('roi_count'),
            'criteria': qg.get('criteria'),
            'unresolved_pixels': qg.get('unresolved_pixels'),
        },
        'closure': analysis.get('closure'),
        'outcome': analysis.get('outcome'),
        'sources': {key: sources.get(key) for key in (
            'inputs', 'report_sources', 'ordered', 'streamed', 'lending',
            'expected_skipped', 'covered_pixels', 'owned_pixels')},
        'output': {key: output.get(key) for key in (
            'width', 'height', 'bit_depth', 'alpha_preserved', 'written',
            'artifact_validation')},
        'output_sha256': analysis.get('output_sha256'),
    }


def signature_sha(signature):
    encoded = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def compare_signatures(expected, actual):
    if expected == actual:
        return []
    failures = []
    keys = sorted(set(expected) | set(actual))
    for key in keys:
        if expected.get(key) != actual.get(key):
            failures.append({'field': key, 'expected': expected.get(key), 'actual': actual.get(key)})
    return failures


def classify_matrix_status(runs):
    """Separate a preflight/runner environment block from a determinism failure."""
    blocked = [run.get('blocking_reason') for run in runs]
    if runs and all(blocked):
        reasons = set(blocked)
        return 'blocked', (next(iter(reasons)) if len(reasons) == 1 else 'multiple_runner_blocks')
    return ('pass' if all(not run['mismatches'] for run in runs) else 'fail'), None


def parse_threads(value):
    values = []
    for token in value.split(','):
        try:
            thread_count = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f'invalid thread count: {token}') from exc
        if thread_count < 1 or thread_count > MAX_THREADS:
            raise argparse.ArgumentTypeError(
                f'threads must be between 1 and {MAX_THREADS} under the exclusive slot contract')
        if thread_count not in values:
            values.append(thread_count)
    if not values:
        raise argparse.ArgumentTypeError('at least one thread count is required')
    return values


def parse_scenarios(value):
    values = []
    for raw in value.split(','):
        token = raw.strip()
        if not token:
            continue
        if token not in SUPPORTED_SCENARIOS:
            raise argparse.ArgumentTypeError(
                f'unsupported scenario "{token}"; supported: {", ".join(SUPPORTED_SCENARIOS)}')
        if token not in values:
            values.append(token)
    if not values:
        raise argparse.ArgumentTypeError('at least one scenario is required')
    return values


def remaining_not_run(scenarios):
    selected = set(scenarios)
    return [item for item in ALL_OPTIONAL_SCENARIOS if item not in selected]


def build_permuted_symlink_view(files, dest_dir, mode='reverse', seed=20261126):
    """Stage read-only symlinks with renamed/permuted order in a scratch dir.

    Never modifies the original source directory.  Preserves exact file
    extensions so dataset discovery and RAW/JPEG decoders see identical bytes.
    """
    source_paths = [Path(item).resolve() for item in files]
    if not source_paths:
        raise ValueError('cannot build permuted symlink view from empty file list')
    for src in source_paths:
        if not src.is_file():
            raise ValueError(f'missing source file for permutation view: {src}')
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    ordered = list(source_paths)
    if mode == 'reverse':
        ordered.reverse()
    elif mode == 'shuffle':
        rng = random.Random(seed)
        rng.shuffle(ordered)
    else:
        raise ValueError(f'unknown permutation mode: {mode}')

    mapping = []
    width = max(4, len(str(len(ordered))))
    for index, src in enumerate(ordered):
        alias = dest_dir / f'perm_{index:0{width}d}_{src.name}'
        alias.symlink_to(src)
        mapping.append({'link': str(alias), 'target': str(src)})
    return mapping


def evaluate_cancellation_probe(exit_code, analysis, tiff_published):
    """Validate that a cancelled run exits non-accepted without publishing output."""
    mismatches = []
    state = (analysis or {}).get('acceptance_state') or (analysis or {}).get('status')
    if exit_code == 0:
        mismatches.append({'field': 'cancellation.exit_code', 'expected': 'non-zero', 'actual': exit_code})
    if state == 'accepted' or (analysis or {}).get('status') == 'quality_accepted':
        mismatches.append({'field': 'cancellation.state', 'expected': 'cancelled/non-accepted', 'actual': state})
    if tiff_published:
        mismatches.append({'field': 'cancellation.tiff_published', 'expected': False, 'actual': True})
    return mismatches


def evaluate_oom_probe(exit_code, analysis, tiff_published):
    """Validate fail-closed memory-gate enforcement without publishing output."""
    mismatches = []
    status = (analysis or {}).get('status')
    state = (analysis or {}).get('acceptance_state')
    memory = (analysis or {}).get('memory') or {}
    exceeded = bool(memory.get('threshold_exceeded')) or state == 'resource_failed' or status == 'resource_failed'
    if exit_code == 0:
        mismatches.append({'field': 'oom.exit_code', 'expected': 'non-zero', 'actual': exit_code})
    if not exceeded:
        mismatches.append({
            'field': 'oom.resource_failed',
            'expected': True,
            'actual': {'status': status, 'acceptance_state': state, 'threshold_exceeded': memory.get('threshold_exceeded')},
        })
    if tiff_published or state == 'accepted':
        mismatches.append({'field': 'oom.tiff_published', 'expected': False, 'actual': True})
    return mismatches


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--lane', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--subset')
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--label', required=True)
    parser.add_argument('--threads', type=parse_threads, default=[1, 2, 5, 8])
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument(
        '--scenarios',
        type=parse_scenarios,
        default=['threads'],
        help=f'comma-separated scenarios from: {", ".join(SUPPORTED_SCENARIOS)}',
    )
    parser.add_argument('--cancel-after-seconds', type=float, default=5.0)
    parser.add_argument('--oom-threshold-bytes', type=int, default=64 * 1024 * 1024)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    for label in (args.binary, args.lane, args.label):
        valid_label(label)
    if args.repeat < 2:
        parser.error('--repeat must be at least 2')
    if not 0 < args.scale <= 1:
        parser.error('--scale must be in (0, 1]')
    if args.cancel_after_seconds <= 0:
        parser.error('--cancel-after-seconds must be positive')
    if args.oom_threshold_bytes <= 0:
        parser.error('--oom-threshold-bytes must be positive')
    dataset_entry(args.dataset)
    return args


def _has_published_tiff(output_dir):
    run_dir = Path(output_dir) / 'run'
    if not run_dir.is_dir():
        return False
    return any(p.is_file() and p.stat().st_size > 0 for p in run_dir.glob('*.tiff'))


def run_once(args, root, output, thread_count, repeat, exclusive=True, extra_env=None, tag=None):
    label_suffix = tag or f't{thread_count}-r{repeat}'
    command = [sys.executable, str(HERE / 'run.py'), '--binary', args.binary,
               '--lane', args.lane, '--dataset', args.dataset, '--scale', str(args.scale),
               '--label', f'{args.label}-{label_suffix}', '--out', str(output)]
    if exclusive:
        command.append('--exclusive')
    if args.subset:
        command += ['--subset', args.subset]
    env = dict(os.environ, RAYON_NUM_THREADS=str(thread_count))
    if extra_env:
        env.update(extra_env)
    log_path = output.with_suffix('.driver.log')
    started = time.time()
    with log_path.open('x') as log:
        exit_code = subprocess.run(command, cwd=root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT).returncode
    driver_output = log_path.read_text(errors='replace')
    blocking_reason = None
    if 'could not start command with required nice priority' in driver_output:
        blocking_reason = 'required_nice_priority_unavailable'
    analysis_path = output / 'analysis.json'
    analysis = read(analysis_path) if analysis_path.is_file() else {
        'status': 'missing_analysis', 'checks': {}, 'output_sha256': None}
    return {
        'scenario': tag or 'threads',
        'thread_count': thread_count,
        'repeat': repeat,
        'directory': str(output),
        'driver_exit_code': exit_code,
        'blocking_reason': blocking_reason,
        'wall_seconds': time.time() - started,
        'analysis': analysis,
        'signature': stable_signature(analysis),
    }


def run_order_permutation_probe(args, root, directory, baseline):
    entry, files = dataset_files(args.dataset, args.subset)
    staging = Path(tempfile.mkdtemp(prefix='stack-lab-perm-', dir=str(directory)))
    dataset_dir = entry.get('directory') or entry['directories'][0]
    target_dir = staging / dataset_dir
    mapping = build_permuted_symlink_view(files, target_dir, mode='reverse')
    ref_name = entry.get('referenceOutput')
    if ref_name:
        ref_src = DATA / dataset_dir / ref_name
        if ref_src.is_file():
            (target_dir / ref_name).symlink_to(ref_src.resolve())

    bin_dir, _ = verify_binary(args.binary)
    out_dir = directory / f'{args.dataset}-perm-reverse'
    out_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable, str(HERE / 'slot.py'), 'exclusive',
        '--lane', args.lane, '--what', f'{args.dataset} order_permutation', '--',
        'node', str(root / 'tests/focus-stack-regression.mjs'),
        '--dataset', args.dataset, '--scale', f'{args.scale:g}',
        '--write-output', '--label', f'{args.lane}-{args.label}-perm',
        '--out-dir', str(out_dir / 'run'),
        '--binary', str(bin_dir / 'native-process-wrapper.py'),
    ]
    env = dict(os.environ, RAW_EDITOR_FOCUS_STACK_DATA_ROOT=str(staging), RAYON_NUM_THREADS=str(args.threads[-1]))
    log_path = out_dir.with_suffix('.driver.log')
    started = time.time()
    with log_path.open('x') as log:
        exit_code = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    driver_output = log_path.read_text(errors='replace')
    blocking_reason = (
        'required_nice_priority_unavailable'
        if 'could not start command with required nice priority' in driver_output
        else None
    )
    tiffs = sorted((out_dir / 'run').glob('*.tiff'))
    perm_sha = sha(tiffs[0]) if tiffs and tiffs[0].stat().st_size > 0 else None
    mismatches = []
    if blocking_reason:
        mismatches.append({'field': 'runner.blocking_reason', 'expected': None, 'actual': blocking_reason})
    elif baseline is not None and perm_sha != baseline.get('output_sha256'):
        mismatches.append({
            'field': 'order_permutation.output_sha256',
            'expected': baseline.get('output_sha256'),
            'actual': perm_sha,
        })
    return {
        'scenario': 'order_permutation',
        'mode': 'reverse_renamed_symlinks',
        'file_count': len(mapping),
        'directory': str(out_dir),
        'driver_exit_code': exit_code,
        'blocking_reason': blocking_reason,
        'wall_seconds': time.time() - started,
        'output_sha256': perm_sha,
        'mismatches': mismatches,
    }


def run_cancellation_probe(args, root, directory):
    out_dir = directory / f'{args.dataset}-cancel-probe'
    command = [
        sys.executable, str(HERE / 'run.py'), '--binary', args.binary,
        '--lane', args.lane, '--dataset', args.dataset, '--scale', str(args.scale),
        '--label', f'{args.label}-cancel', '--out', str(out_dir), '--exclusive',
    ]
    if args.subset:
        command += ['--subset', args.subset]
    log_path = out_dir.with_suffix('.driver.log')
    started = time.time()
    with log_path.open('x') as log:
        proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            proc.wait(timeout=args.cancel_after_seconds)
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=60)
    driver_output = log_path.read_text(errors='replace')
    blocking_reason = (
        'required_nice_priority_unavailable'
        if 'could not start command with required nice priority' in driver_output
        else None
    )
    analysis_path = out_dir / 'analysis.json'
    analysis = read(analysis_path) if analysis_path.is_file() else {}
    tiff_published = _has_published_tiff(out_dir)
    mismatches = (
        [{'field': 'runner.blocking_reason', 'expected': None, 'actual': blocking_reason}]
        if blocking_reason
        else evaluate_cancellation_probe(proc.returncode, analysis, tiff_published)
    )
    return {
        'scenario': 'cancellation',
        'directory': str(out_dir),
        'driver_exit_code': proc.returncode,
        'blocking_reason': blocking_reason,
        'wall_seconds': time.time() - started,
        'tiff_published': tiff_published,
        'mismatches': mismatches,
    }


def run_oom_probe(args, root, directory):
    bin_dir, _ = verify_binary(args.binary)
    out_dir = directory / f'{args.dataset}-oom-probe'
    out_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable, str(HERE / 'slot.py'), 'exclusive',
        '--lane', args.lane, '--what', f'{args.dataset} oom_probe', '--',
        'node', str(root / 'tests/focus-stack-regression.mjs'),
        '--dataset', args.dataset, '--scale', f'{args.scale:g}',
        '--write-output', '--label', f'{args.lane}-{args.label}-oom',
        '--out-dir', str(out_dir / 'run'),
        '--binary', str(bin_dir / 'native-process-wrapper.py'),
    ]
    if args.subset:
        command += ['--subset', args.subset]
    env = dict(
        os.environ,
        RAW_EDITOR_FOCUS_STACK_DATA_ROOT=str(DATA),
        RAW_EDITOR_STACK_MEMORY_THRESHOLD_BYTES=str(args.oom_threshold_bytes),
    )
    log_path = out_dir.with_suffix('.driver.log')
    started = time.time()
    with log_path.open('x') as log:
        exit_code = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    driver_output = log_path.read_text(errors='replace')
    blocking_reason = (
        'required_nice_priority_unavailable'
        if 'could not start command with required nice priority' in driver_output
        else None
    )
    reports = sorted((out_dir / 'run').glob('*stack-reports/stack-report-*.json'))
    report = read(reports[-1]) if reports else {}
    res = report.get('resources') or {}
    pseudo_analysis = {
        'status': 'resource_failed' if res.get('memory_threshold_exceeded') else 'rejected',
        'acceptance_state': 'resource_failed' if res.get('memory_threshold_exceeded') else 'rejected',
        'memory': {'threshold_exceeded': bool(res.get('memory_threshold_exceeded'))},
    }
    tiff_published = _has_published_tiff(out_dir)
    mismatches = (
        [{'field': 'runner.blocking_reason', 'expected': None, 'actual': blocking_reason}]
        if blocking_reason
        else evaluate_oom_probe(exit_code, pseudo_analysis, tiff_published)
    )
    return {
        'scenario': 'oom',
        'threshold_bytes': args.oom_threshold_bytes,
        'directory': str(out_dir),
        'driver_exit_code': exit_code,
        'blocking_reason': blocking_reason,
        'wall_seconds': time.time() - started,
        'tiff_published': tiff_published,
        'mismatches': mismatches,
    }


def run_concurrent_load_probe(args, root, directory, baseline):
    # Spawn a bounded background CPU worker while running under the shared `run`
    # slot contract to verify signature invariance under contention.
    burner = subprocess.Popen([
        sys.executable, '-c',
        'import hashlib, time\n'
        'end = time.time() + 3600\n'
        'b = b"x" * 65536\n'
        'while time.time() < end:\n'
        '    hashlib.sha256(b).digest()\n',
    ])
    try:
        out_dir = directory / f'{args.dataset}-concurrent-load'
        run = run_once(args, root, out_dir, min(5, args.threads[-1]), 1, exclusive=False, tag='concurrent')
        run['mismatches'] = compare_signatures(baseline, run['signature']) if baseline is not None else []
        run['signature_sha256'] = signature_sha(run['signature'])
        return run
    finally:
        burner.terminate()
        try:
            burner.wait(timeout=5)
        except subprocess.TimeoutExpired:
            burner.kill()
            burner.wait()


def main(argv=None):
    args = parse(argv)
    root = repo_root()
    _binary, build = verify_binary(args.binary)
    entry = dataset_entry(args.dataset)
    not_run = remaining_not_run(args.scenarios)
    if args.dry_run:
        print(json.dumps({
            'schema': 1,
            'dataset': args.dataset,
            'subset': args.subset,
            'scale': args.scale,
            'threads': args.threads,
            'repeat': args.repeat,
            'scenarios': args.scenarios,
            'exclusive': True,
            'input_order_policy': entry.get('inputOrder', 'sorted'),
            'not_run': not_run,
        }, ensure_ascii=False))
        return 0
    directory = ACCEPTANCE / args.lane / f'determinism-{args.label}'
    directory.mkdir(parents=True, exist_ok=False)
    result = {
        'schema': 1,
        'status': 'running',
        'dataset': args.dataset,
        'subset': args.subset,
        'scale': args.scale,
        'threads': args.threads,
        'repeat': args.repeat,
        'scenarios': args.scenarios,
        'exclusive': True,
        'input_order_policy': entry.get('inputOrder', 'sorted'),
        'binary': args.binary,
        'binary_sha256': build['sha256'],
        'manifest_path': str((root / 'tests/acceptance/focus-stack-datasets.json').resolve()),
        'manifest_sha256': sha(root / 'tests/acceptance/focus-stack-datasets.json'),
        'not_run': not_run,
        'runs': [],
        'scenario_results': {},
        'started_epoch': time.time(),
    }
    write(directory / 'determinism.json', result)
    baseline = None
    if 'threads' in args.scenarios:
        for thread_count in args.threads:
            for repeat in range(1, args.repeat + 1):
                output = directory / f'{args.dataset}-t{thread_count}-r{repeat}'
                run = run_once(args, root, output, thread_count, repeat)
                run['mismatches'] = compare_signatures(baseline, run['signature']) if baseline is not None else []
                if (run['analysis'].get('status') in ('missing_analysis', 'analysis_error', 'not_run')
                        or run['analysis'].get('acceptance_state') == 'not_run'):
                    run['mismatches'].append({
                        'field': 'analysis.status',
                        'expected': 'a terminal run status',
                        'actual': run['analysis'].get('acceptance_state') or run['analysis'].get('status'),
                    })
                if baseline is None:
                    baseline = run['signature']
                run['signature_sha256'] = signature_sha(run['signature'])
                run['analysis'] = {
                    'status': run['analysis'].get('status'),
                    'quality_gate': run['analysis'].get('quality_gate'),
                    'output_sha256': run['analysis'].get('output_sha256'),
                    'checks': run['analysis'].get('checks'),
                }
                result['runs'].append(run)
                write(directory / 'determinism.json', result)

    scenario_probes = []
    if 'order_permutation' in args.scenarios:
        probe = run_order_permutation_probe(args, root, directory, baseline)
        result['scenario_results']['order_permutation'] = probe
        scenario_probes.append(probe)
        write(directory / 'determinism.json', result)
    if 'concurrent_load' in args.scenarios:
        probe = run_concurrent_load_probe(args, root, directory, baseline)
        result['scenario_results']['concurrent_load'] = probe
        scenario_probes.append(probe)
        write(directory / 'determinism.json', result)
    if 'cancellation' in args.scenarios:
        probe = run_cancellation_probe(args, root, directory)
        result['scenario_results']['cancellation'] = probe
        scenario_probes.append(probe)
        write(directory / 'determinism.json', result)
    if 'oom' in args.scenarios:
        probe = run_oom_probe(args, root, directory)
        result['scenario_results']['oom'] = probe
        scenario_probes.append(probe)
        write(directory / 'determinism.json', result)

    result['baseline_signature_sha256'] = signature_sha(baseline or {})
    all_checks = result['runs'] + scenario_probes
    result['status'], blocked_reason = classify_matrix_status(all_checks)
    result['native_runs_started'] = not (result['status'] == 'blocked')
    if blocked_reason:
        result['blocked_reason'] = blocked_reason
    result['ended_epoch'] = time.time()
    write(directory / 'determinism.json', result)
    print(json.dumps({'matrix': str(directory / 'determinism.json'), 'status': result['status']}, ensure_ascii=False), flush=True)
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
