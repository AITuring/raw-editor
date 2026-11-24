#!/usr/bin/env python3
"""Run a bounded exclusive determinism matrix for one frozen material scope.

This tool deliberately measures only repeatability.  It never changes quality
thresholds or converts a rejected run into an accepted one.  Every invocation
is exclusive so a concurrent build/run cannot contaminate the comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ACCEPTANCE, MATRIX, dataset_entry, read, repo_root, sha, valid_label, verify_binary, write

HERE = Path(__file__).resolve().parent
MAX_THREADS = 8  # exclusive slot contract; higher requests would be capped.


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
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    for label in (args.binary, args.lane, args.label):
        valid_label(label)
    if args.repeat < 2:
        parser.error('--repeat must be at least 2')
    if not 0 < args.scale <= 1:
        parser.error('--scale must be in (0, 1]')
    dataset_entry(args.dataset)
    return args


def run_once(args, root, output, thread_count, repeat):
    command = [sys.executable, str(HERE / 'run.py'), '--binary', args.binary,
               '--lane', args.lane, '--dataset', args.dataset, '--scale', str(args.scale),
               '--label', f'{args.label}-t{thread_count}-r{repeat}', '--out', str(output), '--exclusive']
    if args.subset:
        command += ['--subset', args.subset]
    env = dict(os.environ, RAYON_NUM_THREADS=str(thread_count))
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
        'thread_count': thread_count,
        'repeat': repeat,
        'directory': str(output),
        'driver_exit_code': exit_code,
        'blocking_reason': blocking_reason,
        'wall_seconds': time.time() - started,
        'analysis': analysis,
        'signature': stable_signature(analysis),
    }


def main(argv=None):
    args = parse(argv)
    root = repo_root()
    _binary, build = verify_binary(args.binary)
    entry = dataset_entry(args.dataset)
    if args.dry_run:
        print(json.dumps({
            'schema': 1,
            'dataset': args.dataset,
            'subset': args.subset,
            'scale': args.scale,
            'threads': args.threads,
            'repeat': args.repeat,
            'exclusive': True,
            'input_order_policy': entry.get('inputOrder', 'sorted'),
            'not_run': ['order_permutation', 'concurrent_load', 'cancellation', 'oom', 'provider_fallback'],
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
        'exclusive': True,
        'input_order_policy': entry.get('inputOrder', 'sorted'),
        'binary': args.binary,
        'binary_sha256': build['sha256'],
        'manifest_path': str((root / 'tests/acceptance/focus-stack-datasets.json').resolve()),
        'manifest_sha256': sha(root / 'tests/acceptance/focus-stack-datasets.json'),
        'not_run': ['order_permutation', 'concurrent_load', 'cancellation', 'oom', 'provider_fallback'],
        'runs': [],
        'started_epoch': time.time(),
    }
    write(directory / 'determinism.json', result)
    baseline = None
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
    result['baseline_signature_sha256'] = signature_sha(baseline or {})
    result['status'], blocked_reason = classify_matrix_status(result['runs'])
    result['native_runs_started'] = not (result['status'] == 'blocked')
    if blocked_reason:
        result['blocked_reason'] = blocked_reason
    result['ended_epoch'] = time.time()
    write(directory / 'determinism.json', result)
    print(json.dumps({'matrix': str(directory / 'determinism.json'), 'status': result['status']}), flush=True)
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
