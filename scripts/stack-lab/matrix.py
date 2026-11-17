#!/usr/bin/env python3
"""Evaluate a frozen binary against a versioned regression matrix; retry mismatches exclusively."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from monitor import Interrupted, interrupt
from common import ACCEPTANCE, MAIN, MATRIX, SHARED_TARGET, read, repo_root, sha, valid_label, verify_binary, write

HERE = Path(__file__).resolve().parent


def value_at(actual, path):
    for key in path.split('.'):
        if not isinstance(actual, dict) or key not in actual:
            return None
        actual = actual[key]
    return actual


def compare(actual, expected):
    failures = []
    for path, expected_value in expected.items():
        observed = value_at(actual, path)
        if observed != expected_value or type(observed) is not type(expected_value):
            failures.append({'field': path, 'expected': expected_value, 'actual': observed})
    # Source integrity and resources are never waived for a known QG failure.
    for key in ('inputs_unchanged', 'reference_unchanged', 'binary_unchanged', 'report_source_digests', 'memory_gate_pass'):
        if (actual.get('checks') or {}).get(key) is not True:
            failures.append({'field': 'checks.' + key, 'expected': True, 'actual': (actual.get('checks') or {}).get(key)})
    for key in ('source_cache', 'native_cache'):
        cache = (actual.get('performance') or {}).get(key) or {}
        peak, maximum = cache.get('peak_bytes'), cache.get('max_bytes')
        if not isinstance(peak, int) or not isinstance(maximum, int) or peak < 0 or peak > maximum:
            failures.append({'field': 'performance.' + key, 'expected': '0 <= peak_bytes <= max_bytes', 'actual': cache})
    return failures


def validate(config):
    if config.get('version') != 1 or not config.get('cases'):
        raise ValueError('matrix version 1 and nonempty cases are required')
    seen = set()
    for case in config['cases']:
        valid_label(case['id'])
        if case['id'] in seen:
            raise ValueError('duplicate matrix case')
        seen.add(case['id'])
        if not case.get('tiers') or not set(case['tiers']) <= {'medium', 'extended', 'full'}:
            raise ValueError('invalid case tiers')
        if not isinstance(case.get('expected'), dict) or not case['expected']:
            raise ValueError('case expectations cannot be empty')
        if not {'status', 'quality_gate.verdict', 'closure', 'output_sha256'} <= case['expected'].keys():
            raise ValueError('each case must bind status, QG, closure and TIFF SHA expectations')
        if case['expected']['status'] == 'rejected' and not case.get('known_failure'):
            raise ValueError('rejected expectation must be labelled known_failure')
        if case.get('dataset') == 'mayuan-psd':
            raise ValueError('mayuan-psd is deferred and must not enter this matrix')
    return config


def launch(command, *, log, cwd=None, env=None):
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=cwd, env=env,
                               start_new_session=True)
    try:
        return process.wait()
    except Interrupted as exc:
        process.send_signal(exc.signum)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def invoke(case, args, directory, exclusive):
    command = [sys.executable, str(HERE / 'run.py'), '--binary', args.binary, '--lane', args.lane,
               '--dataset', case['dataset'], '--scale', str(case.get('scale', 1)), '--label',
               args.label + '-' + case['id'], '--out', str(directory)]
    if case.get('subset'):
        command += ['--subset', case['subset']]
    if case.get('framed'):
        command += ['--framed']
    if exclusive:
        command += ['--exclusive']
    with directory.with_suffix('.driver.log').open('x') as log:
        exit_code = launch(command, log=log)
    output = directory / 'lanting' if case.get('framed') else directory
    analysis = read(output / 'analysis.json') if (output / 'analysis.json').is_file() else {'status': 'missing_analysis'}
    return {'directory': str(output), 'driver_exit_code': exit_code, 'analysis': analysis}


def quick_commands(root, directory):
    directory, _build = verify_binary(directory)
    return [
        ['cargo', 'fmt', '--manifest-path', str(root / 'src-tauri/Cargo.toml'), '--check'],
        ['cargo', 'check', '--manifest-path', str(root / 'src-tauri/Cargo.toml'), '--offline'],
        [str(directory / 'native-process-wrapper.py'), '--test-threads=3'],
        *[['node', str(root / 'tests' / name)] for name in ('focus-stack-quality-contract.mjs', 'focus-stack-datasets-contract.mjs')],
        ['node', '--test', str(root / 'tests/focus-stack-regression-verdict.mjs')],
        ['node', '--test', str(root / 'tests/focus-stack-lanting-guard.test.mjs')],
    ]


def quick(args, directory):
    root = repo_root()
    env = {k: v for k, v in os.environ.items() if not k.startswith('RAW_EDITOR_')}
    env['CARGO_TARGET_DIR'] = str(SHARED_TARGET)
    env['RAYON_NUM_THREADS'] = '3'
    # One build slot per command, never hold a slot while asking a child to acquire it.
    results = []
    for i, command in enumerate(quick_commands(root, args.binary)):
        wrapped = [sys.executable, str(HERE / 'slot.py'), 'build', '--lane', args.lane, '--what',
                   f'quick {i + 1}/{7}', '--', *command]
        started = time.time()
        with (directory / f'quick-{i + 1}.log').open('x') as log:
            exit_code = launch(wrapped, cwd=root, env=env, log=log)
        results.append({'command': command, 'exit_code': exit_code, 'wall_seconds': time.time() - started})
        print(f'quick {i + 1}/7: exit {exit_code}', flush=True)
        write(directory / 'quick.json', {'commands': results, 'passed': all(x['exit_code'] == 0 for x in results)})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tier', required=True, choices=['quick', 'medium', 'extended', 'full'])
    parser.add_argument('--binary', required=True)
    parser.add_argument('--matrix', type=Path, default=MATRIX)
    parser.add_argument('--lane', default='task-2')
    parser.add_argument('--label', required=True)
    parser.add_argument('--case', action='append', default=[], help='restrict to named cases within the tier')
    parser.add_argument('--dry-run', action='store_true', help='validate configuration and print selected cases without running')
    args = parser.parse_args(argv)
    directory = None
    result = None
    previous = {}
    try:
        for label in (args.binary, args.lane, args.label):
            valid_label(label)
        config = validate(read(args.matrix))
        _binary, build = verify_binary(args.binary)
        if args.binary == config['baseline']['label'] and build['sha256'] != config['baseline']['binary_sha256']:
            raise ValueError('baseline label is not bound to the expected binary SHA')
        cases = [c for c in config['cases'] if args.tier in c['tiers'] and (not args.case or c['id'] in args.case)]
        if args.tier != 'quick' and (not cases or set(args.case) - {c['id'] for c in cases}):
            raise ValueError('no cases selected, or requested cases are outside this tier')
        if args.dry_run:
            print(json.dumps({'tier': args.tier, 'cases': [c['id'] for c in cases], 'quick_commands': 7 if args.tier == 'quick' else 0}))
            return 0
        directory = ACCEPTANCE / args.lane / f'matrix-{args.label}-{args.tier}'
        directory.mkdir(parents=True, exist_ok=False)
        result = {'schema': 1, 'tier': args.tier, 'binary': args.binary, 'binary_sha256': build['sha256'],
                  'matrix_path': str(args.matrix.resolve()), 'matrix_sha256': sha(args.matrix),
                  'matrix_snapshot': config, 'started_epoch': time.time(), 'cases': [], 'status': 'running'}
        write(directory / 'matrix.json', result)
        previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        if args.tier == 'quick':
            results = quick(args, directory)
            result['commands'] = results
            result['status'] = 'pass' if all(x['exit_code'] == 0 for x in results) else 'fail'
        else:
            for case in cases:
                result['active_case'] = case['id']
                write(directory / 'matrix.json', result)
                item = {'id': case['id'], 'expected': case['expected'], 'attempts': []}
                for attempt in (1, 2):
                    exclusive = attempt == 2 or case.get('exclusive', False)
                    record = invoke(case, args, directory / f'{case["id"]}-attempt-{attempt}', exclusive)
                    record.update(exclusive_requested=exclusive, mismatches=compare(record['analysis'], case['expected']))
                    expected_driver = 0 if record['analysis'].get('status') in ('quality_accepted', 'rendered_without_quality_acceptance') else 1
                    if record['driver_exit_code'] != expected_driver:
                        record['mismatches'].append({'field': 'driver_exit_code', 'expected': expected_driver, 'actual': record['driver_exit_code']})
                    item['attempts'].append(record)
                    print(f'{case["id"]} attempt {attempt} exclusive={exclusive}: '
                          f'{len(record["mismatches"])} mismatches', flush=True)
                    if not record['mismatches']:
                        break
                item['status'] = ('pass_after_exclusive_retry' if len(item['attempts']) == 2 else 'pass') if not item['attempts'][-1]['mismatches'] else 'fail'
                item['quality_accepted'] = item['attempts'][-1]['analysis'].get('status') == 'quality_accepted'
                result['cases'].append(item)
                write(directory / 'matrix.json', result)
            result['status'] = 'pass' if all(c['status'] != 'fail' for c in result['cases']) else 'fail'
        result.pop('active_case', None)
        result['ended_epoch'] = time.time()
        write(directory / 'matrix.json', result)
        print(json.dumps({'matrix': str(directory / 'matrix.json'), 'status': result['status']}), flush=True)
        return 0 if result['status'] == 'pass' else 1
    except Interrupted as exc:
        if directory and result is not None:
            result.update(status='cancelled', signal=exc.signum, ended_epoch=time.time())
            write(directory / 'matrix.json', result)
        return 128 + exc.signum
    except (OSError, ValueError, KeyError, SystemExit) as exc:
        print(f'[matrix] {exc}', file=sys.stderr)
        if directory and result is not None:
            result.update(status='error', error=str(exc), ended_epoch=time.time())
            write(directory / 'matrix.json', result)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    sys.exit(main())
