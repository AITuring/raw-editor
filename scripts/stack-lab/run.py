#!/usr/bin/env python3
"""One immutable real-material run, under an automatically selected resource slot."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import (ACCEPTANCE, DATA, MAIN, MEMORY_GATE_BYTES, SLOTS, dataset_entry, dataset_files,
                    read, repo_root, sha, slot_holders, usage_from_logs, valid_label, verify_binary, write)
from monitor import execute

HERE = Path(__file__).resolve().parent
# GiB estimates apply only to these historically measured scopes, not entire folders.
ESTIMATES = {'sleeping-9-6479': 13, 'langyuan-10': 11, 'wenyuan-10': 12, 'lanting': 12,
             'qixu:0.1': 16, 'fengying:0.1': 16}


def sha_of(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--lane', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--subset')
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--label', required=True)
    parser.add_argument('--framed', action='store_true')
    parser.add_argument('--streaming-diagnostics', action='store_true',
                        help='export test-only streaming owner/coverage maps under the run directory')
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--exclusive', action='store_true')
    parser.add_argument('--estimate-gib', type=float)
    parser.add_argument('--env', action='append', default=[], metavar='RAW_EDITOR_KEY=VALUE')
    parser.add_argument('--out', help='fresh directory under output/acceptance/<lane>')
    args = parser.parse_args(argv)
    for label in (args.binary, args.lane, args.label):
        valid_label(label)
    if not 0 < args.scale <= 1 or not math.isfinite(args.scale):
        parser.error('--scale must be finite and in (0, 1]')
    if args.estimate_gib is not None and (not math.isfinite(args.estimate_gib) or args.estimate_gib <= 0):
        parser.error('--estimate-gib must be finite and positive')
    if args.framed and (args.dataset != 'lanting' or args.subset or args.scale != 1 or args.preview_only):
        parser.error('--framed is the Lanting45 scale=1 native protection scope')
    dataset_entry(args.dataset)
    return args


def historical_estimate(args):
    known = ESTIMATES.get(args.subset or f'{args.dataset}:{args.scale:g}')
    if args.dataset == 'lanting':
        known = ESTIMATES['lanting']
    return known


def estimate(args):
    return args.estimate_gib if args.estimate_gib is not None else historical_estimate(args)


def contention(slot, end):
    if slot.get('slot') == 'exclusive':
        return []
    start = slot['started_epoch']
    found = set(slot.get('contended_with', []))
    history = SLOTS / 'history.jsonl'
    records = []
    if history.exists():
        with history.open() as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            for line in stream:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
    records += list(slot_holders().values())
    for item in records:
        if item.get('pid') == slot.get('pid') or item.get('slot') not in ('build', 'exclusive'):
            continue
        if item.get('state') != 'running':
            continue
        ended = item.get('ended_epoch', end)
        if item.get('started_epoch', end) < end and ended > start:
            found.add(item['lane'] + ': ' + item['what'])
    return sorted(found)


def performance(report):
    perf = report.get('performance') or {}
    return {'raw_decode_count': (perf.get('source_preflight') or {}).get('raw_decode_count'),
            'native_fusion_count': (perf.get('native_fusion_cache') or {}).get('fusion_count'),
            **{key: {k: (perf.get(field) or {}).get(k) for k in ('max_bytes', 'peak_bytes', 'hits')}
               for key, field in [('source_cache', 'source_render_cache'), ('native_cache', 'native_fusion_cache')]}}


def collect(dest):
    reports = [(p, read(p)) for p in sorted((dest / 'run').glob('*stack-reports/stack-report-*.json'))]
    measured = [(p, r) for p, r in reports if (r.get('quality_gate') or {}).get('criteria')]
    if len(measured) > 1:
        raise ValueError('multiple measured Stack_Reports; refusing an ambiguous result')
    report_path, report = measured[0] if measured else (reports[-1] if reports else (None, {}))
    summaries = list((dest / 'run').glob('*.summary.json'))
    if len(summaries) > 1:
        raise ValueError('multiple runner summaries')
    return report_path, report, read(summaries[0]) if summaries else None


def analyse(dest, execution, build, directory, entry):
    report_path, report, summary = collect(dest)
    qg = report.get('quality_gate') or {}
    native = qg.get('legacy_native_evidence') or {}
    output = report.get('output') or {}
    tiff = Path(output['path']) if output.get('path') else None
    output_exists = bool(tiff and tiff.is_file() and tiff.stat().st_size > 0)
    if tiff and not tiff.resolve().is_relative_to(dest):
        raise ValueError('reported output escapes this run directory')
    usage = usage_from_logs(dest / 'run')
    resources = report.get('resources') or {}
    measurements = {'process_tree_sampled': execution.get('sampled_peak_rss_bytes'),
                    'native_getrusage': (usage or {}).get('max_rss_bytes'),
                    'native_report': resources.get('peak_rss_bytes')}
    valid_rss = lambda v: type(v) is int and v > 0
    peaks = [v for v in measurements.values() if valid_rss(v)]
    peak = max(peaks) if peaks else None
    memory_ok = bool(peak is not None and peak <= MEMORY_GATE_BYTES and not execution.get('monitor_error')
                     and not execution.get('memory_threshold_exceeded') and not resources.get('memory_threshold_exceeded')
                     and execution.get('rss_sample_count', 0) > 0 and valid_rss((usage or {}).get('max_rss_bytes')))
    cancelled = execution.get('cancelled_signal')
    inputs_unchanged = (all(Path(x['path']).is_file() and sha(x['path']) == x['sha256'] for x in execution['inputs'])
                        if not cancelled else None)
    reference = execution.get('reference')
    reference_ok = (Path(reference['path']).is_file() and sha(reference['path']) == reference['sha256']) if reference else True
    try:
        verify_binary(execution['binary'])
        binary_ok = True
    except (OSError, ValueError, SystemExit):
        binary_ok = False
    criteria = {c['name']: c.get('status', c.get('verdict')) for c in qg.get('criteria', [])}
    summary_data = (summary or {}).get('summary') or {}
    checks = {
        'inputs_unchanged': inputs_unchanged,
        'reference_unchanged': reference_ok,
        'binary_unchanged': binary_ok,
        'memory_gate_pass': memory_ok,
        'runner_exit_zero': execution['exit_code'] == 0,
        'native_child_exit_zero': (usage or {}).get('child_exit_code') == 0,
        'runner_summary_present': summary is not None,
        'runner_failures_empty': summary is not None and summary.get('failures') == [],
        'quality_gate_pass': qg.get('verdict') == 'pass' and qg.get('roi_count', 0) >= 8,
        'output_contract': bool(output_exists and output.get('written') and output.get('artifact_validation') == 'pass'
                                and output.get('bit_depth') == 16 and output.get('alpha_preserved') is True),
    }
    if execution['preview_only']:
        checks['output_contract'] = output_exists  # preview never qualifies as native acceptance
    report_sources = (report.get('input') or {}).get('sources') or []
    by_path = {item['path']: item['sha256'] for item in execution['inputs']}
    expected_paths = {p for p in by_path if Path(p).name not in (entry.get('expectedSkipped', []) if not execution.get('subset') else [])}
    report_paths = [item.get('path') for item in report_sources]
    checks['report_source_digests'] = (bool(report_sources) and len(set(report_paths)) == len(report_paths)
        and set(report_paths) == expected_paths and all(
        item.get('path') in by_path and item.get('sha256_before') == by_path[item['path']]
        and item.get('sha256_after') == by_path[item['path']] for item in report_sources))
    passed = all(v is True for v in checks.values()) and not execution['preview_only']
    preview_success = execution['preview_only'] and all(checks[k] is True for k in
        ('inputs_unchanged', 'reference_unchanged', 'binary_unchanged', 'report_source_digests', 'memory_gate_pass',
         'runner_exit_zero', 'native_child_exit_zero', 'runner_summary_present', 'runner_failures_empty'))
    status = ('cancelled' if cancelled else 'resource_failed' if not memory_ok else
              'quality_accepted' if passed else 'rendered_without_quality_acceptance' if preview_success else 'rejected')
    comp = report.get('composition') or {}
    input_report = report.get('input') or {}
    contention_with = contention(execution['slot'], time.time())
    return dict(status=status, checks=checks, exit_code=execution['exit_code'], runner_failures=(summary or {}).get('failures'),
                runner_verdict=(summary or {}).get('verdict'), binary_sha256=build['sha256'], commit=build['commit'],
                selected_path=report.get('selected_path'), station_count=(report.get('grouping') or {}).get('station_count'),
                quality_gate={'verdict': qg.get('verdict'), 'roi_count': qg.get('roi_count'), 'criteria': criteria,
                              'non_pass': [f'{n}:{v}' for n, v in criteria.items() if v != 'pass'],
                              'unresolved_pixels': native.get('unresolved_pixel_count'),
                              'detail_repairs': len(native.get('detail_repairs') or [])},
                closure=(report.get('closure') or {}).get('status'), outcome=(report.get('degradation') or {}).get('result'),
                sources={'inputs': len(execution['inputs']), 'report_sources': input_report.get('source_count'),
                         'ordered': summary_data.get('orderedInputs'), 'streamed': summary_data.get('streamedSources'),
                         'lending': summary_data.get('lendingSources'), 'expected_skipped': entry.get('expectedSkipped', []),
                         'covered_pixels': comp.get('source_covered_pixels'), 'owned_pixels': comp.get('source_owned_pixels')},
                output=output, output_sha256=sha(tiff) if output_exists else None,
                report=str(report_path) if report_path else None, report_sha256=sha(report_path) if report_path else None,
                performance=performance(report), memory={'threshold_bytes': MEMORY_GATE_BYTES, 'measurements': measurements,
                    'sample_count': execution.get('rss_sample_count'), 'monitor_error': execution.get('monitor_error'),
                    'threshold_exceeded': bool(execution.get('memory_threshold_exceeded') or resources.get('memory_threshold_exceeded')
                                               or (peak is not None and peak > MEMORY_GATE_BYTES))},
                peak_rss_bytes=peak, memory_gate_pass=memory_ok, wall_seconds=execution['wall_seconds'], usage=usage,
                contended=bool(contention_with), contended_with=contention_with, caps=execution['slot']['caps'])


def destination(args):
    scope = args.subset or args.dataset
    dest = (Path(args.out) if args.out else ACCEPTANCE / args.lane / f'{scope}-{args.label}').resolve()
    if not dest.is_relative_to((ACCEPTANCE / args.lane).resolve()) or dest == (ACCEPTANCE / args.lane).resolve():
        raise ValueError('output must be a fresh directory below output/acceptance/<lane>')
    if dest.is_relative_to(DATA.resolve()):
        raise ValueError('output may not be written to the source directory')
    return dest / 'lanting' if args.framed else dest


def check_disk(lane):
    if shutil.disk_usage(MAIN).free < 40 * 1024 ** 3:
        raise ValueError('less than 40 GiB free; pause new runs and resolve disk space')
    folder = ACCEPTANCE / lane
    used = sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink()) if folder.exists() else 0
    if used >= 20 * 1024 ** 3:
        raise ValueError('lane output reached 20 GiB; pause new runs')


def run(args):
    directory, build = verify_binary(args.binary)
    entry, files = dataset_files(args.dataset, args.subset)
    dest = destination(args)
    check_disk(args.lane)
    dest.mkdir(parents=True, exist_ok=False)
    inputs = [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in files]
    reference_path = DATA / (entry.get('directory') or entry['directories'][0]) / entry['referenceOutput'] if entry.get('referenceOutput') else None
    if reference_path and not reference_path.is_file():
        raise ValueError(f'reference missing: {reference_path}')
    reference = {'path': str(reference_path), 'sha256': sha(reference_path)} if reference_path else None
    root = repo_root()
    env = {k: v for k, v in os.environ.items() if not k.startswith('RAW_EDITOR_')}
    env.update(RAW_EDITOR_FOCUS_STACK_DATA_ROOT=str(DATA), RAW_EDITOR_STACK_ACCEPTANCE_COVERAGE_ALPHA='1',
               RAW_EDITOR_STACK_ACCEPTANCE_ALIGNMENT='auto', RAW_EDITOR_STACK_ACCEPTANCE_BLEND='focus-stack',
               RAW_EDITOR_STACK_ALIGNMENT_CACHE=str(dest / 'source-alignment.json'))
    if args.framed:
        env.update(RAW_EDITOR_ARTWORK_FRAMED_OUTPUT='1', RAW_EDITOR_STREAMING_MOSAIC_DIAGNOSTICS_DIR=str(dest / 'diagnostics'))
    elif args.streaming_diagnostics:
        env['RAW_EDITOR_STREAMING_MOSAIC_DIAGNOSTICS_DIR'] = str(dest / 'diagnostics')
    for item in args.env:
        key, value = item.split('=', 1)
        if not key.startswith('RAW_EDITOR_') or key in env or any(token in key for token in ('OUTPUT', 'REPORT', 'MEMORY', 'THRESHOLD', 'QUALITY_GATE')):
            raise ValueError(f'--env cannot override protected run contracts: {key}')
        env[key] = value
    command = ['node', str(root / 'tests/focus-stack-regression.mjs'), '--dataset', args.dataset, '--scale', f'{args.scale:g}',
               '--write-output', '--label', f'{args.lane}-{args.label}', '--out-dir', str(dest / 'run'),
               '--binary', str(directory / 'native-process-wrapper.py')]
    if args.subset:
        command += ['--subset', args.subset]
    if args.preview_only:
        command += ['--preview-only']
    slot = read(Path(os.environ['STACK_LAB_HOLDER']))
    execution = dict(schema=1, dataset=args.dataset, subset=args.subset, scale=args.scale, framed=args.framed,
                     streaming_diagnostics=args.streaming_diagnostics,
                     resource_contract=build.get('resource_contract'), preview_only=args.preview_only, lane=args.lane, label=args.label, binary=args.binary,
                     binary_sha256=build['sha256'], commit=build['commit'], inputs=inputs, inputs_sha256=sha_of(inputs),
                     reference=reference, command=command, runner_root=str(root), runner_sha256=sha(root / 'tests/focus-stack-regression.mjs'),
                     manifest_sha256=sha(root / 'tests/acceptance/focus-stack-datasets.json'), slot=slot,
                     environment={k: v for k, v in env.items() if k.startswith(('RAW_EDITOR_', 'RAYON_'))}, state='running',
                     started_epoch=time.time(), tool_sha256={name: sha(HERE / name) for name in
                         ('run.py', 'monitor.py', 'common.py', 'lanting_audit.py')})
    write(dest / 'execution.json', execution)
    execution.update(execute(command, cwd=root, env=env, log_path=dest / 'run.log'))
    execution['state'] = 'finished'
    write(dest / 'execution.json', execution)
    try:
        analysis = analyse(dest, execution, build, directory, entry)
        write(dest / 'analysis.json', analysis)
        if args.framed and analysis['checks']['output_contract'] and not execution.get('cancelled_signal'):
            from lanting_audit import audit, guard
            analysis['paper_audit'] = audit(dest)
            analysis['guard'] = {'strict': guard(dest), 'regression_only': guard(dest, True)}
            if not all(v.get('pass') is True for v in analysis['guard'].values()):
                analysis['status'] = 'rejected'
                analysis['checks']['lanting_guard'] = False
            else:
                analysis['checks']['lanting_guard'] = True
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        analysis = dict(status='analysis_error', error=str(exc), exit_code=execution['exit_code'], memory_gate_pass=False)
    write(dest / 'analysis.json', analysis)
    print(json.dumps({'directory': str(dest), 'status': analysis['status'], 'qg': (analysis.get('quality_gate') or {}).get('verdict'),
                      'peak_rss_bytes': analysis.get('peak_rss_bytes'), 'sha256': analysis.get('output_sha256')}, ensure_ascii=False), flush=True)
    return (128 + execution['cancelled_signal']) if execution.get('cancelled_signal') else (
        0 if analysis['status'] in ('quality_accepted', 'rendered_without_quality_acceptance') else 1)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        args = parse(argv)
        destination(args)
        _, build = verify_binary(args.binary)
        verified_budget = (build.get('resource_contract') or {}).get('version') == 1 and (build.get('resource_contract') or {}).get('status') == 'verified'
        current = os.environ.get('STACK_LAB_HOLDER')
        if current:
            slot = read(Path(current))
            if slot['pid'] != os.getppid() or slot.get('slot') not in ('run', 'exclusive'):
                raise ValueError('run must be the direct child of its run/exclusive slot supervisor')
            if not verified_budget and slot['slot'] != 'exclusive':
                raise ValueError('legacy/unverified native thread budget requires exclusive slot')
            return run(args)
        peak = estimate(args)
        mode = 'exclusive' if not verified_budget or args.exclusive or historical_estimate(args) is None or peak is None or peak > 20 else 'run'
        command = [sys.executable, str(HERE / 'slot.py'), mode, '--lane', args.lane, '--what',
                   f'{args.subset or args.dataset} scale={args.scale:g} {args.label}']
        if peak is not None:
            command += ['--estimate-gib', str(peak)]
        os.execv(sys.executable, [*command, '--', sys.executable, str(Path(__file__).resolve()), *argv])
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        print(f'[run] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
