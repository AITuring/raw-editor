#!/usr/bin/env python3
"""Offline Quality_Gate what-if replay over saved Stack_Reports (tasks.md 2B.5).

Read-only. Recomputes, from the fields a Stack_Report already records, the
per-criterion verdicts, the overall Quality_Gate verdict, the output-density
closure decision and the layered native acceptance check, first under the
current rules and then under each candidate contract change (C1..C7 in
tasks.md batch 2B). The current-rule replay must reproduce every recorded
verdict; a mismatch means this tool is wrong and the run reports it.

Rules mirrored (keep in sync when the Rust changes):
- criterion_verdict / overall_verdict:
  src-tauri/src/panorama_utils/stack_pipeline/quality_gate_runner/criteria.rs
- validate_output_closure and the closure limits:
  src-tauri/src/panorama_stitching/output_density.rs, panorama_stitching.rs
- verify_layered_native_acceptance:
  src-tauri/src/panorama_stitching/acceptance_tests/focus_stack_fixture.rs

Only C1 and C3 can be decided from saved fields. C2, C4, C5, C6 and C7 need
new measurements; they are listed with the evidence that a rerun would need.
No production code, threshold, expectation or report is modified.
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

QUALITY_MIN_MEASURABLE = 8
QUALITY_MAX_UNMEASURABLE_RATIO = 0.20
QUALITY_BOUNDARY_MAX = 3.0
CLOSURE_DIRECT_PAIR_P95_LIMIT_PX = 3.0
CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX = 2.0
CLOSURE_MAX_CORNER_CORRECTION_PX = 256.0
TONE_BOUNDARY_DELTA_E_THRESHOLD = 1.5
QUALITY_CRITERIA = (
    'local_scale_median', 'local_scale_pixel_ratio', 'effective_pixel_count',
    'mtf50_normalized', 'gradient_energy_normalized', 'noise_sigma_ratio',
    'roi_delta_e00', 'boundary_stroke_alignment', 'owner_sharpness_coverage',
)
SPARSE = {'mtf50_normalized', 'noise_sigma_ratio', 'boundary_stroke_alignment'}
WHOLE_IMAGE = {'effective_pixel_count', 'owner_sharpness_coverage'}
DENSE = {'local_scale_median', 'local_scale_pixel_ratio', 'gradient_energy_normalized', 'roi_delta_e00'}

# Candidate contract changes. `computable` ones are decided from saved fields.
OPTIONS = {
    'C1': {'computable': True,
           'summary': 'closure limits in base world-native pixels, not multiplied by the density factor'},
    'C2': {'computable': False,
           'summary': 'mixed-magnification local-scale reference',
           'needs': 'per-ROI source magnification and a chosen reference scale; rerun with a new local-scale contract'},
    'C3': {'computable': True,
           'summary': 'boundary with <8 measurable samples: fail only on a sample above 3.0 px, no P95'},
    'C4': {'computable': False,
           'summary': 'tone boundary measured on content-matched paper pixels only',
           'needs': 'per-boundary paper/content masks (2B.9 measurement)'},
    'C5': {'computable': False,
           'summary': 'owner reverse-lookup stencil 3x3 -> 5x5',
           'needs': 'a rerun; task-16 stencil5 A/B measured mayuan-2 unresolved 163 -> 0'},
    'C6': {'computable': False,
           'summary': 'effective_pixel_count for sources without native sampling metadata',
           'needs': 'per-source native_registration inventory (2B.7)'},
    'C7': {'computable': False,
           'summary': 'residual-aware density planning replaces the nonidentity_world_residual bypass',
           'needs': 'composed residual Jacobian samples after the 2B.8 fix'},
}


def criterion_verdict(name, measurable, technical, has_failed, diagnostic, statistics_base_empty):
    """Mirror of criteria.rs criterion_verdict."""
    if diagnostic:
        return 'not_applicable'
    total = measurable + technical
    technical_ratio = 0.0 if total == 0 else technical / total
    whole_image = name in WHOLE_IMAGE
    if (whole_image and statistics_base_empty) or (
            not whole_image and technical_ratio > QUALITY_MAX_UNMEASURABLE_RATIO):
        return 'insufficient_evidence'
    if has_failed:
        return 'fail'
    if name in SPARSE and measurable < QUALITY_MIN_MEASURABLE:
        return 'not_applicable'
    if name in DENSE and measurable < QUALITY_MIN_MEASURABLE:
        return 'insufficient_evidence'
    return 'pass'


def overall_verdict(verdicts):
    """Mirror of criteria.rs overall_verdict over criterion verdict strings."""
    if any(v == 'fail' for v in verdicts):
        return 'fail'
    if not verdicts or any(v not in ('pass', 'not_applicable') for v in verdicts):
        return 'insufficient_evidence'
    return 'pass'


def recompute_criterion(record, c3=False):
    name = record['name']
    measurable = int(record.get('measurable_count') or 0)
    technical = int(record.get('technical_unmeasurable_count') or 0)
    failed = record.get('failed') or []
    has_failed = len(failed) > 0
    base_empty = measurable == 0 or (
        name == 'owner_sharpness_coverage' and int(record.get('unresolved_pixel_count') or 0) > 0)
    if c3 and name == 'boundary_stroke_alignment' and measurable < QUALITY_MIN_MEASURABLE:
        # Every recorded value is either a sample or a path P95, and a P95
        # never exceeds its path's largest sample, so "any recorded value
        # above 3 px" is exactly "a sample above 3 px".
        values = [item.get('measured') for item in (record.get('measured') or []) + failed]
        has_failed = any(isinstance(v, (int, float)) and v > QUALITY_BOUNDARY_MAX for v in values)
    return criterion_verdict(name, measurable, technical, has_failed,
                             bool(record.get('diagnostic')), base_empty)


def density_decision(report, c1=False):
    """Recompute the output-density closure outcome.

    Returns (status, reason). Only a recorded `rejected` density with the
    closure reason, or an `applied` one, can be re-decided; bypasses depend
    on evidence the report does not hold and are carried over unchanged.
    """
    density = (report.get('composition') or {}).get('output_density') or {}
    status = density.get('status')
    reason = density.get('bypass_reason')
    if status not in ('applied', 'rejected', 'unchanged'):
        return status, reason
    if status == 'rejected' and reason != 'output_pixel_closure_limit_exceeded':
        return status, reason
    closure = report.get('closure') or {}
    factor = float(density.get('factor') or 1.0)
    scale = 1.0 if c1 else factor
    try:
        direct = float(closure['max_direct_pair_p95_px']) * scale
        median = float(closure['residual_median_px']) * scale
        corner = float(closure['max_corner_correction_px']) * scale
    except (KeyError, TypeError, ValueError):
        return status, reason
    ok = (math.isfinite(direct) and direct <= CLOSURE_DIRECT_PAIR_P95_LIMIT_PX
          and math.isfinite(median) and median <= CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX
          and math.isfinite(corner) and corner <= CLOSURE_MAX_CORNER_CORRECTION_PX)
    if not ok:
        return 'rejected', 'output_pixel_closure_limit_exceeded'
    return ('applied' if factor > 1.0 else 'unchanged'), None


def native_acceptance(report, verdicts, criteria):
    """Mirror of verify_layered_native_acceptance. Returns (outcome, reason)."""
    if report.get('selected_path') != 'layered_virtual_tile':
        return 'not_layered', 'legacy path: the layered native check does not apply'
    components = ((report.get('station_relations') or {}).get('connectivity') or {}).get('components')
    if components != 1 or (report.get('closure') or {}).get('status') != 'converged':
        return 'rejected', 'native station geometry is not connected and converged'
    if overall_verdict(list(verdicts.values())) != 'pass' or len(verdicts) != len(QUALITY_CRITERIA):
        return 'rejected', 'native output did not pass the complete quality gate'
    for name in QUALITY_CRITERIA:
        if name not in verdicts:
            return 'rejected', f'missing or duplicate quality criterion: {name}'
        verdict = verdicts[name]
        technical = int(criteria[name].get('technical_unmeasurable_count') or 0)
        content_only = name in SPARSE and verdict == 'not_applicable' and technical == 0
        if verdict != 'pass' and not content_only:
            return 'rejected', f'native quality criterion failed: {name}'
    tone = report.get('tone') or {}
    maximum = (tone.get('boundary_delta_e') or {}).get('max')
    if tone.get('status') != 'applied' or not isinstance(maximum, (int, float)) or \
            not math.isfinite(maximum) or maximum > TONE_BOUNDARY_DELTA_E_THRESHOLD:
        return 'rejected', f'native tone boundary failed: {maximum}'
    return 'accepted', None


def evaluate(report, c1=False, c3=False):
    records = (report.get('quality_gate') or {}).get('criteria') or []
    criteria = {r['name']: r for r in records}
    density_status, density_reason = density_decision(report, c1=c1)
    recorded_density = ((report.get('composition') or {}).get('output_density') or {}).get('status')
    qg_ran = bool(records)
    if not qg_ran and recorded_density == 'rejected' and density_status != 'rejected':
        # The run stopped at density; nothing downstream was measured.
        return {'density': density_status, 'density_reason': density_reason,
                'quality_gate': 'needs_rerun', 'criteria': {},
                'native': 'needs_rerun', 'native_reason': 'density would pass; QG was never run'}
    verdicts = {name: recompute_criterion(r, c3=c3) for name, r in criteria.items()}
    qg = overall_verdict(list(verdicts.values())) if qg_ran else 'not_run'
    native, native_reason = native_acceptance(report, verdicts, criteria) if qg_ran else (
        'rejected' if density_status == 'rejected' else 'not_run',
        f'quality gate not run (density {density_status})')
    return {'density': density_status, 'density_reason': density_reason, 'quality_gate': qg,
            'criteria': verdicts, 'native': native, 'native_reason': native_reason}


def recorded(report):
    qg = report.get('quality_gate') or {}
    density = (report.get('composition') or {}).get('output_density') or {}
    return {
        'quality_gate': qg.get('verdict') if qg.get('criteria') else 'not_run',
        'criteria': {r['name']: r.get('verdict') for r in qg.get('criteria') or []},
        'density': density.get('status'),
    }


def consistency(report, current):
    mismatches = []
    rec = recorded(report)
    if rec['quality_gate'] != current['quality_gate']:
        mismatches.append({'field': 'quality_gate', 'recorded': rec['quality_gate'],
                           'replayed': current['quality_gate']})
    for name, verdict in rec['criteria'].items():
        if current['criteria'].get(name) != verdict:
            mismatches.append({'field': f'criteria.{name}', 'recorded': verdict,
                               'replayed': current['criteria'].get(name)})
    if rec['density'] != current['density']:
        mismatches.append({'field': 'density', 'recorded': rec['density'], 'replayed': current['density']})
    return mismatches


def pick_report(target):
    """A report file, or the single authoritative report under a run directory."""
    path = Path(target)
    if path.is_file():
        return path
    candidates = sorted(glob.glob(str(path / '**' / '*stack-reports' / '*.json'), recursive=True))
    if not candidates:
        raise FileNotFoundError(f'no Stack_Report under {target}')
    loaded = []
    for candidate in candidates:
        with open(candidate, encoding='utf-8') as handle:
            data = json.load(handle)
        loaded.append((bool((data.get('quality_gate') or {}).get('criteria')),
                       data.get('finished_at_epoch_ms') or 0, candidate))
    with_qg = [entry for entry in loaded if entry[0]]
    pool = with_qg or loaded
    return Path(max(pool, key=lambda entry: entry[1])[2])


def replay_case(case, root):
    target = case['report'] if os.path.isabs(case['report']) else os.path.join(root, case['report'])
    path = pick_report(target)
    with open(path, encoding='utf-8') as handle:
        report = json.load(handle)
    current = evaluate(report)
    result = {
        'id': case['id'], 'dataset': case.get('dataset'), 'scale': case.get('scale'),
        'report': os.path.relpath(path, root), 'selected_path': report.get('selected_path'),
        'recorded': recorded(report), 'current': current,
        'consistency_mismatches': consistency(report, current),
        'options': {},
    }
    for option, meta in OPTIONS.items():
        if not meta['computable']:
            result['options'][option] = {'status': 'needs_rerun', 'needs': meta['needs']}
            continue
        what_if = evaluate(report, c1=option == 'C1', c3=option == 'C3')
        changed = {key: [current[key], what_if[key]] for key in ('density', 'quality_gate', 'native')
                   if current[key] != what_if[key]}
        changed_criteria = {name: [current['criteria'].get(name), verdict]
                            for name, verdict in what_if['criteria'].items()
                            if current['criteria'].get(name) != verdict}
        result['options'][option] = {'status': 'computed', 'result': what_if,
                                     'changed': changed, 'changed_criteria': changed_criteria}
    both = evaluate(report, c1=True, c3=True)
    result['options']['C1+C3'] = {
        'status': 'computed', 'result': both,
        'changed': {key: [current[key], both[key]] for key in ('density', 'quality_gate', 'native')
                    if current[key] != both[key]},
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases', default=str(Path(__file__).with_name('qg_whatif_cases.json')))
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--output', help='write the full result JSON here')
    args = parser.parse_args(argv)
    with open(args.cases, encoding='utf-8') as handle:
        cases = json.load(handle)['cases']
    acceptance = os.path.join(args.root, 'output', 'acceptance')
    results = [replay_case(case, acceptance) for case in cases]
    inconsistent = [r['id'] for r in results if r['consistency_mismatches']]
    out = {
        'schema': 1, 'tool': 'scripts/stack-lab/qg_whatif.py',
        'options': {key: {k: v for k, v in meta.items()} for key, meta in OPTIONS.items()},
        'consistency': 'pass' if not inconsistent else 'fail', 'inconsistent_cases': inconsistent,
        'cases': results,
    }
    text = json.dumps(out, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
    for r in results:
        line = [r['id'], r['current']['density'] or '-', r['current']['quality_gate'], r['current']['native']]
        for option in ('C1', 'C3', 'C1+C3'):
            changed = r['options'][option].get('changed') or {}
            line.append(f"{option}:{','.join(f'{k}->{v[1]}' for k, v in changed.items()) or 'same'}")
        line.append('CONSISTENT' if not r['consistency_mismatches'] else f"MISMATCH {r['consistency_mismatches']}")
        print(' | '.join(str(x) for x in line))
    return 0 if not inconsistent else 1


if __name__ == '__main__':
    sys.exit(main())
