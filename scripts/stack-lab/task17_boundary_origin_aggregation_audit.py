#!/usr/bin/env python3
"""Read-only attribution of boundary QG failures recorded at world_origin."""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stack-report', required=True)
    ap.add_argument('--neighbourhood', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    report = json.loads(Path(args.stack_report).read_text())
    nb = json.loads(Path(args.neighbourhood).read_text())
    criterion = next(c for c in report['quality_gate']['criteria'] if c['name'] == 'boundary_stroke_alignment')
    origin_value = report.get('closure', {}).get('world_origin') or nb.get('world')
    origin = ([origin_value.get('x'), origin_value.get('y')] if isinstance(origin_value, dict) else origin_value)
    failed = criterion.get('failed', [])
    origin_matches = [x for x in failed if [x['world']['x'], x['world']['y']] == list(origin)] if origin else []
    pixels = nb.get('pixels', [])
    result = {
        'schema': 1,
        'task': 17,
        'method': 'task17_boundary_origin_aggregation_audit_r1',
        'production_impact': 'read-only test-only attribution; no quality gate or threshold change',
        'stack_report': args.stack_report,
        'neighbourhood': args.neighbourhood,
        'quality_gate': {'verdict': report['quality_gate']['verdict'], 'criterion': criterion['name'], 'failed_count': len(failed)},
        'origin': origin,
        'failed_at_origin_count': len(origin_matches),
        'failed_at_origin_examples': origin_matches[:8],
        'neighbourhood': {
            'output': nb.get('output'),
            'owner_path': nb.get('owner_path'),
            'pixel_count': len(pixels),
            'owner_zero_count': sum(p.get('owner', 0) == 0 for p in pixels),
            'coverage_zero_count': sum(p.get('covered', 0) == 0 for p in pixels),
            'luma_zero_count': sum(float(p.get('luma', 0.0)) == 0.0 for p in pixels),
        },
        'classification': 'boundary_origin_uncovered_aggregation',
        'interpretation': 'A native boundary failure is reported at the cropped world origin while the witness neighborhood is owner=0 and coverage=0. This is an attribution/crop semantic lead; it does not change the native QG verdict.',
        'next_step': 'Use a test-only criterion fixture or explicit crop-boundary exclusion audit before considering any production semantic change; do not widen thresholds or bounds.',
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'classification': result['classification'], 'failed_at_origin_count': result['failed_at_origin_count']}))


if __name__ == '__main__':
    main()
