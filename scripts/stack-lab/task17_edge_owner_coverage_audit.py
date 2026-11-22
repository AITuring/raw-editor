#!/usr/bin/env python3
"""Test-only attribution of a native boundary witness to output-edge coverage."""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--owner-map', required=True)
    ap.add_argument('--neighbourhood', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    owner_map = json.loads(Path(args.owner_map).read_text())
    neighbourhood = json.loads(Path(args.neighbourhood).read_text())
    pixels = neighbourhood.get('pixels', [])
    owner_zero = sum(p.get('owner', 0) == 0 for p in pixels)
    coverage_zero = sum(p.get('covered', 0) == 0 for p in pixels)
    luma_zero = sum(float(p.get('luma', 0.0)) == 0.0 for p in pixels)
    n = len(pixels)
    result = {
        'schema': 1,
        'task': 17,
        'method': 'task17_edge_owner_coverage_audit_r1',
        'production_impact': 'read-only test-only attribution; no threshold, bounds, fallback or QG change',
        'inputs': {'owner_map': args.owner_map, 'neighbourhood': args.neighbourhood},
        'canvas': owner_map.get('canvas'),
        'owner_region': owner_map.get('region'),
        'anchor_world': owner_map.get('anchor_world'),
        'anchor_output': owner_map.get('anchor_output'),
        'owner_region_unique_count': len(set(owner_map.get('owners', []))),
        'neighbourhood': {
            'world': neighbourhood.get('world'),
            'output': neighbourhood.get('output'),
            'owner_path': neighbourhood.get('owner_path'),
            'pixel_count': n,
            'owner_zero_count': owner_zero,
            'coverage_zero_count': coverage_zero,
            'luma_zero_count': luma_zero,
            'owner_zero_fraction': owner_zero / n if n else None,
            'coverage_zero_fraction': coverage_zero / n if n else None,
        },
        'attribution': {
            'classification': 'edge_uncovered_owner_zero',
            'passes_edge_attribution': bool(n and owner_zero == n and coverage_zero == n),
            'native_qg_result_preserved': True,
        },
        'conclusion': 'The witness is an uncovered output edge with owner=0; this explains the empty source provenance at the point but does not erase the native boundary-stroke QG failure.',
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'classification': result['attribution']['classification'], 'passes_edge_attribution': result['attribution']['passes_edge_attribution']}))


if __name__ == '__main__':
    main()
