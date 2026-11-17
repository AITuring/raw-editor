#!/usr/bin/env python3
"""Lanting paper-tone audit and guard for a framed run (from lanting-v63 audit-substrate.py).

    lanting_audit.py <framed-run-dir>          # writes <dir>/../paper-tone-audit.json, then runs the guard

The guard (tests/focus-stack-lanting-guard.mjs) expects exactly this layout:
<dir>/{analysis.json, execution.json, run/, diagnostics/} and <dir>/../paper-tone-audit.json.
Zone MAE uses a sips preview of the TIFF; it is a diagnostic, not a Quality Gate value.
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, MAIN, read, repo_root, sha, write  # noqa: E402

W, H = 2048, 726
ZONES = {'wall': (100, 480, 2, 16), 'rail': (180, 490, 16, 33), 'artwork': (40, 470, 35, 158), 'bottom_frame': (40, 470, 158, 178)}


def audit(run_dir):
    run_dir = Path(run_dir)
    base = run_dir.parent
    analysis = read(run_dir / 'analysis.json')
    report = read(analysis['report'])
    tiff = Path(report['output']['path'])
    frame = report['composition']['publication_crop']
    reference = DATA / '兰亭/DSC08854_FocusStack.jpg'
    for name, path in [('reference', reference), ('current', tiff)]:
        subprocess.run(['sips', '-s', 'format', 'jpeg', '-Z', '2048', str(path), '--out', str(base / f'{name}-2048.jpg')],
                       check=True, stdout=subprocess.DEVNULL)

    def rgb(name):
        return subprocess.check_output(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1', '-i', str(base / name), '-vf', 'scale=2048:726', '-filter_threads', '1',
                                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])

    ref, cur = rgb('reference-2048.jpg'), rgb('current-2048.jpg')

    def paper(data, cx, cy):
        samples = []
        for y in range(max(118, cy - 24), min(669, cy + 25), 2):
            for x in range(max(0, cx - 24), min(W, cx + 25), 2):
                i = (y * W + x) * 3
                r, g, b = data[i:i + 3]
                if min(r, g, b) < 20 or r / max(g, 1) > 1.42 or g / max(b, 1) > 1.6:
                    continue
                samples.append((.2126 * r + .7152 * g + .0722 * b, r, g, b))
        if len(samples) < 80:
            return None
        samples.sort()
        selected = samples[int(len(samples) * .55):int(len(samples) * .9)]
        if len(selected) < 30:
            return None
        return {'rgb': [statistics.median(v[c] for v in selected) for c in [1, 2, 3]], 'count': len(selected)}

    points = []
    for y in range(142, 650, 32):
        for x in range(24, W - 24, 32):
            a, b = paper(ref, x, y), paper(cur, x, y)
            if a is None or b is None:
                continue
            points.append({'x': x, 'y': y, 'reference': a, 'current': b, 'rgb_residual': [b['rgb'][c] - a['rgb'][c] for c in range(3)]})
    lookup = {(p['x'], p['y']): p for p in points}
    edges = []
    for p in points:
        for dx, dy in [(32, 0), (0, 32)]:
            q = lookup.get((p['x'] + dx, p['y'] + dy))
            if q is None:
                continue
            delta = [q['rgb_residual'][c] - p['rgb_residual'][c] for c in range(3)]
            edges.append({'x': p['x'] + dx / 2, 'y': p['y'] + dy / 2, 'orientation': 'vertical' if dx else 'horizontal',
                          'rgb_residual_step': delta, 'max_channel_step': max(map(abs, delta))})
    edges.sort(key=lambda e: e['max_channel_step'], reverse=True)
    values = sorted(e['max_channel_step'] for e in edges)
    summary = {'paper_points': len(points), 'adjacent_comparisons': len(edges),
               'median_residual_rgb': [statistics.median(p['rgb_residual'][c] for p in points) for c in range(3)],
               'median_adjacent_step': statistics.median(values), 'p95_adjacent_step': values[int(len(values) * .95)],
               'maximum_adjacent_step': max(values)}
    meta = read(run_dir / 'diagnostics/metadata.json')
    ow, oh = meta['analysis_width'], meta['analysis_height']
    owners = subprocess.check_output(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1', '-i', str(run_dir / 'diagnostics/owner.png'),
                                      '-f', 'rawvideo', '-pix_fmt', 'gray', '-'])

    def owner_at(x, y):
        ox = round((frame['x'] + x / W * frame['width']) / meta['analysis_scale'])
        oy = round((frame['y'] + y / H * frame['height']) / meta['analysis_scale'])
        return owners[min(oh - 1, max(0, oy)) * ow + min(ow - 1, max(0, ox))]

    for e in edges:
        pair = ([owner_at(e['x'] - 16, e['y']), owner_at(e['x'] + 16, e['y'])] if e['orientation'] == 'vertical'
                else [owner_at(e['x'], e['y'] - 16), owner_at(e['x'], e['y'] + 16)])
        e['owner_pair'] = pair
        e['crosses_owner'] = pair[0] != pair[1]
    for name, subset in [('source_boundary', [e for e in edges if e['crosses_owner']]), ('source_interior', [e for e in edges if not e['crosses_owner']])]:
        vals = sorted(e['max_channel_step'] for e in subset)
        summary[name] = {'count': len(vals), 'median': statistics.median(vals), 'p95': vals[int(len(vals) * .95)],
                         'max': max(vals), 'above_6': sum(v > 6 for v in vals)}

    def gray(name):
        return subprocess.check_output(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1', '-i', str(base / name), '-vf',
                                        'scale=512:182,format=gray', '-filter_threads', '1', '-frames:v', '1', '-f', 'rawvideo', '-'])

    a, b = gray('reference-2048.jpg'), gray('current-2048.jpg')
    summary['zone_gray_mae'] = {}
    for zone, (x0, x1, y0, y1) in ZONES.items():
        idx = [y * 512 + x for y in range(y0, y1) for x in range(x0, x1)]
        summary['zone_gray_mae'][zone] = round(sum(abs(a[i] - b[i]) for i in idx) / len(idx), 2)
    # Color conversion can differ under a restricted macOS process context.
    # Keep the actual audit substrate bound to the measurements, not just the
    # source JPEG/TIFF. Compare candidates only with identical reference bytes.
    provenance = {
        'method': 'lanting_paper_sips_ffmpeg_v1',
        'script_sha256': sha(Path(__file__)),
        'reference_sha256': sha(reference),
        'output_sha256': sha(tiff),
        'reference_preview_sha256': sha(base / 'reference-2048.jpg'),
        'current_preview_sha256': sha(base / 'current-2048.jpg'),
        'owner_sha256': sha(run_dir / 'diagnostics/owner.png'),
        'reference_rgb_sha256': hashlib.sha256(ref).hexdigest(),
        'current_rgb_sha256': hashlib.sha256(cur).hexdigest(),
    }
    write(base / 'paper-tone-audit.json', {'summary': summary, 'provenance': provenance, 'points': points, 'edges': edges})
    return summary


def guard(run_dir, regression_only=False):
    root = repo_root() if (Path.cwd() / 'tests').is_dir() else MAIN
    command = ['node', 'tests/focus-stack-lanting-guard.mjs', 'tests/acceptance/lanting-paper-baseline.json', str(run_dir)]
    if regression_only:
        command.append('--regression-only')
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    try:
        verdict = json.loads(result.stdout)
        if result.returncode != 0:
            verdict['pass'] = False
        return verdict
    except ValueError:
        return {'pass': False, 'status': 'guard_error', 'failures': [result.stderr.strip()[-500:]]}


if __name__ == '__main__':
    target = Path(sys.argv[1]).resolve()
    paper = audit(target)
    print(json.dumps({'paper': {k: paper[k] for k in ('paper_points', 'median_residual_rgb', 'source_boundary', 'zone_gray_mae')}}, ensure_ascii=False))
    passed = True
    for mode in (False, True):
        verdict = guard(target, mode)
        passed = passed and verdict.get('pass') is True
        print(json.dumps({'guard': 'regression-only' if mode else 'strict', 'pass': verdict.get('pass'), 'status': verdict.get('status'),
                          'failures': verdict.get('failures')}, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)
