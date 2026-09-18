#!/usr/bin/env python3
"""Measure source-to-reference homographies; never use reference pixels in a render.

Input is reference-sources.json from the ignored Rust export_prepared_sources_from_env
harness. Requires numpy and opencv-python (SIFT). Output is consumed by
render_reference_registered_stack_from_env. Every source must pass the checks.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def project(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, np.float64)[None], matrix)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, type=Path)
    parser.add_argument('--reference-preview', required=True, type=Path)
    parser.add_argument('--reference-width', required=True, type=int)
    parser.add_argument('--reference-height', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    cv2.setRNGSeed(71)
    reference = cv2.imread(str(args.reference_preview), cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise RuntimeError('Cannot decode reference preview')
    ref_height, ref_width = reference.shape
    reference_scale = np.diag([args.reference_width/ref_width, args.reference_height/ref_height, 1.])
    detector = cv2.SIFT_create(nfeatures=40000, contrastThreshold=.008, edgeThreshold=12)
    ref_keypoints, ref_descriptors = detector.detectAndCompute(reference, None)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=128))
    records = json.loads(args.sources.read_text())['sources']
    output, failures = [], []
    previous = None
    group_id = 0
    for entry in records:
        source = cv2.imread(entry['thumb_filename'], cv2.IMREAD_GRAYSCALE)
        if source is None:
            raise RuntimeError(f"Cannot decode {entry['thumb_filename']}")
        sh, sw = source.shape
        keypoints, descriptors = detector.detectAndCompute(source, None)
        if descriptors is None:
            failures.append(entry['filename']); continue
        pairs = matcher.knnMatch(descriptors, ref_descriptors, k=2)
        matches = [a for a, b in pairs if a.distance < .72*b.distance]
        if len(matches) < 30:
            failures.append(entry['filename']); continue
        src = np.float32([keypoints[m.queryIdx].pt for m in matches])
        dst = np.float32([ref_keypoints[m.trainIdx].pt for m in matches])
        matrix, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 2., maxIters=15000, confidence=.999)
        if matrix is None:
            failures.append(entry['filename']); continue
        inliers = mask.ravel() > 0
        src, dst = src[inliers], dst[inliers]
        errors = np.linalg.norm(project(src, matrix)-dst, axis=1)
        spans = (np.percentile(src, 95, axis=0)-np.percentile(src, 5, axis=0))/[sw, sh]
        support = float(np.prod(spans))
        corners = project([[0,0],[sw,0],[sw,sh],[0,sh]], matrix).astype(np.float32)
        good_geometry = cv2.isContourConvex(corners) and abs(cv2.contourArea(corners)) > sw*sh*.05
        if len(src) < 30 or support < .015 or np.percentile(errors,90) > 2.5 or not good_geometry:
            failures.append(entry['filename']); continue
        full = reference_scale @ matrix @ np.diag([sw/entry['width'], sh/entry['height'], 1.])
        full /= full[2,2]
        same_station = False
        if previous is not None:
            relative = np.linalg.inv(previous['matrix']) @ full
            center = np.array([entry['width']/2, entry['height']/2])
            displacement = np.linalg.norm(project([center], relative)[0]-center)/max(entry['width'],entry['height'])
            same_station = displacement < .06
        if not same_station:
            group_id += 1
        previous = {'matrix': full}
        report = dict(inliers=len(src), matches=len(matches), median_error_preview_px=float(np.median(errors)),
                      p90_error_preview_px=float(np.percentile(errors,90)), source_support_area=support)
        output.append(dict(entry, group_id=group_id, homography=full.ravel().tolist(), registration=report))
        print(f"{Path(entry['filename']).name}: {len(src)}/{len(matches)} inliers, "
              f"median={np.median(errors):.3f}px, p90={np.percentile(errors,90):.3f}px, "
              f"support={support:.3f}, group={group_id}", flush=True)
    if failures:
        raise RuntimeError(f'Unverified sources: {failures}')
    result = dict(reference=str(args.reference_preview), reference_width=args.reference_width,
                  reference_height=args.reference_height, registration_preview_width=ref_width,
                  registration_preview_height=ref_height, sources=output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(f'Wrote {len(output)} verified source poses in {group_id} stations to {args.output}', flush=True)


if __name__ == '__main__':
    main()
