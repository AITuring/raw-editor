#!/usr/bin/env python3
"""Test-only SIFT/AKAZE/ECC comparison on saved Langyuan 0-2 planes.

This produces candidate registration evidence only.  It does not write poses,
Stack_Report fields, pixels, or quality-gate verdicts.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def image(path: Path, long_side: int) -> tuple[np.ndarray, np.ndarray]:
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"cannot decode {path}")
    scale = long_side / max(raw.shape)
    if scale < 1.0:
        resized = cv2.resize(raw, (round(raw.shape[1] * scale), round(raw.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = raw
        scale = 1.0
    return resized, np.diag([scale, scale, 1.0])


def project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.astype(np.float64)[None], matrix)[0]


def candidate(detector, matcher, left: np.ndarray, right: np.ndarray, ratio: float) -> dict:
    left_kp, left_desc = detector.detectAndCompute(left, None)
    right_kp, right_desc = detector.detectAndCompute(right, None)
    if left_desc is None or right_desc is None:
        return {"status": "no_descriptors", "keypoints": [len(left_kp), len(right_kp)]}
    pairs = matcher.knnMatch(left_desc, right_desc, k=2)
    good = [a for a, b in pairs if a.distance < ratio * b.distance]
    if len(good) < 8:
        return {"status": "too_few_matches", "keypoints": [len(left_kp), len(right_kp)], "matches": len(good)}
    src = np.float32([left_kp[m.queryIdx].pt for m in good])
    dst = np.float32([right_kp[m.trainIdx].pt for m in good])
    matrix, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 2.0, maxIters=15000, confidence=0.999)
    if matrix is None or mask is None:
        return {"status": "homography_failed", "keypoints": [len(left_kp), len(right_kp)], "matches": len(good)}
    inlier = mask.ravel().astype(bool)
    errors = np.linalg.norm(project(src[inlier], matrix) - dst[inlier], axis=1)
    hull = cv2.convexHull(src[inlier].reshape(-1, 1, 2)) if inlier.any() else None
    support = float(cv2.contourArea(hull) / max(left.shape[0] * left.shape[1], 1)) if hull is not None else 0.0
    return {
        "status": "ok",
        "keypoints": [len(left_kp), len(right_kp)],
        "matches": len(good),
        "inliers": int(inlier.sum()),
        "median_reprojection_px": float(np.median(errors)) if len(errors) else None,
        "p95_reprojection_px": float(np.percentile(errors, 95)) if len(errors) else None,
        "hull_support": support,
        "homography": matrix.reshape(-1).tolist(),
    }


def ecc(left: np.ndarray, right: np.ndarray, initial: np.ndarray) -> dict:
    warped = cv2.warpPerspective(left, initial, (right.shape[1], right.shape[0]), borderMode=cv2.BORDER_CONSTANT)
    template = right.astype(np.float32) / 255.0
    input_image = warped.astype(np.float32) / 255.0
    warp = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        correlation, refined = cv2.findTransformECC(template, input_image, warp, cv2.MOTION_HOMOGRAPHY, criteria, None, 1)
    except cv2.error as error:
        return {"status": "ecc_failed", "error": str(error).splitlines()[-1]}
    return {"status": "ok", "correlation": float(correlation), "refinement": refined.reshape(-1).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planes", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-side", type=int, default=2048)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(71)
    joint = json.loads(args.joint.read_text())
    planes = [image(args.planes / f"station-{index:03d}.gray.png", args.long_side) for index in range(3)]
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.008, edgeThreshold=12)
    akaze = cv2.AKAZE_create()
    sift_matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=128))
    akaze_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    records = []
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        left, left_scale = planes[left_index]
        right, right_scale = planes[right_index]
        rust_left = np.asarray(joint["poses"][left_index], dtype=np.float64)
        rust_right = np.asarray(joint["poses"][right_index], dtype=np.float64)
        initial = right_scale @ np.linalg.inv(rust_right) @ rust_left @ np.linalg.inv(left_scale)
        records.append({
            "left": left_index,
            "right": right_index,
            "sift": candidate(sift, sift_matcher, left, right, 0.72),
            "akaze": candidate(akaze, akaze_matcher, left, right, 0.80),
            "ecc": ecc(left, right, initial),
            "rust_saved_record": next((r for r in joint["records"] if r["left"] == left_index and r["right"] == right_index), None),
        })
    result = {
        "schema": 1,
        "method": "langyuan_provider_candidate_compare_v1",
        "production_impact": "test_only_candidate_metrics_no_pose_write_no_stack_report_or_qg_change",
        "planes": str(args.planes),
        "joint": str(args.joint),
        "long_side": args.long_side,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "providers": ["sift", "akaze", "ecc"],
        "records": records,
        "decision": "candidate_only; Rust geometry validation remains authoritative and was not rerun through a provider sidecar",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "providers": result["providers"], "pairs": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
