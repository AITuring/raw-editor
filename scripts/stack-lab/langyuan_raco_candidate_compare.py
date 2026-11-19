#!/usr/bin/env python3
"""Test-only RaCo-ALIKED-LightGlue+ ONNX candidate comparison.

The end-to-end model returns keypoints, matches and match scores only.  This
runner therefore records candidate metrics and explicitly does not claim a
Rust FeatureProviderResult or geometry-gate submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from lightglue_dynamo.preprocessors import RaCoPreprocessor, prepare_host_images


PAIR_ORDER = ((0, 1), (0, 2), (1, 2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planes", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--license", type=Path)
    args = parser.parse_args()

    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(
        str(args.model), options, providers=["CPUExecutionProvider"]
    )
    plane_paths = [args.planes / f"station-{index:03d}.gray.png" for index in range(3)]
    records = []
    for left, right in PAIR_ORDER:
        started = time.perf_counter()
        prepared = prepare_host_images(
            (plane_paths[left], plane_paths[right]),
            args.input_size,
            args.input_size,
            RaCoPreprocessor,
        )
        keypoints, matches, scores = session.run(None, {"images": prepared.images})[:3]
        matches = np.asarray(matches, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float32)
        points_left = np.asarray(keypoints[0][matches[:, 1], :2], dtype=np.float32)
        points_right = np.asarray(keypoints[1][matches[:, 2], :2], dtype=np.float32)
        homography = None
        mask = None
        if len(matches) >= 4:
            homography, mask = cv2.findHomography(points_left, points_right, cv2.RANSAC, 3.0)
        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), bool)
        if homography is not None and inlier_mask.any():
            inlier_left = points_left[inlier_mask]
            inlier_right = points_right[inlier_mask]
            projected = cv2.perspectiveTransform(
                inlier_left.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            errors = np.linalg.norm(projected - inlier_right, axis=1)
            hull = cv2.convexHull(inlier_left) if len(inlier_left) >= 3 else None
            hull_support = (
                float(cv2.contourArea(hull) / (args.input_size * args.input_size))
                if hull is not None
                else 0.0
            )
            metrics = {
                "median_reprojection_px": float(np.median(errors)),
                "p95_reprojection_px": float(np.quantile(errors, 0.95)),
                "hull_support": hull_support,
            }
        else:
            metrics = {
                "median_reprojection_px": None,
                "p95_reprojection_px": None,
                "hull_support": 0.0,
            }
        records.append(
            {
                "left": left,
                "right": right,
                "status": "ok",
                "coordinate_frame": [args.input_size, args.input_size],
                "keypoints": [int(keypoints[0].shape[0]), int(keypoints[1].shape[0])],
                "matches": int(len(matches)),
                "inliers": int(inlier_mask.sum()),
                **metrics,
                "elapsed_s": time.perf_counter() - started,
            }
        )

    result = {
        "schema": 1,
        "method": "langyuan_raco_aliked_lightglue_candidate_v1",
        "production_impact": "test_only_candidate_no_pose_write_no_stack_report_or_qg_change",
        "provider": "python_onnx_raco_aliked_lightglue",
        "upstream_repository": "https://github.com/fabio-sim/LightGlue-ONNX",
        "upstream_commit": "d12b4ba1632f558234e3f084e1f3d8bdf9147890",
        "release": "v3.0",
        "planes": str(args.planes),
        "model": {
            "path": str(args.model),
            "filename": args.model.name,
            "sha256": sha256(args.model),
        },
        "license": {
            "path": str(args.license) if args.license else None,
            "sha256": sha256(args.license) if args.license else None,
            "lightglue_code": "Apache-2.0",
            "aliked_component": "BSD-3-Clause",
            "review_required": True,
        },
        "runtime": {
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "onnxruntime": ort.__version__,
            "input_resize": [args.input_size, args.input_size],
            "onnx_intra_op_threads": 1,
            "onnx_inter_op_threads": 1,
        },
        "records": records,
        "outputs": ["keypoints", "matches", "mscores"],
        "descriptors_output": False,
        "rust_feature_provider_result_compatible": False,
        "rust_geometry_gate_rerun": False,
        "decision": "candidate_only; the end-to-end pipeline does not emit descriptors and no Rust sidecar adapter/geometry rerun exists",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "pairs": len(records)}))


if __name__ == "__main__":
    main()
