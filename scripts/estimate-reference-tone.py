#!/usr/bin/env python3
"""Estimate per-source low-frequency colour calibration from a trusted reference.

Only broad tone is measured; no reference pixels are written to the mosaic.
For every source thumbnail, the projective registration is applied, both the
warped source and the reference are Gaussian blurred, and robust per-channel
affine coefficients are fit on their valid overlap::

    reference_low ~= gain * source_low + offset

The coefficients can be applied to the original/full-resolution source before
focus selection.  Samples are clipped to the photographic range and weighted
by local gradient agreement so dark borders, glass highlights, and brush edges
do not drive exposure.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def homography_to_preview(entry, ref_w, ref_h, reference_width, reference_height):
    source_scale = np.diag(
        [entry["width"] / entry["thumb_width"], entry["height"] / entry["thumb_height"], 1.0]
    )
    return np.diag([ref_w / reference_width, ref_h / reference_height, 1.0]) @ np.asarray(
        entry["homography"], np.float64
    ).reshape(3, 3) @ source_scale


def fit_channel(x, y):
    # Iteratively reweighted least squares with Huber-like residual weights.
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    design = np.column_stack([x, np.ones_like(x)])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(4):
        residual = y - design @ coef
        scale = max(1e-3, float(np.median(np.abs(residual - np.median(residual))) * 1.4826))
        weights = np.minimum(1.0, 2.5 * scale / np.maximum(np.abs(residual), 1e-6))
        weighted = design * weights[:, None]
        coef = np.linalg.lstsq(weighted, y * weights, rcond=None)[0]
    return float(coef[0]), float(coef[1])


def estimate(entry, reference, reference_width, reference_height, blur_sigma, max_samples):
    source = cv2.imread(entry["thumb_filename"], cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Cannot decode {entry['thumb_filename']}")
    ref_h, ref_w = reference.shape[:2]
    H = homography_to_preview(entry, ref_w, ref_h, reference_width, reference_height)
    warped = cv2.warpPerspective(source, H, (ref_w, ref_h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(source.shape[:2], 255, np.uint8), H, (ref_w, ref_h))
    valid = cv2.erode(valid, np.ones((31, 31), np.uint8)) > 0
    source_low = cv2.GaussianBlur(warped, (0, 0), blur_sigma).astype(np.float32) / 255.0
    reference_low = cv2.GaussianBlur(reference, (0, 0), blur_sigma).astype(np.float32) / 255.0
    # Avoid black canvas, clipped glass highlights, and edge/ink samples where
    # sub-pixel registration differences would masquerade as exposure.
    gradients = cv2.cvtColor(reference_low, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gradients, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gradients, cv2.CV_32F, 0, 1, ksize=3)
    quiet = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 3) < 0.10
    valid &= quiet
    flat = valid & np.all((source_low > 0.025) & (source_low < 0.97), axis=2)
    flat &= np.all((reference_low > 0.025) & (reference_low < 0.97), axis=2)
    points = np.argwhere(flat)
    if len(points) > max_samples:
        points = points[np.linspace(0, len(points) - 1, max_samples).astype(int)]
    if len(points) < 50:
        raise RuntimeError(f"{Path(entry['filename']).name}: only {len(points)} tone samples")
    x = source_low[points[:, 0], points[:, 1]][:, ::-1]  # RGB channel order
    y = reference_low[points[:, 0], points[:, 1]][:, ::-1]
    coeffs = [fit_channel(x[:, c], y[:, c]) for c in range(3)]
    predicted = np.column_stack([x[:, c] * coeffs[c][0] + coeffs[c][1] for c in range(3)])
    error = np.abs(predicted - y)
    return {
        "gain_rgb": [v[0] for v in coeffs],
        "offset_rgb": [v[1] for v in coeffs],
        "samples": int(len(points)),
        "median_abs_error_rgb": np.median(error, axis=0).tolist(),
        "p90_abs_error_rgb": np.quantile(error, 0.9, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--blur-sigma", type=float, default=24.0)
    parser.add_argument("--max-samples", type=int, default=6000)
    args = parser.parse_args()
    reference = cv2.imread(str(args.reference), cv2.IMREAD_COLOR)
    if reference is None:
        raise RuntimeError(f"Cannot decode {args.reference}")
    manifest = json.loads(args.manifest.read_text())
    output = dict(manifest)
    output["reference_tone"] = {
        "model": "reference_low = gain_rgb * source_low + offset_rgb",
        "blur_sigma_preview_px": args.blur_sigma,
        "source": str(args.reference),
    }
    output["sources"] = []
    for entry in manifest["sources"]:
        item = dict(entry)
        item["tone_calibration"] = estimate(
            entry,
            reference,
            manifest["reference_width"],
            manifest["reference_height"],
            args.blur_sigma,
            args.max_samples,
        )
        t = item["tone_calibration"]
        print(f"{Path(entry['filename']).name}: samples={t['samples']}, gain="
              f"{np.round(t['gain_rgb'], 4)}, offset={np.round(t['offset_rgb'], 4)}", flush=True)
        output["sources"].append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote reference tone manifest to {args.output}")


if __name__ == "__main__":
    main()
