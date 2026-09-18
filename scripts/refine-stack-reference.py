#!/usr/bin/env python3
"""Estimate a small, local deformation on top of reference registrations.

The registration manifest supplies a projective transform for every source.
This tool does not copy reference pixels into the result.  It only stores a
coarse displacement field measured between the warped source thumbnail and
the reference preview.  A renderer can use it as an inverse remap: for an
output/reference point ``p``, sample the already projectively warped source
at ``p + d(p)``.  Fields are local to each source bounding box and therefore
remain compact enough to carry to a full-resolution renderer.

The displacement is estimated with OpenCV DIS after local contrast
normalisation.  Invalid borders, forward/backward inconsistent vectors, and
large vectors are removed before inpainting and coarse-grid sampling.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def local_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.GaussianBlur(gray, (0, 0), 9)
    var = np.maximum(cv2.GaussianBlur(gray * gray, (0, 0), 9) - mean * mean, 0)
    return np.clip(128.0 + 28.0 * (gray - mean) / (np.sqrt(var) + 3.0), 0, 255).astype(np.uint8)


def projective_preview_homography(entry, ref_width, ref_height):
    sw, sh = entry["thumb_width"], entry["thumb_height"]
    sx = np.diag([ref_width / entry["reference_width"], ref_height / entry["reference_height"], 1.0])
    source_to_full = np.asarray(entry["homography"], np.float64).reshape(3, 3)
    source_to_thumb = np.diag([entry["width"] / sw, entry["height"] / sh, 1.0])
    return sx @ source_to_full @ source_to_thumb


def refine_entry(entry, reference, reference_width, reference_height, grid_step, max_flow):
    source = cv2.imread(entry["thumb_filename"], cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Cannot decode {entry['thumb_filename']}")
    height, width = reference.shape[:2]
    # The manifest homography is full-resolution reference coordinates.  The
    # local field is intentionally measured at the dimensions of the supplied
    # preview, so this scale must be the preview width/height here.
    measured = dict(entry, reference_width=reference_width, reference_height=reference_height)
    H = projective_preview_homography(measured, width, height)
    mask = cv2.warpPerspective(np.full(source.shape[:2], 255, np.uint8), H, (width, height))
    x, y, w, h = cv2.boundingRect(mask)
    pad = 24
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
    ref_crop = reference[y0:y1, x0:x1]
    warped = cv2.warpPerspective(source, H, (width, height))[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    # Keep invalid pixels from creating a fake edge in DIS.  They are never
    # sampled because the eroded validity mask below excludes the border.
    source_norm = local_contrast(warped)
    ref_norm = local_contrast(ref_crop)
    source_norm[mask_crop < 1] = ref_norm[mask_crop < 1]

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    flow = dis.calc(ref_norm, source_norm, None).astype(np.float32)
    backward = dis.calc(source_norm, ref_norm, None).astype(np.float32)
    grid = np.mgrid[0 : ref_crop.shape[0], 0 : ref_crop.shape[1]].transpose(1, 2, 0)[:, :, ::-1].astype(np.float32)
    backward_at_flow = cv2.remap(backward, grid + flow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    consistency = np.linalg.norm(flow + backward_at_flow, axis=2)
    valid = cv2.erode(mask_crop, np.ones((25, 25), np.uint8), iterations=1) == 255
    valid &= consistency < 1.5
    valid &= np.linalg.norm(flow, axis=2) < max_flow
    valid_ratio = float(valid.mean())
    if valid_ratio < 0.25:
        raise RuntimeError(f"{Path(entry['filename']).name}: only {valid_ratio:.3f} reliable local flow")

    bad = (~valid).astype(np.uint8) * 255
    for component in (0, 1):
        flow[:, :, component] = cv2.inpaint(flow[:, :, component], bad, 5, cv2.INPAINT_NS)
    flow = cv2.GaussianBlur(flow, (0, 0), 1.5)
    cols = max(2, int(np.ceil((flow.shape[1] - 1) / grid_step)) + 1)
    rows = max(2, int(np.ceil((flow.shape[0] - 1) / grid_step)) + 1)
    # Store nodes at exact multiples of grid_step.  A generic resize samples
    # pixel centres and would shift this field by roughly half a grid cell
    # relative to the Rust renderer's local_x / step interpolation.
    grid_y, grid_x = np.mgrid[0:rows, 0:cols].astype(np.float32)
    grid_flow = cv2.remap(
        flow,
        grid_x * grid_step,
        grid_y * grid_step,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # Clip extrapolated values at the outermost cell.  It prevents a sparse
    # field from bending a source outside its projective footprint.
    grid_flow = np.clip(grid_flow, -max_flow, max_flow)
    return {
        "origin_x": int(x0),
        "origin_y": int(y0),
        "width": int(flow.shape[1]),
        "height": int(flow.shape[0]),
        "step": int(grid_step),
        "cols": int(cols),
        "rows": int(rows),
        "displacements": grid_flow.astype(np.float32).round(4).reshape(-1, 2).tolist(),
        "valid_ratio": valid_ratio,
        "median_magnitude": float(np.median(np.linalg.norm(flow[valid], axis=1))),
        "p90_magnitude": float(np.percentile(np.linalg.norm(flow[valid], axis=1), 90)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reference-preview", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid-step", type=int, default=16)
    parser.add_argument("--max-flow", type=float, default=40.0)
    parser.add_argument("--ids", nargs="*", type=int, help="Only refine these source IDs (for diagnostics)")
    args = parser.parse_args()
    if args.grid_step < 4:
        raise ValueError("--grid-step must be at least 4")
    reference = cv2.imread(str(args.reference_preview), cv2.IMREAD_COLOR)
    if reference is None:
        raise RuntimeError(f"Cannot decode {args.reference_preview}")
    manifest = json.loads(args.manifest.read_text())
    ref_height, ref_width = reference.shape[:2]
    wanted = set(args.ids) if args.ids else None
    output_sources = []
    for source in manifest["sources"]:
        item = dict(source)
        if wanted is None or source["id"] in wanted:
            item["refinement"] = refine_entry(
                item,
                reference,
                manifest["reference_width"],
                manifest["reference_height"],
                args.grid_step,
                args.max_flow,
            )
            r = item["refinement"]
            print(f"{Path(source['filename']).name}: valid={r['valid_ratio']:.3f}, "
                  f"median={r['median_magnitude']:.3f}px, p90={r['p90_magnitude']:.3f}px", flush=True)
        output_sources.append(item)
    manifest["reference_preview_width"] = ref_width
    manifest["reference_preview_height"] = ref_height
    manifest["local_refinement"] = {
        "coordinate_system": "reference_preview",
        "sampling": "inverse: source_sample = projective_warp(source, p + d(p))",
        "grid_step": args.grid_step,
        "max_flow": args.max_flow,
    }
    manifest["sources"] = output_sources
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote refined registration manifest to {args.output}")


if __name__ == "__main__":
    main()
