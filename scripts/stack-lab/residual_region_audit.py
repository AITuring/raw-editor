#!/usr/bin/env python3
"""Audit low residual-Jacobian samples against the task-16 region report.

This is read-only with respect to source material.  It records every sample
below the requested scale threshold and, when the Stack_Report lacks station
poses, uses source-alignment's anchor homography only as an explicitly marked
fallback.  The fallback is diagnostic evidence, not a production mapping.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def project(h: list[float], point: list[float]) -> list[float] | None:
    x, y = point
    z = h[6] * x + h[7] * y + h[8]
    if not math.isfinite(z) or abs(z) <= 1e-12:
        return None
    wx = (h[0] * x + h[1] * y + h[2]) / z
    wy = (h[3] * x + h[4] * y + h[5]) / z
    return [wx, wy] if math.isfinite(wx) and math.isfinite(wy) else None


def region_membership(regions: list[dict[str, Any]], point: list[float] | None) -> dict[str, Any]:
    if point is None:
        return {"count": None, "nearest_boundary_px": None, "region_indices": []}
    x, y = point
    inside: list[int] = []
    distances: list[float] = []
    for index, region in enumerate(regions):
        world = region["world"]
        left = float(world["left"])
        top = float(world["top"])
        right = left + float(world["width"])
        bottom = top + float(world["height"])
        if left <= x <= right and top <= y <= bottom:
            inside.append(index)
            distances.append(min(x - left, right - x, y - top, bottom - y))
        else:
            dx = max(left - x, x - right, 0.0)
            dy = max(top - y, y - bottom, 0.0)
            distances.append(math.hypot(dx, dy))
    return {
        "count": len(inside),
        "nearest_boundary_px": min(distances) if distances else None,
        "region_indices": inside,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/Users/dp/Desktop/raw-editor/output/acceptance/task-16/mayuan-2-mayuan-2-native-residual-density-diagnostic-r1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/2B8-residual-region-audit-r1.json"),
    )
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "diagnostics" / "residual-density" / "summary.json"
    station_paths = sorted((run_dir / "diagnostics" / "residual-density").glob("station-*.json"))
    reports = sorted((run_dir / "run").glob("*stack-reports"), key=lambda p: p.name)
    report_paths = sorted(reports[0].glob("*.json")) if reports else []
    report_path = report_paths[0] if report_paths else None
    alignment_path = run_dir / "source-alignment.json"
    summary = load(summary_path)
    report = load(report_path) if report_path else {}
    alignment = load(alignment_path) if alignment_path.exists() else {}
    regions = report.get("residual_warp", {}).get("regions", [])

    # The report intentionally has no station pose.  Build a diagnostic-only
    # fallback from each station's anchor source and source-alignment matrix.
    source_by_path = {
        os.path.normpath(source["filename"]): source["id"]
        for source in alignment.get("sources", [])
        if "filename" in source and "id" in source
    }
    homographies = alignment.get("global_homographies", {})
    fallback_poses: dict[str, dict[str, Any]] = {}
    for row in report.get("intra_station", []):
        station = str(row.get("station_index"))
        anchor_path = os.path.normpath(row.get("anchor_path", ""))
        source_id = source_by_path.get(anchor_path)
        matrix = homographies.get(str(source_id)) if source_id is not None else None
        fallback_poses[station] = {
            "available": matrix is not None,
            "source": "source-alignment.global_homographies",
            "source_id": source_id,
            "matrix": matrix,
        }

    low_samples: list[dict[str, Any]] = []
    per_station: dict[str, dict[str, Any]] = {}
    for station_path in station_paths:
        station_doc = load(station_path)
        station = str(station_doc.get("station", int(station_path.stem.split("-")[-1])))
        pose_info = fallback_poses.get(station, {"available": False})
        matrix = pose_info.get("matrix")
        rows: list[dict[str, Any]] = []
        for sample in station_doc.get("samples", []):
            scale = sample.get("scale")
            if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) >= args.threshold:
                continue
            station_point = sample.get("point")
            world_point = project(matrix, station_point) if matrix and station_point else None
            membership = region_membership(regions, world_point)
            row = {
                "station": int(station),
                "source_id": sample.get("source_id"),
                "station_point": station_point,
                "world_point": world_point,
                "scale": scale,
                **membership,
            }
            rows.append(row)
            low_samples.append(row)
        per_station[station] = {
            "low_scale_count": len(rows),
            "pose_in_stack_report": False,
            "fallback_pose": pose_info,
            "region_membership_usable": bool(matrix),
        }

    worst = {
        "station": summary.get("worst_station_index"),
        "source_id": summary.get("worst_source_id"),
        "station_point": summary.get("worst_station_point"),
        "scale": summary.get("minimum_scale"),
    }
    worst_station = str(worst["station"]) if worst["station"] is not None else None
    if worst_station in fallback_poses and fallback_poses[worst_station].get("matrix") and worst["station_point"]:
        worst["world_point"] = project(fallback_poses[worst_station]["matrix"], worst["station_point"])
        worst.update(region_membership(regions, worst["world_point"]))
    else:
        worst.update(region_membership(regions, None))

    usable = [row for row in low_samples if row["count"] is not None]
    near_boundary = [row for row in usable if row["nearest_boundary_px"] <= 1.0]
    multiple = [row for row in usable if row["count"] >= 2]
    output = {
        "schema": 1,
        "task": "2B.8",
        "method": "task16_residual_region_audit",
        "inputs": {
            "run_dir": str(run_dir),
            "summary": str(summary_path),
            "stack_report": str(report_path) if report_path else None,
            "alignment": str(alignment_path) if alignment_path.exists() else None,
            "region_count": len(regions),
            "threshold_scale_lt": args.threshold,
        },
        "pose_contract": {
            "stack_report_contains_station_pose": False,
            "missing_fields": ["station_to_world", "right_station"],
            "fallback_used": bool(usable),
            "fallback_source": "source-alignment.global_homographies",
            "fallback_is_production_contract": False,
        },
        "worst_sample": worst,
        "low_scale_sample_count": len(low_samples),
        "low_scale_samples": low_samples,
        "classification": {
            "mapped_sample_count": len(usable),
            "unmapped_sample_count": len(low_samples) - len(usable),
            "multiple_region_count": len(multiple),
            "multiple_region_fraction": len(multiple) / len(usable) if usable else None,
            "near_boundary_count": len(near_boundary),
            "near_boundary_fraction": len(near_boundary) / len(usable) if usable else None,
            "supports_overlapping_boundary_hypothesis": (len(multiple) / len(usable) >= 0.5 and len(near_boundary) / len(usable) >= 0.5) if usable else False,
            "criterion_interpreted_as_both_majority": True,
        },
        "per_station": per_station,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "low_scale_sample_count": len(low_samples),
        "mapped_sample_count": len(usable),
        "multiple_region_count": len(multiple),
        "near_boundary_count": len(near_boundary),
        "supports_overlapping_boundary_hypothesis": (len(multiple) / len(usable) >= 0.5 and len(near_boundary) / len(usable) >= 0.5) if usable else False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
