#!/usr/bin/env python3
"""Test-only owner/coverage stencil audit for the Fengying diagnostic export."""
import argparse
import collections
import json
import subprocess
from pathlib import Path


def raw(path):
    return subprocess.check_output(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--origin-x", type=float, required=True)
    parser.add_argument("--origin-y", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads((args.diagnostics / "metadata.json").read_text())
    width = metadata["analysis_width"]
    height = metadata["analysis_height"]
    scale = metadata["analysis_scale"]
    owner = raw(args.diagnostics / "owner.png")
    coverage = raw(args.diagnostics / "coverage.png")
    expected = width * height
    if len(owner) != expected or len(coverage) != expected:
        raise SystemExit(f"diagnostic byte count mismatch: expected {expected}, owner={len(owner)}, coverage={len(coverage)}")

    analysis = json.loads(args.analysis.read_text())
    report_path = args.report or Path(analysis["report"])
    report = json.loads(report_path.read_text())
    failures = []
    for criterion in (report.get("quality_gate") or {}).get("criteria", []):
        for item in criterion.get("failed", []):
            world = item.get("world")
            if not world:
                continue
            ax = round((world["x"] - args.origin_x) / scale)
            ay = round((world["y"] - args.origin_y) / scale)
            samples = []
            covered = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    xx = max(0, min(width - 1, ax + dx))
                    yy = max(0, min(height - 1, ay + dy))
                    samples.append(owner[yy * width + xx])
                    covered += coverage[yy * width + xx] > 0
            counts = collections.Counter(samples)
            cx = max(0, min(width - 1, ax))
            cy = max(0, min(height - 1, ay))
            center = owner[cy * width + cx]
            alternates = [value for value, count in counts.items() if value not in (0, center) and count >= 2]
            failures.append({
                "criterion": criterion.get("name"),
                "world": world,
                "analysis": [ax, ay],
                "owner_path": item.get("owner_path", ""),
                "center_owner": center,
                "owner_counts": dict(counts),
                "covered_neighbors": covered,
                "alternate_candidates": alternates,
                "fallback_candidate": bool(alternates) and covered == 9,
            })

    scale_failures = [item for item in failures if item["criterion"] in ("local_scale_median", "local_scale_pixel_ratio")]
    boundary_failures = [item for item in failures if item["criterion"] == "boundary_stroke_alignment"]
    owner_failures = [item for item in failures if item["criterion"] == "owner_sharpness_coverage"]
    result = {
        "schema": 1,
        "method": "fengying_saved_owner_3x3_fallback_audit_v1",
        "production_impact": "test_only_no_pixel_write_no_owner_change",
        "diagnostics": str(args.diagnostics),
        "analysis": str(args.analysis),
        "report": str(report_path),
        "metadata": metadata,
        "origin_world": [args.origin_x, args.origin_y],
        "failures": failures,
        "summary": {
            "scale_failure_points": len(scale_failures),
            "scale_points_with_alternate_candidate": sum(item["fallback_candidate"] for item in scale_failures),
            "boundary_failure_points": len(boundary_failures),
            "boundary_points_with_alternate_candidate": sum(item["fallback_candidate"] for item in boundary_failures),
            "owner_coverage_failure_points": len(owner_failures),
            "owner_coverage_points_with_covered_neighbors": sum(item["covered_neighbors"] > 0 for item in owner_failures),
            "decision": "defer_automatic_fallback_until_source_resampling_and_quality_gate_evidence",
            "fallback_scope": "same 3x3 owner stencil, Coverage required, owner=0 excluded, no search expansion",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
