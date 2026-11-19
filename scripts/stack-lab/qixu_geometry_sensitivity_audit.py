#!/usr/bin/env python3
"""Read-only source-to-world scale sensitivity audit for Qixu failures."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    report = json.loads(Path(analysis["report"]).read_text())
    alignment = json.loads(args.alignment.read_text())
    by_name = {Path(item["filename"]).name: item for item in alignment.get("sources", [])}
    global_h = alignment.get("global_homographies", {})
    rows = []
    for criterion in (report.get("quality_gate") or {}).get("criteria", []):
        if criterion.get("name") not in ("local_scale_median", "local_scale_pixel_ratio"):
            continue
        for item in criterion.get("failed", []):
            measured = item.get("measured")
            path = item.get("owner_path", "")
            source = by_name.get(Path(path).name)
            rows.append(
                {
                    "criterion": criterion["name"],
                    "world": item.get("world"),
                    "measured": measured,
                    "owner_path": path,
                    "source_id": source.get("id") if source else None,
                    "source_to_world": global_h.get(str(source.get("id"))) if source else None,
                    "required_isotropic_scale_to_0_95": (
                        0.95 / measured
                        if isinstance(measured, (int, float)) and measured > 0 and measured < 0.95
                        else None
                    ),
                }
            )
    required = [
        row["required_isotropic_scale_to_0_95"]
        for row in rows
        if row["required_isotropic_scale_to_0_95"] is not None
    ]
    result = {
        "schema": 1,
        "method": "qixu_failure_geometry_sensitivity_audit_v1",
        "production_impact": "test_only_read_only_source_to_world_sensitivity_no_pose_write_no_threshold_change",
        "analysis": str(args.analysis),
        "alignment": str(args.alignment),
        "failures": rows,
        "summary": {
            "failure_points": len(rows),
            "points_below_0_95": len(required),
            "minimum_required_scale_factor": min(required, default=None),
            "maximum_required_scale_factor": max(required, default=None),
            "decision": "sensitivity_only; required factors are evidence for a geometry hypothesis, not an instruction to alter source poses",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
