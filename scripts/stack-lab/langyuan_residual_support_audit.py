#!/usr/bin/env python3
"""Read-only audit of Langyuan QG failures against residual-warp support."""
import argparse
import json
from pathlib import Path


def inside(point, region):
    world = region["world"]
    return (
        world["left"] <= point["x"] <= world["left"] + world["width"]
        and world["top"] <= point["y"] <= world["top"] + world["height"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    report = json.loads(Path(analysis["report"]).read_text())
    regions = report.get("residual_warp", {}).get("regions", [])
    failures = []
    for criterion in (report.get("quality_gate") or {}).get("criteria", []):
        if criterion.get("name") not in (
            "local_scale_median",
            "local_scale_pixel_ratio",
            "boundary_stroke_alignment",
        ):
            continue
        for item in criterion.get("failed", []):
            point = item.get("world")
            if not point:
                continue
            failures.append(
                {
                    "criterion": criterion["name"],
                    "world": point,
                    "measured": item.get("measured"),
                    "owner_path": item.get("owner_path", ""),
                    "residual_region_indexes": [
                        index for index, region in enumerate(regions) if inside(point, region)
                    ],
                }
            )
    outside = [item for item in failures if not item["residual_region_indexes"]]
    result = {
        "schema": 1,
        "method": "langyuan_failure_residual_support_audit_v1",
        "production_impact": "test_only_read_only_report_intersection_no_pixel_write_no_model_change",
        "analysis": str(args.analysis),
        "report": analysis["report"],
        "residual_warp": report.get("residual_warp", {}),
        "failures": failures,
        "summary": {
            "failure_points": len(failures),
            "failure_points_inside_residual_region": len(failures) - len(outside),
            "failure_points_outside_residual_region": len(outside),
            "local_scale_failure_points_outside": sum(
                item["criterion"] in ("local_scale_median", "local_scale_pixel_ratio")
                and not item["residual_region_indexes"]
                for item in failures
            ),
            "boundary_failure_points_outside": sum(
                item["criterion"] == "boundary_stroke_alignment"
                and not item["residual_region_indexes"]
                for item in failures
            ),
            "decision": "residual_identity_or_disable_experiment_cannot_explain_current_failures_when_all_failed_points_are_outside_recorded_support",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
