#!/usr/bin/env python3
"""Build a bounded, read-only Wenyuan tone-boundary A/B summary.

The full-scale pair compares the pre-adaptive station-affine report with the
adaptive local-tone report already on disk.  The saved native replay then
isolates one test-only variable (local affine bound scale) without starting a
new full-material run.  This script writes only the requested JSON summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def report_summary(path: Path, analysis_path: Path) -> dict[str, Any]:
    report = load(path)
    analysis = load(analysis_path)
    tone = report.get("tone") or {}
    boundary = tone.get("boundary_delta_e") or {}
    violations = [
        item for item in boundary.get("violations", []) if isinstance(item, dict)
    ]
    pairs: dict[str, dict[str, Any]] = {}
    for item in violations:
        left, right = item.get("left"), item.get("right")
        key = f"{min(left, right)}↔{max(left, right)}" if isinstance(left, int) and isinstance(right, int) else "unknown"
        row = pairs.setdefault(key, {"count": 0, "max_delta_e00": 0.0})
        row["count"] += 1
        row["max_delta_e00"] = max(row["max_delta_e00"], float(item.get("delta_e00", 0.0)))
    clamped = sorted(
        int(tile["station_index"])
        for tile in tone.get("tiles", [])
        if isinstance(tile, dict) and tile.get("clamped") is True and isinstance(tile.get("station_index"), int)
    )
    composition = report.get("composition") or {}
    density = composition.get("output_density") or {}
    return {
        "report": str(path),
        "analysis": str(analysis_path),
        "status": analysis.get("status"),
        "tone_model": tone.get("correction_model"),
        "tone_status": tone.get("status"),
        "native_fitting_relation_count": len(tone.get("native_fitting_relations", [])),
        "boundary": {
            "threshold": boundary.get("threshold"),
            "max_delta_e00": boundary.get("max"),
            "violation_count": len(violations),
            "pair_max": dict(sorted(pairs.items(), key=lambda item: (-item[1]["max_delta_e00"], item[0]))),
        },
        "clamped_station_indices": clamped,
        "scope": {
            "source_count": (report.get("input") or {}).get("source_count"),
            "station_count": (report.get("grouping") or {}).get("station_count"),
            "output_dimensions": [
                (report.get("output") or {}).get("width"),
                (report.get("output") or {}).get("height"),
            ],
            "closure": (report.get("closure") or {}).get("status"),
            "density_status": density.get("status"),
            "density_bypass_reason": density.get("bypass_reason"),
            "qg_verdict": (report.get("quality_gate") or {}).get("verdict"),
            "memory_gate_pass": analysis.get("memory_gate_pass", analysis.get("checks", {}).get("memory_gate_pass")),
            "report_source_digests": analysis.get("checks", {}).get("report_source_digests"),
        },
    }


def replay_summary(path: Path) -> dict[str, Any]:
    value = load(path)
    ranges = value.get("parameter_ranges") or {}
    return {
        "output": str(path),
        "diagnostic_only": value.get("diagnostic_only"),
        "bound_scale": value.get("test_only_bound_scale"),
        "maximum_delta_e00": value.get("maximum"),
        "failed_window_count": len(value.get("failed", [])),
        "fitted_relations": value.get("fitted_relations"),
        "iterations": value.get("iterations"),
        "parameter_ranges": ranges,
    }


def materially_different(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) > tolerance
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return any(materially_different(a, b, tolerance) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        return any(materially_different(left[key], right[key], tolerance) for key in left)
    return left != right


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--control-analysis", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--candidate-analysis", type=Path, required=True)
    parser.add_argument("--replay-control", type=Path, required=True)
    parser.add_argument("--replay-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = report_summary(args.control_report, args.control_analysis)
    candidate = report_summary(args.candidate_report, args.candidate_analysis)
    replay_control = replay_summary(args.replay_control)
    replay_candidate = replay_summary(args.replay_candidate)
    control_max = float(control["boundary"]["max_delta_e00"] or 0.0)
    candidate_max = float(candidate["boundary"]["max_delta_e00"] or 0.0)
    control_count = int(control["boundary"]["violation_count"])
    candidate_count = int(candidate["boundary"]["violation_count"])
    replay_changed = materially_different(
        replay_control["maximum_delta_e00"], replay_candidate["maximum_delta_e00"]
    ) or replay_control["failed_window_count"] != replay_candidate["failed_window_count"] or materially_different(
        replay_control["parameter_ranges"], replay_candidate["parameter_ranges"]
    )
    result = {
        "schema": 1,
        "method": "wenyuan_tone_boundary_bounded_ab_v1",
        "production_impact": "test_only_read_only_comparison; no threshold, fallback, geometry, density, or production bound change",
        "dataset": "wenyuan",
        "scale": 1.0,
        "variable": {
            "name": "local_tone_affine_bound_scale",
            "control": 1.0,
            "candidate": 1.5,
            "candidate_scope": "cfg(test) saved native replay only",
            "hypothesis": "the remaining boundary residual is limited by the local gain/offset box",
        },
        "full_scale_context": {
            "control": control,
            "candidate": candidate,
            "max_delta_e00_change": candidate_max - control_max,
            "max_delta_e00_relative_change": (candidate_max / control_max - 1.0) if control_max else None,
            "violation_count_change": candidate_count - control_count,
            "strict_single_variable": False,
            "interpretation": "the historical full reports show adaptive local fit materially reduced the prior station-affine residual, but their frozen commits differ; use this as context, not causal A/B proof",
        },
        "bounded_replay_ab": {
            "control": replay_control,
            "candidate": replay_candidate,
            "changed": replay_changed,
            "interpretation": "the Wenyuan-10 saved replay has no bound hit; widening the box is therefore non-discriminating on this control",
        },
        "decision": {
            "promotion": "rejected",
            "reason": "candidate full-scale tone boundary remains over threshold and the bounded replay control does not exercise the offending 34–39 station cluster",
            "next_step": "keep production bounds and threshold unchanged; obtain a test-only field export for the full 34–39 cluster or a synthetic bound-hit fixture before any new full run",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
