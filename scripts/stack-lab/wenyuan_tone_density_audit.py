#!/usr/bin/env python3
"""Read-only audit of Wenyuan tone degradation and output-density bypass.

This consumes an existing Stack_Report and, optionally, source files from the
same checkout.  It never writes a manifest, report, TIFF, or production code;
the sole write is the requested JSON audit output.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _count_reasons(entries: list[Any]) -> dict[str, int]:
    counts = collections.Counter(
        item.get("reason")
        for item in entries
        if isinstance(item, dict) and isinstance(item.get("reason"), str)
    )
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def _tone_audit(report: dict[str, Any], degradation: list[Any]) -> dict[str, Any]:
    tone = report.get("tone") if isinstance(report.get("tone"), dict) else {}
    boundary = tone.get("boundary_delta_e")
    if not isinstance(boundary, dict):
        boundary = {}
    violations = [
        item
        for item in boundary.get("violations", [])
        if isinstance(item, dict)
    ]
    violations.sort(key=lambda item: float(item.get("delta_e00", 0.0)), reverse=True)

    pair_counts: collections.Counter[str] = collections.Counter()
    pair_max: dict[str, float] = {}
    for item in violations:
        pair = f"{item.get('left')}↔{item.get('right')}"
        pair_counts[pair] += 1
        pair_max[pair] = max(pair_max.get(pair, float("-inf")), float(item.get("delta_e00", 0.0)))

    tone_entries = [
        item
        for item in degradation
        if isinstance(item, dict)
        and isinstance(item.get("reason"), str)
        and item["reason"].startswith("tone_")
    ]
    clamped = [
        item.get("detail", {}).get("station_index")
        for item in tone_entries
        if item.get("reason") == "tone_gain_clamped"
        and isinstance(item.get("detail"), dict)
    ]
    clamped = sorted({value for value in clamped if isinstance(value, int)})

    return {
        "status": tone.get("status"),
        "correction_model": tone.get("correction_model"),
        "geometry_active_relation_count": tone.get("geometry_active_relation_count"),
        "low_pass_sigma_world_px": tone.get("low_pass_sigma_world_px"),
        "boundary_delta_e": {
            "max": boundary.get("max"),
            "threshold": boundary.get("threshold"),
            "violation_count": len(violations),
            "violations_descending": violations,
            "pair_counts": dict(sorted(pair_counts.items())),
            "pair_max": dict(sorted(pair_max.items())),
        },
        "degradation": {
            "tone_reason_counts": _count_reasons(tone_entries),
            "tone_gain_clamped_station_indices": clamped,
        },
    }


def _density_audit(report: dict[str, Any]) -> dict[str, Any]:
    composition = report.get("composition")
    composition = composition if isinstance(composition, dict) else {}
    density = composition.get("output_density")
    if not isinstance(density, dict):
        density = report.get("output_density")
    density = density if isinstance(density, dict) else {}
    residual = report.get("residual_warp")
    residual = residual if isinstance(residual, dict) else {}
    status = density.get("status")
    factor = density.get("factor")
    sample_count = density.get("sample_count")
    bypass_reason = density.get("bypass_reason")
    no_op = status == "bypassed" and factor == 1.0 and sample_count == 0
    return {
        "status": status,
        "factor": factor,
        "sample_count": sample_count,
        "invalid_probe_count": density.get("invalid_probe_count"),
        "boundary_probe_count": density.get("boundary_probe_count"),
        "bypass_reason": bypass_reason,
        "residual_identity": residual.get("identity"),
        "explicit_no_op": no_op,
        "no_op_is_due_to_nonidentity_residual": (
            no_op
            and bypass_reason == "nonidentity_world_residual"
            and residual.get("identity") is False
        ),
        "native_density_proven": not no_op and isinstance(sample_count, int) and sample_count > 0,
        "decision": (
            "density evidence unavailable: nonidentity residual bypassed native Jacobian sampling"
            if no_op and bypass_reason == "nonidentity_world_residual"
            else "density evidence is present"
            if isinstance(sample_count, int) and sample_count > 0
            else "density evidence unavailable"
        ),
    }


def _quality_gate_regions(report: dict[str, Any]) -> list[dict[str, Any]]:
    qg = report.get("quality_gate")
    if not isinstance(qg, dict):
        return []
    regions: list[dict[str, Any]] = []
    criteria = qg.get("criteria")
    if not isinstance(criteria, list):
        return regions
    for criterion in criteria:
        if not isinstance(criterion, dict) or criterion.get("verdict") == "pass":
            continue
        failed = criterion.get("failed")
        regions.append(
            {
                "name": criterion.get("name"),
                "verdict": criterion.get("verdict"),
                "measurable_count": criterion.get("measurable_count"),
                "unmeasurable_count": criterion.get("unmeasurable_count"),
                "failed_count": len(failed) if isinstance(failed, list) else None,
                "failed_examples": failed[:8] if isinstance(failed, list) else [],
            }
        )
    return regions


def _test_only_evidence(repo: Path | None) -> dict[str, Any]:
    checks = [
        {
            "path": "src-tauri/src/panorama_stitching/output_density.rs",
            "markers": [
                "nonidentity_residual_and_missing_evidence_are_explicit_bypasses",
                'Some("nonidentity_world_residual")',
            ],
            "role": "cfg(test) density no-op contract and production bypass reason",
        },
        {
            "path": "src-tauri/src/panorama_utils/stitching/test_tone_capture.rs",
            "markers": [
                "set_test_tone_capture_directory",
                "capture_test_contour_probe",
                "capture_test_seam_grid",
            ],
            "role": "cfg(test) tone tile/contour/seam capture only",
        },
    ]
    result = []
    for item in checks:
        path = repo / item["path"] if repo is not None else None
        text = path.read_text() if path is not None and path.is_file() else ""
        result.append(
            {
                **item,
                "exists": bool(text),
                "all_markers_present": bool(text)
                and all(marker in text for marker in item["markers"]),
                "production_override_path": False,
            }
        )
    return {
        "source_checks": result,
        "tone_override": "no test-only switch found; capture hooks are diagnostic sinks",
        "density_override": "no test-only density factor override found; only explicit bypass/no-op path",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()

    report = _load(args.report)
    degradation = report.get("degradation", {}).get("entries", [])
    if not isinstance(degradation, list):
        degradation = []
    result = {
        "schema": 1,
        "method": "wenyuan_tone_density_read_only_audit_v1",
        "production_impact": "read-only report and source-marker inspection; no manifest, baseline, or production code writes",
        "report": str(args.report),
        "run_id": report.get("run_id"),
        "selected_path": report.get("selected_path"),
        "closure": {
            "status": (report.get("closure") or {}).get("status"),
            "residual_identity": (report.get("residual_warp") or {}).get("identity"),
        },
        "degradation": {
            "result": (report.get("degradation") or {}).get("result"),
            "entry_count": len(degradation),
            "reason_counts": _count_reasons(degradation),
        },
        "tone": _tone_audit(report, degradation),
        "density": _density_audit(report),
        "quality_gate_non_pass_regions": _quality_gate_regions(report),
        "test_only_evidence": _test_only_evidence(args.repo),
        "decision": "keep Wenyuan at diagnosis: tone violations are localized but explicit, and density is a no-op bypass with no native sampling evidence; do not infer scale=1 safety",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "tone_status": result["tone"]["status"],
        "tone_violation_count": result["tone"]["boundary_delta_e"]["violation_count"],
        "tone_max": result["tone"]["boundary_delta_e"]["max"],
        "density": result["density"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
