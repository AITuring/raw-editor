#!/usr/bin/env python3
"""Audit task-16 output-density/closure multiplication without running images."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src-tauri/src/panorama_stitching.rs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def constants() -> dict[str, float]:
    text = SOURCE.read_text()
    names = {
        "direct_pair_p95_limit_px": "CLOSURE_DIRECT_PAIR_P95_LIMIT_PX",
        "residual_median_limit_px": "CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX",
        "max_corner_correction_limit_px": "CLOSURE_MAX_CORNER_CORRECTION_PX",
    }
    result = {}
    for output_name, source_name in names.items():
        match = re.search(rf"{source_name}: f64 = ([0-9.]+)", text)
        if not match:
            raise RuntimeError(f"missing closure constant: {source_name}")
        result[output_name] = float(match.group(1))
    return result


def audit(path: Path, limits: dict[str, float]) -> dict:
    data = json.loads(path.read_text())
    density = data["output_density"]
    closure = data["closure"]
    direct = float(closure["max_direct_pair_p95_px"])
    median = float(closure["residual_median_px"])
    corner = float(closure["max_corner_correction_px"])
    factor = float(density["factor"])
    caps = {
        "direct_pair_p95": limits["direct_pair_p95_limit_px"] / direct,
        "residual_median": limits["residual_median_limit_px"] / median,
        "max_corner_correction": limits["max_corner_correction_limit_px"] / corner,
    }
    allowed = min(caps.values())
    return {
        "diagnostic": str(path),
        "dataset": data["dataset"],
        "binary": data["binary"],
        "inputs_sha256": data["inputs_sha256"],
        "output_density": {
            "status": density["status"],
            "factor": factor,
            "minimum_scale": density.get("minimum_scale"),
            "sample_count": density.get("sample_count"),
            "worst_station_index": density.get("worst_station_index"),
            "worst_source_id": density.get("worst_source_id"),
            "worst_station_point": density.get("worst_station_point"),
        },
        "base_closure_px": {
            "direct_pair_p95": direct,
            "residual_median": median,
            "max_corner_correction": corner,
        },
        "output_lattice_closure_px": {
            "direct_pair_p95": density["direct_pair_p95_output_px"],
            "residual_median": density["residual_median_output_px"],
            "max_corner_correction": density["max_corner_correction_output_px"],
        },
        "factor_caps_from_existing_limits": caps,
        "allowed_factor_upper_bound": allowed,
        "actual_factor_exceeds_allowed": factor > allowed,
        "decision": "density_factor_is_rejected_by_existing_closure_contract; do_not cap factor or relax thresholds in production",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("diagnostics", nargs="+", type=Path)
    args = parser.parse_args()
    limits = constants()
    result = {
        "schema": 1,
        "task": 16,
        "method": "task16_output_density_closure_contract_audit_v1",
        "production_impact": "test_only_read_only_formula_audit_no_pose_write_no_threshold_change",
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "closure_limits": limits,
        "runs": [audit(path, limits) for path in args.diagnostics],
        "next_step": "write a single-variable station-density brief before any 0.25 rerun; preserve the 3px gate and native-density evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "runs": len(result["runs"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
