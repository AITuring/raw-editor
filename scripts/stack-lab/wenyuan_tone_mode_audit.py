#!/usr/bin/env python3
"""Compare Wenyuan tone-harmonizer mode and clamp evidence across scales."""

import argparse
import json
import statistics
from pathlib import Path


def load(path: Path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def summarize(label, path):
    report = load(path)
    tone = report.get("tone") or {}
    tiles = [item for item in tone.get("tiles", []) if isinstance(item, dict)]
    strengths = [float(item["source_fidelity_strength"]) for item in tiles if isinstance(item.get("source_fidelity_strength"), (int, float))]
    clamped = [item.get("station_index") for item in tiles if item.get("clamped") is True]
    boundary = tone.get("boundary_delta_e") or {}
    density = (report.get("composition") or {}).get("output_density") or {}
    return {
        "label": label,
        "report": str(path),
        "correction_model": tone.get("correction_model"),
        "tile_count": len(tiles),
        "clamped_tile_count": len(clamped),
        "clamped_station_indices": sorted(value for value in clamped if isinstance(value, int)),
        "source_fidelity_strength": {
            "min": min(strengths) if strengths else None,
            "median": statistics.median(strengths) if strengths else None,
            "max": max(strengths) if strengths else None,
        },
        "tone_status": tone.get("status"),
        "boundary_delta_e_max": boundary.get("max"),
        "boundary_violation_count": len(boundary.get("violations", [])),
        "density": {
            "status": density.get("status"),
            "sample_count": density.get("sample_count"),
            "bypass_reason": density.get("bypass_reason"),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale025", type=Path, required=True)
    parser.add_argument("--scale1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scale025 = summarize("scale025", args.scale025)
    scale1 = summarize("scale1", args.scale1)
    mode_changed = scale025["correction_model"] != scale1["correction_model"]
    result = {
        "schema": 1,
        "method": "wenyuan_tone_mode_read_only_audit_v1",
        "production_impact": "read-only report comparison; no tone fallback, threshold, or pixel change",
        "runs": [scale025, scale1],
        "comparison": {
            "correction_model_changed": mode_changed,
            "scale025_to_scale1": {
                "model": [scale025["correction_model"], scale1["correction_model"]],
                "boundary_max": [scale025["boundary_delta_e_max"], scale1["boundary_delta_e_max"]],
                "boundary_violations": [scale025["boundary_violation_count"], scale1["boundary_violation_count"]],
                "clamped_tiles": [scale025["clamped_tile_count"], scale1["clamped_tile_count"]],
                "fidelity_min": [scale025["source_fidelity_strength"]["min"], scale1["source_fidelity_strength"]["min"]],
            },
        },
        "decision": (
            "scale1_tone_mode_changed_to_station_affine_with_widespread_clamping; investigate native-source retention or an explicitly bounded test-only tone-mode hypothesis; do not loosen tone threshold or add production fallback"
            if mode_changed and scale1["clamped_tile_count"] > scale025["clamped_tile_count"]
            else "tone mode did not change; keep investigating boundary evidence"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
