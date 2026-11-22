#!/usr/bin/env python3
"""Summarize a bounded path-only RGB ring around the LiSong witness."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def luma(rgb: list[float]) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ring", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--stack-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ring = load(args.ring)
    summary = load(args.summary)
    report = load(args.stack_report)
    grid: dict[tuple[str, str], dict[str, list[float]]] = {}
    pattern = re.compile(r"^(dsc\d+_station\d+)_dx([+-]\d+)_dy([+-]\d+)$")
    for row in ring.get("records", []):
        match = pattern.match(row.get("label", ""))
        if not match:
            continue
        key = (match.group(2), match.group(3))
        grid.setdefault(key, {})[match.group(1)] = row["rgb_f32"]

    pair_specs = [
        ("station9_vs_station13", "dsc0910_station9", "dsc0912_station13"),
        ("station9_vs_station13_reference", "dsc0910_station9", "dsc0905_station13"),
        ("same_station13", "dsc0912_station13", "dsc0905_station13"),
    ]
    pair_rows = []
    for label, left, right in pair_specs:
        rows = []
        for offset, values in sorted(grid.items()):
            if left not in values or right not in values:
                continue
            left_rgb = values[left]
            right_rgb = values[right]
            rows.append(
                {
                    "offset": {"dx": int(offset[0]), "dy": int(offset[1])},
                    "rgb_ratio_left_over_right": [
                        left_rgb[i] / right_rgb[i] if right_rgb[i] else None for i in range(3)
                    ],
                    "luma_ratio_left_over_right": luma(left_rgb) / luma(right_rgb),
                }
            )
        luma_values = [row["luma_ratio_left_over_right"] for row in rows]
        channel_stats = []
        for channel in range(3):
            values = [row["rgb_ratio_left_over_right"][channel] for row in rows]
            channel_stats.append(
                {
                    "min": min(values) if values else None,
                    "median": statistics.median(values) if values else None,
                    "max": max(values) if values else None,
                    "population_stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
                }
            )
        pair_rows.append(
            {
                "label": label,
                "sample_count": len(rows),
                "channel_stats": channel_stats,
                "luma_stats": {
                    "min": min(luma_values) if luma_values else None,
                    "median": statistics.median(luma_values) if luma_values else None,
                    "max": max(luma_values) if luma_values else None,
                    "population_stddev": statistics.pstdev(luma_values) if len(luma_values) > 1 else 0.0,
                },
                "rows": rows,
            }
        )

    same_station = next(row for row in pair_rows if row["label"] == "same_station13")
    result = {
        "schema": 1,
        "task": 17,
        "method": "task17_lisong_source_rgb_ring_audit_r1",
        "production_impact": "test_only bounded path-only ring audit; no production decision, threshold, bounds or fallback change",
        "ring": str(args.ring),
        "summary": str(args.summary),
        "stack_report": str(args.stack_report),
        "sample_count": len(ring.get("records", [])),
        "pairwise": pair_rows,
        "native_result": {
            "quality_gate": summary["summary"].get("qualityGate"),
            "outcome": summary["summary"].get("outcome"),
            "tone_status": summary["summary"].get("toneStatus"),
            "tone_boundary_delta_e": report.get("tone", {}).get("boundary_delta_e"),
            "density": report.get("composition", {}).get("output_density"),
            "memory": report.get("resources"),
        },
        "classification": {
            "same_station13_luma_median": same_station["luma_stats"]["median"],
            "same_station13_luma_range": [same_station["luma_stats"]["min"], same_station["luma_stats"]["max"]],
            "same_station13_luma_spread": same_station["luma_stats"]["max"] - same_station["luma_stats"]["min"],
            "localized_or_content_sensitive": same_station["luma_stats"]["median"] is not None and abs(same_station["luma_stats"]["median"] - 1.0) < 0.02 and same_station["luma_stats"]["max"] - same_station["luma_stats"]["min"] > 0.05,
            "production_promotion": "rejected",
            "interpretation": "The center witness can be strongly different, but the surrounding same-station ring has a near-unity median with a broad spread. This is more consistent with localized content/boundary sensitivity than a stable uniform tone gain; the ring is diagnostic evidence only.",
        },
        "next_step": "Keep tone bounds, thresholds and fallback unchanged. Treat the witness as localized source-content evidence; move to the next material or collect a larger content-matched patch only if a production decision requires it.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["classification"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
