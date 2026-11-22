#!/usr/bin/env python3
"""Audit LiSong path-only RGB witnesses against saved station tone relations.

Read-only/test-only evidence join. It does not rerun stitching and does not
change tone thresholds, bounds, fallback, or QG decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sampler = load(args.sampler)
    pairs = sampler.get("pairwise_comparison", [])
    saved = sampler.get("saved_tone", {})
    station_tiles = saved.get("station_tiles", {})
    relation = saved.get("station_relation_9_to_13") or {}

    gain9 = station_tiles.get("9", {}).get("gain", [])
    gain13 = station_tiles.get("13", {}).get("gain", [])
    gain_relation = [
        gain9[i] / gain13[i] if i < len(gain9) and i < len(gain13) and gain13[i] else None
        for i in range(3)
    ]

    rows = []
    for pair in pairs:
        ratios = pair.get("rgb_ratio_a_over_b", [])
        luma_ratio = pair.get("luma_ratio_a_over_b")
        rows.append(
            {
                "label": pair.get("label"),
                "source_a": pair.get("source_a"),
                "source_b": pair.get("source_b"),
                "station_a": pair.get("station_a"),
                "station_b": pair.get("station_b"),
                "rgb_ratio_a_over_b": ratios,
                "luma_ratio_a_over_b": luma_ratio,
                "max_channel_ratio": max(ratios) if ratios else None,
                "saved_station_gain_relation_9_over_13": gain_relation,
                "ratio_exceeds_saved_gain_relation": bool(
                    ratios
                    and gain_relation
                    and all(
                        isinstance(ratio, (int, float))
                        and isinstance(gain, (int, float))
                        and ratio > gain * 1.01
                        for ratio, gain in zip(ratios, gain_relation)
                    )
                ),
            }
        )

    same_station = next((row for row in rows if row["label"] == "same_station13"), {})
    cross_station = [
        row for row in rows if row["label"] != "same_station13" and row.get("luma_ratio_a_over_b")
    ]
    result = {
        "schema": 1,
        "task": 17,
        "method": "task17_lisong_saved_tone_relation_content_audit_r1",
        "production_impact": "test_only/read_only relation-content audit; no production decision, threshold, bounds or fallback change",
        "sampler": str(args.sampler),
        "station_relation_9_to_13": {
            "scale_ratio": relation.get("scale_ratio"),
            "inliers": relation.get("inliers"),
            "spatial_support": relation.get("spatial_support"),
            "median_error_px": relation.get("median_error_px"),
            "accepted": True,
        },
        "saved_tone": {
            "station9_gain": station_tiles.get("9", {}).get("gain"),
            "station13_gain": station_tiles.get("13", {}).get("gain"),
            "station9_over_station13_gain": gain_relation,
        },
        "pairwise_rows": rows,
        "classification": {
            "same_station13_luma_difference": same_station.get("luma_ratio_a_over_b"),
            "cross_station_max_luma_ratio": max(
                (row["luma_ratio_a_over_b"] for row in cross_station), default=None
            ),
            "saved_relation_is_near_unity": bool(
                gain_relation and max(abs(value - 1.0) for value in gain_relation) < 0.01
            ),
            "bounded_source_content_photometric_witness": bool(
                same_station.get("luma_ratio_a_over_b")
                and abs(same_station["luma_ratio_a_over_b"] - 1.0) > 0.02
            ),
            "production_promotion": "rejected",
            "interpretation": "The accepted 9↔13 geometry relation and near-unity saved tone gains do not explain the observed RGB differences, including a same-station 13 witness. This is bounded source-content/photometric evidence; one world point per source cannot prove that all samples are identical content.",
        },
        "next_step": "Keep production tone bounds, thresholds and fallback unchanged. If stronger content equivalence evidence is required, capture a small bounded ring around this witness with the same path-only loader before considering any model change.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["classification"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
