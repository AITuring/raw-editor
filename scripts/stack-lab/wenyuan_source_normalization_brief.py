#!/usr/bin/env python3
"""Read-only source-normalization brief for active Wenyuan tone witnesses.

Uses active sampled luminance values to estimate the minimum source-specific
normalization needed to bring each pair into the production gain/offset box.
No production bounds or source pixels are changed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PRODUCTION_GAIN_MIN = 0.8
PRODUCTION_GAIN_MAX = 1.25
PRODUCTION_OFFSET_MAX = 0.02
TARGET_RATIO = 1.25
TARGET_PAIRS = ("36↔37", "37↔38", "36↔38")


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def pair_brief(row: dict[str, Any]) -> dict[str, Any]:
    left = row.get("active_left") or {}
    right = row.get("active_right") or {}
    left_luma = float(left.get("sample_luminance"))
    right_luma = float(right.get("sample_luminance"))
    brighter = max(left_luma, right_luma)
    darker = min(left_luma, right_luma)
    ratio = brighter / max(darker, 1e-12)

    # One-sided gain needed to bring the ratio to the 1.25 production target.
    bright_gain_required = TARGET_RATIO / ratio
    dark_gain_required = ratio / TARGET_RATIO
    symmetric_bright_gain = math.sqrt(bright_gain_required)
    symmetric_dark_gain = 1.0 / symmetric_bright_gain

    # Best ratio available when both sides use the favorable production bounds.
    bounded_bright = max(0.0, brighter * PRODUCTION_GAIN_MIN - PRODUCTION_OFFSET_MAX)
    bounded_dark = min(1.0, darker * PRODUCTION_GAIN_MAX + PRODUCTION_OFFSET_MAX)
    bounded_ratio = bounded_bright / max(bounded_dark, 1e-12)

    # Extra positive offset needed on the dark source after favorable bounded
    # correction, or extra negative offset needed on the bright source.
    dark_offset_required = max(
        0.0,
        (brighter * PRODUCTION_GAIN_MIN - PRODUCTION_OFFSET_MAX) / TARGET_RATIO
        - darker * PRODUCTION_GAIN_MAX,
    )
    bright_offset_required = max(
        0.0,
        brighter * PRODUCTION_GAIN_MIN
        - (darker * PRODUCTION_GAIN_MAX + PRODUCTION_OFFSET_MAX) * TARGET_RATIO,
    )
    return {
        "active_luminance": {"left": left_luma, "right": right_luma},
        "active_luminance_ratio": ratio,
        "target_ratio": TARGET_RATIO,
        "required_one_sided_gain": {
            "bright_source_gain": bright_gain_required,
            "dark_source_gain": dark_gain_required,
            "symmetric_bright_gain": symmetric_bright_gain,
            "symmetric_dark_gain": symmetric_dark_gain,
        },
        "production_box": {
            "gain_min": PRODUCTION_GAIN_MIN,
            "gain_max": PRODUCTION_GAIN_MAX,
            "offset_abs_max": PRODUCTION_OFFSET_MAX,
            "best_bounded_ratio_with_gain_offset": bounded_ratio,
        },
        "extra_offset_beyond_box": {
            "dark_source_positive_offset": dark_offset_required,
            "bright_source_negative_offset": bright_offset_required,
        },
        "sources": {
            "left": left.get("source_path"),
            "right": right.get("source_path"),
            "sampling_observed": bool(left.get("source_sampling_observed")) and bool(right.get("source_sampling_observed")),
            "fallback": bool(left.get("used_fallback")) or bool(right.get("used_fallback")),
        },
        "judgement": {
            "bounded_source_ab_discriminating": bounded_ratio <= TARGET_RATIO,
            "source_specific_ab_worthwhile": bounded_ratio <= TARGET_RATIO,
            "reason": "production gain/offset box cannot reach the target ratio; source-specific A/B would need an unbounded calibration variable"
            if bounded_ratio > TARGET_RATIO
            else "bounded source-specific A/B could reach the target ratio and may be discriminating",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-mismatch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = load(args.active_mismatch)
    pairs = source.get("pairs", {})
    result = {
        "schema": 1,
        "task": 14,
        "method": "wenyuan_active_source_normalization_brief_v1",
        "production_impact": "read-only normalization calculation; no source pixels, production bounds, or tone thresholds changed",
        "input": str(args.active_mismatch),
        "production_box": {
            "gain": [PRODUCTION_GAIN_MIN, PRODUCTION_GAIN_MAX],
            "offset_abs_max": PRODUCTION_OFFSET_MAX,
            "target_pair_ratio": TARGET_RATIO,
        },
        "pairs": {pair: pair_brief(pairs[pair]) for pair in TARGET_PAIRS if pair in pairs},
        "decision": {
            "promotion": "rejected",
            "next_step": "do not run a bounded source-specific A/B: the production box cannot normalize the active pair ratios; use explicit source calibration or relation-model evidence if a test-only experiment is required",
        },
        "limitations": [
            "luminance ratio is a diagnostic scalar, not a color-managed QG metric",
            "bounded ratio uses favorable independent gain/offset extrema and is therefore optimistic",
            "no full 126-source run or formal QG verdict was performed",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value["judgement"] for key, value in result["pairs"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
