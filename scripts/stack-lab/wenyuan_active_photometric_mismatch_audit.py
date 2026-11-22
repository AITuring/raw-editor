#!/usr/bin/env python3
"""Join corrected active point-trace RGB with saved tone replay failures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def luminance(rgb: list[float] | None) -> float | None:
    if not rgb:
        return None
    return float(rgb[0]) * 0.2126 + float(rgb[1]) * 0.7152 + float(rgb[2]) * 0.0722


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace: dict[str, Any] = json.loads(args.trace.read_text())
    replay: dict[str, Any] = json.loads(args.replay.read_text())
    fields = [
        json.loads(path.read_text())
        for path in sorted(args.capture_dir.glob("tone-owner-*.json"))
    ]
    labels = {item.get("station"): item.get("station_label") for item in fields}
    active: dict[str, dict[int, dict[str, Any]]] = {}
    for entry in trace.get("stations", []):
        station = entry.get("station", {})
        label = labels.get(station.get("topology_station_id"))
        if label is None:
            continue
        for point in entry.get("points", []):
            request = point.get("request") or {}
            actual = point.get("actual") or {}
            resolution = actual.get("native_owner_resolution") or {}
            active.setdefault(request.get("label"), {})[label] = {
                "station_label": label,
                "status": actual.get("status"),
                "sample_rgb": actual.get("sample_rgb") or resolution.get("sample_rgb"),
                "sample_luminance": luminance(actual.get("sample_rgb") or resolution.get("sample_rgb")),
                "source_path": resolution.get("source_path"),
                "source_id": resolution.get("source_id"),
                "used_fallback": resolution.get("used_fallback"),
                "source_sampling_observed": bool(point.get("native_candidates", {}).get("source_diagnostics")),
            }
    replay_index_to_label = {
        index: item.get("station_label") for index, item in enumerate(fields)
    }
    pair_rows: dict[str, dict[str, Any]] = {}
    for row in replay.get("failed", []):
        if not isinstance(row, dict):
            continue
        left = replay_index_to_label.get(row.get("first"))
        right = replay_index_to_label.get(row.get("second"))
        if left is None or right is None:
            continue
        key = f"{min(left, right)}↔{max(left, right)}"
        if key not in pair_rows or float(row.get("delta", 0.0)) > float(pair_rows[key].get("saved_delta", 0.0)):
            pair_rows[key] = {
                "pair": key,
                "left_station": left,
                "right_station": right,
                "saved_delta": row.get("delta"),
                "saved_common_rgb": row.get("common_rgb"),
                "saved_first_parameters": row.get("first_parameters"),
                "saved_second_parameters": row.get("second_parameters"),
                "saved_samples": row.get("samples"),
                "active_left": active.get(next((label for label in active if label.endswith(f"{left}-{right}")), ""), {}).get(left),
                "active_right": active.get(next((label for label in active if label.endswith(f"{left}-{right}")), ""), {}).get(right),
            }
    # The labels in the input are stable and avoid relying on pair text parsing.
    for pair, row in pair_rows.items():
        label = f"boundary-{row['left_station']}-{row['right_station']}"
        row["witness_label"] = label
        row["active_left"] = active.get(label, {}).get(row["left_station"])
        row["active_right"] = active.get(label, {}).get(row["right_station"])
        left_luma = (row["active_left"] or {}).get("sample_luminance")
        right_luma = (row["active_right"] or {}).get("sample_luminance")
        row["active_luminance_ratio"] = (
            max(left_luma, right_luma) / min(left_luma, right_luma)
            if left_luma is not None and right_luma is not None and min(left_luma, right_luma) > 1e-9
            else None
        )
        row["active_owner_selection_gap"] = any(
            (row.get(side) or {}).get("status") not in ("mask_written", "native_sample_returned_none")
            for side in ("active_left", "active_right")
        )
    result = {
        "schema": 1,
        "diagnostic_only": True,
        "method": "wenyuan_active_photometric_mismatch_audit_v1",
        "trace": str(args.trace),
        "replay": str(args.replay),
        "capture_dir": str(args.capture_dir),
        "pairs": pair_rows,
        "interpretation": "active RGB/source selection is compared with the saved tone relation at the same witness labels; this is diagnostic evidence, not a QG verdict",
        "decision": {
            "promotion": "rejected",
            "next_step": "separate owner-selection gaps from photometric mismatch before any tone bound change"
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: {"saved_delta": value["saved_delta"], "active_luminance_ratio": value.get("active_luminance_ratio"), "active_owner_selection_gap": value.get("active_owner_selection_gap")} for key, value in pair_rows.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
