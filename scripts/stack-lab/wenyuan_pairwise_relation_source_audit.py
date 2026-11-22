#!/usr/bin/env python3
"""Read-only pairwise relation/source-Jacobian isolation for Wenyuan r10 fields.

Joins the active LayeredVirtualTile witness trace, tone-final-world relation
metrics, and saved owner-fix replays. It does not modify production bounds or
rerun stitching.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

TARGET_PAIRS = ((36, 37), (37, 38), (36, 38))


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def owner_station_map(capture_dir: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    for path in sorted(capture_dir.glob("tone-owner-*.json")):
        value = load(path)
        result[int(value["owner"])] = int(value.get("station_label", value["station"]))
    return result


def replay_pair_summary(path: Path, capture_dir: Path) -> dict[str, dict[str, Any]]:
    tiles = [load(path) for path in sorted(capture_dir.glob("tone-owner-*.json"))]
    labels = [int(tile.get("station_label", tile["station"])) for tile in tiles]
    result: dict[str, dict[str, Any]] = {}
    for row in load(path).get("failed", []):
        if not isinstance(row, dict):
            continue
        left, right = row.get("first"), row.get("second")
        if not isinstance(left, int) or not isinstance(right, int) or left >= len(labels) or right >= len(labels):
            continue
        key = f"{min(labels[left], labels[right])}↔{max(labels[left], labels[right])}"
        item = result.setdefault(key, {"failed_window_count": 0, "maximum_delta": 0.0})
        item["failed_window_count"] += 1
        item["maximum_delta"] = max(item["maximum_delta"], float(row.get("delta", 0.0)))
    return result


def relation_summary(tone_final_world: dict[str, Any], owner_labels: dict[int, int]) -> dict[str, dict[str, Any]]:
    relation_out: dict[str, dict[str, Any]] = {}
    owner_to_station = {
        int(tile["owner_id"]): int(tile["station"]) for tile in tone_final_world.get("tiles", [])
    }
    for relation in tone_final_world.get("relations", []):
        owners = relation.get("owner_ids") or []
        if len(owners) != 2:
            continue
        stations = [owner_labels.get(int(owners[0]), owner_to_station.get(int(owners[0]))), owner_labels.get(int(owners[1]), owner_to_station.get(int(owners[1])))]
        if any(station is None for station in stations):
            continue
        key = f"{min(stations)}↔{max(stations)}"
        relation_out[key] = {
            "owner_ids": owners,
            "station_indices": relation.get("stations"),
            "pre_delta_e00": (relation.get("pre") or {}).get("delta_e00"),
            "post_delta_e00": (relation.get("post") or {}).get("delta_e00"),
            "retained_samples": (relation.get("pair_samples") or {}).get("retained"),
            "minimum_samples": (relation.get("pair_samples") or {}).get("minimum"),
            "holdout_enabled": (relation.get("holdout") or {}).get("enabled"),
            "solver_log_residual": relation.get("solver_log_residual"),
        }
    return relation_out


def active_pair_summary(mismatch: dict[str, Any], trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    trace_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace.get("rows", []):
        label = row.get("label")
        if isinstance(label, str):
            pair = label.removeprefix("boundary-").replace("-", "↔")
            trace_by_pair[pair].append(row)
    out: dict[str, dict[str, Any]] = {}
    for key, value in mismatch.get("pairs", {}).items():
        if key not in {"36↔37", "37↔38", "36↔38"}:
            continue
        rows = trace_by_pair.get(key, [])
        diagnostics = [diagnostic for row in rows for diagnostic in row.get("source_diagnostics", [])]
        out[key] = {
            "active_luminance_ratio": value.get("active_luminance_ratio"),
            "owner_selection_gap": value.get("active_owner_selection_gap"),
            "left_source": (value.get("active_left") or {}).get("source_path"),
            "right_source": (value.get("active_right") or {}).get("source_path"),
            "left_sampling_observed": (value.get("active_left") or {}).get("source_sampling_observed"),
            "right_sampling_observed": (value.get("active_right") or {}).get("source_sampling_observed"),
            "active_rows": sum(row.get("actual_status") == "mask_written" for row in rows),
            "support_diagnostic_rows": len(diagnostics),
            "all_source_footprints_supported": bool(diagnostics) and all(item.get("source_footprint_supported") is True for item in diagnostics),
            "all_global_source_supported": bool(diagnostics) and all(item.get("global_source_supported") is True for item in diagnostics),
            "fallback_rows": sum(item.get("used_fallback") is True for item in diagnostics),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--tone-final-world", type=Path, required=True)
    parser.add_argument("--active-mismatch", type=Path, required=True)
    parser.add_argument("--active-trace", type=Path, required=True)
    parser.add_argument("--replay", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    owner_labels = owner_station_map(args.capture_dir)
    final_world = load(args.tone_final_world)
    relation = relation_summary(final_world, owner_labels)
    replay = {label: replay_pair_summary(Path(path), args.capture_dir) for label, path in args.replay}
    active = active_pair_summary(load(args.active_mismatch), load(args.active_trace))

    pairs: dict[str, Any] = {}
    for left, right in TARGET_PAIRS:
        key = f"{left}↔{right}"
        relation_row = relation.get(key, {})
        active_row = active.get(key, {})
        row = {
            "relation_fit": relation_row,
            "active_source_sampling": active_row,
            "replays": {label: rows.get(key, {"failed_window_count": 0, "maximum_delta": 0.0}) for label, rows in replay.items()},
        }
        post = relation_row.get("post_delta_e00")
        ratio = active_row.get("active_luminance_ratio")
        row["classification"] = {
            "relation_in_support_fit": isinstance(post, (int, float)) and post < 1.5,
            "direct_low_luma_mismatch": isinstance(ratio, (int, float)) and ratio > 100.0,
            "active_source_supported": active_row.get("all_source_footprints_supported") is True and active_row.get("all_global_source_supported") is True,
            "owner_selection_gap": active_row.get("owner_selection_gap") is True,
            "interpretation": "relation fits in-support but active samples show direct photometric mismatch" if isinstance(post, (int, float)) and post < 1.5 and isinstance(ratio, (int, float)) and ratio > 100.0 else "requires further pairwise evidence",
        }
        pairs[key] = row

    result = {
        "schema": 1,
        "diagnostic_only": True,
        "method": "wenyuan_pairwise_relation_source_audit_v1",
        "production_impact": "read-only pairwise relation/source-Jacobian isolation; no production bounds or thresholds changed",
        "inputs": {
            "capture_dir": str(args.capture_dir),
            "tone_final_world": str(args.tone_final_world),
            "active_mismatch": str(args.active_mismatch),
            "active_trace": str(args.active_trace),
            "replays": {label: path for label, path in args.replay},
        },
        "pairs": pairs,
        "decision": {
            "promotion": "rejected",
            "next_step": "treat 36↔37 and 37↔38 as active direct photometric mismatch under supported source sampling; keep bounds unchanged and isolate pairwise source/owner calibration before any production change",
        },
        "limitations": [
            "active trace is diagnostic evidence, not a formal QG verdict",
            "tone-final-world post deltas summarize relation support samples and do not prove boundary witnesses",
            "no full 126-source run was performed",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value["classification"] for key, value in pairs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
