#!/usr/bin/env python3
"""Audit a small, test-only Wenyuan ToneTile capture.

This is intentionally independent of the production runner.  It verifies the
15-source station subset and the owner JSON/native field pairs emitted by
``real_focus_stack_fixture_from_env``; a stopped preview run may legitimately
have no final owner plane or Stack_Report.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    subset = load(args.subset_manifest)
    capture = args.capture_dir
    fields = []
    missing_native = []
    invalid = []
    for path in sorted(capture.glob("tone-owner-*.json")):
        value = load(path)
        owner = value.get("owner")
        station = value.get("station")
        width = value.get("width")
        height = value.get("height")
        native = path.with_suffix(".native.f32")
        expected_native_bytes = width * height * 16 * 3 * 4 if isinstance(width, int) and isinstance(height, int) else None
        if not native.is_file():
            missing_native.append(path.name)
        if expected_native_bytes is None or native.is_file() and native.stat().st_size != expected_native_bytes:
            invalid.append({"file": path.name, "expected_native_bytes": expected_native_bytes,
                            "actual_native_bytes": native.stat().st_size if native.is_file() else None})
        fields.append({
            "file": path.name,
            "native_file": native.name,
            "run_local_station": station,
            "owner_id": owner,
            "dimensions": [width, height],
            "world_origin": value.get("world_origin"),
            "world_size": value.get("world_size"),
            "native_bytes": native.stat().st_size if native.is_file() else None,
        })
    run_local_stations = sorted({item["run_local_station"] for item in fields if isinstance(item["run_local_station"], int)})
    result = {
        "schema": 1,
        "diagnostic_only": True,
        "method": "wenyuan_tone_fields_capture_audit_v1",
        "subset_manifest": str(args.subset_manifest),
        "capture_dir": str(capture),
        "original_station_ids": subset.get("stations", []),
        "original_source_count": subset.get("source_count"),
        "original_sources_all_present": not bool(subset.get("missing_files")),
        "run_local_station_reindexed": True,
        "run_local_stations": run_local_stations,
        "fields": fields,
        "field_json_count": len(fields),
        "missing_native_files": missing_native,
        "invalid_fields": invalid,
        "owner_plane": {
            "path": str(capture / "tone-owners.filtered.png") if (capture / "tone-owners.filtered.png").is_file() else str(capture / "tone-owners.png"),
            "present": (capture / "tone-owners.filtered.png").is_file() or (capture / "tone-owners.png").is_file(),
        },
        "tone_final_world": {
            "path": str(capture / "tone-final-world.json"),
            "present": (capture / "tone-final-world.json").is_file(),
        },
        "filter_manifest": {
            "path": str(capture / "tone-fields-filter-manifest.json"),
            "present": (capture / "tone-fields-filter-manifest.json").is_file(),
        },
        "decision": (
            "field_pairs_complete_for_small_real_subset"
            if len(fields) >= 1 and not missing_native and not invalid and not subset.get("missing_files")
            else "incomplete_or_invalid_capture"
        ),
        "limitations": [
            "run-local station indices are reindexed 0..N-1; use the subset manifest to recover original 34–39 source membership",
            "capture was preview-only and stopped after field pairs; absence of owner plane or tone-final-world is not a production failure",
            "no full 126-source run or native acceptance claim",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"field_json_count": len(fields), "run_local_stations": run_local_stations,
                      "decision": result["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
