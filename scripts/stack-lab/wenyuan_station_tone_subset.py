#!/usr/bin/env python3
"""Select real Wenyuan station members for a test-only tone-field replay.

This is a read-only projection of an existing Stack_Report.  It never changes
the manifest or production runner; the emitted JSON is the auditable input
list for ``real_focus_stack_fixture_from_env``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stations", default="34,35,36,37,38,39")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stations = sorted({int(value) for value in args.stations.split(",") if value.strip()})
    grouping = load(args.report).get("grouping") or {}
    by_index = {
        int(item["index"]): item
        for item in grouping.get("stations", [])
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    selected = []
    members: list[str] = []
    for index in stations:
        item = by_index.get(index)
        if item is None:
            selected.append({"index": index, "status": "missing_station"})
            continue
        paths = [str(path) for path in item.get("members", []) if isinstance(path, str)]
        selected.append({
            "index": index,
            "status": "selected",
            "members": paths,
            "member_count": len(paths),
            "topology_status": item.get("topology_status"),
        })
        members.extend(paths)
    members = list(dict.fromkeys(members))
    files = [{"path": path, "exists": Path(path).is_file(),
              "bytes": Path(path).stat().st_size if Path(path).is_file() else None}
             for path in members]
    selected_station_labels = [item["index"] for item in selected if item["status"] == "selected"]
    result = {
        "schema": 1,
        "diagnostic_only": True,
        "method": "wenyuan_station_tone_subset_v1",
        "report": str(args.report),
        "stations": stations,
        "selected": selected,
        "files": files,
        "source_count": len(members),
        "missing_files": [item["path"] for item in files if not item["exists"]],
        "test_only_env": {
            "RAW_EDITOR_TEST_TONE_STATIONS": ",".join(str(index) for index in range(len(selected_station_labels))),
            "RAW_EDITOR_TEST_TONE_STATION_LABELS": ",".join(str(index) for index in selected_station_labels),
            "RAW_EDITOR_FOCUS_STACK_PATHS": os.pathsep.join(members),
        },
        "decision": (
            "ready_for_small_test_only_replay"
            if len(members) >= 2 and not any(item["path"] for item in files if not item["exists"])
            else "blocked_missing_station_or_source"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"source_count": result["source_count"], "decision": result["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
