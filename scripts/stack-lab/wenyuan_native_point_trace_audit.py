#!/usr/bin/env python3
"""Summarize the cfg(test)-only active LayeredVirtualTile point trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace: dict[str, Any] = json.loads(args.trace.read_text())
    rows = []
    for station_entry in trace.get("stations", []):
        station = station_entry.get("station", {})
        dimensions = station.get("tile_dimensions", [None, None])
        for point in station_entry.get("points", []):
            candidates = point.get("native_candidates") or {}
            diagnostics = candidates.get("source_diagnostics") or []
            actual = point.get("actual") or {}
            rows.append(
                {
                    "label": (point.get("request") or {}).get("label"),
                    "world": point.get("world"),
                    "station_order_index": station.get("station_order_index"),
                    "topology_station_id": station.get("topology_station_id"),
                    "tile_dimensions": dimensions,
                    "diagnostic_world_to_tile": point.get("diagnostic_world_to_tile"),
                    "station_bbox_eligible": point.get("station_bbox_eligible"),
                    "actual_status": actual.get("status"),
                    "native_owner_resolution": actual.get("native_owner_resolution"),
                    "sample_invoked": actual.get("sample_invoked", False),
                    "source_diagnostic_count": len(diagnostics),
                    "source_diagnostics": [
                        {
                            "source": (item.get("source") or {}).get("filename"),
                            "source_footprint_supported": item.get("source_footprint_supported"),
                            "global_source_supported": item.get("global_source_supported"),
                            "source_point": item.get("source_point"),
                            "rejection": item.get("rejection"),
                        }
                        for item in diagnostics
                    ],
                }
            )
    compositor_points = (trace.get("compositor_return") or {}).get("points", [])
    summary = {
        "trace_status": trace.get("iteration_status"),
        "station_count": len(trace.get("stations", [])),
        "witness_count": len({row["label"] for row in rows}),
        "station_point_rows": len(rows),
        "outside_station_bbox_rows": sum(not row["station_bbox_eligible"] for row in rows),
        "inside_station_bbox_rows": sum(bool(row["station_bbox_eligible"]) for row in rows),
        "source_diagnostic_rows": sum(row["source_diagnostic_count"] > 0 for row in rows),
        "native_owner_resolution_rows": sum(row["native_owner_resolution"] is not None for row in rows),
        "sample_invoked_rows": sum(row["sample_invoked"] for row in rows),
        "final_compositor_points": compositor_points,
    }
    result = {
        "schema": 1,
        "diagnostic_only": True,
        "method": "wenyuan_active_layered_native_point_trace_audit_v1",
        "trace": str(args.trace),
        "summary": summary,
        "rows": rows,
        "interpretation": "all witness rows outside station bbox with no active native source resolution means the LayeredVirtualTile path has no source-sampling support at these boundary coordinates; this is evidence for a boundary-local support gap, not a QG verdict",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
