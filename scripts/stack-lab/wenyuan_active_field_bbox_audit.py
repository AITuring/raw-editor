#!/usr/bin/env python3
"""Compare saved ToneTile world rectangles with active station tile bounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace: dict[str, Any] = json.loads(args.trace.read_text())
    fields = {}
    for path in sorted(args.capture_dir.glob("tone-owner-*.json")):
        item = json.loads(path.read_text())
        fields[item.get("station")] = item
    rows = []
    for entry in trace.get("stations", []):
        station = entry.get("station", {})
        local = station.get("topology_station_id")
        field = fields.get(local)
        if field is None:
            continue
        origin = field["world_origin"]
        size = field["world_size"]
        dimensions = station.get("tile_dimensions", [0, 0])
        for point in entry.get("points", []):
            # Saved ToneTile world rectangles and compositor raster bounds are
            # expressed in the active canvas coordinate domain.  The trace's
            # `world` field is the requested canvas point plus
            # reference_canvas_origin, so use the request canvas here.
            world = (point.get("request") or {}).get("canvas", [0.0, 0.0])
            mapped = point.get("diagnostic_world_to_tile")
            field_inside = (
                origin[0] <= world[0] <= origin[0] + size[0]
                and origin[1] <= world[1] <= origin[1] + size[1]
            )
            active_inside = bool(point.get("station_bbox_eligible"))
            active_tile_inside = bool(
                mapped
                and 0.0 <= mapped[0] < dimensions[0]
                and 0.0 <= mapped[1] < dimensions[1]
            )
            rows.append(
                {
                    "label": (point.get("request") or {}).get("label"),
                    "station_local": local,
                    "station_label": field.get("station_label"),
                    "canvas": world,
                    "field_world_origin": origin,
                    "field_world_size": size,
                    "field_inside": field_inside,
                    "active_tile_dimensions": dimensions,
                    "active_mapped_tile": mapped,
                    "active_bbox_inside": active_inside,
                    "active_tile_inside": active_tile_inside,
                    "field_inside_active_outside": field_inside and not active_tile_inside,
                }
            )
    summary = {
        "rows": len(rows),
        "field_inside": sum(row["field_inside"] for row in rows),
        "active_tile_inside": sum(row["active_tile_inside"] for row in rows),
        "field_inside_active_outside": sum(row["field_inside_active_outside"] for row in rows),
    }
    result = {
        "schema": 1,
        "diagnostic_only": True,
        "method": "wenyuan_active_field_bbox_audit_v1",
        "trace": str(args.trace),
        "capture_dir": str(args.capture_dir),
        "summary": summary,
        "rows": rows,
        "interpretation": "field_inside_active_outside means a saved tone field world rectangle covers the witness while the active LayeredVirtualTile station mapping does not; this indicates capture-field/active-geometry mismatch or extrapolation and is not a QG verdict",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
