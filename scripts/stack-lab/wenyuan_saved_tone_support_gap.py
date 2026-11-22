#!/usr/bin/env python3
"""Quantify boundary-local support gaps in saved Wenyuan tone replay.

Read-only diagnostic: joins replay witnesses with saved ToneTile fields and
native verified correspondence bboxes. It does not alter production bounds or
quality gates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def bbox(points: list[list[float]], side: int) -> dict[str, float] | None:
    values = [
        (float(row[side * 2]), float(row[side * 2 + 1]))
        for row in points
        if isinstance(row, list) and len(row) >= side * 2 + 2
    ]
    if not values:
        return None
    xs = [value[0] for value in values]
    ys = [value[1] for value in values]
    return {"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)}


def gap(box: dict[str, float] | None, x: float, y: float) -> dict[str, Any]:
    if box is None:
        return {"available": False}
    dx = max(box["x_min"] - x, 0.0, x - box["x_max"])
    dy = max(box["y_min"] - y, 0.0, y - box["y_max"])
    directions: list[str] = []
    if x < box["x_min"]:
        directions.append("left")
    elif x > box["x_max"]:
        directions.append("right")
    if y < box["y_min"]:
        directions.append("top")
    elif y > box["y_max"]:
        directions.append("bottom")
    return {
        "available": True,
        "bbox": box,
        "gap_x_px": dx,
        "gap_y_px": dy,
        "gap_euclidean_px": math.hypot(dx, dy),
        "direction": directions,
        "inside": dx == 0.0 and dy == 0.0,
    }


def field_sample(tile: dict[str, Any], x: float, y: float) -> dict[str, Any]:
    ox, oy = tile["world_origin"]
    stride = float(tile["world_stride"])
    width, height = int(tile["width"]), int(tile["height"])
    cx = min(max(round((x - float(ox)) / stride), 0), width - 1)
    cy = min(max(round((y - float(oy)) / stride), 0), height - 1)
    index = cy * width + cx
    low = [float(value) for value in tile["low"][index * 3 : index * 3 + 3]]
    luminance = low[0] * 0.2126 + low[1] * 0.7152 + low[2] * 0.0722
    coverage = tile.get("cell_coverage")
    return {
        "cell": [cx, cy],
        "validity": int(tile["validity"][index]),
        "coverage": float(coverage[index]) if isinstance(coverage, list) else None,
        "low_luminance": luminance,
    }


def relation_index(proof: Any) -> dict[tuple[int, int], dict[str, Any]]:
    records = proof.get("records", []) if isinstance(proof, dict) else proof
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        left, right = record.get("left"), record.get("right")
        points = record.get("native_verified_points") or record.get("points") or []
        if not isinstance(left, int) or not isinstance(right, int):
            continue
        result[(left, right)] = {
            "left_bbox": bbox(points, 0),
            "right_bbox": bbox(points, 1),
            "inliers": record.get("inliers"),
            "error_px": record.get("error"),
            "native_verified_points": len(record.get("native_verified_points", [])),
            "hull_support": (record.get("diagnostic") or {}).get("hull_support"),
            "scale": record.get("scale"),
        }
    return result


def localize(row: dict[str, Any], tiles: list[dict[str, Any]], relations: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    left, right = int(row["first"]), int(row["second"])
    relation = relations.get((left, right))
    swapped = False
    if relation is None:
        relation = relations.get((right, left))
        swapped = relation is not None
    x, y = float(row["x"]), float(row["y"])
    if relation is None:
        return {"left_index": left, "right_index": right, "delta": row.get("delta"), "support_available": False}
    left_box = relation["right_bbox"] if swapped else relation["left_bbox"]
    right_box = relation["left_bbox"] if swapped else relation["right_bbox"]
    left_origin, right_origin = tiles[left]["world_origin"], tiles[right]["world_origin"]
    left_local = (x - float(left_origin[0]), y - float(left_origin[1]))
    right_local = (x - float(right_origin[0]), y - float(right_origin[1]))
    left_gap = gap(left_box, *left_local)
    right_gap = gap(right_box, *right_local)
    return {
        "left_index": left,
        "right_index": right,
        "left_station": tiles[left].get("station_label", tiles[left].get("station")),
        "right_station": tiles[right].get("station_label", tiles[right].get("station")),
        "world": [int(row["x"]), int(row["y"])],
        "delta": float(row.get("delta", 0.0)),
        "samples": row.get("samples"),
        "support_available": True,
        "left_local": [left_local[0], left_local[1]],
        "right_local": [right_local[0], right_local[1]],
        "left_gap": left_gap,
        "right_gap": right_gap,
        "outside_both": not left_gap.get("inside", False) and not right_gap.get("inside", False),
        "field": {
            "left": field_sample(tiles[left], x, y),
            "right": field_sample(tiles[right], x, y),
        },
        "relation": {
            "inliers": relation.get("inliers"),
            "error_px": relation.get("error_px"),
            "native_verified_points": relation.get("native_verified_points"),
            "hull_support": relation.get("hull_support"),
            "scale": relation.get("scale"),
        },
    }


def top_per_pair(replay: dict[str, Any], tiles: list[dict[str, Any]], relations: dict[tuple[int, int], dict[str, Any]], top: int) -> list[dict[str, Any]]:
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for row in replay.get("failed", []):
        if not isinstance(row, dict):
            continue
        left, right = row.get("first"), row.get("second")
        if not isinstance(left, int) or not isinstance(right, int):
            continue
        key = (left, right)
        if key not in best or float(row.get("delta", 0.0)) > float(best[key].get("delta", 0.0)):
            best[key] = row
    return [localize(row, tiles, relations) for row in sorted(best.values(), key=lambda value: float(value.get("delta", 0.0)), reverse=True)[:top]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--native-proof", type=Path, required=True)
    parser.add_argument("--replay", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()

    tiles = [load(path) for path in sorted(args.capture_dir.glob("tone-owner-*.json"))]
    relations = relation_index(load(args.native_proof))
    output: dict[str, Any] = {
        "schema": 1,
        "diagnostic_only": True,
        "method": "wenyuan_saved_tone_support_gap_v1",
        "capture_dir": str(args.capture_dir),
        "native_proof": str(args.native_proof),
        "relation_count": len(relations),
        "replays": {},
        "interpretation": "gap is measured from each witness to the axis-aligned bbox of native_verified_points; it quantifies extrapolation support, not a production quality verdict",
    }
    for label, path in args.replay:
        replay = load(Path(path))
        output["replays"][label] = {
            "maximum": replay.get("maximum"),
            "failed_window_count": len(replay.get("failed", [])),
            "top_pair_witnesses": top_per_pair(replay, tiles, relations, args.top),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({label: value["top_pair_witnesses"][:3] for label, value in output["replays"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
