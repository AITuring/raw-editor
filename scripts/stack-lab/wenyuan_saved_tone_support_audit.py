#!/usr/bin/env python3
"""Classify saved Wenyuan tone failures by native correspondence support.

This is read-only and works on replay JSON plus the native joint proof. A
failure is ``in_support`` only when its witness lies inside both sides' axis-
aligned bounding boxes of native_verified_points. This is a conservative
support audit, not a quality-gate verdict.
"""
from __future__ import annotations

import argparse
import json
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
    xs = [v[0] for v in values]
    ys = [v[1] for v in values]
    return {"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)}


def inside(box: dict[str, float] | None, x: float, y: float) -> bool:
    return bool(
        box
        and box["x_min"] <= x <= box["x_max"]
        and box["y_min"] <= y <= box["y_max"]
    )


def bound_hit(parameters: list[float], scale: float) -> bool:
    gain_min, gain_max, offset = 1.0 - 0.2 * scale, 1.0 + 0.25 * scale, 0.02 * scale
    for channel in range(3):
        gain = 1.0 + float(parameters[channel])
        if abs(gain - gain_min) < 2e-4 or abs(gain - gain_max) < 2e-4:
            return True
        if abs(abs(float(parameters[channel + 3])) - offset) < 2e-4:
            return True
    return False


def classify(replay: dict[str, Any], tiles: list[dict[str, Any]], supports: dict[tuple[int, int], tuple[Any, Any]]) -> dict[str, Any]:
    scale = float(replay.get("test_only_bound_scale") or 1.0)
    classes: dict[str, dict[str, Any]] = {
        "in_support": {"count": 0, "max_delta": None, "pair_max": {}, "bound_hit_count": 0},
        "outside_support": {"count": 0, "max_delta": None, "pair_max": {}, "bound_hit_count": 0},
        "support_unavailable": {"count": 0, "max_delta": None, "pair_max": {}, "bound_hit_count": 0},
    }
    for row in replay.get("failed", []):
        if not isinstance(row, dict):
            continue
        left, right = row.get("first"), row.get("second")
        if not isinstance(left, int) or not isinstance(right, int):
            continue
        delta = float(row.get("delta", 0.0))
        key = (left, right)
        pair = supports.get(key)
        swap = False
        if pair is None:
            pair = supports.get((right, left))
            swap = pair is not None
        category = "support_unavailable"
        if pair is not None and 0 <= left < len(tiles) and 0 <= right < len(tiles):
            boxes = (pair[1], pair[0]) if swap else pair
            lx = float(row.get("x", 0.0)) - float(tiles[left]["world_origin"][0])
            ly = float(row.get("y", 0.0)) - float(tiles[left]["world_origin"][1])
            rx = float(row.get("x", 0.0)) - float(tiles[right]["world_origin"][0])
            ry = float(row.get("y", 0.0)) - float(tiles[right]["world_origin"][1])
            category = "in_support" if inside(boxes[0], lx, ly) and inside(boxes[1], rx, ry) else "outside_support"
        bucket = classes[category]
        bucket["count"] += 1
        bucket["max_delta"] = delta if bucket["max_delta"] is None else max(bucket["max_delta"], delta)
        pair_name = f"{min(left, right)}-{max(left, right)}"
        current = bucket["pair_max"].get(pair_name)
        bucket["pair_max"][pair_name] = delta if current is None else max(current, delta)
        if bound_hit(row.get("first_parameters", []), scale):
            bucket["bound_hit_count"] += 1
    for bucket in classes.values():
        bucket["pair_max"] = dict(sorted(bucket["pair_max"].items(), key=lambda item: -item[1]))
    return {"bound_scale": scale, "failed_total": len(replay.get("failed", [])), "classes": classes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--native-proof", type=Path, required=True)
    parser.add_argument("--replay", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tiles = [load(path) for path in sorted(args.capture_dir.glob("tone-owner-*.json"))]
    proof = load(args.native_proof)
    records = proof.get("records", []) if isinstance(proof, dict) else proof
    supports: dict[tuple[int, int], tuple[Any, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        left, right = record.get("left"), record.get("right")
        points = record.get("native_verified_points") or record.get("points") or []
        if isinstance(left, int) and isinstance(right, int):
            supports[(left, right)] = (bbox(points, 0), bbox(points, 1))
    result = {
        "schema": 1,
        "diagnostic_only": True,
        "method": "wenyuan_saved_tone_support_audit_v1",
        "capture_dir": str(args.capture_dir),
        "native_proof": str(args.native_proof),
        "tile_count": len(tiles),
        "relation_count": len(supports),
        "replays": {},
        "interpretation": "in_support requires both witness sides inside native correspondence bboxes; outside_support indicates extrapolation/support mismatch and is not itself a production failure verdict",
    }
    for label, path in args.replay:
        result["replays"][label] = classify(load(Path(path)), tiles, supports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["replays"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
