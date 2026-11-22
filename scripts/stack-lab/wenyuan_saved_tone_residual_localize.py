#!/usr/bin/env python3
"""Localize saved-native Wenyuan tone residuals without rerunning stitching.

The replay JSON stores local tile indices (the order in which owner-001..N
were loaded), while the captured ToneTile JSON stores the original run-local
station and label.  This script joins those records, reports the strongest
boundary witnesses, samples the saved low-frequency fields at each witness,
and attaches the corresponding native relation evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def luminance(rgb: list[float]) -> float:
    return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722


def sample_field(tile: dict[str, Any], x: int, y: int) -> dict[str, Any]:
    origin = tile["world_origin"]
    stride = float(tile["world_stride"])
    width = int(tile["width"])
    height = int(tile["height"])
    cx = min(max(round((x - float(origin[0])) / stride), 0), width - 1)
    cy = min(max(round((y - float(origin[1])) / stride), 0), height - 1)
    index = cy * width + cx
    low = [float(v) for v in tile["low"][index * 3 : index * 3 + 3]]
    mean = [float(v) for v in tile["cell_mean"][index * 3 : index * 3 + 3]]
    validity = int(tile["validity"][index])
    coverage = tile.get("cell_coverage")
    coverage_value = float(coverage[index]) if isinstance(coverage, list) else None
    value = luminance(low)
    return {
        "cell": [cx, cy],
        "low": low,
        "cell_mean": mean,
        "validity": validity,
        "coverage": coverage_value,
        "luminance": value,
    }


def pair_key(left: int, right: int) -> str:
    return f"{min(left, right)}↔{max(left, right)}"


def point_bbox(points: list[list[float]], side: int) -> dict[str, float] | None:
    values = [
        (float(point[side * 2]), float(point[side * 2 + 1]))
        for point in points
        if isinstance(point, list) and len(point) >= side * 2 + 2
    ]
    if not values:
        return None
    xs = [value[0] for value in values]
    ys = [value[1] for value in values]
    return {"x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)}


def support_evidence(
    row: dict[str, Any],
    tiles: list[dict[str, Any]],
    relation_bboxes: dict[tuple[int, int], tuple[dict[str, float] | None, dict[str, float] | None]],
) -> dict[str, Any]:
    left = int(row["first"])
    right = int(row["second"])
    pair = relation_bboxes.get((left, right)) or relation_bboxes.get((right, left))
    if pair is None:
        return {"native_support_available": False}
    witness = (float(row["x"]), float(row["y"]))
    result: dict[str, Any] = {"native_support_available": True}
    for name, index, bbox in (("left", left, pair[0]), ("right", right, pair[1])):
        origin = tiles[index].get("world_origin", [0.0, 0.0])
        local = {"x": witness[0] - float(origin[0]), "y": witness[1] - float(origin[1])}
        inside = bool(
            bbox
            and bbox["x_min"] <= local["x"] <= bbox["x_max"]
            and bbox["y_min"] <= local["y"] <= bbox["y_max"]
        )
        result[f"{name}_witness_local"] = local
        result[f"{name}_native_verified_bbox"] = bbox
        result[f"{name}_inside_native_verified_bbox"] = inside
    result["outside_native_verified_support"] = not (
        result.get("left_inside_native_verified_bbox", False)
        and result.get("right_inside_native_verified_bbox", False)
    )
    return result


def bound_hit(parameters: list[float], scale: float = 1.0) -> bool:
    gain_min = 1.0 - 0.2 * scale
    gain_max = 1.0 + 0.25 * scale
    offset = 0.02 * scale
    for channel in range(3):
        gain = 1.0 + float(parameters[channel])
        if abs(gain - gain_min) < 2e-4 or abs(gain - gain_max) < 2e-4:
            return True
        value = float(parameters[channel + 3])
        if abs(abs(value) - offset) < 2e-4:
            return True
    return False


def witness_rows(
    replay: dict[str, Any],
    tiles: list[dict[str, Any]],
    relation_bboxes: dict[tuple[int, int], tuple[dict[str, float] | None, dict[str, float] | None]],
    top: int,
) -> list[dict[str, Any]]:
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for row in replay.get("failed", []):
        if not isinstance(row, dict):
            continue
        left, right = row.get("first"), row.get("second")
        if not isinstance(left, int) or not isinstance(right, int):
            continue
        if not (0 <= left < len(tiles) and 0 <= right < len(tiles)):
            continue
        key = (left, right)
        if key not in best or float(row.get("delta", 0.0)) > float(best[key].get("delta", 0.0)):
            best[key] = row
    result = []
    for row in sorted(best.values(), key=lambda value: float(value.get("delta", 0.0)), reverse=True)[:top]:
        left = int(row["first"])
        right = int(row["second"])
        left_sample = sample_field(tiles[left], int(row["x"]), int(row["y"]))
        right_sample = sample_field(tiles[right], int(row["x"]), int(row["y"]))
        left_luma = left_sample["luminance"]
        right_luma = right_sample["luminance"]
        ratio = None
        if min(left_luma, right_luma) > 1e-9:
            ratio = max(left_luma, right_luma) / min(left_luma, right_luma)
        result.append(
            {
                "left_index": left,
                "right_index": right,
                "left_station": tiles[left].get("station_label", tiles[left].get("station")),
                "right_station": tiles[right].get("station_label", tiles[right].get("station")),
                "left_run_local_station": tiles[left].get("station"),
                "right_run_local_station": tiles[right].get("station"),
                "x": row.get("x"),
                "y": row.get("y"),
                "delta": row.get("delta"),
                "samples": row.get("samples"),
                "common_rgb": row.get("common_rgb"),
                "first_parameters": row.get("first_parameters"),
                "second_parameters": row.get("second_parameters"),
                "control_bound_hit": bound_hit(row.get("first_parameters", [])),
                "left_field": left_sample,
                "right_field": right_sample,
                "low_luminance_ratio": ratio,
                "native_support": support_evidence(row, tiles, relation_bboxes),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--replay-control", type=Path, required=True)
    parser.add_argument("--replay-candidate", type=Path, required=True)
    parser.add_argument("--native-proof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    tiles = []
    for path in sorted(args.capture_dir.glob("tone-owner-*.json")):
        tiles.append(load(path))
    if len(tiles) < 2:
        raise SystemExit("capture directory must contain at least two tone-owner JSON files")
    control = load(args.replay_control)
    candidate = load(args.replay_candidate)
    proof = load(args.native_proof)
    proof_records = proof.get("records", []) if isinstance(proof, dict) else proof
    relations = []
    relation_bboxes: dict[
        tuple[int, int], tuple[dict[str, float] | None, dict[str, float] | None]
    ] = {}
    for row in proof_records:
        if not isinstance(row, dict):
            continue
        left, right = row.get("left"), row.get("right")
        if not isinstance(left, int) or not isinstance(right, int):
            continue
        relations.append(
            {
                "left_index": left,
                "right_index": right,
                "left_station": tiles[left].get("station_label") if left < len(tiles) else None,
                "right_station": tiles[right].get("station_label") if right < len(tiles) else None,
                "inliers": row.get("inliers"),
                "error": row.get("error"),
                "native_verified_points": len(row.get("native_verified_points", [])),
                "hull_support": (row.get("diagnostic") or {}).get("hull_support"),
                "scale": row.get("scale"),
            }
        )
        points = row.get("native_verified_points") or row.get("points") or []
        relation_bboxes[(left, right)] = (point_bbox(points, 0), point_bbox(points, 1))

    output = {
        "schema": 1,
        "diagnostic_only": True,
        "method": "wenyuan_saved_tone_residual_localization_v1",
        "capture_dir": str(args.capture_dir),
        "tile_count": len(tiles),
        "control": {
            "replay": str(args.replay_control),
            "maximum": control.get("maximum"),
            "failed_window_count": len(control.get("failed", [])),
            "top_witnesses": witness_rows(control, tiles, relation_bboxes, args.top),
        },
        "candidate": {
            "replay": str(args.replay_candidate),
            "maximum": candidate.get("maximum"),
            "failed_window_count": len(candidate.get("failed", [])),
            "top_witnesses": witness_rows(candidate, tiles, relation_bboxes, args.top),
        },
        "native_relations": relations,
        "interpretation": "Use low-frequency field and relation evidence to separate source/Jacobian residuals from bound-hit residuals; this artifact does not change production thresholds or bounds.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "control_maximum": output["control"]["maximum"],
        "candidate_maximum": output["candidate"]["maximum"],
        "control_top_pair": output["control"]["top_witnesses"][0] if output["control"]["top_witnesses"] else None,
        "candidate_top_pair": output["candidate"]["top_witnesses"][0] if output["candidate"]["top_witnesses"] else None,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
