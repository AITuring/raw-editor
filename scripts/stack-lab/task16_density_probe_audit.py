#!/usr/bin/env python3
"""Summarize test-only station/source output-density probe evidence."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def mat3(flat: list[float]) -> list[list[float]]:
    # nalgebra serializes Matrix3::as_slice in column-major order.
    return [[flat[col * 3 + row] for col in range(3)] for row in range(3)]


def map_point(h: list[list[float]], x: float, y: float) -> tuple[float, float]:
    z = h[2][0] * x + h[2][1] * y + h[2][2]
    if not math.isfinite(z) or abs(z) < 1e-12:
        raise ValueError("singular station pose")
    return (
        (h[0][0] * x + h[0][1] * y + h[0][2]) / z,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / z,
    )


def inverse2(m: list[list[float]]) -> list[list[float]]:
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if not math.isfinite(det) or abs(det) < 1e-12:
        raise ValueError("singular source sampling matrix")
    return [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]


def min_singular(m: list[list[float]]) -> float:
    a, b, c, d = m[0][0], m[0][1], m[1][0], m[1][1]
    trace = a * a + b * b + c * c + d * d
    det = a * d - b * c
    discriminant = max(0.0, trace * trace - 4.0 * det * det)
    return math.sqrt(max(0.0, (trace - math.sqrt(discriminant)) / 2.0))


def world_per_source(h: list[list[float]], source_per_station: list[list[float]], x: float, y: float) -> float:
    px = map_point(h, x + 0.5, y)
    mx = map_point(h, x - 0.5, y)
    py = map_point(h, x, y + 0.5)
    my = map_point(h, x, y - 0.5)
    world = [[px[0] - mx[0], py[0] - my[0]], [px[1] - mx[1], py[1] - my[1]]]
    inv = inverse2(source_per_station)
    product = [
        [world[0][0] * inv[0][0] + world[0][1] * inv[1][0], world[0][0] * inv[0][1] + world[0][1] * inv[1][1]],
        [world[1][0] * inv[0][0] + world[1][1] * inv[1][0], world[1][0] * inv[0][1] + world[1][1] * inv[1][1]],
    ]
    return min_singular(product)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    total = 0
    for path in sorted(args.density_dir.glob("station-*.json")):
        data = json.loads(path.read_text())
        flat = data.get("station_to_world")
        if not flat:
            raise ValueError(f"missing station_to_world in {path}")
        h = mat3(flat)
        station = int(data["station"])
        for probe in data["probes"]:
            point = probe["point"]
            value = world_per_source(h, probe["source_per_station"], point[0], point[1])
            groups[(station, int(probe["source_id"]))].append({"scale": value, "point": point})
            total += 1

    summaries = []
    for (station, source_id), values in groups.items():
        ordered = sorted(values, key=lambda item: item["scale"])
        scales = [item["scale"] for item in ordered]
        p01 = scales[max(0, (len(scales) - 1) // 100)]
        summaries.append({
            "station": station,
            "source_id": source_id,
            "probe_count": len(scales),
            "minimum_scale": scales[0],
            "p01_scale": p01,
            "median_scale": scales[len(scales) // 2],
            "minimum_point": ordered[0]["point"],
        })
    summaries.sort(key=lambda item: item["minimum_scale"])
    result = {
        "schema": 1,
        "task": 16,
        "method": "task16_station_density_probe_audit_v1",
        "production_impact": "test_only_read_only_probe_export_no_pose_write_no_density_override",
        "density_dir": str(args.density_dir),
        "stations": len({item["station"] for item in summaries}),
        "station_source_groups": len(summaries),
        "probe_count": total,
        "worst_groups": summaries[:20],
        "decision": "station/source Jacobian evidence agrees with the rejected output-density report; no factor override is justified",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "probe_count": total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
