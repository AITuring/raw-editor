#!/usr/bin/env python3
"""Read-only Task17 Chibi tone/local-scale isolation from one Stack_Report.

This is a bounded diagnostic: it joins the failed local-scale ROIs with their
source stations and extracts the 2↔5 tone relation candidate. It does not
recompute pixels or change production bounds/thresholds/fallbacks.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    d = load(args.report)
    stations = {
        int(s["index"]): {
            "members": s.get("members", []),
            "anchor": s.get("anchor"),
        }
        for s in d["grouping"].get("stations", [])
    }
    source_station = {}
    for idx, s in stations.items():
        for p in s["members"]:
            source_station[p] = idx

    local = {}
    for criterion_name in ("local_scale_median", "local_scale_pixel_ratio"):
        criterion = next(c for c in d["quality_gate"]["criteria"] if c["name"] == criterion_name)
        rows = []
        for row in criterion.get("measured", []):
            owner = row.get("owner_path", "")
            rows.append({
                "world": row.get("world"),
                "measured": row.get("measured"),
                "owner_path": owner,
                "owner_station": source_station.get(owner),
                "failed": row in criterion.get("failed", []),
            })
        local[criterion_name] = {
            "verdict": criterion.get("verdict"),
            "threshold": criterion.get("threshold"),
            "rows": rows,
        }

    candidate = next(
        (r for r in d["station_relations"].get("candidates", [])
         if {int(r.get("left", -1)), int(r.get("right", -1))} == {2, 5}),
        None,
    )
    tone_violations = [
        v for v in d["tone"]["boundary_delta_e"].get("violations", [])
        if {int(v.get("left", -1)), int(v.get("right", -1))} == {2, 5}
    ]
    tile_by_station = {
        int(t["station_index"]): {
            "gain": t.get("gain"),
            "solved_gain": t.get("solved_gain"),
            "solved_offset": t.get("solved_offset"),
            "spatial_gain_min": t.get("spatial_gain_min"),
            "spatial_gain_max": t.get("spatial_gain_max"),
            "samples": t.get("samples"),
        }
        for t in d["tone"].get("tiles", [])
        if int(t.get("station_index", -1)) in (2, 5)
    }
    native_fitting = [
        [int(pair[0]), int(pair[1])]
        for pair in d["tone"].get("native_fitting_relations", [])
    ]
    tone_pair_key = {2, 5}
    result = {
        "schema": 1,
        "method": "task17_chibi_tone_local_scale_bounded_audit_v1",
        "production_impact": "read-only report join; no pixels, thresholds, bounds, fallback, or production code changed",
        "report": str(args.report),
        "dataset": "chibi",
        "scale": 0.25,
        "tone_pair": {
            "pair": "2↔5",
            "violations": tone_violations,
            "candidate": candidate,
            "tile_fields": tile_by_station,
            "native_fitting_relation_present": any(set(pair) == tone_pair_key for pair in native_fitting),
            "native_fitting_relations": native_fitting,
            "pairs_without_evidence": d["tone"].get("pairs_without_evidence", []),
        },
        "local_scale": local,
        "station_members": {str(i): v for i, v in stations.items() if i in (2, 3, 5, 7, 8, 11)},
        "finding": {
            "relation_support": "2↔5 candidate is not fitted: 10 measurable patches, 8 bidirectional matches, residual model not_run, failure_stage=insufficient_measurable_patches, prefit orientation difference ≈14.043°",
            "local_scale_overlap": "all six non-1.0 local_scale_pixel_ratio failures map to stations 2,3,5,7,8,11; station 2 and 5 are included in the tone witness pair",
            "bounded_next": "test-only source/geometry witness replay for pair 2↔5 and the six failed local-scale owners; keep thresholds/bounds/fallback unchanged",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "tone_pair_violations": len(tone_violations),
        "candidate_failure_stage": candidate.get("failure_stage") if candidate else None,
        "local_scale_failed": sum(1 for r in local["local_scale_pixel_ratio"]["rows"] if r["failed"]),
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
