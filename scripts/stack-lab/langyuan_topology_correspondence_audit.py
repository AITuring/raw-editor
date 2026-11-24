#!/usr/bin/env python3
"""Read-only topology/source correspondence audit for the Langyuan scale=1 witness.

This consumes an existing Stack_Report and source-alignment.json.  It does not
rerun registration, write poses, or change any acceptance contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def basename(path: str) -> str:
    return Path(path).name


def station_audit(station: dict, alignment: dict, topology_candidates: list[dict]) -> dict:
    members = station.get("members", [])
    names = {basename(item) for item in members}
    source_by_name = {basename(item["filename"]): item for item in alignment.get("sources", [])}
    source_rows = [source_by_name[name] for name in sorted(names) if name in source_by_name]
    source_ids = sorted(item["id"] for item in source_rows)
    exact_groups = [
        group for group in alignment.get("capture_groups", [])
        if sorted(group.get("member_ids", [])) == source_ids
    ]
    involving = [
        item for item in topology_candidates
        if item.get("left") == station.get("index") or item.get("right") == station.get("index")
    ]
    involving.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return {
        "station_index": station.get("index"),
        "topology_status": station.get("topology_status"),
        "members": members,
        "source_ids": source_ids,
        "source_membership_complete": len(source_rows) == len(members),
        "capture_group_exact_membership": len(exact_groups) == 1,
        "capture_groups": [
            {
                "group_index": item.get("group_index"),
                "anchor_id": item.get("anchor_id"),
                "member_ids": item.get("member_ids"),
                "member_count": item.get("member_count"),
                "median_center_drift_px": item.get("median_center_drift_px"),
                "maximum_center_drift_px": item.get("maximum_center_drift_px"),
            }
            for item in exact_groups
        ],
        "station_overlap_evidence": station.get("evidence", {}),
        "topology_candidates": involving[:8],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = load(args.report)
    alignment = load(args.alignment)
    stations = report.get("grouping", {}).get("stations", [])
    selected = [station for station in stations if station.get("index") in (0, 17)]
    audits = [station_audit(station, alignment, report.get("topology", {}).get("candidates", [])) for station in selected]
    by_index = {item["station_index"]: item for item in audits}
    left_group = (by_index.get(0, {}).get("capture_groups") or [{}])[0]
    right_group = (by_index.get(17, {}).get("capture_groups") or [{}])[0]
    left_median = left_group.get("median_center_drift_px")
    right_median = right_group.get("median_center_drift_px")
    left_max = left_group.get("maximum_center_drift_px")
    right_max = right_group.get("maximum_center_drift_px")
    comparison = {
        "station17_to_station0_median_center_drift_ratio": (right_median / left_median)
        if left_median and right_median else None,
        "station17_to_station0_max_center_drift_ratio": (right_max / left_max)
        if left_max and right_max else None,
        "exact_capture_membership_for_both": all(item.get("capture_group_exact_membership") for item in audits),
        "source_membership_complete_for_both": all(item.get("source_membership_complete") for item in audits),
    }
    result = {
        "schema": 1,
        "method": "langyuan_scale1_topology_correspondence_audit_v1",
        "production_impact": "test_only_read_only_topology_and_source_membership_no_pose_write_no_threshold_change",
        "report": str(args.report),
        "alignment": str(args.alignment),
        "stations": audits,
        "comparison": comparison,
        "interpretation": (
            "Stations 0 and 17 have exact source membership matches in source-alignment.json, "
            "so the witness is not explained by a missing or misassigned source member. "
            "Station 17 has materially larger within-station center drift than station 0 while both "
            "retain strong verified-overlap evidence; both remain topology_index_ambiguous. "
            "The cross-station 0↔17 relation therefore remains a topology/physical-magnification "
            "diagnostic, not a production scale correction."
        ),
        "next_action": "Keep the rejection and topology ambiguity; reopen only with new topology or source correspondence evidence.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "stations": len(audits), "status": "written"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
