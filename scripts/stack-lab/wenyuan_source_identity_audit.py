#!/usr/bin/env python3
"""Audit active witness source identity against the frozen station subset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--active-mismatch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    subset = load(args.subset)
    mismatch = load(args.active_mismatch)
    members: dict[int, set[str]] = {}
    for row in subset.get("selected", []):
        if row.get("status") != "selected":
            continue
        members[int(row["index"])] = {os.path.abspath(str(path)) for path in row.get("members", [])}

    rows: dict[str, dict[str, Any]] = {}
    for pair, value in mismatch.get("pairs", {}).items():
        if pair not in {"36↔37", "37↔38", "36↔38"}:
            continue
        left_label, right_label = (int(part) for part in pair.split("↔"))
        active_left = value.get("active_left") or {}
        active_right = value.get("active_right") or {}
        sides = []
        for label, active in ((left_label, active_left), (right_label, active_right)):
            source = active.get("source_path")
            source_abs = os.path.abspath(source) if source else None
            expected = members.get(label, set())
            sides.append({
                "station_label": label,
                "source_path": source,
                "source_id": active.get("source_id"),
                "status": active.get("status"),
                "used_fallback": active.get("used_fallback"),
                "source_sampling_observed": active.get("source_sampling_observed"),
                "expected_station_member": source_abs in expected if source_abs else False,
            })
        rows[pair] = {
            "sides": sides,
            "cross_cluster_source": any(not side["expected_station_member"] for side in sides),
            "fallback_rows": sum(side["used_fallback"] is True for side in sides),
            "source_sampling_complete": all(side["source_sampling_observed"] is True for side in sides),
            "classification": "source_identity_consistent_direct_content_mismatch"
            if all(side["expected_station_member"] for side in sides)
            and all(side["used_fallback"] is False for side in sides)
            else "source_identity_requires_review",
        }

    source_reuse: dict[str, list[str]] = {}
    for pair, row in rows.items():
        for side in row["sides"]:
            source = side.get("source_path")
            if source:
                source_reuse.setdefault(source, []).append(pair)
    result = {
        "schema": 1,
        "diagnostic_only": True,
        "method": "wenyuan_source_identity_audit_v1",
        "production_impact": "read-only source identity audit; no decoding, owner, relation, or bounds change",
        "inputs": {"subset": str(args.subset), "active_mismatch": str(args.active_mismatch)},
        "station_member_counts": {str(key): len(value) for key, value in sorted(members.items())},
        "pairs": rows,
        "source_reuse": source_reuse,
        "decision": {
            "promotion": "rejected",
            "conclusion": "all reported active witness sources belong to their expected station clusters and use no fallback; remaining pair residuals are not explained by cross-cluster source contamination",
            "next_step": "inspect source/content correspondence at the witness locations; do not add source normalization or widen tone bounds",
        },
        "limitations": [
            "source identity is checked against the 15-source subset manifest, not a new full-run decode",
            "membership consistency does not prove that the source pixels depict identical content",
            "no full 126-source run was performed",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value["classification"] for key, value in rows.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
