#!/usr/bin/env python3
"""Close the N0.3 review queue with an explicit owner for every shared switch."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "docs/benchmarks/n0-3-raw-env-inventory.json"
OUTPUT = ROOT / "docs/benchmarks/n0-3-shared-review-matrix.json"

RESOURCE_CONTRACT = {
    "RAW_EDITOR_STACK_NATIVE_FUSION_CACHE_MAX_BYTES",
    "RAW_EDITOR_STACK_SOURCE_RENDER_CACHE_MAX_BYTES",
}
PRODUCT_OUTPUT = {
    "RAW_EDITOR_ARTWORK_FRAMED_OUTPUT",
    "RAW_EDITOR_STACK_REPORT_DIR",
}
DIAGNOSTIC_CONTRACT = {
    "RAW_EDITOR_EXPOSURE_TRACE",
    "RAW_EDITOR_FOCUS_BOUNDARY_CANDIDATE_DIR",
    "RAW_EDITOR_NATIVE_POINT_TRACE_INPUT",
    "RAW_EDITOR_OWNERSHIP_TRACE",
    "RAW_EDITOR_STREAMING_MOSAIC_DIAGNOSTICS_DIR",
}


def category(name: str) -> tuple[str, str, str]:
    if name in RESOURCE_CONTRACT:
        return (
            "resource_budget_contract",
            "production resource budget",
            "retain as an explicit bounded cache/input budget; it does not select pixels or owners",
        )
    if name in PRODUCT_OUTPUT:
        return (
            "product_output_contract",
            "product output/report path",
            "retain as an explicit output/report contract; do not treat it as a research gate",
        )
    if name in DIAGNOSTIC_CONTRACT:
        return (
            "diagnostic_evidence_contract",
            "test/diagnostic evidence artifact",
            "retain only for evidence output; it must not alter production pixels, owners or geometry",
        )
    return (
        "legacy_comparison_control",
        "explicit legacy/comparison path",
        "retain behind the selected comparison path; the default layered production path does not depend on this switch",
    )


def main() -> int:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    variables = inventory["variables"]
    review_names = sorted(
        name
        for name, value in variables.items()
        if "shared_or_production_source" in value["scopes"] and not value["known_test_only"]
    )
    records = []
    for name in review_names:
        kind, owner, decision = category(name)
        records.append(
            {
                "name": name,
                "category": kind,
                "owner": owner,
                "decision": decision,
                "occurrences": variables[name]["occurrences"],
            }
        )
    counts = {}
    for record in records:
        counts[record["category"]] = counts.get(record["category"], 0) + 1
    result = {
        "schema": 1,
        "task": "N0.3",
        "method": "shared_environment_review_matrix_v1",
        "production_impact": "read_only_ownership_matrix_no_algorithm_change",
        "status": "pass" if len(records) == len(review_names) and all(record["category"] for record in records) else "open",
        "summary": {
            "review_names": len(review_names),
            "classified_names": len(records),
            "category_counts": counts,
        },
        "records": records,
        "decision": "all remaining shared/production environment names have an explicit owner and retention rule; no unowned experiment switch remains",
        "release_boundary": "legacy comparison controls remain outside the default layered production path; diagnostic contracts do not alter pixels, owners or geometry",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
