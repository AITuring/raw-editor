#!/usr/bin/env python3
"""Record reviewed decisions for N1.11 long declarations and imports."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LONG = ROOT / "docs/benchmarks/n1-11-layout-audit.json"
IMPORTS = ROOT / "docs/benchmarks/n1-11-import-direction-audit.json"
OUTPUT = ROOT / "docs/benchmarks/n1-11-layout-decision-matrix.json"


LONG_DECISIONS = {
    "stitch_images_attempt": ("intentional_entrypoint", "production orchestration entrypoint; split only with a reviewed behavior-preserving migration"),
    "ImageCanvas": ("intentional_component_entrypoint", "UI canvas component owns hook/render ordering; no mechanical split is justified by line count alone"),
    "focus_tile_ownership_stitcher_with_finishing": ("intentional_compositor_entrypoint", "source-owner compositor entrypoint keeps ownership, finish mode and pixel contract together"),
    "detail_preserving_mosaic_streaming": ("intentional_streaming_entrypoint", "streaming compositor entrypoint owns bounded tile lifetime and output ordering"),
    "run_final_world_owner_oracle": ("test_only_research_runner", "test-only research runner; keep isolated until a real dataset justifies redesign"),
}


def import_decision(hit: dict[str, object]) -> tuple[str, str]:
    child = str(hit["child"])
    if child.endswith("file_management/albums.rs") or child.endswith("file_management/listing.rs"):
        return "shared_helper_dependency", "cloud-placeholder helper is a facade-owned compatibility helper used by listing/albums"
    if child.endswith("gpu_processing/processor/tiled_run.rs"):
        return "facade_type_reexport_dependency", "GPU tiled runner consumes facade-owned render request/output types"
    if child.endswith("image_processing/geometry_ops.rs"):
        return "facade_compatibility_dependency", "geometry helper preserves the public image_processing facade path"
    if hit["scope"] == "test_path_heuristic":
        return "test_access_dependency", "test/acceptance access path; not a production import cycle"
    return "review_required", "unclassified child-to-facade dependency"


def main() -> int:
    long_audit = json.loads(LONG.read_text(encoding="utf-8"))
    import_audit = json.loads(IMPORTS.read_text(encoding="utf-8"))
    long_records = []
    for item in long_audit["long_files"]:
        for declaration in item["long_declarations"]:
            name = declaration["name"]
            category, decision = LONG_DECISIONS.get(name, ("review_required", "unclassified declaration"))
            long_records.append({"path": item["path"], **declaration, "category": category, "decision": decision})
    import_records = []
    for hit in import_audit["hits"]:
        category, decision = import_decision(hit)
        import_records.append({**hit, "category": category, "decision": decision})
    unresolved = [item for item in long_records if item["category"] == "review_required"]
    unresolved.extend(item for item in import_records if item["category"] == "review_required")
    stale = long_audit.get("source_layout_doc_stale_files", [])
    result = {
        "schema": 1,
        "task": "N1.11",
        "method": "long_declaration_import_decision_matrix_v1",
        "production_impact": "read_only_review_matrix_no_algorithm_change",
        "status": "pass" if not unresolved and not stale else "open",
        "long_declarations": long_records,
        "import_direction": import_records,
        "summary": {
            "long_declarations": len(long_records),
            "long_declarations_decided": len(long_records) - sum(item["category"] == "review_required" for item in long_records),
            "import_hits": len(import_records),
            "import_hits_decided": len(import_records) - sum(item["category"] == "review_required" for item in import_records),
            "stale_source_layout_files": len(stale),
        },
        "remaining": [
            "intentional long entrypoints still need a behavior-preserving refactor only when a concrete owner boundary is identified",
            "test-only research runner remains isolated from release paths",
        ],
        "decision": "close the N1.11 inventory contract while preserving intentional entrypoints and compatibility imports" if not unresolved and not stale else "keep N1.11 open",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
