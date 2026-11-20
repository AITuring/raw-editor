#!/usr/bin/env python3
"""Inventory child-to-facade imports for the N1.11 review queue.

The layout convention prefers child modules to depend on shared siblings.  A
few compatibility re-exports and test access modules legitimately mention
their facade, so this script reports every hit and keeps production/shared hits
open for review instead of treating a grep result as an automatic violation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src-tauri/src"


def test_path(relative: str) -> bool:
    return (
        "/tests/" in relative
        or "/acceptance_tests/" in relative
        or relative.endswith("_tests.rs")
        or relative.endswith("/tests.rs")
        or relative.endswith("/test_support.rs")
    )


def main() -> int:
    hits: list[dict[str, object]] = []
    for facade in sorted(SRC.rglob("*.rs")):
        child_dir = facade.with_suffix("")
        if not child_dir.is_dir():
            continue
        module_path = "crate::" + "::".join(facade.relative_to(SRC).with_suffix("").parts)
        for child in sorted(child_dir.rglob("*.rs")):
            relative = child.relative_to(ROOT).as_posix()
            lines = child.read_text(encoding="utf-8").splitlines()
            module_pattern = re.compile(rf"{re.escape(module_path)}(?![A-Za-z0-9_])")
            for line_number, line in enumerate(lines, 1):
                if module_pattern.search(line) and re.search(r"\b(use|pub\s+use|super::|crate::)", line):
                    hits.append(
                        {
                            "facade": facade.relative_to(ROOT).as_posix(),
                            "child": relative,
                            "line": line_number,
                            "module": module_path,
                            "scope": "test_path_heuristic" if test_path(relative) else "production_or_shared_review",
                            "text": line.strip(),
                        }
                    )

    production_hits = [hit for hit in hits if hit["scope"] == "production_or_shared_review"]
    result = {
        "schema": 1,
        "task": "N1.11",
        "method": "child_to_facade_import_direction_inventory_v1",
        "production_impact": "read_only_import_inventory_no_code_change",
        "summary": {
            "total_hits": len(hits),
            "test_path_hits": len(hits) - len(production_hits),
            "production_or_shared_review_hits": len(production_hits),
        },
        "hits": hits,
        "remaining": [
            "confirm each production_or_shared_review hit is a deliberate compatibility dependency or move the shared item to a sibling module",
            "replace path heuristics with parent-module cfg(test) resolution before using this as a blocking contract",
        ],
        "decision": "record the import direction inventory and keep N1.11 open",
    }
    output = ROOT / "docs/benchmarks/n1-11-import-direction-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
