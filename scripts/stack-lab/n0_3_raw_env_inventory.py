#!/usr/bin/env python3
"""Inventory RAW_EDITOR_* reads before making further N0.3 changes.

The list is intentionally conservative: acceptance/test modules are separated
from shared/production sources, while shared sources remain review candidates.
This prevents a broad grep from deleting valid preview, report, resource-budget
or comparison-path contracts.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENV_RE = re.compile(r"RAW_EDITOR_[A-Z0-9_]+")
SOURCE_ROOT = ROOT / "src-tauri/src"


def is_test_harness(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    name = path.name
    return (
        "/acceptance_tests/" in relative
        or "/tests/" in relative
        or name.endswith("_tests.rs")
        or name == "test_support.rs"
    )


def parent_cfg_test_module(path: Path) -> dict[str, str] | None:
    """Resolve the nearest facade declarations that compile this file."""
    stem = path.stem
    for parent in sorted(path.parent.glob("*.rs")):
        source = parent.read_text(encoding="utf-8")
        for match in re.finditer(
            r"#\[cfg\(test\)\]\s*((?:#\[path\s*=\s*\"[^\"]+\"\]\s*)*)"
            r"mod\s+([A-Za-z0-9_]+)\b",
            source,
        ):
            path_attributes, module = match.groups()
            target_names = [Path(value).name for value in re.findall(r'\"([^\"]+)\"', path_attributes)]
            if module == stem or path.name in target_names:
                return {
                    "parent": parent.relative_to(ROOT).as_posix(),
                    "module": module,
                }
    relative = path.relative_to(SOURCE_ROOT)
    parts = relative.parts
    current = SOURCE_ROOT
    for index, directory in enumerate(parts[:-1]):
        parent = current / f"{directory}.rs"
        child_stem = Path(parts[index + 1]).stem
        if parent.is_file():
            source = parent.read_text(encoding="utf-8")
            pattern = re.compile(
                rf"#\[cfg\(test\)\]\s*(?:#\[path\s*=\s*\"[^\"]+\"\]\s*)*"
                rf"mod\s+{re.escape(child_stem)}\b"
            )
            if pattern.search(source):
                return {
                    "parent": parent.relative_to(ROOT).as_posix(),
                    "module": child_stem,
                }
        current = current / directory
    return None


def main() -> int:
    occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in sorted(SOURCE_ROOT.rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        parent_cfg = parent_cfg_test_module(path)
        test_only = is_test_harness(path) or parent_cfg is not None
        for match in ENV_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            occurrences[match.group(0)].append(
                {
                    "path": relative,
                    "line": line,
                    "scope": "test_harness_module" if test_only else "shared_or_production_source",
                    "path_heuristic_test": is_test_harness(path),
                    "parent_cfg_test": parent_cfg,
                }
            )

    test_only_names = {
        "RAW_EDITOR_STACK_RAW_FIXED_GAIN",
        "RAW_EDITOR_SKIP_STACK_RAW_EXPOSURE_NORMALIZATION",
        "RAW_EDITOR_DISABLE_SHARED_RAW_NORMALIZATION",
        "RAW_EDITOR_STACK_DISABLE_NATIVE_PATCH_REFINEMENT",
        "RAW_EDITOR_FORCE_STREAMING_MOSAIC",
        "RAW_EDITOR_TEST_SOURCE_RESAMPLE_PATH",
        "RAW_EDITOR_TEST_SOURCE_RESAMPLE_DIVISOR",
        "RAW_EDITOR_FOCUS_OWNERSHIP_DIAGNOSTICS",
    }
    grouped = {
        name: {
            "occurrences": values,
            "count": len(values),
            "scopes": sorted({str(value["scope"]) for value in values}),
            "known_test_only": name in test_only_names,
            "parent_cfg_test_occurrences": sum(
                value["parent_cfg_test"] is not None for value in values
            ),
        }
        for name, values in sorted(occurrences.items())
    }
    review_names = [
        name
        for name, value in grouped.items()
        if "shared_or_production_source" in value["scopes"] and not value["known_test_only"]
    ]
    result = {
        "schema": 1,
        "task": "N0.3",
        "method": "raw_editor_environment_inventory_v1",
        "production_impact": "read_only_inventory_no_code_change",
        "summary": {
            "unique_env_names": len(grouped),
            "source_occurrences": sum(int(value["count"]) for value in grouped.values()),
            "known_test_only_names": sorted(name for name, value in grouped.items() if value["known_test_only"]),
            "shared_or_production_review_names": len(review_names),
            "parent_cfg_test_occurrences": sum(
                int(value["parent_cfg_test_occurrences"]) for value in grouped.values()
            ),
        },
        "variables": grouped,
        "remaining": [
            "review shared_or_production_source names one by one; do not delete preview, Stack_Report, resource-budget or comparison-path contracts by name alone",
            "RAW_EDITOR_FINAL_PANORAMA_SHARPEN_AMOUNT remains an explicit legacy comparison-path decision",
            "shared/production names still need one-by-one ownership review; path and parent cfg(test) classification is evidence, not a release gate",
        ],
        "decision": "inventory only; keep N0.3 open until the review candidates have explicit ownership and build-boundary evidence",
    }
    output = ROOT / "docs/benchmarks/n0-3-raw-env-inventory.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
