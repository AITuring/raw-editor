#!/usr/bin/env python3
"""Project an existing Stack_Report into the task14 candidate exclusion schema.

This is a read-only fixture generator. It never edits the manifest, runner, or
source report; the generated JSON is only evidence for schema review.
"""

import argparse
import hashlib
import json
from pathlib import Path


EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def load(path: Path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--excluded", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    execution = load(args.execution)
    report = load(args.report)
    excluded_names = set(args.excluded)
    inputs = {Path(item["path"]).name: item for item in execution["inputs"]}
    report_sources = (report.get("input") or {}).get("sources") or []
    report_by_name = {Path(item.get("path", "")).name: item for item in report_sources}
    active_names = sorted(set(inputs) - excluded_names)

    sources = [report_by_name[name] for name in active_names if name in report_by_name]
    excluded_sources = []
    for name in sorted(excluded_names):
        item = report_by_name.get(name)
        input_item = inputs.get(name)
        if item is None or input_item is None:
            continue
        source_path = Path(input_item["path"])
        excluded_sources.append({
            "path": item.get("path"),
            "reason": "empty_source" if source_path.stat().st_size == 0 else "explicit_exclusion",
            "decoded": item.get("decoded") is True,
            "sha256_before": item.get("sha256_before"),
            "sha256_after": item.get("sha256_after"),
        })

    active_paths_ok = {Path(item.get("path", "")).name for item in sources} == set(active_names)
    active_digests_ok = all(
        item.get("sha256_before") == inputs[Path(item["path"]).name]["sha256"]
        and item.get("sha256_after") == inputs[Path(item["path"]).name]["sha256"]
        for item in sources
    )
    excluded_ok = all(
        item["decoded"] is False
        and item["sha256_before"] in (None, "")
        and item["sha256_after"] == digest(Path(inputs[Path(item["path"]).name]["path"]))
        for item in excluded_sources
    ) and len(excluded_sources) == len(excluded_names)
    candidate = {
        "schema": 1,
        "method": "wenyuan_excluded_source_candidate_fixture_v1",
        "production_impact": "read-only projection; no manifest, runner, Stack_Report, or source write",
        "origin": {"execution": str(args.execution), "report": str(args.report)},
        "input": {
            "selected_source_count": len(inputs),
            "active_source_count": len(sources),
            "excluded_source_count": len(excluded_sources),
            "sources": sources,
            "excluded_sources": excluded_sources,
        },
        "validation": {
            "active_paths_equal_selected_minus_excluded": active_paths_ok,
            "active_digests_match": active_digests_ok,
            "excluded_records_valid": excluded_ok,
            "counts_close": len(inputs) == len(sources) + len(excluded_sources),
            "passes": active_paths_ok and active_digests_ok and excluded_ok and len(inputs) == len(sources) + len(excluded_sources),
        },
        "decision": "fixture_only; review schema and regenerate a fresh baseline before formal adoption",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(candidate["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
