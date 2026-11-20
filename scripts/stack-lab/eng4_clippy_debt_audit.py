#!/usr/bin/env python3
"""Record a strict-clippy debt snapshot without hiding target boundaries.

The all-targets invocation emits separate summaries for the library and the
library test target.  Keeping those counts separate prevents a test-only
module from being mistaken for a production diagnostic.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ERROR_RE = re.compile(r"^error: ")
ARROW_RE = re.compile(r"^\s*--> (.+?):\d+:\d+")
SUMMARY_RE = re.compile(r"could not compile `raw-editor` \((lib(?: test)?)\) due to (\d+) previous errors?")


def parse_snapshot(log: str) -> tuple[int, dict[str, int], list[dict[str, int | str]]]:
    lines = log.splitlines()
    summaries = {
        match.group(1): int(match.group(2))
        for line in lines
        if (match := SUMMARY_RE.search(line))
    }
    files: Counter[str] = Counter()
    for line in lines:
        match = ARROW_RE.match(line)
        if match and not match.group(1).startswith("/private/"):
            files[match.group(1)] += 1
    top_files = [
        {"path": path, "diagnostics": count}
        for path, count in files.most_common(25)
    ]
    return sum(summaries.values()), summaries, top_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    total, targets, top_files = parse_snapshot(args.log.read_text(encoding="utf-8", errors="replace"))
    result = {
        "schema": 2,
        "task": "ENG.4",
        "method": "strict_clippy_debt_inventory_v2",
        "command": "cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets --all-features -- -D warnings",
        "exit_code": 101,
        "diagnostic_count": total,
        "target_diagnostics": targets,
        "top_files": top_files,
        "scope_note": "source_raw_direct, source_raw_direct_basin, source_raw_direct_pairs, source_raw_layer_diagnostics, feature_provider, focus_pyramid, focus_color, station_seam, and tile_source are cfg(test)-isolated research/test contracts; remaining files are not reclassified by this snapshot",
        "source_log": str(args.log),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnostic_count": total, "targets": targets}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
