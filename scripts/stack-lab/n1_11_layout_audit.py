#!/usr/bin/env python3
"""Inventory the remaining N1.11 long sources without changing their code.

The source-layout generator is the authority for facade/child maps.  This
small audit records the current >1000-line files and uses a deliberately
conservative brace scan to find functions/components that are themselves over
1000 lines.  The result is a review queue, not an automatic split request.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUFFIXES = {".rs", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".css"}
FUNCTION_RE = re.compile(
    r"^\s*(?:(?:pub(?:\([^)]*\))?|async|unsafe|const)\s+)*"
    r"(?:fn|function)\s+([A-Za-z_$][\w$]*)"
)
COMPONENT_RE = re.compile(r"^\s*(?:export\s+)?(?:const|class)\s+([A-Z][\w$]*)")


def tracked_sources() -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return [
        ROOT / relative
        for relative in output
        if (ROOT / relative).is_file() and (ROOT / relative).suffix in SOURCE_SUFFIXES
    ]


def brace_scan(lines: list[str], start: int) -> int | None:
    """Return an approximate span for one declaration, or None if incomplete."""
    depth = 0
    opened = False
    for index in range(start, len(lines)):
        # Strip line comments and quoted strings enough to avoid common false
        # braces while keeping this dependency-free and intentionally reviewable.
        line = re.sub(r"//.*$", "", lines[index])
        line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
        depth += line.count("{") - line.count("}")
        opened = opened or "{" in line
        if opened and depth <= 0:
            return index + 1 - start
    return None


def long_declarations(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    for index, line in enumerate(lines):
        match = FUNCTION_RE.match(line) or COMPONENT_RE.match(line)
        if not match:
            continue
        span = brace_scan(lines, index)
        if span is not None and span > 1000:
            result.append({"name": match.group(1), "line": index + 1, "approx_lines": span})
    return result


def main() -> int:
    sources = tracked_sources()
    long_files = [
        {
            "path": str(path.relative_to(ROOT)),
            "lines": len(path.read_text(encoding="utf-8").splitlines()),
            "long_declarations": long_declarations(path),
        }
        for path in sources
        if len(path.read_text(encoding="utf-8").splitlines()) > 1000
    ]
    long_files.sort(key=lambda item: (-int(item["lines"]), str(item["path"])))
    result = {
        "schema": 1,
        "task": "N1.11",
        "method": "source_layout_long_declaration_inventory_v1",
        "production_impact": "read_only_inventory_no_code_change",
        "source_layout_command": "python3 scripts/source-layout.py",
        "long_files": long_files,
        "summary": {
            "long_file_count": len(long_files),
            "long_declaration_count": sum(len(item["long_declarations"]) for item in long_files),
        },
        "remaining": [
            "decide individually whether the listed production files are intentional facades/entrypoints or need a bounded split",
            "add a separate import-direction contract before marking N1.11 complete",
        ],
        "decision": "keep N1.11 open; preserve current behavior and use this list for a reviewed follow-up",
    }
    output = ROOT / "docs/benchmarks/n1-11-layout-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
