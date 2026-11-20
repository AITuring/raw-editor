#!/usr/bin/env python3
"""Audit documented N1 migration evidence without inventing acceptance."""
from __future__ import annotations
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / "docs/focus-stack-architecture.md", ROOT / "docs/archive/focus-stack-tasks-2026-11-04.md"]
MIGRATIONS = ["N1-1", "N1-2", "N1-3", "N1-4", "N1-5", "N1-6", "N1-7", "N1-8", "N1-9", "N1-10"]
PATTERNS = {
    "targeted_tests": r"定向|单测|property|测试",
    "full_regression": r"完整\s*(?:release|库测试)|release\s*全量|cargo check",
    "output_equivalence": r"等价|逐字节|digest|基线|行为.{0,8}未改变|输出.{0,8}不变|保持不变",
}
def context_for(marker: str, text: str) -> str:
    hits = [match.start() for match in re.finditer(re.escape(marker), text)]
    return "\n".join(text[max(0, start - 600): start + 5200] for start in hits)
def main() -> int:
    text = "\n".join(source.read_text(encoding="utf-8") for source in SOURCES)
    records = []
    for marker in MIGRATIONS:
        context = context_for(marker, text)
        evidence = {name: bool(re.search(pattern, context, re.IGNORECASE)) for name, pattern in PATTERNS.items()}
        records.append({"migration": marker, "status": "documented" if context else "missing", "evidence": evidence})
    result = {
        "schema": 1,
        "task": "N1.12",
        "method": "migration_evidence_prose_audit_v1",
        "production_impact": "read_only_documentation_audit_no_code_change",
        "sources": [str(source.relative_to(ROOT)) for source in SOURCES],
        "migrations": records,
        "summary": {
            "migration_count": len(records),
            "documented_count": sum(record["status"] == "documented" for record in records),
            "all_evidence_categories_present": all(all(record["evidence"].values()) for record in records if record["status"] == "documented"),
        },
        "remaining": [
            "historical evidence is prose and is not a machine-checked per-migration manifest",
            "current WIP changes after the N1-10 migration are not treated as historical output-equivalence evidence",
            "keep N1.12 open until a reviewed migration ledger and repeatable before/after comparison entrypoint exist",
        ],
        "decision": "record the documented evidence without marking N1.12 complete",
    }
    output = ROOT / "docs/benchmarks/n1-12-migration-evidence-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
if __name__ == "__main__":
    raise SystemExit(main())
