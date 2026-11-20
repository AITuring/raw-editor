#!/usr/bin/env python3
"""Build and validate the machine-readable N1 migration evidence ledger."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs/benchmarks/n1-12-migration-evidence-audit.json"
OUTPUT = ROOT / "docs/benchmarks/n1-12-migration-ledger.json"


def main() -> int:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    records = []
    for item in audit["migrations"]:
        evidence = item["evidence"]
        records.append(
            {
                "migration": item["migration"],
                "status": "historical_evidence_recorded",
                "evidence_mode": "historical_prose",
                "targeted_tests": bool(evidence["targeted_tests"]),
                "full_regression": bool(evidence["full_regression"]),
                "output_equivalence": bool(evidence["output_equivalence"]),
                "machine_replay": False,
                "current_wip_applies": False,
                "source_docs": audit["sources"],
            }
        )
    complete = len(records) == 10 and all(
        record["targeted_tests"]
        and record["full_regression"]
        and record["output_equivalence"]
        and record["evidence_mode"] == "historical_prose"
        and record["current_wip_applies"] is False
        for record in records
    )
    result = {
        "schema": 1,
        "task": "N1.12",
        "method": "migration_evidence_ledger_v1",
        "production_impact": "read_only_ledger_generation_no_code_change",
        "status": "pass" if complete else "open",
        "records": records,
        "summary": {
            "migration_count": len(records),
            "complete_records": sum(
                record["targeted_tests"]
                and record["full_regression"]
                and record["output_equivalence"]
                for record in records
            ),
            "machine_replay_records": sum(record["machine_replay"] for record in records),
        },
        "boundary": "historical ledger evidence does not certify later WIP; every future migration needs a new record and rerun",
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
