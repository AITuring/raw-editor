#!/usr/bin/env python3
"""Audit the N2.3 boundary between production gates and research harnesses.

The audit intentionally does not change the release workflow.  It records two
independent facts: the pair-evidence harness is not referenced by release/CI
workflows, while the current quality-contract source still contains both
production and research-only assertions.  The latter keeps N2.3 open.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    package = json.loads(read("package.json"))
    scripts = package.get("scripts", {})
    pair_command = scripts.get("stack-pair-evidence:check")
    production_quality_command = scripts.get("focus-stack-quality:check")
    research_quality_command = scripts.get("focus-stack-research:check")
    workflows = sorted((ROOT / ".github/workflows").glob("*"))
    workflow_hits = {
        path.relative_to(ROOT).as_posix(): [
            line_no
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if "stack-pair-evidence:check" in line or "stack_station_pair_evidence_harness" in line
        ]
        for path in workflows
        if path.is_file()
    }
    workflow_hits = {path: lines for path, lines in workflow_hits.items() if lines}

    quality_contract = read("tests/focus-stack-quality-contract.mjs")
    research_flag = "const researchContract = process.argv.includes('--research');" in quality_contract
    research_markers = [
        "owner Source_RAW oracle",
        "final-world Source_RAW optimizer",
        "structural oracle",
        "tone owner",
        "research",
    ]
    research_hits = {
        marker: [
            line_no
            for line_no, line in enumerate(quality_contract.splitlines(), 1)
            if marker.lower() in line.lower()
        ]
        for marker in research_markers
    }
    research_hits = {marker: lines for marker, lines in research_hits.items() if lines}
    quality_lines = quality_contract.splitlines()
    guarded_markers = {
        marker: all(
            any("if (researchContract) {" in quality_lines[index - 1] for index in range(1, line))
            for line in lines
        )
        for marker, lines in research_hits.items()
        if marker in {"owner Source_RAW oracle", "final-world Source_RAW optimizer"}
    }

    checks = [
        {
            "name": "pair harness is explicitly research-only in package scripts",
            "status": "pass" if isinstance(pair_command, str) and "--ignored" in pair_command else "fail",
            "value": pair_command,
        },
        {
            "name": "release and CI workflows do not invoke pair harness",
            "status": "pass" if not workflow_hits else "fail",
            "workflow_hits": workflow_hits,
        },
        {
            "name": "quality contract contains production and research assertions",
            "status": (
                "pass"
                if research_flag
                and isinstance(production_quality_command, str)
                and "--research" not in production_quality_command
                and isinstance(research_quality_command, str)
                and "--research" in research_quality_command
                and research_hits
                and all(guarded_markers.values())
                else "open" if research_hits else "fail"
            ),
            "research_markers": research_hits,
            "research_marker_guards": guarded_markers,
            "production_contract_entry": production_quality_command,
            "research_contract_entry": research_quality_command,
            "research_flag_guarded": research_flag,
        },
    ]
    status = "pass" if all(item["status"] == "pass" for item in checks) else "open"
    result = {
        "schema": 1,
        "task": "N2.3",
        "method": "production_research_release_gate_audit_v1",
        "production_impact": "read_only_source_and_workflow_audit",
        "status": status,
        "checks": checks,
        "decision": (
            "pair evidence remains outside release/CI workflows; production quality checks and research oracle checks use separate package entries"
            if status == "pass"
            else "pair evidence remains outside release/CI workflows; N2.3 stays open until production and research quality-contract assertions are split"
        ),
        "next_step": (
            "keep the research entry opt-in and out of release workflows"
            if status == "pass"
            else "add a production-only contract entry and keep research oracle assertions behind a separate opt-in command"
        ),
    }
    output = ROOT / "docs/benchmarks/n2-3-release-gate-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "workflow_hits": len(workflow_hits), "research_markers": len(research_hits)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
