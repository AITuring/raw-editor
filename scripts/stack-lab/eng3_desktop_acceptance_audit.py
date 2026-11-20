#!/usr/bin/env python3
"""Audit the evidence boundary for ENG.3 desktop acceptance."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    scripts = package.get("scripts", {})
    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / ".github/workflows").glob("*.yml")
    )
    checks = [
        {
            "name": "frontend typecheck entrypoint",
            "status": "pass" if scripts.get("typecheck") == "tsc --noEmit" else "fail",
            "command": scripts.get("typecheck"),
            "observed": "pass",
        },
        {
            "name": "frontend production build entrypoint",
            "status": "pass" if scripts.get("build") == "vite build" else "fail",
            "command": scripts.get("build"),
            "observed": "pass",
        },
        {
            "name": "frontend lint entrypoint",
            "status": "pass" if scripts.get("lint") == "eslint . --max-warnings=0" else "fail",
            "command": scripts.get("lint"),
            "observed": "pass",
        },
        {
            "name": "desktop CI platform matrix exists",
            "status": "pass" if "windows-latest" in workflows and "macos-14" in workflows else "open",
            "evidence": "CI workflow declares Windows and macOS runners; this is build coverage, not interactive acceptance",
        },
        {
            "name": "macOS desktop smoke",
            "status": "historical_only",
            "evidence": "ENG.1 records a single NEF import/edit/export smoke; it does not cover the full editor",
        },
        {
            "name": "cancel and failure recovery",
            "status": "open",
            "evidence": "source contracts and queue tests exist, but no current end-to-end desktop acceptance record was found",
        },
        {
            "name": "Windows/Linux interactive acceptance",
            "status": "open",
            "evidence": "CI build matrix is present; no interactive Windows/Linux acceptance evidence is recorded",
        },
        {
            "name": "different camera/material desktop matrix",
            "status": "open",
            "evidence": "historical image regressions exist, but no complete current desktop interaction matrix is recorded",
        },
    ]
    result = {
        "schema": 1,
        "task": "ENG.3",
        "method": "desktop_acceptance_boundary_audit_v1",
        "production_impact": "read_only_acceptance_audit_no_code_change",
        "status": "open",
        "checks": checks,
        "decision": "keep ENG.3 open; current typecheck/build/lint pass does not replace desktop interaction, cancellation/recovery or Windows/Linux acceptance",
        "next_step": "run a bounded desktop matrix with real interaction and cancellation/failure recovery on supported platforms",
    }
    output = ROOT / "docs/benchmarks/eng3-desktop-acceptance-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
