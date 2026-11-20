#!/usr/bin/env python3
"""Audit ENG.4 strict lint and release/signing evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    lint = read(".github/workflows/lint.yml")
    release = read(".github/workflows/release.yml")
    build = read(".github/workflows/build.yml")
    checks = [
        {
            "name": "strict clippy workflow exists",
            "status": "pass" if "cargo clippy --all-targets --all-features -- -D warnings" in lint else "fail",
        },
        {
            "name": "strict clippy current checkout",
            "status": "fail",
            "command": "cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets --all-features -- -D warnings",
            "exit_code": 101,
            "diagnostics": 326,
            "target_diagnostics": {"lib": 141, "lib test": 185},
            "note": "strict lint remains nonzero; the current all-targets snapshot reports separate lib and lib test totals after cfg(test) isolation. Ordinary cargo check remains separate.",
        },
        {
            "name": "release workflow publishes artifacts",
            "status": "pass" if "gh release create" in release and "tauri-apps/tauri-action" in build else "fail",
        },
        {
            "name": "desktop release platform matrix",
            "status": "pass" if all(platform in release for platform in ["windows-latest", "macos-14", "ubuntu-22.04"]) else "fail",
            "note": "build/publish matrix exists; this is not signing evidence",
        },
        {
            "name": "macOS signing and notarization",
            "status": "open",
            "evidence": "no Apple signing identity, notarization or stapling step found in release/build workflows",
        },
        {
            "name": "Windows signing",
            "status": "open",
            "evidence": "Windows package build exists, but no certificate/signing step is recorded",
        },
        {
            "name": "Linux distribution verification",
            "status": "open",
            "evidence": "Linux artifacts are uploaded, but no install/smoke/distribution acceptance record is present",
        },
    ]
    result = {
        "schema": 1,
        "task": "ENG.4",
        "method": "strict_clippy_release_signing_audit_v1",
        "production_impact": "read_only_release_audit_no_code_change",
        "status": "open",
        "checks": checks,
        "decision": "keep ENG.4 open: strict clippy fails and signing/notarization/distribution evidence is incomplete",
        "next_step": "fix strict clippy in a scoped pass, then add verified signing/notarization/distribution acceptance evidence",
    }
    output = ROOT / "docs/benchmarks/eng4-release-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
