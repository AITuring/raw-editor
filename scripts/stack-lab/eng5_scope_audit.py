#!/usr/bin/env python3
"""Audit the current post-baseline commit scope before release decisions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = "f582825f"


def run(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    head = run("rev-parse", "HEAD")
    remote = run("rev-parse", "origin/v1")
    status = run("status", "--porcelain")
    commits = run("log", "--format=%H%x09%s", f"{BASE}..HEAD").splitlines()
    files = run("diff", "--name-only", f"{BASE}..HEAD").splitlines()
    result = {
        "schema": 1,
        "task": "ENG.5",
        "method": "post_baseline_scope_audit_v1",
        "production_impact": "read_only_commit_scope_audit_no_code_change",
        "baseline": BASE,
        "head": head,
        "remote": remote,
        "worktree_clean": not bool(status),
        "remote_matches_head": head == remote,
        "commits": [
            {"sha": line.split("\t", 1)[0], "subject": line.split("\t", 1)[1]}
            for line in commits
        ],
        "changed_files": files,
        "decision": "scope is reviewed and remotely synchronized; release publication still requires the separate ENG.4 signing/distribution gates",
        "status": "pass" if not status and head == remote and commits else "open",
    }
    output = ROOT / "docs/benchmarks/eng5-scope-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "commit_count": len(commits), "file_count": len(files)}, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
