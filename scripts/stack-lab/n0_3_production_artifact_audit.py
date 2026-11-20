#!/usr/bin/env python3
"""Audit the narrow N0.3 production/test boundary.

N0.3 is intentionally broader than one environment switch: preview and
Stack_Report fields are valid product contracts, while acceptance diagnostics
and experiment knobs must stay out of a shipped renderer.  This audit records
the small boundary cleanup that is mechanically provable and keeps the
remaining legacy knobs explicit instead of deleting them speculatively.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STITCH = ROOT / "src-tauri/src/panorama_stitching/stitch_images.rs"


def main() -> int:
    source = STITCH.read_text(encoding="utf-8")
    checks = []
    switches = {
        "RAW_EDITOR_STACK_RAW_FIXED_GAIN": "test_stack_raw_fixed_gain",
        "RAW_EDITOR_SKIP_STACK_RAW_EXPOSURE_NORMALIZATION": "test_skip_stack_raw_exposure_normalization",
        "RAW_EDITOR_DISABLE_SHARED_RAW_NORMALIZATION": "test_disable_shared_raw_normalization",
    }
    for env_name, helper in switches.items():
        count = source.count(env_name)
        helper_guarded = (
            f"#[cfg(test)]\nfn {helper}" in source
            and f"#[cfg(not(test))]\nfn {helper}" in source
        )
        checks.append(
            {
                "name": f"{env_name} is test-only",
                "status": "pass" if count == 1 and helper_guarded else "fail",
                "source_occurrences": count,
                "helper": helper,
            }
        )

    # These are deliberate follow-up items: they need an explicit product
    # decision before removal because they affect comparison paths or report
    # compatibility.  The audit must not silently treat them as cleaned.
    remaining = [
        "RAW_EDITOR_FINAL_PANORAMA_SHARPEN_AMOUNT remains a legacy comparison-path knob",
        "acceptance/diagnostic RAW_EDITOR_* switches remain in test-only modules and need a complete inventory",
        "preview and Stack_Report artifact fields are product contracts; no speculative deletion is justified",
    ]
    passed = sum(item["status"] == "pass" for item in checks)
    result = {
        "schema": 1,
        "task": "N0.3",
        "method": "production_artifact_boundary_audit_v1",
        "production_impact": "one test-only boundary cleanup; no preview/report schema change",
        "status": "partial" if passed == len(checks) else "open",
        "checks": checks,
        "summary": {"total": len(checks), "passed": passed, "failed": len(checks) - passed},
        "remaining": remaining,
        "decision": "keep N0.3 open; retain historical evidence and product preview/report contracts",
    }
    output = ROOT / "docs/benchmarks/n0-3-production-artifact-audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0 if result["status"] != "open" or result["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
