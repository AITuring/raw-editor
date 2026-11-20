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
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STITCH = ROOT / "src-tauri/src/panorama_stitching/stitch_images.rs"


def main() -> int:
    checks = []
    switches = {
        "RAW_EDITOR_STACK_RAW_FIXED_GAIN": (STITCH, "fn", "test_stack_raw_fixed_gain"),
        "RAW_EDITOR_SKIP_STACK_RAW_EXPOSURE_NORMALIZATION": (STITCH, "fn", "test_skip_stack_raw_exposure_normalization"),
        "RAW_EDITOR_DISABLE_SHARED_RAW_NORMALIZATION": (STITCH, "fn", "test_disable_shared_raw_normalization"),
        "RAW_EDITOR_FOCUS_OWNERSHIP_DIAGNOSTICS": (
            ROOT / "src-tauri/src/panorama_utils/stitching/focus_stack_stitcher.rs",
            "let",
            "ownership_diagnostics",
        ),
        "RAW_EDITOR_FOCUS_TRANSLATION_DIAGNOSTICS": (
            ROOT / "src-tauri/src/panorama_utils/stitching/focus_alignment.rs",
            "let",
            "translation_diagnostics",
        ),
        "RAW_EDITOR_FOCUS_TONE_DIAGNOSTICS": (
            ROOT / "src-tauri/src/panorama_utils/stitching/focus_tone.rs",
            "let",
            "tone_diagnostics",
        ),
        "RAW_EDITOR_STATION_SHARPNESS_TRACE": (
            ROOT / "src-tauri/src/panorama_utils/stitching/focus_stack_entry.rs",
            "let",
            "sharpness_trace",
        ),
        "RAW_EDITOR_FOCUS_BOUNDARY_DEBUG": (
            ROOT / "src-tauri/src/panorama_utils/stack_pipeline/quality_gate_runner/boundary_repair.rs",
            "let",
            "debug_local_relation",
        ),
        "RAW_EDITOR_DEBUG_GROUP_GAINS": (
            ROOT / "src-tauri/src/panorama_utils/mosaic/streaming_harmonization.rs",
            "let",
            "debug_group_gains",
        ),
        "RAW_EDITOR_FOCUS_OWNERSHIP_DEBUG": (
            ROOT / "src-tauri/src/panorama_utils/stack_pipeline/focus_fuser/decisive_cells.rs",
            "let",
            "debug",
        ),
    }
    for env_name, (source_path, mode, symbol) in switches.items():
        source = source_path.read_text(encoding="utf-8")
        count = source.count(env_name)
        helper_guarded = (
            f"#[cfg(test)]\nfn {symbol}" in source
            and f"#[cfg(not(test))]\nfn {symbol}" in source
            if mode == "fn"
            else re.search(rf"#\[cfg\(test\)\]\s+let {symbol}\b", source) is not None
            and re.search(rf"#\[cfg\(not\(test\)\)\]\s+let {symbol}\b", source) is not None
        )
        checks.append(
            {
                "name": f"{env_name} is test-only",
                "status": "pass" if count == 1 and helper_guarded else "fail",
                "source_occurrences": count,
                "binding": symbol,
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
