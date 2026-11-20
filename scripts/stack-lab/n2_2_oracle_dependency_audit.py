#!/usr/bin/env python3
"""Static dependency-direction inventory for N2.2.

This audit is deliberately conservative.  It records the narrow release/test
boundary for the structural, owner and final-world research oracles without
parsing Rust.  A check passes only when the source contains an explicit
``#[cfg(test)]`` immediately before a module/import/call boundary (or in the
small guarded block that owns the call).  The generated JSON is evidence for
the boundary contract; it does not make any oracle a release gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]


def read_lines(relative: str) -> list[str]:
    return (ROOT / relative).read_text(encoding="utf-8").splitlines()


def occurrences(lines: Iterable[str], needle: str) -> list[int]:
    return [index for index, line in enumerate(lines, start=1) if needle in line]


def nearby_cfg_test(lines: list[str], line_number: int, lookback: int = 8) -> bool:
    start = max(0, line_number - 1 - lookback)
    window = "\n".join(lines[start : line_number - 1])
    return "#[cfg(test)]" in window and "#[cfg(not(test))]" not in window


def function_cfg_test(lines: list[str], line_number: int) -> bool:
    """Find the nearest Rust function and require its preceding cfg(test)."""
    index = line_number - 1
    while index >= 0:
        if " fn " in f" {lines[index]}" or lines[index].lstrip().startswith("fn "):
            return direct_cfg_test(lines, index + 1)
        index -= 1
    return False


def direct_cfg_test(lines: list[str], line_number: int) -> bool:
    index = line_number - 2
    while index >= 0 and not lines[index].strip():
        index -= 1
    return index >= 0 and lines[index].strip() == "#[cfg(test)]"


def check(
    checks: list[dict[str, object]],
    *,
    name: str,
    path: str,
    needle: str,
    mode: str = "direct",
    lookback: int = 8,
) -> bool:
    lines = read_lines(path)
    hits = occurrences(lines, needle)
    if not hits:
        checks.append(
            {
                "name": name,
                "status": "fail",
                "path": path,
                "needle": needle,
                "reason": "source pattern not found",
            }
        )
        return False
    if mode == "direct":
        failed = [line for line in hits if not direct_cfg_test(lines, line)]
    elif mode == "function":
        failed = [line for line in hits if not function_cfg_test(lines, line)]
    else:
        failed = [line for line in hits if not nearby_cfg_test(lines, line, lookback)]
    checks.append(
        {
            "name": name,
            "status": "pass" if not failed else "fail",
            "path": path,
            "needle": needle,
            "lines": hits,
            "guard": {
                "direct": "direct_cfg_test",
                "function": "function_cfg_test",
            }.get(mode, "nearby_cfg_test"),
            "failed_lines": failed,
        }
    )
    return not failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/benchmarks/n2-2-oracle-dependency-inventory.json",
    )
    args = parser.parse_args()

    checks: list[dict[str, object]] = []
    # Structural oracle: module declaration, facade import and the harness call.
    check(
        checks,
        name="structural module declaration",
        path="src-tauri/src/panorama_stitching.rs",
        needle='#[path = "panorama_structural_oracle.rs"]',
    )
    check(
        checks,
        name="structural facade import",
        path="src-tauri/src/panorama_stitching/stitch_images.rs",
        needle="structural_oracle,",
        mode="nearby",
        lookback=4,
    )
    check(
        checks,
        name="structural harness call",
        path="src-tauri/src/panorama_stitching/stitch_images.rs",
        needle="structural_oracle::run_structural_oracle",
        mode="nearby",
        lookback=14,
    )

    # Constrained owner oracle: module declaration and every use is test-only.
    check(
        checks,
        name="owner module declaration",
        path="src-tauri/src/panorama_utils/stitching.rs",
        needle='#[path = "owner_quality_oracle.rs"]',
    )
    check(
        checks,
        name="owner probe module declaration",
        path="src-tauri/src/panorama_utils/stitching.rs",
        needle="mod score_noise_probes;",
    )
    check(
        checks,
        name="owner probe import",
        path="src-tauri/src/panorama_utils/stitching/score_noise_probes.rs",
        needle="use super::owner_quality_oracle;",
    )
    check(
        checks,
        name="owner probe references",
        path="src-tauri/src/panorama_utils/stitching/score_noise_probes.rs",
        needle="owner_quality_oracle::",
        mode="function",
    )

    # Final-world oracle: module, runner export and the production facade call.
    check(
        checks,
        name="final-world oracle module declaration",
        path="src-tauri/src/panorama_utils/stitching.rs",
        needle='#[path = "final_world_owner_oracle.rs"]',
    )
    check(
        checks,
        name="final-world runner module declaration",
        path="src-tauri/src/panorama_utils/stitching.rs",
        needle="mod final_world_owner_runner;",
    )
    check(
        checks,
        name="final-world runner export",
        path="src-tauri/src/panorama_utils/stitching.rs",
        needle="run_final_world_owner_oracle;",
    )
    check(
        checks,
        name="final-world facade call",
        path="src-tauri/src/panorama_stitching/stitch_images.rs",
        needle="stitching::run_final_world_owner_oracle(",
        mode="nearby",
        lookback=6,
    )
    check(
        checks,
        name="final-world run reset",
        path="src-tauri/src/panorama_stitching/stitch_images.rs",
        needle="stitching::reset_final_world_owner_oracle();",
        mode="nearby",
        lookback=2,
    )

    # Tone owner evidence is a related owner oracle and must follow the same
    # direction even though N2.2's primary owner module is owner_quality_oracle.
    check(
        checks,
        name="tone owner oracle module declaration",
        path="src-tauri/src/panorama_utils/stack_pipeline/tone.rs",
        needle="mod owner_oracle;",
    )
    check(
        checks,
        name="quality-gate tone owner oracle module declaration",
        path="src-tauri/src/panorama_utils/stack_pipeline/quality_gate_runner.rs",
        needle="mod tone_owner_oracle;",
    )
    check(
        checks,
        name="tone owner field capture",
        path="src-tauri/src/panorama_utils/stack_pipeline/tone/harmonize.rs",
        needle="let owner_oracle_fields =",
        mode="nearby",
        lookback=4,
    )
    check(
        checks,
        name="tone owner snapshot publication",
        path="src-tauri/src/panorama_utils/stack_pipeline/tone/harmonize.rs",
        needle="if let Some(fields) = owner_oracle_fields",
        mode="nearby",
        lookback=4,
    )
    check(
        checks,
        name="tone owner quality-gate import",
        path="src-tauri/src/panorama_utils/stack_pipeline/quality_gate_runner/run_gate.rs",
        needle="write_tone_owner_fidelity_oracle,",
        mode="nearby",
        lookback=4,
    )

    passed = sum(item["status"] == "pass" for item in checks)
    failed = [item for item in checks if item["status"] != "pass"]
    result = {
        "schema": 1,
        "task": "N2.2",
        "method": "oracle_dependency_direction_inventory_v1",
        "production_impact": "read_only_static_source_audit",
        "status": "pass" if not failed else "open",
        "checks": checks,
        "summary": {"total": len(checks), "passed": passed, "failed": len(failed)},
        "remaining": [
            "N2.3 production invariant vs research oracle contract remains open"
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
