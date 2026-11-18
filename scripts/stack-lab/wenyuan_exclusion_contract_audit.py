#!/usr/bin/env python3
"""Compare strict and proposed excluded-source report contracts read-only."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--excluded", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execution = json.loads(args.execution.read_text())
    analysis = json.loads(args.analysis.read_text())
    report = json.loads(Path(analysis["report"]).read_text())
    excluded = set(args.excluded)
    inputs = {Path(item["path"]).name: item for item in execution["inputs"]}
    report_sources = (report.get("input") or {}).get("sources") or []
    report_by_name = {Path(item.get("path", "")).name: item for item in report_sources}
    active_names = set(inputs) - excluded
    report_active_names = set(report_by_name) - excluded
    active_digest_ok = all(
        name in report_by_name
        and report_by_name[name].get("sha256_before") == inputs[name]["sha256"]
        and report_by_name[name].get("sha256_after") == inputs[name]["sha256"]
        for name in active_names
    )
    excluded_records_valid = all(
        name in report_by_name
        and report_by_name[name].get("decoded") is False
        and report_by_name[name].get("sha256_before") in (None, "")
        and report_by_name[name].get("sha256_after") == sha(Path(inputs[name]["path"]))
        for name in excluded
        if name in inputs
    )
    result = {
        "schema": 1,
        "method": "wenyuan_excluded_source_report_contract_audit_v1",
        "production_impact": "test_only_read_only_contract_comparison_no_runner_or_manifest_write",
        "execution": str(args.execution),
        "analysis": str(args.analysis),
        "excluded": sorted(excluded),
        "strict_contract": {
            "report_paths_equal_active_inputs": set(report_by_name) == active_names,
            "analysis_report_source_digests": analysis["checks"].get("report_source_digests"),
            "passes": bool(analysis["checks"].get("report_source_digests")),
        },
        "proposed_contract": {
            "active_report_paths_equal_active_inputs": report_active_names == active_names,
            "active_source_digests_match": active_digest_ok,
            "excluded_records_are_explicit_and_empty": excluded_records_valid,
            "passes": report_active_names == active_names and active_digest_ok and excluded_records_valid,
            "rule": "exclude the named source from active report path equality, but require a report record with decoded=false, sha256_before empty, and sha256_after equal to the immutable empty-file SHA",
        },
        "tone_density_context": {
            "tone_status": report.get("tone", {}).get("status"),
            "tone_boundary_delta_e_max": report.get("tone", {}).get("boundary_delta_e", {}).get("max"),
            "output_density_status": report.get("composition", {}).get("output_density", {}).get("status"),
            "output_density_bypass_reason": report.get("composition", {}).get("output_density", {}).get("bypass_reason"),
        },
        "decision": "design_only; review proposed contract and regenerate a fresh baseline before changing formal manifest or runner behavior",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"strict": result["strict_contract"], "proposed": result["proposed_contract"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
