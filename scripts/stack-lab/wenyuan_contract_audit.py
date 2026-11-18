#!/usr/bin/env python3
"""Read-only shadow-contract audit for Wenyuan's empty source."""
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    entry = next(item for item in manifest["datasets"] if item["id"] == "wenyuan")
    execution = json.loads(args.execution.read_text())
    analysis = json.loads(args.analysis.read_text())
    report = json.loads(Path(analysis["report"]).read_text())
    empty_path = next(
        Path(item["path"])
        for item in execution["inputs"]
        if Path(item["path"]).name == "DSC_4885.NEF"
    )
    report_sources = (report.get("input") or {}).get("sources") or []
    report_names = {Path(item.get("path", "")).name for item in report_sources}
    proposed = list(entry.get("expectedSkipped", []))
    if "DSC_4885.NEF" not in proposed:
        proposed.append("DSC_4885.NEF")
    result = {
        "schema": 1,
        "method": "wenyuan_empty_source_shadow_contract_audit_v1",
        "production_impact": "test_only_read_only_manifest_diff_no_manifest_write_no_baseline_rewrite",
        "manifest": str(args.manifest),
        "execution": str(args.execution),
        "analysis": str(args.analysis),
        "empty_source": {
            "path": str(empty_path),
            "exists": empty_path.is_file(),
            "bytes": empty_path.stat().st_size if empty_path.is_file() else None,
            "sha256": sha(empty_path) if empty_path.is_file() else None,
            "manifest_expected_skipped_before": entry.get("expectedSkipped", []),
            "manifest_expected_skipped_shadow": proposed,
        },
        "report_source": {
            "reported": "DSC_4885.NEF" in report_names,
            "report_record": next(
                (item for item in report_sources if Path(item.get("path", "")).name == "DSC_4885.NEF"),
                None,
            ),
            "report_source_count": len(report_sources),
        },
        "checks": {
            "formal_manifest_declares_exclusion": "DSC_4885.NEF" in entry.get("expectedSkipped", []),
            "shadow_manifest_declares_exclusion": "DSC_4885.NEF" in proposed,
            "native_decode_failed": any(
                item.get("path", "").endswith("DSC_4885.NEF") and item.get("decoded") is False
                for item in report_sources
            ),
            "formal_report_source_digests": analysis["checks"].get("report_source_digests"),
        },
        "decision": "do_not_edit_formal_manifest_in_this_audit; a reviewed manifest update plus fresh baseline is required before treating the empty source as expected exclusion",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
