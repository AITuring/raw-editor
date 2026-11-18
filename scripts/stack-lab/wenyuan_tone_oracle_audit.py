#!/usr/bin/env python3
"""Classify the read-only Wenyuan tone-owner oracle result."""

import argparse
import json
from pathlib import Path


def load(path: Path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--owner-oracle", type=Path, required=True)
    parser.add_argument("--tone-final-world", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    analysis = load(args.analysis)
    oracle_doc = load(args.owner_oracle)
    oracle = oracle_doc.get("oracle") or {}
    boundary = oracle.get("boundary") or {}
    owner = oracle.get("owner_fidelity") or {}
    holdout = oracle.get("checkerboard_holdout") or {}
    final_world = load(args.tone_final_world) if args.tone_final_world else {}
    summary = final_world.get("summary") if isinstance(final_world.get("summary"), dict) else {}

    boundary_required = boundary.get("minimum_feasible_relaxation")
    owner_allowed = owner.get("maximum_feasible_relaxation")
    feasible = oracle.get("feasible_joint_solution") is True
    result = {
        "schema": 1,
        "method": "wenyuan_tone_owner_oracle_audit_v1",
        "production_impact": "read-only classification of cfg(test) tone-owner oracle; no panorama, manifest, runner, or threshold write",
        "analysis": str(args.analysis),
        "owner_oracle": str(args.owner_oracle),
        "tone_final_world": str(args.tone_final_world) if args.tone_final_world else None,
        "run": {
            "status": analysis.get("status"),
            "selected_path": analysis.get("selected_path"),
            "closure": analysis.get("closure"),
            "quality_gate": analysis.get("quality_gate"),
            "native_child_exit_code": (analysis.get("usage") or {}).get("child_exit_code"),
            "peak_rss_bytes": analysis.get("peak_rss_bytes"),
        },
        "oracle": {
            "family": oracle.get("family"),
            "feasible_joint_solution": feasible,
            "identity_fallback": oracle.get("identity_fallback"),
            "selected_relaxation": oracle.get("selected_relaxation"),
            "boundary_required_relaxation": boundary_required,
            "owner_maximum_feasible_relaxation": owner_allowed,
            "pareto_gap": oracle.get("pareto_gap"),
            "relaxation_resolution": oracle.get("relaxation_resolution"),
            "boundary_baseline_max_delta_e00": boundary.get("baseline_max_delta_e00"),
            "boundary_selected_max_delta_e00": boundary.get("selected_max_delta_e00"),
            "boundary_threshold": boundary.get("threshold"),
            "owner_baseline_max": owner.get("baseline_max"),
            "owner_selected_max": owner.get("selected_max"),
            "owner_candidate_full_max": owner.get("candidate_full_max"),
            "owner_candidate_full_violations": owner.get("candidate_full_violations"),
            "owner_formal_pass": owner.get("formal_pass"),
            "holdout_baseline_p95": holdout.get("baseline_p95"),
            "holdout_selected_p95": holdout.get("selected_p95"),
        },
        "tone_final_world": {
            "summary": summary,
        },
        "decision": (
            "hypothesis_rejected_no_joint_tone_relaxation; boundary improvement requires a relaxation outside owner-fidelity feasibility; preserve identity fallback and keep task14 at diagnosis"
            if not feasible
            else "hypothesis_not_rejected; requires a fresh full-QG experiment before any production decision"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "feasible_joint_solution": feasible,
        "boundary_required_relaxation": boundary_required,
        "owner_maximum_feasible_relaxation": owner_allowed,
        "pareto_gap": oracle.get("pareto_gap"),
        "decision": result["decision"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
