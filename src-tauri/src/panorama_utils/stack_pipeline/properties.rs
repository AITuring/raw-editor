//! `proptest` property tests for the Stack_Pipeline Correctness Properties of
//! the `layered-camera-group-focus-stitching` design document.
//!
//! One property test per numbered property, each configured with at least 100
//! cases.  Failing cases are persisted under `proptest-regressions/`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Matrix3, Point2, Point3};
use regex::Regex;

use super::degradation::{FAILURE_REASONS, UNMEASURABLE_REASONS};
use super::report::StackReport;

mod warp_tone;

mod tile_compositor;

use tile_compositor::render_layered_ownership;

mod acceptance_verdict;

use acceptance_verdict::stack_acceptance_failures;

mod acceptance_inputs;

use acceptance_inputs::{
    acceptance_source_snapshot, frozen_station_source_paths, parse_station_pair_cases,
    select_unique_stack_report, validate_acceptance_directories,
};

mod run_outcome;

mod reproducibility;

use reproducibility::{
    SyntheticGeometryOutcome, determinism_images, fit_synthetic_matches, synthetic_geometry_outcome,
};

mod grouping_invariance;

mod tile_structure;

use tile_structure::{CACHE_PROPERTY_MAX_SIDE, CACHE_PROPERTY_MIN_SIDE};

mod tile_cache;

mod station_membership;

use station_membership::{
    REQUIRED_ANCHOR_SHARPNESS_TIE, REQUIRED_MAX_ACCUMULATED_MOTION_RATIO,
    REQUIRED_MAX_STATION_MEMBERS, accepted_link_components,
};

mod station_splits;

mod station_anchor;

mod intra_station_registration;

mod ownership_cells;

use ownership_cells::probe_texture;

mod graph_cut;

mod station_ownership;

use station_ownership::{StationDraw, arb_station_draw, fuse_station};

mod single_frame;

use single_frame::{
    degraded_station_info, render_station_tile, single_frame_info, single_frame_source,
};

mod station_failures;

mod capture_topology;

use capture_topology::{above, accepted_relation_measurements, below};

mod station_pose_graph;

use station_pose_graph::{matrix_bits, translation_pose};

mod closure;

mod quality_gate_checks;

/// The exact shape requirement 12.7 fixes for a machine readable failure reason.
fn reason_identifier_pattern() -> Regex {
    Regex::new(r"^[a-z0-9_]{1,64}$").expect("the reason identifier pattern must compile")
}

#[test]
#[ignore = "acceptance harness requires the supplied real dataset"]
fn stack_acceptance_harness() {
    let source_dir = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_SOURCE_DIR")
        .map(PathBuf::from)
        .expect("RAW_EDITOR_STACK_ACCEPTANCE_SOURCE_DIR is required");
    let report_dir = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_REPORT_DIR")
        .map(PathBuf::from)
        .expect("RAW_EDITOR_STACK_ACCEPTANCE_REPORT_DIR is required");
    let (source_dir, report_dir, paths) = validate_acceptance_directories(&source_dir, &report_dir)
        .unwrap_or_else(|error| panic!("acceptance directory validation failed: {error}"));
    let before = acceptance_source_snapshot(&source_dir);
    let source_strings = paths
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    // The harness owns this process-wide variable for the duration of the ignored run.
    unsafe {
        std::env::set_var("RAW_EDITOR_STACK_REPORT_DIR", &report_dir);
    }
    crate::sidecar_storage::initialize(Path::new(
        "/private/tmp/raw-editor-stack-acceptance-sidecars",
    ))
    .expect("acceptance sidecar storage should initialize");
    let app = tauri::test::mock_app();
    let pipeline_result = crate::panorama_stitching::stitch_images_with_options(
        source_strings,
        app.handle().clone(),
        crate::panorama_stitching::AlignmentMode::Auto,
        crate::panorama_stitching::BlendMode::FocusStack,
        "test-stack-acceptance-progress",
    );
    let after = acceptance_source_snapshot(&source_dir);
    assert_eq!(
        before, after,
        "acceptance source directory changed during run"
    );
    let report_path = select_unique_stack_report(&report_dir)
        .unwrap_or_else(|error| panic!("acceptance report selection failed: {error}"));
    let report: StackReport = serde_json::from_slice(
        &fs::read(&report_path).expect("acceptance report should be readable"),
    )
    .expect("acceptance report should match Stack_Report schema");
    if let Err(error) = &pipeline_result {
        eprintln!("acceptance pipeline stopped: {error}");
    }
    let failures = stack_acceptance_failures(&report);
    if !failures.is_empty() {
        eprintln!("acceptance unmet conditions: {failures:?}");
    }
    assert!(
        failures.is_empty() && pipeline_result.is_ok(),
        "acceptance conditions were not met"
    );
}

#[test]
#[ignore = "pair evidence harness requires the supplied real dataset"]
fn stack_station_pair_evidence_harness() {
    const F1_CANDIDATES: &str = "17-24,9-17,1-2,21-23,24-25";
    const FIRST_EIGHT: &str = "17-24,9-17,0-1,4-8,1-2,21-23,24-25,19-23";
    const F_DIRECT_21_22_STAGE: &str = "F_DIRECT_21_22";
    const STRUCTURAL_ORACLE_STAGE: &str = "A_STRUCTURAL";
    const POSITIVE_CONTROLS: [(usize, usize); 4] = [(0, 9), (4, 6), (18, 25), (21, 22)];
    const NEGATIVE_CONTROL: (usize, usize) = (7, 19);

    let source_dir = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_SOURCE_DIR")
        .map(PathBuf::from)
        .expect("RAW_EDITOR_STACK_ACCEPTANCE_SOURCE_DIR is required");
    let report_root = std::env::var_os("RAW_EDITOR_STACK_PAIR_REPORT_DIR")
        .map(PathBuf::from)
        .expect("RAW_EDITOR_STACK_PAIR_REPORT_DIR is required");
    let stage = std::env::var("RAW_EDITOR_STACK_SOURCE_RAW_DIRECT_STAGE")
        .unwrap_or_else(|_| "F1".to_string());
    let default_cases = match stage.as_str() {
        "F2" => FIRST_EIGHT,
        F_DIRECT_21_22_STAGE | STRUCTURAL_ORACLE_STAGE => "21-22",
        _ => F1_CANDIDATES,
    };
    let requested =
        std::env::var("RAW_EDITOR_STACK_PAIR_CASES").unwrap_or_else(|_| default_cases.to_string());
    let cases = parse_station_pair_cases(&requested);
    assert!(!cases.is_empty(), "at least one pair case is required");
    let (source_dir, report_root, _) = validate_acceptance_directories(&source_dir, &report_root)
        .unwrap_or_else(|error| panic!("pair harness directory validation failed: {error}"));
    assert!(
        fs::read_dir(&report_root).unwrap().next().is_none(),
        "pair report directory must be fresh: {}",
        report_root.display()
    );
    let before = acceptance_source_snapshot(&source_dir);
    crate::sidecar_storage::initialize(Path::new(
        "/private/tmp/raw-editor-stack-pair-evidence-sidecars",
    ))
    .expect("pair evidence sidecar storage should initialize");
    unsafe {
        std::env::set_var("RAW_EDITOR_STACK_PAIR_HARNESS_STOP_AFTER_RELATIONS", "1");
        std::env::set_var("RAW_EDITOR_STACK_SOURCE_RAW_LAYER_DIAGNOSTICS", "1");
    }

    if stage == STRUCTURAL_ORACLE_STAGE {
        assert_eq!(cases, vec![(21, 22)], "A oracle scope is fixed to 21-22");
        unsafe {
            std::env::set_var("RAW_EDITOR_STACK_STRUCTURAL_ORACLE", "1");
            std::env::remove_var("RAW_EDITOR_STACK_SOURCE_RAW_DIRECT_DIAGNOSTICS");
        }
        let mut structural_observations = Vec::new();
        let mut structural_hard_gate_passed = true;
        for (pair, control) in [((21, 22), "positive"), (NEGATIVE_CONTROL, "negative")] {
            let case_dir = report_root.join(format!("{}-{}", pair.0, pair.1));
            fs::create_dir(&case_dir).expect("create fresh structural pair report directory");
            unsafe {
                std::env::set_var("RAW_EDITOR_STACK_REPORT_DIR", &case_dir);
            }
            let mut paths = frozen_station_source_paths(&source_dir, pair.0);
            paths.extend(frozen_station_source_paths(&source_dir, pair.1));
            paths.sort();
            let source_strings = paths
                .iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            let app = tauri::test::mock_app();
            let result = crate::panorama_stitching::stitch_images_with_options(
                source_strings,
                app.handle().clone(),
                crate::panorama_stitching::AlignmentMode::Auto,
                crate::panorama_stitching::BlendMode::FocusStack,
                "test-stack-structural-oracle-progress",
            );
            assert!(
                result.as_ref().is_err_and(|error| error.contains(
                    "structural oracle harness stopped before production relation selection"
                )),
                "pair {pair:?} did not stop at the structural oracle: {}",
                result
                    .as_ref()
                    .err()
                    .map_or("unexpected success", String::as_str)
            );
            let oracle_path = case_dir.join("structural-oracle.json");
            let oracle: serde_json::Value = serde_json::from_slice(
                &fs::read(&oracle_path).expect("structural oracle report should be readable"),
            )
            .expect("structural oracle report schema");
            let accepted = oracle["accepted"].as_bool().unwrap_or(false);
            let elapsed_ms = oracle["elapsed_ms"].as_u64().unwrap_or(u64::MAX);
            let extra_memory = oracle["estimated_extra_memory_bytes"]
                .as_u64()
                .unwrap_or(u64::MAX);
            assert!(
                elapsed_ms < 30_000,
                "pair {pair:?} structural runtime {elapsed_ms}ms"
            );
            assert!(
                extra_memory < 256 * 1024 * 1024,
                "pair {pair:?} structural memory estimate {extra_memory} bytes"
            );
            let empty_rejection_models = |direction: &serde_json::Value| {
                direction["models"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|model| {
                        model["rejection_reasons"]
                            .as_array()
                            .is_some_and(Vec::is_empty)
                    })
                    .count()
            };
            if control == "positive" {
                structural_hard_gate_passed &= accepted;
                if accepted {
                    for direction in [&oracle["forward"], &oracle["reverse"]] {
                        let model = direction["models"]
                            .as_array()
                            .and_then(|models| {
                                models.iter().find(|model| {
                                    model["rejection_reasons"]
                                        .as_array()
                                        .is_some_and(Vec::is_empty)
                                })
                            })
                            .expect("21-22 must have an empty-rejection model in each direction");
                        assert!(model["correspondences"].as_array().unwrap().len() >= 24);
                        assert!(model["median_symmetric_error_px"].as_f64().unwrap() <= 3.0);
                        assert!((0.95..=1.05).contains(&model["scale_ratio"].as_f64().unwrap()));
                        assert!(model["covered_overlap_hull"].as_f64().unwrap() >= 0.20);
                        assert!(model["preserves_convex_orientation"].as_bool().unwrap());
                    }
                    assert!(
                        oracle["forward_reverse_five_point_closure_px"]
                            .as_f64()
                            .unwrap()
                            <= 1.0
                    );
                    assert!(oracle["canonical_swap_disagreement_px"].as_f64().unwrap() <= 1.0);
                }
            } else {
                let negative_passed = !accepted
                    && empty_rejection_models(&oracle["forward"]) == 0
                    && empty_rejection_models(&oracle["reverse"]) == 0;
                structural_hard_gate_passed &= negative_passed;
            }
            println!("STRUCTURAL_ORACLE {}", oracle);
            structural_observations.push(serde_json::json!({
                "pair": format!("{}-{}", pair.0, pair.1),
                "control": control,
                "report": oracle_path,
                "oracle": oracle,
            }));
        }
        assert_eq!(
            acceptance_source_snapshot(&source_dir),
            before,
            "frozen Source_RAW directory changed during structural oracle"
        );
        let summary = serde_json::json!({
            "stage": STRUCTURAL_ORACLE_STAGE,
            "requested": requested,
            "raw_digest_unchanged": true,
            "network_requests": 0,
            "hard_gate_passed": structural_hard_gate_passed,
            "observations": structural_observations,
        });
        fs::write(
            report_root.join("structural-oracle-summary.json"),
            serde_json::to_vec_pretty(&summary).unwrap(),
        )
        .expect("write structural oracle summary");
        assert!(
            structural_hard_gate_passed,
            "traditional-structure A oracle hard gate failed; see {}",
            report_root.join("structural-oracle-summary.json").display()
        );
        return;
    }

    let mut observations = Vec::new();
    let mut run_case = |pair: (usize, usize), control: &str| {
        let case_dir = report_root.join(format!("{}-{}", pair.0, pair.1));
        fs::create_dir(&case_dir).expect("create fresh pair report directory");
        unsafe {
            std::env::set_var("RAW_EDITOR_STACK_REPORT_DIR", &case_dir);
        }
        let mut paths = frozen_station_source_paths(&source_dir, pair.0);
        paths.extend(frozen_station_source_paths(&source_dir, pair.1));
        paths.sort();
        let expected_source_count = paths.len();
        let source_strings = paths
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let app = tauri::test::mock_app();
        let result = crate::panorama_stitching::stitch_images_with_options(
            source_strings,
            app.handle().clone(),
            crate::panorama_stitching::AlignmentMode::Auto,
            crate::panorama_stitching::BlendMode::FocusStack,
            "test-stack-pair-evidence-progress",
        );
        let reached_relations = result
            .as_ref()
            .is_err_and(|error| error.contains("pair evidence harness stopped"));
        assert!(
            reached_relations,
            "pair {pair:?} did not reach Station_Relation: {}",
            result
                .as_ref()
                .err()
                .map_or("unexpected success", String::as_str)
        );
        let report_path = select_unique_stack_report(&case_dir)
            .unwrap_or_else(|error| panic!("pair {pair:?} report selection failed: {error}"));
        let report: StackReport = serde_json::from_slice(
            &fs::read(&report_path).expect("pair report should be readable"),
        )
        .expect("pair report schema");
        assert_eq!(
            report.input.source_count, expected_source_count,
            "pair {pair:?}"
        );
        assert_eq!(report.grouping.station_count, 2, "pair {pair:?}");
        assert!(report.input.sources.iter().all(|source| {
            source.decoded
                && !source.sha256_before.is_empty()
                && source.sha256_before == source.sha256_after
        }));
        let global = report
            .station_relations
            .candidates
            .iter()
            .find(|candidate| candidate.seed == "global_coverage_similarity");
        let global_pass = global.is_some_and(|candidate| {
            !candidate.global_proposals.is_empty()
                && candidate.hypotheses.iter().any(|hypothesis| {
                    hypothesis.selected && hypothesis.rejection_reasons.is_empty()
                })
        });
        let c_clusters = report
            .station_relations
            .source_raw_pair_diagnostics
            .iter()
            .flat_map(|pair| pair.consensus_clusters.iter())
            .collect::<Vec<_>>();
        let c_pass = c_clusters.iter().any(|cluster| {
            cluster.independent_support >= 2 && cluster.scale_compliant && cluster.spatially_broad
        });
        let c_max_support = c_clusters
            .iter()
            .map(|cluster| cluster.independent_support)
            .max()
            .unwrap_or_default();
        let c_layer_models = report
            .station_relations
            .source_raw_pair_diagnostics
            .iter()
            .flat_map(|pair| pair.layer_models.iter())
            .collect::<Vec<_>>();
        let c_fitted_layer_pairs = c_layer_models
            .iter()
            .filter(|model| model.model_fitted)
            .count();
        let direct_pairs = &report.station_relations.source_raw_direct_diagnostics;
        let direct_observations = direct_pairs
            .iter()
            .flat_map(|pair| pair.observations.iter())
            .collect::<Vec<_>>();
        let direct_clusters = direct_pairs
            .iter()
            .flat_map(|pair| pair.consensus_clusters.iter())
            .collect::<Vec<_>>();
        let direct_pass = direct_clusters.iter().any(|cluster| {
            cluster.independent_support >= 2
                && cluster.station_inliers >= 24
                && cluster.station_median_error_px <= 3.0
                && (0.95..=1.05).contains(&cluster.station_scale_ratio)
                && cluster.station_spatial_support >= 0.20
                && cluster.station_preserves_convex_orientation
                && cluster.canonical_disagreement_px <= 3.0
                && cluster.station_bidirectional_closure_px <= 1.0
                && cluster.station_rejection_reasons.is_empty()
        });
        let direct_elapsed_ms = direct_pairs.iter().map(|pair| pair.elapsed_ms).sum::<u64>();
        let direct_peak_working_set_bytes = direct_pairs
            .iter()
            .map(|pair| pair.peak_working_set_bytes)
            .max()
            .unwrap_or_default();
        let accepted = !report.station_relations.accepted.is_empty();
        let selected = global.and_then(|candidate| {
            candidate
                .hypotheses
                .iter()
                .find(|hypothesis| hypothesis.selected)
        });
        let observation = serde_json::json!({
            "pair": format!("{}-{}", pair.0, pair.1),
            "control": control,
            "accepted": accepted,
            "global_pass": global_pass,
            "source_raw_consensus_potential": c_pass,
            "source_raw_fitted_layer_pairs": c_fitted_layer_pairs,
            "source_raw_max_independent_support": c_max_support,
            "source_raw_consensus_clusters": c_clusters,
            "source_raw_layer_attrition": c_layer_models,
            "source_raw_direct_pass": direct_pass,
            "source_raw_direct_observations": direct_observations,
            "source_raw_direct_clusters": direct_clusters,
            "source_raw_direct_elapsed_ms": direct_elapsed_ms,
            "source_raw_direct_peak_working_set_bytes": direct_peak_working_set_bytes,
            "failure_stage": global.map(|candidate| candidate.failure_stage.clone()),
            "proposal_count": global.map_or(0, |candidate| candidate.global_proposals.len()),
            "selected": selected,
            "report": report_path,
        });
        println!("PAIR_EVIDENCE {}", observation);
        observations.push((
            pair,
            control.to_string(),
            accepted,
            c_pass,
            direct_pass,
            observation,
        ));
    };

    for &pair in &cases {
        run_case(
            pair,
            if stage == F_DIRECT_21_22_STAGE {
                "positive"
            } else {
                "candidate"
            },
        );
    }
    if stage != F_DIRECT_21_22_STAGE {
        for pair in POSITIVE_CONTROLS {
            run_case(pair, "positive");
        }
    }
    run_case(NEGATIVE_CONTROL, "negative");

    assert_eq!(
        acceptance_source_snapshot(&source_dir),
        before,
        "frozen Source_RAW directory changed during pair harness"
    );
    for (pair, control, accepted, c_pass, direct_pass, _) in &observations {
        if control == "positive" {
            assert!(*accepted, "positive control {pair:?} regressed");
        } else if control == "negative" {
            assert!(
                !*accepted && !*c_pass && !*direct_pass,
                "negative control {pair:?} was accepted or produced C/F consensus"
            );
        }
    }
    let candidate_successes = observations
        .iter()
        .filter(|(_, control, _, _, direct_pass, _)| control == "candidate" && *direct_pass)
        .map(|(pair, _, _, _, _, _)| *pair)
        .collect::<Vec<_>>();
    let f1_required_success = candidate_successes
        .iter()
        .any(|pair| matches!(*pair, (1, 2) | (21, 23) | (24, 25)));
    let no_overlap_success = candidate_successes
        .iter()
        .any(|pair| matches!(*pair, (24, 25) | (21, 23)));
    let failure_classes = candidate_successes
        .iter()
        .map(|pair| match *pair {
            (24, 25) | (21, 23) => "no_predicted_overlap",
            (1, 2) | (0, 1) | (4, 8) | (19, 23) => "grid_bidirectional",
            (17, 24) | (9, 17) => "fitted_scale_hull",
            _ => "other",
        })
        .collect::<BTreeSet<_>>();
    let hard_gate_passed = match stage.as_str() {
        "F2" => candidate_successes.len() >= 3 && no_overlap_success && failure_classes.len() >= 2,
        F_DIRECT_21_22_STAGE => {
            observations
                .iter()
                .any(|(pair, control, accepted, _, direct_pass, _)| {
                    *pair == (21, 22) && control == "positive" && *accepted && *direct_pass
                })
        }
        _ => f1_required_success,
    };
    let summary = serde_json::json!({
        "stage": stage,
        "requested": requested,
        "candidate_successes": candidate_successes,
        "f1_required_success": f1_required_success,
        "no_overlap_success": no_overlap_success,
        "failure_classes": failure_classes,
        "raw_digest_unchanged": true,
        "network_requests": 0,
        "hard_gate_passed": hard_gate_passed,
        "observations": observations.into_iter().map(|entry| entry.5).collect::<Vec<_>>(),
    });
    fs::write(
        report_root.join("pair-evidence-summary.json"),
        serde_json::to_vec_pretty(&summary).unwrap(),
    )
    .expect("write pair evidence summary");
    assert!(
        hard_gate_passed,
        "pair-level {stage} hard gate failed: {summary}"
    );
}

/// Every identifier a Stack_Report is allowed to carry as a failure or
/// unmeasurable reason.
fn frozen_reason_identifiers() -> BTreeSet<&'static str> {
    FAILURE_REASONS
        .iter()
        .map(|(reason, _)| *reason)
        .chain(UNMEASURABLE_REASONS.iter().copied())
        .collect()
}

fn map_point(transform: &Matrix3<f64>, point: Point2<f64>) -> Option<Point2<f64>> {
    let mapped = transform * Point3::new(point.x, point.y, 1.0);
    (mapped.z.abs() > 1e-8).then(|| Point2::new(mapped.x / mapped.z, mapped.y / mapped.z))
}

/// `SplitMix64`, so a drawn seed expands into a whole grid of per-cell
/// measurements as a pure function on every platform.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

/// Uniform value in `[0, 1)` from the mixer above.
fn unit_from(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}
