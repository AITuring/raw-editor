//! Capture-group pose solving: disconnected station graphs, coarse consensus and its conflicts,
//! the station scale contract, non-invertible canonical transforms and spanning-tree tie-breaks.

use super::*;

#[test]
fn disconnected_station_graph_rejects_full_mosaic_and_reports_components() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    degradation::reset_run_ledger();
    let mut images = vec![
        focus_test_image(0, "/synthetic/station-a-1.nef"),
        focus_test_image(1, "/synthetic/station-a-2.nef"),
        focus_test_image(2, "/synthetic/station-b-1.nef"),
        focus_test_image(3, "/synthetic/station-b-2.nef"),
    ];
    for image in &mut images {
        image.focal_length_35mm = Some(90.0);
    }
    // Each pair is a valid focus bracket, but there is deliberately no
    // accepted Station_Relation between the two Capture_Stations.
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(5.0, 0.0, 40)),
        ((2, 3), observed_translation_match(5.0, 0.0, 40)),
    ]);
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let locked = HashMap::from([
        (0, translation(0.0)),
        (1, translation(-5.0)),
        (2, translation(-300.0)),
        (3, translation(-305.0)),
    ]);
    let mut geometry_verified = false;
    let mut closure_report = stack_report::ClosureReport::default();
    let mut relation_report = stack_report::StationRelationsReport::default();

    let solved = solve_focus_capture_group_poses_with_report(
        &images,
        &matches,
        &locked,
        0,
        &mut geometry_verified,
        &mut closure_report,
        &mut relation_report,
    );

    assert!(
        solved.is_empty(),
        "disconnected geometry is not a pose fallback"
    );
    assert!(!geometry_verified);
    assert_eq!(relation_report.connectivity.components, 2);
    assert_eq!(
        relation_report.connectivity.component_member_counts,
        vec![2, 2]
    );
    let canonical_components = |components: &[Vec<String>]| {
        let mut components = components
            .iter()
            .map(|members| {
                let mut members = members.clone();
                members.sort();
                members
            })
            .collect::<Vec<_>>();
        components.sort();
        components
    };
    assert_eq!(
        canonical_components(&relation_report.connectivity.component_members),
        vec![
            vec![
                "/synthetic/station-a-1.nef".to_string(),
                "/synthetic/station-a-2.nef".to_string(),
            ],
            vec![
                "/synthetic/station-b-1.nef".to_string(),
                "/synthetic/station-b-2.nef".to_string(),
            ],
        ]
    );
    let ledger = degradation::run_ledger_snapshot();
    let rejection = ledger
        .entries()
        .iter()
        .find(|entry| entry.reason == degradation::GEOMETRY_DISCONNECTED)
        .expect("the disconnected rejection must remain in diagnostics");
    assert_eq!(rejection.severity, degradation::Severity::Rejected);
    assert_eq!(rejection.detail["component_count"], serde_json::json!(2));
    assert_eq!(rejection.detail["components"][0]["member_count"], 2);
    assert_eq!(rejection.detail["components"][1]["member_count"], 2);
    let error = geometry_disconnected_user_error(2);
    assert!(error.contains("2 个不连通分量"));
    assert!(error.contains("按连续场景重新分组"));
    assert!(error.contains("补拍增加重叠"));
}

#[test]
fn focus_capture_group_consensus_rejects_a_single_conflicting_edge() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let mut images = vec![
        focus_test_image(0, "DSC_1001.NEF"),
        focus_test_image(1, "DSC_1002.NEF"),
        focus_test_image(2, "DSC_1010.NEF"),
        focus_test_image(3, "DSC_1011.NEF"),
    ];
    for image in &mut images {
        image.focal_length_35mm = Some(90.0);
    }
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(5.0, 0.0, 40)),
        ((2, 3), observed_translation_match(5.0, 0.0, 40)),
        ((0, 2), observed_translation_match(100.0, 0.0, 32)),
        ((1, 3), observed_translation_match(100.0, 0.0, 28)),
        // Repeated content proposes a different group placement, but it
        // has no independent agreement from another member pair.
        ((0, 3), observed_translation_match(400.0, 0.0, 80)),
    ]);
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let locked = HashMap::from([
        (0, translation(0.0)),
        (1, translation(-5.0)),
        (2, translation(-160.0)),
        (3, translation(-165.0)),
    ]);

    let mut translation_verified = false;
    let mut closure_report = stack_report::ClosureReport::default();
    let mut relation_report = stack_report::StationRelationsReport::default();
    let solved = solve_focus_capture_group_poses_with_report(
        &images,
        &matches,
        &locked,
        0,
        &mut translation_verified,
        &mut closure_report,
        &mut relation_report,
    );
    assert!(!translation_verified);
    assert!(solved.is_empty());
    assert_eq!(relation_report.rejected_count, 1);
    assert_eq!(
        relation_report.rejected_by_reason[degradation::STATION_RELATION_PHOTOMETRIC_MISMATCH],
        1
    );
    assert_eq!(
        relation_report.rejected_by_reason[degradation::STATION_RELATION_EDGE_STRENGTH_MISMATCH],
        1
    );
    assert_eq!(
        relation_report.rejected_by_reason[degradation::STATION_RELATION_EDGE_ORIENTATION_MISMATCH],
        1
    );
}

#[test]
fn focus_capture_group_solver_keeps_multi_layer_coarse_consensus() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let mut images = vec![
        focus_test_image(0, "DSC_1001.NEF"),
        focus_test_image(1, "DSC_1002.NEF"),
        focus_test_image(2, "DSC_1010.NEF"),
        focus_test_image(3, "DSC_1011.NEF"),
    ];
    for image in &mut images {
        image.focal_length_35mm = Some(90.0);
    }
    let mut consensus_bridge = observed_translation_match(240.0, 20.0, 12);
    consensus_bridge.coarse_bridge = true;
    // Component recovery uses this non-empty field only after different
    // focus layers on both sides agreed on the coarse transform.
    consensus_bridge.top_candidate_points = consensus_bridge.points.clone();
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(5.0, 0.0, 40)),
        ((2, 3), observed_translation_match(5.0, 0.0, 40)),
        ((0, 2), consensus_bridge),
    ]);
    let translation = |x: f64, y: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0);
    let locked = HashMap::from([
        (0, translation(0.0, 0.0)),
        (1, translation(-5.0, 0.0)),
        (2, translation(-60.0, 0.0)),
        (3, translation(-65.0, 0.0)),
    ]);

    let mut translation_verified = false;
    let mut closure_report = stack_report::ClosureReport::default();
    let mut relation_report = stack_report::StationRelationsReport::default();
    let solved = solve_focus_capture_group_poses_with_report(
        &images,
        &matches,
        &locked,
        0,
        &mut translation_verified,
        &mut closure_report,
        &mut relation_report,
    );
    assert!(!translation_verified);
    assert!(solved.is_empty());
    assert_eq!(relation_report.rejected_count, 1);
    assert_eq!(
        relation_report.rejected_by_reason[degradation::STATION_RELATION_PHOTOMETRIC_MISMATCH],
        1
    );
}

#[test]
fn focus_capture_group_solver_rejects_out_of_contract_station_scale() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let mut images = vec![
        focus_test_image(0, "DSC_1001.NEF"),
        focus_test_image(1, "DSC_1002.NEF"),
        focus_test_image(2, "DSC_1010.NEF"),
        focus_test_image(3, "DSC_1011.NEF"),
    ];
    for image in &mut images {
        image.focal_length_35mm = Some(90.0);
    }
    let rejected_scale = STATION_RELATION_SCALE_RATIO_MIN - 0.01;
    let station_transform = Matrix3::new(
        rejected_scale,
        0.01,
        240.0,
        -0.01,
        rejected_scale,
        20.0,
        0.0,
        0.0,
        1.0,
    );
    let mut consensus_bridge = observed_transform_match(station_transform, 24);
    consensus_bridge.coarse_bridge = true;
    consensus_bridge.top_candidate_points = consensus_bridge.points.clone();
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(5.0, 0.0, 40)),
        ((2, 3), observed_translation_match(5.0, 0.0, 40)),
        ((0, 2), consensus_bridge),
    ]);
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let locked = HashMap::from([
        (0, translation(0.0)),
        (1, translation(-5.0)),
        (2, translation(-60.0)),
        (3, translation(-65.0)),
    ]);

    let mut geometry_verified = false;
    let mut closure_report = stack_report::ClosureReport::default();
    let mut relation_report = stack_report::StationRelationsReport::default();
    let solved = solve_focus_capture_group_poses_with_report(
        &images,
        &matches,
        &locked,
        0,
        &mut geometry_verified,
        &mut closure_report,
        &mut relation_report,
    );
    assert!(!geometry_verified);
    assert!(solved.is_empty());
    assert_eq!(relation_report.rejected_count, 1);
    assert_eq!(
        relation_report.rejected_by_reason[degradation::STATION_RELATION_SCALE_OUT_OF_RANGE],
        1
    );
}

#[test]
fn non_invertible_canonical_transform_is_reported_once_per_station_pair() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let mut images = vec![
        focus_test_image(0, "DSC_1001.NEF"),
        focus_test_image(1, "DSC_1002.NEF"),
        focus_test_image(2, "DSC_1010.NEF"),
        focus_test_image(3, "DSC_1011.NEF"),
    ];
    for image in &mut images {
        image.focal_length_35mm = Some(90.0);
    }
    let mut singular = observed_translation_match(240.0, 20.0, 24);
    singular.homography = Matrix3::zeros();
    singular.coarse_bridge = true;
    singular.top_candidate_points = singular.points.clone();
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(5.0, 0.0, 40)),
        ((2, 3), observed_translation_match(5.0, 0.0, 40)),
        ((0, 2), singular),
    ]);
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let locked = HashMap::from([
        (0, translation(0.0)),
        (1, translation(-5.0)),
        (2, translation(-60.0)),
        (3, translation(-65.0)),
    ]);
    let mut geometry_verified = false;
    let mut closure_report = stack_report::ClosureReport::default();
    let mut relation_report = stack_report::StationRelationsReport::default();
    let solved = solve_focus_capture_group_poses_with_report(
        &images,
        &matches,
        &locked,
        0,
        &mut geometry_verified,
        &mut closure_report,
        &mut relation_report,
    );
    relation_report.summarize_candidate_measurements();
    assert!(!geometry_verified);
    assert!(solved.is_empty());
    assert_eq!(relation_report.rejected_count, 1);
    assert_eq!(relation_report.unmeasured_count, 0);
    assert_eq!(
        relation_report.rejected_by_reason
            [degradation::STATION_RELATION_NON_INVERTIBLE_CANONICAL_TRANSFORM],
        1
    );
    assert_eq!(
        relation_report.rejected_by_failure_stage["canonical_transform_non_invertible"],
        1
    );
    assert!(
        !relation_report
            .rejected_by_reason
            .contains_key(degradation::STATION_RELATION_SCALE_OUT_OF_RANGE)
    );
}

#[test]
fn station_tree_prefers_inlier_count_over_composite_score() {
    let topology = topology::StationTopology {
        row: vec![0, 0, 0],
        column: vec![0, 1, 2],
        ..topology::StationTopology::default()
    };
    // Relation 0->2 has the largest old composite score, but the two local
    // relations carry more accepted geometric inliers and must form the
    // maximum-support initial tree.
    let scored_relations = [(1.0, 0, 1, 80), (2.0, 1, 2, 70), (10_000.0, 0, 2, 60)];
    let relations = scored_relations
        .iter()
        .map(|&(_, left, right, inliers)| (left, right, inliers))
        .collect::<Vec<_>>();

    let selected = maximum_inlier_spanning_tree_indices(3, &topology, &relations);

    assert_eq!(selected, vec![0, 1]);
    assert!(!selected.contains(&2));
}

#[test]
fn station_tree_breaks_equal_inlier_ties_by_row_then_column() {
    // Station ids intentionally oppose the geometric order. This catches
    // regressions to pair-index or insertion-order tie-breaking.
    let topology = topology::StationTopology {
        row: vec![1, 0, 1, 0],
        column: vec![1, 1, 0, 0],
        ..topology::StationTopology::default()
    };
    let relations = vec![(0, 1, 40), (0, 2, 40), (1, 3, 40), (2, 3, 40)];

    let selected = maximum_inlier_spanning_tree_indices(4, &topology, &relations);

    // Topology keys order 1->3 first, 2->3 second, then 0->1. The fourth
    // edge closes a cycle and is excluded.
    assert_eq!(selected, vec![2, 3, 0]);
}
