//! Station closure: projective tree closure and its seed, relation sample points, the robust
//! residual weight, corner limits, translation convergence and closure acceptance reports.

use super::*;

#[test]
fn station_projective_tree_closure_uses_loop_constraints() {
    let images = (0..3)
        .map(|index| focus_test_image(index, &format!("station-{index}.NEF")))
        .collect::<Vec<_>>();
    let groups = (0..3)
        .map(|index| FocusCaptureGroup {
            members: vec![index],
            anchor: index,
            local_to_anchor: HashMap::from([(index, Matrix3::identity())]),
        })
        .collect::<Vec<_>>();
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let initial = vec![translation(0.0), translation(100.0), translation(200.0)];
    let edges = vec![
        (1.0, 0, 1, translation(-100.0), 2, 0.0),
        (1.0, 1, 2, translation(-100.0), 2, 0.0),
        // The long closure is two pixels short and should pull the last
        // station without exceeding the direct-pair acceptance bound.
        (1.0, 0, 2, translation(-198.0), 2, 0.0),
    ];
    let solved =
        refine_focus_group_projective_tree_poses(&images, &groups, &edges, &initial, 0, 1_000.0)
            .expect("connected station loop should optimize");
    assert_eq!(solved.len(), 3);
    assert!(solved[2][(0, 2)] > 198.0);
    assert!(solved[2][(0, 2)] < 200.0);
}

#[test]
fn closure_relation_samples_ignore_out_of_bounds_overlap_extrapolation() {
    let translate = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    // Only the rightmost 20% of the left tile overlaps the right tile.
    // The overlap polygon contributes its vertices, edge midpoints, and
    // centroid; points in the non-overlapping 80% never become closure
    // constraints.
    let samples = closure_relation_sample_points(&translate(-80.0), (100, 100), (100, 100));
    assert_eq!(samples.len(), 9);
    assert!(
        samples
            .iter()
            .all(|point| (80.0..=100.0).contains(&point.x))
    );
    assert!(samples.iter().all(|point| (0.0..=100.0).contains(&point.y)));
    assert!(
        samples
            .iter()
            .any(|point| point.x == 90.0 && point.y == 50.0)
    );
}

#[test]
fn closure_balanced_retry_telemetry_is_backwards_compatible() {
    let report = stack_report::ClosureReport {
        balanced_retry_attempted: true,
        balanced_retry_accepted: true,
        before_retry_max_direct_pair_p95_px: Some(4.18925),
        ..Default::default()
    };
    let mut encoded = serde_json::to_value(&report).unwrap();
    assert_eq!(
        serde_json::from_value::<stack_report::ClosureReport>(encoded.clone()).unwrap(),
        report
    );
    let object = encoded.as_object_mut().unwrap();
    for field in [
        "balanced_retry_attempted",
        "balanced_retry_accepted",
        "before_retry_max_direct_pair_p95_px",
    ] {
        object.remove(field);
    }
    let legacy: stack_report::ClosureReport = serde_json::from_value(encoded).unwrap();
    assert!(!legacy.balanced_retry_attempted);
    assert!(!legacy.balanced_retry_accepted);
    assert_eq!(legacy.before_retry_max_direct_pair_p95_px, None);
}

#[test]
fn closure_residual_weight_is_exact_continuous_and_monotone() {
    assert_eq!(closure_residual_weight(0.0), 1.0);
    assert_eq!(closure_residual_weight(2.0), 1.0);
    assert!((closure_residual_weight(4.0) - 0.55).abs() < 1e-12);
    assert!((closure_residual_weight(6.0) - 0.1).abs() < 1e-12);
    assert!((closure_residual_weight(12.0) - 0.05).abs() < 1e-12);
    assert_eq!(closure_residual_weight(f64::INFINITY), 0.0);

    let epsilon = 1e-9;
    assert!(
        (closure_residual_weight(2.0 - epsilon) - closure_residual_weight(2.0 + epsilon)).abs()
            < epsilon
    );
    assert!(
        (closure_residual_weight(6.0 - epsilon) - closure_residual_weight(6.0 + epsilon)).abs()
            < epsilon
    );
    assert!(closure_residual_weight(2.0) >= 0.9);
    assert!(closure_residual_weight(6.0) <= 0.1);

    let samples = (0..=10_000)
        .map(|sample| closure_residual_weight(sample as f64 / 100.0))
        .collect::<Vec<_>>();
    assert!(samples.windows(2).all(|pair| pair[0] >= pair[1]));
}

#[test]
fn closure_corner_limit_uses_overlap_short_side_and_clamps_instead_of_rejecting() {
    let images = (0..2)
        .map(|index| focus_test_image(index, &format!("capture-{index}.NEF")))
        .collect::<Vec<_>>();
    let groups = (0..2)
        .map(|index| FocusCaptureGroup {
            members: vec![index],
            anchor: index,
            local_to_anchor: HashMap::from([(index, Matrix3::identity())]),
        })
        .collect::<Vec<_>>();
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let edges = vec![(1.0, 0, 1, translation(-800.0), 2, 0.0)];
    let initial = vec![Matrix3::identity(), translation(800.0)];
    let mut corrections = vec![[0.0, 0.0], [100.0, 0.0]];

    let (maximum, clamped) =
        clamp_station_translation_corrections(&images, &groups, &edges, &initial, &mut corrections)
            .expect("the accepted relation has a measurable overlap");

    // The overlap is 200px wide and 1000px high, so 10% of its short
    // side is 20px. The 100px proposal must be retained at that boundary.
    assert_eq!(clamped, 1);
    assert!((maximum - 20.0).abs() < 1e-9);
    assert!((corrections[1][0] - 20.0).abs() < 1e-9);
    let corrected = translation(corrections[1][0]) * initial[1];
    assert!(
        maximum_station_corner_displacement(&initial[1], &corrected, images[1].dimensions())
            .is_some_and(|displacement| (displacement - 20.0).abs() < 1e-9)
    );
}

#[test]
fn closure_corner_displacement_checks_all_four_corners() {
    let dimensions = (1_000, 1_000);
    let initial = Matrix3::identity();
    // The top-left corner is fixed while the other three move. A center-
    // only check would not describe the required station footprint bound.
    let corrected = Matrix3::new(1.1, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.0);
    let maximum = maximum_station_corner_displacement(&initial, &corrected, dimensions)
        .expect("all projected corners are finite");
    assert!((maximum - 100.0f64.hypot(100.0)).abs() < 1e-9);
}

#[test]
fn closure_translation_solver_uses_relative_weighted_residual_convergence() {
    let constraints = vec![
        (0, 1, [100.0, 0.0], 1.0),
        (1, 2, [100.0, 0.0], 1.0),
        (0, 2, [200.0, 0.0], 1.0),
    ];
    let solve = solve_robust_closure_translations(3, 0, &constraints).expect("connected solve");

    assert!(solve.converged);
    assert!(solve.iterations < CLOSURE_MAX_ITERATIONS);
    assert!((solve.corrections[1][0] - 100.0).abs() < 1e-3);
    assert!((solve.corrections[2][0] - 200.0).abs() < 1e-3);
    assert!(closure_relative_residual_decrease_converged(100.0, 99.995));
    assert!(!closure_relative_residual_decrease_converged(100.0, 99.98));
    assert!(!closure_relative_residual_decrease_converged(
        100.0, 100.001
    ));
}

#[test]
fn focus_group_projective_seed_uses_reliable_loop_closure() {
    let images = (0..3)
        .map(|index| focus_test_image(index, &format!("capture-{index}.NEF")))
        .collect::<Vec<_>>();
    let identity_local = |index| HashMap::from([(index, Matrix3::identity())]);
    let groups = (0..3)
        .map(|index| FocusCaptureGroup {
            members: vec![index],
            anchor: index,
            local_to_anchor: identity_local(index),
        })
        .collect::<Vec<_>>();
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let base_poses = vec![translation(0.0), translation(100.0), translation(200.0)];
    // The tree edges agree exactly with the seed. The 0->2 closure
    // observes a 2px residual and must stay inside both closure gates while
    // moving only group 2; scale and rotation remain untouched.
    let edges = vec![
        (1.0, 0, 1, translation(-100.0), 2, 0.0),
        (1.0, 1, 2, translation(-100.0), 2, 0.0),
        (100.0, 0, 2, translation(-202.0), 8, 0.0),
    ];
    let refined =
        refine_focus_group_projective_tree_poses(&images, &groups, &edges, &base_poses, 0, 1_000.0)
            .expect("connected loop graph should refine");
    assert!(refined[2][(0, 2)] > 200.0);
    assert!(refined[2][(0, 2)] < 202.1);
    assert!((refined[2][(0, 0)] - 1.0).abs() < 1e-9);
    assert!((refined[2][(1, 1)] - 1.0).abs() < 1e-9);
}

#[test]
fn closure_acceptance_uses_literal_iteration_residual_and_pair_boundaries() {
    let boundary_pair = [ClosureDirectPairP95 {
        left: 0,
        right: 1,
        p95_px: CLOSURE_DIRECT_PAIR_P95_LIMIT_PX,
    }];
    assert_eq!(
        closure_acceptance(
            3,
            3,
            true,
            CLOSURE_MAX_ITERATIONS,
            CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX,
            &boundary_pair,
        ),
        ClosureAcceptance::Accepted
    );
    assert_eq!(
        closure_acceptance(3, 3, true, 1, 2.000_001, &boundary_pair),
        ClosureAcceptance::UnreliableResidual
    );
    assert_eq!(
        closure_acceptance(3, 3, false, CLOSURE_MAX_ITERATIONS, 0.0, &boundary_pair),
        ClosureAcceptance::UnreliableIterations
    );
    assert!(matches!(
        closure_acceptance(
            3,
            3,
            true,
            1,
            0.0,
            &[ClosureDirectPairP95 {
                left: 0,
                right: 1,
                p95_px: 3.000_001,
            }],
        ),
        ClosureAcceptance::UnreliablePairP95(_)
    ));
}

#[test]
fn closure_accepts_converged_cycle_and_reports_direct_pair_maximum() {
    let images = (0..3)
        .map(|index| focus_test_image(index, &format!("capture-{index}.NEF")))
        .collect::<Vec<_>>();
    let groups = (0..3)
        .map(|index| FocusCaptureGroup {
            members: vec![index],
            anchor: index,
            local_to_anchor: HashMap::from([(index, Matrix3::identity())]),
        })
        .collect::<Vec<_>>();
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let base_poses = vec![translation(0.0), translation(100.0), translation(200.0)];
    let edges = vec![
        (1.0, 0, 1, translation(-100.0), 2, 0.0),
        (1.0, 1, 2, translation(-100.0), 2, 0.0),
        (1.0, 0, 2, translation(-200.0), 2, 0.0),
    ];
    let station_topology = topology::StationTopology {
        row: vec![0, 0, 0],
        column: vec![0, 1, 2],
        ..topology::StationTopology::default()
    };
    let mut closure = stack_report::ClosureReport::default();

    let accepted = refine_focus_group_projective_tree_poses_with_report(
        &images,
        &groups,
        &edges,
        &base_poses,
        0,
        1_000.0,
        &station_topology,
        2,
        &mut closure,
    )
    .expect("consistent cycle must be accepted");

    assert_eq!(closure.status, stack_report::ClosureStatus::Converged);
    assert!(closure.iterations <= CLOSURE_MAX_ITERATIONS);
    assert!(closure.residual_median_px <= CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX);
    assert!(closure.max_direct_pair_p95_px <= CLOSURE_DIRECT_PAIR_P95_LIMIT_PX);
    assert_eq!(closure.fallback_reason, None);
    assert_eq!(closure.offending_pair, None);
    assert_eq!(accepted.len(), base_poses.len());
}

#[test]
fn closure_without_cycle_skips_joint_solve_and_preserves_tree_bits() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let images = (0..3)
        .map(|index| focus_test_image(index, &format!("capture-{index}.NEF")))
        .collect::<Vec<_>>();
    let groups = (0..3)
        .map(|index| FocusCaptureGroup {
            members: vec![index],
            anchor: index,
            local_to_anchor: HashMap::from([(index, Matrix3::identity())]),
        })
        .collect::<Vec<_>>();
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let base_poses = vec![translation(-0.0), translation(100.25), translation(200.5)];
    let tree_edges = vec![
        (1.0, 0, 1, translation(-100.25), 2, 0.0),
        (1.0, 1, 2, translation(-100.25), 2, 0.0),
    ];
    let station_topology = topology::StationTopology {
        row: vec![0, 0, 0],
        column: vec![0, 1, 2],
        ..topology::StationTopology::default()
    };
    let mut closure = stack_report::ClosureReport::default();

    let returned = refine_focus_group_projective_tree_poses_with_report(
        &images,
        &groups,
        &tree_edges,
        &base_poses,
        0,
        1_000.0,
        &station_topology,
        2,
        &mut closure,
    )
    .expect("tree-only graph returns its initial poses");

    assert_eq!(closure.status, stack_report::ClosureStatus::NoConstraints);
    assert_eq!(closure.iterations, 0);
    assert_eq!(closure.max_direct_pair_p95_px, 0.0);
    assert_eq!(
        closure.fallback_reason.as_deref(),
        Some(degradation::CLOSURE_NO_CONSTRAINTS)
    );
    for (actual, expected) in returned.iter().zip(&base_poses) {
        for (actual, expected) in actual.as_slice().iter().zip(expected.as_slice()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn focus_group_projective_seed_keeps_large_residual_relation_in_closure_report() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let images = (0..3)
        .map(|index| focus_test_image(index, &format!("capture-{index}.NEF")))
        .collect::<Vec<_>>();
    let groups = (0..3)
        .map(|index| FocusCaptureGroup {
            members: vec![index],
            anchor: index,
            local_to_anchor: HashMap::from([(index, Matrix3::identity())]),
        })
        .collect::<Vec<_>>();
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let base_poses = vec![translation(0.0), translation(100.0), translation(200.0)];
    // The 0->2 relation disagrees with the tree by 100px, beyond the old
    // coordinate_scale * 0.08 pre-discard threshold of 80px. It remains an
    // accepted, independently confirmed relation and must be downweighted
    // rather than removed.
    let edges = vec![
        (1.0, 0, 1, translation(-100.0), 2, 0.0),
        (1.0, 1, 2, translation(-100.0), 2, 0.0),
        (1.0, 0, 2, translation(-300.0), 2, 0.0),
    ];
    let mut closure = stack_report::ClosureReport::default();
    let station_topology = topology::StationTopology {
        row: vec![0, 0, 0],
        column: vec![0, 1, 2],
        ..topology::StationTopology::default()
    };

    let refined = refine_focus_group_projective_tree_poses_with_report(
        &images,
        &groups,
        &edges,
        &base_poses,
        0,
        1_000.0,
        &station_topology,
        2,
        &mut closure,
    )
    .expect("an unreliable joint solution must return the tree seed");

    assert_eq!(closure.status, stack_report::ClosureStatus::Unreliable);
    assert_eq!(
        closure.fallback_reason.as_deref(),
        Some(degradation::CLOSURE_UNRELIABLE_PAIR_P95)
    );
    let offending = closure
        .offending_pair
        .expect("the worst direct pair must be reported");
    assert_eq!((offending.left_row, offending.left_column), (0, 0));
    assert_eq!((offending.right_row, offending.right_column), (0, 2));
    assert!(offending.p95_px > CLOSURE_DIRECT_PAIR_P95_LIMIT_PX);
    for (actual, expected) in refined.iter().zip(&base_poses) {
        for (actual, expected) in actual.as_slice().iter().zip(expected.as_slice()) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    assert_eq!(closure.participating_constraints, edges.len());
    assert!(closure.iterations > 0);
    assert!(closure.iterations <= CLOSURE_MAX_ITERATIONS);
    assert!(closure.residual_median_px.is_finite());
    assert!(closure.residual_p95_px >= closure.residual_median_px);
    assert!(closure.max_corner_correction_px <= CLOSURE_MAX_CORNER_CORRECTION_PX);
    assert_eq!(closure.constraint_weights.len(), edges.len());
    let large_residual = closure
        .constraint_weights
        .iter()
        .find(|relation| (relation.left, relation.right) == (0, 2))
        .expect("large-residual relation must be reported");
    assert!(large_residual.residual_px > 80.0);
    assert!(large_residual.weight < 0.1);
    assert_eq!(
        large_residual.weight,
        closure_residual_weight(large_residual.residual_px),
        "reported weights must use the same production M-estimator multiplier"
    );

    let original_edges = edges.clone();
    let protected = std::collections::BTreeSet::from([(0, 1), (1, 2)]);
    let mut protected_report = stack_report::ClosureReport::default();
    let protected_result = refine_focus_group_projective_tree_poses_with_report_protected(
        &images,
        &groups,
        &edges,
        &base_poses,
        0,
        1_000.0,
        &station_topology,
        2,
        &protected,
        &mut protected_report,
    )
    .unwrap();
    assert_eq!(
        protected_report.status,
        stack_report::ClosureStatus::Unreliable
    );
    assert!(protected_report.max_direct_pair_p95_px > CLOSURE_DIRECT_PAIR_P95_LIMIT_PX);
    assert!(protected_report.balanced_retry_attempted);
    assert!(!protected_report.balanced_retry_accepted);
    assert!(
        protected_report
            .before_retry_max_direct_pair_p95_px
            .is_some_and(|value| value > CLOSURE_DIRECT_PAIR_P95_LIMIT_PX)
    );
    assert_eq!(protected_result, base_poses);
    assert_eq!(
        edges, original_edges,
        "a contradictory cycle must not be repaired by replacing direct evidence"
    );
}

// Frozen direct Virtual_Tile measurements from the continuous nine-NEF
// acceptance set. These are independent fitted relations, never composed
// from the spanning tree being evaluated.
#[test]
fn closure_reconciles_nine_source_direct_relations_without_rewriting_evidence() {
    let _scope = degradation::begin_run_scope();
    let dimensions = [(4204, 2800), (4205, 2825), (4310, 2931)];
    let images = dimensions
        .iter()
        .enumerate()
        .map(|(index, &(width, height))| {
            let mut image = focus_test_image(index, "frozen-station");
            image.width = width;
            image.height = height;
            image
        })
        .collect::<Vec<_>>();
    let groups = (0..3)
        .map(|index| FocusCaptureGroup {
            members: vec![index],
            anchor: index,
            local_to_anchor: HashMap::from([(index, Matrix3::identity())]),
        })
        .collect::<Vec<_>>();
    let direct_01 = Matrix3::from_row_slice(&[
        0.9988000864418181,
        0.00017634524606939856,
        -49.488326799623756,
        -6.491946260586207e-05,
        0.9981903540579469,
        37.61623382393811,
        0.0,
        -0.0,
        1.0,
    ])
    .try_inverse()
    .unwrap();
    let direct_02 = Matrix3::from_row_slice(&[
        0.94956845881629,
        0.020475259162372854,
        3068.4978836987198,
        0.019354315921857837,
        0.9557928271468688,
        183.23907833473763,
        1.0741563054728558e-08,
        2.210637741704831e-08,
        1.0000374745151706,
    ])
    .try_inverse()
    .unwrap();
    let direct_12 = Matrix3::from_row_slice(&[
        1.0501052879221628,
        -0.02033504423161718,
        -3275.661361581249,
        -0.020767181416482677,
        1.0430175081872481,
        -86.61508997317091,
        0.0,
        0.0,
        1.0,
    ]);
    let edges = vec![
        (1.0, 0, 1, direct_01, 1, 0.8138741651368357),
        (1.0, 1, 2, direct_12, 1, 0.6396504274573706),
        (1.0, 0, 2, direct_02, 1, 1.0531387958310412),
    ];
    let seed1 = direct_01.try_inverse().unwrap();
    let base = vec![
        Matrix3::identity(),
        seed1,
        seed1 * direct_12.try_inverse().unwrap(),
    ];
    let topology = topology::StationTopology {
        row: vec![0, 0, 0],
        column: vec![1, 2, 0],
        ..Default::default()
    };
    let protected = std::collections::BTreeSet::from([(0, 1), (1, 2)]);
    let original_edges = edges.clone();
    let initial_validation = closure_direct_pair_p95s(&images, &groups, &edges, &base);
    assert!(
        initial_validation
            .iter()
            .any(|pair| pair.p95_px > CLOSURE_DIRECT_PAIR_P95_LIMIT_PX)
    );
    let mut report = stack_report::ClosureReport::default();
    let poses = refine_focus_group_projective_tree_poses_with_report_protected(
        &images,
        &groups,
        &edges,
        &base,
        0,
        4310.0,
        &topology,
        1,
        &protected,
        &mut report,
    )
    .unwrap();
    assert_eq!(
        report.status,
        stack_report::ClosureStatus::Converged,
        "{report:?}"
    );
    assert!(report.balanced_retry_attempted);
    assert!(report.balanced_retry_accepted);
    assert!(
        report
            .before_retry_max_direct_pair_p95_px
            .is_some_and(|value| value > CLOSURE_DIRECT_PAIR_P95_LIMIT_PX)
    );
    assert_eq!(
        edges, original_edges,
        "direct observations must remain unchanged"
    );
    let validation = closure_direct_pair_p95s(&images, &groups, &original_edges, &poses);
    assert!(
        validation
            .iter()
            .all(|pair| pair.p95_px <= CLOSURE_DIRECT_PAIR_P95_LIMIT_PX)
    );
    let independent_maximum = validation
        .iter()
        .map(|pair| pair.p95_px)
        .fold(0.0, f64::max);
    assert!((report.max_direct_pair_p95_px - independent_maximum).abs() < 1e-9);
    assert!(
        independent_maximum > 1.0,
        "real direct residuals must not disappear through circular validation"
    );
    assert!(report.max_corner_correction_px <= CLOSURE_MAX_CORNER_CORRECTION_PX);
    assert!(report.offending_pair.is_none());
    assert!(report.fallback_reason.is_none());
}
