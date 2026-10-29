//! Closure_Optimizer properties (35-40, 77).
//!
//! The robust weight, clamped corrections, the termination condition, residuals
//! against an independent oracle, unreliable closures keeping the tree poses,
//! tree-only relations, and closure degradation keeping the Quality_Gate record.

use super::super::degradation;
use super::super::report::QualityGateReport;
use nalgebra::Matrix3;
use nalgebra::Point2;
use proptest::prelude::*;

use super::{map_point, matrix_bits, splitmix64, translation_pose, unit_from};

// ---------------------------------------------------------------------------
// Closure_Optimizer properties 35-39 (requirements 7.3-7.9)
// ---------------------------------------------------------------------------

pub(super) fn closure_weight_numeric_oracle(residual_px: f64) -> f64 {
    if residual_px.is_nan() {
        return 0.0;
    }
    let residual_px = if residual_px < 0.0 { 0.0 } else { residual_px };
    match residual_px {
        residual if residual <= 2.0 => 1.0,
        residual if residual <= 6.0 => 0.1 + 0.225 * (6.0 - residual),
        residual => 0.6 / residual,
    }
}

pub(super) fn closure_stop_numeric_oracle(previous: f64, current: f64) -> bool {
    if !previous.is_finite() || !current.is_finite() || current > previous {
        return false;
    }
    if previous <= f64::EPSILON {
        return current <= f64::EPSILON;
    }
    (previous - current) / previous < 1.0e-4
}

pub(super) fn closure_summary_numeric_oracle(mut residuals: Vec<f64>) -> (f64, f64) {
    assert!(!residuals.is_empty());
    assert!(residuals.iter().all(|residual| residual.is_finite()));
    residuals.sort_by(f64::total_cmp);
    let median = residuals[residuals.len() / 2];
    let p95 = residuals[(residuals.len() * 95).div_ceil(100) - 1];
    (median, p95)
}

pub(super) fn closure_relation_residual_numeric_oracle(
    dimensions: (u32, u32),
    base_poses: &[Matrix3<f64>],
    solved_poses: &[Matrix3<f64>],
    relation: &crate::panorama_stitching::station_relation_test_access::ClosureRelationView,
) -> f64 {
    let sample = Point2::new(f64::from(dimensions.0) * 0.5, f64::from(dimensions.1) * 0.5);
    let left_base = map_point(&base_poses[relation.left], sample).unwrap();
    let right_local = map_point(&relation.left_to_right, sample).unwrap();
    let right_base = map_point(&base_poses[relation.right], right_local).unwrap();
    let observed = left_base - right_base;

    let left_solved = map_point(&solved_poses[relation.left], sample).unwrap();
    let right_solved = map_point(&solved_poses[relation.right], right_local).unwrap();
    let left_correction = left_solved - left_base;
    let right_correction = right_solved - right_base;
    (right_correction - left_correction - observed).norm()
}

pub(super) fn closure_direct_pair_p95_numeric_oracle(
    dimensions: (u32, u32),
    solved_poses: &[Matrix3<f64>],
    relation: &crate::panorama_stitching::station_relation_test_access::ClosureRelationView,
) -> f64 {
    const SAMPLES: usize = 11;
    let mut errors = Vec::with_capacity(SAMPLES * SAMPLES);
    for sample_y in 0..SAMPLES {
        let y = f64::from(dimensions.1.saturating_sub(1)) * sample_y as f64 / (SAMPLES - 1) as f64;
        for sample_x in 0..SAMPLES {
            let x =
                f64::from(dimensions.0.saturating_sub(1)) * sample_x as f64 / (SAMPLES - 1) as f64;
            let source = Point2::new(x, y);
            let Some(target) = map_point(&relation.left_to_right, source) else {
                continue;
            };
            if target.x < 0.0
                || target.y < 0.0
                || target.x >= f64::from(dimensions.0)
                || target.y >= f64::from(dimensions.1)
            {
                continue;
            }
            let left_world = map_point(&solved_poses[relation.left], source).unwrap();
            let right_world = map_point(&solved_poses[relation.right], target).unwrap();
            errors.push((left_world - right_world).norm());
        }
    }
    closure_summary_numeric_oracle(errors).1
}

pub(super) fn accepted_closure_fixture(
    seed: u64,
) -> (
    (u32, u32),
    Vec<u32>,
    Vec<u32>,
    Vec<Matrix3<f64>>,
    Vec<crate::panorama_stitching::station_relation_test_access::ClosureRelationView>,
) {
    use crate::panorama_stitching::station_relation_test_access::ClosureRelationView;

    let dimensions = (512, 384);
    let positions = [(0.0, 0.0), (120.0, 0.0), (240.0, 0.0)];
    let base_poses = positions
        .iter()
        .map(|&(x, y)| translation_pose(x, y))
        .collect::<Vec<_>>();
    let mut state = seed ^ 0x35_36_37_38_39;
    let pairs = [(0, 1), (1, 2), (0, 2)];
    let relations = pairs
        .into_iter()
        .map(|(left, right)| {
            let noise_x = (unit_from(&mut state) - 0.5) * 0.8;
            let noise_y = (unit_from(&mut state) - 0.5) * 0.8;
            let ideal_x = positions[left].0 - positions[right].0;
            let ideal_y = positions[left].1 - positions[right].1;
            ClosureRelationView {
                score: 0.75 + 0.5 * unit_from(&mut state),
                left,
                right,
                left_to_right: Matrix3::new(
                    1.0,
                    0.0,
                    ideal_x + noise_x,
                    0.0,
                    1.0,
                    ideal_y + noise_y,
                    0.0,
                    0.0,
                    1.0,
                ),
                independent_support: 2 + (splitmix64(&mut state) as usize % 5),
                median_error_px: 0.5 * unit_from(&mut state),
            }
        })
        .collect();
    (
        dimensions,
        vec![0, 0, 0],
        vec![0, 1, 2],
        base_poses,
        relations,
    )
}

pub(super) fn assert_pose_bits_equal(actual: &[Matrix3<f64>], expected: &[Matrix3<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(matrix_bits(actual), matrix_bits(expected));
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 35: the exact
    // production closure weight equals an independent piecewise numeric oracle,
    // is monotone on r >= 0, is continuous at 2/6, satisfies both endpoint
    // bounds, and gives finite deterministic values for tails/non-finite input.
    //
    // **Validates: Requirements 7.3**
    #[test]
    fn property_35_closure_weight_is_monotone_with_exact_boundaries(
        first in 0u64..10_000_000,
        second in 0u64..10_000_000,
    ) {
        use crate::panorama_stitching::station_relation_test_access;

        let lower = first.min(second) as f64 / 1_000.0;
        let upper = first.max(second) as f64 / 1_000.0;
        let actual_lower = station_relation_test_access::closure_weight(lower);
        let actual_upper = station_relation_test_access::closure_weight(upper);
        prop_assert!(actual_lower >= actual_upper);
        prop_assert!((actual_lower - closure_weight_numeric_oracle(lower)).abs() <= 4.0 * f64::EPSILON);
        prop_assert!((actual_upper - closure_weight_numeric_oracle(upper)).abs() <= 4.0 * f64::EPSILON);

        for residual in [0.0, 2.0, 6.0, 12.0, f64::MAX, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let actual = station_relation_test_access::closure_weight(residual);
            let expected = closure_weight_numeric_oracle(residual);
            prop_assert!(actual.is_finite(), "weight must be finite for residual {residual:?}");
            prop_assert!((actual - expected).abs() <= 4.0 * f64::EPSILON.max(expected.abs() * f64::EPSILON));
        }
        let epsilon = 1.0e-9;
        prop_assert!((station_relation_test_access::closure_weight(2.0 - epsilon)
            - station_relation_test_access::closure_weight(2.0 + epsilon)).abs() < epsilon);
        prop_assert!((station_relation_test_access::closure_weight(6.0 - epsilon)
            - station_relation_test_access::closure_weight(6.0 + epsilon)).abs() < epsilon);
        prop_assert!(station_relation_test_access::closure_weight(2.0) >= 0.9);
        prop_assert!(station_relation_test_access::closure_weight(6.0) <= 0.1);
        prop_assert!(station_relation_test_access::closure_weight(f64::MAX) >= 0.0);
    }

    // Feature: layered-camera-group-focus-stitching, Property 36: production
    // clamps every station by the maximum of all four corner displacements at
    // min(10% overlap short side, 256px), retaining rather than discarding an
    // above-limit correction and reporting exact count/maximum/reason.
    //
    // **Validates: Requirements 7.4**
    #[test]
    fn property_36_closure_corrections_are_clamped_and_reported(
        seed in any::<u64>(),
        use_absolute_cap in any::<bool>(),
    ) {
        let _run_scope = degradation::begin_run_scope();
        use crate::panorama_stitching::station_relation_test_access::{self, ClosureRelationView};

        let (dimensions, overlap_short_side) = if use_absolute_cap {
            ((4_096, 4_096), 4_096.0)
        } else {
            let width = 600 + (seed as u32 % 600);
            let height = 500 + (seed.rotate_left(13) as u32 % 500);
            let overlap = 40 + (seed.rotate_left(29) as u32 % 300).min(width - 1);
            ((width, height), f64::from(overlap.min(height)))
        };
        let relation_x = if use_absolute_cap {
            0.0
        } else {
            -(f64::from(dimensions.0) - overlap_short_side)
        };
        let base_poses = vec![Matrix3::identity(), translation_pose(-relation_x, 0.0)];
        let relations = vec![ClosureRelationView {
            score: 1.0,
            left: 0,
            right: 1,
            left_to_right: translation_pose(relation_x, 0.0),
            independent_support: 2,
            median_error_px: 0.0,
        }];
        let limit = (0.10 * overlap_short_side).min(256.0);
        let excess = 0.25 + unit_from(&mut (seed ^ 0x36c1_a55e)) * 64.0;
        let on_or_just_below_limit = if use_absolute_cap {
            limit
        } else {
            limit - 1.0e-6
        };
        let proposed = vec![
            [on_or_just_below_limit, 0.0],
            [limit + excess, 0.0],
        ];
        let clamped = station_relation_test_access::clamp_closure_corrections(
            dimensions,
            &base_poses,
            &relations,
            &proposed,
        )
        .expect("the synthetic overlap is measurable");

        prop_assert_eq!(clamped.clamped_stations, 1);
        prop_assert!((clamped.maximum_corner_displacement_px - limit).abs() < 1.0e-8);
        prop_assert_eq!(
            clamped.corrections[0][0].to_bits(),
            on_or_just_below_limit.to_bits(),
            "the 256px exact boundary and ratio-boundary interior are retained"
        );
        prop_assert!((clamped.corrections[1][0] - limit).abs() < 1.0e-8);
        prop_assert!(clamped.corrections[1][0] > 0.0, "above-bound correction is clamped, not discarded");
        for correction in &clamped.corrections {
            prop_assert!(correction[0].hypot(correction[1]) <= limit + 1.0e-8);
        }

        // A small, consistent four-station loop reaches the real reporting
        // boundary with corrections just above its 2px overlap-derived cap.
        degradation::reset_run_ledger();
        let square = [(0.0, 0.0), (380.0, 0.0), (0.0, 380.0), (380.0, 380.0)];
        let base = square.iter().map(|&(x, y)| translation_pose(x, y)).collect::<Vec<_>>();
        let q = 2.05 + 0.20 * unit_from(&mut (seed ^ 0x36de_9a11));
        let desired = [(0.0, 0.0), (q, 0.0), (0.0, q), (q, q)];
        let loop_pairs = [(0, 1), (1, 3), (3, 2), (2, 0)];
        let loop_relations = loop_pairs.into_iter().map(|(left, right)| {
            let observed_x = desired[right].0 - desired[left].0;
            let observed_y = desired[right].1 - desired[left].1;
            ClosureRelationView {
                score: 1.0,
                left,
                right,
                left_to_right: translation_pose(
                    square[left].0 - square[right].0 - observed_x,
                    square[left].1 - square[right].1 - observed_y,
                ),
                independent_support: 2,
                median_error_px: 0.0,
            }
        }).collect::<Vec<_>>();
        let run = station_relation_test_access::closure_run(
            (400, 400),
            &[0, 0, 1, 1],
            &[0, 1, 0, 1],
            &base,
            &loop_relations,
        );
        prop_assert_eq!(run.report.status, crate::panorama_utils::stack_pipeline::report::ClosureStatus::Converged);
        prop_assert!(run.report.clamped_stations > 0);
        prop_assert!(run.report.max_corner_correction_px <= 2.0 + 1.0e-8);
        let ledger = degradation::run_ledger_snapshot();
        let entry = ledger.entries.iter().find(|entry| entry.reason == degradation::CLOSURE_CORRECTION_CLAMPED)
            .expect("a clamped accepted solve records the stable reason");
        prop_assert_eq!(entry.detail["clamped_stations"].as_u64(), Some(run.report.clamped_stations as u64));
        prop_assert!((entry.detail["max_corner_correction_px"].as_f64().unwrap() - run.report.max_corner_correction_px).abs() < 1.0e-12);
    }

    // Feature: layered-camera-group-focus-stitching, Property 37: production
    // terminates exactly on relative robust-weighted decrease < 1e-4 or at the
    // literal 100-iteration cap, and reports exact iterations/status/residuals.
    //
    // **Validates: Requirements 7.5**
    #[test]
    fn property_37_closure_termination_condition_and_report_are_exact(seed in any::<u64>()) {
        use crate::panorama_stitching::station_relation_test_access::{self, ClosureAcceptanceView};

        let mut state = seed ^ 0x37c0_ffee;
        let previous = 1.0 + unit_from(&mut state) * 100_000.0;
        let relative_drop = unit_from(&mut state) * 2.0e-4;
        let current = previous * (1.0 - relative_drop);
        prop_assert_eq!(
            station_relation_test_access::closure_converged(previous, current),
            closure_stop_numeric_oracle(previous, current),
        );
        for (before, after) in [
            (10_000.0, 9_999.000_1),
            (10_000.0, 9_999.0),
            (10_000.0, 9_998.999_9),
            (0.0, 0.0),
            (1.0, f64::INFINITY),
            (f64::NAN, 0.0),
            (1.0, 1.000_001),
        ] {
            prop_assert_eq!(
                station_relation_test_access::closure_converged(before, after),
                closure_stop_numeric_oracle(before, after),
            );
        }
        prop_assert_eq!(
            station_relation_test_access::closure_acceptance_view(3, 3, false, 100, 0.0, &[(0, 1, 3.0)]),
            ClosureAcceptanceView::UnreliableIterations,
        );
        prop_assert_eq!(
            station_relation_test_access::closure_acceptance_view(3, 3, true, 100, 2.0, &[(0, 1, 3.0)]),
            ClosureAcceptanceView::Accepted,
        );

        let (dimensions, rows, columns, base, relations) = accepted_closure_fixture(seed);
        let constraints = relations.iter().map(|relation| {
            let sample = Point2::new(f64::from(dimensions.0) * 0.5, f64::from(dimensions.1) * 0.5);
            let left = map_point(&base[relation.left], sample).unwrap();
            let right_local = map_point(&relation.left_to_right, sample).unwrap();
            let right = map_point(&base[relation.right], right_local).unwrap();
            let observed = left - right;
            let weight = relation.independent_support as f64 * relation.score.sqrt()
                / (1.0 + relation.median_error_px / 3.0);
            (relation.left, relation.right, [observed.x, observed.y], weight)
        }).collect::<Vec<_>>();
        let solve = station_relation_test_access::closure_solve(3, 0, &constraints)
            .expect("connected closure fixture solves");
        prop_assert!(solve.converged || solve.iterations == 100);
        prop_assert!(solve.iterations <= 100);
        prop_assert_eq!(solve.corrections.len(), 3);

        let run = station_relation_test_access::closure_run(dimensions, &rows, &columns, &base, &relations);
        prop_assert_eq!(run.report.iterations, solve.iterations);
        prop_assert_eq!(run.report.status, crate::panorama_utils::stack_pipeline::report::ClosureStatus::Converged);
        prop_assert_eq!(run.report.participating_constraints, relations.len());
        let oracle_residuals = relations.iter().map(|relation| {
            closure_relation_residual_numeric_oracle(dimensions, &base, &run.poses, relation)
        }).collect::<Vec<_>>();
        let (median, p95) = closure_summary_numeric_oracle(oracle_residuals);
        prop_assert!((run.report.residual_median_px - median).abs() < 1.0e-8);
        prop_assert!((run.report.residual_p95_px - p95).abs() < 1.0e-8);
    }

    // Feature: layered-camera-group-focus-stitching, Property 38: every
    // accepted production closure has median <= 2 and each direct pair P95 <=
    // 3; all reported summaries equal an independent overlap-residual oracle.
    //
    // **Validates: Requirements 7.6, 7.8**
    #[test]
    fn property_38_accepted_closure_residuals_match_independent_oracle(seed in any::<u64>()) {
        use crate::panorama_stitching::station_relation_test_access::{self, ClosureAcceptanceView};

        let (dimensions, rows, columns, base, relations) = accepted_closure_fixture(seed);
        let run = station_relation_test_access::closure_run(dimensions, &rows, &columns, &base, &relations);
        prop_assert_eq!(run.report.status, crate::panorama_utils::stack_pipeline::report::ClosureStatus::Converged);
        prop_assert!(run.report.residual_median_px <= 2.0);
        prop_assert!(run.report.max_direct_pair_p95_px <= 3.0);

        let closure_residuals = relations.iter().map(|relation| {
            closure_relation_residual_numeric_oracle(dimensions, &base, &run.poses, relation)
        }).collect::<Vec<_>>();
        let (median, p95) = closure_summary_numeric_oracle(closure_residuals);
        prop_assert!((run.report.residual_median_px - median).abs() < 1.0e-8);
        prop_assert!((run.report.residual_p95_px - p95).abs() < 1.0e-8);

        let pair_p95s = relations.iter().map(|relation| {
            closure_direct_pair_p95_numeric_oracle(dimensions, &run.poses, relation)
        }).collect::<Vec<_>>();
        prop_assert!(pair_p95s.iter().all(|&pair| pair <= 3.0));
        let oracle_maximum = pair_p95s.into_iter().max_by(f64::total_cmp).unwrap();
        prop_assert!((run.report.max_direct_pair_p95_px - oracle_maximum).abs() < 1.0e-8);
        prop_assert_eq!(run.report.offending_pair, None);
        prop_assert_eq!(run.report.fallback_reason, None);

        prop_assert_eq!(
            station_relation_test_access::closure_acceptance_view(3, 3, true, 100, 2.0, &[(0, 1, 3.0)]),
            ClosureAcceptanceView::Accepted,
        );
        prop_assert_eq!(
            station_relation_test_access::closure_acceptance_view(3, 3, true, 1, 2.0 + f64::EPSILON * 4.0, &[(0, 1, 3.0)]),
            ClosureAcceptanceView::UnreliableResidual,
        );
        prop_assert_eq!(
            station_relation_test_access::closure_acceptance_view(3, 3, true, 1, 2.0, &[(0, 1, 3.0 + f64::EPSILON * 4.0)]),
            ClosureAcceptanceView::UnreliablePairP95(0, 1),
        );
    }

    // Feature: layered-camera-group-focus-stitching, Property 39: residual,
    // iteration, and direct-pair-P95 unreliable outcomes all return the
    // spanning-tree initial poses bit-for-bit with exact stable reasons and the
    // correct offending pair coordinates.
    //
    // **Validates: Requirements 7.7, 7.9**
    #[test]
    fn property_39_unreliable_closure_preserves_tree_pose_bits(seed in any::<u64>()) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use crate::panorama_stitching::station_relation_test_access::{self, ClosureRelationView};

        let signed_zero = if seed & 1 == 0 { 0.0 } else { -0.0 };
        let base = vec![translation_pose(signed_zero, -0.0), translation_pose(100.25, 0.0), translation_pose(200.5, 0.0)];
        let relation = |left, right, tx| ClosureRelationView {
            score: 1.0,
            left,
            right,
            left_to_right: translation_pose(tx, 0.0),
            independent_support: 2,
            median_error_px: 0.0,
        };

        // Equal 9px incompatible residuals keep equal robust weights, so the
        // production preliminary gate deterministically selects residual.
        let residual_relations = vec![
            relation(0, 1, -109.25),
            relation(1, 2, -109.25),
            relation(0, 2, -191.5),
        ];
        let residual = station_relation_test_access::closure_run(
            (400, 400), &[2, 0, 1], &[1, 2, 0], &base, &residual_relations,
        );
        prop_assert_eq!(residual.report.status, crate::panorama_utils::stack_pipeline::report::ClosureStatus::Unreliable);
        prop_assert_eq!(residual.report.fallback_reason.as_deref(), Some(degradation::CLOSURE_UNRELIABLE_RESIDUAL));
        prop_assert!(residual.report.residual_median_px > 2.0);
        assert_pose_bits_equal(&residual.poses, &base);

        // More accepted relations than a tree, but two disconnected usable
        // components: the production iteration failure path returns immediately.
        let base_four = vec![
            translation_pose(signed_zero, -0.0),
            translation_pose(100.25, 0.0),
            translation_pose(300.75, 0.0),
            translation_pose(401.0, 0.0),
        ];
        let iteration_relations = vec![
            relation(0, 1, -100.25),
            relation(1, 0, 100.25),
            relation(2, 3, -100.25),
            relation(3, 2, 100.25),
        ];
        let iterations = station_relation_test_access::closure_run(
            (400, 400), &[0, 0, 1, 1], &[0, 1, 0, 1], &base_four, &iteration_relations,
        );
        prop_assert_eq!(iterations.report.status, crate::panorama_utils::stack_pipeline::report::ClosureStatus::Unreliable);
        prop_assert_eq!(iterations.report.fallback_reason.as_deref(), Some(degradation::CLOSURE_UNRELIABLE_ITERATIONS));
        prop_assert_eq!(iterations.report.iterations, 0);
        assert_pose_bits_equal(&iterations.poses, &base_four);

        let pair_relations = vec![
            relation(0, 1, -100.25),
            relation(1, 2, -100.25),
            relation(0, 2, -300.5),
        ];
        let pair = station_relation_test_access::closure_run(
            (400, 400), &[2, 0, 1], &[1, 2, 0], &base, &pair_relations,
        );
        prop_assert_eq!(pair.report.status, crate::panorama_utils::stack_pipeline::report::ClosureStatus::Unreliable);
        prop_assert_eq!(pair.report.fallback_reason.as_deref(), Some(degradation::CLOSURE_UNRELIABLE_PAIR_P95));
        let offending = pair.report.offending_pair.expect("pair-P95 fallback reports its worst direct pair");
        prop_assert_eq!((offending.left, offending.right), (0, 2));
        prop_assert_eq!((offending.left_row, offending.left_column), (2, 1));
        prop_assert_eq!((offending.right_row, offending.right_column), (1, 0));
        prop_assert!(offending.p95_px > 3.0);
        prop_assert!((pair.report.max_direct_pair_p95_px - offending.p95_px).abs() < 1.0e-12);
        assert_pose_bits_equal(&pair.poses, &base);
    }

    // Feature: layered-camera-group-focus-stitching, Property 40: whenever the
    // accepted relation count cannot exceed a spanning tree, the production
    // closure boundary returns the tree seed bit-for-bit without entering the
    // joint solver and records the exact no-constraints status and reason.
    //
    // **Validates: Requirements 7.10**
    #[test]
    fn property_40_tree_only_relations_skip_joint_optimization(
        seed in any::<u64>(),
        station_count in 2usize..10,
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use crate::panorama_stitching::station_relation_test_access::{self, ClosureRelationView};

        let mut state = seed ^ 0x40d7_10c1_05e5_7eed;
        let relation_count = splitmix64(&mut state) as usize % station_count;
        let base = (0..station_count)
            .map(|_| {
                Matrix3::from_iterator(
                    (0..9).map(|_| f64::from_bits(splitmix64(&mut state)))
                )
            })
            .collect::<Vec<_>>();
        let relations = (0..relation_count)
            .map(|edge| {
                let left = edge % station_count;
                let right = (edge + 1) % station_count;
                ClosureRelationView {
                    score: f64::from_bits(splitmix64(&mut state)),
                    left,
                    right,
                    left_to_right: Matrix3::from_iterator(
                        (0..9).map(|_| f64::from_bits(splitmix64(&mut state)))
                    ),
                    independent_support: 2,
                    median_error_px: f64::from_bits(splitmix64(&mut state)),
                }
            })
            .collect::<Vec<_>>();
        let rows = (0..station_count as u32).collect::<Vec<_>>();
        let columns = (0..station_count as u32).rev().collect::<Vec<_>>();

        // Independent oracle: relation_count is constructed in 0..station_count,
        // exactly the predicate accepted_relations <= stations - 1.
        prop_assert!(relations.len() < station_count);
        let run = station_relation_test_access::closure_run(
            (400, 400),
            &rows,
            &columns,
            &base,
            &relations,
        );

        assert_pose_bits_equal(&run.poses, &base);
        prop_assert_eq!(run.report.status, crate::panorama_utils::stack_pipeline::report::ClosureStatus::NoConstraints);
        prop_assert_eq!(run.report.participating_constraints, relations.len());
        prop_assert_eq!(run.report.iterations, 0);
        prop_assert!(run.report.constraint_weights.is_empty());
        prop_assert_eq!(run.report.residual_median_px.to_bits(), 0.0f64.to_bits());
        prop_assert_eq!(run.report.residual_p95_px.to_bits(), 0.0f64.to_bits());
        prop_assert_eq!(run.report.max_direct_pair_p95_px.to_bits(), 0.0f64.to_bits());
        prop_assert_eq!(
            run.report.fallback_reason.as_deref(),
            Some(degradation::CLOSURE_NO_CONSTRAINTS),
        );
        prop_assert_eq!(
            station_relation_test_access::closure_acceptance_view(
                station_count,
                relations.len(),
                false,
                usize::MAX,
                f64::INFINITY,
                &[(0, 1, f64::INFINITY)],
            ),
            station_relation_test_access::ClosureAcceptanceView::NoConstraints,
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 77: closure
    // degradation is recorded as geometry evidence while the Quality_Gate's
    // complete measured criterion set, thresholds, and conclusion remain
    // unchanged.
    //
    // **Validates: Requirements 7.7, 7.9, 11.15, 12.3**
    #[test]
    fn property_77_closure_degradation_preserves_quality_gate_record(
        seed in any::<u64>(),
        reliable in any::<bool>(),
    ) {
        use crate::panorama_stitching::station_relation_test_access::{
            self, ClosureRelationView,
        };
        use crate::panorama_utils::stack_pipeline::quality_gate::{self, SourceGeometry};
        use crate::panorama_utils::stack_pipeline::quality_gate_runner::{run_quality_gate, QualityGateInput, QualitySource};
        use crate::panorama_utils::stack_pipeline::residual_warp::ResidualWarp;

        let _run_scope = degradation::begin_run_scope();
        degradation::reset_run_ledger();

        // Run the production Quality_Gate twice around the production closure
        // boundary. The fixture has one full 512px owner region so all nine
        // criterion records are built by `run_quality_gate`, rather than a
        // hand-written report standing in for its output.
        let run_quality_gate_fixture = || {
            let width = 512;
            let height = 512;
            let output = image::Rgb32FImage::from_fn(width, height, |x, y| {
                let ramp = ((x.wrapping_mul(17) + y.wrapping_mul(31)) % 251) as f32 / 251.0;
                image::Rgb([0.2 + ramp * 0.6, 0.25 + ramp * 0.5, 0.3 + ramp * 0.4])
            });
            let coverage = image::GrayImage::from_pixel(width, height, image::Luma([255]));
            let ownership = vec![1u16; (width * height) as usize];
            let confidence = vec![0.8f32; (width * height) as usize];
            let textured = vec![255u8; (width * height) as usize];
            let shortfall = vec![0u8; (width * height) as usize];
            let disagreement = vec![0u8; (width * height) as usize];
            let residual = ResidualWarp::new();
            let source = QualitySource {
                owner: 1,
                path: "property-77-owner.raw".to_string(),
                geometry: SourceGeometry {
                    member_to_anchor: Matrix3::identity(),
                    tile_to_world: Matrix3::identity(),
                    station_id: 0,
                },
                dimensions: (width, height),
                native_warp: None,
                native_registration: None,
            };
            let input = QualityGateInput {
                output: &output,
                coverage: &coverage,
                ownership: &ownership,
                confidence: &confidence,
                textured: Some(&textured),
                owner_shortfall: Some(&shortfall),
                owner_disagreement: Some(&disagreement),
                unresolved_pixel_count: 0,
                world_origin: (0.0, 0.0),
                sources: std::slice::from_ref(&source),
                accepted_station_relations: &[],
                residual: &residual,
                acceptance_render_scale: 1.0,
                final_sharpen_amount: 0.0,
            };
            run_quality_gate(&input, &mut |_source| Ok(output.clone()))
        };
        let quality_before = run_quality_gate_fixture();

        let (run, expected_status) = if reliable {
            let (dimensions, rows, columns, base, relations) = accepted_closure_fixture(seed);
            (
                station_relation_test_access::closure_run(
                    dimensions,
                    &rows,
                    &columns,
                    &base,
                    &relations,
                ),
                crate::panorama_utils::stack_pipeline::report::ClosureStatus::Converged,
            )
        } else {
            let signed_zero = if seed & 1 == 0 { 0.0 } else { -0.0 };
            let base = vec![
                translation_pose(signed_zero, -0.0),
                translation_pose(100.25, 0.0),
                translation_pose(200.5, 0.0),
            ];
            let relation = |left, right, tx| ClosureRelationView {
                score: 1.0,
                left,
                right,
                left_to_right: translation_pose(tx, 0.0),
                independent_support: 2,
                median_error_px: 0.0,
            };
            (
                station_relation_test_access::closure_run(
                    (400, 400),
                    &[2, 0, 1],
                    &[1, 2, 0],
                    &base,
                    &[
                        relation(0, 1, -109.25),
                        relation(1, 2, -109.25),
                        relation(0, 2, -191.5),
                    ],
                ),
                crate::panorama_utils::stack_pipeline::report::ClosureStatus::Unreliable,
            )
        };
        prop_assert_eq!(run.report.status, expected_status);
        let quality_after = run_quality_gate_fixture();

        let criterion_signature = |report: &QualityGateReport| {
            report
                .criteria
                .iter()
                .map(|criterion| {
                    (
                        criterion.name.clone(),
                        criterion.threshold,
                        criterion.threshold_min,
                        criterion.threshold_max,
                        criterion.verdict,
                    )
                })
                .collect::<Vec<_>>()
        };
        prop_assert_eq!(quality_before.verdict, quality_after.verdict);
        prop_assert_eq!(criterion_signature(&quality_before), criterion_signature(&quality_after));
        prop_assert_eq!(
            quality_before.criteria.iter().map(|criterion| criterion.name.as_str()).collect::<Vec<_>>(),
            quality_gate::QUALITY_CRITERIA.to_vec(),
        );
        for criterion in &quality_after.criteria {
            let expected_threshold = match criterion.name.as_str() {
                "local_scale_median" => quality_gate::QUALITY_LOCAL_SCALE_MEDIAN_MIN,
                "local_scale_pixel_ratio" => quality_gate::QUALITY_LOCAL_SCALE_PIXEL_RATIO_MIN,
                "effective_pixel_count" => quality_gate::QUALITY_EFFECTIVE_PIXEL_RATIO_MIN,
                "mtf50_normalized" => quality_gate::QUALITY_MTF50_RATIO_MIN,
                "gradient_energy_normalized" => quality_gate::QUALITY_GRADIENT_RATIO_MIN,
                "noise_sigma_ratio" => quality_gate::QUALITY_NOISE_RATIO_MIN,
                "roi_delta_e00" => quality_gate::QUALITY_ROI_DELTA_E00_MAX,
                "boundary_stroke_alignment" => quality_gate::QUALITY_BOUNDARY_P95_MAX,
                "owner_sharpness_coverage" => quality_gate::OWNER_SHARPNESS_COVERAGE_MIN,
                _ => unreachable!("run_quality_gate emitted an unknown criterion"),
            };
            prop_assert_eq!(criterion.threshold, expected_threshold);
        }

        let ledger = degradation::run_ledger_snapshot();
        let recorder = crate::panorama_utils::stack_pipeline::report::StackReportRecorder::isolated("property-77", None);
        recorder.update(|report| {
            report.quality_gate = quality_after.clone();
            report.closure = run.report.clone();
        });
        recorder.apply_degradation_ledger(&ledger);
        let report = recorder.snapshot();
        prop_assert_eq!(report.closure, run.report);
        if reliable {
            let has_unreliable_degradation = report.degradation.entries.iter().any(|entry| {
                matches!(
                    entry.reason.as_str(),
                    degradation::CLOSURE_UNRELIABLE_RESIDUAL
                        | degradation::CLOSURE_UNRELIABLE_ITERATIONS
                        | degradation::CLOSURE_UNRELIABLE_PAIR_P95
                )
            });
            prop_assert!(!has_unreliable_degradation);
        } else {
            let has_unreliable_degradation = report.degradation.entries.iter().any(|entry| {
                matches!(
                    entry.reason.as_str(),
                    degradation::CLOSURE_UNRELIABLE_RESIDUAL
                        | degradation::CLOSURE_UNRELIABLE_ITERATIONS
                        | degradation::CLOSURE_UNRELIABLE_PAIR_P95
                )
            });
            prop_assert!(has_unreliable_degradation);
        }
    }
}
