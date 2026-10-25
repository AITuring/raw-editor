// Closure_Optimizer properties 35-39 (requirements 7.3-7.9)
// ---------------------------------------------------------------------------

fn closure_weight_numeric_oracle(residual_px: f64) -> f64 {
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

fn closure_stop_numeric_oracle(previous: f64, current: f64) -> bool {
    if !previous.is_finite() || !current.is_finite() || current > previous {
        return false;
    }
    if previous <= f64::EPSILON {
        return current <= f64::EPSILON;
    }
    (previous - current) / previous < 1.0e-4
}

fn closure_summary_numeric_oracle(mut residuals: Vec<f64>) -> (f64, f64) {
    assert!(!residuals.is_empty());
    assert!(residuals.iter().all(|residual| residual.is_finite()));
    residuals.sort_by(f64::total_cmp);
    let median = residuals[residuals.len() / 2];
    let p95 = residuals[(residuals.len() * 95).div_ceil(100) - 1];
    (median, p95)
}

fn closure_relation_residual_numeric_oracle(
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

fn closure_direct_pair_p95_numeric_oracle(
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

fn accepted_closure_fixture(
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

fn assert_pose_bits_equal(actual: &[Matrix3<f64>], expected: &[Matrix3<f64>]) {
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
        prop_assert_eq!(run.report.status, super::report::ClosureStatus::Converged);
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
        prop_assert_eq!(run.report.status, super::report::ClosureStatus::Converged);
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
        prop_assert_eq!(run.report.status, super::report::ClosureStatus::Converged);
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
        prop_assert_eq!(residual.report.status, super::report::ClosureStatus::Unreliable);
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
        prop_assert_eq!(iterations.report.status, super::report::ClosureStatus::Unreliable);
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
        prop_assert_eq!(pair.report.status, super::report::ClosureStatus::Unreliable);
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
        prop_assert_eq!(run.report.status, super::report::ClosureStatus::NoConstraints);
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

// Stage 7 backfill: task 11.8 (Property 41).
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 41: 局部形变只在 P95 > 3 的区域启用，64px 网格每方向至少四节点。
    // **Validates: Requirements 8.1, 8.2**
    #[test]
    fn property_41_residual_regions_exactly_match_global_p95_gate(
        origin in (-4096i32..4096, -4096i32..4096),
        overlaps in prop::collection::vec(
            (
                1u16..1025,
                1u16..1025,
                prop_oneof![Just(3.0f64), Just(3.0f64 + f64::EPSILON * 2.0), 0.0f64..12.0],
            ),
            0..16,
        ),
    ) {
        use super::report::{WorldPoint, WorldRect};
        use super::residual_warp::{OverlapResidual, ResidualWarp};

        let mut model = ResidualWarp::default();
        prop_assert!(model.regions.is_empty());
        prop_assert!(model.identity());
        let mut expected = Vec::new();
        for (index, &(width, height, p95)) in overlaps.iter().enumerate() {
            let world = WorldRect {
                left: origin.0 as f64 + index as f64 * 2048.0,
                top: origin.1 as f64,
                width: f64::from(width),
                height: f64::from(height),
            };
            let overlap = OverlapResidual::new(index * 2, index * 2 + 1, world, p95);
            let inserted = model.add_overlap(overlap);
            if p95 > 3.0 {
                prop_assert_eq!(inserted, Some(expected.len()));
                expected.push(overlap);
            } else {
                prop_assert_eq!(inserted, None);
                let mut measured = ResidualWarp::default();
                prop_assert_eq!(measured.add_observations(overlap, &[]), None);
                prop_assert!(measured.regions.is_empty());
            }
        }
        prop_assert_eq!(model.regions.len(), expected.len());
        for (region, overlap) in model.regions.iter().zip(&expected) {
            prop_assert_eq!(region.world, overlap.world);
            prop_assert_eq!((region.left_station, region.right_station), (overlap.left_station, overlap.right_station));
            prop_assert_eq!(region.p95_before_px, overlap.p95_px);
            prop_assert_eq!(region.node_step_px, 64);
            prop_assert_eq!(region.columns, ((overlap.world.width / 64.0).ceil() as u32 + 1).max(4));
            prop_assert_eq!(region.rows, ((overlap.world.height / 64.0).ceil() as u32 + 1).max(4));
            prop_assert_eq!(region.node_count(), region.columns as usize * region.rows as usize);
            for (index, node) in region.nodes().iter().enumerate() {
                let column = index % region.columns as usize;
                let row = index / region.columns as usize;
                prop_assert_eq!(node.world, WorldPoint {
                    x: overlap.world.left + column as f64 * 64.0,
                    y: overlap.world.top + row as f64 * 64.0,
                });
                prop_assert_eq!(node.displacement, [0.0, 0.0]);
            }
        }
        // Merely allocating an enabled region must not invent a deformation.
        prop_assert!(model.identity());
        for (station, point) in [
            (usize::MAX, Point2::new(-0.0, origin.1 as f64)),
            (0, Point2::new(origin.0 as f64 - 1.0, origin.1 as f64)),
        ] {
            let mapped = model.warp_inverse(point, station);
            prop_assert_eq!(mapped.x.to_bits(), point.x.to_bits());
            prop_assert_eq!(mapped.y.to_bits(), point.y.to_bits());
        }
    }
}

// Stage 7 backfill snippets for tasks 11.9–11.14. Append each `proptest!`
// block independently to properties.rs after task 11.8 is committed.

// Task 11.9 / Property 42.
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 42: 写入的节点位移同时满足位移、邻接差与往返误差三项约束。
    // **Validates: Requirements 8.3, 8.4, 8.5**
    #[test]
    fn property_42_residual_nodes_satisfy_all_three_gates(
        displacement in (-8.0f64..8.0, -8.0f64..8.0),
        round_trip in 0.0f64..=1.0,
    ) {
        use super::report::WorldRect;
        use super::residual_warp::{OverlapResidual, WarpObservation};
        let overlap = OverlapResidual::new(1, 2, WorldRect { left: 0.0, top: 0.0, width: 192.0, height: 192.0 }, 4.0);
        let observations = (0..4).flat_map(|row| (0..4).map(move |column| WarpObservation::new(
            super::report::WorldPoint { x: column as f64 * 64.0, y: row as f64 * 64.0 },
            [displacement.0, displacement.1], round_trip,
        ))).collect::<Vec<_>>();
        let region = super::residual_warp::WarpRegion::from_observations(overlap, &observations);
        if region.insufficient_evidence {
            prop_assert!(region.is_identity());
        }
        for node in region.nodes() {
            prop_assert!(node.displacement[0].hypot(node.displacement[1]) <= super::residual_warp::RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX + 1e-9);
            if node.valid {
                prop_assert!(node.round_trip_error_px <= super::residual_warp::RESIDUAL_WARP_MAX_ROUND_TRIP_ERROR_PX + 1e-9);
            }
        }
        for row in 0..region.rows as usize {
            for column in 0..region.columns as usize {
                let index = row * region.columns as usize + column;
                let mut neighbours = Vec::with_capacity(2);
                if row > 0 {
                    neighbours.push((row - 1) * region.columns as usize + column);
                }
                if column > 0 {
                    neighbours.push(row * region.columns as usize + column - 1);
                }
                for other in neighbours {
                    let delta = (region.nodes()[index].displacement[0] - region.nodes()[other].displacement[0])
                        .hypot(region.nodes()[index].displacement[1] - region.nodes()[other].displacement[1]);
                    prop_assert!(delta <= super::residual_warp::RESIDUAL_WARP_MAX_NEIGHBOUR_DELTA_PX + 1e-7);
                }
            }
        }
    }
}

// Task 11.10 / Property 43.
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 43: 局部形变不优于全局单应性的网格单元回退且计数准确。
    // **Validates: Requirements 8.6**
    #[test]
    fn property_43_residual_cell_reverts_match_error_ordering(
        global in prop::collection::vec(0.0f64..20.0, 9),
        residual in prop::collection::vec(0.0f64..20.0, 9),
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use super::report::WorldRect;
        use super::residual_warp::{OverlapResidual, WarpObservation, WarpRegion};
        let overlap = OverlapResidual::new(0, 1, WorldRect { left: 0.0, top: 0.0, width: 192.0, height: 192.0 }, 4.0);
        let mut region = WarpRegion::from_observations(overlap, &(0..16).map(|index| WarpObservation::new(
            super::report::WorldPoint { x: (index % 4) as f64 * 64.0, y: (index / 4) as f64 * 64.0 }, [1.0, 0.0], 0.0,
        )).collect::<Vec<_>>());
        region.revert_cells(&global, &residual);
        let expected = global.iter().zip(&residual).filter(|(g, r)| **r >= **g).count();
        prop_assert_eq!(region.reverted_cells as usize, expected);
    }
}

// Task 11.12 / Property 45.
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 45: 无效节点只从三节点半径内有效邻域外推，无邻域时严格为零。
    // **Validates: Requirements 8.8**
    #[test]
    fn property_45_invalid_nodes_use_only_three_node_valid_support(
        value in 0.0f64..8.0,
    ) {
        use super::report::{WorldPoint, WorldRect};
        use super::residual_warp::{OverlapResidual, WarpObservation, WarpRegion};
        let overlap = OverlapResidual::new(0, 1, WorldRect { left: 0.0, top: 0.0, width: 704.0, height: 704.0 }, 4.0);
        let observations = (0..4).flat_map(|row| (0..4).map(move |column| WarpObservation::new(
            WorldPoint { x: column as f64 * 64.0, y: row as f64 * 64.0 }, [value, 0.0], 0.0,
        ))).collect::<Vec<_>>();
        let region = WarpRegion::from_observations(overlap, &observations);
        for node in region.nodes().iter().filter(|node| !node.valid) {
            let supporting = region.nodes().iter().filter(|candidate| candidate.valid)
                .map(|candidate| (candidate.world.x - node.world.x).hypot(candidate.world.y - node.world.y) / 64.0)
                .fold(f64::INFINITY, f64::min);
            if supporting > super::residual_warp::RESIDUAL_WARP_EXTRAPOLATION_RADIUS_NODES {
                prop_assert_eq!(node.displacement, [0.0, 0.0]);
            } else if node.displacement != [0.0, 0.0] {
                prop_assert!(supporting <= super::residual_warp::RESIDUAL_WARP_EXTRAPOLATION_RADIUS_NODES);
            }
        }
    }
}

// Task 11.13 / Property 46.
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 46: 形变沿边界法向在128px内单调衰减为零，区域外恒等。
    // **Validates: Requirements 8.9**
    #[test]
    fn property_46_residual_edge_fade_is_monotonic(
        displacement in 1.0f64..32.0,
    ) {
        use super::report::{WorldPoint, WorldRect};
        use super::residual_warp::{OverlapResidual, WarpObservation, WarpRegion};
        let overlap = OverlapResidual::new(0, 1, WorldRect { left: 0.0, top: 0.0, width: 1024.0, height: 1024.0 }, 4.0);
        let observations = (0..17).flat_map(|row| (0..17).map(move |column| WarpObservation::new(
            WorldPoint { x: column as f64 * 64.0, y: row as f64 * 64.0 }, [displacement, 0.0], 0.0,
        ))).collect::<Vec<_>>();
        let region = WarpRegion::from_observations(overlap, &observations);
        let mut previous = 0.0;
        for distance in (0..=128).step_by(8) {
            let current = region.displacement_at(distance as f64, 512.0)[0];
            prop_assert!(current + 1e-8 >= previous);
            previous = current;
        }
        prop_assert_eq!(region.displacement_at(-1.0, 512.0), [0.0, 0.0]);
        prop_assert_eq!(region.displacement_at(1025.0, 512.0), [0.0, 0.0]);
    }
}

// Task 11.14 / Property 47.
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 100,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::SourceParallel("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    // Feature: layered-camera-group-focus-stitching, Property 47: 证据不足时区域保持恒等并记录不足证据。
    // **Validates: Requirements 8.10**
    #[test]
    fn property_47_insufficient_residual_evidence_is_identity(
        observations in prop::collection::vec(
            (0.0f64..192.0, 0.0f64..192.0, -8.0f64..8.0, -8.0f64..8.0),
            0..16,
        ),
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use super::report::{WorldPoint, WorldRect};
        use super::residual_warp::{OverlapResidual, WarpObservation, WarpRegion, RESIDUAL_WARP_MIN_VERIFIED_POINTS};
        let overlap = OverlapResidual::new(0, 1, WorldRect { left: 0.0, top: 0.0, width: 192.0, height: 192.0 }, 3.01);
        let samples = observations.iter().map(|&(x, y, dx, dy)| WarpObservation::new(WorldPoint { x, y }, [dx, dy], 1.0)).collect::<Vec<_>>();
        let region = WarpRegion::from_observations(overlap, &samples);
        prop_assert!(samples.len() < RESIDUAL_WARP_MIN_VERIFIED_POINTS);
        prop_assert!(region.insufficient_evidence);
        prop_assert!(region.is_identity());
        for node in region.nodes() {
            prop_assert_eq!(node.displacement, [0.0, 0.0]);
        }
    }
}
