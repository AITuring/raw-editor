//! Checkpoint 10 closure on a synthetic two-dimensional scan grid: tree drift, cross-station
//! scale drift and perspective drift are corrected within the residual bounds.

use super::*;

// ---------------------------------------------------------------------------
// checkpoint 10: closure on a synthetic two-dimensional scan grid
// ---------------------------------------------------------------------------

/// Worst reprojection disagreement of one translation relation under `poses`,
/// on a lattice over the left tile clipped to the right tile.
fn translation_pair_error(
    dimensions: (u32, u32),
    poses: &[Matrix3<f64>],
    left: usize,
    right: usize,
    left_to_right: &Matrix3<f64>,
) -> f64 {
    let (width, height) = (f64::from(dimensions.0), f64::from(dimensions.1));
    let mut worst = 0.0f64;
    for row in 0..=10 {
        for column in 0..=10 {
            let source = Point2::new(
                (width - 1.0) * f64::from(column) / 10.0,
                (height - 1.0) * f64::from(row) / 10.0,
            );
            let target = transformed_point(left_to_right, source).unwrap();
            if !(0.0..width).contains(&target.x) || !(0.0..height).contains(&target.y) {
                continue;
            }
            let error = (transformed_point(&poses[left], source).unwrap()
                - transformed_point(&poses[right], target).unwrap())
            .norm();
            worst = worst.max(error);
        }
    }
    worst
}

#[test]
fn checkpoint_ten_scan_grid_closure_corrects_tree_drift_within_the_residual_bounds() {
    use crate::panorama_stitching::station_relation_test_access::{self, ClosureRelationView};

    // A 3×3 serpentine scan; each spanning-tree step along the scan path adds
    // (0.6, −0.4)px of drift, so the tree alone misplaces loop partners by up
    // to five steps (3.6px) while every relation measures the true geometry.
    let dimensions = (512u32, 384u32);
    let serpentine = [
        (0u32, 0u32),
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 1),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
    ];
    let world = |station: usize| {
        let (row, column) = serpentine[station];
        (f64::from(column) * 360.0, f64::from(row) * 270.0)
    };
    let drift = nalgebra::Vector2::new(0.6, -0.4);
    let base = (0..serpentine.len())
        .map(|station| {
            let (x, y) = world(station);
            translation(x + drift.x * station as f64, y + drift.y * station as f64)
        })
        .collect::<Vec<_>>();
    let station_of = |row: u32, column: u32| {
        serpentine
            .iter()
            .position(|&cell| cell == (row, column))
            .unwrap()
    };
    let mut relations = Vec::new();
    for row in 0..3u32 {
        for column in 0..3u32 {
            for (next_row, next_column) in [(row, column + 1), (row + 1, column)] {
                if next_row > 2 || next_column > 2 {
                    continue;
                }
                let (a, b) = (station_of(row, column), station_of(next_row, next_column));
                let (left, right) = (a.min(b), a.max(b));
                let ((left_x, left_y), (right_x, right_y)) = (world(left), world(right));
                let noise = |salt: u32| hashed_noise(left as u32, right as u32, salt, 0.25);
                let tree_edge = right == left + 1;
                // Tree edges carry exactly the drifted placement the tree was
                // built from; loop edges carry the true geometry.
                let (dx, dy) = if tree_edge {
                    (-drift.x, -drift.y)
                } else {
                    (noise(1), noise(2))
                };
                relations.push(ClosureRelationView {
                    score: 1.0,
                    left,
                    right,
                    left_to_right: translation(left_x - right_x + dx, left_y - right_y + dy),
                    independent_support: 3,
                    median_error_px: 0.5,
                });
            }
        }
    }
    assert_eq!(relations.len(), 12, "6 horizontal + 6 vertical relations");
    let tree_worst = relations
        .iter()
        .map(|relation| {
            translation_pair_error(
                dimensions,
                &base,
                relation.left,
                relation.right,
                &relation.left_to_right,
            )
        })
        .fold(0.0f64, f64::max);
    assert!(
        tree_worst > 3.0,
        "the drifted tree alone violates the pair bound: {tree_worst}"
    );

    let rows = serpentine.iter().map(|&(row, _)| row).collect::<Vec<_>>();
    let columns = serpentine
        .iter()
        .map(|&(_, column)| column)
        .collect::<Vec<_>>();
    let run =
        station_relation_test_access::closure_run(dimensions, &rows, &columns, &base, &relations);
    assert_eq!(
        run.report.status,
        report::ClosureStatus::Converged,
        "{:?}",
        run.report
    );
    assert_eq!(run.report.participating_constraints, relations.len());
    assert!(run.report.residual_median_px <= 2.0, "{:?}", run.report);
    assert!(run.report.max_direct_pair_p95_px <= 3.0, "{:?}", run.report);
    assert_eq!(run.report.offending_pair, None);
    assert_eq!(run.report.clamped_stations, 0);
    let closed_worst = relations
        .iter()
        .map(|relation| {
            translation_pair_error(
                dimensions,
                &run.poses,
                relation.left,
                relation.right,
                &relation.left_to_right,
            )
        })
        .fold(0.0f64, f64::max);
    assert!(closed_worst <= 3.0, "{closed_worst}");
    assert!(closed_worst < tree_worst);
}

#[test]
fn global_affine_closure_corrects_cross_station_scale_drift_without_relaxing_gates() {
    use crate::panorama_stitching::station_relation_test_access::{self, ClosureRelationView};

    // Three generic stations with a small per-station affine scale drift.  The
    // local edge models are all exact translations, so a translation-only
    // closure leaves a position-dependent error on the long overlap.  The
    // global affine stage must reconcile every edge before the existing 3px
    // direct-pair gate is evaluated.
    let dimensions = (1200u32, 800u32);
    let world_x = |station: usize| station as f64 * 300.0;
    let scale = |station: usize| 1.0 + station as f64 * 0.02;
    let base = (0..3)
        .map(|station| {
            translation(world_x(station), 0.0)
                * Matrix3::new(
                    scale(station),
                    0.0,
                    0.0,
                    0.0,
                    scale(station),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                )
        })
        .collect::<Vec<_>>();
    let mut relations = Vec::new();
    for left in 0..3 {
        for right in (left + 1)..3 {
            relations.push(ClosureRelationView {
                score: 1.0,
                left,
                right,
                left_to_right: translation(world_x(left) - world_x(right), 0.0),
                independent_support: 3,
                median_error_px: 0.5,
            });
        }
    }
    let base_worst = relations
        .iter()
        .map(|relation| {
            translation_pair_error(
                dimensions,
                &base,
                relation.left,
                relation.right,
                &relation.left_to_right,
            )
        })
        .fold(0.0f64, f64::max);
    assert!(
        base_worst > 3.0,
        "scale drift must violate the direct-pair gate: {base_worst}"
    );

    let rows = vec![0, 0, 0];
    let columns = vec![0, 1, 2];
    let run =
        station_relation_test_access::closure_run(dimensions, &rows, &columns, &base, &relations);
    assert_eq!(
        run.report.status,
        report::ClosureStatus::Converged,
        "{:?}",
        run.report
    );
    assert!(run.report.residual_median_px <= 2.0, "{:?}", run.report);
    assert!(run.report.max_direct_pair_p95_px <= 3.0, "{:?}", run.report);
    assert!(run.report.max_direct_pair_p95_px < base_worst);
}

#[test]
fn global_projective_closure_corrects_perspective_drift_without_relaxing_gates() {
    use crate::panorama_stitching::station_relation_test_access::{self, ClosureRelationView};

    // The spanning-tree seed carries a small perspective term at each station.
    // The redundant direct relations are exact translations, so the accumulated
    // denominator drift is visible on the long overlap and cannot be removed by
    // an affine-only correction.
    let dimensions = (1200u32, 800u32);
    let world_x = |station: usize| station as f64 * 300.0;
    let base = (0..4)
        .map(|station| {
            let perspective = station as f64 * 0.000006;
            translation(world_x(station), 0.0)
                * Matrix3::new(
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    perspective,
                    -perspective * 0.25,
                    1.0,
                )
        })
        .collect::<Vec<_>>();
    let mut relations = Vec::new();
    for left in 0..4 {
        for right in (left + 1)..4 {
            relations.push(ClosureRelationView {
                score: 1.0,
                left,
                right,
                left_to_right: translation(world_x(left) - world_x(right), 0.0),
                independent_support: 3,
                median_error_px: 0.5,
            });
        }
    }
    let base_worst = relations
        .iter()
        .map(|relation| {
            translation_pair_error(
                dimensions,
                &base,
                relation.left,
                relation.right,
                &relation.left_to_right,
            )
        })
        .fold(0.0f64, f64::max);
    assert!(
        base_worst > 3.0,
        "perspective drift must violate the direct-pair gate: {base_worst}"
    );

    let rows = vec![0, 0, 0, 0];
    let columns = vec![0, 1, 2, 3];
    let run =
        station_relation_test_access::closure_run(dimensions, &rows, &columns, &base, &relations);
    assert_eq!(
        run.report.status,
        report::ClosureStatus::Converged,
        "{:?}",
        run.report
    );
    assert!(run.report.residual_median_px <= 2.0, "{:?}", run.report);
    assert!(run.report.max_direct_pair_p95_px <= 3.0, "{:?}", run.report);
    assert!(run.report.max_direct_pair_p95_px < base_worst);
}
