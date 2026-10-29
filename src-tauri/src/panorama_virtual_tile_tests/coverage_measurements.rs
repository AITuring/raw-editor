//! Coverage-aware measurements: requirement 6.8 on covered pixels, directly and through the
//! station solver, and requirement 6.4 over the doubly covered overlap.

use super::*;

// ---------------------------------------------------------------------------
// 4. coverage-aware requirement-6.8 measurements
// ---------------------------------------------------------------------------

/// 256x160 planes: the 60x36 sample grid sits at x = 4c + 2, y = 4r + 2.
const QUALITY_PLANE: (u32, u32) = (256, 160);
const QUALITY_GRID_SAMPLES: usize = 60 * 36;

#[test]
fn focus_overlap_quality_on_coverage_reads_only_covered_payload() {
    let (width, height) = QUALITY_PLANE;
    let texture = textured_plane(width, height);
    let hole = |x: u32| (96..176).contains(&x);
    let source = plane_info(1, texture.clone());
    let target = plane_info(
        2,
        GrayImage::from_fn(width, height, |x, y| {
            if hole(x) {
                image::Luma([0])
            } else {
                *texture.get_pixel(x, y)
            }
        }),
    );
    let source_coverage = full_coverage(width, height);
    let target_coverage = coverage_from_fn(width, height, |x, _| !hole(x));
    let identity = Matrix3::identity();

    let unmasked = focus_overlap_quality_measurements(&source, &target, &identity).unwrap();
    assert_eq!(unmasked.samples, QUALITY_GRID_SAMPLES);
    assert!(
        unmasked.low_frequency_mean_relative_difference > 0.1,
        "uncovered zeros contaminate the mean: {unmasked:?}"
    );
    assert!(
        unmasked.edge_strength_ratio > 1.1,
        "uncovered zeros contaminate the edges: {unmasked:?}"
    );

    let masked = focus_overlap_quality_measurements_on_coverage(
        &source,
        &target,
        &identity,
        Some((&source_coverage, &target_coverage)),
    )
    .unwrap();
    assert!(
        masked.low_frequency_mean_relative_difference < 1e-12,
        "{masked:?}"
    );
    assert!(
        (masked.edge_strength_ratio - 1.0).abs() < 1e-12,
        "{masked:?}"
    );
    assert!(masked.intensity_ncc > 0.999_999, "{masked:?}");
    assert!(masked.median_edge_orientation_difference_degrees < 1e-6);
    // Columns whose ±4px neighbourhood meets [96, 176) are excluded.
    let kept_columns = (2..62u32)
        .map(|column| 4 * column + 2)
        .filter(|&x| x + 4 < 96 || x - 4 > 175)
        .count();
    assert_eq!(kept_columns, 38);
    assert_eq!(masked.samples, kept_columns * 36);

    // Fully covered masks sample exactly what the unmasked measurement does.
    let full = focus_overlap_quality_measurements_on_coverage(
        &source,
        &target,
        &identity,
        Some((&source_coverage, &source_coverage)),
    )
    .unwrap();
    assert_eq!(full.samples, unmasked.samples);
    assert_eq!(
        full.low_frequency_mean_relative_difference,
        unmasked.low_frequency_mean_relative_difference
    );
    assert_eq!(full.edge_strength_ratio, unmasked.edge_strength_ratio);
}

#[test]
fn focus_overlap_quality_on_coverage_excludes_samples_within_four_pixels_of_uncovered_pixels() {
    assert_eq!(VIRTUAL_TILE_QUALITY_COVERAGE_RADIUS_PX, 4.0);
    let (width, height) = QUALITY_PLANE;
    let texture = textured_plane(width, height);
    let source = plane_info(1, texture.clone());
    let target = plane_info(2, texture);
    let full = full_coverage(width, height);
    let identity = Matrix3::identity();
    let baseline = focus_overlap_quality_measurements_on_coverage(
        &source,
        &target,
        &identity,
        Some((&full, &full)),
    )
    .unwrap()
    .samples;
    assert_eq!(baseline, QUALITY_GRID_SAMPLES);
    // (102, 50) is itself a grid sample: x ∈ {98, 102, 106} × y ∈ {46, 50, 54}
    // lie within 4px (radius 3 would exclude 1). (103, 51): only
    // x ∈ {102, 106} × y ∈ {50, 54} (radius 5 would exclude 9).
    for ((hole_x, hole_y), excluded) in [((102u32, 50u32), 9usize), ((103, 51), 4)] {
        let hole = coverage_from_fn(width, height, |x, y| (x, y) != (hole_x, hole_y));
        for (source_coverage, target_coverage, side) in
            [(&hole, &full, "source"), (&full, &hole, "target")]
        {
            let quality = focus_overlap_quality_measurements_on_coverage(
                &source,
                &target,
                &identity,
                Some((source_coverage, target_coverage)),
            )
            .unwrap();
            assert_eq!(
                quality.samples,
                baseline - excluded,
                "{side} hole at ({hole_x}, {hole_y})"
            );
        }
    }
}

#[test]
fn focus_overlap_quality_on_coverage_rejects_mismatched_masks_and_sparse_overlap() {
    let (width, height) = QUALITY_PLANE;
    let texture = textured_plane(width, height);
    let source = plane_info(1, texture.clone());
    let target = plane_info(2, texture);
    let full = full_coverage(width, height);
    let identity = Matrix3::identity();
    let measure = |source_coverage: &stitching::CoverageMask,
                   target_coverage: &stitching::CoverageMask| {
        focus_overlap_quality_measurements_on_coverage(
            &source,
            &target,
            &identity,
            Some((source_coverage, target_coverage)),
        )
    };
    let narrower = full_coverage(width - 1, height);
    let shorter = full_coverage(width, height - 1);
    assert!(measure(&narrower, &full).is_none());
    assert!(measure(&full, &narrower).is_none());
    assert!(measure(&shorter, &full).is_none());
    // Covered strip [0, 24): x ∈ {10, 14, 18} → 108 samples < 120.
    let sparse = coverage_from_fn(width, height, |x, _| x < 24);
    assert!(measure(&full, &sparse).is_none());
    assert!(measure(&sparse, &full).is_none());
    // Covered strip [0, 28): x ∈ {10, 14, 18, 22} → 144 samples.
    let enough = coverage_from_fn(width, height, |x, _| x < 28);
    assert_eq!(measure(&full, &enough).unwrap().samples, 144);
}

// ---------------------------------------------------------------------------
// 4b. coverage-aware requirement-6.8 re-check through the station solver
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_solver_rechecks_requirement_6_8_on_covered_pixels_only() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    // Tile 1 shows tile 0 shifted by (40, 6) but carries a zero-valued,
    // uncovered band x ∈ [150, 250) inside the overlap.
    let (width, height) = (400u32, 300u32);
    let band = |x: u32| (150..250).contains(&x);
    let left = textured_plane(width, height);
    let right = GrayImage::from_fn(width, height, |x, y| {
        if band(x) {
            image::Luma([0])
        } else {
            image::Luma([textured_value(f64::from(x) + 40.0, f64::from(y) + 6.0).round() as u8])
        }
    });
    let tiles = vec![plane_info(0, left), plane_info(1, right)];
    let left_to_right = translation(-40.0, -6.0);
    let points = [60.0, 100.0, 140.0, 180.0, 300.0, 340.0, 380.0]
        .into_iter()
        .flat_map(|x| (0..6).map(move |row| Point2::new(x, 30.0 + 50.0 * f64::from(row))))
        .map(|source| (source, transformed_point(&left_to_right, source).unwrap()))
        .collect::<Vec<_>>();
    let matches = HashMap::from([((0, 1), measured_relation(left_to_right, points))]);
    let initial = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, left_to_right.try_inverse().unwrap()),
    ]);
    let station_topology = candidate_topology(2);
    let solve = |coverages: &[stitching::CoverageMask]| {
        let mut closure = report::ClosureReport::default();
        let mut relations = report::StationRelationsReport::default();
        let solved = solve_virtual_tile_station_poses_with_report(
            &tiles,
            VirtualTileSolverCoverages {
                quality: coverages,
                overlap: coverages,
                overlap_factor: 1,
            },
            &matches,
            &initial,
            &station_topology,
            &mut closure,
            &mut relations,
        );
        (solved, relations)
    };

    let masked = [
        full_coverage(width, height),
        coverage_from_fn(width, height, |x, _| !band(x)),
    ];
    let (solved, relations) = solve(&masked);
    assert_eq!(solved.len(), 2, "{relations:?}");
    assert_eq!(relations.accepted.len(), 1);

    // Treating the band as content lets its zeros reach the low-frequency and
    // edge-strength statistics, which then reject the same relation.
    let unmasked = [full_coverage(width, height), full_coverage(width, height)];
    let (solved, relations) = solve(&unmasked);
    assert!(solved.is_empty(), "{relations:?}");
    assert!(relations.accepted.is_empty());
    assert!(
        relations
            .rejected_by_reason
            .contains_key("station_relation_photometric_mismatch"),
        "{relations:?}"
    );
}

// ---------------------------------------------------------------------------
// requirement 6.4 over the doubly covered overlap
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_covered_overlap_area_counts_only_doubly_covered_pixels() {
    let (width, height) = (100u32, 80u32);
    let source = coverage_from_fn(width, height, |x, y| !(10..30).contains(&x) || y >= 40);
    let target = coverage_from_fn(width, height, |x, y| x < 90 && !(50..60).contains(&y));
    let count = |shift: u32| {
        (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                x + shift < width && source.is_covered(x, y) && target.is_covered(x + shift, y)
            })
            .count() as f64
    };
    let area = |relation: &Matrix3<f64>| {
        virtual_tile_covered_overlap_area(
            &source,
            (width, height),
            &target,
            (width, height),
            1,
            relation,
        )
        .unwrap()
    };
    assert_eq!(area(&Matrix3::identity()), count(0));
    assert_eq!(area(&translation(20.0, 0.0)), count(20));

    // A factor-2 mask pixel is a 2×2 block of relation pixels.
    let coarse_source = full_coverage(50, 40);
    let coarse_target = coverage_from_fn(50, 40, |x, _| x < 25);
    let coarse = virtual_tile_covered_overlap_area(
        &coarse_source,
        (100, 80),
        &coarse_target,
        (101, 81),
        2,
        &Matrix3::identity(),
    )
    .unwrap();
    assert_eq!(coarse, 25.0 * 40.0 * 4.0);

    // A mask that does not tile its plane is no measurement.
    assert!(
        virtual_tile_covered_overlap_area(
            &source,
            (width + 2, height),
            &target,
            (width, height),
            1,
            &Matrix3::identity(),
        )
        .is_none()
    );
}

#[test]
fn covered_overlap_support_ignores_the_uncovered_part_of_the_rectangle_overlap() {
    let (width, height) = (200u32, 100u32);
    // Identity relation, the right half of the source uncovered: the covered
    // overlap is half of the rectangle overlap.
    let source = coverage_from_fn(width, height, |x, _| x < 100);
    let target = full_coverage(width, height);
    let points = [
        (20.0, 20.0),
        (80.0, 20.0),
        (80.0, 70.0),
        (20.0, 70.0),
        (50.0, 45.0),
    ]
    .map(|(x, y)| (Point2::new(x, y), Point2::new(x, y)));
    let hull = 60.0 * 50.0;
    let rectangles = station_relation_spatial_support(
        &points,
        &Matrix3::identity(),
        (width, height),
        (width, height),
    );
    let covered = station_relation_spatial_support_with(
        &points,
        &Matrix3::identity(),
        (width, height),
        (width, height),
        StationOverlapMeasure::Covered {
            source: &source,
            target: &target,
            factor: 1,
        },
    );
    assert!((rectangles - hull / 20_000.0).abs() < 1e-9, "{rectangles}");
    assert!((covered - hull / 10_000.0).abs() < 1e-9, "{covered}");
    // Fully covered masks reproduce the rectangle measure.
    let full = station_relation_spatial_support_with(
        &points,
        &Matrix3::identity(),
        (width, height),
        (width, height),
        StationOverlapMeasure::Covered {
            source: &target,
            target: &target,
            factor: 1,
        },
    );
    assert!((full - rectangles).abs() < 1e-9);
    // No doubly covered pixel: no support at all.
    let empty = coverage_from_fn(width, height, |_, _| false);
    assert_eq!(
        station_relation_spatial_support_with(
            &points,
            &Matrix3::identity(),
            (width, height),
            (width, height),
            StationOverlapMeasure::Covered {
                source: &empty,
                target: &target,
                factor: 1,
            },
        ),
        0.0
    );
}

#[test]
fn virtual_tile_solver_measures_requirement_6_4_on_the_doubly_covered_overlap() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    // Tile 1 shows tile 0 shifted by (40, 6) everywhere, but tile 0 is only
    // covered for x < 220: half of the rectangle overlap holds no tile content.
    let (width, height) = (400u32, 300u32);
    let left = textured_plane(width, height);
    let right = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([textured_value(f64::from(x) + 40.0, f64::from(y) + 6.0).round() as u8])
    });
    let tiles = vec![plane_info(0, left), plane_info(1, right)];
    let left_to_right = translation(-40.0, -6.0);
    // 36 inliers spanning a 100×150 hull: 14% of the 360×294 rectangle
    // overlap, 28% of the 180×294 doubly covered overlap.
    let points = (0..6)
        .flat_map(|row| (0..6).map(move |column| (row, column)))
        .map(|(row, column)| {
            let source = Point2::new(
                80.0 + 20.0 * f64::from(column),
                60.0 + 30.0 * f64::from(row),
            );
            (source, transformed_point(&left_to_right, source).unwrap())
        })
        .collect::<Vec<_>>();
    let matches = HashMap::from([((0, 1), measured_relation(left_to_right, points))]);
    let initial = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, left_to_right.try_inverse().unwrap()),
    ]);
    let station_topology = candidate_topology(2);
    let solve = |coverages: &[stitching::CoverageMask]| {
        let mut closure = report::ClosureReport::default();
        let mut relations = report::StationRelationsReport::default();
        let solved = solve_virtual_tile_station_poses_with_report(
            &tiles,
            VirtualTileSolverCoverages {
                quality: coverages,
                overlap: coverages,
                overlap_factor: 1,
            },
            &matches,
            &initial,
            &station_topology,
            &mut closure,
            &mut relations,
        );
        (solved, relations)
    };

    let covered = [
        coverage_from_fn(width, height, |x, _| x < 220),
        full_coverage(width, height),
    ];
    let (solved, relations) = solve(&covered);
    assert_eq!(solved.len(), 2, "{relations:?}");
    let support = relations.accepted[0].spatial_support;
    assert!(
        (support - 15_000.0 / (180.0 * 294.0)).abs() < 0.01,
        "{support}"
    );

    // Counting the uncovered half as overlap drops the same evidence to 14%.
    let pretended = [full_coverage(width, height), full_coverage(width, height)];
    let (solved, relations) = solve(&pretended);
    assert!(solved.is_empty(), "{relations:?}");
    assert!(
        relations
            .rejected_by_reason
            .contains_key("station_relation_low_spatial_support"),
        "{relations:?}"
    );
}
