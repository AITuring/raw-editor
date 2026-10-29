//! Multi-hypothesis Station_Relation selection: a compliant model when the projective one fails
//! scale, holdout keys, low-hull rejection, canonical inversion, closed scale endpoints and
//! coordinate diagnostics.

use super::*;

fn hypothesis_fixture() -> (
    Vec<ImageInfo>,
    Vec<stitching::CoverageMask>,
    topology::StationTopology,
) {
    let dimensions = (400, 300);
    let texture = textured_plane(dimensions.0, dimensions.1);
    (
        vec![plane_info(0, texture.clone()), plane_info(1, texture)],
        vec![
            full_coverage(dimensions.0, dimensions.1),
            full_coverage(dimensions.0, dimensions.1),
        ],
        candidate_topology(2),
    )
}

#[test]
fn multi_hypothesis_selector_keeps_compliant_model_when_projective_fails_scale() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let (tiles, coverages, topology) = hypothesis_fixture();
    let passing = Matrix3::identity();
    let scale_failing_projective =
        Matrix3::new(1.08, 0.0, 0.0, 0.0, 1.08, 0.0, 0.000_01, -0.000_01, 1.0);
    let hypotheses = vec![
        measured_relation(
            scale_failing_projective,
            hypothesis_points(&scale_failing_projective, tiles[0].dimensions()),
        ),
        measured_relation(passing, hypothesis_points(&passing, tiles[0].dimensions())),
    ];
    let solver_coverages = VirtualTileSolverCoverages {
        quality: &coverages,
        overlap: &coverages,
        overlap_factor: 1,
    };
    let projective_canonical = canonical_relation_view(&tiles, 0, 1, &hypotheses[0]).unwrap();
    let projective_measurement = measure_station_relation_candidate(
        hypotheses[0].inliers,
        &projective_canonical.homography,
        &hypotheses[0].points,
        tiles[0].dimensions(),
        tiles[1].dimensions(),
        solver_coverages.overlap_measure(0, 1),
        focus_overlap_quality_measurements_on_coverage(
            &tiles[0],
            &tiles[1],
            &projective_canonical.homography,
            Some((&coverages[0], &coverages[1])),
        ),
        false,
    )
    .unwrap();
    assert!(
        projective_measurement.measurements.spatial_support >= STATION_RELATION_MIN_SPATIAL_SUPPORT
    );
    assert!(
        station_relation_rejection_reasons(&projective_measurement.measurements)
            .contains(&degradation::STATION_RELATION_SCALE_OUT_OF_RANGE)
    );

    let selection =
        select_virtual_tile_relation_hypothesis(&tiles, 0, 1, &hypotheses, solver_coverages)
            .expect("the compliant hypothesis is measurable");
    assert!(selection.passes);
    assert_eq!(selection.relation.homography, passing);
    let selected_outcome = selection
        .outcomes
        .iter()
        .find(|outcome| outcome.selected)
        .expect("one hypothesis is selected");
    assert!(selected_outcome.holdout_fit_count >= STATION_RELATION_MIN_INLIERS);
    assert!(selected_outcome.holdout_count >= STATION_RELATION_MIN_INLIERS);
    assert!(
        selected_outcome
            .holdout_median_error_px
            .is_some_and(|error| error <= STATION_RELATION_MAX_MEDIAN_ERROR_PX)
    );
    assert!(
        selected_outcome
            .holdout_spatial_support
            .is_some_and(|support| support >= STATION_RELATION_MIN_SPATIAL_SUPPORT)
    );

    let relations = VirtualTileRelationHypotheses::from([((0, 1), hypotheses)]);
    assert_eq!(
        virtual_tile_authoritative_pairs(&relations, &tiles, solver_coverages),
        [(0, 1)]
    );
    let initial = HashMap::from([(0, Matrix3::identity()), (1, Matrix3::identity())]);
    let mut closure = report::ClosureReport::default();
    let mut relation_report = report::StationRelationsReport::default();
    let solved = solve_virtual_tile_station_hypotheses_with_report(
        &tiles,
        solver_coverages,
        &relations,
        &initial,
        &topology,
        &mut closure,
        &mut relation_report,
    );
    assert_eq!(solved.len(), 2);
    assert_eq!(relation_report.accepted.len(), 1);
    assert_eq!(relation_report.accepted[0].scale_ratio, 1.0);
}

#[test]
fn multi_hypothesis_holdout_key_prefers_measured_independent_support() {
    let relation = measured_relation(Matrix3::identity(), Vec::new());
    let measured = MeasuredStationRelationCandidate {
        measurements: StationRelationMeasurements {
            inliers: STATION_RELATION_MIN_INLIERS,
            median_error_px: 1.0,
            scale_ratio: 1.0,
            spatial_support: STATION_RELATION_MIN_SPATIAL_SUPPORT,
            low_frequency_mean_relative_difference: 0.0,
            edge_strength_ratio: 1.0,
            median_edge_orientation_difference_degrees: 0.0,
            preserves_convex_orientation: true,
        },
        quality: 1.0,
    };
    let candidate =
        |holdout_fit_count, holdout_count, error, support| MeasuredVirtualTileHypothesis {
            relation: relation.clone(),
            canonical_source: 0,
            canonical_target: 1,
            canonical_homography: Matrix3::identity(),
            correspondence_count: 0,
            correspondence_sample: Vec::new(),
            holdout_fit_count,
            holdout_count,
            holdout_median_error_px: error,
            holdout_spatial_support: support,
            holdout_transform: None,
            measured,
            rejection_count: 0,
            model_rank: 0,
        };
    let better = candidate(
        STATION_RELATION_MIN_INLIERS,
        STATION_RELATION_MIN_INLIERS,
        Some(1.0),
        Some(0.8),
    );
    let worse = candidate(
        STATION_RELATION_MIN_INLIERS,
        STATION_RELATION_MIN_INLIERS,
        Some(2.0),
        Some(0.9),
    );
    let unusable = candidate(2, 2, Some(0.0), Some(1.0));
    let low_support = candidate(
        STATION_RELATION_MIN_INLIERS,
        STATION_RELATION_MIN_INLIERS,
        Some(1.0),
        Some(STATION_RELATION_MIN_SPATIAL_SUPPORT - 0.001),
    );
    let high_error = candidate(
        STATION_RELATION_MIN_INLIERS,
        STATION_RELATION_MIN_INLIERS,
        Some(STATION_RELATION_MAX_MEDIAN_ERROR_PX + 0.001),
        Some(0.8),
    );
    let better_key = virtual_tile_hypothesis_holdout_key(&better);
    let worse_key = virtual_tile_hypothesis_holdout_key(&worse);
    let unusable_key = virtual_tile_hypothesis_holdout_key(&unusable);
    assert!(better_key.0 && worse_key.0);
    assert!(better_key.1 < worse_key.1);
    assert!(!unusable_key.0);
    assert!(better_key.0 > unusable_key.0);
    assert!(!virtual_tile_hypothesis_holdout_key(&low_support).0);
    assert!(!virtual_tile_hypothesis_holdout_key(&high_error).0);
}

#[test]
fn multi_hypothesis_low_hull_is_rejected_once_at_pair_level() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let (tiles, coverages, topology) = hypothesis_fixture();
    let clustered = (0..24)
        .map(|index| {
            let point = Point2::new(20.0 + f64::from(index % 6), 20.0 + f64::from(index / 6));
            (point, point)
        })
        .collect::<Vec<_>>();
    let relation = measured_relation(Matrix3::identity(), clustered);
    let hypotheses = vec![relation.clone(), relation];
    assert!(virtual_tile_hypotheses_are_equivalent(
        &hypotheses[0].homography,
        &hypotheses[1].homography,
        tiles[0].dimensions(),
    ));
    let solver_coverages = VirtualTileSolverCoverages {
        quality: &coverages,
        overlap: &coverages,
        overlap_factor: 1,
    };
    let selection =
        select_virtual_tile_relation_hypothesis(&tiles, 0, 1, &hypotheses, solver_coverages)
            .unwrap();
    assert!(!selection.passes);
    let relations = VirtualTileRelationHypotheses::from([((0, 1), hypotheses)]);
    assert!(virtual_tile_authoritative_pairs(&relations, &tiles, solver_coverages).is_empty());
    let initial = HashMap::from([(0, Matrix3::identity()), (1, Matrix3::identity())]);
    let mut closure = report::ClosureReport::default();
    let mut relation_report = report::StationRelationsReport::default();
    let solved = solve_virtual_tile_station_hypotheses_with_report(
        &tiles,
        solver_coverages,
        &relations,
        &initial,
        &topology,
        &mut closure,
        &mut relation_report,
    );
    assert!(solved.is_empty());
    assert_eq!(relation_report.rejected_count, 1);
    assert_eq!(
        relation_report.rejected_by_reason[degradation::STATION_RELATION_LOW_SPATIAL_SUPPORT],
        1
    );
}

#[test]
fn multi_hypothesis_canonical_inversion_recovers_from_singular_and_is_order_stable() {
    let (mut tiles, coverages, _) = hypothesis_fixture();
    tiles[0].filename = "virtual://z".to_string();
    tiles[1].filename = "virtual://a".to_string();
    let valid = Matrix3::identity();
    let valid_relation = measured_relation(valid, hypothesis_points(&valid, tiles[0].dimensions()));
    let singular = measured_relation(Matrix3::zeros(), valid_relation.points.clone());
    let solver_coverages = VirtualTileSolverCoverages {
        quality: &coverages,
        overlap: &coverages,
        overlap_factor: 1,
    };
    let first = select_virtual_tile_relation_hypothesis(
        &tiles,
        0,
        1,
        &[
            singular.clone(),
            valid_relation.clone(),
            valid_relation.clone(),
        ],
        solver_coverages,
    )
    .unwrap();
    let second = select_virtual_tile_relation_hypothesis(
        &tiles,
        0,
        1,
        &[valid_relation.clone(), singular, valid_relation],
        solver_coverages,
    )
    .unwrap();
    assert!(first.passes && second.passes);
    assert_eq!(first.relation.homography, second.relation.homography);
    let canonical = canonical_relation_view(&tiles, 0, 1, &first.relation).unwrap();
    assert!(canonical.inverted);
    assert_eq!(canonical.homography, Matrix3::identity());
}

#[test]
fn station_relation_scale_endpoints_are_closed_without_relaxing_adjacent_values() {
    let passing = |scale_ratio| StationRelationMeasurements {
        inliers: STATION_RELATION_MIN_INLIERS,
        median_error_px: STATION_RELATION_MAX_MEDIAN_ERROR_PX,
        scale_ratio,
        spatial_support: STATION_RELATION_MIN_SPATIAL_SUPPORT,
        low_frequency_mean_relative_difference:
            STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE,
        edge_strength_ratio: STATION_RELATION_EDGE_STRENGTH_RATIO_MIN,
        median_edge_orientation_difference_degrees:
            STATION_RELATION_MAX_MEDIAN_EDGE_ORIENTATION_DIFFERENCE_DEGREES,
        preserves_convex_orientation: true,
    };
    for endpoint in [
        STATION_RELATION_SCALE_RATIO_MIN,
        STATION_RELATION_SCALE_RATIO_MAX,
    ] {
        assert!(station_relation_rejection_reasons(&passing(endpoint)).is_empty());
    }
    for outside in [
        f64::from_bits(STATION_RELATION_SCALE_RATIO_MIN.to_bits() - 1),
        f64::from_bits(STATION_RELATION_SCALE_RATIO_MAX.to_bits() + 1),
    ] {
        assert_eq!(
            station_relation_rejection_reasons(&passing(outside)),
            [degradation::STATION_RELATION_SCALE_OUT_OF_RANGE]
        );
    }
}

#[test]
fn station_relation_coordinate_diagnostics_preserve_unequal_tiles_crop_and_reverse_orientation() {
    let mut tiles = vec![
        plane_info(10, textured_plane(420, 300)),
        plane_info(11, textured_plane(360, 280)),
    ];
    // Force the content-derived canonical direction to reverse the pair key.
    tiles[0].filename = "virtual://z".to_string();
    tiles[1].filename = "virtual://a".to_string();
    let coverages = vec![
        coverage_from_fn(105, 75, |x, y| x >= 3 && y >= 2),
        coverage_from_fn(90, 70, |x, y| x < 87 && y < 68),
    ];
    let solver_coverages = VirtualTileSolverCoverages {
        quality: &coverages,
        overlap: &coverages,
        overlap_factor: 4,
    };
    let crop_left = translation(123.0, 47.0);
    let crop_right = translation(-31.0, 82.0);
    let physical = about_center((210.0, 150.0), rotation(0.03) * uniform_scale(1.05));
    let key_direction = crop_right.try_inverse().unwrap() * physical * crop_left;
    let relation = measured_relation(
        key_direction,
        hypothesis_points(&key_direction, tiles[0].dimensions()),
    );
    let canonical = canonical_relation_view(&tiles, 0, 1, &relation).unwrap();
    assert!(canonical.inverted);
    assert_eq!((canonical.source_index, canonical.target_index), (1, 0));
    assert!((homography_scale_ratio(&canonical.homography).unwrap() - 1.0 / 1.05).abs() < 1e-12);

    let mut diagnostic = sample_candidate_record();
    let tile_to_world = HashMap::from([(10, crop_left), (11, crop_right)]);
    populate_station_relation_candidate_geometry(
        &mut diagnostic,
        &tiles,
        solver_coverages,
        &tile_to_world,
    );
    assert_eq!(diagnostic.left_tile.dimensions, [420, 300]);
    assert_eq!(diagnostic.right_tile.dimensions, [360, 280]);
    assert_eq!(
        diagnostic.left_tile.effective_bounds,
        Some([12, 8, 408, 292])
    );
    assert_eq!(
        diagnostic.right_tile.effective_bounds,
        Some([0, 0, 348, 272])
    );
    assert_eq!(
        diagnostic.left_tile.tile_to_world,
        Some(matrix3_row_major(&crop_left))
    );
    assert_eq!(
        diagnostic.right_tile.tile_to_world,
        Some(matrix3_row_major(&crop_right))
    );
}
