//! Multi-hypothesis Station_Relation selection: a compliant model when the projective one fails
//! scale, holdout keys, low-hull rejection, canonical inversion, closed scale endpoints and
//! coordinate diagnostics.

use super::*;
use std::borrow::Cow;

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
            scale_evidence: None,
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
        scale_evidence: None,
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

// Ported from codex/layered-focus-modularize-2026-10-26 (7e3a0488, fc22136f):
// selector and solver share one canonical measurement, independent of import order.
fn projective_hypothesis_fixture(
    transform: Matrix3<f64>,
    target_gain: f64,
) -> (
    Vec<ImageInfo>,
    Vec<stitching::CoverageMask>,
    MatchInfo,
    topology::StationTopology,
    HashMap<usize, Matrix3<f64>>,
) {
    let dimensions = (800u32, 600u32);
    let texture = |x: f64, y: f64| {
        (128.0
            + 42.0 * (x * 0.051 + y * 0.037).sin()
            + 31.0 * (x * 0.023 - y * 0.079).cos()
            + 18.0 * (x * 0.131 + y * 0.097).sin())
        .clamp(1.0, 254.0)
    };
    let inverse = transform.try_inverse().unwrap();
    let left = GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
        image::Luma([texture(f64::from(x), f64::from(y)).round() as u8])
    });
    let right = GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
        let value = transformed_point(&inverse, Point2::new(f64::from(x), f64::from(y)))
            .filter(|point| {
                point.x >= 0.0
                    && point.y >= 0.0
                    && point.x < f64::from(dimensions.0)
                    && point.y < f64::from(dimensions.1)
            })
            .map_or(0.0, |point| texture(point.x, point.y) * target_gain);
        image::Luma([value.clamp(0.0, 255.0).round() as u8])
    });
    let mut tiles = vec![plane_info(0, left), plane_info(1, right)];
    for tile in &mut tiles {
        tile.width = dimensions.0;
        tile.height = dimensions.1;
    }
    let target_coverage = coverage_from_fn(dimensions.0, dimensions.1, |x, y| {
        transformed_point(&inverse, Point2::new(f64::from(x), f64::from(y))).is_some_and(|point| {
            point.x >= 0.0
                && point.y >= 0.0
                && point.x < f64::from(dimensions.0)
                && point.y < f64::from(dimensions.1)
        })
    });
    let coverages = vec![full_coverage(dimensions.0, dimensions.1), target_coverage];
    let points = (0..8)
        .flat_map(|row| {
            (0..8).map(move |column| {
                Point2::new(
                    200.0 + f64::from(column) * 400.0 / 7.0,
                    80.0 + f64::from(row) * 440.0 / 7.0,
                )
            })
        })
        .map(|source| (source, transformed_point(&transform, source).unwrap()))
        .collect();
    let relation = measured_relation(transform, points);
    let initial = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, transform.try_inverse().unwrap()),
    ]);
    (tiles, coverages, relation, candidate_topology(2), initial)
}

fn selector_and_solver_measurements(
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    relation: MatchInfo,
    topology: &topology::StationTopology,
    initial: &HashMap<usize, Matrix3<f64>>,
) -> (
    VirtualTileHypothesisSelection,
    HashMap<usize, Matrix3<f64>>,
    report::StationRelationsReport,
) {
    let solver_coverages = VirtualTileSolverCoverages {
        quality: coverages,
        overlap: coverages,
        overlap_factor: 1,
    };
    let selection = select_virtual_tile_relation_hypothesis(
        tiles,
        0,
        1,
        std::slice::from_ref(&relation),
        solver_coverages,
    )
    .unwrap();
    let hypotheses = VirtualTileRelationHypotheses::from([((0, 1), vec![relation])]);
    let mut closure = report::ClosureReport::default();
    let mut relation_report = report::StationRelationsReport::default();
    let solved = solve_virtual_tile_station_hypotheses_with_report(
        tiles,
        solver_coverages,
        &hypotheses,
        initial,
        topology,
        &mut closure,
        &mut relation_report,
    );
    (selection, solved, relation_report)
}

#[test]
fn selector_and_solver_share_overlap_scale_when_origin_scale_disagrees() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    for (transform, expected_pass) in [
        (
            Matrix3::new(0.9, 0.0, 0.0, 0.0, 0.9, 0.0, -0.00017, 0.0, 1.0),
            true,
        ),
        (
            Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -0.00016, 0.0, 1.0),
            false,
        ),
    ] {
        let (tiles, coverages, relation, topology, initial) =
            projective_hypothesis_fixture(transform, 1.0);
        let origin_scale = homography_scale_ratio(&transform).unwrap();
        let overlap_scale = station_relation_overlap_scale(&transform, &relation.points).unwrap();
        assert_ne!(
            (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
                .contains(&origin_scale),
            (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
                .contains(&overlap_scale),
        );
        let (selection, solved, report) =
            selector_and_solver_measurements(&tiles, &coverages, relation, &topology, &initial);
        let selected = selection
            .outcomes
            .iter()
            .find(|outcome| outcome.selected)
            .unwrap();
        assert_eq!(selection.passes, expected_pass);
        assert_eq!(report.accepted.len() == 1, expected_pass);
        assert_eq!(solved.len() == 2, expected_pass);
        if expected_pass {
            assert_eq!(report.accepted[0].scale_ratio, selected.scale_ratio);
        } else {
            assert!(
                selected
                    .rejection_reasons
                    .iter()
                    .any(|reason| reason == degradation::STATION_RELATION_SCALE_OUT_OF_RANGE)
            );
            assert_eq!(
                report.rejected_by_reason[degradation::STATION_RELATION_SCALE_OUT_OF_RANGE],
                1
            );
        }
    }
}

#[test]
fn selector_and_solver_share_exposure_compensated_photometric_verdict() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let transform = Matrix3::identity();
    let (tiles, coverages, relation, topology, initial) =
        projective_hypothesis_fixture(transform, 0.7);
    let raw = focus_overlap_quality_measurements_on_coverage(
        &tiles[0],
        &tiles[1],
        &transform,
        Some((&coverages[0], &coverages[1])),
    )
    .unwrap();
    assert!(raw.low_frequency_mean_relative_difference > 0.2);
    assert!(virtual_tile_photometric_measurements(raw).0 <= 0.2);
    let (selection, solved, report) =
        selector_and_solver_measurements(&tiles, &coverages, relation, &topology, &initial);
    let selected = selection
        .outcomes
        .iter()
        .find(|outcome| outcome.selected)
        .unwrap();
    assert!(selection.passes, "{:?}", selected.rejection_reasons);
    assert!(selected.low_frequency_mean_relative_difference <= 0.2);
    assert_eq!(report.accepted.len(), 1);
    assert_eq!(
        report.accepted[0].scale_ratio, selected.scale_ratio,
        "selector and solver must retain the same canonical measurement"
    );
    assert_eq!(solved.len(), 2);
}

fn canonical_edge_ratio_fixture(
    source_span: u8,
    target_span: u8,
    swapped: bool,
) -> (
    Vec<ImageInfo>,
    Vec<stitching::CoverageMask>,
    MatchInfo,
    topology::StationTopology,
    HashMap<usize, Matrix3<f64>>,
) {
    let dimensions = (400, 300);
    let contrast_plane = |span: u8| {
        let low = ((255 - u16::from(span)) / 2) as u8;
        let high = low + span;
        GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
            let high_frequency = ((x / 7) + (y / 11) + ((x / 29) * (y / 31))) % 2 == 0;
            image::Luma([if high_frequency { high } else { low }])
        })
    };
    let mut source = plane_info(10, contrast_plane(source_span));
    source.filename = "virtual://canonical-a".to_string();
    let mut target = plane_info(20, contrast_plane(target_span));
    target.filename = "virtual://canonical-z".to_string();
    let tiles = if swapped {
        vec![target, source]
    } else {
        vec![source, target]
    };
    let transform = Matrix3::identity();
    let relation = measured_relation(
        transform,
        hypothesis_points(&transform, tiles[0].dimensions()),
    );
    let coverages = vec![
        full_coverage(dimensions.0, dimensions.1),
        full_coverage(dimensions.0, dimensions.1),
    ];
    let initial = tiles
        .iter()
        .map(|tile| (tile.id, Matrix3::identity()))
        .collect();
    (tiles, coverages, relation, candidate_topology(2), initial)
}

#[test]
fn canonical_edge_strength_ratio_is_import_order_invariant_at_asymmetric_gates() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    for (source_span, target_span, expected_ratio, expected_pass) in [
        (71, 100, 0.71, true),
        (100, 71, 1.0 / 0.71, false),
        (139, 100, 1.39, true),
        (100, 139, 1.0 / 1.39, true),
    ] {
        let runs = [false, true].map(|swapped| {
            let (tiles, coverages, relation, topology, initial) =
                canonical_edge_ratio_fixture(source_span, target_span, swapped);
            let solver_coverages = VirtualTileSolverCoverages {
                quality: &coverages,
                overlap: &coverages,
                overlap_factor: 1,
            };
            let canonical = canonical_relation_view(&tiles, 0, 1, &relation).unwrap();
            assert_eq!(canonical.inverted, swapped);
            let points: Cow<'_, [(Point2<f64>, Point2<f64>)]> = if canonical.inverted {
                Cow::Owned(
                    relation
                        .points
                        .iter()
                        .map(|&(source, target)| (target, source))
                        .collect(),
                )
            } else {
                Cow::Borrowed(&relation.points)
            };
            let dense = focus_overlap_quality_measurements_on_coverage(
                &tiles[canonical.source_index],
                &tiles[canonical.target_index],
                &canonical.homography,
                Some((
                    &coverages[canonical.source_index],
                    &coverages[canonical.target_index],
                )),
            )
            .unwrap();
            let canonical_photometric = virtual_tile_photometric_measurements(dense);
            assert!(
                (canonical_photometric.1 - expected_ratio).abs() < 1e-12,
                "expected {expected_ratio}, measured {canonical_photometric:?}"
            );
            let direct = measure_station_relation_candidate(
                relation.inliers,
                &canonical.homography,
                &points,
                tiles[canonical.source_index].dimensions(),
                tiles[canonical.target_index].dimensions(),
                solver_coverages.overlap_measure(canonical.source_index, canonical.target_index),
                Some(dense),
            )
            .unwrap();
            assert!((direct.measurements.edge_strength_ratio - expected_ratio).abs() < 1e-12);
            assert_eq!(
                station_relation_rejection_reasons(&direct.measurements).is_empty(),
                expected_pass
            );

            let hypotheses =
                VirtualTileRelationHypotheses::from([((0, 1), vec![relation.clone()])]);
            let authoritative =
                virtual_tile_relation_measurements(0, 1, &relation, &tiles, solver_coverages)
                    .is_some_and(|measurements| {
                        station_relation_rejection_reasons(&measurements).is_empty()
                    });
            let provisional = virtual_tile_provisional_authoritative_candidate(
                &tiles,
                0,
                1,
                std::slice::from_ref(&relation),
                solver_coverages,
            );
            assert_eq!(authoritative, expected_pass);
            assert_eq!(provisional, expected_pass);
            let authoritative_pairs =
                virtual_tile_authoritative_pairs(&hypotheses, &tiles, solver_coverages);
            let prior_free_pairs =
                virtual_tile_prior_free_candidate_pairs(2, &authoritative_pairs, &topology);
            assert_eq!(authoritative_pairs.is_empty(), !expected_pass);
            assert_eq!(prior_free_pairs.is_empty(), expected_pass);

            let (selection, solved, report) =
                selector_and_solver_measurements(&tiles, &coverages, relation, &topology, &initial);
            let selected = selection
                .outcomes
                .iter()
                .find(|outcome| outcome.selected)
                .unwrap()
                .clone();
            assert_eq!(selection.passes, expected_pass);
            assert!((selected.edge_strength_ratio - expected_ratio).abs() < 1e-12);
            assert_eq!(solved.len() == 2, expected_pass);
            if expected_pass {
                assert_eq!(report.rejected_count, 0);
                assert_eq!(report.accepted.len(), 1);
            } else {
                assert_eq!(selected.rejection_reasons.len(), 1);
                assert_eq!(
                    selected.rejection_reasons[0],
                    degradation::STATION_RELATION_EDGE_STRENGTH_MISMATCH
                );
                assert_eq!(report.accepted.len(), 0);
                assert_eq!(report.rejected_count, 1);
                assert_eq!(report.rejected_by_reason.len(), 1);
                assert_eq!(
                    report.rejected_by_reason[degradation::STATION_RELATION_EDGE_STRENGTH_MISMATCH],
                    1
                );
            }
            (selection.relation.homography, selected, report)
        });
        assert_eq!(runs[0].0, runs[1].0, "selected hypothesis changed");
        assert_eq!(runs[0].1.model, runs[1].1.model);
        assert_eq!(
            runs[0].1.source_to_target_transform,
            runs[1].1.source_to_target_transform
        );
        assert_eq!(runs[0].1.rejection_reasons, runs[1].1.rejection_reasons);
        assert_eq!(runs[0].2.rejected_count, runs[1].2.rejected_count);
        assert_eq!(runs[0].2.rejected_by_reason, runs[1].2.rejected_by_reason);
    }
}
