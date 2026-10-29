//! Station_Relation acceptance: the literal gate boundaries and one identifier per failed
//! defect, native tone evidence, inlier-hull spatial support, overlap scale and the residual
//! model built from accepted relations.

use super::*;

#[test]
fn native_tone_evidence_does_not_relax_geometry_acceptance() {
    let actual = StationRelationMeasurements {
        inliers: 972,
        median_error_px: 2.266923987432463,
        scale_ratio: 1.0004691316830572,
        spatial_support: 0.5755645891054085,
        low_frequency_mean_relative_difference: 0.26202033364182203,
        edge_strength_ratio: 0.9321190720076769,
        median_edge_orientation_difference_degrees: 5.7927737338494625,
        preserves_convex_orientation: true,
    };
    assert_eq!(
        station_relation_rejection_reasons(&actual),
        vec![degradation::STATION_RELATION_PHOTOMETRIC_MISMATCH]
    );
    assert!(native_photometric_measurements_are_usable(&actual));
    assert!(!native_photometric_measurements_are_usable(
        &StationRelationMeasurements {
            median_error_px: 3.01,
            ..actual
        }
    ));
    assert!(!native_photometric_measurements_are_usable(
        &StationRelationMeasurements {
            low_frequency_mean_relative_difference: f64::NAN,
            ..actual
        }
    ));
}

#[test]
fn station_relation_contract_accepts_every_literal_boundary() {
    let measurements = StationRelationMeasurements {
        inliers: STATION_RELATION_MIN_INLIERS,
        median_error_px: STATION_RELATION_MAX_MEDIAN_ERROR_PX,
        scale_ratio: STATION_RELATION_SCALE_RATIO_MIN,
        spatial_support: STATION_RELATION_MIN_SPATIAL_SUPPORT,
        low_frequency_mean_relative_difference:
            STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE,
        edge_strength_ratio: STATION_RELATION_EDGE_STRENGTH_RATIO_MIN,
        median_edge_orientation_difference_degrees:
            STATION_RELATION_MAX_MEDIAN_EDGE_ORIENTATION_DIFFERENCE_DEGREES,
        preserves_convex_orientation: true,
    };
    assert!(station_relation_rejection_reasons(&measurements).is_empty());

    let upper_scale_and_edge = StationRelationMeasurements {
        scale_ratio: STATION_RELATION_SCALE_RATIO_MAX,
        edge_strength_ratio: STATION_RELATION_EDGE_STRENGTH_RATIO_MAX,
        ..measurements
    };
    assert!(station_relation_rejection_reasons(&upper_scale_and_edge).is_empty());
}

#[test]
fn station_relation_contract_records_one_identifier_per_failed_defect() {
    let accepted = StationRelationMeasurements {
        inliers: 24,
        median_error_px: 3.0,
        scale_ratio: 1.0,
        spatial_support: 0.20,
        low_frequency_mean_relative_difference: 0.20,
        edge_strength_ratio: 1.0,
        median_edge_orientation_difference_degrees: 10.0,
        preserves_convex_orientation: true,
    };
    let cases = [
        (
            StationRelationMeasurements {
                inliers: 23,
                ..accepted
            },
            degradation::STATION_RELATION_LOW_INLIERS,
        ),
        (
            StationRelationMeasurements {
                median_error_px: 3.000_001,
                ..accepted
            },
            degradation::STATION_RELATION_HIGH_REPROJECTION_ERROR,
        ),
        (
            StationRelationMeasurements {
                scale_ratio: STATION_RELATION_SCALE_RATIO_MIN - 0.000_001,
                ..accepted
            },
            degradation::STATION_RELATION_SCALE_OUT_OF_RANGE,
        ),
        (
            StationRelationMeasurements {
                spatial_support: 0.199_999,
                ..accepted
            },
            degradation::STATION_RELATION_LOW_SPATIAL_SUPPORT,
        ),
        (
            StationRelationMeasurements {
                low_frequency_mean_relative_difference: 0.200_001,
                ..accepted
            },
            degradation::STATION_RELATION_PHOTOMETRIC_MISMATCH,
        ),
        (
            StationRelationMeasurements {
                edge_strength_ratio: 1.400_001,
                ..accepted
            },
            degradation::STATION_RELATION_EDGE_STRENGTH_MISMATCH,
        ),
        (
            StationRelationMeasurements {
                median_edge_orientation_difference_degrees: 10.000_001,
                ..accepted
            },
            degradation::STATION_RELATION_EDGE_ORIENTATION_MISMATCH,
        ),
        (
            StationRelationMeasurements {
                preserves_convex_orientation: false,
                ..accepted
            },
            degradation::STATION_RELATION_NON_CONVEX_QUAD,
        ),
    ];
    for (measurements, expected) in cases {
        assert_eq!(
            station_relation_rejection_reasons(&measurements),
            [expected]
        );
    }

    let every_defect = StationRelationMeasurements {
        inliers: 0,
        median_error_px: f64::INFINITY,
        scale_ratio: 0.0,
        spatial_support: 0.0,
        low_frequency_mean_relative_difference: f64::INFINITY,
        edge_strength_ratio: f64::INFINITY,
        median_edge_orientation_difference_degrees: f64::INFINITY,
        preserves_convex_orientation: false,
    };
    let reasons = station_relation_rejection_reasons(&every_defect);
    assert_eq!(reasons.len(), 8);
    assert_eq!(reasons.iter().copied().collect::<HashSet<_>>().len(), 8);
}

#[test]
fn residual_model_builder_uses_accepted_bidirectional_relation_evidence() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let tiles = vec![focus_test_image(0, "a.NEF"), focus_test_image(1, "b.NEF")];
    let homography = Matrix3::new(
        1.0, 0.0, 4.0, // an intentionally large local residual in world pixels
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    );
    let relation = observed_transform_match(homography, 36);
    let relations = HashMap::from([((0usize, 1usize), relation)]);
    let tile_homographies = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, Matrix3::identity()),
    ]);
    let accepted = [stack_report::AcceptedStationRelationRecord {
        left: 0,
        right: 1,
        evidence_kind: stack_report::StationRelationEvidenceKind::VirtualTile,
        inliers: 36,
        spatial_support: 0.5,
        scale_ratio: 1.0,
        median_error_px: 0.0,
        independent_support: 1,
    }];

    let model =
        residual_model_from_station_relations(&tiles, &tile_homographies, &relations, &accepted);
    assert_eq!(model.regions.len(), 1);
    assert!(!model.identity());
    assert_eq!(model.regions[0].right_station, 1);
    assert!((model.regions[0].p95_before_px - 4.0).abs() < 1.0e-12);
}

#[test]
fn residual_model_builder_keeps_low_p95_relation_identity() {
    let tiles = vec![focus_test_image(0, "a.NEF"), focus_test_image(1, "b.NEF")];
    let relation = observed_transform_match(
        Matrix3::new(
            1.0, 0.0, 2.9, // below the strict 3px enable gate
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ),
        36,
    );
    let relations = HashMap::from([((0usize, 1usize), relation)]);
    let tile_homographies = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, Matrix3::identity()),
    ]);
    let accepted = [stack_report::AcceptedStationRelationRecord {
        left: 0,
        right: 1,
        evidence_kind: stack_report::StationRelationEvidenceKind::VirtualTile,
        inliers: 36,
        spatial_support: 0.5,
        scale_ratio: 1.0,
        median_error_px: 0.0,
        independent_support: 1,
    }];

    let model =
        residual_model_from_station_relations(&tiles, &tile_homographies, &relations, &accepted);
    assert!(model.regions.is_empty());
    assert!(model.identity());
}

#[test]
fn station_relation_spatial_support_is_inlier_hull_over_overlap() {
    let points = vec![
        (Point2::new(0.0, 0.0), Point2::new(0.0, 0.0)),
        (Point2::new(50.0, 0.0), Point2::new(50.0, 0.0)),
        (Point2::new(50.0, 50.0), Point2::new(50.0, 50.0)),
        (Point2::new(0.0, 50.0), Point2::new(0.0, 50.0)),
    ];
    let support =
        station_relation_spatial_support(&points, &Matrix3::identity(), (100, 100), (100, 100));
    assert!((support - 0.25).abs() < 1e-12);
}

#[test]
fn station_overlap_scale_is_independent_of_both_tile_origins() {
    let pose = Matrix3::new(0.74, 0.01, 220.0, 0.0, 0.74, 80.0, -0.0002, 0.0, 1.0);
    let points = [
        (500.0, 300.0),
        (740.0, 300.0),
        (500.0, 540.0),
        (740.0, 540.0),
    ]
    .into_iter()
    .map(|(x, y)| {
        let source = Point2::new(x, y);
        (source, transformed_point(&pose, source).unwrap())
    })
    .collect::<Vec<_>>();
    let scale = station_relation_overlap_scale(&pose, &points).unwrap();
    assert!(scale > 0.9 && scale < 1.1, "{scale}");
    assert!(homography_scale_ratio(&pose).unwrap() < STATION_RELATION_SCALE_RATIO_MIN);
    let source_shift = Matrix3::new(1.0, 0.0, 1000.0, 0.0, 1.0, 300.0, 0.0, 0.0, 1.0);
    let target_shift = Matrix3::new(1.0, 0.0, -500.0, 0.0, 1.0, 400.0, 0.0, 0.0, 1.0);
    let shifted = target_shift * pose * source_shift.try_inverse().unwrap();
    let shifted_points = points
        .iter()
        .map(|&(source, target)| {
            (
                transformed_point(&source_shift, source).unwrap(),
                transformed_point(&target_shift, target).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert!(
        (station_relation_overlap_scale(&shifted, &shifted_points).unwrap() - scale).abs() < 1e-10
    );
    let affine = Matrix3::new(1.03, 0.0, 120.0, 0.0, 1.03, -40.0, 0.0, 0.0, 1.0);
    assert!((station_relation_overlap_scale(&affine, &points).unwrap() - 1.03).abs() < 1e-12);
    assert!(station_relation_overlap_scale(&pose, &[]).is_none());
}
