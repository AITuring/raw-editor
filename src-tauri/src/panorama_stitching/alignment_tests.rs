use super::*;
use nalgebra::Point2;

#[test]
fn robust_native_affine_resists_outliers_without_removing_gate_observations() {
    let expected = Matrix3::new(1.001, 0.002, 18.0, -0.001, 0.999, 20.0, 0.0, 0.0, 1.0);
    let mut points = Vec::new();
    for y in (128..896).step_by(64) {
        for x in (128..896).step_by(64) {
            let a = Point2::new(x as f64, y as f64);
            let mut b = transformed_point(&expected, a).unwrap();
            if points.len() % 5 == 0 {
                b.x += 30.0;
                b.y += 20.0;
            }
            points.push((a, b));
        }
    }
    let ordinary = estimate_affine(&points).unwrap();
    assert!(median_symmetric_error(&ordinary, &points) > 3.0);
    let robust = robust_native_affine(&points).unwrap();
    assert!(median_symmetric_error(&robust, &points) < 1.0);
    assert_eq!(points.len(), 144);
    let corner = Point2::new(1024.0, 1024.0);
    assert!(
        (transformed_point(&robust, corner).unwrap()
            - transformed_point(&expected, corner).unwrap())
        .norm()
            < 1.0
    );
}

#[test]
fn native_affine_model_only_replaces_perspective_when_native_gates_hold() {
    let coverage =
        stitching::CoverageMask::from_gray(GrayImage::from_pixel(1024, 1024, image::Luma([255])));
    let points_for = |h: &Matrix3<f64>| {
        (128..896)
            .step_by(64)
            .flat_map(|y| {
                (128..896).step_by(64).map(move |x| {
                    let source = Point2::new(x as f64, y as f64);
                    (source, transformed_point(h, source).unwrap())
                })
            })
            .collect::<Vec<_>>()
    };
    let affine = Matrix3::new(1.001, 0.002, 18.0, -0.001, 0.999, 20.0, 0.0, 0.0, 1.0);
    let points = points_for(&affine);
    let measured =
        native_station_affine_candidate(&points, (1024, 1024), (1024, 1024), &coverage, &coverage)
            .unwrap();
    assert!(median_symmetric_error(&measured, &points) < 1e-6);
    let perspective = Matrix3::new(1.0, 0.0, 18.0, 0.0, 1.0, 20.0, 0.00015, -0.00004, 1.0);
    let native_points = points_for(&perspective);
    let regularized = native_station_regularized_candidate(
        &native_points,
        &perspective,
        (1024, 1024),
        (1024, 1024),
        &coverage,
        &coverage,
    )
    .unwrap();
    assert!(
        regularized[(2, 0)].abs() > 1e-6,
        "genuine perspective was removed"
    );
    assert!(median_symmetric_error(&regularized, &native_points) <= 3.0);
    assert!(
        native_station_affine_candidate(
            &points_for(&perspective),
            (1024, 1024),
            (1024, 1024),
            &coverage,
            &coverage
        )
        .is_none()
    );
}

fn test_image(id: usize, filename: &str) -> ImageInfo {
    ImageInfo {
        id,
        filename: filename.to_string(),
        width: 1,
        height: 1,
        alignment_image: GrayImage::new(1, 1),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: None,
        source_footprint: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

#[test]
fn flat_artwork_render_order_starts_at_center_and_walks_overlaps() {
    let mut images = (0..4)
        .map(|id| test_image(id, &format!("/synthetic/DSC_{id:04}.ARW")))
        .collect::<Vec<_>>();
    for image in &mut images {
        image.width = 100;
        image.height = 100;
    }
    let translations = [(0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (90.0, 90.0)];
    let homographies = translations
        .into_iter()
        .enumerate()
        .map(|(id, (x, y))| (id, Matrix3::new(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0)))
        .collect::<HashMap<_, _>>();
    let groups = HashMap::from([(0, 1u8), (1, 1u8), (2, 2u8), (3, 2u8)]);

    let order =
        flat_artwork_overlap_render_order(&images, &[0, 1, 2, 3], &homographies, Some(&groups));

    assert_eq!(order, vec![1, 0, 2, 3]);
    for (position, &index) in order.iter().enumerate().skip(1) {
        let current = flat_artwork_projected_rect(&images[index], &homographies).unwrap();
        let overlaps_previous = order[..position].iter().any(|previous| {
            let previous = flat_artwork_projected_rect(&images[*previous], &homographies).unwrap();
            current.overlap_area(previous) > 0.0
        });
        assert!(
            overlaps_previous,
            "source {index} should overlap prior coverage"
        );
    }
}

/// Alignment side of [`focus_textured_test_image`]; the full frame is
/// 1000×1000, so `scale_factor` is 4.0 exactly as a downscaled production
/// source would be.
const FOCUS_TEST_ALIGNMENT_SIDE: u32 = 250;

/// Deterministic band-limited texture for an alignment image the overlap
/// correlation of 需求 1.1 can actually be measured on.
///
/// Three incommensurate sinusoids decorrelate over roughly ten alignment
/// pixels: bracket-sized breathing stays near 1.0 while a neighbouring
/// camera position falls well below the 0.60 bar.
fn focus_test_alignment_image() -> GrayImage {
    GrayImage::from_fn(
        FOCUS_TEST_ALIGNMENT_SIDE,
        FOCUS_TEST_ALIGNMENT_SIDE,
        |x, y| {
            let x = f64::from(x);
            let y = f64::from(y);
            let value = 128.0
                + 60.0 * (x * 0.09 + y * 0.05).sin()
                + 45.0 * (x * 0.031 - y * 0.11).sin()
                + 25.0 * (x * 0.17 + y * 0.13).cos();
            image::Luma([value.clamp(0.0, 255.0) as u8])
        },
    )
}

fn focus_test_image(id: usize, filename: &str) -> ImageInfo {
    let mut image = test_image(id, filename);
    image.width = 1_000;
    image.height = 1_000;
    image
}

/// [`focus_test_image`] with a real alignment image, so the overlap
/// correlation of 需求 1.1 is measurable rather than absent.
fn focus_textured_test_image(id: usize, filename: &str) -> ImageInfo {
    let mut image = focus_test_image(id, filename);
    image.alignment_image = focus_test_alignment_image();
    image.scale_factor = f64::from(image.width) / f64::from(FOCUS_TEST_ALIGNMENT_SIDE);
    image.focal_length_35mm = Some(75.0);
    image
}

#[test]
fn shared_raw_display_gain_preserves_bright_scan_channels() {
    let mut highs = vec![0.3; 40];
    highs.extend([0.8, 0.95]);
    let gain = shared_stack_raw_display_gain(&highs).unwrap();
    assert_eq!(gain, 1.0);
    assert!(0.95 * gain < 1.0);
    let warm =
        DynamicImage::ImageRgb32F(Rgb32FImage::from_pixel(32, 32, image::Rgb([0.8, 0.4, 0.2])));
    assert_eq!(raw_display_channel_percentile(&warm), Some(0.8));
    let gain =
        shared_stack_raw_display_gain(&[raw_display_channel_percentile(&warm).unwrap()]).unwrap();
    assert!((gain - 1.125).abs() < 1e-6);
    assert!(shared_stack_raw_display_gain(&[f32::NAN, -1.0]).is_none());
}

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
fn focus_adjacent_same_focal_pair_allows_missing_exif() {
    let mut left = focus_test_image(0, "DSC_3743.png");
    left.focal_length_35mm = Some(50.0);
    let mut right = focus_test_image(1, "DSC_3744.png");
    right.focal_length_35mm = None;
    assert!(focus_adjacent_same_focal_pair(&left, &right));
    let mut mismatched = focus_test_image(2, "DSC_3745.png");
    mismatched.focal_length_35mm = Some(85.0);
    assert!(!focus_compatible_focal_length(&left, &mismatched));
}

#[test]
fn narrow_same_focal_scan_boundary_rejects_low_texture_overlap() {
    assert!(!focus_position_recovery_accepts(
        1, true, 1.0, 0.045, 0.12, 120, 0.35, -0.05, -0.05, false,
    ));
    assert!(!focus_position_recovery_accepts(
        2, true, 1.0, 0.045, 0.12, 120, 0.35, -0.05, -0.05, false,
    ));
    assert!(!focus_position_recovery_accepts(
        1, false, 1.0, 0.20, 0.12, 120, 0.35, -0.05, -0.05, false,
    ));
}

#[test]
fn focus_geometry_guard_rejects_incomplete_or_drifting_scan() {
    let images = (0..4)
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();
    let stable = (0..images.len())
        .map(|index| (images[index].id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();
    assert!(focus_capture_geometry_passes(&images, 3, 3, 0.64, &stable));
    assert!(!focus_capture_geometry_passes(&images, 2, 3, 0.64, &stable));

    let drifting = HashMap::from([
        (0, Matrix3::identity()),
        (
            1,
            Matrix3::new(0.78, 0.0, 0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0),
        ),
        (2, Matrix3::identity()),
        (3, Matrix3::identity()),
    ]);
    assert!(!focus_capture_geometry_passes(
        &images, 3, 3, 0.64, &drifting
    ));
}

#[test]
fn focus_station_geometry_guard_does_not_hide_broken_boundaries() {
    assert!(focus_capture_group_geometry_passes(25, 25, 24, 0.50));
    assert!(!focus_capture_group_geometry_passes(24, 25, 24, 0.50));
    assert!(!focus_capture_group_geometry_passes(25, 25, 23, 0.50));
    assert!(!focus_capture_group_geometry_passes(25, 25, 24, 0.20));
}

fn identity_match(inliers: usize) -> MatchInfo {
    MatchInfo {
        homography: Matrix3::identity(),
        inliers,
        sequence_bridge: false,
        coarse_bridge: false,
        points: Vec::new(),
        candidate_points: Vec::new(),
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
        native_verified_points: Vec::new(),
    }
}

#[test]
fn shifted_focus_structure_requires_geometry_and_rejects_wrong_overlap() {
    let mut source = test_image(0, "source.nef");
    let mut target = test_image(1, "target.nef");
    let texture = |x: i32, y: i32, seed: u32| {
        let xf = x as f64;
        let yf = y as f64;
        let grain = ((x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663) ^ seed)
            .wrapping_mul(83492791);
        let value = 128.0
            + 45.0 * (xf * 0.029 + yf * 0.013).sin()
            + 35.0 * (xf * 0.017 - yf * 0.037).cos()
            + 25.0 * (xf * 0.053 + yf * 0.041).sin()
            + ((grain >> 24) as f64 - 128.0) * 0.30;
        image::Luma([value.clamp(0.0, 255.0) as u8])
    };
    for image in [&mut source, &mut target] {
        image.width = 4800;
        image.height = 3200;
        image.scale_factor = 4.0;
    }
    source.alignment_image = GrayImage::from_fn(1200, 800, |x, y| texture(x as i32, y as i32, 7));
    target.alignment_image =
        GrayImage::from_fn(1200, 800, |x, y| texture(x as i32 - 160, y as i32 - 40, 91));
    let transform = Matrix3::new(1.0, 0.0, 640.0, 0.0, 1.0, 160.0, 0.0, 0.0, 1.0);
    let points = (0..6)
        .flat_map(|row| {
            (0..8).map(move |col| {
                let p = Point2::new(400.0 + col as f64 * 450.0, 400.0 + row as f64 * 450.0);
                (p, Point2::new(p.x + 640.0, p.y + 160.0))
            })
        })
        .collect::<Vec<_>>();
    assert!(shifted_focus_structure_is_verified(
        &source, &target, &transform, &points, true
    ));
    source.focal_length_35mm = Some(24.0);
    target.focal_length_35mm = Some(50.0);
    assert!(!shifted_focus_structure_is_verified(
        &source, &target, &transform, &points, false
    ));
    target.focal_length_35mm = Some(24.0);
    assert!(!shifted_focus_structure_is_verified(
        &source,
        &target,
        &transform,
        &points[..29],
        false
    ));
    let wrong = Matrix3::new(1.0, 0.0, 960.0, 0.0, 1.0, 160.0, 0.0, 0.0, 1.0);
    let wrong_points = points
        .iter()
        .map(|&(p, _)| (p, Point2::new(p.x + 960.0, p.y + 160.0)))
        .collect::<Vec<_>>();
    assert!(!shifted_focus_structure_is_verified(
        &source,
        &target,
        &wrong,
        &wrong_points,
        true
    ));
    let concentrated = (0..48)
        .map(|i| {
            let p = Point2::new(1000.0 + (i % 8) as f64 * 5.0, 1000.0 + (i / 8) as f64 * 5.0);
            (p, Point2::new(p.x + 640.0, p.y + 160.0))
        })
        .collect::<Vec<_>>();
    assert!(!shifted_focus_structure_is_verified(
        &source,
        &target,
        &transform,
        &concentrated,
        false
    ));
    let fixed_points = points.iter().map(|&(p, _)| (p, p)).collect::<Vec<_>>();
    assert!(!shifted_focus_structure_is_verified(
        &source,
        &source,
        &Matrix3::identity(),
        &fixed_points,
        false
    ));
    let scan_match = MatchInfo {
        homography: transform,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: false,
        points: points.clone(),
        candidate_points: points.clone(),
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
        native_verified_points: Vec::new(),
    };
    let matches = HashMap::from([((0, 1), scan_match)]);
    let images = [source, target];
    let grouped = focus_station_grouping(&images, &matches);
    assert!(
        grouped.isolated.is_empty(),
        "a verified scan source is not an isolated source"
    );
    assert_eq!(
        grouped.components,
        [vec![0], vec![1]],
        "scan positions must remain separate stations"
    );
    let misleading_priors = HashMap::from([(0, Matrix3::identity()), (1, Matrix3::identity())]);
    let (groups, _) = focus_capture_groups(&images, &matches, &misleading_priors)
        .expect("connected scan stations");
    assert_eq!(
        groups.len(),
        2,
        "coincident world priors cannot merge distinct scan positions"
    );
}

#[test]
fn low_texture_overlap_requires_boundary_rescue_context() {
    let zhang_haohao_boundary = (0.886, 0.323, 0.134, 1_313);
    assert!(!focus_overlap_is_verified(
        zhang_haohao_boundary,
        13,
        1.887,
        false
    ));
    assert!(focus_overlap_is_verified(
        zhang_haohao_boundary,
        13,
        1.887,
        true
    ));
    assert!(!focus_overlap_is_verified(
        (0.886, 0.323, 0.099, 1_313),
        13,
        1.887,
        true
    ));
    assert!(!focus_overlap_is_verified(
        zhang_haohao_boundary,
        11,
        1.887,
        true
    ));
}

#[test]
fn capture_sequence_edges_outrank_remote_calligraphy_lookalikes() {
    let first = focus_test_image(0, "DSC_1000.NEF");
    let adjacent = focus_test_image(1, "DSC_1001.NEF");
    let remote = focus_test_image(2, "DSC_1200.NEF");
    let local_match = observed_translation_match(180.0, 0.0, 16);
    let remote_match = observed_translation_match(20.0, 0.0, 300);

    assert_eq!(
        focus_graph_capture_sequence_boost(&first, &adjacent, &local_match),
        FOCUS_CAPTURE_SEQUENCE_GRAPH_WEIGHT_BOOST
    );
    assert_eq!(
        focus_graph_capture_sequence_boost(&first, &remote, &remote_match),
        1.0
    );
}

#[test]
fn weak_interior_focus_frame_can_be_skipped_only_across_verified_overlap() {
    let images = vec![
        focus_test_image(0, "DSC_2740.NEF"),
        focus_test_image(1, "DSC_2741.NEF"),
        focus_test_image(2, "DSC_2742.NEF"),
    ];
    let stitched = HashSet::from([0, 2]);
    let verified_bypass = HashMap::from([((0, 2), observed_translation_match(20.0, 0.0, 24))]);

    assert!(focus_unstitched_sources_are_redundant(
        &images,
        &verified_bypass,
        &stitched
    ));
    assert!(!focus_unstitched_sources_are_redundant(
        &images,
        &HashMap::new(),
        &stitched
    ));
}

fn translation_match(dx: f64, dy: f64, inliers: usize) -> MatchInfo {
    MatchInfo {
        homography: Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0),
        inliers,
        sequence_bridge: false,
        coarse_bridge: false,
        points: Vec::new(),
        candidate_points: Vec::new(),
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
        native_verified_points: Vec::new(),
    }
}

/// A 6×6 correspondence grid over a 1000×1000 frame.
///
/// 需求 1.1 asks for at least 30 verified inlier correspondences before two
/// sources may share a Capture_Station, so an observed relation in these
/// tests carries 36 of them spread from 100 to 900 on both axes.
fn observed_transform_match(homography: Matrix3<f64>, inliers: usize) -> MatchInfo {
    let points = (0..6)
        .flat_map(|row| {
            (0..6).map(move |column| {
                let source = Point2::new(100.0 + column as f64 * 160.0, 100.0 + row as f64 * 160.0);
                (source, transformed_point(&homography, source).unwrap())
            })
        })
        .collect::<Vec<_>>();
    MatchInfo {
        homography,
        inliers,
        sequence_bridge: false,
        coarse_bridge: false,
        candidate_points: points.clone(),
        points,
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
        native_verified_points: Vec::new(),
    }
}

fn observed_translation_match(dx: f64, dy: f64, inliers: usize) -> MatchInfo {
    observed_transform_match(
        Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0),
        inliers,
    )
}

#[test]
fn canonical_match_direction_is_independent_of_import_order() {
    let reversed = vec![test_image(0, "z.jpg"), test_image(1, "a.jpg")];
    let (source, target, invert_for_storage) = canonical_match_direction(&reversed, 0, 1);

    assert_eq!(reversed[source].filename, "a.jpg");
    assert_eq!(reversed[target].filename, "z.jpg");
    assert!(invert_for_storage);

    let forward = vec![test_image(0, "a.jpg"), test_image(1, "z.jpg")];
    let (source, target, invert_for_storage) = canonical_match_direction(&forward, 0, 1);

    assert_eq!((source, target), (0, 1));
    assert_eq!(forward[source].filename, "a.jpg");
    assert_eq!(forward[target].filename, "z.jpg");
    assert!(!invert_for_storage);
}

#[test]
fn canonical_relation_view_distinguishes_bounds_and_singular_matrices() {
    let images = vec![test_image(0, "z.jpg"), test_image(1, "a.jpg")];
    let relation = translation_match(12.0, -4.0, 24);
    assert_eq!(
        canonical_relation_view(&images, 0, 2, &relation),
        Err(CanonicalRelationViewError::IndexOutOfBounds {
            left: 0,
            right: 2,
            image_count: 2,
        })
    );

    let singular = MatchInfo {
        homography: Matrix3::zeros(),
        ..relation.clone()
    };
    assert_eq!(
        canonical_relation_view(&images, 0, 1, &singular),
        Err(CanonicalRelationViewError::NonInvertible)
    );

    let canonical = canonical_relation_view(&images, 0, 1, &relation)
        .expect("invertible relation has a canonical view");
    assert_eq!((canonical.source_index, canonical.target_index), (1, 0));
    assert!(canonical.inverted);
    assert_eq!(
        canonical.homography,
        relation.homography.try_inverse().unwrap()
    );
}

#[test]
fn large_stack_matching_scales_with_neighbors_instead_of_all_pairs() {
    let small_pairs = pairs_to_match(SCALABLE_STACK_THRESHOLD);
    assert_eq!(
        small_pairs.len(),
        SCALABLE_STACK_THRESHOLD * (SCALABLE_STACK_THRESHOLD - 1) / 2
    );

    let large_pairs = pairs_to_match(200);
    assert!(large_pairs.len() <= 200 * LARGE_STACK_NEIGHBOR_WINDOW);
    assert!(large_pairs.len() * 10 < 200 * 199 / 2);
    for index in 0..199 {
        assert!(large_pairs.contains(&(index, index + 1)));
    }
}

#[test]
fn large_stack_candidates_include_natural_filename_neighbors() {
    let images = (0..31)
        .map(|index| {
            let number = (index * 13) % 31;
            test_image(index, &format!("tile-{number}.jpg"))
        })
        .collect::<Vec<_>>();
    let pairs = pairs_to_match_for_images(&images, BlendMode::Panorama);
    let index_for_number = |number: usize| {
        images
            .iter()
            .position(|image| image.filename == format!("tile-{number}.jpg"))
            .expect("generated filename should exist")
    };

    for number in 0..30 {
        let left = index_for_number(number);
        let right = index_for_number(number + 1);
        assert!(pairs.contains(&(left.min(right), left.max(right))));
    }
    assert!(pairs.len() <= images.len() * LARGE_STACK_NEIGHBOR_WINDOW * 2);
}

#[test]
fn medium_focus_stack_candidates_are_independent_of_upload_order() {
    let images = (0..45)
        .map(|index| test_image(index, &format!("upload-random-{index}.jpg")))
        .collect::<Vec<_>>();
    let pairs = pairs_to_match_for_images(&images, BlendMode::FocusStack);

    assert_eq!(
        pairs.len(),
        45 * 44 / 2,
        "a medium focus stack must inspect every possible overlap before ordering"
    );
    assert!(pairs.contains(&(0, 44)));
    assert!(pairs.contains(&(7, 31)));
}

#[test]
fn medium_panorama_candidates_cover_a_possible_lens_switch() {
    let images = (0..69)
        .map(|index| test_image(index, &format!("frame-{index}.jpg")))
        .collect::<Vec<_>>();
    let pairs = pairs_to_match_for_images(&images, BlendMode::Panorama);

    assert_eq!(pairs.len(), 69 * 68 / 2);
    assert!(pairs.contains(&(0, 68)));
}

#[test]
fn natural_path_order_compares_numeric_filename_runs() {
    let mut paths = ["tile-10.jpg", "tile-2.jpg", "tile-001.jpg", "tile-1.jpg"];
    paths.sort_by(|left, right| natural_path_cmp(left, right));

    assert_eq!(
        paths,
        ["tile-1.jpg", "tile-001.jpg", "tile-2.jpg", "tile-10.jpg"]
    );
}

#[test]
fn focus_stack_geometry_uses_overlap_evidence_and_rendering_uses_capture_order() {
    let images = vec![
        focus_test_image(0, "random-c.jpg"),
        focus_test_image(1, "random-a.jpg"),
        focus_test_image(2, "random-d.jpg"),
        focus_test_image(3, "random-b.jpg"),
    ];
    let matches = HashMap::from([
        ((0, 2), translation_match(100.0, 0.0, 30)),
        ((2, 3), translation_match(100.0, 0.0, 35)),
        ((3, 1), translation_match(100.0, 0.0, 25)),
        // A long overlap can have more descriptors than its immediate
        // neighbor. Motion-aware graph weights must still prefer the
        // local capture path.
        ((0, 3), translation_match(200.0, 0.0, 400)),
        ((0, 1), translation_match(300.0, 0.0, 350)),
    ]);

    let (order, homographies) = build_focus_stack_stitching_order(&images, &matches);

    assert_eq!(order.len(), images.len());
    assert_eq!(
        order.iter().copied().collect::<HashSet<_>>().len(),
        images.len()
    );
    assert!(order.windows(2).all(|pair| {
        matches.contains_key(&(pair[0], pair[1])) || matches.contains_key(&(pair[1], pair[0]))
    }));
    assert_eq!(homographies.len(), images.len());
    assert!(
        homographies
            .values()
            .any(|matrix| *matrix == Matrix3::identity())
    );
}

#[test]
fn dense_focus_overlap_graph_identifies_a_continuous_scan() {
    let images = (0..12)
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();
    let mut matches = HashMap::new();
    for source in 0..images.len() {
        for target in source + 1..images.len() {
            matches.insert(
                (source, target),
                observed_translation_match(0.0, (target - source) as f64 * 12.0, 32),
            );
        }
    }

    assert!(focus_match_graph_is_dense_continuous_scan(
        &images, &matches
    ));
}

/// Build single-source stations plus image-level world poses for focused
/// compositor-order tests. `centers` are world centres of 1000×1000
/// stations, exactly the observations consumed by production topology
/// inference (任务 9.10).
fn compositor_order_fixture(
    centers: &[(f64, f64)],
) -> (
    Vec<ImageInfo>,
    Vec<FocusCaptureGroup>,
    HashMap<usize, Matrix3<f64>>,
) {
    let images = (0..centers.len())
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 3680 + index)))
        .collect::<Vec<_>>();
    let groups = (0..centers.len())
        .map(|index| FocusCaptureGroup {
            members: vec![index],
            anchor: index,
            local_to_anchor: HashMap::from([(index, Matrix3::identity())]),
        })
        .collect::<Vec<_>>();
    let poses = centers
        .iter()
        .enumerate()
        .map(|(index, &(x, y))| {
            (
                images[index].id,
                Matrix3::new(1.0, 0.0, x - 500.0, 0.0, 1.0, y - 500.0, 0.0, 0.0, 1.0),
            )
        })
        .collect::<HashMap<_, _>>();
    (images, groups, poses)
}

#[test]
fn compositor_order_uses_topology_rows_and_columns_not_world_coordinate_drift() {
    // Per-station drift makes a continuous y/x sort interleave each row in
    // the wrong order. Production topology first clusters columns, assigns
    // rows inside them, then composes top-to-bottom and right-to-left.
    let (images, groups, poses) =
        compositor_order_fixture(&[(900.0, 20.0), (900.0, 900.0), (0.0, 0.0), (0.0, 880.0)]);
    let station_topology = focus_station_topology(&images, &groups, &poses);

    assert_eq!(station_topology.row, vec![0, 1, 0, 1]);
    assert_eq!(station_topology.column, vec![0, 0, 1, 1]);
    assert_eq!(
        focus_source_order_group_compositor_order(groups.len(), &station_topology),
        vec![0, 2, 1, 3]
    );
}

#[test]
fn compositor_order_uses_station_index_only_for_exact_topology_ties() {
    // Requirement 5.1 prevents this malformed tie in production. Keeping a
    // final station key nevertheless makes synthetic/partial topology
    // deterministic without consulting filenames, world coordinates or
    // candidate/import order.
    let station_topology = topology::StationTopology {
        row: vec![0, 0, 1],
        column: vec![0, 0, 0],
        ..topology::StationTopology::default()
    };

    assert_eq!(
        focus_source_order_group_compositor_order(3, &station_topology),
        vec![0, 1, 2]
    );
}

#[test]
fn compositor_order_is_invariant_to_rename_and_import_order() {
    // Physical station identities in canonical topology order: right-top,
    // left-top, right-bottom, left-bottom.
    let physical_centers = [(900.0, 10.0), (0.0, 0.0), (900.0, 900.0), (0.0, 890.0)];
    let (original_images, original_groups, original_poses) =
        compositor_order_fixture(&physical_centers);
    let original_topology =
        focus_station_topology(&original_images, &original_groups, &original_poses);
    let original_order =
        focus_source_order_group_compositor_order(original_groups.len(), &original_topology);

    // Reindex stations by shuffled import order and rename every source.
    // Map the resulting station indices back to physical identities before
    // comparing, because import indices themselves intentionally differ.
    let permutation = [3usize, 0, 2, 1];
    let shuffled_centers = permutation
        .iter()
        .map(|&physical| physical_centers[physical])
        .collect::<Vec<_>>();
    let (mut shuffled_images, shuffled_groups, shuffled_poses) =
        compositor_order_fixture(&shuffled_centers);
    let shuffled_count = shuffled_images.len();
    for (index, image) in shuffled_images.iter_mut().enumerate() {
        image.filename = format!("renamed-{}.NEF", shuffled_count - index);
    }
    let shuffled_topology =
        focus_station_topology(&shuffled_images, &shuffled_groups, &shuffled_poses);
    let shuffled_order =
        focus_source_order_group_compositor_order(shuffled_groups.len(), &shuffled_topology);

    let original_physical = original_order;
    let shuffled_physical = shuffled_order
        .into_iter()
        .map(|station| permutation[station])
        .collect::<Vec<_>>();
    assert_eq!(original_physical, vec![0, 1, 2, 3]);
    assert_eq!(shuffled_physical, original_physical);
}

#[test]
fn raw_physical_exposure_normalisation_uses_camera_settings() {
    let mut metadata = HashMap::from([
        ("ExposureTime".into(), "1/3 s".into()),
        ("FNumber".into(), "f/4".into()),
        ("ISOSpeedRatings".into(), "1000".into()),
    ]);
    let reference = stack_raw_exposure_value(&metadata).unwrap();
    assert!((reference - 1000.0 / 48.0).abs() < 1e-9);
    metadata.insert("ExposureTime".into(), "1/4 s".into());
    assert!((reference / stack_raw_exposure_value(&metadata).unwrap() - 4.0 / 3.0).abs() < 1e-9);
    metadata.insert("ExposureTime".into(), "0.625 s".into());
    metadata.insert("FNumber".into(), "f/5".into());
    assert_eq!(stack_raw_exposure_value(&metadata), Some(25.0));
    metadata.insert("FNumber".into(), "f/0".into());
    assert!(stack_raw_exposure_value(&metadata).is_none());
    metadata.remove("ExposureTime");
    assert!(stack_raw_exposure_value(&metadata).is_none());
}

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
fn sparse_focus_overlap_graph_keeps_evidence_selected_render_order() {
    let images = (0..12)
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();
    let matches = (0..images.len() - 1)
        .map(|source| {
            (
                (source, source + 1),
                observed_translation_match(0.0, 12.0, 32),
            )
        })
        .collect::<HashMap<_, _>>();

    assert!(!focus_match_graph_is_dense_continuous_scan(
        &images, &matches
    ));
}

#[test]
fn compact_three_station_focus_run_executes_strict_station_solver() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let capture_numbers = [1000, 1009, 1001, 1008, 1002, 1007, 1003, 1006, 1004, 1005];
    let images = capture_numbers
        .iter()
        .enumerate()
        .map(|(index, number)| focus_textured_test_image(index, &format!("DSC_{number:04}.NEF")))
        .collect::<Vec<_>>();
    assert!(images.len() < SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES);

    // Three focus brackets (4 + 3 + 3 frames) are connected by two
    // multi-layer consensus bridges. The bridges seed station geometry but
    // cannot merge the brackets into one Capture_Station.
    let mut matches = HashMap::new();
    for &(start, end) in &[(0usize, 4usize), (4, 7), (7, 10)] {
        for index in start..end - 1 {
            matches.insert((index, index + 1), observed_translation_match(0.0, 0.0, 36));
        }
    }
    for &(left, right) in &[(3usize, 4usize), (6, 7)] {
        let mut bridge = observed_translation_match(0.0, 0.0, 36);
        bridge.coarse_bridge = true;
        bridge.top_candidate_points = bridge.points.clone();
        matches.insert((left, right), bridge);
    }

    let mut closure_report = stack_report::ClosureReport::default();
    let mut relation_report = stack_report::StationRelationsReport::default();
    let (render_order, solved) = build_focus_stack_stitching_order_with_report(
        &images,
        &matches,
        &mut closure_report,
        &mut relation_report,
        true,
    );

    assert_eq!(solved.len(), images.len());
    assert_eq!(relation_report.accepted.len(), 2);
    assert_eq!(relation_report.connectivity.components, 1);
    assert_eq!(closure_report.participating_constraints, 2);
    assert_eq!(
        closure_report.status,
        stack_report::ClosureStatus::NoConstraints
    );

    // Solving compact station geometry must not opt the ownership renderer
    // into the large-scan filename heuristic.
    let filename_order = focus_filename_order_with_overview_gaps(&images, &matches);
    assert_eq!(render_order.len(), images.len());
    assert_ne!(render_order, filename_order);
}

#[test]
fn compact_single_station_focus_run_skips_inter_station_solver() {
    let images = (0..10)
        .map(|index| focus_textured_test_image(index, &format!("DSC_{:04}.NEF", 2000 + index)))
        .collect::<Vec<_>>();
    let matches = (0..images.len() - 1)
        .map(|index| ((index, index + 1), observed_translation_match(0.0, 0.0, 36)))
        .collect::<HashMap<_, _>>();

    let mut closure_report = stack_report::ClosureReport::default();
    let mut relation_report = stack_report::StationRelationsReport::default();
    let (render_order, solved) = build_focus_stack_stitching_order_with_report(
        &images,
        &matches,
        &mut closure_report,
        &mut relation_report,
        true,
    );

    assert_eq!(render_order.len(), images.len());
    assert_eq!(solved.len(), images.len());
    assert!(relation_report.accepted.is_empty());
    assert_eq!(relation_report.rejected_count, 0);
    assert_eq!(relation_report.connectivity.components, 0);
    assert_eq!(closure_report.status, stack_report::ClosureStatus::NotRun);
    assert_eq!(closure_report.participating_constraints, 0);
}

#[test]
fn large_focus_scan_keeps_capture_order_when_one_weak_frame_is_disconnected() {
    let images = (0..65)
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();
    let skipped = 32usize;
    let retained = (0..images.len())
        .filter(|index| *index != skipped)
        .collect::<Vec<_>>();
    let matches = retained
        .windows(2)
        .map(|pair| {
            (
                (pair[0].min(pair[1]), pair[0].max(pair[1])),
                translation_match(20.0, 0.0, 32),
            )
        })
        .collect::<HashMap<_, _>>();

    let (order, _) = build_focus_stack_stitching_order(&images, &matches);

    assert_eq!(order, retained);
}

#[test]
fn local_focus_bracket_lock_restores_the_direct_relative_pose() {
    let mut images = vec![
        focus_test_image(0, "DSC_1001.NEF"),
        focus_test_image(1, "DSC_1002.NEF"),
    ];
    for image in &mut images {
        image.focal_length_35mm = Some(90.0);
    }
    let matches = HashMap::from([((0, 1), observed_translation_match(10.0, 4.0, 32))]);
    let global = HashMap::from([
        (0, Matrix3::identity()),
        (
            1,
            Matrix3::new(1.0, 0.0, 90.0, 0.0, 1.0, 40.0, 0.0, 0.0, 1.0),
        ),
    ]);

    let locked = lock_focus_local_bracket_poses(&images, &matches, &global);
    let relative = locked[&1]
        .try_inverse()
        .expect("locked target pose should be invertible")
        * locked[&0];

    assert!((relative[(0, 2)] - 10.0).abs() < 1e-9);
    assert!((relative[(1, 2)] - 4.0).abs() < 1e-9);
}

#[test]
fn capture_stations_do_not_merge_across_a_coarse_sequence_boundary() {
    let images = (0..7)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let mut coarse_boundary = observed_translation_match(120.0, 0.0, 48);
    coarse_boundary.coarse_bridge = true;
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(3.0, 1.0, 600)),
        ((1, 2), observed_translation_match(2.0, 1.0, 540)),
        ((2, 3), coarse_boundary),
        ((3, 4), observed_translation_match(3.0, 0.0, 520)),
        ((4, 5), observed_translation_match(2.0, 1.0, 490)),
        // A weak direct boundary match also cannot define a bracket.
        ((5, 6), observed_translation_match(140.0, 0.0, 95)),
    ]);

    let stations = focus_capture_stations(&images, &matches);
    assert_eq!(stations, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]]);
}

#[test]
fn capture_stations_keep_a_low_texture_direct_focus_layer() {
    let images = (0..3)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(3.0, 1.0, 540)),
        ((1, 2), observed_translation_match(4.0, 2.0, 64)),
    ]);

    assert_eq!(
        focus_capture_stations(&images, &matches),
        vec![vec![0, 1, 2]]
    );
}

#[test]
fn capture_groups_keep_long_focus_brackets() {
    let images = (0..12)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    // 需求 1.4 budgets 0.02 long sides of *accumulated* centre displacement
    // per Capture_Station, so a 12-frame bracket may drift at most
    // 20px/11 ≈ 1.8px per layer.  0.4px of refocus breathing per layer adds
    // up to 0.0049 and stays one station.
    let matches = (0..images.len() - 1)
        .map(|index| ((index, index + 1), observed_translation_match(0.4, 0.2, 80)))
        .collect::<HashMap<_, _>>();
    let locked = images
        .iter()
        .map(|image| (image.id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();

    let (groups, group_for_image) = focus_capture_groups(&images, &matches, &locked).unwrap();

    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].members, (0..12).collect::<Vec<_>>());
    assert_eq!(group_for_image, vec![0; 12]);
}

#[test]
fn capture_groups_use_geometry_when_filenames_have_unrelated_numbers() {
    // A file picker can rename or re-number captures.  The same-position
    // decision must therefore come from the verified overlap and solved
    // centre motion, rather than treating a numeric filename gap as a
    // camera-position boundary.
    let mut first = focus_test_image(0, "capture-1000.NEF");
    first.focal_length_35mm = Some(75.0);
    let mut second = focus_test_image(1, "capture-9000.NEF");
    second.focal_length_35mm = Some(75.0);
    let images = vec![first, second];
    let matches = HashMap::from([((0, 1), observed_translation_match(0.0, 0.0, 16))]);
    let locked = images
        .iter()
        .map(|image| (image.id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();

    let (groups, group_for_image) =
        focus_capture_groups(&images, &matches, &locked).expect("poses are invertible");

    assert_eq!(groups.len(), 1);
    assert_eq!(
        groups[0].members.iter().copied().collect::<HashSet<_>>(),
        HashSet::from([0, 1])
    );
    assert_eq!(group_for_image, vec![0, 0]);
}

#[test]
fn capture_groups_split_accumulated_scan_motion() {
    let images = (0..12)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let matches = (0..images.len() - 1)
        .map(|index| {
            (
                (index, index + 1),
                observed_translation_match(20.0, 0.0, 80),
            )
        })
        .collect::<HashMap<_, _>>();

    let groups = focus_local_bracket_components(&images, &matches);

    assert!(groups.len() > 1);
    assert!(groups.iter().all(|group| group.len() <= 4));
    assert_eq!(groups.iter().map(Vec::len).sum::<usize>(), images.len());

    let locked = images
        .iter()
        .enumerate()
        .map(|(index, image)| {
            (
                image.id,
                Matrix3::new(1.0, 0.0, index as f64 * 20.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            )
        })
        .collect::<HashMap<_, _>>();
    let (groups, _) = focus_capture_groups(&images, &matches, &locked).unwrap();
    assert!(groups.len() > 1);
    assert!(groups.iter().all(|group| group.members.len() <= 4));
}

#[test]
fn homography_scale_ratio_is_the_singular_value_geometric_mean() {
    assert!((homography_scale_ratio(&Matrix3::identity()).unwrap() - 1.0).abs() < 1e-12);

    // A uniform 3% enlargement reports 1.03 regardless of rotation.
    let (sin, cos) = 0.4f64.sin_cos();
    let rotated_scale = Matrix3::new(
        1.03 * cos,
        -1.03 * sin,
        120.0,
        1.03 * sin,
        1.03 * cos,
        -40.0,
        0.0,
        0.0,
        1.0,
    );
    assert!((homography_scale_ratio(&rotated_scale).unwrap() - 1.03).abs() < 1e-9);

    // An anisotropic stretch reports the geometric mean of both axes.
    let anisotropic = Matrix3::new(1.20, 0.0, 0.0, 0.0, 0.80, 0.0, 0.0, 0.0, 1.0);
    let expected = (1.20f64 * 0.80).sqrt();
    assert!((homography_scale_ratio(&anisotropic).unwrap() - expected).abs() < 1e-12);

    // A homography is defined up to an overall scale, so an unnormalised
    // matrix must report the same ratio as its normalised form.
    let unnormalised = Matrix3::identity() * 4.0;
    assert!((homography_scale_ratio(&unnormalised).unwrap() - 1.0).abs() < 1e-12);
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

#[test]
fn prior_free_seed_fallback_requires_support_consistent_with_polish() {
    assert!(virtual_tile_prior_free_seed_fallback_conditions(
        true,
        "insufficient_polished_support",
        141,
        141,
    ));
    assert!(virtual_tile_prior_free_seed_fallback_conditions(
        true,
        "insufficient_polished_support",
        141,
        579,
    ));
    assert!(!virtual_tile_prior_free_seed_fallback_conditions(
        true,
        "insufficient_polished_support",
        579,
        141,
    ));
    assert!(!virtual_tile_prior_free_seed_fallback_conditions(
        true, "fitted", 141, 579,
    ));
}

#[test]
fn prior_repair_bounds_use_finite_inclusive_shift_and_scale_limits() {
    assert!(virtual_tile_prior_repair_is_bounded(
        CLOSURE_MAX_CORNER_CORRECTION_PX,
        STATION_RELATION_SCALE_RATIO_MIN,
    ));
    assert!(virtual_tile_prior_repair_is_bounded(
        0.0,
        STATION_RELATION_SCALE_RATIO_MAX,
    ));
    assert!(!virtual_tile_prior_repair_is_bounded(
        CLOSURE_MAX_CORNER_CORRECTION_PX + 1.0,
        1.0,
    ));
    assert!(!virtual_tile_prior_repair_is_bounded(
        1.0,
        STATION_RELATION_SCALE_RATIO_MIN - f64::EPSILON,
    ));
    assert!(!virtual_tile_prior_repair_is_bounded(f64::NAN, 1.0,));
    assert!(!virtual_tile_prior_repair_is_bounded(1.0, f64::INFINITY,));
}

#[test]
fn station_pose_jacobian_accepts_bounded_projective_local_scale_without_mutation() {
    // Every one of the eight homography parameters is free and non-trivial;
    // the Local_Scale gate observes the matrix but never projects it onto a
    // similarity/affine model.
    let pose = Matrix3::new(
        1.012, -0.018, 73.0, 0.021, 0.997, -41.0, 0.000_015, -0.000_009, 1.0,
    );
    let original = pose;
    let (minimum, maximum) = station_pose_local_scale_range(&pose, (1200, 800))
        .expect("a nonsingular projective pose has a measurable Jacobian");
    let ratio = maximum / minimum;

    assert!(minimum > 0.0);
    assert!(ratio <= STATION_POSE_MAX_LOCAL_SCALE_RATIO);
    assert_eq!(pose, original, "the gate must preserve all eight pose DOF");
    assert_ne!(pose[(2, 0)], 0.0);
    assert_ne!(pose[(2, 1)], 0.0);
}

#[test]
fn station_pose_jacobian_rejects_excessive_local_scale_variation() {
    let pose = Matrix3::new(1.0, 0.01, 0.0, -0.02, 1.0, 0.0, 0.000_35, 0.0, 1.0);
    let (minimum, maximum) = station_pose_local_scale_range(&pose, (1200, 800))
        .expect("the perspective pose remains nonsingular over the tile");
    let ratio = maximum / minimum;

    assert!(ratio > STATION_POSE_MAX_LOCAL_SCALE_RATIO);
}

#[test]
fn capture_station_link_requires_all_four_pieces_of_evidence() {
    let images = (0..2)
        .map(|index| focus_textured_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();

    // Focus breathing: 36 inliers, high overlap correlation, broad support,
    // unit scale.
    let accepted = observed_translation_match(2.0, 1.0, 36);
    let evidence =
        focus_station_link_evidence(&images, 0, 1, &accepted).expect("the overlap is measurable");
    assert!(evidence.overlap_ncc_measured);
    assert!(evidence.inliers >= STATION_MIN_INLIER_MATCHES);
    assert!(evidence.overlap_ncc >= STATION_MIN_OVERLAP_NCC);
    assert!(evidence.spatial_support >= STATION_MIN_INLIER_SPATIAL_SUPPORT);
    assert!((STATION_SCALE_RATIO_MIN..=STATION_SCALE_RATIO_MAX).contains(&evidence.scale_ratio));
    assert!(focus_match_is_capture_station_link(
        &images, 0, 1, &accepted
    ));

    // Fewer than 30 verified correspondences is not station evidence, even
    // with a perfect transform.
    let mut too_few_inliers = accepted.clone();
    too_few_inliers
        .points
        .truncate(STATION_MIN_INLIER_MATCHES - 1);
    assert!(!focus_match_is_capture_station_link(
        &images,
        0,
        1,
        &too_few_inliers
    ));

    // A 5% scale step is a different camera distance, not a focus layer.
    let scaled = observed_transform_match(
        Matrix3::new(1.05, 0.0, 0.0, 0.0, 1.05, 0.0, 0.0, 0.0, 1.0),
        36,
    );
    let scaled_evidence = focus_station_link_evidence(&images, 0, 1, &scaled).expect("measurable");
    assert!(scaled_evidence.scale_ratio > STATION_SCALE_RATIO_MAX);
    assert!(!focus_match_is_capture_station_link(&images, 0, 1, &scaled));

    // Inliers packed into one corner carry no spatial support.
    let mut clustered = accepted.clone();
    clustered.points = (0..36)
        .map(|index| {
            let source = Point2::new(
                120.0 + f64::from(index % 6) * 2.0,
                120.0 + f64::from(index / 6) * 2.0,
            );
            (source, Point2::new(source.x + 2.0, source.y + 1.0))
        })
        .collect();
    let clustered_evidence =
        focus_station_link_evidence(&images, 0, 1, &clustered).expect("measurable");
    assert!(clustered_evidence.spatial_support < STATION_MIN_INLIER_SPATIAL_SUPPORT);
    assert!(!focus_match_is_capture_station_link(
        &images, 0, 1, &clustered
    ));

    // A displacement of a quarter frame decorrelates the overlap well below
    // the 0.60 bar even though the transform is a clean translation.
    let neighbouring_station = observed_translation_match(60.0, 0.0, 36);
    let neighbour_evidence =
        focus_station_link_evidence(&images, 0, 1, &neighbouring_station).expect("measurable");
    assert!(neighbour_evidence.overlap_ncc < STATION_MIN_OVERLAP_NCC);
    assert!(!focus_match_is_capture_station_link(
        &images,
        0,
        1,
        &neighbouring_station
    ));
}

#[test]
fn capture_sequence_weight_boost_never_reaches_station_membership() {
    // 需求 1.2: `FOCUS_BRACKET_MAX_CAPTURE_GAP` is a candidate-order hint for
    // the pose graph.  Two filename-adjacent captures whose overlap evidence
    // fails 需求 1.1 must stay in different Capture_Stations even while the
    // graph builder keeps boosting their edge weight.
    let images = vec![
        focus_textured_test_image(0, "DSC_1001.NEF"),
        focus_textured_test_image(1, "DSC_1002.NEF"),
    ];
    let capture_gap = trailing_capture_number(&images[0].filename)
        .zip(trailing_capture_number(&images[1].filename))
        .map(|(left, right)| left.abs_diff(right))
        .expect("both names carry a trailing capture number");
    assert!(capture_gap <= FOCUS_BRACKET_MAX_CAPTURE_GAP);

    let neighbouring_station = observed_translation_match(60.0, 0.0, 36);
    assert!(
        focus_graph_capture_sequence_boost(&images[0], &images[1], &neighbouring_station) > 1.0,
        "the capture-sequence hint must keep boosting the graph weight"
    );
    assert!(
        !focus_match_is_capture_station_link(&images, 0, 1, &neighbouring_station),
        "a capture-sequence hint must not create station membership"
    );
    assert!(
        focus_station_grouping(&images, &HashMap::from([((0, 1), neighbouring_station)]))
            .components
            .is_empty()
    );
}

#[test]
fn grouping_reports_an_isolated_source_without_moving_the_others() {
    // 需求 1.6: the isolated source is named, and the remaining sources keep
    // the grouping they would have had on their own.
    let images = (0..3)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let bracket = HashMap::from([((0, 1), observed_translation_match(2.0, 1.0, 36))]);
    let reference = focus_station_grouping(&images[..2], &bracket);

    let with_isolated = focus_station_grouping(&images, &bracket);
    assert_eq!(with_isolated.components, reference.components);
    assert_eq!(with_isolated.isolated, vec![2]);
}

#[test]
fn task_7_31_station_member_count_boundaries_are_exact() {
    let outcomes = [1usize, 2, 47, 48, 49].map(station_member_count_is_supported);
    assert_eq!(outcomes, [true, true, true, true, false]);
    assert_eq!(FOCUS_BRACKET_MAX_COMPONENT_SOURCES, 48);
    assert!(!station_member_count_is_supported(0));
}

#[test]
fn oversized_station_is_split_until_every_member_count_fits() {
    // 需求 1.5 / 1.10: a static 50-frame bracket has no motion to split on,
    // so the member ceiling has to do it, and every split is recorded.
    let count = FOCUS_BRACKET_MAX_COMPONENT_SOURCES + 2;
    let images = (0..count)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let matches = (0..count - 1)
        .map(|index| ((index, index + 1), observed_translation_match(0.0, 0.0, 36)))
        .collect::<HashMap<_, _>>();

    let grouping = focus_station_grouping(&images, &matches);

    assert_eq!(
        grouping.components.iter().map(Vec::len).sum::<usize>(),
        count
    );
    assert!(
        grouping
            .components
            .iter()
            .all(|station| station.len() <= FOCUS_BRACKET_MAX_COMPONENT_SOURCES)
    );
    assert!(
        grouping
            .splits
            .iter()
            .any(|split| split.reason == degradation::GROUPING_MEMBER_LIMIT_SPLIT)
    );
}

#[test]
fn accumulated_motion_split_records_its_position_and_ratio() {
    // 需求 1.4: 12 frames drifting 2.236px each cross 0.02 long sides of
    // accumulated displacement at the ninth step (9 × 0.002236 = 0.0201).
    let images = (0..12)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let matches = (0..images.len() - 1)
        .map(|index| ((index, index + 1), observed_translation_match(2.0, 1.0, 36)))
        .collect::<HashMap<_, _>>();

    let grouping = focus_station_grouping(&images, &matches);

    let motion_splits = grouping
        .splits
        .iter()
        .filter(|split| split.reason == degradation::GROUPING_ACCUMULATED_MOTION)
        .collect::<Vec<_>>();
    assert_eq!(motion_splits.len(), 1);
    assert_eq!(motion_splits[0].position, 9);
    assert!(motion_splits[0].motion_ratio > STATION_MAX_ACCUMULATED_CENTER_MOTION_RATIO);
    // The candidate order runs along the recovered world centres, whose
    // direction is a property of the measured relations rather than of the
    // import order, so assert the partition rather than which end it starts
    // from.
    let mut sizes = grouping.components.iter().map(Vec::len).collect::<Vec<_>>();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![3, 9]);
    let mut covered = grouping.components.concat();
    covered.sort_unstable();
    assert_eq!(covered, (0..12).collect::<Vec<_>>());
}

#[test]
fn focus_render_order_finishes_each_capture_group_before_the_next() {
    let images = (0..6)
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();
    let groups = HashMap::from([(0, 1), (1, 1), (2, 1), (3, 2), (4, 2), (5, 2)]);
    let mut render_order = vec![
        &images[4], &images[1], &images[3], &images[0], &images[5], &images[2],
    ];

    order_focus_render_sources_by_capture_group(&mut render_order, &groups);

    assert_eq!(
        render_order
            .iter()
            .map(|image| image.id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4, 5]
    );
}

#[test]
fn capture_groups_keep_an_isolated_layer_out_of_every_station() {
    // 需求 1.6: a source that satisfies the 需求 1.1 overlap evidence with no
    // other source is marked isolated and merged into no Capture_Station —
    // a close solved world centre is not a substitute for the missing
    // evidence — while the remaining sources keep the grouping they would
    // have had anyway.
    let images = (0..5)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(3.0, 1.0, 540)),
        ((3, 4), observed_translation_match(2.0, 1.0, 520)),
    ]);
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let locked = HashMap::from([
        (0, translation(0.0)),
        (1, translation(3.0)),
        // No accepted relation at all for image 2, even though its solved
        // centre sits at the same physical station as images 0 and 1.
        (2, translation(7.0)),
        (3, translation(900.0)),
        (4, translation(904.0)),
    ]);

    let (groups, group_for_image) = focus_capture_groups(&images, &matches, &locked).unwrap();
    assert_eq!(
        groups
            .iter()
            .map(|group| group.members.clone())
            .collect::<Vec<_>>(),
        vec![vec![0, 1], vec![3, 4]]
    );
    assert_eq!(group_for_image[2], usize::MAX);
    assert!(groups.iter().all(|group| !group.members.contains(&2)));
    assert_eq!(
        focus_station_grouping(&images, &matches).isolated,
        vec![2],
        "需求 1.6: the isolated source is still named for the Stack_Report"
    );
}

#[test]
fn capture_groups_keep_a_coarse_boundary_despite_close_solved_centers() {
    let images = (0..4)
        .map(|index| {
            let mut image = focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index));
            image.focal_length_35mm = Some(75.0);
            image
        })
        .collect::<Vec<_>>();
    let mut boundary = observed_translation_match(100.0, 0.0, 48);
    boundary.coarse_bridge = true;
    let matches = HashMap::from([
        ((0, 1), observed_translation_match(3.0, 1.0, 540)),
        ((1, 2), boundary),
        ((2, 3), observed_translation_match(2.0, 1.0, 520)),
    ]);
    let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let locked = HashMap::from([
        (0, translation(0.0)),
        (1, translation(3.0)),
        (2, translation(100.0)),
        (3, translation(104.0)),
    ]);

    let (groups, _) = focus_capture_groups(&images, &matches, &locked).unwrap();
    assert_eq!(
        groups
            .iter()
            .map(|group| group.members.clone())
            .collect::<Vec<_>>(),
        vec![vec![0, 1], vec![2, 3]]
    );
}

#[test]
fn virtual_tile_features_ignore_every_uncovered_pixel() {
    let width = 96;
    let height = 96;
    let covered = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                if (8..88).contains(&x) && (8..88).contains(&y) {
                    u8::MAX
                } else {
                    0
                }
            })
        })
        .collect::<Vec<_>>();
    let coverage = stitching::CoverageMask::from_bytes(width, height, covered).unwrap();
    let covered_value = |x: u32, y: u32| {
        if ((x / 6) + (y / 6)) % 2 == 0 {
            0.1
        } else {
            0.9
        }
    };
    let first = Rgb32FImage::from_fn(width, height, |x, y| {
        let value = if coverage.is_covered(x, y) {
            covered_value(x, y)
        } else {
            0.0
        };
        image::Rgb([value, value, value])
    });
    let second = Rgb32FImage::from_fn(width, height, |x, y| {
        let value = if coverage.is_covered(x, y) {
            covered_value(x, y)
        } else if (x + y) % 2 == 0 {
            1.0
        } else {
            0.35
        };
        image::Rgb([value, value, value])
    });
    let brief_pairs = processing::generate_brief_pairs();

    let (first_gray, first_features) =
        virtual_tile_covered_features(&first, &coverage, &brief_pairs);
    let (second_gray, second_features) =
        virtual_tile_covered_features(&second, &coverage, &brief_pairs);
    let signature = |features: &[Feature]| {
        features
            .iter()
            .map(|feature| (feature.keypoint.x, feature.keypoint.y, feature.descriptor))
            .collect::<Vec<_>>()
    };

    assert_eq!(first_gray, second_gray);
    assert!(
        !first_features.is_empty(),
        "the covered texture must yield evidence"
    );
    assert_eq!(signature(&first_features), signature(&second_features));
    assert!(first_features.iter().all(|feature| {
        let x = feature.keypoint.x;
        let y = feature.keypoint.y;
        (y - VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS..=y + VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS).all(
            |sample_y| {
                (x - VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS
                    ..=x + VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS)
                    .all(|sample_x| coverage.is_covered(sample_x, sample_y))
            },
        )
    }));
}

fn synthetic_virtual_tile_topology(station_count: usize) -> topology::StationTopology {
    topology::StationTopology {
        row: vec![0; station_count],
        column: (0..station_count as u32).collect(),
        candidates: all_image_pairs(station_count)
            .into_iter()
            .map(|(left, right)| topology::CandidateAdjacency {
                left,
                right,
                kind: stack_report::AdjacencyKind::SameRow,
                score: 1.0,
            })
            .collect(),
        ..topology::StationTopology::default()
    }
}

fn synthetic_virtual_tiles(station_count: usize) -> Vec<ImageInfo> {
    (0..station_count)
        .map(|station| {
            let mut tile = focus_textured_test_image(
                usize::MAX - station,
                &format!("virtual://synthetic-station-{station}"),
            );
            // The fused representation is the requirement-6.1 evidence;
            // no individual source feature vector is retained here.
            tile.features.clear();
            tile.top_features.clear();
            tile
        })
        .collect()
}
/// Solver coverages of tiles whose single analysis plane is covered
/// everywhere and doubles as their finest match plane.
fn fully_covered_solver_coverages<'a>(
    tiles: &[ImageInfo],
    masks: &'a [stitching::CoverageMask],
) -> VirtualTileSolverCoverages<'a> {
    VirtualTileSolverCoverages {
        quality: masks,
        overlap: masks,
        overlap_factor: tiles
            .first()
            .map_or(1, |tile| tile.scale_factor.round().max(1.0) as u32),
    }
}
/// Relation-quality Coverage_Masks of tiles whose analysis planes are
/// covered everywhere.
fn fully_covered_quality_coverages(tiles: &[ImageInfo]) -> Vec<stitching::CoverageMask> {
    tiles
        .iter()
        .map(|tile| {
            let (width, height) = tile.alignment_image.dimensions();
            stitching::CoverageMask::from_bytes(
                width,
                height,
                vec![u8::MAX; width as usize * height as usize],
            )
            .expect("a fully covered mask matches its plane")
        })
        .collect()
}

fn guided_repeated_texture_fixture() -> (
    Vec<ImageInfo>,
    Vec<stitching::CoverageMask>,
    HashMap<usize, Matrix3<f64>>,
    topology::StationTopology,
) {
    let (width, height) = (720u32, 560u32);
    let world_texture = |world_x: u32, y: u32| {
        let coarse = ((world_x / 11 + y / 13) % 2) as u8 * 150;
        let fine = ((world_x / 5 + y / 7) % 2) as u8 * 70;
        20u8.saturating_add(coarse).saturating_add(fine)
    };
    let left_gray = GrayImage::from_fn(width, height, |x, y| image::Luma([world_texture(x, y)]));
    let right_gray = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([world_texture(x + 200, y)])
    });
    let points = (70..=490)
        .step_by(70)
        .flat_map(|y| (240..=660).step_by(70).map(move |left_x| (left_x, y)))
        .collect::<Vec<_>>();
    let descriptor_for = |index: usize| {
        let mut descriptor = [0u8; BRIEF_DESCRIPTOR_SIZE / 8];
        descriptor[index / 8] = 1u8 << (index % 8);
        descriptor
    };
    let left_features = points
        .iter()
        .enumerate()
        .map(|(index, &(left_x, y))| Feature {
            keypoint: KeyPoint { x: left_x, y },
            descriptor: descriptor_for(index),
            support_scale: 1.0,
        })
        .collect::<Vec<_>>();
    // Deliberately permute otherwise perfect descriptor identities. A
    // descriptor matcher therefore gets broad but geometrically
    // inconsistent repeated-texture pairs; direct pixels remain aligned.
    let right_features = points
        .iter()
        .enumerate()
        .map(|(index, &(left_x, y))| Feature {
            keypoint: KeyPoint { x: left_x - 200, y },
            descriptor: descriptor_for((index * 17 + 5) % points.len()),
            support_scale: 1.0,
        })
        .collect::<Vec<_>>();
    let mut tiles = synthetic_virtual_tiles(2);
    for tile in &mut tiles {
        tile.width = width;
        tile.height = height;
        tile.scale_factor = 1.0;
    }
    tiles[0].alignment_image = left_gray;
    tiles[0].features = left_features;
    tiles[1].alignment_image = right_gray;
    tiles[1].features = right_features;
    let coverage_bytes = (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| {
                u8::from((20..width - 20).contains(&x) && (20..height - 20).contains(&y)) * u8::MAX
            })
        })
        .collect::<Vec<_>>();
    let coverage = stitching::CoverageMask::from_bytes(width, height, coverage_bytes).unwrap();
    let right_to_world = Matrix3::new(1.0, 0.0, 200.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let initial = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, right_to_world),
    ]);
    (
        tiles,
        vec![coverage.clone(), coverage],
        initial,
        synthetic_virtual_tile_topology(2),
    )
}

fn direct_projective_fixture() -> (
    Vec<ImageInfo>,
    Vec<stitching::CoverageMask>,
    HashMap<usize, Matrix3<f64>>,
    topology::StationTopology,
    Matrix3<f64>,
) {
    let (width, height) = (420u32, 340u32);
    let texture = |x: f64, y: f64| {
        let value = 126.0
            + 47.0 * (x * 0.091 + y * 0.037).sin()
            + 38.0 * (x * 0.027 - y * 0.113).cos()
            + 24.0 * (x * 0.173 + y * 0.149).sin()
            + 13.0 * (x * 0.311 - y * 0.071).cos();
        value.clamp(0.0, 255.0)
    };
    let actual = Matrix3::new(
        1.006, 0.009, -82.0, -0.006, 0.996, 11.0, 0.000_018, -0.000_012, 1.0,
    );
    let inverse = actual.try_inverse().unwrap();
    let left = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([texture(f64::from(x), f64::from(y)).round() as u8])
    });
    let right = GrayImage::from_fn(width, height, |x, y| {
        let point = transformed_point(&inverse, Point2::new(f64::from(x), f64::from(y)));
        let value = point.map_or(0.0, |point| texture(point.x, point.y) * 0.82 + 19.0);
        image::Luma([value.clamp(0.0, 255.0).round() as u8])
    });
    let mut tiles = synthetic_virtual_tiles(2);
    for tile in &mut tiles {
        tile.width = width;
        tile.height = height;
        tile.scale_factor = 2.0;
        tile.features.clear();
    }
    tiles[0].alignment_image = left;
    tiles[1].alignment_image = right;
    let coverage = stitching::CoverageMask::from_bytes(
        width,
        height,
        vec![u8::MAX; width as usize * height as usize],
    )
    .unwrap();
    let mut initial_left_to_right = actual;
    initial_left_to_right[(0, 2)] += 4.5;
    initial_left_to_right[(1, 2)] -= 3.75;
    let initial = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, initial_left_to_right.try_inverse().unwrap()),
    ]);
    (
        tiles,
        vec![coverage.clone(), coverage],
        initial,
        synthetic_virtual_tile_topology(2),
        actual,
    )
}

fn nonzero_origin_virtual_tile_fixture() -> (
    Vec<ImageInfo>,
    Vec<stitching::CoverageMask>,
    HashMap<usize, Matrix3<f64>>,
    topology::StationTopology,
    Matrix3<f64>,
) {
    let left_dimensions = (560u32, 420u32);
    let right_dimensions = (500u32, 460u32);
    let left_to_world = Matrix3::new(1.0, 0.0, 140.0, 0.0, 1.0, -80.0, 0.0, 0.0, 1.0);
    let right_to_world = Matrix3::new(1.0, 0.0, 260.0, 0.0, 1.0, -55.0, 0.0, 0.0, 1.0);
    let actual = right_to_world.try_inverse().unwrap() * left_to_world;
    let world_texture = |x: f64, y: f64| {
        (127.0
            + 48.0 * (x * 0.071 + y * 0.043).sin()
            + 37.0 * (x * 0.029 - y * 0.109).cos()
            + 22.0 * (x * 0.163 + y * 0.137).sin())
        .clamp(0.0, 255.0)
    };
    let render = |dimensions: (u32, u32), tile_to_world: &Matrix3<f64>| {
        GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
            let world =
                transformed_point(tile_to_world, Point2::new(f64::from(x), f64::from(y))).unwrap();
            image::Luma([world_texture(world.x, world.y).round() as u8])
        })
    };
    let mut tiles = synthetic_virtual_tiles(2);
    tiles[0].width = left_dimensions.0;
    tiles[0].height = left_dimensions.1;
    tiles[0].alignment_image = render(left_dimensions, &left_to_world);
    tiles[1].width = right_dimensions.0;
    tiles[1].height = right_dimensions.1;
    tiles[1].alignment_image = render(right_dimensions, &right_to_world);
    let coverages = [left_dimensions, right_dimensions]
        .into_iter()
        .map(|(width, height)| {
            stitching::CoverageMask::from_bytes(
                width,
                height,
                vec![u8::MAX; width as usize * height as usize],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let initial = HashMap::from([(tiles[0].id, left_to_world), (tiles[1].id, right_to_world)]);
    (
        tiles,
        coverages,
        initial,
        synthetic_virtual_tile_topology(2),
        actual,
    )
}

#[test]
fn direct_grid_nonzero_origins_and_different_dimensions_keep_station_mapping() {
    let (tiles, coverages, initial, topology, actual) = nonzero_origin_virtual_tile_fixture();
    let outcome = virtual_tile_station_match_outcome(&tiles, &coverages, &initial, &topology);
    let relation = outcome
        .relations
        .get(&(0, 1))
        .expect("nonzero Virtual_Tile origins must remain in the same station frame");
    assert!(relation.inliers >= STATION_RELATION_MIN_INLIERS);
    for point in [
        Point2::new(180.0, 90.0),
        Point2::new(310.0, 210.0),
        Point2::new(455.0, 350.0),
    ] {
        let expected = transformed_point(&actual, point).unwrap();
        let measured = transformed_point(&relation.homography, point).unwrap();
        assert!((expected - measured).norm() <= 0.75);
        let left_world = transformed_point(&initial[&tiles[0].id], point).unwrap();
        let right_world = transformed_point(&initial[&tiles[1].id], measured).unwrap();
        assert!((left_world - right_world).norm() <= 0.75);
    }

    let (mut reordered_tiles, mut reordered_coverages, reordered_initial, reordered_topology, _) =
        nonzero_origin_virtual_tile_fixture();
    reordered_tiles.swap(0, 1);
    reordered_coverages.swap(0, 1);
    let reordered = virtual_tile_station_matches(
        &reordered_tiles,
        &reordered_coverages,
        &reordered_initial,
        &reordered_topology,
    );
    let reordered_inverse = reordered
        .get(&(0, 1))
        .expect("station import reordering must retain measured geometry")
        .homography
        .try_inverse()
        .unwrap();
    for (forward, reverse) in relation.homography.iter().zip(reordered_inverse.iter()) {
        assert!((forward - reverse).abs() < 1e-8);
    }

    let mut wrong_mapping = initial.clone();
    wrong_mapping.insert(tiles[0].id, initial[&tiles[1].id]);
    wrong_mapping.insert(tiles[1].id, initial[&tiles[0].id]);
    assert!(
        virtual_tile_station_matches(&tiles, &coverages, &wrong_mapping, &topology).is_empty(),
        "swapping station origins must not be accepted as measured evidence"
    );
}

#[test]
fn polished_matcher_accepts_a_forward_shift_beyond_reverse_probe_radius() {
    // A five-pixel forward residual is within the production polish pass
    // (±8px), but outside the old reverse probe (±2px) when that probe was
    // centred on the forward match.  The reverse check must use the source
    // centre so it can verify the same correspondence without widening
    // its independent radius.
    let (width, height) = (420u32, 320u32);
    let texture = |x: f64, y: f64| {
        (127.0
            + 55.0 * (x * 0.071 + y * 0.043).sin()
            + 41.0 * (x * 0.029 - y * 0.109).cos()
            + 23.0 * (x * 0.163 + y * 0.137).sin())
        .clamp(0.0, 255.0)
        .round() as u8
    };
    let left = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([texture(f64::from(x), f64::from(y))])
    });
    let shift = 5.0;
    let right = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([texture(f64::from(x) - shift, f64::from(y))])
    });
    let coverage = stitching::CoverageMask::from_bytes(
        width,
        height,
        vec![u8::MAX; width as usize * height as usize],
    )
    .expect("the synthetic polish planes are fully covered");
    let mut planes = synthetic_virtual_tiles(2);
    let mut left_plane = planes.remove(0);
    let mut right_plane = planes.remove(0);
    left_plane.width = width;
    left_plane.height = height;
    left_plane.alignment_image = left;
    right_plane.width = width;
    right_plane.height = height;
    right_plane.alignment_image = right;
    let (_, mut diagnostic) = virtual_tile_direct_station_match(
        0,
        1,
        &[],
        &[],
        &HashMap::new(),
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    );
    let relation = virtual_tile_polished_station_match(
        &left_plane,
        &coverage,
        &right_plane,
        &coverage,
        1,
        &Matrix3::identity(),
        ((width, height), (width, height)),
        &mut diagnostic,
    )
    .into_iter()
    .last()
    .expect("the production polish matcher must verify the five-pixel shift");
    assert_eq!(diagnostic.failure_stage, "fitted");
    assert!(relation.inliers >= STATION_RELATION_MIN_INLIERS);
    let measured = transformed_point(&relation.homography, Point2::new(210.0, 160.0))
        .expect("fitted homography maps the source point");
    assert!((measured.x - 215.0).abs() < 1.0);
    assert!((measured.y - 160.0).abs() < 1.0);
}

#[test]
fn coarse_pyramid_retry_accepts_finer_sampling_failures_without_relaxing_gates() {
    for stage in [
        "insufficient_grid_probes",
        "insufficient_measurable_patches",
        "insufficient_bidirectional_matches",
        "insufficient_grid_coverage",
        "residual_model_no_fit",
        "insufficient_matches_after_model_quota",
        "insufficient_fitted_hull_support",
    ] {
        assert!(virtual_tile_coarse_failure_can_retry(stage));
    }
    for stage in [
        "no_predicted_overlap",
        "fitted_orientation_mismatch",
        "non_invertible_seed",
    ] {
        assert!(!virtual_tile_coarse_failure_can_retry(stage));
    }
}

#[test]
fn direct_grid_rejects_inverse_coordinate_direction() {
    let (tiles, coverages, mut initial, topology, actual) = nonzero_origin_virtual_tile_fixture();
    initial.insert(tiles[0].id, Matrix3::identity());
    initial.insert(tiles[1].id, actual);
    assert!(
        virtual_tile_station_matches(&tiles, &coverages, &initial, &topology).is_empty(),
        "using target-to-source as source-to-target must fail strict pixel evidence"
    );
}

#[test]
fn measured_reverse_transform_is_the_inverse_of_forward_residual_composition() {
    let initial = Matrix3::new(
        1.004, 0.009, -83.0, -0.006, 0.997, 12.0, 0.000_01, -0.000_02, 1.0,
    );
    let source = Point2::new(210.5, 143.25);
    let predicted = transformed_point(&initial, source).unwrap();
    let measured = predicted + nalgebra::Vector2::new(11.25, -7.5);
    let reverse = virtual_tile_measured_reverse_transform(&initial, source, measured).unwrap();
    assert!((transformed_point(&reverse, measured).unwrap() - source).norm() < 1e-9);
    assert!(
        (transformed_point(&initial.try_inverse().unwrap(), measured).unwrap() - source).norm()
            > VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX,
        "the inverse prior is not the inverse of a measured residual match"
    );
}

#[test]
fn direct_grid_recovers_projective_photometric_overlap() {
    let (tiles, coverages, initial, topology, actual) = direct_projective_fixture();
    let outcome = virtual_tile_station_match_outcome(&tiles, &coverages, &initial, &topology);
    let relation = outcome
        .relations
        .get(&(0, 1))
        .expect("direct fused pixels must recover the projective overlap");
    let diagnostic = &outcome.diagnostics[0];
    assert!(diagnostic.grid_probe_count <= VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES);
    assert!(diagnostic.measurable_patch_count >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.bidirectional_match_count >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.fitted_inliers >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.hull_support >= STATION_RELATION_MIN_SPATIAL_SUPPORT);
    for point in [
        Point2::new(100.0, 70.0),
        Point2::new(210.0, 170.0),
        Point2::new(335.0, 275.0),
    ] {
        let expected = transformed_point(&actual, point).unwrap();
        let measured = transformed_point(&relation.homography, point).unwrap();
        assert!(
            (expected - measured).norm() <= 1.0,
            "{expected:?} != {measured:?}"
        );
    }
}

#[test]
fn virtual_tile_direct_relations_leave_the_canonical_direction_to_the_solver() {
    let (tiles, coverages, initial, _, _) = direct_projective_fixture();
    let (relation, _) = virtual_tile_direct_station_match(
        0,
        1,
        &tiles,
        &coverages,
        &initial,
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    );
    let relation = relation.expect("the fixture overlap is measurable");
    // The fit is stored untouched in the (left, right) key direction.
    assert!(relation.canonical_homography.is_none());
}

#[test]
fn virtual_tile_solver_reads_a_key_direction_relation_in_either_content_direction() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    // Tile 1 shows tile 0's content shifted by `shift`: left p ↔ right p − shift.
    let (side, shift) = (1_000u32, nalgebra::Vector2::new(40.0, 6.0));
    let texture = |x: f64, y: f64| {
        (126.0
            + 47.0 * (x * 0.091 + y * 0.037).sin()
            + 38.0 * (x * 0.027 - y * 0.113).cos()
            + 24.0 * (x * 0.173 + y * 0.149).sin())
        .clamp(0.0, 255.0)
        .round() as u8
    };
    let mut tiles = synthetic_virtual_tiles(2);
    for (tile, offset) in tiles.iter_mut().zip([nalgebra::Vector2::zeros(), shift]) {
        tile.alignment_image = GrayImage::from_fn(side, side, |x, y| {
            image::Luma([texture(f64::from(x) + offset.x, f64::from(y) + offset.y)])
        });
        tile.width = side;
        tile.height = side;
        tile.scale_factor = 1.0;
    }
    let coverages = fully_covered_quality_coverages(&tiles);
    let left_to_right = Matrix3::new(1.0, 0.0, -shift.x, 0.0, 1.0, -shift.y, 0.0, 0.0, 1.0);
    let initial = HashMap::from([
        (tiles[0].id, Matrix3::identity()),
        (tiles[1].id, left_to_right.try_inverse().unwrap()),
    ]);
    // The production matcher's key-direction relation, not a hand-built one.
    let (relation, diagnostic) = virtual_tile_direct_station_match(
        0,
        1,
        &tiles,
        &coverages,
        &initial,
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    );
    let relation = relation.unwrap_or_else(|| panic!("measurable shift: {diagnostic:?}"));
    let matches = HashMap::from([((0, 1), relation)]);
    let station_topology = synthetic_virtual_tile_topology(2);

    let mut accepted = Vec::new();
    for (left_name, inverted) in [
        ("virtual://synthetic-station-0", false),
        ("virtual://z", true),
    ] {
        tiles[0].filename = left_name.to_string();
        // The content-derived measurement direction runs right→left when
        // `inverted`; the key-direction fit must then be read inverted.
        assert_eq!(canonical_match_direction(&tiles, 0, 1).2, inverted);
        let mut closure = stack_report::ClosureReport::default();
        let mut report = stack_report::StationRelationsReport::default();
        let solved = solve_virtual_tile_station_poses_with_report(
            &tiles,
            fully_covered_solver_coverages(&tiles, &coverages),
            &matches,
            &initial,
            &station_topology,
            &mut closure,
            &mut report,
        );
        assert_eq!(solved.len(), 2, "inverted {inverted}: {report:?}");
        assert_eq!(report.accepted.len(), 1, "inverted {inverted}: {report:?}");
        accepted.push(report.accepted[0]);
    }
    let (forward, reversed) = (&accepted[0], &accepted[1]);
    assert!(forward.median_error_px <= STATION_RELATION_MAX_MEDIAN_ERROR_PX);
    assert!((forward.median_error_px - reversed.median_error_px).abs() < 1e-6);
    assert_eq!(forward.inliers, reversed.inliers);
}

#[test]
fn anchor_frame_fusion_is_independent_of_global_station_placement() {
    let sources = (0..4)
        .map(|id| test_image(id, &format!("{id}.nef")))
        .collect::<Vec<_>>();
    let groups = vec![
        FocusCaptureGroup {
            members: vec![0, 1],
            anchor: 0,
            local_to_anchor: HashMap::new(),
        },
        FocusCaptureGroup {
            members: vec![2, 3],
            anchor: 2,
            local_to_anchor: HashMap::new(),
        },
    ];
    let child = Matrix3::new(1.001, 0.0, 5.0, 0.0, 1.001, -3.0, 0.0, 0.0, 1.0);
    let first = Matrix3::new(1.4, 0.2, 1000.0, 0.1, 1.2, 700.0, 0.0003, 0.0002, 1.0);
    let second = Matrix3::new(0.8, -0.3, 3000.0, 0.2, 0.9, 200.0, -0.0001, 0.0003, 1.0);
    let global = HashMap::from([
        (0, first),
        (1, first * child),
        (2, second),
        (3, second * child),
    ]);
    let local = focus_anchor_render_frames(&groups, &sources, &global, None).unwrap();
    for (id, expected) in [
        (0, Matrix3::identity()),
        (1, child),
        (2, Matrix3::identity()),
        (3, child),
    ] {
        assert!((local.homographies[&id] - expected).norm() < 1e-8);
    }
    assert!((local.anchor_to_world[0] - Matrix3::identity()).norm() < 1e-8);
    let mut moved = global.clone();
    let movement = Matrix3::new(1.3, 0.1, -800.0, 0.0, 1.1, 600.0, 0.0002, 0.0, 1.0);
    moved.insert(2, movement * second);
    moved.insert(3, movement * second * child);
    let other = focus_anchor_render_frames(&groups, &sources, &moved, None).unwrap();
    assert!((other.homographies[&3] - local.homographies[&3]).norm() < 1e-8);
    // Local source pixels and source provenance share the same composition.
    let expected_world = first.try_inverse().unwrap() * moved[&3];
    assert!(
        (normalized_homography(&(other.anchor_to_world[1] * other.homographies[&3])).unwrap()
            - normalized_homography(&expected_world).unwrap())
        .norm()
            < 1e-8
    );
}

#[test]
fn projective_repair_scale_is_measured_at_the_station_center() {
    let correction = Matrix3::new(1.5, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0005, 0.0, 1.0);
    let center = Point2::new(600.0, 400.0);
    assert!(!virtual_tile_prior_repair_is_bounded(
        38.9,
        homography_scale_ratio(&correction).unwrap()
    ));
    let scale = virtual_tile_repair_local_scale_ratio(&correction, center).unwrap();
    assert!(virtual_tile_prior_repair_is_bounded(38.9, scale));
    let translation = Matrix3::new(1.0, 0.0, 1000.0, 0.0, 1.0, 500.0, 0.0, 0.0, 1.0);
    let relocated = translation * correction * translation.try_inverse().unwrap();
    let relocated_center = transformed_point(&translation, center).unwrap();
    assert!(
        (virtual_tile_repair_local_scale_ratio(&relocated, relocated_center).unwrap() - scale)
            .abs()
            < 1e-8
    );
}

#[test]
fn virtual_tile_exposure_compensation_requires_verified_structure_and_held_out_pixels() {
    let mut tiles = synthetic_virtual_tiles(2);
    let texture = |x: f64, y: f64| {
        128.0 + 45.0 * (x * 0.075 + y * 0.035).sin() + 35.0 * (x * 0.027 - y * 0.081).cos()
    };
    for tile in &mut tiles {
        tile.width = 500;
        tile.height = 400;
        tile.scale_factor = 1.0;
    }
    tiles[0].alignment_image = GrayImage::from_fn(500, 400, |x, y| {
        image::Luma([texture(x as f64, y as f64).round() as u8])
    });
    tiles[1].alignment_image = GrayImage::from_fn(500, 400, |x, y| {
        image::Luma([(0.7 * texture(x as f64, y as f64)).round() as u8])
    });
    let good =
        focus_overlap_quality_measurements(&tiles[0], &tiles[1], &Matrix3::identity()).unwrap();
    assert!(good.low_frequency_mean_relative_difference > 0.29);
    let compensated = virtual_tile_photometric_measurements(good);
    assert!(
        compensated.0 < 0.01 && (compensated.1 - 1.0).abs() < 0.03,
        "{good:?}"
    );
    let wrong = Matrix3::new(1.0, 0.0, 30.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let bad = focus_overlap_quality_measurements(&tiles[0], &tiles[1], &wrong).unwrap();
    assert_eq!(
        virtual_tile_photometric_measurements(bad).0,
        bad.low_frequency_mean_relative_difference
    );
    assert!(virtual_tile_photometric_measurements(bad).0 > 0.2);
    let excessive = FocusOverlapQuality {
        exposure_gain: 3.0,
        ..good
    };
    assert_eq!(
        virtual_tile_photometric_measurements(excessive).0,
        excessive.low_frequency_mean_relative_difference
    );
}

#[test]
fn overlap_gradient_measurement_uses_the_registered_pixel_grid() {
    let mut tiles = synthetic_virtual_tiles(2);
    let angle = 0.35f64;
    let level_transform = Matrix3::new(
        angle.cos(),
        -angle.sin(),
        90.0,
        angle.sin(),
        angle.cos(),
        -60.0,
        0.0,
        0.0,
        1.0,
    );
    let inverse = level_transform.try_inverse().unwrap();
    let texture = |x: f64, y: f64| {
        image::Luma([(128.0
            + 45.0 * (x * 0.075 + y * 0.035).sin()
            + 35.0 * (x * 0.027 - y * 0.081).cos()) as u8])
    };
    tiles[0].alignment_image = GrayImage::from_fn(500, 400, |x, y| texture(x as f64, y as f64));
    tiles[1].alignment_image = GrayImage::from_fn(500, 400, |x, y| {
        let p = transformed_point(&inverse, Point2::new(x as f64, y as f64)).unwrap();
        texture(p.x, p.y)
    });
    for (tile, factor) in tiles.iter_mut().zip([4.0, 6.0]) {
        tile.width = (500.0 * factor) as u32;
        tile.height = (400.0 * factor) as u32;
        tile.scale_factor = factor;
    }
    let native = Matrix3::new(6.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0)
        * level_transform
        * Matrix3::new(0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0);
    let quality = focus_overlap_quality_measurements(&tiles[0], &tiles[1], &native).unwrap();
    assert!(quality.intensity_ncc > 0.99, "{quality:?}");
    // Quantized, independently sampled 8-bit gradients retain some
    // rounding noise even though the continuous texture is identical.
    assert!(quality.edge_ncc > 0.97, "{quality:?}");
    assert!(
        quality.median_edge_orientation_difference_degrees < 3.0,
        "{quality:?}"
    );
    assert!(
        (quality.edge_strength_ratio - 1.0).abs() < 0.03,
        "{quality:?}"
    );
}

#[test]
fn single_level_relation_quality_distinguishes_registered_and_shifted_structure() {
    let (mut tiles, coverages, _, _, actual) = direct_projective_fixture();
    let mut quality_coverages = Vec::new();
    for (tile, coverage) in tiles.iter_mut().zip(&coverages) {
        let plane = VirtualTileMatchPlane::from_native(&tile.alignment_image, coverage, 1).unwrap();
        let (quality, factor) =
            virtual_tile_relation_quality_plane(&[plane], &tile.alignment_image, coverage);
        assert_eq!(factor, 4);
        quality_coverages.push(
            VirtualTileMatchPlane::from_native(&tile.alignment_image, coverage, factor)
                .unwrap()
                .coverage,
        );
        tile.scale_factor *= f64::from(factor);
        tile.alignment_image = quality;
    }
    let scale = Matrix3::new(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0);
    let native = scale * actual * scale.try_inverse().unwrap();
    let good = focus_overlap_quality_measurements_on_coverage(
        &tiles[0],
        &tiles[1],
        &native,
        Some((&quality_coverages[0], &quality_coverages[1])),
    )
    .unwrap();
    let mut shifted = native;
    shifted[(0, 2)] += 80.0;
    let bad = focus_overlap_quality_measurements_on_coverage(
        &tiles[0],
        &tiles[1],
        &shifted,
        Some((&quality_coverages[0], &quality_coverages[1])),
    )
    .unwrap();
    assert!(
        good.median_edge_orientation_difference_degrees < 10.0,
        "{:?}",
        good.median_edge_orientation_difference_degrees
    );
    assert!(
        bad.median_edge_orientation_difference_degrees > 10.0,
        "{:?}",
        bad.median_edge_orientation_difference_degrees
    );
    assert!(good.intensity_ncc > bad.intensity_ncc + 0.2);
}

#[test]
fn direct_grid_structure_planes_only_carry_the_orientation_measurement() {
    let (tiles, coverages, initial, _, _) = direct_projective_fixture();
    let threshold = STATION_RELATION_MAX_MEDIAN_ERROR_PX;
    let (probe, probe_diagnostic) =
        virtual_tile_direct_station_match(0, 1, &tiles, &coverages, &initial, threshold);
    let probe = probe.expect("the probe planes measure the projective overlap");

    // σ = 1 structural planes (the coarse-octave rule) keep the probe and
    // the fitted relation; only the orientation measurement moves.
    let structure = tiles
        .iter()
        .map(|tile| {
            imageproc::filter::gaussian_blur_f32(
                &tile.alignment_image,
                VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
            )
        })
        .collect::<Vec<_>>();
    let (structural, structural_diagnostic) = virtual_tile_direct_station_match_with_structure(
        0,
        1,
        &tiles,
        &coverages,
        &initial,
        threshold,
        Some((&structure[0], &structure[1])),
    );
    let structural = structural.expect("structural planes keep the verified relation");
    assert_eq!(
        structural_diagnostic.bidirectional_match_count,
        probe_diagnostic.bidirectional_match_count
    );
    assert!(
        structural_diagnostic
            .median_fitted_orientation_difference_degrees
            .is_some_and(|degrees| degrees <= VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES)
    );
    for point in [Point2::new(100.0, 70.0), Point2::new(335.0, 275.0)] {
        let expected = transformed_point(&probe.homography, point).unwrap();
        let measured = transformed_point(&structural.homography, point).unwrap();
        assert!(
            (expected - measured).norm() <= 1.0,
            "{expected:?} != {measured:?}"
        );
    }

    // The gate reads the structure planes: edges that cross the probe
    // planes' edges reject the same probe matches.
    let (width, height) = tiles[1].alignment_image.dimensions();
    let rows = GrayImage::from_fn(width, height, |_, y| {
        image::Luma([(127.0 + 100.0 * (f64::from(y) * 0.2).sin()).round() as u8])
    });
    let columns = GrayImage::from_fn(width, height, |x, _| {
        image::Luma([(127.0 + 100.0 * (f64::from(x) * 0.2).sin()).round() as u8])
    });
    let (crossed, crossed_diagnostic) = virtual_tile_direct_station_match_with_structure(
        0,
        1,
        &tiles,
        &coverages,
        &initial,
        threshold,
        Some((&rows, &columns)),
    );
    assert!(crossed.is_none());
    assert_eq!(
        crossed_diagnostic.failure_stage,
        "fitted_orientation_mismatch"
    );
    assert_eq!(
        crossed_diagnostic.bidirectional_match_count,
        probe_diagnostic.bidirectional_match_count
    );
    assert!(
        crossed_diagnostic
            .median_fitted_orientation_difference_degrees
            .is_some_and(|degrees| degrees > VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES)
    );
}

fn residual_model_fixture(
    initial: Matrix3<f64>,
    correction: Matrix3<f64>,
    columns: usize,
    rows: usize,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let measured = correction * initial;
    (0..rows)
        .flat_map(|row| {
            (0..columns).map(move |column| {
                let source = Point2::new(70.0 + column as f64 * 65.0, 55.0 + row as f64 * 52.0);
                (source, transformed_point(&measured, source).unwrap())
            })
        })
        .collect()
}

#[test]
fn direct_residual_translation_uses_only_measured_correspondences() {
    let initial = Matrix3::new(
        1.002, 0.004, -96.0, -0.003, 0.998, 8.0, 0.000_006, -0.000_004, 1.0,
    );
    let correction = Matrix3::new(1.0, 0.0, 6.25, 0.0, 1.0, -4.5, 0.0, 0.0, 1.0);
    let points = residual_model_fixture(initial, correction, 6, 6);
    let outcome =
        virtual_tile_fit_measured_residual_model(&points, &initial, (500, 400), (500, 400));
    let maximum_consensus = outcome.max_translation_consensus;
    let fit = outcome.fit.expect("coherent measured residuals must fit");
    assert_eq!(fit.model, "translation");
    assert_eq!(maximum_consensus, points.len());
    assert_eq!(fit.inlier_indices.len(), points.len());
    assert!(fit.spatial_support >= STATION_RELATION_MIN_SPATIAL_SUPPORT);
    for (source, target) in &points {
        let predicted = transformed_point(&fit.measured_homography, *source).unwrap();
        assert!((predicted - target).norm() < 1e-9);
    }

    // Mutation proof: a perfect prior plus only 23 measured pairs still has
    // only 23 pieces of evidence. The prior itself must never become the
    // 24th point or an accepted inlier.
    let prior_only_points = residual_model_fixture(initial, Matrix3::identity(), 23, 1);
    let outcome = virtual_tile_fit_measured_residual_model(
        &prior_only_points,
        &initial,
        (1_600, 400),
        (1_600, 400),
    );
    assert!(outcome.fit.is_none());
    assert_eq!(
        outcome.max_translation_consensus,
        STATION_RELATION_MIN_INLIERS - 1
    );
}

#[test]
fn direct_residual_escalates_to_bounded_affine_only_when_translation_fails() {
    let initial = Matrix3::new(1.0, 0.0, -90.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0);
    let correction = Matrix3::new(1.015, 0.008, 1.5, -0.006, 1.012, -2.0, 0.0, 0.0, 1.0);
    let points = residual_model_fixture(initial, correction, 6, 6);
    let outcome =
        virtual_tile_fit_measured_residual_model(&points, &initial, (500, 400), (500, 400));
    let translation_consensus = outcome.max_translation_consensus;
    let fit = outcome
        .fit
        .expect("bounded spatially coherent affine residual must fit");
    assert!(translation_consensus < STATION_RELATION_MIN_INLIERS);
    assert_eq!(fit.model, "affine");
    assert_eq!(fit.inlier_indices.len(), points.len());
    assert!(virtual_tile_residual_correction_is_bounded(
        &correction,
        &initial,
        virtual_tile_full_tile_bounds((500, 400))
    ));
}

#[test]
fn direct_residual_rejects_incoherent_measurements_and_is_order_deterministic() {
    let initial = Matrix3::new(1.0, 0.0, -80.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0);
    let correction = Matrix3::new(1.0, 0.0, 5.0, 0.0, 1.0, -3.0, 0.0, 0.0, 1.0);
    let points = residual_model_fixture(initial, correction, 6, 6);
    let forward =
        virtual_tile_fit_measured_residual_model(&points, &initial, (500, 400), (500, 400));
    let mut reversed = points.clone();
    reversed.reverse();
    let reverse =
        virtual_tile_fit_measured_residual_model(&reversed, &initial, (500, 400), (500, 400));
    let forward = forward.fit.unwrap();
    let reverse = reverse.fit.unwrap();
    assert_eq!(forward.model, reverse.model);
    assert_eq!(forward.inlier_indices.len(), reverse.inlier_indices.len());
    assert_eq!(
        forward
            .measured_homography
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        reverse
            .measured_homography
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let incoherent = (0..48)
        .map(|index| {
            let source = Point2::new(
                60.0 + f64::from(index % 8) * 52.0,
                45.0 + f64::from(index / 8) * 58.0,
            );
            let predicted = transformed_point(&initial, source).unwrap();
            let dx = f64::from((index * 37) % 31) - 15.0;
            let dy = f64::from((index * 53) % 29) - 14.0;
            (source, predicted + nalgebra::Vector2::new(dx, dy))
        })
        .collect::<Vec<_>>();
    let outcome =
        virtual_tile_fit_measured_residual_model(&incoherent, &initial, (500, 400), (500, 400));
    assert!(outcome.max_translation_consensus < STATION_RELATION_MIN_INLIERS);
    assert!(
        outcome.fit.is_none(),
        "an incorrect prior is not acceptance evidence"
    );
}

#[test]
fn residual_model_selects_a_spatially_broad_mode_from_multimodal_texture() {
    let initial = Matrix3::new(
        1.001, 0.003, -74.0, -0.002, 0.999, 6.0, 0.000_004, -0.000_003, 1.0,
    );
    let coherent_correction = Matrix3::new(1.0, 0.0, 6.5, 0.0, 1.0, -4.25, 0.0, 0.0, 1.0);
    let mut points = residual_model_fixture(initial, coherent_correction, 6, 5);
    // Add stronger-looking but mutually inconsistent repeated-texture modes.
    // None may borrow the prior or combine with the 30-point broad mode.
    points.extend((0..72).map(|index| {
        let source = Point2::new(
            45.0 + f64::from(index % 12) * 36.0,
            38.0 + f64::from(index / 12) * 57.0,
        );
        let predicted = transformed_point(&initial, source).unwrap();
        let mode = index % 6;
        let residual = nalgebra::Vector2::new(
            -14.0 + f64::from(mode) * 5.1,
            13.0 - f64::from((index * 5) % 7) * 4.2,
        );
        (source, predicted + residual)
    }));
    let outcome =
        virtual_tile_fit_measured_residual_model(&points, &initial, (520, 390), (500, 410));
    let fit = outcome
        .fit
        .expect("the spatially broad measured residual mode must be recovered");
    assert_eq!(fit.model, "translation");
    assert!(fit.inlier_indices.len() >= 30);
    assert!(fit.spatial_support >= STATION_RELATION_MIN_SPATIAL_SUPPORT);
    assert_eq!(outcome.translation.rejection_reason, "accepted");
    assert_eq!(outcome.translation.bounded_correction, Some(true));
}

#[test]
fn residual_model_reports_each_rejected_bounded_stage() {
    let initial = Matrix3::new(1.0, 0.0, -80.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0);
    let unbounded = Matrix3::new(1.0, 0.0, 40.0, 0.0, 1.0, -35.0, 0.0, 0.0, 1.0);
    let points = residual_model_fixture(initial, unbounded, 6, 6);
    let outcome =
        virtual_tile_fit_measured_residual_model(&points, &initial, (500, 400), (500, 400));
    assert!(outcome.fit.is_none());
    assert_eq!(outcome.translation.candidate_inliers, points.len());
    assert_eq!(outcome.translation.bounded_correction, Some(false));
    assert_eq!(
        outcome.translation.rejection_reason,
        "correction_out_of_bounds"
    );
    assert!(!outcome.affine.rejection_reason.is_empty());
    assert!(!outcome.projective.rejection_reason.is_empty());
}

#[test]
fn direct_grid_round_trip_mutation_is_rejected() {
    let source = Point2::new(113.25, 84.5);
    assert!(virtual_tile_round_trip_is_consistent(
        source,
        source + nalgebra::Vector2::new(0.6, -0.6)
    ));
    assert!(!virtual_tile_round_trip_is_consistent(
        source,
        source + nalgebra::Vector2::new(1.01, 0.0)
    ));
}

#[test]
fn direct_grid_recovers_repeated_texture_despite_inconsistent_descriptors() {
    let (tiles, coverages, initial, topology) = guided_repeated_texture_fixture();
    assert!(
        virtual_tile_unseeded_station_match(0, 1, &tiles).is_none(),
        "global ratio matching must remain ambiguous on repeated descriptors"
    );
    let outcome = virtual_tile_station_match_outcome(&tiles, &coverages, &initial, &topology);
    let relation = outcome
        .relations
        .get(&(0, 1))
        .expect("guided overlap must fit");
    let diagnostic = &outcome.diagnostics[0];
    assert_eq!(
        diagnostic.evidence_kind,
        stack_report::StationRelationEvidenceKind::VirtualTile
    );
    assert_eq!(diagnostic.failure_stage, "fitted");
    assert!(diagnostic.grid_probe_count >= 24);
    assert!(diagnostic.measurable_patch_count >= 24);
    assert!(diagnostic.bidirectional_match_count >= 24);
    assert!(diagnostic.occupied_grid_cells >= VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS);
    assert!(diagnostic.search_radius_px > 0.0);
    assert_eq!(diagnostic.residual_model, "translation");
    assert_eq!(
        diagnostic.max_residual_translation_consensus,
        diagnostic.residual_translation.candidate_inliers
    );
    assert!(
        diagnostic.max_residual_translation_consensus >= diagnostic.fitted_inliers,
        "the post-model spatial quota may only reduce a measured consensus"
    );
    assert!(diagnostic.residual_dx_px.count >= diagnostic.fitted_inliers);
    assert_eq!(
        diagnostic.residual_dx_px.count,
        diagnostic.residual_dy_px.count
    );
    assert_eq!(
        diagnostic.residual_dx_px.count,
        diagnostic.residual_magnitude_px.count
    );
    assert!(diagnostic.fitted_inliers >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.hull_support >= STATION_RELATION_MIN_SPATIAL_SUPPORT);
    assert!(relation.inliers >= STATION_RELATION_MIN_INLIERS);
    assert!(relation.points.iter().all(|(left, right)| {
        (right.x - (left.x - 200.0)).abs() < 1e-9 && (right.y - left.y).abs() < 1e-9
    }));
    assert!(
        station_relation_spatial_support(
            &relation.points,
            &relation.homography,
            tiles[0].dimensions(),
            tiles[1].dimensions(),
        ) >= STATION_RELATION_MIN_SPATIAL_SUPPORT
    );
}

#[test]
fn direct_grid_ignores_descriptor_matches_concentrated_in_repeated_stroke_cells() {
    let (mut tiles, coverages, initial, topology) = guided_repeated_texture_fixture();
    let descriptor = [0u8; BRIEF_DESCRIPTOR_SIZE / 8];
    tiles[0].features = (0..30)
        .map(|index| Feature {
            keypoint: KeyPoint {
                x: 240 + index * 14,
                y: 270 + index % 3,
            },
            descriptor,
            support_scale: 1.0,
        })
        .collect();
    tiles[1].features = tiles[0]
        .features
        .iter()
        .map(|feature| Feature {
            keypoint: KeyPoint {
                x: feature.keypoint.x - 200,
                y: feature.keypoint.y,
            },
            descriptor,
            support_scale: 1.0,
        })
        .collect();

    let outcome = virtual_tile_station_match_outcome(&tiles, &coverages, &initial, &topology);
    let relation = outcome
        .relations
        .get(&(0, 1))
        .expect("direct grid pixels must not depend on descriptor concentration");
    let diagnostic = &outcome.diagnostics[0];
    assert!(diagnostic.grid_probe_count > tiles[0].features.len());
    assert!(diagnostic.measurable_patch_count >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.bidirectional_match_count >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.occupied_grid_cells >= VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS);
    assert!(relation.inliers >= STATION_RELATION_MIN_INLIERS);
}

#[test]
fn guided_overlap_rejects_wrong_prior_and_wrong_content() {
    let (tiles, coverages, mut initial, topology) = guided_repeated_texture_fixture();
    initial.insert(
        tiles[1].id,
        Matrix3::new(1.0, 0.0, 260.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    );
    assert!(
        virtual_tile_station_matches(&tiles, &coverages, &initial, &topology).is_empty(),
        "a prior outside the bounded structural search must not become evidence"
    );

    let (mut tiles, coverages, initial, topology) = guided_repeated_texture_fixture();
    tiles[1].alignment_image =
        GrayImage::from_pixel(tiles[1].width, tiles[1].height, image::Luma([127]));
    assert!(
        virtual_tile_station_matches(&tiles, &coverages, &initial, &topology).is_empty(),
        "descriptor identity without native patch agreement must reject"
    );
}

#[test]
fn direct_grid_rejects_locally_wrong_45_degree_orientation() {
    let (mut tiles, coverages, initial, topology) = guided_repeated_texture_fixture();
    let (width, height) = tiles[0].dimensions();
    tiles[0].alignment_image = GrayImage::from_fn(width, height, |x, _| {
        image::Luma([(40.0 + f64::from(x % 100) * 2.0).round() as u8])
    });
    tiles[1].alignment_image = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([(40.0 + f64::from((x + y) % 100) * std::f64::consts::SQRT_2).round() as u8])
    });
    let left_to_right = Matrix3::new(1.0, 0.0, -200.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let structure = virtual_tile_structural_score(
        &tiles[0].alignment_image,
        &tiles[1].alignment_image,
        &coverages[0],
        &coverages[1],
        tiles[0].features[1].keypoint,
        tiles[1].features[1].keypoint,
        &left_to_right,
    )
    .expect("the two ramp patches are measurable");
    assert!(structure.patch_ncc >= VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC);
    assert!(structure.gradient_ncc >= VIRTUAL_TILE_GUIDED_MIN_GRADIENT_NCC);
    assert!(structure.orientation_difference_degrees > 35.0);

    let outcome = virtual_tile_station_match_outcome(&tiles, &coverages, &initial, &topology);
    assert!(outcome.relations.is_empty());
    let diagnostic = &outcome.diagnostics[0];
    assert_ne!(diagnostic.failure_stage, "fitted");
    // A one-dimensional ramp can be rejected earlier as an ambiguous patch
    // peak. If enough unique peaks survive to produce a model, the retained
    // fitted-Jacobian orientation gate must be the rejecting stage.
    if diagnostic.fitted_inliers >= STATION_RELATION_MIN_INLIERS {
        assert_eq!(diagnostic.failure_stage, "fitted_orientation_mismatch");
        assert!(
            diagnostic
                .median_fitted_orientation_difference_degrees
                .is_some_and(|difference| difference > 35.0)
        );
    }
}

#[test]
fn direct_grid_flat_overlap_reports_unmeasurable_patches() {
    let (mut tiles, coverages, initial, topology) = guided_repeated_texture_fixture();
    for tile in &mut tiles {
        tile.alignment_image = GrayImage::from_pixel(tile.width, tile.height, image::Luma([127]));
    }
    let outcome = virtual_tile_station_match_outcome(&tiles, &coverages, &initial, &topology);
    assert!(outcome.relations.is_empty());
    assert_eq!(outcome.diagnostics.len(), 1);
    let diagnostic = &outcome.diagnostics[0];
    assert_eq!(
        diagnostic.evidence_kind,
        stack_report::StationRelationEvidenceKind::VirtualTile
    );
    assert!(diagnostic.grid_probe_count >= STATION_RELATION_MIN_INLIERS);
    assert_eq!(diagnostic.measurable_patch_count, 0);
    assert_eq!(diagnostic.bidirectional_match_count, 0);
    assert_eq!(diagnostic.fitted_inliers, 0);
    assert_eq!(diagnostic.failure_stage, "insufficient_measurable_patches");
}

#[test]
fn guided_overlap_ignores_uncovered_payload_and_import_order() {
    let (tiles, coverages, initial, topology) = guided_repeated_texture_fixture();
    let original = virtual_tile_station_matches(&tiles, &coverages, &initial, &topology);
    let mut mutated = tiles;
    let coverage = &coverages[1];
    for (x, y, pixel) in mutated[1].alignment_image.enumerate_pixels_mut() {
        if !coverage.is_covered(x, y) {
            pixel[0] = ((x.wrapping_mul(31) + y.wrapping_mul(17)) % 251) as u8;
        }
    }
    let repeated = virtual_tile_station_matches(&mutated, &coverages, &initial, &topology);
    let original_relation = original.get(&(0, 1)).unwrap();
    let repeated_relation = repeated.get(&(0, 1)).unwrap();
    assert_eq!(original_relation.points, repeated_relation.points);
    assert_eq!(
        original_relation
            .homography
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        repeated_relation
            .homography
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let (mut swapped_tiles, mut swapped_coverages, swapped_initial, swapped_topology) =
        guided_repeated_texture_fixture();
    swapped_tiles.swap(0, 1);
    swapped_coverages.swap(0, 1);
    let swapped = virtual_tile_station_matches(
        &swapped_tiles,
        &swapped_coverages,
        &swapped_initial,
        &swapped_topology,
    );
    let reverse = swapped.get(&(0, 1)).unwrap();
    let reverse_inverse = reverse.homography.try_inverse().unwrap();
    for (left, right) in original_relation
        .homography
        .iter()
        .zip(reverse_inverse.iter())
    {
        assert!((left - right).abs() < 1e-8);
    }
}

#[test]
fn virtual_tile_no_fit_candidate_reports_unmeasured_without_inventing_reasons() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let tiles = synthetic_virtual_tiles(2);
    let initial = tiles
        .iter()
        .map(|tile| (tile.id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();
    let mut closure = stack_report::ClosureReport::default();
    let mut report = stack_report::StationRelationsReport::default();
    let solved = solve_virtual_tile_station_poses_with_report(
        &tiles,
        fully_covered_solver_coverages(&tiles, &fully_covered_quality_coverages(&tiles)),
        &HashMap::new(),
        &initial,
        &synthetic_virtual_tile_topology(2),
        &mut closure,
        &mut report,
    );
    assert!(solved.is_empty());
    assert_eq!(report.accepted.len(), 0);
    assert_eq!(report.rejected_count, 1);
    assert!(report.rejected_by_reason.is_empty());
    let (_, diagnostic) = virtual_tile_direct_station_match(
        0,
        1,
        &[],
        &[],
        &HashMap::new(),
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    );
    report.candidates = vec![diagnostic];
    report.summarize_candidate_measurements();
    assert_eq!(report.unmeasured_count, 1);
    assert_eq!(
        report.rejected_by_failure_stage,
        BTreeMap::from([("missing_input".to_string(), 1)])
    );
    assert!(report.rejected_by_reason.is_empty());
}

#[test]
fn default_two_stage_flow_solves_ten_sources_as_three_fused_stations() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let images = (0..10)
        .map(|index| focus_textured_test_image(index, &format!("source-{index}.raw")))
        .collect::<Vec<_>>();
    let mut source_matches = HashMap::new();
    for &(start, end) in &[(0usize, 4usize), (4, 7), (7, 10)] {
        for index in start..end - 1 {
            source_matches.insert((index, index + 1), observed_translation_match(0.0, 0.0, 36));
        }
    }
    // Each station boundary has only one Source_RAW observation.  The old
    // early authoritative solve discarded both as support=1 and rejected
    // the three components before the fused station tiles existed.
    source_matches.insert((3, 4), observed_translation_match(120.0, 0.0, 36));
    source_matches.insert((6, 7), observed_translation_match(120.0, 0.0, 36));
    let mut early_closure = stack_report::ClosureReport::default();
    let mut early_relations = stack_report::StationRelationsReport::default();
    let (order, local_geometry) = build_focus_stack_stitching_order_with_report(
        &images,
        &source_matches,
        &mut early_closure,
        &mut early_relations,
        false,
    );
    assert_eq!(order.len(), 10);
    assert_eq!(local_geometry.len(), 10);
    assert!(early_relations.accepted.is_empty());
    assert_eq!(early_relations.connectivity.components, 0);
    assert_eq!(early_closure.status, stack_report::ClosureStatus::NotRun);

    let tiles = synthetic_virtual_tiles(3);
    let tile_matches = HashMap::from([
        ((0, 1), observed_translation_match(0.0, 0.0, 36)),
        ((1, 2), observed_translation_match(0.0, 0.0, 36)),
    ]);
    let initial = tiles
        .iter()
        .map(|tile| (tile.id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();
    let mut final_closure = stack_report::ClosureReport::default();
    let mut final_relations = stack_report::StationRelationsReport::default();
    let solved = solve_virtual_tile_station_poses_with_report(
        &tiles,
        fully_covered_solver_coverages(&tiles, &fully_covered_quality_coverages(&tiles)),
        &tile_matches,
        &initial,
        &synthetic_virtual_tile_topology(3),
        &mut final_closure,
        &mut final_relations,
    );

    assert_eq!(solved.len(), 3);
    assert_eq!(final_relations.connectivity.components, 1);
    assert_eq!(final_relations.accepted.len(), 2);
    assert!(final_relations.discarded_single_layer_evidence.is_empty());
    assert!(final_relations.accepted.iter().all(|relation| {
        relation.evidence_kind == stack_report::StationRelationEvidenceKind::VirtualTile
            && relation.independent_support == 1
    }));
    assert_eq!(
        final_closure.status,
        stack_report::ClosureStatus::NoConstraints
    );
}

#[test]
fn true_virtual_tile_disconnection_rejects_before_composition() {
    let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
    let tiles = synthetic_virtual_tiles(3);
    let tile_matches = HashMap::from([((0, 1), observed_translation_match(0.0, 0.0, 36))]);
    let initial = tiles
        .iter()
        .map(|tile| (tile.id, Matrix3::identity()))
        .collect::<HashMap<_, _>>();
    let mut closure = stack_report::ClosureReport::default();
    let mut relations = stack_report::StationRelationsReport::default();
    let solved = solve_virtual_tile_station_poses_with_report(
        &tiles,
        fully_covered_solver_coverages(&tiles, &fully_covered_quality_coverages(&tiles)),
        &tile_matches,
        &initial,
        &synthetic_virtual_tile_topology(3),
        &mut closure,
        &mut relations,
    );

    assert!(solved.is_empty());
    assert_eq!(relations.connectivity.components, 2);
    assert_eq!(relations.connectivity.component_member_counts, vec![2, 1]);
}

#[test]
fn single_virtual_tile_skips_station_relations_and_closure() {
    let tiles = synthetic_virtual_tiles(1);
    let initial = HashMap::from([(tiles[0].id, Matrix3::identity())]);
    let mut closure = stack_report::ClosureReport::default();
    let mut relations = stack_report::StationRelationsReport::default();
    let solved = solve_virtual_tile_station_poses_with_report(
        &tiles,
        fully_covered_solver_coverages(&tiles, &fully_covered_quality_coverages(&tiles)),
        &HashMap::new(),
        &initial,
        &synthetic_virtual_tile_topology(1),
        &mut closure,
        &mut relations,
    );

    assert_eq!(solved, initial);
    assert!(relations.accepted.is_empty());
    assert_eq!(relations.connectivity.components, 0);
    assert_eq!(closure.status, stack_report::ClosureStatus::NotRun);
}

#[test]
fn station_source_evidence_requires_two_independent_layers() {
    let discarded = single_layer_station_evidence_discard(1, 3, 8)
        .expect("one Source_RAW without consensus must be discarded");
    assert_eq!(discarded["count"], serde_json::json!(1));
    assert_eq!(discarded["stations"], serde_json::json!([3, 8]));
    assert_eq!(discarded["independent_support"], serde_json::json!(1));
    assert!(
        single_layer_station_evidence_discard(2, 3, 8).is_none(),
        "two independent focus layers are valid supplemental evidence"
    );
}

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
}

#[test]
fn focus_component_bridge_candidates_keep_both_sequence_boundaries() {
    let images = (0..12)
        .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
        .collect::<Vec<_>>();
    let component = (0..images.len()).collect::<Vec<_>>();

    let candidates = focus_component_bridge_candidates(&component, &images);

    assert!(candidates.contains(&0));
    assert!(candidates.contains(&1));
    assert!(candidates.contains(&10));
    assert!(candidates.contains(&11));
}

#[test]
fn focus_stack_motion_detects_a_scan_without_reclassifying_a_fixed_stack() {
    let points = (0..3)
        .flat_map(|row| {
            (0..3).map(move |column| {
                let source = Point2::new(
                    1_000.0 + column as f64 * 2_000.0,
                    900.0 + row as f64 * 2_000.0,
                );
                (source, source + nalgebra::Vector2::new(4.0, 2.0))
            })
        })
        .collect::<Vec<_>>();
    assert!(!focus_stack_motion_is_shifted_mosaic(
        &points,
        (9_504, 6_336)
    ));

    let shifted = points
        .iter()
        .map(|(source, target)| (*source, *target + nalgebra::Vector2::new(300.0, 0.0)))
        .collect::<Vec<_>>();
    assert!(focus_stack_motion_is_shifted_mosaic(
        &shifted,
        (9_504, 6_336)
    ));

    let scale_only = points
        .iter()
        .map(|(source, _)| {
            let centered = *source - Point2::new(4_752.0, 3_168.0);
            (*source, Point2::new(4_752.0, 3_168.0) + centered * 1.24)
        })
        .collect::<Vec<_>>();
    assert!(focus_stack_motion_is_shifted_mosaic(
        &scale_only,
        (9_504, 6_336)
    ));
}

#[test]
fn large_stack_alignment_budget_tightens_as_source_count_grows() {
    assert_eq!(scalable_alignment_budget(64), (2_400, 1_600));
    assert_eq!(scalable_alignment_budget(128), (1_800, 1_100));
    assert_eq!(scalable_alignment_budget(200), (1_536, 800));
}

#[test]
fn focus_stack_cache_key_includes_proxy_contract_and_input_budget() {
    let one = focus_stack_cache_pipeline_version(1);
    let two = focus_stack_cache_pipeline_version(2);
    assert!(one.contains("proxy_version=proxy-v1"));
    assert!(one.contains("resize_filter=triangle"));
    assert_ne!(one, two);
}

#[test]
fn mixed_focal_detection_distinguishes_lens_changes_from_rounding_noise() {
    assert!(focal_lengths_span_multiple_lenses([35.0, 85.0]));
    assert!(!focal_lengths_span_multiple_lenses([34.0, 35.0, 36.0]));
    assert!(!focal_lengths_span_multiple_lenses([f64::NAN, -1.0, 85.0]));
}

#[test]
fn scalable_preparation_workers_are_bounded_by_images_cpu_and_memory() {
    let abundant_memory = 32 * PREPARATION_RAM_PER_WORKER_BYTES;

    assert_eq!(
        bounded_preparation_worker_count(185, 10, abundant_memory),
        MAX_SCALABLE_PREPARATION_WORKERS
    );
    assert_eq!(bounded_preparation_worker_count(185, 4, abundant_memory), 4);
    assert_eq!(
        bounded_preparation_worker_count(185, 10, 2 * PREPARATION_RAM_PER_WORKER_BYTES),
        2
    );
    assert_eq!(bounded_preparation_worker_count(3, 10, abundant_memory), 3);
    assert_eq!(bounded_preparation_worker_count(185, 10, 0), 1);
}

#[test]
fn oversized_panorama_canvas_is_scaled_to_the_memory_budget() {
    assert_eq!(memory_safe_panorama_render_scale(12_000, 8_000), 1.0);
    let scale = memory_safe_panorama_render_scale(248_296, 9_843);
    let scaled_pixels = (248_296.0 * scale).ceil() as u64 * (9_843.0 * scale).ceil() as u64;

    assert!(scale < 0.32);
    assert!(scaled_pixels <= MAX_IN_MEMORY_PANORAMA_PIXELS + 100_000);
}

#[test]
fn render_scale_is_applied_after_the_global_image_transform() {
    let source = HashMap::from([(
        7,
        Matrix3::new(1.0, 0.0, 400.0, 0.0, 1.0, -80.0, 0.0, 0.0, 1.0),
    )]);
    let scaled = scaled_homographies(&source, 0.25);
    let transform = scaled.get(&7).expect("scaled transform should exist");

    assert_eq!(transform[(0, 0)], 0.25);
    assert_eq!(transform[(1, 1)], 0.25);
    assert_eq!(transform[(0, 2)], 100.0);
    assert_eq!(transform[(1, 2)], -20.0);
}

#[test]
fn scaled_render_sources_preserve_the_output_coordinate_system() {
    let source = HashMap::from([(
        7,
        Matrix3::new(1.0, 0.0, 400.0, 0.0, 1.0, -80.0, 0.0, 0.0, 1.0),
    )]);
    let scale = 0.25;
    let scaled = scaled_source_render_homographies(&source, scale);
    let source_point = nalgebra::Point3::new(800.0, 200.0, 1.0);
    let expected =
        Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0) * source[&7] * source_point;
    let actual =
        scaled[&7] * nalgebra::Point3::new(source_point.x * scale, source_point.y * scale, 1.0);

    assert!((actual.x - expected.x).abs() < 1e-9);
    assert!((actual.y - expected.y).abs() < 1e-9);
    assert!((actual.z - expected.z).abs() < 1e-9);

    let mut image = test_image(7, "large.jpg");
    image.width = 4_672;
    image.height = 7_008;
    let rendered = scaled_render_image_info(&image, scale);
    assert_eq!(rendered.dimensions(), (1_168, 1_752));
    assert!(rendered.alignment_image.is_empty());
    assert!(rendered.features.is_empty());
}

#[test]
fn parallel_area_resize_preserves_constant_rgb8_sources() {
    let source = RgbImage::from_pixel(7, 5, image::Rgb([51, 102, 204]));
    let resized = resize_rgb8_area_to_rgb32f(&source, 3, 2);

    assert_eq!(resized.dimensions(), (3, 2));
    for pixel in resized.pixels() {
        assert!((pixel[0] - 0.2).abs() < 1e-6);
        assert!((pixel[1] - 0.4).abs() < 1e-6);
        assert!((pixel[2] - 0.8).abs() < 1e-6);
    }
}

#[test]
fn panorama_refinement_rejects_multi_pixel_correspondence_errors() {
    let expected = Matrix3::new(1.0, 0.0, 4_273.25, 0.0, 1.0, 71.75, 0.0, 0.0, 1.0);
    let mut points = Vec::new();
    for row in 0..4 {
        for column in 0..6 {
            let source = Point2::new(
                480.0 + column as f64 * 1_350.0,
                420.0 + row as f64 * 1_420.0,
            );
            let mut target = transformed_point(&expected, source).unwrap();
            target.x += ((row * 7 + column * 5) as f64).sin() * 0.18;
            target.y += ((row * 3 + column * 11) as f64).cos() * 0.18;
            points.push((source, target));
        }
    }
    for index in 0..8 {
        let source = Point2::new(700.0 + index as f64 * 780.0, 850.0 + index as f64 * 510.0);
        let mut target = transformed_point(&expected, source).unwrap();
        target.x += 5.5;
        target.y -= 4.5;
        points.push((source, target));
    }

    let refined = refine_homography_inliers(
        &mut points,
        FULL_RES_REFINEMENT_THRESHOLD,
        processing::MIN_INLIERS_FOR_CONNECTION,
    )
    .expect("the accurate correspondence grid should remain connected");

    assert_eq!(points.len(), 24);
    assert!(symmetric_reprojection_rmse(&refined, &points) < 0.35);
}

#[test]
fn allocation_free_patch_ncc_preserves_correlation_range() {
    let source = GrayImage::from_fn(9, 9, |x, y| {
        image::Luma([((x * 17 + y * 29 + x * y * 3) % 255) as u8])
    });
    let inverted = GrayImage::from_fn(9, 9, |x, y| image::Luma([255 - source.get_pixel(x, y)[0]]));
    let source_plane = LumaPlane::Gray(&source);
    let inverted_plane = LumaPlane::Gray(&inverted);

    assert!(patch_ncc(&source_plane, &source_plane, 4, 4, 4, 4, 3) > 0.999_999);
    assert!(patch_ncc(&source_plane, &inverted_plane, 4, 4, 4, 4, 3) < -0.999_999);
}

#[test]
fn stitching_order_uses_stable_filenames_for_equivalent_graphs() {
    let first_images = vec![
        test_image(0, "c.jpg"),
        test_image(1, "a.jpg"),
        test_image(2, "b.jpg"),
    ];
    let first_matches = HashMap::from([((1, 2), identity_match(30)), ((0, 2), identity_match(20))]);
    let (first_order, _) = build_stitching_order(&first_images, &first_matches);

    let second_images = vec![
        test_image(0, "b.jpg"),
        test_image(1, "c.jpg"),
        test_image(2, "a.jpg"),
    ];
    let second_matches =
        HashMap::from([((0, 2), identity_match(30)), ((0, 1), identity_match(20))]);
    let (second_order, _) = build_stitching_order(&second_images, &second_matches);

    let first_names = first_order
        .iter()
        .map(|&index| first_images[index].filename.as_str())
        .collect::<Vec<_>>();
    let second_names = second_order
        .iter()
        .map(|&index| second_images[index].filename.as_str())
        .collect::<Vec<_>>();

    assert_eq!(first_names, vec!["a.jpg", "b.jpg", "c.jpg"]);
    assert_eq!(second_names, first_names);
}

#[test]
fn disconnected_stitch_graph_selects_the_largest_component() {
    let images = (0..7)
        .map(|index| test_image(index, &format!("tile-{index}.jpg")))
        .collect::<Vec<_>>();
    let matches = HashMap::from([
        ((0, 1), identity_match(80)),
        // The four-image component must win even though every one of its
        // edges has fewer inliers than the isolated two-image component.
        ((2, 3), identity_match(30)),
        ((3, 4), identity_match(30)),
        ((4, 5), identity_match(30)),
    ]);

    let (order, homographies) = build_stitching_order(&images, &matches);

    assert_eq!(order.len(), 4);
    assert_eq!(
        order.iter().copied().collect::<HashSet<_>>(),
        HashSet::from([2, 3, 4, 5])
    );
    assert_eq!(homographies.len(), 4);
}

#[test]
fn robust_affine_fit_rejects_false_correspondences() {
    let expected = Matrix3::new(1.012, -0.008, 420.0, 0.006, 0.994, -730.0, 0.0, 0.0, 1.0);
    let mut points = Vec::new();
    for row in 0..5 {
        for column in 0..6 {
            let source = Point2::new(
                300.0 + column as f64 * 1_420.0,
                240.0 + row as f64 * 1_080.0,
            );
            let mut target = transformed_point(&expected, source).unwrap();
            target.x += ((row * 7 + column * 3) as f64).sin() * 0.18;
            target.y += ((row * 5 + column * 11) as f64).cos() * 0.18;
            points.push((source, target));
        }
    }
    for index in 0..12 {
        points.push((
            Point2::new(index as f64 * 613.0, index as f64 * 277.0),
            Point2::new(8_000.0 - index as f64 * 193.0, 900.0 + index as f64 * 421.0),
        ));
    }

    let fit = robust_transform_fit(&points, 3, 42, estimate_affine)
        .expect("the inlier grid should produce an affine consensus");
    assert!(fit.inlier_indices.len() >= 29);
    assert!(fit.median_error < 0.5);
    let probe = Point2::new(4_500.0, 2_900.0);
    let expected_probe = transformed_point(&expected, probe).unwrap();
    let actual_probe = transformed_point(&fit.transform, probe).unwrap();
    assert!((actual_probe - expected_probe).norm() < 0.5);
}

#[test]
fn large_panorama_auto_alignment_preserves_real_focal_scale() {
    let expected = Matrix3::new(2.35, -0.08, 410.0, 0.08, 2.35, -270.0, 0.0, 0.0, 1.0);
    let points = (0..5)
        .flat_map(|row| {
            (0..6).map(move |column| {
                let source = Point2::new(100.0 + column as f64 * 140.0, 80.0 + row as f64 * 120.0);
                (source, transformed_point(&expected, source).unwrap())
            })
        })
        .collect::<Vec<_>>();
    let projective = processing::compute_homography(&points).unwrap();

    let selected = select_large_panorama_transform(&projective, &points, (1_000, 800), false);
    let (scale, _, _) = panorama_model_linear_characteristics(&selected);

    assert!((scale - 2.351).abs() < 0.02, "scale={scale}");
    assert!(median_symmetric_error(&selected, &points) < 0.05);
}

#[test]
fn large_panorama_auto_alignment_keeps_true_translation_stable() {
    let expected = Matrix3::new(1.0, 0.0, 330.0, 0.0, 1.0, -140.0, 0.0, 0.0, 1.0);
    let points = (0..5)
        .flat_map(|row| {
            (0..6).map(move |column| {
                let source = Point2::new(100.0 + column as f64 * 140.0, 80.0 + row as f64 * 120.0);
                (source, transformed_point(&expected, source).unwrap())
            })
        })
        .collect::<Vec<_>>();
    let projective = processing::compute_homography(&points).unwrap();

    let selected = select_large_panorama_transform(&projective, &points, (1_000, 800), false);

    assert_eq!(selected, expected);
}

#[test]
fn focus_alignment_accepts_a_real_lens_scale_change() {
    let expected = Matrix3::new(2.35, 0.0, -640.0, 0.0, 2.35, -510.0, 0.0, 0.0, 1.0);
    let points = (0..5)
        .flat_map(|row| {
            (0..6).map(move |column| {
                let source = Point2::new(100.0 + column as f64 * 140.0, 80.0 + row as f64 * 120.0);
                (source, transformed_point(&expected, source).unwrap())
            })
        })
        .collect::<Vec<_>>();
    let projective = processing::compute_homography(&points).unwrap();

    let selected =
        select_focus_stack_transform(&projective, &points, (1_000, 800), AlignmentMode::Auto);
    let (scale, _, _) = panorama_model_linear_characteristics(&selected);

    assert!((scale - 2.35).abs() < 0.03, "scale={scale}");
}

#[test]
fn focus_alignment_auto_accepts_broad_handheld_perspective() {
    // A small planar tilt is common when a focus stack is shot by hand.
    // The projective residual is real even though the camera centre does
    // not move enough to classify the stack as a shifted mosaic.
    let expected = Matrix3::new(
        1.0, 0.002, 3.0, -0.001, 1.0, -2.0, 0.000_018, -0.000_012, 1.0,
    );
    let points = (0..6)
        .flat_map(|row| {
            (0..7).map(move |column| {
                let source = Point2::new(70.0 + column as f64 * 140.0, 60.0 + row as f64 * 110.0);
                (source, transformed_point(&expected, source).unwrap())
            })
        })
        .collect::<Vec<_>>();
    let projective = processing::compute_homography(&points).unwrap();
    let selected =
        select_focus_stack_transform(&projective, &points, (1_000, 800), AlignmentMode::Auto);
    let error = median_symmetric_error(&selected, &points);
    assert!(error < 0.05, "selected model error={error}");
    assert!(selected[(2, 0)].abs() > 1e-6 || selected[(2, 1)].abs() > 1e-6);
}

#[test]
fn focus_alignment_auto_rejects_compact_projective_stroke_support() {
    // Four or five keypoints on one repeated calligraphic stroke can fit a
    // projective model by accident.  The broad support gate must keep that
    // local warp from bending the whole focus layer.
    let expected = Matrix3::new(1.0, 0.003, 0.0, -0.002, 1.0, 0.0, 0.000_12, -0.000_09, 1.0);
    let points = (0..3)
        .flat_map(|row| {
            (0..4).map(move |column| {
                let source = Point2::new(470.0 + column as f64 * 18.0, 370.0 + row as f64 * 16.0);
                (source, transformed_point(&expected, source).unwrap())
            })
        })
        .collect::<Vec<_>>();
    let projective = processing::compute_homography(&points).unwrap();
    let selected =
        select_focus_stack_transform(&projective, &points, (1_000, 800), AlignmentMode::Auto);
    assert!(
        selected[(2, 0)].abs() < 1e-8 && selected[(2, 1)].abs() < 1e-8,
        "compact support unexpectedly selected projective model: {:?}",
        selected
    );
}

#[test]
fn mixed_focal_registration_rejects_a_repeated_texture_bridge() {
    let identity = Matrix3::identity();
    assert!(!mixed_focal_scale_is_plausible(&identity, 35.0, 85.0));
    let expected = Matrix3::new(
        85.0 / 35.0,
        0.0,
        -640.0,
        0.0,
        85.0 / 35.0,
        -510.0,
        0.0,
        0.0,
        1.0,
    );
    assert!(mixed_focal_scale_is_plausible(&expected, 35.0, 85.0));
}

#[test]
fn focus_alignment_does_not_trade_fit_precision_for_a_few_more_inliers() {
    let selected = RobustTransformFit {
        transform: Matrix3::identity(),
        inlier_indices: (0..22).collect(),
        median_error: 2.0,
    };
    let overfit = RobustTransformFit {
        transform: Matrix3::identity(),
        inlier_indices: (0..28).collect(),
        median_error: 2.45,
    };
    let precise = RobustTransformFit {
        transform: Matrix3::identity(),
        inlier_indices: (0..18).collect(),
        median_error: 1.6,
    };

    assert!(!focus_fit_is_competitive(&overfit, &selected));
    assert!(focus_fit_is_competitive(&precise, &selected));
}

#[test]
fn focus_stack_stability_rejects_extrapolated_projective_warp() {
    let stable = Matrix3::new(1.01, -0.01, 180.0, 0.01, 1.01, -90.0, 0.0, 0.0, 1.0);
    let unstable = Matrix3::new(1.0, 0.0, 180.0, 0.0, 1.0, -90.0, 0.000_18, -0.000_12, 1.0);
    assert!(transform_is_stable_for_focus_stack(&stable, (9_504, 6_336)));
    assert!(!transform_is_stable_for_focus_stack(
        &unstable,
        (9_504, 6_336)
    ));
}

#[test]
fn focus_registration_rejects_mirrored_image_orientation() {
    let identity = Matrix3::identity();
    let mirrored = Matrix3::new(-1.0, 0.0, 9_504.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

    assert!(homography_preserves_focus_orientation(
        &identity,
        (9_504, 6_336)
    ));
    assert!(!homography_preserves_focus_orientation(
        &mirrored,
        (9_504, 6_336)
    ));
}

#[test]
fn generic_focus_registration_samples_frames_without_a_detected_depth_layer() {
    let source = GrayImage::from_fn(320, 240, |x, y| {
        image::Luma([((x * 17 + y * 29 + x * y * 3) % 255) as u8])
    });
    let target = GrayImage::from_fn(320, 240, |x, y| {
        image::Luma([source.get_pixel(x.saturating_sub(5), y)[0]])
    });
    let mut source_info = test_image(0, "source.jpg");
    source_info.width = 320;
    source_info.height = 240;
    source_info.alignment_image = source;
    let mut target_info = test_image(1, "target.jpg");
    target_info.width = 320;
    target_info.height = 240;
    target_info.alignment_image = target;

    let points = collect_dense_focus_region_points(
        &source_info,
        &target_info,
        &Matrix3::new(1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        Projection::Planar,
    );

    assert!(source_info.foreground_range.is_none());
    assert!(target_info.foreground_range.is_none());
    assert!(points.len() >= 8);
}

#[test]
fn long_edge_consensus_groups_only_distinct_images() {
    let lines = vec![
        FocusHorizontalEdgeLine {
            image_id: 0,
            source_row: 0.20,
            world_x_center: 500.0,
            slope: 0.01,
            intercept: 995.0,
            median_error: 2.0,
        },
        FocusHorizontalEdgeLine {
            image_id: 1,
            source_row: 0.18,
            world_x_center: 700.0,
            slope: 0.011,
            intercept: 994.0,
            median_error: 2.0,
        },
        FocusHorizontalEdgeLine {
            image_id: 2,
            source_row: 0.22,
            world_x_center: 900.0,
            slope: 0.009,
            intercept: 996.0,
            median_error: 2.0,
        },
        FocusHorizontalEdgeLine {
            image_id: 0,
            source_row: 0.40,
            world_x_center: 500.0,
            slope: 0.01,
            intercept: 2_000.0,
            median_error: 2.0,
        },
    ];

    let clusters = focus_horizontal_edge_line_clusters(&lines, 9_504.0);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 3);
}

#[test]
fn long_edge_duplicate_candidates_keep_the_best_fit_per_image() {
    let lines = vec![
        FocusHorizontalEdgeLine {
            image_id: 0,
            source_row: 0.20,
            world_x_center: 500.0,
            slope: 0.01,
            intercept: 995.0,
            median_error: 4.0,
        },
        FocusHorizontalEdgeLine {
            image_id: 0,
            source_row: 0.201,
            world_x_center: 500.0,
            slope: 0.0105,
            intercept: 995.2,
            median_error: 1.5,
        },
        FocusHorizontalEdgeLine {
            image_id: 0,
            source_row: 0.31,
            world_x_center: 500.0,
            slope: 0.01,
            intercept: 1_500.0,
            median_error: 2.0,
        },
    ];

    let deduplicated = deduplicate_focus_horizontal_edge_lines(lines, 9_504.0);
    assert_eq!(deduplicated.len(), 2);
    assert!(
        deduplicated
            .iter()
            .any(|line| (line.median_error - 1.5).abs() < f64::EPSILON)
    );
    assert!(
        deduplicated
            .iter()
            .any(|line| (line.source_row - 0.31).abs() < f64::EPSILON)
    );
}

#[test]
fn vertical_long_edge_consensus_groups_only_distinct_images() {
    let lines = vec![
        FocusVerticalEdgeLine {
            image_id: 0,
            source_column: 0.20,
            world_y_center: 500.0,
            slope: 0.01,
            intercept: 995.0,
            median_error: 2.0,
        },
        FocusVerticalEdgeLine {
            image_id: 1,
            source_column: 0.18,
            world_y_center: 700.0,
            slope: 0.011,
            intercept: 994.0,
            median_error: 2.0,
        },
        FocusVerticalEdgeLine {
            image_id: 2,
            source_column: 0.22,
            world_y_center: 900.0,
            slope: 0.009,
            intercept: 996.0,
            median_error: 2.0,
        },
        FocusVerticalEdgeLine {
            image_id: 0,
            source_column: 0.40,
            world_y_center: 500.0,
            slope: 0.01,
            intercept: 2_000.0,
            median_error: 2.0,
        },
    ];

    let clusters = focus_vertical_edge_line_clusters(&lines, 9_504.0);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 3);
}

#[test]
fn vertical_edge_duplicate_candidates_keep_the_best_fit_per_image() {
    let lines = vec![
        FocusVerticalEdgeLine {
            image_id: 0,
            source_column: 0.20,
            world_y_center: 500.0,
            slope: 0.01,
            intercept: 995.0,
            median_error: 4.0,
        },
        FocusVerticalEdgeLine {
            image_id: 0,
            source_column: 0.201,
            world_y_center: 500.0,
            slope: 0.0105,
            intercept: 995.2,
            median_error: 1.5,
        },
        FocusVerticalEdgeLine {
            image_id: 0,
            source_column: 0.31,
            world_y_center: 500.0,
            slope: 0.01,
            intercept: 1_500.0,
            median_error: 2.0,
        },
    ];

    let deduplicated = deduplicate_focus_vertical_edge_lines(lines, 9_504.0);
    assert_eq!(deduplicated.len(), 2);
    assert!(
        deduplicated
            .iter()
            .any(|line| (line.median_error - 1.5).abs() < f64::EPSILON)
    );
    assert!(
        deduplicated
            .iter()
            .any(|line| (line.source_column - 0.31).abs() < f64::EPSILON)
    );
}

#[test]
fn generated_stack_outputs_and_macos_sidecars_are_not_reused_as_sources() {
    assert!(is_generated_stitch_output("/tmp/DSC08897_FocusStack1.jpg"));
    assert!(is_generated_stitch_output(
        "/tmp/scan_Panorama_20260905.jpg"
    ));
    assert!(is_generated_stitch_output("/tmp/scan_Pano.png"));
    assert!(!is_generated_stitch_output("/tmp/DSC08897.jpg"));
    assert!(!is_generated_stitch_output("/tmp/focusstack_reference.jpg"));
    assert!(is_auxiliary_stitch_file("/tmp/._DSC08897.jpg"));
    assert!(!is_auxiliary_stitch_file("/tmp/DSC08897.jpg"));
}
