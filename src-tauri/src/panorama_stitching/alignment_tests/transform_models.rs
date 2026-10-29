//! Transform estimation and model selection: native and robust affine fits, panorama and
//! focus-stack transform choice, registration guards, homography refinement and patch NCC.

use super::*;

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
