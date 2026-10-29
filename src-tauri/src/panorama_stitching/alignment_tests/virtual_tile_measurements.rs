//! Virtual_Tile coverage and measurements: covered features, exposure compensation and gradient
//! sampling on the registered grid, the prior-free seed fallback and bounded prior repairs.

use super::*;

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
