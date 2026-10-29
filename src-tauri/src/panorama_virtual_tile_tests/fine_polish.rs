//! The fine polish stage: NCC and best window offsets, low-texture templates, projective polish
//! fits, polished station matches on noisy planes, and the final-polish fallback after a
//! recoverable coarse failure.

use super::*;

// ---------------------------------------------------------------------------
// 8. polish helpers
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_ncc_and_best_window_offset_recover_integer_offsets() {
    let run = [1.0, 4.0, 2.0, 8.0, 5.0];
    assert!((virtual_tile_ncc(&run, &run).unwrap() - 1.0).abs() < 1e-12);
    assert!((virtual_tile_ncc(&run, &run.map(|value| -value)).unwrap() + 1.0).abs() < 1e-12);
    assert!(
        (virtual_tile_ncc(&run, &run.map(|value| 3.0 * value + 7.0)).unwrap() - 1.0).abs() < 1e-12
    );
    assert!(virtual_tile_ncc(&run, &[5.0; 5]).is_none(), "constant run");
    assert!(
        virtual_tile_ncc(&run, &run[..4]).is_none(),
        "length mismatch"
    );
    assert!(virtual_tile_ncc(&[], &[]).is_none());

    let (half, search) = (3i32, 4i32);
    let side = (2 * (half + search) + 1) as usize;
    let template_side = (2 * half + 1) as usize;
    let window = (0..side * side)
        .map(|index| textured_value((index % side) as f64 * 3.1, (index / side) as f64 * 2.7))
        .collect::<Vec<_>>();
    for (offset_x, offset_y) in [(0, 0), (2, -3), (-4, 4), (1, 1)] {
        let template = (0..template_side * template_side)
            .map(|index| {
                let x = (index % template_side) as i32 + search + offset_x;
                let y = (index / template_side) as i32 + search + offset_y;
                window[y as usize * side + x as usize]
            })
            .collect::<Vec<_>>();
        let ((best_x, best_y), best, second, scores) =
            virtual_tile_best_window_offset(&template, &window, half, search).unwrap();
        assert_eq!((best_x, best_y), (offset_x, offset_y));
        assert!((best - 1.0).abs() < 1e-9);
        assert!(second < best);
        assert_eq!(scores.len(), ((2 * search + 1) * (2 * search + 1)) as usize);
    }
}

#[test]
fn virtual_tile_polish_correspondences_skip_low_texture_templates() {
    let full = full_coverage(96, 96);
    let textured = textured_plane(96, 96);
    // Alternating 127/129 has a standard deviation of 1 < 3 levels.
    let low_texture = GrayImage::from_fn(96, 96, |x, y| {
        image::Luma([if (x + y) % 2 == 0 { 127 } else { 129 }])
    });
    let bounds = (0.0, 0.0, 95.0, 95.0);
    let mut counts = VirtualTilePolishCounts::default();
    let matches = virtual_tile_polish_correspondences(
        &low_texture,
        &full,
        &textured,
        &full,
        &Matrix3::identity(),
        bounds,
        4,
        &mut counts,
    );
    assert_eq!(counts.probes, 25);
    assert_eq!(counts.textured, 0);
    assert_eq!(counts.matched, 0);
    assert!(matches.is_empty());

    let mut counts = VirtualTilePolishCounts::default();
    virtual_tile_polish_correspondences(
        &textured,
        &full,
        &textured,
        &full,
        &Matrix3::identity(),
        bounds,
        4,
        &mut counts,
    );
    assert_eq!(counts.probes, 25);
    assert_eq!(counts.textured, counts.probes);
}

// ---------------------------------------------------------------------------
// 8. polish fit and polished station match
// ---------------------------------------------------------------------------

fn polish_truth() -> Matrix3<f64> {
    translation(-60.0, 8.0)
        * about_center((240.0, 180.0), rotation(0.008) * uniform_scale(1.004))
        * Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.000_012, -0.000_008, 1.0)
}

#[test]
fn virtual_tile_polish_fit_recovers_a_projective_relation_and_drops_outliers() {
    let truth = polish_truth();
    let mut correspondences = (0..8)
        .flat_map(|row| (0..8).map(move |column| (row, column)))
        .map(|(row, column)| {
            let source = Point2::new(90.0 + 45.0 * column as f64, 40.0 + 40.0 * row as f64);
            let target = transformed_point(&truth, source).unwrap()
                + nalgebra::Vector2::new(
                    hashed_noise(column, row, 1, 0.3),
                    hashed_noise(column, row, 2, 0.3),
                );
            (source, target)
        })
        .collect::<Vec<_>>();
    let true_count = correspondences.len();
    for outlier in 0..16u32 {
        let source = Point2::new(
            100.0 + 21.0 * f64::from(outlier),
            60.0 + 13.0 * f64::from(outlier),
        );
        let offset = nalgebra::Vector2::new(
            25.0 + 2.0 * f64::from(outlier),
            -30.0 + 3.0 * f64::from(outlier % 5),
        );
        correspondences.push((source, transformed_point(&truth, source).unwrap() + offset));
    }
    let threshold = VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX;
    let (fitted, inliers) = virtual_tile_polish_fit(&correspondences, threshold, (480, 360))
        .expect("64 consistent pairs among 16 outliers");
    assert_eq!(inliers, (0..true_count).collect::<Vec<_>>());
    for (x, y) in [
        (90.0, 40.0),
        (405.0, 40.0),
        (90.0, 320.0),
        (405.0, 320.0),
        (240.0, 180.0),
    ] {
        let point = Point2::new(x, y);
        let error = (transformed_point(&fitted, point).unwrap()
            - transformed_point(&truth, point).unwrap())
        .norm();
        assert!(error < 0.3, "{error} px at ({x}, {y})");
    }

    // Fewer than the configured minimum consistent pairs is no relation.
    assert!(
        virtual_tile_polish_fit(
            &correspondences[..STATION_RELATION_MIN_INLIERS - 1],
            threshold,
            (480, 360)
        )
        .is_none()
    );
    // A mirrored relation is never a valid planar station relation.
    let mirror = Matrix3::new(-1.0, 0.0, 480.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let mirrored = correspondences[..true_count]
        .iter()
        .map(|&(source, _)| (source, transformed_point(&mirror, source).unwrap()))
        .collect::<Vec<_>>();
    assert!(virtual_tile_polish_fit(&mirrored, threshold, (480, 360)).is_none());
}

#[test]
fn virtual_tile_polished_station_match_recovers_a_perturbed_seed_on_noisy_planes() {
    let (width, height) = (480u32, 360u32);
    let truth = polish_truth();
    let inverse = truth.try_inverse().unwrap();
    let left = GrayImage::from_fn(width, height, |x, y| {
        let value = textured_value(f64::from(x), f64::from(y)) + hashed_noise(x, y, 3, 3.0);
        image::Luma([value.clamp(0.0, 255.0).round() as u8])
    });
    let right = GrayImage::from_fn(width, height, |x, y| {
        let source = transformed_point(&inverse, Point2::new(f64::from(x), f64::from(y))).unwrap();
        let value = textured_value(source.x, source.y) + hashed_noise(x, y, 4, 3.0);
        image::Luma([value.clamp(0.0, 255.0).round() as u8])
    });
    let (left_info, right_info) = (plane_info(0, left), plane_info(1, right));
    let coverage = full_coverage(width, height);
    // The seed misses by a 2.5px shift and a 0.15° rotation: inside the
    // first ±8px pass, outside the requirement-6.3 3px tolerance at the edges.
    let seed = translation(2.5, -1.5) * about_center((240.0, 180.0), rotation(0.0026)) * truth;
    let mut diagnostic = sample_candidate_record();
    let hypotheses = virtual_tile_polished_station_match(
        &left_info,
        &coverage,
        &right_info,
        &coverage,
        1,
        &seed,
        ((width, height), (width, height)),
        &mut diagnostic,
    );
    let mut model_ranks = hypotheses
        .iter()
        .map(|relation| virtual_tile_hypothesis_model_rank(&relation.homography))
        .collect::<Vec<_>>();
    model_ranks.sort_unstable();
    model_ranks.dedup();
    assert_eq!(model_ranks, [0, 1, 2], "all model families must survive");
    let relation = hypotheses
        .into_iter()
        .filter(|relation| virtual_tile_hypothesis_model_rank(&relation.homography) == 2)
        .last()
        .unwrap_or_else(|| panic!("polish must recover the projective seed: {diagnostic:?}"));
    assert_eq!(diagnostic.failure_stage, "fitted");
    assert_eq!(diagnostic.residual_model, "polished_multi_hypothesis");
    assert!(relation.inliers >= STATION_RELATION_MIN_INLIERS);
    assert!(diagnostic.hull_support >= STATION_RELATION_MIN_SPATIAL_SUPPORT);
    assert!(relation.canonical_homography.is_none());
    for (x, y) in [
        (120.0, 60.0),
        (400.0, 60.0),
        (120.0, 300.0),
        (400.0, 300.0),
        (260.0, 180.0),
    ] {
        let point = Point2::new(x, y);
        let error = (transformed_point(&relation.homography, point).unwrap()
            - transformed_point(&truth, point).unwrap())
        .norm();
        assert!(error < 0.75, "{error} px at ({x}, {y})");
    }
}

// ---------------------------------------------------------------------------
// 9. final-polish fallback after a recoverable coarse failure
// ---------------------------------------------------------------------------

/// Build the two levels consumed by `virtual_tile_pyramid_station_match`
/// directly.  The coarse target contains a smooth, non-affine sinusoidal warp
/// that defeats a single bounded residual model while leaving every local
/// patch measurable.  The finest planes carry the same texture under one
/// known native translation, so the independent model-warped polish can
/// verify it when the last coarse level gives up.
fn final_polish_fallback_fixture(
    sparse_finest_target: bool,
) -> (VirtualTileMatchPyramid, Matrix3<f64>) {
    let coarse_dimensions = (288u32, 288u32);
    let finest_dimensions = (576u32, 576u32);
    let native_dimensions = (2_304u32, 2_304u32);
    let coarse_left = textured_plane(coarse_dimensions.0, coarse_dimensions.1);
    let coarse_right = GrayImage::from_fn(coarse_dimensions.0, coarse_dimensions.1, |x, y| {
        let phase = f64::from(y) * std::f64::consts::TAU / 96.0;
        let dx = (4.0 * phase.sin()).round() as i32;
        let source_x = (x as i32 - dx).clamp(0, coarse_dimensions.0 as i32 - 1) as u32;
        *coarse_left.get_pixel(source_x, y)
    });

    // The known relation is left -> right = translate(+8, -4) in native
    // pixels, i.e. (+2, -1) on the factor-4 finest planes.
    let known_native = translation(8.0, -4.0);
    let finest_left = textured_plane(finest_dimensions.0, finest_dimensions.1);
    let finest_right = GrayImage::from_fn(finest_dimensions.0, finest_dimensions.1, |x, y| {
        let source_x = (x as i32 - 2).clamp(0, finest_dimensions.0 as i32 - 1) as u32;
        let source_y = (y as i32 + 1).clamp(0, finest_dimensions.1 as i32 - 1) as u32;
        *finest_left.get_pixel(source_x, source_y)
    });
    let full = full_coverage(finest_dimensions.0, finest_dimensions.1);
    let finest_target_coverage = if sparse_finest_target {
        coverage_from_fn(finest_dimensions.0, finest_dimensions.1, |x, _| x < 40)
    } else {
        full.clone()
    };

    let mut pyramid = VirtualTileMatchPyramid::new(vec![8, 4], 2);
    pyramid.native_dimensions = vec![native_dimensions; 2];
    pyramid.levels[0] = vec![plane_info(0, coarse_left), plane_info(1, coarse_right)];
    pyramid.coverages[0] = vec![
        full_coverage(coarse_dimensions.0, coarse_dimensions.1),
        full_coverage(coarse_dimensions.0, coarse_dimensions.1),
    ];
    pyramid.levels[1] = vec![plane_info(0, finest_left), plane_info(1, finest_right)];
    pyramid.coverages[1] = vec![full, finest_target_coverage];
    pyramid.quality_coverages = pyramid.coverages[1].clone();
    (pyramid, known_native)
}

#[test]
fn virtual_tile_pyramid_polishes_after_final_coarse_residual_failure() {
    let (pyramid, known_native) = final_polish_fallback_fixture(false);
    let initial = HashMap::from([(0usize, Matrix3::identity()), (1usize, Matrix3::identity())]);
    let (_, coarse_diagnostic) = virtual_tile_direct_station_match_with_structure(
        0,
        1,
        &pyramid.levels[0],
        &pyramid.coverages[0],
        &initial,
        VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX,
        None,
    );
    assert!(
        matches!(
            coarse_diagnostic.failure_stage.as_str(),
            "residual_model_no_fit"
                | "insufficient_matches_after_model_quota"
                | "insufficient_fitted_hull_support"
        ),
        "fixture must exercise the recoverable final-coarse failure: {coarse_diagnostic:?}"
    );
    let (relation, diagnostic) = virtual_tile_pyramid_station_match(
        0,
        1,
        &pyramid,
        &Matrix3::identity(),
        VIRTUAL_TILE_SEED_RENDERING_PRIOR,
    );
    let relation = relation.into_iter().last().unwrap_or_else(|| {
        panic!("the finest polish must recover the known transform: {diagnostic:?}")
    });
    assert_eq!(diagnostic.failure_stage, "fitted", "{diagnostic:?}");
    assert!(
        diagnostic.fitted_inliers >= STATION_RELATION_MIN_INLIERS,
        "{diagnostic:?}"
    );
    assert!(
        diagnostic.hull_support >= STATION_RELATION_MIN_SPATIAL_SUPPORT,
        "{diagnostic:?}"
    );
    for (x, y) in [
        (0.0, 0.0),
        (1_152.0, 0.0),
        (0.0, 1_152.0),
        (1_152.0, 1_152.0),
    ] {
        let point = Point2::new(x, y);
        let expected = transformed_point(&known_native, point).unwrap();
        let measured = transformed_point(&relation.homography, point).unwrap();
        assert!(
            (expected - measured).norm() < 0.75,
            "coarse fallback: {expected:?} vs {measured:?} at ({x}, {y})"
        );
    }
}

#[test]
fn virtual_tile_pyramid_does_not_promote_insufficient_finest_evidence() {
    let (pyramid, _) = final_polish_fallback_fixture(true);
    let (relation, diagnostic) = virtual_tile_pyramid_station_match(
        0,
        1,
        &pyramid,
        &Matrix3::identity(),
        VIRTUAL_TILE_SEED_RENDERING_PRIOR,
    );
    assert!(
        relation.is_empty() || diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT,
        "{diagnostic:?}"
    );
    if relation.is_empty() {
        assert_ne!(diagnostic.failure_stage, "fitted", "{diagnostic:?}");
    }
    assert!(
        diagnostic.measurable_patch_count < STATION_RELATION_MIN_INLIERS
            || diagnostic.bidirectional_match_count < STATION_RELATION_MIN_INLIERS
            || diagnostic.fitted_inliers < STATION_RELATION_MIN_INLIERS
            || diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT,
        "insufficient support must remain visible in the diagnostic: {diagnostic:?}"
    );
}
