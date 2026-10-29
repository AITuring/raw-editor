//! Source_RAW direct photometric refinement: reciprocal multi-peak pairing, a brute-force oracle,
//! gain, bias and blur, partial coverage, holdout evidence, canonical closure, repeated RAW ids
//! and bit determinism across thread counts.

use super::*;

fn source_raw_direct_test_pyramid(
    gray: &GrayImage,
    coverage: &stitching::CoverageMask,
) -> SourceRawDirectPyramid {
    SourceRawDirectPyramid::from_native(gray, coverage).expect("direct pyramid")
}

fn source_raw_direct_refined_pair(
    left: &GrayImage,
    right: &GrayImage,
    right_coverage: &stitching::CoverageMask,
    forward_seed: Matrix3<f64>,
    reverse_seed: Matrix3<f64>,
) -> (SourceRawDirectFold, SourceRawDirectFold) {
    let left_pyramid =
        source_raw_direct_test_pyramid(left, &full_coverage(left.width(), left.height()));
    let right_pyramid = source_raw_direct_test_pyramid(right, right_coverage);
    let forward = source_raw_direct_refine(&left_pyramid, &right_pyramid, forward_seed, 0, 0)
        .expect("forward direct refinement");
    let reverse = source_raw_direct_refine(&right_pyramid, &left_pyramid, reverse_seed, 0, 0)
        .expect("reverse direct refinement");
    (forward, reverse)
}

fn source_raw_direct_test_seed(
    transform: Matrix3<f64>,
    score: f64,
    overlap_samples: usize,
) -> SourceRawCorrelationProposal {
    SourceRawCorrelationProposal {
        proposal: VirtualTileGlobalProposal {
            left_to_right: transform,
            score,
            overlap_samples,
            left_factor: 1,
            right_factor: 1,
            scale: 1.0,
            rotation_degrees: 0.0,
        },
        second_peak_margin: 0.0,
    }
}

#[test]
fn source_raw_direct_reciprocal_multi_peak_pairing_rejects_top_aliases() {
    let (width, height) = (128, 96);
    let left = textured_plane(width, height);
    let expected = translation(31.0, 1.0);
    let (right, right_coverage) = transformed_fixture(&left, &expected);
    let left_info = plane_info(0, left.clone());
    let right_info = plane_info(1, right.clone());
    let left_pyramid = source_raw_direct_test_pyramid(&left, &full_coverage(width, height));
    let right_pyramid = source_raw_direct_test_pyramid(&right, &right_coverage);
    let forward_seeds = vec![
        source_raw_direct_test_seed(translation(23.0, 1.0), 0.99, 500),
        source_raw_direct_test_seed(expected, 0.93, 420),
    ];
    let reverse_seeds = vec![
        source_raw_direct_test_seed(translation(-63.0, -1.0), 0.99, 500),
        source_raw_direct_test_seed(translation(-31.0, -1.0), 0.93, 420),
    ];
    let selected = source_raw_direct_select_basin(
        &left_info,
        &right_info,
        &left_pyramid,
        &right_pyramid,
        &forward_seeds,
        &reverse_seeds,
    )
    .expect("a reciprocal image-derived basin");
    assert_eq!(
        (selected.forward[0].seed_rank, selected.reverse[0].seed_rank),
        (1, 1),
        "one-way scores and overlap must not let aliases win"
    );
    assert!(selected.closures.iter().all(|closure| *closure <= 1.0));
    assert!(selected.union_points >= STATION_RELATION_MIN_INLIERS);
    assert!(
        focus_transform_disagreement_px(
            &selected.forward[0].transform,
            &expected,
            left.dimensions(),
        )
        .unwrap()
            <= 1.0
    );
}

/// Independent of F-Direct's robust objective: exhaustive Pearson correlation
/// over the complete scale/rotation/translation grid used by this fixture.
fn source_raw_direct_brute_force_oracle(
    left: &GrayImage,
    right: &GrayImage,
    right_coverage: &stitching::CoverageMask,
) -> Matrix3<f64> {
    let mut best = (f64::NEG_INFINITY, Matrix3::identity());
    for scale_percent in 98..=102 {
        let scale = scale_percent as f64 / 100.0;
        for angle_degrees in -2..=2 {
            let theta = (angle_degrees as f64).to_radians();
            let (sin, cos) = theta.sin_cos();
            for ty in -8..=2 {
                for tx in 6..=16 {
                    let transform = Matrix3::new(
                        scale * cos,
                        -scale * sin,
                        tx as f64,
                        scale * sin,
                        scale * cos,
                        ty as f64,
                        0.0,
                        0.0,
                        1.0,
                    );
                    let mut left_values = Vec::new();
                    let mut right_values = Vec::new();
                    for y in (4..left.height().saturating_sub(4)).step_by(3) {
                        for x in (4..left.width().saturating_sub(4)).step_by(3) {
                            let source = Point2::new(f64::from(x), f64::from(y));
                            let Some(target) = transformed_point(&transform, source) else {
                                continue;
                            };
                            if target.x < 0.0
                                || target.y < 0.0
                                || target.x >= f64::from(right.width().saturating_sub(1))
                                || target.y >= f64::from(right.height().saturating_sub(1))
                                || !right_coverage
                                    .is_covered(target.x.round() as u32, target.y.round() as u32)
                            {
                                continue;
                            }
                            let Some(value) = registration::sample_gray(right, target.x, target.y)
                            else {
                                continue;
                            };
                            left_values.push(f64::from(left.get_pixel(x, y)[0]));
                            right_values.push(value);
                        }
                    }
                    if left_values.len() >= 96
                        && let Some(score) = normalized_correlation(&left_values, &right_values)
                        && score > best.0
                    {
                        best = (score, transform);
                    }
                }
            }
        }
    }
    best.1
}

#[test]
fn source_raw_direct_brute_force_oracle_and_refinement_recover_similarity() {
    let left = textured_plane(96, 72);
    let expected = translation(11.0, -4.0) * rotation(1f64.to_radians()) * uniform_scale(1.01);
    let (right, right_coverage) = transformed_fixture(&left, &expected);
    let oracle = source_raw_direct_brute_force_oracle(&left, &right, &right_coverage);
    assert!(focus_transform_disagreement_px(&oracle, &expected, left.dimensions()).unwrap() <= 1.0);
    let seed = translation(9.5, -2.5) * rotation(0.5f64.to_radians()) * uniform_scale(1.0);
    let (forward, reverse) = source_raw_direct_refined_pair(
        &left,
        &right,
        &right_coverage,
        seed,
        expected.try_inverse().unwrap() * translation(0.75, -0.5),
    );
    assert!(
        focus_transform_disagreement_px(&forward.transform, &oracle, left.dimensions()).unwrap()
            <= 1.0,
        "refined={:?} oracle={oracle:?}",
        forward.transform
    );
    let closure =
        source_raw_direct_closure(&forward.transform, &reverse.transform, left.dimensions())
            .unwrap();
    assert!(
        closure <= 1.0,
        "closure={closure} forward={:?} reverse={:?}",
        forward.transform,
        reverse.transform
    );
}

#[test]
fn source_raw_direct_gain_bias_and_blur_keep_the_geometry() {
    let left = textured_plane(96, 72);
    let expected = translation(10.0, -3.0);
    let (warped, coverage) = transformed_fixture(&left, &expected);
    let photometric = GrayImage::from_fn(warped.width(), warped.height(), |x, y| {
        image::Luma([
            ((f64::from(warped.get_pixel(x, y)[0]) * 0.78 + 28.0).clamp(0.0, 255.0)) as u8,
        ])
    });
    let blurred = imageproc::filter::gaussian_blur_f32(&photometric, 0.8);
    let left_blurred = imageproc::filter::gaussian_blur_f32(&left, 0.45);
    let (forward, _) = source_raw_direct_refined_pair(
        &left_blurred,
        &blurred,
        &coverage,
        translation(8.5, -2.0),
        translation(-9.0, 3.5),
    );
    assert!(
        focus_transform_disagreement_px(&forward.transform, &expected, left.dimensions()).unwrap()
            <= 1.0,
        "{:?}",
        forward.transform
    );
    assert!((forward.holdout.gain - 0.78).abs() < 0.2);
}

#[test]
fn source_raw_direct_partial_coverage_ignores_every_uncovered_payload_bit() {
    let (width, height) = (96, 72);
    let coverage = coverage_from_fn(width, height, |x, y| {
        (8..88).contains(&x) && (6..66).contains(&y) && !(40..50).contains(&x)
    });
    let plane = |payload| {
        GrayImage::from_fn(width, height, |x, y| {
            image::Luma([if coverage.is_covered(x, y) {
                textured_value(f64::from(x), f64::from(y)).round() as u8
            } else {
                payload
            }])
        })
    };
    let first = source_raw_direct_test_pyramid(&plane(0), &coverage);
    let second = source_raw_direct_test_pyramid(&plane(255), &coverage);
    for (left, right) in first.levels.iter().zip(&second.levels) {
        assert_eq!(left.plane.gray, right.plane.gray);
        assert_eq!(left.gx, right.gx);
        assert_eq!(left.gy, right.gy);
    }
    let first_fit = source_raw_direct_refine(&first, &first, Matrix3::identity(), 0, 0).unwrap();
    let second_fit = source_raw_direct_refine(&second, &second, Matrix3::identity(), 0, 0).unwrap();
    assert_eq!(
        virtual_tile_hypothesis_matrix_key(&first_fit.transform),
        virtual_tile_hypothesis_matrix_key(&second_fit.transform)
    );
    assert_eq!(first_fit.fit.loss.to_bits(), second_fit.fit.loss.to_bits());
    assert_eq!(
        first_fit.holdout.loss.to_bits(),
        second_fit.holdout.loss.to_bits()
    );
}

#[test]
fn source_raw_direct_repeated_texture_wrong_peak_has_no_holdout_evidence() {
    let (width, height) = (128, 96);
    let left = GrayImage::from_fn(width, height, |x, y| {
        let periodic = if (x / 4 + y / 4) % 2 == 0 { 45 } else { 210 };
        image::Luma([if (9..14).contains(&x) && (8..13).contains(&y) {
            ((x * 37 + y * 19) % 251) as u8
        } else {
            periodic
        }])
    });
    let expected = translation(31.0, 1.0);
    let (right, coverage) = transformed_fixture(&left, &expected);
    let left_info = plane_info(0, left.clone());
    let right_info = plane_info(1, right.clone());
    let left_pyramid = source_raw_direct_test_pyramid(&left, &full_coverage(width, height));
    let right_pyramid = source_raw_direct_test_pyramid(&right, &coverage);
    let wrong = translation(23.0, 1.0);
    let points = source_raw_direct_localize_holdout(
        &left_info,
        &right_info,
        &left_pyramid,
        &right_pyramid,
        &wrong,
        &wrong.try_inverse().unwrap(),
        0,
    );
    assert!(points.len() < STATION_RELATION_MIN_INLIERS, "{points:?}");
}

#[test]
fn source_raw_direct_non_overlap_cannot_produce_twenty_four_holdout_points() {
    let left = textured_plane(96, 72);
    let right = GrayImage::from_fn(96, 72, |x, y| {
        image::Luma([((x * 71 + y * 43 + x * y * 5) % 251) as u8])
    });
    let left_info = plane_info(0, left.clone());
    let right_info = plane_info(1, right.clone());
    let left_pyramid = source_raw_direct_test_pyramid(&left, &full_coverage(96, 72));
    let right_pyramid = source_raw_direct_test_pyramid(&right, &full_coverage(96, 72));
    let forward_seeds = vec![
        source_raw_direct_test_seed(Matrix3::identity(), 0.99, 400),
        source_raw_direct_test_seed(translation(12.0, -5.0), 0.95, 350),
    ];
    let reverse_seeds = vec![
        source_raw_direct_test_seed(Matrix3::identity(), 0.99, 400),
        source_raw_direct_test_seed(translation(-12.0, 5.0), 0.95, 350),
    ];
    let selected = source_raw_direct_select_basin(
        &left_info,
        &right_info,
        &left_pyramid,
        &right_pyramid,
        &forward_seeds,
        &reverse_seeds,
    );
    assert!(selected.is_none_or(|selection| {
        selection.union_points < STATION_RELATION_MIN_INLIERS
            || selection.minimum_spatial_support < STATION_RELATION_MIN_SPATIAL_SUPPORT
    }));
}

#[test]
fn source_raw_direct_holdout_points_are_measured_not_synthesized() {
    let left = textured_plane(128, 96);
    let expected = translation(8.0, 2.0);
    let (right, coverage) = transformed_fixture(&left, &expected);
    let left_info = plane_info(0, left.clone());
    let right_info = plane_info(1, right.clone());
    let left_pyramid = source_raw_direct_test_pyramid(&left, &full_coverage(128, 96));
    let right_pyramid = source_raw_direct_test_pyramid(&right, &coverage);
    let baseline = source_raw_direct_localize_holdout(
        &left_info,
        &right_info,
        &left_pyramid,
        &right_pyramid,
        &expected,
        &expected.try_inverse().unwrap(),
        0,
    );
    assert!(baseline.len() >= STATION_RELATION_MIN_INLIERS);
    let witness = baseline[baseline.len() / 2].clone();
    let mut damaged = right;
    let center = Point2::new(witness.target[0], witness.target[1]);
    for y in (center.y as i32 - 8).max(0)..=(center.y as i32 + 8).min(95) {
        for x in (center.x as i32 - 8).max(0)..=(center.x as i32 + 8).min(127) {
            damaged.put_pixel(
                x as u32,
                y as u32,
                image::Luma([((x * 83 + y * 47) % 251) as u8]),
            );
        }
    }
    let damaged_info = plane_info(1, damaged.clone());
    let damaged_pyramid = source_raw_direct_test_pyramid(&damaged, &coverage);
    let measured = source_raw_direct_localize_holdout(
        &left_info,
        &damaged_info,
        &left_pyramid,
        &damaged_pyramid,
        &expected,
        &expected.try_inverse().unwrap(),
        0,
    );
    assert!(
        measured.iter().all(|point| point.cell != witness.cell)
            || measured
                .iter()
                .find(|point| point.cell == witness.cell)
                .is_some_and(|point| point.target != witness.target),
        "a damaged hold-out block cannot retain target=H(source)"
    );
}

#[test]
fn source_raw_direct_canonicalization_and_independent_reverse_close_once() {
    let raw_forward = translation(18.0, -4.0) * uniform_scale(1.02);
    let raw_reverse = raw_forward.try_inverse().unwrap();
    let left_local = translation(5.0, 2.0);
    let right_local = translation(-3.0, 1.0);
    let canonical = right_local * raw_forward * left_local.try_inverse().unwrap();
    let canonical_reverse = left_local * raw_reverse * right_local.try_inverse().unwrap();
    assert!(source_raw_direct_closure(&canonical, &canonical_reverse, (120, 90)).unwrap() < 1e-9);
    let point = Point2::new(20.0, 30.0);
    let target = transformed_point(&raw_forward, point).unwrap();
    assert_eq!(
        transformed_point(&canonical, transformed_point(&left_local, point).unwrap()).unwrap(),
        transformed_point(&right_local, target).unwrap()
    );
}

#[test]
fn source_raw_direct_support_never_counts_a_repeated_raw_id() {
    let models = [(0, 10), (0, 11), (1, 11), (1, 12)]
        .into_iter()
        .enumerate()
        .map(
            |(record_index, (left_source, right_source))| SourceRawLayerDiagnosticModel {
                left_source,
                right_source,
                record_index,
                transform: Matrix3::identity(),
                spatial_support: 0.5,
            },
        )
        .collect::<Vec<_>>();
    let selected = source_raw_independent_observations(&models, &[0, 1, 2, 3]);
    assert_eq!(selected.len(), 2);
    assert_eq!(
        selected
            .iter()
            .map(|&index| models[index].left_source)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        2
    );
    assert_eq!(
        selected
            .iter()
            .map(|&index| models[index].right_source)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        2
    );
}

#[test]
fn source_raw_direct_is_bit_deterministic_across_thread_counts() {
    let left = textured_plane(96, 72);
    let expected = translation(10.0, -3.0);
    let (right, coverage) = transformed_fixture(&left, &expected);
    let left_info = plane_info(0, left.clone());
    let right_info = plane_info(1, right.clone());
    let forward_seeds = vec![
        source_raw_direct_test_seed(translation(18.0, -3.0), 0.99, 500),
        source_raw_direct_test_seed(expected, 0.94, 420),
    ];
    let reverse_seeds = vec![
        source_raw_direct_test_seed(translation(-2.0, 3.0), 0.99, 500),
        source_raw_direct_test_seed(translation(-10.0, 3.0), 0.94, 420),
    ];
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let left_pyramid = source_raw_direct_test_pyramid(
                    &left,
                    &full_coverage(left.width(), left.height()),
                );
                let right_pyramid = source_raw_direct_test_pyramid(&right, &coverage);
                let selected = source_raw_direct_select_basin(
                    &left_info,
                    &right_info,
                    &left_pyramid,
                    &right_pyramid,
                    &forward_seeds,
                    &reverse_seeds,
                )
                .unwrap();
                (
                    selected.forward[0].seed_rank,
                    selected.reverse[0].seed_rank,
                    selected
                        .forward
                        .iter()
                        .map(|fold| virtual_tile_hypothesis_matrix_key(&fold.transform))
                        .collect::<Vec<_>>(),
                    selected
                        .reverse
                        .iter()
                        .map(|fold| virtual_tile_hypothesis_matrix_key(&fold.transform))
                        .collect::<Vec<_>>(),
                    selected
                        .closures
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    selected
                        .points
                        .iter()
                        .map(|points| points.len())
                        .collect::<Vec<_>>(),
                    selected.minimum_spatial_support.to_bits(),
                )
            })
    };
    let expected_bits = run(1);
    for threads in [2, 8, 16] {
        assert_eq!(run(threads), expected_bits, "threads={threads}");
    }
}
