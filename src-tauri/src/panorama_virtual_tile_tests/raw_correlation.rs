//! Source_RAW correlation proposals: large translations, repeated-texture peaks, uncovered
//! payload, unvalidated negatives and the canonical inversion of a validated match.

use super::*;

#[test]
fn source_raw_correlation_recalls_true_large_translation() {
    let (width, height) = (56, 44);
    let left = textured_plane(width, height);
    let expected = translation(40.0, 2.0);
    let (right, right_coverage) = transformed_fixture(&left, &expected);
    let left =
        SourceRawCorrelationPyramid::from_native(&left, &full_coverage(width, height)).unwrap();
    let right = SourceRawCorrelationPyramid::from_native(&right, &right_coverage).unwrap();
    let proposals = source_raw_correlation_proposals(&left, &right);
    assert!(
        proposals.iter().any(|proposal| {
            transformed_point(&proposal.proposal.left_to_right, Point2::new(24.0, 20.0))
                .is_some_and(|actual| (actual - Point2::new(64.0, 22.0)).norm() <= 1.5)
        }),
        "{proposals:?}"
    );
    assert!(proposals.iter().all(|proposal| {
        (0.95..=1.05).contains(&proposal.proposal.scale)
            && (-10.0..=10.0).contains(&proposal.proposal.rotation_degrees)
    }));
}

#[test]
fn source_raw_correlation_retains_multiple_repeated_texture_peaks() {
    let (width, height) = (64, 48);
    let left = GrayImage::from_fn(width, height, |x, y| {
        let periodic = if (x / 4 + y / 4) % 2 == 0 { 55 } else { 205 };
        image::Luma([if (9..15).contains(&x) && (7..13).contains(&y) {
            ((x * 19 + y * 23) % 251) as u8
        } else {
            periodic
        }])
    });
    let expected = translation(29.0, 3.0);
    let (right, right_coverage) = transformed_fixture(&left, &expected);
    let left =
        SourceRawCorrelationPyramid::from_native(&left, &full_coverage(width, height)).unwrap();
    let right = SourceRawCorrelationPyramid::from_native(&right, &right_coverage).unwrap();
    let proposals = source_raw_correlation_proposals(&left, &right);
    assert!(
        proposals.len() > 1,
        "multi-peak oracle collapsed: {proposals:?}"
    );
    assert!(proposals.iter().any(|proposal| {
        transformed_point(&proposal.proposal.left_to_right, Point2::new(24.0, 20.0))
            .is_some_and(|actual| (actual - Point2::new(53.0, 23.0)).norm() <= 1.5)
    }));
    assert!(
        proposals
            .iter()
            .all(|proposal| proposal.second_peak_margin >= 0.0)
    );
}

#[test]
fn source_raw_correlation_ignores_uncovered_payload() {
    let (width, height) = (320, 240);
    let coverage = coverage_from_fn(width, height, |x, y| x % 17 == 0 || (80..90).contains(&y));
    let plane = |uncovered| {
        GrayImage::from_fn(width, height, |x, y| {
            image::Luma([if coverage.is_covered(x, y) {
                ((x * 7 + y * 3) % 251) as u8
            } else {
                uncovered
            }])
        })
    };
    let first = SourceRawCorrelationPyramid::from_native(&plane(0), &coverage).unwrap();
    let second = SourceRawCorrelationPyramid::from_native(&plane(255), &coverage).unwrap();
    assert_eq!(first.coarse.factor, second.coarse.factor);
    assert_eq!(first.coarse.luminance, second.coarse.luminance);
    assert_eq!(first.coarse.gradient, second.coarse.gradient);
    assert_eq!(first.coarse.covered_count, second.coarse.covered_count);
}

#[test]
fn source_raw_correlation_proposals_cannot_accept_an_unvalidated_negative() {
    let (width, height) = (64, 48);
    let left = textured_plane(width, height);
    let right = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([((x * 71 + y * 29 + x * y * 3) % 251) as u8])
    });
    let left_pyramid =
        SourceRawCorrelationPyramid::from_native(&left, &full_coverage(width, height)).unwrap();
    let right_pyramid =
        SourceRawCorrelationPyramid::from_native(&right, &full_coverage(width, height)).unwrap();
    let proposals = source_raw_correlation_proposals(&left_pyramid, &right_pyramid);
    assert!(
        !proposals.is_empty(),
        "proposal recall is not an acceptance gate"
    );
    let left = plane_info(0, left);
    let right = plane_info(1, right);
    assert!(proposals.iter().all(|proposal| {
        source_raw_correlation_guided_matches(&left, &right, &proposal.proposal).is_empty()
    }));
}

#[test]
fn source_raw_validated_match_canonical_inversion_swaps_points_once() {
    let transform = translation(31.0, -7.0) * uniform_scale(1.02);
    let points = vec![
        (Point2::new(4.0, 8.0), Point2::new(35.08, 1.16)),
        (Point2::new(20.0, 12.0), Point2::new(51.4, 5.24)),
    ];
    let relation = MatchInfo {
        native_verified_points: Vec::new(),
        homography: transform,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: false,
        points: points.clone(),
        candidate_points: Vec::new(),
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
    };
    let (forward, forward_points, _) =
        source_raw_normalize_validated_match(&relation, false).unwrap();
    assert_projectively_equal(&forward, &transform, "forward canonical relation");
    assert_eq!(forward_points, points);
    let (reverse, reverse_points, _) =
        source_raw_normalize_validated_match(&relation, true).unwrap();
    assert_projectively_equal(
        &reverse,
        &transform.try_inverse().unwrap(),
        "reversed canonical relation",
    );
    assert_eq!(reverse_points[0], (points[0].1, points[0].0));
}
