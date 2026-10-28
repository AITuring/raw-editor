//! Regression tests for the checkpoint-10 Virtual_Tile station matcher:
//! coarse-to-fine match planes, the fine polish stage, prior-free relation
//! proposals, rendering-prior repair planning and the structural quality plane.

use super::*;
use crate::panorama_utils::stack_pipeline::{intra_station, report};
use nalgebra::Point2;

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------

fn full_coverage(width: u32, height: u32) -> stitching::CoverageMask {
    stitching::CoverageMask::from_bytes(
        width,
        height,
        vec![u8::MAX; width as usize * height as usize],
    )
    .expect("a fully covered mask matches its plane")
}

fn coverage_from_fn(
    width: u32,
    height: u32,
    covered: impl Fn(u32, u32) -> bool,
) -> stitching::CoverageMask {
    let bytes = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| if covered(x, y) { u8::MAX } else { 0 })
        .collect::<Vec<_>>();
    stitching::CoverageMask::from_bytes(width, height, bytes).expect("mask dimensions match")
}

fn coverage_bits(mask: &stitching::CoverageMask) -> Vec<bool> {
    let (width, height) = mask.dimensions();
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| mask.is_covered(x, y))
        .collect()
}

fn textured_value(x: f64, y: f64) -> f64 {
    (126.0
        + 47.0 * (x * 0.091 + y * 0.037).sin()
        + 38.0 * (x * 0.027 - y * 0.113).cos()
        + 24.0 * (x * 0.173 + y * 0.149).sin()
        + 13.0 * (x * 0.311 - y * 0.071).cos())
    .clamp(0.0, 255.0)
}

fn textured_plane(width: u32, height: u32) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        image::Luma([textured_value(f64::from(x), f64::from(y)).round() as u8])
    })
}

fn plane_info(id: usize, gray: GrayImage) -> ImageInfo {
    ImageInfo {
        id,
        filename: format!("virtual://pyramid-test-{id}"),
        width: gray.width(),
        height: gray.height(),
        alignment_image: gray,
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

fn planes_for(
    factors: &[u32],
    gray: &GrayImage,
    coverage: &stitching::CoverageMask,
) -> Vec<VirtualTileMatchPlane> {
    factors
        .iter()
        .map(|&factor| {
            VirtualTileMatchPlane::from_native(gray, coverage, factor).expect("usable level")
        })
        .collect()
}

fn translation(dx: f64, dy: f64) -> Matrix3<f64> {
    Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0)
}

/// `linear` applied about `center`: T(c) · L · T(−c).
fn about_center(center: (f64, f64), linear: Matrix3<f64>) -> Matrix3<f64> {
    translation(center.0, center.1) * linear * translation(-center.0, -center.1)
}

fn rotation(theta: f64) -> Matrix3<f64> {
    let (sin, cos) = theta.sin_cos();
    Matrix3::new(cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0)
}

fn uniform_scale(scale: f64) -> Matrix3<f64> {
    Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0)
}

/// `sqrt(|det J|)` of a homography at `point`, by central differences.
fn numeric_local_scale(transform: &Matrix3<f64>, point: Point2<f64>) -> f64 {
    let step = 1e-3;
    let at = |dx: f64, dy: f64| {
        transformed_point(transform, Point2::new(point.x + dx, point.y + dy)).expect("finite point")
    };
    let ddx = (at(step, 0.0) - at(-step, 0.0)) / (2.0 * step);
    let ddy = (at(0.0, step) - at(0.0, -step)) / (2.0 * step);
    (ddx.x * ddy.y - ddx.y * ddy.x).abs().sqrt()
}

fn assert_projectively_equal(left: &Matrix3<f64>, right: &Matrix3<f64>, context: &str) {
    for (x, y) in [
        (0.0, 0.0),
        (400.0, 0.0),
        (0.0, 300.0),
        (400.0, 300.0),
        (123.0, 77.0),
    ] {
        let point = Point2::new(x, y);
        let a = transformed_point(left, point).expect("finite left point");
        let b = transformed_point(right, point).expect("finite right point");
        assert!(
            (a - b).norm() < 1e-6,
            "{context}: {a:?} vs {b:?} at ({x}, {y})"
        );
    }
}

fn line_topology(station_count: usize) -> topology::StationTopology {
    topology::StationTopology {
        row: vec![0; station_count],
        column: (0..station_count as u32).collect(),
        ..topology::StationTopology::default()
    }
}

fn prior_proposal(
    left: usize,
    right: usize,
    left_to_right: Matrix3<f64>,
    inliers: usize,
) -> VirtualTilePriorProposal {
    VirtualTilePriorProposal {
        left,
        right,
        left_to_right,
        inliers,
    }
}

/// Four station placements with small projective parts.
fn sample_tile_to_world() -> Vec<Matrix3<f64>> {
    vec![
        Matrix3::identity(),
        Matrix3::new(1.0, 0.0, 380.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0),
        Matrix3::new(1.01, 0.002, 770.0, -0.001, 0.99, -8.0, 0.000_001, 0.0, 1.0),
        Matrix3::new(1.0, 0.0, 1_150.0, 0.0, 1.0, 12.0, 0.0, 0.0, 1.0),
    ]
}

fn sample_relation(dx: f64, dy: f64) -> Matrix3<f64> {
    Matrix3::new(
        0.998, 0.004, dx, 0.003, 1.001, dy, 0.000_002, -0.000_001, 1.0,
    )
}

fn repaired_stations(repairs: &[VirtualTilePriorRepair]) -> Vec<usize> {
    repairs.iter().map(|repair| repair.station).collect()
}

// ---------------------------------------------------------------------------
// 0. Source_RAW independent-support oracle
// ---------------------------------------------------------------------------

fn source_raw_test_record(left: usize, right: usize) -> report::SourceRawLayerModelRecord {
    report::SourceRawLayerModelRecord {
        left_source: format!("left-{left}"),
        right_source: format!("right-{right}"),
        model_fitted: true,
        inliers: 24,
        median_error_px: 0.5,
        scale_ratio: 1.0,
        spatial_support: 0.4,
        anchor_transform: matrix3_row_major(&Matrix3::identity()),
        consensus_cluster: None,
        model_provenance: "stable_matcher".to_string(),
        ..Default::default()
    }
}

#[test]
fn source_raw_spatial_proposals_are_per_pair_and_preserve_match_provenance() {
    let mut matches = Vec::new();
    let mut points = Vec::new();
    for index in 0..12 {
        let source = Point2::new((index % 4) as f64 * 100.0, (index / 4) as f64 * 120.0);
        matches.push(Match {
            index1: index,
            index2: 100 + index,
        });
        points.push((
            source,
            Point2::new(source.x + 305.0 + (index % 2) as f64, source.y - 181.0),
        ));
    }
    for index in 12..48 {
        let source = Point2::new((index % 8) as f64 * 80.0, (index / 8) as f64 * 90.0);
        matches.push(Match {
            index1: index,
            index2: 100 + index,
        });
        points.push((
            source,
            Point2::new(
                source.x - 900.0 + index as f64 * 47.0,
                source.y + 800.0 - index as f64 * 53.0,
            ),
        ));
    }
    let proposals = source_raw_descriptor_proposals(&matches, &points, (1000, 800));
    assert!(!proposals.is_empty());
    let recovered = &proposals[0].1;
    assert_eq!(recovered.len(), 12);
    assert_eq!(
        recovered
            .iter()
            .map(|matched| (matched.index1, matched.index2))
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        recovered.len(),
        "proposal filtering must preserve one-to-one descriptor provenance"
    );
    assert!(recovered.iter().all(|matched| matched.index1 < 12));
}

#[test]
fn source_raw_proposal_scores_and_duplicate_provenance_never_inflate_support() {
    let mut records = (0..32)
        .map(|_| {
            let mut record = source_raw_test_record(0, 10);
            record.model_provenance = "per_raw_coarse_luminance_gradient_correlation".to_string();
            record.proposal_seeded = true;
            record.proposal_score = Some(f64::MAX);
            record.correlation_proposals = (0..8)
                .map(|rank| report::SourceRawCorrelationProposalRecord {
                    rank,
                    score: f64::MAX,
                    full_resolution_validated: true,
                    ..Default::default()
                })
                .collect();
            record
        })
        .collect::<Vec<_>>();
    let models = records
        .iter()
        .enumerate()
        .map(|(record_index, _)| SourceRawLayerDiagnosticModel {
            left_source: 0,
            right_source: 10,
            record_index,
            transform: Matrix3::identity(),
            spatial_support: 0.9,
        })
        .collect::<Vec<_>>();
    let cluster = (0..models.len()).collect::<Vec<_>>();
    assert_eq!(
        source_raw_independent_observations(&models, &cluster).len(),
        1
    );
    let clusters = source_raw_consensus_clusters(&models, &mut records, (100, 100));
    assert!(
        clusters.is_empty(),
        "proposal count and score cannot create support=2"
    );
}

#[test]
fn source_raw_independent_support_never_counts_one_raw_twice() {
    let pairs = [(0, 10), (0, 11), (1, 11)];
    let models = pairs
        .iter()
        .enumerate()
        .map(
            |(record_index, &(left_source, right_source))| SourceRawLayerDiagnosticModel {
                left_source,
                right_source,
                record_index,
                transform: Matrix3::identity(),
                spatial_support: 0.4,
            },
        )
        .collect::<Vec<_>>();
    assert_eq!(
        source_raw_independent_observations(&models, &[0, 1]).len(),
        1,
        "two layer pairs sharing one left RAW are one independent observation"
    );
    let selected = source_raw_independent_observations(&models, &[0, 1, 2]);
    assert_eq!(selected.len(), 2);
    let selected_left = selected
        .iter()
        .map(|&index| models[index].left_source)
        .collect::<std::collections::BTreeSet<_>>();
    let selected_right = selected
        .iter()
        .map(|&index| models[index].right_source)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(selected_left.len(), 2);
    assert_eq!(selected_right.len(), 2);

    let mut records = pairs
        .iter()
        .map(|&(left, right)| source_raw_test_record(left, right))
        .collect::<Vec<_>>();
    let clusters = source_raw_consensus_clusters(&models, &mut records, (100, 100));
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].observation_count, 3);
    assert_eq!(clusters[0].independent_support, 2);
    assert!(clusters[0].scale_compliant);
    assert!(clusters[0].spatially_broad);
    assert_eq!(clusters[0].supporting_layer_pairs.len(), 2);
}

#[test]
fn source_raw_layer_models_use_verified_local_to_anchor_coordinates() {
    let images = (0..4)
        .map(|id| plane_info(id, textured_plane(100, 100)))
        .collect::<Vec<_>>();
    let left_locals = [Matrix3::identity(), translation(5.0, 0.0)];
    let right_locals = [Matrix3::identity(), translation(-4.0, 0.0)];
    let expected_anchor = translation(20.0, 0.0);
    let mut matches = HashMap::new();
    for (left, right, left_local, right_local) in [
        (0usize, 2usize, left_locals[0], right_locals[0]),
        (1usize, 3usize, left_locals[1], right_locals[1]),
    ] {
        let raw_transform = right_local.try_inverse().unwrap() * expected_anchor * left_local;
        let points = [
            (10.0, 10.0),
            (50.0, 10.0),
            (80.0, 10.0),
            (10.0, 50.0),
            (80.0, 50.0),
            (10.0, 80.0),
            (50.0, 80.0),
            (80.0, 80.0),
        ]
        .into_iter()
        .map(|(x, y)| {
            let source = Point2::new(x, y);
            (source, transformed_point(&raw_transform, source).unwrap())
        })
        .collect::<Vec<_>>();
        matches.insert(
            (left, right),
            MatchInfo {
                native_verified_points: Vec::new(),
                homography: raw_transform,
                inliers: points.len(),
                sequence_bridge: false,
                coarse_bridge: false,
                points,
                candidate_points: Vec::new(),
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
                canonical_homography: Some(raw_transform),
            },
        );
    }
    let groups = vec![
        FocusCaptureGroup {
            members: vec![0, 1],
            anchor: 0,
            local_to_anchor: HashMap::from([(0, left_locals[0]), (1, left_locals[1])]),
        },
        FocusCaptureGroup {
            members: vec![2, 3],
            anchor: 2,
            local_to_anchor: HashMap::from([(2, right_locals[0]), (3, right_locals[1])]),
        },
    ];
    let diagnostics = source_raw_layer_pair_diagnostics(&images, &matches, &groups);
    assert_eq!(diagnostics.len(), 1);
    let fitted = diagnostics[0]
        .layer_models
        .iter()
        .filter(|model| model.model_fitted)
        .collect::<Vec<_>>();
    assert_eq!(fitted.len(), 2);
    for model in fitted {
        assert_projectively_equal(
            &Matrix3::from_row_slice(&model.anchor_transform),
            &expected_anchor,
            "layer model canonicalization",
        );
    }
    assert_eq!(diagnostics[0].consensus_clusters.len(), 1);
    assert_eq!(diagnostics[0].consensus_clusters[0].independent_support, 2);
}

// ---------------------------------------------------------------------------
// 1. independent global-search oracle
// ---------------------------------------------------------------------------

fn transformed_fixture(
    source: &GrayImage,
    source_to_target: &Matrix3<f64>,
) -> (GrayImage, stitching::CoverageMask) {
    let (width, height) = source.dimensions();
    let inverse = source_to_target.try_inverse().expect("fixture transform");
    let mut covered = vec![0u8; width as usize * height as usize];
    let target = GrayImage::from_fn(width, height, |x, y| {
        let Some(point) = transformed_point(&inverse, Point2::new(f64::from(x), f64::from(y)))
        else {
            return image::Luma([233]);
        };
        let (sx, sy) = (point.x.round() as i32, point.y.round() as i32);
        if sx >= 0 && sy >= 0 && sx < width as i32 && sy < height as i32 {
            covered[y as usize * width as usize + x as usize] = u8::MAX;
            *source.get_pixel(sx as u32, sy as u32)
        } else {
            image::Luma([233])
        }
    });
    (
        target,
        stitching::CoverageMask::from_bytes(width, height, covered).unwrap(),
    )
}

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

/// Deliberately independent of the production scorer: exhaustive integer
/// translation and a direct Pearson correlation over native covered pixels.
fn brute_force_translation_oracle(
    left: &GrayImage,
    right: &GrayImage,
    right_coverage: &stitching::CoverageMask,
) -> (i32, i32) {
    let (width, height) = left.dimensions();
    let mut best = (f64::NEG_INFINITY, 0, 0);
    for dy in -(height as i32 - 1)..height as i32 {
        for dx in -(width as i32 - 1)..width as i32 {
            let mut pairs = Vec::new();
            for y in 0..height as i32 {
                for x in 0..width as i32 {
                    let (tx, ty) = (x + dx, y + dy);
                    if tx >= 0
                        && ty >= 0
                        && tx < width as i32
                        && ty < height as i32
                        && right_coverage.is_covered(tx as u32, ty as u32)
                    {
                        pairs.push((
                            f64::from(left.get_pixel(x as u32, y as u32)[0]),
                            f64::from(right.get_pixel(tx as u32, ty as u32)[0]),
                        ));
                    }
                }
            }
            if pairs.len() < VIRTUAL_TILE_GLOBAL_MIN_OVERLAP_SAMPLES {
                continue;
            }
            let count = pairs.len() as f64;
            let left_mean = pairs.iter().map(|pair| pair.0).sum::<f64>() / count;
            let right_mean = pairs.iter().map(|pair| pair.1).sum::<f64>() / count;
            let numerator = pairs
                .iter()
                .map(|pair| (pair.0 - left_mean) * (pair.1 - right_mean))
                .sum::<f64>();
            let left_energy = pairs
                .iter()
                .map(|pair| (pair.0 - left_mean).powi(2))
                .sum::<f64>();
            let right_energy = pairs
                .iter()
                .map(|pair| (pair.1 - right_mean).powi(2))
                .sum::<f64>();
            let score = numerator / (left_energy * right_energy).sqrt();
            if score > best.0 {
                best = (score, dx, dy);
            }
        }
    }
    (best.1, best.2)
}

fn global_plane(
    gray: &GrayImage,
    coverage: &stitching::CoverageMask,
) -> VirtualTileGlobalSearchPlane {
    VirtualTileGlobalSearchPlane::from_native(gray, coverage).expect("global plane")
}

fn proposal_reaches(
    proposals: &[VirtualTileGlobalProposal],
    expected: &Matrix3<f64>,
    tolerance: f64,
) -> bool {
    let centre = Point2::new(24.0, 20.0);
    let expected = transformed_point(expected, centre).unwrap();
    proposals.iter().any(|proposal| {
        transformed_point(&proposal.left_to_right, centre)
            .is_some_and(|actual| (actual - expected).norm() <= tolerance)
    })
}

#[test]
fn global_similarity_recalls_full_translation_and_agrees_with_independent_oracle() {
    let (width, height) = (56, 44);
    let left = textured_plane(width, height);
    let expected = translation(40.0, 2.0);
    let (right, right_coverage) = transformed_fixture(&left, &expected);
    let oracle = brute_force_translation_oracle(&left, &right, &right_coverage);
    assert_eq!(oracle, (40, 2));
    let proposals = virtual_tile_global_similarity_proposals(
        &global_plane(&left, &full_coverage(width, height)),
        &global_plane(&right, &right_coverage),
    );
    assert!(
        proposal_reaches(&proposals, &expected, 1.5),
        "{proposals:?}"
    );
}

#[test]
fn global_similarity_includes_closed_scale_and_rotation_boundaries() {
    let (width, height) = (56, 44);
    let left = textured_plane(width, height);
    for (scale, angle, dx, dy) in [
        (0.95f64, -10.0f64, 18.0f64, 11.0f64),
        (1.05, 10.0, 14.0, -3.0),
    ] {
        let expected = translation(dx, dy) * rotation(angle.to_radians()) * uniform_scale(scale);
        let (right, right_coverage) = transformed_fixture(&left, &expected);
        let proposals = virtual_tile_global_similarity_proposals(
            &global_plane(&left, &full_coverage(width, height)),
            &global_plane(&right, &right_coverage),
        );
        assert!(
            proposal_reaches(&proposals, &expected, 2.0),
            "{scale} {angle}: {proposals:?}"
        );
        assert!(
            proposals
                .iter()
                .all(|proposal| (0.95..=1.05).contains(&proposal.scale))
        );
        assert!(
            proposals
                .iter()
                .all(|proposal| (-10.0..=10.0).contains(&proposal.rotation_degrees))
        );
    }
}

#[test]
fn global_plane_uses_partial_coverage_without_reading_uncovered_payload() {
    let (width, height) = (320, 240);
    let coverage = coverage_from_fn(width, height, |x, y| x % 17 == 0 || (80..90).contains(&y));
    let first = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([if coverage.is_covered(x, y) {
            ((x * 7 + y * 3) % 251) as u8
        } else {
            0
        }])
    });
    let second = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([if coverage.is_covered(x, y) {
            ((x * 7 + y * 3) % 251) as u8
        } else {
            255
        }])
    });
    let first = global_plane(&first, &coverage);
    let second = global_plane(&second, &coverage);
    assert_eq!(first.factor, 2);
    assert_eq!(first.luminance, second.luminance);
    assert_eq!(first.gradient, second.gradient);
    assert_eq!(first.covered_count, second.covered_count);
    assert!(
        first
            .covered_count
            .iter()
            .any(|&count| count > 0 && count < first.factor * first.factor)
    );
}

#[test]
fn source_pairwise_seed_is_composed_in_station_tile_coordinates() {
    let left = plane_info(7, textured_plane(100, 80));
    let right = plane_info(11, textured_plane(100, 80));
    let source_to_target = translation(23.0, -11.0) * uniform_scale(1.01);
    let points = hypothesis_points(&source_to_target, (100, 80));
    let matches = HashMap::from([(
        (7usize, 11usize),
        MatchInfo {
            native_verified_points: Vec::new(),
            homography: source_to_target,
            inliers: points.len(),
            sequence_bridge: false,
            coarse_bridge: false,
            points,
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            canonical_homography: Some(source_to_target),
        },
    )]);
    let source_to_world = HashMap::from([
        (7usize, translation(120.0, 40.0)),
        (11usize, translation(310.0, 95.0)),
    ]);
    let left_tile_to_world = translation(-35.0, 12.0);
    let right_tile_to_world = translation(80.0, -26.0);
    let proposals = virtual_tile_source_match_proposals(
        &[&left],
        &[&right],
        &matches,
        &source_to_world,
        &left_tile_to_world,
        &right_tile_to_world,
        1.0,
    );
    assert_eq!(proposals.len(), 1);
    let expected = right_tile_to_world.try_inverse().unwrap()
        * source_to_world[&11]
        * source_to_target
        * source_to_world[&7].try_inverse().unwrap()
        * left_tile_to_world;
    assert_projectively_equal(
        &proposals[0].left_to_right,
        &expected,
        "source pair seed should use tile-local coordinates",
    );
    assert_eq!(proposals[0].inliers, 64);
}

#[test]
fn source_pairwise_seed_scales_full_resolution_match_into_render_coordinates() {
    let left = plane_info(7, textured_plane(100, 80));
    let right = plane_info(11, textured_plane(100, 80));
    let source_to_target = translation(23.0, -11.0) * uniform_scale(1.01);
    let points = hypothesis_points(&source_to_target, (100, 80));
    let matches = HashMap::from([(
        (7usize, 11usize),
        MatchInfo {
            native_verified_points: Vec::new(),
            homography: source_to_target,
            inliers: points.len(),
            sequence_bridge: false,
            coarse_bridge: false,
            points,
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            canonical_homography: Some(source_to_target),
        },
    )]);
    let source_to_world = HashMap::from([
        (7usize, translation(120.0, 40.0)),
        (11usize, translation(310.0, 95.0)),
    ]);
    let left_tile_to_world = translation(-35.0, 12.0);
    let right_tile_to_world = translation(80.0, -26.0);
    let render_scale = 0.5;
    let proposals = virtual_tile_source_match_proposals(
        &[&left],
        &[&right],
        &matches,
        &source_to_world,
        &left_tile_to_world,
        &right_tile_to_world,
        render_scale,
    );
    assert_eq!(proposals.len(), 1);
    let source_scale = Matrix3::new(
        render_scale,
        0.0,
        0.0,
        0.0,
        render_scale,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let expected_source_to_target =
        source_scale * source_to_target * source_scale.try_inverse().unwrap();
    let expected = right_tile_to_world.try_inverse().unwrap()
        * source_to_world[&11]
        * expected_source_to_target
        * source_to_world[&7].try_inverse().unwrap()
        * left_tile_to_world;
    assert_projectively_equal(
        &proposals[0].left_to_right,
        &expected,
        "source pair seed should scale full-resolution matches into render coordinates",
    );
}

#[test]
fn source_pairwise_seed_requires_consensus_for_coarse_bridge() {
    let left = plane_info(7, textured_plane(100, 80));
    let right = plane_info(11, textured_plane(100, 80));
    let source_to_target = translation(23.0, -11.0) * uniform_scale(1.01);
    let points = hypothesis_points(&source_to_target, (100, 80));
    let mut match_info = MatchInfo {
        native_verified_points: Vec::new(),
        homography: source_to_target,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: true,
        points: points.clone(),
        candidate_points: Vec::new(),
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: Some(source_to_target),
    };
    let mut matches = HashMap::from([((7usize, 11usize), match_info.clone())]);
    let source_to_world = HashMap::from([
        (7usize, translation(120.0, 40.0)),
        (11usize, translation(310.0, 95.0)),
    ]);
    let left_tile_to_world = translation(-35.0, 12.0);
    let right_tile_to_world = translation(80.0, -26.0);
    assert!(
        virtual_tile_source_match_proposals(
            &[&left],
            &[&right],
            &matches,
            &source_to_world,
            &left_tile_to_world,
            &right_tile_to_world,
            1.0,
        )
        .is_empty(),
        "ordinary coarse bridges must not seed station geometry"
    );
    match_info.top_candidate_points = points.into_iter().take(2).collect();
    matches.insert((7, 11), match_info);
    assert_eq!(
        virtual_tile_source_match_proposals(
            &[&left],
            &[&right],
            &matches,
            &source_to_world,
            &left_tile_to_world,
            &right_tile_to_world,
            1.0,
        )
        .len(),
        1,
        "consensus coarse bridges may seed a basin, but still need the final selector"
    );
}

#[test]
fn global_similarity_keeps_true_peak_for_repeated_texture_and_is_deterministic() {
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
    let left_global = global_plane(&left, &full_coverage(width, height));
    let right_global = global_plane(&right, &right_coverage);
    let covered = (0..left_global.height)
        .flat_map(|y| (0..left_global.width).map(move |x| (x, y)))
        .filter(|&(x, y)| left_global.covered_count[(y * left_global.width + x) as usize] > 0)
        .collect::<Vec<_>>();
    let stride = covered
        .len()
        .div_ceil(VIRTUAL_TILE_GLOBAL_MAX_SAMPLES)
        .max(1);
    let samples = covered
        .into_iter()
        .step_by(stride)
        .take(VIRTUAL_TILE_GLOBAL_MAX_SAMPLES)
        .collect::<Vec<_>>();
    let expected_score =
        virtual_tile_masked_similarity_score(&left_global, &right_global, &expected, &samples);
    assert!(
        expected_score.is_some_and(|(score, _)| score > 0.95),
        "expected score: {expected_score:?}"
    );
    let run = || virtual_tile_global_similarity_proposals(&left_global, &right_global);
    let first = run();
    let second = run();
    assert!(proposal_reaches(&first, &expected, 1.5), "{first:?}");
    let threaded = [1usize, 2, 8, 16]
        .into_iter()
        .map(|threads| {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(run)
        })
        .collect::<Vec<_>>();
    let proposal_bits = |proposals: &[VirtualTileGlobalProposal]| {
        proposals
            .iter()
            .map(|proposal| {
                (
                    proposal.score.to_bits(),
                    proposal.overlap_samples,
                    virtual_tile_hypothesis_matrix_key(&proposal.left_to_right),
                )
            })
            .collect::<Vec<_>>()
    };
    for proposals in &threaded {
        assert_eq!(proposal_bits(proposals), proposal_bits(&first));
    }
    let inverse = expected.try_inverse().unwrap();
    let reverse = virtual_tile_global_similarity_proposals(&right_global, &left_global);
    assert!(proposal_reaches(&reverse, &inverse, 1.5), "{reverse:?}");
    assert_eq!(first.len(), second.len());
    for (left, right) in first.iter().zip(second.iter()) {
        assert_eq!(left.score.to_bits(), right.score.to_bits());
        assert_eq!(
            virtual_tile_hypothesis_matrix_key(&left.left_to_right),
            virtual_tile_hypothesis_matrix_key(&right.left_to_right)
        );
    }
}

#[test]
fn unrelated_global_proposals_do_not_bypass_dense_station_relation_gates() {
    let (width, height) = (64, 48);
    let left = textured_plane(width, height);
    let right = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([((x * 71 + y * 29 + x * y * 3) % 251) as u8])
    });
    let coverages = vec![full_coverage(width, height), full_coverage(width, height)];
    let tiles = vec![plane_info(0, left.clone()), plane_info(1, right.clone())];
    let mut pyramid = VirtualTileMatchPyramid::new(vec![1], 2);
    pyramid
        .set_station(0, &tiles[0], &left, &coverages[0])
        .unwrap();
    pyramid
        .set_station(1, &tiles[1], &right, &coverages[1])
        .unwrap();
    let proposals = virtual_tile_global_similarity_proposals(
        pyramid.global_planes[0].as_ref().unwrap(),
        pyramid.global_planes[1].as_ref().unwrap(),
    );
    assert!(!proposals.is_empty());
    let hypotheses = proposals
        .iter()
        .flat_map(|proposal| {
            virtual_tile_pyramid_station_match(
                0,
                1,
                &pyramid,
                &proposal.left_to_right,
                VIRTUAL_TILE_SEED_GLOBAL_COVERAGE,
            )
            .0
        })
        .collect::<Vec<_>>();
    assert!(
        !select_virtual_tile_relation_hypothesis(
            &tiles,
            0,
            1,
            &hypotheses,
            pyramid.solver_coverages()
        )
        .is_some_and(|selection| selection.passes)
    );
}

// ---------------------------------------------------------------------------
// 1. pyramid factors
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_pyramid_factors_keep_the_probe_window_and_the_smallest_level_size() {
    let window = f64::from(2 * VIRTUAL_TILE_DIRECT_PATCH_RADIUS + 1);
    let wanted = (VIRTUAL_TILE_MIN_MATCH_WINDOW_NATIVE_PX / window).ceil() as u32;
    assert_eq!(wanted, 4, "ceil(48 / 13)");
    assert_eq!(VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE, 384);

    let cases = [
        (0u32, vec![1u32]),
        (383, vec![1]),
        (700, vec![1]),
        (767, vec![1]),
        (768, vec![2]),
        (1_151, vec![2]),
        (1_152, vec![3]),
        (1_200, vec![3]),
        (1_535, vec![3]),
        (1_536, vec![4]),
        (3_071, vec![4]),
        (3_072, vec![8, 4]),
        (6_143, vec![8, 4]),
        (6_144, vec![16, 8, 4]),
        (7_818, vec![16, 8, 4]),
        (12_287, vec![16, 8, 4]),
        (12_288, vec![32, 16, 8, 4]),
    ];
    for (long_side, expected) in cases {
        assert_eq!(
            virtual_tile_pyramid_factors(long_side),
            expected,
            "long side {long_side}"
        );
    }

    for long_side in (0..20_000u32).step_by(37) {
        let factors = virtual_tile_pyramid_factors(long_side);
        let largest = (long_side / VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE).max(1);
        let finest = *factors.last().expect("at least one level");
        assert_eq!(finest, wanted.min(largest), "long side {long_side}");
        assert!(
            factors.windows(2).all(|pair| pair[0] == 2 * pair[1]),
            "octaves coarsest first: {factors:?}"
        );
        assert!(factors[0] <= largest && factors[0] * 2 > largest);
        if factors[0] > 1 {
            assert!(long_side / factors[0] >= VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. match planes and pyramid
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_match_plane_rounds_block_means_and_never_leaks_uncovered_payload() {
    // Three 2x2 blocks: means 1.75 -> 2, 1.25 -> 1, 1.5 -> 2 (round half up).
    let gray = GrayImage::from_raw(6, 2, vec![1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2]).unwrap();
    let plane = VirtualTileMatchPlane::from_native(&gray, &full_coverage(6, 2), 2).unwrap();
    assert_eq!(plane.gray.dimensions(), (3, 1));
    assert_eq!(plane.gray.as_raw(), &vec![2, 1, 2]);
    assert_eq!(coverage_bits(&plane.coverage), vec![true, true, true]);

    // One uncovered pixel carrying a bright payload uncovers only its block.
    let mut leaking = gray.clone();
    leaking.put_pixel(3, 1, image::Luma([255]));
    let coverage = coverage_from_fn(6, 2, |x, y| (x, y) != (3, 1));
    let plane = VirtualTileMatchPlane::from_native(&leaking, &coverage, 2).unwrap();
    assert_eq!(plane.gray.as_raw(), &vec![2, 0, 2]);
    assert_eq!(coverage_bits(&plane.coverage), vec![true, false, true]);

    // Reference block average on a larger plane with scattered holes.
    let (width, height) = (9u32, 7u32);
    let uncovered = [(1u32, 0u32), (6, 5), (4, 4)];
    let mut gray = GrayImage::from_fn(width, height, |x, y| {
        image::Luma([((x * 37 + y * 11) % 200) as u8 + 20])
    });
    for &(x, y) in &uncovered {
        gray.put_pixel(x, y, image::Luma([255]));
    }
    let coverage = coverage_from_fn(width, height, |x, y| !uncovered.contains(&(x, y)));
    for factor in [2u32, 3] {
        let plane = VirtualTileMatchPlane::from_native(&gray, &coverage, factor).unwrap();
        assert_eq!(plane.factor, factor);
        assert_eq!(plane.gray.dimensions(), (width / factor, height / factor));
        assert_eq!(plane.coverage.dimensions(), plane.gray.dimensions());
        for level_y in 0..height / factor {
            for level_x in 0..width / factor {
                let block = (0..factor)
                    .flat_map(|dy| {
                        (0..factor).map(move |dx| (level_x * factor + dx, level_y * factor + dy))
                    })
                    .collect::<Vec<_>>();
                let all_covered = block.iter().all(|&(x, y)| coverage.is_covered(x, y));
                let expected = if all_covered {
                    let sum = block
                        .iter()
                        .map(|&(x, y)| u32::from(gray.get_pixel(x, y)[0]))
                        .sum::<u32>();
                    (f64::from(sum) / f64::from(factor * factor)).round() as u8
                } else {
                    0
                };
                assert_eq!(
                    plane.coverage.is_covered(level_x, level_y),
                    all_covered,
                    "factor {factor} level ({level_x}, {level_y})"
                );
                assert_eq!(
                    plane.gray.get_pixel(level_x, level_y)[0],
                    expected,
                    "factor {factor} level ({level_x}, {level_y})"
                );
            }
        }
    }
}

#[test]
fn virtual_tile_match_plane_identity_factor_floor_dimensions_and_mismatches() {
    let (width, height) = (9u32, 7u32);
    let gray = textured_plane(width, height);
    let coverage = coverage_from_fn(width, height, |x, y| (x + y) % 5 != 0);
    for factor in [0u32, 1] {
        let plane = VirtualTileMatchPlane::from_native(&gray, &coverage, factor).unwrap();
        assert_eq!(plane.factor, 1);
        assert_eq!(plane.gray, gray, "factor {factor} keeps native pixels");
        assert_eq!(coverage_bits(&plane.coverage), coverage_bits(&coverage));
    }
    assert_eq!(
        VirtualTileMatchPlane::from_native(&gray, &coverage, 4)
            .unwrap()
            .gray
            .dimensions(),
        (2, 1)
    );
    assert!(
        VirtualTileMatchPlane::from_native(&gray, &coverage, 8).is_none(),
        "a level without rows is unusable"
    );
    let narrower = full_coverage(width - 1, height);
    for factor in [1u32, 2] {
        assert!(VirtualTileMatchPlane::from_native(&gray, &narrower, factor).is_none());
    }
}

#[test]
fn virtual_tile_match_plane_level_and_native_maps_are_exact_inverses_on_block_centres() {
    for factor in [0u32, 1, 2, 3, 4, 8, 16] {
        let to_native = VirtualTileMatchPlane::level_to_native(factor);
        let to_level = VirtualTileMatchPlane::native_to_level(factor);
        assert!((to_native * to_level - Matrix3::identity()).amax() < 1e-12);
        assert!((to_level * to_native - Matrix3::identity()).amax() < 1e-12);
        let scale = f64::from(factor.max(1));
        for (x, y) in [(0.0, 0.0), (3.0, 2.0), (17.0, 5.0), (0.25, 9.5)] {
            let native = transformed_point(&to_native, Point2::new(x, y)).unwrap();
            assert!((native.x - (scale * x + (scale - 1.0) * 0.5)).abs() < 1e-12);
            assert!((native.y - (scale * y + (scale - 1.0) * 0.5)).abs() < 1e-12);
            let back = transformed_point(&to_level, native).unwrap();
            assert!((back - Point2::new(x, y)).norm() < 1e-12);
        }
    }
    // Level pixel (0, 0) of a 4x level is the centre of native block [0, 4).
    let centre =
        transformed_point(&VirtualTileMatchPlane::level_to_native(4), Point2::origin()).unwrap();
    assert_eq!((centre.x, centre.y), (1.5, 1.5));
}

#[test]
fn virtual_tile_match_pyramid_set_station_fills_every_level_of_that_station_only() {
    let factors = vec![8u32, 4];
    let mut pyramid = VirtualTileMatchPyramid::new(factors.clone(), 3);
    assert_eq!(pyramid.factors, factors);
    assert_eq!(pyramid.levels.len(), 2);
    assert_eq!(pyramid.coverages.len(), 2);
    assert!(pyramid.levels.iter().all(|level| level.len() == 3));
    assert!(pyramid.coverages.iter().all(|level| level.len() == 3));
    assert_eq!(pyramid.native_dimensions, vec![(1, 1); 3]);
    assert_eq!(pyramid.quality_coverages.len(), 3);

    let (width, height) = (130u32, 70u32);
    let gray = textured_plane(width, height);
    let coverage = coverage_from_fn(width, height, |x, y| x >= 9 || y >= 5);
    let mut tile = plane_info(42, gray.clone());
    tile.scale_factor = 3.0;
    let planes = pyramid
        .set_station(1, &tile, &gray, &coverage)
        .expect("both levels are usable");
    assert_eq!(
        planes.iter().map(|plane| plane.factor).collect::<Vec<_>>(),
        factors
    );
    for (level, plane) in planes.iter().enumerate() {
        let factor = factors[level];
        let expected = VirtualTileMatchPlane::from_native(&gray, &coverage, factor).unwrap();
        assert_eq!(plane.gray, expected.gray);
        let info = &pyramid.levels[level][1];
        assert_eq!(info.id, 42);
        assert_eq!((info.width, info.height), (width / factor, height / factor));
        assert_eq!(info.alignment_image, expected.gray);
        assert_eq!(info.scale_factor, 1.0, "a level is its own pixel grid");
        assert_eq!(
            coverage_bits(&pyramid.coverages[level][1]),
            coverage_bits(&expected.coverage)
        );
        for untouched in [0, 2] {
            assert_eq!(pyramid.levels[level][untouched].id, usize::MAX);
            assert_eq!(
                pyramid.levels[level][untouched]
                    .alignment_image
                    .dimensions(),
                (1, 1)
            );
            assert_eq!(pyramid.coverages[level][untouched].dimensions(), (1, 1));
        }
    }
    assert_eq!(
        pyramid.native_dimensions,
        vec![(1, 1), (width, height), (1, 1)]
    );

    let tiny = GrayImage::new(6, 6);
    assert!(
        pyramid
            .set_station(2, &tile, &tiny, &full_coverage(6, 6))
            .is_none(),
        "a level below one pixel makes the station unusable"
    );
}

// ---------------------------------------------------------------------------
// 3. relation-quality plane and its installation
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_relation_quality_plane_blurs_the_plane_two_octaves_above_the_finest() {
    assert_eq!(VIRTUAL_TILE_RELATION_QUALITY_OCTAVES, 2);
    assert_eq!(VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA, 1.0);
    let gray = textured_plane(512, 256);
    let coverage = full_coverage(512, 256);

    let (empty, empty_factor) = virtual_tile_relation_quality_plane(&[], &gray, &coverage);
    assert_eq!((empty.dimensions(), empty_factor), ((1, 1), 1));

    for factor in [1u32, 4] {
        let planes = planes_for(&[factor], &gray, &coverage);
        let (plane, plane_factor) = virtual_tile_relation_quality_plane(&planes, &gray, &coverage);
        assert_eq!(plane_factor, 4);
        let structural = VirtualTileMatchPlane::from_native(&gray, &coverage, 4).unwrap();
        assert_eq!(
            plane,
            imageproc::filter::gaussian_blur_f32(&structural.gray, 1.0),
            "a single matching level still verifies covered structure"
        );
    }

    for (factors, expected) in [
        (vec![16u32, 8, 4], 16u32),
        (vec![8, 4], 8),
        (vec![2, 1], 2),
        (vec![4, 2], 4),
        (vec![32, 16, 8, 4], 16),
    ] {
        let planes = planes_for(&factors, &gray, &coverage);
        let (plane, plane_factor) = virtual_tile_relation_quality_plane(&planes, &gray, &coverage);
        assert_eq!(plane_factor, expected, "factors {factors:?}");
        let source = planes
            .iter()
            .find(|plane| plane.factor == expected)
            .unwrap();
        assert_eq!(
            plane,
            imageproc::filter::gaussian_blur_f32(&source.gray, 1.0),
            "factors {factors:?}"
        );
        assert_ne!(plane, source.gray, "factors {factors:?} must be blurred");
    }
}

#[test]
fn install_virtual_tile_match_planes_pairs_the_quality_plane_with_its_own_coverage() {
    let (width, height) = (320u32, 200u32);
    let coverage = coverage_from_fn(width, height, |x, y| x >= 37 && y < 170);
    let texture = textured_plane(width, height);
    let gray = GrayImage::from_fn(width, height, |x, y| {
        if coverage.is_covered(x, y) {
            *texture.get_pixel(x, y)
        } else {
            image::Luma([0])
        }
    });
    let brief_pairs = processing::generate_brief_pairs();

    let mut pyramid = VirtualTileMatchPyramid::new(vec![8, 4], 2);
    let mut tile = plane_info(7, gray.clone());
    install_virtual_tile_match_planes(&mut tile, &mut pyramid, 1, &gray, &coverage, &brief_pairs)
        .expect("installed");
    let quality_level = VirtualTileMatchPlane::from_native(&gray, &coverage, 8).unwrap();
    assert_eq!(tile.scale_factor, 8.0);
    assert_eq!(tile.alignment_image.dimensions(), (40, 25));
    assert_eq!(
        tile.alignment_image,
        imageproc::filter::gaussian_blur_f32(&quality_level.gray, 1.0)
    );
    assert_eq!(
        pyramid.quality_coverages[1].dimensions(),
        tile.alignment_image.dimensions()
    );
    assert_eq!(
        coverage_bits(&pyramid.quality_coverages[1]),
        coverage_bits(&quality_level.coverage)
    );
    assert_eq!(pyramid.quality_coverages[0].dimensions(), (1, 1));
    let feature_plane =
        VirtualTileMatchPlane::from_native(&gray, &coverage, pyramid.feature_factors[1]).unwrap();
    assert!(tile.features.iter().all(|feature| {
        feature.keypoint.x < feature_plane.gray.width()
            && feature.keypoint.y < feature_plane.gray.height()
    }));

    let mut single = VirtualTileMatchPyramid::new(vec![1], 1);
    let mut native_tile = plane_info(8, gray.clone());
    install_virtual_tile_match_planes(
        &mut native_tile,
        &mut single,
        0,
        &gray,
        &coverage,
        &brief_pairs,
    )
    .expect("installed");
    let structural = VirtualTileMatchPlane::from_native(&gray, &coverage, 3).unwrap();
    assert_eq!(native_tile.scale_factor, 3.0);
    assert_eq!(
        native_tile.alignment_image,
        imageproc::filter::gaussian_blur_f32(&structural.gray, 1.0)
    );
    assert_eq!(
        single.quality_coverages[0].dimensions(),
        native_tile.alignment_image.dimensions()
    );
    assert_eq!(
        coverage_bits(&single.quality_coverages[0]),
        coverage_bits(&structural.coverage)
    );

    let mut failing = VirtualTileMatchPyramid::new(vec![8, 4], 1);
    let tiny = GrayImage::new(5, 5);
    let mut tiny_tile = plane_info(9, tiny.clone());
    assert!(
        install_virtual_tile_match_planes(
            &mut tiny_tile,
            &mut failing,
            0,
            &tiny,
            &full_coverage(5, 5),
            &brief_pairs,
        )
        .is_err()
    );
}

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
// 5. rendering-prior repair planning
// ---------------------------------------------------------------------------

#[test]
fn prior_free_seed_fallback_requires_a_fitted_polish_support_failure() {
    assert!(virtual_tile_prior_free_seed_fallback_conditions(
        true,
        "insufficient_polished_support",
        STATION_RELATION_MIN_INLIERS,
        STATION_RELATION_MIN_INLIERS,
    ));
    for (model_fitted, stage, fitted, prior_free) in [
        (
            false,
            "insufficient_polished_support",
            STATION_RELATION_MIN_INLIERS,
            STATION_RELATION_MIN_INLIERS,
        ),
        (
            true,
            "fitted",
            STATION_RELATION_MIN_INLIERS,
            STATION_RELATION_MIN_INLIERS,
        ),
        (
            true,
            "insufficient_polished_support",
            STATION_RELATION_MIN_INLIERS - 1,
            STATION_RELATION_MIN_INLIERS,
        ),
        (
            true,
            "insufficient_polished_support",
            STATION_RELATION_MIN_INLIERS,
            STATION_RELATION_MIN_INLIERS - 1,
        ),
    ] {
        assert!(!virtual_tile_prior_free_seed_fallback_conditions(
            model_fitted,
            stage,
            fitted,
            prior_free,
        ));
    }
}

#[test]
fn plan_virtual_tile_prior_repairs_needs_proposals_and_two_clusters() {
    let tile_to_world = sample_tile_to_world();
    let topology = line_topology(4);
    let relation = sample_relation(-372.0, 9.0);
    assert!(
        plan_virtual_tile_prior_repairs(4, &[(0, 1), (2, 3)], &[], &tile_to_world, &topology)
            .is_empty()
    );
    assert!(
        plan_virtual_tile_prior_repairs(
            4,
            &[(0, 1), (1, 2), (2, 3)],
            &[prior_proposal(1, 2, relation, 50)],
            &tile_to_world,
            &topology,
        )
        .is_empty(),
        "one authoritative cluster needs no repair"
    );
    assert!(
        plan_virtual_tile_prior_repairs(
            3,
            &[],
            &[prior_proposal(0, 1, relation, 50)],
            &tile_to_world,
            &topology,
        )
        .is_empty(),
        "placements must match the station count"
    );
    assert!(
        plan_virtual_tile_prior_repairs(
            1,
            &[],
            &[prior_proposal(0, 1, relation, 50)],
            &tile_to_world[..1],
            &topology,
        )
        .is_empty()
    );
}

#[test]
fn authoritative_quality_gates_control_production_cluster_and_proposal_behavior() {
    let dimensions = (96, 96);
    let coverage = full_coverage(dimensions.0, dimensions.1);
    let quality_coverages = vec![coverage.clone(), coverage.clone()];
    let overlap_coverages = vec![coverage.clone(), coverage];
    let coverages = VirtualTileSolverCoverages {
        quality: &quality_coverages,
        overlap: &overlap_coverages,
        overlap_factor: 1,
    };
    let points = (0..24)
        .map(|index| {
            let x = 8.0 + f64::from(index % 8) * 10.0;
            let y = 8.0 + f64::from(index / 8) * 30.0;
            (Point2::new(x, y), Point2::new(x, y))
        })
        .collect::<Vec<_>>();
    let relation = measured_relation(Matrix3::identity(), points);
    let mut topology = line_topology(2);
    topology.candidates.push(topology::CandidateAdjacency {
        left: 0,
        right: 1,
        kind: report::AdjacencyKind::SameRow,
        score: 1.0,
    });
    let proposal_pairs = |tiles: &[ImageInfo], relation: MatchInfo| {
        let authoritative = virtual_tile_authoritative_pairs(
            &HashMap::from([((0, 1), vec![relation])]),
            tiles,
            coverages,
        );
        (
            authoritative.clone(),
            virtual_tile_prior_free_candidate_pairs(tiles.len(), &authoritative, &topology),
        )
    };

    let texture = textured_plane(dimensions.0, dimensions.1);
    let mut reversed_tiles = vec![
        plane_info(0, texture.clone()),
        plane_info(1, texture.clone()),
    ];
    reversed_tiles[0].filename = "virtual://z".to_string();
    reversed_tiles[1].filename = "virtual://a".to_string();
    let (authoritative, proposals) = proposal_pairs(&reversed_tiles, relation.clone());
    assert_eq!(
        authoritative,
        [(0, 1)],
        "reversed canonical direction remains authoritative"
    );
    assert!(
        proposals.is_empty(),
        "an accepted pair merges the production clusters"
    );

    let low_contrast = GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
        image::Luma([40 + texture.get_pixel(x, y)[0] / 4])
    });
    let brighter = GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
        image::Luma([140 + texture.get_pixel(x, y)[0] / 4])
    });
    let photometric_tiles = vec![plane_info(0, low_contrast), plane_info(1, brighter)];
    let (authoritative, proposals) = proposal_pairs(&photometric_tiles, relation.clone());
    assert!(
        authoritative.is_empty(),
        "photometric mismatch cannot merge clusters"
    );
    assert_eq!(
        proposals,
        [(0, 1)],
        "photometric failure keeps the prior-free proposal"
    );

    let weak_edges = GrayImage::from_fn(dimensions.0, dimensions.1, |x, y| {
        let value = i16::from(texture.get_pixel(x, y)[0]);
        image::Luma([(128 + (value - 128) / 4) as u8])
    });
    let edge_tiles = vec![plane_info(0, texture.clone()), plane_info(1, weak_edges)];
    let (authoritative, proposals) = proposal_pairs(&edge_tiles, relation.clone());
    assert!(
        authoritative.is_empty(),
        "edge-strength mismatch cannot merge clusters"
    );
    assert_eq!(
        proposals,
        [(0, 1)],
        "edge failure keeps the prior-free proposal"
    );

    let flat = GrayImage::from_pixel(dimensions.0, dimensions.1, image::Luma([128]));
    let orientation_missing_tiles = vec![plane_info(0, flat.clone()), plane_info(1, flat)];
    let (authoritative, proposals) = proposal_pairs(&orientation_missing_tiles, relation);
    assert!(
        authoritative.is_empty(),
        "missing dense orientation cannot merge clusters"
    );
    assert_eq!(
        proposals,
        [(0, 1)],
        "missing orientation keeps the prior-free proposal"
    );
}

#[test]
fn authoritative_pairs_keep_the_inclusive_canonical_scale_contract() {
    let dimensions = (96, 96);
    let gray = textured_plane(dimensions.0, dimensions.1);
    let mut tiles = vec![plane_info(0, gray.clone()), plane_info(1, gray)];
    tiles[0].filename = "virtual://z".to_string();
    tiles[1].filename = "virtual://a".to_string();
    let coverage = full_coverage(dimensions.0, dimensions.1);
    let quality_coverages = vec![coverage.clone(), coverage.clone()];
    let overlap_coverages = vec![coverage.clone(), coverage];
    let coverages = VirtualTileSolverCoverages {
        quality: &quality_coverages,
        overlap: &overlap_coverages,
        overlap_factor: 1,
    };
    let points = (0..24)
        .map(|index| {
            let x = 8.0 + f64::from(index % 8) * 10.0;
            let y = 8.0 + f64::from(index / 8) * 30.0;
            (Point2::new(x, y), Point2::new(x, y))
        })
        .collect::<Vec<_>>();

    assert_eq!(STATION_RELATION_SCALE_RATIO_MIN, 0.95);
    assert_eq!(STATION_RELATION_SCALE_RATIO_MAX, 1.05);
    assert_eq!(
        virtual_tile_authoritative_pairs(
            &HashMap::from([(
                (0, 1,),
                vec![measured_relation(Matrix3::identity(), points.clone())],
            )]),
            &tiles,
            coverages,
        ),
        [(0, 1)]
    );

    for canonical_scale in [
        f64::from_bits(STATION_RELATION_SCALE_RATIO_MIN.to_bits() - 1),
        f64::from_bits(STATION_RELATION_SCALE_RATIO_MAX.to_bits() + 1),
    ] {
        let relation = measured_relation(uniform_scale(1.0 / canonical_scale), points.clone());
        assert!(
            virtual_tile_authoritative_pairs(
                &HashMap::from([((0, 1), vec![relation])]),
                &tiles,
                coverages,
            )
            .is_empty(),
            "canonical scale {canonical_scale}"
        );
    }
}

#[test]
fn virtual_tile_non_invertible_canonical_transform_is_summarized_once() {
    let _run_scope = degradation::begin_run_scope();
    let mut report = report::StationRelationsReport::default();
    report.record_non_invertible_canonical_pair(0, 1);
    report.record_non_invertible_canonical_pair(1, 0);
    report.candidates = vec![sample_candidate_record()];
    report.summarize_candidate_measurements();
    assert_eq!(report.rejected_count, 1);
    assert_eq!(report.unmeasured_count, 0);
    assert_eq!(
        report.rejected_by_reason[degradation::STATION_RELATION_NON_INVERTIBLE_CANONICAL_TRANSFORM],
        1
    );
    assert_eq!(
        report.rejected_by_failure_stage["canonical_transform_non_invertible"],
        1
    );
    assert_eq!(
        report.candidates[0].failure_stage,
        "canonical_transform_non_invertible"
    );
    assert!(report.candidates[0].model_fitted);
    assert!(
        !report
            .rejected_by_reason
            .contains_key(degradation::STATION_RELATION_SCALE_OUT_OF_RANGE)
    );
}

#[test]
fn virtual_tile_connectivity_reports_original_source_members() {
    let stations = vec![
        vec!["/raw/a.nef".to_string(), "/raw/b.nef".to_string()],
        vec!["/raw/c.nef".to_string()],
        vec!["/raw/d.nef".to_string(), "/raw/e.nef".to_string()],
    ];
    assert_eq!(
        station_graph_components_from_members(&stations, &[(0, 1)]),
        vec![
            vec![
                "/raw/a.nef".to_string(),
                "/raw/b.nef".to_string(),
                "/raw/c.nef".to_string(),
            ],
            vec!["/raw/d.nef".to_string(), "/raw/e.nef".to_string()],
        ]
    );
}

#[test]
fn prior_free_relation_uses_stable_ransac_for_large_projective_coordinates() {
    let dimensions = (30_000, 30_000);
    let expected = Matrix3::new(
        1.018, 0.012, 73.0, -0.008, 0.997, -41.0, 1.5e-6, -1.2e-6, 1.0,
    );
    let points = (0..32)
        .map(|index| {
            let source = Point2::new(
                10_000.0 + f64::from(index % 8) * 230.0,
                20_000.0 + f64::from(index / 8) * 190.0,
            );
            let target = transformed_point(&expected, source).expect("finite target");
            (source, target)
        })
        .collect::<Vec<_>>();
    let mut all_points = points.clone();
    all_points.extend((0..12).map(|index| {
        (
            Point2::new(10_100.0 + f64::from(index) * 120.0, 24_500.0),
            Point2::new(
                400.0 + f64::from(index) * 137.0,
                600.0 + f64::from(index) * 29.0,
            ),
        )
    }));
    let descriptor_for = |index: usize| {
        let mut descriptor = [0u8; BRIEF_DESCRIPTOR_SIZE / 8];
        descriptor[index / 8] = 1u8 << (index % 8);
        descriptor
    };
    let left_features = all_points
        .iter()
        .enumerate()
        .map(|(index, &(source, _))| Feature {
            keypoint: KeyPoint {
                x: source.x as u32,
                y: source.y as u32,
            },
            descriptor: descriptor_for(index),
            support_scale: 1.0,
        })
        .collect::<Vec<_>>();
    let right_features = all_points
        .iter()
        .enumerate()
        .map(|(index, &(_, target))| Feature {
            keypoint: KeyPoint {
                x: target.x as u32,
                y: target.y as u32,
            },
            descriptor: descriptor_for(index),
            support_scale: 1.0,
        })
        .collect::<Vec<_>>();
    let (relation, inliers) =
        virtual_tile_prior_free_relation(1, &left_features, 1, &right_features, dimensions)
            .expect("stable RANSAC recovers the large-coordinate projective relation");
    assert_eq!(inliers, points.len());
    let probe = Point2::new(11_234.0, 20_456.0);
    let expected_probe = transformed_point(&expected, probe).expect("finite expected probe");
    let actual_probe = transformed_point(&relation, probe).expect("finite fitted probe");
    assert!((actual_probe - expected_probe).norm() < 2.0);
}

#[test]
fn plan_virtual_tile_prior_repairs_moves_the_whole_unfixed_cluster_onto_the_proposal() {
    let tile_to_world = sample_tile_to_world();
    let topology = line_topology(4);
    let authoritative = [(0, 1), (2, 3)];
    let relation = sample_relation(-372.0, 9.0);

    // The fixed station is the proposal's left: H(u→f) = (left→right)⁻¹.
    let repairs = plan_virtual_tile_prior_repairs(
        4,
        &authoritative,
        &[prior_proposal(1, 2, relation, 50)],
        &tile_to_world,
        &topology,
    );
    assert_eq!(repaired_stations(&repairs), vec![2, 3]);
    for repair in &repairs {
        assert_eq!(repair.fixed_station, 1);
        assert_eq!(repair.relation, (1, 2));
        assert_eq!(repair.relation_inliers, 50);
        assert_eq!(repair.world_correction, repairs[0].world_correction);
    }
    let correction = repairs[0].world_correction;
    assert_projectively_equal(
        &(correction * tile_to_world[2]),
        &(tile_to_world[1] * relation.try_inverse().unwrap()),
        "left fixed",
    );

    // The fixed station is the proposal's right: H(u→f) = left→right.
    let repairs = plan_virtual_tile_prior_repairs(
        4,
        &authoritative,
        &[prior_proposal(2, 1, relation, 50)],
        &tile_to_world,
        &topology,
    );
    assert_eq!(repaired_stations(&repairs), vec![2, 3]);
    assert!(repairs.iter().all(|repair| repair.fixed_station == 1
        && repair.relation == (2, 1)
        && repair.world_correction == repairs[0].world_correction));
    assert_projectively_equal(
        &(repairs[0].world_correction * tile_to_world[2]),
        &(tile_to_world[1] * relation),
        "right fixed",
    );
}

#[test]
fn plan_virtual_tile_prior_repairs_keeps_the_largest_cluster_and_breaks_ties_by_position() {
    let tile_to_world = sample_tile_to_world();
    let relation = sample_relation(-372.0, 9.0);
    let proposals = [prior_proposal(0, 1, relation, 40)];

    // {0} against {1, 2}: the larger cluster is the reference.
    let repairs = plan_virtual_tile_prior_repairs(
        3,
        &[(1, 2)],
        &proposals,
        &tile_to_world[..3],
        &line_topology(3),
    );
    assert_eq!(repaired_stations(&repairs), vec![0]);
    assert_eq!(repairs[0].fixed_station, 1);

    // Equal sizes: the smallest row/column key is the reference.
    let repairs =
        plan_virtual_tile_prior_repairs(2, &[], &proposals, &tile_to_world[..2], &line_topology(2));
    assert_eq!(repaired_stations(&repairs), vec![1]);
    assert_eq!(repairs[0].fixed_station, 0);
    let reordered = topology::StationTopology {
        row: vec![1, 0],
        column: vec![0, 3],
        ..topology::StationTopology::default()
    };
    let repairs =
        plan_virtual_tile_prior_repairs(2, &[], &proposals, &tile_to_world[..2], &reordered);
    assert_eq!(repaired_stations(&repairs), vec![0]);
    assert_eq!(repairs[0].fixed_station, 1);
}

#[test]
fn plan_virtual_tile_prior_repairs_prefers_inliers_and_composes_repairs_transitively() {
    let tile_to_world = sample_tile_to_world();
    let weak_relation = sample_relation(-372.0, 9.0);
    let strong_relation = sample_relation(-375.5, 6.0);
    let weak = prior_proposal(0, 1, weak_relation, 50);
    let strong = prior_proposal(0, 1, strong_relation, 80);
    let topology = line_topology(2);
    let repairs =
        plan_virtual_tile_prior_repairs(2, &[], &[weak, strong], &tile_to_world[..2], &topology);
    assert_eq!(
        repaired_stations(&repairs),
        vec![1],
        "repaired at most once"
    );
    assert_eq!(repairs[0].relation_inliers, 80);
    assert_projectively_equal(
        &(repairs[0].world_correction * tile_to_world[1]),
        &(tile_to_world[0] * strong_relation.try_inverse().unwrap()),
        "strongest proposal",
    );
    assert_eq!(
        plan_virtual_tile_prior_repairs(2, &[], &[strong, weak], &tile_to_world[..2], &topology),
        repairs,
        "proposal order is irrelevant"
    );

    // Three singleton clusters: 0→2 (90) first, then 2→1 (60) beats 0→1 (40)
    // and composes station 2's correction.
    let relation_01 = sample_relation(-372.0, 9.0);
    let relation_02 = sample_relation(-765.0, 14.0);
    let relation_21 = sample_relation(391.0, 12.0);
    let repairs = plan_virtual_tile_prior_repairs(
        3,
        &[],
        &[
            prior_proposal(0, 1, relation_01, 40),
            prior_proposal(0, 2, relation_02, 90),
            prior_proposal(2, 1, relation_21, 60),
        ],
        &tile_to_world[..3],
        &line_topology(3),
    );
    assert_eq!(repaired_stations(&repairs), vec![1, 2], "sorted by station");
    let (repair_1, repair_2) = (&repairs[0], &repairs[1]);
    assert_eq!(
        (
            repair_2.fixed_station,
            repair_2.relation,
            repair_2.relation_inliers
        ),
        (0, (0, 2), 90)
    );
    assert_projectively_equal(
        &(repair_2.world_correction * tile_to_world[2]),
        &(tile_to_world[0] * relation_02.try_inverse().unwrap()),
        "first repair",
    );
    assert_eq!(
        (
            repair_1.fixed_station,
            repair_1.relation,
            repair_1.relation_inliers
        ),
        (2, (2, 1), 60)
    );
    assert_projectively_equal(
        &(repair_1.world_correction * tile_to_world[1]),
        &(repair_2.world_correction * tile_to_world[2] * relation_21.try_inverse().unwrap()),
        "transitive repair",
    );
}

#[test]
fn plan_virtual_tile_prior_repairs_skips_an_unbounded_strong_proposal_for_a_bounded_fallback() {
    let tile_to_world = sample_tile_to_world();
    let topology = line_topology(2);
    let bounded = sample_relation(-372.0, 9.0);
    // Station 0 is fixed. This relation would require station 1's world pose
    // to shrink by 20%, outside the existing inclusive [0.95, 1.05] envelope.
    let unbounded = uniform_scale(1.25) * bounded;
    let repairs = plan_virtual_tile_prior_repairs(
        2,
        &[],
        &[
            prior_proposal(0, 1, unbounded, 99),
            prior_proposal(0, 1, bounded, 30),
        ],
        &tile_to_world[..2],
        &topology,
    );

    assert_eq!(repaired_stations(&repairs), vec![1]);
    assert_eq!(repairs[0].relation_inliers, 30);
    let correction_scale = homography_scale_ratio(&repairs[0].world_correction).unwrap();
    assert!(
        (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
            .contains(&correction_scale),
        "fallback correction scale {correction_scale}"
    );
    assert_projectively_equal(
        &(repairs[0].world_correction * tile_to_world[1]),
        &(tile_to_world[0] * bounded.try_inverse().unwrap()),
        "bounded fallback",
    );
}

#[test]
fn plan_virtual_tile_prior_repairs_ignores_invalid_proposals_and_skips_singular_relations() {
    let tile_to_world = sample_tile_to_world();
    let topology = line_topology(2);
    let relation = sample_relation(-372.0, 9.0);
    assert!(
        plan_virtual_tile_prior_repairs(
            2,
            &[],
            &[
                prior_proposal(0, 2, relation, 99),
                prior_proposal(5, 0, relation, 99),
                prior_proposal(1, 1, relation, 99),
            ],
            &tile_to_world[..2],
            &topology,
        )
        .is_empty()
    );
    // Station 0 is the reference. A zero relation cannot be inverted (fixed
    // left) nor normalised (fixed right); the valid weaker proposal repairs.
    let repairs = plan_virtual_tile_prior_repairs(
        2,
        &[],
        &[
            prior_proposal(0, 1, Matrix3::zeros(), 99),
            prior_proposal(1, 0, Matrix3::zeros(), 98),
            prior_proposal(0, 1, relation, 30),
        ],
        &tile_to_world[..2],
        &topology,
    );
    assert_eq!(repaired_stations(&repairs), vec![1]);
    assert_eq!(repairs[0].relation_inliers, 30);
    assert_projectively_equal(
        &(repairs[0].world_correction * tile_to_world[1]),
        &(tile_to_world[0] * relation.try_inverse().unwrap()),
        "valid fallback",
    );
}

// ---------------------------------------------------------------------------
// 6. bounded residual correction
// ---------------------------------------------------------------------------

#[test]
fn virtual_tile_residual_correction_is_bounded_reads_local_scale_over_the_evidence_region() {
    assert_eq!(VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX, 24.0);
    assert_eq!(VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX, 8.0);
    assert_eq!(VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MIN, 0.98);
    assert_eq!(VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MAX, 1.02);
    let identity = Matrix3::identity();
    let evidence = (400.0, 200.0, 600.0, 400.0);
    let centre = Point2::new(500.0, 300.0);

    // Upper-left block 1.035, but sqrt(det J) = a / w^1.5 = 1 at the centre.
    let block: f64 = 1.035;
    let perspective = (block.powf(2.0 / 3.0) - 1.0) / centre.x;
    let projective = Matrix3::new(block, 0.0, 0.0, 0.0, block, 0.0, perspective, 0.0, 1.0);
    assert!((numeric_local_scale(&projective, centre) - 1.0).abs() < 1e-6);
    assert!(virtual_tile_residual_correction_is_bounded(
        &projective,
        &identity,
        evidence
    ));
    assert_eq!(
        virtual_tile_full_tile_bounds((1000, 700)),
        (0.0, 0.0, 1000.0, 700.0)
    );
    assert!(
        !virtual_tile_residual_correction_is_bounded(
            &projective,
            &identity,
            virtual_tile_full_tile_bounds((1000, 700)),
        ),
        "the full tile reaches the origin, where the local scale is 1.035"
    );
    // The bounds are evaluated where the initial relation predicts the evidence.
    assert!(!virtual_tile_residual_correction_is_bounded(
        &projective,
        &translation(-300.0, 0.0),
        evidence
    ));
    let shifted_perspective = (block.powf(2.0 / 3.0) - 1.0) / 200.0;
    let shifted = Matrix3::new(
        block,
        0.0,
        0.0,
        0.0,
        block,
        0.0,
        shifted_perspective,
        0.0,
        1.0,
    );
    assert!(virtual_tile_residual_correction_is_bounded(
        &shifted,
        &translation(-300.0, 0.0),
        evidence
    ));
}

#[test]
fn virtual_tile_residual_correction_is_bounded_rejects_scale_displacement_and_non_translation() {
    let identity = Matrix3::identity();
    let evidence = (400.0, 200.0, 600.0, 400.0);
    let centre = (500.0, 300.0);
    let bounded = |correction: Matrix3<f64>| {
        virtual_tile_residual_correction_is_bounded(&correction, &identity, evidence)
    };
    // True scale about the centre moves corners by at most 4.3px.
    for (scale, expected) in [
        (1.0, true),
        (1.015, true),
        (1.019, true),
        (1.021, false),
        (1.03, false),
        (0.981, true),
        (0.979, false),
        (0.97, false),
    ] {
        assert_eq!(
            bounded(about_center(centre, uniform_scale(scale))),
            expected,
            "scale {scale}"
        );
    }
    for (dx, dy, expected) in [
        (23.5, 0.0, true),
        (0.0, -23.5, true),
        (24.5, 0.0, false),
        (17.0, 17.0, false),
    ] {
        assert_eq!(
            bounded(translation(dx, dy)),
            expected,
            "translation ({dx}, {dy})"
        );
    }
    // A rotation about the centre moves the corners by 2·sin(θ/2)·141.4px.
    for (theta, expected) in [(0.05, true), (0.06, false), (-0.06, false)] {
        assert_eq!(
            bounded(about_center(centre, rotation(theta))),
            expected,
            "rotation {theta}"
        );
    }
    // Each part passes alone; together a corner moves 25.6px.
    assert!(bounded(translation(20.0, 0.0)));
    assert!(!bounded(
        translation(20.0, 0.0) * about_center(centre, rotation(0.05))
    ));
}

// ---------------------------------------------------------------------------
// 9. intra-station frame records
// ---------------------------------------------------------------------------

#[test]
fn push_frame_record_replaces_a_re_measured_frame_in_place() {
    let frame = |path: &str, inliers: usize| report::IntraStationFrameRecord {
        path: path.to_string(),
        inliers,
        ..report::IntraStationFrameRecord::default()
    };
    let summary = |station: &report::IntraStationReport| {
        station
            .frames
            .iter()
            .map(|frame| (frame.path.clone(), frame.inliers))
            .collect::<Vec<_>>()
    };
    let mut records = Vec::new();
    intra_station::push_frame_record(&mut records, 3, "anchor-a.raw", frame("a.raw", 10));
    intra_station::push_frame_record(&mut records, 3, "anchor-a.raw", frame("b.raw", 20));
    intra_station::push_frame_record(&mut records, 5, "anchor-c.raw", frame("c.raw", 30));
    // A later render re-measures a.raw: replaced in place, anchor unchanged.
    intra_station::push_frame_record(&mut records, 3, "other-anchor.raw", frame("a.raw", 11));
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].station_index, 3);
    assert_eq!(records[0].anchor_path, "anchor-a.raw");
    assert_eq!(
        summary(&records[0]),
        vec![("a.raw".to_string(), 11), ("b.raw".to_string(), 20)]
    );
    intra_station::push_frame_record(&mut records, 3, "anchor-a.raw", frame("d.raw", 40));
    assert_eq!(
        summary(&records[0]),
        vec![
            ("a.raw".to_string(), 11),
            ("b.raw".to_string(), 20),
            ("d.raw".to_string(), 40)
        ]
    );
    assert_eq!(records[1].station_index, 5);
    assert_eq!(summary(&records[1]), vec![("c.raw".to_string(), 30)]);

    // An empty anchor is filled by the first record that names one.
    intra_station::push_frame_record(&mut records, 7, "", frame("e.raw", 1));
    assert_eq!(records[2].anchor_path, "");
    intra_station::push_frame_record(&mut records, 7, "anchor-e.raw", frame("e.raw", 2));
    assert_eq!(records[2].anchor_path, "anchor-e.raw");
    assert_eq!(summary(&records[2]), vec![("e.raw".to_string(), 2)]);
}

// ---------------------------------------------------------------------------
// 10. report serde defaults
// ---------------------------------------------------------------------------

fn sample_candidate_record() -> report::StationRelationCandidateRecord {
    let distribution = report::ResidualDistributionRecord {
        count: 3,
        min: -1.0,
        p10: -0.5,
        median: 0.25,
        p90: 0.75,
        max: 1.5,
    };
    let stage = |reason: &str| report::ResidualModelStageRecord {
        candidate_inliers: 42,
        final_support: 0.4,
        bounded_correction: Some(true),
        rejection_reason: reason.to_string(),
    };
    report::StationRelationCandidateRecord {
        left: 0,
        right: 1,
        left_tile: Default::default(),
        right_tile: Default::default(),
        predicted_overlap_area_px: 12_345.0,
        model_fitted: true,
        evidence_kind: report::StationRelationEvidenceKind::VirtualTile,
        grid_probe_count: 120,
        measurable_patch_count: 90,
        bidirectional_match_count: 70,
        occupied_grid_cells: 12,
        search_radius_px: 16.0,
        residual_dx_px: distribution,
        residual_dy_px: distribution,
        residual_magnitude_px: distribution,
        max_residual_translation_consensus: 60,
        residual_translation: stage("accepted"),
        residual_affine: stage("not_run"),
        residual_projective: stage("not_run"),
        residual_model: "polished_projective".to_string(),
        fitted_inliers: 64,
        hull_support: 0.31,
        median_prefit_structural_orientation_difference_degrees: Some(4.5),
        median_fitted_orientation_difference_degrees: None,
        failure_stage: "fitted".to_string(),
        pyramid_factors: vec![16, 8, 4],
        measurement_factor: 4,
        seed: "prior_free_features".to_string(),
        prior_free_inliers: 77,
        global_proposals: vec![report::StationRelationGlobalProposalRecord {
            rank: 0,
            score: 0.91,
            overlap_samples: 128,
            left_factor: 32,
            right_factor: 31,
            translation_x_px: 1_024.0,
            translation_y_px: -12.0,
            scale: 1.0,
            rotation_degrees: 0.0,
        }],
        after_prior_repair: true,
        hypotheses: Vec::new(),
    }
}

#[test]
fn station_relation_records_deserialise_without_checkpoint_ten_fields() {
    let candidate = sample_candidate_record();
    let value = serde_json::to_value(&candidate).unwrap();
    assert_eq!(
        serde_json::from_value::<report::StationRelationCandidateRecord>(value.clone()).unwrap(),
        candidate
    );
    let mut legacy = value;
    let object = legacy.as_object_mut().unwrap();
    for key in [
        "predicted_overlap_area_px",
        "model_fitted",
        "left_tile",
        "right_tile",
        "pyramid_factors",
        "measurement_factor",
        "seed",
        "prior_free_inliers",
        "global_proposals",
        "after_prior_repair",
        "hypotheses",
    ] {
        assert!(object.remove(key).is_some(), "{key} is serialised");
    }
    let parsed = serde_json::from_value::<report::StationRelationCandidateRecord>(legacy).unwrap();
    assert_eq!(
        parsed,
        report::StationRelationCandidateRecord {
            predicted_overlap_area_px: 0.0,
            model_fitted: false,
            left_tile: Default::default(),
            right_tile: Default::default(),
            pyramid_factors: Vec::new(),
            measurement_factor: 0,
            seed: String::new(),
            prior_free_inliers: 0,
            global_proposals: Vec::new(),
            after_prior_repair: false,
            hypotheses: Vec::new(),
            ..candidate.clone()
        }
    );

    let repair = report::StationPriorRepairRecord {
        station: 2,
        fixed_station: 1,
        relation_left: 1,
        relation_right: 2,
        relation_inliers: 64,
        center_shift_px: 12.5,
        correction_scale_ratio: 1.004,
    };
    let text = serde_json::to_string(&repair).unwrap();
    assert_eq!(
        serde_json::from_str::<report::StationPriorRepairRecord>(&text).unwrap(),
        repair
    );

    let source_raw_diagnostic = report::SourceRawPairDiagnosticRecord {
        left: 1,
        right: 2,
        layer_models: vec![source_raw_test_record(3, 7)],
        consensus_clusters: vec![report::SourceRawConsensusClusterRecord {
            cluster: 0,
            observation_count: 2,
            independent_support: 2,
            supporting_layer_pairs: vec![["left-3".to_string(), "right-7".to_string()]],
            median_disagreement_px: 0.5,
            scale_ratio: 1.0,
            spatial_support_potential: 0.4,
            scale_compliant: true,
            spatially_broad: true,
            anchor_transform: matrix3_row_major(&Matrix3::identity()),
        }],
    };
    let relations = report::StationRelationsReport {
        candidates: vec![candidate],
        source_raw_pair_diagnostics: vec![source_raw_diagnostic],
        prior_repairs: vec![repair],
        ..report::StationRelationsReport::default()
    };
    let value = serde_json::to_value(&relations).unwrap();
    assert_eq!(
        serde_json::from_value::<report::StationRelationsReport>(value.clone()).unwrap(),
        relations
    );
    let mut legacy = value;
    legacy.as_object_mut().unwrap().remove("prior_repairs");
    legacy
        .as_object_mut()
        .unwrap()
        .remove("source_raw_pair_diagnostics");
    assert_eq!(
        serde_json::from_value::<report::StationRelationsReport>(legacy).unwrap(),
        report::StationRelationsReport {
            source_raw_pair_diagnostics: Vec::new(),
            prior_repairs: Vec::new(),
            ..relations
        }
    );
}

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

/// Deterministic zero-mean pseudo-noise in `[-amplitude, amplitude]`.
fn hashed_noise(x: u32, y: u32, seed: u32, amplitude: f64) -> f64 {
    let mut value = x
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(y.wrapping_mul(0x85EB_CA77))
        .wrapping_add(seed.wrapping_mul(0xC2B2_AE3D));
    value ^= value >> 15;
    value = value.wrapping_mul(0x2C1B_3C6D);
    value ^= value >> 12;
    (f64::from(value % 2_001) / 1_000.0 - 1.0) * amplitude
}

fn candidate_topology(station_count: usize) -> topology::StationTopology {
    topology::StationTopology {
        row: vec![0; station_count],
        column: (0..station_count as u32).collect(),
        candidates: (0..station_count)
            .flat_map(|left| {
                (left + 1..station_count).map(move |right| topology::CandidateAdjacency {
                    left,
                    right,
                    kind: report::AdjacencyKind::SameRow,
                    score: 1.0,
                })
            })
            .collect(),
        ..topology::StationTopology::default()
    }
}

fn measured_relation(
    homography: Matrix3<f64>,
    points: Vec<(Point2<f64>, Point2<f64>)>,
) -> MatchInfo {
    MatchInfo {
        homography,
        inliers: points.len(),
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

fn hypothesis_points(
    transform: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> Vec<(Point2<f64>, Point2<f64>)> {
    (0..8)
        .flat_map(|row| {
            (0..8).map(move |column| {
                let source = Point2::new(
                    30.0 + column as f64 * (f64::from(dimensions.0) - 60.0) / 7.0,
                    30.0 + row as f64 * (f64::from(dimensions.1) - 60.0) / 7.0,
                );
                (source, transformed_point(transform, source).unwrap())
            })
        })
        .collect()
}

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

// ---------------------------------------------------------------------------
// 7. residual model selection
// ---------------------------------------------------------------------------

/// `rows × columns` measured pairs of `correction · initial` on a 500×400 tile.
fn residual_selection_points(
    initial: &Matrix3<f64>,
    correction: &Matrix3<f64>,
    columns: usize,
    rows: usize,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let measured = correction * initial;
    (0..rows)
        .flat_map(|row| {
            (0..columns).map(move |column| {
                let source = Point2::new(
                    40.0 + column as f64 * 420.0 / (columns - 1) as f64,
                    35.0 + row as f64 * 330.0 / (rows - 1) as f64,
                );
                (source, transformed_point(&measured, source).unwrap())
            })
        })
        .collect()
}

fn fit_residual_model(
    points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    selection: VirtualTileResidualSelection,
) -> VirtualTileResidualModelOutcome {
    virtual_tile_fit_measured_residual_model_with_threshold(
        points,
        initial,
        (500, 400),
        (500, 400),
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
        virtual_tile_full_tile_bounds((500, 400)),
        selection,
        StationOverlapMeasure::Rectangles,
    )
}

#[test]
fn most_inliers_selection_keeps_the_richer_model_a_passing_translation_would_truncate() {
    let initial = translation(-90.0, 5.0);
    // A 0.9° rotation about the tile centre: a translation explains only the
    // central band within 3px; the bounded affine explains every pair.
    let correction = translation(1.5, -2.0) * about_center((250.0, 200.0), rotation(0.0157));
    let points = residual_selection_points(&initial, &correction, 10, 8);

    let lowest = fit_residual_model(
        &points,
        &initial,
        VirtualTileResidualSelection::LowestComplexity,
    );
    let lowest_fit = lowest.fit.expect("the translation stage passes on its own");
    assert_eq!(lowest_fit.model, "translation");
    assert_eq!(lowest.translation.rejection_reason, "accepted");
    assert_eq!(lowest.affine.rejection_reason, "not_run");
    assert!(lowest_fit.inlier_indices.len() < points.len());

    let most = fit_residual_model(&points, &initial, VirtualTileResidualSelection::MostInliers);
    let most_fit = most.fit.expect("every stage is evaluated");
    assert_eq!(most.translation.rejection_reason, "accepted");
    assert_eq!(most.affine.rejection_reason, "accepted");
    assert_ne!(most_fit.model, "translation");
    assert_eq!(most_fit.inlier_indices.len(), points.len());
    assert!(most_fit.inlier_indices.len() > lowest_fit.inlier_indices.len());
    for (source, target) in &points {
        let predicted = transformed_point(&most_fit.measured_homography, *source).unwrap();
        assert!((predicted - target).norm() < 1e-6);
    }
}

#[test]
fn most_inliers_selection_breaks_an_inlier_tie_by_strictly_lower_median_error() {
    let initial = translation(-90.0, 5.0);
    // Small enough that the translation keeps every pair within 3px, yet
    // leaves a measurable error that the affine stage removes.
    let correction = translation(1.5, -2.0) * about_center((250.0, 200.0), rotation(0.004));
    let points = residual_selection_points(&initial, &correction, 10, 8);

    let lowest = fit_residual_model(
        &points,
        &initial,
        VirtualTileResidualSelection::LowestComplexity,
    );
    let lowest_fit = lowest.fit.expect("translation passes");
    assert_eq!(lowest_fit.model, "translation");
    assert_eq!(lowest_fit.inlier_indices.len(), points.len());
    assert!(lowest_fit.median_error_px > 0.1);

    let most = fit_residual_model(&points, &initial, VirtualTileResidualSelection::MostInliers);
    let most_fit = most.fit.expect("every stage is evaluated");
    assert_eq!(most_fit.inlier_indices.len(), points.len());
    assert_ne!(
        most_fit.model, "translation",
        "the tie goes to the lower error"
    );
    assert!(most_fit.median_error_px < lowest_fit.median_error_px);
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
// checkpoint 10: closure on a synthetic two-dimensional scan grid
// ---------------------------------------------------------------------------

/// Worst reprojection disagreement of one translation relation under `poses`,
/// on a lattice over the left tile clipped to the right tile.
fn translation_pair_error(
    dimensions: (u32, u32),
    poses: &[Matrix3<f64>],
    left: usize,
    right: usize,
    left_to_right: &Matrix3<f64>,
) -> f64 {
    let (width, height) = (f64::from(dimensions.0), f64::from(dimensions.1));
    let mut worst = 0.0f64;
    for row in 0..=10 {
        for column in 0..=10 {
            let source = Point2::new(
                (width - 1.0) * f64::from(column) / 10.0,
                (height - 1.0) * f64::from(row) / 10.0,
            );
            let target = transformed_point(left_to_right, source).unwrap();
            if !(0.0..width).contains(&target.x) || !(0.0..height).contains(&target.y) {
                continue;
            }
            let error = (transformed_point(&poses[left], source).unwrap()
                - transformed_point(&poses[right], target).unwrap())
            .norm();
            worst = worst.max(error);
        }
    }
    worst
}

#[test]
fn checkpoint_ten_scan_grid_closure_corrects_tree_drift_within_the_residual_bounds() {
    use super::station_relation_test_access::{self, ClosureRelationView};

    // A 3×3 serpentine scan; each spanning-tree step along the scan path adds
    // (0.6, −0.4)px of drift, so the tree alone misplaces loop partners by up
    // to five steps (3.6px) while every relation measures the true geometry.
    let dimensions = (512u32, 384u32);
    let serpentine = [
        (0u32, 0u32),
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 1),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
    ];
    let world = |station: usize| {
        let (row, column) = serpentine[station];
        (f64::from(column) * 360.0, f64::from(row) * 270.0)
    };
    let drift = nalgebra::Vector2::new(0.6, -0.4);
    let base = (0..serpentine.len())
        .map(|station| {
            let (x, y) = world(station);
            translation(x + drift.x * station as f64, y + drift.y * station as f64)
        })
        .collect::<Vec<_>>();
    let station_of = |row: u32, column: u32| {
        serpentine
            .iter()
            .position(|&cell| cell == (row, column))
            .unwrap()
    };
    let mut relations = Vec::new();
    for row in 0..3u32 {
        for column in 0..3u32 {
            for (next_row, next_column) in [(row, column + 1), (row + 1, column)] {
                if next_row > 2 || next_column > 2 {
                    continue;
                }
                let (a, b) = (station_of(row, column), station_of(next_row, next_column));
                let (left, right) = (a.min(b), a.max(b));
                let ((left_x, left_y), (right_x, right_y)) = (world(left), world(right));
                let noise = |salt: u32| hashed_noise(left as u32, right as u32, salt, 0.25);
                let tree_edge = right == left + 1;
                // Tree edges carry exactly the drifted placement the tree was
                // built from; loop edges carry the true geometry.
                let (dx, dy) = if tree_edge {
                    (-drift.x, -drift.y)
                } else {
                    (noise(1), noise(2))
                };
                relations.push(ClosureRelationView {
                    score: 1.0,
                    left,
                    right,
                    left_to_right: translation(left_x - right_x + dx, left_y - right_y + dy),
                    independent_support: 3,
                    median_error_px: 0.5,
                });
            }
        }
    }
    assert_eq!(relations.len(), 12, "6 horizontal + 6 vertical relations");
    let tree_worst = relations
        .iter()
        .map(|relation| {
            translation_pair_error(
                dimensions,
                &base,
                relation.left,
                relation.right,
                &relation.left_to_right,
            )
        })
        .fold(0.0f64, f64::max);
    assert!(
        tree_worst > 3.0,
        "the drifted tree alone violates the pair bound: {tree_worst}"
    );

    let rows = serpentine.iter().map(|&(row, _)| row).collect::<Vec<_>>();
    let columns = serpentine
        .iter()
        .map(|&(_, column)| column)
        .collect::<Vec<_>>();
    let run =
        station_relation_test_access::closure_run(dimensions, &rows, &columns, &base, &relations);
    assert_eq!(
        run.report.status,
        report::ClosureStatus::Converged,
        "{:?}",
        run.report
    );
    assert_eq!(run.report.participating_constraints, relations.len());
    assert!(run.report.residual_median_px <= 2.0, "{:?}", run.report);
    assert!(run.report.max_direct_pair_p95_px <= 3.0, "{:?}", run.report);
    assert_eq!(run.report.offending_pair, None);
    assert_eq!(run.report.clamped_stations, 0);
    let closed_worst = relations
        .iter()
        .map(|relation| {
            translation_pair_error(
                dimensions,
                &run.poses,
                relation.left,
                relation.right,
                &relation.left_to_right,
            )
        })
        .fold(0.0f64, f64::max);
    assert!(closed_worst <= 3.0, "{closed_worst}");
    assert!(closed_worst < tree_worst);
}

#[test]
fn global_affine_closure_corrects_cross_station_scale_drift_without_relaxing_gates() {
    use super::station_relation_test_access::{self, ClosureRelationView};

    // Three generic stations with a small per-station affine scale drift.  The
    // local edge models are all exact translations, so a translation-only
    // closure leaves a position-dependent error on the long overlap.  The
    // global affine stage must reconcile every edge before the existing 3px
    // direct-pair gate is evaluated.
    let dimensions = (1200u32, 800u32);
    let world_x = |station: usize| station as f64 * 300.0;
    let scale = |station: usize| 1.0 + station as f64 * 0.02;
    let base = (0..3)
        .map(|station| {
            translation(world_x(station), 0.0)
                * Matrix3::new(
                    scale(station),
                    0.0,
                    0.0,
                    0.0,
                    scale(station),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                )
        })
        .collect::<Vec<_>>();
    let mut relations = Vec::new();
    for left in 0..3 {
        for right in (left + 1)..3 {
            relations.push(ClosureRelationView {
                score: 1.0,
                left,
                right,
                left_to_right: translation(world_x(left) - world_x(right), 0.0),
                independent_support: 3,
                median_error_px: 0.5,
            });
        }
    }
    let base_worst = relations
        .iter()
        .map(|relation| {
            translation_pair_error(
                dimensions,
                &base,
                relation.left,
                relation.right,
                &relation.left_to_right,
            )
        })
        .fold(0.0f64, f64::max);
    assert!(
        base_worst > 3.0,
        "scale drift must violate the direct-pair gate: {base_worst}"
    );

    let rows = vec![0, 0, 0];
    let columns = vec![0, 1, 2];
    let run =
        station_relation_test_access::closure_run(dimensions, &rows, &columns, &base, &relations);
    assert_eq!(
        run.report.status,
        report::ClosureStatus::Converged,
        "{:?}",
        run.report
    );
    assert!(run.report.residual_median_px <= 2.0, "{:?}", run.report);
    assert!(run.report.max_direct_pair_p95_px <= 3.0, "{:?}", run.report);
    assert!(run.report.max_direct_pair_p95_px < base_worst);
}

#[test]
fn global_projective_closure_corrects_perspective_drift_without_relaxing_gates() {
    use super::station_relation_test_access::{self, ClosureRelationView};

    // The spanning-tree seed carries a small perspective term at each station.
    // The redundant direct relations are exact translations, so the accumulated
    // denominator drift is visible on the long overlap and cannot be removed by
    // an affine-only correction.
    let dimensions = (1200u32, 800u32);
    let world_x = |station: usize| station as f64 * 300.0;
    let base = (0..4)
        .map(|station| {
            let perspective = station as f64 * 0.000006;
            translation(world_x(station), 0.0)
                * Matrix3::new(
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    perspective,
                    -perspective * 0.25,
                    1.0,
                )
        })
        .collect::<Vec<_>>();
    let mut relations = Vec::new();
    for left in 0..4 {
        for right in (left + 1)..4 {
            relations.push(ClosureRelationView {
                score: 1.0,
                left,
                right,
                left_to_right: translation(world_x(left) - world_x(right), 0.0),
                independent_support: 3,
                median_error_px: 0.5,
            });
        }
    }
    let base_worst = relations
        .iter()
        .map(|relation| {
            translation_pair_error(
                dimensions,
                &base,
                relation.left,
                relation.right,
                &relation.left_to_right,
            )
        })
        .fold(0.0f64, f64::max);
    assert!(
        base_worst > 3.0,
        "perspective drift must violate the direct-pair gate: {base_worst}"
    );

    let rows = vec![0, 0, 0, 0];
    let columns = vec![0, 1, 2, 3];
    let run =
        station_relation_test_access::closure_run(dimensions, &rows, &columns, &base, &relations);
    assert_eq!(
        run.report.status,
        report::ClosureStatus::Converged,
        "{:?}",
        run.report
    );
    assert!(run.report.residual_median_px <= 2.0, "{:?}", run.report);
    assert!(run.report.max_direct_pair_p95_px <= 3.0, "{:?}", run.report);
    assert!(run.report.max_direct_pair_p95_px < base_worst);
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
