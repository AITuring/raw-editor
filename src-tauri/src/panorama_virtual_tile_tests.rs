//! Regression tests for the checkpoint-10 Virtual_Tile station matcher:
//! coarse-to-fine match planes, the fine polish stage, prior-free relation
//! proposals, rendering-prior repair planning and the structural quality plane.

use super::*;
use crate::panorama_utils::stack_pipeline::{intra_station, report};
use nalgebra::Point2;

#[path = "panorama_virtual_tile_tests/coverage_measurements.rs"]
mod coverage_measurements;
#[path = "panorama_virtual_tile_tests/fine_polish.rs"]
mod fine_polish;
#[path = "panorama_virtual_tile_tests/global_search.rs"]
mod global_search;
#[path = "panorama_virtual_tile_tests/hypothesis_selection.rs"]
mod hypothesis_selection;
#[path = "panorama_virtual_tile_tests/match_pyramid.rs"]
mod match_pyramid;
#[path = "panorama_virtual_tile_tests/prior_repair_planning.rs"]
mod prior_repair_planning;
#[path = "panorama_virtual_tile_tests/raw_correlation.rs"]
mod raw_correlation;
#[path = "panorama_virtual_tile_tests/raw_direct_refinement.rs"]
mod raw_direct_refinement;
#[path = "panorama_virtual_tile_tests/raw_independent_support.rs"]
mod raw_independent_support;
#[path = "panorama_virtual_tile_tests/residual_correction.rs"]
mod residual_correction;
#[path = "panorama_virtual_tile_tests/scan_grid_closure.rs"]
mod scan_grid_closure;
#[path = "panorama_virtual_tile_tests/station_records.rs"]
mod station_records;

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
