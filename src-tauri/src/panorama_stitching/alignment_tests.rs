use super::*;
use nalgebra::Point2;

mod capture_grouping;
mod closure_optimizer;
mod edge_bands;
mod focus_scan_order;
mod group_pose_solver;
mod match_pairs;
mod relation_acceptance;
mod render_ordering;
mod source_preparation;
mod transform_models;
mod virtual_tile_matching;
mod virtual_tile_measurements;
mod virtual_tile_residual_model;
mod virtual_tile_solver_flow;

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
