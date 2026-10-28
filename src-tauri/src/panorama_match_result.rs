//! Shared result assembly for pair matching.

use nalgebra::{Matrix3, Point2};

use super::MatchInfo;

pub(super) fn build_match_info(
    homography: Matrix3<f64>,
    points: Vec<(Point2<f64>, Point2<f64>)>,
    candidate_points: Vec<(Point2<f64>, Point2<f64>)>,
    top_candidate_points: Vec<(Point2<f64>, Point2<f64>)>,
    dense_focus_points: Vec<(Point2<f64>, Point2<f64>)>,
    foreground_feature_points: Vec<(Point2<f64>, Point2<f64>)>,
    coarse_bridge: bool,
) -> MatchInfo {
    MatchInfo {
        native_verified_points: Vec::new(),
        homography,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge,
        points,
        candidate_points,
        top_candidate_points,
        dense_focus_points,
        foreground_feature_points,
        canonical_homography: None,
    }
}
