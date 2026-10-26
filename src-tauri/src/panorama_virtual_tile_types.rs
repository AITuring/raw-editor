//! Virtual-tile solver shared types and feature constants.

use super::*;

/// re-check it makes on a tile pair looks only at pixels both tiles cover
/// (需求 6.1).
#[derive(Clone, Copy)]
pub(crate) struct VirtualTileSolverCoverages<'a> {
    /// `quality[i]` covers `tiles[i].alignment_image` (the requirement-6.8
    /// relation-quality plane).
    pub(crate) quality: &'a [stitching::CoverageMask],
    /// `overlap[i]` covers tile `i`'s finest match plane, `overlap_factor`
    /// native pixels per plane pixel (the requirement-6.4 overlap area).
    pub(crate) overlap: &'a [stitching::CoverageMask],
    pub(crate) overlap_factor: u32,
}

impl<'a> VirtualTileSolverCoverages<'a> {
    pub(crate) fn overlap_measure(
        &self,
        source: usize,
        target: usize,
    ) -> Option<StationOverlapMeasure<'a>> {
        Some(StationOverlapMeasure::Covered {
            source: self.overlap.get(source)?,
            target: self.overlap.get(target)?,
            factor: self.overlap_factor,
        })
    }
}

pub(crate) const VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS: u32 = 20;

/// Convert a rendered Virtual_Tile to its native-pixel station-matching
/// representation.
///
/// Uncovered RGB values are never read. The deterministic zero outside the
/// mask prevents transparent-corner payloads from changing detection, while
/// the final erosion keeps both FAST and BRIEF support wholly inside genuine
/// Coverage_Mask pixels (需求 6.1). Production matches the area-averaged planes
/// of [`install_virtual_tile_match_planes`]; this native form backs the tests.
#[cfg(test)]
pub(crate) fn virtual_tile_covered_features(
    pixels: &Rgb32FImage,
    coverage: &stitching::CoverageMask,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
) -> (GrayImage, Vec<Feature>) {
    let (width, height) = pixels.dimensions();
    if coverage.dimensions() != (width, height) {
        return (GrayImage::new(width, height), Vec::new());
    }
    let gray = virtual_tile_covered_luminance(pixels, coverage);
    let radius = VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS;
    let features = processing::find_features(&gray, brief_pairs)
        .into_iter()
        .filter(|feature| {
            let x = feature.keypoint.x;
            let y = feature.keypoint.y;
            x >= radius
                && y >= radius
                && x.saturating_add(radius) < width
                && y.saturating_add(radius) < height
                && (y - radius..=y + radius).all(|sample_y| {
                    (x - radius..=x + radius)
                        .all(|sample_x| coverage.is_covered(sample_x, sample_y))
                })
        })
        .collect();
    (gray, features)
}

pub(crate) const VIRTUAL_TILE_GUIDED_PATCH_RADIUS: i32 = 6;
pub(crate) const VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC: f64 = 0.55;
pub(crate) const VIRTUAL_TILE_GUIDED_MIN_GRADIENT_NCC: f64 = 0.35;
pub(crate) const VIRTUAL_TILE_GUIDED_MIN_PEAK_MARGIN: f64 = 0.025;
pub(crate) const VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES: f64 = 15.0;
pub(crate) const VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE: u32 = 64;
pub(crate) const VIRTUAL_TILE_GUIDED_RATIO: f64 = 0.90;
pub(crate) const VIRTUAL_TILE_GUIDED_MIN_SEARCH_RADIUS_PX: f64 = 12.0;
pub(crate) const VIRTUAL_TILE_GUIDED_MAX_SEARCH_RADIUS_PX: f64 = 96.0;
pub(crate) const VIRTUAL_TILE_GUIDED_GRID_COLUMNS: usize = 6;
pub(crate) const VIRTUAL_TILE_GUIDED_GRID_ROWS: usize = 6;
pub(crate) const VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA: usize = 2;
pub(crate) const VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS: usize = 12;

#[derive(Clone, Copy, Debug)]
pub(crate) struct VirtualTileStructuralScore {
    pub(crate) patch_ncc: f64,
    pub(crate) gradient_ncc: f64,
    pub(crate) orientation_difference_degrees: f64,
}

impl VirtualTileStructuralScore {
    pub(crate) fn rank(self, descriptor_distance: u32) -> f64 {
        self.patch_ncc + 0.5 * self.gradient_ncc
            - f64::from(descriptor_distance)
                / f64::from(VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE.max(1))
                * 0.05
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VirtualTileGuidedFeatureMatch {
    pub(crate) target_index: usize,
    pub(crate) rank: f64,
    pub(crate) orientation_difference_degrees: f64,
}

#[cfg(test)]
pub(crate) struct VirtualTileStationMatchOutcome {
    pub(crate) relations: VirtualTileRelationHypotheses,
    pub(crate) diagnostics: Vec<stack_report::StationRelationCandidateRecord>,
}

#[cfg(test)]
pub(crate) fn virtual_tile_unseeded_station_match(
    left: usize,
    right: usize,
    tiles: &[ImageInfo],
) -> Option<MatchInfo> {
    let descriptor_matches =
        processing::match_features(&tiles[left].features, &tiles[right].features);
    let left_keypoints = tiles[left]
        .features
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let right_keypoints = tiles[right]
        .features
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let (homography, inliers) =
        processing::find_homography_ransac(&descriptor_matches, &left_keypoints, &right_keypoints)?;
    let points = inliers
        .iter()
        .map(|matched| {
            let source = left_keypoints[matched.index1];
            let target = right_keypoints[matched.index2];
            (
                Point2::new(source.x as f64, source.y as f64),
                Point2::new(target.x as f64, target.y as f64),
            )
        })
        .collect::<Vec<_>>();
    Some(MatchInfo {
        homography,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: false,
        points,
        candidate_points: Vec::new(),
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        // Key-direction Virtual_Tile fit; see `MatchInfo::canonical_homography`.
        canonical_homography: None,
    })
}
