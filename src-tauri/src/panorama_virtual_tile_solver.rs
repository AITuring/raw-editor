//! Virtual-tile matching, relation fitting, and station-solver support.
//!
//! This module is kept behind the panorama_stitching facade so existing
//! callers and test paths continue to resolve through the parent module.

use super::*;

/// Coverage_Masks the station solver reads for Virtual_Tile evidence: every
/// re-check it makes on a tile pair looks only at pixels both tiles cover
/// (需求 6.1).
#[derive(Clone, Copy)]
pub(super) struct VirtualTileSolverCoverages<'a> {
    /// `quality[i]` covers `tiles[i].alignment_image` (the requirement-6.8
    /// relation-quality plane).
    pub(super) quality: &'a [stitching::CoverageMask],
    /// `overlap[i]` covers tile `i`'s finest match plane, `overlap_factor`
    /// native pixels per plane pixel (the requirement-6.4 overlap area).
    pub(super) overlap: &'a [stitching::CoverageMask],
    pub(super) overlap_factor: u32,
}

impl<'a> VirtualTileSolverCoverages<'a> {
    pub(super) fn overlap_measure(
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

pub(super) const VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS: u32 = 20;

/// Convert a rendered Virtual_Tile to its native-pixel station-matching
/// representation.
///
/// Uncovered RGB values are never read. The deterministic zero outside the
/// mask prevents transparent-corner payloads from changing detection, while
/// the final erosion keeps both FAST and BRIEF support wholly inside genuine
/// Coverage_Mask pixels (需求 6.1). Production matches the area-averaged planes
/// of [`install_virtual_tile_match_planes`]; this native form backs the tests.
#[cfg(test)]
pub(super) fn virtual_tile_covered_features(
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

pub(super) const VIRTUAL_TILE_GUIDED_PATCH_RADIUS: i32 = 6;
pub(super) const VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC: f64 = 0.55;
pub(super) const VIRTUAL_TILE_GUIDED_MIN_GRADIENT_NCC: f64 = 0.35;
pub(super) const VIRTUAL_TILE_GUIDED_MIN_PEAK_MARGIN: f64 = 0.025;
pub(super) const VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES: f64 = 15.0;
pub(super) const VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE: u32 = 64;
pub(super) const VIRTUAL_TILE_GUIDED_RATIO: f64 = 0.90;
pub(super) const VIRTUAL_TILE_GUIDED_MIN_SEARCH_RADIUS_PX: f64 = 12.0;
pub(super) const VIRTUAL_TILE_GUIDED_MAX_SEARCH_RADIUS_PX: f64 = 96.0;
pub(super) const VIRTUAL_TILE_GUIDED_GRID_COLUMNS: usize = 6;
pub(super) const VIRTUAL_TILE_GUIDED_GRID_ROWS: usize = 6;
pub(super) const VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA: usize = 2;
pub(super) const VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS: usize = 12;

#[derive(Clone, Copy, Debug)]
pub(super) struct VirtualTileStructuralScore {
    pub(super) patch_ncc: f64,
    pub(super) gradient_ncc: f64,
    pub(super) orientation_difference_degrees: f64,
}

impl VirtualTileStructuralScore {
    pub(super) fn rank(self, descriptor_distance: u32) -> f64 {
        self.patch_ncc + 0.5 * self.gradient_ncc
            - f64::from(descriptor_distance)
                / f64::from(VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE.max(1))
                * 0.05
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct VirtualTileGuidedFeatureMatch {
    pub(super) target_index: usize,
    pub(super) rank: f64,
    pub(super) orientation_difference_degrees: f64,
}

#[cfg(test)]
pub(super) struct VirtualTileStationMatchOutcome {
    pub(super) relations: VirtualTileRelationHypotheses,
    pub(super) diagnostics: Vec<stack_report::StationRelationCandidateRecord>,
}

#[cfg(test)]
pub(super) fn virtual_tile_unseeded_station_match(
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

pub(super) fn virtual_tile_point_is_covered(
    coverage: &stitching::CoverageMask,
    point: Point2<f64>,
) -> bool {
    let (width, height) = coverage.dimensions();
    point.x.is_finite()
        && point.y.is_finite()
        && point.x >= 0.0
        && point.y >= 0.0
        && point.x < width as f64
        && point.y < height as f64
        && coverage.is_covered(point.x.round() as u32, point.y.round() as u32)
}

pub(super) fn virtual_tile_local_jacobian(
    transform: &Matrix3<f64>,
    point: Point2<f64>,
) -> Option<nalgebra::Matrix2<f64>> {
    let center = transformed_point(transform, point)?;
    let horizontal = transformed_point(transform, point + nalgebra::Vector2::new(1.0, 0.0))?;
    let vertical = transformed_point(transform, point + nalgebra::Vector2::new(0.0, 1.0))?;
    let jacobian = nalgebra::Matrix2::new(
        horizontal.x - center.x,
        vertical.x - center.x,
        horizontal.y - center.y,
        vertical.y - center.y,
    );
    (jacobian.iter().all(|value| value.is_finite()) && jacobian.determinant().abs() > 1e-8)
        .then_some(jacobian)
}

pub(super) fn virtual_tile_gradient(
    image: &GrayImage,
    point: Point2<f64>,
) -> Option<nalgebra::Vector2<f64>> {
    let gx = registration::sample_gray(image, point.x + 1.0, point.y)?
        - registration::sample_gray(image, point.x - 1.0, point.y)?;
    let gy = registration::sample_gray(image, point.x, point.y + 1.0)?
        - registration::sample_gray(image, point.x, point.y - 1.0)?;
    let gradient = nalgebra::Vector2::new(gx, gy);
    gradient
        .iter()
        .all(|value| value.is_finite())
        .then_some(gradient)
}

/// A projective repair's scale belongs at the station, not at the unrelated
/// world origin. Reading its normalized linear block made the bound depend on
/// canvas translation and rejected a near-unit correction as a 1.5x jump.
pub(super) fn virtual_tile_repair_local_scale_ratio(
    transform: &Matrix3<f64>,
    center: Point2<f64>,
) -> Option<f64> {
    let ratio = virtual_tile_local_jacobian(transform, center)?
        .determinant()
        .abs()
        .sqrt();
    (ratio.is_finite() && ratio > 0.0).then_some(ratio)
}

pub(super) fn axial_orientation_difference_degrees(
    left: nalgebra::Vector2<f64>,
    right: nalgebra::Vector2<f64>,
) -> Option<f64> {
    let denominator = left.norm() * right.norm();
    if !denominator.is_finite() || denominator <= 1e-8 {
        return None;
    }
    let cosine = (left.dot(&right) / denominator).abs().clamp(0.0, 1.0);
    Some(cosine.acos().to_degrees())
}

pub(super) fn virtual_tile_gradient_magnitude_ncc(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    if let Some(correlation) = normalized_correlation(left, right) {
        return Some(correlation);
    }
    // A constant-strength edge has zero variance but still carries orientation
    // and location evidence. Treat equal non-zero magnitudes as perfect NCC;
    // unequal or flat gradients remain non-matches.
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let scale = left_mean.abs().max(right_mean.abs());
    (scale > 1e-8).then(|| (1.0 - (left_mean - right_mean).abs() / scale).clamp(-1.0, 1.0))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_structural_score_points(
    left: &GrayImage,
    right: &GrayImage,
    left_coverage: &stitching::CoverageMask,
    right_coverage: &stitching::CoverageMask,
    center: Point2<f64>,
    right_center: Point2<f64>,
    left_to_right: &Matrix3<f64>,
) -> Option<VirtualTileStructuralScore> {
    let projected_center = transformed_point(left_to_right, center)?;
    let residual = right_center - projected_center;
    let jacobian = virtual_tile_local_jacobian(left_to_right, center)?;
    let gradient_transform = jacobian.try_inverse()?.transpose();
    let radius = VIRTUAL_TILE_GUIDED_PATCH_RADIUS;
    let capacity = ((radius * 2 + 1) * (radius * 2 + 1)) as usize;
    let mut left_values = Vec::with_capacity(capacity);
    let mut right_values = Vec::with_capacity(capacity);
    let mut left_gradient_magnitudes = Vec::with_capacity(capacity);
    let mut right_gradient_magnitudes = Vec::with_capacity(capacity);
    let mut orientation_differences = Vec::with_capacity(capacity);

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let source = center + nalgebra::Vector2::new(f64::from(dx), f64::from(dy));
            let target = transformed_point(left_to_right, source)? + residual;
            if !virtual_tile_point_is_covered(left_coverage, source)
                || !virtual_tile_point_is_covered(right_coverage, target)
            {
                return None;
            }
            left_values.push(registration::sample_gray(left, source.x, source.y)?);
            right_values.push(registration::sample_gray(right, target.x, target.y)?);

            let left_gradient = virtual_tile_gradient(left, source)?;
            let right_gradient = virtual_tile_gradient(right, target)?;
            let transformed_gradient = gradient_transform * left_gradient;
            left_gradient_magnitudes.push(transformed_gradient.norm());
            right_gradient_magnitudes.push(right_gradient.norm());
            if transformed_gradient.norm().min(right_gradient.norm()) >= 4.0
                && let Some(difference) =
                    axial_orientation_difference_degrees(transformed_gradient, right_gradient)
            {
                orientation_differences.push(difference);
            }
        }
    }

    let patch_ncc = normalized_correlation(&left_values, &right_values)?;
    let gradient_ncc =
        virtual_tile_gradient_magnitude_ncc(&left_gradient_magnitudes, &right_gradient_magnitudes)?;
    let orientation_difference_degrees = median_value(&mut orientation_differences)?;
    Some(VirtualTileStructuralScore {
        patch_ncc,
        gradient_ncc,
        orientation_difference_degrees,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_structural_score(
    left: &GrayImage,
    right: &GrayImage,
    left_coverage: &stitching::CoverageMask,
    right_coverage: &stitching::CoverageMask,
    left_point: KeyPoint,
    right_point: KeyPoint,
    left_to_right: &Matrix3<f64>,
) -> Option<VirtualTileStructuralScore> {
    virtual_tile_structural_score_points(
        left,
        right,
        left_coverage,
        right_coverage,
        Point2::new(f64::from(left_point.x), f64::from(left_point.y)),
        Point2::new(f64::from(right_point.x), f64::from(right_point.y)),
        left_to_right,
    )
}

pub(super) fn virtual_tile_descriptor_distance(left: &Descriptor, right: &Descriptor) -> u32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (left ^ right).count_ones())
        .sum()
}

/// Search radius in the Virtual_Tile analysis plane.
///
/// The source-stage station geometry supplies a measured uncertainty of at
/// most `FOCUS_MODEL_INLIER_THRESHOLD`; three such sigmas cover its current
/// geometric uncertainty. Sparse instruments add two standard sampling errors:
/// `sqrt(overlap_area / count) / sqrt(count)`. The result is converted through
/// the actual analysis scale and bounded to eight percent of the shorter tile
/// side (and 96 analysis pixels) so a prior can never turn into an unrestricted
/// repeated-texture search.
pub(super) fn virtual_tile_guided_search_radius(
    left: &ImageInfo,
    right: &ImageInfo,
    predicted_overlap_area: f64,
    instrument_count: usize,
) -> f64 {
    let analysis_scale = left.scale_factor.max(right.scale_factor).max(1.0);
    let count = instrument_count.max(1) as f64;
    let sampling_error = (predicted_overlap_area.max(1.0) / count).sqrt() / count.sqrt();
    let measured_geometry_error = 3.0 * FOCUS_MODEL_INLIER_THRESHOLD / analysis_scale;
    let dimension_cap = 0.08
        * f64::from(
            left.width
                .min(left.height)
                .min(right.width.min(right.height)),
        );
    (measured_geometry_error + 2.0 * sampling_error).clamp(
        VIRTUAL_TILE_GUIDED_MIN_SEARCH_RADIUS_PX,
        VIRTUAL_TILE_GUIDED_MAX_SEARCH_RADIUS_PX
            .min(dimension_cap.max(VIRTUAL_TILE_GUIDED_MIN_SEARCH_RADIUS_PX)),
    )
}

pub(super) fn virtual_tile_feature_buckets(
    features: &[Feature],
    eligible: &[bool],
    cell_size: f64,
) -> BTreeMap<(i32, i32), Vec<usize>> {
    let mut buckets = BTreeMap::<(i32, i32), Vec<usize>>::new();
    for (index, feature) in features.iter().enumerate() {
        if !eligible.get(index).copied().unwrap_or(false) {
            continue;
        }
        let key = (
            (f64::from(feature.keypoint.x) / cell_size).floor() as i32,
            (f64::from(feature.keypoint.y) / cell_size).floor() as i32,
        );
        buckets.entry(key).or_default().push(index);
    }
    buckets
}

#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_best_guided_feature(
    source_index: usize,
    source_features: &[Feature],
    target_features: &[Feature],
    target_buckets: &BTreeMap<(i32, i32), Vec<usize>>,
    source_gray: &GrayImage,
    target_gray: &GrayImage,
    source_coverage: &stitching::CoverageMask,
    target_coverage: &stitching::CoverageMask,
    source_to_target: &Matrix3<f64>,
    search_radius: f64,
    orientation_observations: &mut Vec<f64>,
) -> Option<VirtualTileGuidedFeatureMatch> {
    let source = source_features.get(source_index)?;
    let projected = transformed_point(
        source_to_target,
        Point2::new(f64::from(source.keypoint.x), f64::from(source.keypoint.y)),
    )?;
    let center = (
        (projected.x / search_radius).floor() as i32,
        (projected.y / search_radius).floor() as i32,
    );
    let mut candidates = Vec::<(f64, u32, usize, f64)>::new();
    for bucket_y in center.1 - 1..=center.1 + 1 {
        for bucket_x in center.0 - 1..=center.0 + 1 {
            for &target_index in target_buckets
                .get(&(bucket_x, bucket_y))
                .into_iter()
                .flatten()
            {
                let target = &target_features[target_index];
                let displacement = Point2::new(
                    f64::from(target.keypoint.x) - projected.x,
                    f64::from(target.keypoint.y) - projected.y,
                );
                if displacement.coords.norm() > search_radius {
                    continue;
                }
                let descriptor_distance =
                    virtual_tile_descriptor_distance(&source.descriptor, &target.descriptor);
                if descriptor_distance > VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE {
                    continue;
                }
                let Some(structure) = virtual_tile_structural_score(
                    source_gray,
                    target_gray,
                    source_coverage,
                    target_coverage,
                    source.keypoint,
                    target.keypoint,
                    source_to_target,
                ) else {
                    continue;
                };
                // The rough station pose is only a search prior. Its local
                // Jacobian may have the wrong rotation, so pre-fit orientation
                // is recorded for diagnostics but cannot reject a measured
                // correspondence. Patch/gradient structure remains evidence.
                orientation_observations.push(structure.orientation_difference_degrees);
                if structure.patch_ncc < VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC
                    || structure.gradient_ncc < VIRTUAL_TILE_GUIDED_MIN_GRADIENT_NCC
                {
                    continue;
                }
                candidates.push((
                    structure.rank(descriptor_distance),
                    descriptor_distance,
                    target_index,
                    structure.orientation_difference_degrees,
                ));
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| {
                let left_point = target_features[left.2].keypoint;
                let right_point = target_features[right.2].keypoint;
                (left_point.y, left_point.x, left.2).cmp(&(right_point.y, right_point.x, right.2))
            })
    });
    let &(best_rank, best_distance, best_index, orientation_difference_degrees) =
        candidates.first()?;
    if let Some(&(second_rank, second_distance, _, _)) = candidates.get(1)
        && (second_distance == 0
            || f64::from(best_distance) / f64::from(second_distance) >= VIRTUAL_TILE_GUIDED_RATIO
            || best_rank - second_rank < VIRTUAL_TILE_GUIDED_MIN_PEAK_MARGIN)
    {
        return None;
    }
    Some(VirtualTileGuidedFeatureMatch {
        target_index: best_index,
        rank: best_rank,
        orientation_difference_degrees,
    })
}

pub(super) fn virtual_tile_predicted_overlap_bounds(
    left_to_right: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
) -> Option<(f64, f64, f64, f64)> {
    let right_to_left = left_to_right.try_inverse()?;
    let footprint = [
        Point2::new(0.0, 0.0),
        Point2::new(f64::from(right_dimensions.0), 0.0),
        Point2::new(f64::from(right_dimensions.0), f64::from(right_dimensions.1)),
        Point2::new(0.0, f64::from(right_dimensions.1)),
    ]
    .into_iter()
    .map(|point| transformed_point(&right_to_left, point))
    .collect::<Option<Vec<_>>>()?;
    let clipped = clip_polygon_to_rectangle(&footprint, left_dimensions);
    if clipped.len() < 3 {
        return None;
    }
    let min_x = clipped
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let max_x = clipped
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = clipped
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min);
    let max_y = clipped
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max);
    (min_x.is_finite()
        && max_x.is_finite()
        && min_y.is_finite()
        && max_y.is_finite()
        && max_x > min_x
        && max_y > min_y)
        .then_some((min_x, min_y, max_x, max_y))
}

pub(super) fn virtual_tile_overlap_grid_cell(
    point: Point2<f64>,
    bounds: (f64, f64, f64, f64),
) -> (usize, usize) {
    let (min_x, min_y, max_x, max_y) = bounds;
    let x = (((point.x - min_x) / (max_x - min_x)) * VIRTUAL_TILE_GUIDED_GRID_COLUMNS as f64)
        .floor()
        .clamp(0.0, (VIRTUAL_TILE_GUIDED_GRID_COLUMNS - 1) as f64) as usize;
    let y = (((point.y - min_y) / (max_y - min_y)) * VIRTUAL_TILE_GUIDED_GRID_ROWS as f64)
        .floor()
        .clamp(0.0, (VIRTUAL_TILE_GUIDED_GRID_ROWS - 1) as f64) as usize;
    (x, y)
}

#[allow(dead_code)]
pub(super) fn virtual_tile_descriptor_guided_station_match(
    left: usize,
    right: usize,
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
) -> (
    Option<MatchInfo>,
    stack_report::StationRelationCandidateRecord,
) {
    let mut diagnostic = stack_report::StationRelationCandidateRecord {
        left,
        right,
        left_tile: Default::default(),
        right_tile: Default::default(),
        predicted_overlap_area_px: 0.0,
        model_fitted: false,
        evidence_kind: stack_report::StationRelationEvidenceKind::VirtualTile,
        grid_probe_count: 0,
        measurable_patch_count: 0,
        bidirectional_match_count: 0,
        occupied_grid_cells: 0,
        search_radius_px: 0.0,
        residual_dx_px: Default::default(),
        residual_dy_px: Default::default(),
        residual_magnitude_px: Default::default(),
        max_residual_translation_consensus: 0,
        residual_translation: Default::default(),
        residual_affine: Default::default(),
        residual_projective: Default::default(),
        residual_model: "not_run".to_string(),
        fitted_inliers: 0,
        hull_support: 0.0,
        median_prefit_structural_orientation_difference_degrees: None,
        median_fitted_orientation_difference_degrees: None,
        failure_stage: "missing_input".to_string(),
        pyramid_factors: Vec::new(),
        measurement_factor: 1,
        seed: VIRTUAL_TILE_SEED_RENDERING_PRIOR.to_string(),
        prior_free_inliers: 0,
        after_prior_repair: false,
        hypotheses: Vec::new(),
    };
    let Some(left_tile) = tiles.get(left) else {
        return (None, diagnostic);
    };
    let Some(right_tile) = tiles.get(right) else {
        return (None, diagnostic);
    };
    let Some(left_coverage) = coverages.get(left) else {
        return (None, diagnostic);
    };
    let Some(right_coverage) = coverages.get(right) else {
        return (None, diagnostic);
    };
    diagnostic.failure_stage = "missing_initial_pose".to_string();
    let Some(left_to_world) = initial_homographies.get(&left_tile.id) else {
        return (None, diagnostic);
    };
    let Some(right_to_world) = initial_homographies.get(&right_tile.id) else {
        return (None, diagnostic);
    };
    let Some(world_to_right) = right_to_world.try_inverse() else {
        return (None, diagnostic);
    };
    let left_to_right = world_to_right * left_to_world;
    let Some(right_to_left) = left_to_right.try_inverse() else {
        return (None, diagnostic);
    };
    diagnostic.failure_stage = "no_predicted_overlap".to_string();
    let Some(overlap_bounds) = virtual_tile_predicted_overlap_bounds(
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
    ) else {
        return (None, diagnostic);
    };
    let Some(predicted_overlap_area) = station_relation_overlap_area(
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
        true,
    ) else {
        return (None, diagnostic);
    };
    diagnostic.predicted_overlap_area_px = predicted_overlap_area;

    let left_eligible = left_tile
        .features
        .iter()
        .map(|feature| {
            let point = Point2::new(f64::from(feature.keypoint.x), f64::from(feature.keypoint.y));
            virtual_tile_point_is_covered(left_coverage, point)
                && transformed_point(&left_to_right, point)
                    .is_some_and(|mapped| virtual_tile_point_is_covered(right_coverage, mapped))
        })
        .collect::<Vec<_>>();
    let right_eligible = right_tile
        .features
        .iter()
        .map(|feature| {
            let point = Point2::new(f64::from(feature.keypoint.x), f64::from(feature.keypoint.y));
            virtual_tile_point_is_covered(right_coverage, point)
                && transformed_point(&right_to_left, point)
                    .is_some_and(|mapped| virtual_tile_point_is_covered(left_coverage, mapped))
        })
        .collect::<Vec<_>>();
    let instrument_count = left_eligible
        .iter()
        .filter(|&&eligible| eligible)
        .count()
        .min(right_eligible.iter().filter(|&&eligible| eligible).count());
    diagnostic.grid_probe_count = instrument_count;
    diagnostic.failure_stage = "insufficient_jointly_covered_probes".to_string();
    if instrument_count < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }

    let search_radius = virtual_tile_guided_search_radius(
        left_tile,
        right_tile,
        predicted_overlap_area,
        instrument_count,
    );
    diagnostic.search_radius_px = search_radius;
    let left_buckets =
        virtual_tile_feature_buckets(&left_tile.features, &left_eligible, search_radius);
    let right_buckets =
        virtual_tile_feature_buckets(&right_tile.features, &right_eligible, search_radius);
    let mut by_cell = BTreeMap::<(usize, usize), Vec<(f64, usize, usize, f64)>>::new();
    let mut orientation_observations = Vec::new();
    for (left_index, &eligible) in left_eligible.iter().enumerate() {
        if !eligible {
            continue;
        }
        let Some(forward) = virtual_tile_best_guided_feature(
            left_index,
            &left_tile.features,
            &right_tile.features,
            &right_buckets,
            &left_tile.alignment_image,
            &right_tile.alignment_image,
            left_coverage,
            right_coverage,
            &left_to_right,
            search_radius,
            &mut orientation_observations,
        ) else {
            continue;
        };
        let mut reverse_orientations = Vec::new();
        let reverse = virtual_tile_best_guided_feature(
            forward.target_index,
            &right_tile.features,
            &left_tile.features,
            &left_buckets,
            &right_tile.alignment_image,
            &left_tile.alignment_image,
            right_coverage,
            left_coverage,
            &right_to_left,
            search_radius,
            &mut reverse_orientations,
        );
        orientation_observations.extend(reverse_orientations);
        if reverse.is_none_or(|reverse| reverse.target_index != left_index) {
            continue;
        }
        let source = left_tile.features[left_index].keypoint;
        let cell = virtual_tile_overlap_grid_cell(
            Point2::new(f64::from(source.x), f64::from(source.y)),
            overlap_bounds,
        );
        by_cell.entry(cell).or_default().push((
            forward.rank,
            left_index,
            forward.target_index,
            forward.orientation_difference_degrees,
        ));
    }
    diagnostic.median_prefit_structural_orientation_difference_degrees =
        median_value(&mut orientation_observations);
    diagnostic.bidirectional_match_count = by_cell.values().map(Vec::len).sum();
    diagnostic.occupied_grid_cells = by_cell.len();
    diagnostic.failure_stage = "insufficient_bidirectional_match_count".to_string();
    if diagnostic.bidirectional_match_count < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    diagnostic.failure_stage = "insufficient_grid_coverage".to_string();
    if diagnostic.occupied_grid_cells < VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS {
        return (None, diagnostic);
    }

    for proposals in by_cell.values_mut() {
        proposals.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| {
                    let left_source = left_tile.features[left.1].keypoint;
                    let right_source = left_tile.features[right.1].keypoint;
                    (left_source.y, left_source.x).cmp(&(right_source.y, right_source.x))
                })
                .then_with(|| {
                    let left_target = right_tile.features[left.2].keypoint;
                    let right_target = right_tile.features[right.2].keypoint;
                    (left_target.y, left_target.x).cmp(&(right_target.y, right_target.x))
                })
        });
        proposals.truncate(VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA);
    }
    // Round-robin by quota rank gives RANSAC one observation from every cell
    // before any repeated structure receives its second slot.
    let mut selected = Vec::new();
    for rank in 0..VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA {
        for proposals in by_cell.values() {
            if let Some(proposal) = proposals.get(rank) {
                selected.push(*proposal);
            }
        }
    }
    diagnostic.failure_stage = "insufficient_matches_after_cell_quota".to_string();
    if selected.len() < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    let candidate_points = selected
        .iter()
        .map(|&(_, left_index, right_index, _)| {
            let source = left_tile.features[left_index].keypoint;
            let target = right_tile.features[right_index].keypoint;
            (
                Point2::new(f64::from(source.x), f64::from(source.y)),
                Point2::new(f64::from(target.x), f64::from(target.y)),
            )
        })
        .collect::<Vec<_>>();
    diagnostic.hull_support = station_relation_spatial_support(
        &candidate_points,
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
    );
    diagnostic.failure_stage = "insufficient_prefit_hull_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        return (None, diagnostic);
    }

    // The initial pose ends here. RANSAC and its refit receive only measured
    // fused-pixel correspondences, so the prior can neither become an inlier
    // nor contribute acceptance evidence.
    diagnostic.failure_stage = "ransac_no_fit".to_string();
    let Some((seed_fit, seed_inliers)) =
        processing::find_homography_ransac_points_stable_with_min_inliers(
            &candidate_points,
            STATION_RELATION_MAX_MEDIAN_ERROR_PX,
            STATION_RELATION_MIN_INLIERS,
        )
    else {
        return (None, diagnostic);
    };
    diagnostic.model_fitted = true;
    let seed_points = seed_inliers
        .iter()
        .map(|&index| candidate_points[index])
        .collect::<Vec<_>>();
    let refined = processing::compute_homography(&seed_points).unwrap_or(seed_fit);
    let refined_inliers = symmetric_inlier_indices(
        &refined,
        &candidate_points,
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    );
    diagnostic.fitted_inliers = refined_inliers.len();
    diagnostic.failure_stage = "insufficient_fitted_inliers".to_string();
    if refined_inliers.len() < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    let points = refined_inliers
        .iter()
        .map(|&index| candidate_points[index])
        .collect::<Vec<_>>();
    diagnostic.hull_support = station_relation_spatial_support(
        &points,
        &refined,
        left_tile.dimensions(),
        right_tile.dimensions(),
    );
    diagnostic.failure_stage = "insufficient_fitted_hull_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        return (None, diagnostic);
    }

    // Orientation becomes an acceptance gate only after fused-pixel
    // correspondences have produced a measured homography. Re-measure every
    // accepted inlier patch with the fitted local Jacobian; the rough prior is
    // deliberately absent from this decision.
    let mut fitted_orientation_observations = refined_inliers
        .iter()
        .filter_map(|&index| {
            let &(_, left_index, right_index, _) = selected.get(index)?;
            virtual_tile_structural_score(
                &left_tile.alignment_image,
                &right_tile.alignment_image,
                left_coverage,
                right_coverage,
                left_tile.features[left_index].keypoint,
                right_tile.features[right_index].keypoint,
                &refined,
            )
            .map(|score| score.orientation_difference_degrees)
        })
        .collect::<Vec<_>>();
    diagnostic.median_fitted_orientation_difference_degrees =
        median_value(&mut fitted_orientation_observations);
    diagnostic.failure_stage = "fitted_orientation_mismatch".to_string();
    if diagnostic
        .median_fitted_orientation_difference_degrees
        .is_some_and(|difference| difference > VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES)
    {
        return (None, diagnostic);
    }

    diagnostic.failure_stage = "fitted".to_string();
    (
        Some(MatchInfo {
            homography: refined,
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
        }),
        diagnostic,
    )
}

pub(super) const VIRTUAL_TILE_DIRECT_PATCH_RADIUS: i32 = 6;
pub(super) const VIRTUAL_TILE_DIRECT_GRID_SPACING_PX: f64 = 24.0;
pub(super) const VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES: usize = 1_024;
pub(super) const VIRTUAL_TILE_DIRECT_MAX_GRID_AXIS: usize = 64;
pub(super) const VIRTUAL_TILE_DIRECT_MAX_SEARCH_RADIUS_PX: f64 = 16.0;
pub(super) const VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX: f64 = 1.0;
pub(super) const VIRTUAL_TILE_DIRECT_PATCH_GATES: registration::PatchMatchGates =
    registration::PatchMatchGates {
        min_correlation: VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC,
        min_peak_margin: VIRTUAL_TILE_GUIDED_MIN_PEAK_MARGIN,
        min_source_variance: 2.0,
        min_target_variance: 1.0,
    };

pub(super) fn virtual_tile_direct_grid_points(bounds: (f64, f64, f64, f64)) -> Vec<Point2<f64>> {
    let (min_x, min_y, max_x, max_y) = bounds;
    let width = max_x - min_x;
    let height = max_y - min_y;
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        return Vec::new();
    }
    let area = width * height;
    let desired = (area
        / (VIRTUAL_TILE_DIRECT_GRID_SPACING_PX * VIRTUAL_TILE_DIRECT_GRID_SPACING_PX))
        .ceil()
        .clamp(
            STATION_RELATION_MIN_INLIERS as f64,
            VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES as f64,
        ) as usize;
    let aspect = width / height;
    let mut columns = ((desired as f64 * aspect).sqrt().round() as usize)
        .clamp(1, VIRTUAL_TILE_DIRECT_MAX_GRID_AXIS);
    let mut rows = desired
        .div_ceil(columns)
        .clamp(1, VIRTUAL_TILE_DIRECT_MAX_GRID_AXIS);
    while columns * rows > VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES {
        if columns >= rows && columns > 1 {
            columns -= 1;
        } else if rows > 1 {
            rows -= 1;
        } else {
            break;
        }
    }
    let mut points = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        let y = min_y + (row as f64 + 0.5) * height / rows as f64;
        for column in 0..columns {
            let x = min_x + (column as f64 + 0.5) * width / columns as f64;
            points.push(Point2::new(x, y));
        }
    }
    points
}

pub(super) fn virtual_tile_coverage_contains_rectangle(
    coverage: &stitching::CoverageMask,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> bool {
    let (width, height) = coverage.dimensions();
    if ![min_x, min_y, max_x, max_y]
        .iter()
        .all(|value| value.is_finite())
        || min_x < 0.0
        || min_y < 0.0
        || max_x >= f64::from(width.saturating_sub(1))
        || max_y >= f64::from(height.saturating_sub(1))
        || max_x < min_x
        || max_y < min_y
    {
        return false;
    }
    let start_x = min_x.floor() as u32;
    let end_x = max_x.ceil() as u32;
    let start_y = min_y.floor() as u32;
    let end_y = max_y.ceil() as u32;
    (start_y..=end_y).all(|y| (start_x..=end_x).all(|x| coverage.is_covered(x, y)))
}

pub(super) fn virtual_tile_direct_patch_support_is_covered(
    source_coverage: &stitching::CoverageMask,
    target_coverage: &stitching::CoverageMask,
    source_to_target: &Matrix3<f64>,
    center: Point2<f64>,
    patch_radius: i32,
    search_radius: i32,
) -> bool {
    let Some(jacobian) = virtual_tile_local_jacobian(source_to_target, center) else {
        return false;
    };
    let scale = (jacobian.column(0).norm() * jacobian.column(1).norm()).sqrt();
    if !(0.15..6.0).contains(&scale) {
        return false;
    }
    let spacing = scale.recip().max(1.0);
    // Include one pixel for gradient sampling and another for bilinear access.
    let source_extent = f64::from(patch_radius) * spacing + 2.0;
    if !virtual_tile_coverage_contains_rectangle(
        source_coverage,
        center.x - source_extent,
        center.y - source_extent,
        center.x + source_extent,
        center.y + source_extent,
    ) {
        return false;
    }
    let corners = [
        Point2::new(center.x - source_extent, center.y - source_extent),
        Point2::new(center.x + source_extent, center.y - source_extent),
        Point2::new(center.x + source_extent, center.y + source_extent),
        Point2::new(center.x - source_extent, center.y + source_extent),
    ];
    let Some(projected) = corners
        .into_iter()
        .map(|point| transformed_point(source_to_target, point))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let padding = f64::from(search_radius) + 2.0;
    let min_x = projected
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min)
        - padding;
    let max_x = projected
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max)
        + padding;
    let min_y = projected
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min)
        - padding;
    let max_y = projected
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max)
        + padding;
    virtual_tile_coverage_contains_rectangle(target_coverage, min_x, min_y, max_x, max_y)
}

pub(super) fn virtual_tile_round_trip_is_consistent(
    source: Point2<f64>,
    returned: Point2<f64>,
) -> bool {
    let distance = (returned - source).norm();
    distance.is_finite() && distance <= VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX
}

/// Invert the *measured* local forward map, not merely the rough station prior.
///
/// `probe_warped_patch` returns `target = H_initial(source) + residual` in the
/// target tile's coordinates. The corresponding local forward homography is
/// therefore `T(residual) * H_initial`; its inverse is the only transform whose
/// source/target convention maps that measured target back to the original
/// source. Using `H_initial.inverse()` here incorrectly asks the reverse probe
/// to rediscover the already measured residual and made valid pairs depend on
/// which station happened to be the source (任务 10 checkpoint).
pub(super) fn virtual_tile_measured_reverse_transform(
    initial_source_to_target: &Matrix3<f64>,
    source: Point2<f64>,
    measured_target: Point2<f64>,
) -> Option<Matrix3<f64>> {
    let predicted_target = transformed_point(initial_source_to_target, source)?;
    let residual = measured_target - predicted_target;
    if !residual.iter().all(|value| value.is_finite()) {
        return None;
    }
    let residual_translation =
        Matrix3::new(1.0, 0.0, residual.x, 0.0, 1.0, residual.y, 0.0, 0.0, 1.0);
    (residual_translation * initial_source_to_target).try_inverse()
}

pub(super) const VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX: f64 = 8.0;
pub(super) const VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX: f64 = 24.0;
pub(super) const VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MIN: f64 = 0.98;
pub(super) const VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MAX: f64 = 1.02;

#[derive(Debug, Clone)]
pub(super) struct VirtualTileResidualFit {
    pub(super) measured_homography: Matrix3<f64>,
    pub(super) inlier_indices: Vec<usize>,
    pub(super) median_error_px: f64,
    pub(super) spatial_support: f64,
    pub(super) model: &'static str,
}

pub(super) fn residual_distribution(
    mut values: Vec<f64>,
) -> stack_report::ResidualDistributionRecord {
    values.retain(|value| value.is_finite());
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return Default::default();
    }
    let at = |quantile: f64| {
        let index = ((values.len() - 1) as f64 * quantile).round() as usize;
        values[index]
    };
    stack_report::ResidualDistributionRecord {
        count: values.len(),
        min: values[0],
        p10: at(0.10),
        median: at(0.50),
        p90: at(0.90),
        max: values[values.len() - 1],
    }
}

pub(super) fn virtual_tile_measured_residuals(
    points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
) -> Vec<(usize, nalgebra::Vector2<f64>)> {
    points
        .iter()
        .enumerate()
        .filter_map(|(index, (source, target))| {
            let predicted = transformed_point(initial, *source)?;
            let residual = target - predicted;
            residual
                .iter()
                .all(|value| value.is_finite())
                .then_some((index, residual))
        })
        .collect()
}

pub(super) fn virtual_tile_residual_candidate(
    correction: &Matrix3<f64>,
    initial: &Matrix3<f64>,
    measured_points: &[(Point2<f64>, Point2<f64>)],
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    model: &'static str,
    inlier_threshold_px: f64,
) -> Option<VirtualTileResidualFit> {
    // Both matrices map left -> right, so a correction measured in right-tile
    // coordinates composes on the left: H_measured = H_residual * H_initial.
    let measured_homography = correction * initial;
    let inlier_indices =
        symmetric_inlier_indices(&measured_homography, measured_points, inlier_threshold_px);
    let inlier_points = inlier_indices
        .iter()
        .map(|&index| measured_points[index])
        .collect::<Vec<_>>();
    let median_error_px = median_symmetric_error(&measured_homography, &inlier_points);
    let spatial_support = station_relation_spatial_support(
        &inlier_points,
        &measured_homography,
        left_dimensions,
        right_dimensions,
    );
    Some(VirtualTileResidualFit {
        measured_homography,
        inlier_indices,
        median_error_px,
        spatial_support,
        model,
    })
}

pub(super) fn virtual_tile_residual_fit_is_better(
    candidate: &VirtualTileResidualFit,
    current: &VirtualTileResidualFit,
) -> bool {
    candidate.inlier_indices.len() > current.inlier_indices.len()
        || (candidate.inlier_indices.len() == current.inlier_indices.len()
            && (candidate.median_error_px < current.median_error_px
                || (candidate.median_error_px == current.median_error_px
                    && candidate
                        .measured_homography
                        .iter()
                        .map(|value| value.to_bits())
                        .cmp(
                            current
                                .measured_homography
                                .iter()
                                .map(|value| value.to_bits()),
                        )
                        .is_lt())))
}

/// Whole left tile as the bounded-correction evidence region.
#[cfg(test)]
pub(super) fn virtual_tile_full_tile_bounds(left_dimensions: (u32, u32)) -> (f64, f64, f64, f64) {
    (
        0.0,
        0.0,
        f64::from(left_dimensions.0),
        f64::from(left_dimensions.1),
    )
}

/// The correction may move the evidence region's corners by at most
/// `VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX`, differ from its centre's
/// translation by at most `VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX`,
/// and keep its scale within the residual scale range. `evidence_bounds` is
/// `(min_x, min_y, max_x, max_y)` in left-tile pixels.
pub(super) fn virtual_tile_residual_correction_is_bounded(
    correction: &Matrix3<f64>,
    initial: &Matrix3<f64>,
    evidence_bounds: (f64, f64, f64, f64),
) -> bool {
    let (min_x, min_y, max_x, max_y) = evidence_bounds;
    let source_samples = [
        Point2::new(min_x, min_y),
        Point2::new(max_x, min_y),
        Point2::new(max_x, max_y),
        Point2::new(min_x, max_y),
        Point2::new((min_x + max_x) * 0.5, (min_y + max_y) * 0.5),
    ];
    let Some(displacements) = source_samples
        .into_iter()
        .map(|source| {
            let predicted = transformed_point(initial, source)?;
            let corrected = transformed_point(correction, predicted)?;
            Some(corrected - predicted)
        })
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let center = displacements[4];
    // The correction's scale is its local one where the evidence lies. The
    // upper-left block of a projective correction is not a Jacobian anywhere
    // but at the origin: at 1/8 on langyuan-10 it read 1.035 for a correction
    // whose local scale over its 900px evidence region stayed near 1.0.
    let local_scales = source_samples
        .into_iter()
        .map(|source| {
            let predicted = transformed_point(initial, source)?;
            let jacobian = virtual_tile_local_jacobian(correction, predicted)?;
            Some(jacobian.determinant().abs().sqrt())
        })
        .collect::<Option<Vec<_>>>();
    displacements.iter().all(|displacement| {
        displacement.norm() <= VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX
            && (*displacement - center).norm()
                <= VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX
    }) && local_scales.is_some_and(|scales| {
        scales.iter().all(|scale| {
            (VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MIN..=VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MAX)
                .contains(scale)
        })
    })
}

pub(super) fn virtual_tile_robust_translation_residual(
    measured_points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    inlier_threshold_px: f64,
) -> Option<VirtualTileResidualFit> {
    let residuals = virtual_tile_measured_residuals(measured_points, initial);
    let mut hypotheses = residuals
        .iter()
        .map(|(_, residual)| *residual)
        .collect::<Vec<_>>();
    hypotheses.sort_by(|left, right| {
        left.x
            .total_cmp(&right.x)
            .then_with(|| left.y.total_cmp(&right.y))
    });
    hypotheses.dedup_by(|left, right| left.x == right.x && left.y == right.y);

    let mut best = None::<VirtualTileResidualFit>;
    for residual in hypotheses {
        let correction = Matrix3::new(1.0, 0.0, residual.x, 0.0, 1.0, residual.y, 0.0, 0.0, 1.0);
        let candidate = virtual_tile_residual_candidate(
            &correction,
            initial,
            measured_points,
            left_dimensions,
            right_dimensions,
            "translation",
            inlier_threshold_px,
        )?;
        if best
            .as_ref()
            .is_none_or(|current| virtual_tile_residual_fit_is_better(&candidate, current))
        {
            best = Some(candidate);
        }
    }

    let best = best?;
    let best_indices = best.inlier_indices.iter().copied().collect::<HashSet<_>>();
    let mut dx = residuals
        .iter()
        .filter(|(index, _)| best_indices.contains(index))
        .map(|(_, residual)| residual.x)
        .collect::<Vec<_>>();
    let mut dy = residuals
        .iter()
        .filter(|(index, _)| best_indices.contains(index))
        .map(|(_, residual)| residual.y)
        .collect::<Vec<_>>();
    dx.sort_by(f64::total_cmp);
    dy.sort_by(f64::total_cmp);
    if dx.is_empty() || dy.is_empty() {
        return Some(best);
    }
    let correction = Matrix3::new(
        1.0,
        0.0,
        dx[dx.len() / 2],
        0.0,
        1.0,
        dy[dy.len() / 2],
        0.0,
        0.0,
        1.0,
    );
    let refined = virtual_tile_residual_candidate(
        &correction,
        initial,
        measured_points,
        left_dimensions,
        right_dimensions,
        "translation",
        inlier_threshold_px,
    )?;
    if virtual_tile_residual_fit_is_better(&refined, &best) {
        Some(refined)
    } else {
        Some(best)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_refit_residual_affine(
    seed: Matrix3<f64>,
    residual_points: &[(Point2<f64>, Point2<f64>)],
    measured_points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    inlier_threshold_px: f64,
    evidence_bounds: (f64, f64, f64, f64),
) -> Option<VirtualTileResidualFit> {
    let mut correction = seed;
    let mut fit = virtual_tile_residual_candidate(
        &correction,
        initial,
        measured_points,
        left_dimensions,
        right_dimensions,
        "affine",
        inlier_threshold_px,
    )?;
    for _ in 0..2 {
        let points = fit
            .inlier_indices
            .iter()
            .map(|&index| residual_points[index])
            .collect::<Vec<_>>();
        let Some(refitted) = estimate_affine(&points) else {
            break;
        };
        if !virtual_tile_residual_correction_is_bounded(&refitted, initial, evidence_bounds) {
            break;
        }
        correction = refitted;
        fit = virtual_tile_residual_candidate(
            &correction,
            initial,
            measured_points,
            left_dimensions,
            right_dimensions,
            "affine",
            inlier_threshold_px,
        )?;
    }
    virtual_tile_residual_correction_is_bounded(&correction, initial, evidence_bounds)
        .then_some(fit)
}

/// How the residual model chooses among its passing stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VirtualTileResidualSelection {
    /// Stop at the lowest-DOF stage that passes (single-level rule).
    LowestComplexity,
    /// Evaluate every stage and keep the passing one with the most measured
    /// inliers; on an inlier tie a strictly lower median error wins, otherwise
    /// the lower-DOF stage stays. A coarse-to-fine level must carry
    /// the seed's rotation, scale and perspective error forward: a translation
    /// that explains only the overlap's central band would leave the finer level
    /// a correction it cannot bound.
    MostInliers,
}

#[derive(Debug)]
pub(super) struct VirtualTileResidualModelOutcome {
    pub(super) fit: Option<VirtualTileResidualFit>,
    pub(super) max_translation_consensus: usize,
    pub(super) translation: stack_report::ResidualModelStageRecord,
    pub(super) affine: stack_report::ResidualModelStageRecord,
    pub(super) projective: stack_report::ResidualModelStageRecord,
}

pub(super) fn residual_model_stage(
    fit: Option<&VirtualTileResidualFit>,
    bounded_correction: Option<bool>,
) -> stack_report::ResidualModelStageRecord {
    let candidate_inliers = fit.map_or(0, |fit| fit.inlier_indices.len());
    let final_support = fit.map_or(0.0, |fit| fit.spatial_support);
    let rejection_reason = if bounded_correction == Some(false) {
        "correction_out_of_bounds"
    } else if fit.is_none() {
        "no_fit"
    } else if candidate_inliers < STATION_RELATION_MIN_INLIERS {
        "insufficient_inliers"
    } else if final_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        "insufficient_support"
    } else {
        "accepted"
    };
    stack_report::ResidualModelStageRecord {
        candidate_inliers,
        final_support,
        bounded_correction,
        rejection_reason: rejection_reason.to_string(),
    }
}

pub(super) fn residual_model_not_run_stage() -> stack_report::ResidualModelStageRecord {
    stack_report::ResidualModelStageRecord {
        rejection_reason: "not_run".to_string(),
        ..Default::default()
    }
}

/// Residual model fit whose inlier tolerance is the requirement-6.3 world
/// pixel bound itself; the points are native tile pixels and the whole left
/// tile bounds the correction (the single-level rule).
#[cfg(test)]
pub(super) fn virtual_tile_fit_measured_residual_model(
    measured_points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
) -> VirtualTileResidualModelOutcome {
    virtual_tile_fit_measured_residual_model_with_threshold(
        measured_points,
        initial,
        left_dimensions,
        right_dimensions,
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
        virtual_tile_full_tile_bounds(left_dimensions),
        VirtualTileResidualSelection::LowestComplexity,
        StationOverlapMeasure::Rectangles,
    )
}

/// Re-measure a stage fit's requirement-6.4 support over `overlap`. The fits
/// themselves rank hypotheses by inliers and error only, so the (costlier)
/// covered overlap is measured once per stage rather than per hypothesis.
pub(super) fn virtual_tile_fit_with_overlap_support(
    mut fit: VirtualTileResidualFit,
    measured_points: &[(Point2<f64>, Point2<f64>)],
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    overlap: StationOverlapMeasure<'_>,
) -> VirtualTileResidualFit {
    if matches!(overlap, StationOverlapMeasure::Covered { .. }) {
        let inlier_points = fit
            .inlier_indices
            .iter()
            .map(|&index| measured_points[index])
            .collect::<Vec<_>>();
        fit.spatial_support = station_relation_spatial_support_with(
            &inlier_points,
            &fit.measured_homography,
            left_dimensions,
            right_dimensions,
            overlap,
        );
    }
    fit
}

/// Residual model fit in the pixels of one probe level. `inlier_threshold_px`
/// is a symmetric error in those pixels, so the finest pyramid level passes the
/// requirement-6.3 bound divided by its area-averaging factor. The correction
/// bounds are evaluated over `evidence_bounds` (left-tile pixels), the region
/// the probes actually measured; each stage's support over `overlap`.
#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_fit_measured_residual_model_with_threshold(
    measured_points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    inlier_threshold_px: f64,
    evidence_bounds: (f64, f64, f64, f64),
    selection: VirtualTileResidualSelection,
    overlap: StationOverlapMeasure<'_>,
) -> VirtualTileResidualModelOutcome {
    let with_support = |fit: VirtualTileResidualFit| {
        virtual_tile_fit_with_overlap_support(
            fit,
            measured_points,
            left_dimensions,
            right_dimensions,
            overlap,
        )
    };
    let translation = virtual_tile_robust_translation_residual(
        measured_points,
        initial,
        left_dimensions,
        right_dimensions,
        inlier_threshold_px,
    )
    .map(with_support);
    let max_translation_consensus = translation
        .as_ref()
        .map_or(0, |fit| fit.inlier_indices.len());
    let translation_correction = translation.as_ref().and_then(|fit| {
        initial
            .try_inverse()
            .map(|initial_inverse| fit.measured_homography * initial_inverse)
    });
    let translation_bounded = translation_correction.as_ref().map(|correction| {
        virtual_tile_residual_correction_is_bounded(correction, initial, evidence_bounds)
    });
    let translation_stage = residual_model_stage(translation.as_ref(), translation_bounded);
    if selection == VirtualTileResidualSelection::LowestComplexity
        && translation_stage.rejection_reason == "accepted"
    {
        return VirtualTileResidualModelOutcome {
            fit: translation,
            max_translation_consensus,
            translation: translation_stage,
            affine: residual_model_not_run_stage(),
            projective: residual_model_not_run_stage(),
        };
    }

    // Escalate model complexity only when the lower-DOF correction cannot pass
    // the unchanged inlier/support/bounds gates (or, for a coarse-to-fine
    // level, evaluate every stage). The fit uses predicted target points as
    // coordinates, but every accepted inlier is recomputed from the original
    // measured source/target pair under H_residual * H_initial.
    let mut indexed_residual_points = measured_points
        .iter()
        .enumerate()
        .filter_map(|(index, (source, target))| {
            transformed_point(initial, *source).map(|predicted| (index, predicted, *target))
        })
        .collect::<Vec<_>>();
    indexed_residual_points.sort_by(|left, right| {
        left.1
            .y
            .total_cmp(&right.1.y)
            .then_with(|| left.1.x.total_cmp(&right.1.x))
            .then_with(|| left.2.y.total_cmp(&right.2.y))
            .then_with(|| left.2.x.total_cmp(&right.2.x))
    });
    let sorted_pairs = indexed_residual_points
        .iter()
        .map(|(_, predicted, target)| (*predicted, *target))
        .collect::<Vec<_>>();
    let residual_points = measured_points
        .iter()
        .map(|(source, target)| {
            (
                transformed_point(initial, *source).unwrap_or(*source),
                *target,
            )
        })
        .collect::<Vec<_>>();

    // The single-level rule seeds the affine with the source-stage tolerance;
    // a pyramid level seeds it at its own precision so its spurious peaks
    // cannot pull the seed out of bounds.
    let affine_seed_threshold = match selection {
        VirtualTileResidualSelection::LowestComplexity => FOCUS_MODEL_INLIER_THRESHOLD,
        VirtualTileResidualSelection::MostInliers => inlier_threshold_px,
    };
    let affine_seed = robust_transform_fit_with_threshold(
        &sorted_pairs,
        3,
        0xD1EC_7A11_AFF1_4E01,
        affine_seed_threshold,
        estimate_affine,
    );
    let affine_seed_bounded = affine_seed.as_ref().map(|seed| {
        virtual_tile_residual_correction_is_bounded(&seed.transform, initial, evidence_bounds)
    });
    let affine = affine_seed
        .filter(|_| affine_seed_bounded == Some(true))
        .and_then(|seed| {
            virtual_tile_refit_residual_affine(
                seed.transform,
                &residual_points,
                measured_points,
                initial,
                left_dimensions,
                right_dimensions,
                inlier_threshold_px,
                evidence_bounds,
            )
        })
        .map(with_support);
    let affine_bounded = affine
        .as_ref()
        .map(|fit| {
            initial.try_inverse().is_some_and(|initial_inverse| {
                let correction = fit.measured_homography * initial_inverse;
                virtual_tile_residual_correction_is_bounded(&correction, initial, evidence_bounds)
            })
        })
        .or(affine_seed_bounded);
    let affine_stage = residual_model_stage(affine.as_ref(), affine_bounded);
    if selection == VirtualTileResidualSelection::LowestComplexity
        && affine_stage.rejection_reason == "accepted"
    {
        return VirtualTileResidualModelOutcome {
            fit: affine,
            max_translation_consensus,
            translation: translation_stage,
            affine: affine_stage,
            projective: residual_model_not_run_stage(),
        };
    }

    let projective_seed = processing::find_homography_ransac_points_stable_with_min_inliers(
        &sorted_pairs,
        inlier_threshold_px,
        STATION_RELATION_MIN_INLIERS,
    );
    let projective_correction = projective_seed.as_ref().map(|(seed, seed_inliers)| {
        let seed_points = seed_inliers
            .iter()
            .map(|&index| sorted_pairs[index])
            .collect::<Vec<_>>();
        processing::compute_homography(&seed_points).unwrap_or(*seed)
    });
    let projective_bounded = projective_correction.as_ref().map(|correction| {
        virtual_tile_residual_correction_is_bounded(correction, initial, evidence_bounds)
    });
    let projective = projective_correction
        .filter(|_| projective_bounded == Some(true))
        .and_then(|correction| {
            virtual_tile_residual_candidate(
                &correction,
                initial,
                measured_points,
                left_dimensions,
                right_dimensions,
                "small_projective",
                inlier_threshold_px,
            )
        })
        .map(with_support);
    let projective_stage = residual_model_stage(projective.as_ref(), projective_bounded);
    let fit = match selection {
        VirtualTileResidualSelection::LowestComplexity => (projective_stage.rejection_reason
            == "accepted")
            .then_some(projective)
            .flatten(),
        VirtualTileResidualSelection::MostInliers => {
            // Lowest DOF first: a later stage replaces it with strictly more
            // measured inliers, or with as many and a strictly lower median
            // error. Under a coarse tolerance a translation and an affine can
            // explain the same points; only the lower error shows which one
            // carries the seed's rotation and scale remainder.
            let mut chosen = None::<VirtualTileResidualFit>;
            for (fit, stage) in [
                (translation, &translation_stage),
                (affine, &affine_stage),
                (projective, &projective_stage),
            ] {
                let Some(fit) = fit.filter(|_| stage.rejection_reason == "accepted") else {
                    continue;
                };
                if chosen.as_ref().is_none_or(|current| {
                    fit.inlier_indices.len() > current.inlier_indices.len()
                        || (fit.inlier_indices.len() == current.inlier_indices.len()
                            && fit.median_error_px < current.median_error_px)
                }) {
                    chosen = Some(fit);
                }
            }
            chosen
        }
    };
    VirtualTileResidualModelOutcome {
        fit,
        max_translation_consensus,
        translation: translation_stage,
        affine: affine_stage,
        projective: projective_stage,
    }
}

/// Bounded bidirectional grid probe of one tile pair in the pixels of the
/// supplied planes. `inlier_threshold_px` is the symmetric residual-model
/// tolerance in those pixels (see [`virtual_tile_pyramid_station_match`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_direct_station_match(
    left: usize,
    right: usize,
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    inlier_threshold_px: f64,
) -> (
    Option<MatchInfo>,
    stack_report::StationRelationCandidateRecord,
) {
    virtual_tile_direct_station_match_with_structure(
        left,
        right,
        tiles,
        coverages,
        initial_homographies,
        inlier_threshold_px,
        None,
    )
}

/// [`virtual_tile_direct_station_match`] whose structural edge-orientation
/// measurements (prefit diagnostic and fitted gate) read `structure_planes`
/// `(left, right)` instead of the probe planes. The patch and gradient NCC
/// filters always read the probe planes. `None` measures on the probe planes.
#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_direct_station_match_with_structure(
    left: usize,
    right: usize,
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    inlier_threshold_px: f64,
    structure_planes: Option<(&GrayImage, &GrayImage)>,
) -> (
    Option<MatchInfo>,
    stack_report::StationRelationCandidateRecord,
) {
    let mut diagnostic = stack_report::StationRelationCandidateRecord {
        left,
        right,
        left_tile: Default::default(),
        right_tile: Default::default(),
        predicted_overlap_area_px: 0.0,
        model_fitted: false,
        evidence_kind: stack_report::StationRelationEvidenceKind::VirtualTile,
        grid_probe_count: 0,
        measurable_patch_count: 0,
        bidirectional_match_count: 0,
        occupied_grid_cells: 0,
        search_radius_px: 0.0,
        residual_dx_px: Default::default(),
        residual_dy_px: Default::default(),
        residual_magnitude_px: Default::default(),
        max_residual_translation_consensus: 0,
        residual_translation: Default::default(),
        residual_affine: Default::default(),
        residual_projective: Default::default(),
        residual_model: "not_run".to_string(),
        fitted_inliers: 0,
        hull_support: 0.0,
        median_prefit_structural_orientation_difference_degrees: None,
        median_fitted_orientation_difference_degrees: None,
        failure_stage: "missing_input".to_string(),
        pyramid_factors: Vec::new(),
        measurement_factor: 1,
        seed: VIRTUAL_TILE_SEED_RENDERING_PRIOR.to_string(),
        prior_free_inliers: 0,
        after_prior_repair: false,
        hypotheses: Vec::new(),
    };
    let (Some(left_tile), Some(right_tile), Some(left_coverage), Some(right_coverage)) = (
        tiles.get(left),
        tiles.get(right),
        coverages.get(left),
        coverages.get(right),
    ) else {
        return (None, diagnostic);
    };
    diagnostic.failure_stage = "missing_initial_pose".to_string();
    let (Some(left_to_world), Some(right_to_world)) = (
        initial_homographies.get(&left_tile.id),
        initial_homographies.get(&right_tile.id),
    ) else {
        return (None, diagnostic);
    };
    let Some(world_to_right) = right_to_world.try_inverse() else {
        return (None, diagnostic);
    };
    let left_to_right = world_to_right * left_to_world;
    if left_to_right.try_inverse().is_none() {
        return (None, diagnostic);
    }
    let (left_structure, right_structure) =
        structure_planes.unwrap_or((&left_tile.alignment_image, &right_tile.alignment_image));
    diagnostic.failure_stage = "no_predicted_overlap".to_string();
    let Some(overlap_bounds) = virtual_tile_predicted_overlap_bounds(
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
    ) else {
        return (None, diagnostic);
    };
    let Some(predicted_overlap_area) = station_relation_overlap_area(
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
        true,
    ) else {
        return (None, diagnostic);
    };
    diagnostic.predicted_overlap_area_px = predicted_overlap_area;
    let grid = virtual_tile_direct_grid_points(overlap_bounds);
    diagnostic.grid_probe_count = grid.len();
    diagnostic.failure_stage = "insufficient_grid_probes".to_string();
    if grid.len() < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    let search_radius = virtual_tile_guided_search_radius(
        left_tile,
        right_tile,
        predicted_overlap_area,
        grid.len(),
    )
    .min(VIRTUAL_TILE_DIRECT_MAX_SEARCH_RADIUS_PX)
    .ceil() as i32;
    diagnostic.search_radius_px = f64::from(search_radius);

    let mut by_cell = BTreeMap::<(usize, usize), Vec<(f64, Point2<f64>, Point2<f64>, f64)>>::new();
    let mut prefit_orientations = Vec::new();
    for source in grid {
        if !virtual_tile_direct_patch_support_is_covered(
            left_coverage,
            right_coverage,
            &left_to_right,
            source,
            VIRTUAL_TILE_DIRECT_PATCH_RADIUS,
            search_radius,
        ) {
            continue;
        }
        let forward_probe = registration::probe_warped_patch(
            &left_tile.alignment_image,
            &right_tile.alignment_image,
            &left_to_right,
            source,
            VIRTUAL_TILE_DIRECT_PATCH_RADIUS,
            search_radius,
            VIRTUAL_TILE_DIRECT_PATCH_GATES,
        );
        if forward_probe.best_correlation.is_some() {
            diagnostic.measurable_patch_count += 1;
        }
        let Some(forward) = forward_probe.matched else {
            continue;
        };
        let Some(structure) = virtual_tile_structural_score_points(
            &left_tile.alignment_image,
            &right_tile.alignment_image,
            left_coverage,
            right_coverage,
            source,
            forward.target,
            &left_to_right,
        ) else {
            continue;
        };
        let prefit_orientation = if structure_planes.is_some() {
            virtual_tile_structural_score_points(
                left_structure,
                right_structure,
                left_coverage,
                right_coverage,
                source,
                forward.target,
                &left_to_right,
            )
            .map(|score| score.orientation_difference_degrees)
        } else {
            Some(structure.orientation_difference_degrees)
        };
        if let Some(orientation) = prefit_orientation {
            prefit_orientations.push(orientation);
        }
        if structure.patch_ncc < VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC
            || structure.gradient_ncc < VIRTUAL_TILE_GUIDED_MIN_GRADIENT_NCC
        {
            continue;
        }
        let Some(measured_right_to_left) =
            virtual_tile_measured_reverse_transform(&left_to_right, source, forward.target)
        else {
            continue;
        };
        if !virtual_tile_direct_patch_support_is_covered(
            right_coverage,
            left_coverage,
            &measured_right_to_left,
            forward.target,
            VIRTUAL_TILE_DIRECT_PATCH_RADIUS,
            search_radius,
        ) {
            continue;
        }
        let reverse_probe = registration::probe_warped_patch(
            &right_tile.alignment_image,
            &left_tile.alignment_image,
            &measured_right_to_left,
            forward.target,
            VIRTUAL_TILE_DIRECT_PATCH_RADIUS,
            search_radius,
            VIRTUAL_TILE_DIRECT_PATCH_GATES,
        );
        let Some(reverse) = reverse_probe.matched else {
            continue;
        };
        if !virtual_tile_round_trip_is_consistent(source, reverse.target) {
            continue;
        }
        diagnostic.bidirectional_match_count += 1;
        let cell = virtual_tile_overlap_grid_cell(source, overlap_bounds);
        let rank = structure.patch_ncc + 0.5 * structure.gradient_ncc
            - 0.05 * (reverse.target - source).norm();
        by_cell.entry(cell).or_default().push((
            rank,
            source,
            forward.target,
            structure.orientation_difference_degrees,
        ));
    }
    diagnostic.median_prefit_structural_orientation_difference_degrees =
        median_value(&mut prefit_orientations);
    diagnostic.occupied_grid_cells = by_cell.len();
    diagnostic.failure_stage = "insufficient_measurable_patches".to_string();
    if diagnostic.measurable_patch_count < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    diagnostic.failure_stage = "insufficient_bidirectional_matches".to_string();
    if diagnostic.bidirectional_match_count < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    diagnostic.failure_stage = "insufficient_grid_coverage".to_string();
    if diagnostic.occupied_grid_cells < VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS {
        return (None, diagnostic);
    }

    // Rank locally, but do not discard candidates before finding a globally
    // coherent residual model. Repeated artwork can give every cell a strong
    // yet different local peak; truncating each cell first made those modes
    // indistinguishable to RANSAC. The strict measured model selects one mode,
    // then the per-cell quota only limits its already verified inliers.
    for proposals in by_cell.values_mut() {
        proposals.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.y.total_cmp(&right.1.y))
                .then_with(|| left.1.x.total_cmp(&right.1.x))
                .then_with(|| left.2.y.total_cmp(&right.2.y))
                .then_with(|| left.2.x.total_cmp(&right.2.x))
        });
    }
    let maximum_cell_candidates = by_cell.values().map(Vec::len).max().unwrap_or(0);
    let mut all_selected = Vec::with_capacity(diagnostic.bidirectional_match_count);
    for rank in 0..maximum_cell_candidates {
        for (&cell, proposals) in &by_cell {
            if let Some(proposal) = proposals.get(rank) {
                all_selected.push((cell, *proposal));
            }
        }
    }
    let candidate_points = all_selected
        .iter()
        .map(|(_, (_, source, target, _))| (*source, *target))
        .collect::<Vec<_>>();
    // Requirement 6.4 on the pixels both tiles cover, in this plane's pixels.
    let overlap_measure = StationOverlapMeasure::Covered {
        source: left_coverage,
        target: right_coverage,
        factor: 1,
    };
    diagnostic.hull_support = station_relation_spatial_support_with(
        &candidate_points,
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
        overlap_measure,
    );
    diagnostic.failure_stage = "insufficient_prefit_hull_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        return (None, diagnostic);
    }

    let measured_residuals = virtual_tile_measured_residuals(&candidate_points, &left_to_right);
    diagnostic.residual_dx_px = residual_distribution(
        measured_residuals
            .iter()
            .map(|(_, residual)| residual.x)
            .collect(),
    );
    diagnostic.residual_dy_px = residual_distribution(
        measured_residuals
            .iter()
            .map(|(_, residual)| residual.y)
            .collect(),
    );
    diagnostic.residual_magnitude_px = residual_distribution(
        measured_residuals
            .iter()
            .map(|(_, residual)| residual.norm())
            .collect(),
    );

    diagnostic.failure_stage = "residual_model_no_fit".to_string();
    // Bound the correction where the probes measured it: the extent of the
    // bidirectional matches inside the predicted overlap. The far corners of a
    // tile that overlaps along one strip would turn any rotation or perspective
    // remainder of the seed into an apparent large correction there.
    let evidence_bounds = candidate_points.iter().fold(
        (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ),
        |(min_x, min_y, max_x, max_y), (source, _)| {
            (
                min_x.min(source.x),
                min_y.min(source.y),
                max_x.max(source.x),
                max_y.max(source.y),
            )
        },
    );
    let model = virtual_tile_fit_measured_residual_model_with_threshold(
        &candidate_points,
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
        inlier_threshold_px,
        evidence_bounds,
        VirtualTileResidualSelection::MostInliers,
        overlap_measure,
    );
    diagnostic.max_residual_translation_consensus = model.max_translation_consensus;
    diagnostic.residual_translation = model.translation.clone();
    diagnostic.residual_affine = model.affine.clone();
    diagnostic.residual_projective = model.projective.clone();
    let Some(fit) = model.fit else {
        return (None, diagnostic);
    };
    diagnostic.model_fitted = true;
    diagnostic.residual_model = fit.model.to_string();
    let refined = fit.measured_homography;

    let fitted_indices = fit.inlier_indices.into_iter().collect::<HashSet<_>>();
    let mut fitted_by_cell =
        BTreeMap::<(usize, usize), Vec<(f64, Point2<f64>, Point2<f64>, f64)>>::new();
    let mut fitted_proposals = Vec::with_capacity(fitted_indices.len());
    for (index, (cell, proposal)) in all_selected.into_iter().enumerate() {
        if fitted_indices.contains(&index) {
            fitted_by_cell.entry(cell).or_default().push(proposal);
            fitted_proposals.push(proposal);
        }
    }
    for proposals in fitted_by_cell.values_mut() {
        proposals.truncate(VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA);
    }
    let mut selected = Vec::new();
    for rank in 0..VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA {
        for proposals in fitted_by_cell.values() {
            if let Some(proposal) = proposals.get(rank) {
                selected.push(*proposal);
            }
        }
    }
    // Requirement 6.4 measures the convex hull of the *inliers*. The per-cell
    // quota only limits how strongly a textured cell weighs in the pose graph,
    // so it must not shrink that hull: keep the fitted inliers that are hull
    // vertices on either station plane as well.
    let hull_vertex = |vertices: &[Point2<f64>], point: Point2<f64>| {
        vertices
            .iter()
            .any(|vertex| vertex.x == point.x && vertex.y == point.y)
    };
    let source_hull =
        convex_hull_vertices(fitted_proposals.iter().map(|&(_, source, _, _)| source));
    let target_hull =
        convex_hull_vertices(fitted_proposals.iter().map(|&(_, _, target, _)| target));
    for &proposal in &fitted_proposals {
        let (_, source, target, _) = proposal;
        let already = selected
            .iter()
            .any(|&(_, selected_source, selected_target, _)| {
                selected_source == source && selected_target == target
            });
        if !already && (hull_vertex(&source_hull, source) || hull_vertex(&target_hull, target)) {
            selected.push(proposal);
        }
    }
    diagnostic.failure_stage = "insufficient_matches_after_model_quota".to_string();
    if selected.len() < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    let points = selected
        .iter()
        .map(|&(_, source, target, _)| (source, target))
        .collect::<Vec<_>>();
    diagnostic.fitted_inliers = points.len();
    diagnostic.hull_support = station_relation_spatial_support_with(
        &points,
        &refined,
        left_tile.dimensions(),
        right_tile.dimensions(),
        overlap_measure,
    );
    diagnostic.failure_stage = "insufficient_fitted_hull_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        return (None, diagnostic);
    }

    let mut fitted_orientations = selected
        .iter()
        .filter_map(|&(_, source, target, _)| {
            virtual_tile_structural_score_points(
                left_structure,
                right_structure,
                left_coverage,
                right_coverage,
                source,
                target,
                &refined,
            )
            .map(|score| score.orientation_difference_degrees)
        })
        .collect::<Vec<_>>();
    diagnostic.median_fitted_orientation_difference_degrees =
        median_value(&mut fitted_orientations);
    diagnostic.failure_stage = "fitted_orientation_mismatch".to_string();
    if diagnostic
        .median_fitted_orientation_difference_degrees
        .is_some_and(|difference| difference > VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES)
    {
        return (None, diagnostic);
    }

    diagnostic.failure_stage = "fitted".to_string();
    (
        Some(MatchInfo {
            homography: refined,
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
        }),
        diagnostic,
    )
}

// ---------------------------------------------------------------------------
// Coarse-to-fine Virtual_Tile station matching (任务 10 checkpoint)
//
// `langyuan-10` measured two independent failures of a native-pixel probe
// centred on the rendering prior (the source-level world poses the tiles are
// rendered with):
//
// * the fine texture of two focus-fused stations is not comparable. Seeded by
//   an independent SIFT relation (median error ~1.6px), 13px and 33px native
//   windows still gave only 58–92 of ~900 bidirectional matches, with a 32–35°
//   structural orientation difference (spurious peaks). The same 13-sample
//   window on 4×4 area averages gave 408–605 with 13–22°, and 0→2 was
//   accepted from the rough prior alone at 1/8 and 1/4;
// * the prior itself can be grossly wrong: a false source-level coarse bridge
//   put one station ~3,400px away at the wrong scale, where no bounded search
//   can reach it. Covered-pixel FAST/BRIEF at analysis resolution found its
//   true neighbour (253–260 RANSAC inliers) without any prior.
//
// The authoritative relation always comes from a bounded pixel probe, run
// coarse to fine on area-averaged planes. A prior-free feature relation only
// seeds that measurement; it never becomes a Station_Relation itself. The
// default anchor-grid fusion keeps placement out of the pixels, so the
// independently measured relation can go through the solver's gates directly.
// Globally warped comparison tiles instead repair the whole cluster's prior,
// render again, and measure around the repaired prior.
// ---------------------------------------------------------------------------

/// Stable seed identifiers of a Virtual_Tile relation measurement.
pub(super) const VIRTUAL_TILE_SEED_RENDERING_PRIOR: &str = "rendering_prior";
pub(super) const VIRTUAL_TILE_SEED_PRIOR_FREE: &str = "prior_free_features";
/// Smallest native-pixel extent one direct-probe window must span before two
/// focus-fused stations present comparable structure (measured above; the
/// intra-station registrar needed 51px windows for the same reason, 任务 7.5).
pub(super) const VIRTUAL_TILE_MIN_MATCH_WINDOW_NATIVE_PX: f64 = 48.0;
/// Smallest long side on which a pyramid level still carries a probe grid.
pub(super) const VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE: u32 = 384;
/// Symmetric residual-model tolerance of the octaves coarser than the finest,
/// in their own pixels.
pub(super) const VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX: f64 = 1.5;
/// Covered-feature budget and support scales of the prior-free proposal,
/// identical to the scalable source analysis path.
pub(super) const VIRTUAL_TILE_PRIOR_FREE_MAX_FEATURES: usize = 1_600;
pub(super) const VIRTUAL_TILE_PRIOR_FREE_SCALES: [f32; 3] =
    [0.38, 0.5, std::f32::consts::FRAC_1_SQRT_2];

/// One area-averaged level of a Virtual_Tile's covered luminance plane.
///
/// A level pixel is covered only when every native pixel of its block is, so
/// uncovered payload never reaches a probe (需求 6.1). Level pixel `p` is the
/// mean of the native block `[f·p, f·p + f)`, whose centre is
/// `f·p + (f − 1) / 2`; [`Self::level_to_native`] is exactly that map.
#[derive(Clone)]
pub(super) struct VirtualTileMatchPlane {
    pub(super) factor: u32,
    pub(super) gray: GrayImage,
    pub(super) coverage: stitching::CoverageMask,
}

impl VirtualTileMatchPlane {
    pub(super) fn from_native(
        gray: &GrayImage,
        coverage: &stitching::CoverageMask,
        factor: u32,
    ) -> Option<Self> {
        let factor = factor.max(1);
        let (width, height) = gray.dimensions();
        if coverage.dimensions() != (width, height) {
            return None;
        }
        if factor == 1 {
            return Some(Self {
                factor,
                gray: gray.clone(),
                coverage: coverage.clone(),
            });
        }
        let level_width = width / factor;
        let level_height = height / factor;
        if level_width == 0 || level_height == 0 {
            return None;
        }
        let area = f64::from(factor * factor);
        let row_len = level_width as usize;
        let mut values = vec![0u8; row_len * level_height as usize];
        let mut covered = vec![0u8; row_len * level_height as usize];
        // Every output element is written by exactly one task and no value is
        // reduced across tasks, so the result is independent of scheduling.
        values
            .par_chunks_mut(row_len)
            .zip(covered.par_chunks_mut(row_len))
            .enumerate()
            .for_each(|(y, (value_row, covered_row))| {
                let y = y as u32;
                for x in 0..level_width {
                    let mut sum = 0u32;
                    let mut all_covered = true;
                    'block: for dy in 0..factor {
                        for dx in 0..factor {
                            let (sample_x, sample_y) = (x * factor + dx, y * factor + dy);
                            if !coverage.is_covered(sample_x, sample_y) {
                                all_covered = false;
                                break 'block;
                            }
                            sum += u32::from(gray.get_pixel(sample_x, sample_y)[0]);
                        }
                    }
                    if all_covered {
                        value_row[x as usize] = (f64::from(sum) / area).round() as u8;
                        covered_row[x as usize] = u8::MAX;
                    }
                }
            });
        Some(Self {
            factor,
            gray: GrayImage::from_raw(level_width, level_height, values)?,
            coverage: stitching::CoverageMask::from_bytes(level_width, level_height, covered)
                .ok()?,
        })
    }

    pub(super) fn level_to_native(factor: u32) -> Matrix3<f64> {
        let scale = f64::from(factor.max(1));
        let offset = (scale - 1.0) * 0.5;
        Matrix3::new(scale, 0.0, offset, 0.0, scale, offset, 0.0, 0.0, 1.0)
    }

    pub(super) fn native_to_level(factor: u32) -> Matrix3<f64> {
        let scale = f64::from(factor.max(1));
        let offset = (scale - 1.0) * 0.5;
        Matrix3::new(
            1.0 / scale,
            0.0,
            -offset / scale,
            0.0,
            1.0 / scale,
            -offset / scale,
            0.0,
            0.0,
            1.0,
        )
    }
}

/// Coarse-to-fine area-averaging factors, coarsest first.
///
/// The finest level makes the 13-sample probe window span at least
/// [`VIRTUAL_TILE_MIN_MATCH_WINDOW_NATIVE_PX`], but never shrinks a tile below
/// [`VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE`]. Coarser octaves are added while
/// they still fit: each doubles how far the ±16px search and the bounded
/// residual correction reach, so a prior that is off by a rotation or a
/// perspective remainder is corrected in steps that stay inside the bounds.
/// Small (synthetic) tiles therefore keep a single native level.
pub(super) fn virtual_tile_pyramid_factors(shortest_long_side: u32) -> Vec<u32> {
    let window = f64::from(2 * VIRTUAL_TILE_DIRECT_PATCH_RADIUS + 1);
    let wanted = (VIRTUAL_TILE_MIN_MATCH_WINDOW_NATIVE_PX / window)
        .ceil()
        .max(1.0) as u32;
    let largest = (shortest_long_side / VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE).max(1);
    let mut factors = vec![wanted.min(largest).max(1)];
    while factors[0].saturating_mul(2) <= largest {
        factors.insert(0, factors[0] * 2);
    }
    factors
}

/// Probe-level metadata for one station: the bounded matcher treats a level as
/// its own pixel grid, so the plane is exposed with `scale_factor = 1`.
pub(super) fn virtual_tile_plane_image_info(
    tile: &ImageInfo,
    plane: &VirtualTileMatchPlane,
) -> ImageInfo {
    ImageInfo {
        id: tile.id,
        filename: tile.filename.clone(),
        width: plane.gray.width(),
        height: plane.gray.height(),
        alignment_image: plane.gray.clone(),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

/// Covered luminance of a rendered Virtual_Tile; uncovered payload is zero.
pub(super) fn virtual_tile_covered_luminance(
    pixels: &Rgb32FImage,
    coverage: &stitching::CoverageMask,
) -> GrayImage {
    let (width, height) = pixels.dimensions();
    if coverage.dimensions() != (width, height) {
        return GrayImage::new(width, height);
    }
    GrayImage::from_fn(width, height, |x, y| {
        if !coverage.is_covered(x, y) {
            return image::Luma([0]);
        }
        let pixel = pixels.get_pixel(x, y);
        let luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
        image::Luma([(luminance.clamp(0.0, 1.0) * 255.0).round() as u8])
    })
}

/// FAST/BRIEF features of a covered plane, keeping only keypoints whose
/// [`VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS`] window is covered, the same rule
/// as the native covered-feature path. (Scaling that window by
/// `1 / support_scale` discarded two thirds of the `langyuan-10` 1→2 inliers:
/// the overlaps are strips along tile borders.) Uncovered plane pixels are
/// zero, so a descriptor can only see a flat border, never uncovered payload.
pub(super) fn virtual_tile_plane_features(
    plane: &VirtualTileMatchPlane,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
) -> Vec<Feature> {
    let (width, height) = plane.gray.dimensions();
    processing::find_features_multiscale(
        &plane.gray,
        brief_pairs,
        VIRTUAL_TILE_PRIOR_FREE_MAX_FEATURES,
        &VIRTUAL_TILE_PRIOR_FREE_SCALES,
    )
    .into_iter()
    .filter(|feature| {
        let radius = VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS;
        let (x, y) = (feature.keypoint.x, feature.keypoint.y);
        x >= radius
            && y >= radius
            && x.saturating_add(radius) < width
            && y.saturating_add(radius) < height
            && (y - radius..=y + radius).all(|sample_y| {
                (x - radius..=x + radius)
                    .all(|sample_x| plane.coverage.is_covered(sample_x, sample_y))
            })
    })
    .collect()
}

/// Station planes of every probe level, `[level][station]`, coarsest first.
pub(super) struct VirtualTileMatchPyramid {
    pub(super) factors: Vec<u32>,
    pub(super) levels: Vec<Vec<ImageInfo>>,
    pub(super) coverages: Vec<Vec<stitching::CoverageMask>>,
    /// Native tile dimensions per station.
    pub(super) native_dimensions: Vec<(u32, u32)>,
    /// Feature proposal grids are independent of the bounded probe grids.
    pub(super) feature_factors: Vec<u32>,
    /// Native structural images for the final, one-native-pixel round trip.
    pub(super) native_refinement: Vec<Option<(GrayImage, stitching::CoverageMask)>>,
    /// Coverage_Mask of each station's relation-quality plane, in the pixels
    /// of the `alignment_image` [`install_virtual_tile_match_planes`] installs.
    pub(super) quality_coverages: Vec<stitching::CoverageMask>,
}

impl VirtualTileMatchPyramid {
    pub(super) fn new(factors: Vec<u32>, station_count: usize) -> Self {
        let empty_coverage = || {
            stitching::CoverageMask::from_bytes(1, 1, vec![0])
                .expect("a one-pixel empty Coverage_Mask is valid")
        };
        let empty_info = || ImageInfo {
            id: usize::MAX,
            filename: String::new(),
            width: 1,
            height: 1,
            alignment_image: GrayImage::new(1, 1),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        };
        let level_count = factors.len();
        Self {
            factors,
            levels: (0..level_count)
                .map(|_| (0..station_count).map(|_| empty_info()).collect())
                .collect(),
            coverages: (0..level_count)
                .map(|_| (0..station_count).map(|_| empty_coverage()).collect())
                .collect(),
            native_dimensions: vec![(1, 1); station_count],
            feature_factors: vec![1; station_count],
            native_refinement: vec![None; station_count],
            quality_coverages: (0..station_count).map(|_| empty_coverage()).collect(),
        }
    }

    /// The masks the station solver re-checks each relation on: the
    /// relation-quality planes (requirement 6.8) and the finest match planes
    /// (requirement 6.4's overlap area).
    pub(super) fn solver_coverages(&self) -> VirtualTileSolverCoverages<'_> {
        VirtualTileSolverCoverages {
            quality: &self.quality_coverages,
            overlap: self.coverages.last().map_or(&[][..], Vec::as_slice),
            overlap_factor: self.factors.last().copied().unwrap_or(1),
        }
    }

    /// Replace one station's planes and return them, coarsest first.
    pub(super) fn set_station(
        &mut self,
        station: usize,
        tile: &ImageInfo,
        gray: &GrayImage,
        coverage: &stitching::CoverageMask,
    ) -> Option<Vec<VirtualTileMatchPlane>> {
        let mut planes = Vec::with_capacity(self.factors.len());
        self.native_dimensions[station] = gray.dimensions();
        self.native_refinement[station] = (gray.width().max(gray.height()) >= 4096
            && self.factors.last().is_some_and(|factor| *factor > 1))
        .then(|| {
            (
                imageproc::filter::gaussian_blur_f32(gray, 1.0),
                coverage.clone(),
            )
        });
        for (level, &factor) in self.factors.iter().enumerate() {
            let plane = VirtualTileMatchPlane::from_native(gray, coverage, factor)?;
            self.levels[level][station] = virtual_tile_plane_image_info(tile, &plane);
            self.coverages[level][station] = plane.coverage.clone();
            planes.push(plane);
        }
        (!planes.is_empty()).then_some(planes)
    }
}

/// Octaves above the finest one at which the station solver compares edges.
pub(super) const VIRTUAL_TILE_RELATION_QUALITY_OCTAVES: u32 = 2;
/// Gaussian σ, in plane pixels, of the structural relation-quality plane.
pub(super) const VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA: f32 = 1.0;
/// Half-width, in relation-quality plane pixels, of the neighbourhood that must
/// be covered around a requirement-6.8 sample: the ±1px gradient taps with
/// bilinear access read 2 pixels around it and the σ = 1 blur (imageproc
/// kernel radius ⌈2σ⌉) 2 more. On langyuan-10 the uncovered payload inside
/// that reach alone moved the verified 1→2 relation's edge-strength ratio
/// from 1.01 to 1.40 and its low-frequency difference from 0.002 to 0.13.
pub(super) const VIRTUAL_TILE_QUALITY_COVERAGE_RADIUS_PX: f64 = 4.0;

/// The plane on which the station solver re-checks requirement 6.8.
///
/// Its `focus_overlap_quality_measurements` compares ±1-pixel gradients at a
/// sparse grid. On langyuan-10 the verified 0→2 and 1→2 relations read a
/// median edge-orientation difference of 21–38° on the unfiltered 1/4–1/16
/// planes — the level of a 40px-shifted wrong relation (38–46°), because single
/// pixel gradients follow fine silk texture. Two octaves above the finest level
/// and after a σ = 1 blur they read 4.6–5.9°, against 27–28° for the shifted
/// relation. A single matching level still needs a separate structural quality
/// plane: the probe pyramid's minimum size must not switch verification back
/// to fine silk texture. Keep at least 64 pixels on its shorter side so the
/// dense grid, gradient taps and covered blur neighbourhood remain measurable.
pub(super) fn virtual_tile_relation_quality_plane(
    planes: &[VirtualTileMatchPlane],
    native: &GrayImage,
    coverage: &stitching::CoverageMask,
) -> (GrayImage, u32) {
    let Some(finest) = planes.last() else {
        return (GrayImage::new(1, 1), 1);
    };
    if planes.len() < 2 {
        // The probe's factor can be truncated to fit a small preview. It is
        // not a structural scale: multiplying that factor by four gave only
        // ~100 pixels across real previews and followed illumination gradients
        // instead of artwork edges. Choose the quality grid from native pixels.
        let wanted = native
            .width()
            .max(native.height())
            .div_ceil(600)
            .max(1)
            .next_power_of_two()
            .max(1 << VIRTUAL_TILE_RELATION_QUALITY_OCTAVES);
        let factor = wanted.min((native.width().min(native.height()) / 64).max(1));
        let structural = VirtualTileMatchPlane::from_native(native, coverage, factor)
            .expect("a validated probe plane has valid grayscale and coverage dimensions");
        return (
            imageproc::filter::gaussian_blur_f32(
                &structural.gray,
                VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
            ),
            factor,
        );
    }
    let wanted = finest
        .factor
        .saturating_mul(1 << VIRTUAL_TILE_RELATION_QUALITY_OCTAVES);
    let plane = planes
        .iter()
        .filter(|plane| plane.factor <= wanted)
        .max_by_key(|plane| plane.factor)
        .unwrap_or(finest);
    (
        imageproc::filter::gaussian_blur_f32(&plane.gray, VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA),
        plane.factor,
    )
}

/// A coarse pyramid failure may reflect the level's sparse residual model,
/// while a finer level can still measure the same prior.  Keep this retry
/// list deliberately narrow: a missing overlap or missing texture is not
/// evidence that becomes stronger at a finer level.
pub(super) fn virtual_tile_coarse_failure_can_retry(failure_stage: &str) -> bool {
    matches!(
        failure_stage,
        // At the coarsest octave a narrow, but valid, overlap can contain
        // fewer than the final minimum number of grid probes.  That is a
        // sampling-density failure of this octave, not evidence that the
        // station pair is unrelated.  Retry at the next finer bounded octave
        // and let the unchanged final polish/Station_Relation gates decide.
        "insufficient_grid_probes"
            | "insufficient_measurable_patches"
            | "insufficient_bidirectional_matches"
            | "insufficient_grid_coverage"
            | "residual_model_no_fit"
            | "insufficient_matches_after_model_quota"
            | "insufficient_fitted_hull_support"
    )
}

/// Probe one pair coarse to fine around `seed_left_to_right` (native tile
/// pixels). Every octave but the finest runs the bounded bidirectional probe
/// (edge orientation read on the octave's σ = 1 structural plane) and re-seeds
/// the next with its measured model (tolerance
/// [`VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX`] in its own pixels); the
/// finest octave polishes that model with model-warped templates
/// ([`virtual_tile_polished_station_match`]). A single native level (small
/// tiles) keeps the bounded probe with the requirement-6.3 tolerance. The
/// relation is expressed in native tile pixels; the diagnostic describes the
/// last stage that ran.
pub(super) fn virtual_tile_pyramid_station_match(
    left: usize,
    right: usize,
    pyramid: &VirtualTileMatchPyramid,
    seed_left_to_right: &Matrix3<f64>,
    seed_kind: &'static str,
) -> (Vec<MatchInfo>, stack_report::StationRelationCandidateRecord) {
    let empty_diagnostic = || {
        let (_, mut diagnostic) = virtual_tile_direct_station_match(
            left,
            right,
            &[],
            &[],
            &HashMap::new(),
            STATION_RELATION_MAX_MEDIAN_ERROR_PX,
        );
        diagnostic.pyramid_factors = pyramid.factors.clone();
        diagnostic.seed = seed_kind.to_string();
        diagnostic
    };
    let level_count = pyramid.factors.len();
    if level_count == 0
        || left >= pyramid.native_dimensions.len()
        || right >= pyramid.native_dimensions.len()
    {
        return (Vec::new(), empty_diagnostic());
    }
    let single_level = level_count == 1;
    let bounded_levels = if single_level { 1 } else { level_count - 1 };
    let mut seed = *seed_left_to_right;
    let mut diagnostic = empty_diagnostic();
    for level in 0..bounded_levels {
        let factor = pyramid.factors[level];
        let to_native = VirtualTileMatchPlane::level_to_native(factor);
        let from_native = VirtualTileMatchPlane::native_to_level(factor);
        let level_tiles = &pyramid.levels[level];
        let Some(level_seed_inverse) = (from_native * seed * to_native).try_inverse() else {
            diagnostic.failure_stage = "non_invertible_seed".to_string();
            return (Vec::new(), diagnostic);
        };
        let initial = HashMap::from([
            (level_tiles[left].id, Matrix3::identity()),
            (level_tiles[right].id, level_seed_inverse),
        ]);
        let inlier_threshold_px = if single_level {
            STATION_RELATION_MAX_MEDIAN_ERROR_PX
        } else {
            VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX
        };
        // A coarse octave reads edge orientation on its structural (σ = 1)
        // plane, the rule of the relation-quality plane: on langyuan-10 ±1px
        // gradients of the unfiltered 1/8 planes read 13.4–15.5° for verified
        // relations (the prior-free 1→2 proposal failed the 15° gate at 15.5°),
        // the structural planes 5.1–6.0°, and a 2-level-pixel shift 27–36°.
        let structure = (!single_level).then(|| {
            (
                imageproc::filter::gaussian_blur_f32(
                    &level_tiles[left].alignment_image,
                    VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
                ),
                imageproc::filter::gaussian_blur_f32(
                    &level_tiles[right].alignment_image,
                    VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
                ),
            )
        });
        let (relation, level_diagnostic) = virtual_tile_direct_station_match_with_structure(
            left,
            right,
            level_tiles,
            &pyramid.coverages[level],
            &initial,
            inlier_threshold_px,
            structure
                .as_ref()
                .map(|(left_plane, right_plane)| (left_plane, right_plane)),
        );
        diagnostic = level_diagnostic;
        diagnostic.pyramid_factors = pyramid.factors.clone();
        diagnostic.measurement_factor = factor;
        diagnostic.seed = seed_kind.to_string();
        let Some(relation) = relation else {
            // A later octave has a smaller residual in its own pixels and can
            // recover a model that was under-supported at this coarse level.
            // Keep the original seed until a measured relation is available;
            // no prior is ever promoted to acceptance evidence.
            if !single_level && virtual_tile_coarse_failure_can_retry(&diagnostic.failure_stage) {
                if level + 1 < bounded_levels {
                    continue;
                }
                // The last bounded octave has no finer bounded probe left.
                // Its residual-model/support failure is still a recoverable
                // measurement failure: the finest polish has independent
                // patch, reverse-consistency, hull, scale, and orientation
                // gates and must get the chance to verify the original prior.
                // Do not return the coarse diagnostic here, because that
                // would make the finest evidence unreachable for exactly the
                // sparse, shifted overlaps this retry is meant to recover.
                break;
            }
            return (Vec::new(), diagnostic);
        };
        let Some(native) = normalized_homography(&(to_native * relation.homography * from_native))
        else {
            diagnostic.failure_stage = "non_finite_native_relation".to_string();
            return (Vec::new(), diagnostic);
        };
        seed = native;
        if single_level && factor == 1 {
            let points = relation
                .points
                .iter()
                .filter_map(|(source, target)| {
                    transformed_point(&to_native, *source)
                        .zip(transformed_point(&to_native, *target))
                })
                .collect::<Vec<_>>();
            return (
                vec![MatchInfo {
                    homography: native,
                    inliers: points.len(),
                    points,
                    // Key-direction fit; see `MatchInfo::canonical_homography`.
                    canonical_homography: None,
                    ..relation
                }],
                diagnostic,
            );
        }
    }
    let finest = level_count - 1;
    let factor = pyramid.factors[finest];
    diagnostic.measurement_factor = factor;
    let relation = virtual_tile_polished_station_match(
        &pyramid.levels[finest][left],
        &pyramid.coverages[finest][left],
        &pyramid.levels[finest][right],
        &pyramid.coverages[finest][right],
        factor,
        &seed,
        (
            pyramid.native_dimensions[left],
            pyramid.native_dimensions[right],
        ),
        &mut diagnostic,
    );
    let relation = match (
        &pyramid.native_refinement[left],
        &pyramid.native_refinement[right],
    ) {
        (Some(left_native), Some(right_native)) => relation
            .into_iter()
            .filter_map(|relation| {
                virtual_tile_native_station_polish(
                    left_native,
                    right_native,
                    &relation,
                    &mut diagnostic,
                )
            })
            .collect(),
        _ => relation,
    };
    (relation, diagnostic)
}

/// Coarse pixels establish a search model. Final camera geometry is measured
/// on native pixels: a one-level-pixel reverse check at factor four otherwise
/// admits four-native-pixel errors and biases long projective tree chains.
pub(super) fn virtual_tile_native_station_polish(
    left: &(GrayImage, stitching::CoverageMask),
    right: &(GrayImage, stitching::CoverageMask),
    seed: &MatchInfo,
    diagnostic: &mut stack_report::StationRelationCandidateRecord,
) -> Option<MatchInfo> {
    let inverse = seed.homography.try_inverse()?;
    let step = (seed.points.len() / 1000).max(1);
    let supported = |mask: &stitching::CoverageMask, point: Point2<f64>| {
        let (w, h) = mask.dimensions();
        let radius = 40i32;
        let (x, y) = (point.x.round() as i32, point.y.round() as i32);
        x >= radius
            && y >= radius
            && x + radius < w as i32
            && y + radius < h as i32
            && (-radius..=radius).step_by(8).all(|dy| {
                (-radius..=radius)
                    .step_by(8)
                    .all(|dx| mask.is_covered((x + dx) as u32, (y + dy) as u32))
            })
    };
    let points = seed
        .points
        .par_iter()
        .enumerate()
        .filter(|(index, _)| index % step == 0)
        .filter_map(|(_, &(source, target))| {
            if !supported(&left.1, source) || !supported(&right.1, target) {
                return None;
            }
            let measured = residual_warp::verify_round_trip_match_with_template(
                &left.0,
                &right.0,
                &seed.homography,
                &inverse,
                source,
                31,
            )?;
            supported(&right.1, measured.forward).then_some((source, measured.forward))
        })
        .collect::<Vec<_>>();
    diagnostic.failure_stage = "insufficient_native_round_trip_matches".to_string();
    diagnostic.bidirectional_match_count = points.len();
    diagnostic.measurement_factor = 1;
    if points.len() < STATION_RELATION_MIN_INLIERS {
        return None;
    }
    let (homography, indices) = virtual_tile_polish_fit(
        &points,
        VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX,
        left.0.dimensions(),
    )?;
    let points = indices
        .into_iter()
        .map(|index| points[index])
        .collect::<Vec<_>>();
    diagnostic.fitted_inliers = points.len();
    diagnostic.hull_support = station_relation_spatial_support_with(
        &points,
        &homography,
        left.0.dimensions(),
        right.0.dimensions(),
        StationOverlapMeasure::Covered {
            source: &left.1,
            target: &right.1,
            factor: 1,
        },
    );
    diagnostic.residual_magnitude_px = residual_distribution(match homography.try_inverse() {
        Some(inverse) => points
            .iter()
            .map(|(a, b)| symmetric_point_error(&homography, &inverse, *a, *b))
            .collect(),
        None => Vec::new(),
    });
    diagnostic.failure_stage = "insufficient_native_polished_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT
        || median_symmetric_error(&homography, &points) > STATION_RELATION_MAX_MEDIAN_ERROR_PX
    {
        return None;
    }
    diagnostic.residual_model = "native_polished_projective".to_string();
    diagnostic.failure_stage = "fitted".to_string();
    Some(MatchInfo {
        homography,
        inliers: points.len(),
        points,
        canonical_homography: None,
        ..seed.clone()
    })
}

// ---------------------------------------------------------------------------
// Finest-octave polish
//
// The bounded 13-sample probe is a good *seed* but a poor final instrument on
// the finest octave: on langyuan-10 only 23–30% of its 1/4 bidirectional
// matches agreed with an independent relation, and the ones that did covered
// 16–19% of the 0→2 overlap. Matching 15×15 templates of the left plane against
// the right plane resampled through the current relation (so the template and
// the window share one geometry) gave 846–1011 inliers at a 0.9–1.1px one-way
// median, and 0→2 reached 0.22–0.28 support at a 1.2–1.9px one-way median once
// the inlier tolerance admitted the overlap's non-homographic remainder.
// Requirement 6.3 bounds the inliers' *median*; the solver re-checks it.
// ---------------------------------------------------------------------------

pub(super) const VIRTUAL_TILE_POLISH_TEMPLATE_HALF: i32 = 7;
pub(super) const VIRTUAL_TILE_POLISH_GRID_STEP_PX: f64 = 16.0;
/// Search radii of the two polish passes, in finest-plane pixels. The first
/// absorbs what the coarser octaves left; the second re-matches around it.
pub(super) const VIRTUAL_TILE_POLISH_SEARCH_PASSES: [i32; 2] = [8, 4];
pub(super) const VIRTUAL_TILE_POLISH_MIN_NCC: f64 = 0.6;
pub(super) const VIRTUAL_TILE_POLISH_MIN_PEAK_MARGIN: f64 = 0.03;
/// Smallest template standard deviation in 8-bit levels.
pub(super) const VIRTUAL_TILE_POLISH_MIN_TEMPLATE_STD: f64 = 3.0;
/// Symmetric inlier tolerance of the polished relation in native pixels: both
/// one-way errors within the requirement-6.3 3.0 world pixels. (Twice that let
/// the refits drift to a looser model: 0→2 read a 3.43px symmetric median,
/// against 2.09px at 0.29 support with this tolerance.)
pub(super) const VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX: f64 =
    std::f64::consts::SQRT_2 * STATION_RELATION_MAX_MEDIAN_ERROR_PX;

#[derive(Default)]
pub(super) struct VirtualTilePolishCounts {
    pub(super) probes: usize,
    pub(super) textured: usize,
    pub(super) matched: usize,
}

/// Zero-mean normalised cross-correlation of two equally long sample runs.
pub(super) fn virtual_tile_ncc(left: &[f64], right: &[f64]) -> Option<f64> {
    let count = left.len() as f64;
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / count;
    let right_mean = right.iter().sum::<f64>() / count;
    let mut product = 0.0;
    let mut left_energy = 0.0;
    let mut right_energy = 0.0;
    for (&l, &r) in left.iter().zip(right) {
        let (l, r) = (l - left_mean, r - right_mean);
        product += l * r;
        left_energy += l * l;
        right_energy += r * r;
    }
    let denominator = (left_energy * right_energy).sqrt();
    (denominator > f64::EPSILON).then(|| product / denominator)
}

/// Best integer offset of `template` inside a square `window` of side
/// `2·(half + search) + 1`, its NCC, and the best NCC at least two pixels away.
#[allow(clippy::type_complexity)]
pub(super) fn virtual_tile_best_window_offset(
    template: &[f64],
    window: &[f64],
    half: i32,
    search: i32,
) -> Option<((i32, i32), f64, f64, Vec<f64>)> {
    let side = (2 * (half + search) + 1) as usize;
    let template_side = (2 * half + 1) as usize;
    let offsets = (2 * search + 1) as usize;
    let mut scores = vec![-1.0; offsets * offsets];
    let mut sample = vec![0.0; template.len()];
    for oy in 0..offsets {
        for ox in 0..offsets {
            for ty in 0..template_side {
                let row = (oy + ty) * side + ox;
                sample[ty * template_side..(ty + 1) * template_side]
                    .copy_from_slice(&window[row..row + template_side]);
            }
            if let Some(score) = virtual_tile_ncc(template, &sample) {
                scores[oy * offsets + ox] = score;
            }
        }
    }
    let (best_index, &best) =
        scores
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })?;
    let (bx, by) = ((best_index % offsets) as i32, (best_index / offsets) as i32);
    let second = scores
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            let (x, y) = ((index % offsets) as i32, (index / offsets) as i32);
            (x - bx).abs() >= 2 || (y - by).abs() >= 2
        })
        .map(|(_, &score)| score)
        .fold(-1.0f64, f64::max);
    Some(((bx - search, by - search), best, second, scores))
}

/// Model-warped template correspondences `(left plane, right plane)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_polish_correspondences(
    left: &GrayImage,
    left_coverage: &stitching::CoverageMask,
    right: &GrayImage,
    right_coverage: &stitching::CoverageMask,
    left_to_right: &Matrix3<f64>,
    bounds: (f64, f64, f64, f64),
    search: i32,
    counts: &mut VirtualTilePolishCounts,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let half = VIRTUAL_TILE_POLISH_TEMPLATE_HALF;
    #[cfg(test)]
    let half = std::env::var("RAW_EDITOR_TILE_REPLAY_TEMPLATE_HALF")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(half);
    let reach = half + search;
    let (min_x, min_y, max_x, max_y) = bounds;
    let mut centers = Vec::new();
    let mut y = (min_y + f64::from(reach)).ceil();
    while y <= max_y - f64::from(reach) {
        let mut x = (min_x + f64::from(reach)).ceil();
        while x <= max_x - f64::from(reach) {
            centers.push((x as i32, y as i32));
            x += VIRTUAL_TILE_POLISH_GRID_STEP_PX;
        }
        y += VIRTUAL_TILE_POLISH_GRID_STEP_PX;
    }
    counts.probes += centers.len();
    let right_sample = |point: Point2<f64>| -> Option<f64> {
        let (x0, y0) = (point.x.floor(), point.y.floor());
        if x0 < 0.0 || y0 < 0.0 {
            return None;
        }
        let (x0, y0) = (x0 as u32, y0 as u32);
        [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)]
            .iter()
            .all(|&(x, y)| right_coverage.is_covered(x, y))
            .then(|| registration::sample_gray(right, point.x, point.y))
            .flatten()
    };
    // One task per centre; results are collected in centre order.
    let results = centers
        .par_iter()
        .map(|&(cx, cy)| {
            let mut template = Vec::with_capacity(((2 * half + 1) * (2 * half + 1)) as usize);
            for dy in -half..=half {
                for dx in -half..=half {
                    let (x, y) = (cx + dx, cy + dy);
                    if x < 0 || y < 0 || !left_coverage.is_covered(x as u32, y as u32) {
                        return (false, None);
                    }
                    template.push(f64::from(left.get_pixel(x as u32, y as u32)[0]));
                }
            }
            let mean = template.iter().sum::<f64>() / template.len() as f64;
            let std = (template.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / template.len() as f64)
                .sqrt();
            if std < VIRTUAL_TILE_POLISH_MIN_TEMPLATE_STD {
                return (false, None);
            }
            let side = (2 * reach + 1) as usize;
            let mut window = Vec::with_capacity(side * side);
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let point = Point2::new(f64::from(cx + dx), f64::from(cy + dy));
                    let Some(value) =
                        transformed_point(left_to_right, point).and_then(right_sample)
                    else {
                        return (true, None);
                    };
                    window.push(value);
                }
            }
            let Some(((ox, oy), best, second, scores)) =
                virtual_tile_best_window_offset(&template, &window, half, search)
            else {
                return (true, None);
            };
            if best < VIRTUAL_TILE_POLISH_MIN_NCC
                || best - second < VIRTUAL_TILE_POLISH_MIN_PEAK_MARGIN
                || ox.abs() >= search
                || oy.abs() >= search
            {
                return (true, None);
            }
            // Reverse: the matched warped-right content must find its way back
            // to the template centre in the left plane.
            let reverse_template = {
                let mut values = Vec::with_capacity(template.len());
                for dy in -half..=half {
                    for dx in -half..=half {
                        let row = (oy + dy + reach) as usize;
                        let column = (ox + dx + reach) as usize;
                        values.push(window[row * side + column]);
                    }
                }
                values
            };
            let back_search = 2;
            let back_reach = half + back_search;
            let mut back_window = Vec::with_capacity(((2 * back_reach + 1).pow(2)) as usize);
            for dy in -back_reach..=back_reach {
                for dx in -back_reach..=back_reach {
                    // The reverse template should land back at the original
                    // source centre.  Searching around the forward match
                    // (`cx + ox`) made the reverse offset equal to `-ox`,
                    // although `back_search` is intentionally only two
                    // pixels wide.  A valid forward correction of five
                    // pixels was therefore rejected before fitting.  Keep
                    // the reverse search in the source coordinate frame and
                    // require its offset to return to that centre.
                    let (x, y) = (cx + dx, cy + dy);
                    if x < 0 || y < 0 || !left_coverage.is_covered(x as u32, y as u32) {
                        return (true, None);
                    }
                    back_window.push(f64::from(left.get_pixel(x as u32, y as u32)[0]));
                }
            }
            let Some(((bx, by), _, _, _)) =
                virtual_tile_best_window_offset(&reverse_template, &back_window, half, back_search)
            else {
                return (true, None);
            };
            if f64::from(bx).hypot(f64::from(by)) > 1.0 {
                return (true, None);
            }
            // Separable parabolic sub-pixel peak on the forward score surface.
            let offsets = (2 * search + 1) as usize;
            let at =
                |x: i32, y: i32| scores[(y + search) as usize * offsets + (x + search) as usize];
            let parabola = |minus: f64, center: f64, plus: f64| {
                let denominator = minus - 2.0 * center + plus;
                if denominator.abs() > f64::EPSILON {
                    (0.5 * (minus - plus) / denominator).clamp(-0.5, 0.5)
                } else {
                    0.0
                }
            };
            let sx = parabola(at(ox - 1, oy), best, at(ox + 1, oy));
            let sy = parabola(at(ox, oy - 1), best, at(ox, oy + 1));
            let warped = Point2::new(f64::from(cx + ox) + sx, f64::from(cy + oy) + sy);
            let Some(target) = transformed_point(left_to_right, warped) else {
                return (true, None);
            };
            (
                true,
                Some((Point2::new(f64::from(cx), f64::from(cy)), target)),
            )
        })
        .collect::<Vec<_>>();
    counts.textured += results.iter().filter(|(textured, _)| *textured).count();
    let matches = results
        .into_iter()
        .filter_map(|(_, correspondence)| correspondence)
        .collect::<Vec<_>>();
    counts.matched += matches.len();
    matches
}

/// Robust homography over polished correspondences: RANSAC with a per-direction
/// tolerance of `threshold / √2`, then least-squares refits on the symmetric
/// inlier set.
pub(super) fn virtual_tile_polish_fit(
    correspondences: &[(Point2<f64>, Point2<f64>)],
    threshold: f64,
    left_dimensions: (u32, u32),
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    let (seed, mut inliers) = processing::find_homography_ransac_points_stable_with_min_inliers(
        correspondences,
        threshold / std::f64::consts::SQRT_2,
        STATION_RELATION_MIN_INLIERS,
    )?;
    let mut homography = seed;
    for _ in 0..3 {
        let points = inliers
            .iter()
            .map(|&index| correspondences[index])
            .collect::<Vec<_>>();
        let Some(refitted) = processing::compute_homography(&points) else {
            break;
        };
        let refitted_inliers = symmetric_inlier_indices(&refitted, correspondences, threshold);
        if refitted_inliers.len() < inliers.len() {
            break;
        }
        let converged = refitted_inliers == inliers;
        homography = refitted;
        inliers = refitted_inliers;
        if converged {
            break;
        }
    }
    let homography = normalized_homography(&homography)?;
    (inliers.len() >= STATION_RELATION_MIN_INLIERS
        && homography_preserves_focus_orientation(&homography, left_dimensions))
    .then_some((homography, inliers))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum VirtualTilePolishModel {
    Similarity,
    Affine,
    Projective,
}

impl VirtualTilePolishModel {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Similarity => "polished_similarity",
            Self::Affine => "polished_affine",
            Self::Projective => "polished_projective",
        }
    }

    pub(super) fn fit(
        self,
        correspondences: &[(Point2<f64>, Point2<f64>)],
        threshold: f64,
        left_dimensions: (u32, u32),
    ) -> Option<(Matrix3<f64>, Vec<usize>)> {
        if self == Self::Projective {
            // Preserve the retained project's exact projective RANSAC/refit
            // path. Multi-hypothesis polish adds lower-DOF competitors; it
            // must not perturb the projective evidence that already passed.
            return virtual_tile_polish_fit(correspondences, threshold, left_dimensions);
        }
        let fit = match self {
            Self::Similarity => robust_transform_fit_with_threshold(
                correspondences,
                2,
                0x504f_4c49_5348_5349,
                threshold,
                estimate_similarity,
            ),
            Self::Affine => robust_transform_fit_with_threshold(
                correspondences,
                3,
                0x504f_4c49_5348_4146,
                threshold,
                estimate_affine,
            ),
            Self::Projective => unreachable!("handled above"),
        }?;
        let homography = normalized_homography(&fit.transform)?;
        (fit.inlier_indices.len() >= STATION_RELATION_MIN_INLIERS
            && homography.try_inverse().is_some()
            && homography_preserves_focus_orientation(&homography, left_dimensions))
        .then_some((homography, fit.inlier_indices))
    }
}

#[derive(Clone)]
pub(super) struct VirtualTilePolishHypothesis {
    pub(super) model: VirtualTilePolishModel,
    pub(super) homography: Matrix3<f64>,
    pub(super) inliers: Vec<usize>,
    pub(super) correspondences: Vec<(Point2<f64>, Point2<f64>)>,
}

pub(super) fn virtual_tile_hypotheses_are_equivalent(
    left: &Matrix3<f64>,
    right: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> bool {
    focus_transform_disagreement_px(left, right, dimensions)
        .is_some_and(|difference| difference <= 1e-6)
}

/// Polish a pyramid relation on the finest planes. Every independently
/// measurable similarity, affine, and projective model survives to the shared
/// Station_Relation gate; this stage does not let one model's scale or hull
/// result suppress another model before dense covered-overlap measurement.
#[allow(clippy::too_many_arguments)]
pub(super) fn virtual_tile_polished_station_match(
    left_plane: &ImageInfo,
    left_coverage: &stitching::CoverageMask,
    right_plane: &ImageInfo,
    right_coverage: &stitching::CoverageMask,
    factor: u32,
    seed_native: &Matrix3<f64>,
    native_dimensions: ((u32, u32), (u32, u32)),
    diagnostic: &mut stack_report::StationRelationCandidateRecord,
) -> Vec<MatchInfo> {
    let to_native = VirtualTileMatchPlane::level_to_native(factor);
    let from_native = VirtualTileMatchPlane::native_to_level(factor);
    let threshold = VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX / f64::from(factor.max(1));
    let Some(seed) = normalized_homography(&(from_native * seed_native * to_native)) else {
        diagnostic.failure_stage = "non_invertible_seed".to_string();
        return Vec::new();
    };
    let mut hypotheses = Vec::<VirtualTilePolishHypothesis>::new();
    let mut representative_counts = VirtualTilePolishCounts::default();

    for (pass_index, search) in VIRTUAL_TILE_POLISH_SEARCH_PASSES.into_iter().enumerate() {
        let seeds = if pass_index == 0 {
            vec![(None, seed)]
        } else {
            hypotheses
                .iter()
                .map(|hypothesis| (Some(hypothesis.model), hypothesis.homography))
                .collect::<Vec<_>>()
        };
        if seeds.is_empty() {
            break;
        }
        let mut next = Vec::new();
        for (restricted_model, relation) in seeds {
            diagnostic.failure_stage = "no_predicted_overlap".to_string();
            let Some(bounds) = virtual_tile_predicted_overlap_bounds(
                &relation,
                left_plane.dimensions(),
                right_plane.dimensions(),
            ) else {
                continue;
            };
            diagnostic.predicted_overlap_area_px = diagnostic.predicted_overlap_area_px.max(
                station_relation_overlap_area(
                    &relation,
                    left_plane.dimensions(),
                    right_plane.dimensions(),
                    true,
                )
                .unwrap_or(0.0),
            );
            let mut counts = VirtualTilePolishCounts::default();
            let correspondences = virtual_tile_polish_correspondences(
                &left_plane.alignment_image,
                left_coverage,
                &right_plane.alignment_image,
                right_coverage,
                &relation,
                bounds,
                search,
                &mut counts,
            );
            if counts.matched > representative_counts.matched {
                representative_counts = counts;
            }
            if correspondences.len() < STATION_RELATION_MIN_INLIERS {
                continue;
            }
            for model in [
                VirtualTilePolishModel::Similarity,
                VirtualTilePolishModel::Affine,
                VirtualTilePolishModel::Projective,
            ] {
                if restricted_model.is_some_and(|restricted| restricted != model) {
                    continue;
                }
                let Some((homography, inliers)) =
                    model.fit(&correspondences, threshold, left_plane.dimensions())
                else {
                    continue;
                };
                next.push(VirtualTilePolishHypothesis {
                    model,
                    homography,
                    inliers,
                    correspondences: correspondences.clone(),
                });
            }
        }
        if pass_index == 0 {
            hypotheses = next;
        } else {
            // A first-pass lower-DOF model can cover the complete overlap even
            // when its own narrower second-pass search locks onto only one
            // repeated-texture band. Keep both independently measured fits;
            // the shared final gate, not polish iteration order, decides.
            hypotheses.extend(next);
        }
    }

    diagnostic.grid_probe_count = representative_counts.probes;
    diagnostic.measurable_patch_count = representative_counts.textured;
    diagnostic.bidirectional_match_count = representative_counts.matched;
    diagnostic.search_radius_px =
        f64::from(*VIRTUAL_TILE_POLISH_SEARCH_PASSES.last().unwrap_or(&0));
    diagnostic.failure_stage = "polished_fit_failed".to_string();

    let (left_native, right_native) = native_dimensions;
    let mut measured = hypotheses
        .into_iter()
        .filter_map(|hypothesis| {
            let native = normalized_homography(&(to_native * hypothesis.homography * from_native))?;
            let points = hypothesis
                .inliers
                .iter()
                .filter_map(|&index| {
                    let (source, target) = hypothesis.correspondences[index];
                    transformed_point(&to_native, source).zip(transformed_point(&to_native, target))
                })
                .collect::<Vec<_>>();
            (points.len() >= STATION_RELATION_MIN_INLIERS).then_some((
                hypothesis.model,
                native,
                points,
            ))
        })
        .collect::<Vec<_>>();
    measured.sort_by(|left, right| left.0.cmp(&right.0));
    let mut deduplicated = Vec::<(
        VirtualTilePolishModel,
        Matrix3<f64>,
        Vec<(Point2<f64>, Point2<f64>)>,
    )>::new();
    for hypothesis in measured {
        if deduplicated.iter().any(|existing| {
            virtual_tile_hypotheses_are_equivalent(&existing.1, &hypothesis.1, left_native)
        }) {
            continue;
        }
        deduplicated.push(hypothesis);
    }
    if deduplicated.is_empty() {
        return Vec::new();
    }

    let overlap = StationOverlapMeasure::Covered {
        source: left_coverage,
        target: right_coverage,
        factor,
    };
    let representative = deduplicated
        .iter()
        .max_by(|left, right| {
            left.2
                .len()
                .cmp(&right.2.len())
                .then_with(|| right.0.cmp(&left.0))
        })
        .expect("a non-empty hypothesis set has a representative");
    diagnostic.model_fitted = true;
    diagnostic.residual_model = "polished_multi_hypothesis".to_string();
    diagnostic.fitted_inliers = representative.2.len();
    diagnostic.hull_support = station_relation_spatial_support_with(
        &representative.2,
        &representative.1,
        left_native,
        right_native,
        overlap,
    );
    diagnostic.residual_magnitude_px =
        residual_distribution(match representative.1.try_inverse() {
            Some(inverse) => representative
                .2
                .iter()
                .map(|(source, target)| {
                    symmetric_point_error(&representative.1, &inverse, *source, *target)
                })
                .collect(),
            None => Vec::new(),
        });
    diagnostic.failure_stage = "fitted".to_string();

    deduplicated
        .into_iter()
        .map(|(model, homography, points)| {
            let _model_name = model.name();
            MatchInfo {
                homography,
                inliers: points.len(),
                sequence_bridge: false,
                coarse_bridge: false,
                points,
                candidate_points: Vec::new(),
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
                canonical_homography: None,
            }
        })
        .collect()
}

pub(super) fn normalized_homography(transform: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let scale = transform[(2, 2)];
    if !scale.is_finite() || scale.abs() <= f64::EPSILON {
        return None;
    }
    let normalized = transform / scale;
    normalized
        .iter()
        .all(|value| value.is_finite())
        .then_some(normalized)
}

/// Rendering-prior relation `left tile -> right tile` in native tile pixels.
pub(super) fn virtual_tile_prior_relation(
    left: &ImageInfo,
    right: &ImageInfo,
    tile_to_world: &HashMap<usize, Matrix3<f64>>,
) -> Option<Matrix3<f64>> {
    let left_to_world = tile_to_world.get(&left.id)?;
    let world_to_right = tile_to_world.get(&right.id)?.try_inverse()?;
    normalized_homography(&(world_to_right * left_to_world))
}

/// Prior-free covered-feature relation between two feature planes, in native
/// tile pixels, with its RANSAC inlier count. It is a proposal only.
pub(super) fn virtual_tile_prior_free_relation(
    left_factor: u32,
    left_features: &[Feature],
    right_factor: u32,
    right_features: &[Feature],
    left_native_dimensions: (u32, u32),
) -> Option<(Matrix3<f64>, usize)> {
    let matches = processing::match_features(left_features, right_features);
    let correspondences = matches
        .iter()
        .map(|matched| {
            let left = left_features[matched.index1].keypoint;
            let right = right_features[matched.index2].keypoint;
            (
                Point2::new(f64::from(left.x), f64::from(left.y)),
                Point2::new(f64::from(right.x), f64::from(right.y)),
            )
        })
        .collect::<Vec<_>>();
    let (homography, inliers) = processing::find_homography_ransac_points_stable_with_min_inliers(
        &correspondences,
        5.0,
        STATION_RELATION_MIN_INLIERS,
    )?;
    if inliers.len() < STATION_RELATION_MIN_INLIERS {
        return None;
    }
    let native = normalized_homography(
        &(VirtualTileMatchPlane::level_to_native(right_factor)
            * homography
            * VirtualTileMatchPlane::native_to_level(left_factor)),
    )?;
    (native.try_inverse().is_some()
        && homography_preserves_focus_orientation(&native, left_native_dimensions))
    .then_some((native, inliers.len()))
}

/// A covered-feature seed can still repair a rendering prior when the
/// subsequent finest-plane polish fitted a model but its inlier hull missed
/// the fixed spatial-support gate. The seed remains a proposal only: the
/// repaired station is rendered again and the ordinary bounded matcher must
/// accept the resulting relation before it enters the pose solver.
pub(super) fn virtual_tile_prior_free_seed_fallback_conditions(
    model_fitted: bool,
    failure_stage: &str,
    fitted_inliers: usize,
    prior_free_inliers: usize,
) -> bool {
    model_fitted
        && failure_stage == "insufficient_polished_support"
        && fitted_inliers >= STATION_RELATION_MIN_INLIERS
        // A covered-feature seed is only a repair proposal when it carries at
        // least as much support as the model that the finest-plane polish
        // already fitted.  A smaller seed can move an entire station cluster
        // on a repeated texture even when the polished model has hundreds of
        // mutually consistent matches but fails only its spatial-support gate.
        && prior_free_inliers >= fitted_inliers
}

pub(super) fn virtual_tile_prior_free_seed_fallback_allowed(
    diagnostic: &stack_report::StationRelationCandidateRecord,
) -> bool {
    virtual_tile_prior_free_seed_fallback_conditions(
        diagnostic.model_fitted,
        &diagnostic.failure_stage,
        diagnostic.fitted_inliers,
        diagnostic.prior_free_inliers,
    )
}

pub(super) fn log_virtual_tile_candidate(
    diagnostic: &stack_report::StationRelationCandidateRecord,
    accepted: bool,
) {
    println!(
        "  - Virtual_Tile candidate {}->{}: seed={} level=1/{} of {:?} accepted={} prior_free_inliers={} repaired={} grid_probes={} measurable={} bidirectional={} cells={} radius={:.1}px residual_mag=[{:.2},{:.2},{:.2}] translation={}/{:.4}/{:?}/{} affine={}/{:.4}/{:?}/{} projective={}/{:.4}/{:?}/{} residual_model={} fit_inliers={} support={:.4} prefit_structural_orientation={} fitted_orientation={} stage={}",
        diagnostic.left,
        diagnostic.right,
        diagnostic.seed,
        diagnostic.measurement_factor,
        diagnostic.pyramid_factors,
        accepted,
        diagnostic.prior_free_inliers,
        diagnostic.after_prior_repair,
        diagnostic.grid_probe_count,
        diagnostic.measurable_patch_count,
        diagnostic.bidirectional_match_count,
        diagnostic.occupied_grid_cells,
        diagnostic.search_radius_px,
        diagnostic.residual_magnitude_px.p10,
        diagnostic.residual_magnitude_px.median,
        diagnostic.residual_magnitude_px.p90,
        diagnostic.residual_translation.candidate_inliers,
        diagnostic.residual_translation.final_support,
        diagnostic.residual_translation.bounded_correction,
        diagnostic.residual_translation.rejection_reason,
        diagnostic.residual_affine.candidate_inliers,
        diagnostic.residual_affine.final_support,
        diagnostic.residual_affine.bounded_correction,
        diagnostic.residual_affine.rejection_reason,
        diagnostic.residual_projective.candidate_inliers,
        diagnostic.residual_projective.final_support,
        diagnostic.residual_projective.bounded_correction,
        diagnostic.residual_projective.rejection_reason,
        diagnostic.residual_model,
        diagnostic.fitted_inliers,
        diagnostic.hull_support,
        diagnostic
            .median_prefit_structural_orientation_difference_degrees
            .map_or_else(|| "n/a".to_string(), |value| format!("{value:.2}deg")),
        diagnostic
            .median_fitted_orientation_difference_degrees
            .map_or_else(|| "n/a".to_string(), |value| format!("{value:.2}deg")),
        diagnostic.failure_stage,
    );
}

pub(super) type VirtualTileRelationHypotheses = HashMap<(usize, usize), Vec<MatchInfo>>;

pub(super) fn station_relation_tile_geometry_record(
    tile: &ImageInfo,
    coverage: Option<&stitching::CoverageMask>,
    factor: u32,
    tile_to_world: Option<&Matrix3<f64>>,
) -> stack_report::StationRelationTileGeometryRecord {
    let factor = factor.max(1);
    let effective_bounds = coverage
        .and_then(stitching::CoverageMask::covered_bounds)
        .map(|(x, y, width, height)| {
            let native_x = x.saturating_mul(factor).min(tile.width);
            let native_y = y.saturating_mul(factor).min(tile.height);
            let native_right = x
                .saturating_add(width)
                .saturating_mul(factor)
                .min(tile.width);
            let native_bottom = y
                .saturating_add(height)
                .saturating_mul(factor)
                .min(tile.height);
            [
                native_x,
                native_y,
                native_right.saturating_sub(native_x),
                native_bottom.saturating_sub(native_y),
            ]
        });
    stack_report::StationRelationTileGeometryRecord {
        dimensions: [tile.width, tile.height],
        effective_bounds,
        tile_to_world: tile_to_world.map(matrix3_row_major),
    }
}

pub(super) fn populate_station_relation_candidate_geometry(
    diagnostic: &mut stack_report::StationRelationCandidateRecord,
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
    tile_to_world: &HashMap<usize, Matrix3<f64>>,
) {
    let factor = coverages.overlap_factor;
    if let Some(tile) = tiles.get(diagnostic.left) {
        diagnostic.left_tile = station_relation_tile_geometry_record(
            tile,
            coverages.overlap.get(diagnostic.left),
            factor,
            tile_to_world.get(&tile.id),
        );
    }
    if let Some(tile) = tiles.get(diagnostic.right) {
        diagnostic.right_tile = station_relation_tile_geometry_record(
            tile,
            coverages.overlap.get(diagnostic.right),
            factor,
            tile_to_world.get(&tile.id),
        );
    }
}

/// Authoritative relations: bounded coarse-to-fine hypotheses around the
/// rendering prior for every topology candidate (or for `only` when given).
/// Returns `(relations, diagnostics)`, diagnostics in candidate order.
pub(super) fn virtual_tile_measure_prior_relations(
    tiles: &[ImageInfo],
    pyramid: &VirtualTileMatchPyramid,
    tile_to_world: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
    only: Option<&BTreeSet<(usize, usize)>>,
    after_prior_repair: bool,
) -> (
    VirtualTileRelationHypotheses,
    Vec<stack_report::StationRelationCandidateRecord>,
) {
    let mut relations = VirtualTileRelationHypotheses::new();
    let mut diagnostics = Vec::with_capacity(station_topology.candidates.len());
    for candidate in &station_topology.candidates {
        let (left, right) = (candidate.left, candidate.right);
        if only.is_some_and(|pairs| !pairs.contains(&(left.min(right), left.max(right)))) {
            continue;
        }
        let prior = tiles
            .get(left)
            .zip(tiles.get(right))
            .and_then(|(left_tile, right_tile)| {
                virtual_tile_prior_relation(left_tile, right_tile, tile_to_world)
            });
        let (relation, mut diagnostic) = match prior {
            Some(prior) => virtual_tile_pyramid_station_match(
                left,
                right,
                pyramid,
                &prior,
                VIRTUAL_TILE_SEED_RENDERING_PRIOR,
            ),
            None => {
                let (_, mut diagnostic) = virtual_tile_direct_station_match(
                    left,
                    right,
                    &[],
                    &[],
                    &HashMap::new(),
                    STATION_RELATION_MAX_MEDIAN_ERROR_PX,
                );
                diagnostic.failure_stage = "missing_initial_pose".to_string();
                (Vec::new(), diagnostic)
            }
        };
        diagnostic.after_prior_repair = after_prior_repair;
        if let Some(selection) = select_virtual_tile_relation_hypothesis(
            tiles,
            left,
            right,
            &relation,
            pyramid.solver_coverages(),
        ) {
            diagnostic.hypotheses = selection.outcomes;
        }
        populate_station_relation_candidate_geometry(
            &mut diagnostic,
            tiles,
            pyramid.solver_coverages(),
            tile_to_world,
        );
        log_virtual_tile_candidate(&diagnostic, !relation.is_empty());
        if !relation.is_empty() {
            relations.insert((left, right), relation);
        }
        diagnostics.push(diagnostic);
    }
    (relations, diagnostics)
}

/// A densely verified prior-free relation between two stations whose rendering
/// priors disagree with it. It may only repair a prior; it is never a
/// Station_Relation.
#[derive(Clone, Copy, Debug)]
pub(super) struct VirtualTilePriorProposal {
    pub(super) left: usize,
    pub(super) right: usize,
    /// Measured `left tile -> right tile` relation in native tile pixels.
    pub(super) left_to_right: Matrix3<f64>,
    pub(super) inliers: usize,
}

/// One cluster member's rendering-prior repair: `world_correction` maps the
/// station's current world placement onto the placement the proposal measured
/// relative to an already consistent station.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct VirtualTilePriorRepair {
    pub(super) station: usize,
    pub(super) cluster_root: usize,
    pub(super) fixed_station: usize,
    pub(super) relation: (usize, usize),
    pub(super) relation_inliers: usize,
    pub(super) world_correction: Matrix3<f64>,
}

/// A rendering-prior repair may move a station only within the same bounded
/// geometry envelope used by closure corrections and the station scale gate.
/// Keep this decision pure so production and property tests share the exact
/// finite-value and inclusive-boundary semantics.
pub(super) fn virtual_tile_prior_repair_is_bounded(
    center_shift_px: f64,
    correction_scale_ratio: f64,
) -> bool {
    center_shift_px.is_finite()
        && center_shift_px >= 0.0
        && center_shift_px <= CLOSURE_MAX_CORNER_CORRECTION_PX
        && correction_scale_ratio.is_finite()
        && (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
            .contains(&correction_scale_ratio)
}

pub(super) fn virtual_tile_prior_repair_measurements(
    correction: &Matrix3<f64>,
    station_world_center: Point2<f64>,
) -> Option<(f64, f64)> {
    let corrected_center = transformed_point(correction, station_world_center)?;
    let center_shift_px = (corrected_center - station_world_center).norm();
    let correction_scale_ratio =
        virtual_tile_repair_local_scale_ratio(correction, station_world_center)?;
    (center_shift_px.is_finite() && correction_scale_ratio.is_finite())
        .then_some((center_shift_px, correction_scale_ratio))
}

pub(super) fn virtual_tile_prior_repair_is_bounded_at_center(
    correction: &Matrix3<f64>,
    station_world_center: Point2<f64>,
) -> bool {
    virtual_tile_prior_repair_measurements(correction, station_world_center).is_some_and(
        |(center_shift_px, correction_scale_ratio)| {
            virtual_tile_prior_repair_is_bounded(center_shift_px, correction_scale_ratio)
        },
    )
}

/// Apply the exact pure Station_Relation gates to a Virtual_Tile match before
/// it is allowed to merge rendering-prior repair clusters. This mirrors the
/// measurements in `solve_station_poses_for_evidence_with_report` without
/// recording a second report or running any degradation side effects.
pub(super) fn virtual_tile_relation_is_authoritative(
    left: usize,
    right: usize,
    relation: &MatchInfo,
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> bool {
    if left >= tiles.len()
        || right >= tiles.len()
        || left == right
        || relation.sequence_bridge
        || relation.points.len() < FOCUS_MODEL_MIN_INLIERS
    {
        return false;
    }
    let canonical = match canonical_relation_view(tiles, left, right, relation) {
        Ok(view) => view,
        Err(CanonicalRelationViewError::NonInvertible) => return false,
        Err(CanonicalRelationViewError::IndexOutOfBounds {
            left,
            right,
            image_count,
        }) => panic!("canonical relation pair ({left}, {right}) is outside {image_count} tiles"),
    };
    let quality_source = canonical.source_index;
    let quality_target = canonical.target_index;
    let quality_points = if canonical.inverted {
        relation
            .points
            .iter()
            .map(|&(source, target)| (target, source))
            .collect::<Vec<_>>()
    } else {
        relation.points.clone()
    };
    let quality_coverages = coverages
        .quality
        .get(quality_source)
        .zip(coverages.quality.get(quality_target));
    let dense_measurements = focus_overlap_quality_measurements_on_coverage(
        &tiles[quality_source],
        &tiles[quality_target],
        &canonical.homography,
        quality_coverages,
    );
    let overlap_measure = coverages.overlap_measure(quality_source, quality_target);
    measure_station_relation_candidate(
        relation.inliers.max(quality_points.len()),
        &canonical.homography,
        &quality_points,
        tiles[quality_source].dimensions(),
        tiles[quality_target].dimensions(),
        overlap_measure,
        dense_measurements,
    )
    .is_some_and(|candidate| station_relation_rejection_reasons(&candidate.measurements).is_empty())
}

pub(super) fn matrix3_row_major(transform: &Matrix3<f64>) -> [f64; 9] {
    [
        transform[(0, 0)],
        transform[(0, 1)],
        transform[(0, 2)],
        transform[(1, 0)],
        transform[(1, 1)],
        transform[(1, 2)],
        transform[(2, 0)],
        transform[(2, 1)],
        transform[(2, 2)],
    ]
}

pub(super) fn virtual_tile_hypothesis_model_rank(transform: &Matrix3<f64>) -> u8 {
    let Some(transform) = normalized_homography(transform) else {
        return u8::MAX;
    };
    if transform[(2, 0)].abs() > 1e-12 || transform[(2, 1)].abs() > 1e-12 {
        return 2;
    }
    let first = nalgebra::Vector2::new(transform[(0, 0)], transform[(1, 0)]);
    let second = nalgebra::Vector2::new(transform[(0, 1)], transform[(1, 1)]);
    let scale = first.norm().max(second.norm()).max(f64::EPSILON);
    if first.dot(&second).abs() <= 1e-10 * scale * scale
        && (first.norm() - second.norm()).abs() <= 1e-10 * scale
    {
        0
    } else {
        1
    }
}

pub(super) fn virtual_tile_hypothesis_matrix_key(transform: &Matrix3<f64>) -> [u64; 9] {
    let mut key = [0; 9];
    for (slot, value) in key.iter_mut().zip(transform.iter()) {
        *slot = value.to_bits();
    }
    key
}

#[derive(Clone)]
pub(super) struct MeasuredVirtualTileHypothesis {
    pub(super) relation: MatchInfo,
    pub(super) canonical_source: usize,
    pub(super) canonical_target: usize,
    pub(super) canonical_homography: Matrix3<f64>,
    pub(super) correspondence_count: usize,
    pub(super) correspondence_sample: Vec<stack_report::StationRelationCorrespondenceRecord>,
    pub(super) holdout_fit_count: usize,
    pub(super) holdout_count: usize,
    pub(super) holdout_median_error_px: Option<f64>,
    pub(super) holdout_spatial_support: Option<f64>,
    pub(super) holdout_transform: Option<Matrix3<f64>>,
    pub(super) measured: MeasuredStationRelationCandidate,
    pub(super) rejection_count: usize,
    pub(super) model_rank: u8,
}

#[derive(Clone)]
pub(super) struct VirtualTileHypothesisSelection {
    pub(super) relation: MatchInfo,
    pub(super) passes: bool,
    pub(super) outcomes: Vec<stack_report::StationRelationHypothesisRecord>,
}

#[allow(clippy::type_complexity)]
pub(super) fn virtual_tile_hypothesis_holdout(
    model_rank: u8,
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    overlap_measure: Option<StationOverlapMeasure<'_>>,
) -> (usize, usize, Option<f64>, Option<f64>, Option<Matrix3<f64>>) {
    let cell = f64::from(source_dimensions.0.max(source_dimensions.1).max(1)) / 12.0;
    let mut fit = Vec::new();
    let mut holdout = Vec::new();
    for &point in points {
        let column = (point.0.x / cell).floor() as i64;
        let row = (point.0.y / cell).floor() as i64;
        if (column + row).rem_euclid(2) == 0 {
            fit.push(point);
        } else {
            holdout.push(point);
        }
    }
    let model = match model_rank {
        0 => VirtualTilePolishModel::Similarity,
        1 => VirtualTilePolishModel::Affine,
        2 => VirtualTilePolishModel::Projective,
        _ => return (fit.len(), holdout.len(), None, None, None),
    };
    let transform = model
        .fit(
            &fit,
            VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX,
            source_dimensions,
        )
        .map(|(transform, _)| transform);
    let median_error_px = transform
        .as_ref()
        .map(|transform| median_symmetric_error(transform, &holdout))
        .filter(|error| error.is_finite());
    let spatial_support = transform.as_ref().and_then(|transform| {
        overlap_measure.map(|measure| {
            station_relation_spatial_support_with(
                &holdout,
                transform,
                source_dimensions,
                target_dimensions,
                measure,
            )
        })
    });
    (
        fit.len(),
        holdout.len(),
        median_error_px,
        spatial_support,
        transform,
    )
}

/// Measure, canonicalize, deduplicate, and rank one pair's independent model
/// hypotheses. Fully passing hypotheses always outrank failing hypotheses;
/// ties use the complete deterministic quality tuple and then intrinsic model
/// complexity/matrix bits, never input order. Both provisional clustering and
/// the final solver call this exact selector.
pub(super) fn select_virtual_tile_relation_hypothesis(
    tiles: &[ImageInfo],
    left: usize,
    right: usize,
    hypotheses: &[MatchInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> Option<VirtualTileHypothesisSelection> {
    let mut measured = Vec::<MeasuredVirtualTileHypothesis>::new();
    for relation in hypotheses {
        if left == right
            || relation.sequence_bridge
            || relation.points.len() < FOCUS_MODEL_MIN_INLIERS
        {
            continue;
        }
        let canonical = match canonical_relation_view(tiles, left, right, relation) {
            Ok(view) => view,
            Err(CanonicalRelationViewError::NonInvertible) => continue,
            Err(CanonicalRelationViewError::IndexOutOfBounds {
                left,
                right,
                image_count,
            }) => {
                panic!("canonical relation pair ({left}, {right}) is outside {image_count} tiles")
            }
        };
        let points: Cow<'_, [(Point2<f64>, Point2<f64>)]> = if canonical.inverted {
            Cow::Owned(
                relation
                    .points
                    .iter()
                    .map(|&(source, target)| (target, source))
                    .collect(),
            )
        } else {
            Cow::Borrowed(&relation.points)
        };
        let quality_coverages = coverages
            .quality
            .get(canonical.source_index)
            .zip(coverages.quality.get(canonical.target_index));
        let dense_measurements = focus_overlap_quality_measurements_on_coverage(
            &tiles[canonical.source_index],
            &tiles[canonical.target_index],
            &canonical.homography,
            quality_coverages,
        );
        let overlap_measure =
            coverages.overlap_measure(canonical.source_index, canonical.target_index);
        let Some(candidate) = measure_station_relation_candidate(
            relation.inliers.max(points.len()),
            &canonical.homography,
            &points,
            tiles[canonical.source_index].dimensions(),
            tiles[canonical.target_index].dimensions(),
            overlap_measure,
            dense_measurements,
        ) else {
            continue;
        };
        let correspondence_count = points.len();
        let sample_count = correspondence_count.min(64);
        let correspondence_sample = (0..sample_count)
            .map(|sample| {
                let index = if sample_count <= 1 {
                    0
                } else {
                    sample * (correspondence_count - 1) / (sample_count - 1)
                };
                let (source, target) = points[index];
                let local_scale = virtual_tile_local_jacobian(&canonical.homography, source)
                    .map(|jacobian| jacobian.determinant().abs().sqrt());
                stack_report::StationRelationCorrespondenceRecord {
                    source: [source.x, source.y],
                    target: [target.x, target.y],
                    local_scale,
                }
            })
            .collect();
        let (
            holdout_fit_count,
            holdout_count,
            holdout_median_error_px,
            holdout_spatial_support,
            holdout_transform,
        ) = virtual_tile_hypothesis_holdout(
            virtual_tile_hypothesis_model_rank(&canonical.homography),
            &points,
            tiles[canonical.source_index].dimensions(),
            tiles[canonical.target_index].dimensions(),
            overlap_measure,
        );
        measured.push(MeasuredVirtualTileHypothesis {
            relation: relation.clone(),
            canonical_source: canonical.source_index,
            canonical_target: canonical.target_index,
            canonical_homography: canonical.homography,
            correspondence_count,
            correspondence_sample,
            holdout_fit_count,
            holdout_count,
            holdout_median_error_px,
            holdout_spatial_support,
            holdout_transform,
            measured: candidate,
            rejection_count: station_relation_rejection_reasons(&candidate.measurements).len(),
            model_rank: virtual_tile_hypothesis_model_rank(&canonical.homography),
        });
    }
    measured.sort_by(|left, right| {
        left.rejection_count
            .cmp(&right.rejection_count)
            .then_with(|| right.measured.quality.total_cmp(&left.measured.quality))
            .then_with(|| {
                right
                    .measured
                    .measurements
                    .spatial_support
                    .total_cmp(&left.measured.measurements.spatial_support)
            })
            .then_with(|| {
                right
                    .measured
                    .measurements
                    .inliers
                    .cmp(&left.measured.measurements.inliers)
            })
            .then_with(|| {
                left.measured
                    .measurements
                    .median_error_px
                    .total_cmp(&right.measured.measurements.median_error_px)
            })
            .then_with(|| left.model_rank.cmp(&right.model_rank))
            .then_with(|| {
                virtual_tile_hypothesis_matrix_key(&left.canonical_homography).cmp(
                    &virtual_tile_hypothesis_matrix_key(&right.canonical_homography),
                )
            })
    });
    let mut deduplicated = Vec::<MeasuredVirtualTileHypothesis>::new();
    for hypothesis in measured {
        if deduplicated.iter().any(|existing| {
            virtual_tile_hypotheses_are_equivalent(
                &existing.canonical_homography,
                &hypothesis.canonical_homography,
                tiles[left].dimensions(),
            )
        }) {
            continue;
        }
        deduplicated.push(hypothesis);
    }
    let outcomes = deduplicated
        .iter()
        .enumerate()
        .map(|(index, hypothesis)| {
            let measurements = hypothesis.measured.measurements;
            stack_report::StationRelationHypothesisRecord {
                model: match hypothesis.model_rank {
                    0 => "similarity",
                    1 => "affine",
                    2 => "projective",
                    _ => "unknown",
                }
                .to_string(),
                selected: index == 0,
                inliers: measurements.inliers,
                median_error_px: measurements.median_error_px,
                scale_ratio: measurements.scale_ratio,
                spatial_support: measurements.spatial_support,
                low_frequency_mean_relative_difference: measurements
                    .low_frequency_mean_relative_difference,
                edge_strength_ratio: measurements.edge_strength_ratio,
                median_edge_orientation_difference_degrees: measurements
                    .median_edge_orientation_difference_degrees,
                preserves_convex_orientation: measurements.preserves_convex_orientation,
                rejection_reasons: station_relation_rejection_reasons(&measurements)
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                canonical_source: hypothesis.canonical_source,
                canonical_target: hypothesis.canonical_target,
                source_to_target_transform: matrix3_row_major(&hypothesis.canonical_homography),
                correspondence_count: hypothesis.correspondence_count,
                correspondence_sample: hypothesis.correspondence_sample.clone(),
                holdout_fit_count: hypothesis.holdout_fit_count,
                holdout_count: hypothesis.holdout_count,
                holdout_median_error_px: hypothesis.holdout_median_error_px,
                holdout_spatial_support: hypothesis.holdout_spatial_support,
                holdout_transform: hypothesis.holdout_transform.as_ref().map(matrix3_row_major),
            }
        })
        .collect::<Vec<_>>();
    deduplicated
        .first()
        .map(|hypothesis| VirtualTileHypothesisSelection {
            relation: hypothesis.relation.clone(),
            passes: hypothesis.rejection_count == 0,
            outcomes,
        })
}

pub(super) fn select_virtual_tile_relation_hypotheses(
    tiles: &[ImageInfo],
    relations: &VirtualTileRelationHypotheses,
    coverages: VirtualTileSolverCoverages<'_>,
) -> HashMap<(usize, usize), MatchInfo> {
    let mut selected = HashMap::new();
    let mut pairs = relations.iter().collect::<Vec<_>>();
    pairs.sort_by_key(|(pair, _)| **pair);
    for (&pair, hypotheses) in pairs {
        if let Some(selection) =
            select_virtual_tile_relation_hypothesis(tiles, pair.0, pair.1, hypotheses, coverages)
        {
            selected.insert(pair, selection.relation);
        } else if let Some(relation) = hypotheses.first() {
            // Preserve one singular measured hypothesis so the final solver can
            // report the pair once as a canonical inversion failure.
            selected.insert(pair, relation.clone());
        }
    }
    selected
}

/// Apply the complete canonical Station_Relation measurement and rejection
/// gate before a prior-seeded Virtual_Tile relation may merge repair clusters.
/// This is intentionally side-effect free: final report counts remain owned by
/// the solver, so provisional filtering cannot double-count a rejection.
pub(super) fn virtual_tile_provisional_authoritative_candidate(
    tiles: &[ImageInfo],
    left: usize,
    right: usize,
    hypotheses: &[MatchInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> bool {
    select_virtual_tile_relation_hypothesis(tiles, left, right, hypotheses, coverages)
        .is_some_and(|selection| selection.passes)
}

pub(super) fn virtual_tile_authoritative_pairs(
    relations: &VirtualTileRelationHypotheses,
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> Vec<(usize, usize)> {
    let mut pairs = relations
        .iter()
        .filter_map(|(&(left, right), hypotheses)| {
            virtual_tile_provisional_authoritative_candidate(
                tiles, left, right, hypotheses, coverages,
            )
            .then_some((left, right))
        })
        .collect::<Vec<_>>();
    pairs.sort_unstable();
    pairs
}

/// Topology pairs that remain eligible for an actual prior-free proposal after
/// authoritative relations form their prior-consistent clusters.
pub(super) fn virtual_tile_prior_free_candidate_pairs(
    station_count: usize,
    authoritative_pairs: &[(usize, usize)],
    station_topology: &topology::StationTopology,
) -> Vec<(usize, usize)> {
    let mut clusters = Dsu::new(station_count);
    for &(left, right) in authoritative_pairs {
        if left < station_count && right < station_count && left != right {
            clusters.union(left, right);
        }
    }
    station_topology
        .candidates
        .iter()
        .filter_map(|candidate| {
            let (left, right) = (candidate.left, candidate.right);
            (left < station_count
                && right < station_count
                && clusters.find(left) != clusters.find(right))
            .then_some((left, right))
        })
        .collect()
}

/// Plan rendering-prior repairs from prior-free proposals.
///
/// Stations joined by authoritative (prior-seeded) relations form clusters
/// whose priors agree. The largest cluster is the reference (ties: smallest
/// canonical row/column key). Each proposal joining a fixed cluster to an
/// unfixed one — strongest first, ties by canonical pair key — moves the whole
/// unfixed cluster so that the proposal's measured relation holds; a cluster is
/// repaired at most once and a station therefore re-rendered at most once.
/// Stations no proposal reaches keep their prior (and stay disconnected).
pub(super) fn plan_virtual_tile_prior_repairs_with_centres(
    station_count: usize,
    authoritative_pairs: &[(usize, usize)],
    proposals: &[VirtualTilePriorProposal],
    tile_to_world: &[Matrix3<f64>],
    station_world_centres: &[Point2<f64>],
    station_topology: &topology::StationTopology,
) -> Vec<VirtualTilePriorRepair> {
    if station_count < 2
        || tile_to_world.len() != station_count
        || station_world_centres.len() != station_count
    {
        return Vec::new();
    }
    let mut clusters = Dsu::new(station_count);
    for &(left, right) in authoritative_pairs {
        if left < station_count && right < station_count && left != right {
            clusters.union(left, right);
        }
    }
    let mut members = BTreeMap::<usize, Vec<usize>>::new();
    for station in 0..station_count {
        members
            .entry(clusters.find(station))
            .or_default()
            .push(station);
    }
    if members.len() < 2 {
        return Vec::new();
    }
    let cluster_key = |stations: &[usize]| {
        stations
            .iter()
            .map(|&station| station_tree_position_key(station_topology, station))
            .min()
    };
    let Some(reference) = members
        .iter()
        .max_by(|(_, left), (_, right)| {
            left.len()
                .cmp(&right.len())
                .then_with(|| cluster_key(right).cmp(&cluster_key(left)))
        })
        .map(|(&root, _)| root)
    else {
        return Vec::new();
    };
    let mut corrections = BTreeMap::from([(reference, Matrix3::identity())]);
    let mut remaining = proposals
        .iter()
        .copied()
        .filter(|proposal| {
            proposal.left < station_count
                && proposal.right < station_count
                && proposal.left != proposal.right
        })
        .collect::<Vec<_>>();
    let mut repairs = Vec::new();
    loop {
        let next = remaining
            .iter()
            .enumerate()
            .filter(|(_, proposal)| {
                corrections.contains_key(&clusters.find(proposal.left))
                    != corrections.contains_key(&clusters.find(proposal.right))
            })
            .max_by(|(_, left), (_, right)| {
                left.inliers.cmp(&right.inliers).then_with(|| {
                    station_tree_edge_key(station_topology, right.left, right.right).cmp(
                        &station_tree_edge_key(station_topology, left.left, left.right),
                    )
                })
            })
            .map(|(index, _)| index);
        let Some(index) = next else {
            break;
        };
        let proposal = remaining.remove(index);
        let left_root = clusters.find(proposal.left);
        let right_root = clusters.find(proposal.right);
        // tile pixel q of the unfixed station lands where the fixed station's
        // (already corrected) placement puts the measured corresponding pixel.
        let (fixed, unfixed, unfixed_root, fixed_to_unfixed_tile) =
            if corrections.contains_key(&left_root) {
                (
                    proposal.left,
                    proposal.right,
                    right_root,
                    proposal.left_to_right.try_inverse(),
                )
            } else {
                (
                    proposal.right,
                    proposal.left,
                    left_root,
                    Some(proposal.left_to_right),
                )
            };
        let Some(unfixed_to_fixed) = fixed_to_unfixed_tile else {
            continue;
        };
        let fixed_correction = corrections[&clusters.find(fixed)];
        let Some(unfixed_world_to_tile) = tile_to_world[unfixed].try_inverse() else {
            continue;
        };
        let Some(correction) = normalized_homography(
            &(fixed_correction * tile_to_world[fixed] * unfixed_to_fixed * unfixed_world_to_tile),
        ) else {
            continue;
        };
        if correction.try_inverse().is_none()
            || members[&unfixed_root].iter().any(|&station| {
                !virtual_tile_prior_repair_is_bounded_at_center(
                    &correction,
                    station_world_centres[station],
                )
            })
        {
            // Reject the proposal atomically before this cluster is marked as
            // repaired, so a weaker bounded proposal remains eligible.
            continue;
        }
        corrections.insert(unfixed_root, correction);
        for &station in &members[&unfixed_root] {
            repairs.push(VirtualTilePriorRepair {
                station,
                cluster_root: unfixed_root,
                fixed_station: fixed,
                relation: (proposal.left, proposal.right),
                relation_inliers: proposal.inliers,
                world_correction: correction,
            });
        }
    }
    repairs.sort_by_key(|repair| repair.station);
    repairs
}

#[cfg(test)]
pub(super) fn plan_virtual_tile_prior_repairs(
    station_count: usize,
    authoritative_pairs: &[(usize, usize)],
    proposals: &[VirtualTilePriorProposal],
    tile_to_world: &[Matrix3<f64>],
    station_topology: &topology::StationTopology,
) -> Vec<VirtualTilePriorRepair> {
    let station_world_centres = tile_to_world
        .iter()
        .filter_map(|pose| transformed_point(pose, Point2::new(500.0, 400.0)))
        .collect::<Vec<_>>();
    plan_virtual_tile_prior_repairs_with_centres(
        station_count,
        authoritative_pairs,
        proposals,
        tile_to_world,
        &station_world_centres,
        station_topology,
    )
}

/// Render one Capture_Station for station matching only and return its covered
/// luminance plane and Coverage_Mask; the fused colour pixels are dropped.
#[allow(clippy::too_many_arguments)]
pub(super) fn render_virtual_tile_match_plane<R: Runtime, F>(
    group: &[&ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    station_index: usize,
    expected: stitching::FocusVirtualTileGeometry,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<(GrayImage, stitching::CoverageMask), String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let rendered = stitching::focus_stack_stitcher_unfilled(
        group,
        homographies,
        projection,
        focus_warp,
        None,
        station_index,
        false,
        app_handle,
        progress_event,
        load_image,
    )?;
    let rendered_dimensions = rendered.image.dimensions();
    if rendered_dimensions != (expected.width, expected.height) {
        return Err(format!(
            "Virtual tile station {station_index} geometry/render mismatch: expected {}x{} at world origin ({:.3},{:.3}), rendered {}x{}",
            expected.width,
            expected.height,
            expected.tile_to_world[(0, 2)],
            expected.tile_to_world[(1, 2)],
            rendered_dimensions.0,
            rendered_dimensions.1,
        ));
    }
    let coverage = match rendered.masks.as_ref() {
        Some(masks) => masks.coverage.clone(),
        None => stitching::CoverageMask::from_bytes(
            expected.width,
            expected.height,
            vec![0; expected.width as usize * expected.height as usize],
        )?,
    };
    let gray = virtual_tile_covered_luminance(&rendered.image, &coverage);
    Ok((gray, coverage))
}

/// Install one station's probe planes. The tile keeps the structural
/// relation-quality plane as its analysis image (`scale_factor` = that plane's
/// factor, the source `ImageInfo` convention) and independent covered features.
/// Small preview probe grids must not downsample feature descriptors twice.
/// The feature factor is explicit; nothing reads keypoints in quality-plane
/// coordinates or promotes a feature fit to solver evidence.
pub(super) fn install_virtual_tile_match_planes(
    tile: &mut ImageInfo,
    pyramid: &mut VirtualTileMatchPyramid,
    station: usize,
    gray: &GrayImage,
    coverage: &stitching::CoverageMask,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
) -> Result<(), String> {
    let planes = pyramid
        .set_station(station, tile, gray, coverage)
        .ok_or_else(|| format!("Virtual tile station {station} has no usable matching plane"))?;
    let finest = planes
        .last()
        .expect("set_station returns at least one plane");
    let feature_factor = finest
        .factor
        .min((gray.width().max(gray.height()) / 1200).max(1));
    let feature_plane = VirtualTileMatchPlane::from_native(gray, coverage, feature_factor)
        .ok_or_else(|| format!("Virtual tile station {station} has no feature proposal plane"))?;
    tile.features = virtual_tile_plane_features(&feature_plane, brief_pairs);
    pyramid.feature_factors[station] = feature_factor;
    let (quality, factor) = virtual_tile_relation_quality_plane(&planes, gray, coverage);
    #[cfg(test)]
    if let Some(directory) = std::env::var_os("RAW_EDITOR_VIRTUAL_TILE_DIAGNOSTICS_DIR") {
        let directory = PathBuf::from(directory);
        std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        gray.save(directory.join(format!("station-{station:03}.gray.png")))
            .map_err(|error| error.to_string())?;
        quality
            .save(directory.join(format!("station-{station:03}.quality.png")))
            .map_err(|error| error.to_string())?;
        let (w, h) = coverage.dimensions();
        GrayImage::from_raw(w, h, coverage.covered().to_vec())
            .ok_or("invalid diagnostic coverage")?
            .save(directory.join(format!("station-{station:03}.coverage.png")))
            .map_err(|error| error.to_string())?;
    }
    let quality_coverage = planes
        .iter()
        .find(|plane| plane.factor == factor)
        .map(|plane| plane.coverage.clone())
        .or_else(|| {
            VirtualTileMatchPlane::from_native(gray, coverage, factor).map(|plane| plane.coverage)
        })
        .ok_or_else(|| format!("Virtual tile station {station} has no relation-quality plane"))?;
    pyramid.quality_coverages[station] = quality_coverage;
    tile.scale_factor = f64::from(factor);
    tile.alignment_image = quality;
    Ok(())
}

/// Move one station's rendering prior: every member source pose (and its
/// focus-band poses) is composed with `world_correction`. Returns false, and
/// changes nothing, when a corrected pose would not be a valid planar pose.
pub(super) fn apply_virtual_tile_prior_repair(
    group: &[&ImageInfo],
    world_correction: &Matrix3<f64>,
    homographies: &mut HashMap<usize, Matrix3<f64>>,
    focus_warp: Option<&mut FocusLayerWarp>,
) -> bool {
    let mut corrected = Vec::with_capacity(group.len());
    for image in group {
        let Some(pose) = homographies.get(&image.id) else {
            return false;
        };
        let Some(pose) = normalized_homography(&(world_correction * pose)) else {
            return false;
        };
        if pose.try_inverse().is_none()
            || !homography_preserves_focus_orientation(&pose, image.dimensions())
        {
            return false;
        }
        corrected.push((image.id, pose));
    }
    for (id, pose) in corrected {
        homographies.insert(id, pose);
    }
    if let Some(warp) = focus_warp {
        for band in &mut warp.bands {
            for image in group {
                if let Some(pose) = band.homographies.get_mut(&image.id)
                    && let Some(corrected) = normalized_homography(&(world_correction * *pose))
                {
                    *pose = corrected;
                }
            }
        }
    }
    true
}

pub(super) fn replace_virtual_tile_candidate_record(
    records: &mut Vec<stack_report::StationRelationCandidateRecord>,
    record: stack_report::StationRelationCandidateRecord,
) {
    let key = (record.left.min(record.right), record.left.max(record.right));
    match records.iter_mut().find(|existing| {
        (
            existing.left.min(existing.right),
            existing.left.max(existing.right),
        ) == key
    }) {
        Some(existing) => *existing = record,
        None => records.push(record),
    }
}

/// Fit station relations directly from focus-fused Virtual_Tiles. The topology
/// controls deterministic attempt order and budget. Initial poses select the
/// jointly covered grid and predict bounded target search centres only; every
/// returned point and fitted model is measured directly from fused tile pixels,
/// coarse to fine on area-averaged planes of the supplied native planes.
/// Production drives the same measurement through
/// [`virtual_tile_measure_prior_relations`] with its per-station render loop.
#[cfg(test)]
pub(super) fn virtual_tile_station_match_outcome(
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
) -> VirtualTileStationMatchOutcome {
    let shortest_long_side = tiles
        .iter()
        .map(|tile| {
            tile.alignment_image
                .width()
                .max(tile.alignment_image.height())
        })
        .min()
        .unwrap_or(0);
    let mut pyramid = VirtualTileMatchPyramid::new(
        virtual_tile_pyramid_factors(shortest_long_side),
        tiles.len(),
    );
    for (station, tile) in tiles.iter().enumerate() {
        if let Some(coverage) = coverages.get(station) {
            pyramid.set_station(station, tile, &tile.alignment_image, coverage);
        }
    }
    let (relations, diagnostics) = virtual_tile_measure_prior_relations(
        tiles,
        &pyramid,
        initial_homographies,
        station_topology,
        None,
        false,
    );
    VirtualTileStationMatchOutcome {
        relations,
        diagnostics,
    }
}

#[cfg(test)]
pub(super) fn virtual_tile_station_matches(
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
) -> HashMap<(usize, usize), MatchInfo> {
    let outcome = virtual_tile_station_match_outcome(
        tiles,
        coverages,
        initial_homographies,
        station_topology,
    );
    select_virtual_tile_relation_hypotheses(
        tiles,
        &outcome.relations,
        VirtualTileSolverCoverages {
            quality: coverages,
            overlap: coverages,
            overlap_factor: 1,
        },
    )
}
