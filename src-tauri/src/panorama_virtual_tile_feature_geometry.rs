//! Virtual-tile feature geometry helpers.

use super::*;

pub(crate) fn virtual_tile_point_is_covered(
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

pub(crate) fn virtual_tile_local_jacobian(
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

pub(crate) fn virtual_tile_gradient(
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
pub(crate) fn virtual_tile_repair_local_scale_ratio(
    transform: &Matrix3<f64>,
    center: Point2<f64>,
) -> Option<f64> {
    let ratio = virtual_tile_local_jacobian(transform, center)?
        .determinant()
        .abs()
        .sqrt();
    (ratio.is_finite() && ratio > 0.0).then_some(ratio)
}

pub(crate) fn axial_orientation_difference_degrees(
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

pub(crate) fn virtual_tile_gradient_magnitude_ncc(left: &[f64], right: &[f64]) -> Option<f64> {
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
pub(crate) fn virtual_tile_structural_score_points(
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
pub(crate) fn virtual_tile_structural_score(
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

pub(crate) fn virtual_tile_descriptor_distance(left: &Descriptor, right: &Descriptor) -> u32 {
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
pub(crate) fn virtual_tile_guided_search_radius(
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

pub(crate) fn virtual_tile_feature_buckets(
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
pub(crate) fn virtual_tile_best_guided_feature(
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

pub(crate) fn virtual_tile_predicted_overlap_bounds(
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

pub(crate) fn virtual_tile_overlap_grid_cell(
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
