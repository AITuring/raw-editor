//! Geometry, overlap quality, and transform-family selection helpers.
//!
//! This module keeps panorama registration evidence separate from the graph
//! order solver and the focus execution pipeline. It intentionally shares the
//! parent module's constants and image types through `use super::*`.

use super::*;

pub(super) fn panorama_transform_is_stable(
    transform: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> bool {
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    if width <= 1.0 || height <= 1.0 || transform.try_inverse().is_none() {
        return false;
    }
    let corners = [
        Point2::new(0.0, 0.0),
        Point2::new(width, 0.0),
        Point2::new(width, height),
        Point2::new(0.0, height),
    ]
    .into_iter()
    .map(|point| transformed_point(transform, point))
    .collect::<Option<Vec<_>>>();
    let Some(corners) = corners else {
        return false;
    };

    let edge_scales = [
        (corners[1] - corners[0]).norm() / width.max(1.0),
        (corners[2] - corners[1]).norm() / height.max(1.0),
        (corners[2] - corners[3]).norm() / width.max(1.0),
        (corners[3] - corners[0]).norm() / height.max(1.0),
    ];
    if edge_scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale < 0.18 || *scale > 5.5)
    {
        return false;
    }
    let min_scale = edge_scales.iter().copied().fold(f64::INFINITY, f64::min);
    let max_scale = edge_scales
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !min_scale.is_finite() || max_scale / min_scale > 6.0 {
        return false;
    }

    let signed_double_area = corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(4)
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f64>();
    let area_ratio = signed_double_area / (2.0 * width * height);
    area_ratio.is_finite() && area_ratio.abs() >= 0.025 && area_ratio.abs() <= 30.0
}

pub(super) fn panorama_model_fit(
    transform: Matrix3<f64>,
    points: &[(Point2<f64>, Point2<f64>)],
) -> Option<(Matrix3<f64>, Vec<usize>, f64)> {
    let inliers = symmetric_inlier_indices(&transform, points, PANORAMA_MODEL_INLIER_THRESHOLD);
    if inliers.len() < PANORAMA_MODEL_MIN_INLIERS {
        return None;
    }
    let inlier_points = inliers
        .iter()
        .map(|&index| points[index])
        .collect::<Vec<_>>();
    // One deterministic refit removes the small bias introduced when a model
    // was estimated from a mixed set of descriptor inliers.
    let refitted = if transform[(2, 0)].abs() < 1e-14 && transform[(2, 1)].abs() < 1e-14 {
        // Keep the model family (translation/similarity/affine) selected by the
        // caller; a generic projective refit here would reintroduce drift.
        transform
    } else {
        processing::compute_homography(&inlier_points).unwrap_or(transform)
    };
    let error = median_symmetric_error(&refitted, &inlier_points);
    error.is_finite().then_some((refitted, inliers, error))
}

pub(super) fn panorama_model_linear_characteristics(transform: &Matrix3<f64>) -> (f64, f64, f64) {
    let a = transform[(0, 0)];
    let b = transform[(0, 1)];
    let c = transform[(1, 0)];
    let d = transform[(1, 1)];
    let scale_x = (a * a + c * c).sqrt();
    let scale_y = (b * b + d * d).sqrt();
    let scale = ((scale_x * scale_y).max(0.0)).sqrt();
    let rotation = 0.5 * (c - b).atan2(a + d);
    let anisotropy = if scale_x.max(scale_y) > f64::EPSILON {
        (scale_x - scale_y).abs() / scale_x.max(scale_y)
    } else {
        f64::INFINITY
    };
    (scale, rotation, anisotropy)
}

pub(super) fn mixed_focal_scale_is_plausible(
    transform: &Matrix3<f64>,
    source_focal: f64,
    target_focal: f64,
) -> bool {
    if !source_focal.is_finite()
        || !target_focal.is_finite()
        || source_focal <= 0.0
        || target_focal <= 0.0
    {
        return true;
    }
    let expected = target_focal / source_focal;
    let (scale, _, anisotropy) = panorama_model_linear_characteristics(transform);
    if !expected.is_finite() || !scale.is_finite() || anisotropy > 0.35 {
        return false;
    }
    let ratio = scale / expected;
    ratio.is_finite()
        && (MIXED_FOCAL_SCALE_MIN_RATIO..=MIXED_FOCAL_SCALE_MAX_RATIO).contains(&ratio)
}

pub(super) fn estimate_mixed_focal_coarse_registration(
    source: &ImageInfo,
    target: &ImageInfo,
    source_focal: f64,
    target_focal: f64,
    allow_adjacent_defocus: bool,
    log_match: bool,
) -> Option<(Matrix3<f64>, Vec<(Point2<f64>, Point2<f64>)>, f64)> {
    if !source_focal.is_finite()
        || !target_focal.is_finite()
        || source_focal <= 0.0
        || target_focal <= 0.0
    {
        return None;
    }
    let focal_scale = target_focal / source_focal;
    if !focal_scale.is_finite() || focal_scale <= 0.0 {
        return None;
    }

    // Work at a deliberately small common resolution. The source is resized
    // by the metadata-predicted focal ratio before the translation search, so
    // this is a real coarse registration rather than a same-size image hash.
    // It recovers a close-up inside a wider frame even when BRIEF selects a
    // repeated character as its only cross-lens consensus.
    let target_alignment_width = target.alignment_image.width().max(1) as f64;
    let target_alignment_height = target.alignment_image.height().max(1) as f64;
    let target_low_width = 128u32.min(target.alignment_image.width().max(32));
    let target_low_height = ((target_alignment_height / target_alignment_width)
        * target_low_width as f64)
        .round()
        .clamp(32.0, 128.0) as u32;
    let target_low = image::imageops::resize(
        &target.alignment_image,
        target_low_width,
        target_low_height,
        image::imageops::FilterType::Triangle,
    );
    let target_low_scale = target_low_width as f64 / target_alignment_width;
    // EXIF focal ratio is a useful prior, but a handheld capture can also
    // change camera distance while zooming. Search a bounded scale band around
    // that prior; otherwise a valid close-up in a wide overview is rejected by
    // the fixed-ratio translation search before dense validation sees it.
    let focal_ratio = (source_focal / target_focal).max(target_focal / source_focal);
    let scale_candidates = if focal_ratio <= 1.03 {
        // Same lens module: do not let EXIF noise expand the search to 1.08/1.20
        // scales that mimic a false lens switch on dark, low-texture artwork.
        [0.94_f64, 0.97, 1.0, 1.03, 1.06]
            .into_iter()
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .collect::<Vec<_>>()
    } else {
        [0.70, 0.82, 0.94, 1.0, 1.08, 1.20, 1.34]
            .into_iter()
            .map(|adjustment| focal_scale * adjustment)
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .collect::<Vec<_>>()
    };
    let mut best: Option<(i32, i32, f64, f64)> = None;
    let mut candidate_scores: Vec<(i32, i32, f64, f64)> = Vec::new();

    for scale in scale_candidates {
        let source_alignment_scale =
            scale * source.scale_factor / target.scale_factor.max(f64::EPSILON);
        let source_low_width =
            (source.alignment_image.width() as f64 * source_alignment_scale * target_low_scale)
                .round()
                .clamp(24.0, 360.0) as u32;
        let source_low_height =
            (source.alignment_image.height() as f64 * source_alignment_scale * target_low_scale)
                .round()
                .clamp(24.0, 280.0) as u32;
        let source_low = image::imageops::resize(
            &source.alignment_image,
            source_low_width,
            source_low_height,
            image::imageops::FilterType::Triangle,
        );
        let minimum_overlap_width = (source_low.width().min(target_low.width()) as f64 * 0.30)
            .round()
            .max(12.0) as u32;
        let minimum_overlap_height = (source_low.height().min(target_low.height()) as f64 * 0.30)
            .round()
            .max(12.0) as u32;
        let x_min = -(source_low.width() as i32 - minimum_overlap_width as i32);
        let x_max = target_low.width() as i32 - minimum_overlap_width as i32;
        let y_min = -(source_low.height() as i32 - minimum_overlap_height as i32);
        let y_max = target_low.height() as i32 - minimum_overlap_height as i32;

        for offset_y in y_min..=y_max {
            for offset_x in x_min..=x_max {
                let source_start_x = offset_x.max(0) as u32;
                let source_start_y = offset_y.max(0) as u32;
                let target_start_x = (-offset_x).max(0) as u32;
                let target_start_y = (-offset_y).max(0) as u32;
                if source_start_x >= source_low.width()
                    || source_start_y >= source_low.height()
                    || target_start_x >= target_low.width()
                    || target_start_y >= target_low.height()
                {
                    continue;
                }
                let overlap_width =
                    (source_low.width() - source_start_x).min(target_low.width() - target_start_x);
                let overlap_height = (source_low.height() - source_start_y)
                    .min(target_low.height() - target_start_y);
                if overlap_width < minimum_overlap_width || overlap_height < minimum_overlap_height
                {
                    continue;
                }
                let stride = 2u32;
                let mut values = Vec::new();
                let mut target_values = Vec::new();
                for y in (0..overlap_height).step_by(stride as usize) {
                    for x in (0..overlap_width).step_by(stride as usize) {
                        values.push(f64::from(
                            source_low.get_pixel(source_start_x + x, source_start_y + y)[0],
                        ));
                        target_values.push(f64::from(
                            target_low.get_pixel(target_start_x + x, target_start_y + y)[0],
                        ));
                    }
                }
                if values.len() < 120 {
                    continue;
                }
                let mean_source = values.iter().sum::<f64>() / values.len() as f64;
                let mean_target = target_values.iter().sum::<f64>() / values.len() as f64;
                let mut covariance = 0.0;
                let mut source_variance = 0.0;
                let mut target_variance = 0.0;
                for (source_value, target_value) in values.iter().zip(target_values.iter()) {
                    let source_delta = source_value - mean_source;
                    let target_delta = target_value - mean_target;
                    covariance += source_delta * target_delta;
                    source_variance += source_delta * source_delta;
                    target_variance += target_delta * target_delta;
                }
                if source_variance <= f64::EPSILON || target_variance <= f64::EPSILON {
                    continue;
                }
                let score = covariance / (source_variance * target_variance).sqrt();
                if !score.is_finite() {
                    continue;
                }
                candidate_scores.push((offset_x, offset_y, scale, score));
                if best
                    .as_ref()
                    .is_none_or(|(_, _, _, best_score)| score > *best_score)
                {
                    best = Some((offset_x, offset_y, scale, score));
                }
            }
        }
    }

    let (best_offset_x, best_offset_y, best_scale, best_score) = best?;
    let alternate = candidate_scores
        .iter()
        .filter(|(candidate_x, candidate_y, candidate_scale, _)| {
            (*candidate_x - best_offset_x).abs() >= 8
                || (*candidate_y - best_offset_y).abs() >= 8
                || (*candidate_scale - best_scale).abs() >= best_scale * 0.08
        })
        .map(|(_, _, _, candidate_score)| *candidate_score)
        .fold(f64::NEG_INFINITY, f64::max);
    if log_match {
        println!(
            "  - Mixed-focal coarse search best offset=({}, {}), NCC {:.3}, margin {:.3}",
            best_offset_x,
            best_offset_y,
            best_score,
            best_score - alternate
        );
    }
    if best_score < 0.40 {
        return None;
    }
    // The best low-resolution luminance peak is not necessarily the best
    // geometric peak: a repeated stroke can win the cheap search while a
    // weaker peak is the actual shared patch. Re-score a small set of
    // spatially separated peaks against the denser source/target images.
    let mut ranked_candidates = candidate_scores;
    ranked_candidates.sort_by(|left, right| right.3.total_cmp(&left.3));
    let mut selected = None;
    let mut best_overlap = None;
    let mut tried: Vec<(i32, i32, f64)> = Vec::new();
    for (offset_x, offset_y, scale, score) in ranked_candidates {
        if score < (best_score - 0.16).max(0.40) || tried.len() >= 16 {
            break;
        }
        if tried
            .iter()
            .any(|(previous_x, previous_y, previous_scale)| {
                (offset_x - *previous_x).abs() < 8
                    && (offset_y - *previous_y).abs() < 8
                    && (scale - *previous_scale).abs() < scale * 0.08
            })
        {
            continue;
        }
        tried.push((offset_x, offset_y, scale));
        // `offset_*` above is expressed as the crop displacement applied to
        // the scaled source. A positive source crop aligns with target zero,
        // so the source-to-target transform has the opposite translation.
        // Using the same sign validates a mirrored search pose and was the
        // reason genuine overview/close-up bridges never survived the dense
        // overlap check.
        let tx_alignment = -offset_x as f64 / target_low_scale;
        let ty_alignment = -offset_y as f64 / target_low_scale;
        let tx_full = tx_alignment * target.scale_factor;
        let ty_full = ty_alignment * target.scale_factor;
        let homography = Matrix3::new(scale, 0.0, tx_full, 0.0, scale, ty_full, 0.0, 0.0, 1.0);
        let Some((intensity_ncc, edge_ncc, edge_orientation, samples)) =
            focus_overlap_quality(source, target, &homography)
        else {
            continue;
        };
        let validation_score = intensity_ncc * 0.45 + edge_ncc * 0.35 + edge_orientation * 0.20;
        if best_overlap
            .as_ref()
            .is_none_or(|(_, _, _, _, _, best_validation)| validation_score > *best_validation)
        {
            best_overlap = Some((
                offset_x,
                offset_y,
                intensity_ncc,
                edge_ncc,
                edge_orientation,
                validation_score,
            ));
        }
        let structurally_verified = if allow_adjacent_defocus {
            // A heavily defocused bracket endpoint preserves coarse luminance
            // and edge magnitude but not reliable gradient orientation. This
            // relaxed profile is reachable only for adjacent, same-focal
            // captures at the caller and still requires both image channels.
            (score >= 0.60
                && intensity_ncc >= 0.30
                && edge_ncc >= -0.10
                && edge_orientation >= -0.10)
                || (intensity_ncc >= 0.60 && edge_ncc >= 0.20 && edge_orientation >= 0.0)
                || (intensity_ncc >= 0.85 && edge_ncc >= 0.05 && edge_orientation >= 0.10)
        } else {
            intensity_ncc >= MIXED_FOCAL_COARSE_MIN_INTENSITY_NCC
                && edge_ncc >= MIXED_FOCAL_COARSE_MIN_EDGE_NCC
                && edge_orientation >= MIXED_FOCAL_COARSE_MIN_EDGE_ORIENTATION
        };
        if !structurally_verified {
            continue;
        }
        if selected
            .as_ref()
            .is_none_or(|(_, _, _, _, _, _, _, _, _, best_validation)| {
                validation_score > *best_validation
            })
        {
            selected = Some((
                offset_x,
                offset_y,
                scale,
                score,
                homography,
                intensity_ncc,
                edge_ncc,
                edge_orientation,
                samples,
                validation_score,
            ));
        }
    }
    let Some((_, _, _, score, homography, intensity_ncc, edge_ncc, edge_orientation, samples, _)) =
        selected
    else {
        if log_match {
            if let Some((offset_x, offset_y, intensity_ncc, edge_ncc, edge_orientation, _)) =
                best_overlap
            {
                println!(
                    "  - Rejecting mixed-focal coarse bridge: best structural candidate at ({offset_x}, {offset_y}) had intensity NCC {intensity_ncc:.3}, edge NCC {edge_ncc:.3}, edge orientation {edge_orientation:.3}"
                );
            } else {
                println!(
                    "  - Rejecting mixed-focal coarse bridge: overlap structure is not verified"
                );
            }
        }
        return None;
    };
    if log_match {
        println!(
            "  - Mixed-focal coarse overlap validation: intensity NCC {:.3}, edge NCC {:.3}, edge orientation {:.3} ({} samples)",
            intensity_ncc, edge_ncc, edge_orientation, samples
        );
    }
    let mut points = Vec::new();
    for row in 1..=6 {
        for column in 1..=8 {
            let source_point = Point2::new(
                source.width as f64 * column as f64 / 9.0,
                source.height as f64 * row as f64 / 7.0,
            );
            let Some(target_point) = transformed_point(&homography, source_point) else {
                continue;
            };
            if target_point.x >= 0.0
                && target_point.y >= 0.0
                && target_point.x < target.width as f64
                && target_point.y < target.height as f64
            {
                points.push((source_point, target_point));
            }
        }
    }
    // A narrow same-focal scan strip can cover only a handful of the coarse
    // grid cells.  It is still independently validated by the dense
    // luminance/edge checks above; do not discard it solely because the
    // synthetic grid has fewer than the cross-lens minimum correspondences.
    let minimum_points = if allow_adjacent_defocus && focal_ratio <= 1.03 {
        // A very narrow but well-correlated bracket boundary may expose no
        // coarse grid cell after the transform is scaled back to the source
        // dimensions. The dense overlap validator remains the evidence gate;
        // these points are optional and only carry the recovered placement.
        0
    } else {
        MIXED_FOCAL_MIN_INLIERS
    };
    (points.len() >= minimum_points).then_some((homography, points, score))
}

/// Validate a metadata-scaled translation against the actual alignment images.
/// The low-resolution search above is intentionally cheap, but repeated
/// calligraphy can make a wrong translation look unique. Sampling the proposed
/// overlap at a denser grid and comparing both luminance and edge structure
/// catches that failure before synthetic points can be promoted to a match.
#[derive(Clone, Copy, Debug)]
pub(super) struct FocusOverlapQuality {
    pub(super) intensity_ncc: f64,
    pub(super) edge_ncc: f64,
    pub(super) mean_edge_orientation_cosine: f64,
    pub(super) samples: usize,
    pub(super) low_frequency_mean_relative_difference: f64,
    pub(super) edge_strength_ratio: f64,
    pub(super) median_edge_orientation_difference_degrees: f64,
    pub(super) exposure_gain: f64,
    pub(super) exposure_normalized_difference: f64,
    pub(super) exposure_normalized_mean_relative_difference: f64,
}

impl FocusOverlapQuality {
    pub(super) fn legacy_tuple(self) -> (f64, f64, f64, usize) {
        (
            self.intensity_ncc,
            self.edge_ncc,
            self.mean_edge_orientation_cosine,
            self.samples,
        )
    }
}

pub(super) fn focus_overlap_quality_measurements(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
) -> Option<FocusOverlapQuality> {
    focus_overlap_quality_measurements_on_coverage(source, target, source_to_target, None)
}

/// [`focus_overlap_quality_measurements`] on the pixels both analysis planes
/// cover. `coverages` holds the `(source, target)` Coverage_Masks in
/// `alignment_image` pixels; a sample counts only when every pixel it reads,
/// directly or through the plane's blur, is covered in both (需求 6.1, 6.8:
/// uncovered payload is not overlap). `None` samples every in-bounds position.
pub(super) fn focus_overlap_quality_measurements_on_coverage(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
    coverages: Option<(&stitching::CoverageMask, &stitching::CoverageMask)>,
) -> Option<FocusOverlapQuality> {
    let inverse = source_to_target.try_inverse()?;
    let target_width = target.alignment_image.width();
    let target_height = target.alignment_image.height();
    if target_width < 32 || target_height < 32 {
        return None;
    }
    if coverages.is_some_and(|(source_coverage, target_coverage)| {
        source_coverage.dimensions() != source.alignment_image.dimensions()
            || target_coverage.dimensions() != target.alignment_image.dimensions()
    }) {
        return None;
    }
    let sample_is_covered = |coverage: &stitching::CoverageMask, x: f64, y: f64| {
        let radius = VIRTUAL_TILE_QUALITY_COVERAGE_RADIUS_PX;
        virtual_tile_coverage_contains_rectangle(
            coverage,
            x - radius,
            y - radius,
            x + radius,
            y + radius,
        )
    };
    let columns = 64u32;
    let rows = 40u32;
    let mut source_values = Vec::new();
    let mut target_values = Vec::new();
    let mut source_edges = Vec::new();
    let mut target_edges = Vec::new();
    let mut edge_orientation_cosines = Vec::new();
    let mut edge_orientation_differences = Vec::new();
    for row in 2..rows.saturating_sub(2) {
        let target_y = (row as f64 + 0.5) * target_height as f64 / rows as f64;
        for column in 2..columns.saturating_sub(2) {
            let target_x = (column as f64 + 0.5) * target_width as f64 / columns as f64;
            let target_full = Point2::new(
                target_x * target.scale_factor,
                target_y * target.scale_factor,
            );
            let Some(source_full) = transformed_point(&inverse, target_full) else {
                continue;
            };
            let source_x = source_full.x / source.scale_factor.max(f64::EPSILON);
            let source_y = source_full.y / source.scale_factor.max(f64::EPSILON);
            if let Some((source_coverage, target_coverage)) = coverages
                && !(sample_is_covered(target_coverage, target_x, target_y)
                    && sample_is_covered(source_coverage, source_x, source_y))
            {
                continue;
            }
            let Some(target_value) =
                registration::sample_gray(&target.alignment_image, target_x, target_y)
            else {
                continue;
            };
            let Some(source_value) =
                registration::sample_gray(&source.alignment_image, source_x, source_y)
            else {
                continue;
            };
            let Some(target_dx_left) =
                registration::sample_gray(&target.alignment_image, target_x - 1.0, target_y)
            else {
                continue;
            };
            let Some(target_dx_right) =
                registration::sample_gray(&target.alignment_image, target_x + 1.0, target_y)
            else {
                continue;
            };
            let Some(target_dy_up) =
                registration::sample_gray(&target.alignment_image, target_x, target_y - 1.0)
            else {
                continue;
            };
            let Some(target_dy_down) =
                registration::sample_gray(&target.alignment_image, target_x, target_y + 1.0)
            else {
                continue;
            };
            let Some(source_dx_left) =
                registration::sample_gray(&source.alignment_image, source_x - 1.0, source_y)
            else {
                continue;
            };
            let Some(source_dx_right) =
                registration::sample_gray(&source.alignment_image, source_x + 1.0, source_y)
            else {
                continue;
            };
            let Some(source_dy_up) =
                registration::sample_gray(&source.alignment_image, source_x, source_y - 1.0)
            else {
                continue;
            };
            let Some(source_dy_down) =
                registration::sample_gray(&source.alignment_image, source_x, source_y + 1.0)
            else {
                continue;
            };
            // Gradients are covectors: compare them in the target pixel grid,
            // not in the two independently rotated/scaled source grids. The
            // patch matcher already applies this inverse-transpose Jacobian;
            // the dense acceptance check must use the same convention.
            let Some(gradient_transform) =
                virtual_tile_local_jacobian(source_to_target, source_full)
                    .and_then(|jacobian| jacobian.try_inverse())
                    .map(|inverse| inverse.transpose())
            else {
                continue;
            };
            let source_gradient = gradient_transform
                * nalgebra::Vector2::new(
                    source_dx_right - source_dx_left,
                    source_dy_down - source_dy_up,
                )
                * (target.scale_factor / source.scale_factor.max(f64::EPSILON));
            let source_dx = source_gradient.x;
            let source_dy = source_gradient.y;
            let target_dx = target_dx_right - target_dx_left;
            let target_dy = target_dy_down - target_dy_up;
            let source_edge = source_dx.hypot(source_dy);
            let target_edge = target_dx.hypot(target_dy);
            if source_edge.is_finite() && target_edge.is_finite() {
                source_edges.push(source_edge);
                target_edges.push(target_edge);
                if source_edge > 4.0 && target_edge > 4.0 {
                    let cosine = ((source_dx * target_dx + source_dy * target_dy)
                        / (source_edge * target_edge))
                        .clamp(-1.0, 1.0);
                    edge_orientation_cosines.push(cosine);
                    // Edge orientation is axial rather than directional: a
                    // 180-degree gradient reversal describes the same edge.
                    edge_orientation_differences.push(cosine.abs().acos().to_degrees());
                }
            }
            source_values.push(source_value);
            target_values.push(target_value);
        }
    }
    if source_values.len() < 120 || source_edges.len() != source_values.len() {
        return None;
    }
    let intensity_ncc = normalized_correlation(&source_values, &target_values)?;
    let edge_ncc = normalized_correlation(&source_edges, &target_edges)?;
    let mean_edge_orientation_cosine = if edge_orientation_cosines.is_empty() {
        0.0
    } else {
        edge_orientation_cosines.iter().sum::<f64>() / edge_orientation_cosines.len() as f64
    };
    let source_mean = source_values.iter().sum::<f64>() / source_values.len() as f64;
    let target_mean = target_values.iter().sum::<f64>() / target_values.len() as f64;
    // A mean is the zero-frequency component of the sampled overlap. Dividing
    // by the larger magnitude makes the comparison symmetric and bounded.
    let low_frequency_mean_relative_difference = (source_mean - target_mean).abs()
        / source_mean.abs().max(target_mean.abs()).max(f64::EPSILON);
    // Estimate a scalar exposure gain on alternating overlap samples and
    // validate it on the held-out samples. Unlike matching the two means,
    // this residual cannot be made zero for arbitrary different images.
    let mut exposure_ratios = source_values
        .iter()
        .zip(&target_values)
        .enumerate()
        .filter_map(|(index, (&source, &target))| {
            (index % 2 == 0 && source > 8.0 && target > 8.0).then_some(target / source)
        })
        .collect::<Vec<_>>();
    let exposure_gain = median_value(&mut exposure_ratios).unwrap_or(f64::NAN);
    let mut exposure_residual = 0.0;
    let mut exposure_magnitude = 0.0;
    let mut predicted_sum = 0.0;
    let mut held_out_target_sum = 0.0;
    for (index, (&source, &target)) in source_values.iter().zip(&target_values).enumerate() {
        if index % 2 == 1 {
            let predicted = source * exposure_gain;
            exposure_residual += (predicted - target).abs();
            exposure_magnitude += predicted.abs().max(target.abs());
            predicted_sum += predicted;
            held_out_target_sum += target;
        }
    }
    let exposure_normalized_difference = exposure_residual / exposure_magnitude.max(f64::EPSILON);
    let exposure_normalized_mean_relative_difference = (predicted_sum - held_out_target_sum).abs()
        / predicted_sum
            .abs()
            .max(held_out_target_sum.abs())
            .max(f64::EPSILON);
    let source_edge_mean = source_edges.iter().sum::<f64>() / source_edges.len() as f64;
    let target_edge_mean = target_edges.iter().sum::<f64>() / target_edges.len() as f64;
    let edge_strength_ratio = if target_edge_mean > f64::EPSILON {
        source_edge_mean / target_edge_mean
    } else if source_edge_mean <= f64::EPSILON {
        1.0
    } else {
        f64::INFINITY
    };
    let median_edge_orientation_difference_degrees =
        median_value(&mut edge_orientation_differences).unwrap_or(f64::INFINITY);
    Some(FocusOverlapQuality {
        intensity_ncc,
        edge_ncc,
        mean_edge_orientation_cosine,
        samples: source_values.len(),
        low_frequency_mean_relative_difference,
        edge_strength_ratio,
        median_edge_orientation_difference_degrees,
        exposure_gain,
        exposure_normalized_difference,
        exposure_normalized_mean_relative_difference,
    })
}

/// Geometry is checked before the tone harmonizer runs. A confirmed structural
/// overlap must therefore survive a moderate camera-exposure change. Keep the
/// ordinary raw photometric gate unless correlation, held-out compensated
/// pixels and compensated edge strength independently verify that change.
pub(super) fn virtual_tile_photometric_measurements(quality: FocusOverlapQuality) -> (f64, f64) {
    let compensated_strength = quality.edge_strength_ratio * quality.exposure_gain;
    let exposure_structure_verified = (quality.intensity_ncc >= 0.95
        && quality.edge_ncc >= 0.65
        && quality.exposure_normalized_difference <= 0.20)
        || (quality.intensity_ncc >= 0.85
            && quality.edge_ncc >= 0.45
            && quality.exposure_normalized_difference <= 0.10);
    if quality.low_frequency_mean_relative_difference
        > STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE
        && exposure_structure_verified
        && (0.5..=2.0).contains(&quality.exposure_gain)
        && quality.exposure_normalized_mean_relative_difference
            <= STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE
        && (STATION_RELATION_EDGE_STRENGTH_RATIO_MIN..=STATION_RELATION_EDGE_STRENGTH_RATIO_MAX)
            .contains(&compensated_strength)
    {
        (
            quality.exposure_normalized_mean_relative_difference,
            compensated_strength,
        )
    } else {
        (
            quality.low_frequency_mean_relative_difference,
            quality.edge_strength_ratio,
        )
    }
}

/// Legacy overlap score used by non-Station_Relation paths. Station relation
/// acceptance reads the complete measurement structure above.
pub(super) fn focus_overlap_quality(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
) -> Option<(f64, f64, f64, usize)> {
    focus_overlap_quality_measurements(source, target, source_to_target)
        .map(FocusOverlapQuality::legacy_tuple)
}

pub(super) fn normalized_correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (&left_value, &right_value) in left.iter().zip(right) {
        let left_delta = left_value - left_mean;
        let right_delta = right_value - right_mean;
        covariance += left_delta * right_delta;
        left_variance += left_delta * left_delta;
        right_variance += right_delta * right_delta;
    }
    (left_variance > f64::EPSILON && right_variance > f64::EPSILON)
        .then(|| covariance / (left_variance * right_variance).sqrt())
        .filter(|value| value.is_finite())
}

pub(super) fn find_mixed_focal_similarity_ransac(
    points: &[(Point2<f64>, Point2<f64>)],
    source_focal: f64,
    target_focal: f64,
    inlier_threshold: f64,
    minimum_inliers: usize,
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    if points.len() < 2 || !inlier_threshold.is_finite() || inlier_threshold <= 0.0 {
        return None;
    }

    // See `find_homography_ransac_points_with_solver`: site constant mixed with
    // the content derived run seed, no time and no thread id (需求 14.6).
    let mut rng = StdRng::seed_from_u64(
        0xB4D4_5EED_5CA1_E001u64
            ^ (points.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ determinism::run_random_seed_mix(),
    );
    let all_indices: Vec<usize> = (0..points.len()).collect();
    let required_inliers = minimum_inliers.max(2);
    let mut best_transform = None;
    let mut best_inliers = Vec::new();
    let mut best_median = f64::INFINITY;

    for _ in 0..MIXED_FOCAL_SIMILARITY_RANSAC_ITERATIONS {
        let sample_indices: Vec<usize> = all_indices.sample(&mut rng, 2).copied().collect();
        if sample_indices.len() != 2 {
            continue;
        }
        let sample = sample_indices
            .iter()
            .map(|&index| points[index])
            .collect::<Vec<_>>();
        let source_span = (sample[1].0 - sample[0].0).norm();
        let target_span = (sample[1].1 - sample[0].1).norm();
        if source_span < 8.0 || target_span < 8.0 {
            continue;
        }
        let Some(transform) = estimate_similarity(&sample) else {
            continue;
        };
        if !mixed_focal_scale_is_plausible(&transform, source_focal, target_focal) {
            continue;
        }
        let inliers = symmetric_inlier_indices(&transform, points, inlier_threshold);
        if inliers.len() < required_inliers {
            continue;
        }
        let inlier_points = inliers
            .iter()
            .map(|&index| points[index])
            .collect::<Vec<_>>();
        let median = median_symmetric_error(&transform, &inlier_points);
        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len() && median < best_median)
        {
            best_transform = Some(transform);
            best_inliers = inliers;
            best_median = median;
        }
    }

    if best_inliers.len() < required_inliers {
        return None;
    }
    let mut transform = best_transform?;
    let mut inliers = best_inliers;
    for _ in 0..4 {
        let inlier_points = inliers
            .iter()
            .map(|&index| points[index])
            .collect::<Vec<_>>();
        let Some(refitted) = estimate_similarity(&inlier_points) else {
            break;
        };
        if !mixed_focal_scale_is_plausible(&refitted, source_focal, target_focal) {
            break;
        }
        let refitted_inliers = symmetric_inlier_indices(&refitted, points, inlier_threshold);
        if refitted_inliers.len() < required_inliers {
            break;
        }
        transform = refitted;
        if refitted_inliers == inliers {
            break;
        }
        inliers = refitted_inliers;
    }

    Some((transform, inliers))
}

pub(super) fn select_large_panorama_transform(
    projective: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    source_dimensions: (u32, u32),
    log_selection: bool,
) -> Matrix3<f64> {
    if points.len() < PANORAMA_MODEL_MIN_INLIERS {
        return *projective;
    }

    let translation = panorama_model_fit(estimate_translation(points), points);
    let similarity = estimate_similarity(points).and_then(|model| {
        panorama_transform_is_stable(&model, source_dimensions)
            .then(|| panorama_model_fit(model, points))
            .flatten()
    });
    let affine = estimate_affine(points).and_then(|model| {
        panorama_transform_is_stable(&model, source_dimensions)
            .then(|| panorama_model_fit(model, points))
            .flatten()
    });
    let projective_fit = panorama_transform_is_stable(projective, source_dimensions)
        .then(|| panorama_model_fit(*projective, points))
        .flatten();

    let mut selected = translation
        .clone()
        .or_else(|| similarity.clone())
        .or_else(|| affine.clone())
        .or_else(|| projective_fit.clone());
    let mut selected_name = if translation.is_some() {
        "translation"
    } else if similarity.is_some() {
        "similarity"
    } else if affine.is_some() {
        "affine"
    } else {
        "projective"
    };

    // A pure translation is still the safest model for a conventional long
    // panorama.  Promote it only when the data demonstrates a real scale or
    // rotation change, or when the lower-DOF model cannot explain the overlap.
    if let (Some(current), Some(candidate)) = (selected.as_ref(), similarity.as_ref()) {
        let (scale, rotation, _) = panorama_model_linear_characteristics(&candidate.0);
        let current_error = current.2;
        let meaningful_motion = (scale - 1.0).abs() > PANORAMA_MODEL_SCALE_EPSILON
            || rotation.abs() > PANORAMA_MODEL_ROTATION_EPSILON;
        let materially_better = candidate.2 + PANORAMA_MODEL_MIN_ERROR_GAIN < current_error
            || candidate.2 <= current_error * PANORAMA_MODEL_RELATIVE_ERROR_GAIN;
        let sufficient_support = candidate.1.len() + 2 >= current.1.len();
        if meaningful_motion && materially_better && sufficient_support {
            selected = Some(candidate.clone());
            selected_name = "similarity";
        }
    }

    if let (Some(current), Some(candidate)) = (selected.as_ref(), affine.as_ref()) {
        let (_, _, anisotropy) = panorama_model_linear_characteristics(&candidate.0);
        let meaningful_deformation = anisotropy > 0.018
            || candidate.0[(2, 0)].abs() > 1e-8
            || candidate.0[(2, 1)].abs() > 1e-8;
        let materially_better = candidate.2 + PANORAMA_MODEL_MIN_ERROR_GAIN < current.2
            && candidate.2 <= current.2 * 0.92;
        if meaningful_deformation && materially_better && candidate.1.len() + 2 >= current.1.len() {
            selected = Some(candidate.clone());
            selected_name = "affine";
        }
    }

    if let (Some(current), Some(candidate)) = (selected.as_ref(), projective_fit.as_ref()) {
        let perspective_strength = projective[(2, 0)].abs() + projective[(2, 1)].abs();
        let materially_better = candidate.2 + 0.2 < current.2 && candidate.2 <= current.2 * 0.82;
        if perspective_strength > 1e-8
            && materially_better
            && candidate.1.len() + 3 >= current.1.len()
        {
            selected = Some(candidate.clone());
            selected_name = "projective";
        }
    }

    let selected = selected.map(|fit| fit.0).unwrap_or_else(|| {
        // Never hand an unbounded projective fit to the graph merely because
        // every scored model missed the minimum support. A finite translation
        // is the conservative fallback and cannot fold the output canvas.
        if panorama_transform_is_stable(projective, source_dimensions) {
            *projective
        } else {
            estimate_translation(points)
        }
    });
    if log_selection {
        let describe = |fit: &Option<(Matrix3<f64>, Vec<usize>, f64)>| {
            fit.as_ref()
                .map(|(_, inliers, error)| format!("{error:.3}px/{}", inliers.len()))
                .unwrap_or_else(|| "n/a".to_string())
        };
        println!(
            "  - Large-panorama alignment selected {selected_name}: translation {}, similarity {}, affine {}, projective {}",
            describe(&translation),
            describe(&similarity),
            describe(&affine),
            describe(&projective_fit),
        );
    }
    selected
}
