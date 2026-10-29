//! Direct photometric refinement for Source_RAW layer pairs.
//!
//! Correlation and direct pyramids, the coverage-safe objective and the staged
//! optimisation behind the Source_RAW pair diagnostics.

use crate::panorama_utils::registration;
use crate::panorama_utils::stitching;
use image::GrayImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::BTreeMap;

use super::{
    ImageInfo, STATION_RELATION_SCALE_RATIO_MAX, STATION_RELATION_SCALE_RATIO_MIN,
    VIRTUAL_TILE_GLOBAL_TOP_K, VirtualTileGlobalProposal, VirtualTileGlobalSearchPlane,
    VirtualTileMatchPlane, homography_preserves_focus_orientation, homography_scale_ratio,
    normalized_homography, transformed_point, virtual_tile_global_similarity_proposals,
    virtual_tile_local_jacobian, virtual_tile_point_is_covered, virtual_tile_translation,
};

pub(super) const SOURCE_RAW_CORRELATION_TARGET_LONG_SIDE: u32 = 64;

/// A coverage-safe coarse pyramid built from exactly one Source_RAW. It uses
/// the generic luminance/gradient correlation plane implementation, but never
/// reads a fused VirtualTile, station prior, or another RAW layer.
#[derive(Clone)]
pub(super) struct SourceRawCorrelationPyramid {
    pub(super) coarse: VirtualTileGlobalSearchPlane,
}

impl SourceRawCorrelationPyramid {
    pub(super) fn from_native(
        gray: &GrayImage,
        coverage: &stitching::CoverageMask,
    ) -> Option<Self> {
        Some(Self {
            coarse: VirtualTileGlobalSearchPlane::from_native_target(
                gray,
                coverage,
                SOURCE_RAW_CORRELATION_TARGET_LONG_SIDE,
            )?,
        })
    }

    pub(super) fn from_image(image: &ImageInfo) -> Option<Self> {
        let (width, height) = image.alignment_image.dimensions();
        let coverage = stitching::CoverageMask::from_bytes(
            width,
            height,
            vec![u8::MAX; width as usize * height as usize],
        )
        .ok()?;
        Self::from_native(&image.alignment_image, &coverage)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SourceRawCorrelationProposal {
    pub(super) proposal: VirtualTileGlobalProposal,
    pub(super) second_peak_margin: f64,
}

pub(super) fn source_raw_correlation_proposals(
    left: &SourceRawCorrelationPyramid,
    right: &SourceRawCorrelationPyramid,
) -> Vec<SourceRawCorrelationProposal> {
    let proposals = virtual_tile_global_similarity_proposals(&left.coarse, &right.coarse);
    proposals
        .iter()
        .enumerate()
        .map(|(index, &proposal)| {
            let second_score = proposals.get(index + 1).map(|other| other.score);
            SourceRawCorrelationProposal {
                proposal,
                second_peak_margin: second_score.map_or(0.0, |score| proposal.score - score),
            }
        })
        .collect()
}

pub(super) const SOURCE_RAW_DIRECT_TARGET_LONG_SIDES: [u32; 4] = [64, 128, 256, 512];

pub(super) const SOURCE_RAW_DIRECT_GRID: (usize, usize) = (16, 12);

pub(super) const SOURCE_RAW_DIRECT_MAX_SEEDS: usize = VIRTUAL_TILE_GLOBAL_TOP_K;

pub(super) const SOURCE_RAW_DIRECT_PATCH_RADIUS: i32 = 4;

pub(super) const SOURCE_RAW_DIRECT_SEARCH_RADIUS: i32 = 3;

pub(super) const SOURCE_RAW_DIRECT_PATCH_GATES: registration::PatchMatchGates =
    registration::PatchMatchGates {
        min_correlation: 0.72,
        min_peak_margin: 0.01,
        min_source_variance: 2.0,
        min_target_variance: 1.0,
    };

#[derive(Clone)]
pub(super) struct SourceRawDirectLevel {
    pub(super) plane: VirtualTileMatchPlane,
    pub(super) gx: Vec<f32>,
    pub(super) gy: Vec<f32>,
}

impl SourceRawDirectLevel {
    pub(super) fn new(plane: VirtualTileMatchPlane) -> Self {
        let (width, height) = plane.gray.dimensions();
        let mut gx = vec![0.0; width as usize * height as usize];
        let mut gy = vec![0.0; gx.len()];
        if width >= 3 && height >= 3 {
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    if [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                        .into_iter()
                        .all(|(x, y)| plane.coverage.is_covered(x, y))
                    {
                        let index = (y * width + x) as usize;
                        gx[index] = 0.5
                            * (f32::from(plane.gray.get_pixel(x + 1, y)[0])
                                - f32::from(plane.gray.get_pixel(x - 1, y)[0]));
                        gy[index] = 0.5
                            * (f32::from(plane.gray.get_pixel(x, y + 1)[0])
                                - f32::from(plane.gray.get_pixel(x, y - 1)[0]));
                    }
                }
            }
        }
        Self { plane, gx, gy }
    }

    pub(super) fn gradient(&self, point: Point2<f64>) -> Option<nalgebra::Vector2<f64>> {
        let (width, height) = self.plane.gray.dimensions();
        let x = point.x.round() as i32;
        let y = point.y.round() as i32;
        if x <= 0 || y <= 0 || x >= width as i32 - 1 || y >= height as i32 - 1 {
            return None;
        }
        let index = y as usize * width as usize + x as usize;
        let gradient = nalgebra::Vector2::new(f64::from(self.gx[index]), f64::from(self.gy[index]));
        (gradient.norm() > 1e-6).then_some(gradient)
    }
}

#[derive(Clone)]
pub(super) struct SourceRawDirectPyramid {
    pub(super) levels: Vec<SourceRawDirectLevel>,
}

impl SourceRawDirectPyramid {
    pub(super) fn from_native(
        gray: &GrayImage,
        coverage: &stitching::CoverageMask,
    ) -> Option<Self> {
        let long_side = gray.width().max(gray.height());
        if coverage.dimensions() != gray.dimensions() {
            return None;
        }
        // Sanitize before even the factor-1 level is cloned. This makes every
        // stored luma/gradient bit independent of uncovered payload, not only
        // every objective sample.
        let safe_gray = GrayImage::from_fn(gray.width(), gray.height(), |x, y| {
            if coverage.is_covered(x, y) {
                *gray.get_pixel(x, y)
            } else {
                image::Luma([0])
            }
        });
        let mut factors = SOURCE_RAW_DIRECT_TARGET_LONG_SIDES
            .into_iter()
            .map(|target| long_side.div_ceil(target).max(1))
            .collect::<Vec<_>>();
        factors.sort_unstable_by(|left, right| right.cmp(left));
        factors.dedup();
        let levels = factors
            .into_iter()
            .filter_map(|factor| VirtualTileMatchPlane::from_native(&safe_gray, coverage, factor))
            .map(SourceRawDirectLevel::new)
            .collect::<Vec<_>>();
        (!levels.is_empty()).then_some(Self { levels })
    }

    pub(super) fn from_image(image: &ImageInfo) -> Option<Self> {
        let (width, height) = image.alignment_image.dimensions();
        let coverage = stitching::CoverageMask::from_bytes(
            width,
            height,
            vec![u8::MAX; width as usize * height as usize],
        )
        .ok()?;
        Self::from_native(&image.alignment_image, &coverage)
    }

    pub(super) fn working_set_bytes(&self) -> usize {
        self.levels
            .iter()
            .map(|level| {
                let pixels = level.plane.gray.width() as usize * level.plane.gray.height() as usize;
                pixels * (1 + 1 + 4 + 4)
            })
            .sum()
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct SourceRawDirectLoss {
    pub(super) loss: f64,
    pub(super) gain: f64,
    pub(super) bias: f64,
    pub(super) samples: usize,
    pub(super) cells: usize,
}

pub(super) fn source_raw_direct_level_transform(
    native: &Matrix3<f64>,
    left_factor: u32,
    right_factor: u32,
) -> Matrix3<f64> {
    VirtualTileMatchPlane::native_to_level(right_factor)
        * native
        * VirtualTileMatchPlane::level_to_native(left_factor)
}

pub(super) fn source_raw_direct_native_transform(
    level: &Matrix3<f64>,
    left_factor: u32,
    right_factor: u32,
) -> Matrix3<f64> {
    VirtualTileMatchPlane::level_to_native(right_factor)
        * level
        * VirtualTileMatchPlane::native_to_level(left_factor)
}

pub(super) fn source_raw_direct_raw_transform(
    analysis: &Matrix3<f64>,
    left_scale: f64,
    right_scale: f64,
) -> Matrix3<f64> {
    let left_to_raw = Matrix3::new(left_scale, 0.0, 0.0, 0.0, left_scale, 0.0, 0.0, 0.0, 1.0);
    let right_to_raw = Matrix3::new(right_scale, 0.0, 0.0, 0.0, right_scale, 0.0, 0.0, 0.0, 1.0);
    right_to_raw * analysis * left_to_raw.try_inverse().unwrap_or_else(Matrix3::identity)
}

pub(super) fn source_raw_direct_huber(value: f64, delta: f64) -> f64 {
    let absolute = value.abs();
    if absolute <= delta {
        0.5 * value * value
    } else {
        delta * (absolute - 0.5 * delta)
    }
}

/// Coverage-safe F-Direct objective. Gain and bias are solved independently in
/// each spatial cell before robust intensity and normalized-gradient-vector
/// residuals are accumulated. No descriptor or correspondence enters here.
pub(super) fn source_raw_direct_objective(
    left: &SourceRawDirectLevel,
    right: &SourceRawDirectLevel,
    transform: &Matrix3<f64>,
    fold: usize,
    holdout: bool,
) -> Option<SourceRawDirectLoss> {
    let (width, height) = left.plane.gray.dimensions();
    let mut cells = BTreeMap::<(usize, usize), Vec<(f64, f64, f64)>>::new();
    let (columns, rows) = (32usize, 24usize);
    for row in 1..rows - 1 {
        let y = (row as f64 + 0.5) * f64::from(height) / rows as f64;
        for column in 1..columns - 1 {
            let cell = (
                column * SOURCE_RAW_DIRECT_GRID.0 / columns,
                row * SOURCE_RAW_DIRECT_GRID.1 / rows,
            );
            let is_holdout = (cell.0 + cell.1) % 2 != fold % 2;
            if is_holdout != holdout {
                continue;
            }
            let x = (column as f64 + 0.5) * f64::from(width) / columns as f64;
            let source = Point2::new(x, y);
            let Some(target) = transformed_point(transform, source) else {
                continue;
            };
            if !virtual_tile_point_is_covered(&left.plane.coverage, source)
                || !virtual_tile_point_is_covered(&right.plane.coverage, target)
            {
                continue;
            }
            let Some(source_value) =
                registration::sample_gray(&left.plane.gray, source.x, source.y)
            else {
                continue;
            };
            let Some(target_value) =
                registration::sample_gray(&right.plane.gray, target.x, target.y)
            else {
                continue;
            };
            let Some(source_gradient) = left.gradient(source) else {
                continue;
            };
            let Some(target_gradient) = right.gradient(target) else {
                continue;
            };
            let Some(jacobian) = virtual_tile_local_jacobian(transform, source) else {
                continue;
            };
            let Some(inverse) = jacobian.try_inverse() else {
                continue;
            };
            let warped_gradient = inverse.transpose() * source_gradient;
            let denominator = warped_gradient.norm() * target_gradient.norm();
            if denominator <= 1e-8 {
                continue;
            }
            let ngf = 1.0
                - (warped_gradient.dot(&target_gradient) / denominator)
                    .abs()
                    .clamp(0.0, 1.0);
            cells
                .entry(cell)
                .or_default()
                .push((source_value, target_value, ngf));
        }
    }
    let mut total_loss = 0.0;
    let mut total_samples = 0usize;
    let mut gains = Vec::new();
    let mut biases = Vec::new();
    for values in cells.values().filter(|values| values.len() >= 4) {
        let mut weights = vec![1.0; values.len()];
        let mut gain = 1.0;
        let mut bias = 0.0;
        for _ in 0..4 {
            let sw = weights.iter().sum::<f64>();
            let sx = values
                .iter()
                .zip(&weights)
                .map(|((source, _, _), weight)| source * weight)
                .sum::<f64>();
            let sy = values
                .iter()
                .zip(&weights)
                .map(|((_, target, _), weight)| target * weight)
                .sum::<f64>();
            let sxx = values
                .iter()
                .zip(&weights)
                .map(|((source, _, _), weight)| source * source * weight)
                .sum::<f64>();
            let sxy = values
                .iter()
                .zip(&weights)
                .map(|((source, target, _), weight)| source * target * weight)
                .sum::<f64>();
            let denominator = sw * sxx - sx * sx;
            if denominator.abs() > 1e-8 {
                gain = (sw * sxy - sx * sy) / denominator;
                bias = (sy - gain * sx) / sw;
            }
            for (weight, (source, target, _)) in weights.iter_mut().zip(values) {
                let residual = (gain * source + bias - target) / 24.0;
                *weight = if residual.abs() <= 1.5 {
                    1.0
                } else {
                    1.5 / residual.abs()
                };
            }
        }
        if !gain.is_finite() || !bias.is_finite() {
            continue;
        }
        gains.push(gain);
        biases.push(bias);
        for &(source, target, ngf) in values {
            let intensity = (gain * source + bias - target) / 24.0;
            total_loss += 0.65 * source_raw_direct_huber(intensity, 1.5)
                + 0.35 * source_raw_direct_huber(ngf * 3.0, 1.0);
            total_samples += 1;
        }
    }
    if total_samples < 24 {
        return None;
    }
    Some(SourceRawDirectLoss {
        loss: total_loss / total_samples as f64,
        gain: gains.iter().sum::<f64>() / gains.len().max(1) as f64,
        bias: biases.iter().sum::<f64>() / biases.len().max(1) as f64,
        samples: total_samples,
        cells: gains.len(),
    })
}

pub(super) fn source_raw_direct_symmetric_objective(
    left: &SourceRawDirectLevel,
    right: &SourceRawDirectLevel,
    transform: &Matrix3<f64>,
    fold: usize,
    holdout: bool,
) -> Option<SourceRawDirectLoss> {
    let forward = source_raw_direct_objective(left, right, transform, fold, holdout)?;
    let reverse_transform = transform.try_inverse()?;
    let reverse = source_raw_direct_objective(right, left, &reverse_transform, fold, holdout)?;
    Some(SourceRawDirectLoss {
        loss: 0.5 * (forward.loss + reverse.loss),
        samples: forward.samples.min(reverse.samples),
        cells: forward.cells.min(reverse.cells),
        gain: forward.gain,
        bias: forward.bias,
    })
}

#[derive(Clone)]
pub(super) struct SourceRawDirectFold {
    pub(super) transform: Matrix3<f64>,
    pub(super) model: &'static str,
    pub(super) iterations: usize,
    pub(super) fit: SourceRawDirectLoss,
    pub(super) holdout: SourceRawDirectLoss,
    pub(super) factors: Vec<u32>,
    pub(super) seed_rank: usize,
}

pub(super) fn source_raw_direct_geometry_valid(
    transform: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> bool {
    transform.try_inverse().is_some()
        && homography_scale_ratio(transform).is_some_and(|scale| {
            (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX).contains(&scale)
        })
        && homography_preserves_focus_orientation(transform, dimensions)
}

pub(super) fn source_raw_direct_adjustments(
    model: &'static str,
    translation_step: f64,
    mapped_center: Point2<f64>,
) -> Vec<Matrix3<f64>> {
    let around = |linear: Matrix3<f64>| {
        virtual_tile_translation(mapped_center.x, mapped_center.y)
            * linear
            * virtual_tile_translation(-mapped_center.x, -mapped_center.y)
    };
    let mut adjustments = vec![
        virtual_tile_translation(translation_step, 0.0),
        virtual_tile_translation(-translation_step, 0.0),
        virtual_tile_translation(0.0, translation_step),
        virtual_tile_translation(0.0, -translation_step),
    ];
    let angle = (translation_step * 0.12).min(1.0).to_radians();
    let scale = (translation_step * 0.0015).min(0.01);
    for sign in [-1.0, 1.0] {
        let (sin, cos) = (sign * angle).sin_cos();
        adjustments.push(around(Matrix3::new(
            cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0,
        )));
        adjustments.push(around(Matrix3::new(
            1.0 + sign * scale,
            0.0,
            0.0,
            0.0,
            1.0 + sign * scale,
            0.0,
            0.0,
            0.0,
            1.0,
        )));
    }
    if matches!(model, "affine" | "projective") {
        let shear = (translation_step * 0.0008).min(0.004);
        for sign in [-1.0, 1.0] {
            adjustments.push(around(Matrix3::new(
                1.0 + sign * shear,
                sign * shear,
                0.0,
                0.0,
                1.0 - sign * shear,
                0.0,
                0.0,
                0.0,
                1.0,
            )));
        }
    }
    if model == "projective" {
        let projective = (translation_step * 0.000_002).min(0.000_01);
        for sign in [-1.0, 1.0] {
            adjustments.push(Matrix3::new(
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                sign * projective,
                0.0,
                1.0,
            ));
            adjustments.push(Matrix3::new(
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                sign * projective,
                1.0,
            ));
        }
    }
    adjustments
}

pub(super) fn source_raw_direct_optimize_stage(
    left: &SourceRawDirectLevel,
    right: &SourceRawDirectLevel,
    initial: Matrix3<f64>,
    fold: usize,
    model: &'static str,
) -> Option<(Matrix3<f64>, SourceRawDirectLoss, usize)> {
    let mut transform = initial;
    let mut best = source_raw_direct_symmetric_objective(left, right, &transform, fold, false)?;
    let center = Point2::new(
        f64::from(left.plane.gray.width()) * 0.5,
        f64::from(left.plane.gray.height()) * 0.5,
    );
    let mapped_center = transformed_point(&transform, center)?;
    let mut iterations = 0usize;
    for step in [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625] {
        for _ in 0..4 {
            iterations += 1;
            let mut improved = false;
            for adjustment in source_raw_direct_adjustments(model, step, mapped_center) {
                let Some(candidate) = normalized_homography(&(adjustment * transform)) else {
                    continue;
                };
                if !source_raw_direct_geometry_valid(&candidate, left.plane.gray.dimensions()) {
                    continue;
                }
                let Some(loss) =
                    source_raw_direct_symmetric_objective(left, right, &candidate, fold, false)
                else {
                    continue;
                };
                if loss.loss.total_cmp(&best.loss).is_lt() {
                    transform = candidate;
                    best = loss;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }
    Some((transform, best, iterations.min(20)))
}

pub(super) fn source_raw_direct_refine(
    left: &SourceRawDirectPyramid,
    right: &SourceRawDirectPyramid,
    seed: Matrix3<f64>,
    seed_rank: usize,
    fold: usize,
) -> Option<SourceRawDirectFold> {
    let mut native = seed;
    let mut model = "similarity";
    let mut iterations = 0usize;
    let mut final_fit = None;
    let mut final_holdout = None;
    let mut factors = Vec::new();
    for (left_level, right_level) in left.levels.iter().zip(&right.levels) {
        factors.push(left_level.plane.factor.max(right_level.plane.factor));
        let mut level_transform = source_raw_direct_level_transform(
            &native,
            left_level.plane.factor,
            right_level.plane.factor,
        );
        let (similarity, fit, count) = source_raw_direct_optimize_stage(
            left_level,
            right_level,
            level_transform,
            fold,
            "similarity",
        )?;
        level_transform = similarity;
        iterations += count;
        let mut holdout = source_raw_direct_symmetric_objective(
            left_level,
            right_level,
            &level_transform,
            fold,
            true,
        )?;
        model = "similarity";
        for next_model in ["affine", "projective"] {
            let Some((candidate, candidate_fit, count)) = source_raw_direct_optimize_stage(
                left_level,
                right_level,
                level_transform,
                fold,
                next_model,
            ) else {
                continue;
            };
            let Some(candidate_holdout) = source_raw_direct_symmetric_objective(
                left_level,
                right_level,
                &candidate,
                fold,
                true,
            ) else {
                continue;
            };
            if candidate_holdout.loss < holdout.loss * 0.90
                && source_raw_direct_geometry_valid(&candidate, left_level.plane.gray.dimensions())
            {
                level_transform = candidate;
                holdout = candidate_holdout;
                model = next_model;
                iterations += count;
                final_fit = Some(candidate_fit);
            }
        }
        native = source_raw_direct_native_transform(
            &level_transform,
            left_level.plane.factor,
            right_level.plane.factor,
        );
        final_fit = final_fit.or(Some(fit));
        final_holdout = Some(holdout);
    }
    Some(SourceRawDirectFold {
        transform: native,
        model,
        iterations,
        fit: final_fit?,
        holdout: final_holdout?,
        factors,
        seed_rank,
    })
}

pub(super) fn source_raw_direct_grid_centers(
    dimensions: (u32, u32),
    fold: usize,
) -> Vec<((usize, usize), Point2<f64>)> {
    let (columns, rows) = SOURCE_RAW_DIRECT_GRID;
    (0..rows)
        .flat_map(|row| {
            (0..columns).filter_map(move |column| {
                ((column + row) % 2 != fold % 2).then(|| {
                    (
                        (column, row),
                        Point2::new(
                            (column as f64 + 0.5) * f64::from(dimensions.0) / columns as f64,
                            (row as f64 + 0.5) * f64::from(dimensions.1) / rows as f64,
                        ),
                    )
                })
            })
        })
        .collect()
}
