use crate::panorama_stitching::{FOCUS_FOREGROUND_LUMA_THRESHOLD, FocusLayerWarp, ImageInfo};
use image::{GrayImage, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Point3};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use tauri::{AppHandle, Emitter, Runtime};

const PANORAMA_BLEND_BANDS: usize = 9;
const PANORAMA_DETAIL_SEAM_FEATHER_RADIUS: f32 = 4.0;
// Laplacian levels 0..4 contain structure up to roughly 32 px wide. Keep those
// levels tied to the optimal seam; broader bands may follow the overlap-wide
// illumination ramp without visibly doubling normal photographic detail.
const PANORAMA_GLOBAL_TONE_FIRST_BAND: usize = 5;
// Keep enough native detail in the focus decision for thin brush strokes and
// seal edges. The final canvas may be much wider than one source frame, so the
// analysis scale is still bounded by the per-source dimension in
// `focus_analysis_dimensions`.
const FOCUS_ANALYSIS_MAX_DIMENSION: u32 = 2048;
const FOCUS_ANALYSIS_MAX_PIXELS: u64 = 24_000_000;
const FOCUS_DECISIVE_ADVANTAGE: f32 = 0.20;
const FOCUS_CONFIDENCE_MARGIN: f32 = 0.04;
const FOCUS_EDGE_PROTECTION_AT_1024: f32 = 12.0;
const FOCUS_EDGE_CLAIM_THRESHOLD: f32 = 0.08;
// The analysis canvas is still sampled from the native frame (roughly 4-5
// source pixels per analysis pixel for the Lanting fixture). Keep the focus
// metric local at that scale: a large blur makes a sharp character claim its
// neighbour and is the main way a focus stack can become softer than either
// input frame.
const FOCUS_SCORE_BLUR_RADIUS_DIVISOR: f32 = 1_024.0;
const FOCUS_SCORE_MAX_BLUR_RADIUS: usize = 4;
const FOCUS_DECISION_COHERENCE_RADIUS_DIVISOR: f32 = 1_024.0;
const FOCUS_DECISION_MAX_COHERENCE_RADIUS: usize = 5;
const FOCUS_EDGE_PROTECTION_SCALE: f32 = 0.35;
// The broad tone transition is limited to the canvas-only low-frequency mask
// below. Keep enough room for a source-sized exposure step to meet smoothly,
// while all foreground pixels and all detail bands remain source-selected.
const FOCUS_SEAM_BLEND_RADIUS: usize = 512;
const FOCUS_FOREGROUND_HARD_EDGE_RADIUS_AT_2400: f32 = 8.0;
const FOCUS_SEAM_TONE_MIN_SAMPLES: usize = 128;
const FOCUS_SEAM_TONE_MAX_ADJUSTMENT: f32 = 0.075;
const FOCUS_COLOR_SAMPLE_BUDGET: u64 = 12_000;
const FOCUS_COLOR_MIN_SAMPLES: usize = 96;
const FOCUS_COLOR_MIN_CHANNEL_VALUE: f32 = 0.025;
const FOCUS_COLOR_MIN_GAIN: f32 = 0.78;
const FOCUS_COLOR_MAX_GAIN: f32 = 1.28;
const FOCUS_COLOR_MAX_OFFSET: f32 = 0.06;
// Bright plastic, glass, or paper edges can sit above the normal colour-sample
// ceiling. Admit those pixels only inside a matched foreground consensus band;
// ordinary bright highlights remain excluded from exposure estimation.
const FOCUS_COLOR_MAX_RELAXED_LUMA: f32 = 0.995;
// Focus stacks must not average a displaced subject at a seam. The seam path
// therefore blends only the low-frequency tone bands and restores protected
// foreground pixels after reconstruction; the visible detail bands stay on a
// single source. This lets a shifted scan remove exposure blocks without
// turning a displaced brush stroke into a double contour.
const FOCUS_ALLOW_LOW_FREQUENCY_SEAM_BLEND: bool = false;
// Correct broad local illumination differences without allowing individual
// brush strokes or the canvas weave to become a colour reference. The field is
// sampled in candidate-image space, then interpolated while the layer is
// copied into the final canvas.
const FOCUS_COLOR_CELL_SIZE: u32 = 768;
const FOCUS_COLOR_MIN_CELL_SAMPLES: usize = 12;
const FOCUS_COLOR_SPATIAL_VARIATION_THRESHOLD: f32 = 0.015;
const FOCUS_COLOR_SPATIAL_OFFSET_THRESHOLD: f32 = 0.004;
// Overlap samples do not cover the newly exposed side of every shifted frame.
// Let a measured local correction fade into nearby unsampled cells instead of
// stopping at a grid boundary, but keep the reach and strength bounded so an
// isolated overlap cannot recolour an entire new region.
const FOCUS_COLOR_PROPAGATION_RADIUS_CELLS: i32 = 3;
const FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE: f32 = 0.65;
// The final canvas can still contain a broad, source-sized exposure step when
// a newly exposed background region has no direct overlap samples. Work on a
// bounded analysis image and correct only a slowly varying background field;
// the canvas weave and all selected foreground detail remain untouched.
const FOCUS_BACKGROUND_TONE_LOCAL_RADIUS: usize = 4;
const FOCUS_BACKGROUND_TONE_SMOOTH_RADIUS: usize = 192;
const FOCUS_BACKGROUND_TONE_GLOBAL_WEIGHT: f32 = 0.65;
const FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT: f32 = 0.12;
const FOCUS_BACKGROUND_TONE_FOREGROUND_RADIUS_RATIO: f32 = 0.018;
// A depth layer can move by more than a pixel when its source frame is aligned
// against the paper plane. Keep an ownership buffer around an already selected
// foreground object so a later frame cannot re-introduce the same object as a
// parallel strip. The radius is derived from the source layer, not from any
// particular colour or holder shape.
const FOCUS_FOREGROUND_OWNERSHIP_RADIUS_RATIO: f32 = 0.025;
// The mask-only foreground refinement is a last-mile correction, not a second
// registration pass. A larger search can match a repeated rail/seal edge and
// move an otherwise valid foreground layer onto the wrong instance. Residual
// motion beyond this bound must be resolved by the image-backed homography and
// edge-consensus stages above.
const FOCUS_FOREGROUND_REFINEMENT_MAX_SHIFT: i32 = 32;
// Scalable stacks register on a reduced analysis image. Refine the committed
// full-resolution layer only when several independent overlap patches agree on
// the same small translation; ambiguous texture and isolated repeated strokes
// are deliberately left unchanged.
const FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS: i32 = 8;
const FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT: i32 = 16;
const FOCUS_FULL_RES_ALIGNMENT_GRID_SIZE: i32 = 4;
const FOCUS_FULL_RES_ALIGNMENT_MIN_NCC: f64 = 0.52;
const FOCUS_FULL_RES_ALIGNMENT_MIN_MARGIN: f64 = 0.025;
const FOCUS_FULL_RES_ALIGNMENT_MIN_ENERGY: f64 = 0.008;
const FOCUS_FULL_RES_ALIGNMENT_MIN_PATCHES: usize = 4;
const FOCUS_FULL_RES_ALIGNMENT_MAX_FOREGROUND_FRACTION: f64 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    Planar,
    Cylindrical,
    Spherical,
}

pub fn project_point(
    image: &ImageInfo,
    x: f64,
    y: f64,
    projection: Projection,
) -> Option<Point2<f64>> {
    let width = image.width() as f64;
    let height = image.height() as f64;
    let center_x = width * 0.5;
    let center_y = height * 0.5;
    let focal = width.max(height).max(1.0) * 0.85;
    let normalized_x = (x - center_x) / focal;
    let normalized_y = (y - center_y) / focal;

    let (projected_x, projected_y) = match projection {
        Projection::Planar => (x, y),
        Projection::Cylindrical => (
            normalized_x.atan() * focal + center_x,
            (normalized_y / (1.0 + normalized_x * normalized_x).sqrt()) * focal + center_y,
        ),
        Projection::Spherical => (
            normalized_x.atan() * focal + center_x,
            (normalized_y / (1.0 + normalized_x * normalized_x).sqrt()).atan() * focal + center_y,
        ),
    };

    if projected_x.is_finite() && projected_y.is_finite() {
        Some(Point2::new(projected_x, projected_y))
    } else {
        None
    }
}

fn unproject_point(
    image: &ImageInfo,
    x: f64,
    y: f64,
    projection: Projection,
) -> Option<Point2<f64>> {
    if projection == Projection::Planar {
        return if x.is_finite() && y.is_finite() {
            Some(Point2::new(x, y))
        } else {
            None
        };
    }

    let width = image.width() as f64;
    let height = image.height() as f64;
    let center_x = width * 0.5;
    let center_y = height * 0.5;
    let focal = width.max(height).max(1.0) * 0.85;
    let projected_x = (x - center_x) / focal;
    let projected_y = (y - center_y) / focal;

    let normalized_x = projected_x.tan();
    let vertical_scale = (1.0 + normalized_x * normalized_x).sqrt();
    let normalized_y = match projection {
        Projection::Cylindrical => projected_y * vertical_scale,
        Projection::Spherical => projected_y.tan() * vertical_scale,
        Projection::Planar => unreachable!("planar projection returned above"),
    };
    let source_x = normalized_x * focal + center_x;
    let source_y = normalized_y * focal + center_y;

    if source_x.is_finite() && source_y.is_finite() {
        Some(Point2::new(source_x, source_y))
    } else {
        None
    }
}

pub(super) fn map_target_to_source(
    inverse_homography: &Matrix3<f64>,
    target: Point3<f64>,
    image: &ImageInfo,
    projection: Projection,
) -> Option<Point2<f64>> {
    let projected_source = inverse_homography * target;
    if projected_source.z.abs() < 1e-8 {
        return None;
    }
    unproject_point(
        image,
        projected_source.x / projected_source.z,
        projected_source.y / projected_source.z,
        projection,
    )
}

pub(super) fn output_bounds(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for &image in images {
        let h = global_homographies[&image.id];
        let (width, height) = image.dimensions();
        let corners = [
            (0.0, 0.0),
            (width as f64, 0.0),
            (width as f64, height as f64),
            (0.0, height as f64),
        ];
        for (x, y) in corners {
            let Some(projected) = project_point(image, x, y, projection) else {
                continue;
            };
            let mapped = h * Point3::new(projected.x, projected.y, 1.0);
            if mapped.z.abs() < 1e-8 {
                continue;
            }
            let mapped_x = mapped.x / mapped.z;
            let mapped_y = mapped.y / mapped.z;
            min_x = min_x.min(mapped_x);
            max_x = max_x.max(mapped_x);
            min_y = min_y.min(mapped_y);
            max_y = max_y.max(mapped_y);
        }
    }

    (min_x, max_x, min_y, max_y)
}

pub(super) fn pixel_aligned_canvas(minimum: f64, maximum: f64) -> (f64, u32) {
    // The first image is the global reference and normally has an identity
    // transform. Keep its samples on integer output coordinates; using the exact
    // fractional bound as the offset would unnecessarily interpolate every
    // reference pixel.
    let offset = (-minimum).ceil();
    let size = (maximum + offset).ceil().max(1.0) as u32;
    (offset, size)
}

pub(crate) fn output_canvas_dimensions(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> (u32, u32) {
    if images.is_empty() {
        return (0, 0);
    }
    let (min_x, max_x, min_y, max_y) = output_bounds(images, global_homographies, projection);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return (0, 0);
    }
    let (_, width) = pixel_aligned_canvas(min_x, max_x);
    let (_, height) = pixel_aligned_canvas(min_y, max_y);
    (width, height)
}

pub(crate) fn output_canvas_dimensions_with_focus_warp(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
) -> (u32, u32) {
    if images.is_empty() {
        return (0, 0);
    }
    let (min_x, max_x, min_y, max_y) =
        focus_output_bounds(images, global_homographies, projection, focus_warp);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return (0, 0);
    }
    let (_, width) = pixel_aligned_canvas(min_x, max_x);
    let (_, height) = pixel_aligned_canvas(min_y, max_y);
    (width, height)
}

pub(super) fn transformed_image_region(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
) -> Option<(u32, u32, u32, u32)> {
    if out_width == 0 || out_height == 0 {
        return None;
    }
    let (width, height) = image.dimensions();
    let corners = [
        (0.0, 0.0),
        (width as f64, 0.0),
        (width as f64, height as f64),
        (0.0, height as f64),
    ];
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in corners {
        let projected = project_point(image, x, y, projection)?;
        let mapped = homography * Point3::new(projected.x, projected.y, 1.0);
        if mapped.z.abs() < 1e-8 {
            continue;
        }
        let mapped_x = mapped.x / mapped.z + offset_x;
        let mapped_y = mapped.y / mapped.z + offset_y;
        if !mapped_x.is_finite() || !mapped_y.is_finite() {
            continue;
        }
        min_x = min_x.min(mapped_x);
        max_x = max_x.max(mapped_x);
        min_y = min_y.min(mapped_y);
        max_y = max_y.max(mapped_y);
    }
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return None;
    }
    let left = min_x.floor().max(0.0).min((out_width - 1) as f64) as u32;
    let right = max_x.ceil().max(0.0).min((out_width - 1) as f64) as u32;
    let top = min_y.floor().max(0.0).min((out_height - 1) as f64) as u32;
    let bottom = max_y.ceil().max(0.0).min((out_height - 1) as f64) as u32;
    (left <= right && top <= bottom).then_some((left, right, top, bottom))
}

fn transformed_source_bounds(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    minimum_source_y: f64,
    maximum_source_y: f64,
) -> Option<(f64, f64, f64, f64)> {
    transformed_source_bounds_xy(
        image,
        homography,
        projection,
        0.0,
        1.0,
        minimum_source_y,
        maximum_source_y,
    )
}

fn transformed_source_bounds_xy(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    minimum_source_x: f64,
    maximum_source_x: f64,
    minimum_source_y: f64,
    maximum_source_y: f64,
) -> Option<(f64, f64, f64, f64)> {
    let (width, height) = image.dimensions();
    let x0 = (width as f64 * minimum_source_x).clamp(0.0, width as f64);
    let x1 = (width as f64 * maximum_source_x).clamp(0.0, width as f64);
    let y0 = (height as f64 * minimum_source_y).clamp(0.0, height as f64);
    let y1 = (height as f64 * maximum_source_y).clamp(0.0, height as f64);
    let corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for (x, y) in corners {
        let projected = project_point(image, x, y, projection)?;
        let mapped = homography * Point3::new(projected.x, projected.y, 1.0);
        if mapped.z.abs() < 1e-8 {
            continue;
        }
        let mapped_x = mapped.x / mapped.z;
        let mapped_y = mapped.y / mapped.z;
        if !mapped_x.is_finite() || !mapped_y.is_finite() {
            continue;
        }
        min_x = min_x.min(mapped_x);
        max_x = max_x.max(mapped_x);
        min_y = min_y.min(mapped_y);
        max_y = max_y.max(mapped_y);
    }
    if min_x.is_finite() && max_x.is_finite() && min_y.is_finite() && max_y.is_finite() {
        Some((min_x, max_x, min_y, max_y))
    } else {
        None
    }
}

fn focus_output_bounds(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
) -> (f64, f64, f64, f64) {
    let (mut min_x, mut max_x, mut min_y, mut max_y) =
        output_bounds(images, global_homographies, projection);
    let Some(focus_warp) = focus_warp else {
        return (min_x, max_x, min_y, max_y);
    };
    for image in images {
        for band in &focus_warp.bands {
            let Some(homography) = band.homographies.get(&image.id) else {
                continue;
            };
            let Some(&(minimum_source_y, maximum_source_y)) = band.source_ranges.get(&image.id)
            else {
                continue;
            };
            let (minimum_source_x, maximum_source_x) = band
                .source_x_ranges
                .get(&image.id)
                .copied()
                .unwrap_or((0.0, 1.0));
            let Some((band_min_x, band_max_x, band_min_y, band_max_y)) =
                transformed_source_bounds_xy(
                    image,
                    homography,
                    projection,
                    minimum_source_x,
                    maximum_source_x,
                    minimum_source_y,
                    maximum_source_y,
                )
            else {
                continue;
            };
            min_x = min_x.min(band_min_x);
            max_x = max_x.max(band_max_x);
            min_y = min_y.min(band_min_y);
            max_y = max_y.max(band_max_y);
        }
    }
    (min_x, max_x, min_y, max_y)
}

fn apply_exposure_gain(pixel: Rgb<f32>, gain: f32) -> Rgb<f32> {
    Rgb([pixel[0] * gain, pixel[1] * gain, pixel[2] * gain])
}

fn panorama_detail_alpha(candidate_signed_distance: f32) -> f32 {
    if candidate_signed_distance.is_infinite() {
        return if candidate_signed_distance.is_sign_positive() {
            1.0
        } else {
            0.0
        };
    }
    (0.5 + candidate_signed_distance / (PANORAMA_DETAIL_SEAM_FEATHER_RADIUS * 2.0)).clamp(0.0, 1.0)
}

struct ExposureOverlap<'a> {
    panorama: &'a Rgb32FImage,
    panorama_mask: &'a GrayImage,
    candidate: &'a ImageInfo,
    candidate_image: &'a Rgb32FImage,
    candidate_inverse: &'a Matrix3<f64>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
}

struct ExposureCompensation {
    cell_size: u32,
    grid_width: usize,
    grid_height: usize,
    gains: Vec<f32>,
    representative_gain: f32,
}

impl ExposureCompensation {
    fn gain_at(&self, x: u32, y: u32) -> f32 {
        if self.gains.is_empty() || self.grid_width == 0 || self.grid_height == 0 {
            return 1.0;
        }
        let grid_x = x as f64 / self.cell_size as f64;
        let grid_y = y as f64 / self.cell_size as f64;
        let x0 = (grid_x.floor() as usize).min(self.grid_width - 1);
        let y0 = (grid_y.floor() as usize).min(self.grid_height - 1);
        let x1 = (x0 + 1).min(self.grid_width - 1);
        let y1 = (y0 + 1).min(self.grid_height - 1);
        let tx = (grid_x - x0 as f64) as f32;
        let ty = (grid_y - y0 as f64) as f32;
        let value = |gx: usize, gy: usize| self.gains[gy * self.grid_width + gx];
        let top = value(x0, y0) * (1.0 - tx) + value(x1, y0) * tx;
        let bottom = value(x0, y1) * (1.0 - tx) + value(x1, y1) * tx;
        top * (1.0 - ty) + bottom * ty
    }
}

fn estimate_overlap_exposure_compensation(ctx: ExposureOverlap<'_>) -> ExposureCompensation {
    const CELL_SIZE: u32 = 256;
    let (out_width, out_height) = ctx.panorama.dimensions();
    let grid_width = out_width.div_ceil(CELL_SIZE) as usize + 1;
    let grid_height = out_height.div_ceil(CELL_SIZE) as usize + 1;
    let cell_count = grid_width * grid_height;
    let sample_step = out_width.max(out_height).div_ceil(720).max(8) as usize;
    let mut log_sums = vec![0.0f64; cell_count];
    let mut counts = vec![0u32; cell_count];
    let mut ratios = Vec::new();
    let candidate_homography = ctx.candidate_inverse.try_inverse();
    let candidate_region = candidate_homography.as_ref().and_then(|homography| {
        transformed_image_region(
            ctx.candidate,
            homography,
            ctx.projection,
            ctx.offset_x,
            ctx.offset_y,
            out_width,
            out_height,
        )
    });
    let (left, right, top, bottom) = candidate_region.unwrap_or((
        0,
        out_width.saturating_sub(1),
        0,
        out_height.saturating_sub(1),
    ));
    let sample_step_u32 = sample_step as u32;
    let sample_start = |value: u32| value.div_ceil(sample_step_u32) * sample_step_u32;
    for y in (sample_start(top)..=bottom).step_by(sample_step) {
        for x in (sample_start(left)..=right).step_by(sample_step) {
            if ctx.panorama_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let target = Point3::new(x as f64 - ctx.offset_x, y as f64 - ctx.offset_y, 1.0);
            let Some(source) =
                map_target_to_source(ctx.candidate_inverse, target, ctx.candidate, ctx.projection)
            else {
                continue;
            };
            if source.x < 0.0
                || source.y < 0.0
                || source.x >= ctx.candidate_image.width() as f64 - 1.0
                || source.y >= ctx.candidate_image.height() as f64 - 1.0
            {
                continue;
            }
            let base_luma = luminance(ctx.panorama.get_pixel(x, y));
            let candidate_luma = luminance(&get_interpolated_pixel(
                ctx.candidate_image,
                source.x,
                source.y,
            ));
            if (0.025..0.92).contains(&base_luma) && (0.025..0.92).contains(&candidate_luma) {
                let ratio = base_luma / candidate_luma;
                if (0.55..1.8).contains(&ratio) {
                    ratios.push(ratio);
                    let grid_x = (x / CELL_SIZE) as usize;
                    let grid_y = (y / CELL_SIZE) as usize;
                    let index = grid_y * grid_width + grid_x;
                    log_sums[index] += (ratio as f64).ln();
                    counts[index] += 1;
                }
            }
        }
    }
    let representative_gain = if ratios.len() < 32 {
        1.0
    } else {
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ratios[ratios.len() / 2].clamp(0.75, 1.35)
    };
    let mut gains = vec![representative_gain; cell_count];
    for index in 0..cell_count {
        if counts[index] >= 3 {
            gains[index] = (log_sums[index] / counts[index] as f64).exp().clamp(
                (representative_gain * 0.78) as f64,
                (representative_gain * 1.28) as f64,
            ) as f32;
        }
    }

    // The field models only broad illumination. Repeated neighbor averaging removes
    // local texture and registration noise while retaining vignetting and shadows.
    for _ in 0..5 {
        let mut smoothed = gains.clone();
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                let mut weighted_sum = 0.0f32;
                let mut weight_sum = 0.0f32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let neighbor_x = grid_x as i32 + dx;
                        let neighbor_y = grid_y as i32 + dy;
                        if neighbor_x < 0
                            || neighbor_y < 0
                            || neighbor_x >= grid_width as i32
                            || neighbor_y >= grid_height as i32
                        {
                            continue;
                        }
                        let neighbor_index = neighbor_y as usize * grid_width + neighbor_x as usize;
                        let weight = if dx == 0 && dy == 0 { 4.0 } else { 1.0 };
                        weighted_sum += gains[neighbor_index] * weight;
                        weight_sum += weight;
                    }
                }
                smoothed[grid_y * grid_width + grid_x] = weighted_sum / weight_sum;
            }
        }
        gains = smoothed;
    }

    ExposureCompensation {
        cell_size: CELL_SIZE,
        grid_width,
        grid_height,
        gains,
        representative_gain,
    }
}

struct SeamContext<'a> {
    pano: &'a Rgb32FImage,
    pano_mask: &'a GrayImage,
    img_to_add_info: &'a ImageInfo,
    img_to_add: &'a Rgb32FImage,
    h_add: &'a Matrix3<f64>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
    exposure: &'a ExposureCompensation,
}

#[derive(Clone, Copy)]
enum SeamOrientation {
    Vertical,
    Horizontal,
}

struct SeamInfo {
    orientation: SeamOrientation,
    coords: Vec<i32>,
    dx: f64,
    dy: f64,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
}

pub fn progressive_seam_stitcher<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    if images.is_empty() {
        return Ok(Rgb32FImage::new(0, 0));
    }

    let (min_x, max_x, min_y, max_y) = output_bounds(images, global_homographies, projection);

    let (offset_x, out_width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, out_height) = pixel_aligned_canvas(min_y, max_y);
    println!("  - Output canvas size: {}x{}", out_width, out_height);

    let mut panorama = Rgb32FImage::new(out_width, out_height);
    let mut panorama_mask = GrayImage::new(out_width, out_height);

    let base_img_info = images[0];
    let base_image = load_image(base_img_info)?;
    let h_base = &global_homographies[&base_img_info.id];
    let h_base_inv = h_base
        .try_inverse()
        .ok_or_else(|| "The base image alignment is not invertible.".to_string())?;
    println!("  - Placing base image: '{}'", base_img_info.filename);

    let num_pixels_per_row = out_width as usize * 3;
    let (base_left, base_right, base_top, base_bottom) = transformed_image_region(
        base_img_info,
        h_base,
        projection,
        offset_x,
        offset_y,
        out_width,
        out_height,
    )
    .unwrap_or((
        0,
        out_width.saturating_sub(1),
        0,
        out_height.saturating_sub(1),
    ));
    panorama
        .par_chunks_mut(num_pixels_per_row)
        .zip(panorama_mask.par_chunks_mut(out_width as usize))
        .enumerate()
        .skip(base_top as usize)
        .take((base_bottom - base_top + 1) as usize)
        .for_each(|(y, (row_slice, mask_row))| {
            for x in base_left..=base_right {
                let target_p = Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
                if let Some(source) =
                    map_target_to_source(&h_base_inv, target_p, base_img_info, projection)
                {
                    let sx = source.x;
                    let sy = source.y;
                    if sx < 0.0
                        || sx >= base_image.width() as f64
                        || sy < 0.0
                        || sy >= base_image.height() as f64
                    {
                        continue;
                    }
                    let color = get_high_quality_interpolated_pixel(&base_image, sx, sy);
                    let start = x as usize * 3;
                    row_slice[start..start + 3].copy_from_slice(&color.0);
                    mask_row[x as usize] = 255;
                }
            }
        });
    drop(base_image);

    for (i, &img_to_add_info) in images.iter().skip(1).enumerate() {
        let progress_msg = format!(
            "Stitching image {} of {}: {}",
            i + 2,
            images.len(),
            Path::new(&img_to_add_info.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        );
        let _ = app_handle.emit(progress_event, &progress_msg);
        println!("  - Progressively stitching '{}'", img_to_add_info.filename);

        let h_add = &global_homographies[&img_to_add_info.id];
        let h_add_inv = h_add.try_inverse().ok_or_else(|| {
            format!(
                "The alignment for '{}' is not invertible.",
                img_to_add_info.filename
            )
        })?;
        let img_to_add = load_image(img_to_add_info)?;
        let (candidate_left, candidate_right, candidate_top, candidate_bottom) =
            transformed_image_region(
                img_to_add_info,
                h_add,
                projection,
                offset_x,
                offset_y,
                out_width,
                out_height,
            )
            .unwrap_or((
                0,
                out_width.saturating_sub(1),
                0,
                out_height.saturating_sub(1),
            ));
        let exposure = estimate_overlap_exposure_compensation(ExposureOverlap {
            panorama: &panorama,
            panorama_mask: &panorama_mask,
            candidate: img_to_add_info,
            candidate_image: &img_to_add,
            candidate_inverse: &h_add_inv,
            projection,
            offset_x,
            offset_y,
        });
        println!(
            "    - Overlap exposure gain: {:.3} with local illumination correction",
            exposure.representative_gain
        );

        let ctx = SeamContext {
            pano: &panorama,
            pano_mask: &panorama_mask,
            img_to_add_info,
            img_to_add: &img_to_add,
            h_add,
            projection,
            offset_x,
            offset_y,
            out_width,
            out_height,
            exposure: &exposure,
        };
        let seam_info = find_adaptive_seam(&ctx);

        let use_seam = if let Some(ref info) = seam_info {
            !info.coords.is_empty()
        } else {
            false
        };

        if !use_seam {
            println!("    - Warning: Could not find seam. Using simple overwrite.");
        }

        let (orientation, seam_coords, new_image_is_dominant_side, seam_bounds) =
            if let Some(info) = seam_info {
                let dominant = match info.orientation {
                    SeamOrientation::Vertical => info.dx > 0.0,
                    SeamOrientation::Horizontal => info.dy > 0.0,
                };
                (
                    info.orientation,
                    info.coords,
                    dominant,
                    Some((info.min_x, info.max_x, info.min_y, info.max_y)),
                )
            } else {
                (SeamOrientation::Vertical, vec![], true, None)
            };

        if use_seam {
            let side = match orientation {
                SeamOrientation::Vertical => {
                    if new_image_is_dominant_side {
                        "right"
                    } else {
                        "left"
                    }
                }
                SeamOrientation::Horizontal => {
                    if new_image_is_dominant_side {
                        "bottom"
                    } else {
                        "top"
                    }
                }
            };
            println!("    - New image is on the {} side of the seam.", side);
        }

        panorama
            .par_chunks_mut(num_pixels_per_row)
            .zip(panorama_mask.par_chunks_mut(out_width as usize))
            .enumerate()
            .skip(candidate_top as usize)
            .take((candidate_bottom - candidate_top + 1) as usize)
            .for_each(|(y, (row_slice, mask_row))| {
                for x in candidate_left..=candidate_right {
                    let target_p = Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
                    let Some(source_add) =
                        map_target_to_source(&h_add_inv, target_p, img_to_add_info, projection)
                    else {
                        continue;
                    };
                    let sx = source_add.x;
                    let sy = source_add.y;
                    let is_on_add = sx >= 0.0
                        && sx < img_to_add.width() as f64
                        && sy >= 0.0
                        && sy < img_to_add.height() as f64;
                    let is_on_pano = mask_row[x as usize] > 0;

                    if !is_on_add && !is_on_pano {
                        continue;
                    }
                    if is_on_add && is_on_pano && use_seam {
                        // Preserve both inputs across the overlap. The multiband
                        // stage below needs the untouched base to blend broad
                        // illumination independently from the detail seam.
                        continue;
                    }
                    if is_on_add {
                        let color_to_add = apply_exposure_gain(
                            get_high_quality_interpolated_pixel(&img_to_add, sx, sy),
                            exposure.gain_at(x, y as u32),
                        );
                        let start = x as usize * 3;
                        row_slice[start..start + 3].copy_from_slice(&color_to_add.0);
                        mask_row[x as usize] = 255;
                    }
                }
            });

        if let (true, Some((min_x, max_x, min_y, max_y))) = (use_seam, seam_bounds) {
            blend_panorama_seam_band(SeamBandBlend {
                panorama: &mut panorama,
                panorama_mask: &mut panorama_mask,
                img_to_add_info,
                img_to_add: &img_to_add,
                h_add,
                projection,
                offset_x,
                offset_y,
                orientation,
                seam_coords: &seam_coords,
                new_image_is_dominant_side,
                min_x,
                max_x,
                min_y,
                max_y,
                exposure: &exposure,
            });
        }
    }

    let panorama_dimensions = panorama.dimensions();
    let cropped = crop_to_valid_rectangle(panorama, &panorama_mask);
    if cropped.dimensions() != panorama_dimensions {
        println!(
            "  - Cropped invalid projection margins: {}x{} -> {}x{}",
            panorama_dimensions.0,
            panorama_dimensions.1,
            cropped.width(),
            cropped.height()
        );
    }
    Ok(cropped)
}

struct SeamBandBlend<'a> {
    panorama: &'a mut Rgb32FImage,
    panorama_mask: &'a mut GrayImage,
    img_to_add_info: &'a ImageInfo,
    img_to_add: &'a Rgb32FImage,
    h_add: &'a Matrix3<f64>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
    orientation: SeamOrientation,
    seam_coords: &'a [i32],
    new_image_is_dominant_side: bool,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
    exposure: &'a ExposureCompensation,
}

fn blend_panorama_seam_band(ctx: SeamBandBlend<'_>) {
    let SeamBandBlend {
        panorama,
        panorama_mask,
        img_to_add_info,
        img_to_add,
        h_add,
        projection,
        offset_x,
        offset_y,
        orientation,
        seam_coords,
        new_image_is_dominant_side,
        min_x,
        max_x,
        min_y,
        max_y,
        exposure,
    } = ctx;
    let (out_width, out_height) = panorama.dimensions();
    if out_width == 0 || out_height == 0 || seam_coords.is_empty() {
        return;
    }

    let overlap_left = min_x.min(out_width - 1);
    let overlap_right = max_x.min(out_width - 1);
    let overlap_top = min_y.min(out_height - 1);
    let overlap_bottom = max_y.min(out_height - 1);
    if overlap_left > overlap_right || overlap_top > overlap_bottom {
        return;
    }

    let seam_value = |index: usize| -> Option<u32> {
        seam_coords
            .get(index)
            .copied()
            .map(|value| value.max(0) as u32)
    };

    match orientation {
        SeamOrientation::Horizontal if seam_coords.len() <= overlap_right as usize => return,
        SeamOrientation::Vertical if seam_coords.len() <= overlap_bottom as usize => return,
        _ => {}
    }
    // Keep the complete overlap so the low-frequency mask can transition across the
    // whole shared field rather than inheriting a tonal step around the detail seam.
    let (patch_left, patch_top, patch_right, patch_bottom) =
        (overlap_left, overlap_top, overlap_right, overlap_bottom);

    if patch_left > patch_right || patch_top > patch_bottom {
        return;
    }
    let patch_width = patch_right - patch_left + 1;
    let patch_height = patch_bottom - patch_top + 1;
    if patch_width < 8 || patch_height < 8 {
        return;
    }

    let Some(h_add_inv) = h_add.try_inverse() else {
        return;
    };
    let patch_pixel_count = patch_width as usize * patch_height as usize;
    let mut base_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut candidate_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut blend_mask = vec![0u8; patch_pixel_count];
    let mut low_frequency_mask = vec![0u8; patch_pixel_count];
    let image_width = img_to_add.width() as f64;
    let image_height = img_to_add.height() as f64;
    let patch_rgb_stride = patch_width as usize * 3;
    let patch_mask_stride = patch_width as usize;
    let panorama_ref: &Rgb32FImage = panorama;
    let panorama_mask_ref: &GrayImage = panorama_mask;

    base_pixels
        .par_chunks_mut(patch_rgb_stride)
        .zip(candidate_pixels.par_chunks_mut(patch_rgb_stride))
        .zip(blend_mask.par_chunks_mut(patch_mask_stride))
        .zip(low_frequency_mask.par_chunks_mut(patch_mask_stride))
        .enumerate()
        .for_each(
            |(local_y, (((base_row, candidate_row), blend_row), low_frequency_row))| {
                let local_y = local_y as u32;
                let global_y = patch_top + local_y;
                for local_x in 0..patch_width {
                    let global_x = patch_left + local_x;
                    let target =
                        Point3::new(global_x as f64 - offset_x, global_y as f64 - offset_y, 1.0);
                    let candidate_source =
                        map_target_to_source(&h_add_inv, target, img_to_add_info, projection);
                    let candidate_valid = candidate_source.as_ref().is_some_and(|source| {
                        source.x >= 0.0
                            && source.x < image_width
                            && source.y >= 0.0
                            && source.y < image_height
                    });
                    let candidate_pixel = if let Some(source) = candidate_source {
                        if candidate_valid {
                            apply_exposure_gain(
                                get_high_quality_interpolated_pixel(img_to_add, source.x, source.y),
                                exposure.gain_at(global_x, global_y),
                            )
                        } else {
                            Rgb([0.0, 0.0, 0.0])
                        }
                    } else {
                        Rgb([0.0, 0.0, 0.0])
                    };
                    let panorama_valid = panorama_mask_ref.get_pixel(global_x, global_y)[0] > 0;
                    let current_pixel = *panorama_ref.get_pixel(global_x, global_y);
                    let candidate_pixel = if !candidate_valid && panorama_valid {
                        current_pixel
                    } else {
                        candidate_pixel
                    };
                    let base_pixel = if panorama_valid || !candidate_valid {
                        current_pixel
                    } else {
                        candidate_pixel
                    };

                    let candidate_signed_distance = if !candidate_valid {
                        f32::NEG_INFINITY
                    } else if !panorama_valid {
                        f32::INFINITY
                    } else {
                        match orientation {
                            SeamOrientation::Horizontal => {
                                let seam_y = seam_value(global_x as usize).unwrap_or(global_y);
                                let distance = global_y as f32 - seam_y as f32;
                                if new_image_is_dominant_side {
                                    distance
                                } else {
                                    -distance
                                }
                            }
                            SeamOrientation::Vertical => {
                                let seam_x = seam_value(global_y as usize).unwrap_or(global_x);
                                let distance = global_x as f32 - seam_x as f32;
                                if new_image_is_dominant_side {
                                    distance
                                } else {
                                    -distance
                                }
                            }
                        }
                    };
                    let detail_alpha = panorama_detail_alpha(candidate_signed_distance);

                    let base_start = local_x as usize * 3;
                    base_row[base_start..base_start + 3].copy_from_slice(&base_pixel.0);
                    candidate_row[base_start..base_start + 3].copy_from_slice(&candidate_pixel.0);
                    blend_row[local_x as usize] = (detail_alpha * 255.0).round() as u8;
                    let low_frequency_alpha = if !candidate_valid {
                        0.0
                    } else if !panorama_valid {
                        1.0
                    } else {
                        match orientation {
                            SeamOrientation::Horizontal => {
                                let span = (overlap_bottom - overlap_top).max(1) as f32;
                                let position = (global_y.saturating_sub(overlap_top)) as f32 / span;
                                if new_image_is_dominant_side {
                                    position
                                } else {
                                    1.0 - position
                                }
                            }
                            SeamOrientation::Vertical => {
                                let span = (overlap_right - overlap_left).max(1) as f32;
                                let position =
                                    (global_x.saturating_sub(overlap_left)) as f32 / span;
                                if new_image_is_dominant_side {
                                    position
                                } else {
                                    1.0 - position
                                }
                            }
                        }
                    };
                    low_frequency_row[local_x as usize] =
                        (low_frequency_alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            },
        );

    let base = Rgb32FImage::from_raw(patch_width, patch_height, base_pixels)
        .expect("seam base patch dimensions must match");
    let candidate = Rgb32FImage::from_raw(patch_width, patch_height, candidate_pixels)
        .expect("seam candidate patch dimensions must match");
    let mask = GrayImage::from_raw(patch_width, patch_height, blend_mask.clone())
        .expect("seam mask dimensions must match");
    let low_frequency_mask = GrayImage::from_raw(patch_width, patch_height, low_frequency_mask)
        .expect("low-frequency seam mask dimensions must match");
    // Keep fine and middle-frequency Laplacian detail on one source side of the
    // path. Averaging those bands across the complete overlap softens strokes,
    // seals, foliage, and other structure whenever registration is not literally
    // pixel-identical. Only broad tonal bands may transition across the complete
    // overlap; visible detail transitions stay local to the optimal seam.
    let blended = multiband_blend(
        base,
        candidate,
        mask,
        Some(low_frequency_mask),
        PANORAMA_BLEND_BANDS,
        // A residual sub-pixel registration error must never be averaged into
        // the visible detail bands. Keep the source ownership hard for the
        // finest and middle frequencies; only the coarse tone pyramid is
        // allowed to feather across the seam.
        true,
    );

    let panorama_rgb_stride = out_width as usize * 3;
    let blended_pixels = blended.as_raw();
    panorama
        .as_mut()
        .par_chunks_mut(panorama_rgb_stride)
        .zip(panorama_mask.as_mut().par_chunks_mut(out_width as usize))
        .enumerate()
        .skip(patch_top as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, (panorama_row, panorama_mask_row))| {
            let local_y = global_y - patch_top as usize;
            let blended_row =
                &blended_pixels[local_y * patch_rgb_stride..(local_y + 1) * patch_rgb_stride];
            let destination_start = patch_left as usize * 3;
            let destination_end = destination_start + patch_rgb_stride;
            panorama_row[destination_start..destination_end].copy_from_slice(blended_row);

            let blend_row =
                &blend_mask[local_y * patch_mask_stride..(local_y + 1) * patch_mask_stride];
            for (local_x, candidate_owns_pixel) in blend_row.iter().copied().enumerate() {
                let global_x = patch_left as usize + local_x;
                if panorama_mask_row[global_x] > 0 || candidate_owns_pixel > 0 {
                    panorama_mask_row[global_x] = 255;
                }
            }
        });
}

fn luminance(pixel: &Rgb<f32>) -> f32 {
    pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114
}

fn focus_stack_pixel_is_canvas_like(pixel: &[f32]) -> bool {
    if pixel.len() < 3 || pixel[..3].iter().any(|value| !value.is_finite()) {
        return false;
    }
    let red = pixel[0].clamp(0.0, 1.0);
    let green = pixel[1].clamp(0.0, 1.0);
    let blue = pixel[2].clamp(0.0, 1.0);
    let luma = red * 0.299 + green * 0.587 + blue * 0.114;
    let red_floor = red.max(0.001);
    let green_ratio = green / red_floor;
    let blue_ratio = blue / red_floor;
    let red_dominance = red - 2.0 * green + blue;
    // The source canvas is a warm, moderately desaturated brown. This gate
    // deliberately rejects the red skirt, green sash, near-black robe, and
    // pale faces/hands before a low-frequency exposure field is applied. The
    // limits are broad enough to retain the darker and lighter canvas tiles.
    luma >= 0.12
        && luma <= 0.78
        && red >= green
        && green >= blue
        && green_ratio >= 0.42
        && blue_ratio >= 0.24
        && red - green >= 0.025
        && green - blue >= 0.015
        && red_dominance <= 0.08
}

fn focus_stack_pixel_is_tone_foreground(pixel: &[f32]) -> bool {
    if pixel.len() < 3 || pixel[..3].iter().any(|value| !value.is_finite()) {
        return false;
    }
    let red = pixel[0].clamp(0.0, 1.0);
    let green = pixel[1].clamp(0.0, 1.0);
    let blue = pixel[2].clamp(0.0, 1.0);
    let luma = red * 0.299 + green * 0.587 + blue * 0.114;
    let chroma = red.max(green).max(blue) - red.min(green).min(blue);
    let red_dominance = red - 2.0 * green + blue;
    let red_paint = red - green >= 0.14 && red - blue >= 0.17 && red - 2.0 * green + blue >= 0.08;
    let green_or_cool_paint = green - red >= 0.05 || blue - green >= 0.05;
    let dark_neutral_paint =
        luma <= 0.18 && (red - green).abs() <= 0.07 && (green - blue).abs() <= 0.07;
    // Faces and hands can have nearly the same average luminance as the
    // canvas, but their blue channel is much closer to green and their red
    // dominance is stronger. Keep this warm-pale paint out of the canvas-only
    // correction without admitting the flatter brown paper tone.
    let warm_light_paint = luma >= 0.22
        && red - green >= 0.05
        && red - blue >= 0.08
        && green - blue <= 0.08
        && red_dominance >= 0.04;
    // Very pale sleeves, hands, and faces are outside the brown canvas range.
    // This is intentionally narrower than a generic bright-pixel detector so
    // a lighter canvas exposure can still be harmonized.
    let pale_paint = luma >= 0.78 && chroma <= 0.22;
    red_paint || green_or_cool_paint || dark_neutral_paint || warm_light_paint || pale_paint
}

struct RenderedFocusLayer {
    image: Rgb32FImage,
    mask: GrayImage,
    foreground_mask: GrayImage,
    // Pixels in a repeated long-edge silhouette may be switched between
    // aligned sources. Other detected foreground remains first-owner-wins.
    relaxed_foreground_mask: GrayImage,
    left: u32,
    top: u32,
}

struct RenderedFocusAnalysisLayer {
    image: Rgb32FImage,
    mask: GrayImage,
    foreground_mask: GrayImage,
    relaxed_foreground_mask: GrayImage,
}

#[derive(Clone, Debug, PartialEq)]
struct FocusColorCorrection {
    gains: [f32; 3],
    offsets: [f32; 3],
    cell_size: u32,
    grid_width: usize,
    grid_height: usize,
    spatial_gains: Option<Vec<[f32; 3]>>,
    spatial_offsets: Option<Vec<[f32; 3]>>,
}

impl FocusColorCorrection {
    const IDENTITY: Self = Self {
        gains: [1.0; 3],
        offsets: [0.0; 3],
        cell_size: 1,
        grid_width: 0,
        grid_height: 0,
        spatial_gains: None,
        spatial_offsets: None,
    };

    fn interpolate_field(&self, field: &[[f32; 3]], x: u32, y: u32) -> [f32; 3] {
        if self.cell_size == 0 || self.grid_width == 0 || self.grid_height == 0 {
            return [0.0; 3];
        }

        let grid_x = x as f64 / self.cell_size as f64;
        let grid_y = y as f64 / self.cell_size as f64;
        let x0 = (grid_x.floor() as usize).min(self.grid_width - 1);
        let y0 = (grid_y.floor() as usize).min(self.grid_height - 1);
        let x1 = (x0 + 1).min(self.grid_width - 1);
        let y1 = (y0 + 1).min(self.grid_height - 1);
        let tx = if x0 == x1 {
            0.0
        } else {
            (grid_x - x0 as f64).clamp(0.0, 1.0) as f32
        };
        let ty = if y0 == y1 {
            0.0
        } else {
            (grid_y - y0 as f64).clamp(0.0, 1.0) as f32
        };
        let value = |gx: usize, gy: usize| field[gy * self.grid_width + gx];
        let top = value(x0, y0);
        let top_right = value(x1, y0);
        let bottom = value(x0, y1);
        let bottom_right = value(x1, y1);
        let mut interpolated = [0.0f32; 3];
        for channel in 0..3 {
            let top_value = top[channel] * (1.0 - tx) + top_right[channel] * tx;
            let bottom_value = bottom[channel] * (1.0 - tx) + bottom_right[channel] * tx;
            interpolated[channel] = top_value * (1.0 - ty) + bottom_value * ty;
        }
        interpolated
    }

    fn gain_at(&self, x: u32, y: u32) -> [f32; 3] {
        self.spatial_gains
            .as_ref()
            .map_or(self.gains, |gains| self.interpolate_field(gains, x, y))
    }

    fn offset_at(&self, x: u32, y: u32) -> [f32; 3] {
        self.spatial_offsets
            .as_ref()
            .map_or(self.offsets, |offsets| {
                self.interpolate_field(offsets, x, y)
            })
    }
}

fn median_f32(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f32::total_cmp);
    Some(values[values.len() / 2])
}

fn estimate_focus_color_correction(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    canvas_only: bool,
) -> FocusColorCorrection {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0
        || layer_height == 0
        || base.dimensions() != base_mask.dimensions()
        || merged_foreground_mask.dimensions() != base.dimensions()
        || candidate.left >= base.width()
        || candidate.top >= base.height()
        || candidate.left + layer_width > base.width()
        || candidate.top + layer_height > base.height()
    {
        return FocusColorCorrection::IDENTITY;
    }

    let sample_area = u64::from(layer_width) * u64::from(layer_height);
    let sample_step = ((sample_area as f64 / FOCUS_COLOR_SAMPLE_BUDGET as f64)
        .sqrt()
        .ceil() as u32)
        .max(1);
    let grid_width = layer_width.div_ceil(FOCUS_COLOR_CELL_SIZE) as usize;
    let grid_height = layer_height.div_ceil(FOCUS_COLOR_CELL_SIZE) as usize;
    let cell_count = grid_width * grid_height;
    let mut luminance_ratios = Vec::new();
    let mut channel_ratios = [Vec::new(), Vec::new(), Vec::new()];
    let mut channel_values = [Vec::new(), Vec::new(), Vec::new()];
    let mut cell_luminance_ratios: Vec<Vec<f32>> = (0..cell_count).map(|_| Vec::new()).collect();
    let mut cell_channel_ratios: Vec<[Vec<f32>; 3]> = (0..cell_count)
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();
    let mut cell_channel_values: Vec<[Vec<(f32, f32)>; 3]> = (0..cell_count)
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();
    for y in (0..layer_height).step_by(sample_step as usize) {
        for x in (0..layer_width).step_by(sample_step as usize) {
            if candidate.mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let global_x = candidate.left + x;
            let global_y = candidate.top + y;
            if base_mask.get_pixel(global_x, global_y)[0] == 0 {
                continue;
            }
            let base_is_foreground = merged_foreground_mask.get_pixel(global_x, global_y)[0] > 0;
            let candidate_is_foreground = candidate.foreground_mask.get_pixel(x, y)[0] > 0;
            let relaxed_foreground_overlap = candidate.relaxed_foreground_mask.get_pixel(x, y)[0]
                > 0
                && base_is_foreground
                && candidate_is_foreground;
            // Foreground is normally excluded because a repeated stroke or an
            // occluder is not a safe colour reference. A long-edge consensus is
            // different: it is an explicitly matched, repeated physical edge,
            // so its shared overlap is useful for correcting the rail/holder
            // colour that the paper-only samples cannot observe. Keep ordinary
            // foreground protected even when a focus stack contains it.
            if (base_is_foreground || candidate_is_foreground) && !relaxed_foreground_overlap {
                continue;
            }
            let base_pixel = base.get_pixel(global_x, global_y);
            let candidate_pixel = candidate.image.get_pixel(x, y);
            if canvas_only
                && (!focus_stack_pixel_is_canvas_like(base_pixel.0.as_slice())
                    || !focus_stack_pixel_is_canvas_like(candidate_pixel.0.as_slice()))
            {
                continue;
            }
            let base_luma = luminance(base_pixel);
            let candidate_luma = luminance(candidate_pixel);
            let maximum_luma = if relaxed_foreground_overlap {
                FOCUS_COLOR_MAX_RELAXED_LUMA
            } else {
                0.92
            };
            if !(0.04..=maximum_luma).contains(&base_luma)
                || !(0.04..=maximum_luma).contains(&candidate_luma)
            {
                continue;
            }
            let luma_ratio = base_luma / candidate_luma;
            if !(0.55..=1.8).contains(&luma_ratio) {
                continue;
            }
            luminance_ratios.push(luma_ratio);
            let cell_index = (y / FOCUS_COLOR_CELL_SIZE) as usize * grid_width
                + (x / FOCUS_COLOR_CELL_SIZE) as usize;
            cell_luminance_ratios[cell_index].push(luma_ratio);
            for channel in 0..3 {
                let base_value = base_pixel[channel];
                let candidate_value = candidate_pixel[channel];
                if base_value < FOCUS_COLOR_MIN_CHANNEL_VALUE
                    || candidate_value < FOCUS_COLOR_MIN_CHANNEL_VALUE
                {
                    continue;
                }
                let ratio = base_value / candidate_value;
                if (0.55..=1.8).contains(&ratio) {
                    channel_ratios[channel].push(ratio);
                    channel_values[channel].push((base_value, candidate_value));
                    cell_channel_ratios[cell_index][channel].push(ratio);
                    cell_channel_values[cell_index][channel].push((base_value, candidate_value));
                }
            }
        }
    }

    let luma_gain = if luminance_ratios.len() >= FOCUS_COLOR_MIN_SAMPLES {
        median_f32(&mut luminance_ratios)
            .unwrap_or(1.0)
            .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN)
    } else {
        1.0
    };
    let mut gains = [luma_gain; 3];
    for (channel, ratios) in channel_ratios.iter_mut().enumerate() {
        if ratios.len() >= FOCUS_COLOR_MIN_SAMPLES {
            let channel_gain = median_f32(ratios)
                .unwrap_or(luma_gain)
                .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
            // Most exposure changes are achromatic. Let the measured channel
            // ratio correct a real white-balance shift, but keep it anchored to
            // the luminance ratio so ink/seal colour cannot dominate a sparse
            // overlap.
            gains[channel] = (channel_gain * 0.72 + luma_gain * 0.28)
                .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
        }
    }

    // A ratio-only correction cannot remove a small black-level or flare
    // offset. Estimate it after the bounded gain, using the same robust
    // overlap samples, and keep it deliberately small so paper texture and
    // ink density are not flattened.
    let mut offsets = [0.0f32; 3];
    for channel in 0..3 {
        let mut differences = channel_values[channel]
            .iter()
            .map(|(base_value, candidate_value)| *base_value - *candidate_value * gains[channel])
            .collect::<Vec<_>>();
        if differences.len() >= FOCUS_COLOR_MIN_SAMPLES {
            offsets[channel] = median_f32(&mut differences)
                .unwrap_or(0.0)
                .clamp(-FOCUS_COLOR_MAX_OFFSET, FOCUS_COLOR_MAX_OFFSET);
        }
    }

    let mut spatial_gains = vec![gains; cell_count];
    let mut spatial_offsets = vec![offsets; cell_count];
    let mut known_cells = vec![false; cell_count];
    let mut known_cell_count = 0usize;
    for cell_index in 0..cell_count {
        let luma_ratios = &mut cell_luminance_ratios[cell_index];
        if luma_ratios.len() < FOCUS_COLOR_MIN_CELL_SAMPLES {
            continue;
        }
        let local_luma_gain = median_f32(luma_ratios)
            .unwrap_or(luma_gain)
            .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
        let mut local_gains = [local_luma_gain; 3];
        for (channel, ratios) in cell_channel_ratios[cell_index].iter_mut().enumerate() {
            if ratios.len() >= FOCUS_COLOR_MIN_CELL_SAMPLES {
                let channel_gain = median_f32(ratios)
                    .unwrap_or(local_luma_gain)
                    .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
                local_gains[channel] = (channel_gain * 0.72 + local_luma_gain * 0.28)
                    .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
            }
        }
        spatial_gains[cell_index] = local_gains;
        for channel in 0..3 {
            let values = &cell_channel_values[cell_index][channel];
            if values.len() >= FOCUS_COLOR_MIN_CELL_SAMPLES {
                let mut differences = values
                    .iter()
                    .map(|(base_value, candidate_value)| {
                        *base_value - *candidate_value * local_gains[channel]
                    })
                    .collect::<Vec<_>>();
                spatial_offsets[cell_index][channel] = median_f32(&mut differences)
                    .unwrap_or(offsets[channel])
                    .clamp(-FOCUS_COLOR_MAX_OFFSET, FOCUS_COLOR_MAX_OFFSET);
            }
        }
        known_cells[cell_index] = true;
        known_cell_count += 1;
    }

    // Smooth only measured cells first. Unknown cells start at the global
    // correction, so an overlap cannot impose its colour cast on a newly added
    // region without further evidence.
    for _ in 0..2 {
        let mut smoothed = spatial_gains.clone();
        let mut smoothed_offsets = spatial_offsets.clone();
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                let index = grid_y * grid_width + grid_x;
                if !known_cells[index] {
                    continue;
                }
                let mut weight_sum = 0.0f32;
                let mut weighted = [0.0f32; 3];
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let neighbor_x = grid_x as i32 + dx;
                        let neighbor_y = grid_y as i32 + dy;
                        if neighbor_x < 0
                            || neighbor_y < 0
                            || neighbor_x >= grid_width as i32
                            || neighbor_y >= grid_height as i32
                        {
                            continue;
                        }
                        let neighbor_index = neighbor_y as usize * grid_width + neighbor_x as usize;
                        if !known_cells[neighbor_index] {
                            continue;
                        }
                        let weight = if dx == 0 && dy == 0 { 4.0 } else { 1.0 };
                        weight_sum += weight;
                        for channel in 0..3 {
                            weighted[channel] += spatial_gains[neighbor_index][channel] * weight;
                        }
                    }
                }
                if weight_sum > 0.0 {
                    for channel in 0..3 {
                        smoothed[index][channel] = (weighted[channel] / weight_sum)
                            .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
                    }
                }

                let mut offset_weight_sum = 0.0f32;
                let mut weighted_offsets = [0.0f32; 3];
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let neighbor_x = grid_x as i32 + dx;
                        let neighbor_y = grid_y as i32 + dy;
                        if neighbor_x < 0
                            || neighbor_y < 0
                            || neighbor_x >= grid_width as i32
                            || neighbor_y >= grid_height as i32
                        {
                            continue;
                        }
                        let neighbor_index = neighbor_y as usize * grid_width + neighbor_x as usize;
                        if !known_cells[neighbor_index] {
                            continue;
                        }
                        let weight = if dx == 0 && dy == 0 { 4.0 } else { 1.0 };
                        offset_weight_sum += weight;
                        for channel in 0..3 {
                            weighted_offsets[channel] +=
                                spatial_offsets[neighbor_index][channel] * weight;
                        }
                    }
                }
                if offset_weight_sum > 0.0 {
                    for channel in 0..3 {
                        smoothed_offsets[index][channel] = (weighted_offsets[channel]
                            / offset_weight_sum)
                            .clamp(-FOCUS_COLOR_MAX_OFFSET, FOCUS_COLOR_MAX_OFFSET);
                    }
                }
            }
        }
        spatial_gains = smoothed;
        spatial_offsets = smoothed_offsets;
    }

    // The measured overlap is often a narrow strip at one side of a shifted
    // frame. Leaving every other cell at the global value creates a visible
    // low-frequency block even though the correction itself is bilinearly
    // interpolated. Propagate only into nearby unknown cells, with confidence
    // determined by distance to measured cells and capped well below 1.0.
    // This keeps the correction continuous at the overlap boundary while
    // preserving the global calibration in genuinely unobserved areas.
    if known_cell_count >= 2 {
        let radius = FOCUS_COLOR_PROPAGATION_RADIUS_CELLS.max(1);
        let mut propagated = spatial_gains.clone();
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                let index = grid_y * grid_width + grid_x;
                if known_cells[index] {
                    continue;
                }

                let mut weight_sum = 0.0f32;
                let mut weighted = [0.0f32; 3];
                let mut nearest_distance = f32::INFINITY;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let distance = ((dx * dx + dy * dy) as f32).sqrt();
                        if distance > radius as f32 {
                            continue;
                        }
                        let neighbor_x = grid_x as i32 + dx;
                        let neighbor_y = grid_y as i32 + dy;
                        if neighbor_x < 0
                            || neighbor_y < 0
                            || neighbor_x >= grid_width as i32
                            || neighbor_y >= grid_height as i32
                        {
                            continue;
                        }
                        let neighbor_index = neighbor_y as usize * grid_width + neighbor_x as usize;
                        if !known_cells[neighbor_index] {
                            continue;
                        }
                        let weight = (1.0 - distance / (radius as f32 + 1.0)).powi(2);
                        weight_sum += weight;
                        nearest_distance = nearest_distance.min(distance);
                        for channel in 0..3 {
                            weighted[channel] += spatial_gains[neighbor_index][channel] * weight;
                        }
                    }
                }
                if weight_sum == 0.0 || !weight_sum.is_finite() {
                    continue;
                }

                let distance_confidence =
                    (1.0 - nearest_distance / (radius as f32 + 1.0)).clamp(0.0, 1.0);
                let confidence = (distance_confidence * FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE)
                    .clamp(0.0, FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE);
                for channel in 0..3 {
                    let local_value = weighted[channel] / weight_sum;
                    propagated[index][channel] = (gains[channel] * (1.0 - confidence)
                        + local_value * confidence)
                        .clamp(FOCUS_COLOR_MIN_GAIN, FOCUS_COLOR_MAX_GAIN);
                }
            }
        }
        let mut propagated_offsets = spatial_offsets.clone();
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                let index = grid_y * grid_width + grid_x;
                if known_cells[index] {
                    continue;
                }

                let mut weight_sum = 0.0f32;
                let mut weighted = [0.0f32; 3];
                let mut nearest_distance = f32::INFINITY;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let distance = ((dx * dx + dy * dy) as f32).sqrt();
                        if distance > radius as f32 {
                            continue;
                        }
                        let neighbor_x = grid_x as i32 + dx;
                        let neighbor_y = grid_y as i32 + dy;
                        if neighbor_x < 0
                            || neighbor_y < 0
                            || neighbor_x >= grid_width as i32
                            || neighbor_y >= grid_height as i32
                        {
                            continue;
                        }
                        let neighbor_index = neighbor_y as usize * grid_width + neighbor_x as usize;
                        if !known_cells[neighbor_index] {
                            continue;
                        }
                        let weight = (1.0 - distance / (radius as f32 + 1.0)).powi(2);
                        weight_sum += weight;
                        nearest_distance = nearest_distance.min(distance);
                        for channel in 0..3 {
                            weighted[channel] += spatial_offsets[neighbor_index][channel] * weight;
                        }
                    }
                }
                if weight_sum == 0.0 || !weight_sum.is_finite() {
                    continue;
                }

                let distance_confidence =
                    (1.0 - nearest_distance / (radius as f32 + 1.0)).clamp(0.0, 1.0);
                let confidence = (distance_confidence * FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE)
                    .clamp(0.0, FOCUS_COLOR_PROPAGATION_MAX_CONFIDENCE);
                for channel in 0..3 {
                    let local_value = weighted[channel] / weight_sum;
                    propagated_offsets[index][channel] = (offsets[channel] * (1.0 - confidence)
                        + local_value * confidence)
                        .clamp(-FOCUS_COLOR_MAX_OFFSET, FOCUS_COLOR_MAX_OFFSET);
                }
            }
        }
        spatial_gains = propagated;
        spatial_offsets = propagated_offsets;
    }

    let has_spatial_variation = known_cell_count >= 2
        && known_cells.iter().enumerate().any(|(index, known)| {
            *known
                && spatial_gains[index]
                    .iter()
                    .zip(gains.iter())
                    .any(|(local_gain, global_gain)| {
                        (local_gain - global_gain).abs() >= FOCUS_COLOR_SPATIAL_VARIATION_THRESHOLD
                    })
        });
    let has_spatial_offset_variation = known_cell_count >= 2
        && known_cells.iter().enumerate().any(|(index, known)| {
            *known
                && spatial_offsets[index].iter().zip(offsets.iter()).any(
                    |(local_offset, global_offset)| {
                        (local_offset - global_offset).abs() >= FOCUS_COLOR_SPATIAL_OFFSET_THRESHOLD
                    },
                )
        });
    FocusColorCorrection {
        gains,
        offsets,
        cell_size: FOCUS_COLOR_CELL_SIZE,
        grid_width,
        grid_height,
        spatial_gains: has_spatial_variation.then_some(spatial_gains),
        spatial_offsets: has_spatial_offset_variation.then_some(spatial_offsets),
    }
}

fn apply_focus_color_correction(
    image: &mut Rgb32FImage,
    correction: &FocusColorCorrection,
    protected_mask: &GrayImage,
    canvas_only: bool,
) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return;
    }
    let has_protected_mask = protected_mask.dimensions() == (width, height);

    // Apply the exposure match to a low-frequency copy and add only that
    // correction back to the source. Applying gain/offset directly to every
    // source pixel compresses brush-stroke and weave contrast, which is not
    // acceptable for a high-resolution cultural-relic scan.
    let (analysis_width, analysis_height) =
        focus_analysis_dimensions(width, height, width.max(height));
    if (analysis_width, analysis_height) == (width, height) {
        let width = width as usize;
        image
            .as_mut()
            .par_chunks_mut(3)
            .enumerate()
            .for_each(|(index, pixel)| {
                let x = (index % width) as u32;
                let y = (index / width) as u32;
                if has_protected_mask && protected_mask.get_pixel(x, y)[0] > 0 {
                    return;
                }
                if canvas_only && !focus_stack_pixel_is_canvas_like(pixel) {
                    return;
                }
                let gains = correction.gain_at(x, y);
                let offsets = correction.offset_at(x, y);
                for (channel, value) in pixel.iter_mut().enumerate() {
                    *value = (*value * gains[channel] + offsets[channel]).clamp(0.0, 1.0);
                }
            });
        return;
    }

    let analysis = resize_rgb(image, analysis_width, analysis_height);
    let analysis_protected = if has_protected_mask {
        resize_binary_mask(protected_mask, analysis_width, analysis_height)
    } else {
        GrayImage::new(analysis_width, analysis_height)
    };
    let mut low_frequency_delta = Rgb32FImage::new(analysis_width, analysis_height);
    let analysis_width_usize = analysis_width as usize;
    let source_width = width as f64;
    let source_height = height as f64;
    low_frequency_delta
        .as_mut()
        .par_chunks_mut(3)
        .enumerate()
        .for_each(|(index, delta)| {
            if analysis_protected.as_raw()[index] > 0
                || (canvas_only
                    && !focus_stack_pixel_is_canvas_like(
                        &analysis.as_raw()[index * 3..index * 3 + 3],
                    ))
            {
                return;
            }
            let x = (index % analysis_width_usize) as f64;
            let y = (index / analysis_width_usize) as f64;
            let source_x = ((x + 0.5) * source_width / analysis_width as f64)
                .floor()
                .min(source_width - 1.0) as u32;
            let source_y = ((y + 0.5) * source_height / analysis_height as f64)
                .floor()
                .min(source_height - 1.0) as u32;
            let gains = correction.gain_at(source_x, source_y);
            let offsets = correction.offset_at(source_x, source_y);
            let source = &analysis.as_raw()[index * 3..index * 3 + 3];
            for channel in 0..3 {
                let corrected =
                    (source[channel] * gains[channel] + offsets[channel]).clamp(0.0, 1.0);
                delta[channel] = corrected - source[channel];
            }
        });
    // The correction field is already low-frequency. Sample it directly with
    // bilinear interpolation instead of allocating and resizing another full
    // source-sized RGB image for every frame in a long stack.
    let x_samples = linear_samples(analysis_width, width);
    let y_samples = linear_samples(analysis_height, height);
    let analysis_stride = analysis_width as usize * 3;
    let low_frequency_delta = low_frequency_delta.as_raw();
    image
        .as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            let top_row = y_sample.lower * analysis_stride;
            let bottom_row = y_sample.upper * analysis_stride;
            let y_weight = y_sample.upper_weight;
            for (x, x_sample) in x_samples.iter().copied().enumerate() {
                let index = y * width as usize + x;
                if has_protected_mask && protected_mask.as_raw()[index] > 0 {
                    continue;
                }
                let top_left = &low_frequency_delta
                    [top_row + x_sample.lower * 3..top_row + x_sample.lower * 3 + 3];
                let top_right = &low_frequency_delta
                    [top_row + x_sample.upper * 3..top_row + x_sample.upper * 3 + 3];
                let bottom_left = &low_frequency_delta
                    [bottom_row + x_sample.lower * 3..bottom_row + x_sample.lower * 3 + 3];
                let bottom_right = &low_frequency_delta
                    [bottom_row + x_sample.upper * 3..bottom_row + x_sample.upper * 3 + 3];
                let start = x * 3;
                if canvas_only && !focus_stack_pixel_is_canvas_like(&row[start..start + 3]) {
                    continue;
                }
                let x_weight = x_sample.upper_weight;
                for channel in 0..3 {
                    let top = top_left[channel] * (1.0 - x_weight) + top_right[channel] * x_weight;
                    let bottom =
                        bottom_left[channel] * (1.0 - x_weight) + bottom_right[channel] * x_weight;
                    let delta = top * (1.0 - y_weight) + bottom * y_weight;
                    row[start + channel] = (row[start + channel] + delta).clamp(0.0, 1.0);
                }
            }
        });
}

fn source_is_focus_foreground(image: &ImageInfo, source: Point2<f64>) -> bool {
    let Some((minimum_y, maximum_y)) = image.foreground_range else {
        return false;
    };
    let source_width = image.width().max(1) as f64;
    let source_height = image.height().max(1) as f64;
    if source.x < 0.0 || source.y < 0.0 || source.x >= source_width || source.y >= source_height {
        return false;
    }
    let source_y = source.y / source_height;
    if !(minimum_y..=maximum_y).contains(&source_y) {
        return false;
    }
    let alignment_scale = image.scale_factor.max(1.0);
    let alignment_x = (source.x / alignment_scale).round() as i32;
    let alignment_y = (source.y / alignment_scale).round() as i32;
    let (alignment_width, alignment_height) = image.alignment_image.dimensions();
    if alignment_x < 0
        || alignment_y < 0
        || alignment_x >= alignment_width as i32
        || alignment_y >= alignment_height as i32
    {
        return false;
    }
    image.foreground_mask.as_ref().map_or_else(
        || {
            image
                .alignment_image
                .get_pixel(alignment_x as u32, alignment_y as u32)[0]
                >= FOCUS_FOREGROUND_LUMA_THRESHOLD
        },
        |mask| mask.get_pixel(alignment_x as u32, alignment_y as u32)[0] > 0,
    )
}

fn focus_transformed_image_region(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
) -> Option<(u32, u32, u32, u32)> {
    if out_width == 0 || out_height == 0 {
        return None;
    }
    let Some((mut min_x, mut max_x, mut min_y, mut max_y)) =
        transformed_source_bounds(image, homography, projection, 0.0, 1.0)
    else {
        return None;
    };
    if let Some(focus_warp) = focus_warp {
        for band in &focus_warp.bands {
            let Some(band_homography) = band.homographies.get(&image.id) else {
                continue;
            };
            let Some(&(minimum_source_y, maximum_source_y)) = band.source_ranges.get(&image.id)
            else {
                continue;
            };
            let (minimum_source_x, maximum_source_x) = band
                .source_x_ranges
                .get(&image.id)
                .copied()
                .unwrap_or((0.0, 1.0));
            let Some((band_min_x, band_max_x, band_min_y, band_max_y)) =
                transformed_source_bounds_xy(
                    image,
                    band_homography,
                    projection,
                    minimum_source_x,
                    maximum_source_x,
                    minimum_source_y,
                    maximum_source_y,
                )
            else {
                continue;
            };
            min_x = min_x.min(band_min_x);
            max_x = max_x.max(band_max_x);
            min_y = min_y.min(band_min_y);
            max_y = max_y.max(band_max_y);
        }
    }
    let min_x = min_x + offset_x;
    let max_x = max_x + offset_x;
    let min_y = min_y + offset_y;
    let max_y = max_y + offset_y;
    let left = min_x.floor().max(0.0).min((out_width - 1) as f64) as u32;
    let right = max_x.ceil().max(0.0).min((out_width - 1) as f64) as u32;
    let top = min_y.floor().max(0.0).min((out_height - 1) as f64) as u32;
    let bottom = max_y.ceil().max(0.0).min((out_height - 1) as f64) as u32;
    (left <= right && top <= bottom).then_some((left, right, top, bottom))
}

#[allow(clippy::too_many_arguments)]
fn render_focus_layer(
    image: &ImageInfo,
    source_image: &Rgb32FImage,
    homography: &Matrix3<f64>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
) -> RenderedFocusLayer {
    let global_inverse = homography.try_inverse().unwrap_or_else(Matrix3::identity);
    let Some((left, right, top, bottom)) = focus_transformed_image_region(
        image, homography, projection, focus_warp, offset_x, offset_y, out_width, out_height,
    ) else {
        return RenderedFocusLayer {
            image: Rgb32FImage::new(0, 0),
            mask: GrayImage::new(0, 0),
            foreground_mask: GrayImage::new(0, 0),
            relaxed_foreground_mask: GrayImage::new(0, 0),
            left: 0,
            top: 0,
        };
    };
    let band_inverses = focus_warp
        .into_iter()
        .flat_map(|warp| warp.bands.iter())
        .filter_map(|band| {
            let homography = band.homographies.get(&image.id)?;
            let &(minimum_source_y, maximum_source_y) = band.source_ranges.get(&image.id)?;
            let (minimum_source_x, maximum_source_x) = band
                .source_x_ranges
                .get(&image.id)
                .copied()
                .unwrap_or((0.0, 1.0));
            let inverse = homography.try_inverse()?;
            let correction_inverse = (*homography * global_inverse).try_inverse()?;
            let axis = if maximum_source_x - minimum_source_x < maximum_source_y - minimum_source_y
            {
                0u8 // vertical edge: narrow source-x band
            } else {
                1u8 // horizontal/depth band: narrow source-y band
            };
            Some((
                minimum_source_x,
                maximum_source_x,
                minimum_source_y,
                maximum_source_y,
                inverse,
                correction_inverse,
                axis,
                band.relax_foreground_seam,
                band.foreground_only,
            ))
        })
        .collect::<Vec<_>>();
    let layer_width = right - left + 1;
    let layer_height = bottom - top + 1;
    let mut pixels = vec![0.0f32; layer_width as usize * layer_height as usize * 3];
    let mut mask = vec![0u8; layer_width as usize * layer_height as usize];
    let mut foreground_mask = vec![0u8; layer_width as usize * layer_height as usize];
    let mut relaxed_foreground_mask = vec![0u8; layer_width as usize * layer_height as usize];

    pixels
        .par_chunks_mut(layer_width as usize * 3)
        .zip(mask.par_chunks_mut(layer_width as usize))
        .zip(foreground_mask.par_chunks_mut(layer_width as usize))
        .zip(relaxed_foreground_mask.par_chunks_mut(layer_width as usize))
        .enumerate()
        .for_each(
            |(y, (((row, mask_row), foreground_mask_row), relaxed_row))| {
                let global_y = top + y as u32;
                for local_x in 0..layer_width {
                    let global_x = left + local_x;
                    let target =
                        Point3::new(global_x as f64 - offset_x, global_y as f64 - offset_y, 1.0);
                    let global_source =
                        map_target_to_source(&global_inverse, target, image, projection).filter(
                            |candidate| {
                                candidate.x >= 0.0
                                    && candidate.y >= 0.0
                                    && candidate.x < source_image.width() as f64
                                    && candidate.y < source_image.height() as f64
                            },
                        );
                    // A local model moves a depth layer relative to the paper. Do
                    // not use the global inverse to decide whether that moved pixel
                    // belongs to the band: doing so drops the local model exactly
                    // along the displaced silhouette and creates a step back to the
                    // paper transform. Instead, validate each local inverse in its
                    // own source domain, then let the narrowest active band win.
                    // This is content-agnostic; when no local bands exist the global
                    // inverse remains the only mapping.
                    let mut local_candidates = Vec::new();
                    for &(
                        minimum_source_x,
                        maximum_source_x,
                        minimum_source_y,
                        maximum_source_y,
                        ref band_inverse,
                        ref correction_inverse,
                        axis,
                        relax_foreground_seam,
                        foreground_only,
                    ) in &band_inverses
                    {
                        let Some(candidate) =
                            map_target_to_source(band_inverse, target, image, projection)
                        else {
                            continue;
                        };
                        if candidate.x < 0.0
                            || candidate.y < 0.0
                            || candidate.x >= source_image.width() as f64
                            || candidate.y >= source_image.height() as f64
                        {
                            continue;
                        }
                        if foreground_only && !source_is_focus_foreground(image, candidate) {
                            continue;
                        }
                        let normalized_x = candidate.x / image.width().max(1) as f64;
                        let normalized_y = candidate.y / image.height().max(1) as f64;
                        if normalized_x < minimum_source_x - 0.02
                            || normalized_x > maximum_source_x + 0.02
                            || normalized_y < minimum_source_y - 0.02
                            || normalized_y > maximum_source_y + 0.02
                        {
                            continue;
                        }
                        let x_span = (maximum_source_x - minimum_source_x).max(1e-6);
                        let y_span = (maximum_source_y - minimum_source_y).max(1e-6);
                        let (span, normalized_axis, center_axis) = if x_span < y_span {
                            (
                                x_span,
                                normalized_x,
                                (minimum_source_x + maximum_source_x) * 0.5,
                            )
                        } else {
                            (
                                y_span,
                                normalized_y,
                                (minimum_source_y + maximum_source_y) * 0.5,
                            )
                        };
                        local_candidates.push((
                            span,
                            normalized_axis,
                            center_axis,
                            axis,
                            candidate,
                            *correction_inverse,
                            relax_foreground_seam,
                            foreground_only,
                        ));
                    }
                    let select_local_candidate = |axis: u8| {
                        let narrowest_span = local_candidates
                            .iter()
                            .filter(|candidate| candidate.3 == axis)
                            .map(|candidate| candidate.0)
                            .fold(f64::INFINITY, f64::min);
                        if !narrowest_span.is_finite() {
                            return None;
                        }
                        let maximum_span = narrowest_span * 1.25;
                        // Several generic/edge bands can overlap in source space.
                        // Keep one candidate per axis: prefer the narrowest valid
                        // band, then the band whose centre is closest to this
                        // sample. Orthogonal bands are composed below rather than
                        // allowing one axis to discard the other.
                        local_candidates
                            .iter()
                            .filter(|candidate| candidate.3 == axis && candidate.0 <= maximum_span)
                            .min_by(|left, right| {
                                left.0.total_cmp(&right.0).then_with(|| {
                                    (left.1 - left.2)
                                        .abs()
                                        .total_cmp(&(right.1 - right.2).abs())
                                })
                            })
                    };
                    let selected_vertical = select_local_candidate(0);
                    let selected_horizontal = select_local_candidate(1);
                    let local_strength_for =
                        |candidate: &(f64, f64, f64, u8, Point2<f64>, Matrix3<f64>, bool, bool)| {
                            (1.0 - (candidate.1 - candidate.2).abs() / (candidate.0 * 0.5))
                                .clamp(0.0, 1.0)
                        };
                    let vertical_strength = selected_vertical.map(local_strength_for);
                    let horizontal_strength = selected_horizontal.map(local_strength_for);
                    let mut local_source = None;
                    let mut local_strength = 0.0f64;
                    let mut relax_foreground_seam = false;
                    match (selected_vertical, selected_horizontal) {
                        (Some(vertical), Some(horizontal)) => {
                            // The stored correction is a world-space correction
                            // on top of the global pose. Apply inverse horizontal
                            // then inverse vertical correction before the global
                            // inverse; this preserves both sides of a detected
                            // corner/endpoint without introducing an averaged pose.
                            let corrected_target = horizontal.5 * target;
                            let corrected_target = vertical.5 * corrected_target;
                            local_source = map_target_to_source(
                                &global_inverse,
                                corrected_target,
                                image,
                                projection,
                            )
                            .filter(|candidate| {
                                candidate.x >= 0.0
                                    && candidate.y >= 0.0
                                    && candidate.x < source_image.width() as f64
                                    && candidate.y < source_image.height() as f64
                            });
                            local_strength = vertical_strength
                                .unwrap_or(0.0)
                                .min(horizontal_strength.unwrap_or(0.0));
                            relax_foreground_seam = vertical.6 || horizontal.6;
                        }
                        (Some(vertical), None) => {
                            local_source = Some(vertical.4);
                            local_strength = vertical_strength.unwrap_or(0.0);
                            relax_foreground_seam = vertical.6;
                        }
                        (None, Some(horizontal)) => {
                            local_source = Some(horizontal.4);
                            local_strength = horizontal_strength.unwrap_or(0.0);
                            relax_foreground_seam = horizontal.6;
                        }
                        (None, None) => {}
                    }
                    if local_strength <= 0.0 {
                        local_source = None;
                        relax_foreground_seam = false;
                    }
                    // Blend each selected regional correction into the global
                    // mapping at the edges of its source band. A hard switch is
                    // itself visible as a seam, especially when the corrected
                    // object is a rail or paper boundary. Each axis' influence
                    // falls continuously to zero at its band boundary; when both
                    // axes are active their composed correction uses the smaller
                    // overlap weight. If the global inverse is unavailable, the
                    // local mapping remains a valid fallback.
                    let source = match (global_source, local_source) {
                        (Some(global), Some(local)) => {
                            let local_weight = local_strength.clamp(0.0, 1.0);
                            Some(Point2::new(
                                global.x * (1.0 - local_weight) + local.x * local_weight,
                                global.y * (1.0 - local_weight) + local.y * local_weight,
                            ))
                        }
                        (Some(global), None) => Some(global),
                        (None, Some(local)) => Some(local),
                        (None, None) => None,
                    };
                    let Some(source) = source else {
                        continue;
                    };
                    if source.x < 1.0
                        || source.y < 1.0
                        || source.x >= source_image.width() as f64 - 2.0
                        || source.y >= source_image.height() as f64 - 2.0
                    {
                        continue;
                    }
                    let pixel =
                        get_high_quality_interpolated_pixel(source_image, source.x, source.y);
                    let start = local_x as usize * 3;
                    row[start..start + 3].copy_from_slice(&pixel.0);
                    mask_row[local_x as usize] = 255;
                    // This mask controls focus ownership and duplicate suppression;
                    // it must stay geometric. Colour classification belongs only
                    // to the final canvas-tone protection below. Marking every
                    // black stroke or red seal here would make the first soft
                    // sample win forever and produce the reported ghosting.
                    if source_is_focus_foreground(image, source) {
                        foreground_mask_row[local_x as usize] = 255;
                        if relax_foreground_seam {
                            relaxed_row[local_x as usize] = 255;
                        }
                    }
                }
            },
        );

    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(layer_width, layer_height, pixels)
            .expect("focus layer buffer dimensions must match"),
        mask: GrayImage::from_raw(layer_width, layer_height, mask)
            .expect("focus layer mask dimensions must match"),
        foreground_mask: GrayImage::from_raw(layer_width, layer_height, foreground_mask)
            .expect("focus foreground mask buffer dimensions must match"),
        relaxed_foreground_mask: GrayImage::from_raw(
            layer_width,
            layer_height,
            relaxed_foreground_mask,
        )
        .expect("focus relaxed foreground mask dimensions must match"),
        left,
        top,
    }
}

fn focus_analysis_dimensions(width: u32, height: u32, source_longest_dimension: u32) -> (u32, u32) {
    let longest_side = width.max(height).max(1);
    // Keep the focus-analysis sampling density tied to an individual source,
    // not to the total mosaic length. Otherwise a long shifted stack compresses
    // each 9504px frame to a few hundred analysis pixels and erases the very
    // sharpness differences the stacker is supposed to select.
    let source_scale = FOCUS_ANALYSIS_MAX_DIMENSION as f64 / source_longest_dimension.max(1) as f64;
    let mut scale = source_scale.min(1.0);
    let scaled_pixels = width as f64 * height as f64 * scale * scale;
    if scaled_pixels > FOCUS_ANALYSIS_MAX_PIXELS as f64 {
        scale *= (FOCUS_ANALYSIS_MAX_PIXELS as f64 / scaled_pixels).sqrt();
    }
    if scale >= 1.0 && longest_side <= FOCUS_ANALYSIS_MAX_DIMENSION {
        return (width.max(1), height.max(1));
    }
    (
        (width as f64 * scale).round().max(1.0) as u32,
        (height as f64 * scale).round().max(1.0) as u32,
    )
}

fn box_blur_focus_map(source: &[f32], width: u32, height: u32, radius: usize) -> Vec<f32> {
    if radius == 0 || width == 0 || height == 0 {
        return source.to_vec();
    }
    let width = width as usize;
    let height = height as usize;
    let window_size = radius * 2 + 1;
    let divisor = window_size as f32;
    let mut horizontal = vec![0.0f32; source.len()];
    horizontal
        .par_chunks_mut(width)
        .zip(source.par_chunks(width))
        .for_each(|(output_row, source_row)| {
            let mut sum = 0.0f32;
            for offset in 0..window_size {
                sum += source_row[offset.saturating_sub(radius).min(width - 1)];
            }
            output_row[0] = sum / divisor;
            for (x, output) in output_row.iter_mut().enumerate().skip(1) {
                let add_x = (x + radius).min(width - 1);
                let remove_x = x.saturating_sub(radius + 1);
                sum += source_row[add_x] - source_row[remove_x];
                *output = sum / divisor;
            }
        });

    let mut output = vec![0.0f32; source.len()];
    for x in 0..width {
        let mut sum = 0.0f32;
        for offset in 0..window_size {
            let y = offset.saturating_sub(radius).min(height - 1);
            sum += horizontal[y * width + x];
        }
        output[x] = sum / divisor;
        for y in 1..height {
            let add_y = (y + radius).min(height - 1);
            let remove_y = y.saturating_sub(radius + 1);
            sum += horizontal[add_y * width + x] - horizontal[remove_y * width + x];
            output[y * width + x] = sum / divisor;
        }
    }
    output
}

fn focus_score_map(image: &Rgb32FImage, mask: &GrayImage) -> Vec<f32> {
    let (width, height) = image.dimensions();
    let pixel_count = width as usize * height as usize;
    let mut luminance_map = vec![0.0f32; pixel_count];
    luminance_map
        .par_iter_mut()
        .zip(image.as_raw().par_chunks(3))
        .for_each(|(output, pixel)| {
            *output = pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114;
        });

    let mut raw_score = vec![0.0f32; pixel_count];
    if width >= 3 && height >= 3 {
        raw_score
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                if y == 0 || y + 1 >= height as usize {
                    return;
                }
                for (x, output) in row.iter_mut().enumerate().take(width as usize - 1).skip(1) {
                    let index = y * width as usize + x;
                    if mask.as_raw()[index] == 0
                        || mask.as_raw()[index - 1] == 0
                        || mask.as_raw()[index + 1] == 0
                        || mask.as_raw()[index - width as usize] == 0
                        || mask.as_raw()[index + width as usize] == 0
                    {
                        continue;
                    }
                    *output = (4.0 * luminance_map[index]
                        - luminance_map[index - 1]
                        - luminance_map[index + 1]
                        - luminance_map[index - width as usize]
                        - luminance_map[index + width as usize])
                        .abs();
                }
            });
    }

    // Focus ownership is decided on a bounded analysis canvas. Keep the score
    // window local enough that one clear stroke cannot claim an adjacent
    // character; the final pixels still come from the full-resolution aligned
    // layer.
    let radius = ((width.max(height) as f32 / FOCUS_SCORE_BLUR_RADIUS_DIVISOR).round() as usize)
        .clamp(1, FOCUS_SCORE_MAX_BLUR_RADIUS);
    box_blur_focus_map(&raw_score, width, height, radius)
}

fn focus_decision_mask(
    base_focus: &[f32],
    candidate_focus: &[f32],
    base_mask: &GrayImage,
    candidate_mask: &GrayImage,
) -> GrayImage {
    let (width, height) = base_mask.dimensions();
    let pixel_count = width as usize * height as usize;
    let mut fine_advantage = vec![0.0f32; pixel_count];
    fine_advantage
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, output)| {
            let base_valid = base_mask.as_raw()[index] > 0;
            let candidate_valid = candidate_mask.as_raw()[index] > 0;
            *output = match (base_valid, candidate_valid) {
                (false, true) => 1.0,
                (true, false) => -1.0,
                (true, true) => {
                    let base = base_focus[index];
                    let candidate = candidate_focus[index];
                    (candidate - base) / (candidate + base + 1e-6)
                }
                (false, false) => 0.0,
            };
        });

    // The old radius was effectively ~100 source pixels for a 9504px frame.
    // That let a clear stroke in one region claim a neighbouring blurred
    // character. Keep the decision coherent, but make the neighbourhood local
    // enough that each character can select its own sharp source.
    let coherence_radius = ((width.max(height) as f32 / FOCUS_DECISION_COHERENCE_RADIUS_DIVISOR)
        .round() as usize)
        .clamp(2, FOCUS_DECISION_MAX_COHERENCE_RADIUS);
    let coarse_advantage = box_blur_focus_map(&fine_advantage, width, height, coherence_radius);

    let sample_step = (pixel_count / 4096).max(1);
    let mut magnitude_samples = (0..pixel_count)
        .step_by(sample_step)
        .filter_map(|index| {
            (base_mask.as_raw()[index] > 0 || candidate_mask.as_raw()[index] > 0)
                .then_some(base_focus[index].max(candidate_focus[index]))
        })
        .collect::<Vec<_>>();
    magnitude_samples.sort_unstable_by(f32::total_cmp);
    let confidence_floor = magnitude_samples
        .get(((magnitude_samples.len().saturating_sub(1)) as f32 * 0.9) as usize)
        .copied()
        .unwrap_or(0.0)
        * 0.04;

    // A defocused subject spills contrast beyond its true silhouette. That halo can
    // look locally sharper than the clean background in the focused layer, leaving
    // detached slivers around depth discontinuities even with hard pixel selection.
    // Let decisive, valid structure claim a small adjacent ambiguous band so the
    // focused silhouette also supplies the clean pixels immediately around it.
    let mut strong_base = vec![0.0f32; pixel_count];
    let mut strong_candidate = vec![0.0f32; pixel_count];
    strong_base
        .par_iter_mut()
        .zip(strong_candidate.par_iter_mut())
        .enumerate()
        .for_each(|(index, (base_claim, candidate_claim))| {
            let both_valid = base_mask.as_raw()[index] > 0 && candidate_mask.as_raw()[index] > 0;
            let confident = base_focus[index].max(candidate_focus[index]) > confidence_floor;
            if !both_valid || !confident {
                return;
            }
            let advantage = fine_advantage[index];
            if advantage < -FOCUS_DECISIVE_ADVANTAGE {
                *base_claim = 1.0;
            } else if advantage > FOCUS_DECISIVE_ADVANTAGE {
                *candidate_claim = 1.0;
            }
        });
    let protection_radius = ((width.max(height) as f32 / 2048.0)
        * (FOCUS_EDGE_PROTECTION_AT_1024 * FOCUS_EDGE_PROTECTION_SCALE))
        .round()
        .clamp(2.0, 6.0) as usize;
    let base_claim = box_blur_focus_map(&strong_base, width, height, protection_radius);
    let candidate_claim = box_blur_focus_map(&strong_candidate, width, height, protection_radius);

    GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        let base_valid = base_mask.as_raw()[index] > 0;
        let candidate_valid = candidate_mask.as_raw()[index] > 0;
        let confident = base_focus[index].max(candidate_focus[index]) > confidence_floor;
        // Always use the coherent neighborhood decision for ownership. A single
        // high-contrast stroke can be displaced by a couple of pixels between
        // captures; using its raw per-pixel score would then alternate sources
        // across the stroke and recreate a double contour.
        let advantage = coarse_advantage[index];
        let nearby_base_claim = base_claim[index];
        let nearby_candidate_claim = candidate_claim[index];
        let structural_winner = if nearby_candidate_claim > FOCUS_EDGE_CLAIM_THRESHOLD
            && nearby_candidate_claim > nearby_base_claim
        {
            Some(true)
        } else if nearby_base_claim > FOCUS_EDGE_CLAIM_THRESHOLD
            && nearby_base_claim > nearby_candidate_claim
        {
            Some(false)
        } else {
            None
        };
        let candidate_wins = candidate_valid
            && (!base_valid
                || structural_winner.unwrap_or(confident && advantage > FOCUS_CONFIDENCE_MARGIN));
        image::Luma([if candidate_wins { 255 } else { 0 }])
    })
}

fn suppress_focus_canvas_switches(
    decision_mask: &mut GrayImage,
    base: &Rgb32FImage,
    candidate: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate_mask: &GrayImage,
) {
    if decision_mask.dimensions() != base.dimensions()
        || base.dimensions() != candidate.dimensions()
        || base_mask.dimensions() != base.dimensions()
        || candidate_mask.dimensions() != base.dimensions()
    {
        return;
    }

    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(base.as_raw().par_chunks(3))
        .zip(candidate.as_raw().par_chunks(3))
        .zip(base_mask.as_raw().par_iter())
        .zip(candidate_mask.as_raw().par_iter())
        .for_each(
            |((((decision, base_pixel), candidate_pixel), base_valid), candidate_valid)| {
                // The canvas is one planar surface. Sharpness fluctuations in its
                // weave must not make the focus score swap the source for an
                // already-covered pixel; those swaps are the rectangular exposure
                // blocks visible in a shifted scan. A painted pixel can still win
                // normally, and a new candidate can still fill an uncovered hole.
                if *decision > 0
                    && *base_valid > 0
                    && *candidate_valid > 0
                    && focus_stack_pixel_is_canvas_like(base_pixel)
                    && focus_stack_pixel_is_canvas_like(candidate_pixel)
                    && !focus_stack_pixel_is_tone_foreground(base_pixel)
                    && !focus_stack_pixel_is_tone_foreground(candidate_pixel)
                {
                    *decision = 0;
                }
            },
        );
}

fn resize_binary_mask(mask: &GrayImage, width: u32, height: u32) -> GrayImage {
    image::imageops::resize(
        mask,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Nearest,
    )
}

fn dilate_focus_binary_mask(mask: &GrayImage, radius: usize) -> GrayImage {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return GrayImage::new(width, height);
    }
    let seed = mask
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let dilated = box_blur_focus_map(&seed, width, height, radius);
    GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        image::Luma([u8::from(dilated[index] > 0.0) * 255])
    })
}

fn open_focus_binary_mask(mask: &GrayImage, radius: usize) -> GrayImage {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 || radius == 0 {
        return mask.clone();
    }
    let seed = mask
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let eroded = box_blur_focus_map(&seed, width, height, radius);
    let eroded = GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        image::Luma([u8::from(eroded[index] >= 0.999) * 255])
    });
    dilate_focus_binary_mask(&eroded, radius)
}

fn fill_focus_foreground_holes(mask: &GrayImage) -> GrayImage {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return mask.clone();
    }
    let width = width as usize;
    let height = height as usize;
    let source = mask.as_raw();
    let mut outside = vec![false; source.len()];
    let mut pending = Vec::new();
    let enqueue = |index: usize, outside: &mut [bool], pending: &mut Vec<usize>| {
        if source[index] == 0 && !outside[index] {
            outside[index] = true;
            pending.push(index);
        }
    };
    for x in 0..width {
        enqueue(x, &mut outside, &mut pending);
        enqueue((height - 1) * width + x, &mut outside, &mut pending);
    }
    for y in 0..height {
        enqueue(y * width, &mut outside, &mut pending);
        enqueue(y * width + width - 1, &mut outside, &mut pending);
    }
    while let Some(index) = pending.pop() {
        let x = index % width;
        let y = index / width;
        if x > 0 {
            enqueue(index - 1, &mut outside, &mut pending);
        }
        if x + 1 < width {
            enqueue(index + 1, &mut outside, &mut pending);
        }
        if y > 0 {
            enqueue(index - width, &mut outside, &mut pending);
        }
        if y + 1 < height {
            enqueue(index + width, &mut outside, &mut pending);
        }
    }

    let mut filled = source.to_vec();
    for (index, value) in filled.iter_mut().enumerate() {
        if *value == 0 && !outside[index] {
            *value = 255;
        }
    }
    GrayImage::from_raw(width as u32, height as u32, filled)
        .expect("focus hole-filled foreground mask dimensions must match")
}

fn retain_focus_foreground_components(
    foreground_mask: &GrayImage,
    seed_mask: &GrayImage,
) -> GrayImage {
    let (width, height) = foreground_mask.dimensions();
    if (width, height) != seed_mask.dimensions() || width == 0 || height == 0 {
        return GrayImage::new(width, height);
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    let foreground = foreground_mask.as_raw();
    let seeds = seed_mask.as_raw();
    let mut visited = vec![false; width_usize * height_usize];
    let mut retained = vec![0u8; width_usize * height_usize];
    let minimum_component_size = ((width as u64 * height as u64) as f64 * 0.00018)
        .round()
        .clamp(24.0, 512.0) as usize;
    for start in 0..foreground.len() {
        if foreground[start] == 0 || visited[start] {
            continue;
        }
        visited[start] = true;
        let mut pending = vec![start];
        let mut component = Vec::new();
        let mut has_seed = false;
        let mut minimum_x = start % width_usize;
        let mut maximum_x = minimum_x;
        let mut minimum_y = start / width_usize;
        let mut maximum_y = minimum_y;
        while let Some(index) = pending.pop() {
            component.push(index);
            has_seed |= seeds[index] > 0;
            let x = index % width_usize;
            let y = index / width_usize;
            minimum_x = minimum_x.min(x);
            maximum_x = maximum_x.max(x);
            minimum_y = minimum_y.min(y);
            maximum_y = maximum_y.max(y);
            if x > 0 {
                let neighbor = index - 1;
                if foreground[neighbor] > 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
            if x + 1 < width_usize {
                let neighbor = index + 1;
                if foreground[neighbor] > 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
            if y > 0 {
                let neighbor = index - width_usize;
                if foreground[neighbor] > 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
            if y + 1 < height_usize {
                let neighbor = index + width_usize;
                if foreground[neighbor] > 0 && !visited[neighbor] {
                    visited[neighbor] = true;
                    pending.push(neighbor);
                }
            }
        }
        // Exposure-shifted canvas noise can satisfy a colour seed locally, but
        // it should not become a protected island. Real painted regions in this
        // stack are much larger and remain connected at analysis resolution.
        let component_width = maximum_x - minimum_x + 1;
        let component_height = maximum_y - minimum_y + 1;
        let minimum_component_width = (width as f32 * 0.006).round().clamp(8.0, 32.0) as usize;
        if has_seed
            && component.len() >= minimum_component_size
            && component_width >= minimum_component_width
            && component_height >= minimum_component_width
        {
            for index in component {
                retained[index] = 255;
            }
        }
    }

    GrayImage::from_raw(width, height, retained)
        .expect("focus retained foreground mask dimensions must match")
}

fn build_focus_background_foreground_envelope(
    analysis_image: &Rgb32FImage,
    analysis_mask: &GrayImage,
    existing_foreground_mask: &GrayImage,
) -> GrayImage {
    let (width, height) = analysis_image.dimensions();
    if width == 0
        || height == 0
        || analysis_mask.dimensions() != (width, height)
        || existing_foreground_mask.dimensions() != (width, height)
    {
        return GrayImage::new(width, height);
    }

    // A raw colour mask contains isolated canvas-weave outliers. Keep only
    // connected components that have enough support to be painted content,
    // then expand the retained silhouette so skin/highlights between the dark
    // hair and robe are protected as one object rather than treated as paper.
    let tone_seed = GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        let covered = analysis_mask.as_raw()[index] > 0;
        let start = index * 3;
        image::Luma([u8::from(
            covered
                && focus_stack_pixel_is_tone_foreground(&analysis_image.as_raw()[start..start + 3]),
        ) * 255])
    });
    // Canvas weave can create long, one-pixel colour runs that connect to a
    // real silhouette. Open the seed before connected-component filtering so
    // those runs cannot become foreground exclusions across an entire source
    // tile. The radius is only a few analysis pixels and is applied to the
    // mask, never to the final image.
    let seed_cleanup_radius = (width.max(height) as f32 * 0.0015).round().clamp(2.0, 4.0) as usize;
    let cleaned_tone_seed = open_focus_binary_mask(&tone_seed, seed_cleanup_radius);
    let retained_seed = retain_focus_foreground_components(&cleaned_tone_seed, &cleaned_tone_seed);
    let retained_seed = fill_focus_foreground_holes(&retained_seed);
    let retained_count = retained_seed
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    let mut envelope = if retained_count >= 24 {
        let radius = (width.max(height) as f32 * FOCUS_BACKGROUND_TONE_FOREGROUND_RADIUS_RATIO)
            .round()
            .clamp(8.0, 64.0) as usize;
        dilate_focus_binary_mask(&retained_seed, radius)
    } else {
        GrayImage::new(width, height)
    };

    // The geometric mask is an additional safety net when a caller has a
    // detected depth layer. It is intentionally dilated by the same bounded
    // halo so the tone pass cannot nibble at its silhouette edge.
    let existing_count = existing_foreground_mask
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if existing_count > 0 {
        let radius = (width.max(height) as f32 * 0.01).round().clamp(4.0, 32.0) as usize;
        let existing = dilate_focus_binary_mask(existing_foreground_mask, radius);
        for (destination, source) in envelope.as_mut().iter_mut().zip(existing.as_raw()) {
            *destination = (*destination).max(*source);
        }
    }
    envelope
}

fn build_focus_background_reference_mask(
    analysis_image: &Rgb32FImage,
    analysis_mask: &GrayImage,
    existing_foreground_mask: &GrayImage,
) -> GrayImage {
    let (width, height) = analysis_image.dimensions();
    if width == 0
        || height == 0
        || analysis_mask.dimensions() != (width, height)
        || existing_foreground_mask.dimensions() != (width, height)
    {
        return GrayImage::new(width, height);
    }

    // Do not use a colour threshold here. The same paper can be considerably
    // darker or lighter in different source frames, which is exactly the
    // exposure step this reference is meant to remove. The foreground
    // envelope is content-derived and is the only exclusion needed here.
    let foreground_envelope = build_focus_background_foreground_envelope(
        analysis_image,
        analysis_mask,
        existing_foreground_mask,
    );
    GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        image::Luma([u8::from(
            analysis_mask.as_raw()[index] > 0 && foreground_envelope.as_raw()[index] == 0,
        ) * 255])
    })
}

fn update_focus_background_reference(
    reference_image: &mut Rgb32FImage,
    reference_mask: &mut GrayImage,
    reference_counts: &mut [u16],
    candidate_image: &Rgb32FImage,
    candidate_mask: &GrayImage,
) -> [f32; 3] {
    let (width, height) = reference_image.dimensions();
    if width == 0
        || height == 0
        || candidate_image.dimensions() != (width, height)
        || reference_mask.dimensions() != (width, height)
        || candidate_mask.dimensions() != (width, height)
        || reference_counts.len() != width as usize * height as usize
    {
        return [0.0; 3];
    }

    let reference_pixels = reference_image.as_raw();
    let reference_mask_pixels = reference_mask.as_raw();
    let candidate_pixels = candidate_image.as_raw();
    let candidate_mask_pixels = candidate_mask.as_raw();
    let sample_step = ((u64::from(width) * u64::from(height) / 120_000) as f64)
        .sqrt()
        .ceil()
        .max(1.0) as usize;
    let mut channel_differences = [Vec::new(), Vec::new(), Vec::new()];
    for y in (0..height as usize).step_by(sample_step) {
        for x in (0..width as usize).step_by(sample_step) {
            let index = y * width as usize + x;
            if reference_mask_pixels[index] == 0 || candidate_mask_pixels[index] == 0 {
                continue;
            }
            let start = index * 3;
            for channel in 0..3 {
                channel_differences[channel]
                    .push(reference_pixels[start + channel] - candidate_pixels[start + channel]);
            }
        }
    }

    let correction = [
        median_f32(&mut channel_differences[0])
            .unwrap_or(0.0)
            .clamp(
                -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
            ),
        median_f32(&mut channel_differences[1])
            .unwrap_or(0.0)
            .clamp(
                -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
            ),
        median_f32(&mut channel_differences[2])
            .unwrap_or(0.0)
            .clamp(
                -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
            ),
    ];

    let reference_pixels = reference_image.as_mut();
    let reference_mask_pixels = reference_mask.as_mut();
    for index in 0..candidate_mask_pixels.len() {
        if candidate_mask_pixels[index] == 0 {
            continue;
        }
        let start = index * 3;
        let mut normalized = [0.0f32; 3];
        for channel in 0..3 {
            normalized[channel] =
                (candidate_pixels[start + channel] + correction[channel]).clamp(0.0, 1.0);
        }
        let count = reference_counts[index];
        if count == 0 || reference_mask_pixels[index] == 0 {
            reference_pixels[start..start + 3].copy_from_slice(&normalized);
            reference_counts[index] = 1;
            reference_mask_pixels[index] = 255;
            continue;
        }
        let old_weight = f32::from(count);
        let next_weight = old_weight + 1.0;
        for channel in 0..3 {
            reference_pixels[start + channel] = (reference_pixels[start + channel] * old_weight
                + normalized[channel])
                / next_weight;
        }
        reference_counts[index] = count.saturating_add(1);
    }
    correction
}

fn hard_select_focus_pixels(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    decision_mask: &GrayImage,
) {
    debug_assert_eq!(base.dimensions(), candidate.dimensions());
    debug_assert_eq!(base.dimensions(), base_mask.dimensions());
    debug_assert_eq!(base.dimensions(), candidate_mask.dimensions());
    debug_assert_eq!(base.dimensions(), decision_mask.dimensions());
    let base_pixels: &mut [f32] = base.as_mut();
    let base_mask_pixels: &mut [u8] = base_mask.as_mut();
    base_pixels
        .par_chunks_mut(3)
        .zip(base_mask_pixels.par_iter_mut())
        .zip(candidate.as_raw().par_chunks(3))
        .zip(candidate_mask.as_raw().par_iter())
        .zip(decision_mask.as_raw().par_iter())
        .for_each(
            |((((base_pixel, base_valid), candidate_pixel), candidate_valid), decision)| {
                if *candidate_valid > 0 && (*base_valid == 0 || *decision > 0) {
                    base_pixel.copy_from_slice(candidate_pixel);
                }
                if *candidate_valid > 0 {
                    *base_valid = 255;
                }
            },
        );
}

fn place_focus_layer(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    candidate: &RenderedFocusLayer,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = base.width();
    let base_height = base.height();
    if candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }

    let base_stride = base_width as usize * 3;
    let layer_stride = layer_width as usize * 3;
    base.as_mut()
        .par_chunks_mut(base_stride)
        .zip(base_mask.as_mut().par_chunks_mut(base_width as usize))
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(candidate.image.as_raw().par_chunks(layer_stride))
        .zip(candidate.mask.as_raw().par_chunks(layer_width as usize))
        .for_each(
            |(((base_row, base_mask_row), candidate_row), candidate_mask_row)| {
                let base_start = candidate.left as usize * 3;
                let base_end = base_start + layer_stride;
                base_row[base_start..base_end].copy_from_slice(candidate_row);
                let mask_start = candidate.left as usize;
                let mask_end = mask_start + layer_width as usize;
                base_mask_row[mask_start..mask_end].copy_from_slice(candidate_mask_row);
            },
        );
}

fn place_focus_foreground_mask(base: &mut GrayImage, candidate: &RenderedFocusLayer) {
    let (layer_width, layer_height) = candidate.foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = base.width();
    let base_height = base.height();
    if candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }

    base.as_mut()
        .par_chunks_mut(base_width as usize)
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(
            candidate
                .foreground_mask
                .as_raw()
                .par_chunks(layer_width as usize),
        )
        .for_each(|(base_row, candidate_row)| {
            let start = candidate.left as usize;
            let end = start + layer_width as usize;
            base_row[start..end].copy_from_slice(candidate_row);
        });
}

fn focus_rgb_luma_at(image: &Rgb32FImage, x: i32, y: i32) -> Option<f64> {
    let (width, height) = image.dimensions();
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return None;
    }
    let pixel = image.get_pixel(x as u32, y as u32);
    pixel
        .0
        .iter()
        .all(|value| value.is_finite())
        .then_some(pixel[0] as f64 * 0.299 + pixel[1] as f64 * 0.587 + pixel[2] as f64 * 0.114)
}

fn focus_alignment_gradient_at(
    image: &Rgb32FImage,
    mask: &GrayImage,
    x: i32,
    y: i32,
) -> Option<(f64, f64)> {
    let (width, height) = image.dimensions();
    if x < 1 || y < 1 || x + 1 >= width as i32 || y + 1 >= height as i32 {
        return None;
    }
    let valid =
        |sample_x: i32, sample_y: i32| mask.get_pixel(sample_x as u32, sample_y as u32)[0] > 0;
    if !valid(x, y) || !valid(x - 1, y) || !valid(x + 1, y) || !valid(x, y - 1) || !valid(x, y + 1)
    {
        return None;
    }
    let horizontal = focus_rgb_luma_at(image, x + 1, y)? - focus_rgb_luma_at(image, x - 1, y)?;
    let vertical = focus_rgb_luma_at(image, x, y + 1)? - focus_rgb_luma_at(image, x, y - 1)?;
    Some((horizontal, vertical))
}

fn focus_full_resolution_patch_score(
    candidate: &RenderedFocusLayer,
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
    center_x: i32,
    center_y: i32,
    delta_x: i32,
    delta_y: i32,
    radius: i32,
) -> Option<(f64, f64, f64)> {
    let (candidate_width, candidate_height) = candidate.image.dimensions();
    let mut candidate_horizontal_sum = 0.0;
    let mut candidate_vertical_sum = 0.0;
    let mut merged_horizontal_sum = 0.0;
    let mut merged_vertical_sum = 0.0;
    let mut candidate_squared_sum = 0.0;
    let mut merged_squared_sum = 0.0;
    let mut product_sum = 0.0;
    let mut sample_count = 0usize;
    let mut foreground_count = 0usize;
    let mut patch_count = 0usize;
    for offset_y in -radius..=radius {
        for offset_x in -radius..=radius {
            let local_x = center_x + offset_x;
            let local_y = center_y + offset_y;
            if local_x >= 0
                && local_y >= 0
                && local_x < candidate_width as i32
                && local_y < candidate_height as i32
            {
                patch_count += 1;
                if candidate
                    .foreground_mask
                    .get_pixel(local_x as u32, local_y as u32)[0]
                    > 0
                {
                    foreground_count += 1;
                }
            }
            let Some((candidate_horizontal, candidate_vertical)) =
                focus_alignment_gradient_at(&candidate.image, &candidate.mask, local_x, local_y)
            else {
                continue;
            };
            let merged_x = candidate.left as i32 + local_x + delta_x;
            let merged_y = candidate.top as i32 + local_y + delta_y;
            let Some((merged_horizontal, merged_vertical)) =
                focus_alignment_gradient_at(merged, merged_mask, merged_x, merged_y)
            else {
                continue;
            };
            candidate_horizontal_sum += candidate_horizontal;
            candidate_vertical_sum += candidate_vertical;
            merged_horizontal_sum += merged_horizontal;
            merged_vertical_sum += merged_vertical;
            candidate_squared_sum += candidate_horizontal * candidate_horizontal
                + candidate_vertical * candidate_vertical;
            merged_squared_sum +=
                merged_horizontal * merged_horizontal + merged_vertical * merged_vertical;
            product_sum +=
                candidate_horizontal * merged_horizontal + candidate_vertical * merged_vertical;
            sample_count += 1;
        }
    }
    let minimum_samples = (((radius * 2 + 1) * (radius * 2 + 1)) as f64 * 0.45)
        .round()
        .max(48.0) as usize;
    if sample_count < minimum_samples || patch_count == 0 {
        return None;
    }
    let sample_count_f64 = sample_count as f64;
    let candidate_variance = candidate_squared_sum
        - (candidate_horizontal_sum * candidate_horizontal_sum
            + candidate_vertical_sum * candidate_vertical_sum)
            / sample_count_f64;
    let merged_variance = merged_squared_sum
        - (merged_horizontal_sum * merged_horizontal_sum
            + merged_vertical_sum * merged_vertical_sum)
            / sample_count_f64;
    if candidate_variance <= f64::EPSILON || merged_variance <= f64::EPSILON {
        return None;
    }
    let covariance = product_sum
        - (candidate_horizontal_sum * merged_horizontal_sum
            + candidate_vertical_sum * merged_vertical_sum)
            / sample_count_f64;
    let score = covariance / (candidate_variance * merged_variance).sqrt();
    let candidate_energy = (candidate_variance / sample_count_f64).sqrt();
    let merged_energy = (merged_variance / sample_count_f64).sqrt();
    let foreground_fraction = foreground_count as f64 / patch_count as f64;
    (score.is_finite() && candidate_energy.is_finite() && merged_energy.is_finite()).then_some((
        score,
        candidate_energy.min(merged_energy),
        foreground_fraction,
    ))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_focus_full_resolution_shift(
    candidate: &RenderedFocusLayer,
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
    center_x: i32,
    center_y: i32,
    delta_x: i32,
    delta_y: i32,
    radius: i32,
    best: &mut Option<(f64, f64, i32, i32)>,
    second_best: &mut f64,
    current_score: &mut f64,
) {
    let Some((score, energy, foreground_fraction)) = focus_full_resolution_patch_score(
        candidate,
        merged,
        merged_mask,
        center_x,
        center_y,
        delta_x,
        delta_y,
        radius,
    ) else {
        return;
    };
    if foreground_fraction > FOCUS_FULL_RES_ALIGNMENT_MAX_FOREGROUND_FRACTION
        || energy < FOCUS_FULL_RES_ALIGNMENT_MIN_ENERGY
        || score < FOCUS_FULL_RES_ALIGNMENT_MIN_NCC
    {
        return;
    }
    if delta_x == 0 && delta_y == 0 {
        *current_score = score;
    }
    let replaces_best = best.as_ref().is_none_or(|(best_score, _, best_x, best_y)| {
        score > *best_score
            || (score == *best_score
                && (delta_x.abs() + delta_y.abs() < best_x.abs() + best_y.abs()))
    });
    if replaces_best {
        if let Some((best_score, _, _, _)) = *best {
            *second_best = (*second_best).max(best_score);
        }
        *best = Some((score, energy, delta_x, delta_y));
    } else {
        *second_best = (*second_best).max(score);
    }
}

fn estimate_focus_layer_translation(
    candidate: &RenderedFocusLayer,
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
) -> Option<(i32, i32, f64, usize, usize)> {
    let (candidate_width, candidate_height) = candidate.image.dimensions();
    let (merged_width, merged_height) = merged.dimensions();
    if candidate_width < 2 * (FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS + 2) as u32
        || candidate_height < 2 * (FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS + 2) as u32
        || merged_width < 2
        || merged_height < 2
        || merged_mask.dimensions() != merged.dimensions()
    {
        return None;
    }

    let patch_radius = FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS;
    let overlap_left = candidate.left.max(0).min(merged_width.saturating_sub(1)) as i32;
    let overlap_top = candidate.top.max(0).min(merged_height.saturating_sub(1)) as i32;
    let overlap_right = (candidate
        .left
        .saturating_add(candidate_width)
        .saturating_sub(1))
    .min(merged_width.saturating_sub(1)) as i32;
    let overlap_bottom = (candidate
        .top
        .saturating_add(candidate_height)
        .saturating_sub(1))
    .min(merged_height.saturating_sub(1)) as i32;
    if overlap_right - overlap_left < patch_radius * 2 + 2
        || overlap_bottom - overlap_top < patch_radius * 2 + 2
    {
        return None;
    }

    let grid_size = FOCUS_FULL_RES_ALIGNMENT_GRID_SIZE;
    let mut patch_centers = Vec::with_capacity((grid_size * grid_size) as usize);
    for grid_y in 0..grid_size {
        let global_y =
            overlap_top + ((overlap_bottom - overlap_top) * (grid_y + 1) / (grid_size + 1));
        for grid_x in 0..grid_size {
            let global_x =
                overlap_left + ((overlap_right - overlap_left) * (grid_x + 1) / (grid_size + 1));
            let local_x = global_x - candidate.left as i32;
            let local_y = global_y - candidate.top as i32;
            patch_centers.push((local_x, local_y));
        }
    }

    let mut patch_deltas = Vec::new();
    let mut score_sum = 0.0;
    for (center_x, center_y) in patch_centers.iter().copied() {
        let mut best: Option<(f64, f64, i32, i32)> = None;
        let mut second_best = f64::NEG_INFINITY;
        let mut current_score = f64::NEG_INFINITY;
        // A coarse pass limits the expensive full-resolution correlation to a
        // small neighbourhood around its best integer displacement. The fine
        // pass still checks every pixel shift around that candidate, so a
        // one-pixel residual is not rounded away.
        let coarse_step = 4;
        for delta_y in (-FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT..=FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT)
            .step_by(coarse_step as usize)
        {
            for delta_x in (-FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT
                ..=FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT)
                .step_by(coarse_step as usize)
            {
                evaluate_focus_full_resolution_shift(
                    candidate,
                    merged,
                    merged_mask,
                    center_x,
                    center_y,
                    delta_x,
                    delta_y,
                    patch_radius,
                    &mut best,
                    &mut second_best,
                    &mut current_score,
                );
            }
        }
        let Some((_, _, coarse_x, coarse_y)) = best else {
            continue;
        };
        let fine_min_x = (coarse_x - coarse_step + 1).max(-FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT);
        let fine_max_x = (coarse_x + coarse_step - 1).min(FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT);
        let fine_min_y = (coarse_y - coarse_step + 1).max(-FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT);
        let fine_max_y = (coarse_y + coarse_step - 1).min(FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT);
        for delta_y in fine_min_y..=fine_max_y {
            for delta_x in fine_min_x..=fine_max_x {
                if delta_x == coarse_x && delta_y == coarse_y {
                    continue;
                }
                evaluate_focus_full_resolution_shift(
                    candidate,
                    merged,
                    merged_mask,
                    center_x,
                    center_y,
                    delta_x,
                    delta_y,
                    patch_radius,
                    &mut best,
                    &mut second_best,
                    &mut current_score,
                );
            }
        }
        let Some((best_score, _, best_x, best_y)) = best else {
            continue;
        };
        let margin = best_score - second_best.max(current_score);
        if best_score < FOCUS_FULL_RES_ALIGNMENT_MIN_NCC
            || !margin.is_finite()
            || margin < FOCUS_FULL_RES_ALIGNMENT_MIN_MARGIN
            || (best_x != 0 || best_y != 0)
                && (!current_score.is_finite()
                    || best_score - current_score < FOCUS_FULL_RES_ALIGNMENT_MIN_MARGIN)
        {
            continue;
        }
        patch_deltas.push((best_x, best_y));
        score_sum += best_score;
    }
    if patch_deltas.is_empty() {
        return None;
    }

    let mut x_values = patch_deltas.iter().map(|(x, _)| *x).collect::<Vec<_>>();
    let mut y_values = patch_deltas.iter().map(|(_, y)| *y).collect::<Vec<_>>();
    x_values.sort_unstable();
    y_values.sort_unstable();
    let median_x = x_values[x_values.len() / 2];
    let median_y = y_values[y_values.len() / 2];
    let support = patch_deltas
        .iter()
        .filter(|(x, y)| (x - median_x).abs() <= 1 && (y - median_y).abs() <= 1)
        .count();
    let minimum_support = FOCUS_FULL_RES_ALIGNMENT_MIN_PATCHES;
    if support < minimum_support
        || (support as f64) < patch_deltas.len() as f64 * 0.5
        || (median_x == 0 && median_y == 0)
    {
        return None;
    }
    Some((
        median_x,
        median_y,
        score_sum / patch_deltas.len() as f64,
        support,
        patch_deltas.len(),
    ))
}

fn translate_rendered_focus_layer(
    candidate: RenderedFocusLayer,
    delta_x: i32,
    delta_y: i32,
    out_width: u32,
    out_height: u32,
) -> RenderedFocusLayer {
    if delta_x == 0 && delta_y == 0 {
        return candidate;
    }
    let (width, height) = candidate.image.dimensions();
    if width == 0 || height == 0 || out_width == 0 || out_height == 0 {
        return candidate;
    }
    let old_left = candidate.left as i64;
    let old_top = candidate.top as i64;
    let old_right = old_left + width as i64 - 1;
    let old_bottom = old_top + height as i64 - 1;
    let new_left = (old_left + i64::from(delta_x)).clamp(0, i64::from(out_width - 1));
    let new_top = (old_top + i64::from(delta_y)).clamp(0, i64::from(out_height - 1));
    let new_right = (old_right + i64::from(delta_x)).clamp(0, i64::from(out_width - 1));
    let new_bottom = (old_bottom + i64::from(delta_y)).clamp(0, i64::from(out_height - 1));
    if new_left > new_right || new_top > new_bottom {
        return RenderedFocusLayer {
            image: Rgb32FImage::new(0, 0),
            mask: GrayImage::new(0, 0),
            foreground_mask: GrayImage::new(0, 0),
            relaxed_foreground_mask: GrayImage::new(0, 0),
            left: 0,
            top: 0,
        };
    }
    let new_width = (new_right - new_left + 1) as u32;
    let new_height = (new_bottom - new_top + 1) as u32;
    let mut image_pixels = vec![0.0f32; new_width as usize * new_height as usize * 3];
    let mut mask_pixels = vec![0u8; new_width as usize * new_height as usize];
    let mut foreground_pixels = vec![0u8; new_width as usize * new_height as usize];
    let mut relaxed_pixels = vec![0u8; new_width as usize * new_height as usize];
    let source_image = candidate.image.as_raw();
    let source_mask = candidate.mask.as_raw();
    let source_foreground = candidate.foreground_mask.as_raw();
    let source_relaxed = candidate.relaxed_foreground_mask.as_raw();
    for source_y in 0..height as i64 {
        for source_x in 0..width as i64 {
            let destination_x = old_left + source_x + i64::from(delta_x);
            let destination_y = old_top + source_y + i64::from(delta_y);
            if destination_x < new_left
                || destination_x > new_right
                || destination_y < new_top
                || destination_y > new_bottom
            {
                continue;
            }
            let source_index = source_y as usize * width as usize + source_x as usize;
            let destination_index = (destination_y - new_top) as usize * new_width as usize
                + (destination_x - new_left) as usize;
            let source_start = source_index * 3;
            let destination_start = destination_index * 3;
            image_pixels[destination_start..destination_start + 3]
                .copy_from_slice(&source_image[source_start..source_start + 3]);
            mask_pixels[destination_index] = source_mask[source_index];
            foreground_pixels[destination_index] = source_foreground[source_index];
            relaxed_pixels[destination_index] = source_relaxed[source_index];
        }
    }
    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(new_width, new_height, image_pixels)
            .expect("translated focus layer image dimensions must match"),
        mask: GrayImage::from_raw(new_width, new_height, mask_pixels)
            .expect("translated focus layer mask dimensions must match"),
        foreground_mask: GrayImage::from_raw(new_width, new_height, foreground_pixels)
            .expect("translated focus foreground mask dimensions must match"),
        relaxed_foreground_mask: GrayImage::from_raw(new_width, new_height, relaxed_pixels)
            .expect("translated focus relaxed foreground mask dimensions must match"),
        left: new_left as u32,
        top: new_top as u32,
    }
}

fn align_focus_foreground_layer_to_existing(
    candidate: RenderedFocusLayer,
    merged_foreground_mask: &GrayImage,
) -> RenderedFocusLayer {
    let (layer_width, layer_height) = candidate.foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 || merged_foreground_mask.width() == 0 {
        return candidate;
    }

    let foreground = candidate.foreground_mask.as_raw();
    let foreground_count = foreground.iter().filter(|value| **value > 0).count();
    let existing_count = merged_foreground_mask
        .as_raw()
        .iter()
        .filter(|value| **value > 0)
        .count();
    if foreground_count < 32 || existing_count < 32 {
        return candidate;
    }

    // Use a bounded, deterministic set of mask anchors. This makes the
    // refinement cheap even for a full 60MP source while retaining enough
    // support to distinguish a real overlap from an accidental nearby edge.
    let anchor_stride = (foreground_count / 2_048).max(1);
    let mut anchors = Vec::with_capacity(foreground_count.min(2_048));
    let mut foreground_index = 0usize;
    for (index, value) in foreground.iter().enumerate() {
        if *value == 0 {
            continue;
        }
        if foreground_index.is_multiple_of(anchor_stride) {
            let local_x = (index % layer_width as usize) as u32;
            let local_y = (index / layer_width as usize) as u32;
            anchors.push((candidate.left + local_x, candidate.top + local_y));
        }
        foreground_index += 1;
    }
    if anchors.len() < 32 {
        return candidate;
    }

    let base_width = merged_foreground_mask.width() as i32;
    let base_height = merged_foreground_mask.height() as i32;
    let base_pixels = merged_foreground_mask.as_raw();
    let search_radius = ((layer_width.max(layer_height) as f32) * 0.01)
        .round()
        .clamp(4.0, FOCUS_FOREGROUND_REFINEMENT_MAX_SHIFT as f32) as i32;
    let coarse_step = (search_radius / 24).max(1);
    let hit_count = |delta_x: i32, delta_y: i32| -> usize {
        anchors
            .iter()
            .filter(|&&(x, y)| {
                let x = x as i32 + delta_x;
                let y = y as i32 + delta_y;
                x >= 0
                    && y >= 0
                    && x < base_width
                    && y < base_height
                    && base_pixels[y as usize * base_width as usize + x as usize] > 0
            })
            .count()
    };

    let current_hits = hit_count(0, 0);
    let mut best_delta = (0i32, 0i32);
    let mut best_hits = current_hits;
    for delta_y in (-search_radius..=search_radius).step_by(coarse_step as usize) {
        for delta_x in (-search_radius..=search_radius).step_by(coarse_step as usize) {
            let hits = hit_count(delta_x, delta_y);
            if hits > best_hits {
                best_hits = hits;
                best_delta = (delta_x, delta_y);
            }
        }
    }
    // Refine around the best coarse translation at pixel precision. The
    // foreground mask is only a registration cue; the final source pixels are
    // still selected by the focus score below.
    if coarse_step > 1 {
        let (coarse_x, coarse_y) = best_delta;
        let fine_min_y = (coarse_y - coarse_step + 1).max(-search_radius);
        let fine_max_y = (coarse_y + coarse_step - 1).min(search_radius);
        let fine_min_x = (coarse_x - coarse_step + 1).max(-search_radius);
        let fine_max_x = (coarse_x + coarse_step - 1).min(search_radius);
        for delta_y in fine_min_y..=fine_max_y {
            for delta_x in fine_min_x..=fine_max_x {
                let hits = hit_count(delta_x, delta_y);
                if hits > best_hits {
                    best_hits = hits;
                    best_delta = (delta_x, delta_y);
                }
            }
        }
    }

    let minimum_gain = (anchors.len() / 20).max(12);
    if best_delta == (0, 0)
        || best_hits < current_hits.saturating_add(minimum_gain)
        || best_hits < anchors.len() / 20
    {
        return candidate;
    }

    let (delta_x, delta_y) = best_delta;
    let mut image_pixels = candidate.image.into_raw();
    let mut candidate_mask = candidate.mask.into_raw();
    let mut foreground_mask = vec![0u8; foreground.len()];
    let relaxed_foreground = candidate.relaxed_foreground_mask.as_raw();
    let mut relaxed_foreground_mask = vec![0u8; foreground.len()];
    let mut moved_pixels = Vec::with_capacity(foreground_count);
    for (index, value) in foreground.iter().enumerate() {
        if *value == 0 {
            continue;
        }
        let local_x = (index % layer_width as usize) as i32;
        let local_y = (index / layer_width as usize) as i32;
        let destination_x = local_x + delta_x;
        let destination_y = local_y + delta_y;
        if destination_x < 0
            || destination_y < 0
            || destination_x >= layer_width as i32
            || destination_y >= layer_height as i32
        {
            continue;
        }
        let destination_index =
            destination_y as usize * layer_width as usize + destination_x as usize;
        let source_start = index * 3;
        moved_pixels.push((
            destination_index,
            [
                image_pixels[source_start],
                image_pixels[source_start + 1],
                image_pixels[source_start + 2],
            ],
        ));
        candidate_mask[index] = 0;
        foreground_mask[destination_index] = 255;
        if relaxed_foreground[index] > 0 {
            relaxed_foreground_mask[destination_index] = 255;
        }
    }
    for (destination_index, pixel) in moved_pixels {
        let destination_start = destination_index * 3;
        image_pixels[destination_start..destination_start + 3].copy_from_slice(&pixel);
        candidate_mask[destination_index] = 255;
    }

    println!(
        "  - Focus foreground mask refinement: delta=({delta_x},{delta_y}), overlap={current_hits}->{best_hits} anchors={}",
        anchors.len()
    );
    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(layer_width, layer_height, image_pixels)
            .expect("aligned focus layer image dimensions must match"),
        mask: GrayImage::from_raw(layer_width, layer_height, candidate_mask)
            .expect("aligned focus layer mask dimensions must match"),
        foreground_mask: GrayImage::from_raw(layer_width, layer_height, foreground_mask)
            .expect("aligned focus foreground mask dimensions must match"),
        relaxed_foreground_mask: GrayImage::from_raw(
            layer_width,
            layer_height,
            relaxed_foreground_mask,
        )
        .expect("aligned focus relaxed foreground mask dimensions must match"),
        left: candidate.left,
        top: candidate.top,
    }
}

fn suppress_focus_foreground_switches(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
) {
    suppress_focus_foreground_switches_in_region(
        decision_mask,
        merged_foreground_mask,
        candidate.left,
        candidate.top,
        &candidate.foreground_mask,
        &candidate.relaxed_foreground_mask,
    );
}

fn suppress_focus_foreground_switches_in_region(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    candidate_foreground_mask: &GrayImage,
    relaxed_foreground_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate_foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = merged_foreground_mask.width();
    let base_height = merged_foreground_mask.height();
    if candidate_left >= base_width
        || candidate_top >= base_height
        || candidate_left + layer_width > base_width
        || candidate_top + layer_height > base_height
    {
        return;
    }
    debug_assert_eq!(decision_mask.dimensions(), (layer_width, layer_height));

    let overlap_mask = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        candidate_left,
        candidate_top,
        (layer_width, layer_height),
    );
    if relaxed_foreground_mask.dimensions() != (layer_width, layer_height) {
        return;
    }

    decision_mask
        .as_mut()
        .par_chunks_mut(layer_width as usize)
        .enumerate()
        .for_each(|(local_y, decision_row)| {
            let foreground_row = &candidate_foreground_mask.as_raw()
                [local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
            let overlap_row =
                &overlap_mask[local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
            let relaxed_row = &relaxed_foreground_mask.as_raw()
                [local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
            for (local_x, decision) in decision_row.iter_mut().enumerate() {
                if foreground_row[local_x] > 0
                    && relaxed_row[local_x] == 0
                    && overlap_row[local_x] > 0
                {
                    *decision = 0;
                }
            }
        });
}

fn suppress_focus_foreground_switches_on_canvas(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_foreground_mask: &GrayImage,
    relaxed_foreground_mask: &GrayImage,
) {
    if decision_mask.dimensions() != merged_foreground_mask.dimensions()
        || decision_mask.dimensions() != candidate_foreground_mask.dimensions()
        || candidate_foreground_mask.dimensions() != relaxed_foreground_mask.dimensions()
    {
        return;
    }
    let overlap_mask = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        0,
        0,
        candidate_foreground_mask.dimensions(),
    );
    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(candidate_foreground_mask.as_raw().par_iter())
        .zip(overlap_mask.par_iter())
        .zip(relaxed_foreground_mask.as_raw().par_iter())
        .for_each(|(((decision, candidate_foreground), overlap), relaxed)| {
            if *candidate_foreground > 0 && *relaxed == 0 && *overlap > 0 {
                *decision = 0;
            }
        });
}

fn suppress_focus_foreground_ownership_switches(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    relaxed_foreground_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate_mask.dimensions();
    if layer_width == 0
        || layer_height == 0
        || decision_mask.dimensions() != (layer_width, layer_height)
        || relaxed_foreground_mask.dimensions() != (layer_width, layer_height)
    {
        return;
    }
    let overlap_mask = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        candidate_left,
        candidate_top,
        (layer_width, layer_height),
    );
    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(candidate_mask.as_raw().par_iter())
        .zip(overlap_mask.par_iter())
        .zip(relaxed_foreground_mask.as_raw().par_iter())
        .for_each(|(((decision, candidate_valid), overlap), relaxed)| {
            if *candidate_valid > 0 && *relaxed == 0 && *overlap > 0 {
                *decision = 0;
            }
        });
}

fn suppress_focus_foreground_ownership_switches_on_canvas(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_mask: &GrayImage,
    relaxed_foreground_mask: &GrayImage,
) {
    if decision_mask.dimensions() != merged_foreground_mask.dimensions()
        || decision_mask.dimensions() != candidate_mask.dimensions()
        || candidate_mask.dimensions() != relaxed_foreground_mask.dimensions()
    {
        return;
    }
    let overlap_mask = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        0,
        0,
        candidate_mask.dimensions(),
    );
    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(candidate_mask.as_raw().par_iter())
        .zip(overlap_mask.par_iter())
        .zip(relaxed_foreground_mask.as_raw().par_iter())
        .for_each(|(((decision, candidate_valid), overlap), relaxed)| {
            if *candidate_valid > 0 && *relaxed == 0 && *overlap > 0 {
                *decision = 0;
            }
        });
}

fn focus_foreground_overlap_mask_for_region(
    merged_foreground_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    candidate_dimensions: (u32, u32),
) -> Vec<u8> {
    let (layer_width, layer_height) = candidate_dimensions;
    let pixel_count = layer_width as usize * layer_height as usize;
    let mut overlap = vec![0u8; pixel_count];
    if layer_width == 0 || layer_height == 0 {
        return overlap;
    }

    let radius = ((layer_width.max(layer_height) as f32) * FOCUS_FOREGROUND_OWNERSHIP_RADIUS_RATIO)
        .round()
        .clamp(8.0, 256.0) as usize;
    let base_width = merged_foreground_mask.width() as usize;
    let base_height = merged_foreground_mask.height() as usize;
    let left = candidate_left as usize;
    let top = candidate_top as usize;
    let width = layer_width as usize;
    let height = layer_height as usize;
    if left >= base_width
        || top >= base_height
        || left + width > base_width
        || top + height > base_height
    {
        return overlap;
    }

    // First apply a horizontal max filter to the already selected foreground
    // mask. This is equivalent to testing the whole rectangular neighbourhood,
    // but is linear in the candidate size and does not allocate a full canvas.
    let mut horizontal = vec![0u8; pixel_count];
    let base_pixels = merged_foreground_mask.as_raw();
    for local_y in 0..height {
        let base_row = &base_pixels[(top + local_y) * base_width..(top + local_y + 1) * base_width];
        let initial_start = left.saturating_sub(radius);
        let initial_end = (left + radius).min(base_width - 1);
        let mut window = base_row[initial_start..=initial_end]
            .iter()
            .filter(|value| **value > 0)
            .count() as u32;
        for local_x in 0..width {
            if local_x > 0 {
                let center_x = left + local_x;
                let add_x = (center_x + radius).min(base_width - 1);
                window += u32::from(base_row[add_x] > 0);
                if center_x > radius {
                    let remove_x = center_x - radius - 1;
                    window -= u32::from(base_row[remove_x] > 0);
                }
            }
            horizontal[local_y * width + local_x] = u8::from(window > 0);
        }
    }

    // Then apply the same max filter vertically. A later frame only loses a
    // foreground switch when it really overlaps this expanded ownership area;
    // unrelated artwork remains governed by the normal focus score.
    for local_x in 0..width {
        let initial_end = radius.min(height - 1);
        let mut window = (0..=initial_end)
            .map(|row| u32::from(horizontal[row * width + local_x] > 0))
            .sum::<u32>();
        for local_y in 0..height {
            if local_y > 0 {
                let add_y = (local_y + radius).min(height - 1);
                window += u32::from(horizontal[add_y * width + local_x] > 0);
                if local_y > radius {
                    let remove_y = local_y.saturating_sub(radius + 1);
                    window -= u32::from(horizontal[remove_y * width + local_x] > 0);
                }
            }
            overlap[local_y * width + local_x] = u8::from(window > 0);
        }
    }
    overlap
}

fn focus_foreground_edge_mask(mask: &[u8], width: u32, height: u32, radius: usize) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    if width == 0 || height == 0 || mask.len() != width * height {
        return Vec::new();
    }
    if radius == 0 {
        return mask.to_vec();
    }

    // Erode the protected foreground with separable prefix-sum windows. The
    // pixels left outside that erosion are the silhouette band; the interior
    // can participate in low-frequency exposure blending without averaging a
    // displaced edge.
    let mut horizontal = vec![0u8; mask.len()];
    for y in 0..height {
        let row_start = y * width;
        let mut prefix = vec![0u32; width + 1];
        for x in 0..width {
            prefix[x + 1] = prefix[x] + u32::from(mask[row_start + x] > 0);
        }
        for x in 0..width {
            let start = x.saturating_sub(radius);
            let end = (x + radius).min(width - 1);
            let window_length = (end - start + 1) as u32;
            if prefix[end + 1] - prefix[start] == window_length {
                horizontal[row_start + x] = 255;
            }
        }
    }

    let mut eroded = vec![0u8; mask.len()];
    for x in 0..width {
        let mut prefix = vec![0u32; height + 1];
        for y in 0..height {
            prefix[y + 1] = prefix[y] + u32::from(horizontal[y * width + x] > 0);
        }
        for y in 0..height {
            let start = y.saturating_sub(radius);
            let end = (y + radius).min(height - 1);
            let window_length = (end - start + 1) as u32;
            if prefix[end + 1] - prefix[start] == window_length {
                eroded[y * width + x] = 255;
            }
        }
    }

    mask.iter()
        .zip(eroded)
        .map(|(value, interior)| u8::from(*value > 0 && interior == 0))
        .collect()
}

fn mark_focus_foreground_pixels(
    merged_foreground_mask: &mut GrayImage,
    merged_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
) {
    mark_focus_foreground_pixels_in_region(
        merged_foreground_mask,
        merged_mask,
        candidate.left,
        candidate.top,
        &candidate.foreground_mask,
        decision_mask,
    );
}

fn mark_focus_foreground_pixels_in_region(
    merged_foreground_mask: &mut GrayImage,
    merged_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    candidate_foreground_mask: &GrayImage,
    decision_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate_foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = merged_foreground_mask.width();
    let base_height = merged_foreground_mask.height();
    if candidate_left >= base_width
        || candidate_top >= base_height
        || candidate_left + layer_width > base_width
        || candidate_top + layer_height > base_height
    {
        return;
    }
    debug_assert_eq!(
        merged_mask.dimensions(),
        merged_foreground_mask.dimensions()
    );
    debug_assert_eq!(decision_mask.dimensions(), (layer_width, layer_height));

    merged_foreground_mask
        .as_mut()
        .par_chunks_mut(base_width as usize)
        .skip(candidate_top as usize)
        .take(layer_height as usize)
        .zip(
            merged_mask
                .as_raw()
                .par_chunks(base_width as usize)
                .skip(candidate_top as usize),
        )
        .zip(
            candidate_foreground_mask
                .as_raw()
                .par_chunks(layer_width as usize),
        )
        .zip(decision_mask.as_raw().par_chunks(layer_width as usize))
        .for_each(
            |(((foreground_row, merged_mask_row), candidate_foreground_row), decision_row)| {
                for local_x in 0..layer_width as usize {
                    if candidate_foreground_row[local_x] > 0
                        && (merged_mask_row[candidate_left as usize + local_x] == 0
                            || decision_row[local_x] > 0)
                    {
                        foreground_row[candidate_left as usize + local_x] = 255;
                    }
                }
            },
        );
}

fn hard_select_focus_layer(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    debug_assert_eq!(candidate.mask.dimensions(), (layer_width, layer_height));
    debug_assert_eq!(decision_mask.dimensions(), (layer_width, layer_height));
    let base_width = base.width();
    let base_height = base.height();
    if candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }

    let base_stride = base_width as usize * 3;
    let layer_stride = layer_width as usize * 3;
    base.as_mut()
        .par_chunks_mut(base_stride)
        .zip(base_mask.as_mut().par_chunks_mut(base_width as usize))
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(candidate.image.as_raw().par_chunks(layer_stride))
        .zip(candidate.mask.as_raw().par_chunks(layer_width as usize))
        .zip(decision_mask.as_raw().par_chunks(layer_width as usize))
        .for_each(
            |((((base_row, base_mask_row), candidate_row), candidate_mask_row), decision_row)| {
                for local_x in 0..layer_width as usize {
                    if candidate_mask_row[local_x] == 0 {
                        continue;
                    }
                    let global_x = candidate.left as usize + local_x;
                    let base_pixel_start = global_x * 3;
                    let candidate_pixel_start = local_x * 3;
                    if base_mask_row[global_x] == 0 || decision_row[local_x] > 0 {
                        base_row[base_pixel_start..base_pixel_start + 3].copy_from_slice(
                            &candidate_row[candidate_pixel_start..candidate_pixel_start + 3],
                        );
                    }
                    base_mask_row[global_x] = 255;
                }
            },
        );
}

fn mark_focus_owner_pixels(
    owner: &mut GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
    owner_id: u8,
) {
    let (layer_width, layer_height) = candidate.mask.dimensions();
    if layer_width == 0
        || layer_height == 0
        || decision_mask.dimensions() != (layer_width, layer_height)
        || candidate.left + layer_width > owner.width()
        || candidate.top + layer_height > owner.height()
    {
        return;
    }
    let owner_width = owner.width() as usize;
    owner
        .as_mut()
        .par_chunks_mut(owner_width)
        .skip(candidate.top as usize)
        .take(layer_height as usize)
        .zip(candidate.mask.as_raw().par_chunks(layer_width as usize))
        .zip(decision_mask.as_raw().par_chunks(layer_width as usize))
        .for_each(|((owner_row, candidate_row), decision_row)| {
            let start = candidate.left as usize;
            for local_x in 0..layer_width as usize {
                if candidate_row[local_x] > 0 && decision_row[local_x] > 0 {
                    owner_row[start + local_x] = owner_id;
                }
            }
        });
}

fn scaled_region_start(value: u32, source_size: u32, target_size: u32) -> u32 {
    ((value as u64 * target_size as u64) / source_size.max(1) as u64) as u32
}

fn scaled_region_end(value: u32, source_size: u32, target_size: u32) -> u32 {
    (value as u64 * target_size as u64).div_ceil(source_size.max(1) as u64) as u32
}

fn render_focus_analysis_layer(
    layer: &RenderedFocusLayer,
    out_width: u32,
    out_height: u32,
    analysis_width: u32,
    analysis_height: u32,
) -> RenderedFocusAnalysisLayer {
    let mut analysis = Rgb32FImage::new(analysis_width, analysis_height);
    let mut analysis_mask = GrayImage::new(analysis_width, analysis_height);
    let mut analysis_foreground_mask = GrayImage::new(analysis_width, analysis_height);
    let (layer_width, layer_height) = layer.image.dimensions();
    if layer_width == 0 || layer_height == 0 || out_width == 0 || out_height == 0 {
        return RenderedFocusAnalysisLayer {
            image: analysis,
            mask: analysis_mask,
            foreground_mask: analysis_foreground_mask,
            relaxed_foreground_mask: GrayImage::new(analysis_width, analysis_height),
        };
    }

    let left = scaled_region_start(layer.left, out_width, analysis_width).min(analysis_width);
    let top = scaled_region_start(layer.top, out_height, analysis_height).min(analysis_height);
    let right = scaled_region_end(
        layer.left.saturating_add(layer_width),
        out_width,
        analysis_width,
    )
    .min(analysis_width);
    let bottom = scaled_region_end(
        layer.top.saturating_add(layer_height),
        out_height,
        analysis_height,
    )
    .min(analysis_height);
    if left >= right || top >= bottom {
        return RenderedFocusAnalysisLayer {
            image: analysis,
            mask: analysis_mask,
            foreground_mask: analysis_foreground_mask,
            relaxed_foreground_mask: GrayImage::new(analysis_width, analysis_height),
        };
    }

    let resized_width = right - left;
    let resized_height = bottom - top;
    let resized = resize_rgb(&layer.image, resized_width, resized_height);
    let resized_mask = resize_binary_mask(&layer.mask, resized_width, resized_height);
    let resized_foreground_mask =
        resize_binary_mask(&layer.foreground_mask, resized_width, resized_height);
    let resized_relaxed_foreground_mask = resize_binary_mask(
        &layer.relaxed_foreground_mask,
        resized_width,
        resized_height,
    );
    let analysis_stride = analysis_width as usize * 3;
    let resized_stride = resized_width as usize * 3;
    analysis
        .as_mut()
        .par_chunks_mut(analysis_stride)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(resized.as_raw().par_chunks(resized_stride))
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize * 3;
            let end = start + resized_stride;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    analysis_mask
        .as_mut()
        .par_chunks_mut(analysis_width as usize)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(resized_mask.as_raw().par_chunks(resized_width as usize))
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize;
            let end = start + resized_width as usize;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    analysis_foreground_mask
        .as_mut()
        .par_chunks_mut(analysis_width as usize)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(
            resized_foreground_mask
                .as_raw()
                .par_chunks(resized_width as usize),
        )
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize;
            let end = start + resized_width as usize;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    let analysis_relaxed_stride = analysis_width as usize;
    let mut relaxed_foreground_mask = GrayImage::new(analysis_width, analysis_height);
    relaxed_foreground_mask
        .as_mut()
        .par_chunks_mut(analysis_relaxed_stride)
        .skip(top as usize)
        .take(resized_height as usize)
        .zip(
            resized_relaxed_foreground_mask
                .as_raw()
                .par_chunks(resized_width as usize),
        )
        .for_each(|(analysis_row, resized_row)| {
            let start = left as usize;
            let end = start + resized_width as usize;
            analysis_row[start..end].copy_from_slice(resized_row);
        });
    RenderedFocusAnalysisLayer {
        image: analysis,
        mask: analysis_mask,
        foreground_mask: analysis_foreground_mask,
        relaxed_foreground_mask,
    }
}

fn focus_decision_for_layer(
    analysis_decision: &GrayImage,
    layer: &RenderedFocusLayer,
    out_width: u32,
    out_height: u32,
) -> GrayImage {
    let (width, height) = layer.image.dimensions();
    let analysis_width = analysis_decision.width();
    let analysis_height = analysis_decision.height();
    GrayImage::from_fn(width, height, |x, y| {
        if out_width == 0 || out_height == 0 || analysis_width == 0 || analysis_height == 0 {
            return image::Luma([0]);
        }
        let global_x = layer.left.saturating_add(x);
        let global_y = layer.top.saturating_add(y);
        let analysis_x = ((global_x as u64 * analysis_width as u64) / out_width as u64)
            .min(analysis_width.saturating_sub(1) as u64) as u32;
        let analysis_y = ((global_y as u64 * analysis_height as u64) / out_height as u64)
            .min(analysis_height.saturating_sub(1) as u64) as u32;
        *analysis_decision.get_pixel(analysis_x, analysis_y)
    })
}

fn commit_focus_analysis_scores(
    best_focus: &mut [f32],
    candidate_focus: &[f32],
    candidate_analysis_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    full_resolution_decision: &GrayImage,
    out_width: u32,
    out_height: u32,
    analysis_width: u32,
    analysis_height: u32,
) {
    if out_width == 0
        || out_height == 0
        || analysis_width == 0
        || analysis_height == 0
        || candidate_analysis_mask.dimensions() != (analysis_width, analysis_height)
        || candidate_focus.len() != analysis_width as usize * analysis_height as usize
        || best_focus.len() != candidate_focus.len()
        || full_resolution_decision.dimensions() != candidate.image.dimensions()
    {
        return;
    }

    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0
        || layer_height == 0
        || candidate.left >= out_width
        || candidate.top >= out_height
        || candidate.left.saturating_add(layer_width) > out_width
        || candidate.top.saturating_add(layer_height) > out_height
    {
        return;
    }

    // The full-resolution ownership guards can veto part of an analysis cell.
    // Downsample the committed decision, rather than blindly copying the
    // low-resolution decision, so the score table remains tied to pixels that
    // really entered the merged image.
    let left = scaled_region_start(candidate.left, out_width, analysis_width).min(analysis_width);
    let top = scaled_region_start(candidate.top, out_height, analysis_height).min(analysis_height);
    let right = scaled_region_end(
        candidate.left.saturating_add(layer_width),
        out_width,
        analysis_width,
    )
    .min(analysis_width);
    let bottom = scaled_region_end(
        candidate.top.saturating_add(layer_height),
        out_height,
        analysis_height,
    )
    .min(analysis_height);
    if left >= right || top >= bottom {
        return;
    }
    let committed_width = right - left;
    let committed_height = bottom - top;
    let committed = resize_binary_mask(full_resolution_decision, committed_width, committed_height);
    let candidate_mask = candidate_analysis_mask.as_raw();
    for local_y in 0..committed_height as usize {
        for local_x in 0..committed_width as usize {
            let analysis_x = left as usize + local_x;
            let analysis_y = top as usize + local_y;
            let analysis_index = analysis_y * analysis_width as usize + analysis_x;
            if committed.as_raw()[local_y * committed_width as usize + local_x] > 0
                && candidate_mask[analysis_index] > 0
            {
                best_focus[analysis_index] = candidate_focus[analysis_index];
            }
        }
    }
}

fn resize_rgb(image: &Rgb32FImage, width: u32, height: u32) -> Rgb32FImage {
    image::imageops::resize(
        image,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Triangle,
    )
}

pub(super) fn downsample_rgb_half(image: &Rgb32FImage) -> Rgb32FImage {
    let (source_width, source_height) = image.dimensions();
    let target_width = source_width.div_ceil(2).max(1);
    let target_height = source_height.div_ceil(2).max(1);
    if (source_width, source_height) == (target_width, target_height) {
        return image.clone();
    }

    let source = image.as_raw();
    let source_stride = source_width as usize * 3;
    let target_stride = target_width as usize * 3;
    let mut output = vec![0.0f32; target_stride * target_height as usize];
    output
        .par_chunks_mut(target_stride)
        .enumerate()
        .for_each(|(target_y, row)| {
            let source_y0 = (target_y * 2).min(source_height as usize - 1);
            let source_y1 = (source_y0 + 1).min(source_height as usize - 1);
            for target_x in 0..target_width as usize {
                let source_x0 = (target_x * 2).min(source_width as usize - 1);
                let source_x1 = (source_x0 + 1).min(source_width as usize - 1);
                let top_left = source_y0 * source_stride + source_x0 * 3;
                let top_right = source_y0 * source_stride + source_x1 * 3;
                let bottom_left = source_y1 * source_stride + source_x0 * 3;
                let bottom_right = source_y1 * source_stride + source_x1 * 3;
                let output_start = target_x * 3;
                for channel in 0..3 {
                    row[output_start + channel] = (source[top_left + channel]
                        + source[top_right + channel]
                        + source[bottom_left + channel]
                        + source[bottom_right + channel])
                        * 0.25;
                }
            }
        });
    Rgb32FImage::from_raw(target_width, target_height, output)
        .expect("half-resolution RGB buffer dimensions must match")
}

fn downsample_mask_half(mask: &GrayImage) -> GrayImage {
    let (source_width, source_height) = mask.dimensions();
    let target_width = source_width.div_ceil(2).max(1);
    let target_height = source_height.div_ceil(2).max(1);
    if (source_width, source_height) == (target_width, target_height) {
        return mask.clone();
    }

    let source = mask.as_raw();
    let source_stride = source_width as usize;
    let target_stride = target_width as usize;
    let mut output = vec![0u8; target_stride * target_height as usize];
    output
        .par_chunks_mut(target_stride)
        .enumerate()
        .for_each(|(target_y, row)| {
            let source_y0 = (target_y * 2).min(source_height as usize - 1);
            let source_y1 = (source_y0 + 1).min(source_height as usize - 1);
            for (target_x, output) in row.iter_mut().enumerate() {
                let source_x0 = (target_x * 2).min(source_width as usize - 1);
                let source_x1 = (source_x0 + 1).min(source_width as usize - 1);
                let total = u16::from(source[source_y0 * source_stride + source_x0])
                    + u16::from(source[source_y0 * source_stride + source_x1])
                    + u16::from(source[source_y1 * source_stride + source_x0])
                    + u16::from(source[source_y1 * source_stride + source_x1]);
                *output = ((total + 2) / 4) as u8;
            }
        });
    GrayImage::from_raw(target_width, target_height, output)
        .expect("half-resolution mask buffer dimensions must match")
}

#[derive(Clone, Copy)]
struct LinearSample {
    lower: usize,
    upper: usize,
    upper_weight: f32,
}

fn linear_samples(source_length: u32, target_length: u32) -> Vec<LinearSample> {
    let scale = source_length as f64 / target_length.max(1) as f64;
    (0..target_length.max(1))
        .map(|target| {
            let source = ((f64::from(target) + 0.5) * scale - 0.5)
                .clamp(0.0, f64::from(source_length.saturating_sub(1)));
            let lower = source.floor() as usize;
            let upper = (lower + 1).min(source_length.saturating_sub(1) as usize);
            LinearSample {
                lower,
                upper,
                upper_weight: (source - lower as f64) as f32,
            }
        })
        .collect()
}

fn subtract_upsampled_rgb(fine: &Rgb32FImage, coarse: &Rgb32FImage) -> Rgb32FImage {
    let (width, height) = fine.dimensions();
    let x_samples = linear_samples(coarse.width(), width);
    let y_samples = linear_samples(coarse.height(), height);
    let coarse_stride = coarse.width() as usize * 3;
    let output_stride = width as usize * 3;
    let coarse_pixels = coarse.as_raw();
    let fine_pixels = fine.as_raw();
    let mut output = vec![0.0f32; output_stride * height as usize];

    output
        .par_chunks_mut(output_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            let y_weight = y_sample.upper_weight;
            let fine_row = &fine_pixels[y * output_stride..(y + 1) * output_stride];
            for (x, x_sample) in x_samples.iter().copied().enumerate() {
                let x_weight = x_sample.upper_weight;
                let top_left = y_sample.lower * coarse_stride + x_sample.lower * 3;
                let top_right = y_sample.lower * coarse_stride + x_sample.upper * 3;
                let bottom_left = y_sample.upper * coarse_stride + x_sample.lower * 3;
                let bottom_right = y_sample.upper * coarse_stride + x_sample.upper * 3;
                let output_start = x * 3;
                for channel in 0..3 {
                    let top = coarse_pixels[top_left + channel] * (1.0 - x_weight)
                        + coarse_pixels[top_right + channel] * x_weight;
                    let bottom = coarse_pixels[bottom_left + channel] * (1.0 - x_weight)
                        + coarse_pixels[bottom_right + channel] * x_weight;
                    let low_frequency = top * (1.0 - y_weight) + bottom * y_weight;
                    row[output_start + channel] = fine_row[output_start + channel] - low_frequency;
                }
            }
        });

    Rgb32FImage::from_raw(width, height, output)
        .expect("Laplacian detail buffer dimensions must match")
}

fn upsample_and_add_rgb(coarse: &Rgb32FImage, detail: &Rgb32FImage) -> Rgb32FImage {
    let (width, height) = detail.dimensions();
    let x_samples = linear_samples(coarse.width(), width);
    let y_samples = linear_samples(coarse.height(), height);
    let coarse_stride = coarse.width() as usize * 3;
    let output_stride = width as usize * 3;
    let coarse_pixels = coarse.as_raw();
    let detail_pixels = detail.as_raw();
    let mut output = vec![0.0f32; output_stride * height as usize];

    output
        .par_chunks_mut(output_stride)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            let y_weight = y_sample.upper_weight;
            let detail_row = &detail_pixels[y * output_stride..(y + 1) * output_stride];
            for (x, x_sample) in x_samples.iter().copied().enumerate() {
                let x_weight = x_sample.upper_weight;
                let top_left = y_sample.lower * coarse_stride + x_sample.lower * 3;
                let top_right = y_sample.lower * coarse_stride + x_sample.upper * 3;
                let bottom_left = y_sample.upper * coarse_stride + x_sample.lower * 3;
                let bottom_right = y_sample.upper * coarse_stride + x_sample.upper * 3;
                let output_start = x * 3;
                for channel in 0..3 {
                    let top = coarse_pixels[top_left + channel] * (1.0 - x_weight)
                        + coarse_pixels[top_right + channel] * x_weight;
                    let bottom = coarse_pixels[bottom_left + channel] * (1.0 - x_weight)
                        + coarse_pixels[bottom_right + channel] * x_weight;
                    row[output_start + channel] = top * (1.0 - y_weight)
                        + bottom * y_weight
                        + detail_row[output_start + channel];
                }
            }
        });

    Rgb32FImage::from_raw(width, height, output).expect("reconstructed image dimensions must match")
}

fn combine_rgb(
    base: &Rgb32FImage,
    candidate: &Rgb32FImage,
    mask: &GrayImage,
    hard_mask: bool,
) -> Rgb32FImage {
    let (width, height) = base.dimensions();
    let mut output = vec![0.0f32; width as usize * height as usize * 3];
    output
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                let mask_value = if hard_mask {
                    if mask.get_pixel(x, y)[0] >= 128 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    mask.get_pixel(x, y)[0] as f32 / 255.0
                };
                let base_pixel = base.get_pixel(x, y);
                let candidate_pixel = candidate.get_pixel(x, y);
                let start = x as usize * 3;
                for channel in 0..3 {
                    row[start + channel] = base_pixel[channel] * (1.0 - mask_value)
                        + candidate_pixel[channel] * mask_value;
                }
            }
        });
    Rgb32FImage::from_raw(width, height, output).expect("combined image dimensions must match")
}

pub(super) fn crop_to_valid_rectangle(image: Rgb32FImage, mask: &GrayImage) -> Rgb32FImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return image;
    }

    let mut heights = vec![0usize; width as usize];
    let mut stack = Vec::with_capacity(width as usize + 1);
    let mut best_area = 0usize;
    let mut best_left = 0usize;
    let mut best_top = 0usize;
    let mut best_width = width as usize;
    let mut best_height = height as usize;

    for y in 0..height as usize {
        for (x, height_value) in heights.iter_mut().enumerate() {
            *height_value = if mask.get_pixel(x as u32, y as u32)[0] > 0 {
                height_value.saturating_add(1)
            } else {
                0
            };
        }

        stack.clear();
        for x in 0..=width as usize {
            let current_height = if x < width as usize { heights[x] } else { 0 };
            while let Some(&bar_index) = stack.last() {
                if heights[bar_index] <= current_height {
                    break;
                }
                stack.pop();
                let left = stack.last().map_or(0, |&index| index + 1);
                let rectangle_width = x - left;
                let rectangle_height = heights[bar_index];
                let area = rectangle_width * rectangle_height;
                if area > best_area {
                    best_area = area;
                    best_left = left;
                    best_top = y + 1 - rectangle_height;
                    best_width = rectangle_width;
                    best_height = rectangle_height;
                }
            }
            stack.push(x);
        }
    }

    if best_area == 0 {
        return image;
    }

    if best_left == 0
        && best_top == 0
        && best_width == width as usize
        && best_height == height as usize
    {
        return image;
    }

    image::imageops::crop_imm(
        &image,
        best_left as u32,
        best_top as u32,
        best_width as u32,
        best_height as u32,
    )
    .to_image()
}

fn reflected_run_source_index(
    target: usize,
    limit: usize,
    left: Option<usize>,
    right: Option<usize>,
) -> usize {
    match (left, right) {
        (Some(left), Some(right)) => {
            let distance_from_left = target - left;
            let distance_from_right = right - target;
            if distance_from_left <= distance_from_right {
                left.saturating_sub(distance_from_left)
            } else {
                right
                    .saturating_add(distance_from_right)
                    .min(limit.saturating_sub(1))
            }
        }
        (Some(left), None) => left.saturating_sub(target - left),
        (None, Some(right)) => right
            .saturating_add(right - target)
            .min(limit.saturating_sub(1)),
        (None, None) => target,
    }
}

const FOCUS_FILL_TONE_RADIUS: usize = 64;

fn fill_tone_adjusted_pixel(
    pixel: [f32; 3],
    source_tone: [f32; 3],
    left_tone: [f32; 3],
    right_tone: [f32; 3],
    progress: f32,
) -> [f32; 3] {
    let progress = progress.clamp(0.0, 1.0);
    let blend = progress * progress * (3.0 - 2.0 * progress);
    let mut output = [0.0f32; 3];
    for channel in 0..3 {
        let target_tone = left_tone[channel] * (1.0 - blend) + right_tone[channel] * blend;
        output[channel] = (pixel[channel] + target_tone - source_tone[channel]).clamp(0.0, 1.0);
    }
    output
}

fn horizontal_fill_edge_tone(
    image_row: &[f32],
    edge: usize,
    from_left: bool,
    radius: usize,
) -> [f32; 3] {
    let width = image_row.len() / 3;
    let radius = radius.max(1).min(width);
    let (start, end) = if from_left {
        (edge.saturating_sub(radius - 1), edge)
    } else {
        (edge, (edge + radius - 1).min(width.saturating_sub(1)))
    };
    let mut tone = [0.0f32; 3];
    let sample_count = (end - start + 1) as f32;
    for x in start..=end {
        let pixel_start = x * 3;
        for channel in 0..3 {
            tone[channel] += image_row[pixel_start + channel];
        }
    }
    for channel in &mut tone {
        *channel /= sample_count;
    }
    tone
}

fn vertical_fill_edge_tone(
    image_pixels: &[f32],
    width: usize,
    edge: usize,
    x: usize,
    from_top: bool,
    radius: usize,
) -> [f32; 3] {
    let height = image_pixels.len() / (width * 3);
    let radius = radius.max(1).min(height);
    let (start, end) = if from_top {
        (edge.saturating_sub(radius - 1), edge)
    } else {
        (edge, (edge + radius - 1).min(height.saturating_sub(1)))
    };
    let mut tone = [0.0f32; 3];
    let sample_count = (end - start + 1) as f32;
    for y in start..=end {
        let pixel_start = (y * width + x) * 3;
        for channel in 0..3 {
            tone[channel] += image_pixels[pixel_start + channel];
        }
    }
    for channel in &mut tone {
        *channel /= sample_count;
    }
    tone
}

fn fill_invalid_runs_horizontally(image: &mut Rgb32FImage, mask: &mut GrayImage) -> usize {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return 0;
    }
    let width = width as usize;
    let image_stride = width * 3;
    image
        .as_mut()
        .par_chunks_mut(image_stride)
        .zip(mask.as_mut().par_chunks_mut(width))
        .map(|(image_row, mask_row)| {
            let mut filled = 0usize;
            let mut run_start = 0usize;
            while run_start < width {
                if mask_row[run_start] > 0 {
                    run_start += 1;
                    continue;
                }
                let mut run_end = run_start + 1;
                while run_end < width && mask_row[run_end] == 0 {
                    run_end += 1;
                }
                let left = run_start.checked_sub(1).filter(|&x| mask_row[x] > 0);
                let right = (run_end < width && mask_row[run_end] > 0).then_some(run_end);
                if left.is_none() && right.is_none() {
                    run_start = run_end;
                    continue;
                }
                let edge_tones = left.zip(right).map(|(left, right)| {
                    (
                        horizontal_fill_edge_tone(image_row, left, true, FOCUS_FILL_TONE_RADIUS),
                        horizontal_fill_edge_tone(image_row, right, false, FOCUS_FILL_TONE_RADIUS),
                    )
                });
                for target in run_start..run_end {
                    let mut source = reflected_run_source_index(target, width, left, right);
                    if mask_row[source] == 0 {
                        source = left.or(right).expect("a fill run has a valid neighbour");
                    }
                    let source_start = source * 3;
                    let target_start = target * 3;
                    let pixel = [
                        image_row[source_start],
                        image_row[source_start + 1],
                        image_row[source_start + 2],
                    ];
                    let pixel = edge_tones.map_or(pixel, |(left_tone, right_tone)| {
                        let source_tone =
                            if left.is_some_and(|left| target - left <= right.unwrap() - target) {
                                left_tone
                            } else {
                                right_tone
                            };
                        let progress =
                            (target - run_start + 1) as f32 / (run_end - run_start + 1) as f32;
                        fill_tone_adjusted_pixel(
                            pixel,
                            source_tone,
                            left_tone,
                            right_tone,
                            progress,
                        )
                    });
                    image_row[target_start..target_start + 3].copy_from_slice(&pixel);
                    mask_row[target] = 255;
                    filled += 1;
                }
                run_start = run_end;
            }
            filled
        })
        .sum()
}

fn fill_invalid_runs_vertically(image: &mut Rgb32FImage, mask: &mut GrayImage) -> usize {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return 0;
    }
    let width = width as usize;
    let height = height as usize;
    let image_pixels = image.as_mut();
    let mask_pixels = mask.as_mut();
    let mut filled = 0usize;
    for x in 0..width {
        let mut run_start = 0usize;
        while run_start < height {
            if mask_pixels[run_start * width + x] > 0 {
                run_start += 1;
                continue;
            }
            let mut run_end = run_start + 1;
            while run_end < height && mask_pixels[run_end * width + x] == 0 {
                run_end += 1;
            }
            let top = run_start
                .checked_sub(1)
                .filter(|&y| mask_pixels[y * width + x] > 0);
            let bottom =
                (run_end < height && mask_pixels[run_end * width + x] > 0).then_some(run_end);
            if top.is_none() && bottom.is_none() {
                run_start = run_end;
                continue;
            }
            let edge_tones = top.zip(bottom).map(|(top, bottom)| {
                (
                    vertical_fill_edge_tone(
                        image_pixels,
                        width,
                        top,
                        x,
                        true,
                        FOCUS_FILL_TONE_RADIUS,
                    ),
                    vertical_fill_edge_tone(
                        image_pixels,
                        width,
                        bottom,
                        x,
                        false,
                        FOCUS_FILL_TONE_RADIUS,
                    ),
                )
            });
            for target in run_start..run_end {
                let mut source = reflected_run_source_index(target, height, top, bottom);
                if mask_pixels[source * width + x] == 0 {
                    source = top.or(bottom).expect("a fill run has a valid neighbour");
                }
                let source_start = (source * width + x) * 3;
                let target_start = (target * width + x) * 3;
                let pixel = [
                    image_pixels[source_start],
                    image_pixels[source_start + 1],
                    image_pixels[source_start + 2],
                ];
                let pixel = edge_tones.map_or(pixel, |(top_tone, bottom_tone)| {
                    let source_tone =
                        if top.is_some_and(|top| target - top <= bottom.unwrap() - target) {
                            top_tone
                        } else {
                            bottom_tone
                        };
                    let progress =
                        (target - run_start + 1) as f32 / (run_end - run_start + 1) as f32;
                    fill_tone_adjusted_pixel(pixel, source_tone, top_tone, bottom_tone, progress)
                });
                image_pixels[target_start..target_start + 3].copy_from_slice(&pixel);
                mask_pixels[target * width + x] = 255;
                filled += 1;
            }
            run_start = run_end;
        }
    }
    filled
}

fn fill_focus_canvas_margins(image: &mut Rgb32FImage, mask: &mut GrayImage) -> (usize, usize) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return (0, 0);
    }
    let mut filled = 0usize;
    // Reflection extends the local canvas weave and broad tone from the nearest
    // valid edge. Alternating directions also fills the triangular corner areas
    // created by a projective scan without allocating another full-size image.
    for _ in 0..3 {
        filled += fill_invalid_runs_horizontally(image, mask);
        filled += fill_invalid_runs_vertically(image, mask);
        if !mask.as_raw().contains(&0) {
            break;
        }
    }
    let remaining = mask.as_raw().iter().filter(|&&value| value == 0).count();
    (filled, remaining)
}

fn masked_box_blur_rgb(source: &Rgb32FImage, mask: &GrayImage, radius: usize) -> Rgb32FImage {
    let (width, height) = source.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) || radius == 0 {
        return source.clone();
    }

    let width = width as usize;
    let height = height as usize;
    let pixel_count = width * height;
    let window_size = radius.saturating_mul(2).saturating_add(1);
    let source_pixels = source.as_raw();
    let mask_pixels = mask.as_raw();
    let mut horizontal_values = vec![0.0f32; pixel_count * 3];
    let mut horizontal_weights = vec![0u32; pixel_count];

    horizontal_values
        .par_chunks_mut(width * 3)
        .zip(horizontal_weights.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (output_row, weight_row))| {
            let source_row_start = y * width * 3;
            let mask_row_start = y * width;
            let mut sums = [0.0f32; 3];
            let mut weight = 0u32;
            for offset in 0..window_size {
                let x = offset.saturating_sub(radius).min(width - 1);
                if mask_pixels[mask_row_start + x] > 0 {
                    weight += 1;
                    let start = source_row_start + x * 3;
                    for channel in 0..3 {
                        sums[channel] += source_pixels[start + channel];
                    }
                }
            }
            let write_pixel = |x: usize,
                               output_row: &mut [f32],
                               weight_row: &mut [u32],
                               sums: [f32; 3],
                               weight: u32| {
                let start = x * 3;
                if weight > 0 {
                    output_row[start..start + 3].copy_from_slice(&sums);
                }
                weight_row[x] = weight;
            };
            write_pixel(0, output_row, weight_row, sums, weight);
            for x in 1..width {
                let add_x = (x + radius).min(width - 1);
                let remove_x = x.saturating_sub(radius + 1);
                if mask_pixels[mask_row_start + add_x] > 0 {
                    weight += 1;
                    let start = source_row_start + add_x * 3;
                    for channel in 0..3 {
                        sums[channel] += source_pixels[start + channel];
                    }
                }
                if mask_pixels[mask_row_start + remove_x] > 0 {
                    weight = weight.saturating_sub(1);
                    let start = source_row_start + remove_x * 3;
                    for channel in 0..3 {
                        sums[channel] -= source_pixels[start + channel];
                    }
                }
                write_pixel(x, output_row, weight_row, sums, weight);
            }
        });

    let mut output = vec![0.0f32; pixel_count * 3];
    for x in 0..width {
        let mut sums = [0.0f32; 3];
        let mut weight = 0u32;
        for offset in 0..window_size {
            let y = offset.saturating_sub(radius).min(height - 1);
            let index = y * width + x;
            if horizontal_weights[index] > 0 {
                weight += horizontal_weights[index];
                let start = index * 3;
                for channel in 0..3 {
                    sums[channel] += horizontal_values[start + channel];
                }
            }
        }
        let write_pixel = |y: usize, output: &mut [f32], sums: [f32; 3], weight: u32| {
            let index = (y * width + x) * 3;
            if weight > 0 {
                for channel in 0..3 {
                    output[index + channel] = sums[channel] / weight as f32;
                }
            } else {
                let source_index = index;
                output[index..index + 3]
                    .copy_from_slice(&source_pixels[source_index..source_index + 3]);
            }
        };
        write_pixel(0, &mut output, sums, weight);
        for y in 1..height {
            let add_y = (y + radius).min(height - 1);
            let remove_y = y.saturating_sub(radius + 1);
            let add_index = add_y * width + x;
            let remove_index = remove_y * width + x;
            if horizontal_weights[add_index] > 0 {
                weight += horizontal_weights[add_index];
                let start = add_index * 3;
                for channel in 0..3 {
                    sums[channel] += horizontal_values[start + channel];
                }
            }
            if horizontal_weights[remove_index] > 0 {
                weight = weight.saturating_sub(horizontal_weights[remove_index]);
                let start = remove_index * 3;
                for channel in 0..3 {
                    sums[channel] -= horizontal_values[start + channel];
                }
            }
            write_pixel(y, &mut output, sums, weight);
        }
    }

    Rgb32FImage::from_raw(width as u32, height as u32, output)
        .expect("masked RGB blur dimensions must match")
}

fn masked_box_blur_focus_map(
    source: &[f32],
    mask: &[u8],
    width: u32,
    height: u32,
    radius: usize,
) -> Vec<f32> {
    let width = width as usize;
    let height = height as usize;
    if width == 0 || height == 0 || source.len() != width * height || mask.len() != source.len() {
        return source.to_vec();
    }
    if radius == 0 {
        return source.to_vec();
    }

    let window_size = radius.saturating_mul(2).saturating_add(1);
    let mut horizontal_values = vec![0.0f32; source.len()];
    let mut horizontal_weights = vec![0u32; source.len()];
    horizontal_values
        .par_chunks_mut(width)
        .zip(horizontal_weights.par_chunks_mut(width))
        .enumerate()
        .for_each(|(y, (output_row, weight_row))| {
            let row_start = y * width;
            let mut sum = 0.0f32;
            let mut weight = 0u32;
            for offset in 0..window_size {
                let x = offset.saturating_sub(radius).min(width - 1);
                if mask[row_start + x] > 0 {
                    sum += source[row_start + x];
                    weight += 1;
                }
            }
            output_row[0] = sum;
            weight_row[0] = weight;
            for x in 1..width {
                let add_x = (x + radius).min(width - 1);
                let remove_x = x.saturating_sub(radius + 1);
                if mask[row_start + add_x] > 0 {
                    sum += source[row_start + add_x];
                    weight += 1;
                }
                if mask[row_start + remove_x] > 0 {
                    sum -= source[row_start + remove_x];
                    weight = weight.saturating_sub(1);
                }
                output_row[x] = sum;
                weight_row[x] = weight;
            }
        });

    let mut output = vec![0.0f32; source.len()];
    for x in 0..width {
        let mut sum = 0.0f32;
        let mut weight = 0u32;
        for offset in 0..window_size {
            let y = offset.saturating_sub(radius).min(height - 1);
            let index = y * width + x;
            sum += horizontal_values[index];
            weight += horizontal_weights[index];
        }
        let first_index = x;
        output[first_index] = if weight > 0 {
            sum / weight as f32
        } else {
            source[first_index]
        };
        for y in 1..height {
            let add_y = (y + radius).min(height - 1);
            let remove_y = y.saturating_sub(radius + 1);
            let add_index = add_y * width + x;
            let remove_index = remove_y * width + x;
            sum += horizontal_values[add_index] - horizontal_values[remove_index];
            weight += horizontal_weights[add_index];
            weight = weight.saturating_sub(horizontal_weights[remove_index]);
            let index = y * width + x;
            output[index] = if weight > 0 {
                sum / weight as f32
            } else {
                source[index]
            };
        }
    }
    output
}

#[cfg(test)]
fn harmonize_focus_background_tone(
    image: &mut Rgb32FImage,
    image_mask: &GrayImage,
    foreground_mask: &GrayImage,
) {
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || image_mask.dimensions() != (width, height)
        || foreground_mask.dimensions() != (width, height)
    {
        return;
    }

    let (analysis_width, analysis_height) =
        focus_analysis_dimensions(width, height, width.max(height));
    let analysis_image = resize_rgb(image, analysis_width, analysis_height);
    let analysis_mask = resize_binary_mask(image_mask, analysis_width, analysis_height);
    // Do not derive the correction domain from the focus ownership mask. That
    // mask is made from sharpness decisions and may contain source-shaped
    // islands on the canvas. Instead, sample only pixels that still look like
    // the warm-brown paper itself; this makes an exposure block eligible for
    // correction while keeping painted colours out of the estimator.
    let analysis_background_like = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
        let covered = analysis_mask.get_pixel(x, y)[0] > 0;
        let index = y as usize * analysis_width as usize + x as usize;
        let pixel_start = index * 3;
        let pixel = &analysis_image.as_raw()[pixel_start..pixel_start + 3];
        let canvas_like = focus_stack_pixel_is_canvas_like(pixel);
        let tone_foreground = focus_stack_pixel_is_tone_foreground(pixel);
        image::Luma([u8::from(covered && canvas_like && !tone_foreground) * 255])
    });
    let foreground_soft_radius = (analysis_width.max(analysis_height) as f32 * 0.005)
        .round()
        .clamp(2.0, 12.0) as usize;
    let analysis_background_values = analysis_background_like
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let analysis_background_weight = box_blur_focus_map(
        &analysis_background_values,
        analysis_width,
        analysis_height,
        foreground_soft_radius,
    )
    .into_iter()
    .map(|value| value.clamp(0.0, 1.0))
    .collect::<Vec<_>>();
    let analysis_background = analysis_background_like.clone();
    let background_samples = analysis_background
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if background_samples < 128 {
        return;
    }

    let background_pixels = analysis_background.as_raw();
    let mut global_channels = [Vec::new(), Vec::new(), Vec::new()];
    for (index, &background) in background_pixels.iter().enumerate() {
        if background == 0 {
            continue;
        }
        let start = index * 3;
        for channel in 0..3 {
            global_channels[channel].push(analysis_image.as_raw()[start + channel]);
        }
    }
    let global_tone = [
        median_f32(&mut global_channels[0]).unwrap_or(0.0),
        median_f32(&mut global_channels[1]).unwrap_or(0.0),
        median_f32(&mut global_channels[2]).unwrap_or(0.0),
    ];
    let analysis_background_like = analysis_background.clone();
    let background_like_pixels = analysis_background_like
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    println!(
        "  - Background tone sample coverage: {background_like_pixels}/{background_samples} analysis pixels"
    );
    if background_like_pixels < 128 {
        return;
    }

    // The first blur removes weave-scale variation from the estimate. The
    // second, much wider blur provides a continuous target across source-sized
    // exposure blocks. Their difference is a low-frequency correction, so the
    // original high-frequency canvas texture is retained pixel-for-pixel.
    let local_tone = masked_box_blur_rgb(
        &analysis_image,
        &analysis_background_like,
        FOCUS_BACKGROUND_TONE_LOCAL_RADIUS,
    );
    let smooth_tone = masked_box_blur_rgb(
        &local_tone,
        &analysis_background_like,
        FOCUS_BACKGROUND_TONE_SMOOTH_RADIUS,
    );
    let local_pixels = local_tone.as_raw();
    let smooth_pixels = smooth_tone.as_raw();
    let mut correction = vec![[0.0f32; 3]; analysis_width as usize * analysis_height as usize];
    correction
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, output)| {
            if background_like_pixels == 0 || analysis_background_like.as_raw()[index] == 0 {
                return;
            }
            let start = index * 3;
            for channel in 0..3 {
                let target = smooth_pixels[start + channel]
                    * (1.0 - FOCUS_BACKGROUND_TONE_GLOBAL_WEIGHT)
                    + global_tone[channel] * FOCUS_BACKGROUND_TONE_GLOBAL_WEIGHT;
                output[channel] = (target - local_pixels[start + channel]).clamp(
                    -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                    FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                );
            }
        });

    let x_samples = linear_samples(analysis_width, width);
    let y_samples = linear_samples(analysis_height, height);
    let analysis_stride = analysis_width as usize;
    let background_weight_ref = &analysis_background_weight;
    let correction_ref = &correction;
    let full_mask = image_mask.as_raw();
    image
        .as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            for x in 0..width as usize {
                let full_index = y * width as usize + x;
                if full_mask[full_index] == 0 {
                    continue;
                }
                let start = x * 3;
                if !focus_stack_pixel_is_canvas_like(&row[start..start + 3])
                    || focus_stack_pixel_is_tone_foreground(&row[start..start + 3])
                {
                    continue;
                }
                let x_sample = x_samples[x];
                let top_left_weight =
                    background_weight_ref[y_sample.lower * analysis_stride + x_sample.lower];
                let top_right_weight =
                    background_weight_ref[y_sample.lower * analysis_stride + x_sample.upper];
                let bottom_left_weight =
                    background_weight_ref[y_sample.upper * analysis_stride + x_sample.lower];
                let bottom_right_weight =
                    background_weight_ref[y_sample.upper * analysis_stride + x_sample.upper];
                let top_weight = top_left_weight * (1.0 - x_sample.upper_weight)
                    + top_right_weight * x_sample.upper_weight;
                let bottom_weight = bottom_left_weight * (1.0 - x_sample.upper_weight)
                    + bottom_right_weight * x_sample.upper_weight;
                let background_weight = (top_weight * (1.0 - y_sample.upper_weight)
                    + bottom_weight * y_sample.upper_weight)
                    .clamp(0.0, 1.0);
                if background_weight <= 0.001 {
                    continue;
                }
                let top_left = correction_ref[y_sample.lower * analysis_stride + x_sample.lower];
                let top_right = correction_ref[y_sample.lower * analysis_stride + x_sample.upper];
                let bottom_left = correction_ref[y_sample.upper * analysis_stride + x_sample.lower];
                let bottom_right =
                    correction_ref[y_sample.upper * analysis_stride + x_sample.upper];
                let mut delta = [0.0f32; 3];
                for channel in 0..3 {
                    let top = top_left[channel] * (1.0 - x_sample.upper_weight)
                        + top_right[channel] * x_sample.upper_weight;
                    let bottom = bottom_left[channel] * (1.0 - x_sample.upper_weight)
                        + bottom_right[channel] * x_sample.upper_weight;
                    delta[channel] = (top * (1.0 - y_sample.upper_weight)
                        + bottom * y_sample.upper_weight)
                        * background_weight;
                }
                for channel in 0..3 {
                    row[start + channel] = (row[start + channel] + delta[channel]).clamp(0.0, 1.0);
                }
            }
        });
}

fn harmonize_focus_background_tone_with_owners(
    image: &mut Rgb32FImage,
    image_mask: &GrayImage,
    foreground_mask: &GrayImage,
    owner_map: &GrayImage,
) {
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || image_mask.dimensions() != (width, height)
        || foreground_mask.dimensions() != (width, height)
        || owner_map.dimensions() != (width, height)
    {
        return;
    }

    let (analysis_width, analysis_height) =
        focus_analysis_dimensions(width, height, width.max(height));
    let analysis_image = resize_rgb(image, analysis_width, analysis_height);
    let analysis_mask = resize_binary_mask(image_mask, analysis_width, analysis_height);
    let analysis_foreground = resize_binary_mask(foreground_mask, analysis_width, analysis_height);
    let foreground_envelope = build_focus_background_foreground_envelope(
        &analysis_image,
        &analysis_mask,
        &analysis_foreground,
    );
    let background = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
        let index = y as usize * analysis_width as usize + x as usize;
        let start = index * 3;
        let pixel = &analysis_image.as_raw()[start..start + 3];
        image::Luma([u8::from(
            analysis_mask.as_raw()[index] > 0
                && foreground_envelope.as_raw()[index] == 0
                && focus_stack_pixel_is_canvas_like(pixel)
                && !focus_stack_pixel_is_tone_foreground(pixel),
        ) * 255])
    });
    let background_pixels = background
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if background_pixels < 128 {
        return;
    }

    let owner_analysis = image::imageops::resize(
        owner_map,
        analysis_width,
        analysis_height,
        image::imageops::FilterType::Nearest,
    );
    let sample_step = ((u64::from(analysis_width) * u64::from(analysis_height) / 120_000) as f64)
        .sqrt()
        .ceil()
        .max(1.0) as usize;
    let mut global_channels = [Vec::new(), Vec::new(), Vec::new()];
    let mut owner_channels: Vec<[Vec<f32>; 3]> = (0..=u8::MAX as usize)
        .map(|_| [Vec::new(), Vec::new(), Vec::new()])
        .collect();
    for y in (0..analysis_height as usize).step_by(sample_step) {
        for x in (0..analysis_width as usize).step_by(sample_step) {
            let index = y * analysis_width as usize + x;
            if background.as_raw()[index] == 0 {
                continue;
            }
            let start = index * 3;
            let owner = owner_analysis.as_raw()[index] as usize;
            for channel in 0..3 {
                let value = analysis_image.as_raw()[start + channel];
                global_channels[channel].push(value);
                if owner > 0 {
                    owner_channels[owner][channel].push(value);
                }
            }
        }
    }
    let global_tone = [
        median_f32(&mut global_channels[0]).unwrap_or(0.0),
        median_f32(&mut global_channels[1]).unwrap_or(0.0),
        median_f32(&mut global_channels[2]).unwrap_or(0.0),
    ];
    let mut owner_corrections = vec![[0.0f32; 3]; u8::MAX as usize + 1];
    let mut corrected_owner_count = 0usize;
    for (owner, channels) in owner_channels.iter_mut().enumerate().skip(1) {
        if channels[0].len() < FOCUS_COLOR_MIN_SAMPLES {
            continue;
        }
        let mut has_adjustment = false;
        for channel in 0..3 {
            if channels[channel].len() < FOCUS_COLOR_MIN_SAMPLES {
                continue;
            }
            let owner_tone = median_f32(&mut channels[channel]).unwrap_or(global_tone[channel]);
            let correction = (global_tone[channel] - owner_tone).clamp(
                -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
            );
            owner_corrections[owner][channel] = correction;
            has_adjustment |= correction.abs() >= 0.002;
        }
        if has_adjustment {
            corrected_owner_count += 1;
        }
    }
    // Estimate a low-frequency tone field from the final image itself. These
    // frames cover one continuous piece of paper, so a broad source-sized
    // brightness step is an acquisition artefact, not detail that should
    // survive stacking. The small blur measures each tile's current tone; the
    // wide blur only softens the correction at the edge of an owner region.
    let local_tone = masked_box_blur_rgb(
        &analysis_image,
        &background,
        FOCUS_BACKGROUND_TONE_LOCAL_RADIUS,
    );
    let smooth_tone = masked_box_blur_rgb(
        &local_tone,
        &background,
        FOCUS_BACKGROUND_TONE_SMOOTH_RADIUS,
    );
    let local_pixels = local_tone.as_raw();
    let smooth_pixels = smooth_tone.as_raw();

    // Store the measured low-frequency correction only on protected background
    // samples, then blur that field across the same mask. The blur removes the
    // source-owner rectangle as a colour boundary while adding the correction
    // back to the full-resolution image leaves its weave and brush detail
    // untouched.
    let mut correction_pixels =
        vec![0.0f32; analysis_width as usize * analysis_height as usize * 3];
    for index in 0..background.as_raw().len() {
        if background.as_raw()[index] == 0 {
            continue;
        }
        let start = index * 3;
        let owner_correction = owner_corrections[owner_analysis.as_raw()[index] as usize];
        for channel in 0..3 {
            let target = smooth_pixels[start + channel]
                * (1.0 - FOCUS_BACKGROUND_TONE_GLOBAL_WEIGHT)
                + global_tone[channel] * FOCUS_BACKGROUND_TONE_GLOBAL_WEIGHT;
            let local_correction = (target - local_pixels[start + channel]).clamp(
                -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
            );
            // The owner median stabilizes very large regions whose local blur
            // is interrupted by a figure; it is only a supplement to the
            // pixel-local low-frequency estimate.
            correction_pixels[start + channel] =
                local_correction * 0.8 + owner_correction[channel] * 0.2;
        }
    }
    let correction_image =
        Rgb32FImage::from_raw(analysis_width, analysis_height, correction_pixels)
            .expect("focus owner correction dimensions must match");
    let correction_radius = (analysis_width.max(analysis_height) as f32 * 0.006)
        .round()
        .clamp(4.0, 16.0) as usize;
    let smoothed_correction =
        masked_box_blur_rgb(&correction_image, &background, correction_radius);
    let background_values = background
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let background_radius = (analysis_width.max(analysis_height) as f32 * 0.006)
        .round()
        .clamp(4.0, 16.0) as usize;
    let background_weight = box_blur_focus_map(
        &background_values,
        analysis_width,
        analysis_height,
        background_radius,
    );
    let x_samples = linear_samples(analysis_width, width);
    let y_samples = linear_samples(analysis_height, height);
    let analysis_stride = analysis_width as usize;
    let correction_ref = smoothed_correction.as_raw();
    let background_weight_ref = &background_weight;
    let image_mask_ref = image_mask.as_raw();
    image
        .as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            for x in 0..width as usize {
                if image_mask_ref[y * width as usize + x] == 0 {
                    continue;
                }
                // The cleaned analysis envelope is the protection boundary.
                // Do not gate this by the current full-resolution colour: the
                // darkest/lightest exposure tiles are precisely the background
                // pixels that need correction, and the low-frequency field does
                // not replace their weave or brush detail.
                let start = x * 3;
                let x_sample = x_samples[x];
                let top_weight = background_weight_ref
                    [y_sample.lower * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + background_weight_ref[y_sample.lower * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let bottom_weight = background_weight_ref
                    [y_sample.upper * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + background_weight_ref[y_sample.upper * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let background_weight = (top_weight * (1.0 - y_sample.upper_weight)
                    + bottom_weight * y_sample.upper_weight)
                    .clamp(0.0, 1.0);
                if background_weight <= 0.001 {
                    continue;
                }
                let top_left = &correction_ref[(y_sample.lower * analysis_stride + x_sample.lower)
                    * 3
                    ..(y_sample.lower * analysis_stride + x_sample.lower) * 3 + 3];
                let top_right = &correction_ref[(y_sample.lower * analysis_stride + x_sample.upper)
                    * 3
                    ..(y_sample.lower * analysis_stride + x_sample.upper) * 3 + 3];
                let bottom_left =
                    &correction_ref[(y_sample.upper * analysis_stride + x_sample.lower) * 3
                        ..(y_sample.upper * analysis_stride + x_sample.lower) * 3 + 3];
                let bottom_right =
                    &correction_ref[(y_sample.upper * analysis_stride + x_sample.upper) * 3
                        ..(y_sample.upper * analysis_stride + x_sample.upper) * 3 + 3];
                for channel in 0..3 {
                    let top = top_left[channel] * (1.0 - x_sample.upper_weight)
                        + top_right[channel] * x_sample.upper_weight;
                    let bottom = bottom_left[channel] * (1.0 - x_sample.upper_weight)
                        + bottom_right[channel] * x_sample.upper_weight;
                    row[start + channel] = (row[start + channel]
                        + (top * (1.0 - y_sample.upper_weight) + bottom * y_sample.upper_weight)
                            * background_weight)
                        .clamp(0.0, 1.0);
                }
            }
        });
    println!(
        "  - Owner-based background harmonization: samples={} owners_adjusted={} radius={}px",
        background_pixels, corrected_owner_count, correction_radius
    );
}

fn harmonize_focus_background_with_reference(
    image: &mut Rgb32FImage,
    image_mask: &GrayImage,
    foreground_mask: &GrayImage,
    reference_image: &Rgb32FImage,
    reference_mask: &GrayImage,
    owner_map: &GrayImage,
    owner_corrections: &[[f32; 3]],
) {
    let (width, height) = image.dimensions();
    let (analysis_width, analysis_height) = reference_image.dimensions();
    if width == 0
        || height == 0
        || analysis_width == 0
        || analysis_height == 0
        || image_mask.dimensions() != (width, height)
        || foreground_mask.dimensions() != (width, height)
        || reference_mask.dimensions() != (analysis_width, analysis_height)
        || owner_map.dimensions() != (width, height)
        || owner_corrections.is_empty()
    {
        return;
    }

    let analysis_image = resize_rgb(image, analysis_width, analysis_height);
    let analysis_mask = resize_binary_mask(image_mask, analysis_width, analysis_height);
    let analysis_foreground = resize_binary_mask(foreground_mask, analysis_width, analysis_height);
    let foreground_envelope = build_focus_background_foreground_envelope(
        &analysis_image,
        &analysis_mask,
        &analysis_foreground,
    );
    let background = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
        let index = y as usize * analysis_width as usize + x as usize;
        image::Luma([u8::from(
            analysis_mask.as_raw()[index] > 0
                && reference_mask.as_raw()[index] > 0
                && foreground_envelope.as_raw()[index] == 0,
        ) * 255])
    });
    let background_pixels = background
        .as_raw()
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if background_pixels < 128 {
        return;
    }

    let all_background = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
        let index = y as usize * analysis_width as usize + x as usize;
        image::Luma([u8::from(
            analysis_mask.as_raw()[index] > 0 && foreground_envelope.as_raw()[index] == 0,
        ) * 255])
    });
    let owner_analysis = image::imageops::resize(
        owner_map,
        analysis_width,
        analysis_height,
        image::imageops::FilterType::Nearest,
    );
    let owner_correction_pixels = all_background
        .as_raw()
        .iter()
        .enumerate()
        .flat_map(|(index, &value)| {
            if value == 0 {
                return [0.0; 3];
            }
            owner_corrections
                .get(owner_analysis.as_raw()[index] as usize)
                .copied()
                .unwrap_or([0.0; 3])
        })
        .collect::<Vec<_>>();
    let owner_correction_image =
        Rgb32FImage::from_raw(analysis_width, analysis_height, owner_correction_pixels)
            .expect("focus owner correction dimensions must match");
    let owner_correction_radius = (analysis_width.max(analysis_height) as f32 * 0.06)
        .round()
        .clamp(24.0, 96.0) as usize;
    let smoothed_owner_correction = masked_box_blur_rgb(
        &owner_correction_image,
        &all_background,
        owner_correction_radius,
    );

    // The reference is an average of every source's canvas-only pixels after a
    // robust per-frame offset. Compare it to the selected focus image only at a
    // broad scale. The final image keeps its own high-frequency samples, so the
    // operation cannot soften a face, sleeve, brush line, or the canvas weave.
    let focus_low = masked_box_blur_rgb(
        &analysis_image,
        &background,
        FOCUS_BACKGROUND_TONE_LOCAL_RADIUS,
    );
    let reference_low = masked_box_blur_rgb(
        reference_image,
        &background,
        FOCUS_BACKGROUND_TONE_LOCAL_RADIUS,
    );
    let focus_pixels = focus_low.as_raw();
    let reference_pixels = reference_low.as_raw();
    let mut correction_pixels =
        vec![0.0f32; analysis_width as usize * analysis_height as usize * 3];
    for index in 0..background.as_raw().len() {
        if background.as_raw()[index] == 0 {
            continue;
        }
        let start = index * 3;
        for channel in 0..3 {
            correction_pixels[start + channel] =
                (reference_pixels[start + channel] - focus_pixels[start + channel]).clamp(
                    -FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                    FOCUS_BACKGROUND_TONE_MAX_ADJUSTMENT,
                );
        }
    }
    let correction_image =
        Rgb32FImage::from_raw(analysis_width, analysis_height, correction_pixels)
            .expect("focus reference correction dimensions must match");
    let correction_radius = (analysis_width.max(analysis_height) as f32 * 0.06)
        .round()
        .clamp(24.0, 96.0) as usize;
    let smoothed_correction =
        masked_box_blur_rgb(&correction_image, &background, correction_radius);
    let reference_background_values = background
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let all_background_values = all_background
        .as_raw()
        .iter()
        .map(|&value| f32::from(value > 0))
        .collect::<Vec<_>>();
    let background_radius = (analysis_width.max(analysis_height) as f32 * 0.006)
        .round()
        .clamp(4.0, 16.0) as usize;
    let reference_background_weight = box_blur_focus_map(
        &reference_background_values,
        analysis_width,
        analysis_height,
        background_radius,
    );
    let all_background_weight = box_blur_focus_map(
        &all_background_values,
        analysis_width,
        analysis_height,
        background_radius,
    );
    let x_samples = linear_samples(analysis_width, width);
    let y_samples = linear_samples(analysis_height, height);
    let analysis_stride = analysis_width as usize;
    let correction_ref = smoothed_correction.as_raw();
    let owner_correction_ref = smoothed_owner_correction.as_raw();
    let reference_background_weight_ref = &reference_background_weight;
    let all_background_weight_ref = &all_background_weight;
    let image_mask_ref = image_mask.as_raw();
    image
        .as_mut()
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y_sample = y_samples[y];
            for x in 0..width as usize {
                if image_mask_ref[y * width as usize + x] == 0 {
                    continue;
                }
                let x_sample = x_samples[x];
                let top_weight = all_background_weight_ref
                    [y_sample.lower * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + all_background_weight_ref[y_sample.lower * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let bottom_weight = all_background_weight_ref
                    [y_sample.upper * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + all_background_weight_ref[y_sample.upper * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let all_background_weight = (top_weight * (1.0 - y_sample.upper_weight)
                    + bottom_weight * y_sample.upper_weight)
                    .clamp(0.0, 1.0);
                if all_background_weight <= 0.001 {
                    continue;
                }
                let reference_top_weight = reference_background_weight_ref
                    [y_sample.lower * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + reference_background_weight_ref
                        [y_sample.lower * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let reference_bottom_weight = reference_background_weight_ref
                    [y_sample.upper * analysis_stride + x_sample.lower]
                    * (1.0 - x_sample.upper_weight)
                    + reference_background_weight_ref
                        [y_sample.upper * analysis_stride + x_sample.upper]
                        * x_sample.upper_weight;
                let reference_background_weight = (reference_top_weight
                    * (1.0 - y_sample.upper_weight)
                    + reference_bottom_weight * y_sample.upper_weight)
                    .clamp(0.0, 1.0);
                let top_left = &correction_ref[(y_sample.lower * analysis_stride + x_sample.lower)
                    * 3
                    ..(y_sample.lower * analysis_stride + x_sample.lower) * 3 + 3];
                let top_right = &correction_ref[(y_sample.lower * analysis_stride + x_sample.upper)
                    * 3
                    ..(y_sample.lower * analysis_stride + x_sample.upper) * 3 + 3];
                let bottom_left =
                    &correction_ref[(y_sample.upper * analysis_stride + x_sample.lower) * 3
                        ..(y_sample.upper * analysis_stride + x_sample.lower) * 3 + 3];
                let bottom_right =
                    &correction_ref[(y_sample.upper * analysis_stride + x_sample.upper) * 3
                        ..(y_sample.upper * analysis_stride + x_sample.upper) * 3 + 3];
                let start = x * 3;
                for channel in 0..3 {
                    let top = top_left[channel] * (1.0 - x_sample.upper_weight)
                        + top_right[channel] * x_sample.upper_weight;
                    let bottom = bottom_left[channel] * (1.0 - x_sample.upper_weight)
                        + bottom_right[channel] * x_sample.upper_weight;
                    let delta = (top * (1.0 - y_sample.upper_weight)
                        + bottom * y_sample.upper_weight)
                        * reference_background_weight;
                    let owner_top_left = &owner_correction_ref[(y_sample.lower * analysis_stride
                        + x_sample.lower)
                        * 3
                        ..(y_sample.lower * analysis_stride + x_sample.lower) * 3 + 3];
                    let owner_top_right = &owner_correction_ref[(y_sample.lower * analysis_stride
                        + x_sample.upper)
                        * 3
                        ..(y_sample.lower * analysis_stride + x_sample.upper) * 3 + 3];
                    let owner_bottom_left = &owner_correction_ref[(y_sample.upper * analysis_stride
                        + x_sample.lower)
                        * 3
                        ..(y_sample.upper * analysis_stride + x_sample.lower) * 3 + 3];
                    let owner_bottom_right =
                        &owner_correction_ref[(y_sample.upper * analysis_stride + x_sample.upper)
                            * 3
                            ..(y_sample.upper * analysis_stride + x_sample.upper) * 3 + 3];
                    let owner_top = owner_top_left[channel] * (1.0 - x_sample.upper_weight)
                        + owner_top_right[channel] * x_sample.upper_weight;
                    let owner_bottom = owner_bottom_left[channel] * (1.0 - x_sample.upper_weight)
                        + owner_bottom_right[channel] * x_sample.upper_weight;
                    let owner_delta = (owner_top * (1.0 - y_sample.upper_weight)
                        + owner_bottom * y_sample.upper_weight)
                        * all_background_weight;
                    row[start + channel] =
                        (row[start + channel] + owner_delta + delta).clamp(0.0, 1.0);
                }
            }
        });
    println!(
        "  - Reference background harmonization: samples={} radius={}px owner_radius={}px",
        background_pixels, correction_radius, owner_correction_radius
    );
}

fn multiband_blend(
    base: Rgb32FImage,
    candidate: Rgb32FImage,
    mask: GrayImage,
    low_frequency_mask: Option<GrayImage>,
    max_bands: usize,
    hard_finest_band: bool,
) -> Rgb32FImage {
    let (width, height) = base.dimensions();
    let mut current_base = base;
    let mut current_candidate = candidate;
    let mut current_mask = mask;
    let mut current_low_frequency_mask = low_frequency_mask.unwrap_or_else(|| current_mask.clone());
    let mut base_laplacian = Vec::new();
    let mut candidate_laplacian = Vec::new();
    let mut masks = Vec::new();
    let mut low_frequency_masks = Vec::new();

    while base_laplacian.len() + 1 < max_bands {
        // Continue far enough for the coarsest band to absorb broad illumination and
        // vignetting differences. Stopping at 32px left low-frequency exposure steps
        // visible even though the high-frequency seam itself was well placed.
        if current_base.width() <= 4 || current_base.height() <= 4 {
            break;
        }
        let next_base = downsample_rgb_half(&current_base);
        let next_candidate = downsample_rgb_half(&current_candidate);
        let next_mask = downsample_mask_half(&current_mask);
        let next_low_frequency_mask = downsample_mask_half(&current_low_frequency_mask);
        base_laplacian.push(subtract_upsampled_rgb(&current_base, &next_base));
        candidate_laplacian.push(subtract_upsampled_rgb(&current_candidate, &next_candidate));
        masks.push(current_mask);
        low_frequency_masks.push(current_low_frequency_mask);
        current_base = next_base;
        current_candidate = next_candidate;
        current_mask = next_mask;
        current_low_frequency_mask = next_low_frequency_mask;
    }

    let mut reconstructed = combine_rgb(
        &current_base,
        &current_candidate,
        &current_low_frequency_mask,
        false,
    );
    for level in (0..base_laplacian.len()).rev() {
        let blend_mask = if level >= PANORAMA_GLOBAL_TONE_FIRST_BAND {
            &low_frequency_masks[level]
        } else {
            &masks[level]
        };
        let blended_detail = combine_rgb(
            &base_laplacian[level],
            &candidate_laplacian[level],
            blend_mask,
            // Focus ownership is a source-selection problem. Once a seam has
            // been chosen, averaging middle-frequency detail across it can
            // create a second contour when the two registered samples differ
            // by even a fraction of a pixel. Keep all visible detail bands on
            // one side; only the broad tone bands may feather continuously.
            hard_finest_band && level < PANORAMA_GLOBAL_TONE_FIRST_BAND,
        );
        reconstructed = upsample_and_add_rgb(&reconstructed, &blended_detail);
    }
    if reconstructed.dimensions() == (width, height) {
        reconstructed
    } else {
        resize_rgb(&reconstructed, width, height)
    }
}

fn blend_focus_seam_band(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width < 3 || layer_height < 3 {
        return;
    }
    debug_assert_eq!(candidate.mask.dimensions(), (layer_width, layer_height));
    debug_assert_eq!(decision_mask.dimensions(), (layer_width, layer_height));

    let base_width = base.width();
    let base_height = base.height();
    if candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }

    // A seam may cross a detected depth/occlusion layer, but that layer must
    // never be reconstructed by averaging two different source positions. The
    // mask is optional and content-derived; for ordinary flat stacks this is an
    // all-zero buffer and the generic seam path is unchanged.
    let foreground_overlap = focus_foreground_overlap_mask_for_region(
        merged_foreground_mask,
        candidate.left,
        candidate.top,
        (layer_width, layer_height),
    );
    let protected_foreground = candidate
        .foreground_mask
        .as_raw()
        .iter()
        .zip(foreground_overlap.iter())
        .zip(candidate.relaxed_foreground_mask.as_raw().iter())
        .map(
            |((candidate_foreground, existing_foreground), relaxed_foreground)| {
                u8::from(
                    (*candidate_foreground > 0 || *existing_foreground > 0)
                        && *relaxed_foreground == 0,
                )
            },
        )
        .collect::<Vec<_>>();
    let hard_foreground_edge_radius = ((layer_width.max(layer_height) as f32 / 2_400.0)
        * FOCUS_FOREGROUND_HARD_EDGE_RADIUS_AT_2400)
        .round()
        .clamp(3.0, 24.0) as usize;
    let protected_foreground_edges = focus_foreground_edge_mask(
        &protected_foreground,
        layer_width,
        layer_height,
        hard_foreground_edge_radius,
    );
    let is_protected = |x: u32, y: u32| -> bool {
        protected_foreground_edges[y as usize * layer_width as usize + x as usize] > 0
    };

    let owns_candidate = |x: u32, y: u32| -> bool {
        if x >= layer_width || y >= layer_height || candidate.mask.get_pixel(x, y)[0] == 0 {
            return false;
        }
        let global_x = candidate.left + x;
        let global_y = candidate.top + y;
        base_mask.get_pixel(global_x, global_y)[0] == 0 || decision_mask.get_pixel(x, y)[0] > 0
    };

    let mut seam_left = layer_width;
    let mut seam_right = 0u32;
    let mut seam_top = layer_height;
    let mut seam_bottom = 0u32;
    for y in 1..layer_height.saturating_sub(1) {
        for x in 1..layer_width.saturating_sub(1) {
            // Do not let a transition on a foreground silhouette expand the
            // seam's bounding box over the whole object. Its ownership is
            // restored pixel-for-pixel after the paper seam is blended.
            if is_protected(x, y) {
                continue;
            }
            let ownership = owns_candidate(x, y);
            let has_transition = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                .into_iter()
                .any(|(neighbor_x, neighbor_y)| {
                    !is_protected(neighbor_x, neighbor_y)
                        && owns_candidate(neighbor_x, neighbor_y) != ownership
                });
            if has_transition {
                seam_left = seam_left.min(x);
                seam_right = seam_right.max(x);
                seam_top = seam_top.min(y);
                seam_bottom = seam_bottom.max(y);
            }
        }
    }
    if seam_left > seam_right || seam_top > seam_bottom {
        hard_select_focus_layer(base, base_mask, candidate, decision_mask);
        return;
    }

    let radius = FOCUS_SEAM_BLEND_RADIUS.min(layer_width.max(layer_height) as usize);
    // The focus seam is used for detail ownership, but the exposure step can
    // extend across the complete shared canvas area. Build the low-frequency
    // pyramid over that overlap, not only over a rectangle around the sharpness
    // transition; otherwise a source-sized canvas block outside the seam box is
    // copied unchanged into the final stack.
    let mut overlap_left = layer_width;
    let mut overlap_right = 0u32;
    let mut overlap_top = layer_height;
    let mut overlap_bottom = 0u32;
    for y in 0..layer_height {
        for x in 0..layer_width {
            if candidate.mask.get_pixel(x, y)[0] == 0
                || base_mask.get_pixel(candidate.left + x, candidate.top + y)[0] == 0
            {
                continue;
            }
            overlap_left = overlap_left.min(x);
            overlap_right = overlap_right.max(x);
            overlap_top = overlap_top.min(y);
            overlap_bottom = overlap_bottom.max(y);
        }
    }
    let patch_left = if overlap_left <= overlap_right {
        overlap_left
    } else {
        seam_left.saturating_sub(radius as u32)
    };
    let patch_right = if overlap_left <= overlap_right {
        overlap_right
    } else {
        (seam_right + radius as u32).min(layer_width - 1)
    };
    let patch_top = if overlap_top <= overlap_bottom {
        overlap_top
    } else {
        seam_top.saturating_sub(radius as u32)
    };
    let patch_bottom = if overlap_top <= overlap_bottom {
        overlap_bottom
    } else {
        (seam_bottom + radius as u32).min(layer_height - 1)
    };
    let patch_width = patch_right - patch_left + 1;
    let patch_height = patch_bottom - patch_top + 1;
    if patch_width < 8 || patch_height < 8 {
        hard_select_focus_layer(base, base_mask, candidate, decision_mask);
        return;
    }

    let patch_pixel_count = patch_width as usize * patch_height as usize;
    let patch_rgb_stride = patch_width as usize * 3;
    let patch_mask_stride = patch_width as usize;
    let mut base_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut candidate_pixels = vec![0.0f32; patch_pixel_count * 3];
    let mut blend_values = vec![0.0f32; patch_pixel_count];
    let mut tone_blend_values = vec![0u8; patch_pixel_count];
    let mut tone_sample_values = vec![0u8; patch_pixel_count];
    let mut protected_values = vec![0u8; patch_pixel_count];
    let mut protected_pixels = vec![0.0f32; patch_pixel_count * 3];
    let base_ref: &Rgb32FImage = base;
    let base_mask_ref: &GrayImage = base_mask;

    base_pixels
        .par_chunks_mut(patch_rgb_stride)
        .zip(candidate_pixels.par_chunks_mut(patch_rgb_stride))
        .zip(blend_values.par_chunks_mut(patch_mask_stride))
        .zip(tone_blend_values.par_chunks_mut(patch_mask_stride))
        .zip(tone_sample_values.par_chunks_mut(patch_mask_stride))
        .zip(protected_values.par_chunks_mut(patch_mask_stride))
        .zip(protected_pixels.par_chunks_mut(patch_rgb_stride))
        .enumerate()
        .for_each(
            |(
                local_y,
                (
                    (
                        ((((base_row, candidate_row), blend_row), tone_blend_row), tone_sample_row),
                        protected_row,
                    ),
                    protected_pixel_row,
                ),
            )| {
                let source_y = patch_top + local_y as u32;
                for local_x in 0..patch_width {
                    let source_x = patch_left + local_x;
                    let global_x = candidate.left + source_x;
                    let global_y = candidate.top + source_y;
                    let base_valid = base_mask_ref.get_pixel(global_x, global_y)[0] > 0;
                    let candidate_valid = candidate.mask.get_pixel(source_x, source_y)[0] > 0;
                    let current_pixel = *base_ref.get_pixel(global_x, global_y);
                    let candidate_pixel = *candidate.image.get_pixel(source_x, source_y);
                    let base_pixel = if base_valid || !candidate_valid {
                        current_pixel
                    } else {
                        candidate_pixel
                    };
                    let candidate_pixel = if candidate_valid {
                        candidate_pixel
                    } else {
                        base_pixel
                    };
                    let owns = candidate_valid
                        && (!base_valid || decision_mask.get_pixel(source_x, source_y)[0] > 0);
                    let candidate_foreground =
                        candidate.foreground_mask.get_pixel(source_x, source_y)[0] > 0;
                    let existing_foreground = foreground_overlap
                        [source_y as usize * layer_width as usize + source_x as usize]
                        > 0;
                    // Black ink, red seals, and other painted content are not part
                    // of the geometric depth mask. They still must not enter the
                    // low-frequency pyramid: averaging two sub-pixel-shifted
                    // strokes is exactly the blur/double-contour regression this
                    // path is intended to prevent. Keep the canvas gate separate
                    // so a darker/lighter paper exposure can still be harmonized.
                    let tone_foreground =
                        focus_stack_pixel_is_tone_foreground(base_pixel.0.as_slice())
                            || focus_stack_pixel_is_tone_foreground(candidate_pixel.0.as_slice());
                    let artwork_foreground =
                        candidate_foreground || existing_foreground || tone_foreground;
                    // Blend the broad tone over the union of the two valid
                    // canvas regions. When only one source exists, the
                    // missing side below is copied from that same source, so
                    // extending the mask is harmless and lets the low
                    // frequency transition continue through a tile edge.
                    // All artwork pixels use hard source ownership, including
                    // their low-frequency bands.
                    tone_blend_row[local_x as usize] =
                        u8::from((base_valid || candidate_valid) && !artwork_foreground);
                    tone_sample_row[local_x as usize] =
                        u8::from(base_valid && candidate_valid && !artwork_foreground);
                    protected_row[local_x as usize] = u8::from(artwork_foreground);
                    let start = local_x as usize * 3;
                    base_row[start..start + 3].copy_from_slice(&base_pixel.0);
                    candidate_row[start..start + 3].copy_from_slice(&candidate_pixel.0);
                    if artwork_foreground {
                        let selected_pixel = if owns { candidate_pixel } else { base_pixel };
                        protected_pixel_row[start..start + 3].copy_from_slice(&selected_pixel.0);
                    }
                    blend_row[local_x as usize] = if owns { 1.0 } else { 0.0 };
                }
            },
        );

    let low_frequency = masked_box_blur_focus_map(
        &blend_values,
        &tone_blend_values,
        patch_width,
        patch_height,
        (radius / 2).max(1),
    );

    // Match only the source-sized tone step measured from the shared canvas
    // samples. This is deliberately a constant correction for this local seam:
    // it is applied to the candidate before the pyramid is built, so it lands
    // in the coarsest tone band while the weave, brush strokes, and all hard
    // selected foreground detail remain in their original frequency bands.
    let tone_sample_count = tone_sample_values
        .iter()
        .filter(|&&value| value > 0)
        .count();
    if tone_sample_count >= FOCUS_SEAM_TONE_MIN_SAMPLES {
        let mut channel_differences = [Vec::new(), Vec::new(), Vec::new()];
        for (index, &sample) in tone_sample_values.iter().enumerate() {
            if sample == 0 {
                continue;
            }
            let start = index * 3;
            for channel in 0..3 {
                channel_differences[channel]
                    .push(base_pixels[start + channel] - candidate_pixels[start + channel]);
            }
        }
        let mut tone_correction = [0.0f32; 3];
        for channel in 0..3 {
            let differences = &mut channel_differences[channel];
            differences.sort_unstable_by(f32::total_cmp);
            tone_correction[channel] = differences[differences.len() / 2].clamp(
                -FOCUS_SEAM_TONE_MAX_ADJUSTMENT,
                FOCUS_SEAM_TONE_MAX_ADJUSTMENT,
            );
        }
        for (index, &protected) in protected_values.iter().enumerate() {
            if protected > 0 {
                continue;
            }
            let local_x = (index % patch_width as usize) as u32;
            let local_y = (index / patch_width as usize) as u32;
            if candidate
                .mask
                .get_pixel(patch_left + local_x, patch_top + local_y)[0]
                == 0
            {
                continue;
            }
            let start = index * 3;
            // `low_frequency` is the smoothed hard ownership mask. Its middle
            // values identify the actual seam; using 4*a*(1-a) keeps the
            // measured correction out of the candidate's remote canvas and
            // out of the base side of the patch. The previous tone-union mask
            // was one across the whole rectangle, which silently turned a
            // large seam bounding box into a full candidate-wide recolour.
            let ownership = low_frequency[index].clamp(0.0, 1.0);
            let weight = (4.0 * ownership * (1.0 - ownership)).clamp(0.0, 1.0);
            if weight <= 0.0 {
                continue;
            }
            for channel in 0..3 {
                candidate_pixels[start + channel] += tone_correction[channel] * weight;
            }
        }
    }

    let blend_mask = GrayImage::from_fn(patch_width, patch_height, |x, y| {
        image::Luma([(blend_values[y as usize * patch_width as usize + x as usize] * 255.0) as u8])
    });
    let low_frequency_mask = GrayImage::from_fn(patch_width, patch_height, |x, y| {
        let index = y as usize * patch_width as usize + x as usize;
        let value = if tone_blend_values[index] > 0 {
            low_frequency[index]
        } else {
            blend_values[index]
        };
        image::Luma([(value.clamp(0.0, 1.0) * 255.0) as u8])
    });
    let blended = multiband_blend(
        Rgb32FImage::from_raw(patch_width, patch_height, base_pixels)
            .expect("focus seam base dimensions must match"),
        Rgb32FImage::from_raw(patch_width, patch_height, candidate_pixels)
            .expect("focus seam candidate dimensions must match"),
        blend_mask,
        Some(low_frequency_mask),
        PANORAMA_BLEND_BANDS,
        true,
    );

    let blended_pixels = blended.as_raw();
    base.as_mut()
        .par_chunks_mut(base_width as usize * 3)
        .enumerate()
        .skip((candidate.top + patch_top) as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, row)| {
            let local_y = global_y - (candidate.top + patch_top) as usize;
            let source_start = local_y * patch_rgb_stride;
            let destination_start = (candidate.left + patch_left) as usize * 3;
            row[destination_start..destination_start + patch_rgb_stride]
                .copy_from_slice(&blended_pixels[source_start..source_start + patch_rgb_stride]);
        });

    // Re-apply hard ownership to the protected layer. This is deliberately
    // after multiband reconstruction: the low-frequency seam mask is allowed
    // to smooth paper illumination, but not to average two displaced copies of
    // a depth-discontinuous object.
    base.as_mut()
        .par_chunks_mut(base_width as usize * 3)
        .enumerate()
        .skip((candidate.top + patch_top) as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, row)| {
            let local_y = global_y - (candidate.top + patch_top) as usize;
            for local_x in 0..patch_width as usize {
                let patch_index = local_y * patch_width as usize + local_x;
                if protected_values[patch_index] == 0 {
                    continue;
                }
                let source_start = patch_index * 3;
                let destination_start = (candidate.left + patch_left + local_x as u32) as usize * 3;
                row[destination_start..destination_start + 3]
                    .copy_from_slice(&protected_pixels[source_start..source_start + 3]);
            }
        });
    base_mask
        .as_mut()
        .par_chunks_mut(base_width as usize)
        .enumerate()
        .skip((candidate.top + patch_top) as usize)
        .take(patch_height as usize)
        .for_each(|(global_y, row)| {
            let local_y = global_y - (candidate.top + patch_top) as usize;
            for local_x in 0..patch_width {
                let source_x = patch_left + local_x;
                if candidate
                    .mask
                    .get_pixel(source_x, patch_top + local_y as u32)[0]
                    > 0
                {
                    row[(candidate.left + source_x) as usize] = 255;
                }
            }
        });

    // Pixels outside the narrow seam patch still need the normal hard ownership
    // decision; only the low-frequency transition is allowed to cross the seam.
    hard_select_focus_layer_outside_patch(
        base,
        base_mask,
        candidate,
        decision_mask,
        patch_left,
        patch_right,
        patch_top,
        patch_bottom,
    );
}

fn hard_select_focus_layer_outside_patch(
    base: &mut Rgb32FImage,
    base_mask: &mut GrayImage,
    candidate: &RenderedFocusLayer,
    decision_mask: &GrayImage,
    excluded_left: u32,
    excluded_right: u32,
    excluded_top: u32,
    excluded_bottom: u32,
) {
    let (layer_width, layer_height) = candidate.image.dimensions();
    if layer_width == 0 || layer_height == 0 {
        return;
    }
    let base_width = base.width();
    let base_height = base.height();
    if candidate.left >= base_width
        || candidate.top >= base_height
        || candidate.left + layer_width > base_width
        || candidate.top + layer_height > base_height
    {
        return;
    }

    let base_stride = base_width as usize * 3;
    let layer_stride = layer_width as usize * 3;
    let base_pixels = base.as_mut();
    let base_mask_pixels = base_mask.as_mut();
    let candidate_pixels = candidate.image.as_raw();
    let candidate_mask_pixels = candidate.mask.as_raw();
    let decision_pixels = decision_mask.as_raw();
    for local_y in 0..layer_height as usize {
        let global_y = candidate.top as usize + local_y;
        let base_row = &mut base_pixels[global_y * base_stride..(global_y + 1) * base_stride];
        let base_mask_row = &mut base_mask_pixels
            [global_y * base_width as usize..(global_y + 1) * base_width as usize];
        let candidate_row = &candidate_pixels[local_y * layer_stride..(local_y + 1) * layer_stride];
        let candidate_mask_row = &candidate_mask_pixels
            [local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
        let decision_row =
            &decision_pixels[local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
        for local_x in 0..layer_width as usize {
            if (excluded_left as usize..=excluded_right as usize).contains(&local_x)
                && (excluded_top as usize..=excluded_bottom as usize).contains(&local_y)
            {
                continue;
            }
            if candidate_mask_row[local_x] == 0 {
                continue;
            }
            let global_x = candidate.left as usize + local_x;
            let base_pixel_start = global_x * 3;
            let candidate_pixel_start = local_x * 3;
            if base_mask_row[global_x] == 0 || decision_row[local_x] > 0 {
                base_row[base_pixel_start..base_pixel_start + 3].copy_from_slice(
                    &candidate_row[candidate_pixel_start..candidate_pixel_start + 3],
                );
            }
            base_mask_row[global_x] = 255;
        }
    }
}

#[derive(Clone, Copy)]
struct MappedFrameGeometry {
    center: Point2<f64>,
    scale: f64,
    rotation: f64,
    area: f64,
}

fn mapped_frame_geometry(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
) -> Option<MappedFrameGeometry> {
    let width = image.width.max(1) as f64;
    let height = image.height.max(1) as f64;
    let source_corners = [
        Point2::new(0.0, 0.0),
        Point2::new(width, 0.0),
        Point2::new(width, height),
        Point2::new(0.0, height),
    ];
    let mut corners = [Point2::new(0.0, 0.0); 4];
    for (slot, source) in source_corners.into_iter().enumerate() {
        let projected = project_point(image, source.x, source.y, projection)?;
        let mapped = homography * Point3::new(projected.x, projected.y, 1.0);
        if mapped.z.abs() < 1e-8 {
            return None;
        }
        let point = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
        if !point.x.is_finite() || !point.y.is_finite() {
            return None;
        }
        corners[slot] = point;
    }

    let center = corners
        .iter()
        .copied()
        .fold(Point2::new(0.0, 0.0), |sum, point| sum + point.coords)
        / 4.0;
    let horizontal_scale =
        ((corners[1] - corners[0]).norm() + (corners[2] - corners[3]).norm()) / (2.0 * width);
    let vertical_scale =
        ((corners[3] - corners[0]).norm() + (corners[2] - corners[1]).norm()) / (2.0 * height);
    let scale = (horizontal_scale.max(0.0) * vertical_scale.max(0.0)).sqrt();
    let top_edge = corners[1] - corners[0];
    let rotation = top_edge.y.atan2(top_edge.x);
    let signed_double_area = corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(4)
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f64>();
    let area = (signed_double_area / (2.0 * width * height)).abs();
    (scale.is_finite() && rotation.is_finite() && area.is_finite()).then_some(MappedFrameGeometry {
        center,
        scale,
        rotation,
        area,
    })
}

fn wrapped_angle_delta(left: f64, right: f64) -> f64 {
    let mut delta = left - right;
    while delta > std::f64::consts::PI {
        delta -= 2.0 * std::f64::consts::PI;
    }
    while delta < -std::f64::consts::PI {
        delta += 2.0 * std::f64::consts::PI;
    }
    delta.abs()
}

fn focus_stack_is_shifted_mosaic(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> bool {
    let Some(first) = images.first() else {
        return false;
    };
    let Some(reference_homography) = global_homographies.get(&first.id) else {
        return false;
    };
    let Some(reference) = mapped_frame_geometry(first, reference_homography, projection) else {
        return false;
    };
    let reference_width = first.width().max(1) as f64;
    let reference_height = first.height().max(1) as f64;
    // Some exported frames lose the equivalent-focal tag. Use the first tag
    // available in the stack as a comparison anchor so a missing tag on the
    // first frame does not hide a later 35mm/85mm switch.
    let reference_focal = first
        .focal_length_35mm
        .or_else(|| images.iter().find_map(|image| image.focal_length_35mm));
    images.iter().skip(1).any(|image| {
        let Some(homography) = global_homographies.get(&image.id) else {
            return false;
        };
        let Some(current) = mapped_frame_geometry(image, homography, projection) else {
            return false;
        };
        let center_shift = (current.center.x - reference.center.x).abs() > reference_width * 0.08
            || (current.center.y - reference.center.y).abs() > reference_height * 0.08;
        let relative_scale = current.scale / reference.scale.max(1e-8);
        let scale_shift = relative_scale.is_finite() && !(0.90..=1.10).contains(&relative_scale);
        let relative_area = current.area / reference.area.max(1e-8);
        let area_shift = relative_area.is_finite() && !(0.82..=1.22).contains(&relative_area);
        let rotation_shift = wrapped_angle_delta(current.rotation, reference.rotation) > 0.035;
        let focal_shift = reference_focal
            .zip(image.focal_length_35mm)
            .map(|(reference_focal, current_focal)| {
                let ratio = current_focal / reference_focal.max(1e-8);
                ratio.is_finite() && !(0.90..=1.10).contains(&ratio)
            })
            .unwrap_or(false);
        center_shift || scale_shift || area_shift || rotation_shift || focal_shift
    })
}

pub fn focus_stack_stitcher<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    capture_group_ids: Option<&HashMap<usize, u8>>,
    sequence_gap_aware: bool,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    if images.is_empty() {
        return Ok(Rgb32FImage::new(0, 0));
    }
    let capture_group_count = capture_group_ids
        .map(|ids| {
            ids.values()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len()
        })
        .unwrap_or(0);
    // Focus breathing and small tripod movement can trip the framing/scale
    // heuristic even when every source belongs to one focus bracket.  The
    // hard, position-owned mosaic is only appropriate when registration has
    // independently recovered multiple camera positions (or an explicit
    // sequence gap).  A single capture group must retain the mature focus
    // fusion path; otherwise its cell-wise local warp visibly fragments
    // strokes that are intentionally present at different focal depths.
    let shifted_mosaic = sequence_gap_aware
        || (capture_group_count > 1
            && focus_stack_is_shifted_mosaic(images, global_homographies, projection));
    if !shifted_mosaic {
        println!(
            "  - Compact focus capture detected ({} inferred group(s)); preserving standard focus fusion",
            capture_group_count.max(1)
        );
    }
    if shifted_mosaic {
        println!("  - Framing/lens shift detected; using local registration and detail ownership");
        let _ = app_handle.emit(
            progress_event,
            "Framing or lens change detected; refining registration and selecting sharp detail...",
        );
        return super::mosaic::detail_preserving_mosaic(
            images,
            global_homographies,
            projection,
            capture_group_ids,
            sequence_gap_aware,
            app_handle,
            progress_event,
            load_image,
        );
    }
    let (min_x, max_x, min_y, max_y) =
        focus_output_bounds(images, global_homographies, projection, focus_warp);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return Ok(Rgb32FImage::new(0, 0));
    }
    let (offset_x, out_width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, out_height) = pixel_aligned_canvas(min_y, max_y);
    let first_source = load_image(images[0])?;
    let first_layer = render_focus_layer(
        images[0],
        &first_source,
        &global_homographies[&images[0].id],
        projection,
        focus_warp,
        offset_x,
        offset_y,
        out_width,
        out_height,
    );
    drop(first_source);
    // The merged image occupies the final canvas, but each subsequent source
    // layer is kept local to its transformed bounds. This is important for a
    // long scan: allocating a full-canvas candidate for every 60MP source can
    // multiply memory use by the number of frames.
    let mut merged = Rgb32FImage::new(out_width, out_height);
    let mut merged_mask = GrayImage::new(out_width, out_height);
    let mut merged_foreground_mask = GrayImage::new(out_width, out_height);
    let mut merged_owner = GrayImage::new(out_width, out_height);
    place_focus_layer(&mut merged, &mut merged_mask, &first_layer);
    place_focus_foreground_mask(&mut merged_foreground_mask, &first_layer);
    merged_owner
        .as_mut()
        .par_chunks_mut(out_width as usize)
        .zip(merged_mask.as_raw().par_chunks(out_width as usize))
        .for_each(|(owner_row, mask_row)| {
            for (owner, mask) in owner_row.iter_mut().zip(mask_row) {
                if *mask > 0 {
                    *owner = 1;
                }
            }
        });
    let source_longest_dimension = images
        .iter()
        .map(|image| image.width().max(image.height()))
        .max()
        .unwrap_or(1);
    let (analysis_width, analysis_height) =
        focus_analysis_dimensions(out_width, out_height, source_longest_dimension);
    let mut merged_analysis = resize_rgb(&merged, analysis_width, analysis_height);
    let mut merged_analysis_mask =
        resize_binary_mask(&merged_mask, analysis_width, analysis_height);
    let mut merged_analysis_foreground_mask =
        resize_binary_mask(&merged_foreground_mask, analysis_width, analysis_height);
    let mut merged_focus = focus_score_map(&merged_analysis, &merged_analysis_mask);

    for (index, image) in images.iter().enumerate().skip(1) {
        let _ = app_handle.emit(
            progress_event,
            format!("Focus-stacking image {} of {}", index + 1, images.len()),
        );
        let source_image = load_image(image)?;
        let mut candidate = render_focus_layer(
            image,
            &source_image,
            &global_homographies[&image.id],
            projection,
            focus_warp,
            offset_x,
            offset_y,
            out_width,
            out_height,
        );
        drop(source_image);
        if let Some((delta_x, delta_y, score, support, patch_count)) =
            estimate_focus_layer_translation(&candidate, &merged, &merged_mask)
        {
            println!(
                "  - Full-resolution focus refinement: delta=({delta_x},{delta_y}), score={score:.3}, consensus={support}/{patch_count} patches"
            );
            candidate =
                translate_rendered_focus_layer(candidate, delta_x, delta_y, out_width, out_height);
        }
        // A regional warp can still leave a small residual translation on a
        // depth-discontinuous layer. Refine only the detected foreground
        // ownership mask against the already selected foreground when the
        // stack is a normal, mostly-overlapping focus capture. In a shifted
        // mosaic the existing mask can contain a different physical instance
        // of a repeated rail/seal, so mask-only matching is not image-backed
        // registration and can move the whole foreground onto the wrong copy.
        // The shifted-mosaic path has already used the bounded edge-consensus
        // geometry above; leave its residuals to that model and the focus
        // ownership pass.
        if !shifted_mosaic {
            candidate =
                align_focus_foreground_layer_to_existing(candidate, &merged_foreground_mask);
        }
        // Match only the slowly varying canvas illumination before focus scoring.
        // The estimator excludes detected foreground and the application is
        // canvas-gated, so brush strokes, seals, and the source weave remain
        // unchanged while the sharpness comparison is not biased by exposure
        // blocks between frames.
        let color_correction = estimate_focus_color_correction(
            &merged,
            &merged_mask,
            &merged_foreground_mask,
            &candidate,
            true,
        );
        apply_focus_color_correction(
            &mut candidate.image,
            &color_correction,
            &candidate.foreground_mask,
            true,
        );
        // Keep the source pixels untouched until focus ownership is decided.
        // The correction above is deliberately low-frequency and canvas-only;
        // final ownership still decides all detail pixels independently.
        let candidate_analysis = render_focus_analysis_layer(
            &candidate,
            out_width,
            out_height,
            analysis_width,
            analysis_height,
        );
        let candidate_focus = focus_score_map(&candidate_analysis.image, &candidate_analysis.mask);
        let mut analysis_decision = focus_decision_mask(
            &merged_focus,
            &candidate_focus,
            &merged_analysis_mask,
            &candidate_analysis.mask,
        );
        // Keep the low-resolution ownership model in lockstep with the final
        // image. If focus scores are allowed to switch a near-field object at
        // analysis resolution and the full-resolution pass vetoes that switch,
        // the next source can still see the stale analysis winner and add a
        // second copy of the object. The mask is deliberately generic and
        // applies to any detected near-field/occlusion layer.
        suppress_focus_foreground_switches_on_canvas(
            &mut analysis_decision,
            &merged_analysis_foreground_mask,
            &candidate_analysis.foreground_mask,
            &candidate_analysis.relaxed_foreground_mask,
        );
        suppress_focus_foreground_ownership_switches_on_canvas(
            &mut analysis_decision,
            &merged_analysis_foreground_mask,
            &candidate_analysis.mask,
            &candidate_analysis.relaxed_foreground_mask,
        );
        suppress_focus_canvas_switches(
            &mut analysis_decision,
            &merged_analysis,
            &candidate_analysis.image,
            &merged_analysis_mask,
            &candidate_analysis.mask,
        );
        let mut full_resolution_decision =
            focus_decision_for_layer(&analysis_decision, &candidate, out_width, out_height);
        // A detected near-field layer is geometrically separate from the main
        // image plane. Do not let focus sharpness make later frames replace an
        // already placed instance of that layer; doing so creates block-shaped
        // brightness and edge jumps in a moving scan. New pixels can still be
        // added where the mosaic has no ownership yet.
        suppress_focus_foreground_switches(
            &mut full_resolution_decision,
            &merged_foreground_mask,
            &candidate,
        );
        suppress_focus_foreground_ownership_switches(
            &mut full_resolution_decision,
            &merged_foreground_mask,
            &candidate.mask,
            candidate.left,
            candidate.top,
            &candidate.relaxed_foreground_mask,
        );
        // A shifted scan can put the ownership boundary through a face, sleeve,
        // or painted contour. Averaging the full-resolution detail there turns
        // two slightly displaced sharp samples into a soft double exposure.
        // The seam path therefore blends only low-frequency canvas tone while
        // keeping detail-band source ownership hard and deterministic.
        let seam_blend_enabled = shifted_mosaic && FOCUS_ALLOW_LOW_FREQUENCY_SEAM_BLEND;
        if seam_blend_enabled {
            blend_focus_seam_band(
                &mut merged,
                &mut merged_mask,
                &merged_foreground_mask,
                &candidate,
                &full_resolution_decision,
            );
        } else {
            hard_select_focus_layer(
                &mut merged,
                &mut merged_mask,
                &candidate,
                &full_resolution_decision,
            );
        }
        mark_focus_foreground_pixels(
            &mut merged_foreground_mask,
            &merged_mask,
            &candidate,
            &full_resolution_decision,
        );
        mark_focus_owner_pixels(
            &mut merged_owner,
            &candidate,
            &full_resolution_decision,
            (index + 2).min(u8::MAX as usize) as u8,
        );
        commit_focus_analysis_scores(
            &mut merged_focus,
            &candidate_focus,
            &candidate_analysis.mask,
            &candidate,
            &full_resolution_decision,
            out_width,
            out_height,
            analysis_width,
            analysis_height,
        );
        // Rebuild the analysis image from the authoritative full-resolution
        // result for the canvas/foreground guards. Focus scores intentionally
        // stay in the separate best-score table above: recomputing them from
        // this image would make a selected seam, a sub-pixel warp, or a cubic
        // resample look like a new focus layer and would reintroduce order-
        // dependent blur.
        merged_analysis = resize_rgb(&merged, analysis_width, analysis_height);
        merged_analysis_mask = resize_binary_mask(&merged_mask, analysis_width, analysis_height);
        merged_analysis_foreground_mask =
            resize_binary_mask(&merged_foreground_mask, analysis_width, analysis_height);
    }
    let merged_dimensions = merged.dimensions();
    let (filled_pixels, remaining_invalid_pixels) =
        fill_focus_canvas_margins(&mut merged, &mut merged_mask);
    println!(
        "  - Kept full focus-stack canvas {}x{}; filled {} invalid margin pixels{}",
        merged_dimensions.0,
        merged_dimensions.1,
        filled_pixels,
        if remaining_invalid_pixels > 0 {
            format!(" ({} remain)", remaining_invalid_pixels)
        } else {
            String::new()
        }
    );
    // A newly exposed canvas region may have no overlap samples from which to
    // estimate a source-specific colour transform. Once ownership is final,
    // remove only the remaining source-sized low-frequency tone steps across
    // the complete mosaic. The strict canvas gate preserves the original
    // weave and painted foreground instead of treating it as a background.
    harmonize_focus_background_tone_with_owners(
        &mut merged,
        &merged_mask,
        &merged_foreground_mask,
        &merged_owner,
    );
    Ok(merged)
}

fn find_adaptive_seam(ctx: &SeamContext) -> Option<SeamInfo> {
    let h_add_inv = ctx.h_add.try_inverse()?;
    let (w_add, h_add_img) = ctx.img_to_add.dimensions();

    let mut min_ox = u32::MAX;
    let mut max_ox = 0;
    let mut min_oy = u32::MAX;
    let mut max_oy = 0;
    let mut has_overlap = false;

    let (candidate_left, candidate_right, candidate_top, candidate_bottom) =
        transformed_image_region(
            ctx.img_to_add_info,
            ctx.h_add,
            ctx.projection,
            ctx.offset_x,
            ctx.offset_y,
            ctx.out_width,
            ctx.out_height,
        )?;
    for y in candidate_top..=candidate_bottom {
        for x in candidate_left..=candidate_right {
            if ctx.pano_mask.get_pixel(x, y)[0] > 0 {
                let target_p = Point3::new(x as f64 - ctx.offset_x, y as f64 - ctx.offset_y, 1.0);
                let Some(source) =
                    map_target_to_source(&h_add_inv, target_p, ctx.img_to_add_info, ctx.projection)
                else {
                    continue;
                };
                let sx = source.x;
                let sy = source.y;
                if sx >= 0.0 && sx < w_add as f64 && sy >= 0.0 && sy < h_add_img as f64 {
                    has_overlap = true;
                    min_ox = min_ox.min(x);
                    max_ox = max_ox.max(x);
                    min_oy = min_oy.min(y);
                    max_oy = max_oy.max(y);
                }
            }
        }
    }

    if !has_overlap {
        return None;
    }

    println!(
        "    - Overlap bounds: x={}..{}, y={}..{}",
        min_ox, max_ox, min_oy, max_oy
    );

    let center_source = project_point(
        ctx.img_to_add_info,
        w_add as f64 / 2.0,
        h_add_img as f64 / 2.0,
        ctx.projection,
    )?;
    let center_p_source = Point3::new(center_source.x, center_source.y, 1.0);
    let center_p_target = ctx.h_add * center_p_source;
    let center_add_x = (center_p_target.x / center_p_target.z) + ctx.offset_x;
    let center_add_y = (center_p_target.y / center_p_target.z) + ctx.offset_y;

    let center_overlap_x = (min_ox + max_ox) as f64 / 2.0;
    let center_overlap_y = (min_oy + max_oy) as f64 / 2.0;

    let dx = center_add_x - center_overlap_x;
    let dy = center_add_y - center_overlap_y;

    if dx.abs() > dy.abs() {
        println!("    - Overlap is vertical. Finding vertical seam...");
        let seam = find_pairwise_seam_dp(
            ctx,
            SeamOrientation::Vertical,
            min_ox,
            max_ox,
            min_oy,
            max_oy,
        );
        if let Some(active) = seam.get(min_oy as usize..=max_oy as usize) {
            let minimum = active.iter().copied().min().unwrap_or_default();
            let maximum = active.iter().copied().max().unwrap_or_default();
            println!("    - Vertical seam range: x={minimum}..{maximum}");
        }
        Some(SeamInfo {
            orientation: SeamOrientation::Vertical,
            coords: seam,
            dx,
            dy,
            min_x: min_ox,
            max_x: max_ox,
            min_y: min_oy,
            max_y: max_oy,
        })
    } else {
        println!("    - Overlap is horizontal. Finding horizontal seam...");
        let seam = find_pairwise_seam_dp(
            ctx,
            SeamOrientation::Horizontal,
            min_ox,
            max_ox,
            min_oy,
            max_oy,
        );
        if let Some(active) = seam.get(min_ox as usize..=max_ox as usize) {
            let minimum = active.iter().copied().min().unwrap_or_default();
            let maximum = active.iter().copied().max().unwrap_or_default();
            println!("    - Horizontal seam range: y={minimum}..{maximum}");
        }
        Some(SeamInfo {
            orientation: SeamOrientation::Horizontal,
            coords: seam,
            dx,
            dy,
            min_x: min_ox,
            max_x: max_ox,
            min_y: min_oy,
            max_y: max_oy,
        })
    }
}

fn seam_energy_at(ctx: &SeamContext, h_add_inv: &Matrix3<f64>, x: u32, y: u32) -> Option<f64> {
    if ctx.pano_mask.get_pixel(x, y)[0] == 0 {
        return None;
    }
    let target = Point3::new(x as f64 - ctx.offset_x, y as f64 - ctx.offset_y, 1.0);
    let source = map_target_to_source(h_add_inv, target, ctx.img_to_add_info, ctx.projection)?;
    if source.x < 0.0
        || source.y < 0.0
        || source.x >= ctx.img_to_add.width() as f64 - 1.0
        || source.y >= ctx.img_to_add.height() as f64 - 1.0
    {
        return None;
    }
    let base = ctx.pano.get_pixel(x, y);
    let candidate = apply_exposure_gain(
        get_interpolated_pixel(ctx.img_to_add, source.x, source.y),
        ctx.exposure.gain_at(x, y),
    );
    Some(
        ((base[0] as f64 - candidate[0] as f64).powi(2)
            + (base[1] as f64 - candidate[1] as f64).powi(2)
            + (base[2] as f64 - candidate[2] as f64).powi(2))
        .sqrt(),
    )
}

fn find_pairwise_seam_dp(
    ctx: &SeamContext,
    orientation: SeamOrientation,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
) -> Vec<i32> {
    const MAX_GRID_DIMENSION: u32 = 2_400;
    let (along_min, along_max, cross_min, cross_max, output_length) = match orientation {
        SeamOrientation::Vertical => (min_y, max_y, min_x, max_x, ctx.out_height),
        SeamOrientation::Horizontal => (min_x, max_x, min_y, max_y, ctx.out_width),
    };
    if along_min > along_max || cross_min > cross_max {
        return Vec::new();
    }

    let along_span = along_max - along_min;
    let cross_span = cross_max - cross_min;
    let step = along_span
        .max(cross_span)
        .div_ceil(MAX_GRID_DIMENSION)
        .max(1);
    let along_count = along_span.div_ceil(step) as usize + 1;
    let cross_count = cross_span.div_ceil(step) as usize + 1;
    if along_count < 2 || cross_count < 2 {
        return Vec::new();
    }
    println!(
        "    - Seam search grid: {}x{} ({}px sampling)",
        cross_count, along_count, step
    );

    let coordinate = |minimum: u32, maximum: u32, index: usize| {
        minimum
            .saturating_add((index as u32).saturating_mul(step))
            .min(maximum)
    };
    let Some(h_add_inv) = ctx.h_add.try_inverse() else {
        return Vec::new();
    };
    let mut previous = vec![f64::INFINITY; cross_count];
    let mut current = vec![f64::INFINITY; cross_count];
    let mut predecessors = vec![i8::MAX; along_count * cross_count];
    let mut last_active_index = None;
    let mut last_active_costs = Vec::new();

    for along_index in 0..along_count {
        current.fill(f64::INFINITY);
        let has_previous_path = previous.iter().any(|cost| cost.is_finite());
        let along = coordinate(along_min, along_max, along_index);
        for cross_index in 0..cross_count {
            let cross = coordinate(cross_min, cross_max, cross_index);
            let (x, y) = match orientation {
                SeamOrientation::Vertical => (cross, along),
                SeamOrientation::Horizontal => (along, cross),
            };
            let Some(mut energy) = seam_energy_at(ctx, &h_add_inv, x, y) else {
                continue;
            };

            // Discourage paths that merely trace a warped image border. Such paths make
            // rectangular exposure changes visible even when the geometry is correct.
            let edge_distance = cross_index.min(cross_count - 1 - cross_index) as f64;
            energy += (6.0 - edge_distance).max(0.0) * 0.01;

            if !has_previous_path {
                current[cross_index] = energy;
                predecessors[along_index * cross_count + cross_index] = 2;
                continue;
            }
            let first_neighbor = cross_index.saturating_sub(1);
            let last_neighbor = (cross_index + 1).min(cross_count - 1);
            let mut best_previous = f64::INFINITY;
            let mut best_index = cross_index;
            for (previous_index, &previous_cost) in previous
                .iter()
                .enumerate()
                .take(last_neighbor + 1)
                .skip(first_neighbor)
            {
                if previous_cost < best_previous {
                    best_previous = previous_cost;
                    best_index = previous_index;
                }
            }
            if best_previous.is_finite() {
                current[cross_index] = best_previous + energy;
                predecessors[along_index * cross_count + cross_index] =
                    (best_index as i32 - cross_index as i32) as i8;
            }
        }
        std::mem::swap(&mut previous, &mut current);
        if previous.iter().any(|cost| cost.is_finite()) {
            last_active_index = Some(along_index);
            last_active_costs.clone_from(&previous);
        }
    }

    let Some(last_active_index) = last_active_index else {
        return Vec::new();
    };
    let Some((mut current_cross, _)) = last_active_costs
        .iter()
        .enumerate()
        .filter(|(_, cost)| cost.is_finite())
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Vec::new();
    };
    let mut sampled_path = vec![0usize; along_count];
    sampled_path[last_active_index] = current_cross;
    let mut first_active_index = last_active_index;
    for along_index in (1..=last_active_index).rev() {
        let predecessor = predecessors[along_index * cross_count + current_cross];
        if predecessor == 2 {
            first_active_index = along_index;
            break;
        }
        if predecessor == i8::MAX {
            return Vec::new();
        }
        current_cross =
            (current_cross as i32 + predecessor as i32).clamp(0, cross_count as i32 - 1) as usize;
        sampled_path[along_index - 1] = current_cross;
        first_active_index = along_index - 1;
    }
    for index in 0..first_active_index {
        sampled_path[index] = sampled_path[first_active_index];
    }
    for index in (last_active_index + 1)..along_count {
        sampled_path[index] = sampled_path[last_active_index];
    }

    let sampled_cross: Vec<f64> = sampled_path
        .iter()
        .map(|&index| coordinate(cross_min, cross_max, index) as f64)
        .collect();
    let mut seam = vec![sampled_cross[0].round() as i32; output_length as usize];
    for along in along_min..=along_max {
        let relative = along - along_min;
        let lower_index = ((relative / step) as usize).min(along_count - 2);
        let lower_along = coordinate(along_min, along_max, lower_index);
        let upper_along = coordinate(along_min, along_max, lower_index + 1);
        let interpolation = if upper_along == lower_along {
            0.0
        } else {
            (along - lower_along) as f64 / (upper_along - lower_along) as f64
        };
        let cross = sampled_cross[lower_index] * (1.0 - interpolation)
            + sampled_cross[lower_index + 1] * interpolation;
        seam[along as usize] = cross.round() as i32;
    }
    let last_cross = sampled_cross.last().copied().unwrap_or(sampled_cross[0]);
    for value in seam.iter_mut().skip(along_max as usize + 1) {
        *value = last_cross.round() as i32;
    }
    seam
}

pub fn warp_image_homography(
    source: &Rgb32FImage,
    homography: &Matrix3<f64>,
    width: u32,
    height: u32,
) -> Rgb32FImage {
    assert!(width > 0 && height > 0, "warp output must be non-empty");
    let mut buffer = vec![0.0f32; (width as usize) * (height as usize) * 3];
    buffer
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let mapped = homography * Point3::new(x as f64, y as f64, 1.0);
                let pixel = if mapped.z.abs() < 1e-8 {
                    Rgb([0.0, 0.0, 0.0])
                } else {
                    get_high_quality_interpolated_pixel(
                        source,
                        mapped.x / mapped.z,
                        mapped.y / mapped.z,
                    )
                };
                let base = x as usize * 3;
                row[base] = pixel[0];
                row[base + 1] = pixel[1];
                row[base + 2] = pixel[2];
            }
        });
    Rgb32FImage::from_raw(width, height, buffer)
        .expect("warp buffer dimensions must match output image")
}

fn get_interpolated_pixel(img: &Rgb32FImage, x: f64, y: f64) -> Rgb<f32> {
    let (width, height) = img.dimensions();
    let x_floor = x.floor() as u32;
    let y_floor = y.floor() as u32;
    if x_floor + 1 >= width || y_floor + 1 >= height || x < 0.0 || y < 0.0 {
        return *img.get_pixel(
            x.max(0.0).min(width as f64 - 1.0) as u32,
            y.max(0.0).min(height as f64 - 1.0) as u32,
        );
    }
    let dx = x - x_floor as f64;
    let dy = y - y_floor as f64;
    let p00 = img.get_pixel(x_floor, y_floor);
    let p10 = img.get_pixel(x_floor + 1, y_floor);
    let p01 = img.get_pixel(x_floor, y_floor + 1);
    let p11 = img.get_pixel(x_floor + 1, y_floor + 1);
    let mut final_pixel = [0.0; 3];
    for i in 0..3 {
        let c00 = p00[i] as f64;
        let c10 = p10[i] as f64;
        let c01 = p01[i] as f64;
        let c11 = p11[i] as f64;
        let top = c00 * (1.0 - dx) + c10 * dx;
        let bottom = c01 * (1.0 - dx) + c11 * dx;
        final_pixel[i] = top * (1.0 - dy) + bottom * dy;
    }
    Rgb([
        final_pixel[0] as f32,
        final_pixel[1] as f32,
        final_pixel[2] as f32,
    ])
}

fn cubic_sample(p0: f64, p1: f64, p2: f64, p3: f64, amount: f64) -> f64 {
    p1 + 0.5
        * amount
        * (p2 - p0
            + amount * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + amount * (3.0 * (p1 - p2) + p3 - p0)))
}

pub(super) fn get_high_quality_interpolated_pixel(img: &Rgb32FImage, x: f64, y: f64) -> Rgb<f32> {
    let (width, height) = img.dimensions();
    if width < 4
        || height < 4
        || x < 1.0
        || y < 1.0
        || x >= width as f64 - 2.0
        || y >= height as f64 - 2.0
    {
        return get_interpolated_pixel(img, x, y);
    }

    let x_floor = x.floor() as i32;
    let y_floor = y.floor() as i32;
    let amount_x = x - x_floor as f64;
    let amount_y = y - y_floor as f64;
    if amount_x.abs() < 1e-9 && amount_y.abs() < 1e-9 {
        return *img.get_pixel(x_floor as u32, y_floor as u32);
    }

    let mut output = [0.0f32; 3];
    for (channel, output_channel) in output.iter_mut().enumerate() {
        let mut rows = [0.0f64; 4];
        let mut local_min = f64::INFINITY;
        let mut local_max = f64::NEG_INFINITY;
        for (row_index, sample_y) in ((y_floor - 1)..=(y_floor + 2)).enumerate() {
            let mut samples = [0.0f64; 4];
            for (column_index, sample_x) in ((x_floor - 1)..=(x_floor + 2)).enumerate() {
                let value = img.get_pixel(sample_x as u32, sample_y as u32)[channel] as f64;
                samples[column_index] = value;
                local_min = local_min.min(value);
                local_max = local_max.max(value);
            }
            rows[row_index] =
                cubic_sample(samples[0], samples[1], samples[2], samples[3], amount_x);
        }
        *output_channel = cubic_sample(rows[0], rows[1], rows[2], rows[3], amount_y)
            .clamp(local_min, local_max) as f32;
    }
    Rgb(output)
}

#[cfg(test)]
mod interpolation_tests {
    use super::*;

    fn geometry_test_image(id: usize, focal_length_35mm: Option<f64>) -> ImageInfo {
        ImageInfo {
            id,
            filename: format!("geometry-{id}.jpg"),
            width: 1_000,
            height: 800,
            alignment_image: GrayImage::new(1, 1),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        }
    }

    #[test]
    fn pixel_aligned_canvas_preserves_integer_reference_coordinates() {
        let (offset, size) = pixel_aligned_canvas(-123.4, 876.2);
        assert_eq!(offset, 124.0);
        assert_eq!(size, 1001);
        assert_eq!(offset.fract(), 0.0);
        assert!(offset - 123.4 >= 0.0);
    }

    #[test]
    fn cubic_warp_sampling_preserves_more_edge_contrast_than_bilinear() {
        let image = Rgb32FImage::from_fn(8, 8, |x, _| {
            let value = if x < 4 { 0.0 } else { 1.0 };
            Rgb([value, value, value])
        });
        let bilinear_low = get_interpolated_pixel(&image, 3.25, 3.5)[0];
        let bilinear_high = get_interpolated_pixel(&image, 3.75, 3.5)[0];
        let cubic_low = get_high_quality_interpolated_pixel(&image, 3.25, 3.5)[0];
        let cubic_high = get_high_quality_interpolated_pixel(&image, 3.75, 3.5)[0];

        assert!(cubic_high - cubic_low > bilinear_high - bilinear_low);
        assert!((0.0..=1.0).contains(&cubic_low));
        assert!((0.0..=1.0).contains(&cubic_high));
    }

    #[test]
    fn focus_box_blur_preserves_a_constant_score_map() {
        let source = vec![3.5f32; 5 * 4];
        let blurred = box_blur_focus_map(&source, 5, 4, 2);

        assert!(blurred.iter().all(|value| (*value - 3.5).abs() < 1e-6));
    }

    #[test]
    fn focus_analysis_keeps_source_sampling_density_on_long_mosaics() {
        let (analysis_width, analysis_height) = focus_analysis_dimensions(31_121, 11_034, 9_504);

        assert!(analysis_width >= 4_500);
        assert!(analysis_height >= 1_500);
        assert!(
            u64::from(analysis_width) * u64::from(analysis_height) <= FOCUS_ANALYSIS_MAX_PIXELS
        );
    }

    #[test]
    fn shifted_mosaic_detection_catches_a_lens_scale_change() {
        let first = geometry_test_image(0, Some(35.0));
        let second = geometry_test_image(1, Some(85.0));
        let center_x = first.width as f64 * 0.5;
        let center_y = first.height as f64 * 0.5;
        let scale = 1.25;
        let scale_about_center = Matrix3::new(
            scale,
            0.0,
            center_x * (1.0 - scale),
            0.0,
            scale,
            center_y * (1.0 - scale),
            0.0,
            0.0,
            1.0,
        );
        let homographies = HashMap::from([(0, Matrix3::identity()), (1, scale_about_center)]);
        let images = vec![&first, &second];

        assert!(focus_stack_is_shifted_mosaic(
            &images,
            &homographies,
            Projection::Planar
        ));
    }

    #[test]
    fn shifted_mosaic_detection_uses_focal_metadata_when_geometry_is_centered() {
        let first = geometry_test_image(0, Some(35.0));
        let second = geometry_test_image(1, Some(85.0));
        let homographies = HashMap::from([(0, Matrix3::identity()), (1, Matrix3::identity())]);
        let images = vec![&first, &second];

        assert!(focus_stack_is_shifted_mosaic(
            &images,
            &homographies,
            Projection::Planar
        ));
    }

    #[test]
    fn shifted_mosaic_detection_keeps_small_focus_breathing_on_the_focus_path() {
        let first = geometry_test_image(0, Some(85.0));
        let second = geometry_test_image(1, Some(85.0));
        let center_x = first.width as f64 * 0.5;
        let center_y = first.height as f64 * 0.5;
        let scale = 1.025;
        let small_breathing = Matrix3::new(
            scale,
            0.0,
            center_x * (1.0 - scale) + 18.0,
            0.0,
            scale,
            center_y * (1.0 - scale) + 12.0,
            0.0,
            0.0,
            1.0,
        );
        let homographies = HashMap::from([(0, Matrix3::identity()), (1, small_breathing)]);
        let images = vec![&first, &second];

        assert!(!focus_stack_is_shifted_mosaic(
            &images,
            &homographies,
            Projection::Planar
        ));
    }

    #[test]
    fn parallel_pyramid_downsample_preserves_constant_pixels_and_odd_edges() {
        let source = Rgb32FImage::from_pixel(5, 7, Rgb([0.2, 0.4, 0.8]));
        let downsampled = downsample_rgb_half(&source);

        assert_eq!(downsampled.dimensions(), (3, 4));
        assert!(downsampled.pixels().all(|pixel| pixel.0 == [0.2, 0.4, 0.8]));
    }

    #[test]
    fn parallel_pyramid_detail_reconstructs_the_original_pixels() {
        let source = Rgb32FImage::from_fn(17, 11, |x, y| {
            let value = (x as f32 * 0.07 + y as f32 * 0.03).sin();
            Rgb([value, value * 0.5, 1.0 - value])
        });
        let coarse = downsample_rgb_half(&source);
        let detail = subtract_upsampled_rgb(&source, &coarse);
        let reconstructed = upsample_and_add_rgb(&coarse, &detail);

        assert_eq!(reconstructed.dimensions(), source.dimensions());
        let maximum_error = reconstructed
            .as_raw()
            .iter()
            .zip(source.as_raw())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(maximum_error < 1e-6, "maximum error: {maximum_error}");
    }

    #[test]
    fn panorama_detail_feather_is_local_and_symmetric() {
        let radius = PANORAMA_DETAIL_SEAM_FEATHER_RADIUS;

        assert_eq!(panorama_detail_alpha(f32::NEG_INFINITY), 0.0);
        assert_eq!(panorama_detail_alpha(-radius), 0.0);
        assert_eq!(panorama_detail_alpha(0.0), 0.5);
        assert_eq!(panorama_detail_alpha(radius), 1.0);
        assert_eq!(panorama_detail_alpha(f32::INFINITY), 1.0);
    }

    #[test]
    fn panorama_global_tone_blend_preserves_medium_frequency_detail() {
        const WIDTH: u32 = 256;
        const HEIGHT: u32 = 64;
        let pattern = |x: u32| if (x / 4).is_multiple_of(2) { 0.2 } else { 0.8 };
        let base = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
            let value = pattern(x);
            Rgb([value, value, value])
        });
        let candidate = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
            let value = (pattern((x + 4).min(WIDTH - 1)) + 0.08).min(1.0);
            Rgb([value, value, value])
        });
        let seam_mask = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
            image::Luma([if x >= WIDTH / 2 { 255 } else { 0 }])
        });
        let tone_mask = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
            image::Luma([((x as f32 / (WIDTH - 1) as f32) * 255.0).round() as u8])
        });
        let blended = multiband_blend(
            base.clone(),
            candidate,
            seam_mask,
            Some(tone_mask),
            PANORAMA_BLEND_BANDS,
            false,
        );
        let horizontal_variation = |image: &Rgb32FImage| {
            (32..96)
                .map(|x| {
                    (image.get_pixel(x, HEIGHT / 2)[0] - image.get_pixel(x - 1, HEIGHT / 2)[0])
                        .abs()
                })
                .sum::<f32>()
        };

        let source_variation = horizontal_variation(&base);
        let blended_variation = horizontal_variation(&blended);
        assert!(
            blended_variation >= source_variation * 0.9,
            "medium-frequency contrast fell from {source_variation:.3} to {blended_variation:.3}",
        );
    }

    #[test]
    fn focus_decision_prefers_the_locally_sharper_layer() {
        let base_focus = vec![0.9, 0.8, 0.1, 0.1];
        let candidate_focus = vec![0.1, 0.1, 0.8, 0.9];
        let base_mask = GrayImage::from_pixel(4, 1, image::Luma([255]));
        let candidate_mask = GrayImage::from_pixel(4, 1, image::Luma([255]));

        let decision =
            focus_decision_mask(&base_focus, &candidate_focus, &base_mask, &candidate_mask);

        assert_eq!(decision.as_raw(), &[0, 0, 255, 255]);
    }

    #[test]
    fn focus_decision_extends_clear_structure_over_an_adjacent_halo() {
        let base_focus = vec![0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.2];
        let candidate_focus = vec![0.1, 0.1, 0.1, 0.8, 0.8, 0.22, 0.2];
        let base_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));
        let candidate_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));

        let decision =
            focus_decision_mask(&base_focus, &candidate_focus, &base_mask, &candidate_mask);

        assert_eq!(decision.as_raw()[2], 0);
        assert_eq!(decision.as_raw()[4], 255);
        assert_eq!(decision.as_raw()[5], 255);
    }

    #[test]
    fn focus_decision_protects_base_structure_symmetrically() {
        let base_focus = vec![0.1, 0.1, 0.8, 0.8, 0.22, 0.2, 0.2];
        let candidate_focus = vec![0.9, 0.9, 0.1, 0.1, 0.2, 0.2, 0.2];
        let base_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));
        let candidate_mask = GrayImage::from_pixel(7, 1, image::Luma([255]));

        let decision =
            focus_decision_mask(&base_focus, &candidate_focus, &base_mask, &candidate_mask);

        assert_eq!(decision.as_raw()[0], 255);
        assert_eq!(decision.as_raw()[2], 0);
        assert_eq!(decision.as_raw()[4], 0);
    }

    #[test]
    fn focus_canvas_switch_suppression_keeps_canvas_owner_but_allows_ink() {
        let mut decision = GrayImage::from_pixel(2, 1, image::Luma([255]));
        let base = Rgb32FImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgb([0.32, 0.28, 0.24])
            } else {
                Rgb([0.05, 0.05, 0.05])
            }
        });
        let candidate = Rgb32FImage::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgb([0.40, 0.36, 0.32])
            } else {
                Rgb([0.01, 0.01, 0.01])
            }
        });
        let mask = GrayImage::from_pixel(2, 1, image::Luma([255]));

        suppress_focus_canvas_switches(&mut decision, &base, &candidate, &mask, &mask);

        assert_eq!(decision.as_raw(), &[0, 255]);
    }

    #[test]
    fn focus_seam_keeps_painted_detail_on_one_source() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 64;
        let canvas = Rgb([0.32, 0.28, 0.24]);
        let candidate_canvas = Rgb([0.40, 0.36, 0.32]);
        let mut base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, canvas);
        let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
            if x == 32 && (16..48).contains(&y) {
                Rgb([0.01, 0.01, 0.01])
            } else {
                candidate_canvas
            }
        });
        let candidate = RenderedFocusLayer {
            image: candidate_image,
            mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            foreground_mask: GrayImage::new(WIDTH, HEIGHT),
            relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
            left: 0,
            top: 0,
        };
        let mut base_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
        let merged_foreground_mask = GrayImage::new(WIDTH, HEIGHT);
        let decision = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
            image::Luma([u8::from(x >= WIDTH / 2) * 255])
        });

        blend_focus_seam_band(
            &mut base,
            &mut base_mask,
            &merged_foreground_mask,
            &candidate,
            &decision,
        );

        assert_eq!(base.get_pixel(32, 32).0, [0.01, 0.01, 0.01]);
    }

    #[test]
    fn hard_focus_selection_copies_source_pixels_without_averaging() {
        let mut base =
            Rgb32FImage::from_raw(3, 1, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
        let candidate =
            Rgb32FImage::from_raw(3, 1, vec![0.0, 0.0, 1.0, 0.2, 0.3, 0.9, 0.7, 0.1, 0.8]).unwrap();
        let mut base_mask = GrayImage::from_raw(3, 1, vec![255, 255, 0]).unwrap();
        let candidate_mask = GrayImage::from_pixel(3, 1, image::Luma([255]));
        let decision = GrayImage::from_raw(3, 1, vec![0, 255, 0]).unwrap();

        hard_select_focus_pixels(
            &mut base,
            &mut base_mask,
            &candidate,
            &candidate_mask,
            &decision,
        );

        assert_eq!(base.get_pixel(0, 0).0, [1.0, 0.0, 0.0]);
        assert_eq!(base.get_pixel(1, 0).0, [0.2, 0.3, 0.9]);
        assert_eq!(base.get_pixel(2, 0).0, [0.7, 0.1, 0.8]);
        assert_eq!(base_mask.as_raw(), &[255, 255, 255]);
    }

    #[test]
    fn focus_color_correction_removes_a_measured_channel_gain() {
        let base = Rgb32FImage::from_pixel(40, 16, Rgb([0.4, 0.5, 0.6]));
        let mut candidate_image = Rgb32FImage::from_pixel(32, 16, Rgb([0.5, 0.625, 0.75]));
        let base_mask = GrayImage::from_pixel(40, 16, image::Luma([255]));
        let candidate = RenderedFocusLayer {
            image: candidate_image.clone(),
            mask: GrayImage::from_pixel(32, 16, image::Luma([255])),
            foreground_mask: GrayImage::new(32, 16),
            relaxed_foreground_mask: GrayImage::new(32, 16),
            left: 4,
            top: 0,
        };

        let correction = estimate_focus_color_correction(
            &base,
            &base_mask,
            &GrayImage::new(40, 16),
            &candidate,
            false,
        );
        assert!(
            correction
                .gains
                .iter()
                .all(|gain| (*gain - 0.8).abs() < 0.01)
        );

        apply_focus_color_correction(
            &mut candidate_image,
            &correction,
            &GrayImage::new(32, 16),
            false,
        );
        let corrected = candidate_image.get_pixel(0, 0);
        assert!((corrected[0] - 0.4).abs() < 0.01);
        assert!((corrected[1] - 0.5).abs() < 0.01);
        assert!((corrected[2] - 0.6).abs() < 0.01);
    }

    #[test]
    fn focus_color_correction_uses_only_consensus_foreground_overlap() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 32;
        let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.4, 0.4, 0.4]));
        let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |_, y| {
            if y < 24 {
                Rgb([0.5, 0.5, 0.5])
            } else {
                Rgb([0.4, 0.4, 0.4])
            }
        });
        let foreground_mask = GrayImage::from_fn(WIDTH, HEIGHT, |_, y| {
            image::Luma([if y < 24 { 255 } else { 0 }])
        });
        let candidate = RenderedFocusLayer {
            image: candidate_image,
            mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            foreground_mask: foreground_mask.clone(),
            relaxed_foreground_mask: foreground_mask.clone(),
            left: 0,
            top: 0,
        };

        let correction = estimate_focus_color_correction(
            &base,
            &GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            &foreground_mask,
            &candidate,
            false,
        );

        assert!(
            correction
                .gains
                .iter()
                .all(|gain| (*gain - 0.8).abs() < 0.01)
        );
    }

    #[test]
    fn focus_color_correction_can_use_bright_consensus_edge_pixels() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 32;
        let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.94, 0.94, 0.94]));
        let candidate = RenderedFocusLayer {
            image: Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.98, 0.98, 0.98])),
            mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            foreground_mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            relaxed_foreground_mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            left: 0,
            top: 0,
        };

        let correction = estimate_focus_color_correction(
            &base,
            &GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            &GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            &candidate,
            false,
        );

        assert!(
            correction
                .gains
                .iter()
                .all(|gain| (*gain - (0.94 / 0.98)).abs() < 0.01)
        );
    }

    #[test]
    fn focus_color_correction_reduces_a_broad_local_gain_step() {
        const WIDTH: u32 = 1_536;
        const HEIGHT: u32 = 32;
        let base = Rgb32FImage::from_pixel(WIDTH, HEIGHT, Rgb([0.4, 0.5, 0.6]));
        let candidate_image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, _| {
            if x < WIDTH / 2 {
                Rgb([0.5, 0.625, 0.75])
            } else {
                Rgb([0.4, 0.5, 0.6])
            }
        });
        let candidate = RenderedFocusLayer {
            image: candidate_image.clone(),
            mask: GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255])),
            foreground_mask: GrayImage::new(WIDTH, HEIGHT),
            relaxed_foreground_mask: GrayImage::new(WIDTH, HEIGHT),
            left: 0,
            top: 0,
        };
        let base_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
        let correction = estimate_focus_color_correction(
            &base,
            &base_mask,
            &GrayImage::new(WIDTH, HEIGHT),
            &candidate,
            false,
        );
        assert!(correction.spatial_gains.is_some());

        let mut corrected = candidate_image;
        apply_focus_color_correction(
            &mut corrected,
            &correction,
            &GrayImage::new(WIDTH, HEIGHT),
            false,
        );
        let input_step = 0.5f32 - 0.4;
        let corrected_step =
            (corrected.get_pixel(100, 10)[0] - corrected.get_pixel(1_300, 10)[0]).abs();
        assert!(corrected_step < input_step * 0.8);
    }

    #[test]
    fn focus_background_tone_harmonization_reduces_step_without_touching_foreground() {
        const WIDTH: u32 = 192;
        const HEIGHT: u32 = 96;
        let mut image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
            if (80..112).contains(&x) && (32..64).contains(&y) {
                Rgb([0.65, 0.10, 0.08])
            } else if x < WIDTH / 2 {
                Rgb([0.32, 0.28, 0.24])
            } else {
                Rgb([0.40, 0.36, 0.32])
            }
        });
        let original_foreground_pixel = *image.get_pixel(96, 48);
        let image_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
        let foreground_mask = GrayImage::from_fn(WIDTH, HEIGHT, |x, y| {
            image::Luma([u8::from((80..112).contains(&x) && (32..64).contains(&y)) * 255])
        });

        let input_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();
        harmonize_focus_background_tone(&mut image, &image_mask, &foreground_mask);
        let output_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();

        assert!(output_step < input_step * 0.5);
        assert_eq!(*image.get_pixel(96, 48), original_foreground_pixel);
    }

    #[test]
    fn focus_owner_tone_harmonization_matches_source_blocks_without_touching_detail() {
        const WIDTH: u32 = 192;
        const HEIGHT: u32 = 96;
        let mut image = Rgb32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
            if (80..112).contains(&x) && (32..64).contains(&y) {
                Rgb([0.65, 0.10, 0.08])
            } else if x < WIDTH / 2 {
                Rgb([0.32, 0.28, 0.24])
            } else {
                Rgb([0.40, 0.36, 0.32])
            }
        });
        let original_foreground_pixel = *image.get_pixel(96, 48);
        let image_mask = GrayImage::from_pixel(WIDTH, HEIGHT, image::Luma([255]));
        let foreground_mask = GrayImage::new(WIDTH, HEIGHT);
        let owner_map = GrayImage::from_fn(WIDTH, HEIGHT, |x, _| {
            image::Luma([if x < WIDTH / 2 { 1 } else { 2 }])
        });

        let input_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();
        harmonize_focus_background_tone_with_owners(
            &mut image,
            &image_mask,
            &foreground_mask,
            &owner_map,
        );
        let output_step = (image.get_pixel(20, 48)[0] - image.get_pixel(172, 48)[0]).abs();

        assert!(output_step < input_step * 0.5);
        assert_eq!(*image.get_pixel(96, 48), original_foreground_pixel);
    }

    #[test]
    fn focus_canvas_margin_fill_keeps_the_full_canvas_and_extends_texture() {
        let mut image = Rgb32FImage::from_fn(8, 6, |x, y| {
            let value = (x as f32 * 0.07 + y as f32 * 0.11).fract();
            Rgb([value, value * 0.8, value * 0.6])
        });
        let mut mask = GrayImage::new(8, 6);
        for y in 1..5 {
            for x in 1..7 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }

        let dimensions = image.dimensions();
        let (filled, remaining) = fill_focus_canvas_margins(&mut image, &mut mask);

        assert_eq!(image.dimensions(), dimensions);
        assert!(filled > 0);
        assert_eq!(remaining, 0);
        assert!(mask.as_raw().iter().all(|&value| value > 0));
        assert!(image.get_pixel(0, 0).0.iter().any(|&value| value != 0.0));
    }
}
