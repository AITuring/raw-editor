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
const FOCUS_ANALYSIS_MAX_DIMENSION: u32 = 1536;
const FOCUS_ANALYSIS_MAX_PIXELS: u64 = 24_000_000;
const FOCUS_DECISIVE_ADVANTAGE: f32 = 0.20;
const FOCUS_CONFIDENCE_MARGIN: f32 = 0.04;
const FOCUS_EDGE_PROTECTION_AT_1024: f32 = 12.0;
const FOCUS_EDGE_CLAIM_THRESHOLD: f32 = 0.08;
const FOCUS_SEAM_BLEND_RADIUS: usize = 48;
const FOCUS_FOREGROUND_HARD_EDGE_RADIUS_AT_2400: f32 = 8.0;
// A depth layer can move by more than a pixel when its source frame is aligned
// against the paper plane. Keep an ownership buffer around an already selected
// foreground object so a later frame cannot re-introduce the same object as a
// parallel strip. The radius is derived from the source layer, not from any
// particular colour or holder shape.
const FOCUS_FOREGROUND_OWNERSHIP_RADIUS_RATIO: f32 = 0.025;

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

fn map_target_to_source(
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

fn output_bounds(
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

fn pixel_aligned_canvas(minimum: f64, maximum: f64) -> (f64, u32) {
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

fn transformed_image_region(
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
    let (width, height) = image.dimensions();
    let y0 = (height as f64 * minimum_source_y).clamp(0.0, height as f64);
    let y1 = (height as f64 * maximum_source_y).clamp(0.0, height as f64);
    let corners = [(0.0, y0), (width as f64, y0), (width as f64, y1), (0.0, y1)];
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
            let Some((band_min_x, band_max_x, band_min_y, band_max_y)) = transformed_source_bounds(
                image,
                homography,
                projection,
                minimum_source_y,
                maximum_source_y,
            ) else {
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
    let h_base_inv = h_base.try_inverse().unwrap();
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
        let h_add_inv = h_add.try_inverse().unwrap();
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
        false,
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

struct RenderedFocusLayer {
    image: Rgb32FImage,
    mask: GrayImage,
    foreground_mask: GrayImage,
    left: u32,
    top: u32,
}

struct RenderedFocusAnalysisLayer {
    image: Rgb32FImage,
    mask: GrayImage,
    foreground_mask: GrayImage,
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
            let Some((band_min_x, band_max_x, band_min_y, band_max_y)) = transformed_source_bounds(
                image,
                band_homography,
                projection,
                minimum_source_y,
                maximum_source_y,
            ) else {
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
    let inverse = homography.try_inverse().unwrap_or_else(Matrix3::identity);
    let Some((left, right, top, bottom)) = focus_transformed_image_region(
        image, homography, projection, focus_warp, offset_x, offset_y, out_width, out_height,
    ) else {
        return RenderedFocusLayer {
            image: Rgb32FImage::new(0, 0),
            mask: GrayImage::new(0, 0),
            foreground_mask: GrayImage::new(0, 0),
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
            let inverse = homography.try_inverse()?;
            Some((minimum_source_y, maximum_source_y, inverse))
        })
        .collect::<Vec<_>>();
    let layer_width = right - left + 1;
    let layer_height = bottom - top + 1;
    let mut pixels = vec![0.0f32; layer_width as usize * layer_height as usize * 3];
    let mut mask = vec![0u8; layer_width as usize * layer_height as usize];
    let mut foreground_mask = vec![0u8; layer_width as usize * layer_height as usize];

    pixels
        .par_chunks_mut(layer_width as usize * 3)
        .zip(mask.par_chunks_mut(layer_width as usize))
        .zip(foreground_mask.par_chunks_mut(layer_width as usize))
        .enumerate()
        .for_each(|(y, ((row, mask_row), foreground_mask_row))| {
            let global_y = top + y as u32;
            for local_x in 0..layer_width {
                let global_x = left + local_x;
                let target =
                    Point3::new(global_x as f64 - offset_x, global_y as f64 - offset_y, 1.0);
                let global_source = map_target_to_source(&inverse, target, image, projection);
                // A local model moves a depth layer relative to the paper. Do
                // not use the global inverse to decide whether that moved pixel
                // belongs to the band: doing so drops the local model exactly
                // along the displaced silhouette and creates a step back to the
                // paper transform. Instead, validate each local inverse in its
                // own source domain, then let the narrowest active band win.
                // This is content-agnostic; when no local bands exist the global
                // inverse remains the only mapping.
                let mut source = global_source;
                let mut local_candidates = Vec::new();
                for &(minimum_source_y, maximum_source_y, ref band_inverse) in &band_inverses {
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
                    let normalized_y = candidate.y / image.height().max(1) as f64;
                    if normalized_y < minimum_source_y - 0.02
                        || normalized_y > maximum_source_y + 0.02
                    {
                        continue;
                    }
                    local_candidates.push((
                        (maximum_source_y - minimum_source_y).max(1e-6),
                        normalized_y,
                        (minimum_source_y + maximum_source_y) * 0.5,
                        candidate,
                    ));
                }
                let narrowest_span = local_candidates
                    .iter()
                    .map(|(span, _, _, _)| *span)
                    .fold(f64::INFINITY, f64::min);
                if narrowest_span.is_finite() {
                    let maximum_span = narrowest_span * 1.25;
                    let mut weighted_source_x = 0.0;
                    let mut weighted_source_y = 0.0;
                    let mut total_weight = 0.0;
                    for (span, normalized_y, center, candidate) in local_candidates {
                        if span > maximum_span {
                            continue;
                        }
                        // Keep the mapping continuous at a band boundary, but
                        // do not allow a broad artwork band to dilute a narrow
                        // detected depth layer that covers the same source row.
                        let weight =
                            (1.0 - (normalized_y - center).abs() / (span * 0.5)).clamp(0.05, 1.0);
                        weighted_source_x += candidate.x * weight;
                        weighted_source_y += candidate.y * weight;
                        total_weight += weight;
                    }
                    if total_weight > 0.0 {
                        source = Some(Point2::new(
                            weighted_source_x / total_weight,
                            weighted_source_y / total_weight,
                        ));
                    }
                } else if source.is_none() {
                    // In the unlikely event that the global inverse is invalid,
                    // retain the previous fallback behavior and use the first
                    // valid local model.
                    source = band_inverses.iter().find_map(|&(_, _, ref band_inverse)| {
                        map_target_to_source(band_inverse, target, image, projection)
                    });
                }
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
                let pixel = get_high_quality_interpolated_pixel(source_image, source.x, source.y);
                let start = local_x as usize * 3;
                row[start..start + 3].copy_from_slice(&pixel.0);
                mask_row[local_x as usize] = 255;
                if source_is_focus_foreground(image, source) {
                    foreground_mask_row[local_x as usize] = 255;
                }
            }
        });

    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(layer_width, layer_height, pixels)
            .expect("focus layer buffer dimensions must match"),
        mask: GrayImage::from_raw(layer_width, layer_height, mask)
            .expect("focus layer mask dimensions must match"),
        foreground_mask: GrayImage::from_raw(layer_width, layer_height, foreground_mask)
            .expect("focus foreground mask buffer dimensions must match"),
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

    // Focus ownership is decided on a bounded analysis canvas. An approximately
    // 8px window at 1536px suppresses sensor/JPEG noise and texture-scale flips,
    // while the final pixels still come from the full-resolution aligned layer.
    let radius = ((width.max(height) as f32 / 192.0).round() as usize).clamp(1, 8);
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

    let coherence_radius = ((width.max(height) as f32 / 96.0).round() as usize).clamp(2, 16);
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
    let protection_radius = ((width.max(height) as f32 / 1024.0) * FOCUS_EDGE_PROTECTION_AT_1024)
        .round()
        .max(1.0) as usize;
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

fn resize_binary_mask(mask: &GrayImage, width: u32, height: u32) -> GrayImage {
    image::imageops::resize(
        mask,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Nearest,
    )
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
        .clamp(4.0, 96.0) as i32;
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
        for delta_y in (coarse_y - coarse_step + 1)..=(coarse_y + coarse_step - 1) {
            for delta_x in (coarse_x - coarse_step + 1)..=(coarse_x + coarse_step - 1) {
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
    );
}

fn suppress_focus_foreground_switches_in_region(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_left: u32,
    candidate_top: u32,
    candidate_foreground_mask: &GrayImage,
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

    decision_mask
        .as_mut()
        .par_chunks_mut(layer_width as usize)
        .enumerate()
        .for_each(|(local_y, decision_row)| {
            let foreground_row = &candidate_foreground_mask.as_raw()
                [local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
            let overlap_row =
                &overlap_mask[local_y * layer_width as usize..(local_y + 1) * layer_width as usize];
            for (local_x, decision) in decision_row.iter_mut().enumerate() {
                if foreground_row[local_x] > 0 && overlap_row[local_x] > 0 {
                    *decision = 0;
                }
            }
        });
}

fn suppress_focus_foreground_switches_on_canvas(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_foreground_mask: &GrayImage,
) {
    if decision_mask.dimensions() != merged_foreground_mask.dimensions()
        || decision_mask.dimensions() != candidate_foreground_mask.dimensions()
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
        .for_each(|((decision, candidate_foreground), overlap)| {
            if *candidate_foreground > 0 && *overlap > 0 {
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
) {
    let (layer_width, layer_height) = candidate_mask.dimensions();
    if layer_width == 0
        || layer_height == 0
        || decision_mask.dimensions() != (layer_width, layer_height)
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
        .for_each(|((decision, candidate_valid), overlap)| {
            if *candidate_valid > 0 && *overlap > 0 {
                *decision = 0;
            }
        });
}

fn suppress_focus_foreground_ownership_switches_on_canvas(
    decision_mask: &mut GrayImage,
    merged_foreground_mask: &GrayImage,
    candidate_mask: &GrayImage,
) {
    if decision_mask.dimensions() != merged_foreground_mask.dimensions()
        || decision_mask.dimensions() != candidate_mask.dimensions()
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
        .for_each(|((decision, candidate_valid), overlap)| {
            if *candidate_valid > 0 && *overlap > 0 {
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
        };
    }

    let resized_width = right - left;
    let resized_height = bottom - top;
    let resized = resize_rgb(&layer.image, resized_width, resized_height);
    let resized_mask = resize_binary_mask(&layer.mask, resized_width, resized_height);
    let resized_foreground_mask =
        resize_binary_mask(&layer.foreground_mask, resized_width, resized_height);
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
    RenderedFocusAnalysisLayer {
        image: analysis,
        mask: analysis_mask,
        foreground_mask: analysis_foreground_mask,
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

fn resize_rgb(image: &Rgb32FImage, width: u32, height: u32) -> Rgb32FImage {
    image::imageops::resize(
        image,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Triangle,
    )
}

fn downsample_rgb_half(image: &Rgb32FImage) -> Rgb32FImage {
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

fn crop_to_valid_rectangle(image: Rgb32FImage, mask: &GrayImage) -> Rgb32FImage {
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
        .map(|(candidate_foreground, existing_foreground)| {
            u8::from(*candidate_foreground > 0 || *existing_foreground > 0)
        })
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
    let patch_left = seam_left.saturating_sub(radius as u32);
    let patch_right = (seam_right + radius as u32).min(layer_width - 1);
    let patch_top = seam_top.saturating_sub(radius as u32);
    let patch_bottom = (seam_bottom + radius as u32).min(layer_height - 1);
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
    let mut protected_values = vec![0u8; patch_pixel_count];
    let mut protected_pixels = vec![0.0f32; patch_pixel_count * 3];
    let base_ref: &Rgb32FImage = base;
    let base_mask_ref: &GrayImage = base_mask;

    base_pixels
        .par_chunks_mut(patch_rgb_stride)
        .zip(candidate_pixels.par_chunks_mut(patch_rgb_stride))
        .zip(blend_values.par_chunks_mut(patch_mask_stride))
        .zip(protected_values.par_chunks_mut(patch_mask_stride))
        .zip(protected_pixels.par_chunks_mut(patch_rgb_stride))
        .enumerate()
        .for_each(
            |(
                local_y,
                ((((base_row, candidate_row), blend_row), protected_row), protected_pixel_row),
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
                    let protected = is_protected(source_x, source_y);
                    protected_row[local_x as usize] = u8::from(protected);
                    let start = local_x as usize * 3;
                    base_row[start..start + 3].copy_from_slice(&base_pixel.0);
                    candidate_row[start..start + 3].copy_from_slice(&candidate_pixel.0);
                    if protected {
                        let selected_pixel = if owns { candidate_pixel } else { base_pixel };
                        protected_pixel_row[start..start + 3].copy_from_slice(&selected_pixel.0);
                    }
                    blend_row[local_x as usize] = if owns { 1.0 } else { 0.0 };
                }
            },
        );

    let low_frequency = box_blur_focus_map(
        &blend_values,
        patch_width,
        patch_height,
        (radius / 2).max(1),
    );
    let blend_mask = GrayImage::from_fn(patch_width, patch_height, |x, y| {
        image::Luma([(blend_values[y as usize * patch_width as usize + x as usize] * 255.0) as u8])
    });
    let low_frequency_mask = GrayImage::from_fn(patch_width, patch_height, |x, y| {
        image::Luma([
            (low_frequency[y as usize * patch_width as usize + x as usize].clamp(0.0, 1.0) * 255.0)
                as u8,
        ])
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

fn mapped_image_center(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
) -> Option<Point2<f64>> {
    let center = project_point(
        image,
        image.width() as f64 * 0.5,
        image.height() as f64 * 0.5,
        projection,
    )?;
    let mapped = homography * Point3::new(center.x, center.y, 1.0);
    if mapped.z.abs() < 1e-8 {
        None
    } else {
        Some(Point2::new(mapped.x / mapped.z, mapped.y / mapped.z))
    }
}

fn focus_stack_is_shifted_mosaic(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> bool {
    let Some(first) = images.first() else {
        return false;
    };
    let Some(reference_center) =
        mapped_image_center(first, &global_homographies[&first.id], projection)
    else {
        return false;
    };
    let reference_width = first.width().max(1) as f64;
    let reference_height = first.height().max(1) as f64;
    images.iter().skip(1).any(|image| {
        mapped_image_center(image, &global_homographies[&image.id], projection).is_some_and(
            |center| {
                (center.x - reference_center.x).abs() > reference_width * 0.08
                    || (center.y - reference_center.y).abs() > reference_height * 0.08
            },
        )
    })
}

pub fn focus_stack_stitcher<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
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
    let shifted_mosaic = focus_stack_is_shifted_mosaic(images, global_homographies, projection);
    if shifted_mosaic {
        println!(
            "  - Large framing shift detected; keeping local sharpness ownership across overlaps"
        );
        let _ = app_handle.emit(
            progress_event,
            "Large framing shift detected; selecting the sharpest source in each overlap...",
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
    place_focus_layer(&mut merged, &mut merged_mask, &first_layer);
    place_focus_foreground_mask(&mut merged_foreground_mask, &first_layer);
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
        let candidate = render_focus_layer(
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
        // A regional warp can still leave a small residual translation on a
        // depth-discontinuous layer. Refine only the detected foreground
        // ownership mask against the already selected foreground; ordinary
        // artwork pixels keep the global/local geometric mapping unchanged.
        let candidate =
            align_focus_foreground_layer_to_existing(candidate, &merged_foreground_mask);
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
        );
        suppress_focus_foreground_ownership_switches_on_canvas(
            &mut analysis_decision,
            &merged_analysis_foreground_mask,
            &candidate_analysis.mask,
        );

        // Keep the focus score synchronized with the actual selected source. The old
        // pipeline stored max scores while repeatedly averaging pixels, so later layers
        // were compared against a score that no longer described the image underneath.
        merged_focus
            .par_iter_mut()
            .zip(candidate_focus.par_iter())
            .zip(merged_analysis_mask.as_raw().par_iter())
            .zip(candidate_analysis.mask.as_raw().par_iter())
            .zip(analysis_decision.as_raw().par_iter())
            .for_each(
                |((((base_focus, candidate_focus), base_valid), candidate_valid), decision)| {
                    if *candidate_valid > 0 && (*base_valid == 0 || *decision > 0) {
                        *base_focus = *candidate_focus;
                    }
                },
            );
        hard_select_focus_pixels(
            &mut merged_analysis,
            &mut merged_analysis_mask,
            &candidate_analysis.image,
            &candidate_analysis.mask,
            &analysis_decision,
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
        );
        let seam_blend_enabled = shifted_mosaic;
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
        // The full-resolution pass is authoritative for ownership. Rebuilding
        // the analysis mask from it prevents a later candidate from inheriting
        // a foreground claim that was rejected after upsampling the decision.
        merged_analysis_foreground_mask =
            resize_binary_mask(&merged_foreground_mask, analysis_width, analysis_height);
    }
    let merged_dimensions = merged.dimensions();
    let cropped = crop_to_valid_rectangle(merged, &merged_mask);
    if cropped.dimensions() != merged_dimensions {
        println!(
            "  - Cropped invalid focus-stack margins: {}x{} -> {}x{}",
            merged_dimensions.0,
            merged_dimensions.1,
            cropped.width(),
            cropped.height()
        );
    }
    Ok(cropped)
}

fn find_adaptive_seam(ctx: &SeamContext) -> Option<SeamInfo> {
    let h_add_inv = ctx.h_add.try_inverse().unwrap();
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

fn get_high_quality_interpolated_pixel(img: &Rgb32FImage, x: f64, y: f64) -> Rgb<f32> {
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
}
