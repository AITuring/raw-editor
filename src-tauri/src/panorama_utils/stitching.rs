use crate::panorama_stitching::ImageInfo;
use image::{GrayImage, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Point3};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use tauri::{AppHandle, Emitter, Runtime};

const PANORAMA_BLEND_BANDS: usize = 9;

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
    let width = image.image.width() as f64;
    let height = image.image.height() as f64;
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

    let width = image.image.width() as f64;
    let height = image.image.height() as f64;
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
        let (width, height) = image.image.dimensions();
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

fn apply_exposure_gain(pixel: Rgb<f32>, gain: f32) -> Rgb<f32> {
    Rgb([pixel[0] * gain, pixel[1] * gain, pixel[2] * gain])
}

struct ExposureOverlap<'a> {
    panorama: &'a Rgb32FImage,
    panorama_mask: &'a GrayImage,
    candidate: &'a ImageInfo,
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
    for y in (0..out_height).step_by(sample_step) {
        for x in (0..out_width).step_by(sample_step) {
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
                || source.x >= ctx.candidate.image.width() as f64 - 1.0
                || source.y >= ctx.candidate.image.height() as f64 - 1.0
            {
                continue;
            }
            let base_luma = luminance(ctx.panorama.get_pixel(x, y));
            let candidate_luma = luminance(&get_interpolated_pixel(
                &ctx.candidate.image,
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

pub fn progressive_seam_stitcher<R: Runtime>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    app_handle: AppHandle<R>,
    progress_event: &str,
) -> Rgb32FImage {
    if images.is_empty() {
        return Rgb32FImage::new(0, 0);
    }

    let (min_x, max_x, min_y, max_y) = output_bounds(images, global_homographies, projection);

    let offset_x = -min_x;
    let offset_y = -min_y;
    let out_width = (max_x - min_x).ceil().max(1.0) as u32;
    let out_height = (max_y - min_y).ceil().max(1.0) as u32;
    println!("  - Output canvas size: {}x{}", out_width, out_height);

    let mut panorama = Rgb32FImage::new(out_width, out_height);
    let mut panorama_mask = GrayImage::new(out_width, out_height);

    let base_img_info = images[0];
    let h_base = &global_homographies[&base_img_info.id];
    let h_base_inv = h_base.try_inverse().unwrap();
    println!("  - Placing base image: '{}'", base_img_info.filename);

    let num_pixels_per_row = out_width as usize * 3;
    panorama
        .par_chunks_mut(num_pixels_per_row)
        .zip(panorama_mask.par_chunks_mut(out_width as usize))
        .enumerate()
        .for_each(|(y, (row_slice, mask_row))| {
            for x in 0..out_width {
                let target_p = Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
                if let Some(source) =
                    map_target_to_source(&h_base_inv, target_p, base_img_info, projection)
                {
                    let sx = source.x;
                    let sy = source.y;
                    if sx < 0.0
                        || sx >= base_img_info.image.width() as f64
                        || sy < 0.0
                        || sy >= base_img_info.image.height() as f64
                    {
                        continue;
                    }
                    let color = get_interpolated_pixel(&base_img_info.image, sx, sy);
                    let start = x as usize * 3;
                    row_slice[start..start + 3].copy_from_slice(&color.0);
                    mask_row[x as usize] = 255;
                }
            }
        });

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
        let img_to_add = &img_to_add_info.image;
        let exposure = estimate_overlap_exposure_compensation(ExposureOverlap {
            panorama: &panorama,
            panorama_mask: &panorama_mask,
            candidate: img_to_add_info,
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
            img_to_add,
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

        match orientation {
            SeamOrientation::Vertical => {
                panorama
                    .par_chunks_mut(num_pixels_per_row)
                    .zip(panorama_mask.par_chunks_mut(out_width as usize))
                    .enumerate()
                    .for_each(|(y, (row_slice, mask_row))| {
                        for x in 0..out_width {
                            let target_p =
                                Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);

                            let Some(source_add) = map_target_to_source(
                                &h_add_inv,
                                target_p,
                                img_to_add_info,
                                projection,
                            ) else {
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
                            } else if is_on_add {
                                let color_to_add = apply_exposure_gain(
                                    get_interpolated_pixel(img_to_add, sx, sy),
                                    exposure.gain_at(x, y as u32),
                                );
                                let start = x as usize * 3;
                                row_slice[start..start + 3].copy_from_slice(&color_to_add.0);
                                mask_row[x as usize] = 255;
                            }
                        }
                    });
            }
            SeamOrientation::Horizontal => {
                panorama
                    .par_chunks_mut(num_pixels_per_row)
                    .zip(panorama_mask.par_chunks_mut(out_width as usize))
                    .enumerate()
                    .for_each(|(y, (row_slice, mask_row))| {
                        for x in 0..out_width {
                            let target_p =
                                Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);

                            let Some(source_add) = map_target_to_source(
                                &h_add_inv,
                                target_p,
                                img_to_add_info,
                                projection,
                            ) else {
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
                            } else if is_on_add {
                                let color_to_add = apply_exposure_gain(
                                    get_interpolated_pixel(img_to_add, sx, sy),
                                    exposure.gain_at(x, y as u32),
                                );
                                let start = x as usize * 3;
                                row_slice[start..start + 3].copy_from_slice(&color_to_add.0);
                                mask_row[x as usize] = 255;
                            }
                        }
                    });
            }
        }

        if let (true, Some((min_x, max_x, min_y, max_y))) = (use_seam, seam_bounds) {
            blend_panorama_seam_band(SeamBandBlend {
                panorama: &mut panorama,
                panorama_mask: &mut panorama_mask,
                img_to_add_info,
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

    let cropped = crop_to_valid_rectangle(&panorama, &panorama_mask);
    if cropped.dimensions() != panorama.dimensions() {
        println!(
            "  - Cropped invalid projection margins: {}x{} -> {}x{}",
            panorama.width(),
            panorama.height(),
            cropped.width(),
            cropped.height()
        );
    }
    cropped
}

struct SeamBandBlend<'a> {
    panorama: &'a mut Rgb32FImage,
    panorama_mask: &'a mut GrayImage,
    img_to_add_info: &'a ImageInfo,
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
    let image_width = img_to_add_info.image.width() as f64;
    let image_height = img_to_add_info.image.height() as f64;

    for local_y in 0..patch_height {
        for local_x in 0..patch_width {
            let global_x = patch_left + local_x;
            let global_y = patch_top + local_y;
            let index = (local_y as usize * patch_width as usize) + local_x as usize;
            let target = Point3::new(global_x as f64 - offset_x, global_y as f64 - offset_y, 1.0);
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
                        get_interpolated_pixel(&img_to_add_info.image, source.x, source.y),
                        exposure.gain_at(global_x, global_y),
                    )
                } else {
                    Rgb([0.0, 0.0, 0.0])
                }
            } else {
                Rgb([0.0, 0.0, 0.0])
            };
            let panorama_valid = panorama_mask.get_pixel(global_x, global_y)[0] > 0;
            let current_pixel = *panorama.get_pixel(global_x, global_y);
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

            let candidate_owns_pixel = if !candidate_valid {
                false
            } else if !panorama_valid {
                true
            } else {
                match orientation {
                    SeamOrientation::Horizontal => {
                        let seam_y = seam_value(global_x as usize).unwrap_or(global_y);
                        if new_image_is_dominant_side {
                            global_y > seam_y
                        } else {
                            global_y < seam_y
                        }
                    }
                    SeamOrientation::Vertical => {
                        let seam_x = seam_value(global_y as usize).unwrap_or(global_x);
                        if new_image_is_dominant_side {
                            global_x > seam_x
                        } else {
                            global_x < seam_x
                        }
                    }
                }
            };

            let base_start = index * 3;
            base_pixels[base_start..base_start + 3].copy_from_slice(&base_pixel.0);
            candidate_pixels[base_start..base_start + 3].copy_from_slice(&candidate_pixel.0);
            blend_mask[index] = if candidate_owns_pixel { 255 } else { 0 };
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
                        let position = (global_x.saturating_sub(overlap_left)) as f32 / span;
                        if new_image_is_dominant_side {
                            position
                        } else {
                            1.0 - position
                        }
                    }
                }
            };
            low_frequency_mask[index] = (low_frequency_alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }

    let base = Rgb32FImage::from_raw(patch_width, patch_height, base_pixels)
        .expect("seam base patch dimensions must match");
    let candidate = Rgb32FImage::from_raw(patch_width, patch_height, candidate_pixels)
        .expect("seam candidate patch dimensions must match");
    let mask = GrayImage::from_raw(patch_width, patch_height, blend_mask.clone())
        .expect("seam mask dimensions must match");
    let low_frequency_mask = GrayImage::from_raw(patch_width, patch_height, low_frequency_mask)
        .expect("low-frequency seam mask dimensions must match");
    // The optimal path keeps the transition away from strong subject edges. A narrow
    // feather at the finest band then removes fabric/sky phase discontinuities without
    // averaging detail across the photographed subject.
    let feathered_mask = feather_mask(&mask, 24);
    let blended = multiband_blend(
        base,
        candidate,
        feathered_mask,
        Some(low_frequency_mask),
        PANORAMA_BLEND_BANDS,
        false,
    );

    for local_y in 0..patch_height {
        for local_x in 0..patch_width {
            let global_x = patch_left + local_x;
            let global_y = patch_top + local_y;
            let index = (local_y as usize * patch_width as usize) + local_x as usize;
            *panorama.get_pixel_mut(global_x, global_y) = *blended.get_pixel(local_x, local_y);
            if panorama_mask.get_pixel(global_x, global_y)[0] > 0 || blend_mask[index] > 0 {
                panorama_mask.put_pixel(global_x, global_y, image::Luma([255]));
            }
        }
    }
}

fn luminance(pixel: &Rgb<f32>) -> f32 {
    pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114
}

fn focus_score_at(image: &Rgb32FImage, x: f64, y: f64) -> f32 {
    let mut sum = 0.0f32;
    let mut sum_squared = 0.0f32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let value = luminance(&get_interpolated_pixel(image, x + dx as f64, y + dy as f64));
            sum += value;
            sum_squared += value * value;
        }
    }
    let mean = sum / 9.0;
    let variance = (sum_squared / 9.0 - mean * mean).max(0.0);
    let center = luminance(&get_interpolated_pixel(image, x, y));
    let left = luminance(&get_interpolated_pixel(image, x - 1.0, y));
    let right = luminance(&get_interpolated_pixel(image, x + 1.0, y));
    let top = luminance(&get_interpolated_pixel(image, x, y - 1.0));
    let bottom = luminance(&get_interpolated_pixel(image, x, y + 1.0));
    let laplacian = (4.0 * center - left - right - top - bottom).abs();
    variance + laplacian * 0.25
}

fn render_focus_layer(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
) -> (Rgb32FImage, GrayImage, Vec<f32>) {
    let inverse = homography.try_inverse().unwrap_or_else(Matrix3::identity);
    let mut pixels = vec![0.0f32; out_width as usize * out_height as usize * 3];
    let mut mask = vec![0u8; out_width as usize * out_height as usize];
    let mut focus = vec![0.0f32; out_width as usize * out_height as usize];

    pixels
        .par_chunks_mut(out_width as usize * 3)
        .zip(mask.par_chunks_mut(out_width as usize))
        .zip(focus.par_chunks_mut(out_width as usize))
        .enumerate()
        .for_each(|(y, ((row, mask_row), focus_row))| {
            for x in 0..out_width {
                let target = Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
                let Some(source) = map_target_to_source(&inverse, target, image, projection) else {
                    continue;
                };
                if source.x < 1.0
                    || source.y < 1.0
                    || source.x >= image.image.width() as f64 - 2.0
                    || source.y >= image.image.height() as f64 - 2.0
                {
                    continue;
                }
                let pixel = get_interpolated_pixel(&image.image, source.x, source.y);
                let start = x as usize * 3;
                row[start..start + 3].copy_from_slice(&pixel.0);
                mask_row[x as usize] = 255;
                focus_row[x as usize] = focus_score_at(&image.image, source.x, source.y);
            }
        });

    (
        Rgb32FImage::from_raw(out_width, out_height, pixels)
            .expect("focus layer buffer dimensions must match"),
        GrayImage::from_raw(out_width, out_height, mask)
            .expect("focus layer mask dimensions must match"),
        focus,
    )
}

fn resize_rgb(image: &Rgb32FImage, width: u32, height: u32) -> Rgb32FImage {
    image::imageops::resize(
        image,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Triangle,
    )
}

fn resize_mask(mask: &GrayImage, width: u32, height: u32) -> GrayImage {
    image::imageops::resize(
        mask,
        width.max(1),
        height.max(1),
        image::imageops::FilterType::Triangle,
    )
}

fn feather_mask(mask: &GrayImage, radius: u32) -> GrayImage {
    if radius <= 1 || mask.width() <= 2 || mask.height() <= 2 {
        return mask.clone();
    }
    let reduced_width = mask.width().div_ceil(radius).max(1);
    let reduced_height = mask.height().div_ceil(radius).max(1);
    let reduced = resize_mask(mask, reduced_width, reduced_height);
    resize_mask(&reduced, mask.width(), mask.height())
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

fn add_rgb(base: &Rgb32FImage, detail: &Rgb32FImage) -> Rgb32FImage {
    let (width, height) = base.dimensions();
    let mut output = vec![0.0f32; width as usize * height as usize * 3];
    output
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                let base_pixel = base.get_pixel(x, y);
                let detail_pixel = detail.get_pixel(x, y);
                let start = x as usize * 3;
                for channel in 0..3 {
                    row[start + channel] = base_pixel[channel] + detail_pixel[channel];
                }
            }
        });
    Rgb32FImage::from_raw(width, height, output).expect("reconstructed image dimensions must match")
}

fn subtract_rgb(base: &Rgb32FImage, reference: &Rgb32FImage) -> Rgb32FImage {
    let (width, height) = base.dimensions();
    let mut output = vec![0.0f32; width as usize * height as usize * 3];
    output
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let y = y as u32;
            for x in 0..width {
                let base_pixel = base.get_pixel(x, y);
                let reference_pixel = reference.get_pixel(x, y);
                let start = x as usize * 3;
                for channel in 0..3 {
                    row[start + channel] = base_pixel[channel] - reference_pixel[channel];
                }
            }
        });
    Rgb32FImage::from_raw(width, height, output).expect("detail image dimensions must match")
}

fn crop_to_valid_rectangle(image: &Rgb32FImage, mask: &GrayImage) -> Rgb32FImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || mask.dimensions() != (width, height) {
        return image.clone();
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
        return image.clone();
    }

    image::imageops::crop_imm(
        image,
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
    let has_distinct_low_frequency_mask = low_frequency_mask.is_some();
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
        let next_width = (current_base.width() / 2).max(1);
        let next_height = (current_base.height() / 2).max(1);
        let next_base = resize_rgb(&current_base, next_width, next_height);
        let next_candidate = resize_rgb(&current_candidate, next_width, next_height);
        let next_mask = resize_mask(&current_mask, next_width, next_height);
        let next_low_frequency_mask =
            resize_mask(&current_low_frequency_mask, next_width, next_height);
        let base_up = resize_rgb(&next_base, current_base.width(), current_base.height());
        let candidate_up = resize_rgb(
            &next_candidate,
            current_candidate.width(),
            current_candidate.height(),
        );
        base_laplacian.push(subtract_rgb(&current_base, &base_up));
        candidate_laplacian.push(subtract_rgb(&current_candidate, &candidate_up));
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
        let blend_mask = if has_distinct_low_frequency_mask && level >= 2 {
            &low_frequency_masks[level]
        } else {
            &masks[level]
        };
        let blended_detail = combine_rgb(
            &base_laplacian[level],
            &candidate_laplacian[level],
            blend_mask,
            hard_finest_band && level == 0,
        );
        reconstructed = add_rgb(
            &resize_rgb(
                &reconstructed,
                base_laplacian[level].width(),
                base_laplacian[level].height(),
            ),
            &blended_detail,
        );
    }
    if reconstructed.dimensions() == (width, height) {
        reconstructed
    } else {
        resize_rgb(&reconstructed, width, height)
    }
}

fn mapped_image_center(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
) -> Option<Point2<f64>> {
    let center = project_point(
        image,
        image.image.width() as f64 * 0.5,
        image.image.height() as f64 * 0.5,
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
    let reference_width = first.image.width().max(1) as f64;
    let reference_height = first.image.height().max(1) as f64;
    images.iter().skip(1).any(|image| {
        mapped_image_center(image, &global_homographies[&image.id], projection).is_some_and(
            |center| {
                (center.x - reference_center.x).abs() > reference_width * 0.08
                    || (center.y - reference_center.y).abs() > reference_height * 0.08
            },
        )
    })
}

pub fn focus_stack_stitcher<R: Runtime>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    app_handle: AppHandle<R>,
    progress_event: &str,
) -> Rgb32FImage {
    if images.is_empty() {
        return Rgb32FImage::new(0, 0);
    }
    if focus_stack_is_shifted_mosaic(images, global_homographies, projection) {
        println!(
            "  - Large framing shift detected; using coherent optimal seams to preserve detail"
        );
        let _ = app_handle.emit(
            progress_event,
            "Large framing shift detected; optimizing overlap seams...",
        );
        return progressive_seam_stitcher(
            images,
            global_homographies,
            projection,
            app_handle,
            progress_event,
        );
    }
    let (min_x, max_x, min_y, max_y) = output_bounds(images, global_homographies, projection);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return Rgb32FImage::new(0, 0);
    }
    let offset_x = -min_x;
    let offset_y = -min_y;
    let out_width = (max_x - min_x).ceil().max(1.0) as u32;
    let out_height = (max_y - min_y).ceil().max(1.0) as u32;
    let (mut merged, mut merged_mask, mut merged_focus) = render_focus_layer(
        images[0],
        &global_homographies[&images[0].id],
        projection,
        offset_x,
        offset_y,
        out_width,
        out_height,
    );

    for (index, image) in images.iter().enumerate().skip(1) {
        let _ = app_handle.emit(
            progress_event,
            format!("Focus-stacking image {} of {}", index + 1, images.len()),
        );
        let (mut candidate, candidate_mask, candidate_focus) = render_focus_layer(
            image,
            &global_homographies[&image.id],
            projection,
            offset_x,
            offset_y,
            out_width,
            out_height,
        );
        let mut blend_mask = GrayImage::new(out_width, out_height);
        for y in 0..out_height {
            for x in 0..out_width {
                let base_valid = merged_mask.get_pixel(x, y)[0] > 0;
                let candidate_valid = candidate_mask.get_pixel(x, y)[0] > 0;
                let candidate_wins = candidate_valid
                    && (!base_valid
                        || candidate_focus[(y * out_width + x) as usize]
                            > merged_focus[(y * out_width + x) as usize] * 1.04);
                blend_mask.put_pixel(x, y, image::Luma([if candidate_wins { 255 } else { 0 }]));

                // Multiband filtering samples across the mask boundary. Mirror the valid
                // source into the invalid side so the pyramid never blends against the
                // zero-filled area outside a warped layer (which otherwise produces a
                // visible dark/soft rectangle at every source-image edge).
                if base_valid && !candidate_valid {
                    *candidate.get_pixel_mut(x, y) = *merged.get_pixel(x, y);
                } else if candidate_valid && !base_valid {
                    *merged.get_pixel_mut(x, y) = *candidate.get_pixel(x, y);
                }
                if candidate_valid {
                    merged_mask.put_pixel(x, y, image::Luma([255]));
                }
                let focus_index = (y * out_width + x) as usize;
                merged_focus[focus_index] =
                    merged_focus[focus_index].max(candidate_focus[focus_index]);
            }
        }
        merged = multiband_blend(merged, candidate, blend_mask, None, 5, true);
    }
    let cropped = crop_to_valid_rectangle(&merged, &merged_mask);
    if cropped.dimensions() != merged.dimensions() {
        println!(
            "  - Cropped invalid focus-stack margins: {}x{} -> {}x{}",
            merged.width(),
            merged.height(),
            cropped.width(),
            cropped.height()
        );
    }
    cropped
}

fn find_adaptive_seam(ctx: &SeamContext) -> Option<SeamInfo> {
    let h_add_inv = ctx.h_add.try_inverse().unwrap();
    let (w_add, h_add_img) = ctx.img_to_add.dimensions();

    let mut min_ox = u32::MAX;
    let mut max_ox = 0;
    let mut min_oy = u32::MAX;
    let mut max_oy = 0;
    let mut has_overlap = false;

    for y in 0..ctx.out_height {
        for x in 0..ctx.out_width {
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
            for previous_index in first_neighbor..=last_neighbor {
                if previous[previous_index] < best_previous {
                    best_previous = previous[previous_index];
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
                    get_interpolated_pixel(source, mapped.x / mapped.z, mapped.y / mapped.z)
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
