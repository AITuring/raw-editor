use crate::panorama_stitching::ImageInfo;
use image::{GrayImage, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Point3};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use tauri::{AppHandle, Emitter, Runtime};

const SEAM_BLEND_RADIUS: u32 = 128;
const PANORAMA_BLEND_BANDS: usize = 5;

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
                                let seam_x_val = seam_coords[y];
                                let new_image_owns_pixel = if new_image_is_dominant_side {
                                    x as i32 > seam_x_val
                                } else {
                                    (x as i32) < seam_x_val
                                };
                                if new_image_owns_pixel {
                                    let color_to_add = get_interpolated_pixel(img_to_add, sx, sy);
                                    let start = x as usize * 3;
                                    row_slice[start..start + 3].copy_from_slice(&color_to_add.0);
                                }
                            } else if is_on_add {
                                let color_to_add = get_interpolated_pixel(img_to_add, sx, sy);
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
                                let seam_y_val = seam_coords[x as usize];
                                let new_image_owns_pixel = if new_image_is_dominant_side {
                                    y as i32 > seam_y_val
                                } else {
                                    (y as i32) < seam_y_val
                                };
                                if new_image_owns_pixel {
                                    let color_to_add = get_interpolated_pixel(img_to_add, sx, sy);
                                    let start = x as usize * 3;
                                    row_slice[start..start + 3].copy_from_slice(&color_to_add.0);
                                }
                            } else if is_on_add {
                                let color_to_add = get_interpolated_pixel(img_to_add, sx, sy);
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

    let (patch_left, patch_top, patch_right, patch_bottom) = match orientation {
        SeamOrientation::Horizontal => {
            if seam_coords.len() <= overlap_right as usize {
                return;
            }
            let mut seam_min = u32::MAX;
            let mut seam_max = 0;
            for x in overlap_left..=overlap_right {
                let Some(value) = seam_value(x as usize) else {
                    return;
                };
                seam_min = seam_min.min(value);
                seam_max = seam_max.max(value);
            }
            (
                overlap_left,
                overlap_top.max(seam_min.saturating_sub(SEAM_BLEND_RADIUS)),
                overlap_right,
                overlap_bottom.min(
                    seam_max
                        .saturating_add(SEAM_BLEND_RADIUS)
                        .min(out_height - 1),
                ),
            )
        }
        SeamOrientation::Vertical => {
            if seam_coords.len() <= overlap_bottom as usize {
                return;
            }
            let mut seam_min = u32::MAX;
            let mut seam_max = 0;
            for y in overlap_top..=overlap_bottom {
                let Some(value) = seam_value(y as usize) else {
                    return;
                };
                seam_min = seam_min.min(value);
                seam_max = seam_max.max(value);
            }
            (
                overlap_left.max(seam_min.saturating_sub(SEAM_BLEND_RADIUS)),
                overlap_top,
                overlap_right.min(
                    seam_max
                        .saturating_add(SEAM_BLEND_RADIUS)
                        .min(out_width - 1),
                ),
                overlap_bottom,
            )
        }
    };

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
                    get_interpolated_pixel(&img_to_add_info.image, source.x, source.y)
                } else {
                    Rgb([0.0, 0.0, 0.0])
                }
            } else {
                Rgb([0.0, 0.0, 0.0])
            };
            let panorama_valid = panorama_mask.get_pixel(global_x, global_y)[0] > 0;
            let current_pixel = *panorama.get_pixel(global_x, global_y);
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
        }
    }

    let base = Rgb32FImage::from_raw(patch_width, patch_height, base_pixels)
        .expect("seam base patch dimensions must match");
    let candidate = Rgb32FImage::from_raw(patch_width, patch_height, candidate_pixels)
        .expect("seam candidate patch dimensions must match");
    let mask = GrayImage::from_raw(patch_width, patch_height, blend_mask.clone())
        .expect("seam mask dimensions must match");
    let blended = multiband_blend(base, candidate, mask, PANORAMA_BLEND_BANDS);

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
    max_bands: usize,
) -> Rgb32FImage {
    let (width, height) = base.dimensions();
    let mut current_base = base;
    let mut current_candidate = candidate;
    let mut current_mask = mask;
    let mut base_laplacian = Vec::new();
    let mut candidate_laplacian = Vec::new();
    let mut masks = Vec::new();

    while base_laplacian.len() + 1 < max_bands {
        if current_base.width() <= 32 || current_base.height() <= 32 {
            break;
        }
        let next_width = (current_base.width() / 2).max(1);
        let next_height = (current_base.height() / 2).max(1);
        let next_base = resize_rgb(&current_base, next_width, next_height);
        let next_candidate = resize_rgb(&current_candidate, next_width, next_height);
        let next_mask = resize_mask(&current_mask, next_width, next_height);
        let base_up = resize_rgb(&next_base, current_base.width(), current_base.height());
        let candidate_up = resize_rgb(
            &next_candidate,
            current_candidate.width(),
            current_candidate.height(),
        );
        base_laplacian.push(subtract_rgb(&current_base, &base_up));
        candidate_laplacian.push(subtract_rgb(&current_candidate, &candidate_up));
        masks.push(current_mask);
        current_base = next_base;
        current_candidate = next_candidate;
        current_mask = next_mask;
    }

    let mut reconstructed = combine_rgb(&current_base, &current_candidate, &current_mask, false);
    for level in (0..base_laplacian.len()).rev() {
        let blended_detail = combine_rgb(
            &base_laplacian[level],
            &candidate_laplacian[level],
            &masks[level],
            level == 0,
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
        let (candidate, candidate_mask, candidate_focus) = render_focus_layer(
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
                if candidate_valid {
                    merged_mask.put_pixel(x, y, image::Luma([255]));
                }
                let focus_index = (y * out_width + x) as usize;
                merged_focus[focus_index] =
                    merged_focus[focus_index].max(candidate_focus[focus_index]);
            }
        }
        merged = multiband_blend(merged, candidate, blend_mask, 5);
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
        let seam = find_pairwise_seam_dp_vertical(ctx);
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
        let seam = find_pairwise_seam_dp_horizontal(ctx);
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

fn find_pairwise_seam_dp_vertical(ctx: &SeamContext) -> Vec<i32> {
    let h_add_inv = ctx.h_add.try_inverse().unwrap();
    let (w_add, h_add_img) = ctx.img_to_add.dimensions();
    let out_width = ctx.out_width;
    let out_height = ctx.out_height;
    let mut cost_matrix = vec![vec![f64::INFINITY; out_width as usize]; out_height as usize];
    let mut path_matrix = vec![vec![0i32; out_width as usize]; out_height as usize];
    let mut first_overlap_row = usize::MAX;
    let mut last_overlap_row = 0;

    for (y_out, cost_row) in cost_matrix.iter_mut().enumerate() {
        let mut row_has_overlap = false;
        for (x_out, cost_val) in cost_row.iter_mut().enumerate() {
            if ctx.pano_mask.get_pixel(x_out as u32, y_out as u32)[0] == 0 {
                continue;
            }
            let target_p = Point3::new(
                x_out as f64 - ctx.offset_x,
                y_out as f64 - ctx.offset_y,
                1.0,
            );
            let Some(source) =
                map_target_to_source(&h_add_inv, target_p, ctx.img_to_add_info, ctx.projection)
            else {
                continue;
            };
            let sx = source.x;
            let sy = source.y;
            if sx >= 0.0 && sx < w_add as f64 - 1.0 && sy >= 0.0 && sy < h_add_img as f64 - 1.0 {
                let p_pano = ctx.pano.get_pixel(x_out as u32, y_out as u32);
                let p_add = get_interpolated_pixel(ctx.img_to_add, sx, sy);
                let energy = ((p_pano[0] as f64 - p_add[0] as f64).powi(2)
                    + (p_pano[1] as f64 - p_add[1] as f64).powi(2)
                    + (p_pano[2] as f64 - p_add[2] as f64).powi(2))
                .sqrt();
                *cost_val = energy;
                row_has_overlap = true;
            }
        }
        if row_has_overlap {
            if first_overlap_row == usize::MAX {
                first_overlap_row = y_out;
            }
            last_overlap_row = y_out;
        }
    }
    if first_overlap_row == usize::MAX {
        return vec![];
    }

    for y in (first_overlap_row + 1)..=last_overlap_row {
        for x in 0..out_width as usize {
            if cost_matrix[y][x] != f64::INFINITY {
                let up_left = if x > 0 {
                    cost_matrix[y - 1][x - 1]
                } else {
                    f64::INFINITY
                };
                let up = cost_matrix[y - 1][x];
                let up_right = if x < (out_width - 1) as usize {
                    cost_matrix[y - 1][x + 1]
                } else {
                    f64::INFINITY
                };
                let min_cost = up.min(up_left).min(up_right);
                if min_cost == f64::INFINITY {
                    continue;
                }
                cost_matrix[y][x] += min_cost;
                if min_cost == up {
                    path_matrix[y][x] = 0;
                } else if min_cost == up_left {
                    path_matrix[y][x] = -1;
                } else {
                    path_matrix[y][x] = 1;
                }
            }
        }
    }

    let mut seam = vec![0i32; out_height as usize];
    let (mut min_cost, mut current_x) = (f64::INFINITY, 0);
    for (x, &cost) in cost_matrix[last_overlap_row].iter().enumerate() {
        if cost < min_cost {
            min_cost = cost;
            current_x = x as i32;
        }
    }
    if min_cost == f64::INFINITY {
        return vec![];
    }

    for y in (first_overlap_row..=last_overlap_row).rev() {
        seam[y] = current_x;
        let path_dir = path_matrix[y][current_x as usize];
        current_x += path_dir;
        current_x = current_x.clamp(0, (out_width - 1) as i32);
    }
    for y in (0..first_overlap_row).rev() {
        seam[y] = seam[first_overlap_row];
    }
    for y in (last_overlap_row + 1)..out_height as usize {
        seam[y] = seam[last_overlap_row];
    }
    seam
}

fn find_pairwise_seam_dp_horizontal(ctx: &SeamContext) -> Vec<i32> {
    let h_add_inv = ctx.h_add.try_inverse().unwrap();
    let (w_add, h_add_img) = ctx.img_to_add.dimensions();
    let out_width = ctx.out_width;
    let out_height = ctx.out_height;
    let mut cost_matrix = vec![vec![f64::INFINITY; out_width as usize]; out_height as usize];
    let mut path_matrix = vec![vec![0i32; out_width as usize]; out_height as usize];
    let mut first_overlap_col = usize::MAX;
    let mut last_overlap_col = 0;

    for (y_out, cost_row) in cost_matrix.iter_mut().enumerate() {
        for (x_out, cost_val) in cost_row.iter_mut().enumerate() {
            if ctx.pano_mask.get_pixel(x_out as u32, y_out as u32)[0] == 0 {
                continue;
            }
            let target_p = Point3::new(
                x_out as f64 - ctx.offset_x,
                y_out as f64 - ctx.offset_y,
                1.0,
            );
            let Some(source) =
                map_target_to_source(&h_add_inv, target_p, ctx.img_to_add_info, ctx.projection)
            else {
                continue;
            };
            let sx = source.x;
            let sy = source.y;
            if sx >= 0.0 && sx < w_add as f64 - 1.0 && sy >= 0.0 && sy < h_add_img as f64 - 1.0 {
                let p_pano = ctx.pano.get_pixel(x_out as u32, y_out as u32);
                let p_add = get_interpolated_pixel(ctx.img_to_add, sx, sy);
                let energy = ((p_pano[0] as f64 - p_add[0] as f64).powi(2)
                    + (p_pano[1] as f64 - p_add[1] as f64).powi(2)
                    + (p_pano[2] as f64 - p_add[2] as f64).powi(2))
                .sqrt();
                *cost_val = energy;
                first_overlap_col = first_overlap_col.min(x_out);
                last_overlap_col = last_overlap_col.max(x_out);
            }
        }
    }
    if first_overlap_col == usize::MAX {
        return vec![];
    }

    for x in (first_overlap_col + 1)..=last_overlap_col {
        for y in 0..out_height as usize {
            if cost_matrix[y][x] != f64::INFINITY {
                let left_up = if y > 0 {
                    cost_matrix[y - 1][x - 1]
                } else {
                    f64::INFINITY
                };
                let left = cost_matrix[y][x - 1];
                let left_down = if y < (out_height - 1) as usize {
                    cost_matrix[y + 1][x - 1]
                } else {
                    f64::INFINITY
                };
                let min_cost = left.min(left_up).min(left_down);
                if min_cost == f64::INFINITY {
                    continue;
                }
                cost_matrix[y][x] += min_cost;
                if min_cost == left {
                    path_matrix[y][x] = 0;
                } else if min_cost == left_up {
                    path_matrix[y][x] = -1;
                } else {
                    path_matrix[y][x] = 1;
                }
            }
        }
    }

    let mut seam = vec![0i32; out_width as usize];
    let (mut min_cost, mut current_y) = (f64::INFINITY, 0);
    for (y, cost_row) in cost_matrix.iter().enumerate() {
        if cost_row[last_overlap_col] < min_cost {
            min_cost = cost_row[last_overlap_col];
            current_y = y as i32;
        }
    }
    if min_cost == f64::INFINITY {
        return vec![];
    }

    for x in (first_overlap_col..=last_overlap_col).rev() {
        seam[x] = current_y;
        let path_dir = path_matrix[current_y as usize][x];
        current_y += path_dir;
        current_y = current_y.clamp(0, (out_height - 1) as i32);
    }
    for x in (0..first_overlap_col).rev() {
        seam[x] = seam[first_overlap_col];
    }
    for x in (last_overlap_col + 1)..out_width as usize {
        seam[x] = seam[last_overlap_col];
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
