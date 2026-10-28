//! Image-level geometry operations.
//!
//! Orientation, warp and unwarp, coarse and fine rotation, crop and flip on
//! whole images, and the inverse mapping of masks and points back to the
//! source frame.

use image::DynamicImage;
use image::GenericImageView;
use image::Rgb32FImage;
use image::Rgba;
use imageproc::geometric_transformations::Border;
use imageproc::geometric_transformations::Interpolation;
use imageproc::geometric_transformations::rotate_about_center;
use nalgebra::Vector3 as NaVector3;
use rawler::decoders::Orientation;
use rayon::prelude::*;
use serde_json::Value;
use std::borrow::Cow;
use std::f32::consts::PI;

use super::{
    GeometryParams, IntoCowImage, build_transform_matrices, get_geometry_params_from_json,
    interpolate_pixel, is_geometry_identity, project_geometry_point, resolve_geometry_crop_rect,
    solve_generic_distortion_inv, unproject_geometry_point, warp_image_geometry,
};

pub fn unwarp_image_geometry(warped_image: &DynamicImage, params: GeometryParams) -> DynamicImage {
    let src_img = warped_image.to_rgb32f();
    let (width, height) = src_img.dimensions();
    let mut out_buffer = vec![0.0f32; (width * height * 3) as usize];

    let (forward_transform, cx, cy, half_diagonal) =
        build_transform_matrices(&params, width as f32, height as f32);
    let max_radius_sq_inv = 1.0 / ((cx * cx + cy * cy) as f64);
    let hd = half_diagonal;

    let k_distortion = (params.distortion as f64 / 100.0) * 2.5;
    let lk1 = params.lens_dist_k1 as f64;
    let lk2 = params.lens_dist_k2 as f64;
    let lk3 = params.lens_dist_k3 as f64;
    let lens_dist_amt = (params.lens_distortion_amount as f64) * 2.5;

    let has_lens_correction = params.lens_distortion_enabled
        && (lk1.abs() > 1e-6 || lk2.abs() > 1e-6 || lk3.abs() > 1e-6);
    let is_ptlens = params.lens_model == 1;

    let src_raw = src_img.as_raw();
    let width_usize = width as usize;
    let height_usize = height as usize;

    out_buffer
        .par_chunks_exact_mut(width_usize * 3)
        .enumerate()
        .for_each(|(y, row_pixel_data)| {
            let y_f = y as f32;

            for (x, pixel) in row_pixel_data.chunks_exact_mut(3).enumerate() {
                let x_f = x as f32;
                let mut current_x = x_f;
                let mut current_y = y_f;

                if k_distortion.abs() > 1e-5 {
                    let dx = (current_x - cx) as f64;
                    let dy = (current_y - cy) as f64;
                    let r_distorted = (dx * dx + dy * dy).sqrt();

                    if r_distorted > 1e-6 {
                        let k_effective = k_distortion * max_radius_sq_inv;
                        let r_straight = solve_generic_distortion_inv(r_distorted, k_effective);

                        let scale = r_straight / r_distorted;
                        current_x = cx + (dx * scale) as f32;
                        current_y = cy + (dy * scale) as f32;
                    }
                }

                if has_lens_correction {
                    let dx = (current_x - cx) as f64;
                    let dy = (current_y - cy) as f64;
                    let rd = (dx * dx + dy * dy).sqrt();

                    if rd > 1e-6 {
                        let mut ru = rd;

                        for _ in 0..8 {
                            let ru_norm = ru / hd;
                            let ru_norm2 = ru_norm * ru_norm;

                            let (f_val, f_prime) = if is_ptlens {
                                let a = lk1;
                                let b = lk2;
                                let c = lk3;
                                let d = 1.0 - a - b - c;
                                let poly = a * ru_norm2 * ru_norm + b * ru_norm2 + c * ru_norm + d;

                                let val = ru * poly;
                                let prime = 4.0 * a * ru_norm2 * ru_norm
                                    + 3.0 * b * ru_norm2
                                    + 2.0 * c * ru_norm
                                    + d;
                                (val, prime)
                            } else {
                                let poly = 1.0
                                    + lk1 * ru_norm2
                                    + lk2 * (ru_norm2 * ru_norm2)
                                    + lk3 * (ru_norm2 * ru_norm2 * ru_norm2);
                                let val = ru * poly;
                                let poly_prime = 2.0 * lk1 * ru_norm
                                    + 4.0 * lk2 * ru_norm2 * ru_norm
                                    + 6.0 * lk3 * (ru_norm2 * ru_norm2) * ru_norm;
                                let prime = poly + ru_norm * poly_prime;
                                (val, prime)
                            };

                            let g_val = ru + (f_val - ru) * lens_dist_amt - rd;
                            let g_prime = 1.0 + (f_prime - 1.0) * lens_dist_amt;

                            if g_prime.abs() < 1e-7 {
                                break;
                            }
                            let delta = g_val / g_prime;
                            ru -= delta;
                            if delta.abs() < 1e-4 {
                                break;
                            }
                        }

                        let scale = ru / rd;
                        current_x = cx + (dx * scale) as f32;
                        current_y = cy + (dy * scale) as f32;
                    }
                }

                let (projected_x, projected_y) = project_geometry_point(
                    current_x as f64,
                    current_y as f64,
                    cx as f64,
                    cy as f64,
                    hd,
                    params.projection,
                );
                current_x = projected_x as f32;
                current_y = projected_y as f32;

                let target_vec = forward_transform * NaVector3::new(current_x, current_y, 1.0);

                if target_vec.z.abs() > 1e-6 {
                    let inv_z = 1.0 / target_vec.z;

                    let src_x = target_vec.x * inv_z;
                    let src_y = target_vec.y * inv_z;

                    interpolate_pixel(src_raw, width_usize, height_usize, src_x, src_y, pixel);
                }
            }
        });

    let out_img = Rgb32FImage::from_vec(width, height, out_buffer).unwrap();
    DynamicImage::ImageRgb32F(out_img)
}

pub fn inverse_transform_mask(
    mask: image::GrayImage,
    adjustments: &serde_json::Value,
) -> image::GrayImage {
    let rotation_degrees = adjustments
        .get("rotation")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let mask_dyn = image::DynamicImage::ImageLuma8(mask);

    let unrotated_fine = if rotation_degrees.abs() > 1e-5 {
        crate::image_processing::apply_rotation(mask_dyn, -rotation_degrees).into_owned()
    } else {
        mask_dyn
    };

    let flip_h = adjustments
        .get("flipHorizontal")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let flip_v = adjustments
        .get("flipVertical")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let flipped = apply_flip(unrotated_fine, flip_h, flip_v).into_owned();

    let steps = adjustments
        .get("orientationSteps")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u8;
    let inverse_steps = (4 - (steps % 4)) % 4;
    let unrotated_coarse = apply_coarse_rotation(flipped, inverse_steps).into_owned();

    let unwarped = apply_unwarp_geometry(unrotated_coarse, adjustments).into_owned();

    unwarped.into_luma8()
}

pub fn inverse_transform_point(
    mut x: f64,
    mut y: f64,
    mut curr_w: f64,
    mut curr_h: f64,
    adjustments: &serde_json::Value,
) -> (f64, f64) {
    let rotation_degrees = adjustments
        .get("rotation")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    if rotation_degrees.abs() > 1e-5 {
        let cx = curr_w / 2.0;
        let cy = curr_h / 2.0;
        let theta_rad = -rotation_degrees * std::f64::consts::PI / 180.0;
        let cos_t = theta_rad.cos();
        let sin_t = theta_rad.sin();

        let dx = x - cx;
        let dy = y - cy;
        x = cx + dx * cos_t - dy * sin_t;
        y = cy + dx * sin_t + dy * cos_t;
    }

    let flip_h = adjustments
        .get("flipHorizontal")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let flip_v = adjustments
        .get("flipVertical")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if flip_h {
        x = curr_w - x;
    }
    if flip_v {
        y = curr_h - y;
    }

    let steps = adjustments
        .get("orientationSteps")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u8;
    let inverse_steps = (4 - (steps % 4)) % 4;
    for _ in 0..inverse_steps {
        let new_x = curr_h - y;
        let new_y = x;
        x = new_x;
        y = new_y;
        std::mem::swap(&mut curr_w, &mut curr_h);
    }

    let params = get_geometry_params_from_json(adjustments);
    let width = curr_w as f32;
    let height = curr_h as f32;

    let (forward_transform, cx_f32, cy_f32, hd) = build_transform_matrices(&params, width, height);
    let cx = cx_f32 as f64;
    let cy = cy_f32 as f64;
    let inv = forward_transform
        .try_inverse()
        .unwrap_or(nalgebra::Matrix3::identity());

    let vec = inv * nalgebra::Vector3::new(x as f32, y as f32, 1.0);
    if vec.z.abs() > 1e-6 {
        let inv_z = 1.0 / (vec.z as f64);
        let mut src_x = (vec.x as f64) * inv_z;
        let mut src_y = (vec.y as f64) * inv_z;

        (src_x, src_y) = unproject_geometry_point(src_x, src_y, cx, cy, hd, params.projection);

        let k_distortion = (params.distortion as f64 / 100.0) * 2.5;
        let lk1 = params.lens_dist_k1 as f64;
        let lk2 = params.lens_dist_k2 as f64;
        let lk3 = params.lens_dist_k3 as f64;
        let lens_dist_amt = (params.lens_distortion_amount as f64) * 2.5;

        let has_lens_correction = params.lens_distortion_enabled
            && (lk1.abs() > 1e-6 || lk2.abs() > 1e-6 || lk3.abs() > 1e-6);
        let is_ptlens = params.lens_model == 1;

        if has_lens_correction {
            let dx = src_x - cx;
            let dy = src_y - cy;
            let ru = (dx * dx + dy * dy).sqrt();

            if ru > 1e-6 {
                let ru_norm = ru / hd;
                let ru_norm2 = ru_norm * ru_norm;

                let rd_norm = if is_ptlens {
                    let a = lk1;
                    let b = lk2;
                    let c = lk3;
                    let d = 1.0 - a - b - c;
                    ru_norm * (a * ru_norm2 * ru_norm + b * ru_norm2 + c * ru_norm + d)
                } else {
                    ru_norm
                        * (1.0
                            + lk1 * ru_norm2
                            + lk2 * (ru_norm2 * ru_norm2)
                            + lk3 * (ru_norm2 * ru_norm2 * ru_norm2))
                };

                let effective_r_norm = ru_norm + (rd_norm - ru_norm) * lens_dist_amt;
                let scale = effective_r_norm / ru_norm;

                src_x = cx + (dx * scale);
                src_y = cy + (dy * scale);
            }
        }

        if k_distortion.abs() > 1e-5 {
            let max_radius_sq_inv = 1.0 / (cx * cx + cy * cy);
            let dx = src_x - cx;
            let dy = src_y - cy;
            let r2_norm = (dx * dx + dy * dy) * max_radius_sq_inv;
            let f = 1.0 + k_distortion * r2_norm;

            src_x = cx + (dx * f);
            src_y = cy + (dy * f);
        }

        return (src_x, src_y);
    }

    (x, y)
}

pub fn apply_orientation(image: DynamicImage, orientation: Orientation) -> DynamicImage {
    match orientation {
        Orientation::Normal | Orientation::Unknown => image,
        Orientation::HorizontalFlip => image.fliph(),
        Orientation::Rotate180 => image.rotate180(),
        Orientation::VerticalFlip => image.flipv(),
        Orientation::Transpose => image.rotate90().fliph(),
        Orientation::Rotate90 => image.rotate90(),
        Orientation::Transverse => image.rotate270().fliph(),
        Orientation::Rotate270 => image.rotate270(),
    }
}

pub fn apply_geometry_warp<'a>(
    image: impl IntoCowImage<'a>,
    adjustments: &serde_json::Value,
) -> Cow<'a, DynamicImage> {
    let image = image.into_cow();
    let params = get_geometry_params_from_json(adjustments);
    if !is_geometry_identity(&params) {
        Cow::Owned(warp_image_geometry(image.as_ref(), params))
    } else {
        image
    }
}

pub fn apply_unwarp_geometry<'a>(
    image: impl IntoCowImage<'a>,
    adjustments: &serde_json::Value,
) -> Cow<'a, DynamicImage> {
    let image = image.into_cow();
    let params = get_geometry_params_from_json(adjustments);
    if !is_geometry_identity(&params) {
        Cow::Owned(unwarp_image_geometry(image.as_ref(), params))
    } else {
        image
    }
}

pub fn apply_coarse_rotation<'a>(
    image: impl IntoCowImage<'a>,
    orientation_steps: u8,
) -> Cow<'a, DynamicImage> {
    let image = image.into_cow();
    match orientation_steps {
        1 => Cow::Owned(image.rotate90()),
        2 => Cow::Owned(image.rotate180()),
        3 => Cow::Owned(image.rotate270()),
        _ => image,
    }
}

pub fn apply_rotation<'a>(
    image: impl IntoCowImage<'a>,
    rotation_degrees: f32,
) -> Cow<'a, DynamicImage> {
    let image = image.into_cow();
    if rotation_degrees % 360.0 == 0.0 {
        return image;
    }

    let rgba_image = image.to_rgba32f();
    let rotated = rotate_about_center(
        &rgba_image,
        rotation_degrees * PI / 180.0,
        Interpolation::Bilinear,
        Border::Constant(Rgba([0.0f32, 0.0, 0.0, 0.0])),
    );

    Cow::Owned(DynamicImage::ImageRgba32F(rotated))
}

pub fn apply_crop<'a>(image: impl IntoCowImage<'a>, crop_value: &Value) -> Cow<'a, DynamicImage> {
    let image = image.into_cow();
    let (image_width, image_height) = image.dimensions();
    let crop = resolve_geometry_crop_rect(image_width, image_height, crop_value);
    if crop.x == 0 && crop.y == 0 && crop.width == image_width && crop.height == image_height {
        return image;
    }
    Cow::Owned(image.crop_imm(crop.x, crop.y, crop.width, crop.height))
}

pub fn apply_flip<'a>(
    image: impl IntoCowImage<'a>,
    horizontal: bool,
    vertical: bool,
) -> Cow<'a, DynamicImage> {
    let image = image.into_cow();
    if !horizontal && !vertical {
        return image;
    }

    let mut img = image.into_owned();
    if horizontal {
        img = img.fliph();
    }
    if vertical {
        img = img.flipv();
    }
    Cow::Owned(img)
}
