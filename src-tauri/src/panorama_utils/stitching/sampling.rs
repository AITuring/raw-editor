//! Homography warping and high-quality source sampling.
//!
//! Bilinear and bicubic sampling with support checks, plus the full-image
//! homography warp.

use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point3;
use rayon::prelude::*;

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

pub(super) fn get_interpolated_pixel(img: &Rgb32FImage, x: f64, y: f64) -> Rgb<f32> {
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

pub(super) fn cubic_sample(p0: f64, p1: f64, p2: f64, p3: f64, amount: f64) -> f64 {
    p1 + 0.5
        * amount
        * (p2 - p0
            + amount * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + amount * (3.0 * (p1 - p2) + p3 - p0)))
}

/// Return whether a source coordinate has actual image support.
///
/// Warped virtual tiles are stored in a rectangular RGB buffer even when the
/// projective footprint is triangular or otherwise clipped.  Those uncovered
/// corners are represented by exact black pixels, so a rectangle-only bounds
/// check would incorrectly feed them into the seam and its low-frequency
/// pyramid.  Real artwork may contain very dark strokes, but a decoded image
/// sample still has non-zero energy in at least one channel; requiring a small
/// positive floor keeps those samples while rejecting the transparent zero
/// support used by the renderer.
#[inline]
pub(super) fn source_sample_has_support(image: &Rgb32FImage, x: f64, y: f64) -> bool {
    const SUPPORT_EPSILON: f32 = 1e-6;
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= width as f64
        || y >= height as f64
    {
        return false;
    }
    // Use the nearest source sample rather than the interpolated value.  A
    // cubic sample near a transparent corner can be non-zero solely because
    // its support straddles the valid footprint; the nearest source pixel is
    // the stable ownership signal and avoids re-introducing that corner.
    let source_x = (x.floor() as u32).min(width - 1);
    let source_y = (y.floor() as u32).min(height - 1);
    let pixel = image.get_pixel(source_x, source_y);
    pixel.0.iter().all(|value| value.is_finite())
        && pixel.0.iter().any(|value| *value > SUPPORT_EPSILON)
}

pub(crate) fn get_high_quality_interpolated_pixel(img: &Rgb32FImage, x: f64, y: f64) -> Rgb<f32> {
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
