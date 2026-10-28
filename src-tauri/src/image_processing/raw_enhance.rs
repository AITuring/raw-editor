//! RAW artifact removal and gentle detail enhancement.
//!
//! Luma/chroma processing of a demosaiced float buffer before editing.

use image::DynamicImage;
use image::Rgb32FImage;
use rayon::prelude::*;

#[inline(always)]
pub(super) fn rgb_to_yc_only(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168736 * r - 0.331264 * g + 0.5 * b;
    let cr = 0.5 * r - 0.418688 * g - 0.081312 * b;
    (y, cb, cr)
}

#[inline(always)]
pub(super) fn yc_to_rgb(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
    let r = y + 1.402 * cr;
    let g = y - 0.344136 * cb - 0.714136 * cr;
    let b = y + 1.772 * cb;
    (r, g, b)
}

pub fn remove_raw_artifacts_and_enhance(
    image: &mut DynamicImage,
    color_nr_inv_sigma: f32,
    sharpening_amount: f32,
) {
    if let Some(buffer) = image.as_mut_rgb32f() {
        enhance_rgb32f_buffer(buffer, color_nr_inv_sigma, sharpening_amount);
        return;
    }

    let mut buffer = image.to_rgb32f();
    enhance_rgb32f_buffer(&mut buffer, color_nr_inv_sigma, sharpening_amount);
    *image = DynamicImage::ImageRgb32F(buffer);
}

pub(super) fn enhance_rgb32f_buffer(
    buffer: &mut Rgb32FImage,
    color_nr_inv_sigma: f32,
    sharpening_amount: f32,
) {
    let w = buffer.width() as usize;
    let h = buffer.height() as usize;

    let mut ycbcr_buffer = vec![0.0f32; w * h * 3];

    let src = buffer.as_raw();

    ycbcr_buffer
        .par_chunks_mut(3)
        .zip(src.par_chunks(3))
        .for_each(|(dest, pixel)| {
            let (y, cb, cr) = rgb_to_yc_only(pixel[0], pixel[1], pixel[2]);
            dest[0] = y;
            dest[1] = cb;
            dest[2] = cr;
        });

    if color_nr_inv_sigma > 0.0 {
        let base_inv_sigma = color_nr_inv_sigma;
        const OFFSETS: [isize; 3] = [-5, -1, 3];
        const OFFSET_SQUARES: [f32; 3] = [25.0, 1.0, 9.0];

        buffer
            .par_chunks_mut(w * 3)
            .enumerate()
            .for_each(|(y, row)| {
                let row_offset = y * w;
                let h_isize = h as isize;
                let w_isize = w as isize;
                let y_isize = y as isize;

                for x in 0..w {
                    let center_idx = (row_offset + x) * 3;

                    let cy = ycbcr_buffer[center_idx];
                    let ccb = ycbcr_buffer[center_idx + 1];
                    let ccr = ycbcr_buffer[center_idx + 2];

                    let mut cb_sum = 0.0;
                    let mut cr_sum = 0.0;
                    let mut w_sum = 0.0;

                    for (ki, &ky) in OFFSETS.iter().enumerate() {
                        let sy = y_isize + ky;
                        if sy < 0 || sy >= h_isize {
                            continue;
                        }

                        let neighbor_row_idx = (sy as usize) * w;
                        let ky_sq_div_50 = OFFSET_SQUARES[ki] * 0.02;

                        for (kj, &kx) in OFFSETS.iter().enumerate() {
                            let sx = (x as isize) + kx;
                            if sx < 0 || sx >= w_isize {
                                continue;
                            }

                            let neighbor_idx = (neighbor_row_idx + sx as usize) * 3;

                            let neighbor_y = ycbcr_buffer[neighbor_idx];
                            let y_diff = (cy - neighbor_y).abs();

                            let val = y_diff * base_inv_sigma;
                            let spatial_penalty = OFFSET_SQUARES[kj] * 0.02 + ky_sq_div_50;

                            let weight = 1.0 / (1.0 + val * val + spatial_penalty);

                            cb_sum += ycbcr_buffer[neighbor_idx + 1] * weight;
                            cr_sum += ycbcr_buffer[neighbor_idx + 2] * weight;
                            w_sum += weight;
                        }
                    }

                    let (out_cb, out_cr) = if w_sum > 1e-4 {
                        let inv_w_sum = 1.0 / w_sum;
                        let filtered_cb = cb_sum * inv_w_sum;
                        let filtered_cr = cr_sum * inv_w_sum;

                        let orig_mag_sq = ccb * ccb + ccr * ccr;
                        let filt_mag_sq = filtered_cb * filtered_cb + filtered_cr * filtered_cr;

                        if filt_mag_sq > orig_mag_sq && orig_mag_sq > 1e-12 {
                            let scale = (orig_mag_sq / filt_mag_sq).sqrt();
                            (filtered_cb * scale, filtered_cr * scale)
                        } else {
                            (filtered_cb, filtered_cr)
                        }
                    } else {
                        (ccb, ccr)
                    };

                    let (r, g, b) = yc_to_rgb(cy, out_cb, out_cr);

                    let o = x * 3;
                    row[o] = r.clamp(0.0, 1.0);
                    row[o + 1] = g.clamp(0.0, 1.0);
                    row[o + 2] = b.clamp(0.0, 1.0);
                }
            });
    }

    if sharpening_amount > 0.0 {
        apply_gentle_detail_enhance(buffer, &ycbcr_buffer, sharpening_amount);
    }
}

pub(super) fn apply_gentle_detail_enhance(
    buffer: &mut image::ImageBuffer<image::Rgb<f32>, Vec<f32>>,
    ycbcr_source: &[f32],
    amount: f32,
) {
    let w = buffer.width() as usize;
    let h = buffer.height() as usize;

    let mut temp_blur = vec![0.0; w * h];
    let radius = 2i32;

    temp_blur
        .par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let row_offset = y * w;
            for (x, row_val) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                let mut count = 0;
                for kx in -radius..=radius {
                    let sx = (x as i32 + kx).clamp(0, (w as i32) - 1) as usize;
                    sum += ycbcr_source[(row_offset + sx) * 3];
                    count += 1;
                }
                *row_val = sum / count as f32;
            }
        });

    let output = buffer.as_mut();

    output
        .par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, rgb_row)| {
            for x in 0..w {
                let mut blur_sum = 0.0;
                let mut count = 0;
                for ky in -radius..=radius {
                    let sy = (y as i32 + ky).clamp(0, (h as i32) - 1) as usize;
                    blur_sum += temp_blur[sy * w + x];
                    count += 1;
                }
                let blurred_val = blur_sum / count as f32;

                let original_luma = ycbcr_source[(y * w + x) * 3];

                let detail = original_luma - blurred_val;

                let edge_strength = detail.abs();
                let adaptive_amount = if edge_strength > 0.1 {
                    amount * 0.3
                } else {
                    amount
                };
                let boost = detail * adaptive_amount;

                let r_idx = x * 3;
                let g_idx = r_idx + 1;
                let b_idx = r_idx + 2;

                let r = rgb_row[r_idx];
                let g = rgb_row[g_idx];
                let b = rgb_row[b_idx];

                let new_r = r + boost;
                let new_g = g + boost;
                let new_b = b + boost;

                let max_val = new_r.max(new_g).max(new_b);
                let min_val = new_r.min(new_g).min(new_b);

                let scale = if max_val > 1.0 || min_val < 0.0 {
                    if max_val > 1.0 && min_val < 0.0 {
                        0.0
                    } else if max_val > 1.0 {
                        (1.0 - r.max(g).max(b)) / boost.max(0.001)
                    } else {
                        r.min(g).min(b) / (-boost).max(0.001)
                    }
                } else {
                    1.0
                };

                let safe_boost = boost * scale.clamp(0.0, 1.0);

                rgb_row[r_idx] = (r + safe_boost).clamp(0.0, 1.0);
                rgb_row[g_idx] = (g + safe_boost).clamp(0.0, 1.0);
                rgb_row[b_idx] = (b + safe_boost).clamp(0.0, 1.0);
            }
        });
}
