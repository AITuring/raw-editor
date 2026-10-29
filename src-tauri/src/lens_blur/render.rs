//! Lens-blur entry point and the layered depth-of-field renderer.
//!
//! Reads the lens-blur adjustments and depth map, builds the signed CoC field
//! and renders it on a downsampled working image: far and near layers are
//! composited per radius band, optionally diffused, and merged back with the
//! sharp source.

use crate::cache_utils::is_section_visible_with_legacy;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::DynamicImage;
use image::GenericImageView;
use image::Rgb32FImage;
use image::Rgba;
use image::Rgba32FImage;
use rayon::prelude::*;
use std::borrow::Cow;

use super::{
    blur_layer_bokeh, build_bokeh_kernel, build_coc_field, dof_box_filter, dof_smoothstep,
};

pub fn apply_lens_blur<'a>(
    image: Cow<'a, DynamicImage>,
    adjustments: &serde_json::Value,
) -> Cow<'a, DynamicImage> {
    let lens_blur_visible = is_section_visible_with_legacy(adjustments, "lensBlur", "effects");

    if !adjustments["lensBlurEnabled"].as_bool().unwrap_or(false) || !lens_blur_visible {
        return image;
    }

    let depth_b64 = adjustments["lensBlurDepthMap"].as_str().unwrap_or("");
    if depth_b64.is_empty() {
        return image;
    }

    let amount = adjustments["lensBlurAmount"].as_f64().unwrap_or(50.0) as f32;
    if amount <= 0.0 {
        return image;
    }

    let b64_data = match depth_b64.find(',') {
        Some(idx) => &depth_b64[idx + 1..],
        None => depth_b64,
    };
    let decoded = match BASE64.decode(b64_data) {
        Ok(b) => b,
        Err(_) => return image,
    };
    let depth_map = match image::load_from_memory(&decoded) {
        Ok(img) => img.into_luma8(),
        Err(_) => return image,
    };
    if depth_map.width() < 2 || depth_map.height() < 2 {
        return image;
    }

    let (w, h) = image.dimensions();
    if w < 8 || h < 8 {
        return image;
    }

    let max_radius = (amount / 100.0) * (w.max(h) as f32) * 0.012;
    if max_radius < 0.35 {
        return image;
    }

    let shape = adjustments["lensBlurShape"].as_str().unwrap_or("circle");
    let min_depth = adjustments["lensBlurMinDepth"].as_f64().unwrap_or(20.0) as f32 / 100.0;
    let max_depth = adjustments["lensBlurMaxDepth"].as_f64().unwrap_or(100.0) as f32 / 100.0;
    let min_fade = adjustments["lensBlurMinFade"].as_f64().unwrap_or(15.0) as f32 / 100.0;
    let max_fade = adjustments["lensBlurMaxFade"].as_f64().unwrap_or(15.0) as f32 / 100.0;
    let diffusion =
        (adjustments["lensBlurDiffusion"].as_f64().unwrap_or(0.0) as f32 / 100.0).clamp(0.0, 1.0);

    let start = std::time::Instant::now();
    let src = image.as_ref().to_rgb32f();

    let coc = build_coc_field(
        &src, &depth_map, min_depth, max_depth, min_fade, max_fade, max_radius,
    );

    let blurred_px = coc.par_iter().filter(|c| c.abs() > 0.4).count();
    if blurred_px * 400 < coc.len() && diffusion <= 0.0 {
        return image;
    }

    let out = render_depth_of_field(&src, &coc, max_radius, shape, diffusion);

    log::info!(
        "lens blur ({}x{}, r_max {:.1}px, diffusion {:.2}) took {:.2?}",
        w,
        h,
        max_radius,
        diffusion,
        start.elapsed()
    );

    if image.color().has_alpha() {
        let source_rgba = image.as_ref().to_rgba32f();
        let with_alpha = Rgba32FImage::from_fn(w, h, |x, y| {
            let rgb = out.get_pixel(x, y);
            let alpha = source_rgba.get_pixel(x, y)[3];
            Rgba([rgb[0], rgb[1], rgb[2], alpha])
        });
        Cow::Owned(DynamicImage::ImageRgba32F(with_alpha))
    } else {
        Cow::Owned(DynamicImage::ImageRgb32F(out))
    }
}

#[inline(always)]
pub(super) fn dof_to_energy(v: f32) -> f32 {
    let x = v.max(0.0);
    let e = x * x;
    e - (e - 12.0).max(0.0) * 0.85
}

#[inline(always)]
pub(super) fn dof_tent(c: f32, radii: &[f32; 5], k: usize) -> f32 {
    if c <= radii[k] {
        if k == 0 {
            1.0
        } else {
            ((c - radii[k - 1]) / (radii[k] - radii[k - 1]).max(1e-5)).clamp(0.0, 1.0)
        }
    } else if k == 4 {
        1.0
    } else {
        1.0 - ((c - radii[k]) / (radii[k + 1] - radii[k]).max(1e-5)).clamp(0.0, 1.0)
    }
}

pub(super) fn render_depth_of_field(
    src: &Rgb32FImage,
    coc: &[f32],
    max_radius: f32,
    shape: &str,
    diffusion: f32,
) -> Rgb32FImage {
    let w = src.width() as usize;
    let h = src.height() as usize;
    let n = w * h;
    let raw = src.as_raw();

    let mut ds = (max_radius / 18.0).ceil().max(1.0) as usize;
    while ds < 6 {
        let rw = max_radius / ds as f32;
        let taps = (std::f32::consts::PI * rw * rw).min(128.0) as f64;
        if (n as f64 / (ds * ds) as f64) * taps <= 7.0e8 {
            break;
        }
        ds += 1;
    }
    let ww = (w / ds).max(4);
    let wh = (h / ds).max(4);

    let (base, coc_small) = dof_downsample_fused(raw, coc, w, h, ww, wh);

    let mut radii = [0.0f32; 5];
    for (k, r) in radii.iter_mut().enumerate() {
        *r = max_radius * (k as f32 / 4.0).powf(1.7);
    }

    let (far_rgb, far_a) = dof_composite_stack(
        &base, &coc_small, ww, wh, &radii, ds, shape, diffusion, false,
    );
    let (near_rgb, near_a) = dof_composite_stack(
        &base, &coc_small, ww, wh, &radii, ds, shape, diffusion, true,
    );

    let diffusion_on = diffusion > 1.0e-3;
    let bloom = if diffusion_on {
        let mut bl = vec![0.0f32; ww * wh * 4];
        bl.par_chunks_exact_mut(4).enumerate().for_each(|(i, px)| {
            px[0] = base[i * 3];
            px[1] = base[i * 3 + 1];
            px[2] = base[i * 3 + 2];
            px[3] = 1.0 - dof_tent(coc_small[i].abs(), &radii, 0);
        });
        let cap = (ww.min(wh) / 2).max(2);
        let br1 = (((max_radius / ds as f32) * 0.8).max(6.0) as usize).min(cap);
        let br2 = ((br1 * 2) / 3).max(3).min(cap);
        dof_box_filter(&mut bl, ww, wh, 4, br1);
        dof_box_filter(&mut bl, ww, wh, 4, br2);
        bl.par_chunks_exact_mut(4).for_each(|px| {
            px[0] = px[0].max(0.0).sqrt();
            px[1] = px[1].max(0.0).sqrt();
            px[2] = px[2].max(0.0).sqrt();
            px[3] = px[3].clamp(0.0, 1.0);
        });
        bl
    } else {
        Vec::new()
    };

    let mut far_lin = vec![0.0f32; ww * wh * 4];
    far_lin
        .par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(i, px)| {
            let a = far_a[i].clamp(0.0, 1.0);
            if a > 1.0e-4 {
                let inv_a = 1.0 / a;
                for c in 0..3 {
                    px[c] = (far_rgb[i * 3 + c] * inv_a).max(0.0).sqrt();
                }
            } else {
                for c in 0..3 {
                    px[c] = base[i * 3 + c].max(0.0).sqrt();
                }
            }
            px[3] = dof_tent(coc_small[i].abs(), &radii, 0);
        });

    let mut near_lin = vec![0.0f32; ww * wh * 4];
    near_lin
        .par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(i, px)| {
            let a = near_a[i].clamp(0.0, 1.0);
            if a > 1.0e-4 {
                let inv_a = 1.0 / a;
                for c in 0..3 {
                    px[c] = (near_rgb[i * 3 + c] * inv_a).max(0.0).sqrt() * a;
                }
            }
            px[3] = a;
        });

    let mut out = vec![0.0f32; n * 3];
    let sx = ww as f32 / w as f32;
    let sy = wh as f32 / h as f32;
    let veil = 0.75 * diffusion;

    out.par_chunks_exact_mut(w * 3)
        .enumerate()
        .for_each(|(y, row)| {
            let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (wh - 1) as f32);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(wh - 1);
            let wy = fy - y0 as f32;
            let r0 = y0 * ww;
            let r1 = y1 * ww;
            let coc_row = y * w;
            let src_row = coc_row * 3;

            for x in 0..w {
                let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (ww - 1) as f32);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(ww - 1);
                let wx = fx - x0 as f32;

                let i00 = (r0 + x0) * 4;
                let i10 = (r0 + x1) * 4;
                let i01 = (r1 + x0) * 4;
                let i11 = (r1 + x1) * 4;

                let mut fs = [0.0f32; 4];
                let mut ns = [0.0f32; 4];
                for c in 0..4 {
                    let ft = far_lin[i00 + c] + (far_lin[i10 + c] - far_lin[i00 + c]) * wx;
                    let fb = far_lin[i01 + c] + (far_lin[i11 + c] - far_lin[i01 + c]) * wx;
                    fs[c] = ft + (fb - ft) * wy;

                    let nt = near_lin[i00 + c] + (near_lin[i10 + c] - near_lin[i00 + c]) * wx;
                    let nb = near_lin[i01 + c] + (near_lin[i11 + c] - near_lin[i01 + c]) * wx;
                    ns[c] = nt + (nb - nt) * wy;
                }

                let fa = dof_tent(coc[coc_row + x].abs(), &radii, 0).min(fs[3]);
                let na = ns[3].clamp(0.0, 1.0);
                let inv_na = 1.0 - na;

                if diffusion_on {
                    let j00 = (r0 + x0) * 4;
                    let j10 = (r0 + x1) * 4;
                    let j01 = (r1 + x0) * 4;
                    let j11 = (r1 + x1) * 4;

                    let mut bv = [0.0f32; 3];
                    let mut b_mask = 0.0f32;

                    for c in 0..4 {
                        let bt = bloom[j00 + c] + (bloom[j10 + c] - bloom[j00 + c]) * wx;
                        let bb = bloom[j01 + c] + (bloom[j11 + c] - bloom[j01 + c]) * wx;
                        let val = bt + (bb - bt) * wy;
                        if c < 3 {
                            bv[c] = val;
                        } else {
                            b_mask = val;
                        }
                    }

                    let final_mask = dof_smoothstep(0.0, 1.0, b_mask.clamp(0.0, 1.0));

                    let local_veil = veil * final_mask;
                    let local_keep = 1.0 - (0.55 * diffusion * final_mask);

                    for c in 0..3 {
                        let sharp = raw[src_row + x * 3 + c];
                        let mid = sharp * fa + fs[c] * (1.0 - fa);
                        let base_px = ns[c] + mid * inv_na;

                        row[x * 3 + c] = (base_px * local_keep + bv[c] * local_veil).max(0.0);
                    }
                } else {
                    for c in 0..3 {
                        let sharp = raw[src_row + x * 3 + c];
                        let mid = sharp * fa + fs[c] * (1.0 - fa);
                        row[x * 3 + c] = (ns[c] + mid * inv_na).max(0.0);
                    }
                }
            }
        });

    Rgb32FImage::from_raw(w as u32, h as u32, out).expect("dof buffer size mismatch")
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dof_composite_stack(
    base: &[f32],
    coc: &[f32],
    ww: usize,
    wh: usize,
    radii: &[f32; 5],
    ds: usize,
    shape: &str,
    diffusion: f32,
    near_side: bool,
) -> (Vec<f32>, Vec<f32>) {
    let np = ww * wh;
    let mut acc_rgb = vec![0.0f32; np * 3];
    let mut acc_a = vec![0.0f32; np];

    let order: Vec<usize> = if near_side {
        (1..5).collect()
    } else {
        (1..5).rev().collect()
    };

    for k in order {
        let mut layer = vec![0.0f32; np * 4];
        let row_bounds: Vec<(usize, usize)> = layer
            .par_chunks_exact_mut(ww * 4)
            .enumerate()
            .map(|(y, row)| {
                let mut min_x = usize::MAX;
                let mut max_x = 0usize;
                for x in 0..ww {
                    let c = coc[y * ww + x];
                    if (near_side && c >= 0.0) || (!near_side && c <= 0.0) {
                        continue;
                    }
                    let wgt = dof_tent(c.abs(), radii, k);
                    if wgt <= 1.0e-4 {
                        continue;
                    }
                    let si = (y * ww + x) * 3;
                    let di = x * 4;
                    row[di] = base[si] * wgt;
                    row[di + 1] = base[si + 1] * wgt;
                    row[di + 2] = base[si + 2] * wgt;
                    row[di + 3] = wgt;
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                }
                (min_x, max_x)
            })
            .collect();

        if row_bounds.iter().all(|&(mn, _)| mn == usize::MAX) {
            continue;
        }

        let r_work = (radii[k] / ds as f32) * (1.0 + 0.3 * diffusion);
        let (blurred, spans) = if r_work >= 0.6 {
            let (taps, ext) = build_bokeh_kernel(r_work, shape, diffusion);
            let spans = dof_dilate_spans(&row_bounds, ww, wh, ext);
            let b = blur_layer_bokeh(&layer, ww, wh, &taps, &spans, ext, r_work, diffusion);
            (b, spans)
        } else {
            let spans = dof_dilate_spans(&row_bounds, ww, wh, 0);
            (layer, spans)
        };

        acc_rgb
            .par_chunks_exact_mut(ww * 3)
            .zip(acc_a.par_chunks_exact_mut(ww))
            .enumerate()
            .for_each(|(y, (rgb_row, a_row))| {
                let (xs, xe) = spans[y];
                if xs == usize::MAX {
                    return;
                }
                for x in xs..=xe {
                    let si = (y * ww + x) * 4;
                    let sa = blurred[si + 3];
                    if sa <= 1.0e-5 {
                        continue;
                    }
                    let inv = 1.0 - sa.min(1.0);
                    for c in 0..3 {
                        rgb_row[x * 3 + c] = blurred[si + c] + rgb_row[x * 3 + c] * inv;
                    }
                    a_row[x] = (sa + a_row[x] * inv).min(1.0);
                }
            });
    }

    (acc_rgb, acc_a)
}

pub(super) fn dof_dilate_spans(
    row_bounds: &[(usize, usize)],
    ww: usize,
    wh: usize,
    ext: usize,
) -> Vec<(usize, usize)> {
    let pad = ext + 1;
    let mut spans = vec![(usize::MAX, 0usize); wh];
    for (y, span) in spans.iter_mut().enumerate() {
        let y0 = y.saturating_sub(pad);
        let y1 = (y + pad).min(wh - 1);
        let mut mn = usize::MAX;
        let mut mx = 0usize;
        for &(a, b) in &row_bounds[y0..=y1] {
            if a == usize::MAX {
                continue;
            }
            mn = mn.min(a);
            mx = mx.max(b);
        }
        if mn != usize::MAX {
            *span = (mn.saturating_sub(pad), (mx + pad).min(ww - 1));
        }
    }
    spans
}

pub(super) fn dof_downsample_fused(
    src: &[f32],
    coc: &[f32],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut base = vec![0.0f32; dw * dh * 3];
    let mut coc_small = vec![0.0f32; dw * dh];
    let x_ratio = sw as f32 / dw as f32;
    let y_ratio = sh as f32 / dh as f32;

    base.par_chunks_exact_mut(dw * 3)
        .zip(coc_small.par_chunks_exact_mut(dw))
        .enumerate()
        .for_each(|(dy, (brow, crow))| {
            let sy0 = (dy as f32 * y_ratio).floor() as usize;
            let sy1 = ((((dy + 1) as f32) * y_ratio).ceil() as usize)
                .min(sh)
                .max(sy0 + 1);
            for dx in 0..dw {
                let sx0 = (dx as f32 * x_ratio).floor() as usize;
                let sx1 = ((((dx + 1) as f32) * x_ratio).ceil() as usize)
                    .min(sw)
                    .max(sx0 + 1);

                let mut r = 0.0f32;
                let mut g = 0.0f32;
                let mut b = 0.0f32;
                let mut cv = 0.0f32;
                let mut cnt = 0.0f32;

                for sy in sy0..sy1 {
                    let row = sy * sw;
                    for sx in sx0..sx1 {
                        let i = (row + sx) * 3;
                        r += dof_to_energy(src[i]);
                        g += dof_to_energy(src[i + 1]);
                        b += dof_to_energy(src[i + 2]);
                        cv += coc[row + sx];
                        cnt += 1.0;
                    }
                }

                let inv = 1.0 / cnt.max(1.0);
                brow[dx * 3] = r * inv;
                brow[dx * 3 + 1] = g * inv;
                brow[dx * 3 + 2] = b * inv;
                crow[dx] = cv * inv;
            }
        });

    (base, coc_small)
}
