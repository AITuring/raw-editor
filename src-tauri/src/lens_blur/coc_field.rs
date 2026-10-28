//! Signed circle-of-confusion field from the depth map.
//!
//! Maps depth to a signed blur radius between the focus range fades, refines
//! it with an image-guided filter and removes isolated speckles.

use image::Rgb32FImage;
use rayon::prelude::*;

use super::{dof_box_filter, dof_smoothstep};

#[inline(always)]
pub(super) fn dof_soft_pow(x: f32) -> f32 {
    x * (0.8 + 0.2 * x)
}

#[inline(always)]
pub(super) fn dof_med3(a: f32, b: f32, c: f32) -> f32 {
    a.min(b).max(a.max(b).min(c))
}

#[inline(always)]
pub(super) fn depth_to_signed_coc(
    d: f32,
    min_depth: f32,
    max_depth: f32,
    min_fade: f32,
    max_fade: f32,
    max_radius: f32,
) -> f32 {
    let far = 1.0 - dof_smoothstep(min_depth - min_fade.max(1e-4), min_depth, d);
    let near = dof_smoothstep(max_depth, max_depth + max_fade.max(1e-4), d);
    (dof_soft_pow(far) - dof_soft_pow(near)) * max_radius
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_coc_field(
    src: &Rgb32FImage,
    depth: &image::GrayImage,
    min_depth: f32,
    max_depth: f32,
    min_fade: f32,
    max_fade: f32,
    max_radius: f32,
) -> Vec<f32> {
    let w = src.width() as usize;
    let h = src.height() as usize;
    let raw = src.as_raw();

    let mut luma_full = vec![0.0f32; w * h];
    luma_full
        .par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let base = y * w * 3;
            for (x, out) in row.iter_mut().enumerate() {
                let i = base + x * 3;
                *out = (0.2126 * raw[i] + 0.7152 * raw[i + 1] + 0.0722 * raw[i + 2])
                    .max(0.0)
                    .sqrt();
            }
        });
    dof_box_filter(&mut luma_full, w, h, 1, 2);

    let scale = (1024.0 / w.max(h) as f32).min(1.0);
    let gw = ((w as f32 * scale).round() as usize).clamp(4, w);
    let gh = ((h as f32 * scale).round() as usize).clamp(4, h);

    let x_ratio = w as f32 / gw as f32;
    let y_ratio = h as f32 / gh as f32;

    let mut guide = vec![0.0f32; gw * gh];
    guide
        .par_chunks_exact_mut(gw)
        .enumerate()
        .for_each(|(gy, row)| {
            let sy0 = (gy as f32 * y_ratio).floor() as usize;
            let sy1 = ((((gy + 1) as f32) * y_ratio).ceil() as usize)
                .min(h)
                .max(sy0 + 1);
            for (gx, out) in row.iter_mut().enumerate() {
                let sx0 = (gx as f32 * x_ratio).floor() as usize;
                let sx1 = ((((gx + 1) as f32) * x_ratio).ceil() as usize)
                    .min(w)
                    .max(sx0 + 1);
                let mut acc = 0.0f32;
                let mut cnt = 0.0f32;
                for sy in sy0..sy1 {
                    let row_off = sy * w;
                    for sx in sx0..sx1 {
                        acc += luma_full[row_off + sx];
                        cnt += 1.0;
                    }
                }
                *out = acc / cnt.max(1.0);
            }
        });

    let depth_small = dof_depth_to_f32(depth, gw, gh);
    let radius = ((gw.max(gh) as f32) * 0.015).round().max(2.0) as usize;
    let packed = build_guided_model(&guide, &depth_small, gw, gh, radius);

    let mut coc = vec![0.0f32; w * h];
    let gx_scale = gw as f32 / w as f32;
    let gy_scale = gh as f32 / h as f32;

    coc.par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let fy = ((y as f32 + 0.5) * gy_scale - 0.5).clamp(0.0, (gh - 1) as f32);
            let gy0 = fy.floor() as usize;
            let gy1 = (gy0 + 1).min(gh - 1);
            let wy = fy - gy0 as f32;
            let r0 = gy0 * gw;
            let r1 = gy1 * gw;
            let lrow = y * w;

            for (x, out) in row.iter_mut().enumerate() {
                let fx = ((x as f32 + 0.5) * gx_scale - 0.5).clamp(0.0, (gw - 1) as f32);
                let gx0 = fx.floor() as usize;
                let gx1 = (gx0 + 1).min(gw - 1);
                let wx = fx - gx0 as f32;

                let i00 = (r0 + gx0) * 4;
                let i10 = (r0 + gx1) * 4;
                let i01 = (r1 + gx0) * 4;
                let i11 = (r1 + gx1) * 4;

                let mut m = [0.0f32; 4];
                for (c, mc) in m.iter_mut().enumerate() {
                    let top = packed[i00 + c] + (packed[i10 + c] - packed[i00 + c]) * wx;
                    let bot = packed[i01 + c] + (packed[i11 + c] - packed[i01 + c]) * wx;
                    *mc = top + (bot - top) * wy;
                }

                let luma = luma_full[lrow + x];
                let d = (m[0] * luma + m[1]).clamp(m[2], m[3]).clamp(0.0, 1.0);
                *out = depth_to_signed_coc(d, min_depth, max_depth, min_fade, max_fade, max_radius);
            }
        });

    dof_despeckle(&mut coc, w, h);
    coc
}

pub(super) fn build_guided_model(
    guide: &[f32],
    p: &[f32],
    w: usize,
    h: usize,
    radius: usize,
) -> Vec<f32> {
    let n = w * h;
    let eps = 4.0e-3f32;

    let mut mean_i = guide.to_vec();
    dof_box_filter(&mut mean_i, w, h, 1, radius);
    let mut mean_p = p.to_vec();
    dof_box_filter(&mut mean_p, w, h, 1, radius);

    let mut corr_ii: Vec<f32> = guide.iter().map(|v| v * v).collect();
    dof_box_filter(&mut corr_ii, w, h, 1, radius);
    let mut corr_ip: Vec<f32> = guide.iter().zip(p.iter()).map(|(a, b)| a * b).collect();
    dof_box_filter(&mut corr_ip, w, h, 1, radius);
    let mut corr_pp: Vec<f32> = p.iter().map(|v| v * v).collect();
    dof_box_filter(&mut corr_pp, w, h, 1, radius);

    let mut gx = vec![0.0f32; n];
    let mut gy = vec![0.0f32; n];
    gx.par_chunks_exact_mut(w)
        .zip(gy.par_chunks_exact_mut(w))
        .enumerate()
        .for_each(|(y, (rx, ry))| {
            let yc = y * w;
            let yu = y.saturating_sub(1) * w;
            let yd = (y + 1).min(h - 1) * w;
            for x in 0..w {
                let xl = x.saturating_sub(1);
                let xr = (x + 1).min(w - 1);
                rx[x] = (guide[yc + xr] - guide[yc + xl]) * 0.5;
                ry[x] = (guide[yd + x] - guide[yu + x]) * 0.5;
            }
        });
    dof_box_filter(&mut gx, w, h, 1, radius);
    dof_box_filter(&mut gy, w, h, 1, radius);

    let span = (2 * radius + 1) as f32;
    let span2 = span * span;

    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    let mut lo = vec![0.0f32; n];
    let mut hi = vec![0.0f32; n];

    for i in 0..n {
        let var = (corr_ii[i] - mean_i[i] * mean_i[i]).max(0.0);
        let cov = corr_ip[i] - mean_i[i] * mean_p[i];

        let structure = (gx[i] * gx[i] + gy[i] * gy[i]) * span2;
        let conf = structure / (structure + 1.4 * var + 1.0e-5);

        a[i] = conf * cov / (var + eps);
        b[i] = mean_p[i] - a[i] * mean_i[i];

        let sd = (corr_pp[i] - mean_p[i] * mean_p[i]).max(0.0).sqrt();
        let band = 1.1 * sd + 0.004;
        lo[i] = mean_p[i] - band;
        hi[i] = mean_p[i] + band;
    }

    dof_box_filter(&mut a, w, h, 1, radius);
    dof_box_filter(&mut b, w, h, 1, radius);

    let mut packed = vec![0.0f32; n * 4];
    packed
        .par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(i, px)| {
            px[0] = a[i];
            px[1] = b[i];
            px[2] = lo[i];
            px[3] = hi[i];
        });
    packed
}

pub(super) fn dof_despeckle(buf: &mut [f32], w: usize, h: usize) {
    if w < 3 || h < 3 {
        return;
    }

    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let base = y * w;
            row[0] = buf[base];
            for x in 1..w - 1 {
                row[x] = dof_med3(buf[base + x - 1], buf[base + x], buf[base + x + 1]);
            }
            row[w - 1] = buf[base + w - 1];
        });

    buf.par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            if y == 0 || y == h - 1 {
                row.copy_from_slice(&tmp[y * w..y * w + w]);
                return;
            }
            let up = (y - 1) * w;
            let cc = y * w;
            let dn = (y + 1) * w;
            for x in 0..w {
                row[x] = dof_med3(tmp[up + x], tmp[cc + x], tmp[dn + x]);
            }
        });
}

pub(super) fn dof_depth_to_f32(depth: &image::GrayImage, dw: usize, dh: usize) -> Vec<f32> {
    let sw = depth.width() as usize;
    let sh = depth.height() as usize;
    let raw = depth.as_raw();
    let mut out = vec![0.0f32; dw * dh];

    out.par_chunks_exact_mut(dw)
        .enumerate()
        .for_each(|(y, row)| {
            let fy = (((y as f32 + 0.5) / dh as f32) * sh as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(sh - 1);
            let wy = fy - y0 as f32;

            for (x, out_px) in row.iter_mut().enumerate() {
                let fx =
                    (((x as f32 + 0.5) / dw as f32) * sw as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(sw - 1);
                let wx = fx - x0 as f32;

                let p00 = raw[y0 * sw + x0] as f32;
                let p10 = raw[y0 * sw + x1] as f32;
                let p01 = raw[y1 * sw + x0] as f32;
                let p11 = raw[y1 * sw + x1] as f32;
                let top = p00 + (p10 - p00) * wx;
                let bot = p01 + (p11 - p01) * wx;
                *out_px = (top + (bot - top) * wy) / 255.0;
            }
        });
    out
}
