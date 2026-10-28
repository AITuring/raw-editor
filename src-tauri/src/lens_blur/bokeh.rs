//! Aperture-shaped bokeh kernels and per-layer blur.
//!
//! Builds the tap list of a circular or polygonal aperture with its intensity
//! profile and blurs one premultiplied layer with it.

use rayon::prelude::*;

use super::{dof_box_filter, dof_smoothstep};

#[derive(Clone, Copy)]
pub(super) struct BokehTap {
    pub(super) x: i32,
    pub(super) y: i32,
    pub(super) ux: f32,
    pub(super) uy: f32,
    pub(super) weight: f32,
}

#[inline(always)]
pub(super) fn dof_polygon_scale(theta: f32, blades: u32) -> f32 {
    let seg = std::f32::consts::TAU / blades as f32;
    let half = seg * 0.5;
    let a = theta.rem_euclid(seg) - half;
    half.cos() / a.cos().max(1e-4)
}

#[inline(always)]
pub(super) fn dof_aperture_intensity(rr: f32, shape: &str, diffusion: f32) -> f32 {
    let crisp = if shape == "ring" {
        0.22 + 3.2 * dof_smoothstep(0.60, 0.93, rr)
    } else {
        1.0 + 0.12 * dof_smoothstep(0.55, 1.0, rr)
    };
    if diffusion <= 1.0e-3 {
        return crisp;
    }
    let soft = (-2.6 * rr * rr).exp() + 0.05;
    crisp * (1.0 - diffusion) + soft * diffusion
}

pub(super) fn build_bokeh_kernel(
    radius: f32,
    shape: &str,
    diffusion: f32,
) -> (Vec<BokehTap>, usize) {
    let r = radius.max(0.6);
    let blades: u32 = match shape {
        "hexagon" => 6,
        "octagon" => 8,
        _ => 0,
    };

    let mut taps: Vec<BokehTap> = Vec::new();
    let mut sum = 0.0f32;

    if r <= 3.5 {
        let ri = r.ceil() as i32;
        for y in -ri..=ri {
            for x in -ri..=ri {
                let d = ((x * x + y * y) as f32).sqrt();
                let cov = (r + 0.5 - d).clamp(0.0, 1.0);
                if cov <= 0.0 {
                    continue;
                }
                let wgt = cov * dof_aperture_intensity((d / r).min(1.0), shape, diffusion);
                sum += wgt;
                taps.push(BokehTap {
                    x,
                    y,
                    ux: x as f32 / r,
                    uy: y as f32 / r,
                    weight: wgt,
                });
            }
        }
    } else {
        let area = std::f32::consts::PI * r * r;
        let n = (area.ceil() as usize).clamp(24, 128);
        let golden = 2.399_963_2f32;

        for i in 0..n {
            let t = (i as f32 + 0.5) / n as f32;
            let rr = t.sqrt();
            let ang = i as f32 * golden;
            let (s, c) = ang.sin_cos();
            let (mut ux, mut uy) = (rr * c, rr * s);

            let mut jw = 1.0f32;
            if blades > 0 {
                let poly = dof_polygon_scale(ang, blades);
                let shaped = 1.0 + (poly - 1.0) * (1.0 - diffusion * 0.6);
                ux *= shaped;
                uy *= shaped;
                jw = shaped * shaped;
            }

            let wgt = jw * dof_aperture_intensity(rr, shape, diffusion);
            sum += wgt;
            taps.push(BokehTap {
                x: (ux * r).round() as i32,
                y: (uy * r).round() as i32,
                ux,
                uy,
                weight: wgt,
            });
        }
    }

    if sum > 0.0 {
        let inv = 1.0 / sum;
        for t in taps.iter_mut() {
            t.weight *= inv;
        }
    }

    let ext = taps
        .iter()
        .map(|t| t.x.unsigned_abs().max(t.y.unsigned_abs()) as usize)
        .max()
        .unwrap_or(0);

    (taps, ext)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn blur_layer_bokeh(
    layer: &[f32],
    ww: usize,
    wh: usize,
    taps: &[BokehTap],
    spans: &[(usize, usize)],
    ext: usize,
    radius: f32,
    diffusion: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; ww * wh * 4];
    let inv_ww = 1.0 / ww as f32;
    let inv_wh = 1.0 / wh as f32;
    let cat_eye = 0.38f32 * (1.0 - 0.7 * diffusion);
    let use_cat = cat_eye > 0.02;

    out.par_chunks_exact_mut(ww * 4)
        .enumerate()
        .for_each(|(y, row)| {
            let (xs, xe) = spans[y];
            if xs == usize::MAX {
                return;
            }
            let ny_img = (y as f32 + 0.5) * inv_wh * 2.0 - 1.0;
            let ccy = cat_eye * ny_img;
            let y_safe = y >= ext && y + ext < wh;

            for x in xs..=xe {
                let nx_img = (x as f32 + 0.5) * inv_ww * 2.0 - 1.0;
                let ccx = cat_eye * nx_img;

                let mut acc = [0.0f32; 4];
                let mut wsum = 0.0f32;

                if y_safe && x >= ext && x + ext < ww {
                    let center = (y * ww + x) as i32;
                    for t in taps {
                        let mut wgt = t.weight;
                        if use_cat {
                            let ox = t.ux - ccx;
                            let oy = t.uy - ccy;
                            let d2 = ox * ox + oy * oy;
                            if d2 >= 1.0 {
                                continue;
                            }
                            if d2 > 0.82 {
                                wgt *= 1.0 - (d2 - 0.82) / 0.18;
                            }
                        }

                        let si = ((center + t.y * ww as i32 + t.x) as usize) * 4;
                        unsafe {
                            acc[0] += layer.get_unchecked(si) * wgt;
                            acc[1] += layer.get_unchecked(si + 1) * wgt;
                            acc[2] += layer.get_unchecked(si + 2) * wgt;
                            acc[3] += layer.get_unchecked(si + 3) * wgt;
                        }
                        wsum += wgt;
                    }
                } else {
                    for t in taps {
                        let sx = x as i32 + t.x;
                        let sy = y as i32 + t.y;
                        if sx < 0 || sy < 0 || sx >= ww as i32 || sy >= wh as i32 {
                            continue;
                        }
                        let mut wgt = t.weight;
                        if use_cat {
                            let ox = t.ux - ccx;
                            let oy = t.uy - ccy;
                            let d2 = ox * ox + oy * oy;
                            if d2 >= 1.0 {
                                continue;
                            }
                            if d2 > 0.82 {
                                wgt *= 1.0 - (d2 - 0.82) / 0.18;
                            }
                        }

                        let si = (sy as usize * ww + sx as usize) * 4;
                        acc[0] += layer[si] * wgt;
                        acc[1] += layer[si + 1] * wgt;
                        acc[2] += layer[si + 2] * wgt;
                        acc[3] += layer[si + 3] * wgt;
                        wsum += wgt;
                    }
                }

                if wsum > 1.0e-8 {
                    let inv = 1.0 / wsum;
                    let di = x * 4;
                    row[di] = acc[0] * inv;
                    row[di + 1] = acc[1] * inv;
                    row[di + 2] = acc[2] * inv;
                    row[di + 3] = acc[3] * inv;
                }
            }
        });

    if radius > 7.0 {
        dof_box_filter(&mut out, ww, wh, 4, 1);
    }
    out
}
