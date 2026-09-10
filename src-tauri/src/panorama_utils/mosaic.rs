//! Focus mosaics: refine residual geometry, choose sharp source detail, match
//! only broad colour. The output is never an average of displaced fine detail.
use super::registration::refine_warped_patch;
use super::seam_cut;
use super::stitching::{
    Projection, crop_to_valid_rectangle, downsample_rgb_half, get_high_quality_interpolated_pixel,
    map_target_to_source, output_bounds, pixel_aligned_canvas, transformed_image_region,
};
use crate::panorama_stitching::ImageInfo;
use image::{GrayImage, Luma, Rgb, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Point3};
use rayon::prelude::*;
use std::collections::HashMap;
use tauri::{AppHandle, Emitter, Runtime};

const ANALYSIS_LONG_SIDE: u32 = 1400;
const SELECTION_LONG_SIDE: u32 = 320;
const GRID_STEP: u32 = 48;

fn luma(p: Rgb<f32>) -> f64 {
    f64::from(p[0] * 0.299 + p[1] * 0.587 + p[2] * 0.114)
}

fn rgb_at(image: &Rgb32FImage, x: f64, y: f64) -> Option<Rgb<f32>> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= image.width() as f64
        || y >= image.height() as f64
    {
        return None;
    }
    Some(get_high_quality_interpolated_pixel(image, x, y))
}

#[derive(Clone)]
struct Field<const N: usize> {
    width: usize,
    height: usize,
    step: f64,
    values: Vec<[f64; N]>,
}

impl<const N: usize> Field<N> {
    fn new(width: u32, height: u32, step: f64) -> Self {
        let width = (width as f64 / step).ceil() as usize + 1;
        let height = (height as f64 / step).ceil() as usize + 1;
        Self {
            width,
            height,
            step,
            values: vec![[0.0; N]; width * height],
        }
    }

    fn at(&self, x: f64, y: f64) -> [f64; N] {
        let gx = (x / self.step).clamp(0.0, self.width as f64 - 1.0);
        let gy = (y / self.step).clamp(0.0, self.height as f64 - 1.0);
        let x0 = gx as usize;
        let y0 = gy as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = gx - x0 as f64;
        let fy = gy - y0 as f64;
        std::array::from_fn(|c| {
            let a = self.values[y0 * self.width + x0][c] * (1.0 - fx)
                + self.values[y0 * self.width + x1][c] * fx;
            let b = self.values[y1 * self.width + x0][c] * (1.0 - fx)
                + self.values[y1 * self.width + x1][c] * fx;
            a * (1.0 - fy) + b * fy
        })
    }
}

struct LayerSampler<'a> {
    info: &'a ImageInfo,
    source: &'a Rgb32FImage,
    source_divisor: f64,
    inverse: Matrix3<f64>,
    projection: Projection,
    offset: (f64, f64),
    left: u32,
    top: u32,
    scale: f64,
    residual: Field<2>,
}

impl LayerSampler<'_> {
    fn sample(&self, x: f64, y: f64) -> Option<Rgb<f32>> {
        let delta = self.residual.at(x / self.scale, y / self.scale);
        let target = Point3::new(
            self.left as f64 + x + delta[0] * self.scale - self.offset.0,
            self.top as f64 + y + delta[1] * self.scale - self.offset.1,
            1.0,
        );
        let source = map_target_to_source(&self.inverse, target, self.info, self.projection)?;
        if source.x < 0.0
            || source.y < 0.0
            || source.x >= self.info.width as f64
            || source.y >= self.info.height as f64
        {
            return None;
        }
        // Each prefilter level represents the centre of a 2x2 source block.
        // Retain the half-pixel offset to avoid a lens-dependent sampling shift.
        let sx = ((source.x + 0.5) / self.source_divisor - 0.5)
            .clamp(0.0, self.source.width() as f64 - 1.0);
        let sy = ((source.y + 0.5) / self.source_divisor - 0.5)
            .clamp(0.0, self.source.height() as f64 - 1.0);
        rgb_at(self.source, sx, sy)
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn analysis_pair(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    width: u32,
    height: u32,
) -> (GrayImage, GrayImage, GrayImage) {
    let mut a = GrayImage::new(width, height);
    let mut b = GrayImage::new(width, height);
    let mut valid = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let lx = x as f64 * sampler.scale;
            let ly = y as f64 * sampler.scale;
            let gx = (sampler.left as f64 + lx) as u32;
            let gy = (sampler.top as f64 + ly) as u32;
            if gx >= base.width() || gy >= base.height() {
                continue;
            }
            let Some(candidate) = sampler.sample(lx, ly) else {
                continue;
            };
            b.put_pixel(
                x,
                y,
                Luma([(luma(candidate) * 255.0).round().clamp(0.0, 255.0) as u8]),
            );
            if base_mask.get_pixel(gx, gy)[0] == 0 {
                continue;
            }
            let current = *base.get_pixel(gx, gy);
            a.put_pixel(
                x,
                y,
                Luma([(luma(current) * 255.0).round().clamp(0.0, 255.0) as u8]),
            );
            valid.put_pixel(x, y, Luma([255]));
        }
    }
    // Avoid letting resampled weave/noise dominate registration at coarse scale.
    (
        imageproc::filter::gaussian_blur_f32(&a, 0.8),
        imageproc::filter::gaussian_blur_f32(&b, 0.8),
        valid,
    )
}

fn refine_layer(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &mut LayerSampler<'_>,
    width: u32,
    height: u32,
) {
    let (a, b, valid) = analysis_pair(base, base_mask, sampler, width, height);
    let mut observations = Vec::<(Point2<f64>, [f64; 2], f64)>::new();
    for y in (16..height.saturating_sub(16)).step_by(GRID_STEP as usize) {
        for x in (16..width.saturating_sub(16)).step_by(GRID_STEP as usize) {
            if [-14i32, 0, 14].iter().any(|dy| {
                [-14i32, 0, 14].iter().any(|dx| {
                    valid.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32)[0] == 0
                })
            }) {
                continue;
            }
            let p = Point2::new(x as f64, y as f64);
            let Some(found) = refine_warped_patch(&a, &b, &Matrix3::identity(), p, 9, 7) else {
                continue;
            };
            if found.correlation < 0.88 {
                continue;
            }
            let Some(back) = refine_warped_patch(&b, &a, &Matrix3::identity(), found.target, 9, 7)
            else {
                continue;
            };
            if (back.target - p).norm() > 0.65 {
                continue;
            }
            let d = found.target - p;
            observations.push((p, [d.x, d.y], found.correlation));
        }
    }
    // A repeated stroke can win NCC locally. Require neighbouring measurements
    // to agree before allowing it to move a whole region of the source image.
    let supported = observations
        .iter()
        .filter(|(p, d, _)| {
            let neighbours = observations
                .iter()
                .filter(|(q, _, _)| (q - p).norm() <= 150.0)
                .collect::<Vec<_>>();
            if neighbours.len() < 3 {
                return false;
            }
            let mx = median(&mut neighbours.iter().map(|(_, d, _)| d[0]).collect::<Vec<_>>());
            let my = median(&mut neighbours.iter().map(|(_, d, _)| d[1]).collect::<Vec<_>>());
            (d[0] - mx).hypot(d[1] - my) <= 2.0
        })
        .copied()
        .collect::<Vec<_>>();
    if supported.len() < 4 {
        println!(
            "    - Local registration: insufficient reliable texture; keeping verified global alignment"
        );
        return;
    }
    let mut field = sampler.residual.clone();
    for gy in 0..field.height {
        for gx in 0..field.width {
            let p = Point2::new(gx as f64 * field.step, gy as f64 * field.step);
            let mut sum = [0.0; 2];
            let mut total = 0.0f64;
            for (q, d, correlation) in &supported {
                let distance = (p - q).norm_squared();
                let weight = (-distance / (2.0 * 72.0 * 72.0)).exp() * correlation.powi(4);
                sum[0] += weight * d[0];
                sum[1] += weight * d[1];
                total += weight;
            }
            // Unobserved space returns continuously to the global model.
            field.values[gy * field.width + gx][0] += sum[0] / total.max(0.05);
            field.values[gy * field.width + gx][1] += sum[1] / total.max(0.05);
        }
    }
    sampler.residual = field;
    let med = median(
        &mut supported
            .iter()
            .map(|(_, d, _)| d[0].hypot(d[1]) * sampler.scale)
            .collect::<Vec<_>>(),
    );
    println!(
        "    - Local registration: {} consistent patches, median correction {med:.2} output pixels",
        supported.len()
    );
}

fn refine_native_layer(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &mut LayerSampler<'_>,
    width: u32,
    height: u32,
) {
    if sampler.scale <= 2.0 {
        return;
    }
    // Refine against the selected, native output samples. Upsampling a coarse
    // displacement field alone leaves several pixels of error on 200MP input.
    // These tiny patches keep native refinement independent of canvas size.
    for spacing in [2.0, 1.0] {
        let mut observations = Vec::<(Point2<f64>, [f64; 2], f64)>::new();
        for y in (20..height.saturating_sub(20)).step_by(GRID_STEP as usize) {
            for x in (20..width.saturating_sub(20)).step_by(GRID_STEP as usize) {
                let lx = x as f64 * sampler.scale;
                let ly = y as f64 * sampler.scale;
                let mut a = GrayImage::new(57, 57);
                let mut b = GrayImage::new(57, 57);
                let mut valid = true;
                'patch: for py in 0..57 {
                    for px in 0..57 {
                        let dx = (px as f64 - 28.0) * spacing;
                        let dy = (py as f64 - 28.0) * spacing;
                        let gx = sampler.left as f64 + lx + dx;
                        let gy = sampler.top as f64 + ly + dy;
                        let Some(current) = rgb_at(base, gx, gy) else {
                            valid = false;
                            break 'patch;
                        };
                        if base_mask.get_pixel(gx as u32, gy as u32)[0] == 0 {
                            valid = false;
                            break 'patch;
                        }
                        let Some(candidate) = sampler.sample(lx + dx, ly + dy) else {
                            valid = false;
                            break 'patch;
                        };
                        a.put_pixel(
                            px,
                            py,
                            Luma([(luma(current) * 255.0).round().clamp(0.0, 255.0) as u8]),
                        );
                        b.put_pixel(
                            px,
                            py,
                            Luma([(luma(candidate) * 255.0).round().clamp(0.0, 255.0) as u8]),
                        );
                    }
                }
                if !valid {
                    continue;
                }
                let p = Point2::new(28.0, 28.0);
                let Some(found) = refine_warped_patch(&a, &b, &Matrix3::identity(), p, 10, 7)
                else {
                    continue;
                };
                if found.correlation < 0.90 {
                    continue;
                }
                let Some(back) =
                    refine_warped_patch(&b, &a, &Matrix3::identity(), found.target, 10, 7)
                else {
                    continue;
                };
                if (back.target - p).norm() > 0.40 {
                    continue;
                }
                let d = (found.target - p) * (spacing / sampler.scale);
                observations.push((
                    Point2::new(x as f64, y as f64),
                    [d.x, d.y],
                    found.correlation,
                ));
            }
        }
        if observations.len() < 6 {
            continue;
        }
        let mut field = sampler.residual.clone();
        for gy in 0..field.height {
            for gx in 0..field.width {
                let p = Point2::new(gx as f64 * field.step, gy as f64 * field.step);
                let mut sum = [0.0; 2];
                let mut total = 0.0f64;
                for (q, d, correlation) in &observations {
                    let weight =
                        (-(p - q).norm_squared() / (2.0 * 48.0 * 48.0)).exp() * correlation.powi(8);
                    sum[0] += weight * d[0];
                    sum[1] += weight * d[1];
                    total += weight;
                }
                for (c, value) in sum.into_iter().enumerate() {
                    field.values[gy * field.width + gx][c] += value / total.max(0.1);
                }
            }
        }
        sampler.residual = field;
        let correction = median(
            &mut observations
                .iter()
                .map(|(_, d, _)| d[0].hypot(d[1]) * sampler.scale)
                .collect::<Vec<_>>(),
        );
        println!(
            "    - Native refinement ({spacing:.0}px samples): {} patches, median correction {correction:.2}px",
            observations.len()
        );
    }
}

/// Smooth per-channel log gain. Corrections are estimated from matched overlap
/// blocks; fine texture never becomes a correction field. Robust medians keep
/// an occlusion or a displaced contour from tinting its surrounding region.
fn tone_field(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    width: u32,
    height: u32,
) -> Field<3> {
    let mut field = Field::new(width, height, 56.0);
    let mut samples = vec![Vec::<[f64; 3]>::new(); field.values.len()];
    for y in (0..height).step_by(4) {
        for x in (0..width).step_by(4) {
            let lx = x as f64 * sampler.scale;
            let ly = y as f64 * sampler.scale;
            let gx = (sampler.left as f64 + lx) as u32;
            let gy = (sampler.top as f64 + ly) as u32;
            if gx >= base.width() || gy >= base.height() || base_mask.get_pixel(gx, gy)[0] == 0 {
                continue;
            }
            let Some(candidate) = sampler.sample(lx, ly) else {
                continue;
            };
            let current = base.get_pixel(gx, gy);
            if current
                .0
                .iter()
                .chain(candidate.0.iter())
                .any(|v| !(0.015..0.97).contains(v))
            {
                continue;
            }
            let d = std::array::from_fn(|c| f64::from(current[c] / candidate[c]).ln());
            if d.iter().any(|v| v.abs() > 1.2) {
                continue;
            }
            let ix = ((x as f64 / field.step).round() as usize).min(field.width - 1);
            let iy = ((y as f64 / field.step).round() as usize).min(field.height - 1);
            samples[iy * field.width + ix].push(d);
        }
    }
    let observations = samples
        .iter()
        .enumerate()
        .filter(|(_, v)| v.len() >= 8)
        .map(|(i, v)| {
            let value =
                std::array::from_fn(|c| median(&mut v.iter().map(|d| d[c]).collect::<Vec<_>>()));
            (i, value)
        })
        .collect::<Vec<(usize, [f64; 3])>>();
    let global: [f64; 3] = std::array::from_fn(|c| {
        median(
            &mut samples
                .iter()
                .flatten()
                .map(|value| value[c])
                .collect::<Vec<_>>(),
        )
    });
    for iy in 0..field.height {
        for ix in 0..field.width {
            // The overlap still supplies a reliable frame-wide exposure when
            // the newly covered side has no local samples. Returning to unity
            // there would put the original exposure step back into the result.
            let mut total = 0.02f64;
            let mut sum = global.map(|gain| gain * total);
            for (i, value) in &observations {
                let dx = ix as f64 - (i % field.width) as f64;
                let dy = iy as f64 - (i / field.width) as f64;
                let weight = (-(dx * dx + dy * dy) / 2.0).exp();
                for c in 0..3 {
                    sum[c] += value[c] * weight;
                }
                total += weight;
            }
            // Extrapolate continuously over the nearby non-overlap border,
            // while bounding unsupported changes far outside the overlap.
            field.values[iy * field.width + ix] =
                std::array::from_fn(|c| (sum[c] / total.max(0.001)).clamp(-0.9, 0.9));
        }
    }
    field
}

fn adjusted(pixel: Rgb<f32>, delta: [f64; 3]) -> Rgb<f32> {
    Rgb(std::array::from_fn(|c| {
        (pixel[c] * delta[c].exp() as f32).clamp(0.0, 1.0)
    }))
}

// Native-output-pixel acutance, measured after a small binomial smoothing
// kernel. A reduced thumbnail can make a blurred 200MP frame look just as
// sharp as a telephoto detail tile, so ownership must inspect native samples.
fn acutance(mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>, x: f64, y: f64) -> f64 {
    let mut patch = [[0.0; 11]; 11];
    for (j, row) in patch.iter_mut().enumerate() {
        for (i, value) in row.iter_mut().enumerate() {
            let Some(p) = sample(x + i as f64 - 5.0, y + j as f64 - 5.0) else {
                return 0.0;
            };
            *value = luma(p);
        }
    }
    let mut energy = 0.0;
    let mut level = 0.0;
    for y in 2..9 {
        for x in 2..9 {
            let dx = (patch[y - 1][x + 1] + 2.0 * patch[y][x + 1] + patch[y + 1][x + 1]
                - patch[y - 1][x - 1]
                - 2.0 * patch[y][x - 1]
                - patch[y + 1][x - 1])
                / 8.0;
            let dy = (patch[y + 1][x - 1] + 2.0 * patch[y + 1][x] + patch[y + 1][x + 1]
                - patch[y - 1][x - 1]
                - 2.0 * patch[y - 1][x]
                - patch[y - 1][x + 1])
                / 8.0;
            energy += dx * dx + dy * dy;
            level += patch[y][x];
        }
    }
    (energy / 49.0).sqrt() / (level / 49.0).max(0.04)
}

/// Binary ownership regularised on a bounded grid. Pairwise terms penalise
/// seams through high contrast or disagreement. Fixed coverage labels ensure
/// every newly observed pixel is admitted, including fully enclosed detail
/// frames which a one-direction panorama seam can otherwise discard.
fn ownership(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    width: u32,
    height: u32,
) -> GrayImage {
    let size = (width.max(height) as f64 / SELECTION_LONG_SIDE as f64).max(1.0);
    let w = (width as f64 / size).ceil() as u32;
    let h = (height as f64 / size).ceil() as u32;
    let count = (w * h) as usize;
    let mut preference = vec![0.0f64; count];
    let mut disagreement = vec![0.0f64; count];
    let mut fixed = vec![0i8; count];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let ax = x as f64 * size;
            let ay = y as f64 * size;
            let lx = ax * sampler.scale;
            let ly = ay * sampler.scale;
            let gx = sampler.left as f64 + lx;
            let gy = sampler.top as f64 + ly;
            let base_valid = gx >= 0.0
                && gy >= 0.0
                && gx < base.width() as f64
                && gy < base.height() as f64
                && base_mask.get_pixel(gx as u32, gy as u32)[0] > 0;
            let candidate = sampler.sample(lx, ly);
            if candidate.is_none() {
                fixed[i] = -1;
                continue;
            }
            if !base_valid {
                fixed[i] = 1;
                continue;
            }
            // Keep a one-cell guard at the candidate's source boundary. The
            // neighbouring cells can still choose the sharper source, while a
            // hard edge prevents a whole tile from ending as a visible square.
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                fixed[i] = -1;
                continue;
            }
            let candidate = adjusted(candidate.unwrap(), tone.at(ax, ay));
            let base_pixel = *base.get_pixel(gx as u32, gy as u32);
            let base_focus = acutance(|x, y| rgb_at(base, x, y), gx, gy);
            let candidate_focus = acutance(
                |x, y| sampler.sample(x, y).map(|p| adjusted(p, tone.at(ax, ay))),
                lx,
                ly,
            );
            // Ties retain existing detail, making duplicate/identical frames
            // idempotent. Weak texture never wins solely from sensor noise.
            preference[i] = ((candidate_focus + 0.002) / (base_focus + 0.002))
                .ln()
                .clamp(-2.0, 2.0)
                - 0.06;
            disagreement[i] = candidate
                .0
                .iter()
                .zip(base_pixel.0)
                .map(|(a, b)| (*a - b).abs() as f64)
                .sum::<f64>()
                / 3.0;
        }
    }
    // Average evidence over neighbouring native patches, never image pixels.
    let raw = preference.clone();
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = y * w as usize + x;
            if fixed[i] != 0 {
                continue;
            }
            let mut sum = 0.0f64;
            let mut total = 0.0f64;
            for yy in y.saturating_sub(2)..=(y + 2).min(h as usize - 1) {
                for xx in x.saturating_sub(2)..=(x + 2).min(w as usize - 1) {
                    let j = yy * w as usize + xx;
                    if fixed[j] == 0 {
                        sum += raw[j];
                        total += 1.0;
                    }
                }
            }
            preference[i] = sum / total.max(1.0);
        }
    }
    // Solve the ownership globally so a seam can route around a contour and a
    // contained frame can be admitted without a block-shaped boundary.
    let labels = seam_cut::cut_grid(w as usize, h as usize, &preference, &disagreement, &fixed);
    GrayImage::from_raw(w, h, labels).expect("ownership dimensions")
}

pub(super) fn detail_preserving_mosaic<R: Runtime, F>(
    images: &[&ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    app: AppHandle<R>,
    event: &str,
    load: &mut F,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let (min_x, max_x, min_y, max_y) = output_bounds(images, homographies, projection);
    if ![min_x, max_x, min_y, max_y].iter().all(|x| x.is_finite()) {
        return Err("The stack does not have a finite aligned canvas.".into());
    }
    let (offset_x, width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, height) = pixel_aligned_canvas(min_y, max_y);
    let mut result = Rgb32FImage::new(width, height);
    let mut mask = GrayImage::new(width, height);
    println!("  - Detail-preserving mosaic canvas: {width}x{height}");
    for (index, &info) in images.iter().enumerate() {
        let _ = app.emit(
            event,
            format!(
                "Refining and selecting detail {} of {}",
                index + 1,
                images.len()
            ),
        );
        let mut source = load(info)?;
        let transform = &homographies[&info.id];
        let mut source_divisor = 1.0;
        if projection == Projection::Planar {
            let center = Point3::new(info.width as f64 * 0.5, info.height as f64 * 0.5, 1.0);
            let c = transform * center;
            let dx = transform * (center + nalgebra::Vector3::new(1.0, 0.0, 0.0));
            let dy = transform * (center + nalgebra::Vector3::new(0.0, 1.0, 0.0));
            let sx = (dx.x / dx.z - c.x / c.z).hypot(dx.y / dx.z - c.y / c.z);
            let sy = (dy.x / dy.z - c.x / c.z).hypot(dy.y / dy.z - c.y / c.z);
            let scale = sx.min(sy);
            // Prefilter a telephoto tile before minification. Otherwise weave
            // aliases into false sharpness and defeats native focus selection.
            while scale.is_finite()
                && scale > 0.0
                && scale * source_divisor < 0.67
                && source.width().min(source.height()) > 8
            {
                source = downsample_rgb_half(&source);
                source_divisor *= 2.0;
            }
        }
        let inverse = transform
            .try_inverse()
            .ok_or("The stack contains a non-invertible alignment.")?;
        let Some((left, right, top, bottom)) = transformed_image_region(
            info, transform, projection, offset_x, offset_y, width, height,
        ) else {
            continue;
        };
        let layer_width = right - left + 1;
        let layer_height = bottom - top + 1;
        let scale = (layer_width.max(layer_height) as f64 / ANALYSIS_LONG_SIDE as f64).max(1.0);
        let aw = (layer_width as f64 / scale).ceil() as u32;
        let ah = (layer_height as f64 / scale).ceil() as u32;
        let mut sampler = LayerSampler {
            info,
            source: &source,
            source_divisor,
            inverse,
            projection,
            offset: (offset_x, offset_y),
            left,
            top,
            scale,
            residual: Field::new(aw, ah, GRID_STEP as f64),
        };
        println!("  - Selecting '{}'", info.filename);
        if index > 0 {
            refine_layer(&result, &mask, &mut sampler, aw, ah);
            refine_layer(&result, &mask, &mut sampler, aw, ah);
            refine_native_layer(&result, &mask, &mut sampler, aw, ah);
        }
        let tone = if index > 0 {
            tone_field(&result, &mask, &sampler, aw, ah)
        } else {
            Field::new(aw, ah, 56.0)
        };
        let decision = ownership(&result, &mask, &sampler, &tone, aw, ah);
        let stride = width as usize * 3;
        result
            .as_mut()
            .par_chunks_mut(stride)
            .zip(mask.as_mut().par_chunks_mut(width as usize))
            .enumerate()
            .skip(top as usize)
            .take(layer_height as usize)
            .for_each(|(y, (row, covered))| {
                let ly = (y - top as usize) as f64;
                let ay = ly / scale;
                let dy =
                    ((ay / ah as f64 * decision.height() as f64) as u32).min(decision.height() - 1);
                for x in left..=right {
                    let lx = (x - left) as f64;
                    let ax = lx / scale;
                    let dx = ((ax / aw as f64 * decision.width() as f64) as u32)
                        .min(decision.width() - 1);
                    if covered[x as usize] != 0 && decision.get_pixel(dx, dy)[0] == 0 {
                        continue;
                    }
                    let Some(pixel) = sampler.sample(lx, ly) else {
                        continue;
                    };
                    let pixel = adjusted(pixel, tone.at(ax, ay));
                    row[x as usize * 3..x as usize * 3 + 3].copy_from_slice(&pixel.0);
                    covered[x as usize] = 255;
                }
            });
    }
    let covered = mask.as_raw().iter().filter(|&&v| v != 0).count();
    if covered == 0 {
        return Err("The aligned stack has no covered pixels.".into());
    }
    // An unsupported canvas margin is not photographic detail. Export only an
    // entirely covered rectangle; do not hide holes with stretched/reflected
    // source pixels, which looks like an unfused blur at the image boundary.
    let output = crop_to_valid_rectangle(result, &mask);
    println!(
        "  - Verified covered output: {}x{}; every pixel has source ownership",
        output.width(),
        output.height()
    );
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_info(id: usize, image: &Rgb32FImage) -> ImageInfo {
        ImageInfo {
            id,
            filename: format!("source-{id}.png"),
            width: image.width(),
            height: image.height(),
            alignment_image: image::DynamicImage::ImageRgb32F(image.clone()).to_luma8(),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        }
    }

    fn sharp_fixture() -> Rgb32FImage {
        Rgb32FImage::from_fn(160, 120, |x, y| {
            let value = 0.36
                + 0.12 * (x as f32 * 0.75).sin()
                + 0.10 * (y as f32 * 0.65).cos()
                + 0.03 * ((x + y) as f32 * 0.19).sin();
            Rgb([value * 1.1, value, value * 0.8])
        })
    }

    #[test]
    fn enclosed_sharp_tile_replaces_blur_without_losing_coverage() {
        let sharp = sharp_fixture();
        let base = image::imageops::blur(&sharp, 2.0);
        let candidate = image::imageops::crop_imm(&sharp, 40, 20, 80, 80).to_image();
        let sources = [base.clone(), candidate];
        let infos = [image_info(0, &sources[0]), image_info(1, &sources[1])];
        let transforms = HashMap::from([
            (0, Matrix3::identity()),
            (
                1,
                Matrix3::new(1.0, 0.0, 40.0, 0.0, 1.0, 20.0, 0.0, 0.0, 1.0),
            ),
        ]);
        let app = tauri::test::mock_app();
        let output = detail_preserving_mosaic(
            &[&infos[0], &infos[1]],
            &transforms,
            Projection::Planar,
            app.handle().clone(),
            "test-progress",
            &mut |info| Ok(sources[info.id].clone()),
        )
        .unwrap();
        assert_eq!(
            output.dimensions(),
            sharp.dimensions(),
            "coverage must include the last row and column"
        );
        let mut before = 0.0;
        let mut after = 0.0;
        for y in 30..90 {
            for x in 50..110 {
                before += (base.get_pixel(x, y)[1] - sharp.get_pixel(x, y)[1]).powi(2);
                after += (output.get_pixel(x, y)[1] - sharp.get_pixel(x, y)[1]).powi(2);
            }
        }
        assert!(
            after < before * 0.05,
            "the sharp interior must replace the blur: {after} vs {before}"
        );
        assert!(
            output
                .pixels()
                .all(|p| p.0.iter().all(|v| v.is_finite() && *v > 0.0))
        );
    }

    #[test]
    fn blurred_candidate_cannot_replace_an_already_sharp_interior() {
        let base = sharp_fixture();
        let source = image::imageops::blur(&base, 2.0);
        let info = image_info(0, &source);
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let mask = GrayImage::from_pixel(160, 120, Luma([255]));
        let selected = ownership(
            &base,
            &mask,
            &sampler,
            &Field::new(160, 120, 56.0),
            160,
            120,
        );
        let switched = (10..110)
            .flat_map(|y| (10..150).map(move |x| (x, y)))
            .filter(|&(x, y)| selected.get_pixel(x, y)[0] != 0)
            .count();
        assert_eq!(switched, 0);
    }

    #[test]
    fn smooth_tone_correction_preserves_selected_detail_relative_contrast() {
        let p = Rgb([0.2, 0.3, 0.4]);
        let q = Rgb([0.4, 0.5, 0.6]);
        let a = adjusted(p, [0.1, -0.05, 0.02]);
        let b = adjusted(q, [0.1, -0.05, 0.02]);
        for c in 0..3 {
            assert!((b[c] / a[c] - q[c] / p[c]).abs() < 1e-6);
        }
    }

    #[test]
    fn displacement_field_interpolation_is_continuous_at_cell_edges() {
        let mut field = Field::<2>::new(200, 200, 48.0);
        for (i, value) in field.values.iter_mut().enumerate() {
            *value = [
                (i % field.width) as f64 * 0.3,
                (i / field.width) as f64 * -0.2,
            ];
        }
        let a = field.at(48.0 - 1e-5, 73.0);
        let b = field.at(48.0 + 1e-5, 73.0);
        assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
    }
}
