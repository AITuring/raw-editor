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

#[cfg(test)]
#[path = "mosaic_diagnostics.rs"]
mod diagnostics;

const ANALYSIS_LONG_SIDE: u32 = 1400;
// Bound the grid per source layer, not by the full panorama width. At a
// roughly 9504px layer footprint this yields approximately 19px cells, fine
// enough for seams to route around individual strokes.
const SELECTION_LONG_SIDE: u32 = 512;
const GRID_STEP: u32 = 32;
const NATIVE_REFINE_STEP: u32 = 16;
const NATIVE_FIELD_RADIUS: f64 = 32.0;
const OWNERSHIP_MISMATCH_PENALTY: f64 = 2.4;

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
        for y in (20..height.saturating_sub(20)).step_by(NATIVE_REFINE_STEP as usize) {
            for x in (20..width.saturating_sub(20)).step_by(NATIVE_REFINE_STEP as usize) {
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
                    let weight = (-(p - q).norm_squared()
                        / (2.0 * NATIVE_FIELD_RADIUS * NATIVE_FIELD_RADIUS))
                        .exp()
                        * correlation.powi(8);
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

// Native focus evidence after a separable low-pass filter. Single-pixel
// gradients confuse sensor grain with detail and miss faded coloured marks
// whose luminance nearly matches the substrate. Keep luminance and two colour
// differences, suppress pixel noise in both axes, then measure coherent edges.
fn acutance(mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>, x: f64, y: f64) -> f64 {
    let mut patch = [[[0.0; 3]; 13]; 13];
    for (j, row) in patch.iter_mut().enumerate() {
        for (i, value) in row.iter_mut().enumerate() {
            let Some(p) = sample(x + i as f64 - 6.0, y + j as f64 - 6.0) else {
                return 0.0;
            };
            *value = [
                luma(p),
                f64::from(p[0] - p[1]) * 0.5,
                f64::from(p[2] - p[1]) * 0.5,
            ];
        }
    }
    const KERNEL: [f64; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let mut horizontal = [[[0.0; 3]; 9]; 13];
    for y in 0..13 {
        for x in 0..9 {
            for c in 0..3 {
                horizontal[y][x][c] = (0..5).map(|k| patch[y][x + k][c] * KERNEL[k]).sum();
            }
        }
    }
    let mut smooth = [[[0.0; 3]; 9]; 9];
    for y in 0..9 {
        for x in 0..9 {
            for c in 0..3 {
                smooth[y][x][c] = (0..5).map(|k| horizontal[y + k][x][c] * KERNEL[k]).sum();
            }
        }
    }
    let mut energy = 0.0;
    let mut level = 0.0;
    for y in 2..7 {
        for x in 2..7 {
            for c in 0..3 {
                let dx = (smooth[y][x + 2][c] - smooth[y][x - 2][c]) * 0.25;
                let dy = (smooth[y + 2][x][c] - smooth[y - 2][x][c]) * 0.25;
                energy += dx * dx + dy * dy;
            }
            level += smooth[y][x][0];
        }
    }
    (energy / 25.0).sqrt() / (level / 25.0).max(0.04)
}

/// Evaluate focus at several locations inside an ownership cell. The old
/// single centre sample could land on quiet paper while the candidate held a
/// sharp brush stroke a few pixels away. Average the two strongest spatial
/// probes so sparse detail contributes; each probe filters pixel noise before
/// measuring its edge response.
fn cell_focus(
    mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    cell_size: f64,
) -> f64 {
    let radius = (cell_size * 0.28).clamp(2.0, 18.0);
    // Five probes cover the centre and all four corners. A stroke can cross
    // any cell edge, so sampling only one diagonal can still land entirely
    // on quiet paper and hide an in-focus candidate.
    let offsets = [
        (-radius, -radius),
        (radius, -radius),
        (0.0, 0.0),
        (-radius, radius),
        (radius, radius),
    ];
    let mut values = offsets
        .iter()
        .filter_map(|(dx, dy)| Some(acutance(&mut sample, x + dx, y + dy)))
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    // Preserve sparse in-focus strokes even when other probes see only paper.
    let first = values.len().saturating_sub(2);
    (values[first] + values[first + 1.min(values.len() - 1)]) * 0.5
}

fn ownership_disagreement(
    base: &Rgb32FImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    ax: f64,
    ay: f64,
    cell_size: f64,
) -> f64 {
    // A single sample at the cell centre can land on canvas and miss a brush
    // stroke that crosses the same cell near an edge. Sample the whole cell
    // instead, and retain a high percentile so a displaced contour vetoes the
    // candidate even when most of the cell is quiet paper.
    const OFFSETS: [f64; 5] = [0.15, 0.325, 0.5, 0.675, 0.85];
    let side = OFFSETS.len();
    let mut samples = vec![None; side * side];
    for (yi, y_offset) in OFFSETS.iter().copied().enumerate() {
        for (xi, x_offset) in OFFSETS.iter().copied().enumerate() {
            let sample_index = yi * side + xi;
            let sample_ax = ax + cell_size * x_offset;
            let sample_ay = ay + cell_size * y_offset;
            let lx = sample_ax * sampler.scale;
            let ly = sample_ay * sampler.scale;
            let gx = sampler.left as f64 + lx;
            let gy = sampler.top as f64 + ly;
            let Some(candidate) = sampler
                .sample(lx, ly)
                .map(|pixel| adjusted(pixel, tone.at(sample_ax, sample_ay)))
            else {
                continue;
            };
            let Some(base_pixel) = rgb_at(base, gx, gy) else {
                continue;
            };
            let candidate = candidate.0.map(f64::from);
            let base_pixel = base_pixel.0.map(f64::from);
            if candidate
                .iter()
                .chain(base_pixel.iter())
                .all(|v| v.is_finite())
            {
                samples[sample_index] = Some((candidate, base_pixel));
            }
        }
    }
    if samples.is_empty() {
        return 1.0;
    }
    // Compare low-pass colour evidence rather than individual high-frequency
    // pixels. Defocus changes brush-edge samples substantially even when the
    // geometry is correct; a displaced contour still changes the local mean
    // over a 3×3 neighbourhood and remains penalised.
    let side = side as isize;
    let mut disagreements = Vec::with_capacity(samples.len());
    for index in 0..samples.len() {
        let row = index as isize / side;
        let column = index as isize % side;
        let mut candidate_sum = [0.0; 3];
        let mut base_sum = [0.0; 3];
        let mut count = 0.0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let y = row + dy;
                let x = column + dx;
                if y < 0 || x < 0 || y >= side || x >= side {
                    continue;
                }
                let neighbour = y as usize * OFFSETS.len() + x as usize;
                let Some(Some((candidate, base))) = samples.get(neighbour) else {
                    continue;
                };
                for channel in 0..3 {
                    candidate_sum[channel] += candidate[channel];
                    base_sum[channel] += base[channel];
                }
                count += 1.0;
            }
        }
        if count > 0.0 {
            disagreements.push(
                (0..3)
                    .map(|channel| (candidate_sum[channel] - base_sum[channel]).abs() / count)
                    .sum::<f64>()
                    / 3.0,
            );
        }
    }
    if disagreements.is_empty() {
        return 1.0;
    }
    disagreements.sort_by(f64::total_cmp);
    let mean = disagreements.iter().sum::<f64>() / disagreements.len() as f64;
    let high = disagreements[(disagreements.len() * 3 / 4).min(disagreements.len() - 1)];
    (mean * 0.45 + high * 0.55).clamp(0.0, 1.0)
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
    ownership_grid(base, base_mask, sampler, tone, width, height, size)
}

fn ownership_grid(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    width: u32,
    height: u32,
    size: f64,
) -> GrayImage {
    let w = (width as f64 / size).ceil() as u32;
    let h = (height as f64 / size).ceil() as u32;
    let count = (w * h) as usize;
    let mut preference = vec![0.0f64; count];
    let mut disagreement = vec![0.0f64; count];
    let mut fixed = vec![0i8; count];
    let mut energy_sum = 0.0;
    let mut energy_count = 0usize;
    let mut mismatch = vec![0.0f64; count];
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
            let native_cell = size * sampler.scale;
            let base_focus = cell_focus(
                |x, y| rgb_at(base, x, y),
                gx + native_cell * 0.5,
                gy + native_cell * 0.5,
                native_cell,
            );
            let candidate_focus = cell_focus(
                |x, y| sampler.sample(x, y).map(|p| adjusted(p, tone.at(ax, ay))),
                lx + native_cell * 0.5,
                ly + native_cell * 0.5,
                native_cell,
            );
            // Ties retain existing detail, making duplicate/identical frames
            // idempotent. Weak texture never wins solely from sensor noise.
            // Compare edge energy rather than per-cell log ratios. Defocus
            // spreads a contour over more cells: ratios give its weak halo
            // as many votes as a focused edge and can reject the sharp frame.
            // Energy keeps quiet substrate votes small; the relative margin
            // retains an existing source when its detail is equivalent.
            preference[i] = candidate_focus.powi(2) - base_focus.powi(2) * 1.06;
            energy_sum += (candidate_focus.powi(2) + base_focus.powi(2)) * 0.5;
            energy_count += 1;
            disagreement[i] = ownership_disagreement(base, sampler, tone, ax, ay, size);
            // A sharp but displaced candidate can win the acutance test even
            // though it would put a second contour next to the existing
            // stroke. Penalise that candidate directly; otherwise the graph
            // cut can legally place a seam through the high-contrast subject
            // and leave a visible rectangular ghost.
            // Defocus changes high-frequency samples even when geometry is
            // correct. When the candidate is demonstrably sharper, discount
            // that photometric disagreement; a sharp source must be allowed
            // to replace a soft source instead of being vetoed as a ghost.
            let mismatch_scale = if candidate_focus > base_focus * 1.12 {
                0.30
            } else {
                1.0
            };
            mismatch[i] = (disagreement[i] * OWNERSHIP_MISMATCH_PENALTY * mismatch_scale).min(1.0);
        }
    }
    // One shared scale preserves energy comparisons across the overlap. A
    // cell-local denominator would again amplify weak paper/blur-halo votes.
    let energy_scale = (energy_sum / energy_count.max(1) as f64).max(1e-6);
    for i in 0..count {
        preference[i] = (preference[i] / energy_scale).clamp(-12.0, 12.0) - mismatch[i];
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
        #[cfg(test)]
        diagnostics::capture_layer(index, &sampler, &tone, &decision, aw, ah);
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
    #[cfg(test)]
    diagnostics::capture_crop(&mask);
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

    fn noisy_colour_stroke_pair() -> (Rgb32FImage, Rgb32FImage) {
        let sharp = Rgb32FImage::from_fn(192, 160, |x, y| {
            let stroke = (x as i32 - 96).abs() <= 3 && (20..140).contains(&y);
            // A faded red mark can have almost the same luminance as its
            // brown substrate; colour edges still carry genuine focus.
            if stroke {
                Rgb([0.46, 0.31, 0.27])
            } else {
                Rgb([0.40, 0.34, 0.27])
            }
        });
        let blurred = image::imageops::blur(&sharp, 3.2);
        let noisy = |im: &Rgb32FImage, amplitude: f32| {
            Rgb32FImage::from_fn(im.width(), im.height(), |x, y| {
                let hash = x.wrapping_mul(0x9e3779b9) ^ y.wrapping_mul(0x85ebca6b);
                let hash = (hash ^ (hash >> 16)).wrapping_mul(0x7feb352d);
                let noise = ((hash & 65535) as f32 / 65535.0 - 0.5) * amplitude;
                Rgb(im.get_pixel(x, y).0.map(|v| (v + noise).clamp(0.0, 1.0)))
            })
        };
        (noisy(&sharp, 0.020), noisy(&blurred, 0.080))
    }

    #[test]
    fn sensor_noise_cannot_hide_a_sharper_faint_colour_stroke() {
        let (sharp, noisy_blur) = noisy_colour_stroke_pair();
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(192, 160, 48.0),
        };
        let mask = GrayImage::from_pixel(192, 160, Luma([255]));
        let chosen = ownership(
            &noisy_blur,
            &mask,
            &sampler,
            &Field::new(192, 160, 56.0),
            192,
            160,
        );
        let selected = (30..130)
            .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
            .count();
        assert!(
            selected >= 90,
            "faint focused colour must beat noisy defocus ({selected}/100)"
        );
    }

    #[test]
    fn extra_sensor_noise_cannot_reclaim_an_already_focused_stroke() {
        let (sharp, noisy_blur) = noisy_colour_stroke_pair();
        let info = image_info(1, &noisy_blur);
        let sampler = LayerSampler {
            info: &info,
            source: &noisy_blur,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(192, 160, 48.0),
        };
        let mask = GrayImage::from_pixel(192, 160, Luma([255]));
        let chosen = ownership(
            &sharp,
            &mask,
            &sampler,
            &Field::new(192, 160, 56.0),
            192,
            160,
        );
        let selected = (30..130)
            .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
            .count();
        assert_eq!(
            selected, 0,
            "noisy defocus must not overwrite the focused stroke"
        );
    }

    #[test]
    fn diffuse_halo_cannot_outvote_a_narrow_focused_contour() {
        let sharp = Rgb32FImage::from_fn(192, 160, |x, y| {
            let line = (x as i32 - 96).abs() <= 1 && (20..140).contains(&y);
            let paper = 0.40 + 0.003 * (x as f32 * 0.17).sin();
            let value = if line { 0.28 } else { paper };
            Rgb([value, value * 0.9, value * 0.8])
        });
        let base = image::imageops::blur(&sharp, 5.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(192, 160, 48.0),
        };
        let chosen = ownership(
            &base,
            &GrayImage::from_pixel(192, 160, Luma([255])),
            &sampler,
            &Field::new(192, 160, 56.0),
            192,
            160,
        );
        let selected = (30..130)
            .filter(|&y| chosen.get_pixel(96, y)[0] > 0)
            .count();
        assert!(
            selected >= 90,
            "focused contour lost to its spread halo: {selected}/100"
        );
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
    fn sparse_in_focus_stroke_is_not_lost_to_cell_background() {
        let sharp = Rgb32FImage::from_fn(160, 120, |x, y| {
            let line = (x as i32 - 80).unsigned_abs() <= 1 && (12..108).contains(&y);
            let value = if line { 0.04 } else { 0.42 };
            Rgb([value, value, value])
        });
        let base = image::imageops::blur(&sharp, 3.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
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
        let selected_on_stroke = (12..108)
            .filter(|&y| selected.get_pixel(80, y)[0] != 0)
            .count();
        assert!(
            selected_on_stroke > 70,
            "a sparse sharp stroke must own its pixels ({selected_on_stroke}/96)"
        );
    }

    #[test]
    fn disagreement_is_lower_for_aligned_defocus_than_for_shifted_detail() {
        let sharp = Rgb32FImage::from_fn(160, 120, |x, y| {
            let value = 0.35 + 0.22 * (x as f32 * 0.37).sin() + 0.16 * (y as f32 * 0.29).cos();
            Rgb([value, value * 0.9, value * 0.75])
        });
        let base = image::imageops::blur(&sharp, 2.5);
        let info = image_info(1, &sharp);
        let tone = Field::new(160, 120, 56.0);
        let aligned = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let aligned_error = ownership_disagreement(&base, &aligned, &tone, 32.0, 32.0, 40.0);
        let shifted = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::new(1.0, 0.0, 4.0, 0.0, 1.0, -3.0, 0.0, 0.0, 1.0),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(160, 120, 48.0),
        };
        let shifted_error = ownership_disagreement(&base, &shifted, &tone, 32.0, 32.0, 40.0);
        assert!(
            aligned_error < shifted_error,
            "local averaging should preserve geometric mismatch signal: {aligned_error} vs {shifted_error}"
        );
    }

    #[test]
    fn invalid_sample_gap_cannot_cancel_disconnected_colour_mismatches() {
        let base = Rgb32FImage::from_pixel(100, 100, Rgb([0.5; 3]));
        let source = Rgb32FImage::from_fn(100, 100, |x, _| {
            // Invalid samples must retain their positions. If the valid
            // samples are packed into a smaller grid, these separated light
            // and dark strips become neighbours and falsely cancel out.
            let value = if x < 30 {
                0.8
            } else if x > 60 {
                0.2
            } else {
                f32::NAN
            };
            Rgb([value; 3])
        });
        let info = image_info(1, &source);
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
            residual: Field::new(100, 100, 48.0),
        };
        let error =
            ownership_disagreement(&base, &sampler, &Field::new(100, 100, 56.0), 0.0, 0.0, 80.0);
        assert!(
            (error - 0.3).abs() < 1e-6,
            "missing samples must not hide a real colour mismatch: {error}"
        );
    }

    #[test]
    fn fully_invalid_overlap_is_not_treated_as_an_aligned_match() {
        let source = Rgb32FImage::from_pixel(32, 32, Rgb([0.5; 3]));
        let info = image_info(1, &source);
        let sampler = LayerSampler {
            info: &info,
            source: &source,
            source_divisor: 1.0,
            inverse: Matrix3::new(1.0, 0.0, 64.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 1.0,
            residual: Field::new(32, 32, 48.0),
        };
        assert_eq!(
            ownership_disagreement(&source, &sampler, &Field::new(32, 32, 56.0), 0.0, 0.0, 24.0,),
            1.0,
            "absence of common samples must not remove the mismatch penalty"
        );
    }

    #[test]
    fn scaled_sparse_detail_is_sampled_at_the_native_cell_centre() {
        let sharp = Rgb32FImage::from_fn(3_200, 1_600, |x, _| {
            let value = if (x as i32 - 84).unsigned_abs() <= 2 {
                0.04
            } else {
                0.42
            };
            Rgb([value; 3])
        });
        let base = image::imageops::blur(&sharp, 4.0);
        let info = image_info(1, &sharp);
        let sampler = LayerSampler {
            info: &info,
            source: &sharp,
            source_divisor: 1.0,
            inverse: Matrix3::identity(),
            projection: Projection::Planar,
            offset: (0.0, 0.0),
            left: 0,
            top: 0,
            scale: 8.0,
            residual: Field::new(400, 200, 48.0),
        };
        let selected = ownership(
            &base,
            &GrayImage::from_pixel(3_200, 1_600, Luma([255])),
            &sampler,
            &Field::new(400, 200, 56.0),
            400,
            200,
        );
        let selected_on_stroke = (20..180)
            .filter(|&y| selected.get_pixel(10, y)[0] != 0)
            .count();
        assert!(
            selected_on_stroke > 100,
            "a native sharp stroke at a scaled cell centre must own its detail ({selected_on_stroke}/160)"
        );
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
