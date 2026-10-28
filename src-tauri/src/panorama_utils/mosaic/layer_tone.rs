//! Per-layer tone fields.
//!
//! Smooth per-channel log-gain fields measured on matched overlap blocks; the
//! streaming variant adds a robust affine trend prior so a one-sided overlap
//! can still carry its exposure change through the layer interior.

use image::GrayImage;
use image::Rgb32FImage;

use super::{Field, LayerSampler, median};

pub(super) fn streaming_tone_field_from_analysis(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    sigma_cells: f64,
) -> Field<3> {
    let width = base.width().min(candidate.width());
    let height = base.height().min(candidate.height());
    // Estimate exposure from the paper/illumination field rather than from
    // individual brush strokes.  The old per-pixel ratios were very sensitive
    // to a dark line landing one analysis sample away after registration; that
    // error was then interpolated over the whole source tile and appeared as
    // a rectangular tonal patch.  Keep this blur confined to the solve: the
    // source pixels written to the mosaic remain untouched at high frequency.
    let base_low = image::imageops::blur(base, 3.0);
    let candidate_low = image::imageops::blur(candidate, 3.0);
    let mut field = Field::new(width, height, 112.0);
    let mut samples = vec![Vec::<[f64; 3]>::new(); field.values.len()];
    for y in (0..height).step_by(8) {
        for x in (0..width).step_by(8) {
            if base_mask.get_pixel(x, y)[0] == 0 || candidate_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let current = *base_low.get_pixel(x, y);
            let source = *candidate_low.get_pixel(x, y);
            if current
                .0
                .iter()
                .chain(source.0.iter())
                .any(|value| !(0.015..0.97).contains(value))
            {
                continue;
            }
            let delta =
                std::array::from_fn(|channel| f64::from(current[channel] / source[channel]).ln());
            if delta.iter().any(|value| value.abs() > 1.2) {
                continue;
            }
            let field_x = ((x as f64 / field.step).round() as usize).min(field.width - 1);
            let field_y = ((y as f64 / field.step).round() as usize).min(field.height - 1);
            samples[field_y * field.width + field_x].push(delta);
        }
    }
    let observations = samples
        .iter()
        .enumerate()
        .filter(|(_, values)| values.len() >= 4)
        .map(|(index, values)| {
            let value: [f64; 3] = std::array::from_fn(|channel| {
                median(
                    &mut values
                        .iter()
                        .map(|delta| delta[channel])
                        .collect::<Vec<_>>(),
                )
            });
            (index, value)
        })
        .collect::<Vec<_>>();
    // When a source overlaps the existing mosaic only along one side, a
    // short Gaussian kernel cannot carry the measured illumination change
    // through the source interior; unsupported cells fall back to one global
    // median and retain a frame-sized step.  Fit a conservative affine
    // low-frequency trend to the robust cell medians as a prior.  This changes
    // only the tone field used for ownership and source writes; source detail
    // is still sampled verbatim below.  A rank check rejects narrow or
    // degenerate overlaps where a plane would be an extrapolation without
    // evidence.
    let tone_plane = fit_streaming_tone_plane(&observations, field.width, field.height);
    let global: [f64; 3] = std::array::from_fn(|channel| {
        median(
            &mut samples
                .iter()
                .flatten()
                .map(|delta| delta[channel])
                .collect::<Vec<_>>(),
        )
    });
    for field_y in 0..field.height {
        for field_x in 0..field.width {
            let (prior, prior_weight) = tone_plane
                .as_ref()
                .map(|plane| {
                    (
                        std::array::from_fn(|channel| {
                            predict_streaming_tone_plane(
                                plane[channel],
                                field_x,
                                field_y,
                                field.width,
                                field.height,
                            )
                        }),
                        0.18f64,
                    )
                })
                .unwrap_or((global, 0.02f64));
            let mut total = prior_weight;
            let mut sum = prior.map(|value| value * total);
            for (index, value) in &observations {
                let dx = field_x as f64 - (index % field.width) as f64;
                let dy = field_y as f64 - (index / field.width) as f64;
                let sigma = sigma_cells.max(1.0);
                let weight = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                for channel in 0..3 {
                    sum[channel] += value[channel] * weight;
                }
                total += weight;
            }
            field.values[field_y * field.width + field_x] =
                std::array::from_fn(|channel| (sum[channel] / total).clamp(-0.9, 0.9));
        }
    }
    field
}

/// Fit three independent affine planes to robust per-cell tone observations.
/// Coordinates are normalized to [-1, 1] so the normal equations stay
/// well-conditioned for a 1400x1400 analysis layer.  The returned array is
/// `[intercept, x_slope, y_slope]` for each RGB channel.
pub(super) fn fit_streaming_tone_plane(
    observations: &[(usize, [f64; 3])],
    width: usize,
    height: usize,
) -> Option<[[f64; 3]; 3]> {
    if observations.len() < 12 || width < 2 || height < 2 {
        return None;
    }
    let mut normal = [[0.0f64; 3]; 3];
    let mut rhs = [[0.0f64; 3]; 3];
    for &(index, value) in observations {
        let x = (index % width) as f64 / (width - 1) as f64 * 2.0 - 1.0;
        let y = (index / width) as f64 / (height - 1) as f64 * 2.0 - 1.0;
        let features = [1.0, x, y];
        for row in 0..3 {
            for column in 0..3 {
                normal[row][column] += features[row] * features[column];
            }
            for channel in 0..3 {
                rhs[channel][row] += features[row] * value[channel];
            }
        }
    }
    let mut result = [[0.0f64; 3]; 3];
    for channel in 0..3 {
        let mut augmented = [
            [normal[0][0], normal[0][1], normal[0][2], rhs[channel][0]],
            [normal[1][0], normal[1][1], normal[1][2], rhs[channel][1]],
            [normal[2][0], normal[2][1], normal[2][2], rhs[channel][2]],
        ];
        for pivot in 0..3 {
            let mut pivot_row = pivot;
            for row in pivot + 1..3 {
                if augmented[row][pivot].abs() > augmented[pivot_row][pivot].abs() {
                    pivot_row = row;
                }
            }
            if augmented[pivot_row][pivot].abs() < 1e-6 {
                return None;
            }
            augmented.swap(pivot, pivot_row);
            let divisor = augmented[pivot][pivot];
            for column in pivot..4 {
                augmented[pivot][column] /= divisor;
            }
            for row in 0..3 {
                if row == pivot {
                    continue;
                }
                let factor = augmented[row][pivot];
                for column in pivot..4 {
                    augmented[row][column] -= factor * augmented[pivot][column];
                }
            }
        }
        result[channel] = [augmented[0][3], augmented[1][3], augmented[2][3]];
    }
    // Reject planes which would apply an unsupported multi-stop correction to
    // a source.  The field is log gain, so ±0.9 is already a 2.5x ratio.
    let corners = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
    if result.iter().any(|plane| {
        corners
            .iter()
            .any(|&[x, y]| (plane[0] + plane[1] * x + plane[2] * y).abs() > 0.75)
    }) {
        return None;
    }
    Some(result)
}

pub(super) fn predict_streaming_tone_plane(
    plane: [f64; 3],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> f64 {
    let nx = x as f64 / width.saturating_sub(1).max(1) as f64 * 2.0 - 1.0;
    let ny = y as f64 / height.saturating_sub(1).max(1) as f64 * 2.0 - 1.0;
    (plane[0] + plane[1] * nx + plane[2] * ny).clamp(-0.9, 0.9)
}

/// Smooth per-channel log gain. Corrections are estimated from matched overlap
/// blocks; fine texture never becomes a correction field. Robust medians keep
/// an occlusion or a displaced contour from tinting its surrounding region.
pub(super) fn tone_field(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    width: u32,
    height: u32,
) -> Field<3> {
    tone_field_with_sampling(base, base_mask, sampler, width, height, 56.0, 4, 8)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn tone_field_with_sampling(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    width: u32,
    height: u32,
    field_step: f64,
    sample_step: usize,
    minimum_samples: usize,
) -> Field<3> {
    let mut field = Field::new(width, height, field_step);
    let mut samples = vec![Vec::<[f64; 3]>::new(); field.values.len()];
    for y in (0..height).step_by(sample_step) {
        for x in (0..width).step_by(sample_step) {
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
        .filter(|(_, v)| v.len() >= minimum_samples)
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
