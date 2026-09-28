//! Exposure and white-balance calibration from geometrically corresponding pixels.
//!
//! Every measurement compares the same world position in two previews. There is
//! deliberately no foreground/background classification and no comparison of
//! whole-image histograms: real illumination and pigment variation in the scene
//! must survive the calibration. Supply previews in the same RGB encoding; linear
//! RGB is preferable when the correction is intended to represent exposure.

use image::{Rgb, Rgb32FImage};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

/// Maximum affine offset allowed by the stack Tone_Harmonizer.
///
/// This is intentionally a small linear-light value.  Keeping it next to the
/// photometric solver gives both the old progressive compositor and the new
/// stack pipeline the same bound without an environment-variable switch.
pub(crate) const PHOTOMETRIC_MAX_ABS_OFFSET: f64 = 0.02;

#[derive(Clone, Debug)]
pub(crate) struct PhotometricOptions {
    pub(crate) max_samples_per_pair: usize,
    pub(crate) min_samples_per_pair: usize,
    /// Numerical black and clipping limits, applied identically to all channels.
    pub(crate) min_sample_value: f64,
    pub(crate) max_sample_value: f64,
    pub(crate) max_pair_log_scatter: f64,
    pub(crate) max_abs_log_gain: f64,
    /// Spatial terms are disabled unless their held-out overlap error improves.
    pub(crate) allow_linear: bool,
    pub(crate) max_linear_log_gain: f64,
    pub(crate) linear_regularization: f64,
}

impl Default for PhotometricOptions {
    fn default() -> Self {
        Self {
            max_samples_per_pair: 2048,
            min_samples_per_pair: 1024,
            min_sample_value: 0.02,
            max_sample_value: 0.98,
            max_pair_log_scatter: 0.18,
            max_abs_log_gain: 1.25_f64.ln(),
            allow_linear: false,
            max_linear_log_gain: 0.10,
            linear_regularization: 0.10,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PhotometricModel {
    /// For each RGB channel: constant, normalized x, normalized y log gain.
    pub(crate) log_gain: [[f64; 3]; 3],
    pub(crate) preview_width: u32,
    pub(crate) preview_height: u32,
    max_abs_log_gain: f64,
    /// The bounded affine offset used by [`Self::apply`].
    pub(crate) offset: [f64; 3],
    /// Unbounded least-squares result, retained for Stack_Report diagnostics.
    pub(crate) solved_offset: [f64; 3],
    pub(crate) offset_clamped: bool,
}

impl PhotometricModel {
    /// Coordinates refer to this model's bounded preview, not the final raster.
    pub(crate) fn gain_at(&self, preview_x: f64, preview_y: f64) -> [f32; 3] {
        let basis = normalized_basis(
            self.preview_width,
            self.preview_height,
            preview_x,
            preview_y,
        );
        std::array::from_fn(|channel| {
            dot(self.log_gain[channel], basis)
                .clamp(-self.max_abs_log_gain, self.max_abs_log_gain)
                .exp() as f32
        })
    }

    /// No clipping or tone mapping is applied here.
    pub(crate) fn apply(&self, pixel: Rgb<f32>, x: f64, y: f64) -> Rgb<f32> {
        let gain = self.gain_at(x, y);
        Rgb(std::array::from_fn(|channel| {
            pixel[channel] * gain[channel] + self.offset[channel] as f32
        }))
    }

    /// Return the solved affine offset without applying it to a pixel.
    #[allow(dead_code)]
    pub(crate) fn offset_at(&self, _preview_x: f64, _preview_y: f64) -> [f32; 3] {
        self.offset.map(|value| value as f32)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PhotometricPairDiagnostic {
    pub(crate) first: usize,
    pub(crate) second: usize,
    pub(crate) samples: usize,
    pub(crate) median_log_ratio: [f64; 3],
    pub(crate) log_scatter: f64,
    pub(crate) reliable: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct PhotometricCalibration {
    pub(crate) models: Vec<PhotometricModel>,
    pub(crate) pairs: Vec<PhotometricPairDiagnostic>,
    pub(crate) linear_used: bool,
    pub(crate) held_out_constant_error: f64,
    pub(crate) held_out_corrected_error: f64,
}

#[derive(Clone)]
struct Observation {
    first_basis: [f64; 3],
    second_basis: [f64; 3],
    // gain(first) - gain(second) = log(pixel(second)/pixel(first)).
    delta: [f64; 3],
    first_rgb: [f64; 3],
    second_rgb: [f64; 3],
}

struct Pair {
    first: usize,
    second: usize,
    observations: Vec<Observation>,
    median: [f64; 3],
    scatter: f64,
    weight: f64,
}

/// Calibrate all reliable overlaps jointly. Each connected component has zero
/// mean constant log gain, so neither an arbitrary anchor nor input order changes
/// its brightness. Isolated or invalid tiles retain exact identity corrections.
pub(crate) fn calibrate_overlap_photometry(
    tiles: &[(Rgb32FImage, Matrix3<f64>)],
    options: &PhotometricOptions,
) -> PhotometricCalibration {
    let mut models: Vec<_> = tiles
        .iter()
        .map(|(image, _)| PhotometricModel {
            log_gain: [[0.0; 3]; 3],
            preview_width: image.width(),
            preview_height: image.height(),
            max_abs_log_gain: options.max_abs_log_gain.max(0.0),
            offset: [0.0; 3],
            solved_offset: [0.0; 3],
            offset_clamped: false,
        })
        .collect();
    let geometry: Vec<_> = tiles
        .iter()
        .map(|(image, transform)| {
            Some((
                transform.try_inverse()?,
                world_bounds(image.width(), image.height(), transform)?,
            ))
        })
        .collect();
    let mut pairs = Vec::new();
    let mut diagnostics = Vec::new();
    for first in 0..tiles.len() {
        for second in first + 1..tiles.len() {
            let (Some((first_inverse, first_bounds)), Some((second_inverse, second_bounds))) =
                (&geometry[first], &geometry[second])
            else {
                continue;
            };
            let Some(bounds) = intersect_bounds(*first_bounds, *second_bounds) else {
                continue;
            };
            let observations = sample_overlap(
                &tiles[first].0,
                first_inverse,
                &tiles[second].0,
                second_inverse,
                bounds,
                options,
            );
            if observations.is_empty() {
                continue;
            }
            let median = std::array::from_fn(|channel| {
                median_value(observations.iter().map(|s| s.delta[channel]).collect())
            });
            let scatter = median_value(
                observations
                    .iter()
                    .map(|s| rgb_distance(s.delta, median))
                    .collect(),
            );
            let reliable = observations.len() >= options.min_samples_per_pair.max(3)
                && scatter <= options.max_pair_log_scatter.max(0.0);
            diagnostics.push(PhotometricPairDiagnostic {
                first,
                second,
                samples: observations.len(),
                median_log_ratio: median,
                log_scatter: scatter,
                reliable,
            });
            if reliable {
                let support = observations.len() as f64
                    / (observations.len() + options.min_samples_per_pair.max(1)) as f64;
                pairs.push(Pair {
                    first,
                    second,
                    observations,
                    median,
                    scatter,
                    weight: support / (1.0 + (scatter / 0.04).powi(2)),
                });
            }
        }
    }

    let components = connected_components(tiles.len(), &pairs);
    solve_constants(&mut models, &pairs, &components);
    limit_constants(&mut models, &components, options.max_abs_log_gain.max(0.0));
    let constant_error = validation_error(&models, &pairs);
    let mut corrected_error = constant_error;
    let mut linear_used = false;
    if options.allow_linear && !pairs.is_empty() {
        let mut candidate = models.clone();
        solve_linear(&mut candidate, &pairs, &components, options);
        let error = validation_error(&candidate, &pairs);
        // A held-out improvement must be both material and proportionate. This
        // prevents ordinary texture/noise from creating extrapolated gradients.
        if error + 0.004 < constant_error && error < constant_error * 0.80 {
            models = candidate;
            corrected_error = error;
            linear_used = true;
        }
    }
    solve_offsets(&mut models, &pairs, &components, PHOTOMETRIC_MAX_ABS_OFFSET);
    PhotometricCalibration {
        models,
        pairs: diagnostics,
        linear_used,
        held_out_constant_error: constant_error,
        held_out_corrected_error: corrected_error,
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn normalized_basis(width: u32, height: u32, x: f64, y: f64) -> [f64; 3] {
    [
        1.0,
        2.0 * x / width.saturating_sub(1).max(1) as f64 - 1.0,
        2.0 * y / height.saturating_sub(1).max(1) as f64 - 1.0,
    ]
}

fn project(matrix: &Matrix3<f64>, x: f64, y: f64) -> Option<(f64, f64)> {
    let point = matrix * Vector3::new(x, y, 1.0);
    if point.iter().any(|v| !v.is_finite()) || point.z.abs() < 1e-10 {
        return None;
    }
    Some((point.x / point.z, point.y / point.z))
}

fn world_bounds(width: u32, height: u32, matrix: &Matrix3<f64>) -> Option<[f64; 4]> {
    if width < 2 || height < 2 {
        return None;
    }
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut denominator_sign = 0.0_f64;
    for (x, y) in [
        (0.0, 0.0),
        ((width - 1) as f64, 0.0),
        (0.0, (height - 1) as f64),
        ((width - 1) as f64, (height - 1) as f64),
    ] {
        let denominator = matrix[(2, 0)] * x + matrix[(2, 1)] * y + matrix[(2, 2)];
        if denominator_sign != 0.0 && denominator.signum() != denominator_sign {
            return None; // A projective pole crosses the preview.
        }
        denominator_sign = denominator.signum();
        let (wx, wy) = project(matrix, x, y)?;
        bounds[0] = bounds[0].min(wx);
        bounds[1] = bounds[1].min(wy);
        bounds[2] = bounds[2].max(wx);
        bounds[3] = bounds[3].max(wy);
    }
    Some(bounds)
}

fn intersect_bounds(first: [f64; 4], second: [f64; 4]) -> Option<[f64; 4]> {
    let bounds = [
        first[0].max(second[0]),
        first[1].max(second[1]),
        first[2].min(second[2]),
        first[3].min(second[3]),
    ];
    (bounds[2] > bounds[0] && bounds[3] > bounds[1]).then_some(bounds)
}

fn sample_rgb(image: &Rgb32FImage, x: f64, y: f64) -> Option<[f64; 3]> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= image.width().saturating_sub(1) as f64
        || y >= image.height().saturating_sub(1) as f64
    {
        return None;
    }
    let ix = x as u32;
    let iy = y as u32;
    let fx = x - ix as f64;
    let fy = y - iy as f64;
    Some(std::array::from_fn(|channel| {
        let a = image.get_pixel(ix, iy)[channel] as f64;
        let b = image.get_pixel(ix + 1, iy)[channel] as f64;
        let c = image.get_pixel(ix, iy + 1)[channel] as f64;
        let d = image.get_pixel(ix + 1, iy + 1)[channel] as f64;
        (a + fx * (b - a)) * (1.0 - fy) + (c + fx * (d - c)) * fy
    }))
}

fn sample_overlap(
    first: &Rgb32FImage,
    first_inverse: &Matrix3<f64>,
    second: &Rgb32FImage,
    second_inverse: &Matrix3<f64>,
    bounds: [f64; 4],
    options: &PhotometricOptions,
) -> Vec<Observation> {
    let budget = options.max_samples_per_pair.clamp(1, 65536);
    let aspect = (bounds[2] - bounds[0]) / (bounds[3] - bounds[1]);
    if !aspect.is_finite() || aspect <= 0.0 {
        return Vec::new();
    }
    let columns = ((budget as f64 * aspect).sqrt().round() as usize).clamp(1, budget);
    let rows = (budget / columns).max(1);
    let mut samples = Vec::with_capacity(rows * columns);
    // This grid depends only on the common world intersection. Swapping the
    // previews gives identical sample locations and exactly opposite ratios.
    for row in 0..rows {
        for column in 0..columns {
            let wx = bounds[0] + (column as f64 + 0.5) / columns as f64 * (bounds[2] - bounds[0]);
            let wy = bounds[1] + (row as f64 + 0.5) / rows as f64 * (bounds[3] - bounds[1]);
            let (Some((ax, ay)), Some((bx, by))) = (
                project(first_inverse, wx, wy),
                project(second_inverse, wx, wy),
            ) else {
                continue;
            };
            let (Some(a), Some(b)) = (sample_rgb(first, ax, ay), sample_rgb(second, bx, by)) else {
                continue;
            };
            // These are sensor/numerical validity limits, never a hue, chroma,
            // paper, or foreground selection rule.
            if a.iter().chain(b.iter()).any(|v| {
                !v.is_finite() || *v <= options.min_sample_value || *v >= options.max_sample_value
            }) {
                continue;
            }
            samples.push(Observation {
                first_basis: normalized_basis(first.width(), first.height(), ax, ay),
                second_basis: normalized_basis(second.width(), second.height(), bx, by),
                delta: std::array::from_fn(|channel| b[channel].ln() - a[channel].ln()),
                first_rgb: a,
                second_rgb: b,
            });
        }
    }
    samples
}

fn median_value(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn rgb_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>() / 3.0).sqrt()
}

fn connected_components(count: usize, pairs: &[Pair]) -> Vec<Vec<usize>> {
    let mut labels: Vec<_> = (0..count).collect();
    for pair in pairs {
        let old = labels[pair.second];
        let new = labels[pair.first];
        for label in &mut labels {
            if *label == old {
                *label = new;
            }
        }
    }
    let mut components: Vec<Vec<usize>> = Vec::new();
    for index in 0..count {
        if labels[index] == index {
            components.push((0..count).filter(|other| labels[*other] == index).collect());
        }
    }
    components
}

fn huber_weight(residual: f64, scale: f64) -> f64 {
    if residual.abs() <= scale {
        1.0
    } else {
        scale / residual.abs()
    }
}

fn add_equation(
    normal: &mut DMatrix<f64>,
    rhs: &mut DVector<f64>,
    terms: &[(usize, f64)],
    value: f64,
    weight: f64,
) {
    for &(row, a) in terms {
        rhs[row] += weight * a * value;
        for &(column, b) in terms {
            normal[(row, column)] += weight * a * b;
        }
    }
}

fn add_gauges(normal: &mut DMatrix<f64>, components: &[Vec<usize>], stride: usize) {
    for component in components {
        let weight = 1.0 / component.len() as f64;
        for &first in component {
            for &second in component {
                normal[(first * stride, second * stride)] += weight;
            }
        }
    }
}

fn solve_constants(models: &mut [PhotometricModel], pairs: &[Pair], components: &[Vec<usize>]) {
    if models.is_empty() {
        return;
    }
    for channel in 0..3 {
        let mut solution = DVector::zeros(models.len());
        for _ in 0..12 {
            let mut normal = DMatrix::zeros(models.len(), models.len());
            let mut rhs = DVector::zeros(models.len());
            for pair in pairs {
                let residual = solution[pair.first] - solution[pair.second] - pair.median[channel];
                let weight = pair.weight * huber_weight(residual, 0.025 + pair.scatter);
                add_equation(
                    &mut normal,
                    &mut rhs,
                    &[(pair.first, 1.0), (pair.second, -1.0)],
                    pair.median[channel],
                    weight,
                );
            }
            add_gauges(&mut normal, components, 1);
            let Some(decomposition) = normal.cholesky() else {
                break;
            };
            let next = decomposition.solve(&rhs);
            let change = (&next - &solution).amax();
            solution = next;
            if change < 1e-8 {
                break;
            }
        }
        for (index, model) in models.iter_mut().enumerate() {
            model.log_gain[channel][0] = solution[index];
        }
    }
}

fn limit_constants(models: &mut [PhotometricModel], components: &[Vec<usize>], limit: f64) {
    for component in components {
        for channel in 0..3 {
            let peak = component
                .iter()
                .map(|&i| models[i].log_gain[channel][0].abs())
                .fold(0.0_f64, f64::max);
            if peak > limit && peak > 0.0 {
                // A common scale preserves the gauge; per-tile clipping does not.
                for &i in component {
                    models[i].log_gain[channel][0] *= limit / peak;
                }
            }
        }
    }
}

fn solve_linear(
    models: &mut [PhotometricModel],
    pairs: &[Pair],
    components: &[Vec<usize>],
    options: &PhotometricOptions,
) {
    let count = models.len() * 3;
    for channel in 0..3 {
        let mut solution = DVector::from_iterator(
            count,
            models.iter().flat_map(|model| model.log_gain[channel]),
        );
        for _ in 0..8 {
            let mut normal = DMatrix::zeros(count, count);
            let mut rhs = DVector::zeros(count);
            for pair in pairs {
                let training_count = pair.observations.len() - pair.observations.len().div_ceil(5);
                let sample_weight = pair.weight / training_count.max(1) as f64;
                for (index, observation) in pair.observations.iter().enumerate() {
                    if index % 5 == 0 {
                        continue;
                    }
                    let terms = [
                        (pair.first * 3, 1.0),
                        (pair.first * 3 + 1, observation.first_basis[1]),
                        (pair.first * 3 + 2, observation.first_basis[2]),
                        (pair.second * 3, -1.0),
                        (pair.second * 3 + 1, -observation.second_basis[1]),
                        (pair.second * 3 + 2, -observation.second_basis[2]),
                    ];
                    let predicted: f64 = terms.iter().map(|&(i, value)| solution[i] * value).sum();
                    let residual = predicted - observation.delta[channel];
                    add_equation(
                        &mut normal,
                        &mut rhs,
                        &terms,
                        observation.delta[channel],
                        sample_weight * huber_weight(residual, 0.025),
                    );
                }
            }
            add_gauges(&mut normal, components, 3);
            for index in 0..models.len() {
                for term in 1..3 {
                    normal[(index * 3 + term, index * 3 + term)] +=
                        options.linear_regularization.max(1e-4);
                }
            }
            let Some(decomposition) = normal.cholesky() else {
                break;
            };
            let next = decomposition.solve(&rhs);
            let change = (&next - &solution).amax();
            solution = next;
            if change < 1e-8 {
                break;
            }
        }
        for (index, model) in models.iter_mut().enumerate() {
            model.log_gain[channel] = std::array::from_fn(|term| solution[index * 3 + term]);
            let slope = model.log_gain[channel][1].abs() + model.log_gain[channel][2].abs();
            let limit = options.max_linear_log_gain.max(0.0);
            if slope > limit && slope > 0.0 {
                model.log_gain[channel][1] *= limit / slope;
                model.log_gain[channel][2] *= limit / slope;
            }
        }
    }
    limit_constants(models, components, options.max_abs_log_gain.max(0.0));
}

/// Solve the additive part of the affine tone model after the multiplicative
/// gains have converged.  A pair contributes the robust median of
/// `gain(second) * second - gain(first) * first`, so the equation has the same
/// direction as the log-gain relation (`offset[first] - offset[second]`).
/// Components retain a zero-mean gauge, and the final value is clipped to the
/// small linear-light bound required by Tone_Harmonizer.
fn solve_offsets(
    models: &mut [PhotometricModel],
    pairs: &[Pair],
    components: &[Vec<usize>],
    max_abs_offset: f64,
) {
    if models.is_empty() || pairs.is_empty() {
        return;
    }
    for channel in 0..3 {
        let mut normal = DMatrix::zeros(models.len(), models.len());
        let mut rhs = DVector::zeros(models.len());
        for pair in pairs {
            let mut values = Vec::with_capacity(pair.observations.len());
            for observation in &pair.observations {
                let first_gain = dot(
                    models[pair.first].log_gain[channel],
                    observation.first_basis,
                )
                .clamp(
                    -models[pair.first].max_abs_log_gain,
                    models[pair.first].max_abs_log_gain,
                )
                .exp();
                let second_gain = dot(
                    models[pair.second].log_gain[channel],
                    observation.second_basis,
                )
                .clamp(
                    -models[pair.second].max_abs_log_gain,
                    models[pair.second].max_abs_log_gain,
                )
                .exp();
                values.push(
                    second_gain * observation.second_rgb[channel]
                        - first_gain * observation.first_rgb[channel],
                );
            }
            let value = median_value(values);
            add_equation(
                &mut normal,
                &mut rhs,
                &[(pair.first, 1.0), (pair.second, -1.0)],
                value,
                pair.weight,
            );
        }
        add_gauges(&mut normal, components, 1);
        let Some(decomposition) = normal.cholesky() else {
            continue;
        };
        let solution = decomposition.solve(&rhs);
        for (index, model) in models.iter_mut().enumerate() {
            let solved = solution[index];
            let bounded = solved.clamp(-max_abs_offset, max_abs_offset);
            model.solved_offset[channel] = solved;
            model.offset[channel] = bounded;
            model.offset_clamped |= (solved - bounded).abs() > 1e-12;
        }
    }
    for (index, model) in models.iter().enumerate() {
        if model.offset_clamped {
            crate::panorama_utils::stack_pipeline::degradation::record_run_degradation(
                crate::panorama_utils::stack_pipeline::degradation::TONE_GAIN_CLAMPED,
                serde_json::json!({
                    "stage": "photometric_affine",
                    "tile": index,
                    "solved_offset": model.solved_offset,
                    "offset": model.offset,
                    "limit": max_abs_offset,
                }),
            );
        }
    }
}

fn validation_error(models: &[PhotometricModel], pairs: &[Pair]) -> f64 {
    let mut weighted_error = 0.0;
    let mut weight = 0.0;
    for pair in pairs {
        let errors = pair
            .observations
            .iter()
            .enumerate()
            .filter(|(index, _)| index % 5 == 0)
            .map(|(_, observation)| {
                let corrected = std::array::from_fn(|channel| {
                    dot(
                        models[pair.first].log_gain[channel],
                        observation.first_basis,
                    ) - dot(
                        models[pair.second].log_gain[channel],
                        observation.second_basis,
                    )
                });
                rgb_distance(corrected, observation.delta)
            })
            .collect();
        weighted_error += median_value(errors) * pair.weight;
        weight += pair.weight;
    }
    if weight > 0.0 {
        weighted_error / weight
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(x: f64, y: f64) -> [f64; 3] {
        // Unequal scene brightness across the panorama is intentional.
        let base = 0.15 + 0.0012 * x + 0.02 * (y * 0.13).sin();
        [base, base * 0.78 + 0.02 * (x * 0.10).sin(), base * 0.62]
    }

    fn tile(offset: f64, gain: [f64; 3]) -> (Rgb32FImage, Matrix3<f64>) {
        (
            Rgb32FImage::from_fn(120, 80, |x, y| {
                let color = scene(x as f64 + offset, y as f64);
                Rgb(std::array::from_fn(|c| (color[c] * gain[c]) as f32))
            }),
            Matrix3::new(1.0, 0.0, offset, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        )
    }

    #[test]
    fn real_scene_brightness_difference_is_not_equalized() {
        let tiles = vec![
            tile(0.0, [1.0; 3]),
            tile(70.0, [1.0; 3]),
            tile(140.0, [1.0; 3]),
        ];
        let calibration = calibrate_overlap_photometry(&tiles, &PhotometricOptions::default());
        assert_eq!(calibration.pairs.iter().filter(|p| p.reliable).count(), 2);
        for model in calibration.models {
            for gain in model.gain_at(60.0, 40.0) {
                assert!((gain - 1.0).abs() < 1e-6, "scene gradient changed: {gain}");
            }
        }
        let left = tiles[0].0.get_pixel(30, 40)[0];
        let right = tiles[2].0.get_pixel(30, 40)[0];
        assert!(right > left * 1.8);
    }

    #[test]
    fn exposure_and_white_balance_are_aligned_from_correspondences() {
        let gains = [[0.80, 1.12, 0.94], [1.20, 0.84, 1.17], [0.97, 0.91, 1.05]];
        let tiles: Vec<_> = gains
            .iter()
            .enumerate()
            .map(|(i, &g)| tile(i as f64 * 45.0, g))
            .collect();
        let calibration = calibrate_overlap_photometry(&tiles, &PhotometricOptions::default());
        assert_eq!(calibration.pairs.iter().filter(|p| p.reliable).count(), 3);
        for channel in 0..3 {
            let geometric_mean = (gains.iter().map(|g| g[channel].ln()).sum::<f64>() / 3.0).exp();
            for (index, model) in calibration.models.iter().enumerate() {
                let corrected_gain =
                    model.gain_at(60.0, 40.0)[channel] as f64 * gains[index][channel];
                assert!((corrected_gain - geometric_mean).abs() < 2e-6);
            }
            assert!(
                calibration
                    .models
                    .iter()
                    .map(|m| m.log_gain[channel][0])
                    .sum::<f64>()
                    .abs()
                    < 1e-9
            );
        }
        assert!(calibration.held_out_corrected_error < 1e-6);
    }

    #[test]
    fn isolated_tile_and_invalid_transform_keep_identity() {
        let mut invalid = tile(0.0, [1.4; 3]);
        invalid.1 = Matrix3::zeros();
        let tiles = vec![
            tile(0.0, [0.8; 3]),
            tile(40.0, [1.2; 3]),
            tile(400.0, [1.3; 3]),
            invalid,
        ];
        let calibration = calibrate_overlap_photometry(&tiles, &PhotometricOptions::default());
        assert_eq!(calibration.models[2].gain_at(60.0, 40.0), [1.0; 3]);
        assert_eq!(calibration.models[3].gain_at(60.0, 40.0), [1.0; 3]);
    }

    #[test]
    fn input_permutation_has_equivalent_corrections() {
        let tiles = vec![
            tile(0.0, [0.8, 1.1, 0.9]),
            tile(45.0, [1.1, 0.9, 1.2]),
            tile(90.0, [1.0, 0.95, 1.0]),
        ];
        let options = PhotometricOptions {
            allow_linear: true,
            ..Default::default()
        };
        let original = calibrate_overlap_photometry(&tiles, &options);
        let order = [2, 0, 1];
        let reordered: Vec<_> = order.iter().map(|&i| tiles[i].clone()).collect();
        let changed = calibrate_overlap_photometry(&reordered, &options);
        for (new, &old) in order.iter().enumerate() {
            for channel in 0..3 {
                for term in 0..3 {
                    assert!(
                        (changed.models[new].log_gain[channel][term]
                            - original.models[old].log_gain[channel][term])
                            .abs()
                            < 1e-8
                    );
                }
            }
        }
        assert!(!original.linear_used);
        assert!(!changed.linear_used);
    }

    #[test]
    fn localized_correspondence_outliers_do_not_drive_exposure() {
        let first = tile(0.0, [0.8; 3]);
        let mut second = tile(30.0, [1.2; 3]);
        for y in 0..15 {
            for x in 0..second.0.width() {
                second.0.put_pixel(x, y, Rgb([0.65, 0.13, 0.40]));
            }
        }
        let calibration =
            calibrate_overlap_photometry(&[first, second], &PhotometricOptions::default());
        for channel in 0..3 {
            let relative = calibration.models[0].gain_at(60.0, 40.0)[channel]
                / calibration.models[1].gain_at(60.0, 40.0)[channel];
            assert!((relative - 1.5).abs() < 1e-5);
        }
    }

    #[test]
    fn bounded_linear_correction_requires_held_out_evidence() {
        let first = tile(0.0, [1.0; 3]);
        let mut second = tile(0.0, [1.0; 3]);
        for (x, y, pixel) in second.0.enumerate_pixels_mut() {
            let basis = normalized_basis(120, 80, x as f64, y as f64);
            let gain = (0.16 * basis[1] + 0.06 * basis[2]).exp() as f32;
            for channel in &mut pixel.0 {
                *channel *= gain;
            }
        }
        let options = PhotometricOptions {
            allow_linear: true,
            max_linear_log_gain: 0.13,
            linear_regularization: 0.001,
            ..Default::default()
        };
        let calibration = calibrate_overlap_photometry(&[first, second], &options);
        assert!(calibration.linear_used);
        assert!(calibration.held_out_corrected_error < calibration.held_out_constant_error * 0.3);
        for model in calibration.models {
            for coefficients in model.log_gain {
                assert!(coefficients[1].abs() + coefficients[2].abs() <= 0.1300001);
            }
        }
    }

    #[test]
    fn empty_and_clipped_inputs_do_not_create_corrections() {
        assert!(
            calibrate_overlap_photometry(&[], &PhotometricOptions::default())
                .models
                .is_empty()
        );
        let clipped = (
            Rgb32FImage::from_pixel(80, 80, Rgb([1.0; 3])),
            Matrix3::identity(),
        );
        let calibration = calibrate_overlap_photometry(
            &[clipped.clone(), clipped],
            &PhotometricOptions::default(),
        );
        assert!(calibration.pairs.is_empty());
        assert!(
            calibration
                .models
                .iter()
                .all(|m| m.gain_at(30.0, 30.0) == [1.0; 3])
        );
    }
}
