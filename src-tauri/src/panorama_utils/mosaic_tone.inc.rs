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
    tone_field_with_sampling(base, base_mask, sampler, width, height, 56.0, 4, 8)
}

#[allow(clippy::too_many_arguments)]
fn tone_field_with_sampling(
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

fn adjusted(pixel: Rgb<f32>, delta: [f64; 3]) -> Rgb<f32> {
    Rgb(std::array::from_fn(|c| {
        (pixel[c] * delta[c].exp() as f32).clamp(0.0, 1.0)
    }))
}

// Native focus evidence after a separable low-pass filter. Single-pixel
// gradients confuse sensor grain with detail and miss faded coloured marks
// whose luminance nearly matches the substrate. Keep luminance and two colour
// differences, suppress pixel noise in both axes, then measure coherent edges.
fn acutance(sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>, x: f64, y: f64) -> f64 {
    acutance_with_step(sample, x, y, 1.0)
}

/// Number of patch grid points per axis of one [`acutance`] measurement.
pub(super) const ACUTANCE_PATCH_POINTS: u32 = 13;

/// [`acutance`] with an explicit spacing between the 13×13 patch grid points.
///
/// `step = 1.0` reproduces the legacy measurement exactly (one grid point per
/// native pixel, a ~13px window).  The Focus_Fuser passes a larger step so the
/// effective window reaches the 32 native pixels 需求 3.2 asks for; the gradient
/// stays "four sampling steps apart", so only the sample *positions* scale.
pub(super) fn acutance_with_step(
    mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    step: f64,
) -> f64 {
    let step = if step.is_finite() && step > 0.0 {
        step
    } else {
        1.0
    };
    let mut patch = [[[0.0; 3]; 13]; 13];
    for (j, row) in patch.iter_mut().enumerate() {
        for (i, value) in row.iter_mut().enumerate() {
            let Some(p) = sample(x + (i as f64 - 6.0) * step, y + (j as f64 - 6.0) * step) else {
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
