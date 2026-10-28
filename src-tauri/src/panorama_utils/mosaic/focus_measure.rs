//! Focus measures shared by the mosaic and the Focus_Fuser.
//!
//! Native-resolution acutance, per-cell focus probes and low-pass difference
//! samples, plus the bounded-resolution sharpness that ranks the members of a
//! degraded station.

use super::super::stack_pipeline::intra_station;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Rgb;

use super::luma;

/// Sharpness_Score of a gray analysis image, sampled with `spacing` pixels
/// between probes so a measurement can be taken at a bounded analysis
/// resolution (需求 2.1 / 2.2).  `None` where the probe window would leave the
/// image, which is what "valid pixels" means for a full frame.
pub(crate) fn gray_analysis_sharpness(
    image: &GrayImage,
    x: f64,
    y: f64,
    spacing: f64,
) -> Option<f64> {
    let half = 6.0 * spacing + 1.0;
    if !(x - half >= 0.0
        && y - half >= 0.0
        && x + half < f64::from(image.width())
        && y + half < f64::from(image.height()))
    {
        return None;
    }
    Some(acutance(
        |sample_x, sample_y| {
            intra_station::gray_sample(
                image,
                x + (sample_x - x) * spacing,
                y + (sample_y - y) * spacing,
            )
        },
        x,
        y,
    ))
}

/// Median Sharpness_Score over the valid pixels of one Source_RAW, measured the
/// same way the anchor selection of 需求 2.1 measures it.
///
/// This is what 需求 12.1 ranks the members of a degraded Capture_Station by.
/// The measurement runs on the already prepared analysis image and on the fixed
/// probe grid of [`intra_station::median_valid_sharpness`], so it costs a few
/// thousand samples per frame and is independent of the canvas the station
/// happens to be rendered on.
///
/// Shared with the default station path in [`crate::panorama_utils::stitching`], so a degraded
/// station ranks its members by the same number whichever compositor rendered
/// it.
pub(crate) fn station_member_median_sharpness(info: &ImageInfo) -> f64 {
    let analysis = &info.alignment_image;
    let (width, height) = (analysis.width(), analysis.height());
    let spacing = intra_station::analysis_probe_spacing(width.max(height));
    intra_station::median_valid_sharpness(width, height, |x, y| {
        gray_analysis_sharpness(analysis, x, y, spacing)
    })
}

// Native focus evidence after a separable low-pass filter. Single-pixel
// gradients confuse sensor grain with detail and miss faded coloured marks
// whose luminance nearly matches the substrate. Keep luminance and two colour
// differences, suppress pixel noise in both axes, then measure coherent edges.
pub(super) fn acutance(sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>, x: f64, y: f64) -> f64 {
    acutance_with_step(sample, x, y, 1.0)
}

/// Number of patch grid points per axis of one [`acutance`] measurement.
pub(in crate::panorama_utils) const ACUTANCE_PATCH_POINTS: u32 = 13;

/// [`acutance`] with an explicit spacing between the 13×13 patch grid points.
///
/// `step = 1.0` reproduces the legacy measurement exactly (one grid point per
/// native pixel, a ~13px window).  The Focus_Fuser passes a larger step so the
/// effective window reaches the 32 native pixels 需求 3.2 asks for; the gradient
/// stays "four sampling steps apart", so only the sample *positions* scale.
pub(in crate::panorama_utils) fn acutance_with_step(
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

/// Evaluate focus at several locations inside an ownership cell. The old
/// single centre sample could land on quiet paper while the candidate held a
/// sharp brush stroke a few pixels away. Average the two strongest spatial
/// probes so sparse detail contributes; each probe filters pixel noise before
/// measuring its edge response.
pub(super) fn cell_focus(
    sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    cell_size: f64,
) -> f64 {
    cell_focus_with_step(sample, x, y, cell_size, 1.0)
}

/// Probe positions of one ownership cell, relative to the cell centre.
///
/// Five probes cover the centre and all four corners. A stroke can cross any
/// cell edge, so sampling only one diagonal can still land entirely on quiet
/// paper and hide an in-focus candidate.  Published so the Focus_Fuser can
/// assert that a candidate and the current composite measure the *same*
/// positions inside a cell (需求 3.2) instead of assuming it.
pub(in crate::panorama_utils) fn cell_probe_offsets(cell_size: f64) -> [(f64, f64); 5] {
    let radius = (cell_size * 0.28).clamp(2.0, 18.0);
    [
        (-radius, -radius),
        (radius, -radius),
        (0.0, 0.0),
        (-radius, radius),
        (radius, radius),
    ]
}

/// [`cell_focus`] with an explicit [`acutance_with_step`] window spacing.
pub(in crate::panorama_utils) fn cell_focus_with_step(
    mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    cell_size: f64,
    step: f64,
) -> f64 {
    let mut values = cell_probe_offsets(cell_size)
        .iter()
        .map(|(dx, dy)| acutance_with_step(&mut sample, x + dx, y + dy, step))
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

// Streaming stacks may evaluate hundreds of thousands of ownership cells.
// Measure the same five spatial probes with coherent, low-pass cross
// gradients instead of rebuilding a full 13x13 patch for every probe.  Each
// gradient averages along its tangent before measuring energy, so isolated
// sensor noise does not win over a focused brush edge.
pub(super) fn streaming_acutance(
    sample: &mut impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
) -> Option<f64> {
    const TANGENT: [f64; 3] = [-2.0, 0.0, 2.0];
    const GRADIENTS: [(f64, f64); 2] = [(-3.0, 1.0), (-1.0, 3.0)];
    let mut energy = 0.0;
    let mut level = 0.0;
    let mut level_count = 0.0f64;
    for (near, far) in GRADIENTS {
        let mut dx = [0.0; 3];
        let mut dy = [0.0; 3];
        for tangent in TANGENT {
            let left = sample(x + near, y + tangent)?;
            let right = sample(x + far, y + tangent)?;
            let top = sample(x + tangent, y + near)?;
            let bottom = sample(x + tangent, y + far)?;
            let channels = |pixel: Rgb<f32>| {
                [
                    luma(pixel),
                    f64::from(pixel[0] - pixel[1]) * 0.5,
                    f64::from(pixel[2] - pixel[1]) * 0.5,
                ]
            };
            let left_channels = channels(left);
            let right_channels = channels(right);
            let top_channels = channels(top);
            let bottom_channels = channels(bottom);
            for channel in 0..3 {
                dx[channel] += right_channels[channel] - left_channels[channel];
                dy[channel] += bottom_channels[channel] - top_channels[channel];
            }
            level += luma(left) + luma(right) + luma(top) + luma(bottom);
            level_count += 4.0;
        }
        for channel in 0..3 {
            let dx = dx[channel] / TANGENT.len() as f64;
            let dy = dy[channel] / TANGENT.len() as f64;
            energy += dx * dx + dy * dy;
        }
    }
    Some((energy / (GRADIENTS.len() * 3) as f64).sqrt() / (level / level_count.max(1.0)).max(0.04))
}

pub(super) fn streaming_cell_focus(
    mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    cell_size: f64,
) -> f64 {
    let mut values = cell_probe_offsets(cell_size)
        .into_iter()
        .filter_map(|(dx, dy)| streaming_acutance(&mut sample, x + dx, y + dy))
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let first = values.len().saturating_sub(2);
    (values[first] + values[(first + 1).min(values.len() - 1)]) * 0.5
}

/// Fractional positions inside an ownership cell at which the candidate and the
/// current composite are compared.  A single sample at the cell centre can land
/// on canvas and miss a brush stroke that crosses the same cell near an edge.
pub(in crate::panorama_utils) const CELL_DIFFERENCE_OFFSETS: [f64; 5] =
    [0.15, 0.325, 0.5, 0.675, 0.85];

/// Low-pass absolute colour differences inside one ownership cell.
///
/// Both closures are called with the *same* fractional cell offsets, which is
/// how 需求 3.3's "candidate versus current composite in the same cell" stays a
/// property of the code rather than of the caller: each closure maps the shared
/// offset into its own coordinate space and returns `None` outside its coverage.
///
/// Individual high-frequency pixels are deliberately not compared. Defocus
/// changes brush-edge samples substantially even when the geometry is correct; a
/// displaced contour still changes the local mean over a 3×3 neighbourhood of
/// probes and remains measurable.  The reduction of these values (high
/// percentile for the legacy mosaic path, median for the Focus_Fuser) is left to
/// the caller.
pub(in crate::panorama_utils) fn cell_low_pass_differences(
    mut candidate_at: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    mut base_at: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
) -> Vec<f64> {
    let side = CELL_DIFFERENCE_OFFSETS.len();
    let mut samples = vec![None; side * side];
    for (yi, y_offset) in CELL_DIFFERENCE_OFFSETS.iter().copied().enumerate() {
        for (xi, x_offset) in CELL_DIFFERENCE_OFFSETS.iter().copied().enumerate() {
            let Some(candidate) = candidate_at(x_offset, y_offset) else {
                continue;
            };
            let Some(base_pixel) = base_at(x_offset, y_offset) else {
                continue;
            };
            let candidate = candidate.0.map(f64::from);
            let base_pixel = base_pixel.0.map(f64::from);
            if candidate
                .iter()
                .chain(base_pixel.iter())
                .all(|v| v.is_finite())
            {
                samples[yi * side + xi] = Some((candidate, base_pixel));
            }
        }
    }
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
                let neighbour = y as usize * CELL_DIFFERENCE_OFFSETS.len() + x as usize;
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
    disagreements
}
