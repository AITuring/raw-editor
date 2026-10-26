/// Evaluate focus at several locations inside an ownership cell. The old
/// single centre sample could land on quiet paper while the candidate held a
/// sharp brush stroke a few pixels away. Average the two strongest spatial
/// probes so sparse detail contributes; each probe filters pixel noise before
/// measuring its edge response.
fn cell_focus(
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
pub(super) fn cell_probe_offsets(cell_size: f64) -> [(f64, f64); 5] {
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
pub(super) fn cell_focus_with_step(
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
fn streaming_acutance(
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

fn streaming_cell_focus(
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
pub(super) const CELL_DIFFERENCE_OFFSETS: [f64; 5] = [0.15, 0.325, 0.5, 0.675, 0.85];

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
pub(super) fn cell_low_pass_differences(
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

fn ownership_disagreement(
    base: &Rgb32FImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    ax: f64,
    ay: f64,
    cell_size: f64,
) -> f64 {
    // Retain a high percentile so a displaced contour vetoes the candidate even
    // when most of the cell is quiet paper.  This is the legacy comparison
    // path's reduction; the Focus_Fuser uses the median 需求 3.3 prescribes.
    let mut disagreements = cell_low_pass_differences(
        |x_offset, y_offset| {
            let sample_ax = ax + cell_size * x_offset;
            let sample_ay = ay + cell_size * y_offset;
            sampler
                .sample(sample_ax * sampler.scale, sample_ay * sampler.scale)
                .map(|pixel| adjusted(pixel, tone.at(sample_ax, sample_ay)))
        },
        |x_offset, y_offset| {
            let sample_ax = ax + cell_size * x_offset;
            let sample_ay = ay + cell_size * y_offset;
            rgb_at(
                base,
                sampler.left as f64 + sample_ax * sampler.scale,
                sampler.top as f64 + sample_ay * sampler.scale,
            )
        },
    );
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
    ownership_with_long_side(
        base,
        base_mask,
        sampler,
        tone,
        width,
        height,
        SELECTION_LONG_SIDE,
    )
}

fn ownership_with_long_side(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    width: u32,
    height: u32,
    selection_long_side: u32,
) -> GrayImage {
    let size = (width.max(height) as f64 / selection_long_side.max(1) as f64).max(1.0);
    ownership_grid(base, base_mask, sampler, tone, width, height, size, false)
}

fn streaming_ownership_with_long_side(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    width: u32,
    height: u32,
    selection_long_side: u32,
) -> GrayImage {
    let size = (width.max(height) as f64 / selection_long_side.max(1) as f64).max(1.0);
    ownership_grid(base, base_mask, sampler, tone, width, height, size, true)
}

fn ownership_grid(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    sampler: &LayerSampler<'_>,
    tone: &Field<3>,
    width: u32,
    height: u32,
    size: f64,
    streaming: bool,
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
            // Keep a one-cell guard at the *photographic source* boundary.
            // `width`/`height` can describe a streaming backing tile rather
            // than the source footprint; treating every tile edge as a source
            // edge forced the old owner into a visible 1024px square grid.
            let native_cell = size * sampler.scale;
            if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                let source_boundary = [
                    (-native_cell, 0.0),
                    (native_cell, 0.0),
                    (0.0, -native_cell),
                    (0.0, native_cell),
                ]
                .iter()
                .any(|(dx, dy)| sampler.sample(lx + dx, ly + dy).is_none());
                if source_boundary {
                    fixed[i] = -1;
                    continue;
                }
            }
            let base_focus = if streaming {
                streaming_cell_focus(
                    |x, y| rgb_at(base, x, y),
                    gx + native_cell * 0.5,
                    gy + native_cell * 0.5,
                    native_cell,
                )
            } else {
                cell_focus(
                    |x, y| rgb_at(base, x, y),
                    gx + native_cell * 0.5,
                    gy + native_cell * 0.5,
                    native_cell,
                )
            };
            let candidate_focus = if streaming {
                streaming_cell_focus(
                    |x, y| sampler.sample(x, y).map(|p| adjusted(p, tone.at(ax, ay))),
                    lx + native_cell * 0.5,
                    ly + native_cell * 0.5,
                    native_cell,
                )
            } else {
                cell_focus(
                    |x, y| sampler.sample(x, y).map(|p| adjusted(p, tone.at(ax, ay))),
                    lx + native_cell * 0.5,
                    ly + native_cell * 0.5,
                    native_cell,
                )
            };
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

