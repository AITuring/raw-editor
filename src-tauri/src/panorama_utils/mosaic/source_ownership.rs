//! Source ownership decisions.
//!
//! Per-cell focus, disagreement and transition-structure costs regularised by
//! the global seam cut, on the in-memory grid and on the bounded streaming
//! analysis; fixed coverage labels admit every newly observed pixel.

use super::super::seam_cut;
use image::GrayImage;
use image::Rgb;
use image::Rgb32FImage;
use std::collections::VecDeque;

use super::{
    Field, LayerSampler, OWNERSHIP_MISMATCH_PENALTY, OutputFocus, SELECTION_LONG_SIDE,
    STREAMING_SELECTION_LONG_SIDE, adjusted, cell_focus, cell_low_pass_differences, luma, rgb_at,
    streaming_cell_focus,
};

// Low-pass (analysis pixels) under which two registered camera positions
// agree even though one is defocused or shaken. Wider than that blur, narrower
// than a brush stroke.
const SCAN_TRANSITION_LOW_PASS_SIGMA: f32 = 3.0;
// Correlation of the low-passed views at and above which a transition cell
// counts as registered, and at and below which it keeps the base.
const SCAN_TRANSITION_AGREEING_CORRELATION: f64 = 0.9;
const SCAN_TRANSITION_DISAGREEING_CORRELATION: f64 = 0.5;
// Relative contrast under which a low-passed window has no structure that
// could be misregistered.
const SCAN_TRANSITION_FLAT_CONTRAST: f64 = 0.02;
// Bound of one cell's normalised focus vote in a transition, so a few cells
// with extreme contrast cannot outweigh the seam costs around them.
const SCAN_TRANSITION_FOCUS_LIMIT: f64 = 2.0;
// Weight of the spatial seam prior in a transition that weighs focus. The
// prior only places the cut where the focus evidence does not decide it.
const SCAN_TRANSITION_PRIOR_WEIGHT: f64 = 0.5;

pub(super) fn masked_rgb_at(
    image: &Rgb32FImage,
    mask: &GrayImage,
    x: f64,
    y: f64,
) -> Option<Rgb<f32>> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= image.width() as f64
        || y >= image.height() as f64
        || mask.get_pixel(x as u32, y as u32)[0] == 0
    {
        return None;
    }
    rgb_at(image, x, y)
}

pub(super) fn streaming_analysis_disagreement(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    tone: &Field<3>,
    x: f64,
    y: f64,
    cell_size: f64,
) -> f64 {
    let mut values = Vec::new();
    for y_offset in [0.2, 0.5, 0.8] {
        for x_offset in [0.2, 0.5, 0.8] {
            let sample_x = x + cell_size * x_offset;
            let sample_y = y + cell_size * y_offset;
            let Some(current) = masked_rgb_at(base, base_mask, sample_x, sample_y) else {
                continue;
            };
            let Some(source) = masked_rgb_at(candidate, candidate_mask, sample_x, sample_y) else {
                continue;
            };
            let source = adjusted(source, tone.at(sample_x, sample_y));
            values.push(
                (0..3)
                    .map(|channel| f64::from((current[channel] - source[channel]).abs()))
                    .sum::<f64>()
                    / 3.0,
            );
        }
    }
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let high = values[(values.len() * 3 / 4).min(values.len() - 1)];
    (mean * 0.45 + high * 0.55).clamp(0.0, 1.0)
}

pub(super) fn streaming_transition_structure(
    mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    x: f64,
    y: f64,
    cell_size: f64,
) -> f64 {
    let mut levels = Vec::with_capacity(9);
    for y_offset in [0.18, 0.5, 0.82] {
        for x_offset in [0.18, 0.5, 0.82] {
            if let Some(pixel) = sample(x + cell_size * x_offset, y + cell_size * y_offset) {
                levels.push(luma(pixel));
            }
        }
    }
    if levels.len() < 5 {
        return 0.0;
    }
    levels.sort_by(f64::total_cmp);
    let dark = levels[levels.len() / 8];
    let paper = levels[levels.len() * 3 / 4].max(0.04);
    ((paper - dark) / paper).clamp(0.0, 1.0)
}

pub(super) fn streaming_distance_to_fixed_label(
    width: u32,
    height: u32,
    fixed: &[i8],
    label: i8,
) -> Option<Vec<u32>> {
    let mut distances = vec![u32::MAX; fixed.len()];
    let mut pending = VecDeque::new();
    for (index, &value) in fixed.iter().enumerate() {
        if value == label {
            distances[index] = 0;
            pending.push_back(index);
        }
    }
    if pending.is_empty() {
        return None;
    }
    while let Some(index) = pending.pop_front() {
        let x = index as u32 % width;
        let y = index as u32 / width;
        let next_distance = distances[index].saturating_add(1);
        for neighbor in [
            x.checked_sub(1).map(|nx| (y * width + nx) as usize),
            (x + 1 < width).then_some((y * width + x + 1) as usize),
            y.checked_sub(1).map(|ny| (ny * width + x) as usize),
            (y + 1 < height).then_some(((y + 1) * width + x) as usize),
        ]
        .into_iter()
        .flatten()
        {
            if next_distance < distances[neighbor] {
                distances[neighbor] = next_distance;
                pending.push_back(neighbor);
            }
        }
    }
    Some(distances)
}

/// Whether two camera positions agree after the low-pass of a focus-scan
/// transition, from 0 (structure displaced) to 1 (registered).
///
/// The views are compared by the correlation of their luma over a 5×5
/// window, so a frame-edge band that is darker from lens falloff still agrees
/// with a sharper registered frame and can be replaced; a displaced stroke
/// does not correlate. A window without structure in either view agrees.
fn low_pass_agreement(
    low_base: &Rgb32FImage,
    low_candidate: &Rgb32FImage,
    tone: &Field<3>,
    x: f64,
    y: f64,
) -> f64 {
    const OFFSETS: [f64; 5] = [-4.0, -2.0, 0.0, 2.0, 4.0];
    let mut pairs = Vec::with_capacity(OFFSETS.len() * OFFSETS.len());
    for dy in OFFSETS {
        for dx in OFFSETS {
            let (sample_x, sample_y) = (x + dx, y + dy);
            let (Some(base), Some(candidate)) = (
                rgb_at(low_base, sample_x, sample_y),
                rgb_at(low_candidate, sample_x, sample_y),
            ) else {
                continue;
            };
            pairs.push((
                luma(base),
                luma(adjusted(candidate, tone.at(sample_x, sample_y))),
            ));
        }
    }
    if pairs.len() * 2 < OFFSETS.len() * OFFSETS.len() {
        return 0.0;
    }
    let count = pairs.len() as f64;
    let base_mean = pairs.iter().map(|pair| pair.0).sum::<f64>() / count;
    let candidate_mean = pairs.iter().map(|pair| pair.1).sum::<f64>() / count;
    let (mut base_variance, mut candidate_variance, mut covariance) = (0.0, 0.0, 0.0);
    for &(base, candidate) in &pairs {
        let (base, candidate) = (base - base_mean, candidate - candidate_mean);
        base_variance += base * base;
        candidate_variance += candidate * candidate;
        covariance += base * candidate;
    }
    let level = base_mean.max(candidate_mean).max(0.04);
    let flat = |variance: f64| (variance / count).sqrt() / level < SCAN_TRANSITION_FLAT_CONTRAST;
    if flat(base_variance) && flat(candidate_variance) {
        return 1.0;
    }
    let correlation = covariance / (base_variance * candidate_variance).sqrt().max(1e-12);
    ((correlation - SCAN_TRANSITION_DISAGREEING_CORRELATION)
        / (SCAN_TRANSITION_AGREEING_CORRELATION - SCAN_TRANSITION_DISAGREEING_CORRELATION))
        .clamp(0.0, 1.0)
}

/// `(columns, rows, analysis pixels per cell)` of the streaming ownership grid
/// for a `width`×`height` analysis layer.
pub(super) fn streaming_ownership_grid(width: u32, height: u32) -> (u32, u32, f64) {
    let size = (width.max(height) as f64 / STREAMING_SELECTION_LONG_SIDE.max(1) as f64).max(1.0);
    (
        (width as f64 / size).ceil() as u32,
        (height as f64 / size).ceil() as u32,
        size,
    )
}

/// Ownership of one layer on the streaming analysis grid.
///
/// `base_owner` holds the Capture_Station of every base pixel.
/// `panorama_transition` places a seam between camera positions instead of
/// running the focus contest of a later focus bracket; on a transition,
/// `transition_focus_weight > 0` adds the bounded, normalised focus difference
/// of every cell where both positions agree after a low-pass, and a vote for
/// the base where they do not, to the spatial seam prior. `0.0` keeps the
/// seam purely spatial. `edge_margin_px` (analysis pixels) keeps the
/// candidate's own footprint border out of pixels that are already covered;
/// `0.0` keeps only the one-cell guard. `output_focus` replaces the
/// analysis-resolution focus probes by measurements on output pixels where it
/// has a value for a cell.
#[allow(clippy::too_many_arguments)]
pub(super) fn streaming_ownership_from_analysis(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    tone: &Field<3>,
    base_owner: &GrayImage,
    capture_group_id: u8,
    panorama_transition: bool,
    transition_focus_weight: f64,
    edge_margin_px: f64,
    output_focus: Option<&OutputFocus>,
) -> GrayImage {
    let width = base.width().min(candidate.width());
    let height = base.height().min(candidate.height());
    let (grid_width, grid_height, size) = streaming_ownership_grid(width, height);
    let count = (grid_width * grid_height) as usize;
    // Focus measured on output pixels, when the caller supplied it for this
    // very grid; the analysis-resolution probes remain the fallback.
    let output_focus =
        output_focus.filter(|focus| focus.base.len() == count && focus.candidate.len() == count);
    let measured_focus = |index: usize| {
        output_focus.and_then(|focus| {
            let (base, candidate) = (focus.base[index], focus.candidate[index]);
            (base.is_finite() && candidate.is_finite()).then_some((base, candidate))
        })
    };
    let focus_pair = |index: usize, center_x: f64, center_y: f64| {
        measured_focus(index).unwrap_or_else(|| {
            (
                streaming_cell_focus(
                    |sample_x, sample_y| masked_rgb_at(base, base_mask, sample_x, sample_y),
                    center_x,
                    center_y,
                    size,
                ),
                streaming_cell_focus(
                    |sample_x, sample_y| {
                        masked_rgb_at(candidate, candidate_mask, sample_x, sample_y)
                            .map(|pixel| adjusted(pixel, tone.at(sample_x, sample_y)))
                    },
                    center_x,
                    center_y,
                    size,
                ),
            )
        })
    };
    let mut preference = vec![0.0f64; count];
    let mut disagreement = vec![0.0f64; count];
    let mut fixed = vec![0i8; count];
    let mut mismatch = vec![0.0f64; count];
    let mut energy_sum = 0.0;
    let mut energy_count = 0usize;
    #[cfg(test)]
    let tracing = trace::enabled();
    #[cfg(test)]
    let mut trace_focus = vec![(f64::NAN, f64::NAN); if tracing { count } else { 0 }];
    #[cfg(test)]
    let mut trace_focus_term = Vec::new();
    // A focus-scan transition compares two camera positions. Defocus and
    // shake change their fine detail, but after a low-pass wider than that
    // blur two registered frames agree; a registration or parallax error
    // still moves whole strokes. The agreement of these low-passed views
    // gates the focus evidence of a transition.
    let transition_low_pass = (panorama_transition && transition_focus_weight > 0.0).then(|| {
        (
            image::imageops::blur(base, SCAN_TRANSITION_LOW_PASS_SIGMA),
            image::imageops::blur(candidate, SCAN_TRANSITION_LOW_PASS_SIGMA),
        )
    });
    let mut transition_agreement = vec![0.0f64; count];
    // Cells this close to the candidate's own footprint edge never take
    // pixels another layer already covers: a frame border carries lens
    // correction and interpolation residue, which a focus vote would
    // otherwise expose as a bright or dark streak inside the painting.
    let margin_cells = (edge_margin_px / size).ceil().max(0.0) as u32;
    let candidate_edge_distance = (margin_cells > 1).then(|| {
        let outside = (0..count)
            .map(|index| {
                let center_x = ((index as u32 % grid_width) as f64 + 0.5) * size;
                let center_y = ((index as u32 / grid_width) as f64 + 0.5) * size;
                if masked_rgb_at(
                    candidate,
                    candidate_mask,
                    center_x.min(width.saturating_sub(1) as f64),
                    center_y.min(height.saturating_sub(1) as f64),
                )
                .is_some()
                {
                    0i8
                } else {
                    -1i8
                }
            })
            .collect::<Vec<_>>();
        let to_invalid = streaming_distance_to_fixed_label(grid_width, grid_height, &outside, -1);
        // The analysis grid ends at the layer's bounding box, which is the
        // footprint edge wherever the footprint touches it.
        (0..count)
            .map(|index| {
                let grid_x = index as u32 % grid_width;
                let grid_y = index as u32 / grid_width;
                let to_border = grid_x
                    .min(grid_y)
                    .min(grid_width - 1 - grid_x)
                    .min(grid_height - 1 - grid_y);
                to_invalid
                    .as_ref()
                    .map_or(to_border, |distances| distances[index].min(to_border))
            })
            .collect::<Vec<_>>()
    });
    for grid_y in 0..grid_height {
        for grid_x in 0..grid_width {
            let index = (grid_y * grid_width + grid_x) as usize;
            let x = grid_x as f64 * size;
            let y = grid_y as f64 * size;
            let center_x = x + size * 0.5;
            let center_y = y + size * 0.5;
            let candidate_valid = masked_rgb_at(
                candidate,
                candidate_mask,
                center_x.min(width.saturating_sub(1) as f64),
                center_y.min(height.saturating_sub(1) as f64),
            )
            .is_some();
            if !candidate_valid {
                fixed[index] = -1;
                continue;
            }
            let base_valid = masked_rgb_at(
                base,
                base_mask,
                center_x.min(width.saturating_sub(1) as f64),
                center_y.min(height.saturating_sub(1) as f64),
            )
            .is_some();
            if !base_valid {
                fixed[index] = 1;
                continue;
            }
            let owner_x = center_x
                .floor()
                .clamp(0.0, base_owner.width().saturating_sub(1) as f64)
                as u32;
            let owner_y = center_y
                .floor()
                .clamp(0.0, base_owner.height().saturating_sub(1) as f64)
                as u32;
            // Once a panorama position owns a pixel, later focal planes from
            // another camera position may not reclaim it merely because a
            // repeated brush stroke scores as sharper.  A new camera group is
            // introduced once through a connected panorama seam; subsequent
            // members only compete inside that group's owned region.
            if !panorama_transition && base_owner.get_pixel(owner_x, owner_y)[0] != capture_group_id
            {
                fixed[index] = -1;
                continue;
            }
            let source_boundary = [(-size, 0.0), (size, 0.0), (0.0, -size), (0.0, size)]
                .iter()
                .any(|(dx, dy)| {
                    masked_rgb_at(candidate, candidate_mask, center_x + dx, center_y + dy).is_none()
                });
            let inside_edge_margin = candidate_edge_distance
                .as_ref()
                .is_some_and(|distances| distances[index] < margin_cells);
            if source_boundary || inside_edge_margin {
                fixed[index] = -1;
                continue;
            }
            disagreement[index] = streaming_analysis_disagreement(
                base,
                base_mask,
                candidate,
                candidate_mask,
                tone,
                x,
                y,
                size,
            );
            if panorama_transition {
                // Cross-position stitching is a seam-placement problem, not
                // a focus contest. Prefer the established panorama except for
                // the connected region needed to admit genuinely new source
                // coverage. Make ink/edge cells expensive even when the two
                // sources agree photometrically: a seam through an aligned
                // brush stroke can still expose a tiny residual warp as a
                // horizontally cut glyph. Quiet paper remains the cheapest
                // route between characters.
                let base_structure = streaming_transition_structure(
                    |sample_x, sample_y| masked_rgb_at(base, base_mask, sample_x, sample_y),
                    x,
                    y,
                    size,
                );
                let candidate_structure = streaming_transition_structure(
                    |sample_x, sample_y| {
                        masked_rgb_at(candidate, candidate_mask, sample_x, sample_y)
                            .map(|pixel| adjusted(pixel, tone.at(sample_x, sample_y)))
                    },
                    x,
                    y,
                    size,
                );
                disagreement[index] = disagreement[index]
                    .max(base_structure)
                    .max(candidate_structure);
                preference[index] = 0.0;
                if let Some((low_base, low_candidate)) = transition_low_pass.as_ref() {
                    // A focus scan photographs every region from two or three
                    // positions, and handheld frames differ in focus and
                    // shake. Let the sharper position own the overlap, but
                    // only where both agree at low frequency: across camera
                    // positions a disagreement is a registration error, not
                    // defocus.
                    let sample_x = center_x.min(width.saturating_sub(1) as f64);
                    let sample_y = center_y.min(height.saturating_sub(1) as f64);
                    transition_agreement[index] =
                        low_pass_agreement(low_base, low_candidate, tone, sample_x, sample_y);
                    let (base_focus, candidate_focus) = focus_pair(index, center_x, center_y);
                    #[cfg(test)]
                    if tracing {
                        trace_focus[index] = (base_focus, candidate_focus);
                    }
                    preference[index] = candidate_focus.powi(2) - base_focus.powi(2) * 1.06;
                    energy_sum += (candidate_focus.powi(2) + base_focus.powi(2)) * 0.5;
                    energy_count += 1;
                }
                continue;
            }
            let (base_focus, candidate_focus) = focus_pair(index, center_x, center_y);
            #[cfg(test)]
            if tracing {
                trace_focus[index] = (base_focus, candidate_focus);
            }
            preference[index] = candidate_focus.powi(2) - base_focus.powi(2) * 1.06;
            energy_sum += (candidate_focus.powi(2) + base_focus.powi(2)) * 0.5;
            energy_count += 1;
            let mismatch_scale = if candidate_focus > base_focus * 1.12 {
                0.30
            } else {
                1.0
            };
            mismatch[index] =
                (disagreement[index] * OWNERSHIP_MISMATCH_PENALTY * mismatch_scale).min(1.0);
        }
    }
    if panorama_transition {
        // The focus evidence of a transition, on the same normalised scale as
        // the focus contest below; zero when the run does not weigh focus.
        let energy_scale = (energy_sum / energy_count.max(1) as f64).max(1e-6);
        let focus_term = (0..count)
            .map(|index| {
                if transition_low_pass.is_none() || fixed[index] != 0 {
                    return 0.0;
                }
                // Agreeing cells vote with their bounded focus difference;
                // disagreeing cells keep the established position.
                let agreement = transition_agreement[index];
                let focus = (preference[index] / energy_scale)
                    .clamp(-SCAN_TRANSITION_FOCUS_LIMIT, SCAN_TRANSITION_FOCUS_LIMIT);
                transition_focus_weight * (agreement * focus - (1.0 - agreement))
            })
            .collect::<Vec<_>>();
        preference.copy_from_slice(&focus_term);
        #[cfg(test)]
        if tracing {
            trace_focus_term = focus_term.clone();
        }
        // Put the transition near the middle of the real overlap. With a
        // uniform bias toward the old panorama the minimum cut hugs the first
        // newly uncovered cell and exposes the rectangular source footprint.
        // Distances to the mandatory old/new coverage seeds provide a spatial
        // prior without using filename order or inventing image content.
        let prior_weight = if transition_low_pass.is_some() {
            SCAN_TRANSITION_PRIOR_WEIGHT
        } else {
            1.0
        };
        if let (Some(base_distance), Some(candidate_distance)) = (
            streaming_distance_to_fixed_label(grid_width, grid_height, &fixed, -1),
            streaming_distance_to_fixed_label(grid_width, grid_height, &fixed, 1),
        ) {
            for index in 0..count {
                if fixed[index] != 0 {
                    continue;
                }
                let base = base_distance[index] as f64;
                let candidate = candidate_distance[index] as f64;
                // Keep most of an already verified panorama intact. The new
                // camera position should own its genuinely new coverage and a
                // narrow connected overlap for a non-rectangular seam, not the
                // middle of a very large overlap. Mid-overlap ownership let a
                // weak defocused bridge (Wen Yuan Tu 3483/3484) replace the
                // tree roots and display rail with a visibly displaced block.
                // The focus term moves that seam toward whichever position is
                // sharper there and agrees with the other at low frequency.
                preference[index] += prior_weight
                    * (((base - candidate) / (base + candidate).max(1.0)) * 0.60 - 0.30);
            }
        }
    }
    if !panorama_transition {
        let energy_scale = (energy_sum / energy_count.max(1) as f64).max(1e-6);
        for (index, value) in preference.iter_mut().enumerate() {
            if fixed[index] == 0 {
                *value = (*value / energy_scale).clamp(-12.0, 12.0) - mismatch[index];
            }
        }
    }
    let labels = seam_cut::cut_grid(
        grid_width as usize,
        grid_height as usize,
        &preference,
        &disagreement,
        &fixed,
    );
    #[cfg(test)]
    if tracing {
        trace::store(trace::OwnershipTrace {
            grid_width,
            size,
            fixed: fixed.clone(),
            agreement: transition_agreement.clone(),
            focus: trace_focus,
            focus_term: trace_focus_term,
            preference: preference.clone(),
            disagreement: disagreement.clone(),
            labels: labels.clone(),
        });
    }
    GrayImage::from_raw(grid_width, grid_height, labels).expect("ownership dimensions")
}

pub(super) fn ownership_disagreement(
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
pub(super) fn ownership(
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

pub(super) fn ownership_with_long_side(
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

pub(super) fn streaming_ownership_with_long_side(
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

pub(super) fn ownership_grid(
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

/// Test-only record of the last streaming ownership decision on this thread,
/// so a real-image run can explain one region cell by cell
/// (`RAW_EDITOR_OWNERSHIP_TRACE`, see the mosaic diagnostics).
#[cfg(test)]
pub(super) mod trace {
    use std::cell::RefCell;

    #[derive(Clone, Default)]
    pub(in crate::panorama_utils) struct OwnershipTrace {
        pub(in crate::panorama_utils) grid_width: u32,
        pub(in crate::panorama_utils) size: f64,
        pub(in crate::panorama_utils) fixed: Vec<i8>,
        pub(in crate::panorama_utils) agreement: Vec<f64>,
        pub(in crate::panorama_utils) focus: Vec<(f64, f64)>,
        pub(in crate::panorama_utils) focus_term: Vec<f64>,
        pub(in crate::panorama_utils) preference: Vec<f64>,
        pub(in crate::panorama_utils) disagreement: Vec<f64>,
        pub(in crate::panorama_utils) labels: Vec<u8>,
    }

    thread_local! {
        static LAST: RefCell<Option<OwnershipTrace>> = const { RefCell::new(None) };
    }

    pub(in crate::panorama_utils) fn enabled() -> bool {
        std::env::var_os("RAW_EDITOR_OWNERSHIP_TRACE").is_some()
    }

    pub(in crate::panorama_utils) fn store(trace: OwnershipTrace) {
        LAST.with(|slot| *slot.borrow_mut() = Some(trace));
    }

    pub(in crate::panorama_utils) fn take() -> Option<OwnershipTrace> {
        LAST.with(|slot| slot.borrow_mut().take())
    }
}
