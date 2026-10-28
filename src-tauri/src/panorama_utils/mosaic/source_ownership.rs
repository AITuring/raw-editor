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
    Field, LayerSampler, OWNERSHIP_MISMATCH_PENALTY, SELECTION_LONG_SIDE,
    STREAMING_SELECTION_LONG_SIDE, adjusted, cell_focus, cell_low_pass_differences, luma, rgb_at,
    streaming_cell_focus,
};

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

pub(super) fn streaming_ownership_from_analysis(
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
    tone: &Field<3>,
    base_owner: &GrayImage,
    capture_group_id: u8,
    panorama_transition: bool,
) -> GrayImage {
    let width = base.width().min(candidate.width());
    let height = base.height().min(candidate.height());
    let size = (width.max(height) as f64 / STREAMING_SELECTION_LONG_SIDE.max(1) as f64).max(1.0);
    let grid_width = (width as f64 / size).ceil() as u32;
    let grid_height = (height as f64 / size).ceil() as u32;
    let count = (grid_width * grid_height) as usize;
    let mut preference = vec![0.0f64; count];
    let mut disagreement = vec![0.0f64; count];
    let mut fixed = vec![0i8; count];
    let mut mismatch = vec![0.0f64; count];
    let mut energy_sum = 0.0;
    let mut energy_count = 0usize;
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
            if source_boundary {
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
                continue;
            }
            let base_focus = streaming_cell_focus(
                |sample_x, sample_y| masked_rgb_at(base, base_mask, sample_x, sample_y),
                center_x,
                center_y,
                size,
            );
            let candidate_focus = streaming_cell_focus(
                |sample_x, sample_y| {
                    masked_rgb_at(candidate, candidate_mask, sample_x, sample_y)
                        .map(|pixel| adjusted(pixel, tone.at(sample_x, sample_y)))
                },
                center_x,
                center_y,
                size,
            );
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
        // Put the transition near the middle of the real overlap. With a
        // uniform bias toward the old panorama the minimum cut hugs the first
        // newly uncovered cell and exposes the rectangular source footprint.
        // Distances to the mandatory old/new coverage seeds provide a spatial
        // prior without using filename order or inventing image content.
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
                preference[index] =
                    ((base - candidate) / (base + candidate).max(1.0)) * 0.60 - 0.30;
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
