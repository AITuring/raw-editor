//! Focus score maps and per-pixel focus decisions.
//!
//! Scores sharpness at the analysis scale, smooths the decision and suppresses
//! isolated owner switches on canvas-like background.

use image::GrayImage;
use image::Rgb32FImage;
use rayon::prelude::*;

use super::{
    FOCUS_ANALYSIS_MAX_DIMENSION, FOCUS_ANALYSIS_MAX_PIXELS, FOCUS_CONFIDENCE_MARGIN,
    FOCUS_DECISION_COHERENCE_RADIUS_DIVISOR, FOCUS_DECISION_MAX_COHERENCE_RADIUS,
    FOCUS_DECISIVE_ADVANTAGE, FOCUS_EDGE_CLAIM_THRESHOLD, FOCUS_EDGE_PROTECTION_AT_1024,
    FOCUS_EDGE_PROTECTION_SCALE, FOCUS_SCORE_BLUR_RADIUS_DIVISOR, FOCUS_SCORE_MAX_BLUR_RADIUS,
    focus_stack_pixel_is_canvas_like, focus_stack_pixel_is_tone_foreground,
};

pub(super) fn focus_analysis_dimensions(
    width: u32,
    height: u32,
    source_longest_dimension: u32,
) -> (u32, u32) {
    let longest_side = width.max(height).max(1);
    // Keep the focus-analysis sampling density tied to an individual source,
    // not to the total mosaic length. Otherwise a long shifted stack compresses
    // each 9504px frame to a few hundred analysis pixels and erases the very
    // sharpness differences the stacker is supposed to select.
    let source_scale = FOCUS_ANALYSIS_MAX_DIMENSION as f64 / source_longest_dimension.max(1) as f64;
    let mut scale = source_scale.min(1.0);
    let scaled_pixels = width as f64 * height as f64 * scale * scale;
    if scaled_pixels > FOCUS_ANALYSIS_MAX_PIXELS as f64 {
        scale *= (FOCUS_ANALYSIS_MAX_PIXELS as f64 / scaled_pixels).sqrt();
    }
    if scale >= 1.0 && longest_side <= FOCUS_ANALYSIS_MAX_DIMENSION {
        return (width.max(1), height.max(1));
    }
    (
        (width as f64 * scale).round().max(1.0) as u32,
        (height as f64 * scale).round().max(1.0) as u32,
    )
}

pub(super) fn box_blur_focus_map(
    source: &[f32],
    width: u32,
    height: u32,
    radius: usize,
) -> Vec<f32> {
    if radius == 0 || width == 0 || height == 0 {
        return source.to_vec();
    }
    let width = width as usize;
    let height = height as usize;
    let window_size = radius * 2 + 1;
    let divisor = window_size as f32;
    let mut horizontal = vec![0.0f32; source.len()];
    horizontal
        .par_chunks_mut(width)
        .zip(source.par_chunks(width))
        .for_each(|(output_row, source_row)| {
            let mut sum = 0.0f32;
            for offset in 0..window_size {
                sum += source_row[offset.saturating_sub(radius).min(width - 1)];
            }
            output_row[0] = sum / divisor;
            for (x, output) in output_row.iter_mut().enumerate().skip(1) {
                let add_x = (x + radius).min(width - 1);
                let remove_x = x.saturating_sub(radius + 1);
                sum += source_row[add_x] - source_row[remove_x];
                *output = sum / divisor;
            }
        });

    let mut output = vec![0.0f32; source.len()];
    for x in 0..width {
        let mut sum = 0.0f32;
        for offset in 0..window_size {
            let y = offset.saturating_sub(radius).min(height - 1);
            sum += horizontal[y * width + x];
        }
        output[x] = sum / divisor;
        for y in 1..height {
            let add_y = (y + radius).min(height - 1);
            let remove_y = y.saturating_sub(radius + 1);
            sum += horizontal[add_y * width + x] - horizontal[remove_y * width + x];
            output[y * width + x] = sum / divisor;
        }
    }
    output
}

pub(super) fn focus_score_map(image: &Rgb32FImage, mask: &GrayImage) -> Vec<f32> {
    let (width, height) = image.dimensions();
    let pixel_count = width as usize * height as usize;
    let mut luminance_map = vec![0.0f32; pixel_count];
    luminance_map
        .par_iter_mut()
        .zip(image.as_raw().par_chunks(3))
        .for_each(|(output, pixel)| {
            *output = pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114;
        });

    let mut raw_score = vec![0.0f32; pixel_count];
    if width >= 3 && height >= 3 {
        raw_score
            .par_chunks_mut(width as usize)
            .enumerate()
            .for_each(|(y, row)| {
                if y == 0 || y + 1 >= height as usize {
                    return;
                }
                for (x, output) in row.iter_mut().enumerate().take(width as usize - 1).skip(1) {
                    let index = y * width as usize + x;
                    if mask.as_raw()[index] == 0
                        || mask.as_raw()[index - 1] == 0
                        || mask.as_raw()[index + 1] == 0
                        || mask.as_raw()[index - width as usize] == 0
                        || mask.as_raw()[index + width as usize] == 0
                    {
                        continue;
                    }
                    let laplacian = (4.0 * luminance_map[index]
                        - luminance_map[index - 1]
                        - luminance_map[index + 1]
                        - luminance_map[index - width as usize]
                        - luminance_map[index + width as usize])
                        .abs();
                    // A Laplacian alone under-scores broad, low-contrast
                    // brush strokes and woven paper texture.  Add a bounded
                    // Tenengrad term so a genuinely focused layer can win on
                    // those structures without making isolated sensor noise
                    // dominate the ownership map.
                    let gradient_x = luminance_map[index + 1] - luminance_map[index - 1];
                    let gradient_y = luminance_map[index + width as usize]
                        - luminance_map[index - width as usize];
                    let gradient = (gradient_x * gradient_x + gradient_y * gradient_y).sqrt();
                    *output = laplacian + gradient * 0.5;
                }
            });
    }

    // Focus ownership is decided on a bounded analysis canvas. Keep the score
    // window local enough that one clear stroke cannot claim an adjacent
    // character; the final pixels still come from the full-resolution aligned
    // layer.
    let radius = ((width.max(height) as f32 / FOCUS_SCORE_BLUR_RADIUS_DIVISOR).round() as usize)
        .clamp(1, FOCUS_SCORE_MAX_BLUR_RADIUS);
    box_blur_focus_map(&raw_score, width, height, radius)
}

pub(super) fn focus_decision_mask(
    base_focus: &[f32],
    candidate_focus: &[f32],
    base_mask: &GrayImage,
    candidate_mask: &GrayImage,
) -> GrayImage {
    let (width, height) = base_mask.dimensions();
    let pixel_count = width as usize * height as usize;
    let mut fine_advantage = vec![0.0f32; pixel_count];
    fine_advantage
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, output)| {
            let base_valid = base_mask.as_raw()[index] > 0;
            let candidate_valid = candidate_mask.as_raw()[index] > 0;
            *output = match (base_valid, candidate_valid) {
                (false, true) => 1.0,
                (true, false) => -1.0,
                (true, true) => {
                    let base = base_focus[index];
                    let candidate = candidate_focus[index];
                    (candidate - base) / (candidate + base + 1e-6)
                }
                (false, false) => 0.0,
            };
        });

    // The old radius was effectively ~100 source pixels for a 9504px frame.
    // That let a clear stroke in one region claim a neighbouring blurred
    // character. Keep the decision coherent, but make the neighbourhood local
    // enough that each character can select its own sharp source.
    let coherence_radius = ((width.max(height) as f32 / FOCUS_DECISION_COHERENCE_RADIUS_DIVISOR)
        .round() as usize)
        .clamp(2, FOCUS_DECISION_MAX_COHERENCE_RADIUS);
    let coarse_advantage = box_blur_focus_map(&fine_advantage, width, height, coherence_radius);

    let sample_step = (pixel_count / 4096).max(1);
    let mut magnitude_samples = (0..pixel_count)
        .step_by(sample_step)
        .filter_map(|index| {
            (base_mask.as_raw()[index] > 0 || candidate_mask.as_raw()[index] > 0)
                .then_some(base_focus[index].max(candidate_focus[index]))
        })
        .collect::<Vec<_>>();
    magnitude_samples.sort_unstable_by(f32::total_cmp);
    let confidence_floor = magnitude_samples
        .get(((magnitude_samples.len().saturating_sub(1)) as f32 * 0.9) as usize)
        .copied()
        .unwrap_or(0.0)
        * 0.04;

    // A defocused subject spills contrast beyond its true silhouette. That halo can
    // look locally sharper than the clean background in the focused layer, leaving
    // detached slivers around depth discontinuities even with hard pixel selection.
    // Let decisive, valid structure claim a small adjacent ambiguous band so the
    // focused silhouette also supplies the clean pixels immediately around it.
    let mut strong_base = vec![0.0f32; pixel_count];
    let mut strong_candidate = vec![0.0f32; pixel_count];
    strong_base
        .par_iter_mut()
        .zip(strong_candidate.par_iter_mut())
        .enumerate()
        .for_each(|(index, (base_claim, candidate_claim))| {
            let both_valid = base_mask.as_raw()[index] > 0 && candidate_mask.as_raw()[index] > 0;
            let confident = base_focus[index].max(candidate_focus[index]) > confidence_floor;
            if !both_valid || !confident {
                return;
            }
            let advantage = fine_advantage[index];
            if advantage < -FOCUS_DECISIVE_ADVANTAGE {
                *base_claim = 1.0;
            } else if advantage > FOCUS_DECISIVE_ADVANTAGE {
                *candidate_claim = 1.0;
            }
        });
    let protection_radius = ((width.max(height) as f32 / 2048.0)
        * (FOCUS_EDGE_PROTECTION_AT_1024 * FOCUS_EDGE_PROTECTION_SCALE))
        .round()
        .clamp(2.0, 6.0) as usize;
    let base_claim = box_blur_focus_map(&strong_base, width, height, protection_radius);
    let candidate_claim = box_blur_focus_map(&strong_candidate, width, height, protection_radius);

    GrayImage::from_fn(width, height, |x, y| {
        let index = y as usize * width as usize + x as usize;
        let base_valid = base_mask.as_raw()[index] > 0;
        let candidate_valid = candidate_mask.as_raw()[index] > 0;
        let confident = base_focus[index].max(candidate_focus[index]) > confidence_floor;
        // Always use the coherent neighborhood decision for ownership. A single
        // high-contrast stroke can be displaced by a couple of pixels between
        // captures; using its raw per-pixel score would then alternate sources
        // across the stroke and recreate a double contour.
        let advantage = coarse_advantage[index];
        let nearby_base_claim = base_claim[index];
        let nearby_candidate_claim = candidate_claim[index];
        let structural_winner = if nearby_candidate_claim > FOCUS_EDGE_CLAIM_THRESHOLD
            && nearby_candidate_claim > nearby_base_claim
        {
            Some(true)
        } else if nearby_base_claim > FOCUS_EDGE_CLAIM_THRESHOLD
            && nearby_base_claim > nearby_candidate_claim
        {
            Some(false)
        } else {
            None
        };
        let candidate_wins = candidate_valid
            && (!base_valid
                || structural_winner.unwrap_or(confident && advantage > FOCUS_CONFIDENCE_MARGIN));
        image::Luma([if candidate_wins { 255 } else { 0 }])
    })
}

pub(super) fn suppress_focus_canvas_switches(
    decision_mask: &mut GrayImage,
    base: &Rgb32FImage,
    candidate: &Rgb32FImage,
    base_mask: &GrayImage,
    candidate_mask: &GrayImage,
) {
    if decision_mask.dimensions() != base.dimensions()
        || base.dimensions() != candidate.dimensions()
        || base_mask.dimensions() != base.dimensions()
        || candidate_mask.dimensions() != base.dimensions()
    {
        return;
    }

    decision_mask
        .as_mut()
        .par_iter_mut()
        .zip(base.as_raw().par_chunks(3))
        .zip(candidate.as_raw().par_chunks(3))
        .zip(base_mask.as_raw().par_iter())
        .zip(candidate_mask.as_raw().par_iter())
        .for_each(
            |((((decision, base_pixel), candidate_pixel), base_valid), candidate_valid)| {
                // The canvas is one planar surface. Sharpness fluctuations in its
                // weave must not make the focus score swap the source for an
                // already-covered pixel; those swaps are the rectangular exposure
                // blocks visible in a shifted scan. A painted pixel can still win
                // normally, and a new candidate can still fill an uncovered hole.
                if *decision > 0
                    && *base_valid > 0
                    && *candidate_valid > 0
                    && focus_stack_pixel_is_canvas_like(base_pixel)
                    && focus_stack_pixel_is_canvas_like(candidate_pixel)
                    && !focus_stack_pixel_is_tone_foreground(base_pixel)
                    && !focus_stack_pixel_is_tone_foreground(candidate_pixel)
                {
                    *decision = 0;
                }
            },
        );
}
