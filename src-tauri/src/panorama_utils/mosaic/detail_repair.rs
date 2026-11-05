//! Detail repair of a composed focus scan.
//!
//! The seam compositor decides region by region which frame owns the canvas,
//! and it keeps a soft frame wherever a sharper one disagrees with it after a
//! low-pass: across handheld frames of wavy silk such a disagreement is often
//! a local displacement of tens of pixels, which the compositor must not
//! paste in. Once the canvas is final, this pass compares every cell of a fine
//! grid with the sharpest source measured there. Where the canvas falls short
//! it aligns that source densely to the canvas (see `dense_align`) and
//! transfers only its fine detail: the canvas keeps its own low frequencies,
//! so tone and stroke positions stay continuous and no seam can appear. A
//! cell is repaired only where the aligned source agrees with the canvas at
//! low frequency and carries clearly more detail.
//!
//! Cells the pass cannot bring close to their sharpest source are returned for
//! the soft-region report.

use super::super::stitching::Projection;
use super::super::stitching::transformed_image_region;
use crate::panorama_utils::shared::ImageInfo;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use super::{
    DetailGrid, Displacement, Field, FrameDetail, GRID_STEP, LayerSampler, Plane, align, luma,
    measure_image_detail, measure_image_tone, neighbourhood_reference, planar_canvas_scale,
    prefilter_minified_source, sample_detail,
};

/// Cells of the repair grid along the canvas long side (about 140 output
/// pixels on 祁序2 at full size), and the smallest cell, in canvas pixels.
pub(super) const REPAIR_CELLS_LONG_SIDE: f64 = 256.0;
pub(super) const REPAIR_MIN_CELL: f64 = 48.0;
/// A cell needs repair when the canvas holds less than this share of the
/// detail of its sharpest source...
pub(super) const REPAIR_DEFICIT_RATIO: f64 = 0.8;
/// ...and less than this share of the detail the canvas reaches around it
/// (see `neighbourhood_reference`). Frames light the silk weave differently,
/// so a frame may measure more detail than a canvas that looks just as
/// sharp; only a cell softer than its surroundings is a soft spot a viewer
/// sees.
const REPAIR_LOCAL_SHARE: f64 = 0.8;
/// Neighbours count as the same material when their mean luma differs by at
/// most this factor.
const LEVEL_SIMILARITY: f64 = 1.5;
/// The surroundings agree with an aligned donor when this share of their
/// structured cells correlates at least this well after the low-pass.
const RING_MIN_CORRELATION: f64 = 0.85;
const RING_MIN_SHARE: f64 = 0.75;
/// Surrounding cells within this many cells of a soft cell decide whether it
/// may be replaced.
const RING_REACH: i64 = 2;
/// Cells of context around a group of soft cells in a repair window.
const WINDOW_CONTEXT_CELLS: usize = 2;
/// A repaired cell must gain at least this factor of detail.
const REPAIR_MIN_GAIN: f64 = 1.15;
/// Donor margin around a repair window, in output pixels: the largest
/// displacement the alignment can find.
const ALIGN_MARGIN: usize = 96;
/// Largest side of one repair window, in output pixels.
const WINDOW_MAX_SIDE: usize = 1024;
/// Low-pass sigma (output pixels) under which an aligned donor must agree
/// with the canvas, and the correlation it needs there.
const VERIFY_SIGMA: f32 = 4.0;
const VERIFY_MIN_CORRELATION: f64 = 0.8;
/// Relative contrast under which a low-passed cell has no structure that
/// could be misaligned.
const VERIFY_FLAT_CONTRAST: f64 = 0.01;
/// Low-pass residual, relative to the level, under which an aligned donor
/// agrees with the canvas whatever their correlation: on plain silk the
/// correlation of what little structure is left measures noise and sheen
/// (祁序1 plain silk of 2-3 % contrast correlated 0.7-0.8 with a well
/// aligned 祁序2 frame, residual about 2 %), and structure this faint cannot
/// show a displaced stroke.
const VERIFY_MAX_RESIDUAL: f64 = 0.025;
/// Frames tried per deficit cell, sharpest first. A cell the sharpest frame
/// cannot repair (it does not align there, or disagrees with the canvas)
/// tries the next frame that is still clearly sharper than the canvas.
const REPAIR_CANDIDATES: usize = 3;
/// Sigma (output pixels) splitting transferred detail from kept tone.
const DETAIL_SPLIT_SIGMA: f32 = 3.0;
/// Sigma (output pixels, measured at quarter resolution) of the gain that
/// scales donor detail to the canvas level.
const GAIN_SIGMA: f32 = 24.0;
const GAIN_LIMITS: (f32, f32) = (0.67, 1.5);
/// Feather (output pixels) of the repair weight around accepted cells.
const FEATHER_SIGMA: f32 = 10.0;
/// Largest change of the alignment per output pixel in and around a
/// repaired cell. Beyond it the donor's weave is bent or stretched, not
/// moved: on soft canvas with little structure a few wrong matches drew
/// squiggles into 祁序1 (DSC_6768 at 6800,5750) through fields bending 1 to
/// 2 px per px, while fields that hold measure about 0.13 (0.1 turned away
/// two thirds of the good repairs).
const MAX_FIELD_STRAIN: f64 = 0.35;

/// What happened to one deficit cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CellOutcome {
    /// Detail transferred from the donor.
    Repaired,
    /// The canvas replaced by the donor, aligned to the surroundings.
    Replaced,
    /// The donor could not be aligned to the canvas here.
    Unaligned,
    /// The aligned donor disagrees with the canvas at low frequency.
    Disagrees,
    /// The aligned donor carries too little extra detail.
    NoGain,
}

/// One cell the pass looked at.
#[derive(Clone, Copy, Debug)]
pub(super) struct RepairCell {
    pub(super) cell: usize,
    pub(super) donor: usize,
    pub(super) best: f64,
    /// Detail of the canvas before the pass (diagnostics).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) before: f64,
    pub(super) after: f64,
    pub(super) outcome: CellOutcome,
    /// Largest strain of the donor's alignment in and around the cell
    /// (diagnostics).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) strain: f64,
}

/// The run's geometry, shared by every donor.
pub(super) struct RepairScene<'a> {
    pub(super) images: &'a [&'a ImageInfo],
    /// Ids of the frames of other captures than the composing one.
    pub(super) donor_frames: &'a HashSet<usize>,
    pub(super) homographies: &'a HashMap<usize, Matrix3<f64>>,
    pub(super) projection: Projection,
    pub(super) offset: (f64, f64),
    pub(super) canvas: (u32, u32),
}

/// Summary of the pass.
#[derive(Debug, Default)]
pub(super) struct RepairOutcome {
    pub(super) measured_cells: usize,
    pub(super) cells: Vec<RepairCell>,
    /// Detail of the canvas per grid cell after the pass, `NaN` where not
    /// measured.
    pub(super) detail_after: Vec<f64>,
    /// Detail of the sharpest source per grid cell and its frame.
    pub(super) best: Vec<(f64, usize)>,
    /// Mean luma of the canvas per grid cell, `NaN` where not measured.
    pub(super) level: Vec<f64>,
    /// Share of the canvas clipped to white per grid cell, `NaN` where not
    /// measured.
    pub(super) clipped: Vec<f64>,
}

/// Connected groups of cells, 8-connected on the grid.
fn components(cells: &[usize], columns: usize) -> Vec<Vec<usize>> {
    let set = cells.iter().copied().collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    let mut groups = Vec::new();
    for &start in cells {
        if !seen.insert(start) {
            continue;
        }
        let mut group = Vec::new();
        let mut pending = VecDeque::from([start]);
        while let Some(cell) = pending.pop_front() {
            group.push(cell);
            let (x, y) = ((cell % columns) as i64, (cell / columns) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= columns as i64 {
                        continue;
                    }
                    let neighbour = ny as usize * columns + nx as usize;
                    if set.contains(&neighbour) && seen.insert(neighbour) {
                        pending.push_back(neighbour);
                    }
                }
            }
        }
        group.sort_unstable();
        groups.push(group);
    }
    groups
}

/// Pearson correlation of two low-passed planes over a window: `None` when
/// the window is flat in both, `1` when `b`, scaled to the level of `a`,
/// matches `a` within `VERIFY_MAX_RESIDUAL`.
fn window_correlation(
    a: &Plane,
    b: &Plane,
    (x0, y0, x1, y1): (usize, usize, usize, usize),
) -> Option<f64> {
    let (mut pairs, mut sum_a, mut sum_b) = (Vec::new(), 0.0, 0.0);
    for y in y0..y1 {
        for x in x0..x1 {
            let (va, vb) = (a.values[y * a.width + x], b.values[y * b.width + x]);
            if va.is_finite() && vb.is_finite() {
                pairs.push((f64::from(va), f64::from(vb)));
                sum_a += f64::from(va);
                sum_b += f64::from(vb);
            }
        }
    }
    if pairs.len() < 16 {
        return Some(0.0);
    }
    let count = pairs.len() as f64;
    let (mean_a, mean_b) = (sum_a / count, sum_b / count);
    let (mut var_a, mut var_b, mut covariance) = (0.0, 0.0, 0.0);
    for (va, vb) in &pairs {
        let (da, db) = (va - mean_a, vb - mean_b);
        var_a += da * da;
        var_b += db * db;
        covariance += da * db;
    }
    let contrast = |variance: f64, mean: f64| (variance / count).sqrt() / mean.abs().max(0.04);
    if contrast(var_a, mean_a) < VERIFY_FLAT_CONTRAST
        && contrast(var_b, mean_b) < VERIFY_FLAT_CONTRAST
    {
        return None;
    }
    let gain = if mean_b.abs() > 1e-6 {
        mean_a / mean_b
    } else {
        1.0
    };
    let residual = pairs
        .iter()
        .map(|(va, vb)| (va - gain * vb).powi(2))
        .sum::<f64>()
        / count;
    if residual.sqrt() / mean_a.abs().max(0.04) < VERIFY_MAX_RESIDUAL {
        return Some(1.0);
    }
    Some(covariance / (var_a * var_b).sqrt().max(1e-12))
}

/// Quarter-resolution Gaussian of a full-resolution plane, sampled back at
/// full resolution (bilinear).
fn broad_blur(plane: &Plane, sigma: f32) -> Plane {
    let quarter = {
        let mut half = Plane::new(plane.width / 2, plane.height / 2);
        for y in 0..half.height {
            for x in 0..half.width {
                let mut sum = 0.0;
                let mut count = 0;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let value = plane.values[(2 * y + dy) * plane.width + 2 * x + dx];
                    if value.is_finite() {
                        sum += value;
                        count += 1;
                    }
                }
                if count > 0 {
                    half.values[y * half.width + x] = sum / count as f32;
                }
            }
        }
        let mut quarter = Plane::new(half.width / 2, half.height / 2);
        for y in 0..quarter.height {
            for x in 0..quarter.width {
                let mut sum = 0.0;
                let mut count = 0;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let value = half.values[(2 * y + dy) * half.width + 2 * x + dx];
                    if value.is_finite() {
                        sum += value;
                        count += 1;
                    }
                }
                if count > 0 {
                    quarter.values[y * quarter.width + x] = sum / count as f32;
                }
            }
        }
        quarter.blurred(sigma / 4.0)
    };
    let mut full = Plane::new(plane.width, plane.height);
    if quarter.width == 0 || quarter.height == 0 {
        return plane.clone();
    }
    for y in 0..plane.height {
        let gy = ((y as f32 + 0.5) / 4.0 - 0.5).clamp(0.0, (quarter.height - 1) as f32);
        let (y0, fy) = (gy as usize, gy.fract());
        let y1 = (y0 + 1).min(quarter.height - 1);
        for x in 0..plane.width {
            let gx = ((x as f32 + 0.5) / 4.0 - 0.5).clamp(0.0, (quarter.width - 1) as f32);
            let (x0, fx) = (gx as usize, gx.fract());
            let x1 = (x0 + 1).min(quarter.width - 1);
            let at = |px: usize, py: usize| quarter.values[py * quarter.width + px];
            let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
            let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
            full.values[y * plane.width + x] = top * (1.0 - fy) + bottom * fy;
        }
    }
    full
}

/// Repairs one window of the canvas from one donor. `window` is in canvas
/// coordinates, `cells` the cells of this donor to repair inside it and
/// `soft` every cell of the run that needs repair (any donor).
///
/// The donor is aligned to the canvas around the soft cells only, so the
/// displacement inside them continues that of their surroundings. A soft cell
/// whose canvas agrees with the aligned donor at low frequency (the canvas
/// is defocused where it should be) receives the donor's fine detail on top
/// of its own tone. One whose canvas disagrees (a soft frame displaced from
/// its neighbours holds it) is replaced by the donor wherever the donor agrees
/// with the surrounding canvas, with the tone the surroundings set. A cell
/// whose alignment bends the donor (see `MAX_FIELD_STRAIN`) is left alone.
/// Returns the outcome of every cell and the strain of its alignment.
#[allow(clippy::too_many_arguments)]
fn repair_window(
    output: &mut Rgb32FImage,
    crop: (u32, u32),
    sampler: &LayerSampler<'_>,
    window: (usize, usize, usize, usize),
    cells: &[usize],
    soft: &HashSet<usize>,
    grid: &DetailGrid,
) -> Vec<(usize, CellOutcome, f64)> {
    let (x0, y0, x1, y1) = window;
    let (width, height) = (x1 - x0, y1 - y0);
    let mut canvas = Vec::with_capacity(width * height);
    for y in y0..y1 {
        for x in x0..x1 {
            canvas.push(
                *output.get_pixel((x - crop.0 as usize) as u32, (y - crop.1 as usize) as u32),
            );
        }
    }
    let cell_size = grid.cell_size();
    let (columns, _) = grid.cell_dimensions();
    // Window rectangle of a grid cell, if it reaches into the window.
    let cell_rect = |cell: usize| {
        let (column, row) = (cell % columns, cell / columns);
        let cx0 = ((column as f64 * cell_size) as usize).clamp(x0, x1);
        let cy0 = ((row as f64 * cell_size) as usize).clamp(y0, y1);
        let cx1 = (((column + 1) as f64 * cell_size) as usize).clamp(x0, x1);
        let cy1 = (((row + 1) as f64 * cell_size) as usize).clamp(y0, y1);
        (cx1 > cx0 && cy1 > cy0).then_some((cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0))
    };
    let mut base = Plane::new(width, height);
    for (value, pixel) in base.values.iter_mut().zip(&canvas) {
        *value = luma(*pixel) as f32;
    }
    // The surroundings the donor is aligned to: the window without its soft
    // cells.
    let mut inside_soft = vec![false; width * height];
    let first_column = (x0 as f64 / cell_size) as usize;
    let last_column = ((x1 as f64 / cell_size) as usize).min(columns - 1);
    let first_row = (y0 as f64 / cell_size) as usize;
    let last_row = (y1 as f64 / cell_size) as usize;
    for row in first_row..=last_row {
        for column in first_column..=last_column {
            let cell = row * columns + column;
            if !soft.contains(&cell) {
                continue;
            }
            if let Some((rx0, ry0, rx1, ry1)) = cell_rect(cell) {
                for y in ry0..ry1 {
                    inside_soft[y * width + rx0..y * width + rx1].fill(true);
                }
            }
        }
    }
    let mut surroundings = base.clone();
    for (value, &soft) in surroundings.values.iter_mut().zip(&inside_soft) {
        if soft {
            *value = f32::NAN;
        }
    }
    // Layer-local coordinates of a canvas position.
    let local = |cx: f64, cy: f64| (cx - f64::from(sampler.left), cy - f64::from(sampler.top));
    let margin = ALIGN_MARGIN;
    let (moving_width, moving_height) = (width + 2 * margin, height + 2 * margin);
    let mut moving = Plane::new(moving_width, moving_height);
    moving
        .values
        .par_chunks_mut(moving_width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, value) in row.iter_mut().enumerate() {
                let (lx, ly) = local(
                    x0 as f64 + x as f64 - margin as f64,
                    y0 as f64 + y as f64 - margin as f64,
                );
                if let Some(pixel) = sampler.sample(lx, ly) {
                    *value = luma(pixel) as f32;
                }
            }
        });
    // Aligned to the surroundings where they carry enough structure, else to
    // the whole window, else the global pose; the checks below decide what
    // each alignment may change.
    let aligned = align(&surroundings, &moving, margin).or_else(|| align(&base, &moving, margin));
    let unaligned = aligned.is_none();
    let displacement = aligned.unwrap_or_else(Displacement::zero);
    let base_low = base.blurred(VERIFY_SIGMA);
    let in_window =
        |cx: f64, cy: f64| cx >= x0 as f64 && cy >= y0 as f64 && cx < x1 as f64 && cy < y1 as f64;
    let feather = (2.0 * FEATHER_SIGMA) as usize;
    // The donor resampled through `displacement`, and what every cell may
    // take from it.
    let evaluate = |displacement: &Displacement| {
        let mut donor = vec![None; width * height];
        donor
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, value) in row.iter_mut().enumerate() {
                    let [dx, dy] = displacement.at(x as f64, y as f64);
                    let (lx, ly) = local(x0 as f64 + x as f64 + dx, y0 as f64 + y as f64 + dy);
                    *value = sampler.sample(lx, ly);
                }
            });
        let mut donor_luma = Plane::new(width, height);
        for (index, pixel) in donor.iter().enumerate() {
            if let Some(pixel) = pixel {
                donor_luma.values[index] = luma(*pixel) as f32;
            }
        }
        let donor_low = donor_luma.blurred(VERIFY_SIGMA);
        // The surroundings next to a soft cell must agree with the aligned
        // donor before that cell may take the donor's own low frequencies:
        // they are what its replacement has to continue. A flat neighbour
        // has no structure that could break.
        let ring = (first_row..=last_row)
            .flat_map(|row| (first_column..=last_column).map(move |column| row * columns + column))
            .filter(|cell| !soft.contains(cell))
            .filter_map(|cell| {
                cell_rect(cell).map(|rect| (cell, window_correlation(&base_low, &donor_low, rect)))
            })
            .collect::<Vec<_>>();
        let ring_agrees_near = |cell: usize| {
            let (column, row) = ((cell % columns) as i64, (cell / columns) as i64);
            let near = ring
                .iter()
                .filter(|(neighbour, _)| {
                    let (nx, ny) = ((neighbour % columns) as i64, (neighbour / columns) as i64);
                    (nx - column).abs() <= RING_REACH && (ny - row).abs() <= RING_REACH
                })
                .filter_map(|(_, correlation)| *correlation)
                .collect::<Vec<_>>();
            near.iter()
                .filter(|&&correlation| correlation >= RING_MIN_CORRELATION)
                .count() as f64
                >= RING_MIN_SHARE * near.len() as f64
        };
        let mut evaluation = Evaluation {
            outcomes: Vec::with_capacity(cells.len()),
            transfer: Vec::new(),
            replace: Vec::new(),
            donor: Vec::new(),
            donor_luma: Plane::new(0, 0),
        };
        for &cell in cells {
            let Some(rect) = cell_rect(cell) else {
                continue;
            };
            let around = (
                rect.0.saturating_sub(feather),
                rect.1.saturating_sub(feather),
                (rect.2 + feather).min(width - 1),
                (rect.3 + feather).min(height - 1),
            );
            let strain = displacement.max_strain(around);
            let centre = grid.cell_centre(cell);
            let donor_detail = sample_detail(
                |cx, cy| {
                    in_window(cx, cy)
                        .then(|| donor[(cy as usize - y0) * width + (cx as usize - x0)])
                        .flatten()
                },
                centre,
                grid,
            );
            let canvas_detail = sample_detail(
                |cx, cy| {
                    in_window(cx, cy)
                        .then(|| canvas[(cy as usize - y0) * width + (cx as usize - x0)])
                },
                centre,
                grid,
            );
            let gains = matches!(
                (donor_detail, canvas_detail),
                (Some(donor), Some(canvas)) if donor >= REPAIR_MIN_GAIN * canvas
            );
            let inner = window_correlation(&base_low, &donor_low, rect);
            let outcome = if strain > MAX_FIELD_STRAIN {
                CellOutcome::Unaligned
            } else if !gains {
                CellOutcome::NoGain
            } else if inner.is_none_or(|correlation| correlation >= VERIFY_MIN_CORRELATION) {
                evaluation.transfer.push(rect);
                CellOutcome::Repaired
            } else if ring_agrees_near(cell) {
                evaluation.replace.push(rect);
                CellOutcome::Replaced
            } else if unaligned {
                CellOutcome::Unaligned
            } else {
                CellOutcome::Disagrees
            };
            evaluation.outcomes.push((cell, outcome, strain));
        }
        evaluation.donor = donor;
        evaluation.donor_luma = donor_luma;
        evaluation
    };
    let mut evaluation = evaluate(&displacement);
    // Where the dense field bends the donor, its affine fit cannot, and may
    // still carry the donor close enough for the checks to pass: on weakly
    // structured silk a few wrong matches fold the field while the donor
    // as a whole is placed right. Only cells the dense field leaves
    // unrepaired take the affine donor (with their feather); a repaired cell
    // keeps the dense one.
    let bent = evaluation.outcomes.iter().any(|&(_, outcome, strain)| {
        outcome == CellOutcome::Unaligned && strain > MAX_FIELD_STRAIN
    });
    if !unaligned
        && bent
        && let Some(smooth) = displacement.affine_fit()
    {
        let alternative = evaluate(&smooth);
        let taken =
            |outcome: CellOutcome| matches!(outcome, CellOutcome::Repaired | CellOutcome::Replaced);
        let mut from_alternative = vec![false; width * height];
        let mut kept_dense = vec![false; width * height];
        let mut merged = Evaluation {
            outcomes: Vec::with_capacity(evaluation.outcomes.len()),
            transfer: Vec::new(),
            replace: Vec::new(),
            donor: Vec::new(),
            donor_luma: Plane::new(0, 0),
        };
        for (&dense, &smooth) in evaluation.outcomes.iter().zip(&alternative.outcomes) {
            let use_smooth = !taken(dense.1) && taken(smooth.1);
            let chosen = if use_smooth { smooth } else { dense };
            let Some(rect) = cell_rect(chosen.0) else {
                continue;
            };
            match chosen.1 {
                CellOutcome::Repaired => merged.transfer.push(rect),
                CellOutcome::Replaced => merged.replace.push(rect),
                _ => {}
            }
            if use_smooth {
                let (fx1, fy1) = (
                    (rect.2 + feather).min(width),
                    (rect.3 + feather).min(height),
                );
                for y in rect.1.saturating_sub(feather)..fy1 {
                    from_alternative[y * width + rect.0.saturating_sub(feather)..y * width + fx1]
                        .fill(true);
                }
            } else if taken(dense.1) {
                for y in rect.1..rect.3 {
                    kept_dense[y * width + rect.0..y * width + rect.2].fill(true);
                }
            }
            merged.outcomes.push(chosen);
        }
        if merged.repaired() > evaluation.repaired() {
            let mut donor = std::mem::take(&mut evaluation.donor);
            let mut donor_luma = std::mem::replace(&mut evaluation.donor_luma, Plane::new(0, 0));
            for index in 0..width * height {
                if from_alternative[index] && !kept_dense[index] {
                    donor[index] = alternative.donor[index];
                    donor_luma.values[index] = alternative.donor_luma.values[index];
                }
            }
            merged.donor = donor;
            merged.donor_luma = donor_luma;
            evaluation = merged;
        }
    }
    #[cfg(test)]
    super::diagnostics::write_repair_probe(
        window,
        sampler.info,
        &base,
        &moving,
        margin,
        &evaluation.donor_luma,
        &displacement,
    );
    let Evaluation {
        outcomes,
        transfer,
        replace,
        donor,
        donor_luma,
    } = evaluation;
    if transfer.is_empty() && replace.is_empty() {
        return outcomes;
    }
    let weight_of = |rects: &[(usize, usize, usize, usize)]| {
        let mut weight = Plane::new(width, height);
        weight.values.fill(0.0);
        for &(ax0, ay0, ax1, ay1) in rects {
            for y in ay0..ay1 {
                weight.values[y * width + ax0..y * width + ax1].fill(1.0);
            }
        }
        weight.blurred(FEATHER_SIGMA)
    };
    let transfer_weight = weight_of(&transfer);
    let replace_weight = weight_of(&replace);
    // Detail gain: the canvas level over the donor level, broadly.
    let gain = {
        let canvas_level = broad_blur(&base, GAIN_SIGMA);
        let donor_level = broad_blur(&donor_luma, GAIN_SIGMA);
        canvas_level
            .values
            .iter()
            .zip(&donor_level.values)
            .map(|(canvas, donor)| {
                if canvas.is_finite() && donor.is_finite() && *donor > 1e-4 {
                    (canvas / donor).clamp(GAIN_LIMITS.0, GAIN_LIMITS.1)
                } else {
                    1.0
                }
            })
            .collect::<Vec<_>>()
    };
    // Replacement tone: per channel, the median ratio of canvas to donor over
    // the surroundings, so a replaced cell keeps the level around it.
    let channel_gain = |channel: usize| {
        let mut ratios = canvas
            .iter()
            .zip(&donor)
            .zip(&inside_soft)
            .filter(|(_, soft)| !**soft)
            .filter_map(|((canvas, donor), _)| {
                donor.and_then(|donor| {
                    (donor[channel] > 1e-3).then(|| f64::from(canvas[channel] / donor[channel]))
                })
            })
            .collect::<Vec<_>>();
        if ratios.len() < 256 {
            return 1.0f32;
        }
        ratios.sort_by(f64::total_cmp);
        (ratios[ratios.len() / 2] as f32).clamp(GAIN_LIMITS.0, GAIN_LIMITS.1)
    };
    for channel in 0..3 {
        let replace_gain = channel_gain(channel);
        let mut canvas_channel = Plane::new(width, height);
        let mut donor_channel = Plane::new(width, height);
        for index in 0..width * height {
            let value = canvas[index][channel];
            canvas_channel.values[index] = value;
            // Outside the donor the canvas stands in, so its detail is
            // replaced by itself.
            donor_channel.values[index] = donor[index].map_or(value, |pixel| pixel[channel]);
        }
        let canvas_low = canvas_channel.blurred(DETAIL_SPLIT_SIGMA);
        let donor_low = donor_channel.blurred(DETAIL_SPLIT_SIGMA);
        for index in 0..width * height {
            if donor[index].is_none() {
                continue;
            }
            let (transfer, replace) = (transfer_weight.values[index], replace_weight.values[index]);
            if !(transfer > 1e-3 || replace > 1e-3) {
                continue;
            }
            let canvas_value = canvas_channel.values[index];
            let canvas_detail = canvas_value - canvas_low.values[index];
            let donor_detail = donor_channel.values[index] - donor_low.values[index];
            if !(canvas_detail.is_finite() && donor_detail.is_finite()) {
                continue;
            }
            let transferred = canvas_value + gain[index] * donor_detail - canvas_detail;
            let replaced = donor_channel.values[index] * replace_gain;
            let replace = replace.min(1.0);
            let transfer = transfer.min(1.0 - replace);
            let value = canvas_value * (1.0 - transfer - replace)
                + transferred * transfer
                + replaced * replace;
            let (x, y) = (index % width, index / width);
            let pixel = output.get_pixel_mut(
                (x0 + x - crop.0 as usize) as u32,
                (y0 + y - crop.1 as usize) as u32,
            );
            pixel[channel] = value.max(0.0);
        }
    }
    outcomes
}

/// A canvas rectangle `(x0, y0, x1, y1)`, end-exclusive.
type Rect = (usize, usize, usize, usize);

/// One donor alignment tried on a repair window: the resampled donor and
/// what every cell may take from it (window rectangles of the cells whose
/// detail is transferred and of those replaced).
struct Evaluation {
    outcomes: Vec<(usize, CellOutcome, f64)>,
    transfer: Vec<Rect>,
    replace: Vec<Rect>,
    donor: Vec<Option<image::Rgb<f32>>>,
    donor_luma: Plane,
}

impl Evaluation {
    fn repaired(&self) -> usize {
        self.transfer.len() + self.replace.len()
    }
}

/// Windows of about `WINDOW_MAX_SIDE` canvas pixels at most that cover
/// `group`, each with the cells it repairs. A window holds its cells with
/// `WINDOW_CONTEXT_CELLS` of context on every side (inside `bounds`), so the
/// patch every cell is measured on lies in its window.
fn group_windows(
    group: &[usize],
    columns: usize,
    cell_size: f64,
    bounds: Rect,
) -> Vec<(Rect, Vec<usize>)> {
    let span = ((WINDOW_MAX_SIDE as f64 / cell_size) as usize)
        .saturating_sub(2 * WINDOW_CONTEXT_CELLS)
        .max(1);
    let mut tiles = HashMap::<(usize, usize), Vec<usize>>::new();
    for &cell in group {
        tiles
            .entry(((cell % columns) / span, (cell / columns) / span))
            .or_default()
            .push(cell);
    }
    let to_px = |cells: usize| (cells as f64 * cell_size) as usize;
    let mut windows = tiles
        .into_values()
        .filter_map(|cells| {
            let (mut cx0, mut cy0, mut cx1, mut cy1) = (usize::MAX, usize::MAX, 0, 0);
            for &cell in &cells {
                cx0 = cx0.min(cell % columns);
                cx1 = cx1.max(cell % columns + 1);
                cy0 = cy0.min(cell / columns);
                cy1 = cy1.max(cell / columns + 1);
            }
            let x0 = to_px(cx0.saturating_sub(WINDOW_CONTEXT_CELLS)).max(bounds.0);
            let y0 = to_px(cy0.saturating_sub(WINDOW_CONTEXT_CELLS)).max(bounds.1);
            let x1 = to_px(cx1 + WINDOW_CONTEXT_CELLS).min(bounds.2);
            let y1 = to_px(cy1 + WINDOW_CONTEXT_CELLS).min(bounds.3);
            (x1 > x0 + 32 && y1 > y0 + 32).then_some(((x0, y0, x1, y1), cells))
        })
        .collect::<Vec<_>>();
    windows.sort_by_key(|&((x0, y0, _, _), _)| (y0, x0));
    windows
}

/// Repairs `output` (canvas pixel (crop.0, crop.1) at its origin) from the
/// sharpest measured sources of every deficit cell, trying up to
/// `REPAIR_CANDIDATES` frames per cell. `details[i]` is frame `i`'s detail
/// on `grid`; `load` decodes a frame.
pub(super) fn repair_detail<F>(
    output: &mut Rgb32FImage,
    crop: (u32, u32),
    scene: &RepairScene<'_>,
    grid: &DetailGrid,
    details: &[FrameDetail],
    load: &mut F,
    mut progress: impl FnMut(usize, usize),
) -> Result<RepairOutcome, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let count = grid.cell_count();
    let (columns, _) = grid.cell_dimensions();
    let mut best = vec![(0.0f64, usize::MAX); count];
    for (frame, detail) in details.iter().enumerate() {
        for &(cell, value) in &detail.cells {
            if value > best[cell].0 {
                best[cell] = (value, frame);
            }
        }
    }
    let mut before = vec![f64::NAN; count];
    for (cell, value) in measure_image_detail(output, crop, grid) {
        before[cell] = value;
    }
    let measured_cells = before.iter().filter(|value| value.is_finite()).count();
    let (mut level, mut clipped) = (vec![f64::NAN; count], vec![f64::NAN; count]);
    for (cell, mean, clipped_share) in measure_image_tone(output, crop, grid) {
        level[cell] = mean;
        clipped[cell] = clipped_share;
    }
    let surroundings = neighbourhood_reference(&before, &level, grid, LEVEL_SIMILARITY);
    // Frames clearly sharper than the canvas in every deficit cell: first
    // those of the composing capture that are as sharp as the canvas around
    // the cell (they match its weave and light), then the rest, sharpest
    // first. Another capture lends its detail only where the composing one
    // has none sharp enough.
    let mut candidates = (0..count)
        .filter(|&cell| {
            best[cell].1 != usize::MAX
                && before[cell].is_finite()
                && surroundings[cell].is_finite()
                && before[cell] < REPAIR_DEFICIT_RATIO * best[cell].0
                && before[cell] < REPAIR_LOCAL_SHARE * surroundings[cell]
        })
        .map(|cell| (cell, Vec::new()))
        .collect::<HashMap<usize, Vec<(f64, usize)>>>();
    for (frame, detail) in details.iter().enumerate() {
        for &(cell, value) in &detail.cells {
            if let Some(list) = candidates.get_mut(&cell)
                && before[cell] < REPAIR_DEFICIT_RATIO * value
            {
                list.push((value, frame));
            }
        }
    }
    for (&cell, list) in &mut candidates {
        let tier = |&(value, frame): &(f64, usize)| {
            let composing = !scene.donor_frames.contains(&scene.images[frame].id);
            usize::from(!(composing && value >= REPAIR_LOCAL_SHARE * surroundings[cell]))
        };
        list.sort_by(|a, b| {
            tier(a)
                .cmp(&tier(b))
                .then(b.0.total_cmp(&a.0))
                .then(a.1.cmp(&b.1))
        });
        list.truncate(REPAIR_CANDIDATES);
    }
    let mut outcomes = HashMap::<usize, (usize, CellOutcome, f64)>::new();
    let cell_size = grid.cell_size();
    let (crop_x1, crop_y1) = (
        crop.0 as usize + output.width() as usize,
        crop.1 as usize + output.height() as usize,
    );
    let mut pending = candidates.keys().copied().collect::<HashSet<_>>();
    let mut steps = (0, 0);
    for round in 0..REPAIR_CANDIDATES {
        let mut by_donor = HashMap::<usize, Vec<usize>>::new();
        for &cell in &pending {
            if let Some(&(_, frame)) = candidates[&cell].get(round) {
                by_donor.entry(frame).or_default().push(cell);
            }
        }
        let mut donors = by_donor.into_iter().collect::<Vec<_>>();
        for (_, cells) in &mut donors {
            cells.sort_unstable();
        }
        donors.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
        steps.1 += donors.len();
        // What is still soft at the start of the round; the donors align to
        // the canvas around it.
        let soft = pending.clone();
        for (frame, cells) in &donors {
            steps.0 += 1;
            progress(steps.0, steps.1);
            let info = scene.images[*frame];
            let transform = &scene.homographies[&info.id];
            let Some((left, right, top, bottom)) = transformed_image_region(
                info,
                transform,
                scene.projection,
                scene.offset.0,
                scene.offset.1,
                scene.canvas.0,
                scene.canvas.1,
            ) else {
                continue;
            };
            let (source, source_divisor) =
                prefilter_minified_source(load(info)?, planar_canvas_scale(info, transform));
            let sampler = LayerSampler {
                info,
                source: &source,
                source_divisor,
                inverse: transform
                    .try_inverse()
                    .ok_or("The stack contains a non-invertible alignment.")?,
                native_mapping: None,
                projection: scene.projection,
                offset: scene.offset,
                left,
                top,
                scale: 1.0,
                residual: Field::new(1, 1, GRID_STEP as f64),
            };
            // Windows inside both the donor and the output.
            let bounds = (
                (crop.0 as usize).max(left as usize),
                (crop.1 as usize).max(top as usize),
                crop_x1.min(right as usize + 1),
                crop_y1.min(bottom as usize + 1),
            );
            for group in components(cells, columns) {
                for (window, inside) in group_windows(&group, columns, cell_size, bounds) {
                    for (cell, outcome, strain) in
                        repair_window(output, crop, &sampler, window, &inside, &soft, grid)
                    {
                        outcomes.insert(cell, (*frame, outcome, strain));
                    }
                }
            }
        }
        pending.retain(|cell| {
            !matches!(
                outcomes.get(cell),
                Some((_, CellOutcome::Repaired | CellOutcome::Replaced, _))
            )
        });
        if pending.is_empty() {
            break;
        }
    }
    let mut detail_after = before.clone();
    let looked_at = outcomes.keys().copied().collect::<HashSet<_>>();
    if !looked_at.is_empty() {
        for (cell, value) in measure_image_detail(output, crop, grid) {
            if looked_at.contains(&cell) {
                detail_after[cell] = value;
            }
        }
    }
    let mut cells = outcomes
        .into_iter()
        .map(|(cell, (donor, outcome, strain))| RepairCell {
            cell,
            donor,
            best: best[cell].0,
            before: before[cell],
            after: detail_after[cell],
            outcome,
            strain,
        })
        .collect::<Vec<_>>();
    cells.sort_by_key(|cell| cell.cell);
    Ok(RepairOutcome {
        measured_cells,
        cells,
        detail_after,
        best,
        level,
        clipped,
    })
}

/// One line for the run log: how many cells needed repair and what became of
/// them.
pub(super) fn repair_summary(outcome: &RepairOutcome) -> String {
    let count = |wanted: CellOutcome| {
        outcome
            .cells
            .iter()
            .filter(|cell| cell.outcome == wanted)
            .count()
    };
    let still_short = outcome
        .cells
        .iter()
        .filter(|cell| cell.after.is_finite() && cell.after < REPAIR_DEFICIT_RATIO * cell.best)
        .count();
    let donors = outcome
        .cells
        .iter()
        .map(|cell| cell.donor)
        .collect::<HashSet<_>>()
        .len();
    format!(
        "Detail repair: {} of {} cells softer than their surroundings and below {:.0}% of their \
         sharpest source, from {} donor(s); detail transferred {}, replaced {}, unaligned {}, \
         disagreeing {}, no gain {}; still below after repair {}",
        outcome.cells.len(),
        outcome.measured_cells,
        REPAIR_DEFICIT_RATIO * 100.0,
        donors,
        count(CellOutcome::Repaired),
        count(CellOutcome::Replaced),
        count(CellOutcome::Unaligned),
        count(CellOutcome::Disagrees),
        count(CellOutcome::NoGain),
        still_short,
    )
}
