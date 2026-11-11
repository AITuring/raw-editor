//! Mostly soft frames of a focus scan.
//!
//! A handheld focus scan photographs every region two or three times, and some
//! frames are out of focus over much of their area: a 120 mm frame shaken at
//! 1/25 s, or a focal plane that missed the wavy silk. Ownership replaces such
//! a frame only where a later layer is clearly sharper, so the frame that
//! covered a region first kept it otherwise, and 祁序1 at full size still
//! showed frame-sized soft blocks. Before composing, every frame's detail is
//! measured at output resolution, in the run's [`FocusBand`], on a coarse
//! canvas grid and compared with the sharpest frame in each cell; a frame soft
//! in at least [`SOFT_FRAME_SHARE`] of its cells is left out, unless leaving it
//! out would uncover a part of the canvas or leave a cell without a frame near
//! the sharpest one there.
//!
//! Frames are only skipped by the compositor. Grouping and registration still
//! see every input, so leaving frames out of the selection itself cannot split
//! the station graph (祁序1 without 16 frames fell apart into three
//! components).

use image::Rgb;
use rayon::prelude::*;

use super::{FocusBand, LayerSampler, luma, weave_band};

/// A frame is soft in a cell when its detail there is below this share of the
/// sharpest frame measured in the same cell.
pub(super) const SOFT_FRAME_RATIO: f64 = 0.6;
/// A frame is near the best in a cell when its detail there is at least this
/// share of the sharpest frame measured in the same cell. A frame that is the
/// last kept one near the best in any cell is never left out: on 祁序2 in the
/// silk-weave band, frames soft over half their area were still the sharpest
/// of a whole region, and dropping them handed it to a frame 0.6 as sharp
/// (DSC_6758, DSC_6790 and DSC_6793).
pub(super) const NEAR_BEST_RATIO: f64 = 0.85;
/// Share of a frame's measured cells that must be soft before it is left out.
pub(super) const SOFT_FRAME_SHARE: f64 = 0.45;
/// A frame measured in fewer cells than this is never left out.
const SOFT_FRAME_MIN_CELLS: usize = 8;
/// Cells of the detail grid along the canvas long side.
const DETAIL_CELLS_LONG_SIDE: f64 = 64.0;
/// Coverage probes along the canvas long side; finer than the detail grid so
/// a narrow strip only one frame covers is still protected.
const COVERAGE_PROBES_LONG_SIDE: f64 = 512.0;
/// Side of the measured patch at a cell centre, in canvas pixels, at most.
const DETAIL_PATCH_PX: u32 = 96;
/// Lower bound of the luma a contrast-normalised detail is divided by.
const DETAIL_LEVEL_FLOOR: f64 = 0.04;

/// The canvas grids every frame is measured on, and the band it is measured
/// in.
#[derive(Clone, Copy, Debug)]
pub(super) struct DetailGrid {
    width: u32,
    height: u32,
    cell: f64,
    probe: f64,
    band: FocusBand,
}

impl DetailGrid {
    pub(super) fn new(width: u32, height: u32, band: FocusBand) -> Self {
        Self::with_cells(width, height, band, DETAIL_CELLS_LONG_SIDE, 32.0)
    }

    /// A grid of `cells_long_side` cells along the canvas long side, each at
    /// least `min_cell` canvas pixels.
    pub(super) fn with_cells(
        width: u32,
        height: u32,
        band: FocusBand,
        cells_long_side: f64,
        min_cell: f64,
    ) -> Self {
        let long_side = f64::from(width.max(height)).max(1.0);
        Self {
            width,
            height,
            cell: (long_side / cells_long_side).max(min_cell),
            probe: (long_side / COVERAGE_PROBES_LONG_SIDE).max(8.0),
            band,
        }
    }

    /// Side of a cell, in canvas pixels.
    pub(super) fn cell_size(&self) -> f64 {
        self.cell
    }

    fn columns(&self, step: f64) -> usize {
        (f64::from(self.width) / step).ceil() as usize
    }

    fn rows(&self, step: f64) -> usize {
        (f64::from(self.height) / step).ceil() as usize
    }

    pub(super) fn cell_count(&self) -> usize {
        self.columns(self.cell) * self.rows(self.cell)
    }

    /// Detail cell holding canvas position (`x`, `y`), if it lies on the
    /// canvas.
    pub(super) fn cell_at(&self, x: f64, y: f64) -> Option<usize> {
        if !(x >= 0.0 && y >= 0.0 && x < f64::from(self.width) && y < f64::from(self.height)) {
            return None;
        }
        let columns = self.columns(self.cell);
        let column = ((x / self.cell) as usize).min(columns - 1);
        let row = ((y / self.cell) as usize).min(self.rows(self.cell) - 1);
        Some(row * columns + column)
    }

    /// Canvas position of the centre of detail cell `index`.
    pub(super) fn cell_centre(&self, index: usize) -> (f64, f64) {
        let columns = self.columns(self.cell);
        (
            ((index % columns) as f64 + 0.5) * self.cell,
            ((index / columns) as f64 + 0.5) * self.cell,
        )
    }

    /// `(columns, rows)` of the detail grid.
    pub(super) fn cell_dimensions(&self) -> (usize, usize) {
        (self.columns(self.cell), self.rows(self.cell))
    }

    /// Side of the patch measured at a cell centre, in canvas pixels.
    fn patch(&self) -> u32 {
        DETAIL_PATCH_PX.min((self.cell * 0.5) as u32).max(8)
    }

    /// Half the side of the square sampled to measure a patch, in canvas
    /// pixels.
    fn sampled_half(&self) -> f64 {
        f64::from(self.patch()) * 0.5 + f64::from(self.band.margin())
    }

    fn probe_count(&self) -> usize {
        self.columns(self.probe) * self.rows(self.probe)
    }
}

/// One frame's contrast-normalised detail per measured cell and the coverage
/// probes it covers.
#[derive(Clone, Debug, Default)]
pub(super) struct FrameDetail {
    pub(super) cells: Vec<(usize, f64)>,
    pub(super) probes: Vec<usize>,
}

/// Contrast-normalised RMS detail of luma in `band` over the `patch` pixels
/// square centred on canvas position `centre`, or `None` when `sample`
/// misses any pixel the band needs.
fn patch_detail(
    sample: impl Fn(f64, f64) -> Option<Rgb<f32>>,
    centre: (f64, f64),
    patch: u32,
    band: FocusBand,
) -> Option<f64> {
    let half = f64::from(patch) * 0.5 + f64::from(band.margin());
    let side = patch as usize + 2 * band.margin() as usize;
    let mut values = Vec::with_capacity(side * side);
    for y in 0..side {
        for x in 0..side {
            let pixel = sample(
                (centre.0 - half).floor() + x as f64,
                (centre.1 - half).floor() + y as f64,
            )?;
            values.push(luma(pixel));
        }
    }
    if band == FocusBand::Weave {
        let values = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
        let weave = weave_band(&values, side, side);
        let (mut energy, mut level, mut count) = (0.0, 0.0, 0usize);
        for (&band, &luma) in weave.band.iter().zip(&weave.level) {
            if band.is_finite() {
                energy += f64::from(band * band);
                level += f64::from(luma);
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        let count = count as f64;
        return Some((energy / count).sqrt() / (level / count).max(DETAIL_LEVEL_FLOOR));
    }
    let (mut energy, mut level) = (0.0, 0.0);
    for y in 1..side - 1 {
        for x in 1..side - 1 {
            let centre = values[y * side + x];
            let differences = [
                values[y * side + x + 1] - centre,
                centre - values[y * side + x - 1],
                values[(y + 1) * side + x] - centre,
                centre - values[(y - 1) * side + x],
            ];
            energy += differences.iter().map(|value| value * value).sum::<f64>() * 0.5;
            level += centre;
        }
    }
    let count = ((side - 2) * (side - 2)) as f64;
    Some((energy / count).sqrt() / (level / count).max(DETAIL_LEVEL_FLOOR))
}

/// Detail of `sampler`'s layer (canvas bounds `layer`, inclusive) on `grid`.
///
/// A cell is measured on a patch at its centre that lies wholly inside the
/// frame, in the grid's band: the measure of the output-pixel focus evidence,
/// on the pixels the warp would write.
pub(super) fn measure_frame_detail(
    sampler: &LayerSampler<'_>,
    layer: (u32, u32, u32, u32),
    grid: &DetailGrid,
) -> FrameDetail {
    let (left, right, top, bottom) = layer;
    let inside = |x: f64, y: f64| {
        x >= f64::from(left)
            && x <= f64::from(right)
            && y >= f64::from(top)
            && y <= f64::from(bottom)
    };
    let sample = |x: f64, y: f64| {
        inside(x, y)
            .then(|| sampler.sample(x - f64::from(left), y - f64::from(top)))
            .flatten()
    };
    let patch = grid.patch();
    let half = grid.sampled_half();
    let cells = (0..grid.cell_count())
        .into_par_iter()
        .filter_map(|index| {
            let centre = grid.cell_centre(index);
            // Most cells lie outside the layer; skip them before sampling.
            if !inside(centre.0 - half, centre.1 - half)
                || !inside(centre.0 + half, centre.1 + half)
            {
                return None;
            }
            patch_detail(sample, centre, patch, grid.band).map(|value| (index, value))
        })
        .collect::<Vec<_>>();
    let probe_columns = grid.columns(grid.probe);
    let probes = (0..grid.probe_count())
        .into_par_iter()
        .filter(|&index| {
            let x = ((index % probe_columns) as f64 + 0.5) * grid.probe;
            let y = ((index / probe_columns) as f64 + 0.5) * grid.probe;
            inside(x, y) && sample(x, y).is_some()
        })
        .collect::<Vec<_>>();
    FrameDetail { cells, probes }
}

/// Every frame's detail relative to the sharpest frame of each cell, one
/// `grid.cell_count()` row per frame; `NaN` where the frame was not measured.
pub(super) fn detail_ratios(details: &[FrameDetail], grid: &DetailGrid) -> Vec<Vec<f32>> {
    let mut best = vec![0.0f64; grid.cell_count()];
    for detail in details {
        for &(cell, value) in &detail.cells {
            best[cell] = best[cell].max(value);
        }
    }
    details
        .iter()
        .map(|detail| {
            let mut ratios = vec![f32::NAN; grid.cell_count()];
            for &(cell, value) in &detail.cells {
                if best[cell] > 0.0 {
                    ratios[cell] = (value / best[cell]) as f32;
                }
            }
            ratios
        })
        .collect()
}

/// Frame indices in the order a focus scan composes them: sharpest first,
/// by the median over a frame's measured cells of its detail relative to the
/// sharpest frame in that cell. Ties and unmeasured frames keep their order.
///
/// The first frame to cover a region becomes the canvas every later frame is
/// compared with, and a later frame replaces it only where the two agree after
/// the transition low-pass. In capture order a blurred frame could take a
/// region first and keep it: 祁序2 DSC_6759, defocused and a few pixels off,
/// held a branch area against three sharper frames that disagreed with it.
pub(super) fn sharpness_order(details: &[FrameDetail], grid: &DetailGrid) -> Vec<usize> {
    let mut best = vec![0.0f64; grid.cell_count()];
    for detail in details {
        for &(cell, value) in &detail.cells {
            best[cell] = best[cell].max(value);
        }
    }
    let score = details
        .iter()
        .map(|detail| {
            let mut ratios = detail
                .cells
                .iter()
                .filter(|&&(cell, _)| best[cell] > 0.0)
                .map(|&(cell, value)| value / best[cell])
                .collect::<Vec<_>>();
            if ratios.is_empty() {
                return 0.0;
            }
            ratios.sort_by(f64::total_cmp);
            ratios[ratios.len() / 2]
        })
        .collect::<Vec<_>>();
    let mut order = (0..details.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
    order
}

/// Indices of the frames to leave out, softest first, with the share of their
/// measured cells that are soft.
///
/// A frame is left out when at least [`SOFT_FRAME_SHARE`] of its cells are
/// soft, unless every remaining frame would then miss a coverage probe it
/// covers or a cell would keep no frame within [`NEAR_BEST_RATIO`] of the
/// sharpest one there.
pub(super) fn soft_frames(details: &[FrameDetail], grid: &DetailGrid) -> Vec<(usize, f64)> {
    let mut best = vec![0.0f64; grid.cell_count()];
    for detail in details {
        for &(cell, value) in &detail.cells {
            best[cell] = best[cell].max(value);
        }
    }
    let shares = details
        .iter()
        .map(|detail| {
            let soft = detail
                .cells
                .iter()
                .filter(|&&(cell, value)| value < SOFT_FRAME_RATIO * best[cell])
                .count();
            (
                detail.cells.len(),
                soft as f64 / detail.cells.len().max(1) as f64,
            )
        })
        .collect::<Vec<_>>();
    let mut probe_cover = vec![0u32; grid.probe_count()];
    let mut sharp_cover = vec![0u32; grid.cell_count()];
    for detail in details {
        for &probe in &detail.probes {
            probe_cover[probe] += 1;
        }
        for &(cell, value) in &detail.cells {
            if value >= NEAR_BEST_RATIO * best[cell] {
                sharp_cover[cell] += 1;
            }
        }
    }
    let mut order = (0..details.len())
        .filter(|&index| {
            shares[index].0 >= SOFT_FRAME_MIN_CELLS && shares[index].1 >= SOFT_FRAME_SHARE
        })
        .collect::<Vec<_>>();
    order.sort_by(|&a, &b| shares[b].1.total_cmp(&shares[a].1).then(a.cmp(&b)));
    let mut dropped = Vec::new();
    for index in order {
        let detail = &details[index];
        let uncovers = detail.probes.iter().any(|&probe| probe_cover[probe] <= 1);
        let loses_sharp = detail
            .cells
            .iter()
            .any(|&(cell, value)| value >= NEAR_BEST_RATIO * best[cell] && sharp_cover[cell] <= 1);
        if uncovers || loses_sharp {
            continue;
        }
        for &probe in &detail.probes {
            probe_cover[probe] -= 1;
        }
        for &(cell, value) in &detail.cells {
            if value >= NEAR_BEST_RATIO * best[cell] {
                sharp_cover[cell] -= 1;
            }
        }
        dropped.push((index, shares[index].1));
    }
    dropped
}

/// A display-referred canvas value at or above this is clipped to white.
const CLIPPED_VALUE: f32 = 0.98;

/// Mean luma of a composed `image` (canvas pixel `origin` at its (0, 0)) on
/// the patch of every cell of `grid` where the patch is wholly covered, and
/// the share of the patch's pixels clipped to white in some channel.
pub(super) fn measure_image_tone(
    image: &image::Rgb32FImage,
    origin: (u32, u32),
    grid: &DetailGrid,
) -> Vec<(usize, f64, f64)> {
    let patch = grid.patch();
    let half = f64::from(patch) * 0.5;
    (0..grid.cell_count())
        .into_par_iter()
        .filter_map(|index| {
            let (cx, cy) = grid.cell_centre(index);
            let (x0, y0) = (
                (cx - half).floor() - f64::from(origin.0),
                (cy - half).floor() - f64::from(origin.1),
            );
            if x0 < 0.0
                || y0 < 0.0
                || x0 + f64::from(patch) > f64::from(image.width())
                || y0 + f64::from(patch) > f64::from(image.height())
            {
                return None;
            }
            let (mut sum, mut count, mut clipped) = (0.0, 0.0, 0.0);
            for y in y0 as u32..y0 as u32 + patch {
                for x in x0 as u32..x0 as u32 + patch {
                    let pixel = *image.get_pixel(x, y);
                    let peak = pixel[0].max(pixel[1]).max(pixel[2]);
                    if peak <= 0.0 {
                        return None;
                    }
                    if peak >= CLIPPED_VALUE {
                        clipped += 1.0;
                    }
                    sum += luma(pixel);
                    count += 1.0;
                }
            }
            Some((index, sum / count, clipped / count))
        })
        .collect()
}

/// Detail of whatever `sample` returns (canvas coordinates) on the patch of
/// cell centre `centre`, measured exactly like a frame.
pub(super) fn sample_detail(
    sample: impl Fn(f64, f64) -> Option<Rgb<f32>>,
    centre: (f64, f64),
    grid: &DetailGrid,
) -> Option<f64> {
    patch_detail(sample, centre, grid.patch(), grid.band)
}

/// Detail of a composed `image` whose pixel (0, 0) is canvas pixel `origin`,
/// on `grid`, measured exactly like a frame. Black pixels count as uncovered.
pub(super) fn measure_image_detail(
    image: &image::Rgb32FImage,
    origin: (u32, u32),
    grid: &DetailGrid,
) -> Vec<(usize, f64)> {
    let sample = |x: f64, y: f64| {
        let (x, y) = (x - f64::from(origin.0), y - f64::from(origin.1));
        if x < 0.0 || y < 0.0 || x >= f64::from(image.width()) || y >= f64::from(image.height()) {
            return None;
        }
        let pixel = *image.get_pixel(x as u32, y as u32);
        (pixel[0].max(pixel[1]).max(pixel[2]) > 0.0).then_some(pixel)
    };
    let patch = grid.patch();
    (0..grid.cell_count())
        .into_par_iter()
        .filter_map(|index| {
            patch_detail(sample, grid.cell_centre(index), patch, grid.band)
                .map(|value| (index, value))
        })
        .collect()
}
