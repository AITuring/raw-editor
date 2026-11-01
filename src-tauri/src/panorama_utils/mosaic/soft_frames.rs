//! Mostly soft frames of a focus scan.
//!
//! A handheld focus scan photographs every region two or three times, and some
//! frames are out of focus over much of their area: a 120 mm frame shaken at
//! 1/25 s, or a focal plane that missed the wavy silk. Ownership replaces such
//! a frame only where a later layer is clearly sharper, so the frame that
//! covered a region first kept it otherwise, and 祁序1 at full size still
//! showed frame-sized soft blocks. Before composing, every frame's detail is
//! measured at output resolution on a coarse canvas grid and compared with the
//! sharpest frame in each cell; a frame soft in at least
//! [`SOFT_FRAME_SHARE`] of its cells is left out, unless leaving it out would
//! uncover a part of the canvas or leave a cell without a sharp frame.
//!
//! Frames are only skipped by the compositor. Grouping and registration still
//! see every input, so leaving frames out of the selection itself cannot split
//! the station graph (祁序1 without 16 frames fell apart into three
//! components).

use image::Rgb;
use rayon::prelude::*;

use super::{LayerSampler, luma};

/// A frame is soft in a cell when its detail there is below this share of the
/// sharpest frame measured in the same cell.
pub(super) const SOFT_FRAME_RATIO: f64 = 0.6;
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

/// The canvas grids every frame is measured on.
#[derive(Clone, Copy, Debug)]
pub(super) struct DetailGrid {
    width: u32,
    height: u32,
    cell: f64,
    probe: f64,
}

impl DetailGrid {
    pub(super) fn new(width: u32, height: u32) -> Self {
        let long_side = f64::from(width.max(height)).max(1.0);
        Self {
            width,
            height,
            cell: (long_side / DETAIL_CELLS_LONG_SIDE).max(32.0),
            probe: (long_side / COVERAGE_PROBES_LONG_SIDE).max(8.0),
        }
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

    /// Canvas position of the centre of detail cell `index`.
    pub(super) fn cell_centre(&self, index: usize) -> (f64, f64) {
        let columns = self.columns(self.cell);
        (
            ((index % columns) as f64 + 0.5) * self.cell,
            ((index / columns) as f64 + 0.5) * self.cell,
        )
    }

    /// `(columns, rows)` of the detail grid.
    #[cfg(test)]
    pub(super) fn cell_dimensions(&self) -> (usize, usize) {
        (self.columns(self.cell), self.rows(self.cell))
    }

    /// Side of the patch measured at a cell centre, in canvas pixels.
    fn patch(&self) -> u32 {
        DETAIL_PATCH_PX.min((self.cell * 0.5) as u32).max(8)
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

/// Detail of `sampler`'s layer (canvas bounds `layer`, inclusive) on `grid`.
///
/// A cell is measured on a patch at its centre that lies wholly inside the
/// frame, as the RMS neighbour difference of luma over the mean luma: the
/// measure of the output-pixel focus evidence, on the pixels the warp would
/// write.
/// Contrast-normalised RMS neighbour difference of luma over the `patch`
/// pixels square centred on canvas position `centre`, or `None` when
/// `sample` misses any of its pixels.
fn patch_detail(
    sample: impl Fn(f64, f64) -> Option<Rgb<f32>>,
    centre: (f64, f64),
    patch: u32,
) -> Option<f64> {
    let half = f64::from(patch) * 0.5 + 1.0;
    let side = patch as usize + 2;
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
    let half = f64::from(patch) * 0.5 + 1.0;
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
            patch_detail(sample, centre, patch).map(|value| (index, value))
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

/// Indices of the frames to leave out, softest first, with the share of their
/// measured cells that are soft.
///
/// A frame is left out when at least [`SOFT_FRAME_SHARE`] of its cells are
/// soft, unless every remaining frame would then miss a coverage probe it
/// covers or a cell would keep no frame within [`SOFT_FRAME_RATIO`] of the
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
            if value >= SOFT_FRAME_RATIO * best[cell] {
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
            .any(|&(cell, value)| value >= SOFT_FRAME_RATIO * best[cell] && sharp_cover[cell] <= 1);
        if uncovers || loses_sharp {
            continue;
        }
        for &probe in &detail.probes {
            probe_cover[probe] -= 1;
        }
        for &(cell, value) in &detail.cells {
            if value >= SOFT_FRAME_RATIO * best[cell] {
                sharp_cover[cell] -= 1;
            }
        }
        dropped.push((index, shares[index].1));
    }
    dropped
}

/// Detail of a composed `image` whose pixel (0, 0) is canvas pixel `origin`,
/// on `grid`, measured exactly like a frame. Black pixels count as uncovered.
#[cfg(test)]
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
            patch_detail(sample, grid.cell_centre(index), patch).map(|value| (index, value))
        })
        .collect()
}
