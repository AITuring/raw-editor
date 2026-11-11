//! Exposure of every frame of a focus scan, solved jointly before composing.
//!
//! The compositor tones each layer against the canvas written before it, on
//! their overlap. That chains: a frame first to cover its region keeps its
//! own level, and where two such regions meet the layer between them can
//! match only one, which left frame-sized light and dark blocks across
//! 枫鹰雉鸡 (its frames ran from 1/2 s to 1/8 s, and the camera's exposure
//! times did not predict the levels either way). Here every pair of
//! overlapping frames is compared on a coarse canvas grid, cell by cell on
//! the same canvas positions, and one gain plane per frame and channel (a log
//! gain varying linearly across the frame) is solved for all pairs at once,
//! so no frame and no order sets the level. A plane, because the lamps lit
//! every frame of 枫鹰雉鸡 unevenly: neighbouring frames differ by 5% at one
//! end of their overlap and 20% at the other, which no constant settles. The
//! per-layer tone then only settles what is left.

use image::Rgb32FImage;
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use rayon::prelude::*;
use std::collections::HashMap;

use super::LayerSampler;

/// Cells of the exposure grid along the canvas long side.
const EXPOSURE_CELLS_LONG_SIDE: f64 = 128.0;
/// Points sampled across each cell, per side.
const CELL_SAMPLES: usize = 6;
/// Two frames are compared when they share at least this many cells.
const MIN_SHARED_CELLS: usize = 8;
/// Linear levels outside this range (black border, clipped white) are not
/// compared.
const LEVEL_RANGE: (f64, f64) = (0.01, 0.95);
/// Huber scale of one cell's log ratio: a cell where the frames disagree by
/// more (a reflection, a misregistered stroke) loses weight.
const HUBER_LOG: f64 = 0.05;
/// Reweighting passes of the robust solve.
const ROBUST_ITERATIONS: usize = 6;
/// Ridge on every frame's slopes and, far weaker, on its level: the slopes of
/// all frames together could drift into one canvas-wide ramp the pairs cannot
/// see, and a frame without pairs keeps gain 1.
const SLOPE_RIDGE: f64 = 0.5;
const LEVEL_RIDGE: f64 = 1e-4;
/// Largest correction, in log gain (a factor of about 2).
const MAX_LOG_GAIN: f64 = 0.7;

/// The coarse canvas grid frames are compared on.
#[derive(Clone, Copy, Debug)]
pub(super) struct ExposureGrid {
    columns: usize,
    rows: usize,
    cell: f64,
}

impl ExposureGrid {
    pub(super) fn new(width: u32, height: u32) -> Self {
        let cell = (f64::from(width.max(height)) / EXPOSURE_CELLS_LONG_SIDE).max(8.0);
        Self {
            columns: (f64::from(width) / cell).ceil() as usize,
            rows: (f64::from(height) / cell).ceil() as usize,
            cell,
        }
    }

    fn centre(&self, cell: usize) -> (f64, f64) {
        (
            ((cell % self.columns) as f64 + 0.5) * self.cell,
            ((cell / self.columns) as f64 + 0.5) * self.cell,
        )
    }
}

/// A frame's solved exposure: per channel, a log gain `a + b u + c v` over
/// the frame's extent on the canvas (`u`, `v` in -1..1).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameExposure {
    centre: (f64, f64),
    half: (f64, f64),
    planes: [[f64; 3]; 3],
}

impl FrameExposure {
    /// A plane with explicit parameters (tests only).
    #[cfg(test)]
    pub(crate) fn with_planes(centre: (f64, f64), half: (f64, f64), planes: [[f64; 3]; 3]) -> Self {
        Self {
            centre,
            half,
            planes,
        }
    }

    fn identity() -> Self {
        Self {
            centre: (0.0, 0.0),
            half: (1.0, 1.0),
            planes: [[0.0; 3]; 3],
        }
    }

    fn local(&self, (x, y): (f64, f64)) -> (f64, f64) {
        (
            ((x - self.centre.0) / self.half.0).clamp(-1.5, 1.5),
            ((y - self.centre.1) / self.half.1).clamp(-1.5, 1.5),
        )
    }

    /// Log gain per channel at canvas position `point`.
    pub(crate) fn log_gain_at(&self, point: (f64, f64)) -> [f64; 3] {
        let (u, v) = self.local(point);
        self.planes
            .map(|[a, b, c]| (a + b * u + c * v).clamp(-MAX_LOG_GAIN, MAX_LOG_GAIN))
    }

    /// Hash every parameter, for caches keyed on the replayed tone.
    pub(crate) fn hash_into(&self, hasher: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        for value in [self.centre.0, self.centre.1, self.half.0, self.half.1]
            .into_iter()
            .chain(self.planes.into_iter().flatten())
        {
            value.to_bits().hash(hasher);
        }
    }

    /// Luma gain at the frame's centre (diagnostics).
    fn centre_gain(&self) -> f64 {
        let gain = self.log_gain_at(self.centre).map(f64::exp);
        0.2126 * gain[0] + 0.7152 * gain[1] + 0.0722 * gain[2]
    }

    /// Largest luma change of the plane across the frame, as a factor.
    fn spread(&self) -> f64 {
        let [_, b, c] = self.planes[1];
        (b.abs() + c.abs()).exp()
    }
}

/// Mean RGB of `sampler`'s layer (canvas bounds `layer`, inclusive) in every
/// cell of `grid` it covers wholly.
pub(super) fn sample_frame_tone(
    sampler: &LayerSampler<'_>,
    layer: (u32, u32, u32, u32),
    grid: &ExposureGrid,
) -> Vec<(usize, [f64; 3])> {
    let (left, right, top, bottom) = layer;
    (0..grid.columns * grid.rows)
        .into_par_iter()
        .filter_map(|cell| {
            let x0 = (cell % grid.columns) as f64 * grid.cell;
            let y0 = (cell / grid.columns) as f64 * grid.cell;
            if x0 < f64::from(left)
                || y0 < f64::from(top)
                || x0 + grid.cell > f64::from(right)
                || y0 + grid.cell > f64::from(bottom)
            {
                return None;
            }
            let mut sum = [0.0f64; 3];
            for sy in 0..CELL_SAMPLES {
                for sx in 0..CELL_SAMPLES {
                    let x = x0 + (sx as f64 + 0.5) * grid.cell / CELL_SAMPLES as f64;
                    let y = y0 + (sy as f64 + 0.5) * grid.cell / CELL_SAMPLES as f64;
                    let pixel = sampler.sample(x - f64::from(left), y - f64::from(top))?;
                    for channel in 0..3 {
                        sum[channel] += f64::from(pixel[channel]);
                    }
                }
            }
            let count = (CELL_SAMPLES * CELL_SAMPLES) as f64;
            let mean = sum.map(|value| value / count);
            mean.iter()
                .all(|value| (LEVEL_RANGE.0..LEVEL_RANGE.1).contains(value))
                .then_some((cell, mean))
        })
        .collect()
}

/// The exposure of every frame that brings every overlapping pair to the
/// same level, cell by cell, solved over all pairs; frames without an
/// overlap keep gain 1. `tones[i]` holds frame `i`'s cells from
/// [`sample_frame_tone`].
pub(super) fn solve_frame_exposures(
    grid: &ExposureGrid,
    tones: &[Vec<(usize, [f64; 3])>],
) -> Vec<FrameExposure> {
    let count = tones.len();
    let mut frames = vec![FrameExposure::identity(); count];
    for (frame, cells) in frames.iter_mut().zip(tones) {
        if cells.is_empty() {
            continue;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for &(cell, _) in cells {
            let (x, y) = grid.centre(cell);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        frame.centre = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        frame.half = (
            ((x1 - x0) * 0.5).max(grid.cell),
            ((y1 - y0) * 0.5).max(grid.cell),
        );
    }
    let maps = tones
        .iter()
        .map(|cells| cells.iter().copied().collect::<HashMap<_, _>>())
        .collect::<Vec<_>>();
    // Every shared cell of every pair: the two frames, the cell's position in
    // each, its log ratio per channel and its weight. A pair weighs as the
    // square root of its cells, so a large overlap counts more without
    // drowning the small ones that hold the chain together.
    let mut rows = Vec::new();
    for first in 0..count {
        for second in first + 1..count {
            let shared = maps[first]
                .iter()
                .filter_map(|(cell, a)| maps[second].get(cell).map(|b| (*cell, *a, *b)))
                .collect::<Vec<_>>();
            if shared.len() < MIN_SHARED_CELLS {
                continue;
            }
            let weight = 1.0 / (shared.len() as f64).sqrt();
            for (cell, a, b) in shared {
                let point = grid.centre(cell);
                // gain[first] / gain[second] = level[second] / level[first].
                let ratio: [f64; 3] = std::array::from_fn(|channel| (b[channel] / a[channel]).ln());
                rows.push((
                    first,
                    frames[first].local(point),
                    second,
                    frames[second].local(point),
                    ratio,
                    weight,
                ));
            }
        }
    }
    if rows.is_empty() {
        return frames;
    }
    for channel in 0..3 {
        let unknowns = 3 * count;
        let mut solution = DVector::<f64>::zeros(unknowns);
        for _ in 0..ROBUST_ITERATIONS {
            let mut normal = DMatrix::<f64>::zeros(unknowns, unknowns);
            let mut rhs = DVector::<f64>::zeros(unknowns);
            for &(first, (u1, v1), second, (u2, v2), ratio, weight) in &rows {
                let terms = [
                    (3 * first, 1.0),
                    (3 * first + 1, u1),
                    (3 * first + 2, v1),
                    (3 * second, -1.0),
                    (3 * second + 1, -u2),
                    (3 * second + 2, -v2),
                ];
                let predicted = terms
                    .iter()
                    .map(|&(index, value)| solution[index] * value)
                    .sum::<f64>();
                let residual = (predicted - ratio[channel]).abs();
                let weight = weight
                    * if residual > HUBER_LOG {
                        HUBER_LOG / residual
                    } else {
                        1.0
                    };
                for &(i, vi) in &terms {
                    rhs[i] += weight * vi * ratio[channel];
                    for &(j, vj) in &terms {
                        normal[(i, j)] += weight * vi * vj;
                    }
                }
            }
            for frame in 0..count {
                normal[(3 * frame, 3 * frame)] += LEVEL_RIDGE;
                normal[(3 * frame + 1, 3 * frame + 1)] += SLOPE_RIDGE;
                normal[(3 * frame + 2, 3 * frame + 2)] += SLOPE_RIDGE;
            }
            let Some(decomposition) = normal.cholesky() else {
                break;
            };
            solution = decomposition.solve(&rhs);
        }
        for (frame, exposure) in frames.iter_mut().enumerate() {
            exposure.planes[channel] = [
                solution[3 * frame],
                solution[3 * frame + 1],
                solution[3 * frame + 2],
            ];
        }
    }
    frames
}

/// Scales `image` (frame pixels, pose `transform` onto the canvas world,
/// canvas pixel = world + `offset`) by its exposure.
pub(super) fn apply_frame_exposure(
    image: &mut Rgb32FImage,
    exposure: &FrameExposure,
    transform: &Matrix3<f64>,
    offset: (f64, f64),
    frame_size: (u32, u32),
) {
    if exposure.planes == [[0.0; 3]; 3] {
        return;
    }
    let width = image.width() as usize;
    let scale = (
        f64::from(frame_size.0) / image.width() as f64,
        f64::from(frame_size.1) / image.height() as f64,
    );
    image
        .par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, pixel) in row.chunks_mut(3).enumerate() {
                let world = transform
                    * Vector3::new((x as f64 + 0.5) * scale.0, (y as f64 + 0.5) * scale.1, 1.0);
                if world.z.abs() < 1e-12 {
                    continue;
                }
                let point = (world.x / world.z + offset.0, world.y / world.z + offset.1);
                let gain = exposure.log_gain_at(point);
                for channel in 0..3 {
                    pixel[channel] *= gain[channel].exp() as f32;
                }
            }
        });
}

/// Log line: how far the solved gains spread.
pub(super) fn exposure_summary(exposures: &[FrameExposure]) -> String {
    let quantiles = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        let at = |share: f64| values[((values.len() - 1) as f64 * share).round() as usize];
        format!(
            "min {:.3} p10 {:.3} median {:.3} p90 {:.3} max {:.3}",
            at(0.0),
            at(0.1),
            at(0.5),
            at(0.9),
            at(1.0)
        )
    };
    format!(
        "{} frames; gain at the frame centre {}; change across a frame {}",
        exposures.len(),
        quantiles(exposures.iter().map(FrameExposure::centre_gain).collect()),
        quantiles(exposures.iter().map(FrameExposure::spread).collect()),
    )
}

/// Test-only trace (`RAW_EDITOR_EXPOSURE_TRACE=<prefix>`): every frame's
/// plane and every compared pair with its shared cells, its median measured
/// log ratio and the median residual the solve leaves, in green.
#[cfg(test)]
pub(super) fn trace_frame_exposures(
    names: &[String],
    grid: &ExposureGrid,
    tones: &[Vec<(usize, [f64; 3])>],
    exposures: &[FrameExposure],
) {
    let Some(prefix) = std::env::var_os("RAW_EDITOR_EXPOSURE_TRACE") else {
        return;
    };
    let prefix = std::path::PathBuf::from(prefix);
    let mut frames = String::from("frame,centre_gain,spread,a_g,b_g,c_g,cells\n");
    for ((name, exposure), cells) in names.iter().zip(exposures).zip(tones) {
        let [a, b, c] = exposure.planes[1];
        frames.push_str(&format!(
            "{name},{:.4},{:.4},{a:.4},{b:.4},{c:.4},{}\n",
            exposure.centre_gain(),
            exposure.spread(),
            cells.len()
        ));
    }
    let maps = tones
        .iter()
        .map(|cells| cells.iter().copied().collect::<HashMap<_, _>>())
        .collect::<Vec<_>>();
    let median = |mut values: Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let mut pairs = String::from("first,second,cells,log_ratio_g,residual_g\n");
    for first in 0..maps.len() {
        for second in first + 1..maps.len() {
            let shared = maps[first]
                .iter()
                .filter_map(|(cell, a)| maps[second].get(cell).map(|b| (*cell, a[1], b[1])))
                .collect::<Vec<_>>();
            if shared.len() < MIN_SHARED_CELLS {
                continue;
            }
            let ratios = shared
                .iter()
                .map(|&(_, a, b)| (b / a).ln())
                .collect::<Vec<_>>();
            let residuals = shared
                .iter()
                .map(|&(cell, a, b)| {
                    let point = grid.centre(cell);
                    exposures[first].log_gain_at(point)[1]
                        - exposures[second].log_gain_at(point)[1]
                        - (b / a).ln()
                })
                .collect::<Vec<_>>();
            pairs.push_str(&format!(
                "{},{},{},{:.4},{:.4}\n",
                names[first],
                names[second],
                shared.len(),
                median(ratios),
                median(residuals)
            ));
        }
    }
    let _ = std::fs::write(prefix.with_extension("frames.csv"), frames);
    let _ = std::fs::write(prefix.with_extension("pairs.csv"), pairs);
}
