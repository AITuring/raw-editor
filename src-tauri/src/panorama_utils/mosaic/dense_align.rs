//! Dense local alignment of a donor view onto the composed canvas.
//!
//! Handheld frames of wavy silk disagree with their neighbours by tens of
//! output pixels, and the disagreement changes within a few hundred pixels
//! (祁序2 DSC_6758 lay 48 px left and 32 px below DSC_6757 in one 600 px
//! region). A global pose plus one smooth residual field cannot follow that,
//! so a sharp frame displaced from the canvas had to be kept out. Here the
//! displacement is measured densely, coarse to fine, by normalised
//! correlation of low-passed luma: a defocused canvas and a sharp donor agree
//! once both are blurred beyond the defocus, and the low-pass leaves the
//! position of a stroke where it is.

use rayon::prelude::*;

/// Levels of the pyramid, the coarsest at 1/2^LEVELS of output resolution.
const LEVELS: usize = 3;
/// Finest level measured; 1 is half the output resolution.
const FINEST_LEVEL: usize = 1;
/// Low-pass sigma, in pixels of each level from the finest to the coarsest:
/// 4, 6 and 8 output pixels, wider than the silk weave the canvas may lack.
const LEVEL_SIGMA: [f32; LEVELS + 1] = [0.0, 2.0, 1.5, 1.0];
/// Template radius per level (pixels of that level).
const LEVEL_RADIUS: [i64; LEVELS + 1] = [0, 7, 6, 5];
/// Grid spacing of the measured points per level (pixels of that level).
const LEVEL_STEP: [usize; LEVELS + 1] = [0, 4, 4, 3];
/// Search radius around the prediction at the finer levels.
const REFINE_SEARCH: i64 = 2;
/// Correlation a match needs, from the finest level to the coarsest.
const LEVEL_MIN_CORRELATION: [f64; LEVELS + 1] = [0.0, 0.6, 0.55, 0.5];
/// Relative contrast under which a template has no structure to match.
const FLAT_CONTRAST: f64 = 0.01;
/// A measured vector further than this (pixels of its level) from the median
/// of its neighbours is an outlier.
const OUTLIER_DISTANCE: f64 = 1.5;
/// Share of the coarsest grid that must match for the alignment to hold.
const MIN_MATCHED_SHARE: f64 = 0.2;
/// 3×3 box passes over the finest field (a support of about 40 output
/// pixels after three passes).
const SMOOTHING_PASSES: usize = 3;

/// A single-channel image, row-major, `NaN` where uncovered.
#[derive(Clone, Debug)]
pub(super) struct Plane {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) values: Vec<f32>,
}

impl Plane {
    pub(super) fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            values: vec![f32::NAN; width * height],
        }
    }

    fn at(&self, x: i64, y: i64) -> f32 {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return f32::NAN;
        }
        self.values[y as usize * self.width + x as usize]
    }

    /// 2×2 box average; `NaN` where fewer than three of the four are covered.
    fn half(&self) -> Self {
        let (width, height) = (self.width / 2, self.height / 2);
        let mut half = Self::new(width, height);
        half.values
            .par_chunks_mut(width.max(1))
            .enumerate()
            .for_each(|(y, row)| {
                for (x, value) in row.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    let mut count = 0;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let sample = self.values[(2 * y + dy) * self.width + 2 * x + dx];
                        if sample.is_finite() {
                            sum += sample;
                            count += 1;
                        }
                    }
                    if count >= 3 {
                        *value = sum / count as f32;
                    }
                }
            });
        half
    }

    /// Gaussian blur by normalised convolution; `NaN` where less than half
    /// of the kernel weight is covered.
    pub(super) fn blurred(&self, sigma: f32) -> Self {
        if sigma <= 0.0 {
            return self.clone();
        }
        let radius = (3.0 * sigma).ceil() as i64;
        let weights = (-radius..=radius)
            .map(|offset| (-(offset * offset) as f32 / (2.0 * sigma * sigma)).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f32>();
        let pass = |source: &[f32], mask: &[f32], horizontal: bool| {
            let mut values = vec![0.0f32; source.len()];
            let mut covered = vec![0.0f32; source.len()];
            values
                .par_chunks_mut(self.width)
                .zip(covered.par_chunks_mut(self.width))
                .enumerate()
                .for_each(|(y, (row, row_covered))| {
                    for x in 0..self.width {
                        let (mut sum, mut weight) = (0.0f32, 0.0f32);
                        for (index, kernel) in weights.iter().enumerate() {
                            let offset = index as i64 - radius;
                            let (sx, sy) = if horizontal {
                                (x as i64 + offset, y as i64)
                            } else {
                                (x as i64, y as i64 + offset)
                            };
                            if sx < 0
                                || sy < 0
                                || sx >= self.width as i64
                                || sy >= self.height as i64
                            {
                                continue;
                            }
                            let at = sy as usize * self.width + sx as usize;
                            sum += kernel * source[at];
                            weight += kernel * mask[at];
                        }
                        row[x] = sum;
                        row_covered[x] = weight;
                    }
                });
            (values, covered)
        };
        let mask = self
            .values
            .iter()
            .map(|value| if value.is_finite() { 1.0 } else { 0.0 })
            .collect::<Vec<f32>>();
        let filled = self
            .values
            .iter()
            .map(|value| if value.is_finite() { *value } else { 0.0 })
            .collect::<Vec<f32>>();
        let (horizontal, horizontal_mask) = pass(&filled, &mask, true);
        let (vertical, vertical_mask) = pass(&horizontal, &horizontal_mask, false);
        let mut blurred = Self::new(self.width, self.height);
        for (index, value) in blurred.values.iter_mut().enumerate() {
            let weight = vertical_mask[index];
            if weight > 0.5 * total * total {
                *value = vertical[index] / weight;
            }
        }
        blurred
    }
}

/// Displacement from canvas to donor on a regular grid, in output pixels:
/// the donor's content for canvas position `p` lies at `p + at(p)`.
#[derive(Clone, Debug)]
pub(super) struct Displacement {
    columns: usize,
    rows: usize,
    step: f64,
    values: Vec<[f64; 2]>,
    /// Share of the coarsest grid points that found a match (diagnostics).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) matched_share: f64,
}

impl Displacement {
    /// No displacement: the donor's global pose.
    pub(super) fn zero() -> Self {
        Self {
            columns: 1,
            rows: 1,
            step: 1.0,
            values: vec![[0.0, 0.0]],
            matched_share: 0.0,
        }
    }

    /// The affine displacement closest to this one on its grid: a least-squares
    /// fit, refitted twice to the two thirds of the points it fits best, so
    /// the folds of a field gone astray do not pull it. `None` when the grid is
    /// too small to fit.
    pub(super) fn affine_fit(&self) -> Option<Self> {
        let points = (0..self.columns * self.rows)
            .map(|index| {
                let (gx, gy) = (index % self.columns, index / self.columns);
                (
                    gx as f64 * self.step,
                    gy as f64 * self.step,
                    self.values[index],
                )
            })
            .collect::<Vec<_>>();
        if points.len() < 9 {
            return None;
        }
        let mut kept = points.clone();
        let mut model = None;
        for _ in 0..3 {
            let fit = fit_affine(&kept)?;
            let mut ranked = points
                .iter()
                .map(|&(x, y, value)| {
                    let [px, py] = affine_at(&fit, x, y);
                    ((px - value[0]).hypot(py - value[1]), (x, y, value))
                })
                .collect::<Vec<_>>();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
            kept = ranked
                .iter()
                .take((points.len() * 2 / 3).max(9))
                .map(|&(_, point)| point)
                .collect();
            model = Some(fit);
        }
        let model = model?;
        Some(Self {
            columns: self.columns,
            rows: self.rows,
            step: self.step,
            values: points
                .iter()
                .map(|&(x, y, _)| affine_at(&model, x, y))
                .collect(),
            matched_share: self.matched_share,
        })
    }

    /// The largest change of the displacement per output pixel between
    /// neighbouring grid points over `(x0, y0, x1, y1)`: how far the donor is
    /// bent or stretched there rather than moved.
    pub(super) fn max_strain(&self, (x0, y0, x1, y1): (usize, usize, usize, usize)) -> f64 {
        let to_grid = |value: usize, count: usize| {
            ((value as f64 / self.step).floor() as usize).min(count - 1)
        };
        let mut strain = 0.0f64;
        for gy in to_grid(y0, self.rows)..=to_grid(y1, self.rows) {
            for gx in to_grid(x0, self.columns)..=to_grid(x1, self.columns) {
                let here = self.values[gy * self.columns + gx];
                for (nx, ny) in [(gx + 1, gy), (gx, gy + 1)] {
                    if nx < self.columns && ny < self.rows {
                        let there = self.values[ny * self.columns + nx];
                        strain =
                            strain.max((there[0] - here[0]).hypot(there[1] - here[1]) / self.step);
                    }
                }
            }
        }
        strain
    }

    pub(super) fn at(&self, x: f64, y: f64) -> [f64; 2] {
        let gx = (x / self.step).clamp(0.0, (self.columns - 1) as f64);
        let gy = (y / self.step).clamp(0.0, (self.rows - 1) as f64);
        let (x0, y0) = (gx as usize, gy as usize);
        let (x1, y1) = ((x0 + 1).min(self.columns - 1), (y0 + 1).min(self.rows - 1));
        let (fx, fy) = (gx - x0 as f64, gy - y0 as f64);
        std::array::from_fn(|axis| {
            let top = self.values[y0 * self.columns + x0][axis] * (1.0 - fx)
                + self.values[y0 * self.columns + x1][axis] * fx;
            let bottom = self.values[y1 * self.columns + x0][axis] * (1.0 - fx)
                + self.values[y1 * self.columns + x1][axis] * fx;
            top * (1.0 - fy) + bottom * fy
        })
    }
}

/// `[a, b, c]` per axis of the displacement `a x + b y + c`.
type Affine = [[f64; 3]; 2];

fn affine_at(model: &Affine, x: f64, y: f64) -> [f64; 2] {
    std::array::from_fn(|axis| model[axis][0] * x + model[axis][1] * y + model[axis][2])
}

/// Least-squares affine displacement through `points` (x, y, displacement),
/// `None` when they do not span a plane.
fn fit_affine(points: &[(f64, f64, [f64; 2])]) -> Option<Affine> {
    let count = points.len() as f64;
    let (mean_x, mean_y) = (
        points.iter().map(|point| point.0).sum::<f64>() / count,
        points.iter().map(|point| point.1).sum::<f64>() / count,
    );
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    let (mut sxd, mut syd, mut mean_d) = ([0.0; 2], [0.0; 2], [0.0; 2]);
    for &(x, y, value) in points {
        let (u, v) = (x - mean_x, y - mean_y);
        sxx += u * u;
        sxy += u * v;
        syy += v * v;
        for axis in 0..2 {
            sxd[axis] += u * value[axis];
            syd[axis] += v * value[axis];
            mean_d[axis] += value[axis] / count;
        }
    }
    let determinant = sxx * syy - sxy * sxy;
    if determinant.abs() < 1e-9 * (sxx * syy).max(1.0) {
        return None;
    }
    Some(std::array::from_fn(|axis| {
        let a = (sxd[axis] * syy - syd[axis] * sxy) / determinant;
        let b = (syd[axis] * sxx - sxd[axis] * sxy) / determinant;
        [a, b, mean_d[axis] - a * mean_x - b * mean_y]
    }))
}

/// One level's measured grid, in pixels of that level.
struct LevelField {
    columns: usize,
    rows: usize,
    step: usize,
    values: Vec<[f64; 2]>,
    matched: Vec<bool>,
}

impl LevelField {
    fn at(&self, x: f64, y: f64) -> [f64; 2] {
        let gx = (x / self.step as f64).clamp(0.0, (self.columns - 1) as f64);
        let gy = (y / self.step as f64).clamp(0.0, (self.rows - 1) as f64);
        let (x0, y0) = (gx as usize, gy as usize);
        let (x1, y1) = ((x0 + 1).min(self.columns - 1), (y0 + 1).min(self.rows - 1));
        let (fx, fy) = (gx - x0 as f64, gy - y0 as f64);
        std::array::from_fn(|axis| {
            let top = self.values[y0 * self.columns + x0][axis] * (1.0 - fx)
                + self.values[y0 * self.columns + x1][axis] * fx;
            let bottom = self.values[y1 * self.columns + x0][axis] * (1.0 - fx)
                + self.values[y1 * self.columns + x1][axis] * fx;
            top * (1.0 - fy) + bottom * fy
        })
    }
}

/// The best integer shift of the `moving` window against the `base`
/// template at `(x, y)`, searched within `radius` of `prediction`, refined
/// to a sub-pixel peak; `None` when the template is flat or no shift
/// correlates well enough.
#[allow(clippy::too_many_arguments)]
fn match_point(
    base: &Plane,
    moving: &Plane,
    margin: i64,
    (x, y): (i64, i64),
    template_radius: i64,
    prediction: (i64, i64),
    search: i64,
    min_correlation: f64,
) -> Option<[f64; 2]> {
    let full = ((2 * template_radius + 1) as usize).pow(2);
    let mut template = Vec::with_capacity(full);
    let mut offsets = Vec::with_capacity(full);
    for oy in -template_radius..=template_radius {
        for ox in -template_radius..=template_radius {
            let value = base.at(x + ox, y + oy);
            if value.is_finite() {
                template.push(f64::from(value));
                offsets.push((ox, oy));
            }
        }
    }
    if template.len() * 10 < full * 9 {
        return None;
    }
    let count = template.len() as f64;
    let mean = template.iter().sum::<f64>() / count;
    let centred = template
        .iter()
        .map(|value| value - mean)
        .collect::<Vec<_>>();
    let norm = centred
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if (norm / count.sqrt()) / mean.abs().max(0.04) < FLAT_CONTRAST {
        return None;
    }
    let side = (2 * search + 1) as usize;
    let mut scores = vec![f64::NEG_INFINITY; side * side];
    let mut window = Vec::with_capacity(template.len());
    for sy in -search..=search {
        for sx in -search..=search {
            let (dx, dy) = (prediction.0 + sx, prediction.1 + sy);
            window.clear();
            for &(ox, oy) in &offsets {
                window.push(f64::from(
                    moving.at(x + ox + dx + margin, y + oy + dy + margin),
                ));
            }
            if window.iter().any(|value| !value.is_finite()) {
                continue;
            }
            let window_mean = window.iter().sum::<f64>() / count;
            let (mut covariance, mut variance) = (0.0, 0.0);
            for (sample, base) in window.iter().zip(&centred) {
                let value = sample - window_mean;
                covariance += value * base;
                variance += value * value;
            }
            scores[(sy + search) as usize * side + (sx + search) as usize] =
                covariance / (variance.sqrt() * norm).max(1e-12);
        }
    }
    let (best, &score) = scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    if score < min_correlation {
        return None;
    }
    let (bx, by) = ((best % side) as i64, (best / side) as i64);
    // A peak on the search border may lie beyond it.
    if bx == 0 || by == 0 || bx == side as i64 - 1 || by == side as i64 - 1 {
        return None;
    }
    let at = |cx: i64, cy: i64| scores[cy as usize * side + cx as usize];
    let parabola = |minus: f64, centre: f64, plus: f64| {
        let curvature = minus - 2.0 * centre + plus;
        if curvature.abs() < 1e-9 || !minus.is_finite() || !plus.is_finite() {
            0.0
        } else {
            (0.5 * (minus - plus) / curvature).clamp(-0.5, 0.5)
        }
    };
    let sub_x = parabola(at(bx - 1, by), score, at(bx + 1, by));
    let sub_y = parabola(at(bx, by - 1), score, at(bx, by + 1));
    Some([
        (prediction.0 + bx - search) as f64 + sub_x,
        (prediction.1 + by - search) as f64 + sub_y,
    ])
}

/// One 3×3 box pass over a level field.
fn box_smoothed(field: &LevelField) -> Vec<[f64; 2]> {
    let (columns, rows) = (field.columns, field.rows);
    (0..columns * rows)
        .map(|index| {
            let (gx, gy) = (index % columns, index / columns);
            let mut sum = [0.0; 2];
            let mut count = 0.0;
            for ny in gy.saturating_sub(1)..=(gy + 1).min(rows - 1) {
                for nx in gx.saturating_sub(1)..=(gx + 1).min(columns - 1) {
                    let value = field.values[ny * columns + nx];
                    sum[0] += value[0];
                    sum[1] += value[1];
                    count += 1.0;
                }
            }
            [sum[0] / count, sum[1] / count]
        })
        .collect()
}

/// Replaces outliers and unmatched points of a level field by the median of
/// their matched neighbours, or by `fallback` where none matched nearby.
fn regularise(field: &mut LevelField, fallback: &dyn Fn(usize, usize) -> [f64; 2]) {
    let (columns, rows) = (field.columns, field.rows);
    let neighbours = |values: &[[f64; 2]], matched: &[bool], gx: usize, gy: usize, reach: usize| {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for ny in gy.saturating_sub(reach)..=(gy + reach).min(rows - 1) {
            for nx in gx.saturating_sub(reach)..=(gx + reach).min(columns - 1) {
                if (nx, ny) != (gx, gy) && matched[ny * columns + nx] {
                    xs.push(values[ny * columns + nx][0]);
                    ys.push(values[ny * columns + nx][1]);
                }
            }
        }
        (xs, ys)
    };
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let mut matched = field.matched.clone();
    for gy in 0..rows {
        for gx in 0..columns {
            let index = gy * columns + gx;
            if !field.matched[index] {
                continue;
            }
            let (mut xs, mut ys) = neighbours(&field.values, &field.matched, gx, gy, 2);
            if xs.len() < 3 {
                continue;
            }
            let (mx, my) = (median(&mut xs), median(&mut ys));
            let value = field.values[index];
            if (value[0] - mx).hypot(value[1] - my) > OUTLIER_DISTANCE {
                matched[index] = false;
            }
        }
    }
    let mut values = field.values.clone();
    for gy in 0..rows {
        for gx in 0..columns {
            let index = gy * columns + gx;
            if matched[index] {
                continue;
            }
            let mut filled = None;
            for reach in [1, 2, 4] {
                let (mut xs, mut ys) = neighbours(&field.values, &matched, gx, gy, reach);
                if xs.len() >= 2 {
                    filled = Some([median(&mut xs), median(&mut ys)]);
                    break;
                }
            }
            values[index] = filled.unwrap_or_else(|| fallback(gx, gy));
        }
    }
    field.values = values;
    field.matched = matched;
}

/// Measures where `moving` (the donor view, sampled `margin` output pixels
/// beyond the canvas window on every side) shows the content of `base` (the
/// canvas window). Both planes hold output-resolution luma. Returns `None`
/// when too little of the window can be matched.
pub(super) fn align(base: &Plane, moving: &Plane, margin: usize) -> Option<Displacement> {
    let mut base_levels = vec![base.clone()];
    let mut moving_levels = vec![moving.clone()];
    for _ in 0..LEVELS {
        let (base_half, moving_half) = (
            base_levels.last().expect("level").half(),
            moving_levels.last().expect("level").half(),
        );
        base_levels.push(base_half);
        moving_levels.push(moving_half);
    }
    let mut coarser: Option<LevelField> = None;
    let mut coarsest_share = 0.0;
    for level in (FINEST_LEVEL..=LEVELS).rev() {
        let scale = 1usize << level;
        let base_level = base_levels[level].blurred(LEVEL_SIGMA[level]);
        let moving_level = moving_levels[level].blurred(LEVEL_SIGMA[level]);
        let level_margin = (margin / scale) as i64;
        let step = LEVEL_STEP[level];
        let columns = base_level.width.div_ceil(step).max(1);
        let rows = base_level.height.div_ceil(step).max(1);
        let search = if coarser.is_some() {
            REFINE_SEARCH
        } else {
            (level_margin - 1).max(1)
        };
        let points = (0..columns * rows)
            .into_par_iter()
            .map(|index| {
                let (gx, gy) = (index % columns, index / columns);
                let (x, y) = ((gx * step) as i64, (gy * step) as i64);
                let prediction = coarser.as_ref().map_or((0, 0), |field| {
                    let [px, py] = field.at(x as f64 / 2.0, y as f64 / 2.0);
                    ((px * 2.0).round() as i64, (py * 2.0).round() as i64)
                });
                match_point(
                    &base_level,
                    &moving_level,
                    level_margin,
                    (x, y),
                    LEVEL_RADIUS[level],
                    prediction,
                    search,
                    LEVEL_MIN_CORRELATION[level],
                )
            })
            .collect::<Vec<_>>();
        let matched = points.iter().map(Option::is_some).collect::<Vec<_>>();
        let share = matched.iter().filter(|&&value| value).count() as f64 / matched.len() as f64;
        if coarser.is_none() {
            coarsest_share = share;
            if share < MIN_MATCHED_SHARE {
                return None;
            }
        }
        let mut field = LevelField {
            columns,
            rows,
            step,
            values: points
                .iter()
                .map(|point| point.unwrap_or([0.0, 0.0]))
                .collect(),
            matched,
        };
        let global = {
            let mut xs = field
                .values
                .iter()
                .zip(&field.matched)
                .filter(|(_, matched)| **matched)
                .map(|(value, _)| value[0])
                .collect::<Vec<_>>();
            let mut ys = field
                .values
                .iter()
                .zip(&field.matched)
                .filter(|(_, matched)| **matched)
                .map(|(value, _)| value[1])
                .collect::<Vec<_>>();
            xs.sort_by(f64::total_cmp);
            ys.sort_by(f64::total_cmp);
            if xs.is_empty() {
                [0.0, 0.0]
            } else {
                [xs[xs.len() / 2], ys[ys.len() / 2]]
            }
        };
        let prediction_of = |gx: usize, gy: usize| {
            coarser.as_ref().map_or(global, |field| {
                let [px, py] = field.at((gx * step) as f64 / 2.0, (gy * step) as f64 / 2.0);
                [px * 2.0, py * 2.0]
            })
        };
        regularise(&mut field, &prediction_of);
        coarser = Some(field);
    }
    let mut finest = coarser?;
    // Neighbouring vectors still scatter by a fraction of a pixel; folded or
    // stretched weave would show that, so the field is smoothed.
    for _ in 0..SMOOTHING_PASSES {
        finest.values = box_smoothed(&finest);
    }
    let scale = (1usize << FINEST_LEVEL) as f64;
    Some(Displacement {
        columns: finest.columns,
        rows: finest.rows,
        step: finest.step as f64 * scale,
        values: finest
            .values
            .iter()
            .map(|value| [value[0] * scale, value[1] * scale])
            .collect(),
        matched_share: coarsest_share,
    })
}
