//! Sharpness_Confidence and low sharpness regions (需求 3.8, 3.9).
//!
//! Per-cell confidence, the low-sharpness threshold and regions with their
//! report records, and the per-cell sharpness evidence grid.

use super::super::report::LowSharpnessRegionRecord;
use super::super::report::SharpnessConfidenceSummary;
use super::super::report::WorldRect;

use super::OwnershipGridGeometry;

// ---------------------------------------------------------------------------
// Sharpness_Confidence and low sharpness regions (需求 3.8, 3.9)
// ---------------------------------------------------------------------------

/// ε of the Sharpness_Confidence normalisation (需求 3.9).  Only reached in a
/// cell whose every candidate measured a Sharpness_Score of exactly `0`, where
/// the numerator is `0` too and the quotient is therefore `0` either way; the
/// constant exists so the division itself is defined.
pub(super) const CONFIDENCE_SCALE_EPSILON: f64 = 1e-6;

/// Percentile of the winning Sharpness_Score distribution below which a cell
/// counts as a low sharpness region (需求 3.8).
pub(crate) const LOW_SHARPNESS_PERCENTILE: f64 = 0.10;

/// Sharpness_Confidence of one ownership cell (需求 3.9).
///
/// `clamp((s_best − s_second) / max(joint_gradient_scale, ε), 0, 1)` with the
/// joint gradient scale taken as the largest normalised Sharpness_Score the
/// cell's candidates reached.  One candidate leaves no runner-up to compare
/// against, which 需求 3.9 fixes at `0` rather than at "perfectly certain".
pub(crate) fn cell_confidence(best: f64, second: f64, candidate_count: usize) -> f32 {
    if candidate_count <= 1 {
        return 0.0;
    }
    let scale = best.max(CONFIDENCE_SCALE_EPSILON);
    (((best - second) / scale).clamp(0.0, 1.0)) as f32
}

/// 10th percentile of the winning Sharpness_Scores of one Capture_Station
/// (需求 3.8).
///
/// Nearest rank on the sorted sample, so the threshold is always a value the
/// station actually produced and a station of uniform sharpness has no cell
/// strictly below it.
pub(crate) fn low_sharpness_threshold(winning: &[f64]) -> f64 {
    let mut sorted = winning
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(f64::total_cmp);
    let last = sorted.len() - 1;
    let rank = (last as f64 * LOW_SHARPNESS_PERCENTILE).round() as usize;
    sorted[rank.min(last)]
}

/// A connected run of low sharpness cells, in ownership cell coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LowSharpnessRegion {
    pub first_column: u32,
    pub first_row: u32,
    pub last_column: u32,
    pub last_row: u32,
    pub cells: u64,
}

/// Merge the low sharpness cells into 4-connected regions (需求 3.8).
///
/// Seeds are visited in row-major order and each region is flood filled with an
/// explicit stack, so the region list and its order depend only on the mask.
pub(crate) fn low_sharpness_regions(
    columns: u32,
    rows: u32,
    low: &[bool],
) -> Vec<LowSharpnessRegion> {
    let (width, height) = (columns as usize, rows as usize);
    if low.len() != width * height {
        return Vec::new();
    }
    let mut seen = vec![false; low.len()];
    let mut regions = Vec::new();
    let mut stack = Vec::new();
    for seed in 0..low.len() {
        if seen[seed] || !low[seed] {
            continue;
        }
        let (mut first_column, mut last_column) = (u32::MAX, 0u32);
        let (mut first_row, mut last_row) = (u32::MAX, 0u32);
        let mut cells = 0u64;
        seen[seed] = true;
        stack.push(seed);
        while let Some(cell) = stack.pop() {
            let (x, y) = ((cell % width) as u32, (cell / width) as u32);
            first_column = first_column.min(x);
            last_column = last_column.max(x);
            first_row = first_row.min(y);
            last_row = last_row.max(y);
            cells += 1;
            let (x, y) = (cell % width, cell / width);
            for neighbour in [
                (y > 0).then(|| cell - width),
                (x > 0).then(|| cell - 1),
                (x + 1 < width).then(|| cell + 1),
                (y + 1 < height).then(|| cell + width),
            ]
            .into_iter()
            .flatten()
            {
                if !seen[neighbour] && low[neighbour] {
                    seen[neighbour] = true;
                    stack.push(neighbour);
                }
            }
        }
        regions.push(LowSharpnessRegion {
            first_column,
            first_row,
            last_column,
            last_row,
            cells,
        });
    }
    regions
}

/// Place the low sharpness regions in world coordinates (需求 3.8).
///
/// `world_origin` is the world position of the station plane's top-left pixel.
/// A Virtual_Tile's `tile_to_world` is a pure translation, so the bounding
/// rectangle and the area transfer without a change of shape, and the area is
/// the covered cell count times the cell area — not the bounding box, which
/// would overstate an L shaped region.
pub(crate) fn low_sharpness_region_records(
    geometry: &OwnershipGridGeometry,
    regions: &[LowSharpnessRegion],
    world_origin: (f64, f64),
) -> Vec<LowSharpnessRegionRecord> {
    let cell = f64::from(geometry.cell_size_px);
    regions
        .iter()
        .map(|region| LowSharpnessRegionRecord {
            world: WorldRect {
                left: world_origin.0 + f64::from(region.first_column) * cell,
                top: world_origin.1 + f64::from(region.first_row) * cell,
                width: f64::from(region.last_column - region.first_column + 1) * cell,
                height: f64::from(region.last_row - region.first_row + 1) * cell,
            },
            area_px: region.cells as f64 * cell * cell,
        })
        .collect()
}

/// Mean Sharpness_Confidence and the low confidence share over the covered
/// pixels of one Virtual_Tile (需求 11.12 measures the same ratio on the final
/// output).
pub(crate) fn confidence_summary(values: &[f32], covered: &[u8]) -> SharpnessConfidenceSummary {
    let mut total = 0.0f64;
    let mut low = 0u64;
    let mut count = 0u64;
    for (&value, &covered) in values.iter().zip(covered) {
        if covered == 0 {
            continue;
        }
        count += 1;
        total += f64::from(value);
        if f64::from(value) < 0.05 {
            low += 1;
        }
    }
    if count == 0 {
        return SharpnessConfidenceSummary::default();
    }
    SharpnessConfidenceSummary {
        mean: total / count as f64,
        below_0_05_ratio: low as f64 / count as f64,
    }
}

/// The compact per-ownership-cell evidence retained for Quality_Gate
/// diagnostics.  Keeping this at cell resolution avoids another full-size
/// station plane while preserving the exact winner/runner values used by
/// [`cell_confidence`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct SharpnessCellEvidence {
    /// Largest normalised Sharpness_Score observed among the cell's candidates.
    pub(crate) winner_score: f32,
    /// Second-largest normalised Sharpness_Score observed among the candidates.
    pub(crate) runner_up_score: f32,
    /// Score of the source that owns the cell after the fusion decision.
    pub(crate) owner_score: f32,
    /// Number of candidates that supplied a measurement for this cell.
    pub(crate) candidate_count: u32,
    /// Maximum pairwise disagreement observed while this cell was fused.
    /// This is retained for report-only owner shortfall diagnostics and never
    /// participates in the ownership decision.
    pub(crate) disagreement: f32,
}

impl SharpnessCellEvidence {
    /// Fraction of the winning score that the selected owner left on the
    /// table. A single candidate has no shortfall by definition.
    pub(crate) fn owner_shortfall(self) -> f32 {
        if self.candidate_count <= 1 || self.winner_score <= 0.0 {
            return 0.0;
        }
        ((self.winner_score - self.owner_score) / self.winner_score.max(f32::EPSILON))
            .clamp(0.0, 1.0)
    }
}

/// Cell-resolution evidence plus the geometry needed to map output pixels
/// back to their common ownership cell.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SharpnessEvidenceGrid {
    pub(super) columns: u32,
    pub(super) rows: u32,
    pub(super) cell_size_px: u32,
    pub(super) cells: Vec<SharpnessCellEvidence>,
}

impl SharpnessEvidenceGrid {
    pub(crate) fn at_pixel(&self, x: u32, y: u32) -> Option<SharpnessCellEvidence> {
        let column = x / self.cell_size_px.max(1);
        let row = y / self.cell_size_px.max(1);
        if column >= self.columns || row >= self.rows {
            return None;
        }
        self.cells
            .get(row as usize * self.columns as usize + column as usize)
            .copied()
    }

    pub(crate) fn cells(&self) -> &[SharpnessCellEvidence] {
        &self.cells
    }

    pub(crate) fn dimensions(&self) -> (u32, u32, u32) {
        (self.columns, self.rows, self.cell_size_px)
    }
}
