//! Ownership-cell metrics and grid geometry (需求 3.1, 3.2, 3.3, 3.9).
//!
//! Normalised Sharpness_Score, the sampling window and cell size, the shared
//! cell sampling plan, the low-pass disagreement and the mismatch-penalised
//! data cost of one candidate cell.

use super::super::super::mosaic::ACUTANCE_PATCH_POINTS;
use super::super::super::mosaic::OWNERSHIP_MISMATCH_PENALTY;
use super::super::super::mosaic::SELECTION_LONG_SIDE;
use super::super::super::mosaic::acutance_with_step;
use super::super::super::mosaic::cell_focus_with_step;
use super::super::super::mosaic::cell_low_pass_differences;
use super::super::super::mosaic::cell_probe_offsets;
use image::Rgb;

use super::{
    OWNERSHIP_CELL_MAX_PX, OWNERSHIP_CELL_MIN_PX, OWNERSHIP_DISAGREEMENT_VETO,
    SHARPNESS_HALF_SCALE, SHARPNESS_SAMPLE_WINDOW_PX,
};

/// Sharpness_Score of one ownership cell, normalised to `[0, 1]` (需求 3.9).
///
/// `a / (a + SHARPNESS_HALF_SCALE)`: `0 → 0`, `a → ∞ → 1`, strictly increasing
/// in between, so the mapping is order preserving and owner selection is
/// unchanged by it.
pub(crate) fn normalized_sharpness(acutance: f64) -> f64 {
    if !acutance.is_finite() || acutance <= 0.0 {
        return 0.0;
    }
    (acutance / (acutance + SHARPNESS_HALF_SCALE)).clamp(0.0, 1.0)
}

/// Spacing between the 13×13 patch grid points of one Sharpness_Score probe.
///
/// `32 / 13` native pixels. The sampler accepts fractional native
/// coordinates, so rounding this to `2` (a 26px effective window) needlessly
/// violated Requirement 3.2 and changed the evidence scale relative to the
/// graph-cut data term.
pub(crate) fn sharpness_sample_step_px() -> f64 {
    (SHARPNESS_SAMPLE_WINDOW_PX / f64::from(ACUTANCE_PATCH_POINTS)).max(1.0)
}

/// Side of the effective Sharpness_Score window in native pixels.
pub(crate) fn sharpness_window_px() -> f64 {
    f64::from(ACUTANCE_PATCH_POINTS) * sharpness_sample_step_px()
}

/// Ownership cell side in native pixels for a station plane whose long side is
/// `long_side_px` (需求 3.1).
///
/// `clamp(floor(long_side / SELECTION_LONG_SIDE), 8, 64)`.
///
/// The task text and the design sketch both say `ceil`, and both also claim that
/// `SELECTION_LONG_SIDE = 512` then guarantees at least 512 cells along the long
/// side.  That does not hold: on the 9,504px plane the design itself cites,
/// `ceil(9504 / 512) = 19` and `9504 / 19 = 501` cells, below the floor 需求 3.1
/// sets.  Rounding *down* is the only direction that keeps both clauses of the
/// requirement — 18px cells give 528 cells there — and a smaller cell is always
/// the safer half of the trade: it can only let a seam route more precisely.
///
/// The `8` floor wins below 4,096px, where a plane has fewer native pixels than
/// 512 cells of the minimum legal side; the `64` ceiling wins above 32,768px and
/// only makes the grid finer than required.
pub(crate) fn ownership_cell_size_px(long_side_px: u32) -> u32 {
    let cells = SELECTION_LONG_SIDE.max(1);
    let size = (f64::from(long_side_px) / f64::from(cells))
        .floor()
        .max(1.0);
    (size as u32).clamp(OWNERSHIP_CELL_MIN_PX, OWNERSHIP_CELL_MAX_PX)
}

/// The ownership cell grid of a station plane (需求 3.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OwnershipGridGeometry {
    pub(crate) cell_size_px: u32,
    pub(crate) columns: u32,
    pub(crate) rows: u32,
}

impl OwnershipGridGeometry {
    /// Cover a `width × height` station plane with square ownership cells.  The
    /// last column and row may hang over the plane edge; a partially covered
    /// cell still has to be owned.
    pub(crate) fn for_plane(width: u32, height: u32) -> Self {
        let cell_size_px = ownership_cell_size_px(width.max(height));
        let cells = |extent: u32| extent.div_ceil(cell_size_px.max(1));
        Self {
            cell_size_px,
            columns: cells(width),
            rows: cells(height),
        }
    }

    pub(crate) fn cell_count(&self) -> usize {
        self.columns as usize * self.rows as usize
    }

    /// Top-left corner of a cell in plane pixels.
    pub(crate) fn cell_origin(&self, column: u32, row: u32) -> (f64, f64) {
        let size = f64::from(self.cell_size_px);
        (f64::from(column) * size, f64::from(row) * size)
    }

    /// Sampling plan shared by every candidate of one cell (需求 3.2).
    pub(crate) fn sampling_plan(&self) -> CellSamplingPlan {
        CellSamplingPlan::for_cell(f64::from(self.cell_size_px))
    }
}

/// Where and how wide one ownership cell is measured.
///
/// Built once per cell and used for *both* the current composite and every
/// candidate, so 需求 3.2's "same sample positions and same window size in the
/// same cell" holds by construction.  [`CellSamplingPlan::assert_matches`] makes
/// that explicit wherever two measurements are compared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CellSamplingPlan {
    /// Ownership cell side in native pixels.
    pub(super) cell_size_px: f64,
    /// Spacing between the patch grid points of one probe, in native pixels.
    pub(super) step_px: f64,
    /// The five probe centres, relative to the cell centre (需求 3.2).
    pub(super) probes: [(f64, f64); 5],
}

impl CellSamplingPlan {
    pub(crate) fn for_cell(cell_size_px: f64) -> Self {
        Self {
            cell_size_px,
            step_px: sharpness_sample_step_px(),
            probes: cell_probe_offsets(cell_size_px),
        }
    }

    pub(crate) fn cell_size_px(&self) -> f64 {
        self.cell_size_px
    }

    pub(crate) fn step_px(&self) -> f64 {
        self.step_px
    }

    /// Side of the measured window in native pixels (需求 3.2).
    pub(crate) fn window_px(&self) -> f64 {
        f64::from(ACUTANCE_PATCH_POINTS) * self.step_px
    }

    /// The five probe centres relative to the cell centre: the cell centre plus
    /// its four corners.
    pub(crate) fn probes(&self) -> [(f64, f64); 5] {
        self.probes
    }

    /// 需求 3.2: a candidate may only be compared against a measurement that used
    /// the same positions and the same window.  Comparing a 26px window against
    /// a 13px one, or a corner probe against a centre probe, would make defocus
    /// indistinguishable from a different sampling geometry.
    pub(crate) fn assert_matches(&self, other: &Self) {
        assert_eq!(
            self.step_px, other.step_px,
            "candidate and current composite must measure Sharpness_Score with the same window size"
        );
        assert_eq!(
            self.cell_size_px, other.cell_size_px,
            "candidate and current composite must measure the same ownership cell"
        );
        assert_eq!(
            self.probes, other.probes,
            "candidate and current composite must measure the same positions inside the cell"
        );
    }

    /// Normalised Sharpness_Score of one cell (需求 3.2, 3.9).
    ///
    /// `origin_x` / `origin_y` are the cell's top-left corner in the coordinate
    /// space of `sample`; the probes are placed around the cell centre.  `sample`
    /// returns `None` outside its own valid coverage, which drives the score to
    /// `0` exactly as an unfocused cell would.
    pub(crate) fn measure(
        &self,
        origin_x: f64,
        origin_y: f64,
        mut sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    ) -> f64 {
        let half = self.cell_size_px * 0.5;
        let mut incomplete = false;
        let score = cell_focus_with_step(
            |x, y| {
                let pixel = sample(x, y);
                incomplete |= pixel.is_none();
                pixel
            },
            origin_x + half,
            origin_y + half,
            self.cell_size_px,
            self.step_px,
        );
        // All five probes share one native comparison window. Selecting the
        // strongest surviving probe at a cropped sensor edge lets a partial
        // frame replace a fully observed contour with an unrelated boundary.
        if incomplete {
            0.0
        } else {
            normalized_sharpness(score)
        }
    }

    /// Normalised Sharpness_Score of a single probe, at the same window size the
    /// cell measurement uses.  Exists so the anchor-selection probe of
    /// [`crate::panorama_utils::stack_pipeline::intra_station`] and the fusion share one window definition.
    pub(crate) fn measure_probe(
        &self,
        x: f64,
        y: f64,
        sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    ) -> f64 {
        normalized_sharpness(acutance_with_step(sample, x, y, self.step_px))
    }
}

/// Pixel inconsistency of a candidate against the current composite in one
/// ownership cell: the median of the low-pass absolute pixel differences,
/// normalised to `[0, 1]` (需求 3.3).
///
/// Both closures receive the same fractional offsets inside the cell and map
/// them into their own coordinate space.  Composite pixels are float RGB in the
/// `[0, 1]` display range, so the median is already in the unit interval; the
/// clamp only guards an out-of-range source.  An empty comparison (no position
/// is valid in both images) is maximal inconsistency, which keeps an
/// unverifiable candidate from winning a cell.
pub(crate) fn cell_disagreement(
    candidate_at: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    base_at: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
) -> f64 {
    let mut differences = cell_low_pass_differences(candidate_at, base_at);
    if differences.is_empty() {
        return 1.0;
    }
    differences.sort_by(f64::total_cmp);
    median_of_sorted(&differences).clamp(0.0, 1.0)
}

/// Median photometric disagreement and the strongest supported local contour
/// disagreement. The latter prices only ownership boundaries; it must not veto
/// a sharper source through the median data term required by 3.3.
pub(crate) fn cell_disagreement_with_seam_evidence(
    candidate_at: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    base_at: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
) -> (f64, f64) {
    let mut differences = cell_low_pass_differences(candidate_at, base_at);
    if differences.is_empty() {
        return (1.0, 1.0);
    }
    differences.sort_by(f64::total_cmp);
    (
        median_of_sorted(&differences).clamp(0.0, 1.0),
        differences.last().copied().unwrap().clamp(0.0, 1.0),
    )
}

/// Median of an ascending slice.  An even count averages the two middle values,
/// so the result does not depend on which side a dropped probe fell on.
pub(super) fn median_of_sorted(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    }
}

/// Selection cost penalty of a candidate in one ownership cell (需求 3.3).
///
/// Applied only when the candidate is photometrically inconsistent *and*
/// sharper: that combination is the signature of a misregistered sharp frame,
/// which would otherwise win the cell on acutance alone and place a second copy
/// of a stroke next to the existing one.  A sharper *and* consistent candidate
/// is the normal focus-stacking case and must not be penalised, and a soft
/// candidate loses on the data term already.
///
/// The penalty is [`OWNERSHIP_MISMATCH_PENALTY`] (2.4 ≥ 1.0) and shares the
/// dimension of the normalised Sharpness_Score, as 需求 3.3 requires.
pub(crate) fn mismatch_penalty(
    disagreement: f64,
    candidate_sharpness: f64,
    base_sharpness: f64,
) -> f64 {
    let sharper = candidate_sharpness > base_sharpness;
    if disagreement > OWNERSHIP_DISAGREEMENT_VETO && sharper {
        OWNERSHIP_MISMATCH_PENALTY
    } else {
        0.0
    }
}

/// Data term of one candidate in one ownership cell (需求 3.4).
///
/// `(1 − Sharpness_Score) + mismatch penalty`.
pub(crate) fn data_cost(normalized_sharpness: f64, mismatch_penalty: f64) -> f64 {
    (1.0 - normalized_sharpness.clamp(0.0, 1.0)) + mismatch_penalty.max(0.0)
}

/// Per-cell choice between the current owner and one candidate.
///
/// This is the independent per-cell minimum, which 需求 3.5 keeps as the timeout
/// fallback of the graph cut.  A tie keeps the current owner, which makes a
/// duplicate frame idempotent and — because candidates are visited in the
/// deterministic order of [`candidate_order`] — resolves every tie in favour of
/// the anchor frame.
pub(crate) fn candidate_wins_cell(base_cost: f64, candidate_cost: f64) -> bool {
    candidate_cost < base_cost
}
