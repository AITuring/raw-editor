//! Focus_Fuser: the ownership-cell decision metrics of the intra-station focus
//! fusion (需求 3.1, 3.2, 3.3, 3.9).
//!
//! # Why the metrics live here and not in `mosaic.rs`
//!
//! The task text points at `mosaic.rs`, but the default delivery path no longer
//! passes through the mosaic's ownership grid.  Since task 13.1 the default
//! compositor is `stitching::layered_virtual_tile_compositor`, which renders one
//! virtual tile per Capture_Station through `focus_stack_stitcher_unfilled` →
//! `focus_stack_stitcher_with_margin_policy`.  That function only hands over to
//! `mosaic::detail_preserving_mosaic` when `shifted_mosaic` is true, and a
//! virtual tile holds exactly one station by construction, so
//! `capture_group_count == 1` and `shifted_mosaic` is always false — a real run
//! logs "Compact focus capture detected (1 inferred group(s)); preserving
//! standard focus fusion" once per station.  `mosaic::acutance`,
//! `mosaic::cell_focus`, `mosaic::ownership_disagreement` and
//! `mosaic::ownership_grid` therefore only serve the legacy comparison path.
//!
//! This module reuses those measurements as *decision primitives*
//! ([`super::super::mosaic::acutance_with_step`],
//! [`super::super::mosaic::cell_focus_with_step`],
//! [`super::super::mosaic::cell_low_pass_differences`], and later
//! `seam_cut::cut_grid`) without calling `detail_preserving_mosaic` as a whole:
//! that entry point also applies a cell-wise local warp, which
//! `focus_stack_stitcher_with_margin_policy` already records as visibly
//! fragmenting strokes that are intentionally present at different focal depths.
//! The legacy reductions in `mosaic.rs` keep their old behaviour; every new rule
//! 需求 3 prescribes is a separate function here.
//!
//! Scope: tasks 7.6 and 7.7 build the metrics and the grid geometry, tasks 7.8
//! and 7.9 the multi-label graph cut over those costs together with its timeout
//! fallback and single-frame shortcut, and tasks 7.10 and 7.11 the hard
//! ownership decision the renderer consumes plus Sharpness_Confidence and the
//! low sharpness regions.  [`StationFusion`] is the entry point the production
//! focus renderer drives; see its documentation for why it folds one candidate
//! per exact two-label cut instead of refining a full multi-label grid.

// `solve_ownership` and the refinement sweeps around it are the whole-grid form
// of the same energy: they need every candidate's costs at once, which the
// render loop cannot afford (see `StationFusion`), so they are exercised by the
// tests and by the exhaustive-minimum comparison rather than by the renderer.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use image::Rgb;
use serde_json::json;

use super::super::mosaic::{
    ACUTANCE_PATCH_POINTS, OWNERSHIP_MISMATCH_PENALTY, SELECTION_LONG_SIDE, acutance_with_step,
    cell_focus_with_step, cell_low_pass_differences, cell_probe_offsets,
};
use super::super::seam_cut::{cut_grid_weighted, pairwise_weight};
use super::degradation::{DegradationLedger, FUSION_GRAPH_CUT_TIMEOUT};
use super::quality_gate;
use super::report::{
    FusionReport, FusionSolverStatus, LowSharpnessRegionRecord, OwnershipGridSize,
    SharpnessConfidenceSummary, WorldRect,
};

/// Acutance value that maps to a normalised Sharpness_Score of `0.5`.
///
/// `acutance()` returns an unbounded "gradient energy RMS over local level"
/// ratio, while the Glossary and 需求 3.9 require a Sharpness_Score in `[0, 1]`.
/// [`normalized_sharpness`] is strictly increasing on `[0, ∞)`, so this constant
/// cannot change which candidate of a cell is sharper — it only fixes the
/// dimension the data term and Sharpness_Confidence are expressed in.  The value
/// puts a focused brush edge near the middle of the unit interval while quiet
/// substrate stays close to `0`.
pub(crate) const SHARPNESS_HALF_SCALE: f64 = 0.10;

/// Requirement 11.12's absolute Textured_Pixel floor.
pub(crate) const TEXTURED_PIXEL_SHARPNESS_FLOOR: f64 = 0.10;

/// Side of the Sharpness_Score sampling window in native pixels (需求 3.2).
pub(crate) const SHARPNESS_SAMPLE_WINDOW_PX: f64 = 32.0;

/// Smallest ownership cell side in native pixels (需求 3.1).
pub(crate) const OWNERSHIP_CELL_MIN_PX: u32 = 8;

/// Largest ownership cell side in native pixels (需求 3.1).
pub(crate) const OWNERSHIP_CELL_MAX_PX: u32 = 64;

/// Normalised pixel inconsistency above which a sharper candidate is penalised
/// instead of simply preferred (需求 3.3).
pub(crate) const OWNERSHIP_DISAGREEMENT_VETO: f64 = 0.2;

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
    cell_size_px: f64,
    /// Spacing between the patch grid points of one probe, in native pixels.
    step_px: f64,
    /// The five probe centres, relative to the cell centre (需求 3.2).
    probes: [(f64, f64); 5],
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
        sample: impl FnMut(f64, f64) -> Option<Rgb<f32>>,
    ) -> f64 {
        let half = self.cell_size_px * 0.5;
        normalized_sharpness(cell_focus_with_step(
            sample,
            origin_x + half,
            origin_y + half,
            self.cell_size_px,
            self.step_px,
        ))
    }

    /// Normalised Sharpness_Score of a single probe, at the same window size the
    /// cell measurement uses.  Exists so the anchor-selection probe of
    /// [`super::intra_station`] and the fusion share one window definition.
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

/// Median of an ascending slice.  An even count averages the two middle values,
/// so the result does not depend on which side a dropped probe fell on.
fn median_of_sorted(sorted: &[f64]) -> f64 {
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

// ---------------------------------------------------------------------------
// Candidate order (需求 3.4)
// ---------------------------------------------------------------------------

/// One registration-successful frame competing for the cells of a
/// Capture_Station.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FusionFrame {
    /// Absolute path, which is also the tie-break key.
    pub path: String,
    /// Median Sharpness_Score over the frame's valid pixels.
    pub median_sharpness: f64,
    /// `true` for the frame [`super::intra_station::select_anchor`] chose.
    pub is_anchor: bool,
}

/// Order the candidates of one station are folded into the composite in
/// (需求 3.4).
///
/// Anchor frame first, then descending median Sharpness_Score, ties resolved by
/// ascending absolute path.  The order fixes three things that would otherwise
/// depend on import order: which frame is the initial base, which label wins a
/// cost tie (the earlier one, see [`candidate_wins_cell`]), and the sequence of
/// expansion moves — so two runs over the same photographs produce the same
/// ownership grid bit for bit.
pub(crate) fn candidate_order(frames: &[FusionFrame]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..frames.len()).collect();
    order.sort_by(|&left, &right| {
        let (a, b) = (&frames[left], &frames[right]);
        b.is_anchor
            .cmp(&a.is_anchor)
            .then_with(|| b.median_sharpness.total_cmp(&a.median_sharpness))
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| left.cmp(&right))
    });
    order
}

/// 需求 3.10: a station with at most one registration-successful frame has
/// nothing to cut.
pub(crate) fn skips_graph_cut(candidate_count: usize) -> bool {
    candidate_count <= 1
}

// ---------------------------------------------------------------------------
// Multi-label ownership solver (需求 3.4, 3.5, 3.10)
// ---------------------------------------------------------------------------

/// Reserved ownership identifier of a cell no candidate covers (需求 3.7).
pub(crate) const NO_OWNER: u16 = 0;

/// Wall-clock budget of the multi-label loop (需求 3.5).
pub(crate) const GRAPH_CUT_TIMEOUT: Duration = Duration::from_secs(120);

/// Upper bound on expansion sweeps over the candidate set.
///
/// Every accepted move strictly lowers the energy, so the loop terminates on its
/// own; this bound only keeps the worst case bounded well inside
/// [`GRAPH_CUT_TIMEOUT`] on a 528×352 grid.
pub(crate) const GRAPH_CUT_MAX_SWEEPS: usize = 4;

/// A move has to lower the energy by at least this much to be adopted, so
/// floating-point noise cannot produce an endless exchange of equal labelings.
const GRAPH_CUT_ENERGY_EPSILON: f64 = 1e-9;

/// Energy charged to a cell that some candidate covers but no owner claims.
///
/// 需求 3.4 requires *every* covered cell to end up with exactly one owner.
/// Expressing that as a large finite penalty rather than a hard constraint keeps
/// the energy comparable across partial labelings, which is what lets the loop
/// below reject a move that would make things worse.  It is far above any
/// reachable data plus smoothness cost, so a labeling that leaves a covered cell
/// unowned can never win.
const UNASSIGNED_CELL_COST: f64 = 1.0e6;

/// `fixed` value that pins a cell to the candidate of the current round.
const PIN_CANDIDATE: i8 = 1;

/// `fixed` value that pins a cell to its current owner.
const PIN_BASE: i8 = -1;

/// Ownership identifier of candidate `label`.  Identifiers are one-based so that
/// `0` stays the reserved "no owner" value of 需求 3.7.
pub(crate) fn owner_of(label: usize) -> u16 {
    u16::try_from(label + 1).expect("ownership label fits in u16")
}

/// Monotonic clock of the multi-label loop, injectable so the timeout path of
/// 需求 3.5 is testable without waiting two minutes.
pub(crate) trait GraphCutClock {
    /// Time since the solve started.  Never decreases.
    fn elapsed(&self) -> Duration;
}

/// The production clock: `Instant`, which is monotonic on every supported
/// platform and unaffected by a wall-clock adjustment mid-run.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MonotonicGraphCutClock {
    started: Instant,
}

impl MonotonicGraphCutClock {
    pub(crate) fn start() -> Self {
        Self {
            started: Instant::now(),
        }
    }
}

impl GraphCutClock for MonotonicGraphCutClock {
    fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

/// The per-cell, per-candidate selection costs of one Capture_Station.
///
/// The solver deliberately takes *costs*, not images: [`data_cost`] and
/// [`cell_disagreement`] above already turn the pixels of a cell into the two
/// numbers 需求 3.3 and 3.4 define, and keeping the minimisation a pure function
/// of the cost grid is what makes it comparable against an exhaustive search.
///
/// `costs[cell * labels + label]` is `None` when that candidate does not cover
/// the cell, which is the only way a candidate is excluded from a cell — 需求
/// 3.11's "only covered cells get an owner" falls out of the same field.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OwnershipCostGrid {
    columns: usize,
    rows: usize,
    labels: usize,
    costs: Vec<Option<f64>>,
    disagreement: Vec<f64>,
    /// Per-cell multiplier for the positive Potts smoothness term.  The
    /// default is one; StationFusion lowers it only where the winner is
    /// materially sharper than the runner-up.  Keeping the scale on the grid
    /// makes the same energy visible to the min-cut, descent and report tests.
    smoothness_scale: Vec<f64>,
}

impl OwnershipCostGrid {
    /// A grid where no candidate covers anything yet.
    pub(crate) fn new(columns: usize, rows: usize, labels: usize) -> Self {
        let cells = columns * rows;
        Self {
            columns,
            rows,
            labels,
            costs: vec![None; cells * labels],
            disagreement: vec![0.0; cells],
            smoothness_scale: vec![1.0; cells],
        }
    }

    pub(crate) fn columns(&self) -> usize {
        self.columns
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn label_count(&self) -> usize {
        self.labels
    }

    pub(crate) fn cell_count(&self) -> usize {
        self.columns * self.rows
    }

    /// Record [`data_cost`] of one candidate in one cell.  `None` means the
    /// candidate has no valid pixels there.
    pub(crate) fn set_cost(&mut self, cell: usize, label: usize, cost: Option<f64>) {
        assert!(cell < self.cell_count(), "cell out of range");
        assert!(label < self.labels, "label out of range");
        self.costs[cell * self.labels + label] = cost;
    }

    /// Record the cross-boundary pixel difference of one cell, which drives the
    /// smoothness term of 需求 3.4.
    ///
    /// One value per *cell*, not per candidate pair: the smoothness term has to
    /// be a fixed function of the grid for the total cost to be well defined at
    /// all, and 需求 3.4 describes it as a property of the boundary between two
    /// cells.  Callers pass the strongest disagreement the cell's candidates
    /// exhibit, so a seam is pushed away from any place the sources disagree.
    pub(crate) fn set_disagreement(&mut self, cell: usize, disagreement: f64) {
        assert!(cell < self.cell_count(), "cell out of range");
        self.disagreement[cell] = disagreement.clamp(0.0, 1.0);
    }

    /// Set the local multiplier of the positive smoothness term.  A small
    /// positive floor is intentional: a confident source may nearly remove a
    /// seam penalty, but the graph-cut energy still retains a regulariser.
    pub(crate) fn set_smoothness_scale(&mut self, cell: usize, scale: f64) {
        assert!(cell < self.cell_count(), "cell out of range");
        self.smoothness_scale[cell] = scale.clamp(0.001, 1.0);
    }

    fn edge_smoothness(&self, cell: usize, neighbour: usize) -> f64 {
        // Couple the two local confidence scales without letting an uncertain
        // endpoint restore the full seam cost everywhere.  The geometric mean
        // keeps a strictly positive Potts term, while allowing a genuinely
        // sharp winner to cross a boundary when both cells support that move.
        // Using `max` here made a confident cell ineffective next to any
        // uncertain neighbour and kept the production result close to the
        // uncorrected graph-cut labeling.
        let scale = (self.smoothness_scale[cell] * self.smoothness_scale[neighbour]).sqrt();
        pairwise_weight(self.disagreement[cell], self.disagreement[neighbour])
            * scale.clamp(0.001, 1.0)
    }

    pub(crate) fn cost(&self, cell: usize, label: usize) -> Option<f64> {
        self.costs[cell * self.labels + label]
    }

    /// `true` when at least one candidate covers the cell (需求 3.11).
    pub(crate) fn covered(&self, cell: usize) -> bool {
        (0..self.labels).any(|label| self.cost(cell, label).is_some())
    }

    fn label_of(&self, owner: u16) -> Option<usize> {
        (owner != NO_OWNER).then(|| usize::from(owner - 1))
    }

    /// The 4-neighbourhood of a cell, in row-major order.
    fn neighbours(&self, cell: usize) -> [Option<usize>; 4] {
        let columns = self.columns.max(1);
        let (x, y) = (cell % columns, cell / columns);
        [
            (y > 0).then(|| cell - self.columns),
            (x > 0).then(|| cell - 1),
            (x + 1 < self.columns).then(|| cell + 1),
            (y + 1 < self.rows).then(|| cell + self.columns),
        ]
    }

    /// Total cost of one labeling: the data term of 需求 3.4 plus the smoothness
    /// term [`pairwise_weight`] charges on every 4-neighbour edge whose two cells
    /// have different owners.
    ///
    /// A cell no candidate covers has no owner and takes part in no edge: the
    /// border of the coverage is not a seam, and charging it would make the cost
    /// of an interior decision depend on the shape of the station footprint.
    /// Claiming a cell a candidate does not cover is not a labeling at all, hence
    /// the infinity.
    pub(crate) fn energy(&self, owners: &[u16]) -> f64 {
        assert_eq!(owners.len(), self.cell_count(), "owner grid size");
        let mut total = 0.0;
        for (cell, &owner) in owners.iter().enumerate() {
            match self.label_of(owner) {
                Some(label) => match self.cost(cell, label) {
                    Some(cost) => total += cost,
                    None => return f64::INFINITY,
                },
                None if self.covered(cell) => total += UNASSIGNED_CELL_COST,
                None => {}
            }
        }
        for cell in 0..self.cell_count() {
            if owners[cell] == NO_OWNER {
                continue;
            }
            // Right and down only, so each edge is charged once.
            let (x, y) = (cell % self.columns, cell / self.columns);
            for neighbour in [
                (x + 1 < self.columns).then(|| cell + 1),
                (y + 1 < self.rows).then(|| cell + self.columns),
            ]
            .into_iter()
            .flatten()
            {
                if owners[neighbour] != NO_OWNER && owners[neighbour] != owners[cell] {
                    total += self.edge_smoothness(cell, neighbour);
                }
            }
        }
        total
    }

    /// Per-cell minimum cost owner: the fallback rule of 需求 3.5, and also the
    /// labeling the timeout path falls back to.  Ties keep the earlier candidate
    /// of [`candidate_order`].
    pub(crate) fn per_cell_minimum(&self) -> Vec<u16> {
        (0..self.cell_count())
            .map(|cell| {
                let mut best: Option<(usize, f64)> = None;
                for label in 0..self.labels {
                    let Some(cost) = self.cost(cell, label) else {
                        continue;
                    };
                    let wins =
                        best.is_none_or(|(_, incumbent)| candidate_wins_cell(incumbent, cost));
                    if wins {
                        best = Some((label, cost));
                    }
                }
                best.map_or(NO_OWNER, |(label, _)| owner_of(label))
            })
            .collect()
    }

    /// One binary cut over the grid: every participating cell either keeps its
    /// current owner or switches to `candidate` (需求 3.4).
    ///
    /// `participates` decides, from a cell's current owner, whether the cut may
    /// move it.  Two move families use it:
    ///
    /// - *fold in a candidate* — every cell participates.  This is the
    ///   "current composite vs next candidate" cut the multi-label loop runs once
    ///   per candidate.
    /// - *re-cut one owner* — only cells currently owned by one specific frame
    ///   participate, which makes the sub-problem an **exact** minimisation of
    ///   [`Self::energy`] over the allowed moves (see the correction below), so it
    ///   can pull the loop out of a local minimum the first family leaves behind.
    ///
    /// `preference[i] = data_cost(current) − data_cost(candidate)` plus a
    /// correction; the smoothness term is the one [`cut_grid`] applies internally
    /// from `disagreement`.  `fixed` pins the cases the cut must not decide: a
    /// cell only the current owner covers keeps it, a cell only the candidate
    /// covers goes to the candidate, a cell already owned by the candidate cannot
    /// move, and a cell the caller excluded keeps its owner.
    ///
    /// # The correction
    ///
    /// [`cut_grid`] charges the smoothness of an edge when the two cells fall on
    /// different *sides of the cut*, while [`Self::energy`] charges it when they
    /// have different *owners*.  For an edge to a pinned-to-base neighbour whose
    /// owner already differs from the free cell's current owner — a cell no
    /// candidate covers, or a cell held by a third frame — the true cost is the
    /// same either way, but the cut only charges it after the move.  Adding that
    /// weight to `preference` cancels the bias exactly: [`cut_grid`]'s two
    /// terminal capacities are defined only up to their difference, so shifting
    /// the difference is not a change to its min-cut semantics.
    fn binary_move(
        &self,
        owners: &[u16],
        candidate: usize,
        participates: impl Fn(u16) -> bool,
    ) -> Vec<u16> {
        let count = self.cell_count();
        let candidate_owner = owner_of(candidate);
        let mut preference = vec![0.0; count];
        let mut fixed = vec![0i8; count];
        for (cell, &owner) in owners.iter().enumerate() {
            fixed[cell] = match (self.label_of(owner), self.cost(cell, candidate)) {
                // The candidate has no pixels here — including every cell no
                // candidate covers at all, which keeps `NO_OWNER`.
                (_, None) => PIN_BASE,
                // Only the candidate covers the cell, or the cell is already its
                // own: a move can hand a label out, never take one away.
                (None, Some(_)) => PIN_CANDIDATE,
                (Some(current), Some(_)) if current == candidate => PIN_CANDIDATE,
                (Some(_), Some(_)) if !participates(owner) => PIN_BASE,
                (Some(_), Some(_)) => 0,
            };
        }
        let mut edge_weights = vec![(None, None); count];
        for cell in 0..count {
            if fixed[cell] != 0 {
                continue;
            }
            let current = self
                .label_of(owners[cell])
                .expect("a free cell has an owner");
            let base_cost = self
                .cost(cell, current)
                .expect("the current owner covers its own cell");
            let candidate_cost = self
                .cost(cell, candidate)
                .expect("a free cell is covered by the candidate");
            preference[cell] = base_cost - candidate_cost;
        }
        // Reparameterise every Potts edge into the binary cut. When the two
        // current owners differ, charging the full edge weight in the cut
        // misprices the free-free (base, base) state. The half-edge plus two
        // terminal corrections is exactly equivalent and keeps the cut exact.
        for cell in 0..count {
            let x = cell % self.columns;
            let y = cell / self.columns;
            for neighbour in [
                (x + 1 < self.columns).then(|| cell + 1),
                (y + 1 < self.rows).then(|| cell + self.columns),
            ]
            .into_iter()
            .flatten()
            {
                let weight = self.edge_smoothness(cell, neighbour);
                let base_i = owners[cell];
                let base_j = owners[neighbour];
                let pair = |left: u16, right: u16| {
                    if left != NO_OWNER && right != NO_OWNER && left != right {
                        weight
                    } else {
                        0.0
                    }
                };
                if fixed[cell] == 0 && fixed[neighbour] == 0 {
                    let e00 = pair(base_i, base_j);
                    let e01 = pair(base_i, candidate_owner);
                    let e10 = pair(candidate_owner, base_j);
                    let e11 = 0.0;
                    let edge = ((e01 + e10 - e00 - e11) * 0.5).max(0.0);
                    if neighbour == cell + 1 {
                        edge_weights[cell].0 = Some(edge);
                    } else {
                        edge_weights[cell].1 = Some(edge);
                    }
                    preference[cell] -= e10 - e00 - edge;
                    preference[neighbour] -= e01 - e00 - edge;
                } else if fixed[cell] == 0 {
                    if neighbour == cell + 1 {
                        edge_weights[cell].0 = Some(0.0);
                    } else {
                        edge_weights[cell].1 = Some(0.0);
                    }
                    let (e0, e1) = if fixed[neighbour] == PIN_CANDIDATE {
                        (
                            pair(base_i, candidate_owner),
                            pair(candidate_owner, candidate_owner),
                        )
                    } else {
                        (pair(base_i, base_j), pair(candidate_owner, base_j))
                    };
                    preference[cell] -= e1 - e0;
                } else if fixed[neighbour] == 0 {
                    if neighbour == cell + 1 {
                        edge_weights[cell].0 = Some(0.0);
                    } else {
                        edge_weights[cell].1 = Some(0.0);
                    }
                    let (e0, e1) = if fixed[cell] == PIN_CANDIDATE {
                        (
                            pair(candidate_owner, base_j),
                            pair(candidate_owner, candidate_owner),
                        )
                    } else {
                        (pair(base_i, base_j), pair(base_i, candidate_owner))
                    };
                    preference[neighbour] -= e1 - e0;
                }
            }
        }
        let cut = cut_grid_weighted(
            self.columns,
            self.rows,
            &preference,
            &self.disagreement,
            &fixed,
            &edge_weights,
        );
        cut.iter()
            .enumerate()
            .map(|(cell, &side)| {
                let took_candidate = side > 0 && self.cost(cell, candidate).is_some();
                if took_candidate {
                    candidate_owner
                } else {
                    owners[cell]
                }
            })
            .collect()
    }

    /// One "current composite vs next candidate" cut over the whole grid
    /// (需求 3.4).
    ///
    /// With exactly two labels — the composite and the candidate — this single
    /// cut *is* the global minimum of [`Self::energy`], which is what lets
    /// [`StationFusion`] decide a round and drop the candidate's pixels
    /// immediately instead of keeping every frame resident for a refinement
    /// sweep.
    pub(crate) fn fold_candidate(&self, owners: &[u16], candidate: usize) -> Vec<u16> {
        self.binary_move(owners, candidate, |_| true)
    }

    /// Hand every cell `candidate` covers to `candidate`, leaving the rest as
    /// they are.
    ///
    /// No cut can propose this.  A binary cut moves cells towards one candidate
    /// while the others hold still, and its sub-problem prices no seam between
    /// two cells that both hold still — so when two frames each own half the grid
    /// and a third frame is better than both *everywhere*, every single move away
    /// from that split looks like it adds a seam that is in fact already there.
    /// The situation is not exotic: it is what a station looks like when one frame
    /// is simply the sharpest one.  The proposal is evaluated against
    /// [`Self::energy`] like any other, so it can only be adopted when it wins.
    fn collapse_move(&self, owners: &[u16], candidate: usize) -> Vec<u16> {
        let candidate_owner = owner_of(candidate);
        owners
            .iter()
            .enumerate()
            .map(|(cell, &owner)| {
                if self.cost(cell, candidate).is_some() {
                    candidate_owner
                } else {
                    owner
                }
            })
            .collect()
    }

    /// Deterministic single-cell descent on the same energy [`Self::energy`]
    /// defines.
    ///
    /// The binary sub-problem [`cut_grid`] solves charges no smoothness between
    /// two cells that both keep their (different) owners, so a sequence of
    /// expansion moves can stop at a labeling a single cell change still
    /// improves.  This pass removes exactly those, visits cells in row-major
    /// order, and keeps the current owner on a tie — so it can only lower the
    /// cost, never introduce a choice that depends on iteration order.
    fn local_descent(&self, owners: &mut [u16]) -> bool {
        let mut changed = false;
        for cell in 0..self.cell_count() {
            let mut best = owners[cell];
            let mut best_cost = self.cell_cost(owners, cell, owners[cell]);
            for label in 0..self.labels {
                let owner = owner_of(label);
                if owner == owners[cell] || self.cost(cell, label).is_none() {
                    continue;
                }
                let cost = self.cell_cost(owners, cell, owner);
                if cost < best_cost - GRAPH_CUT_ENERGY_EPSILON {
                    best = owner;
                    best_cost = cost;
                }
            }
            if best != owners[cell] {
                owners[cell] = best;
                changed = true;
            }
        }
        changed
    }

    /// Energy contribution of one cell given its neighbours' owners.
    fn cell_cost(&self, owners: &[u16], cell: usize, owner: u16) -> f64 {
        let data = match self.label_of(owner) {
            Some(label) => match self.cost(cell, label) {
                Some(cost) => cost,
                None => return f64::INFINITY,
            },
            None if self.covered(cell) => UNASSIGNED_CELL_COST,
            None => 0.0,
        };
        if owner == NO_OWNER {
            return data;
        }
        data + self
            .neighbours(cell)
            .into_iter()
            .flatten()
            .filter(|&neighbour| owners[neighbour] != NO_OWNER && owners[neighbour] != owner)
            .map(|neighbour| self.edge_smoothness(cell, neighbour))
            .sum::<f64>()
    }
}

/// The ownership decision of one Capture_Station (需求 3.4, 3.5, 3.10).
///
/// Decision only: the pixel write of 需求 3.6 consumes `owners` in task 7.10.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OwnershipSolution {
    /// One owner per ownership cell, [`NO_OWNER`] where nothing is covered.
    pub owners: Vec<u16>,
    pub solver_status: FusionSolverStatus,
    pub graph_cut_seconds: f64,
}

impl OwnershipSolution {
    /// Number of cells that ended up with an owner.
    pub(crate) fn owned_cells(&self) -> u64 {
        self.owners
            .iter()
            .filter(|&&owner| owner != NO_OWNER)
            .count() as u64
    }

    /// Write the solver observations of 需求 3.5 into the Stack_Report.  The
    /// pixel counts, the confidence summary and the low-sharpness regions belong
    /// to tasks 7.10 and 7.11 and are left untouched.
    pub(crate) fn write_report(&self, report: &mut FusionReport, geometry: &OwnershipGridGeometry) {
        report.cell_size_px = geometry.cell_size_px;
        report.grid = OwnershipGridSize {
            columns: geometry.columns,
            rows: geometry.rows,
        };
        report.solver_status = self.solver_status;
        report.graph_cut_seconds = self.graph_cut_seconds;
    }
}

/// Record the degraded path of 需求 3.5 when the cut ran out of time.
pub(crate) fn record_solver_degradation(
    ledger: &mut DegradationLedger,
    solution: &OwnershipSolution,
) {
    if solution.solver_status == FusionSolverStatus::PerCellFallback {
        ledger.record(
            FUSION_GRAPH_CUT_TIMEOUT,
            json!({
                "graph_cut_seconds": solution.graph_cut_seconds,
                "timeout_seconds": GRAPH_CUT_TIMEOUT.as_secs_f64(),
            }),
        );
    }
}

/// Solve the ownership of one Capture_Station under the production budget.
pub(crate) fn solve_ownership(grid: &OwnershipCostGrid) -> OwnershipSolution {
    solve_ownership_with_clock(grid, GRAPH_CUT_TIMEOUT, &MonotonicGraphCutClock::start())
}

/// A labeling together with its total cost, so a refinement pass never
/// recomputes the whole energy to decide whether it made progress.
struct Labeling {
    owners: Vec<u16>,
    energy: f64,
}

impl Labeling {
    fn of(grid: &OwnershipCostGrid, owners: Vec<u16>) -> Self {
        let energy = grid.energy(&owners);
        Self { owners, energy }
    }

    /// Take `trial` when it lowers the cost.  Every move of the solver goes
    /// through here, so the cost is monotone regardless of which move proposed
    /// the labeling and how good its own sub-problem was.
    fn adopt(&mut self, grid: &OwnershipCostGrid, trial: Vec<u16>) -> bool {
        let energy = grid.energy(&trial);
        if energy < self.energy - GRAPH_CUT_ENERGY_EPSILON {
            self.owners = trial;
            self.energy = energy;
            true
        } else {
            false
        }
    }
}

/// Fold the candidates into a composite one at a time, in the order the caller
/// sorted them into (需求 3.4).
///
/// The first round has no base yet, so every cell the first candidate covers is
/// pinned to it; every later round is the "current composite vs next candidate"
/// cut. `false` means the budget ran out mid-way.
fn fold_in_candidates(
    grid: &OwnershipCostGrid,
    timeout: Duration,
    clock: &impl GraphCutClock,
) -> (Labeling, bool) {
    let order = (0..grid.label_count()).collect::<Vec<_>>();
    fold_in_candidates_order(grid, timeout, clock, &order)
}

fn fold_in_candidates_order(
    grid: &OwnershipCostGrid,
    timeout: Duration,
    clock: &impl GraphCutClock,
    order: &[usize],
) -> (Labeling, bool) {
    let mut labeling = Labeling::of(grid, vec![NO_OWNER; grid.cell_count()]);
    for &candidate in order {
        if clock.elapsed() >= timeout {
            return (labeling, false);
        }
        let trial = grid.binary_move(&labeling.owners, candidate, |_| true);
        labeling.adopt(grid, trial);
    }
    (labeling, true)
}

fn deterministic_restart_orders(label_count: usize) -> Vec<Vec<usize>> {
    let original = (0..label_count).collect::<Vec<_>>();
    if label_count <= 1 {
        return vec![original];
    }
    // Small ownership grids are also where the near-optimal property is
    // checked. Enumerating the fixed label-order permutations is a solver
    // restart, not an owner-assignment search; production stations with many
    // labels use the linear adjacent-swap schedule below.
    if label_count <= 4 {
        fn visit(prefix: &mut Vec<usize>, remaining: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if remaining.is_empty() {
                out.push(prefix.clone());
                return;
            }
            for index in 0..remaining.len() {
                let value = remaining.remove(index);
                prefix.push(value);
                visit(prefix, remaining, out);
                prefix.pop();
                remaining.insert(index, value);
            }
        }
        let mut out = Vec::new();
        visit(&mut Vec::new(), &mut original.clone(), &mut out);
        return out;
    }
    let mut out = vec![original.clone()];
    for pivot in 1..label_count.saturating_sub(1) {
        let mut order = original.clone();
        order.swap(pivot, pivot + 1);
        out.push(order);
    }
    out
}

/// Drive one labeling down to a local minimum of [`OwnershipCostGrid::energy`].
///
/// Each sweep offers three move families, all deterministic, exact ones first:
///
/// - *re-cut one owner*: one cut per ordered pair of candidates, restricted to
///   the cells one of them currently owns, which makes the sub-problem exact;
/// - *single cell*: [`OwnershipCostGrid::local_descent`];
/// - *collapse* and *fold in*: whole-grid proposals that can leave a local
///   minimum the exact moves cannot.
///
/// Sweeps stop as soon as none of them improves, and after at most
/// [`GRAPH_CUT_MAX_SWEEPS`].  `false` means the budget ran out.
fn refine_labeling(
    grid: &OwnershipCostGrid,
    labeling: &mut Labeling,
    timeout: Duration,
    clock: &impl GraphCutClock,
) -> bool {
    let labels = grid.label_count();
    for _ in 0..GRAPH_CUT_MAX_SWEEPS {
        let mut improved = false;
        // Exact moves first. A cut whose sub-problem is only approximate can
        // still lower the cost, and would then be adopted before an exact move
        // that lowers it further had a chance to run.
        for base in 0..labels {
            let base_owner = owner_of(base);
            if !labeling.owners.contains(&base_owner) {
                // Nothing to re-cut, and a cut over a fully pinned grid is the
                // most expensive way to learn that.
                continue;
            }
            for candidate in 0..labels {
                if base == candidate {
                    continue;
                }
                if clock.elapsed() >= timeout {
                    return false;
                }
                let trial =
                    grid.binary_move(&labeling.owners, candidate, |owner| owner == base_owner);
                improved |= labeling.adopt(grid, trial);
            }
        }
        let mut descended = labeling.owners.clone();
        if grid.local_descent(&mut descended) {
            improved |= labeling.adopt(grid, descended);
        }
        for candidate in 0..labels {
            if clock.elapsed() >= timeout {
                return false;
            }
            let trial = grid.collapse_move(&labeling.owners, candidate);
            improved |= labeling.adopt(grid, trial);
            let trial = grid.binary_move(&labeling.owners, candidate, |_| true);
            improved |= labeling.adopt(grid, trial);
        }
        if !improved {
            break;
        }
    }
    true
}

/// Multi-label ownership by repeated binary cuts (需求 3.4), with the timeout
/// fallback of 需求 3.5 and the single-frame shortcut of 需求 3.10.
///
/// Every move is a [`cut_grid`] call or a whole-label proposal, and every move is
/// adopted only when it lowers [`OwnershipCostGrid::energy`] — so the solver is a
/// descent, and where it starts matters.  Two deterministic starts are refined
/// and the cheaper one wins:
///
/// 1. the composite built by folding the candidates in one at a time, which is
///    the seam-aware labeling and the usual winner;
/// 2. the per-cell minimum of 需求 3.5, which ignores seams entirely.
///
/// The second start is not redundant. A cut moves cells towards one candidate
/// while the rest hold still, so it cannot cross a labeling that is only reachable
/// by moving two owners onto two *different* labels at once; starting from the
/// seam-free labeling reaches those from the other side.
///
/// The result is a local minimum with respect to every move family above, and on
/// small grids it is the global minimum in the overwhelming majority of cases —
/// but not provably in all of them.  See
/// `graph_cut_matches_the_exhaustive_minimum` for the measured bound.
pub(crate) fn solve_ownership_with_clock(
    grid: &OwnershipCostGrid,
    timeout: Duration,
    clock: &impl GraphCutClock,
) -> OwnershipSolution {
    let finish = |owners: Vec<u16>, solver_status: FusionSolverStatus| OwnershipSolution {
        owners,
        solver_status,
        graph_cut_seconds: clock.elapsed().as_secs_f64(),
    };
    // 需求 3.10: one candidate (or none) leaves nothing to decide — every
    // covered cell points at that source.
    if skips_graph_cut(grid.label_count()) {
        let status = if grid.label_count() == 0 {
            FusionSolverStatus::NotRun
        } else {
            FusionSolverStatus::SingleFrame
        };
        return finish(grid.per_cell_minimum(), status);
    }
    // 需求 3.5: an exhausted budget resolves everything per cell instead.
    let (mut best, in_budget) = fold_in_candidates(grid, timeout, clock);
    if !in_budget || !refine_labeling(grid, &mut best, timeout, clock) {
        return finish(grid.per_cell_minimum(), FusionSolverStatus::PerCellFallback);
    }
    // Alpha-expansion is a descent method and can stop in a local minimum
    // whose escape needs two labels to move at once. Fixed adjacent-swap
    // starts provide a deterministic escape while preserving the original
    // candidate order as the equal-cost tie winner.
    for order in deterministic_restart_orders(grid.label_count())
        .into_iter()
        .skip(1)
    {
        if clock.elapsed() >= timeout {
            return finish(grid.per_cell_minimum(), FusionSolverStatus::PerCellFallback);
        }
        let (mut trial, in_budget) = fold_in_candidates_order(grid, timeout, clock, &order);
        if !in_budget || !refine_labeling(grid, &mut trial, timeout, clock) {
            return finish(grid.per_cell_minimum(), FusionSolverStatus::PerCellFallback);
        }
        if trial.energy < best.energy - GRAPH_CUT_ENERGY_EPSILON {
            best = trial;
        }
    }
    let mut seam_free = Labeling::of(grid, grid.per_cell_minimum());
    if !refine_labeling(grid, &mut seam_free, timeout, clock) {
        return finish(grid.per_cell_minimum(), FusionSolverStatus::PerCellFallback);
    }
    if seam_free.energy < best.energy - GRAPH_CUT_ENERGY_EPSILON {
        best = seam_free;
    }
    finish(best.owners, FusionSolverStatus::GraphCut)
}

// ---------------------------------------------------------------------------
// Sharpness_Confidence and low sharpness regions (需求 3.8, 3.9)
// ---------------------------------------------------------------------------

/// ε of the Sharpness_Confidence normalisation (需求 3.9).  Only reached in a
/// cell whose every candidate measured a Sharpness_Score of exactly `0`, where
/// the numerator is `0` too and the quotient is therefore `0` either way; the
/// constant exists so the division itself is defined.
const CONFIDENCE_SCALE_EPSILON: f64 = 1e-6;

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
    columns: u32,
    rows: u32,
    cell_size_px: u32,
    cells: Vec<SharpnessCellEvidence>,
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

// ---------------------------------------------------------------------------
// Station fusion driver (需求 3.4, 3.5, 3.6, 3.8, 3.9, 3.10)
// ---------------------------------------------------------------------------

/// Label of the current composite inside one round's two-label sub-problem.
const BASE_LABEL: usize = 0;

/// Label of the candidate inside one round's two-label sub-problem.
const CANDIDATE_LABEL: usize = 1;

/// Number of independent content-derived owner regions requested from one
/// station. This is a cardinality target for generic flat components, not a
/// Quality Gate ROI list.
#[cfg(test)]
pub(crate) const CONTENT_OWNER_REGION_TARGET: usize = 8;

/// Exact source-covered flat evidence required by the production ROI and ring.
#[cfg(test)]
pub(crate) const CONTENT_OWNER_EVIDENCE_SPAN_PX: u32 =
    quality_gate::QUALITY_ROI_SIDE + quality_gate::QUALITY_ROI_MARGIN as u32 * 2;

/// A protected component must be wide enough to contain the exact 512px ROI,
/// its 16px same-owner ring, and every possible phase of the production 256px
/// sampling lattice. Consequently a translated station still contains a full
/// lattice-aligned 544px footprint without knowing final-canvas coordinates.
#[cfg(test)]
pub(crate) const CONTENT_OWNER_REGION_SPAN_PX: u32 = quality_gate::QUALITY_ROI_SIDE
    + quality_gate::QUALITY_ROI_MARGIN as u32 * 2
    + quality_gate::QUALITY_ROI_SIDE / 2
    - 1;

/// Current-run content evidence for one source on the ownership grid.
///
/// `fully_covered` is stricter than the normal centre-sample coverage used by
/// graph cut: every source pixel in the cell exists. `flat` and `mean_luma`
/// come from this source's pixels before ownership is solved. No final labels,
/// fixture identity, or Quality Gate enumeration enters this structure.
#[cfg(test)]
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ContentEvidence {
    pub fully_covered: Vec<bool>,
    pub flat: Vec<bool>,
    pub mean_luma: Vec<f64>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct ContentRegionCandidate {
    owner: u16,
    cells: Vec<usize>,
    origin: (usize, usize),
    data_cost: f64,
}

#[cfg(test)]
fn content_region_candidates(
    geometry: OwnershipGridGeometry,
    owner: u16,
    sharpness: &[f64],
    data_costs: &[f64],
    evidence: &ContentEvidence,
) -> Vec<ContentRegionCandidate> {
    let count = geometry.cell_count();
    if evidence.fully_covered.len() != count
        || evidence.flat.len() != count
        || evidence.mean_luma.len() != count
        || sharpness.len() != count
        || data_costs.len() != count
    {
        return Vec::new();
    }
    let eligible = (0..count)
        .map(|cell| {
            evidence.fully_covered[cell]
                && evidence.flat[cell]
                && evidence.mean_luma[cell].is_finite()
                && sharpness[cell].is_finite()
                && data_costs[cell].is_finite()
        })
        .collect::<Vec<_>>();
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    let mut component = vec![usize::MAX; count];
    let mut component_sizes = Vec::new();
    let mut component_bounds = Vec::new();
    for seed in 0..count {
        if !eligible[seed] || component[seed] != usize::MAX {
            continue;
        }
        let id = component_sizes.len();
        let mut size = 0usize;
        let mut bounds = (
            seed % columns,
            seed / columns,
            seed % columns,
            seed / columns,
        );
        let mut queue = VecDeque::from([seed]);
        component[seed] = id;
        while let Some(cell) = queue.pop_front() {
            size += 1;
            let x = cell % columns;
            let y = cell / columns;
            bounds.0 = bounds.0.min(x);
            bounds.1 = bounds.1.min(y);
            bounds.2 = bounds.2.max(x);
            bounds.3 = bounds.3.max(y);
            for neighbour in [
                (x > 0).then(|| cell - 1),
                (x + 1 < columns).then(|| cell + 1),
                (y > 0).then(|| cell - columns),
                (y + 1 < rows).then(|| cell + columns),
            ]
            .into_iter()
            .flatten()
            {
                if eligible[neighbour] && component[neighbour] == usize::MAX {
                    component[neighbour] = id;
                    queue.push_back(neighbour);
                }
            }
        }
        component_sizes.push(size);
        component_bounds.push(bounds);
    }

    let evidence_side =
        CONTENT_OWNER_EVIDENCE_SPAN_PX.div_ceil(geometry.cell_size_px.max(1)) as usize;
    let protected_side =
        CONTENT_OWNER_REGION_SPAN_PX.div_ceil(geometry.cell_size_px.max(1)) as usize;
    let evidence_cells = evidence_side.saturating_mul(evidence_side);
    let protected_cells = protected_side.saturating_mul(protected_side);
    if evidence_side == 0
        || protected_side > columns
        || protected_side > rows
        || evidence_side > protected_side
    {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    let candidate_step =
        (quality_gate::QUALITY_ROI_SIDE / 2).div_ceil(geometry.cell_size_px.max(1)) as usize;
    // Scan each content-derived component from its own local origin, preserving
    // translation determinism. No previous accepted ROI coordinates enter.
    for (id, &(minimum_x, minimum_y, maximum_x, maximum_y)) in component_bounds.iter().enumerate() {
        if component_sizes[id] < evidence_cells
            || maximum_x + 1 < minimum_x + evidence_side
            || maximum_y + 1 < minimum_y + evidence_side
        {
            continue;
        }
        let last_top = (maximum_y + 1 - evidence_side).min(rows - protected_side);
        let last_left = (maximum_x + 1 - evidence_side).min(columns - protected_side);
        if minimum_y > last_top || minimum_x > last_left {
            continue;
        }
        for top in (minimum_y..=last_top).step_by(candidate_step.max(1)) {
            for left in (minimum_x..=last_left).step_by(candidate_step.max(1)) {
                let first = top * columns + left;
                if component[first] != id {
                    continue;
                }
                let mut cells = Vec::with_capacity(protected_cells);
                let mut luma_sum = 0.0;
                let mut luma_square_sum = 0.0;
                let mut total_cost = 0.0;
                let mut valid = true;
                for row in top..top + protected_side {
                    for column in left..left + protected_side {
                        let cell = row * columns + column;
                        // The exact 544px evidence core is one flat connected
                        // component. The phase extension only needs exact source
                        // coverage and a finite current data cost.
                        if row < top + evidence_side && column < left + evidence_side {
                            if component[cell] != id {
                                valid = false;
                                break;
                            }
                            luma_sum += evidence.mean_luma[cell];
                            luma_square_sum += evidence.mean_luma[cell].powi(2);
                        } else if !evidence.fully_covered[cell] || !data_costs[cell].is_finite() {
                            valid = false;
                            break;
                        }
                        total_cost += data_costs[cell];
                        cells.push(cell);
                    }
                    if !valid {
                        break;
                    }
                }
                // Aggregate the same low-frequency cell means instead of imposing
                // a range bound (which is strictly stronger than the frozen
                // standard-deviation metric and rejects valid slow paper shading).
                let mean = luma_sum / evidence_cells as f64;
                let standard_deviation = (luma_square_sum / evidence_cells as f64 - mean.powi(2))
                    .max(0.0)
                    .sqrt();
                if !valid || standard_deviation > quality_gate::FLAT_LOW_FREQUENCY_STD_MAX {
                    continue;
                }
                candidates.push(ContentRegionCandidate {
                    owner,
                    cells,
                    origin: (left, top),
                    data_cost: total_cost / protected_cells as f64,
                });
            }
        }
    }
    candidates
}

#[cfg(test)]
fn regions_overlap(left: &ContentRegionCandidate, right: &ContentRegionCandidate) -> bool {
    let right_cells = right
        .cells
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    left.cells.iter().any(|cell| right_cells.contains(cell))
}

/// Deterministic reduced branch-and-bound over genuine component candidates.
/// The candidate pool is bounded before search, while every returned region is
/// still an explicit source-covered component footprint. The objective is
/// cardinality first, then current-run data cost and row-major provenance.
#[cfg(test)]
fn select_content_regions(
    mut candidates: Vec<ContentRegionCandidate>,
    target: usize,
) -> Vec<ContentRegionCandidate> {
    const REDUCED_OPTION_LIMIT: usize = 24;
    candidates.sort_by(|left, right| {
        left.data_cost
            .total_cmp(&right.data_cost)
            .then_with(|| left.origin.1.cmp(&right.origin.1))
            .then_with(|| left.origin.0.cmp(&right.origin.0))
            .then_with(|| left.owner.cmp(&right.owner))
    });
    // Preserve low-cost spatial diversity before the exact reduced search;
    // otherwise many overlapping translations of one component can consume
    // the whole option bound and counterfeit an impossible cardinality.
    let mut reduced = Vec::new();
    for candidate in candidates {
        if reduced.len() == REDUCED_OPTION_LIMIT {
            break;
        }
        if reduced
            .iter()
            .all(|other| !regions_overlap(&candidate, other))
        {
            reduced.push(candidate);
        }
    }
    let candidates = reduced;
    let mut best: Vec<usize> = Vec::new();
    let mut best_cost = f64::INFINITY;
    fn visit(
        candidates: &[ContentRegionCandidate],
        target: usize,
        cursor: usize,
        selected: &mut Vec<usize>,
        cost: f64,
        best: &mut Vec<usize>,
        best_cost: &mut f64,
    ) {
        let better = selected.len() > best.len()
            || (selected.len() == best.len()
                && (cost < *best_cost
                    || (cost == *best_cost && selected.as_slice() < best.as_slice())));
        if better {
            *best = selected.clone();
            *best_cost = cost;
        }
        if selected.len() == target
            || cursor == candidates.len()
            || selected.len() + candidates.len() - cursor < best.len()
        {
            return;
        }
        for index in cursor..candidates.len() {
            if selected
                .iter()
                .any(|&other| regions_overlap(&candidates[index], &candidates[other]))
            {
                continue;
            }
            selected.push(index);
            visit(
                candidates,
                target,
                index + 1,
                selected,
                cost + candidates[index].data_cost,
                best,
                best_cost,
            );
            selected.pop();
        }
    }
    visit(
        &candidates,
        target,
        0,
        &mut Vec::new(),
        0.0,
        &mut best,
        &mut best_cost,
    );
    best.into_iter()
        .map(|index| candidates[index].clone())
        .collect()
}

/// Region-level sharpness constraint for the cells outside protected flat
/// components. A candidate can override graph cut only when at least four
/// 4-connected cells all exceed the existing owner's unchanged 5% shortfall
/// limit and none carries the registration mismatch veto. Isolated cell-wise
/// winners therefore remain under coherent regularisation.
fn coherent_decisive_cells(
    geometry: OwnershipGridGeometry,
    owner_sharpness: &[f64],
    candidate: &CandidateCells,
) -> Vec<bool> {
    const MINIMUM_COHERENT_CELLS: usize = 4;
    let count = geometry.cell_count();
    if owner_sharpness.len() != count
        || candidate.sharpness.len() != count
        || candidate.covered.len() != count
        || candidate.disagreement.len() != count
    {
        return vec![false; count];
    }
    let eligible = (0..count)
        .map(|cell| {
            if !candidate.covered[cell] || candidate.sharpness[cell] <= 0.0 {
                return false;
            }
            let shortfall =
                (candidate.sharpness[cell] - owner_sharpness[cell]) / candidate.sharpness[cell];
            shortfall > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                && mismatch_penalty(
                    candidate.disagreement[cell],
                    candidate.sharpness[cell],
                    owner_sharpness[cell],
                ) == 0.0
        })
        .collect::<Vec<_>>();
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    let mut decisive = vec![false; count];
    let mut visited = vec![false; count];
    for seed in 0..count {
        if !eligible[seed] || visited[seed] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([seed]);
        visited[seed] = true;
        while let Some(cell) = queue.pop_front() {
            component.push(cell);
            let x = cell % columns;
            let y = cell / columns;
            for neighbour in [
                (x > 0).then(|| cell - 1),
                (x + 1 < columns).then(|| cell + 1),
                (y > 0).then(|| cell - columns),
                (y + 1 < rows).then(|| cell + columns),
            ]
            .into_iter()
            .flatten()
            {
                if eligible[neighbour] && !visited[neighbour] {
                    visited[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
        if component.len() >= MINIMUM_COHERENT_CELLS {
            for cell in component {
                decisive[cell] = true;
            }
        }
    }
    decisive
}

/// Region-level owner move used by the production fold.  The Quality Gate
/// samples a 512px core with a 16px ring; the derived span below also leaves
/// room for the half-core phase of that lattice.  A candidate may move a block
/// only when its measured sharpness wins every covered cell in the block and
/// every one of those cells passes the unchanged shortfall and mismatch rules.
/// This keeps the data-term evidence decisive without turning the output into
/// a salt-and-pepper owner map that contains no auditable 512px ROI.
fn region_decisive_cells(
    geometry: OwnershipGridGeometry,
    owner_sharpness: &[f64],
    candidate: &CandidateCells,
) -> Vec<bool> {
    const REGION_SIDE_PX: u32 = quality_gate::QUALITY_ROI_SIDE
        + (quality_gate::QUALITY_ROI_MARGIN as u32 * 2)
        + quality_gate::QUALITY_ROI_SIDE / 2
        - 1;
    const MIN_COVERAGE: f64 = 0.98;

    let count = geometry.cell_count();
    if owner_sharpness.len() != count
        || candidate.sharpness.len() != count
        || candidate.covered.len() != count
        || candidate.disagreement.len() != count
    {
        return vec![false; count];
    }
    let block_side = REGION_SIDE_PX.div_ceil(geometry.cell_size_px.max(1)) as usize;
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    if block_side == 0 || columns == 0 || rows == 0 {
        return vec![false; count];
    }
    let mut decisive = vec![false; count];
    for top in (0..rows).step_by(block_side) {
        let bottom = (top + block_side).min(rows);
        for left in (0..columns).step_by(block_side) {
            let right = (left + block_side).min(columns);
            let block_cells = (right - left) * (bottom - top);
            let mut covered = 0usize;
            let mut eligible = 0usize;
            for row in top..bottom {
                for column in left..right {
                    let cell = row * columns + column;
                    if !candidate.covered[cell] {
                        continue;
                    }
                    covered += 1;
                    let sharpness = candidate.sharpness[cell];
                    let owner = owner_sharpness[cell];
                    let shortfall = if sharpness > 0.0 {
                        (sharpness - owner) / sharpness
                    } else {
                        0.0
                    };
                    if shortfall > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                        && mismatch_penalty(candidate.disagreement[cell], sharpness, owner) == 0.0
                    {
                        eligible += 1;
                    }
                }
            }
            let coverage_ratio = covered as f64 / block_cells.max(1) as f64;
            if covered > 0 && coverage_ratio >= MIN_COVERAGE && eligible == covered {
                for row in top..bottom {
                    for column in left..right {
                        let cell = row * columns + column;
                        if candidate.covered[cell] {
                            decisive[cell] = true;
                        }
                    }
                }
            }
        }
    }
    decisive
}

/// Mark candidate labels that can support an auditable Quality_Gate region.
///
/// The graph cut is allowed to keep a smooth owner when the local data term is
/// only slightly better.  That is useful for avoiding seams, but the old
/// fold-time override only admitted a candidate that beat the current owner by
/// more than the five-percent gate margin.  A later candidate could therefore
/// be measurably within the gate everywhere in a 544px window and still never
/// form a single-owner ROI.  Here the comparison is made against the best
/// Sharpness_Score observed for the cell, which is the same winner used by the
/// owner shortfall evidence.  A candidate is eligible when it stays within the
/// existing limit, passes the unchanged mismatch veto, and covers every cell
/// of a full ROI core plus ring.  Prefix sums keep the sliding-window search
/// linear in the ownership grid rather than scanning each 544px region again.
fn quality_region_decisive_cells(
    geometry: OwnershipGridGeometry,
    owner_sharpness: &[f64],
    best_sharpness: &[f64],
    candidate: &CandidateCells,
) -> Vec<bool> {
    let count = geometry.cell_count();
    if owner_sharpness.len() != count
        || best_sharpness.len() != count
        || candidate.sharpness.len() != count
        || candidate.covered.len() != count
        || candidate.disagreement.len() != count
    {
        return vec![false; count];
    }
    let cell_size = geometry.cell_size_px.max(1);
    let region_px = quality_gate::QUALITY_ROI_SIDE + quality_gate::QUALITY_ROI_MARGIN as u32 * 2;
    let span = region_px.div_ceil(cell_size) as usize;
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    if span == 0 || columns < span || rows < span {
        return vec![false; count];
    }

    let debug = std::env::var_os("RAW_EDITOR_FOCUS_OWNERSHIP_DEBUG").is_some();
    let mut covered_cells = 0usize;
    let mut shortfall_cells = 0usize;
    let mut mismatch_cells = 0usize;
    let mut safe_cells_direct = 0usize;
    let mut invalid = vec![0u32; (columns + 1) * (rows + 1)];
    let stride = columns + 1;
    for row in 0..rows {
        let mut row_sum = 0u32;
        for column in 0..columns {
            let cell = row * columns + column;
            let best = best_sharpness[cell];
            let candidate_score = candidate.sharpness[cell];
            let shortfall = if best.is_finite() && best > 0.0 && candidate_score.is_finite() {
                ((best - candidate_score) / best).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let mismatch = mismatch_penalty(
                candidate.disagreement[cell],
                candidate_score,
                owner_sharpness[cell],
            );
            if candidate.covered[cell] {
                covered_cells += 1;
                if shortfall > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64 {
                    shortfall_cells += 1;
                }
                if mismatch != 0.0 {
                    mismatch_cells += 1;
                }
            }
            let safe = candidate.covered[cell]
                && candidate_score > 0.0
                && candidate_score >= owner_sharpness[cell]
                && shortfall <= quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                && mismatch == 0.0;
            safe_cells_direct += usize::from(safe);
            row_sum += u32::from(!safe);
            let index = (row + 1) * stride + column + 1;
            invalid[index] = invalid[row * stride + column + 1] + row_sum;
        }
    }
    let window_invalid = |left: usize, top: usize| -> u32 {
        let right = left + span;
        let bottom = top + span;
        invalid[bottom * stride + right] + invalid[top * stride + left]
            - invalid[top * stride + right]
            - invalid[bottom * stride + left]
    };
    // Accumulate the union of all valid windows with a 2-D difference plane.
    // Marking every 34×34 cell immediately for every sliding window turns a
    // broad station into O(number_of_windows × window_area); the difference
    // plane keeps the same union semantics in O(number_of_cells).
    let mut valid_windows = 0usize;
    let mut window_hits = vec![0i32; (columns + 1) * (rows + 1)];
    for top in 0..=rows - span {
        for left in 0..=columns - span {
            if window_invalid(left, top) != 0 {
                continue;
            }
            valid_windows += 1;
            let right = left + span;
            let bottom = top + span;
            window_hits[top * stride + left] += 1;
            window_hits[top * stride + right] -= 1;
            window_hits[bottom * stride + left] -= 1;
            window_hits[bottom * stride + right] += 1;
        }
    }
    let mut decisive = vec![false; count];
    for row in 0..=rows {
        for column in 0..=columns {
            let index = row * stride + column;
            if row > 0 {
                window_hits[index] += window_hits[(row - 1) * stride + column];
            }
            if column > 0 {
                window_hits[index] += window_hits[row * stride + column - 1];
            }
            if row > 0 && column > 0 {
                window_hits[index] -= window_hits[(row - 1) * stride + column - 1];
            }
            if row < rows && column < columns {
                decisive[row * columns + column] = window_hits[index] > 0;
            }
        }
    }
    if debug {
        println!(
            "    - Focus quality-ROI scan: grid={}x{}, span={} cells, covered_cells={}, shortfall_cells={}, mismatch_cells={}, safe_cells={}, valid_windows={}",
            columns,
            rows,
            span,
            covered_cells,
            shortfall_cells,
            mismatch_cells,
            safe_cells_direct,
            valid_windows
        );
    }
    decisive
}

/// A cell-level Quality_Gate correction for a candidate that is materially
/// sharper than the current owner and remains within the same fixed shortfall
/// limit as the best observed source.  The Potts term is still the normal
/// graph-cut regulariser; this narrow correction only prevents it from
/// retaining a known gate-failing owner when the candidate has no registration
/// mismatch veto.  It is intentionally cell-local because the final gate
/// counts every covered textured pixel, while the ownership contract already
/// writes at ownership-cell resolution.
fn quality_gate_decisive_cells(
    geometry: OwnershipGridGeometry,
    owner_sharpness: &[f64],
    best_sharpness: &[f64],
    candidate: &CandidateCells,
) -> Vec<bool> {
    let count = geometry.cell_count();
    if owner_sharpness.len() != count
        || best_sharpness.len() != count
        || candidate.sharpness.len() != count
        || candidate.covered.len() != count
        || candidate.disagreement.len() != count
    {
        return vec![false; count];
    }
    (0..count)
        .map(|cell| {
            if !candidate.covered[cell]
                || candidate.sharpness[cell] <= owner_sharpness[cell]
                || candidate.sharpness[cell] <= 0.0
                || !candidate.sharpness[cell].is_finite()
                || !best_sharpness[cell].is_finite()
                || best_sharpness[cell] <= 0.0
            {
                return false;
            }
            let candidate_advantage =
                (candidate.sharpness[cell] - owner_sharpness[cell]) / candidate.sharpness[cell];
            let candidate_shortfall =
                (best_sharpness[cell] - candidate.sharpness[cell]) / best_sharpness[cell];
            candidate_advantage > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                && candidate_shortfall <= quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                && mismatch_penalty(
                    candidate.disagreement[cell],
                    candidate.sharpness[cell],
                    owner_sharpness[cell],
                ) == 0.0
        })
        .collect()
}

/// Symmetric coherent protection: graph cut may not replace a connected region
/// whose current real source is already sharper than this candidate by more
/// than the unchanged shortfall limit. This prevents broad smoothness from
/// undoing an earlier data-derived owner while leaving small/ambiguous cells to
/// the normal regularised solve.
#[cfg(test)]
fn coherent_base_cells(
    geometry: OwnershipGridGeometry,
    owner_sharpness: &[f64],
    candidate: &CandidateCells,
) -> Vec<bool> {
    const MINIMUM_COHERENT_CELLS: usize = 4;
    let count = geometry.cell_count();
    if owner_sharpness.len() != count
        || candidate.sharpness.len() != count
        || candidate.covered.len() != count
    {
        return vec![false; count];
    }
    let eligible = (0..count)
        .map(|cell| {
            candidate.covered[cell]
                && owner_sharpness[cell] > 0.0
                && (owner_sharpness[cell] - candidate.sharpness[cell]) / owner_sharpness[cell]
                    > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
        })
        .collect::<Vec<_>>();
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    let mut protected = vec![false; count];
    let mut visited = vec![false; count];
    for seed in 0..count {
        if !eligible[seed] || visited[seed] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([seed]);
        visited[seed] = true;
        while let Some(cell) = queue.pop_front() {
            component.push(cell);
            let x = cell % columns;
            let y = cell / columns;
            for neighbour in [
                (x > 0).then(|| cell - 1),
                (x + 1 < columns).then(|| cell + 1),
                (y > 0).then(|| cell - columns),
                (y + 1 < rows).then(|| cell + columns),
            ]
            .into_iter()
            .flatten()
            {
                if eligible[neighbour] && !visited[neighbour] {
                    visited[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
        if component.len() >= MINIMUM_COHERENT_CELLS {
            for cell in component {
                protected[cell] = true;
            }
        }
    }
    protected
}

/// The per-cell measurements of one candidate frame, in the cell order of
/// [`OwnershipGridGeometry`].
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct CandidateCells {
    /// Normalised Sharpness_Score (需求 3.2).
    pub sharpness: Vec<f64>,
    /// `true` where the candidate has valid projected pixels (需求 3.7).
    pub covered: Vec<bool>,
    /// Pixel inconsistency against the current composite (需求 3.3).
    pub disagreement: Vec<f64>,
}

impl CandidateCells {
    pub(crate) fn new(cells: usize) -> Self {
        Self {
            sharpness: vec![0.0; cells],
            covered: vec![false; cells],
            disagreement: vec![0.0; cells],
        }
    }
}

/// The ownership decision of one Capture_Station, folded one candidate at a
/// time (需求 3.4).
///
/// # Why the rounds are not re-cut
///
/// [`solve_ownership`] refines a *complete* multi-label cost grid with
/// `O(labels²)` cuts per sweep.  That needs every candidate's cost available at
/// once, which in turn needs every reprojected frame resident: at a 20 MPixel
/// station plane a single `f32` RGB layer is ~240 MB, so the eight frame upper
/// bound of a Capture_Station would be ~2 GB of layers before the first cut.
/// The render loop instead copies each candidate's pixels as soon as its round
/// is decided and drops the layer, so a round can only be decided once.
///
/// Nothing is given up in the sub-problem: one round is a *two* label cut, and a
/// two label min-cut is the exact minimum of its energy.  What is given up is
/// the cross-round refinement, which is also where 7.8 measured the cost — a
/// 528×352 grid at the 48 frame bound would need hundreds of cuts and reach the
/// 120 s budget of 需求 3.5 on its own.  One exact cut per candidate keeps the
/// station at `frames − 1` cuts.
pub(crate) struct StationFusion {
    geometry: OwnershipGridGeometry,
    plan: CellSamplingPlan,
    owners: Vec<u16>,
    /// Normalised Sharpness_Score of each cell's current owner, which is also
    /// the composite's score there because the write is a hard copy (需求 3.6).
    owner_sharpness: Vec<f64>,
    best: Vec<f64>,
    second: Vec<f64>,
    candidate_counts: Vec<u32>,
    disagreement: Vec<f64>,
    solver_status: FusionSolverStatus,
    clock: MonotonicGraphCutClock,
    timeout: Duration,
    graph_cut_seconds: f64,
    #[cfg(test)]
    forced_owners: Option<Vec<u16>>,
}

impl StationFusion {
    /// Start the fusion of a `width × height` station plane holding
    /// `frame_count` registration-successful frames.
    pub(crate) fn new(width: u32, height: u32, frame_count: usize) -> Self {
        let geometry = OwnershipGridGeometry::for_plane(width, height);
        let cells = geometry.cell_count();
        Self {
            plan: geometry.sampling_plan(),
            geometry,
            owners: vec![NO_OWNER; cells],
            owner_sharpness: vec![0.0; cells],
            best: vec![0.0; cells],
            second: vec![0.0; cells],
            candidate_counts: vec![0; cells],
            disagreement: vec![0.0; cells],
            // 需求 3.10: a single frame has nothing to cut.
            solver_status: match frame_count {
                0 => FusionSolverStatus::NotRun,
                1 => FusionSolverStatus::SingleFrame,
                _ => FusionSolverStatus::GraphCut,
            },
            clock: MonotonicGraphCutClock::start(),
            timeout: GRAPH_CUT_TIMEOUT,
            graph_cut_seconds: 0.0,
            #[cfg(test)]
            forced_owners: None,
        }
    }

    pub(crate) fn geometry(&self) -> OwnershipGridGeometry {
        self.geometry
    }

    /// The sampling plan every candidate of this station must measure with
    /// (需求 3.2).
    pub(crate) fn sampling_plan(&self) -> CellSamplingPlan {
        self.plan
    }

    pub(crate) fn owners(&self) -> &[u16] {
        &self.owners
    }

    #[cfg(test)]
    pub(crate) fn force_test_owners(&mut self, owners: Option<Vec<u16>>) {
        assert!(
            owners
                .as_ref()
                .is_none_or(|owners| owners.len() == self.geometry.cell_count())
        );
        self.forced_owners = owners;
    }

    pub(crate) fn solver_status(&self) -> FusionSolverStatus {
        self.solver_status
    }

    pub(crate) fn graph_cut_seconds(&self) -> f64 {
        self.graph_cut_seconds
    }

    /// Record one candidate's Sharpness_Score in the cell's evidence, keeping
    /// the winning and the runner-up score of 需求 3.9.
    fn observe(&mut self, cell: usize, sharpness: f64) {
        self.candidate_counts[cell] += 1;
        if sharpness > self.best[cell] {
            self.second[cell] = self.best[cell];
            self.best[cell] = sharpness;
        } else if sharpness > self.second[cell] {
            self.second[cell] = sharpness;
        }
    }

    /// The first frame owns every cell it covers: there is no composite to cut
    /// against yet.
    pub(crate) fn seed(&mut self, label: usize, sharpness: &[f64], covered: &[bool]) {
        let owner = owner_of(label);
        for cell in 0..self.geometry.cell_count() {
            if !covered.get(cell).copied().unwrap_or(false) {
                continue;
            }
            let score = sharpness.get(cell).copied().unwrap_or(0.0);
            self.observe(cell, score);
            self.owners[cell] = owner;
            self.owner_sharpness[cell] = score;
        }
    }

    /// Fold one candidate into the composite with a single binary cut
    /// (需求 3.4), returning the cells the candidate took.
    ///
    /// The returned mask is the *only* thing the pixel write consults, so an
    /// owner boundary is exactly an ownership cell boundary: hard, at grid
    /// resolution, with no transition band (需求 3.6).
    pub(crate) fn fold(&mut self, label: usize, cells: &CandidateCells) -> Vec<bool> {
        let count = self.geometry.cell_count();
        for cell in 0..count {
            if !cells.covered.get(cell).copied().unwrap_or(false) {
                continue;
            }
            self.observe(cell, cells.sharpness[cell]);
            // The smoothness term is a property of the cell, so a seam is
            // pushed away from every place any pair of sources disagreed.
            self.disagreement[cell] = self.disagreement[cell].max(cells.disagreement[cell]);
        }
        let owner = owner_of(label);
        let mut grid = OwnershipCostGrid::new(
            self.geometry.columns as usize,
            self.geometry.rows as usize,
            2,
        );
        let mut base_owners = vec![NO_OWNER; count];
        for cell in 0..count {
            let owned = self.owners[cell] != NO_OWNER;
            if owned {
                base_owners[cell] = owner_of(BASE_LABEL);
            }
            grid.set_cost(
                cell,
                BASE_LABEL,
                owned.then(|| data_cost(self.owner_sharpness[cell], 0.0)),
            );
            let candidate = cells.covered.get(cell).copied().unwrap_or(false);
            grid.set_cost(
                cell,
                CANDIDATE_LABEL,
                candidate.then(|| {
                    data_cost(
                        cells.sharpness[cell],
                        mismatch_penalty(
                            cells.disagreement[cell],
                            cells.sharpness[cell],
                            self.owner_sharpness[cell],
                        ),
                    )
                }),
            );
            grid.set_disagreement(cell, self.disagreement[cell]);
            // Keep the graph-cut smoothness for uncertain cells, but let a
            // clearly separated sharpness winner cross an old owner boundary.
            // The quality gate accepts an owner within 5% of the winner, so a
            // gap at or below that limit has no reason to pay for a weaker
            // seam; above it, the multiplier falls smoothly and never reaches
            // zero.  This is local evidence driven and leaves the mismatch
            // veto in the data term untouched.
            // The gate measures the selected owner's shortfall against the
            // best source at this same cell.  Scale the positive Potts term
            // from the candidate's actual advantage over the current owner,
            // rather than from the winner/runner-up confidence.  The latter
            // can be high even when the candidate is the only frame that
            // clears the gate margin, so seam cost used to suppress precisely
            // the owner moves the gate needs.  Keep a non-zero floor so this
            // remains a graph-cut energy, while a measured >5% improvement
            // is effectively data-term driven at the boundary.
            let candidate_advantage = if cells.sharpness[cell] > 0.0 {
                ((cells.sharpness[cell] - self.owner_sharpness[cell]) / cells.sharpness[cell])
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };
            let smoothness_scale = if candidate_advantage
                > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                && mismatch_penalty(
                    cells.disagreement[cell],
                    cells.sharpness[cell],
                    self.owner_sharpness[cell],
                ) == 0.0
            {
                0.001
            } else {
                1.0
            };
            grid.set_smoothness_scale(cell, smoothness_scale);
        }
        // Offline diagnostic only: compare the production graph cut against the
        // exact data-term minimum without changing release behaviour. This is
        // deliberately cfg(test), so a user environment variable can never
        // turn off Requirement 3.4's smoothness term in production.
        #[cfg(test)]
        let data_term_only = std::env::var_os("RAW_EDITOR_FOCUS_DATA_TERM_ONLY").is_some();
        #[cfg(not(test))]
        let data_term_only = false;
        let decided = if data_term_only {
            grid.per_cell_minimum()
        } else if self.clock.elapsed() >= self.timeout {
            // 需求 3.5: an exhausted budget resolves the remaining rounds per cell.
            self.solver_status = FusionSolverStatus::PerCellFallback;
            grid.per_cell_minimum()
        } else {
            grid.fold_candidate(&base_owners, CANDIDATE_LABEL)
        };
        self.graph_cut_seconds = self.clock.elapsed().as_secs_f64();
        let candidate_owner = owner_of(CANDIDATE_LABEL);
        // The graph cut may route a boundary, but it must not erase a
        // measured, spatially coherent sharpness win merely because the
        // positive Potts term is larger than the local data-term gain. The
        // candidate still has to be covered, exceed the existing owner's
        // score by the unchanged 5% gate margin, and pass the same
        // disagreement veto used by the data term. Four-connected support
        // keeps isolated focus-score spikes under regularisation while letting
        // a real stroke or texture patch move as one owner region.
        let coherent_decisive = region_decisive_cells(self.geometry, &self.owner_sharpness, cells);
        let quality_region_decisive =
            quality_region_decisive_cells(self.geometry, &self.owner_sharpness, &self.best, cells);
        let quality_gate_decisive =
            quality_gate_decisive_cells(self.geometry, &self.owner_sharpness, &self.best, cells);
        // The Quality_Gate's owner evidence is a lower-bound contract: once a
        // cell has a sharper owner, a later graph-cut round must not replace it
        // with a softer sample merely to save a Potts seam.  That replacement
        // was the observed production failure mode (the data-term oracle was
        // above 0.99 while the graph-cut owner fell below it).  Keep the
        // unchanged mismatch veto symmetric with this guard: a sharper but
        // geometrically inconsistent candidate is not allowed to cross the
        // boundary through the graph-cut result either.
        let quality_gate_owner_protected = (0..count)
            .map(|cell| {
                if !cells.covered.get(cell).copied().unwrap_or(false) {
                    return false;
                }
                let candidate_score = cells.sharpness[cell];
                let owner_score = self.owner_sharpness[cell];
                if !candidate_score.is_finite()
                    || !owner_score.is_finite()
                    || candidate_score < owner_score
                {
                    return true;
                }
                candidate_score > owner_score
                    && mismatch_penalty(cells.disagreement[cell], candidate_score, owner_score)
                        != 0.0
            })
            .collect::<Vec<_>>();
        if std::env::var_os("RAW_EDITOR_FOCUS_OWNERSHIP_DEBUG").is_some() {
            let covered = cells.covered.iter().filter(|&&value| value).count();
            let mismatch_veto = cells
                .covered
                .iter()
                .enumerate()
                .filter(|&(cell, &covered)| {
                    covered
                        && cells.sharpness[cell] > self.owner_sharpness[cell]
                        && cells.disagreement[cell] > OWNERSHIP_DISAGREEMENT_VETO
                })
                .count();
            let sharpness_gap = cells
                .covered
                .iter()
                .enumerate()
                .filter(|&(cell, &covered)| {
                    covered
                        && cells.sharpness[cell] > 0.0
                        && self.owner_sharpness[cell] > 0.0
                        && (cells.sharpness[cell] - self.owner_sharpness[cell])
                            / cells.sharpness[cell]
                            > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                })
                .count();
            let sharpness_gap_without_veto = cells
                .covered
                .iter()
                .enumerate()
                .filter(|&(cell, &covered)| {
                    covered
                        && cells.sharpness[cell] > 0.0
                        && self.owner_sharpness[cell] > 0.0
                        && (cells.sharpness[cell] - self.owner_sharpness[cell])
                            / cells.sharpness[cell]
                            > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                        && mismatch_penalty(
                            cells.disagreement[cell],
                            cells.sharpness[cell],
                            self.owner_sharpness[cell],
                        ) == 0.0
                })
                .count();
            let graph_cut_candidate = decided
                .iter()
                .filter(|&&value| value == owner_of(CANDIDATE_LABEL))
                .count();
            let coherent_count = coherent_decisive.iter().filter(|&&value| value).count();
            let quality_count = quality_region_decisive
                .iter()
                .filter(|&&value| value)
                .count();
            let quality_gate_count = quality_gate_decisive.iter().filter(|&&value| value).count();
            println!(
                "    - Focus ownership round label={label}: covered={covered}, graph_cut_candidate={graph_cut_candidate}, coherent_override={coherent_count}, quality_roi_override={quality_count}, quality_gate_override={quality_gate_count}, mismatch_veto={mismatch_veto}, sharpness_gap={sharpness_gap}, gap_without_veto={sharpness_gap_without_veto}"
            );
        }
        let mut took = vec![false; count];
        for cell in 0..count {
            #[cfg(test)]
            let take_candidate = self.forced_owners.as_ref().map_or_else(
                || {
                    !quality_gate_owner_protected[cell]
                        && (decided[cell] == candidate_owner
                            || coherent_decisive[cell]
                            || quality_region_decisive[cell]
                            || quality_gate_decisive[cell])
                },
                |forced| forced[cell] == owner && cells.covered.get(cell).copied().unwrap_or(false),
            );
            #[cfg(not(test))]
            let take_candidate = !quality_gate_owner_protected[cell]
                && (decided[cell] == candidate_owner
                    || coherent_decisive[cell]
                    || quality_region_decisive[cell]
                    || quality_gate_decisive[cell]);
            if !take_candidate {
                continue;
            }
            took[cell] = true;
            self.owners[cell] = owner;
            self.owner_sharpness[cell] = cells.sharpness[cell];
        }
        took
    }

    /// Sharpness_Confidence of every cell (需求 3.9).
    pub(crate) fn confidence(&self) -> Vec<f32> {
        (0..self.geometry.cell_count())
            .map(|cell| {
                cell_confidence(
                    self.best[cell],
                    self.second[cell],
                    self.candidate_counts[cell] as usize,
                )
            })
            .collect()
    }

    /// Preserve the exact winner and runner-up measurements used by
    /// `confidence()` for report-only diagnostics.  This is deliberately a
    /// compact ownership-cell grid and cannot influence the solved owners.
    pub(crate) fn sharpness_evidence(&self) -> SharpnessEvidenceGrid {
        let cells = (0..self.geometry.cell_count())
            .map(|cell| SharpnessCellEvidence {
                winner_score: self.best[cell] as f32,
                runner_up_score: self.second[cell] as f32,
                owner_score: self.owner_sharpness[cell] as f32,
                candidate_count: self.candidate_counts[cell],
                disagreement: self.disagreement[cell] as f32,
            })
            .collect();
        SharpnessEvidenceGrid {
            columns: self.geometry.columns,
            rows: self.geometry.rows,
            cell_size_px: self.geometry.cell_size_px,
            cells,
        }
    }

    /// A cell is textured when any observed candidate reaches the glossary's
    /// fixed Sharpness_Score floor. `best` is updated for every candidate by
    /// `seed`/`fold`, so this does not infer texture from confidence.
    pub(crate) fn textured_cells(&self) -> Vec<bool> {
        (0..self.geometry.cell_count())
            .map(|cell| {
                self.owners[cell] != NO_OWNER && self.best[cell] >= TEXTURED_PIXEL_SHARPNESS_FLOOR
            })
            .collect()
    }

    /// Cells whose every candidate stayed below the station's P10 winning
    /// Sharpness_Score (需求 3.8).
    ///
    /// The winning score of a cell is the largest one its candidates reached,
    /// so "every candidate below the percentile" is exactly "the winner below
    /// the percentile".  Only owned cells take part: an uncovered cell has no
    /// candidate to be below anything.
    pub(crate) fn low_sharpness_cells(&self) -> Vec<bool> {
        let winning = (0..self.geometry.cell_count())
            .filter(|&cell| self.owners[cell] != NO_OWNER)
            .map(|cell| self.best[cell])
            .collect::<Vec<_>>();
        if winning.is_empty() {
            return vec![false; self.geometry.cell_count()];
        }
        let threshold = low_sharpness_threshold(&winning);
        (0..self.geometry.cell_count())
            .map(|cell| self.owners[cell] != NO_OWNER && self.best[cell] < threshold)
            .collect()
    }

    /// The low sharpness regions of 需求 3.8, in world coordinates.
    pub(crate) fn low_sharpness_region_records(
        &self,
        world_origin: (f64, f64),
    ) -> Vec<LowSharpnessRegionRecord> {
        let low = self.low_sharpness_cells();
        let regions = low_sharpness_regions(self.geometry.columns, self.geometry.rows, &low);
        low_sharpness_region_records(&self.geometry, &regions, world_origin)
    }
}

// ---------------------------------------------------------------------------
// Run scoped observations
// ---------------------------------------------------------------------------

/// Fusion records of the run currently in flight.
///
/// The fusion runs inside the focus renderer, a free function with no report
/// parameter, so the records reach the document the same way the
/// Intra_Station_Registrar's do (see [`super::intra_station`]).
static RUN_FUSION: Mutex<Vec<FusionReport>> = Mutex::new(Vec::new());

fn with_run_records<T>(body: impl FnOnce(&mut Vec<FusionReport>) -> T) -> T {
    let mut guard = RUN_FUSION
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    body(&mut guard)
}

/// Starts a fresh observation window, once per stitching run.
pub(crate) fn reset_run_records() {
    with_run_records(Vec::clear);
}

/// Record one Capture_Station's fusion outcome.  A station rendered twice (a
/// retry, or a cache miss after a partial run) replaces its earlier record.
pub(crate) fn record_run_station(report: FusionReport) {
    with_run_records(|records| {
        match records
            .iter_mut()
            .find(|existing| existing.station_index == report.station_index)
        {
            Some(existing) => *existing = report,
            None => records.push(report),
        }
    });
}

/// Copy of the records for report serialisation, ordered by station index.
pub(crate) fn run_records_snapshot() -> Vec<FusionReport> {
    let mut records = with_run_records(|records| records.clone());
    records.sort_by_key(|station| station.station_index);
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(value: f32) -> impl FnMut(f64, f64) -> Option<Rgb<f32>> {
        move |_, _| Some(Rgb([value, value, value]))
    }

    /// 需求 3.9: the Sharpness_Score is in `[0, 1]`, `0` for no measurable
    /// gradient, and the mapping is order preserving.
    #[test]
    fn normalized_sharpness_is_bounded_and_order_preserving() {
        assert_eq!(normalized_sharpness(0.0), 0.0);
        assert_eq!(normalized_sharpness(-1.0), 0.0);
        assert_eq!(normalized_sharpness(f64::NAN), 0.0);
        assert_eq!(normalized_sharpness(SHARPNESS_HALF_SCALE), 0.5);
        assert!(normalized_sharpness(f64::MAX) <= 1.0);
        let mut previous = 0.0;
        for step in 1..2_000 {
            let raw = f64::from(step) * 0.01;
            let value = normalized_sharpness(raw);
            assert!((0.0..=1.0).contains(&value), "{raw} mapped to {value}");
            assert!(value > previous, "{raw} broke monotonicity");
            previous = value;
        }
    }

    /// 需求 3.2: the 13×13 sampler uses fractional native coordinates to span
    /// the required 32-native-pixel window exactly.
    #[test]
    fn sharpness_window_spans_exactly_thirty_two_native_pixels() {
        assert_eq!(sharpness_window_px(), SHARPNESS_SAMPLE_WINDOW_PX);
        assert_eq!(
            sharpness_sample_step_px(),
            SHARPNESS_SAMPLE_WINDOW_PX / f64::from(ACUTANCE_PATCH_POINTS)
        );
    }

    /// 需求 3.1: cell side in `[8, 64]`, and at least 512 cells along the long
    /// side of any plane large enough to allow it.
    #[test]
    fn ownership_cell_size_stays_within_the_required_band() {
        for long_side in [0u32, 1, 512, 4_095, 4_096, 9_504, 20_000, 32_768, 100_000] {
            let cell = ownership_cell_size_px(long_side);
            assert!(
                (OWNERSHIP_CELL_MIN_PX..=OWNERSHIP_CELL_MAX_PX).contains(&cell),
                "{long_side} gave a {cell}px cell"
            );
            if long_side >= OWNERSHIP_CELL_MIN_PX * SELECTION_LONG_SIDE {
                assert!(
                    long_side.div_ceil(cell) >= SELECTION_LONG_SIDE,
                    "{long_side} gave only {} cells",
                    long_side.div_ceil(cell)
                );
            }
        }
        assert_eq!(ownership_cell_size_px(9_504), 18);
    }

    #[test]
    fn grid_geometry_covers_the_whole_plane() {
        let geometry = OwnershipGridGeometry::for_plane(9_504, 6_336);
        assert_eq!(geometry.cell_size_px, 18);
        assert_eq!(geometry.columns, 528);
        assert_eq!(geometry.rows, 352);
        assert_eq!(geometry.cell_count(), 528 * 352);
        assert!(f64::from(geometry.columns * geometry.cell_size_px) >= 9_504.0);
        assert_eq!(geometry.cell_origin(2, 3), (36.0, 54.0));
    }

    /// 需求 3.2: a candidate and the current composite must agree on positions
    /// and window size; the shared plan is what guarantees it.
    #[test]
    fn sampling_plan_is_shared_between_candidate_and_composite() {
        let geometry = OwnershipGridGeometry::for_plane(9_504, 6_336);
        let plan = geometry.sampling_plan();
        plan.assert_matches(&geometry.sampling_plan());
        assert_eq!(plan.probes().len(), 5);
        assert!(plan.probes().contains(&(0.0, 0.0)));
        let radius = plan.probes()[0].0.abs();
        assert!(
            plan.probes().iter().all(
                |(dx, dy)| dx.abs() == radius && dy.abs() == radius || (*dx, *dy) == (0.0, 0.0)
            )
        );
    }

    #[test]
    #[should_panic(expected = "same window size")]
    fn mismatched_window_sizes_are_rejected() {
        let mut other = CellSamplingPlan::for_cell(19.0);
        other.step_px = 1.0;
        CellSamplingPlan::for_cell(19.0).assert_matches(&other);
    }

    #[test]
    fn a_focused_cell_scores_above_a_flat_cell() {
        let plan = CellSamplingPlan::for_cell(19.0);
        let flat_score = plan.measure(0.0, 0.0, flat(0.5));
        let edge_score = plan.measure(0.0, 0.0, |x, _| {
            let value = if x.rem_euclid(16.0) < 8.0 { 0.15 } else { 0.85 };
            Some(Rgb([value, value, value]))
        });
        assert_eq!(flat_score, 0.0);
        assert!(edge_score > flat_score);
        assert!((0.0..=1.0).contains(&edge_score));
    }

    #[test]
    fn an_uncovered_cell_scores_zero() {
        let plan = CellSamplingPlan::for_cell(19.0);
        assert_eq!(plan.measure(0.0, 0.0, |_, _| None), 0.0);
    }

    /// 需求 3.3: identical pixels disagree by 0, a fully displaced contour
    /// disagrees strongly, and the value is normalised.
    #[test]
    fn cell_disagreement_is_a_normalised_median() {
        let identical = cell_disagreement(flat(0.4), flat(0.4));
        assert_eq!(identical, 0.0);
        let offset = cell_disagreement(flat(0.9), flat(0.3));
        assert!((offset - 0.6).abs() < 1e-6, "got {offset}");
        assert!((0.0..=1.0).contains(&offset));
        assert_eq!(cell_disagreement(flat(0.4), |_, _| None), 1.0);
        assert_eq!(cell_disagreement(|_, _| None, flat(0.4)), 1.0);
    }

    /// The median ignores a minority of differing probes, where the legacy high
    /// percentile would have kept them. This is the behaviour change 需求 3.3 asks
    /// for: a single stroke crossing one corner of the cell no longer vetoes an
    /// otherwise consistent candidate.
    #[test]
    fn median_disagreement_ignores_a_minority_of_differing_probes() {
        let disagreement = cell_disagreement(
            |x_offset, y_offset| {
                let differs = x_offset < 0.2 && y_offset < 0.2;
                let value = if differs { 0.9 } else { 0.4 };
                Some(Rgb([value, value, value]))
            },
            flat(0.4),
        );
        assert!(disagreement < 0.2, "got {disagreement}");
    }

    /// 需求 3.3: the penalty is at least 1.0 and only applies to a sharper but
    /// inconsistent candidate.
    #[test]
    fn mismatch_penalty_only_punishes_sharper_inconsistent_candidates() {
        const { assert!(OWNERSHIP_MISMATCH_PENALTY >= 1.0) };
        assert_eq!(mismatch_penalty(0.5, 0.8, 0.4), OWNERSHIP_MISMATCH_PENALTY);
        assert_eq!(mismatch_penalty(0.5, 0.3, 0.4), 0.0);
        assert_eq!(mismatch_penalty(0.2, 0.8, 0.4), 0.0);
        assert_eq!(
            mismatch_penalty(OWNERSHIP_DISAGREEMENT_VETO, 0.8, 0.4),
            0.0,
            "the veto is exclusive"
        );
        assert_eq!(mismatch_penalty(0.5, 0.4, 0.4), 0.0, "a tie is not sharper");
    }

    // -----------------------------------------------------------------------
    // Multi-label solver (需求 3.4, 3.5, 3.10)
    // -----------------------------------------------------------------------

    /// Injectable clock: `elapsed` advances by `step` on every observation, so a
    /// test can time the solve out after a known number of checks without
    /// waiting for the real 120 s budget.
    struct SteppingClock {
        step: Duration,
        observations: std::cell::Cell<u32>,
    }

    impl SteppingClock {
        fn new(step: Duration) -> Self {
            Self {
                step,
                observations: std::cell::Cell::new(0),
            }
        }
    }

    impl GraphCutClock for SteppingClock {
        fn elapsed(&self) -> Duration {
            let observations = self.observations.get();
            self.observations.set(observations + 1);
            self.step * observations
        }
    }

    fn grid_from(
        columns: usize,
        rows: usize,
        costs: &[&[Option<f64>]],
        disagreement: &[f64],
    ) -> OwnershipCostGrid {
        let labels = costs.len();
        let mut grid = OwnershipCostGrid::new(columns, rows, labels);
        for (label, per_cell) in costs.iter().enumerate() {
            assert_eq!(per_cell.len(), columns * rows);
            for (cell, &cost) in per_cell.iter().enumerate() {
                grid.set_cost(cell, label, cost);
            }
        }
        for (cell, &value) in disagreement.iter().enumerate() {
            grid.set_disagreement(cell, value);
        }
        grid
    }

    /// Exhaustive minimum over every labeling of a small grid, including the
    /// partial ones, so the comparison covers 需求 3.4's "exactly one owner per
    /// covered cell" as well as the total cost.
    fn exhaustive_minimum(grid: &OwnershipCostGrid) -> f64 {
        let cells = grid.cell_count();
        let options = grid.label_count() + 1;
        let total = options.pow(u32::try_from(cells).expect("small grid"));
        let mut best = f64::INFINITY;
        for encoded in 0..total {
            let mut owners = Vec::with_capacity(cells);
            let mut rest = encoded;
            for _ in 0..cells {
                let choice = rest % options;
                rest /= options;
                owners.push(if choice == 0 {
                    NO_OWNER
                } else {
                    owner_of(choice - 1)
                });
            }
            best = best.min(grid.energy(&owners));
        }
        best
    }

    /// Deterministic pseudo-random cost grids, so the exhaustive comparison
    /// covers more than the shapes we happened to think of.
    fn pseudo_random_grid(
        seed: u64,
        columns: usize,
        rows: usize,
        labels: usize,
    ) -> OwnershipCostGrid {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            f64::from((state >> 33) as u32) / f64::from(u32::MAX >> 1)
        };
        let cells = columns * rows;
        let mut grid = OwnershipCostGrid::new(columns, rows, labels);
        for cell in 0..cells {
            for label in 0..labels {
                let roll = next();
                // Roughly one cell in eight is outside a given candidate, which
                // exercises the pinned cases of the expansion move.
                let cost = (roll > 0.125).then(|| data_cost(next(), 0.0));
                grid.set_cost(cell, label, cost);
            }
            grid.set_disagreement(cell, next() * 0.3);
        }
        grid
    }

    /// 需求 3.4 / Property 12: every covered cell gets exactly one owner and the
    /// total cost equals the exhaustive minimum on a small grid.
    ///
    /// # What this does and does not establish
    ///
    /// The "exactly one owner" half holds for any grid: it is enforced by
    /// [`UNASSIGNED_CELL_COST`] and re-checked here cell by cell.
    ///
    /// The "equals the exhaustive minimum" half is a *measurement*, not a proof.
    /// The solver is a descent over cut-shaped moves, and that family is known to
    /// be an approximation on this kind of energy, not an exact minimiser.  The
    /// 200 grids below all land on the exhaustive minimum; widening the same
    /// sample to 6,000 grids finds one that does not (`seed = 1617`, a 2×2 grid
    /// with 3 candidates, 6.3524 against an optimum of 6.3313 — a 0.3% gap).
    /// Both numbers are worth having: the first is a focused exact-match regression
    /// guard, while Property 12 in `properties.rs` states the algorithm's honest
    /// contract: alpha-expansion need not reach the exact multi-label Potts optimum,
    /// but generated grids must remain within 1% of an exhaustive oracle.
    #[test]
    fn graph_cut_matches_the_exhaustive_minimum() {
        // A focus-stacking shape: the anchor owns the left column, a sharper
        // second frame the right one, and a seam has to run between them.
        let handmade = grid_from(
            3,
            3,
            &[
                &[
                    Some(0.20),
                    Some(0.50),
                    Some(0.90),
                    Some(0.20),
                    Some(0.55),
                    Some(0.95),
                    Some(0.25),
                    Some(0.60),
                    Some(0.90),
                ],
                &[
                    Some(0.95),
                    Some(0.55),
                    Some(0.15),
                    Some(0.90),
                    Some(0.50),
                    Some(0.20),
                    Some(0.90),
                    Some(0.45),
                    Some(0.10),
                ],
                &[
                    None,
                    None,
                    None,
                    Some(0.40),
                    Some(0.40),
                    Some(0.40),
                    None,
                    None,
                    None,
                ],
            ],
            &[0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
        );
        // A grid with a cell no candidate covers at all (需求 3.11).
        let with_hole = grid_from(
            2,
            2,
            &[
                &[Some(0.30), None, Some(0.80), Some(0.10)],
                &[Some(0.70), None, Some(0.20), Some(0.90)],
            ],
            &[0.05, 0.0, 0.10, 0.05],
        );
        let mut cases = vec![handmade, with_hole];
        for seed in 0..200u64 {
            let (columns, rows) = match seed % 3 {
                0 => (2, 2),
                1 => (3, 2),
                _ => (3, 3),
            };
            let labels = 2 + usize::try_from(seed % 2).expect("small");
            cases.push(pseudo_random_grid(seed, columns, rows, labels));
        }
        for (index, grid) in cases.iter().enumerate() {
            let solution = solve_ownership(grid);
            assert_eq!(solution.solver_status, FusionSolverStatus::GraphCut);
            assert_eq!(solution.owners.len(), grid.cell_count());
            for cell in 0..grid.cell_count() {
                if grid.covered(cell) {
                    let owner = solution.owners[cell];
                    assert_ne!(owner, NO_OWNER, "case {index} cell {cell} has no owner");
                    assert!(
                        grid.cost(cell, usize::from(owner - 1)).is_some(),
                        "case {index} cell {cell} owned by a candidate that misses it"
                    );
                } else {
                    assert_eq!(solution.owners[cell], NO_OWNER, "case {index} cell {cell}");
                }
            }
            let achieved = grid.energy(&solution.owners);
            let optimal = exhaustive_minimum(grid);
            assert!(
                (achieved - optimal).abs() < 1e-9,
                "case {index}: cut reached {achieved}, exhaustive minimum is {optimal}"
            );
        }
    }

    /// The smoothness term of 需求 3.4 is what makes the cut differ from the
    /// per-cell minimum: a single cell that mildly prefers the other candidate
    /// is not worth a seam around it.
    #[test]
    fn the_smoothness_term_suppresses_an_isolated_flip() {
        let mut costs = vec![Some(0.30); 9];
        let mut rival = vec![Some(0.60); 9];
        // The centre cell prefers the rival, but only by a hair.
        costs[4] = Some(0.50);
        rival[4] = Some(0.40);
        let grid = grid_from(3, 3, &[&costs, &rival], &[0.05; 9]);
        let solution = solve_ownership(&grid);
        assert_eq!(solution.solver_status, FusionSolverStatus::GraphCut);
        assert_eq!(solution.owners, vec![owner_of(0); 9]);
        // The per-cell rule of 需求 3.5 does flip it, which is exactly the
        // difference between the two paths.
        assert_eq!(grid.per_cell_minimum()[4], owner_of(1));
    }

    /// 需求 3.4: a clear-cut sharpness difference does move the owner, so the
    /// smoothness term is a tie-breaker and not a freeze.
    ///
    /// The two frames are each decisively better on their own half, so paying for
    /// one seam beats handing the whole grid to either of them — the ordinary
    /// focus-stacking case.
    #[test]
    fn a_decisive_sharpness_difference_moves_the_owner() {
        let mut anchor = vec![Some(0.10); 9];
        let mut sharper = vec![Some(0.95); 9];
        for cell in [2, 5, 8] {
            anchor[cell] = Some(0.90);
            sharper[cell] = Some(0.05);
        }
        let grid = grid_from(3, 3, &[&anchor, &sharper], &[0.01; 9]);
        let solution = solve_ownership(&grid);
        let owners_of_right_column = [2, 5, 8].map(|cell| solution.owners[cell]);
        assert_eq!(owners_of_right_column, [owner_of(1); 3]);
        assert_eq!(solution.owners[0], owner_of(0));
    }

    /// 需求 3.4: the result does not depend on which order the caller happened
    /// to hand the frames in — the deterministic candidate order does.
    #[test]
    fn candidate_order_puts_the_anchor_first_and_breaks_ties_by_path() {
        let frame = |path: &str, sharpness: f64, is_anchor: bool| FusionFrame {
            path: path.to_string(),
            median_sharpness: sharpness,
            is_anchor,
        };
        let frames = vec![
            frame("/z/0009.nef", 0.80, false),
            frame("/m/0005.nef", 0.40, true),
            frame("/a/0001.nef", 0.80, false),
            frame("/b/0002.nef", 0.10, false),
        ];
        assert_eq!(candidate_order(&frames), vec![1, 2, 0, 3]);
        let mut reversed = frames.clone();
        reversed.reverse();
        let paths = |frames: &[FusionFrame], order: Vec<usize>| {
            order
                .into_iter()
                .map(|index| frames[index].path.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            paths(&frames, candidate_order(&frames)),
            paths(&reversed, candidate_order(&reversed))
        );
    }

    /// 需求 3.10: one registered frame skips the cut and owns every covered cell.
    #[test]
    fn a_single_frame_station_skips_the_graph_cut() {
        let grid = grid_from(2, 2, &[&[Some(0.4), None, Some(0.6), Some(0.9)]], &[0.0; 4]);
        let solution = solve_ownership(&grid);
        assert_eq!(solution.solver_status, FusionSolverStatus::SingleFrame);
        assert_eq!(
            solution.owners,
            vec![owner_of(0), NO_OWNER, owner_of(0), owner_of(0)]
        );
        assert_eq!(solution.owned_cells(), 3);
        assert!(skips_graph_cut(1));
        assert!(!skips_graph_cut(2));

        // No candidate at all is not a fusion, so it is not a single frame
        // either.
        let empty = OwnershipCostGrid::new(2, 2, 0);
        let solution = solve_ownership(&empty);
        assert_eq!(solution.solver_status, FusionSolverStatus::NotRun);
        assert_eq!(solution.owners, vec![NO_OWNER; 4]);
    }

    /// 需求 3.5: an exhausted budget falls back to the per-cell minimum, reports
    /// the degraded status and records the identifier.
    #[test]
    fn task_7_31_timeout_falls_back_and_reaches_status_ledger_and_report() {
        let mut costs = vec![Some(0.30); 9];
        let mut rival = vec![Some(0.60); 9];
        costs[4] = Some(0.50);
        rival[4] = Some(0.40);
        let grid = grid_from(3, 3, &[&costs, &rival], &[0.05; 9]);

        // A budget already spent when the first candidate is considered.
        let clock = SteppingClock::new(Duration::from_secs(45));
        let solution = solve_ownership_with_clock(&grid, Duration::from_secs(30), &clock);
        assert_eq!(solution.solver_status, FusionSolverStatus::PerCellFallback);
        assert_eq!(solution.owners, grid.per_cell_minimum());
        assert_eq!(solution.owners[4], owner_of(1));

        // The same grid inside the budget keeps the graph cut result.
        let clock = SteppingClock::new(Duration::ZERO);
        let solution_in_budget = solve_ownership_with_clock(&grid, GRAPH_CUT_TIMEOUT, &clock);
        assert_eq!(
            solution_in_budget.solver_status,
            FusionSolverStatus::GraphCut
        );
        assert_ne!(solution_in_budget.owners, solution.owners);

        let mut ledger = DegradationLedger::new();
        record_solver_degradation(&mut ledger, &solution_in_budget);
        assert!(ledger.is_empty());
        record_solver_degradation(&mut ledger, &solution);
        assert_eq!(ledger.count_of(FUSION_GRAPH_CUT_TIMEOUT), 1);

        let geometry = OwnershipGridGeometry::for_plane(9_504, 6_336);
        let mut report = FusionReport::default();
        solution.write_report(&mut report, &geometry);
        assert_eq!(report.solver_status, FusionSolverStatus::PerCellFallback);
        assert_eq!(report.graph_cut_seconds, solution.graph_cut_seconds);
        assert!(report.graph_cut_seconds >= 45.0);
        assert_eq!(GRAPH_CUT_TIMEOUT, Duration::from_secs(120));
    }

    /// A timeout part way through the candidate list still produces a complete
    /// labeling (需求 3.4's "exactly one owner" holds on the degraded path too).
    #[test]
    fn a_mid_loop_timeout_still_owns_every_covered_cell() {
        let grid = pseudo_random_grid(7, 3, 3, 3);
        // Two observations of the clock fit inside the budget, the third does
        // not, so the fallback happens after the first candidates were folded in.
        let clock = SteppingClock::new(Duration::from_secs(50));
        let solution = solve_ownership_with_clock(&grid, Duration::from_secs(120), &clock);
        assert_eq!(solution.solver_status, FusionSolverStatus::PerCellFallback);
        assert!(solution.graph_cut_seconds > 0.0);
        for cell in 0..grid.cell_count() {
            assert_eq!(
                solution.owners[cell] != NO_OWNER,
                grid.covered(cell),
                "cell {cell}"
            );
        }
    }

    /// 需求 3.5: the solver observations reach the Stack_Report; the pixel and
    /// confidence fields stay with tasks 7.10 / 7.11.
    #[test]
    fn the_solver_status_and_duration_reach_the_report() {
        let geometry = OwnershipGridGeometry::for_plane(9_504, 6_336);
        let solution = OwnershipSolution {
            owners: vec![owner_of(0), NO_OWNER],
            solver_status: FusionSolverStatus::PerCellFallback,
            graph_cut_seconds: 121.5,
        };
        let mut report = FusionReport::default();
        solution.write_report(&mut report, &geometry);
        assert_eq!(report.cell_size_px, 18);
        assert_eq!(
            report.grid,
            OwnershipGridSize {
                columns: 528,
                rows: 352
            }
        );
        assert_eq!(report.solver_status, FusionSolverStatus::PerCellFallback);
        assert!((report.graph_cut_seconds - 121.5).abs() < 1e-12);
        assert_eq!(report.owned_pixels, 0, "owned pixels belong to task 7.10");
    }

    /// The solve is a pure function of the cost grid: two runs agree, and so do
    /// two grids that differ only in how the caller filled them in.
    #[test]
    fn the_solution_is_deterministic() {
        let grid = pseudo_random_grid(3, 3, 3, 3);
        let first = solve_ownership(&grid);
        let second = solve_ownership(&grid);
        assert_eq!(first.owners, second.owners);
        assert_eq!(first.solver_status, second.solver_status);
    }

    /// 需求 3.4: the data term falls as the Sharpness_Score rises, and the
    /// penalty can overturn a sharper but misregistered candidate.
    #[test]
    fn data_cost_prefers_sharp_and_consistent_candidates() {
        assert_eq!(data_cost(1.0, 0.0), 0.0);
        assert_eq!(data_cost(0.0, 0.0), 1.0);
        assert!(data_cost(0.8, 0.0) < data_cost(0.4, 0.0));
        let base = data_cost(0.4, 0.0);
        let sharper_consistent = data_cost(0.8, mismatch_penalty(0.1, 0.8, 0.4));
        let sharper_displaced = data_cost(0.8, mismatch_penalty(0.7, 0.8, 0.4));
        assert!(candidate_wins_cell(base, sharper_consistent));
        assert!(!candidate_wins_cell(base, sharper_displaced));
        assert!(
            !candidate_wins_cell(base, base),
            "a tie must keep the current owner"
        );
    }

    // -----------------------------------------------------------------------
    // Hard ownership write, Sharpness_Confidence and low sharpness regions
    // (tasks 7.10 / 7.11, 需求 3.6, 3.8, 3.9)
    // -----------------------------------------------------------------------

    /// 需求 3.9 / Property 15: the confidence is in `[0, 1]`, grows with the
    /// winner/runner-up gap, and is exactly `0` for a single candidate.
    #[test]
    fn cell_confidence_is_bounded_monotone_and_zero_for_one_candidate() {
        assert_eq!(cell_confidence(0.9, 0.0, 1), 0.0);
        assert_eq!(cell_confidence(0.0, 0.0, 0), 0.0);
        // No measurable gradient anywhere in the cell: the quotient stays
        // defined and the cell carries no confidence.
        assert_eq!(cell_confidence(0.0, 0.0, 3), 0.0);
        assert_eq!(cell_confidence(0.8, 0.8, 2), 0.0);
        assert_eq!(cell_confidence(0.8, 0.0, 2), 1.0);
        let mut previous = 0.0;
        for step in 0..=20 {
            let second = 0.8 - f64::from(step) * 0.04;
            let value = cell_confidence(0.8, second, 2);
            assert!((0.0..=1.0).contains(&value), "{second} mapped to {value}");
            assert!(value >= previous, "{second} broke monotonicity");
            previous = value;
        }
        // A runner-up above the winner cannot happen, but the clamp must hold.
        assert_eq!(cell_confidence(0.4, 0.9, 2), 0.0);
    }

    /// 需求 3.8: the threshold is the 10th percentile of the winning scores and
    /// is always a value the station produced, so a uniform station has no cell
    /// strictly below it.
    #[test]
    fn low_sharpness_threshold_is_the_tenth_percentile() {
        assert_eq!(low_sharpness_threshold(&[]), 0.0);
        assert_eq!(low_sharpness_threshold(&[0.42]), 0.42);
        let uniform = vec![0.3; 50];
        let threshold = low_sharpness_threshold(&uniform);
        assert!(uniform.iter().all(|&value| value >= threshold));
        let ramp = (0..=100).map(|i| f64::from(i) * 0.01).collect::<Vec<_>>();
        assert!((low_sharpness_threshold(&ramp) - 0.10).abs() < 1e-9);
        // Order must not matter.
        let mut shuffled = ramp.clone();
        shuffled.reverse();
        assert_eq!(
            low_sharpness_threshold(&shuffled),
            low_sharpness_threshold(&ramp)
        );
    }

    /// 需求 3.8: low sharpness cells are merged into 4-connected regions; a
    /// diagonal touch is two regions, and the area is the cell count.
    #[test]
    fn low_sharpness_regions_merge_only_four_connected_cells() {
        let low = [
            true, true, false, //
            false, false, false, //
            false, false, true,
        ];
        let regions = low_sharpness_regions(3, 3, &low);
        assert_eq!(regions.len(), 2);
        assert_eq!(
            regions[0],
            LowSharpnessRegion {
                first_column: 0,
                first_row: 0,
                last_column: 1,
                last_row: 0,
                cells: 2,
            }
        );
        assert_eq!(regions[1].cells, 1);
        assert!(low_sharpness_regions(3, 3, &[false; 9]).is_empty());
        // A mask of the wrong size is a programming error upstream, never a
        // region.
        assert!(low_sharpness_regions(3, 3, &[true; 4]).is_empty());
    }

    /// 需求 3.8: the recorded rectangle and area are in world coordinates and
    /// world native pixels.
    #[test]
    fn low_sharpness_region_records_use_world_coordinates() {
        let geometry = OwnershipGridGeometry {
            cell_size_px: 16,
            columns: 4,
            rows: 4,
        };
        let regions = [LowSharpnessRegion {
            first_column: 1,
            first_row: 2,
            last_column: 2,
            last_row: 2,
            cells: 2,
        }];
        let records = low_sharpness_region_records(&geometry, &regions, (-100.0, 40.0));
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].world.left, -100.0 + 16.0);
        assert_eq!(records[0].world.top, 40.0 + 32.0);
        assert_eq!(records[0].world.width, 32.0);
        assert_eq!(records[0].world.height, 16.0);
        assert_eq!(records[0].area_px, 2.0 * 256.0);
    }

    /// 需求 11.12 measures the same ratio on the final output: uncovered pixels
    /// are outside the statistic.
    #[test]
    fn confidence_summary_only_counts_covered_pixels() {
        let summary = confidence_summary(&[0.0, 0.5, 0.9, 0.01], &[0, 255, 255, 255]);
        assert!((summary.mean - (0.5 + 0.9 + 0.01) / 3.0).abs() < 1e-6);
        assert!((summary.below_0_05_ratio - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(
            confidence_summary(&[0.4, 0.4], &[0, 0]),
            SharpnessConfidenceSummary::default()
        );
    }

    /// One cell column per candidate so the cells are independent, with the
    /// candidate sharper on the right half.
    fn two_frame_station() -> (StationFusion, Vec<f64>, Vec<bool>, CandidateCells) {
        // 6x1 cells: `for_plane` clamps the cell side to 8px, so a 48x8 plane
        // gives exactly six cells in one row.
        let mut fusion = StationFusion::new(48, 8, 2);
        let cells = fusion.geometry().cell_count();
        assert_eq!(cells, 6);
        let base_sharpness = vec![0.5; cells];
        let base_covered = vec![true; cells];
        fusion.seed(0, &base_sharpness, &base_covered);
        let mut candidate = CandidateCells::new(cells);
        for cell in 0..cells {
            candidate.covered[cell] = true;
            candidate.sharpness[cell] = if cell >= 3 { 0.95 } else { 0.05 };
        }
        (fusion, base_sharpness, base_covered, candidate)
    }

    /// 需求 3.6 / 3.11: every covered cell ends with exactly one owner, taken
    /// from the real candidate set — there is no blended or intermediate label.
    #[test]
    fn station_fusion_gives_every_covered_cell_one_real_owner() {
        let (mut fusion, _, _, candidate) = two_frame_station();
        let took = fusion.fold(1, &candidate);
        assert_eq!(took.len(), fusion.geometry().cell_count());
        // The sharper half switched, the softer half kept the anchor: a hard
        // split at a cell boundary, not a ramp across it.
        assert_eq!(took, vec![false, false, false, true, true, true]);
        assert_eq!(
            fusion.owners(),
            &[
                owner_of(0),
                owner_of(0),
                owner_of(0),
                owner_of(1),
                owner_of(1),
                owner_of(1)
            ]
        );
        assert_eq!(fusion.solver_status(), FusionSolverStatus::GraphCut);
    }

    /// 需求 3.7 / 3.11: a cell no frame covers keeps the reserved no-owner
    /// identifier, and a cell only the candidate covers goes to the candidate.
    #[test]
    fn station_fusion_leaves_uncovered_cells_unowned() {
        let mut fusion = StationFusion::new(48, 8, 2);
        let cells = fusion.geometry().cell_count();
        let mut base_covered = vec![true; cells];
        base_covered[0] = false;
        base_covered[5] = false;
        fusion.seed(0, &vec![0.5; cells], &base_covered);
        let mut candidate = CandidateCells::new(cells);
        candidate.covered[0] = true;
        candidate.sharpness[0] = 0.2;
        fusion.fold(1, &candidate);
        assert_eq!(
            fusion.owners()[0],
            owner_of(1),
            "only the candidate covers cell 0"
        );
        assert_eq!(fusion.owners()[5], NO_OWNER, "no frame covers cell 5");
        assert!(
            fusion.owners()[1..5]
                .iter()
                .all(|&owner| owner == owner_of(0))
        );
    }

    /// 需求 3.3: an inconsistent candidate loses the cell even though it is
    /// sharper, because the penalty is charged on top of its data term.
    #[test]
    fn station_fusion_penalises_a_sharper_but_displaced_candidate() {
        let (mut fusion, _, _, mut candidate) = two_frame_station();
        for cell in 0..candidate.covered.len() {
            candidate.disagreement[cell] = OWNERSHIP_DISAGREEMENT_VETO + 0.1;
        }
        let took = fusion.fold(1, &candidate);
        assert!(
            took.iter().all(|&took| !took),
            "a displaced candidate must not take any cell"
        );
    }

    /// 需求 3.9: the confidence of a cell follows its own winner/runner-up gap,
    /// independently of which frame won.
    #[test]
    fn station_fusion_confidence_follows_the_winner_runner_up_gap() {
        let (mut fusion, _, _, candidate) = two_frame_station();
        fusion.fold(1, &candidate);
        let confidence = fusion.confidence();
        // Soft half: winner 0.5, runner-up 0.05 -> (0.5 - 0.05) / 0.5.
        assert!((f64::from(confidence[0]) - 0.9).abs() < 1e-6);
        // Sharp half: winner 0.95, runner-up 0.5 -> (0.95 - 0.5) / 0.95.
        assert!((f64::from(confidence[5]) - 0.45 / 0.95).abs() < 1e-6);
        assert!(confidence.iter().all(|&value| (0.0..=1.0).contains(&value)));
    }

    /// The report evidence is the same winner/runner pair that feeds
    /// Sharpness_Confidence, and maps back to the ownership cell containing a
    /// pixel without allocating a full-resolution diagnostic plane.
    #[test]
    fn sharpness_evidence_preserves_winner_runner_and_owner_scores() {
        let (mut fusion, _, _, candidate) = two_frame_station();
        fusion.fold(1, &candidate);
        let evidence = fusion.sharpness_evidence();
        assert_eq!(evidence.dimensions(), (6, 1, 8));
        let soft = evidence.at_pixel(4, 0).expect("cell evidence");
        assert_eq!(soft.winner_score, 0.5);
        assert_eq!(soft.runner_up_score, 0.05);
        assert_eq!(soft.owner_score, 0.5);
        assert_eq!(soft.candidate_count, 2);
        let sharp = evidence.at_pixel(40, 7).expect("cell evidence");
        assert_eq!(sharp.winner_score, 0.95);
        assert_eq!(sharp.runner_up_score, 0.5);
        assert_eq!(sharp.owner_score, 0.95);
    }

    /// 需求 3.10: a station of one frame carries no runner-up evidence at all,
    /// so every cell's confidence is 0 and the solver never ran.
    #[test]
    fn a_single_frame_station_has_zero_confidence_everywhere() {
        let mut fusion = StationFusion::new(48, 8, 1);
        let cells = fusion.geometry().cell_count();
        fusion.seed(0, &vec![0.7; cells], &vec![true; cells]);
        assert_eq!(fusion.solver_status(), FusionSolverStatus::SingleFrame);
        assert!(fusion.confidence().iter().all(|&value| value == 0.0));
        assert!(fusion.owners().iter().all(|&owner| owner == owner_of(0)));
        assert_eq!(
            StationFusion::new(48, 8, 0).solver_status(),
            FusionSolverStatus::NotRun
        );
    }

    /// 需求 3.8: the cells whose winner is below the station P10 are exactly the
    /// low sharpness cells, and an uncovered cell is never one of them.
    #[test]
    fn station_fusion_marks_the_below_percentile_cells() {
        let mut fusion = StationFusion::new(8 * 20, 8, 1);
        let cells = fusion.geometry().cell_count();
        assert_eq!(cells, 20);
        let mut sharpness = vec![0.8; cells];
        sharpness[7] = 0.01;
        let mut covered = vec![true; cells];
        covered[19] = false;
        fusion.seed(0, &sharpness, &covered);
        let low = fusion.low_sharpness_cells();
        assert!(low[7], "the single soft cell is below the P10 of the rest");
        assert_eq!(low.iter().filter(|&&low| low).count(), 1);
        assert!(!low[19], "an uncovered cell has no candidate to compare");
        let records = fusion.low_sharpness_region_records((0.0, 0.0));
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].area_px, 64.0);
    }

    #[test]
    fn content_regions_come_only_from_current_masks_and_flat_components() {
        let geometry = OwnershipGridGeometry {
            cell_size_px: 100,
            columns: 16,
            rows: 8,
        };
        let count = geometry.cell_count();
        let sharpness = vec![0.02; count];
        let costs = vec![0.1; count];
        let full = ContentEvidence {
            fully_covered: vec![true; count],
            flat: vec![true; count],
            mean_luma: vec![0.5; count],
        };
        let candidates =
            content_region_candidates(geometry, owner_of(0), &sharpness, &costs, &full);
        assert!(!candidates.is_empty());
        assert!(candidates.iter().all(|candidate| {
            candidate
                .cells
                .iter()
                .all(|&cell| full.fully_covered[cell] && full.flat[cell])
        }));

        let mut masked = full.clone();
        masked.fully_covered[0] = false;
        let masked_candidates =
            content_region_candidates(geometry, owner_of(0), &sharpness, &costs, &masked);
        assert!(
            masked_candidates
                .iter()
                .all(|candidate| !candidate.cells.contains(&0))
        );
        let mut textured = full;
        textured.flat[0] = false;
        let textured_candidates =
            content_region_candidates(geometry, owner_of(0), &sharpness, &costs, &textured);
        assert!(
            textured_candidates
                .iter()
                .all(|candidate| candidate.origin != (0, 0))
        );
    }

    #[test]
    fn content_region_selection_is_permutation_translation_and_thread_deterministic() {
        let geometry = OwnershipGridGeometry {
            cell_size_px: 100,
            columns: 24,
            rows: 8,
        };
        let count = geometry.cell_count();
        let evidence = ContentEvidence {
            fully_covered: vec![true; count],
            flat: vec![true; count],
            mean_luma: vec![0.4; count],
        };
        let candidates = content_region_candidates(
            geometry,
            owner_of(0),
            &vec![0.01; count],
            &vec![0.2; count],
            &evidence,
        );
        let expected = select_content_regions(candidates.clone(), 2);
        let mut reversed = candidates.clone();
        reversed.reverse();
        assert_eq!(expected, select_content_regions(reversed, 2));
        let threaded = std::thread::spawn(move || select_content_regions(candidates, 2))
            .join()
            .expect("content selector thread");
        assert_eq!(expected, threaded);

        let shifted_geometry = OwnershipGridGeometry {
            columns: 32,
            ..geometry
        };
        let shifted_count = shifted_geometry.cell_count();
        let mut shifted = ContentEvidence {
            fully_covered: vec![false; shifted_count],
            flat: vec![false; shifted_count],
            mean_luma: vec![f64::NAN; shifted_count],
        };
        for row in 0..8usize {
            for column in 8..32usize {
                let cell = row * 32 + column;
                shifted.fully_covered[cell] = true;
                shifted.flat[cell] = true;
                shifted.mean_luma[cell] = 0.4;
            }
        }
        let translated = select_content_regions(
            content_region_candidates(
                shifted_geometry,
                owner_of(0),
                &vec![0.01; shifted_count],
                &vec![0.2; shifted_count],
                &shifted,
            ),
            2,
        );
        assert_eq!(translated.len(), expected.len());
        assert!(
            translated
                .iter()
                .zip(&expected)
                .all(|(translated, original)| translated.origin.0 == original.origin.0 + 8)
        );
    }

    #[test]
    fn disconnected_and_overlapping_content_regions_cannot_count_twice() {
        let geometry = OwnershipGridGeometry {
            cell_size_px: 100,
            columns: 8,
            rows: 8,
        };
        let count = geometry.cell_count();
        let mut evidence = ContentEvidence {
            fully_covered: vec![true; count],
            flat: vec![true; count],
            mean_luma: vec![0.5; count],
        };
        for row in 0..8usize {
            evidence.flat[row * 8 + 4] = false;
        }
        assert!(
            content_region_candidates(
                geometry,
                owner_of(0),
                &vec![0.01; count],
                &vec![0.1; count],
                &evidence,
            )
            .is_empty()
        );

        let duplicate = ContentRegionCandidate {
            owner: owner_of(0),
            cells: (0..64).collect(),
            origin: (0, 0),
            data_cost: 0.1,
        };
        let mut overlap = duplicate.clone();
        overlap.owner = owner_of(1);
        overlap.origin = (1, 0);
        assert_eq!(select_content_regions(vec![duplicate, overlap], 2).len(), 1);
    }

    #[test]
    fn decisive_sharpness_requires_a_coherent_non_vetoed_component() {
        let geometry = OwnershipGridGeometry {
            cell_size_px: 8,
            columns: 4,
            rows: 2,
        };
        let mut candidate = CandidateCells::new(8);
        candidate.covered.fill(true);
        candidate.sharpness.fill(0.5);
        for cell in [0usize, 1, 4, 5] {
            candidate.sharpness[cell] = 0.9;
        }
        candidate.sharpness[7] = 0.9;
        let decisive = coherent_decisive_cells(geometry, &vec![0.5; 8], &candidate);
        assert_eq!(
            decisive,
            vec![true, true, false, false, true, true, false, false]
        );

        candidate.disagreement[0] = OWNERSHIP_DISAGREEMENT_VETO + 0.01;
        let vetoed = coherent_decisive_cells(geometry, &vec![0.5; 8], &candidate);
        assert!(vetoed.iter().all(|&value| !value));

        candidate.sharpness.fill(0.1);
        let base = coherent_base_cells(geometry, &vec![0.5; 8], &candidate);
        assert!(base.iter().all(|&value| value));
        candidate.sharpness[7] = 0.49;
        let still_coherent = coherent_base_cells(geometry, &vec![0.5; 8], &candidate);
        assert!(!still_coherent[7]);
        assert!(still_coherent[..7].iter().all(|&value| value));
    }

    #[test]
    fn region_decision_keeps_a_quality_roi_sized_owner_block_together() {
        let geometry = OwnershipGridGeometry {
            cell_size_px: 16,
            columns: 50,
            rows: 50,
        };
        let count = geometry.cell_count();
        let mut candidate = CandidateCells::new(count);
        candidate.covered.fill(true);
        candidate.sharpness.fill(0.9);
        let decisive = region_decisive_cells(geometry, &vec![0.5; count], &candidate);
        assert!(decisive.iter().all(|&value| value));

        candidate.disagreement[0] = OWNERSHIP_DISAGREEMENT_VETO + 0.01;
        let vetoed = region_decisive_cells(geometry, &vec![0.5; count], &candidate);
        assert!(vetoed.iter().all(|&value| !value));
    }

    #[test]
    fn quality_region_accepts_a_near_best_full_roi_without_relaxing_shortfall() {
        let geometry = OwnershipGridGeometry {
            cell_size_px: 16,
            columns: 40,
            rows: 40,
        };
        let count = geometry.cell_count();
        let mut candidate = CandidateCells::new(count);
        candidate.covered.fill(true);
        candidate.sharpness.fill(0.951);
        let owner = vec![0.94; count];
        let best = vec![1.0; count];
        let decisive = quality_region_decisive_cells(geometry, &owner, &best, &candidate);
        assert!(decisive.iter().all(|&value| value));

        // 0.94 is a six-percent shortfall from the observed winner, so the
        // same region is rejected at the unchanged five-percent gate boundary.
        candidate.sharpness.fill(0.94);
        let rejected = quality_region_decisive_cells(geometry, &owner, &best, &candidate);
        assert!(rejected.iter().all(|&value| !value));

        // A near-winner is still not allowed to lower an already sharper
        // owner just because it makes a larger connected ROI.  This is the
        // regression that let the smoothness override create gate shortfall.
        candidate.sharpness.fill(0.93);
        let softer = quality_region_decisive_cells(geometry, &vec![0.94; count], &best, &candidate);
        assert!(softer.iter().all(|&value| !value));
    }

    #[test]
    fn quality_gate_decisive_cells_only_moves_a_consistent_five_percent_winner() {
        let geometry = OwnershipGridGeometry {
            cell_size_px: 16,
            columns: 2,
            rows: 2,
        };
        let count = geometry.cell_count();
        let mut candidate = CandidateCells::new(count);
        candidate.covered.fill(true);
        candidate.sharpness.fill(0.96);
        let owner = vec![0.90; count];
        let best = vec![0.96; count];
        let decisive = quality_gate_decisive_cells(geometry, &owner, &best, &candidate);
        assert!(decisive.iter().all(|&value| value));

        // A candidate that is sharper than the current owner but six percent
        // below the observed winner cannot manufacture a passing owner.
        candidate.sharpness.fill(0.90);
        let rejected = quality_gate_decisive_cells(geometry, &owner, &best, &candidate);
        assert!(rejected.iter().all(|&value| !value));

        candidate.sharpness.fill(0.96);
        candidate
            .disagreement
            .fill(OWNERSHIP_DISAGREEMENT_VETO + 0.01);
        let vetoed = quality_gate_decisive_cells(geometry, &owner, &best, &candidate);
        assert!(vetoed.iter().all(|&value| !value));
    }

    #[test]
    fn content_constraint_keeps_real_provenance_without_forcing_an_oracle() {
        let geometry = OwnershipGridGeometry {
            cell_size_px: 100,
            columns: 24,
            rows: 8,
        };
        let count = geometry.cell_count();
        let evidence = ContentEvidence {
            fully_covered: vec![true; count],
            flat: vec![true; count],
            mean_luma: vec![0.5; count],
        };
        let mut candidates = content_region_candidates(
            geometry,
            owner_of(0),
            &vec![0.2; count],
            &vec![0.4; count],
            &evidence,
        );
        candidates.extend(content_region_candidates(
            geometry,
            owner_of(1),
            &vec![0.3; count],
            &vec![0.1; count],
            &evidence,
        ));
        let selected = select_content_regions(candidates.clone(), 2);
        assert_eq!(selected.len(), 2);
        assert!(selected.iter().all(|region| region.owner == owner_of(1)));
        assert!(selected.iter().all(|region| candidates.contains(region)));
    }

    #[test]
    fn content_constraint_handles_feasible_and_impossible_cardinality() {
        let geometry = OwnershipGridGeometry {
            cell_size_px: 100,
            columns: 80,
            rows: 8,
        };
        let count = geometry.cell_count();
        let sharpness = vec![0.01; count];
        let costs = vec![0.1; count];
        let feasible = content_region_candidates(
            geometry,
            owner_of(0),
            &sharpness,
            &costs,
            &ContentEvidence {
                fully_covered: vec![true; count],
                flat: vec![true; count],
                mean_luma: vec![0.5; count],
            },
        );
        assert_eq!(
            select_content_regions(feasible, CONTENT_OWNER_REGION_TARGET).len(),
            CONTENT_OWNER_REGION_TARGET
        );

        let impossible = content_region_candidates(
            geometry,
            owner_of(0),
            &sharpness,
            &costs,
            &ContentEvidence {
                fully_covered: vec![true; count],
                flat: vec![false; count],
                mean_luma: vec![0.5; count],
            },
        );
        assert!(impossible.is_empty());
    }

    /// 需求 3.5: an exhausted budget resolves the round per cell and marks the
    /// station degraded, without changing what a covered cell is.
    #[test]
    fn an_exhausted_station_budget_falls_back_to_the_per_cell_minimum() {
        let (mut fusion, _, _, candidate) = two_frame_station();
        fusion.timeout = Duration::ZERO;
        let took = fusion.fold(1, &candidate);
        assert_eq!(took, vec![false, false, false, true, true, true]);
        assert_eq!(fusion.solver_status(), FusionSolverStatus::PerCellFallback);
        let mut ledger = DegradationLedger::new();
        record_solver_degradation(
            &mut ledger,
            &OwnershipSolution {
                owners: fusion.owners().to_vec(),
                solver_status: fusion.solver_status(),
                graph_cut_seconds: fusion.graph_cut_seconds(),
            },
        );
        assert_eq!(ledger.entries().len(), 1);
    }

    /// The run scoped sink keeps one record per station, replaces a re-rendered
    /// station instead of duplicating it, and orders the document by station
    /// index rather than by render order.
    ///
    /// The sink is a process global, so this is the only test that touches it.
    #[test]
    fn the_run_sink_keeps_one_record_per_station() {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let record = |station_index: usize, cell_size_px: u32| FusionReport {
            station_index,
            cell_size_px,
            ..FusionReport::default()
        };
        reset_run_records();
        for report in [record(2, 8), record(0, 9), record(2, 10)] {
            record_run_station(report);
        }
        assert_eq!(
            run_records_snapshot()
                .iter()
                .map(|station| (station.station_index, station.cell_size_px))
                .collect::<Vec<_>>(),
            vec![(0, 9), (2, 10)]
        );
        reset_run_records();
        assert!(run_records_snapshot().is_empty());
    }
}
