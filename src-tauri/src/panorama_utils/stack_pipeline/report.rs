//! Stack_Report: the machine readable JSON report produced by every
//! Stack_Pipeline run.
//!
//! Requirement 10.8 requires a report on *every* terminating path (success,
//! degraded output, rejection, cancellation) within 30 seconds, and
//! requirement 11.16 requires the Quality_Gate metrics to land in the same
//! document.  The schema therefore fixes its top-level fields and writes
//! defaults explicitly, so `tests/focus-stack-quality-contract.mjs` can assert
//! the structure without decoding a single RAW file.
//!
//! This stage records only.  No field in here participates in a pass/fail
//! decision yet; the Quality_Gate verdict stays [`QualityGateVerdict::NotRun`]
//! until the gate itself is implemented.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use super::degradation::{self, DegradationLedger, RunResult};
/// Single source of the actual Intra_Station analysis configuration
/// (requirement 2.2); the report echoes the configured 1400 rather than the
/// independent 2048 ceiling.
use super::focus_fuser;
use super::intra_station::{self, INTRA_STATION_ANALYSIS_LONG_SIDE};
use super::residual_warp;

/// Bumped whenever the serialized shape of [`StackReport`] changes.
pub(crate) const STACK_REPORT_SCHEMA: u32 = 1;

/// 64 GiB Virtual_Tile cache budget (requirement 4.4).
pub(crate) const VIRTUAL_TILE_CACHE_LIMIT_BYTES: u64 = 68_719_476_736;
/// Canvas long-side ceiling (requirement 10.11).
pub(crate) const CANVAS_LONG_SIDE_LIMIT: u64 = 262_144;
/// Default peak-RSS threshold of 24 GiB (requirement 14).
pub(crate) const MEMORY_THRESHOLD_DEFAULT_BYTES: u64 = 25_769_803_776;
/// Owner_Region boundary Delta_E00 ceiling (requirement 9.8).
pub(crate) const TONE_BOUNDARY_DELTA_E_THRESHOLD: f64 = 1.5;
/// Low-frequency band sigma in world pixels (requirement 9.2).
pub(crate) const TONE_LOW_PASS_SIGMA_WORLD_PX: f64 = 64.0;
/// Residual_Warp node spacing in world pixels (requirement 8.2).
pub(crate) const RESIDUAL_WARP_NODE_STEP_PX: u32 = 64;
/// Directory (under the app cache dir) that collects the reports.
pub(crate) const STACK_REPORT_DIR_NAME: &str = "stack-reports";

/// Which composition path a run actually executed (requirement 15.11).
///
/// The identifiers are stable and machine readable; they are the same set the
/// `StackCompositorChoice` setting will select from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SelectedPath {
    LayeredVirtualTile,
    ProgressiveSeamTile,
    StreamingMosaic,
    LegacySingleLayerMosaic,
    SingleStation,
}

impl SelectedPath {
    /// Wired into the `StackCompositorChoice` setting by
    /// [`super::compositor::StackCompositorChoice::selected_path`]; the
    /// serialized form already goes through `#[serde(rename_all)]`.
    pub(crate) fn as_identifier(self) -> &'static str {
        match self {
            Self::LayeredVirtualTile => "layered_virtual_tile",
            Self::ProgressiveSeamTile => "progressive_seam_tile",
            Self::StreamingMosaic => "streaming_mosaic",
            Self::LegacySingleLayerMosaic => "legacy_single_layer_mosaic",
            Self::SingleStation => "single_station",
        }
    }
}

/// Outcome of a single run (requirement 12.9: rejection outranks degradation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StackRunResult {
    /// The run has not reached a terminating path yet.
    Running,
    Success,
    Degraded,
    Rejected,
    Cancelled,
}

impl StackRunResult {
    /// Consumed by the Quality_Gate reporting of a later phase.
    #[allow(dead_code)]
    pub(crate) fn as_identifier(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Success => "success",
            Self::Degraded => "degraded",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
        }
    }

    /// How strongly a state claims the run outcome.  `Cancelled` ranks highest
    /// because a user cancellation is the most specific terminating path: a
    /// cancelled run must never be reported as a success, a degraded output or
    /// a plain rejection.
    fn precedence(self) -> u8 {
        match self {
            Self::Running => 0,
            Self::Success => 1,
            Self::Degraded => 2,
            Self::Rejected => 3,
            Self::Cancelled => 4,
        }
    }

    /// Combine the state the stitching path already recorded with the state the
    /// [`DegradationLedger`] derived (requirement 12.9).  The stronger claim
    /// wins, so a ledger that saw nothing (`Success`) leaves an already
    /// recorded `Cancelled` or `Rejected` untouched, while a single ledger
    /// rejection overrides a `Success` written by the compositor.
    fn merge(self, ledger: Self) -> Self {
        if ledger.precedence() > self.precedence() {
            ledger
        } else {
            self
        }
    }
}

/// Translate a ledger verdict into the report's result vocabulary.  A ledger
/// cancellation is a `Rejected` severity entry, but the report keeps it
/// distinguishable as [`StackRunResult::Cancelled`].
fn ledger_run_result(ledger: &DegradationLedger) -> StackRunResult {
    if ledger.count_of(degradation::RUN_CANCELLED_BY_USER) > 0 {
        return StackRunResult::Cancelled;
    }
    match ledger.result() {
        RunResult::Success => StackRunResult::Success,
        RunResult::Degraded => StackRunResult::Degraded,
        RunResult::Rejected => StackRunResult::Rejected,
    }
}

/// Convert the ledger entries into their report records, preserving order.
fn degradation_entry_records(ledger: &DegradationLedger) -> Vec<DegradationEntryRecord> {
    ledger
        .entries()
        .iter()
        .map(|entry| DegradationEntryRecord {
            reason: entry.reason.to_string(),
            severity: entry.severity.as_identifier().to_string(),
            detail: entry.detail.clone(),
        })
        .collect()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct WorldRect {
    pub left: f64,
    pub top: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct WorldPoint {
    pub x: f64,
    pub y: f64,
}

// ---------------------------------------------------------------------------
// input
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SourceRecord {
    pub path: String,
    /// SHA-256 of the file before the run (requirement 4.8 / 12.8).
    pub sha256_before: String,
    /// SHA-256 of the same file after the run finished.
    pub sha256_after: String,
    pub decoded: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct InputReport {
    pub source_count: usize,
    pub max_sources: usize,
    pub sources: Vec<SourceRecord>,
}

// ---------------------------------------------------------------------------
// grouping
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StationEvidenceRecord {
    /// Stable identifier of the evidence kind, e.g. `verified_overlap`.
    pub kind: String,
    pub score: f64,
    pub inliers: usize,
    pub overlap_ncc: f64,
    pub spatial_support: f64,
    pub scale_ratio: f64,
}

impl Default for StationEvidenceRecord {
    fn default() -> Self {
        Self {
            kind: "unmeasured".to_string(),
            score: 0.0,
            inliers: 0,
            overlap_ncc: 0.0,
            spatial_support: 0.0,
            scale_ratio: 1.0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct StationRecord {
    pub index: usize,
    /// Row index once the Capture_Topology_Model infers one (requirement 5.1).
    pub row: Option<usize>,
    /// Column index once the Capture_Topology_Model infers one.
    pub column: Option<usize>,
    /// Stable topology status identifier when the displacement-derived index
    /// is not unique (requirement 5.7).
    pub topology_status: Option<String>,
    pub members: Vec<String>,
    pub anchor: String,
    pub evidence: StationEvidenceRecord,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct ExcludedSourceRecord {
    pub path: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct StationSplitRecord {
    pub station_index: usize,
    pub position: usize,
    pub reason: String,
    pub motion_ratio: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct GroupingReport {
    pub station_count: usize,
    pub stations: Vec<StationRecord>,
    pub isolated: Vec<ExcludedSourceRecord>,
    pub undecodable: Vec<ExcludedSourceRecord>,
    pub splits: Vec<StationSplitRecord>,
}

// ---------------------------------------------------------------------------
// intra_station
// ---------------------------------------------------------------------------

/// Per-frame intra-station registration status (requirement 2.7 / 2.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum IntraStationFrameStatus {
    /// The frame is the anchor and carries the identity transform.
    Anchor,
    /// Native patch refinement was accepted.
    Local,
    /// The local field was discarded; the global model is used instead.
    GlobalFallback,
    /// The frame did not reach the required inlier area coverage.
    Failed,
}

/// Outcome of the 需求 2.7 comparison between the refined and the global median
/// symmetric reprojection error.
///
/// Separate from [`IntraStationFrameStatus`] because 需求 2.7 and 需求 2.8 are
/// independent verdicts: a frame can keep its local field and still fail the
/// coverage floor, and the report has to say which of the two happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum IntraStationLocalFieldVerdict {
    /// The refined median was below the global model's; the field stays.
    Kept,
    /// The refined median was not below it; 需求 2.7 reverted the frame.
    Reverted,
    /// One of the two medians had too few accepted control points to exist, so
    /// the comparison was unevaluable and nothing was reverted on its behalf.
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct IntraStationFrameRecord {
    pub path: String,
    /// Row-major 3x3 transform into the anchor frame.
    pub transform: [f64; 9],
    pub inliers: usize,
    pub inlier_area_coverage: f64,
    /// Measured distance between neighbouring patch refinement control points,
    /// in native pixels of the anchor frame (需求 2.3).  The grid step is fixed
    /// in analysis units, so only a measurement can say what it became natively.
    pub control_point_spacing_px: f64,
    pub rejected_control_point_ratio: f64,
    pub median_symmetric_error_px: f64,
    /// Median symmetric reprojection error of the global model, native pixels,
    /// and the number of control points it was formed from (需求 2.7).  Zero
    /// samples means the comparison had no baseline, which the verdict below
    /// then reports as indeterminate instead of as a revert.
    pub global_median_symmetric_error_px: f64,
    pub global_baseline_samples: usize,
    pub local_baseline_samples: usize,
    /// How 需求 2.7 resolved for this frame.
    pub local_field_verdict: IntraStationLocalFieldVerdict,
    /// Median `softer / sharper` high-frequency energy ratio of the frame's
    /// control-point patch pairs *before* the focal-plane matched low-pass, in
    /// `[0, 1]`.  A Capture_Station is a focus bracket, so a value well below 1
    /// is the normal case and is the reason the raw-intensity matcher could not
    /// see these pairs as the same content.
    pub focal_plane_ratio_before: f64,
    /// The same ratio after the matched low-pass.  This is the measurement that
    /// says whether the two sides were actually made comparable.
    pub focal_plane_ratio_after: f64,
    pub status: IntraStationFrameStatus,
}

impl Default for IntraStationFrameRecord {
    fn default() -> Self {
        Self {
            path: String::new(),
            transform: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            inliers: 0,
            inlier_area_coverage: 0.0,
            control_point_spacing_px: 0.0,
            rejected_control_point_ratio: 0.0,
            median_symmetric_error_px: 0.0,
            global_median_symmetric_error_px: 0.0,
            global_baseline_samples: 0,
            local_baseline_samples: 0,
            local_field_verdict: IntraStationLocalFieldVerdict::Indeterminate,
            focal_plane_ratio_before: 1.0,
            focal_plane_ratio_after: 1.0,
            status: IntraStationFrameStatus::GlobalFallback,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct IntraStationReport {
    pub station_index: usize,
    pub anchor_path: String,
    pub analysis_long_side: u32,
    pub frames: Vec<IntraStationFrameRecord>,
}

impl Default for IntraStationReport {
    fn default() -> Self {
        Self {
            station_index: 0,
            anchor_path: String::new(),
            analysis_long_side: INTRA_STATION_ANALYSIS_LONG_SIDE,
            frames: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// fusion
// ---------------------------------------------------------------------------

/// Ownership solver status for a Capture_Station (requirement 3.5 / 3.10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FusionSolverStatus {
    /// Multi-label graph cut converged.
    GraphCut,
    /// Graph cut timed out; per-cell minimum cost was used instead.
    PerCellFallback,
    /// A single frame covers the station, so no cut was needed.
    SingleFrame,
    /// Fusion has not run for this station.
    NotRun,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct OwnershipGridSize {
    pub columns: u32,
    pub rows: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct LowSharpnessRegionRecord {
    pub world: WorldRect,
    pub area_px: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SharpnessConfidenceSummary {
    pub mean: f64,
    pub below_0_05_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct FusionReport {
    pub station_index: usize,
    pub cell_size_px: u32,
    pub grid: OwnershipGridSize,
    pub solver_status: FusionSolverStatus,
    pub graph_cut_seconds: f64,
    pub owned_pixels: u64,
    pub uncovered_pixels: u64,
    pub low_sharpness_regions: Vec<LowSharpnessRegionRecord>,
    pub confidence: SharpnessConfidenceSummary,
}

impl Default for FusionReport {
    fn default() -> Self {
        Self {
            station_index: 0,
            cell_size_px: 0,
            grid: OwnershipGridSize::default(),
            solver_status: FusionSolverStatus::NotRun,
            graph_cut_seconds: 0.0,
            owned_pixels: 0,
            uncovered_pixels: 0,
            low_sharpness_regions: Vec::new(),
            confidence: SharpnessConfidenceSummary::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// virtual_tiles
// ---------------------------------------------------------------------------

/// Virtual_Tile cache outcome (requirement 4.5 / 4.10 / 4.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum VirtualTileCacheStatus {
    Hit,
    Miss,
    /// The entry existed but failed verification and was re-synthesised.
    Invalid,
    /// The cache directory could not be written; the tile stayed in memory.
    WriteUnavailable,
    /// No cache lookup happened.
    NotRun,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct VirtualTileCacheRecord {
    pub status: VirtualTileCacheStatus,
    pub cache_key: String,
    pub invalid_reason: Option<String>,
}

impl Default for VirtualTileCacheRecord {
    fn default() -> Self {
        Self {
            status: VirtualTileCacheStatus::NotRun,
            cache_key: String::new(),
            invalid_reason: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct ProvenanceRecord {
    pub path: String,
    pub sha256: String,
    pub owned_pixels: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct VirtualTileReport {
    pub station_index: usize,
    pub width: u32,
    pub height: u32,
    pub cache: VirtualTileCacheRecord,
    pub provenance: Vec<ProvenanceRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct VirtualTileCacheSummary {
    pub total_bytes: u64,
    pub limit_bytes: u64,
    pub evicted_entries: u64,
}

impl Default for VirtualTileCacheSummary {
    fn default() -> Self {
        Self {
            total_bytes: 0,
            limit_bytes: VIRTUAL_TILE_CACHE_LIMIT_BYTES,
            evicted_entries: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// topology
// ---------------------------------------------------------------------------

/// Candidate_Adjacency classification (requirement 5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AdjacencyKind {
    SameColumn,
    SameRow,
    CrossColumn,
    /// Pairing forced by an ambiguous row/column index (requirement 5.7).
    Ambiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct CandidateAdjacencyRecord {
    pub left: usize,
    pub right: usize,
    pub kind: AdjacencyKind,
    pub score: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct TopologyReport {
    pub candidates: Vec<CandidateAdjacencyRecord>,
    pub truncated_count: usize,
    pub truncation_min_score: Option<f64>,
    pub ambiguous_stations: Vec<usize>,
}

// ---------------------------------------------------------------------------
// station_relations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StationRelationEvidenceKind {
    SourceRawConsensus,
    VirtualTile,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct AcceptedStationRelationRecord {
    pub left: usize,
    pub right: usize,
    /// Authoritative requirement-6.1 provenance for this relation.
    pub evidence_kind: StationRelationEvidenceKind,
    pub inliers: usize,
    pub spatial_support: f64,
    pub scale_ratio: f64,
    pub median_error_px: f64,
    pub independent_support: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct DiscardedEvidenceRecord {
    pub station_index: usize,
    pub count: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct ConnectivityReport {
    pub components: usize,
    /// One entry per connected component, listing member source paths
    /// (requirement 6.7 / 12.4).
    pub component_members: Vec<Vec<String>>,
    /// Mirrors `component_members[*].len()` explicitly for machine readers.
    pub component_member_counts: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct ResidualDistributionRecord {
    pub count: usize,
    pub min: f64,
    pub p10: f64,
    pub median: f64,
    pub p90: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct ResidualModelStageRecord {
    /// Largest measured correspondence consensus considered by this stage.
    pub candidate_inliers: usize,
    /// Spatial support of that stage's final measured consensus.
    pub final_support: f64,
    /// Whether the residual correction passed the bounded-correction guard.
    /// `None` means the stage could not produce a correction to check.
    pub bounded_correction: Option<bool>,
    /// Stable stage-local outcome: `accepted`, `no_fit`,
    /// `insufficient_inliers`, `insufficient_support`, or
    /// `correction_out_of_bounds`.
    pub rejection_reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StationRelationCandidateRecord {
    pub left: usize,
    pub right: usize,
    /// Candidate provenance is explicit even when no model could be fitted.
    pub evidence_kind: StationRelationEvidenceKind,
    /// Deterministic fused-pixel grid locations considered inside predicted overlap.
    pub grid_probe_count: usize,
    /// Grid patches whose complete forward search support was covered and non-flat.
    pub measurable_patch_count: usize,
    /// Unique forward matches that also localized back within one pixel.
    pub bidirectional_match_count: usize,
    pub occupied_grid_cells: usize,
    pub search_radius_px: f64,
    /// Distribution of `measured_target - initial_prediction` in target pixels.
    pub residual_dx_px: ResidualDistributionRecord,
    pub residual_dy_px: ResidualDistributionRecord,
    pub residual_magnitude_px: ResidualDistributionRecord,
    /// Largest strict measured-point consensus obtained by a residual translation.
    pub max_residual_translation_consensus: usize,
    /// Explicit diagnostics for each lowest-to-highest residual model stage.
    pub residual_translation: ResidualModelStageRecord,
    pub residual_affine: ResidualModelStageRecord,
    pub residual_projective: ResidualModelStageRecord,
    /// Lowest-complexity residual model that passed strict measured-point gates.
    pub residual_model: String,
    pub fitted_inliers: usize,
    pub hull_support: f64,
    /// Structural orientation observed before fitting. This is diagnostic only:
    /// its local Jacobian comes from the rough overlap/search prior.
    pub median_prefit_structural_orientation_difference_degrees: Option<f64>,
    /// Orientation observed over fitted inliers using the measured homography's
    /// local Jacobian. This may reject a provisional fit before relation scoring.
    pub median_fitted_orientation_difference_degrees: Option<f64>,
    /// Stable matcher stage such as `fitted`, `fitted_orientation_mismatch`, or
    /// `insufficient_grid_coverage`.
    pub failure_stage: String,
    /// Area-averaging factors of the coarse-to-fine probe levels, coarsest first.
    /// Every count above belongs to the last level in this list that was run.
    #[serde(default)]
    pub pyramid_factors: Vec<u32>,
    /// Level whose measurements the counts above describe (1 = native pixels).
    #[serde(default)]
    pub measurement_factor: u32,
    /// Seed of the bounded search: `rendering_prior` is the only authoritative
    /// kind; `prior_free_features` only proposes a rendering-prior repair.
    #[serde(default)]
    pub seed: String,
    /// RANSAC inliers of the prior-free covered-feature relation, 0 when unused.
    #[serde(default)]
    pub prior_free_inliers: usize,
    /// True when either station was re-rendered from repaired source poses.
    #[serde(default)]
    pub after_prior_repair: bool,
}

/// One Capture_Station whose rendering prior was repaired from a measured
/// prior-free Virtual_Tile relation before being rendered once more.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct StationPriorRepairRecord {
    pub station: usize,
    /// Station already consistent with the reference cluster.
    pub fixed_station: usize,
    /// Candidate pair `(left, right)` whose prior-free relation was used.
    pub relation_left: usize,
    pub relation_right: usize,
    pub relation_inliers: usize,
    /// World displacement of the repaired tile centre.
    pub center_shift_px: f64,
    /// `sqrt(σ1·σ2)` of the world correction's linear part.
    pub correction_scale_ratio: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct StationRelationsReport {
    pub accepted: Vec<AcceptedStationRelationRecord>,
    /// One record for every topology candidate, including no-fit rejections.
    pub candidates: Vec<StationRelationCandidateRecord>,
    /// Rendering-prior repairs applied before the authoritative measurement.
    #[serde(default)]
    pub prior_repairs: Vec<StationPriorRepairRecord>,
    pub rejected_count: usize,
    /// Rejection counts keyed by stable failure identifier (requirement 6.10).
    pub rejected_by_reason: BTreeMap<String, u64>,
    pub discarded_single_layer_evidence: Vec<DiscardedEvidenceRecord>,
    pub connectivity: ConnectivityReport,
}

// ---------------------------------------------------------------------------
// closure
// ---------------------------------------------------------------------------

/// Closure_Optimizer status (requirement 7.5 / 7.7 / 7.10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ClosureStatus {
    Converged,
    /// The joint solve was rejected and the spanning-tree initial poses stand.
    Unreliable,
    /// Fewer accepted relations than stations - 1, so no joint solve ran.
    NoConstraints,
    NotRun,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct ClosureConstraintWeightRecord {
    pub left: usize,
    pub right: usize,
    pub residual_px: f64,
    /// Final M-estimator multiplier after the last closure iteration.
    pub weight: f64,
}

/// Direct Station_Relation whose overlap reprojection P95 rejected closure.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct ClosureOffendingPairRecord {
    pub left: usize,
    pub left_row: u32,
    pub left_column: u32,
    pub right: usize,
    pub right_row: u32,
    pub right_column: u32,
    pub p95_px: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ClosureReport {
    pub status: ClosureStatus,
    pub spanning_tree_edges: usize,
    pub participating_constraints: usize,
    pub constraint_weights: Vec<ClosureConstraintWeightRecord>,
    pub iterations: usize,
    pub residual_median_px: f64,
    pub residual_p95_px: f64,
    pub max_direct_pair_p95_px: f64,
    pub max_corner_correction_px: f64,
    pub clamped_stations: usize,
    pub fallback_reason: Option<String>,
    pub offending_pair: Option<ClosureOffendingPairRecord>,
}

impl Default for ClosureReport {
    fn default() -> Self {
        Self {
            status: ClosureStatus::NotRun,
            spanning_tree_edges: 0,
            participating_constraints: 0,
            constraint_weights: Vec::new(),
            iterations: 0,
            residual_median_px: 0.0,
            residual_p95_px: 0.0,
            max_direct_pair_p95_px: 0.0,
            max_corner_correction_px: 0.0,
            clamped_stations: 0,
            fallback_reason: None,
            offending_pair: None,
        }
    }
}

// ---------------------------------------------------------------------------
// residual_warp
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct WarpRegionRecord {
    pub world: WorldRect,
    pub node_step_px: u32,
    pub columns: u32,
    pub rows: u32,
    pub p95_before_px: f64,
    pub max_node_displacement_px: f64,
    pub max_neighbour_delta_px: f64,
    pub reverted_cells: u64,
}

impl Default for WarpRegionRecord {
    fn default() -> Self {
        Self {
            world: WorldRect::default(),
            node_step_px: RESIDUAL_WARP_NODE_STEP_PX,
            columns: 0,
            rows: 0,
            p95_before_px: 0.0,
            max_node_displacement_px: 0.0,
            max_neighbour_delta_px: 0.0,
            reverted_cells: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ResidualWarpReport {
    pub regions: Vec<WarpRegionRecord>,
    /// True while the model is the identity map everywhere (requirement 8.1).
    pub identity: bool,
}

impl Default for ResidualWarpReport {
    fn default() -> Self {
        Self {
            regions: Vec::new(),
            identity: true,
        }
    }
}

// ---------------------------------------------------------------------------
// tone
// ---------------------------------------------------------------------------

/// Tone_Harmonizer status (requirement 9.11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ToneStatus {
    Applied,
    /// Applied but a boundary Delta_E00 exceeded the threshold.
    Degraded,
    Identity,
    NotRun,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct ToneTileRecord {
    pub station_index: usize,
    pub gain: [f64; 3],
    pub offset: [f64; 3],
    pub samples: u64,
    pub clamped: bool,
    pub solved_gain: [f64; 3],
    pub solved_offset: [f64; 3],
}

impl Default for ToneTileRecord {
    fn default() -> Self {
        Self {
            station_index: 0,
            gain: [1.0, 1.0, 1.0],
            offset: [0.0, 0.0, 0.0],
            samples: 0,
            clamped: false,
            solved_gain: [1.0, 1.0, 1.0],
            solved_offset: [0.0, 0.0, 0.0],
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct TonePairWithoutEvidenceRecord {
    pub left: usize,
    pub right: usize,
    pub retained_samples: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct ToneBoundaryViolationRecord {
    pub left: usize,
    pub right: usize,
    pub delta_e00: f64,
    pub world: WorldRect,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ToneBoundaryReport {
    pub max: f64,
    pub threshold: f64,
    pub violations: Vec<ToneBoundaryViolationRecord>,
}

impl Default for ToneBoundaryReport {
    fn default() -> Self {
        Self {
            max: 0.0,
            threshold: TONE_BOUNDARY_DELTA_E_THRESHOLD,
            violations: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ToneReport {
    pub status: ToneStatus,
    pub low_pass_sigma_world_px: f64,
    pub tiles: Vec<ToneTileRecord>,
    pub pairs_without_evidence: Vec<TonePairWithoutEvidenceRecord>,
    pub boundary_delta_e: ToneBoundaryReport,
}

impl Default for ToneReport {
    fn default() -> Self {
        Self {
            status: ToneStatus::NotRun,
            low_pass_sigma_world_px: TONE_LOW_PASS_SIGMA_WORLD_PX,
            tiles: Vec::new(),
            pairs_without_evidence: Vec::new(),
            boundary_delta_e: ToneBoundaryReport::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// composition
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct CanvasRecord {
    pub width: u64,
    pub height: u64,
    pub long_side_limit: u64,
}

impl Default for CanvasRecord {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            long_side_limit: CANVAS_LONG_SIDE_LIMIT,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct NarrowOverlapRecord {
    pub left: usize,
    pub right: usize,
    pub overlap_width_px: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct CompositionReport {
    pub canvas: CanvasRecord,
    pub opaque_pixels: u64,
    pub union_projected_pixels: u64,
    pub narrow_overlaps: Vec<NarrowOverlapRecord>,
    pub integer_translation_tiles: usize,
    pub final_sharpen_amount: f64,
}

// ---------------------------------------------------------------------------
// quality_gate
// ---------------------------------------------------------------------------

/// Quality_Gate verdict (requirement 11.15 / 11.14).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum QualityGateVerdict {
    Pass,
    Fail,
    InsufficientEvidence,
    /// The gate is not wired into the run yet (observation-only stage).
    NotRun,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct FailedMeasurementRecord {
    pub world: WorldPoint,
    pub measured: f64,
    pub owner_path: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct QualityGateCriterionRecord {
    /// One of the stable criterion identifiers, e.g. `local_scale_median`.
    pub name: String,
    pub threshold: f64,
    pub measurable_count: usize,
    pub unmeasurable_count: usize,
    pub failed: Vec<FailedMeasurementRecord>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct UnmeasurableRecord {
    pub criterion: String,
    pub world: WorldPoint,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct QualityGateReport {
    pub verdict: QualityGateVerdict,
    pub roi_count: usize,
    pub criteria: Vec<QualityGateCriterionRecord>,
    pub unmeasurable: Vec<UnmeasurableRecord>,
}

impl Default for QualityGateReport {
    fn default() -> Self {
        Self {
            verdict: QualityGateVerdict::NotRun,
            roi_count: 0,
            criteria: Vec::new(),
            unmeasurable: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// output
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct PreviewRecord {
    pub path: String,
    pub max_roi_delta_e: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct OutputReport {
    pub written: bool,
    pub path: String,
    pub bit_depth: u8,
    pub alpha_preserved: bool,
    pub icc: String,
    pub result_id: String,
    pub preview: PreviewRecord,
}

impl Default for OutputReport {
    fn default() -> Self {
        Self {
            written: false,
            path: String::new(),
            bit_depth: 16,
            alpha_preserved: false,
            icc: "sRGB".to_string(),
            result_id: String::new(),
            preview: PreviewRecord::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// resources
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ResourcesReport {
    pub memory_threshold_bytes: u64,
    /// `default` or `setting` (requirement 14).
    pub memory_threshold_source: String,
    pub physical_memory_bytes: u64,
    pub peak_rss_bytes: u64,
    pub rss_sample_count: u64,
    /// Highest number of simultaneously leased full-size Virtual_Tiles
    /// (requirement 14.4).
    pub max_resident_virtual_tiles: usize,
    pub network_requests: u64,
    pub worker_threads: usize,
    /// How the RANSAC seed was derived (requirement 14.6).
    pub random_seed_source: String,
}

impl Default for ResourcesReport {
    fn default() -> Self {
        Self {
            memory_threshold_bytes: MEMORY_THRESHOLD_DEFAULT_BYTES,
            memory_threshold_source: "default".to_string(),
            physical_memory_bytes: 0,
            peak_rss_bytes: 0,
            rss_sample_count: 0,
            max_resident_virtual_tiles: 0,
            network_requests: 0,
            worker_threads: 0,
            random_seed_source: "unrecorded".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// degradation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct DegradationEntryRecord {
    /// A stable machine readable failure identifier (requirement 12.7).
    pub reason: String,
    /// `degraded` or `rejected`.
    pub severity: String,
    pub detail: Value,
}

impl Default for DegradationEntryRecord {
    fn default() -> Self {
        Self {
            reason: String::new(),
            severity: "degraded".to_string(),
            detail: Value::Object(Map::new()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct DegradationReport {
    pub entries: Vec<DegradationEntryRecord>,
    pub result: StackRunResult,
}

impl Default for DegradationReport {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            result: StackRunResult::Running,
        }
    }
}

// ---------------------------------------------------------------------------
// Stack_Report
// ---------------------------------------------------------------------------

/// The full Stack_Report document.
///
/// Every top-level field is always serialized, including defaults, so the
/// structure can be asserted without a RAW decode (requirement 15.9).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StackReport {
    pub schema: u32,
    pub pipeline_version: String,
    pub run_id: String,
    pub started_at_epoch_ms: u64,
    pub finished_at_epoch_ms: u64,
    pub selected_path: SelectedPath,

    pub input: InputReport,
    pub grouping: GroupingReport,
    pub intra_station: Vec<IntraStationReport>,
    pub fusion: Vec<FusionReport>,
    pub virtual_tiles: Vec<VirtualTileReport>,
    pub virtual_tile_cache: VirtualTileCacheSummary,
    pub topology: TopologyReport,
    pub station_relations: StationRelationsReport,
    pub closure: ClosureReport,
    pub residual_warp: ResidualWarpReport,
    pub tone: ToneReport,
    pub composition: CompositionReport,
    pub quality_gate: QualityGateReport,
    pub output: OutputReport,
    pub resources: ResourcesReport,
    pub degradation: DegradationReport,
}

impl Default for StackReport {
    fn default() -> Self {
        Self {
            schema: STACK_REPORT_SCHEMA,
            pipeline_version: String::new(),
            run_id: String::new(),
            started_at_epoch_ms: 0,
            finished_at_epoch_ms: 0,
            selected_path: SelectedPath::LayeredVirtualTile,

            input: InputReport::default(),
            grouping: GroupingReport::default(),
            intra_station: Vec::new(),
            fusion: Vec::new(),
            virtual_tiles: Vec::new(),
            virtual_tile_cache: VirtualTileCacheSummary::default(),
            topology: TopologyReport::default(),
            station_relations: StationRelationsReport::default(),
            closure: ClosureReport::default(),
            residual_warp: ResidualWarpReport::default(),
            tone: ToneReport::default(),
            composition: CompositionReport::default(),
            quality_gate: QualityGateReport::default(),
            output: OutputReport::default(),
            resources: ResourcesReport::default(),
            degradation: DegradationReport::default(),
        }
    }
}

/// Every top-level key the schema guarantees.  Used by the unit test below and
/// mirrored by `tests/focus-stack-quality-contract.mjs`.
#[allow(dead_code)]
pub(crate) const STACK_REPORT_TOP_LEVEL_FIELDS: &[&str] = &[
    "schema",
    "pipeline_version",
    "run_id",
    "started_at_epoch_ms",
    "finished_at_epoch_ms",
    "selected_path",
    "input",
    "grouping",
    "intra_station",
    "fusion",
    "virtual_tiles",
    "virtual_tile_cache",
    "topology",
    "station_relations",
    "closure",
    "residual_warp",
    "tone",
    "composition",
    "quality_gate",
    "output",
    "resources",
    "degradation",
];

impl StackReport {
    pub(crate) fn new(pipeline_version: &str) -> Self {
        Self {
            pipeline_version: pipeline_version.to_string(),
            run_id: Uuid::new_v4().to_string(),
            started_at_epoch_ms: epoch_millis(),
            ..Self::default()
        }
    }

    pub(crate) fn to_json_string(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|error| format!("Failed to serialize the stack report: {error}"))
    }
}

fn epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

/// Resolve `<app_cache_dir>/stack-reports` without forcing the caller to
/// import `tauri::Manager`.
pub(crate) fn resolve_stack_report_dir<R: tauri::Runtime>(
    app_handle: &tauri::AppHandle<R>,
) -> Option<PathBuf> {
    use tauri::Manager;

    app_handle
        .path()
        .app_cache_dir()
        .ok()
        .map(|directory| directory.join(STACK_REPORT_DIR_NAME))
}

/// Accumulates a [`StackReport`] behind a shared lock and guarantees that the
/// document reaches disk on every terminating path.
///
/// Requirement 10.8 asks for a report on success, degraded output, rejection
/// and cancellation.  Rather than threading a write call through every `?` in
/// the stitching entry point, the recorder writes on `Drop`, which covers the
/// early-return and unwind paths as well.  Write failures never change the
/// run's result; they are reported on stdout only.
pub(crate) struct StackReportRecorder {
    report: Mutex<StackReport>,
    directory: Option<PathBuf>,
    written: AtomicBool,
    /// Whether [`Self::write_once`] folds the process-wide run ledger into
    /// `degradation`.  Production always does.  The unit tests below turn it off
    /// because `cargo test` drives the ledger observation points of other
    /// modules concurrently in the same process, which would make an assertion
    /// on the collected entries depend on unrelated tests.
    bridge_run_ledger: bool,
    /// Re-reads every Source_RAW and returns its digest in lower hex, in the
    /// order the paths were handed in (requirement 4.8 / 12.8).  Installed by
    /// the focus-stack entry point and called once, on the terminating path, so
    /// success, failure and cancellation all end up comparing the same two
    /// numbers.  `None` leaves `sha256_after` empty, which is what the panorama
    /// path and the unit tests want.
    source_digest_recheck: Option<SourceDigestRecheck>,
}

/// Maps Source_RAW paths to their current content digests in lower hex.  A path
/// that cannot be read contributes an empty string: a re-check that is itself
/// impossible must not hide the digest that *was* taken before the run.
pub(crate) type SourceDigestRecheck = Box<dyn Fn(&[String]) -> Vec<String> + Send + Sync>;

impl StackReportRecorder {
    pub(crate) fn new(pipeline_version: &str, directory: Option<PathBuf>) -> Self {
        Self {
            report: Mutex::new(StackReport::new(pipeline_version)),
            directory,
            written: AtomicBool::new(false),
            bridge_run_ledger: true,
            source_digest_recheck: None,
        }
    }

    /// Install the read-only digest re-check of requirement 4.8 / 12.8.
    pub(crate) fn with_source_digest_recheck(mut self, recheck: SourceDigestRecheck) -> Self {
        self.source_digest_recheck = Some(recheck);
        self
    }

    /// Test-only recorder that never reads the process-wide run ledger, so an
    /// assertion on `degradation` depends on this test alone.
    #[cfg(test)]
    pub(crate) fn isolated(pipeline_version: &str, directory: Option<PathBuf>) -> Self {
        Self {
            report: Mutex::new(StackReport::new(pipeline_version)),
            directory,
            written: AtomicBool::new(false),
            bridge_run_ledger: false,
            source_digest_recheck: None,
        }
    }

    /// Mutate the in-progress report.  A poisoned lock is recovered from
    /// because losing the report must never mask the original failure.
    pub(crate) fn update<F>(&self, apply: F)
    where
        F: FnOnce(&mut StackReport),
    {
        let mut report = self
            .report
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        apply(&mut report);
    }

    pub(crate) fn set_selected_path(&self, selected_path: SelectedPath) {
        self.update(|report| report.selected_path = selected_path);
    }

    /// Used by the explicit terminating paths a later phase introduces; today
    /// the result is derived from the degradation ledger alone.
    #[allow(dead_code)]
    pub(crate) fn set_result(&self, result: StackRunResult) {
        self.update(|report| report.degradation.result = result);
    }

    /// Copy `ledger` into `degradation.entries` and let it refine
    /// `degradation.result` (requirement 12.9).  Called once per report, right
    /// before the document is serialized, so late terminating paths are still
    /// represented.
    pub(crate) fn apply_degradation_ledger(&self, ledger: &DegradationLedger) {
        let entries = degradation_entry_records(ledger);
        let ledger_result = ledger_run_result(ledger);
        let mut discarded_by_station = BTreeMap::<usize, u64>::new();
        for entry in ledger.entries().iter().filter(|entry| {
            entry.reason == degradation::STATION_RELATION_SINGLE_LAYER_EVIDENCE_DISCARDED
        }) {
            let count = entry
                .detail
                .get("count")
                .and_then(Value::as_u64)
                .unwrap_or(1);
            if let Some(stations) = entry.detail.get("stations").and_then(Value::as_array) {
                for station in stations.iter().filter_map(Value::as_u64) {
                    *discarded_by_station.entry(station as usize).or_default() += count;
                }
            }
        }
        let discarded_single_layer_evidence = discarded_by_station
            .into_iter()
            .map(|(station_index, count)| DiscardedEvidenceRecord {
                station_index,
                count,
            })
            .collect();
        self.update(|report| {
            report.degradation.entries = entries;
            report.degradation.result = report.degradation.result.merge(ledger_result);
            report.station_relations.discarded_single_layer_evidence =
                discarded_single_layer_evidence;
        });
    }

    /// Raise `resources.max_resident_virtual_tiles` to `observed`
    /// (requirement 14.4).  Monotonic: the report keeps the peak of every
    /// observation, so a lease counter that has already fallen back to zero
    /// cannot erase the maximum it reached.
    ///
    /// Called by the Tile_Compositor stage, which owns the
    /// [`super::virtual_tile::VirtualTileStore`] whose leases are counted; until
    /// then the field stays at its default of zero.
    #[allow(dead_code)]
    pub(crate) fn record_max_resident_virtual_tiles(&self, observed: usize) {
        self.update(|report| {
            report.resources.max_resident_virtual_tiles =
                report.resources.max_resident_virtual_tiles.max(observed);
        });
    }

    /// Re-read every Source_RAW listed in `input.sources` and record the digest
    /// in `sha256_after` (requirement 4.8 / 12.8).
    ///
    /// The digests are taken outside the report lock: hashing tens of RAW files
    /// must not block a concurrent observation point, and the closure has no
    /// business seeing the half-written report.  A digest that moved is printed,
    /// because the pipeline only ever opens a source read-only and a difference
    /// therefore means something outside this process rewrote the file mid-run.
    fn recheck_source_digests(&self) {
        let Some(recheck) = self.source_digest_recheck.as_ref() else {
            return;
        };
        let paths = self
            .report
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .input
            .sources
            .iter()
            .map(|source| source.path.clone())
            .collect::<Vec<_>>();
        if paths.is_empty() {
            return;
        }
        let digests = recheck(&paths);
        self.update(|report| {
            for (source, digest) in report.input.sources.iter_mut().zip(digests) {
                source.sha256_after = digest;
                if !source.sha256_before.is_empty()
                    && !source.sha256_after.is_empty()
                    && source.sha256_before != source.sha256_after
                {
                    println!(
                        "  - Source_RAW changed during the run: {} ({} -> {})",
                        source.path, source.sha256_before, source.sha256_after
                    );
                }
            }
        });
    }

    pub(crate) fn snapshot(&self) -> StackReport {
        self.report
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Write the report at most once.  Returns the written path when a file was
    /// produced by this call.
    pub(crate) fn write_once(&self) -> Option<PathBuf> {
        if self.written.swap(true, Ordering::SeqCst) {
            return None;
        }
        if self.bridge_run_ledger {
            self.apply_degradation_ledger(&degradation::run_ledger_snapshot());
            // The Intra_Station_Registrar observation points sit in the
            // refinement free functions and collect into a run scoped sink for
            // the same reason the ledger does (requirement 2.9).  Frames
            // already present win, so an explicit `update` from a later phase
            // is never overwritten by the sink.
            let intra_station = intra_station::run_records_snapshot();
            if !intra_station.is_empty() {
                self.update(|report| {
                    if report.intra_station.is_empty() {
                        report.intra_station = intra_station;
                    }
                });
            }
            // The Focus_Fuser sits inside the focus renderer, which has no
            // report parameter either (需求 3.5 / 3.8 / 3.9).
            let fusion = focus_fuser::run_records_snapshot();
            if !fusion.is_empty() {
                self.update(|report| {
                    if report.fusion.is_empty() {
                        report.fusion = fusion;
                    }
                });
            }
            // Residual_Warp is measured inside the stitching path, which has
            // no report parameter.  Copy the run-scoped records at the same
            // terminal point as Intra_Station and Focus_Fuser.
            let residual_warp = residual_warp::run_records_snapshot();
            if !residual_warp.is_empty() {
                self.update(|report| {
                    if report.residual_warp.regions.is_empty() {
                        report.residual_warp.regions = residual_warp;
                        report.residual_warp.identity = report.residual_warp.regions.is_empty();
                    }
                });
            }
        }
        // Every terminating path lands here, so this single call covers the
        // success, failure and cancellation re-check of requirement 4.8 / 12.8.
        self.recheck_source_digests();
        self.update(|report| report.finished_at_epoch_ms = epoch_millis());
        let report = self.snapshot();
        let directory = self.directory.as_deref()?;
        match write_stack_report(&report, directory) {
            Ok(path) => {
                println!("  - Stack report written to {}", path.display());
                Some(path)
            }
            Err(error) => {
                println!("  - Stack report could not be written: {error}");
                None
            }
        }
    }
}

impl Drop for StackReportRecorder {
    fn drop(&mut self) {
        self.write_once();
    }
}

/// Serialize `report` into `directory` using a temporary file plus rename so a
/// partially written document is never observable.
pub(crate) fn write_stack_report(
    report: &StackReport,
    directory: &Path,
) -> Result<PathBuf, String> {
    let json = report.to_json_string()?;
    fs::create_dir_all(directory).map_err(|error| {
        format!(
            "Failed to create the stack report directory {}: {error}",
            directory.display()
        )
    })?;
    let run_id = if report.run_id.is_empty() {
        "unknown".to_string()
    } else {
        report.run_id.clone()
    };
    let final_path = directory.join(format!("stack-report-{run_id}.json"));
    let temporary_path = directory.join(format!(".stack-report-{run_id}.json.tmp"));
    fs::write(&temporary_path, json.as_bytes()).map_err(|error| {
        format!(
            "Failed to write the stack report {}: {error}",
            temporary_path.display()
        )
    })?;
    fs::rename(&temporary_path, &final_path).map_err(|error| {
        let _ = fs::remove_file(&temporary_path);
        format!(
            "Failed to publish the stack report {}: {error}",
            final_path.display()
        )
    })?;
    Ok(final_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_report_value() -> Value {
        serde_json::to_value(StackReport::default()).expect("the default report must serialize")
    }

    #[test]
    fn stack_report_serializes_every_top_level_field() {
        let value = default_report_value();
        let object = value.as_object().expect("the report must be a JSON object");
        for field in STACK_REPORT_TOP_LEVEL_FIELDS {
            assert!(
                object.contains_key(*field),
                "the stack report schema must always write `{field}`"
            );
        }
        assert_eq!(
            object.len(),
            STACK_REPORT_TOP_LEVEL_FIELDS.len(),
            "the schema field list must stay in sync with the struct"
        );
    }

    #[test]
    fn stack_report_defaults_are_written_explicitly() {
        let value = default_report_value();
        assert_eq!(value["schema"], Value::from(STACK_REPORT_SCHEMA));
        assert_eq!(value["input"]["source_count"], Value::from(0));
        assert_eq!(
            value["input"]["sources"],
            Value::Array(Vec::new()),
            "an empty source list must still be present"
        );
        assert_eq!(value["grouping"]["station_count"], Value::from(0));
        assert_eq!(value["closure"]["status"], Value::from("not_run"));
        assert_eq!(value["closure"]["fallback_reason"], Value::Null);
        assert_eq!(value["residual_warp"]["identity"], Value::from(true));
        assert_eq!(
            value["tone"]["boundary_delta_e"]["threshold"],
            Value::from(TONE_BOUNDARY_DELTA_E_THRESHOLD)
        );
        assert_eq!(
            value["composition"]["canvas"]["long_side_limit"],
            Value::from(CANVAS_LONG_SIDE_LIMIT)
        );
        assert_eq!(value["quality_gate"]["verdict"], Value::from("not_run"));
        assert_eq!(
            value["virtual_tile_cache"]["limit_bytes"],
            Value::from(VIRTUAL_TILE_CACHE_LIMIT_BYTES)
        );
        assert_eq!(
            value["resources"]["memory_threshold_bytes"],
            Value::from(MEMORY_THRESHOLD_DEFAULT_BYTES)
        );
        assert_eq!(value["degradation"]["result"], Value::from("running"));
    }

    #[test]
    fn stack_report_round_trips_through_json() {
        let report = StackReport::new("stack-test-1");
        let json = report.to_json_string().expect("serialization must succeed");
        let parsed: StackReport = serde_json::from_str(&json).expect("the schema must round-trip");
        assert_eq!(parsed, report);
    }

    #[test]
    fn selected_path_and_result_identifiers_are_stable() {
        for (variant, identifier) in [
            (SelectedPath::LayeredVirtualTile, "layered_virtual_tile"),
            (SelectedPath::ProgressiveSeamTile, "progressive_seam_tile"),
            (SelectedPath::StreamingMosaic, "streaming_mosaic"),
            (
                SelectedPath::LegacySingleLayerMosaic,
                "legacy_single_layer_mosaic",
            ),
            (SelectedPath::SingleStation, "single_station"),
        ] {
            assert_eq!(variant.as_identifier(), identifier);
            assert_eq!(
                serde_json::to_value(variant).expect("serialization must succeed"),
                Value::from(identifier),
                "the serialized form must match the stable identifier"
            );
        }
        for (variant, identifier) in [
            (StackRunResult::Running, "running"),
            (StackRunResult::Success, "success"),
            (StackRunResult::Degraded, "degraded"),
            (StackRunResult::Rejected, "rejected"),
            (StackRunResult::Cancelled, "cancelled"),
        ] {
            assert_eq!(variant.as_identifier(), identifier);
            assert_eq!(
                serde_json::to_value(variant).expect("serialization must succeed"),
                Value::from(identifier)
            );
        }
    }

    #[test]
    fn recorder_writes_exactly_one_report_on_drop() {
        let directory = tempfile::tempdir().expect("a temporary directory must be available");
        let path = {
            let recorder =
                StackReportRecorder::isolated("stack-test-1", Some(directory.path().to_path_buf()));
            recorder.update(|report| {
                report.input.source_count = 84;
                report.input.max_sources = 500;
            });
            recorder.set_selected_path(SelectedPath::ProgressiveSeamTile);
            recorder.set_result(StackRunResult::Success);
            recorder.write_once().expect("the report must be written")
        };
        assert!(path.exists(), "the report file must exist after the write");

        let entries = fs::read_dir(directory.path())
            .expect("the directory must be readable")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            entries.len(),
            1,
            "an atomic write must not leave a temporary file behind: {entries:?}"
        );

        let parsed: StackReport =
            serde_json::from_str(&fs::read_to_string(&path).expect("the report must be readable"))
                .expect("the written report must parse");
        assert_eq!(parsed.input.source_count, 84);
        assert_eq!(parsed.selected_path, SelectedPath::ProgressiveSeamTile);
        assert_eq!(parsed.degradation.result, StackRunResult::Success);
        assert!(
            parsed.finished_at_epoch_ms >= parsed.started_at_epoch_ms,
            "the finish timestamp must be recorded when the report is written"
        );
    }

    #[test]
    fn recorder_without_a_directory_stays_silent() {
        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        assert!(recorder.write_once().is_none());
        assert!(recorder.write_once().is_none());
    }

    #[test]
    fn ledger_entries_reach_the_report_with_their_severity() {
        let mut ledger = DegradationLedger::new();
        ledger.record(
            degradation::TONE_INSUFFICIENT_SAMPLES,
            serde_json::json!({ "retained_samples": 12 }),
        );
        ledger.record(
            degradation::GEOMETRY_DISCONNECTED,
            serde_json::json!({ "components": 2 }),
        );

        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.set_result(StackRunResult::Success);
        recorder.apply_degradation_ledger(&ledger);
        let report = recorder.snapshot();

        assert_eq!(report.degradation.entries.len(), 2);
        assert_eq!(
            report.degradation.entries[0].reason,
            degradation::TONE_INSUFFICIENT_SAMPLES
        );
        assert_eq!(report.degradation.entries[0].severity, "degraded");
        assert_eq!(
            report.degradation.entries[0].detail["retained_samples"],
            Value::from(12)
        );
        assert_eq!(
            report.degradation.entries[1].reason,
            degradation::GEOMETRY_DISCONNECTED
        );
        assert_eq!(report.degradation.entries[1].severity, "rejected");
        assert_eq!(
            report.degradation.result,
            StackRunResult::Rejected,
            "a ledger rejection must override the compositor's success"
        );
    }

    #[test]
    fn discarded_single_layer_station_evidence_is_counted_per_station() {
        let mut ledger = DegradationLedger::new();
        ledger.record(
            degradation::STATION_RELATION_SINGLE_LAYER_EVIDENCE_DISCARDED,
            serde_json::json!({
                "count": 1,
                "stations": [2, 5],
                "independent_support": 1,
            }),
        );
        ledger.record(
            degradation::STATION_RELATION_SINGLE_LAYER_EVIDENCE_DISCARDED,
            serde_json::json!({
                "count": 2,
                "stations": [2, 7],
                "independent_support": 1,
            }),
        );

        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.apply_degradation_ledger(&ledger);
        let report = recorder.snapshot();

        assert_eq!(
            report.station_relations.discarded_single_layer_evidence,
            vec![
                DiscardedEvidenceRecord {
                    station_index: 2,
                    count: 3,
                },
                DiscardedEvidenceRecord {
                    station_index: 5,
                    count: 1,
                },
                DiscardedEvidenceRecord {
                    station_index: 7,
                    count: 2,
                },
            ]
        );
        assert_eq!(
            report
                .degradation
                .entries
                .iter()
                .filter(|entry| {
                    entry.reason == degradation::STATION_RELATION_SINGLE_LAYER_EVIDENCE_DISCARDED
                })
                .count(),
            2,
            "the stable discard reason remains countable in the report ledger"
        );
    }

    #[test]
    fn an_empty_ledger_leaves_the_recorded_result_alone() {
        let ledger = DegradationLedger::new();
        for recorded in [
            StackRunResult::Success,
            StackRunResult::Rejected,
            StackRunResult::Cancelled,
        ] {
            let recorder = StackReportRecorder::isolated("stack-test-1", None);
            recorder.set_result(recorded);
            recorder.apply_degradation_ledger(&ledger);
            assert_eq!(recorder.snapshot().degradation.result, recorded);
            assert!(recorder.snapshot().degradation.entries.is_empty());
        }
    }

    #[test]
    fn cancellation_is_never_downgraded_by_the_ledger() {
        let mut ledger = DegradationLedger::new();
        ledger.record(degradation::TONE_GAIN_CLAMPED, Value::Null);
        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.set_result(StackRunResult::Cancelled);
        recorder.apply_degradation_ledger(&ledger);
        assert_eq!(
            recorder.snapshot().degradation.result,
            StackRunResult::Cancelled
        );

        // A ledger that saw the cancellation itself reports it as cancelled
        // rather than as the plain rejection its severity implies.
        let mut cancelled = DegradationLedger::new();
        cancelled.record(degradation::RUN_CANCELLED_BY_USER, Value::Null);
        assert_eq!(ledger_run_result(&cancelled), StackRunResult::Cancelled);

        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.set_result(StackRunResult::Success);
        recorder.apply_degradation_ledger(&cancelled);
        assert_eq!(
            recorder.snapshot().degradation.result,
            StackRunResult::Cancelled
        );
    }

    #[test]
    fn the_terminating_write_rechecks_every_source_digest() {
        let before = ["11".repeat(32), "22".repeat(32)];
        let after = ["11".repeat(32), "33".repeat(32)];
        let observed = after.clone();
        let recorder = StackReportRecorder::isolated("stack-test-1", None)
            .with_source_digest_recheck(Box::new(move |paths| {
                assert_eq!(paths, ["/tmp/a.NEF", "/tmp/b.NEF"]);
                observed.to_vec()
            }));
        recorder.update(|report| {
            report.input.sources = vec![
                SourceRecord {
                    path: "/tmp/a.NEF".to_string(),
                    sha256_before: before[0].clone(),
                    ..Default::default()
                },
                SourceRecord {
                    path: "/tmp/b.NEF".to_string(),
                    sha256_before: before[1].clone(),
                    ..Default::default()
                },
            ];
        });
        assert!(
            recorder
                .snapshot()
                .input
                .sources
                .iter()
                .all(|source| source.sha256_after.is_empty()),
            "the after digest is only taken on the terminating path"
        );

        recorder.write_once();
        let sources = recorder.snapshot().input.sources;
        assert_eq!(sources[0].sha256_before, sources[0].sha256_after);
        assert_eq!(sources[1].sha256_after, after[1]);
        assert_ne!(
            sources[1].sha256_before, sources[1].sha256_after,
            "a source rewritten behind the pipeline's back must stay visible in the report"
        );
    }

    #[test]
    fn a_recorder_without_a_recheck_leaves_the_after_digest_empty() {
        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.update(|report| {
            report.input.sources = vec![SourceRecord {
                path: "/tmp/a.NEF".to_string(),
                sha256_before: "44".repeat(32),
                ..Default::default()
            }];
        });
        recorder.write_once();
        let sources = recorder.snapshot().input.sources;
        assert_eq!(sources[0].sha256_before, "44".repeat(32));
        assert!(sources[0].sha256_after.is_empty());
    }

    #[test]
    fn the_resident_virtual_tile_peak_only_grows() {
        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        for observed in [1, 2, 1, 0] {
            recorder.record_max_resident_virtual_tiles(observed);
        }
        assert_eq!(
            recorder.snapshot().resources.max_resident_virtual_tiles,
            2,
            "the report keeps the peak, not the last observation"
        );
    }

    #[test]
    fn a_degraded_ledger_upgrades_success_but_not_rejection() {
        let mut ledger = DegradationLedger::new();
        ledger.record(degradation::CACHE_WRITE_UNAVAILABLE, Value::Null);

        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.set_result(StackRunResult::Success);
        recorder.apply_degradation_ledger(&ledger);
        assert_eq!(
            recorder.snapshot().degradation.result,
            StackRunResult::Degraded
        );

        let recorder = StackReportRecorder::isolated("stack-test-1", None);
        recorder.set_result(StackRunResult::Rejected);
        recorder.apply_degradation_ledger(&ledger);
        assert_eq!(
            recorder.snapshot().degradation.result,
            StackRunResult::Rejected
        );
    }
}
