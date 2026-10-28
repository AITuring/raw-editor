//! Core Stack_Report schema values: identity, input, grouping and intra-station records.

use serde::{Deserialize, Serialize};

use super::super::intra_station::INTRA_STATION_ANALYSIS_LONG_SIDE;
use super::super::proxy::{ProxyBudget, ProxyImage};

/// Report-level identity for the feature provider boundary.  The native
/// provider is always available; optional Python providers must reduce to this
/// compact record before their keypoints or descriptors reach Rust geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct FeatureProviderReport {
    pub requested: String,
    pub selected: String,
    pub provider_version: String,
    pub fallback_reason: Option<String>,
    pub explicit_copy_bytes: u64,
}

impl Default for FeatureProviderReport {
    fn default() -> Self {
        Self {
            requested: "rust_fast_brief".to_string(),
            selected: "rust_fast_brief".to_string(),
            provider_version: "native".to_string(),
            fallback_reason: None,
            explicit_copy_bytes: 0,
        }
    }
}

/// Bumped whenever the serialized shape of [`StackReport`] changes.
pub(crate) const STACK_REPORT_SCHEMA: u32 = 3;

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
    /// [`super::super::compositor::StackCompositorChoice::selected_path`]; the
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
    pub(crate) fn merge(self, ledger: Self) -> Self {
        if ledger.precedence() > self.precedence() {
            ledger
        } else {
            self
        }
    }
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
    /// The shared proxy budget and per-source coordinate metadata used by
    /// feature, geometry and seam analysis (requirements 16.1–16.2).
    #[serde(default)]
    pub proxy_budget: Option<ProxyBudget>,
    #[serde(default)]
    pub proxy_images: Vec<ProxyImage>,
    /// Provider identity and optional Python fallback evidence (requirements
    /// 17.1–17.2).  This stays nested so the established Stack_Report
    /// top-level schema remains unchanged.
    #[serde(default)]
    pub feature_provider: FeatureProviderReport,
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
