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
