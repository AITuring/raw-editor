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
