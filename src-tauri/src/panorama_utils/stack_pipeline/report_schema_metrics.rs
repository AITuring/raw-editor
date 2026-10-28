//! Rendering, quality-gate, output and degradation report records.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::report_schema_core::*;

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
    /// Independently measured normal constraints for a separate depth layer.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub normal_layers: Vec<super::super::line_layers::NormalAlignmentReport>,
    /// True while the model is the identity map everywhere (requirement 8.1).
    pub identity: bool,
}

impl Default for ResidualWarpReport {
    fn default() -> Self {
        Self {
            regions: Vec::new(),
            normal_layers: Vec::new(),
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
    /// Unbounded station-wide baseline solve. For supported_spatial_affine,
    /// the applied model is described by gain/offset and spatial ranges below.
    pub solved_gain: [f64; 3],
    pub solved_offset: [f64; 3],
    /// Strength of the measured correction after source-fidelity bounds.
    #[serde(default = "full_tone_strength")]
    pub source_fidelity_strength: f64,
    #[serde(default)]
    pub spatial_gain_min: [f64; 3],
    #[serde(default)]
    pub spatial_gain_max: [f64; 3],
    #[serde(default)]
    pub spatial_offset_min: [f64; 3],
    #[serde(default)]
    pub spatial_offset_max: [f64; 3],
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
            source_fidelity_strength: 1.0,
            spatial_gain_min: [1.0; 3],
            spatial_gain_max: [1.0; 3],
            spatial_offset_min: [0.0; 3],
            spatial_offset_max: [0.0; 3],
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
    #[serde(default)]
    pub correction_model: String,
    /// Native-validated radiometric fitting relations, distinct from the
    /// closure optimizer's retained geometric constraints.
    #[serde(default)]
    pub native_fitting_relations: Vec<[usize; 2]>,
    #[serde(default)]
    pub geometry_active_relation_count: usize,
}

impl Default for ToneReport {
    fn default() -> Self {
        Self {
            status: ToneStatus::NotRun,
            low_pass_sigma_world_px: TONE_LOW_PASS_SIGMA_WORLD_PX,
            tiles: Vec::new(),
            pairs_without_evidence: Vec::new(),
            boundary_delta_e: ToneBoundaryReport::default(),
            correction_model: "station_affine".into(),
            native_fitting_relations: Vec::new(),
            geometry_active_relation_count: 0,
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum QualityGateVerdict {
    Pass,
    Fail,
    InsufficientEvidence,
    /// The criterion had no applicable content, but no technical evidence was
    /// lost. This is distinct from insufficient evidence for the whole gate.
    NotApplicable,
    /// No Quality_Gate measurement was produced for this report.
    #[default]
    NotRun,
}

/// Why one of the fixed quality-gate observations could not be measured.
/// These names are part of the report contract so callers can distinguish an
/// empty/content-inapplicable ROI from a broken input or measurement path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UnmeasurableCategory {
    ContentNotApplicable,
    #[default]
    Technical,
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
    /// Lower/upper bounds are written for criteria whose acceptance range is
    /// two-sided (for example Noise_Sigma). `threshold` remains the canonical
    /// bound for backwards-compatible readers.
    #[serde(default)]
    pub threshold_min: Option<f64>,
    #[serde(default)]
    pub threshold_max: Option<f64>,
    pub measurable_count: usize,
    pub unmeasurable_count: usize,
    #[serde(default)]
    pub content_not_applicable_count: usize,
    #[serde(default)]
    pub technical_unmeasurable_count: usize,
    #[serde(default)]
    pub station_statistics: Vec<QualityStationStatistic>,
    #[serde(default)]
    pub verdict: QualityGateVerdict,
    /// Stable reason histogram for every item that could not be measured.
    #[serde(default)]
    pub unmeasurable_reasons: BTreeMap<String, usize>,
    /// Every measurable value, including passing values. The historical
    /// `failed` field is retained as the compact failure diagnostic list.
    #[serde(default)]
    pub measured: Vec<FailedMeasurementRecord>,
    /// Diagnostic-only criteria remain visible but do not establish a gate
    /// failure (for example render scale below one or final sharpening).
    #[serde(default)]
    pub diagnostic: bool,
    pub failed: Vec<FailedMeasurementRecord>,
    /// Requirement 11.12's split owner shortfall evidence. These fields are
    /// populated for `owner_sharpness_coverage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub textured_pixel_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excluded_flat_pixel_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub textured_low_confidence_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flat_low_confidence_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub textured_shortfall_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flat_shortfall_ratio: Option<f64>,
    #[serde(default)]
    pub shortfall_histogram: Vec<u64>,
    #[serde(default)]
    pub disagreement_veto_count: u64,
    #[serde(default)]
    pub unresolved_pixel_count: u64,
    /// Exact numerator/denominator used by the whole-image effective-pixel
    /// criterion.  Keeping these with the criterion lets the terminal
    /// composition report reuse the same scan without a second full-canvas
    /// traversal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_output_pixels: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_projected_pixels: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct UnmeasurableRecord {
    pub criterion: String,
    pub world: WorldPoint,
    pub reason: String,
    #[serde(default)]
    pub category: UnmeasurableCategory,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct QualityStationStatistic {
    pub station_id: usize,
    pub numerator: u64,
    pub denominator: u64,
    pub value: Option<f64>,
    #[serde(default)]
    pub textured_pixel_count: u64,
    #[serde(default)]
    pub excluded_flat_pixel_count: u64,
    #[serde(default)]
    pub textured_low_confidence_count: u64,
    #[serde(default)]
    pub flat_low_confidence_count: u64,
    #[serde(default)]
    pub textured_low_confidence_ratio: Option<f64>,
    #[serde(default)]
    pub flat_low_confidence_ratio: Option<f64>,
    #[serde(default)]
    pub textured_shortfall_count: u64,
    #[serde(default)]
    pub flat_shortfall_count: u64,
    #[serde(default)]
    pub textured_shortfall_ratio: Option<f64>,
    #[serde(default)]
    pub flat_shortfall_ratio: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct ConfidenceScoreDistribution {
    pub textured_low_count: u64,
    pub both_scores_at_least_0_10: u64,
    pub winner_below_0_10: u64,
    pub single_candidate_count: u64,
    pub winner_sum: f64,
    pub runner_up_sum: f64,
    /// Pixel-weighted counts in fixed 0.05-wide bins, including 1.0 in bin 20.
    pub winner_histogram: Vec<u64>,
    pub runner_up_histogram: Vec<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct OwnerReverseLookupFailures {
    #[serde(default)]
    pub compositor_owner_zero: u64,
    #[serde(default)]
    pub station_missing: u64,
    #[serde(default)]
    pub transform_failed: u64,
    #[serde(default)]
    pub out_of_bounds: u64,
    #[serde(default)]
    pub raw_owner_unresolved: u64,
    /// The rounded station pixel had no owner, but an owner was found in the
    /// compositor's exact 4x4 bicubic sampling footprint.
    #[serde(default)]
    pub snapped_to_footprint: u64,
    /// The owner was found but its ownership cell carried no finite
    /// Sharpness_Score evidence.
    #[serde(default)]
    pub evidence_missing: u64,
    /// Non-transparent output pixels whose source sampling footprint included
    /// an uncovered station pixel. These are diagnostic-only dark-edge
    /// measurements and do not alter compositor writes.
    #[serde(default)]
    pub footprint_uncovered_nontransparent: u64,
    #[serde(default)]
    pub footprint_brightness_ratio: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct RoiPhotometryRecord {
    pub world: WorldPoint,
    pub station_id: usize,
    pub owner_path: String,
    pub output_mean_rgb: [f32; 3],
    pub reference_mean_rgb: [f32; 3],
    pub channel_ratio: [Option<f64>; 3],
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct QualityGateTimingRecord {
    pub decode_seconds: f64,
    pub reference_resampling_seconds: f64,
    pub local_scale_seconds: f64,
    pub slanted_edge_mtf_seconds: f64,
    pub gradient_seconds: f64,
    pub noise_seconds: f64,
    pub delta_e_seconds: f64,
    pub roi_sampling_seconds: f64,
    pub boundary_seconds: f64,
    pub effective_coverage_seconds: f64,
    pub owner_source_decode_count: usize,
    pub roi_reference_count: usize,
    pub boundary_sample_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct QualityGateReport {
    pub verdict: QualityGateVerdict,
    pub roi_count: usize,
    pub criteria: Vec<QualityGateCriterionRecord>,
    pub unmeasurable: Vec<UnmeasurableRecord>,
    #[serde(default)]
    pub confidence_scores: ConfidenceScoreDistribution,
    #[serde(default)]
    pub roi_photometry: Vec<RoiPhotometryRecord>,
    #[serde(default)]
    pub timing: QualityGateTimingRecord,
    #[serde(default)]
    pub owner_reverse_lookup_failures: OwnerReverseLookupFailures,
}

impl Default for QualityGateReport {
    fn default() -> Self {
        Self {
            verdict: QualityGateVerdict::NotRun,
            roi_count: 0,
            criteria: Vec::new(),
            unmeasurable: Vec::new(),
            confidence_scores: ConfidenceScoreDistribution::default(),
            roi_photometry: Vec::new(),
            timing: QualityGateTimingRecord::default(),
            owner_reverse_lookup_failures: OwnerReverseLookupFailures::default(),
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
    #[serde(default)]
    pub format: String,
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
            format: String::new(),
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
    /// `default`, `user_configured`, or `auto_calibrated` (requirement 14).
    pub memory_threshold_source: String,
    pub physical_memory_bytes: u64,
    pub peak_rss_bytes: u64,
    pub rss_sample_count: u64,
    /// The sampler observed the threshold before the run stopped at a
    /// cooperative stage boundary.
    pub memory_threshold_exceeded: bool,
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
            memory_threshold_exceeded: false,
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

fn full_tone_strength() -> f64 {
    1.0
}
