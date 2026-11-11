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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct LegacyPublicationCropReport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub covered_pixels_excluded: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct NarrowOverlapRecord {
    pub left: usize,
    pub right: usize,
    pub overlap_width_px: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct OutputDensityReport {
    pub status: String,
    pub factor: f64,
    pub pixel_count_multiplier: f64,
    pub canvas_pixels: u64,
    pub output_plane_lower_bound_bytes: u64,
    pub target_minimum_scale: f64,
    pub probe_step_px: u32,
    pub sampling_contract: String,
    pub source_count: usize,
    pub sample_count: usize,
    pub invalid_probe_count: usize,
    pub boundary_probe_count: usize,
    pub worst_station_index: Option<usize>,
    pub worst_source_id: Option<usize>,
    pub worst_station_point: Option<[f64; 2]>,
    /// Original independent closure evidence converted to output-pixel units.
    pub direct_pair_p95_output_px: Option<f64>,
    pub residual_median_output_px: Option<f64>,
    pub max_corner_correction_output_px: Option<f64>,
    pub minimum_scale: Option<f64>,
    pub p01_scale: Option<f64>,
    pub median_scale: Option<f64>,
    pub bypass_reason: Option<String>,
}

impl Default for OutputDensityReport {
    fn default() -> Self {
        Self {
            status: "not_run".into(),
            factor: 1.0,
            pixel_count_multiplier: 1.0,
            canvas_pixels: 0,
            output_plane_lower_bound_bytes: 0,
            target_minimum_scale: 1.0,
            probe_step_px: 0,
            sampling_contract: String::new(),
            source_count: 0,
            sample_count: 0,
            invalid_probe_count: 0,
            boundary_probe_count: 0,
            worst_station_index: None,
            worst_source_id: None,
            worst_station_point: None,
            direct_pair_p95_output_px: None,
            residual_median_output_px: None,
            max_corner_correction_output_px: None,
            minimum_scale: None,
            p01_scale: None,
            median_scale: None,
            bypass_reason: None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct CompositionReport {
    pub canvas: CanvasRecord,
    /// All nontransparent output pixels; not the physical-domain intersection.
    pub opaque_pixels: u64,
    /// Historical projected canvas-rectangle union. Physical sensor support is
    /// recorded separately in Quality_Gate's effective-pixel criterion.
    pub union_projected_pixels: u64,
    /// The legacy RGB mosaic does not publish its internal source mask. Its
    /// zero counters above must not be interpreted as measured empty coverage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_coverage_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_covered_pixels: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_owned_pixels: Option<u64>,
    /// Area of the full source-union bounding canvas before any explicit
    /// publication crop; `canvas` describes the delivered artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_output_pixels: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub publication_crop: Option<LegacyPublicationCropReport>,
    pub narrow_overlaps: Vec<NarrowOverlapRecord>,
    pub integer_translation_tiles: usize,
    pub final_sharpen_amount: f64,
    /// Configured versus actually selected compositor path, including any
    /// source-scan guard fallback.
    #[serde(default)]
    pub path_selection: PathSelectionObservation,
    /// Output lattice selected before composition from native source Jacobians.
    /// Independent matching evidence stays in its original measurement units.
    #[serde(default)]
    pub output_density: OutputDensityReport,
}

// ---------------------------------------------------------------------------
// performance
// ---------------------------------------------------------------------------

/// Wall-clock durations captured at the stable boundaries of one stack run.
/// These values are observations only; no path selection or quality decision
/// reads them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub(crate) struct StageTimingReport {
    pub preflight_seconds: f64,
    pub prepare_seconds: f64,
    pub matching_seconds: f64,
    pub ordering_seconds: f64,
    pub render_seconds: f64,
    pub total_seconds: f64,
}

impl Default for StageTimingReport {
    fn default() -> Self {
        Self {
            preflight_seconds: 0.0,
            prepare_seconds: 0.0,
            matching_seconds: 0.0,
            ordering_seconds: 0.0,
            render_seconds: 0.0,
            total_seconds: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct PairCountReport {
    /// Input/filename-neighbour pairs before compact retrieval promotion.
    pub base_pairs: usize,
    /// Newly promoted compact-descriptor pairs (deduplicated).
    pub retrieval_pairs: usize,
    pub candidate_pairs: usize,
    pub verified_pairs: usize,
    /// Verified pairs added by disconnected-component rescue.
    pub rescue_pairs: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SourcePreflightReport {
    pub source_count: usize,
    pub stat_count: usize,
    pub file_map_count: usize,
    pub hash_count: usize,
    pub exif_parse_count: usize,
    pub preflight_reuse_count: usize,
    pub raw_decode_count: usize,
    pub failed_count: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct QualityRoiCacheReport {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub resident_bytes: usize,
    pub resident_entries: usize,
    pub peak_bytes: usize,
}

/// Independently bounded temporary disk payloads; metadata is charged to the RAM cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct CacheBackingReport {
    pub max_bytes: usize,
    pub resident_bytes: usize,
    pub peak_bytes: usize,
    pub hits: usize,
    pub writes: usize,
    pub read_bytes: usize,
    pub written_bytes: usize,
    pub read_seconds: f64,
    pub write_seconds: f64,
    pub failures: usize,
    pub evictions: usize,
}

/// Bounded source-image reuse shared by the compositor and Quality_Gate.
/// Values are cumulative for the run, while resident/peak fields describe the
/// current byte-bounded LRU footprint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SourceRenderCacheReport {
    pub backing: CacheBackingReport,
    pub max_bytes: usize,
    pub hits: usize,
    pub misses: usize,
    pub render_hits: usize,
    pub render_misses: usize,
    pub quality_gate_hits: usize,
    pub quality_gate_misses: usize,
    pub evictions: usize,
    pub oversized_bypasses: usize,
    pub resident_bytes: usize,
    pub resident_entries: usize,
    pub peak_bytes: usize,
}

/// Run-local complete station fusion reuse. Byte counts conservatively include
/// source metadata/image allocations retained through native sampling Arcs.
/// Active compositor payloads and the separate source cache are not included.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct NativeFusionCacheReport {
    pub backing: CacheBackingReport,
    pub max_bytes: usize,
    pub hits: usize,
    pub misses: usize,
    pub matching_hits: usize,
    pub matching_misses: usize,
    pub composition_hits: usize,
    pub composition_misses: usize,
    pub fusion_count: usize,
    pub fusion_seconds: f64,
    pub failed_fusions: usize,
    pub evictions: usize,
    pub invalidations: usize,
    pub oversized_bypasses: usize,
    pub unattributed_bypasses: usize,
    pub state_snapshot_bypasses: usize,
    pub resident_bytes: usize,
    pub resident_entries: usize,
    pub peak_bytes: usize,
    pub peak_entries: usize,
    /// Full raster payloads retained in RAM, excluding backed metadata entries.
    pub peak_memory_entries: usize,
}

/// Compositor payload requests, separate from fusion and RAW decode counters.
/// Requests are counted even if a later source lookup or load fails.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct TilePayloadReport {
    pub raster_only_requests: usize,
    pub owner_sampling_requests: usize,
    pub owner_payload_build_requests: usize,
    /// Calls to the prepared-source loader; these may hit its source cache.
    pub owner_source_load_requests: usize,
    /// Attributed station payloads omitted because only the raster was needed.
    pub raster_only_payload_skips: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct PerformanceReport {
    #[serde(default)]
    pub stage_seconds: StageTimingReport,
    #[serde(default)]
    pub pairs: PairCountReport,
    #[serde(default)]
    pub source_preflight: SourcePreflightReport,
    #[serde(default)]
    pub quality_roi_cache: QualityRoiCacheReport,
    #[serde(default)]
    pub source_render_cache: SourceRenderCacheReport,
    #[serde(default)]
    pub native_fusion_cache: NativeFusionCacheReport,
    #[serde(default)]
    pub tile_payload: TilePayloadReport,
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
    /// Number of candidate observations offered to this criterion before its
    /// content-specific eligibility checks.
    #[serde(default)]
    pub candidate_count: usize,
    /// Number of candidates that reached the metric implementation and
    /// produced a measurable value.
    #[serde(default)]
    pub eligible_count: usize,
    /// Candidate observations rejected because the artwork content did not
    /// satisfy this criterion's measurement contract (for example no slanted
    /// edge or no flat patch). This remains separate from technical failures.
    #[serde(default)]
    pub rejected_content_count: usize,
    /// Reasons discarded while searching for content-eligible candidates.
    /// These are selection observations, never substitutes for a metric pass.
    #[serde(default)]
    pub rejected_content_reasons: BTreeMap<String, usize>,
    /// Bounded coordinate examples for rejected candidates (at most 32).
    #[serde(default)]
    pub rejected_content_examples: Vec<UnmeasurableRecord>,
    /// Stable selector identifier for the candidate pool used by this
    /// criterion. Empty means the historical shared ROI path.
    #[serde(default)]
    pub selection_mode: String,
    /// Enumerated candidates not visited after the bounded reference budget
    /// was reached. They do not enter eligibility or failure denominators.
    #[serde(default)]
    pub unvisited_candidate_count: usize,
    #[serde(default)]
    pub reference_roi_limit: usize,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_source_domain: Option<EffectiveSourceDomainRecord>,
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
#[serde(default)]
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
    /// Number of ROIs whose geometric eligibility came from a station-level
    /// owner plane while their reference is assembled from the participating
    /// Source_RAW owners in that station.
    pub station_owner_evidence_roi_count: usize,
    pub raw_owner_evidence_roi_count: usize,
    pub boundary_sample_count: usize,
    /// Number of post-tone Quality_Gate reference ROIs served from the
    /// bounded per-run LRU instead of being sampled from a decoded RAW.
    pub roi_cache_hits: usize,
    /// Cache lookups that required a source load or could not be retained.
    pub roi_cache_misses: usize,
    pub roi_cache_evictions: usize,
    pub roi_cache_resident_bytes: usize,
    pub roi_cache_peak_bytes: usize,
}

/// Evidence from the actual legacy streaming source sampler, separate from
/// focus-station fusion evidence. Missing cells remain unresolved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct LegacyNativeEvidenceReport {
    pub method: String,
    /// Actual source-match edges; filename/sequence bridges are excluded.
    pub accepted_source_pairs: Vec<(usize, usize)>,
    pub focus_population: String,
    pub capture_station_count: usize,
    pub adaptive_pixel_count: u64,
    pub adaptive_cell_count: usize,
    pub adaptive_source_load_count: usize,
    pub cell_size_px: u32,
    pub sample_window_px: f64,
    pub sampled_cell_count: usize,
    pub candidate_measurement_count: usize,
    pub delivered_pixels: usize,
    pub unresolved_pixel_count: u64,
    pub world_origin: (f64, f64),
    pub sources: Vec<LegacyNativeSourceRecord>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct LegacyNativeSourceRecord {
    pub capture_station_id: usize,
    pub source_tone_replayed: bool,
    pub owner_gain_replayed: bool,
    pub owner: u16,
    pub source_id: usize,
    pub path: String,
    pub dimensions: (u32, u32),
    /// Row-major source -> world matrix (before the measured local field).
    pub source_to_world: Vec<f64>,
    pub residual_origin: (u32, u32),
    pub residual_scale: f64,
    pub residual_nodes: (usize, usize),
    pub residual_max_offset_px: f64,
    pub source_prefilter_divisor: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct QualityGateReport {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_native_evidence: Option<LegacyNativeEvidenceReport>,
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
            legacy_native_evidence: None,
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
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    pub bit_depth: u8,
    pub alpha_preserved: bool,
    pub icc: String,
    /// None means no TIFF IFD was verified. Some([]) is verified RGB-only TIFF.
    #[serde(default)]
    pub tiff_extra_samples: Option<Vec<u16>>,
    #[serde(default)]
    pub artifact_validation: OutputArtifactValidation,
    #[serde(default)]
    pub artifact_validation_error: Option<String>,
    pub result_id: String,
    pub preview: PreviewRecord,
}

impl Default for OutputReport {
    fn default() -> Self {
        Self {
            written: false,
            format: String::new(),
            path: String::new(),
            width: 0,
            height: 0,
            bit_depth: 16,
            alpha_preserved: false,
            icc: "sRGB".to_string(),
            tiff_extra_samples: None,
            artifact_validation: OutputArtifactValidation::NotRun,
            artifact_validation_error: None,
            result_id: String::new(),
            preview: PreviewRecord::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutputArtifactValidation {
    #[default]
    NotRun,
    Pass,
    Fail,
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

/// Physical-source coverage audit. The support union never depends on output alpha.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct EffectiveSourceDomainRecord {
    pub method: String,
    pub source_count: usize,
    pub legacy_rectangular_union: u64,
    pub all_output_nontransparent: u64,
    pub covered_inside: u64,
    pub missing_inside: u64,
    pub covered_outside: u64,
}
