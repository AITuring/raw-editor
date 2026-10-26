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
