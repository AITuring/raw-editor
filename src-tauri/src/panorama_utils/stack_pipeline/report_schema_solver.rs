//! Solver and topology records used by Stack_Report.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::super::degradation;
use super::report_schema_core::*;

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct StationRelationCorrespondenceRecord {
    pub source: [f64; 2],
    pub target: [f64; 2],
    pub local_scale: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct StationRelationHypothesisRecord {
    pub model: String,
    pub selected: bool,
    pub inliers: usize,
    pub median_error_px: f64,
    pub scale_ratio: f64,
    pub spatial_support: f64,
    pub low_frequency_mean_relative_difference: f64,
    pub edge_strength_ratio: f64,
    pub median_edge_orientation_difference_degrees: f64,
    pub preserves_convex_orientation: bool,
    pub rejection_reasons: Vec<String>,
    /// Content-derived canonical direction used by every final gate.
    #[serde(default)]
    pub canonical_source: usize,
    #[serde(default)]
    pub canonical_target: usize,
    /// Row-major canonical source-to-target homography in native tile pixels.
    #[serde(default)]
    pub source_to_target_transform: [f64; 9],
    /// Number of measured correspondences behind this hypothesis.
    #[serde(default)]
    pub correspondence_count: usize,
    /// At most 64 evenly spaced measured correspondences and their analytic
    /// Local_Scale. This is diagnostic only and never participates in ranking.
    #[serde(default)]
    pub correspondence_sample: Vec<StationRelationCorrespondenceRecord>,
    /// Independent deterministic checkerboard refit and held-out evaluation.
    /// These diagnostics never participate in hypothesis selection.
    #[serde(default)]
    pub holdout_fit_count: usize,
    #[serde(default)]
    pub holdout_count: usize,
    #[serde(default)]
    pub holdout_median_error_px: Option<f64>,
    #[serde(default)]
    pub holdout_spatial_support: Option<f64>,
    #[serde(default)]
    pub holdout_transform: Option<[f64; 9]>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct StationRelationGlobalProposalRecord {
    pub rank: usize,
    pub score: f64,
    pub overlap_samples: usize,
    pub left_factor: u32,
    pub right_factor: u32,
    pub translation_x_px: f64,
    pub translation_y_px: f64,
    pub scale: f64,
    pub rotation_degrees: f64,
}

/// One proposal from the per-RAW luminance/gradient correlation oracle.
/// Scores and proposal counts are diagnostic only and never contribute to
/// independent support.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SourceRawCorrelationProposalRecord {
    pub rank: usize,
    pub score: f64,
    pub second_peak_margin: f64,
    pub overlap_samples: usize,
    pub left_factor: u32,
    pub right_factor: u32,
    pub translation_x_px: f64,
    pub translation_y_px: f64,
    pub scale: f64,
    pub rotation_degrees: f64,
    pub guided_match_count: usize,
    pub stable_ransac_inliers: usize,
    pub full_resolution_validated: bool,
    pub validation_stage: String,
    pub match_source: String,
    pub match_target: String,
    pub source_to_target_transform: [f64; 9],
}

/// One independently fitted Source_RAW layer pair, canonicalized into the two
/// stations' anchor coordinate systems. A layer pair contributes at most one
/// observation regardless of how many patches, proposals, or inliers it contains.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SourceRawLayerModelRecord {
    pub left_source: String,
    pub right_source: String,
    /// Descriptor counts after extraction for the station-left and station-right RAWs.
    #[serde(default)]
    pub left_descriptor_count: usize,
    #[serde(default)]
    pub right_descriptor_count: usize,
    /// True when the stable matcher's deterministic source direction is station-right to station-left.
    #[serde(default)]
    pub match_direction_reversed: bool,
    /// Source descriptors for which nearest and second-nearest descriptors were retrieved.
    #[serde(default)]
    pub descriptor_retrieval_count: usize,
    /// Retrieved descriptors surviving the stable matcher's ratio test, before mutual filtering.
    #[serde(default)]
    pub ratio_filter_count: usize,
    /// Correspondences surviving both ratio and reverse-best mutual filtering.
    #[serde(default)]
    pub mutual_filter_count: usize,
    #[serde(default)]
    pub ransac_model_fitted: bool,
    #[serde(default)]
    pub ransac_inliers: usize,
    #[serde(default)]
    pub ransac_anchor_transform: Option<[f64; 9]>,
    #[serde(default)]
    pub canonical_conversion_succeeded: bool,
    #[serde(default)]
    pub full_resolution_validated: bool,
    #[serde(default)]
    pub failure_stage: String,
    /// `stable_matcher` today; a future independently proven proposal seed must remain explicit.
    #[serde(default)]
    pub model_provenance: String,
    /// Diagnostic-only proposal provenance. Neither this flag nor its score contributes support.
    #[serde(default)]
    pub proposal_seeded: bool,
    #[serde(default)]
    pub proposal_score: Option<f64>,
    #[serde(default)]
    pub proposal_candidate_count: usize,
    #[serde(default)]
    pub proposal_match_count: usize,
    #[serde(default)]
    pub proposal_ransac_inliers: usize,
    /// Complete deterministic attrition for every correlation peak considered
    /// for this RAW pair.
    #[serde(default)]
    pub correlation_proposals: Vec<SourceRawCorrelationProposalRecord>,
    pub model_fitted: bool,
    pub inliers: usize,
    pub median_error_px: f64,
    pub scale_ratio: f64,
    pub spatial_support: f64,
    pub anchor_transform: [f64; 9],
    pub consensus_cluster: Option<usize>,
}

/// A transform cluster supported by different Source_RAW observations on both
/// sides. Scheme-B peaks and repeated patches are not observations here.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SourceRawConsensusClusterRecord {
    pub cluster: usize,
    pub observation_count: usize,
    pub independent_support: usize,
    pub supporting_layer_pairs: Vec<[String; 2]>,
    pub median_disagreement_px: f64,
    pub scale_ratio: f64,
    pub spatial_support_potential: f64,
    pub scale_compliant: bool,
    pub spatially_broad: bool,
    pub anchor_transform: [f64; 9],
}

/// One direct-localization point measured in a checkerboard hold-out cell.
/// `target` is never generated by applying the fitted model to `source`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SourceRawDirectPointRecord {
    pub cell: [usize; 2],
    pub source: [f64; 2],
    pub target: [f64; 2],
    pub reverse_source: [f64; 2],
    pub correlation: f64,
    pub peak_margin: f64,
    pub round_trip_error_px: f64,
}

/// One deterministic checkerboard fit/hold-out fold of F-Direct.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SourceRawDirectFoldRecord {
    pub fold: usize,
    pub seed_rank: usize,
    #[serde(default)]
    pub reverse_seed_rank: usize,
    #[serde(default)]
    pub seed_pair_coarse_closure_px: f64,
    pub seed_provenance: String,
    pub pyramid_factors: Vec<u32>,
    pub iterations: usize,
    pub fit_cell_count: usize,
    pub holdout_cell_count: usize,
    pub fit_loss: f64,
    pub holdout_loss: f64,
    pub gain: f64,
    pub bias: f64,
    pub model: String,
    pub raw_transform: [f64; 9],
    pub reverse_raw_transform: [f64; 9],
    pub bidirectional_closure_px: f64,
    pub points: Vec<SourceRawDirectPointRecord>,
}

/// One independently measured RAW×RAW F-Direct observation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SourceRawDirectObservationRecord {
    pub left_source: String,
    pub right_source: String,
    pub folds: Vec<SourceRawDirectFoldRecord>,
    pub model: String,
    pub holdout_points: usize,
    pub median_error_px: f64,
    pub scale_ratio: f64,
    pub spatial_support: f64,
    pub preserves_convex_orientation: bool,
    pub fold_disagreement_px: f64,
    pub bidirectional_closure_px: f64,
    pub anchor_transform: [f64; 9],
    pub rejection_reasons: Vec<String>,
}

/// A canonical F-Direct cluster whose support is a bipartite matching of RAW
/// IDs. Repeated folds, seeds, or either-side RAW IDs cannot inflate support.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SourceRawDirectConsensusRecord {
    pub cluster: usize,
    pub observation_count: usize,
    pub independent_support: usize,
    pub supporting_layer_pairs: Vec<[String; 2]>,
    pub canonical_disagreement_px: f64,
    pub anchor_transform: [f64; 9],
    pub station_inliers: usize,
    pub station_median_error_px: f64,
    pub station_scale_ratio: f64,
    pub station_spatial_support: f64,
    pub station_preserves_convex_orientation: bool,
    pub station_bidirectional_closure_px: f64,
    pub station_rejection_reasons: Vec<String>,
}

/// Read-only F-Direct diagnostics for one station pair.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SourceRawDirectPairDiagnosticRecord {
    pub left: usize,
    pub right: usize,
    pub observations: Vec<SourceRawDirectObservationRecord>,
    pub consensus_clusters: Vec<SourceRawDirectConsensusRecord>,
    pub elapsed_ms: u64,
    pub peak_working_set_bytes: u64,
}

/// Read-only experiment-C diagnostics for one station pair.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SourceRawPairDiagnosticRecord {
    pub left: usize,
    pub right: usize,
    pub layer_models: Vec<SourceRawLayerModelRecord>,
    pub consensus_clusters: Vec<SourceRawConsensusClusterRecord>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct StationRelationTileGeometryRecord {
    pub dimensions: [u32; 2],
    /// Native-pixel bounds of the finest relation Coverage_Mask.
    pub effective_bounds: Option<[u32; 4]>,
    /// Row-major native-tile to rendering-world transform. For the current
    /// renderer this records the pixel-aligned crop-origin translation.
    pub tile_to_world: Option<[f64; 9]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StationRelationCandidateRecord {
    pub left: usize,
    pub right: usize,
    /// Native/effective coordinate geometry used by scale and hull checks.
    #[serde(default)]
    pub left_tile: StationRelationTileGeometryRecord,
    #[serde(default)]
    pub right_tile: StationRelationTileGeometryRecord,
    /// Geometric overlap predicted by the rendering prior, in left-tile
    /// analysis pixels at `measurement_factor`, at the search-radius calculation.
    #[serde(default)]
    pub predicted_overlap_area_px: f64,
    /// Whether the diagnostic measurement level fitted a model from pixels.
    #[serde(default)]
    pub model_fitted: bool,
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
    /// Coverage-aware ultra-coarse search peaks. These are proposal diagnostics
    /// only; acceptance always belongs to the shared Station_Relation gates.
    #[serde(default)]
    pub global_proposals: Vec<StationRelationGlobalProposalRecord>,
    /// True when either station was re-rendered from repaired source poses.
    #[serde(default)]
    pub after_prior_repair: bool,
    /// Bounded internal model outcomes. Pair-level accepted/rejected accounting
    /// remains singular; these records explain why alternatives did not win.
    #[serde(default)]
    pub hypotheses: Vec<StationRelationHypothesisRecord>,
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
    /// Read-only Source_RAW layer-pair evidence collected by experiments C-E.
    /// These records do not enter the relation graph or rendering-prior repair.
    #[serde(default)]
    pub source_raw_pair_diagnostics: Vec<SourceRawPairDiagnosticRecord>,
    /// Read-only F-Direct evidence. It remains test-only until the F2 gate has
    /// passed and never enters `tile_relations` or prior repair here.
    #[serde(default)]
    pub source_raw_direct_diagnostics: Vec<SourceRawDirectPairDiagnosticRecord>,
    /// Rendering-prior repairs applied before the authoritative measurement.
    #[serde(default)]
    pub prior_repairs: Vec<StationPriorRepairRecord>,
    pub rejected_count: usize,
    /// Rejection counts keyed by stable failure identifier (requirement 6.10).
    pub rejected_by_reason: BTreeMap<String, u64>,
    /// Candidates for which no residual model could be fitted.
    #[serde(default)]
    pub unmeasured_count: usize,
    /// Rejected candidates grouped by the matcher stage that stopped them.
    #[serde(default)]
    pub rejected_by_failure_stage: BTreeMap<String, u64>,
    /// Final station pairs whose selected canonical transform was singular.
    #[serde(skip)]
    pub(crate) non_invertible_canonical_pairs: BTreeSet<(usize, usize)>,
    pub discarded_single_layer_evidence: Vec<DiscardedEvidenceRecord>,
    pub connectivity: ConnectivityReport,
}

impl StationRelationsReport {
    pub(crate) fn record_non_invertible_canonical_pair(&mut self, left: usize, right: usize) {
        let pair = (left.min(right), left.max(right));
        if !self.non_invertible_canonical_pairs.insert(pair) {
            return;
        }
        self.rejected_count += 1;
        *self
            .rejected_by_reason
            .entry(degradation::STATION_RELATION_NON_INVERTIBLE_CANONICAL_TRANSFORM.to_string())
            .or_default() += 1;
        degradation::record_run_degradation(
            degradation::STATION_RELATION_NON_INVERTIBLE_CANONICAL_TRANSFORM,
            serde_json::json!({ "left": pair.0, "right": pair.1 }),
        );
    }

    /// Summarize the final candidate attempt independently of the measured
    /// Requirement 6 rejection reasons. A fitted model can still fail its
    /// support or orientation gate, or a later solver acceptance criterion.
    pub(crate) fn summarize_candidate_measurements(&mut self) {
        self.unmeasured_count = 0;
        self.rejected_by_failure_stage.clear();
        let mut summarized_non_invertible = BTreeSet::new();
        for candidate in &mut self.candidates {
            let pair = (
                candidate.left.min(candidate.right),
                candidate.left.max(candidate.right),
            );
            let non_invertible = self.non_invertible_canonical_pairs.contains(&pair);
            if non_invertible {
                candidate.failure_stage = "canonical_transform_non_invertible".to_string();
                candidate.model_fitted = true;
                if !summarized_non_invertible.insert(pair) {
                    continue;
                }
            } else if !candidate.model_fitted {
                self.unmeasured_count += 1;
            }
            let accepted = self.accepted.iter().any(|relation| {
                (relation.left == candidate.left && relation.right == candidate.right)
                    || (relation.left == candidate.right && relation.right == candidate.left)
            });
            if !accepted {
                *self
                    .rejected_by_failure_stage
                    .entry(candidate.failure_stage.clone())
                    .or_default() += 1;
            }
        }
        for _ in self
            .non_invertible_canonical_pairs
            .difference(&summarized_non_invertible)
        {
            *self
                .rejected_by_failure_stage
                .entry("canonical_transform_non_invertible".to_string())
                .or_default() += 1;
        }
    }
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
