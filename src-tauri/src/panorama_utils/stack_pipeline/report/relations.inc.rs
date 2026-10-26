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
    #[serde(default)]
    pub canonical_source: usize,
    #[serde(default)]
    pub canonical_target: usize,
    #[serde(default)]
    pub source_to_target_transform: [f64; 9],
    #[serde(default)]
    pub correspondence_count: usize,
    #[serde(default)]
    pub correspondence_sample: Vec<StationRelationCorrespondenceRecord>,
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
pub(crate) struct StationRelationTileGeometryRecord {
    pub dimensions: [u32; 2],
    #[serde(default)]
    pub effective_bounds: Option<[u32; 4]>,
    #[serde(default)]
    pub tile_to_world: Option<[f64; 9]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StationRelationCandidateRecord {
    pub left: usize,
    pub right: usize,
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
