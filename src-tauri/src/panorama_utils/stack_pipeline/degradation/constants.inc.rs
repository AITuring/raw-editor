// Degradation_Manager fragment; included at parent module scope.

// ---------------------------------------------------------------------------
// Input and grouping stage
// ---------------------------------------------------------------------------

/// Source count < 2 or > `IMAGE_STACK_MAX_SOURCES`.
pub const INPUT_SOURCE_COUNT_OUT_OF_RANGE: &str = "input_source_count_out_of_range";
/// Front end and back end pipeline versions disagree.
pub const INPUT_PIPELINE_VERSION_MISMATCH: &str = "input_pipeline_version_mismatch";
/// A Source_RAW could not be decoded and was excluded.
pub const SOURCE_DECODE_FAILED: &str = "source_decode_failed";
/// A Source_RAW satisfies the overlap evidence of no other source.
pub const GROUPING_NO_OVERLAP_EVIDENCE: &str = "grouping_no_overlap_evidence";
/// Accumulated centre motion above 0.02 of the long side forced a split.
pub const GROUPING_ACCUMULATED_MOTION: &str = "grouping_accumulated_motion";
/// A candidate station exceeded 48 members and was split.
pub const GROUPING_MEMBER_LIMIT_SPLIT: &str = "grouping_member_limit_split";

// ---------------------------------------------------------------------------
// Intra station registration and fusion stage
// ---------------------------------------------------------------------------

/// Local refinement was not better than the global model; the frame reverted.
pub const INTRA_STATION_LOCAL_FALLBACK: &str = "intra_station_local_fallback";
/// 需求 2.7 had no comparable baseline: one of the two medians was formed from
/// too few accepted control points to exist.  The measured field is kept, since
/// reverting on an absent comparison is not what the requirement asks for.
pub const INTRA_STATION_LOCAL_BASELINE_INDETERMINATE: &str =
    "intra_station_local_baseline_indeterminate";
/// Inlier spatial coverage of a frame below 20%.
pub const INTRA_STATION_REGISTRATION_FAILED: &str = "intra_station_registration_failed";
/// Spatial support < 20% or median error > 0.01 long side; not joined to a station.
pub const INTRA_STATION_REJECTED_FROM_GROUP: &str = "intra_station_rejected_from_group";
/// Graph cut did not converge within 120 s; per cell minimum cost used.
pub const FUSION_GRAPH_CUT_TIMEOUT: &str = "fusion_graph_cut_timeout";
/// Every non anchor frame failed; the station degraded to a single frame.
pub const FUSION_SINGLE_FRAME_DEGRADED: &str = "fusion_single_frame_degraded";
/// A region whose every candidate is below the station P10 sharpness.
pub const FUSION_LOW_SHARPNESS_REGION: &str = "fusion_low_sharpness_region";

// ---------------------------------------------------------------------------
// Virtual_Tile cache stage
// ---------------------------------------------------------------------------

/// A cache entry is missing a required field.
pub const CACHE_ENTRY_FIELD_MISSING: &str = "cache_entry_field_missing";
/// Recorded dimensions disagree with the stored pixel dimensions.
pub const CACHE_ENTRY_DIMENSION_MISMATCH: &str = "cache_entry_dimension_mismatch";
/// The recorded SHA-256 set could not be re-verified.
pub const CACHE_ENTRY_SHA_MISMATCH: &str = "cache_entry_sha_mismatch";
/// Cache write failed because of missing space or an unwritable directory.
pub const CACHE_WRITE_UNAVAILABLE: &str = "cache_write_unavailable";
/// Cache exceeded 64 GiB and entries were evicted.
pub const CACHE_EVICTED_FOR_CAPACITY: &str = "cache_evicted_for_capacity";

// ---------------------------------------------------------------------------
// Topology and station pose stage
// ---------------------------------------------------------------------------

/// Row/column index could not be determined uniquely.
pub const TOPOLOGY_INDEX_AMBIGUOUS: &str = "topology_index_ambiguous";
/// Candidate adjacencies exceeded the search budget and were truncated.
pub const TOPOLOGY_CANDIDATES_TRUNCATED: &str = "topology_candidates_truncated";
/// Fewer than 24 inliers.
pub const STATION_RELATION_LOW_INLIERS: &str = "station_relation_low_inliers";
/// Median inlier reprojection error above 3.0 world pixels.
pub const STATION_RELATION_HIGH_REPROJECTION_ERROR: &str =
    "station_relation_high_reprojection_error";
/// Scale ratio outside `[0.95, 1.05]`.
pub const STATION_RELATION_SCALE_OUT_OF_RANGE: &str = "station_relation_scale_out_of_range";
/// The selected canonical homography is not invertible.
pub const STATION_RELATION_NON_INVERTIBLE_CANONICAL_TRANSFORM: &str =
    "station_relation_non_invertible_canonical_transform";
/// Inlier spatial support below 20%.
pub const STATION_RELATION_LOW_SPATIAL_SUPPORT: &str = "station_relation_low_spatial_support";
/// Low frequency luminance mean relative difference above 20%.
pub const STATION_RELATION_PHOTOMETRIC_MISMATCH: &str = "station_relation_photometric_mismatch";
/// Edge strength ratio outside `[0.7, 1.4]`.
pub const STATION_RELATION_EDGE_STRENGTH_MISMATCH: &str = "station_relation_edge_strength_mismatch";
/// Median edge orientation difference above 10 degrees.
pub const STATION_RELATION_EDGE_ORIENTATION_MISMATCH: &str =
    "station_relation_edge_orientation_mismatch";
/// The fitted homography projects the tile corners to a non convex quad.
pub const STATION_RELATION_NON_CONVEX_QUAD: &str = "station_relation_non_convex_quad";
/// Single source evidence that no Consensus_Feature confirmed was discarded.
pub const STATION_RELATION_SINGLE_LAYER_EVIDENCE_DISCARDED: &str =
    "station_relation_single_layer_evidence_discarded";
/// Local_Scale ratio inside one tile above 1.10.
pub const STATION_POSE_LOCAL_SCALE_EXCEEDED: &str = "station_pose_local_scale_exceeded";
/// The station graph has more than one connected component.
pub const GEOMETRY_DISCONNECTED: &str = "geometry_disconnected";

// ---------------------------------------------------------------------------
// Closure and residual warp stage
// ---------------------------------------------------------------------------

/// Accepted relations <= station count - 1, so no closure constraint exists.
pub const CLOSURE_NO_CONSTRAINTS: &str = "closure_no_constraints";
/// Median closure residual above 2.0 world pixels.
pub const CLOSURE_UNRELIABLE_RESIDUAL: &str = "closure_unreliable_residual";
/// Joint optimisation did not converge within 100 iterations.
pub const CLOSURE_UNRELIABLE_ITERATIONS: &str = "closure_unreliable_iterations";
/// A directly connected pair has overlap P95 above 3.0 world pixels.
pub const CLOSURE_UNRELIABLE_PAIR_P95: &str = "closure_unreliable_pair_p95";
/// A corner correction hit the clamp limit.
pub const CLOSURE_CORRECTION_CLAMPED: &str = "closure_correction_clamped";
/// Fewer than 16 round trip verified matches in a region needing local warp.
pub const RESIDUAL_WARP_INSUFFICIENT_EVIDENCE: &str = "residual_warp_insufficient_evidence";
/// A grid cell reverted to the global homography.
pub const RESIDUAL_WARP_CELL_REVERTED: &str = "residual_warp_cell_reverted";

// ---------------------------------------------------------------------------
// Tone and composition stage
// ---------------------------------------------------------------------------

/// Fewer than 1024 samples survived the MAD consistency check.
pub const TONE_INSUFFICIENT_SAMPLES: &str = "tone_insufficient_samples";
/// A solved gain or offset was clamped to its bound.
pub const TONE_GAIN_CLAMPED: &str = "tone_gain_clamped";
/// Owner region boundary band Delta_E00 above 1.5.
pub const TONE_BOUNDARY_DELTA_E_EXCEEDED: &str = "tone_boundary_delta_e_exceeded";
/// Effective overlap width below 32 world pixels.
pub const COMPOSITION_NARROW_OVERLAP: &str = "composition_narrow_overlap";
/// Union boundary long side above 262,144 world pixels.
pub const CANVAS_LONG_SIDE_EXCEEDED: &str = "canvas_long_side_exceeded";
/// The target format cannot hold 16 bits per channel.
pub const OUTPUT_BIT_DEPTH_DOWNGRADED: &str = "output_bit_depth_downgraded";
/// The target format cannot hold alpha.
pub const OUTPUT_ALPHA_UNSUPPORTED: &str = "output_alpha_unsupported";

// ---------------------------------------------------------------------------
// Quality_Gate and resource stage
// ---------------------------------------------------------------------------

/// At least one measurable measurement item failed its criterion.
pub const QUALITY_GATE_CRITERION_FAILED: &str = "quality_gate_criterion_failed";
/// Measurable items < 8 or unmeasurable share > 20%.
pub const QUALITY_GATE_INSUFFICIENT_EVIDENCE: &str = "quality_gate_insufficient_evidence";
/// Peak resident memory above the effective threshold.
pub const MEMORY_THRESHOLD_EXCEEDED: &str = "memory_threshold_exceeded";
/// The user cancelled the run.
pub const RUN_CANCELLED_BY_USER: &str = "run_cancelled_by_user";
/// The diagnostics directory is unwritable or a diagnostics write failed.
pub const DIAGNOSTICS_WRITE_FAILED: &str = "diagnostics_write_failed";
/// ROI long side > 4096, outside the crop region, or disjoint from coverage.
pub const DIAGNOSTICS_ROI_INVALID: &str = "diagnostics_roi_invalid";

// ---------------------------------------------------------------------------
// Unmeasurable reason identifiers (requirement 11.13)
// ---------------------------------------------------------------------------

/// The ROI does not contain a qualifying slanted edge.
pub const ROI_NOT_SLANTED_EDGE: &str = "roi_not_slanted_edge";
/// The ROI is not flat enough for a noise measurement.
pub const ROI_NOT_FLAT: &str = "roi_not_flat";
/// The owner Source_RAW could not be decoded for the reference measurement.
pub const OWNER_SOURCE_UNDECODABLE: &str = "owner_source_undecodable";
/// Residual alignment error of the ROI pairing above 0.5 native pixels.
pub const PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED: &str = "pairing_residual_alignment_exceeded";
/// No pairable edge on both sides of an owner region boundary.
pub const BOUNDARY_NO_PAIRABLE_EDGE: &str = "boundary_no_pairable_edge";
/// The slanted edge line fit residual is too large to measure MTF50.
pub const SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED: &str = "slanted_edge_line_fit_residual_exceeded";
/// Local scale could not be estimated, so scale-normalised criteria are not measurable.
pub const LOCAL_SCALE_UNMEASURABLE: &str = "local_scale_unmeasurable";
/// The final ownership plane did not carry Textured_Pixel evidence.
pub const TEXTURED_PIXEL_PLANE_UNAVAILABLE: &str = "textured_pixel_plane_unavailable";
/// Output ownership could not be traced back to a Source_RAW pixel.
pub const OWNER_REVERSE_LOOKUP_UNRESOLVED: &str = "owner_reverse_lookup_unresolved";
/// No sufficiently long edge support was found in the ROI.
pub const SLANTED_EDGE_TOO_SHORT: &str = "slanted_edge_too_short";
/// Edge support exists but its two sides do not reach the contrast floor.
pub const SLANTED_EDGE_CONTRAST_INSUFFICIENT: &str = "slanted_edge_contrast_insufficient";
/// Candidate edges exist but their angle is outside the 3..15 degree band.
pub const SLANTED_EDGE_ANGLE_OUT_OF_RANGE: &str = "slanted_edge_angle_out_of_range";
/// Boundary sampling found no edge on one or both sides.
pub const BOUNDARY_LOW_CONTRAST: &str = "boundary_low_contrast";
/// Multiple similarly strong edges prevent a unique cross-seam stroke pairing.
pub const BOUNDARY_STROKE_NOT_UNIQUE: &str = "boundary_stroke_not_unique";
/// Boundary edges were found but their orientations disagree.
pub const BOUNDARY_ORIENTATION_MISMATCH: &str = "boundary_orientation_mismatch";
/// A covered owner transition could not be tied to accepted station/source provenance.
pub const BOUNDARY_PROVENANCE_UNAVAILABLE: &str = "boundary_provenance_unavailable";

