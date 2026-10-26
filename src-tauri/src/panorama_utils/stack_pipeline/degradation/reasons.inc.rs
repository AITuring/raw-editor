// Degradation_Manager fragment; included at parent module scope.

/// The full set of failure reason identifiers together with their severity.
/// The order matches the tables in the design document.
pub const FAILURE_REASONS: &[(&str, Severity)] = &[
    // Input and grouping
    (INPUT_SOURCE_COUNT_OUT_OF_RANGE, Severity::Rejected),
    (INPUT_PIPELINE_VERSION_MISMATCH, Severity::Rejected),
    (SOURCE_DECODE_FAILED, Severity::Degraded),
    (GROUPING_NO_OVERLAP_EVIDENCE, Severity::Degraded),
    (GROUPING_ACCUMULATED_MOTION, Severity::Degraded),
    (GROUPING_MEMBER_LIMIT_SPLIT, Severity::Degraded),
    // Intra station registration and fusion
    (INTRA_STATION_LOCAL_FALLBACK, Severity::Degraded),
    (
        INTRA_STATION_LOCAL_BASELINE_INDETERMINATE,
        Severity::Degraded,
    ),
    (INTRA_STATION_REGISTRATION_FAILED, Severity::Degraded),
    (INTRA_STATION_REJECTED_FROM_GROUP, Severity::Degraded),
    (FUSION_GRAPH_CUT_TIMEOUT, Severity::Degraded),
    (FUSION_SINGLE_FRAME_DEGRADED, Severity::Degraded),
    (FUSION_LOW_SHARPNESS_REGION, Severity::Degraded),
    // Virtual_Tile cache
    (CACHE_ENTRY_FIELD_MISSING, Severity::Degraded),
    (CACHE_ENTRY_DIMENSION_MISMATCH, Severity::Degraded),
    (CACHE_ENTRY_SHA_MISMATCH, Severity::Degraded),
    (CACHE_WRITE_UNAVAILABLE, Severity::Degraded),
    // Housekeeping only: making room in the cache changes which entries stay on
    // disk, never a pixel, a threshold or a fallback decision of this run. See
    // the `Severity::Informational` note above; `design.md` still shows this row
    // as degraded and needs the matching edit.
    (CACHE_EVICTED_FOR_CAPACITY, Severity::Informational),
    // Topology and station pose
    (TOPOLOGY_INDEX_AMBIGUOUS, Severity::Degraded),
    (TOPOLOGY_CANDIDATES_TRUNCATED, Severity::Degraded),
    (STATION_RELATION_LOW_INLIERS, Severity::Degraded),
    (STATION_RELATION_HIGH_REPROJECTION_ERROR, Severity::Degraded),
    (STATION_RELATION_SCALE_OUT_OF_RANGE, Severity::Degraded),
    (
        STATION_RELATION_NON_INVERTIBLE_CANONICAL_TRANSFORM,
        Severity::Degraded,
    ),
    (STATION_RELATION_LOW_SPATIAL_SUPPORT, Severity::Degraded),
    (STATION_RELATION_PHOTOMETRIC_MISMATCH, Severity::Degraded),
    (STATION_RELATION_EDGE_STRENGTH_MISMATCH, Severity::Degraded),
    (
        STATION_RELATION_EDGE_ORIENTATION_MISMATCH,
        Severity::Degraded,
    ),
    (STATION_RELATION_NON_CONVEX_QUAD, Severity::Degraded),
    (
        STATION_RELATION_SINGLE_LAYER_EVIDENCE_DISCARDED,
        Severity::Degraded,
    ),
    (STATION_POSE_LOCAL_SCALE_EXCEEDED, Severity::Degraded),
    (GEOMETRY_DISCONNECTED, Severity::Rejected),
    // Closure and residual warp
    (CLOSURE_NO_CONSTRAINTS, Severity::Degraded),
    (CLOSURE_UNRELIABLE_RESIDUAL, Severity::Degraded),
    (CLOSURE_UNRELIABLE_ITERATIONS, Severity::Degraded),
    (CLOSURE_UNRELIABLE_PAIR_P95, Severity::Degraded),
    (CLOSURE_CORRECTION_CLAMPED, Severity::Degraded),
    (RESIDUAL_WARP_INSUFFICIENT_EVIDENCE, Severity::Degraded),
    (RESIDUAL_WARP_CELL_REVERTED, Severity::Degraded),
    // Tone and composition
    (TONE_INSUFFICIENT_SAMPLES, Severity::Degraded),
    (TONE_GAIN_CLAMPED, Severity::Degraded),
    (TONE_BOUNDARY_DELTA_E_EXCEEDED, Severity::Degraded),
    (COMPOSITION_NARROW_OVERLAP, Severity::Degraded),
    (CANVAS_LONG_SIDE_EXCEEDED, Severity::Rejected),
    (OUTPUT_BIT_DEPTH_DOWNGRADED, Severity::Degraded),
    (OUTPUT_ALPHA_UNSUPPORTED, Severity::Degraded),
    // Quality_Gate and resources
    (QUALITY_GATE_CRITERION_FAILED, Severity::Rejected),
    (QUALITY_GATE_INSUFFICIENT_EVIDENCE, Severity::Rejected),
    (MEMORY_THRESHOLD_EXCEEDED, Severity::Rejected),
    (RUN_CANCELLED_BY_USER, Severity::Rejected),
    (DIAGNOSTICS_WRITE_FAILED, Severity::Degraded),
    (DIAGNOSTICS_ROI_INVALID, Severity::Degraded),
];

/// Reasons a Quality_Gate measurement item can be unmeasurable (requirement 11.13).
pub const UNMEASURABLE_REASONS: &[&str] = &[
    ROI_NOT_SLANTED_EDGE,
    ROI_NOT_FLAT,
    OWNER_SOURCE_UNDECODABLE,
    PAIRING_RESIDUAL_ALIGNMENT_EXCEEDED,
    BOUNDARY_NO_PAIRABLE_EDGE,
    SLANTED_EDGE_LINE_FIT_RESIDUAL_EXCEEDED,
    LOCAL_SCALE_UNMEASURABLE,
    TEXTURED_PIXEL_PLANE_UNAVAILABLE,
    OWNER_REVERSE_LOOKUP_UNRESOLVED,
    SLANTED_EDGE_TOO_SHORT,
    SLANTED_EDGE_CONTRAST_INSUFFICIENT,
    SLANTED_EDGE_ANGLE_OUT_OF_RANGE,
    BOUNDARY_LOW_CONTRAST,
    BOUNDARY_STROKE_NOT_UNIQUE,
    BOUNDARY_ORIENTATION_MISMATCH,
    BOUNDARY_PROVENANCE_UNAVAILABLE,
];

/// Maximum identifier length allowed by requirement 12.7.
pub const MAX_REASON_IDENTIFIER_LENGTH: usize = 64;

/// `true` when `reason` matches `^[a-z0-9_]{1,64}$` (requirement 12.7).
pub fn is_valid_reason_identifier(reason: &str) -> bool {
    !reason.is_empty()
        && reason.len() <= MAX_REASON_IDENTIFIER_LENGTH
        && reason
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

/// Severity declared for `reason` by the design degradation matrix, or `None`
/// when the identifier is not part of the failure reason set.
pub fn severity_of(reason: &str) -> Option<Severity> {
    FAILURE_REASONS
        .iter()
        .find(|(id, _)| *id == reason)
        .map(|(_, severity)| *severity)
}

