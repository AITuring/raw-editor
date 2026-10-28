//! Degradation_Manager: stable machine readable failure reason identifiers plus
//! The ledger and final outcome decision for every degraded or rejected path
//! of a single run.
//!
//! Recording remains observation-only so deep pipeline stages cannot publish a
//! partial result by themselves. At the final publication boundary,
//! [`DegradationManager`] makes the one authoritative decision and either
//! atomically renames the run-owned temporary output or removes only that
//! temporary file. Stack reports and diagnostic previews live outside that
//! transaction and are never cleanup targets.
//!
//! Requirements: 12.7 (identifier format and stability), 12.9 (all active paths
//! recorded, rejection wins over degradation), 12.10 (no rejected output
//! residue while reports and diagnostics remain).

// The identifier set is declared complete up front so the contract tests can
// guard its format and uniqueness, while the call sites that raise each reason
// arrive with the later rollout stages. Drop this allow once every stage
// consumes its identifiers.
#![allow(dead_code)]

use std::cell::Cell;
use std::io;
use std::path::Path;
#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, TryLockError};
use std::thread;
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

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

/// Severity of a degradation entry. `Rejected` means the run must not write a
/// final result file; `Degraded` means the run continues with reduced evidence;
/// `Informational` means the entry is housekeeping that left the output
/// untouched.
///
/// `Informational` exists because not every recorded path costs the result
/// anything. Evicting an old cache entry to stay inside the 64 GiB budget, for
/// example, only decides what stays on disk: the tile it makes room for is
/// synthesised from the same sources with the same code, so the pixels, the
/// thresholds and the fallback decisions of the run are all identical with and
/// without the eviction. Reporting such a run as `degraded` would tell the user
/// the output is worse than it is.
///
/// Design note: the degradation matrix in `design.md` still lists
/// `cache_evicted_for_capacity` as a degraded path and needs the same edit; the
/// table below is the authority in the meantime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Informational,
    Degraded,
    Rejected,
}

impl Severity {
    /// Stable machine readable form, identical to the serialized value.
    pub fn as_identifier(self) -> &'static str {
        match self {
            Self::Informational => "informational",
            Self::Degraded => "degraded",
            Self::Rejected => "rejected",
        }
    }
}

/// Aggregated result of a run as decided by the ledger (requirement 12.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunResult {
    Success,
    Degraded,
    Rejected,
}

/// Authoritative final decision made from one complete run ledger.
///
/// Cancellation is a rejected-severity path, but stays distinct so the report
/// and caller preserve the user's explicit terminal action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutcomeDecision {
    Success,
    Degraded,
    Rejected,
    Cancelled,
}

impl OutcomeDecision {
    pub(crate) fn permits_final_output(self) -> bool {
        matches!(self, Self::Success | Self::Degraded)
    }

    pub(crate) fn as_identifier(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Degraded => "degraded",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Filesystem boundary used by final-output publication. Tests inject an
/// in-memory implementation so they can prove exactly which paths are touched.
pub(crate) trait FinalOutputFileSystem {
    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()>;
    fn remove_file(&mut self, path: &Path) -> io::Result<()>;
    fn sync_directory(&mut self, path: &Path) -> io::Result<()>;
}

/// Production filesystem for the final output transaction.
pub(crate) struct StandardFinalOutputFileSystem;

impl FinalOutputFileSystem for StandardFinalOutputFileSystem {
    fn rename(&mut self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::rename(from, to)
    }

    fn remove_file(&mut self, path: &Path) -> io::Result<()> {
        std::fs::remove_file(path)
    }

    fn sync_directory(&mut self, path: &Path) -> io::Result<()> {
        #[cfg(unix)]
        {
            std::fs::File::open(path)?.sync_all()
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            Ok(())
        }
    }
}

/// Result of resolving a staged final output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputPublication {
    Published(OutcomeDecision),
    Rejected,
    Cancelled,
}

/// Makes the sole final-output decision for a run.
pub(crate) struct DegradationManager<'a> {
    ledger: &'a DegradationLedger,
}

impl<'a> DegradationManager<'a> {
    pub(crate) const fn new(ledger: &'a DegradationLedger) -> Self {
        Self { ledger }
    }

    /// Rejection outranks success and every degradation. Cancellation is tested
    /// first only to retain its existing, more-specific reporting precedence.
    pub(crate) fn decision(&self) -> OutcomeDecision {
        if self.ledger.count_of(RUN_CANCELLED_BY_USER) > 0 {
            return OutcomeDecision::Cancelled;
        }
        match self.ledger.result() {
            RunResult::Success => OutcomeDecision::Success,
            RunResult::Degraded => OutcomeDecision::Degraded,
            RunResult::Rejected => OutcomeDecision::Rejected,
        }
    }

    /// Publish `temporary_path` only after the accepted decision is known.
    ///
    /// Rejected and cancelled runs remove only their run-owned temporary path;
    /// they never remove or truncate `final_path`, a report, a preview, or any
    /// other pre-existing user file. Accepted runs use one atomic rename and
    /// then sync the containing directory.
    pub(crate) fn publish_staged_output<F: FinalOutputFileSystem>(
        &self,
        file_system: &mut F,
        temporary_path: &Path,
        final_path: &Path,
    ) -> Result<OutputPublication, String> {
        let decision = self.decision();
        if !decision.permits_final_output() {
            match file_system.remove_file(temporary_path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(format!(
                        "Failed to remove rejected image-stack temporary output '{}': {error}",
                        temporary_path.display()
                    ));
                }
            }
            return Ok(match decision {
                OutcomeDecision::Cancelled => OutputPublication::Cancelled,
                OutcomeDecision::Rejected => OutputPublication::Rejected,
                OutcomeDecision::Success | OutcomeDecision::Degraded => unreachable!(),
            });
        }

        file_system
            .rename(temporary_path, final_path)
            .map_err(|error| {
                let _ = file_system.remove_file(temporary_path);
                format!(
                    "Failed to atomically publish image-stack output '{}': {error}",
                    final_path.display()
                )
            })?;
        let parent = final_path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        file_system.sync_directory(parent).map_err(|error| {
            format!(
                "Failed to sync the image-stack output folder '{}': {error}",
                parent.display()
            )
        })?;
        Ok(OutputPublication::Published(decision))
    }
}

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

/// One active degraded or rejected path of a run.
#[derive(Debug, Clone, Serialize)]
pub struct DegradationEntry {
    /// ASCII lowercase, digits and underscore, at most 64 bytes.
    pub reason: &'static str,
    pub severity: Severity,
    /// Free form measured context for the Stack_Report. `Value::Null` when the
    /// call site has no numbers to report yet.
    pub detail: Value,
}

impl DegradationEntry {
    pub fn new(reason: &'static str, severity: Severity, detail: Value) -> Self {
        Self {
            reason,
            severity,
            detail,
        }
    }
}

/// Collects every degraded or rejected path of one run (requirement 12.9).
#[derive(Debug, Clone, Default, Serialize)]
pub struct DegradationLedger {
    pub entries: Vec<DegradationEntry>,
}

impl DegradationLedger {
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Records `reason` with the severity declared by the design matrix.
    /// Unknown identifiers are ignored so an observation point can never
    /// introduce a reason outside the frozen set.
    pub fn record(&mut self, reason: &'static str, detail: Value) {
        if let Some(severity) = severity_of(reason) {
            self.entries
                .push(DegradationEntry::new(reason, severity, detail));
        } else {
            debug_assert!(false, "unknown degradation reason identifier: {reason}");
        }
    }

    /// Records `reason` with an explicit severity. Only used where a call site
    /// must override the table (currently nothing does).
    pub fn record_with_severity(
        &mut self,
        reason: &'static str,
        severity: Severity,
        detail: Value,
    ) {
        self.entries
            .push(DegradationEntry::new(reason, severity, detail));
    }

    pub fn entries(&self) -> &[DegradationEntry] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn has_rejection(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.severity == Severity::Rejected)
    }

    /// `true` when at least one entry actually degraded the output. Entries of
    /// [`Severity::Informational`] are housekeeping and do not count.
    pub fn has_degradation(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.severity == Severity::Degraded)
    }

    /// Number of entries carrying `reason`.
    pub fn count_of(&self, reason: &str) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.reason == reason)
            .count()
    }

    /// Decision rule of requirement 12.9: any `Rejected` entry makes the run a
    /// rejected output, otherwise any `Degraded` entry makes it a degraded
    /// output. [`Severity::Informational`] entries are reported in full but
    /// never move the verdict, so an otherwise clean run that only did some
    /// cache housekeeping stays a success.
    pub fn result(&self) -> RunResult {
        if self.has_rejection() {
            RunResult::Rejected
        } else if self.has_degradation() {
            RunResult::Degraded
        } else {
            RunResult::Success
        }
    }

    /// Return the compact, human-readable Quality_Gate failure context that
    /// callers can attach to a rejected export error.  The machine-readable
    /// copy remains in Stack_Report; this summary keeps the caller message
    /// actionable by naming every failed criterion and its recorded ROI world
    /// coordinates without dumping the complete report.
    pub(crate) fn quality_gate_rejection_detail(&self) -> Option<String> {
        let mut sections = Vec::new();
        for entry in self.entries.iter().filter(|entry| {
            matches!(
                entry.reason,
                QUALITY_GATE_CRITERION_FAILED | QUALITY_GATE_INSUFFICIENT_EVIDENCE
            )
        }) {
            let Some(criteria) = entry.detail.get("criteria").and_then(Value::as_array) else {
                continue;
            };
            for criterion in criteria {
                let Some(name) = criterion.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let verdict = criterion
                    .get("verdict")
                    .and_then(Value::as_str)
                    .unwrap_or("failed");
                let mut locations = Vec::new();
                if let Some(failed) = criterion.get("failed").and_then(Value::as_array) {
                    for item in failed {
                        let Some(world) = item.get("world") else {
                            continue;
                        };
                        let (Some(x), Some(y)) = (
                            world.get("x").and_then(Value::as_f64),
                            world.get("y").and_then(Value::as_f64),
                        ) else {
                            continue;
                        };
                        locations.push(format!("({x:.2},{y:.2})"));
                    }
                }
                if locations.is_empty()
                    && let Some(unmeasurable) =
                        entry.detail.get("unmeasurable").and_then(Value::as_array)
                {
                    for item in unmeasurable
                        .iter()
                        .filter(|item| item.get("criterion").and_then(Value::as_str) == Some(name))
                    {
                        let Some(world) = item.get("world") else {
                            continue;
                        };
                        let (Some(x), Some(y)) = (
                            world.get("x").and_then(Value::as_f64),
                            world.get("y").and_then(Value::as_f64),
                        ) else {
                            continue;
                        };
                        locations.push(format!("({x:.2},{y:.2})"));
                    }
                }
                if locations.is_empty() {
                    sections.push(format!(
                        "{name} [{verdict}; ROI world coordinates unavailable]"
                    ));
                } else {
                    sections.push(format!(
                        "{name} [{verdict}; ROI world coordinates: {}]",
                        locations.join(", ")
                    ));
                }
            }
        }
        (!sections.is_empty()).then(|| sections.join("; "))
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Ledger of the run currently in flight.
///
/// The existing fallback sites are free functions deep inside the stitching
/// code and have no ledger parameter. A run scoped global keeps the new
/// observation additive: no signature changes, no behaviour changes, and a
/// record is a push onto a `Vec` behind an uncontended lock.
static RUN_LEDGER: Mutex<DegradationLedger> = Mutex::new(DegradationLedger::new());

/// Serialises operations that use the process-wide run observation sinks.  The
/// sinks predate concurrent runs and are intentionally kept signature-free;
/// one guard now makes reset -> render -> snapshot an atomic run transaction.
static RUN_SCOPE: Mutex<()> = Mutex::new(());
#[cfg(test)]
static RUN_SCOPE_ACTIVE: AtomicBool = AtomicBool::new(false);
/// The image-stack command increments this generation for every new request.
/// A run that already owns `RUN_SCOPE` observes a newer generation at its
/// cooperative checkpoints and exits instead of holding the process-wide sinks
/// while a replacement request waits for a completed render.
static ACTIVE_RUN_GENERATION: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static OWNED_RUN_GENERATION: Cell<usize> = const { Cell::new(0) };
}

pub(crate) struct RunScope {
    _guard: MutexGuard<'static, ()>,
}

#[cfg(test)]
impl Drop for RunScope {
    fn drop(&mut self) {
        RUN_SCOPE_ACTIVE.store(false, Ordering::Release);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RunScopeCancelled;

/// Acquire the process-wide observation scope while remaining responsive to a
/// newer UI generation. A queued replacement run never takes ownership of a
/// scope after its generation has already been superseded.
pub(crate) fn begin_run_scope() -> RunScope {
    begin_run_scope_for_generation(0).expect("a direct run must acquire its scope")
}

pub(crate) fn begin_run_scope_for_generation(
    generation: usize,
) -> Result<RunScope, RunScopeCancelled> {
    loop {
        if generation_changed_for(generation, ACTIVE_RUN_GENERATION.load(Ordering::Acquire)) {
            return Err(RunScopeCancelled);
        }
        match RUN_SCOPE.try_lock() {
            Ok(guard) => {
                if generation_changed_for(generation, ACTIVE_RUN_GENERATION.load(Ordering::Acquire))
                {
                    drop(guard);
                    return Err(RunScopeCancelled);
                }
                OWNED_RUN_GENERATION.with(|owned| owned.set(generation));
                #[cfg(test)]
                RUN_SCOPE_ACTIVE.store(true, Ordering::Release);
                return Ok(RunScope { _guard: guard });
            }
            Err(TryLockError::Poisoned(poisoned)) => {
                let guard = poisoned.into_inner();
                if generation_changed_for(generation, ACTIVE_RUN_GENERATION.load(Ordering::Acquire))
                {
                    drop(guard);
                    return Err(RunScopeCancelled);
                }
                OWNED_RUN_GENERATION.with(|owned| owned.set(generation));
                #[cfg(test)]
                RUN_SCOPE_ACTIVE.store(true, Ordering::Release);
                return Ok(RunScope { _guard: guard });
            }
            Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(5)),
        }
    }
}

pub(crate) fn active_run_generation() -> usize {
    ACTIVE_RUN_GENERATION.load(Ordering::Acquire)
}

/// Publish the generation of the newest UI request.  The next run still waits
/// for the scope to become available, but the previous owner sees the change at
/// its next checkpoint and releases the scope promptly.
pub(crate) fn set_active_run_generation(generation: usize) {
    ACTIVE_RUN_GENERATION.store(generation, Ordering::Release);
}

/// Whether the run holding the process-wide scope has been superseded by a
/// newer image-stack request.  A generation of zero is the direct/library test
/// path and deliberately has no external cancellation source.
pub(crate) fn run_generation_changed() -> bool {
    OWNED_RUN_GENERATION.with(|owned| {
        let generation = owned.get();
        run_generation_changed_for(generation)
    })
}

pub(crate) fn run_generation_changed_for(owned_generation: usize) -> bool {
    generation_changed_for(
        owned_generation,
        ACTIVE_RUN_GENERATION.load(Ordering::Acquire),
    )
}

fn generation_changed_for(owned_generation: usize, active_generation: usize) -> bool {
    owned_generation != 0 && active_generation != owned_generation
}

#[cfg(test)]
#[track_caller]
fn assert_run_scope() {
    assert!(
        RUN_SCOPE_ACTIVE.load(Ordering::Acquire),
        "run ledger write requires degradation::begin_run_scope()"
    );
}

fn with_run_ledger<T>(body: impl FnOnce(&mut DegradationLedger) -> T) -> T {
    let mut guard = RUN_LEDGER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    body(&mut guard)
}

/// Starts a fresh observation window. Called once per stitching run before any
/// diagnostic point can fire.
#[cfg_attr(test, track_caller)]
pub fn reset_run_ledger() {
    #[cfg(test)]
    assert_run_scope();
    with_run_ledger(DegradationLedger::clear);
}

/// Records an active degraded or rejected path of the current run. Parallel to
/// the existing `println!` diagnostics; it never affects control flow.
#[cfg_attr(test, track_caller)]
pub fn record_run_degradation(reason: &'static str, detail: Value) {
    #[cfg(test)]
    assert_run_scope();
    with_run_ledger(|ledger| ledger.record(reason, detail));
}

/// Copy of the ledger for report serialisation. The window stays open so late
/// terminal paths can still record.
pub fn run_ledger_snapshot() -> DegradationLedger {
    with_run_ledger(|ledger| ledger.clone())
}

#[cfg(test)]
mod tests;
