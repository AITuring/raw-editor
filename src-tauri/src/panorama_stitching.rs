use crate::app_settings::{AppSettings, load_settings_for_runtime};
use crate::app_state::AppState;
use crate::file_management::{parse_virtual_path, read_file_mapped};
use base64::{Engine as _, engine::general_purpose};
use image::ImageFormat;
use image::{ColorType, DynamicImage, GenericImageView, GrayImage, Rgb32FImage, RgbImage};
use nalgebra::{Matrix3, Point2, Point3};
use rand::prelude::*;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Runtime};

use crate::formats::{is_psd_file, is_raw_file};
use crate::image_processing::apply_cpu_default_raw_processing;
use crate::panorama_utils::registration;
use crate::panorama_utils::stack_pipeline;
use crate::panorama_utils::stack_pipeline::compositor::{
    StackCompositorChoice, select_and_record_run_path,
};
use crate::panorama_utils::stack_pipeline::degradation;
use crate::panorama_utils::stack_pipeline::determinism;
use crate::panorama_utils::stack_pipeline::determinism::{sorted_keys, sorted_pairs};
use crate::panorama_utils::stack_pipeline::diagnostics::{DiagnosticItem, DiagnosticsRecorder};
use crate::panorama_utils::stack_pipeline::focus_fuser;
use crate::panorama_utils::stack_pipeline::intra_station;
use crate::panorama_utils::stack_pipeline::joint_warp;
use crate::panorama_utils::stack_pipeline::quality_gate;
use crate::panorama_utils::stack_pipeline::quality_gate_runner;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use crate::panorama_utils::stack_pipeline::residual_warp;
use crate::panorama_utils::stack_pipeline::resources;
use crate::panorama_utils::stack_pipeline::tone;
use crate::panorama_utils::stack_pipeline::topology;
use crate::panorama_utils::stack_pipeline::virtual_tile;
use crate::panorama_utils::stitching::{Projection, project_point};
use crate::panorama_utils::{processing, stitching};

// Keep the historical facade paths stable while the shared contracts live in
// their own module. Lower-level modules can migrate to `shared` incrementally.
pub use crate::panorama_utils::shared::{
    Descriptor, Feature, ImageInfo, KeyPoint, Match, MatchInfo,
};
pub const BRIEF_DESCRIPTOR_SIZE: usize = crate::panorama_utils::shared::BRIEF_DESCRIPTOR_SIZE;
pub(crate) use crate::panorama_utils::shared::{FocusLayerWarp, FocusWarpBand};

#[path = "panorama_feature_pairs.rs"]
mod feature_pairs;

use feature_pairs::{all_image_pairs, natural_path_cmp, pairs_to_match, pairs_to_match_for_images};

#[path = "panorama_focus_feature_prep.rs"]
mod focus_feature_prep;

use focus_feature_prep::{
    build_foreground_mask, detect_foreground_range, find_alignment_features,
    find_top_alignment_features, focus_alignment_keypoint_is_foreground,
};

#[path = "panorama_match_contract.rs"]
mod match_contract;

use match_contract::{
    CanonicalRelationViewError, canonical_match_direction, canonical_relation_view,
    focus_adjacent_same_focal_pair, focus_capture_number_gap, focus_compatible_focal_length,
    focus_image_direction_key, homography_scale_ratio, match_has_spatial_support,
    station_pose_local_scale_range,
};

#[path = "panorama_match_candidates.rs"]
mod match_candidates;

use match_candidates::{MatchCandidates, build_match_candidates};

#[path = "panorama_match_result.rs"]
mod match_result;

use match_result::build_match_info;

#[path = "panorama_match_geometry.rs"]
mod match_geometry;

#[path = "panorama_match_focus_rescue.rs"]
mod match_focus_rescue;

#[path = "panorama_station_relation_gate.rs"]
mod station_relation_gate;

#[cfg(test)]
use station_relation_gate::virtual_tile_covered_overlap_area;
use station_relation_gate::{
    FocusGroupEdgeCandidate, MeasuredStationRelationCandidate, StationOverlapMeasure,
    StationRelationEvidenceKind, StationRelationMeasurements, VirtualTileSolverCoverages,
    clip_polygon_to_rectangle, convex_hull_vertices, measure_station_relation_candidate,
    record_station_relation_no_fit, station_relation_is_accepted_with_report,
    station_relation_overlap_area, station_relation_rejection_reasons,
    station_relation_spatial_support, station_relation_spatial_support_with,
};

#[path = "panorama_station_closure.rs"]
mod station_closure;

#[cfg(test)]
use station_closure::closure_relative_residual_decrease_converged;
use station_closure::{
    ClosureAffineObservation, ClosureProjectiveObservation, affine_delta_apply,
    affine_delta_matrix, closure_residual_weight, projective_delta_matrix,
    projective_delta_projection, solve_robust_closure_affine, solve_robust_closure_projective,
    solve_robust_closure_translations,
};

#[path = "panorama_station_closure_gate.rs"]
mod station_closure_gate;

use station_closure_gate::{
    ClosureAcceptance, clamp_station_affine_corrections, clamp_station_projective_corrections,
    clamp_station_translation_corrections, closure_acceptance, closure_direct_pair_p95s,
    closure_offending_pair_record, closure_relation_direct_sample_points,
    closure_relation_sample_points, closure_residual_summary, projected_polygon_short_side,
    reject_global_closure_relation, station_relation_overlap_polygon,
};
#[cfg(test)]
use station_closure_gate::{ClosureDirectPairP95, maximum_station_corner_displacement};

#[path = "panorama_station_topology.rs"]
mod station_topology;

use station_topology::{
    focus_station_topology, geometry_disconnected_user_error, maximum_inlier_spanning_tree_indices,
    station_graph_components_from_members, station_graph_connectivity_rejection,
    station_tree_edge_key, station_tree_position_key,
};

#[path = "panorama_station_closure_refine.rs"]
mod station_closure_refine;

use station_closure_refine::refine_focus_group_projective_tree_poses_with_report_protected;
#[cfg(test)]
use station_closure_refine::{
    refine_focus_group_projective_tree_poses, refine_focus_group_projective_tree_poses_with_report,
};

#[path = "panorama_station_pose_seed.rs"]
mod station_pose_seed;

use station_pose_seed::{
    correct_focus_group_roll_from_long_edges, select_station_pose_candidate,
    solve_focus_group_translation_poses, station_pose_candidate_has_bounded_local_scale,
};

#[path = "panorama_station_solver.rs"]
mod station_solver;

use station_solver::solve_station_poses_for_evidence_with_report;

#[path = "panorama_station_solve_entry.rs"]
mod station_solve_entry;

#[cfg(test)]
use station_solve_entry::solve_virtual_tile_station_hypotheses_with_report;
use station_solve_entry::{
    solve_focus_capture_group_poses_with_report, solve_virtual_tile_station_poses_with_report,
};

#[path = "panorama_station_residual.rs"]
mod station_residual;

use station_residual::record_residual_warp_for_station_relations;
#[cfg(test)]
use station_residual::residual_model_from_station_relations;

#[path = "panorama_station_grouping.rs"]
mod station_grouping;

use station_grouping::{
    StationEvidence, StationGroupingOutcome, focus_local_bracket_components,
    focus_match_center_motion_ratio, focus_match_is_capture_station_link,
    focus_match_is_local_bracket, focus_station_grouping, focus_station_link_evidence,
};
#[cfg(test)]
use station_grouping::{
    StationSplit, focus_match_has_verified_scan_overlap, split_focus_local_component_by_motion,
};

#[path = "panorama_capture_groups.rs"]
mod capture_groups;

use capture_groups::{
    FocusCaptureGroup, canonicalize_station_geometry_anchors, capture_groups_form_singleton_scan,
    focus_capture_groups, focus_capture_groups_with_grouping, focus_station_binding_evidence,
    lock_focus_local_bracket_poses, order_focus_render_sources_by_capture_group,
};

/// Owns the run-scoped RSS sampler and publishes its final sample even when a
/// focus-stack stage returns early.  The sampler only requests cancellation;
/// the caller checks it at stage boundaries and returns before writing output.
struct FocusRssGuard {
    sampler: Option<resources::RssSampler>,
    threshold_bytes: u64,
    recorder: Option<Arc<stack_report::StackReportRecorder>>,
}

impl FocusRssGuard {
    fn cancelled(&self) -> bool {
        self.sampler
            .as_ref()
            .is_some_and(resources::RssSampler::cancelled)
    }

    fn finish(&mut self) -> Option<resources::RssSample> {
        let sample = self
            .sampler
            .take()
            .map(|sampler| sampler.stop(self.threshold_bytes))?;
        if let Some(recorder) = self.recorder.as_ref() {
            recorder.update(|report| {
                report.resources.peak_rss_bytes = sample.peak_rss_bytes;
                report.resources.rss_sample_count = sample.sample_count;
                report.resources.memory_threshold_exceeded = sample.threshold_exceeded;
            });
        }
        if sample.threshold_exceeded {
            degradation::record_run_degradation(
                degradation::MEMORY_THRESHOLD_EXCEEDED,
                serde_json::json!({
                    "threshold_bytes": self.threshold_bytes,
                    "peak_rss_bytes": sample.peak_rss_bytes,
                    "sample_count": sample.sample_count,
                }),
            );
        }
        Some(sample)
    }
}

impl Drop for FocusRssGuard {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

fn ensure_focus_memory_available(sampler: Option<&FocusRssGuard>) -> Result<(), String> {
    if degradation::run_generation_changed() {
        degradation::record_run_degradation(
            degradation::RUN_CANCELLED_BY_USER,
            serde_json::json!({"reason": "newer_image_stack_generation"}),
        );
        return Err("image-stack run cancelled by a newer request".to_string());
    }
    if sampler.is_some_and(FocusRssGuard::cancelled) {
        return Err("memory threshold exceeded during focus-stack run".to_string());
    }
    Ok(())
}

const FULL_RES_RANSAC_INLIER_THRESHOLD: f64 = 12.0;
const FULL_RES_REFINEMENT_THRESHOLD: f64 = 2.5;
const MATCH_REFINE_PATCH_RADIUS: i32 = 14;
const MATCH_REFINE_SEARCH_RADIUS: i32 = 14;
// Full-resolution focus stacks need a wider local search than ordinary panorama
// matches. The alignment keypoints are extracted from a reduced image, so a
// small residual can still be several full-resolution pixels at a face or a
// painted contour. Refine the native images without changing the cheaper
// panorama path.
const FOCUS_MATCH_REFINE_PATCH_RADIUS: i32 = 20;
const FOCUS_MATCH_REFINE_SEARCH_RADIUS: i32 = 24;
const FOCUS_MODEL_INLIER_THRESHOLD: f64 = 6.0;
const FOCUS_MODEL_RANSAC_ITERATIONS: usize = 1_500;
const FOCUS_MODEL_MIN_INLIERS: usize = 8;
// A handheld angle change over a planar artwork can leave a real projective
// residual even when the optical centre barely moves.  Do not force that
// residual through an affine model: the resulting edge error is large enough
// to make every focus layer look soft.  Requiring broad support keeps a small
// repeated stroke from enabling a projective warp by itself.
const FOCUS_PROJECTIVE_MIN_SPATIAL_SUPPORT: f64 = 0.22;
const FOCUS_LOCAL_MODEL_MIN_INLIERS: usize = 6;
const FOCUS_SHIFTED_MOSAIC_MOTION_RATIO: f64 = 0.015;
const FOCUS_GLOBAL_MAX_POINTS_PER_EDGE: usize = 256;
// A real neighbouring capture can move by almost one frame width when the
// camera advances to the next tile. A descriptor match that moves the frame
// farther than this is almost certainly a repeated character/seal match.
const FOCUS_SEQUENCE_MAX_LINK_MOTION_RATIO: f64 = 1.35;
const FOCUS_GLOBAL_MAX_ITERATIONS: usize = 12;
// The global pose solve operates in coordinates normalised by the largest
// source dimension. A 0.0012 Huber transition is roughly 19px on the phone
// frames used by the focus-stack path; that is wide enough for a repeated
// canvas stroke to pull a whole layer before the local mosaic refinement gets
// a chance to correct it. Keep the robust solve focused on the identity
// correspondences and let the regional/native passes handle the remaining
// sub-pixel deformation.
const FOCUS_GLOBAL_HUBER_THRESHOLD: f64 = 0.00065;
const FOCUS_GLOBAL_PRIOR_WEIGHT: f64 = 0.01;
const FOCUS_GLOBAL_PROJECTIVE_PRIOR_WEIGHT: f64 = 0.5;
const FOCUS_GLOBAL_DAMPING: f64 = 1e-6;
const FOCUS_BRACKET_MAX_CENTER_MOTION_RATIO: f64 = 0.075;
// Membership is stricter than the broad pairwise bracket gate above.  A
// neighbouring scan station can still produce a convincing local match, but
// its accumulated centre displacement is much larger than focus breathing.
// Keep that edge available for the station pose graph while preventing it
// from putting two camera positions into one virtual tile.
// Focus brackets can move by a few percent of the frame when the camera is
// refocused or the rail settles.  Treat that as one station and let the local
// focus registration resolve it; splitting at 1.5% turned genuine brackets
// into separate virtual tiles on the long scan (35 groups instead of 26).
// Adjacent stations in a real scan are much farther apart, so this remains a
// conservative station boundary while retaining the full depth bracket.
const FOCUS_BRACKET_INTERNAL_MOTION_RATIO: f64 = 0.05;
const FOCUS_BRACKET_MIN_OVERLAP_SUPPORT: f64 = 0.60;
const FOCUS_BRACKET_MAX_CAPTURE_GAP: u64 = 6;
// ---------------------------------------------------------------------------
// Capture_Station membership evidence (需求 1.1)
//
// These four thresholds are the *whole* membership judgement. They are
// deliberately separate from the broad pairwise bracket gate above, which stays
// in place for the station pose graph: `FOCUS_BRACKET_MIN_OVERLAP_SUPPORT` is an
// overlap *area* ratio, while the station evidence measures the correlation of
// that overlap, how widely the inliers are spread inside it, and whether the two
// frames were shot at the same distance.
// ---------------------------------------------------------------------------
/// Verified overlap inlier correspondences a station link needs (需求 1.1).
const STATION_MIN_INLIER_MATCHES: usize = 30;
/// Local normalised pixel correlation over the overlap (需求 1.1).
///
/// Distinct from the `0.45`-class bars used to merely *connect* two tiles:
/// connecting and co-owning ownership cells are different decisions.
/// `focus_overlap_quality` already refuses to report a correlation measured from
/// fewer than 120 samples, which is the measurable precondition.
const STATION_MIN_OVERLAP_NCC: f64 = 0.60;
/// Inlier spatial support inside the overlap (需求 1.1).
const STATION_MIN_INLIER_SPATIAL_SUPPORT: f64 = 0.20;
/// Scale-ratio window of two sources of one Capture_Station (需求 1.1).
///
/// 需求 6.3 applies `[0.95, 1.05]` to the *inter*-station relation using the
/// same [`homography_scale_ratio`] measurement.
const STATION_SCALE_RATIO_MIN: f64 = 0.98;
const STATION_SCALE_RATIO_MAX: f64 = 1.02;
// ---------------------------------------------------------------------------
// Station_Relation acceptance contract (需求 6.3 / 6.4 / 6.8 / 6.9)
//
// These are world-coordinate native-pixel and dimensionless limits. They are
// deliberately independent of source dimensions: a 3px defect must not become
// acceptable merely because the camera has a longer sensor edge.
// ---------------------------------------------------------------------------
const STATION_RELATION_MIN_INLIERS: usize = 24;
const STATION_RELATION_MAX_MEDIAN_ERROR_PX: f64 = 3.0;
const STATION_RELATION_SCALE_RATIO_MIN: f64 = 0.95;
const STATION_RELATION_SCALE_RATIO_MAX: f64 = 1.05;
const STATION_RELATION_MIN_SPATIAL_SUPPORT: f64 = 0.20;
const STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE: f64 = 0.20;
const STATION_RELATION_EDGE_STRENGTH_RATIO_MIN: f64 = 0.70;
const STATION_RELATION_EDGE_STRENGTH_RATIO_MAX: f64 = 1.40;
const STATION_RELATION_MAX_MEDIAN_EDGE_ORIENTATION_DIFFERENCE_DEGREES: f64 = 10.0;
/// Maximum Local_Scale variation accepted inside one station tile (需求 6.6).
const STATION_POSE_MAX_LOCAL_SCALE_RATIO: f64 = 1.10;
/// Samples per axis, including every valid-area edge and the tile centre.
const STATION_POSE_LOCAL_SCALE_SAMPLES_PER_AXIS: usize = 9;
/// Accumulated frame-centre displacement, in long sides, that splits a
/// Capture_Station at that candidate (需求 1.4).
const STATION_MAX_ACCUMULATED_CENTER_MOTION_RATIO: f64 = 0.02;
/// Station-evidence score difference below which the absolute path is allowed
/// to break the tie between two candidates (需求 1.2).
const STATION_EVIDENCE_SCORE_TIE_BREAK: f64 = 0.001;
// A high-resolution artwork station can contain a long automated focus
// bracket. Keep a guard against an accidental whole-scan component, but do
// not disable capture grouping for legitimate 10-32 frame brackets.
const FOCUS_BRACKET_MAX_COMPONENT_SOURCES: usize = 48;

/// Whether a member count can be represented by one Capture_Station before the
/// deterministic oversized-component split is applied (需求 1.5 / 1.10).
fn station_member_count_is_supported(count: usize) -> bool {
    (1..=FOCUS_BRACKET_MAX_COMPONENT_SOURCES).contains(&count)
}
const FOCUS_BRACKET_GRAPH_WEIGHT_BOOST: f64 = 24.0;
// A long scan of repeated calligraphy can contain a visually convincing edge
// between distant characters. Once two nearby capture numbers have a verified
// overlap, keep that camera-sequence evidence ahead of a remote lookalike when
// constructing the initial pose tree. Remote edges still connect separate
// capture runs, but cannot replace an already verified local backbone.
const FOCUS_CAPTURE_SEQUENCE_GRAPH_WEIGHT_BOOST: f64 = 128.0;
const FOCUS_BRACKET_OBSERVATION_WEIGHT: f64 = 16.0;
const CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX: f64 = 2.0;
const CLOSURE_DIRECT_PAIR_P95_LIMIT_PX: f64 = 3.0;
const CLOSURE_MAX_CORNER_CORRECTION_PX: f64 = 256.0;
const CLOSURE_OVERLAP_SHORT_SIDE_CORRECTION_RATIO: f64 = 0.10;
const CLOSURE_RELATIVE_RESIDUAL_DECREASE: f64 = 0.0001;
const CLOSURE_MAX_ITERATIONS: usize = 100;
const FOCUS_GLOBAL_MAX_LINEAR_ADJUSTMENT: f64 = 0.12;
const FOCUS_GLOBAL_MAX_TRANSLATION_ADJUSTMENT: f64 = 0.18;
const FOCUS_GLOBAL_MAX_PROJECTIVE_ADJUSTMENT: f64 = 0.02;
const FOCUS_LOCAL_MODEL_MAX_DISPLACEMENT_RATIO: f64 = 0.08;
// A random file-picker order must not become the focus-stack layer order. For
// a medium-sized stack, inspect every pair so the capture path can be rebuilt
// from image evidence rather than from filenames or import order.
const FOCUS_AUTO_ORDER_EXHAUSTIVE_MAX_SOURCES: usize = 64;
const SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES: usize = 65;
const SCALE_ROBUST_EXHAUSTIVE_MAX_SOURCES: usize = 96;
const FOCUS_AUTO_ORDER_MOTION_SCALE_FLOOR: f64 = 0.01;
const FOCUS_AUTO_ORDER_MOTION_EXPONENT: f64 = 4.0;
const FOCUS_AUTO_ORDER_GAP_PENALTY: f64 = 4.0;
const FOCUS_AUTO_ORDER_MISSING_EDGE_PENALTY: f64 = 32.0;
// Regional matches are measured against the already registered global model.
// Keep this close to the dense-search radius after conversion back to the
// source coordinate system. A wide gate can accept a different repeated stroke
// and let an under-constrained local fit bend an entire band.
const FOCUS_LOCAL_MATCH_RESIDUAL_RATIO: f64 = 0.008;
const FOCUS_DENSE_REGION_MIN_NCC: f64 = 0.48;
const FOCUS_FOREGROUND_PATCH_RADIUS: i32 = 12;
const FOCUS_FOREGROUND_SEARCH_RADIUS: i32 = 48;
const FOCUS_FOREGROUND_MIN_GRADIENT_ENERGY: f64 = 2.0;
const FOCUS_FOREGROUND_MIN_CORNER_ENERGY: f64 = 500.0;
pub(crate) const FOCUS_FOREGROUND_LUMA_THRESHOLD: u8 = 150;
const FOCUS_FOREGROUND_MIN_BRIGHT_FRACTION: f64 = 0.35;
const FOCUS_FOREGROUND_SCAN_MAX_Y: f64 = 0.45;
const FOCUS_FOREGROUND_MIN_HEIGHT_RATIO: f64 = 0.025;
const FOCUS_HORIZONTAL_EDGE_MAX_ROWS: usize = 8;
const FOCUS_HORIZONTAL_EDGE_MIN_SEPARATION_RATIO: f64 = 0.025;
const FOCUS_HORIZONTAL_EDGE_SEARCH_RADIUS: i32 = 32;
const FOCUS_HORIZONTAL_EDGE_SAMPLE_COUNT: usize = 64;
const FOCUS_HORIZONTAL_EDGE_MIN_SAMPLES: usize = 12;
const FOCUS_HORIZONTAL_EDGE_MIN_SOURCE_SPAN_RATIO: f64 = 0.45;
const FOCUS_HORIZONTAL_EDGE_FOREGROUND_MIN_SOURCE_SPAN_RATIO: f64 = 0.25;
const FOCUS_HORIZONTAL_EDGE_MIN_GRADIENT: f64 = 6.0;
const FOCUS_HORIZONTAL_EDGE_MAX_LINE_RESIDUAL_RATIO: f64 = 0.004;
const FOCUS_HORIZONTAL_EDGE_CLUSTER_TOLERANCE_RATIO: f64 = 0.012;
const FOCUS_HORIZONTAL_EDGE_DEDUP_TOLERANCE_RATIO: f64 = 0.0025;
const FOCUS_HORIZONTAL_EDGE_MAX_SLOPE_DELTA: f64 = 0.12;
const FOCUS_HORIZONTAL_EDGE_MIN_CLUSTER_IMAGES: usize = 3;
const FOCUS_HORIZONTAL_EDGE_MAX_CONSENSUS_ERROR_PX: f64 = 6.0;
const FOCUS_GROUP_ROLL_MIN_LINES: usize = 2;
const FOCUS_GROUP_ROLL_MAX_SLOPE_MAD: f64 = 0.018;
const FOCUS_GROUP_ROLL_MAX_CORRECTION_RADIANS: f64 = 0.045;
const FOCUS_HORIZONTAL_EDGE_BAND_HALF_HEIGHT_RATIO: f64 = 0.025;
const FOCUS_HORIZONTAL_EDGE_MAX_DISPLACEMENT_RATIO: f64 = 0.035;
const FOCUS_VERTICAL_EDGE_MAX_COLUMNS: usize = 8;
const FOCUS_VERTICAL_EDGE_MIN_SEPARATION_RATIO: f64 = 0.025;
const FOCUS_VERTICAL_EDGE_SEARCH_RADIUS: i32 = 32;
const FOCUS_VERTICAL_EDGE_SAMPLE_COUNT: usize = 64;
const FOCUS_VERTICAL_EDGE_MIN_SAMPLES: usize = 12;
const FOCUS_VERTICAL_EDGE_MIN_SOURCE_SPAN_RATIO: f64 = 0.08;
const FOCUS_VERTICAL_EDGE_MIN_GRADIENT: f64 = 6.0;
const FOCUS_VERTICAL_EDGE_MAX_LINE_RESIDUAL_RATIO: f64 = 0.004;
const FOCUS_VERTICAL_EDGE_CLUSTER_TOLERANCE_RATIO: f64 = 0.012;
const FOCUS_VERTICAL_EDGE_DEDUP_TOLERANCE_RATIO: f64 = 0.0025;
const FOCUS_VERTICAL_EDGE_MAX_SLOPE_DELTA: f64 = 0.12;
const FOCUS_VERTICAL_EDGE_MIN_CLUSTER_IMAGES: usize = 3;
const FOCUS_VERTICAL_EDGE_MAX_CONSENSUS_ERROR_PX: f64 = 6.0;
const FOCUS_VERTICAL_EDGE_BAND_HALF_WIDTH_RATIO: f64 = 0.025;
const FOCUS_VERTICAL_EDGE_MAX_DISPLACEMENT_RATIO: f64 = 0.035;
// Overlapping source-coordinate bands let a moving-camera stack absorb small
// residual lens/plane deformation without making the whole algorithm depend on
// a particular foreground object. A narrower detected depth layer, when it is
// trustworthy, is given precedence by the renderer.
const FOCUS_GENERIC_BAND_RANGES: [(f64, f64); 4] =
    [(0.00, 0.32), (0.22, 0.54), (0.44, 0.76), (0.66, 1.00)];
const FOCUS_GENERIC_BAND_MAX_DISPLACEMENT_RATIO: f64 = 0.012;
const FOCUS_DEPTH_LAYER_MAX_DISPLACEMENT_RATIO: f64 = 0.02;
const SCALABLE_STACK_THRESHOLD: usize = 30;
const LARGE_STACK_NEIGHBOR_WINDOW: usize = 4;
const SCALABLE_LOW_TEXTURE_FEATURE_TARGET: usize = 96;
const SCALABLE_LOW_TEXTURE_FAST_THRESHOLD: u8 = 7;
const SCALABLE_LOW_TEXTURE_NMS_RADIUS: f32 = 10.0;
// A cheap descriptor-retrieval pass supplements filename/input neighbours for
// large selections.  It is deliberately small: only the best few visual
// neighbours per image are promoted to full RANSAC matching.
const LARGE_STACK_RETRIEVAL_FEATURES: usize = 64;
const LARGE_STACK_RETRIEVAL_NEIGHBORS: usize = 6;
const LARGE_STACK_RETRIEVAL_MIN_MATCHES: usize = 6;
const SCALABLE_MATCH_RATIO_THRESHOLD: f32 = 0.88;
// A lens switch changes the BRIEF support scale even after the metadata-guided
// pyramid has supplied a common descriptor level.  Keep the ordinary ratio
// gate strict for same-focal pairs, but allow a mixed-focal fallback to reach
// geometric verification.  The later spatial/RANSAC/full-resolution checks
// remain the authority; this only prevents the descriptor gate from erasing a
// real bridge before those checks can run.
const MIXED_FOCAL_MATCH_RATIO_THRESHOLD: f32 = 0.96;
const MIXED_FOCAL_EMERGENCY_MATCH_RATIO_THRESHOLD: f32 = 0.99;
// A two-point similarity fallback is useful only for the lens-switch bridge.
// Its wider seed gate absorbs the scale-dependent BRIEF/keypoint quantisation;
// the native-patch refinement and the strict mixed-focal acceptance gate below
// still decide whether the bridge is real.
// Cross-module lens changes can carry several alignment-pixel quantisation
// errors before the scale-constrained refit. Keep the final median-error gate
// strict, but let RANSAC collect a wider provisional consensus so a real zoom
// bridge is not discarded prematurely.
const MIXED_FOCAL_SIMILARITY_RANSAC_THRESHOLD: f64 = 96.0;
const MIXED_FOCAL_SIMILARITY_RANSAC_ITERATIONS: usize = 2_500;
const SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION: usize = 10;
const PANORAMA_MODEL_INLIER_THRESHOLD: f64 = 8.0;
const PANORAMA_MODEL_MIN_INLIERS: usize = 8;
const PANORAMA_MODEL_MIN_ERROR_GAIN: f64 = 0.12;
const PANORAMA_MODEL_RELATIVE_ERROR_GAIN: f64 = 0.90;
const PANORAMA_MODEL_SCALE_EPSILON: f64 = 0.012;
const PANORAMA_MODEL_ROTATION_EPSILON: f64 = 0.004;
const MIXED_FOCAL_LENGTH_RATIO: f64 = 1.12;
// EXIF focal lengths are approximate and a phone may crop one module, so keep
// a broad tolerance.  Still reject a visually convincing repeated wall patch
// whose fitted transform has no relationship to the lens switch; accepting
// that edge would create a false bridge between two independent scenes.
const MIXED_FOCAL_SCALE_MIN_RATIO: f64 = 0.62;
const MIXED_FOCAL_SCALE_MAX_RATIO: f64 = 1.62;
// A 35mm/85mm bridge is allowed to connect the two lens groups only when it
// has a broad, high-confidence consensus. Small repeated-texture consensuses
// are particularly dangerous here: they can be geometrically plausible while
// placing an entire telephoto tile on the wrong canvas stroke.
// A close-up/detail frame against a wider frame can expose only a small
// repeated-but-consistent artwork patch. The scale gate, similarity RANSAC,
// spatial-support test, and median residual remain active; requiring 16 points
// here discarded the real 1038<->1041 bridge (12 valid correspondences).
const MIXED_FOCAL_MIN_INLIERS: usize = 10;
const MIXED_FOCAL_MAX_MEDIAN_ERROR: f64 = 4.5;
// Coarse lens-switch registration is only a candidate generator. A wrong
// repeated-stroke correlation usually has near-zero edge agreement even when
// its low-resolution luminance NCC looks deceptively respectable.
const MIXED_FOCAL_COARSE_MIN_INTENSITY_NCC: f64 = 0.30;
const MIXED_FOCAL_COARSE_MIN_EDGE_NCC: f64 = 0.15;
const MIXED_FOCAL_COARSE_MIN_EDGE_ORIENTATION: f64 = 0.25;
const FOCUS_MATCH_MIN_INTENSITY_NCC: f64 = 0.25;
const FOCUS_MATCH_MIN_EDGE_NCC: f64 = 0.10;
const FOCUS_MATCH_MIN_EDGE_ORIENTATION: f64 = 0.18;
const MAX_SCALABLE_PREPARATION_WORKERS: usize = 6;
const PREPARATION_RAM_PER_WORKER_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_IN_MEMORY_PANORAMA_PIXELS: u64 = 240_000_000;
const MAX_STITCH_SOURCE_IMAGES: usize = 500;
const MAX_RETAINED_STACK_PIXELS: u64 = 120_000_000;

fn stack_requires_bounded_memory(image_paths: &[String]) -> bool {
    if image_paths.len() > SCALABLE_STACK_THRESHOLD {
        return true;
    }
    let mut pixels = 0u64;
    for path in image_paths {
        // Unknown/RAW dimensions use the bounded path as well. A count-only
        // limit lets even two 200MP phone frames retain several GiB of floats.
        let Ok((width, height)) = image::image_dimensions(path) else {
            return true;
        };
        pixels = pixels.saturating_add(u64::from(width) * u64::from(height));
        if pixels > MAX_RETAINED_STACK_PIXELS {
            return true;
        }
    }
    false
}

#[derive(Clone, Copy, Debug)]
struct FocusHorizontalEdgeLine {
    image_id: usize,
    source_row: f64,
    world_x_center: f64,
    slope: f64,
    intercept: f64,
    median_error: f64,
}

#[derive(Clone, Copy, Debug)]
struct FocusVerticalEdgeLine {
    image_id: usize,
    source_column: f64,
    world_y_center: f64,
    slope: f64,
    intercept: f64,
    median_error: f64,
}

pub(crate) struct StitchOutcome {
    pub image: DynamicImage,
    /// The layered path's output Coverage_Mask, the same size as `image`;
    /// the export writes it as alpha (需求 10.3 / 10.6).
    pub coverage: Option<GrayImage>,
    pub full_canvas_width: u32,
    pub full_canvas_height: u32,
    pub render_scale: f64,
    pub ordered_paths: Vec<String>,
    /// The report written for this focus-stack run.  Image-stack export fills
    /// in the result identifier and output metadata after the file is
    /// published, while keeping the report tied to the same run.
    pub stack_report_path: Option<PathBuf>,
}

fn scalable_alignment_budget(image_count: usize) -> (u32, usize) {
    if image_count <= 64 {
        (2_400, 1_600)
    } else if image_count <= 128 {
        (1_800, 1_100)
    } else {
        (1_536, 800)
    }
}

fn focal_lengths_span_multiple_lenses<I>(focal_lengths: I) -> bool
where
    I: IntoIterator<Item = f64>,
{
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut count = 0usize;
    for focal_length in focal_lengths {
        if !focal_length.is_finite() || focal_length <= 0.0 {
            continue;
        }
        minimum = minimum.min(focal_length);
        maximum = maximum.max(focal_length);
        count += 1;
    }
    count >= 2
        && minimum.is_finite()
        && maximum.is_finite()
        && maximum / minimum >= MIXED_FOCAL_LENGTH_RATIO
}

fn selection_has_mixed_focal_lengths(image_paths: &[String]) -> bool {
    // Large stacks already use the scale-robust path.  This lightweight EXIF
    // pass exists for small selections, where the old count-only switch would
    // otherwise miss a 35mm/85mm camera-module change.
    focal_lengths_span_multiple_lenses(image_paths.iter().filter_map(|filename| {
        read_file_mapped(Path::new(filename))
            .ok()
            .and_then(|bytes| crate::exif_processing::focal_length_35mm_from_bytes(&bytes))
    }))
}

fn image_pair_has_mixed_focal_lengths(source: &ImageInfo, target: &ImageInfo) -> bool {
    focal_lengths_span_multiple_lenses(
        [source.focal_length_35mm, target.focal_length_35mm]
            .into_iter()
            .flatten(),
    )
}

fn images_have_mixed_focal_lengths(images: &[ImageInfo]) -> bool {
    focal_lengths_span_multiple_lenses(images.iter().filter_map(|image| image.focal_length_35mm))
}

fn bounded_preparation_worker_count(
    image_count: usize,
    available_threads: usize,
    available_memory_bytes: u64,
) -> usize {
    let memory_limit = (available_memory_bytes / PREPARATION_RAM_PER_WORKER_BYTES) as usize;
    image_count
        .max(1)
        .min(available_threads.max(1))
        .min(memory_limit.max(1))
        .min(MAX_SCALABLE_PREPARATION_WORKERS)
}

fn scalable_preparation_worker_count(image_count: usize) -> usize {
    let available_threads = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1);
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    bounded_preparation_worker_count(image_count, available_threads, system.available_memory())
}

fn memory_safe_panorama_render_scale(width: u32, height: u32) -> f64 {
    let pixels = u64::from(width).saturating_mul(u64::from(height));
    if pixels <= MAX_IN_MEMORY_PANORAMA_PIXELS || pixels == 0 {
        1.0
    } else {
        (MAX_IN_MEMORY_PANORAMA_PIXELS as f64 / pixels as f64).sqrt()
    }
}

#[cfg(test)]
fn scaled_homographies(
    homographies: &HashMap<usize, Matrix3<f64>>,
    scale: f64,
) -> HashMap<usize, Matrix3<f64>> {
    let scale_matrix = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
    homographies
        .iter()
        .map(|(&id, homography)| (id, scale_matrix * homography))
        .collect()
}

fn scaled_source_render_homographies(
    homographies: &HashMap<usize, Matrix3<f64>>,
    scale: f64,
) -> HashMap<usize, Matrix3<f64>> {
    let output_scale = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
    let source_scale_inverse = Matrix3::new(
        scale.recip(),
        0.0,
        0.0,
        0.0,
        scale.recip(),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    homographies
        .iter()
        .map(|(&id, homography)| (id, output_scale * homography * source_scale_inverse))
        .collect()
}

fn scaled_render_image_info(image: &ImageInfo, scale: f64) -> ImageInfo {
    ImageInfo {
        id: image.id,
        filename: image.filename.clone(),
        width: (f64::from(image.width) * scale).round().max(1.0) as u32,
        height: (f64::from(image.height) * scale).round().max(1.0) as u32,
        alignment_image: GrayImage::new(0, 0),
        full_image: None,
        scale_factor: image.scale_factor,
        focal_length_35mm: image.focal_length_35mm,
        source_footprint: image.source_footprint,
        overview_reference: image.overview_reference,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: image.foreground_range,
        foreground_mask: image.foreground_mask.clone(),
        horizontal_edge_rows: image.horizontal_edge_rows.clone(),
        vertical_edge_columns: image.vertical_edge_columns.clone(),
    }
}

struct AreaContributions {
    offsets: Vec<usize>,
    indices: Vec<usize>,
    weights: Vec<f32>,
}

fn area_contributions(source_length: u32, target_length: u32) -> AreaContributions {
    let mut offsets = Vec::with_capacity(target_length as usize + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    let scale = f64::from(source_length) / f64::from(target_length.max(1));
    offsets.push(0);
    for target in 0..target_length.max(1) {
        let start = f64::from(target) * scale;
        let end = (f64::from(target) + 1.0) * scale;
        let first = start.floor() as usize;
        let last = (end.ceil() as usize)
            .saturating_sub(1)
            .min(source_length.saturating_sub(1) as usize);
        let normalization = (end - start).recip();
        for source in first..=last {
            let overlap = ((source + 1) as f64).min(end) - (source as f64).max(start);
            if overlap > 0.0 {
                indices.push(source);
                weights.push((overlap * normalization) as f32);
            }
        }
        offsets.push(indices.len());
    }
    AreaContributions {
        offsets,
        indices,
        weights,
    }
}

fn resize_rgb8_area_to_rgb32f(
    source: &RgbImage,
    target_width: u32,
    target_height: u32,
) -> Rgb32FImage {
    let (source_width, source_height) = source.dimensions();
    debug_assert!(target_width > 0 && target_height > 0);
    debug_assert!(target_width <= source_width && target_height <= source_height);

    let horizontal_contributions = area_contributions(source_width, target_width);
    let vertical_contributions = area_contributions(source_height, target_height);
    let source_stride = source_width as usize * 3;
    let horizontal_stride = target_width as usize * 3;
    let mut horizontal = vec![0.0f32; horizontal_stride * source_height as usize];
    horizontal
        .par_chunks_mut(horizontal_stride)
        .zip(source.as_raw().par_chunks(source_stride))
        .for_each(|(output_row, source_row)| {
            for target_x in 0..target_width as usize {
                let contribution_start = horizontal_contributions.offsets[target_x];
                let contribution_end = horizontal_contributions.offsets[target_x + 1];
                let output_start = target_x * 3;
                for contribution in contribution_start..contribution_end {
                    let source_start = horizontal_contributions.indices[contribution] * 3;
                    let weight = horizontal_contributions.weights[contribution] / 255.0;
                    for channel in 0..3 {
                        output_row[output_start + channel] +=
                            f32::from(source_row[source_start + channel]) * weight;
                    }
                }
            }
        });

    let mut output = vec![0.0f32; horizontal_stride * target_height as usize];
    output
        .par_chunks_mut(horizontal_stride)
        .enumerate()
        .for_each(|(target_y, output_row)| {
            let contribution_start = vertical_contributions.offsets[target_y];
            let contribution_end = vertical_contributions.offsets[target_y + 1];
            for contribution in contribution_start..contribution_end {
                let source_y = vertical_contributions.indices[contribution];
                let weight = vertical_contributions.weights[contribution];
                let horizontal_row =
                    &horizontal[source_y * horizontal_stride..(source_y + 1) * horizontal_stride];
                for (output, source) in output_row.iter_mut().zip(horizontal_row) {
                    *output += source * weight;
                }
            }
        });

    Rgb32FImage::from_raw(target_width, target_height, output)
        .expect("area-resized RGB buffer dimensions must match")
}

fn source_to_render_rgb32f(
    source: DynamicImage,
    target_width: u32,
    target_height: u32,
) -> Rgb32FImage {
    if source.dimensions() == (target_width, target_height) {
        return source.to_rgb32f();
    }
    if source.color() == ColorType::Rgb8
        && target_width <= source.width()
        && target_height <= source.height()
    {
        return resize_rgb8_area_to_rgb32f(&source.into_rgb8(), target_width, target_height);
    }
    source
        .resize_exact(
            target_width,
            target_height,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb32f()
}

fn detect_horizontal_edge_rows(alignment_image: &GrayImage) -> Vec<f64> {
    let (width, height) = alignment_image.dimensions();
    if width < 96 || height < 96 {
        return Vec::new();
    }

    // A long paper/holder boundary has vertical gradient support across a large
    // fraction of the frame. Measuring that support rather than looking for a
    // particular luminance keeps this cue useful for white, dark, or coloured
    // borders alike, while suppressing the short horizontal strokes of text.
    let pixels = alignment_image.as_raw();
    let stride = width as usize;
    let interior_width = width.saturating_sub(2) as f64;
    let mut profile = vec![0.0f64; height as usize];
    for y in 1..height.saturating_sub(1) {
        let mut gradient_sum = 0.0f64;
        let mut supported = 0usize;
        for x in 1..width.saturating_sub(1) {
            let above = pixels[(y as usize - 1) * stride + x as usize] as i32;
            let below = pixels[(y as usize + 1) * stride + x as usize] as i32;
            let gradient = (below - above).unsigned_abs() as f64 * 0.5;
            gradient_sum += gradient;
            if gradient >= 14.0 {
                supported += 1;
            }
        }
        let mean_gradient = gradient_sum / interior_width;
        let support_ratio = supported as f64 / interior_width;
        profile[y as usize] = mean_gradient * (0.35 + support_ratio * 0.65);
    }

    let peak = profile.iter().copied().fold(0.0f64, f64::max);
    if !peak.is_finite() || peak < 4.0 {
        return Vec::new();
    }
    // Keep the detector permissive enough for a low-contrast paper/holder
    // boundary. Short character strokes still lose at the later line-fit
    // stage because they cannot form a low-residual line across the frame.
    let threshold = (peak * 0.10).max(2.5);
    let minimum_separation =
        ((height as f64 * FOCUS_HORIZONTAL_EDGE_MIN_SEPARATION_RATIO).round() as usize).max(16);
    let mut candidates = (1..height.saturating_sub(1))
        .filter(|&y| {
            let score = profile[y as usize];
            score >= threshold
                && score >= profile[y as usize - 1]
                && score >= profile[y as usize + 1]
        })
        .map(|y| (y, profile[y as usize]))
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|(_, left), (_, right)| right.total_cmp(left));

    let mut selected = Vec::with_capacity(FOCUS_HORIZONTAL_EDGE_MAX_ROWS);
    for (row, _) in candidates {
        if selected.iter().all(|selected_row: &u32| {
            (*selected_row as usize).abs_diff(row as usize) >= minimum_separation
        }) {
            selected.push(row);
            if selected.len() == FOCUS_HORIZONTAL_EDGE_MAX_ROWS {
                break;
            }
        }
    }
    selected.sort_unstable();
    selected
        .into_iter()
        .map(|row| row as f64 / height.max(1) as f64)
        .collect()
}

fn horizontal_edge_row_candidates(
    alignment_image: &GrayImage,
    optional_region: Option<(f64, f64)>,
) -> Vec<f64> {
    let mut rows = detect_horizontal_edge_rows(alignment_image);
    // The gradient profile is the primary, object-agnostic detector. When the
    // existing optional near-field detector has already found a bright/dark
    // region, also probe its silhouettes: a boundary can be partially cropped
    // or low-contrast enough to miss the broad profile while still being a
    // valid long edge. These are only extra candidates and still need the
    // robust line fit plus multi-image consensus below.
    if let Some((minimum, maximum)) = optional_region {
        for row in [minimum, maximum] {
            if row.is_finite()
                && (0.0..=1.0).contains(&row)
                && rows.iter().all(|existing| (*existing - row).abs() >= 0.005)
            {
                rows.push(row);
            }
        }
    }
    rows.sort_unstable_by(f64::total_cmp);
    rows
}

fn horizontal_edge_line_samples(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    normalized_row: f64,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let (width, height) = image.alignment_image.dimensions();
    if width < 96 || height < 96 || !normalized_row.is_finite() {
        return Vec::new();
    }

    let center_y = (normalized_row.clamp(0.02, 0.98) * height as f64).round() as i32;
    let search_radius = ((height as f64 * 0.025).round() as i32)
        .max(FOCUS_HORIZONTAL_EDGE_SEARCH_RADIUS)
        .min((height as i32 / 5).max(FOCUS_HORIZONTAL_EDGE_SEARCH_RADIUS));
    let x_start = ((width as f64 * 0.04).round() as i32).max(2);
    let x_end = ((width as f64 * 0.96).round() as i32).min(width as i32 - 3);
    if x_end <= x_start {
        return Vec::new();
    }
    let pixels = image.alignment_image.as_raw();
    let stride = width as usize;
    let y_start = (center_y - search_radius).max(2);
    let y_end = (center_y + search_radius).min(height as i32 - 3);
    let responses = (x_start..=x_end)
        .map(|x| {
            (y_start..=y_end)
                .map(|y| {
                    if !image.source_is_supported(Point2::new(
                        x as f64 * image.scale_factor,
                        (y - 1) as f64 * image.scale_factor,
                    )) || !image.source_is_supported(Point2::new(
                        x as f64 * image.scale_factor,
                        (y + 1) as f64 * image.scale_factor,
                    )) {
                        return (y, 0.0);
                    }
                    let above = pixels[(y as usize - 1) * stride + x as usize] as i32;
                    let below = pixels[(y as usize + 1) * stride + x as usize] as i32;
                    (y, (below - above).unsigned_abs() as f64 * 0.5)
                })
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .unwrap_or((center_y, 0.0))
        })
        .collect::<Vec<_>>();
    let maximum_gap = ((width as f64 * 0.01).round() as usize).max(2);
    let mut runs = Vec::<(usize, usize)>::new();
    let mut run_start = None;
    let mut last_supported = None;
    let mut gap = 0usize;
    for (index, (_, gradient)) in responses.iter().enumerate() {
        if *gradient >= FOCUS_HORIZONTAL_EDGE_MIN_GRADIENT {
            if run_start.is_none() {
                run_start = Some(index);
            }
            last_supported = Some(index);
            gap = 0;
        } else if let (Some(start), Some(last)) = (run_start, last_supported) {
            gap += 1;
            if gap > maximum_gap {
                runs.push((start, last));
                run_start = None;
                last_supported = None;
                gap = 0;
            }
        }
    }
    if let (Some(start), Some(last)) = (run_start, last_supported) {
        runs.push((start, last));
    }
    let Some((run_start, run_end)) = runs
        .into_iter()
        .max_by_key(|(start, end)| end.saturating_sub(*start) + 1)
    else {
        return Vec::new();
    };
    let minimum_span = if image.foreground_range.is_some() {
        FOCUS_HORIZONTAL_EDGE_FOREGROUND_MIN_SOURCE_SPAN_RATIO
    } else {
        FOCUS_HORIZONTAL_EDGE_MIN_SOURCE_SPAN_RATIO
    };
    if run_end.saturating_sub(run_start) + 1 < (width as f64 * minimum_span).round() as usize {
        return Vec::new();
    }

    let run_length = run_end.saturating_sub(run_start) + 1;
    let sample_count = FOCUS_HORIZONTAL_EDGE_SAMPLE_COUNT.min(run_length).max(1);
    let mut samples = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        let fraction = if sample_count == 1 {
            0.0
        } else {
            sample_index as f64 / (sample_count - 1) as f64
        };
        let response_index =
            run_start + (run_length.saturating_sub(1) as f64 * fraction).round() as usize;
        let x = x_start + response_index as i32;
        let (best_y, _) = responses[response_index];
        let source = Point2::new(
            x as f64 * image.scale_factor,
            best_y as f64 * image.scale_factor,
        );
        let Some(world) = transformed_point(homography, source) else {
            continue;
        };
        samples.push((source, world));
    }
    samples
}

fn fit_focus_horizontal_edge_line(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    normalized_row: f64,
) -> Option<FocusHorizontalEdgeLine> {
    let samples = horizontal_edge_line_samples(image, homography, normalized_row);
    if samples.len() < FOCUS_HORIZONTAL_EDGE_MIN_SAMPLES {
        return None;
    }
    let source_min_x = samples
        .iter()
        .map(|(source, _)| source.x)
        .fold(f64::INFINITY, f64::min);
    let source_max_x = samples
        .iter()
        .map(|(source, _)| source.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let minimum_source_span = if image.foreground_range.is_some() {
        FOCUS_HORIZONTAL_EDGE_FOREGROUND_MIN_SOURCE_SPAN_RATIO
    } else {
        FOCUS_HORIZONTAL_EDGE_MIN_SOURCE_SPAN_RATIO
    };
    if source_max_x - source_min_x < image.width as f64 * minimum_source_span {
        return None;
    }

    let mut slopes = Vec::new();
    for (index, (_, first)) in samples.iter().enumerate() {
        for (_, second) in samples.iter().skip(index + 1) {
            let delta_x = second.x - first.x;
            if delta_x.abs() < image.width as f64 * 0.02 {
                continue;
            }
            let slope = (second.y - first.y) / delta_x;
            if slope.is_finite() && slope.abs() <= FOCUS_HORIZONTAL_EDGE_MAX_SLOPE_DELTA * 2.0 {
                slopes.push(slope);
            }
        }
    }
    let slope = median_value(&mut slopes)?;
    let mut intercepts = samples
        .iter()
        .map(|(_, world)| world.y - slope * world.x)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let intercept = median_value(&mut intercepts)?;
    let mut errors = samples
        .iter()
        .map(|(_, world)| (world.y - (slope * world.x + intercept)).abs())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let median_error = median_value(&mut errors)?;
    let maximum_error =
        image.width.max(image.height) as f64 * FOCUS_HORIZONTAL_EDGE_MAX_LINE_RESIDUAL_RATIO;
    if !median_error.is_finite() || median_error > maximum_error {
        return None;
    }
    let mut source_rows = samples
        .iter()
        .map(|(source, _)| source.y / image.height.max(1) as f64)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let source_row = median_value(&mut source_rows)?;
    let mut world_xs = samples
        .iter()
        .map(|(_, world)| world.x)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let world_x_center = median_value(&mut world_xs)?;
    Some(FocusHorizontalEdgeLine {
        image_id: image.id,
        source_row,
        world_x_center,
        slope,
        intercept,
        median_error,
    })
}

fn deduplicate_focus_horizontal_edge_lines(
    lines: Vec<FocusHorizontalEdgeLine>,
    coordinate_scale: f64,
) -> Vec<FocusHorizontalEdgeLine> {
    if lines.len() < 2 {
        return lines;
    }
    let tolerance = coordinate_scale.max(1.0) * FOCUS_HORIZONTAL_EDGE_DEDUP_TOLERANCE_RATIO;
    let mut deduplicated = Vec::with_capacity(lines.len());
    for line in lines {
        let duplicate_index = deduplicated
            .iter()
            .position(|existing: &FocusHorizontalEdgeLine| {
                if existing.image_id != line.image_id
                    || (existing.slope - line.slope).abs() > FOCUS_HORIZONTAL_EDGE_MAX_SLOPE_DELTA
                {
                    return false;
                }
                let reference_x = (existing.world_x_center + line.world_x_center) * 0.5;
                (focus_horizontal_edge_line_y(existing, reference_x)
                    - focus_horizontal_edge_line_y(&line, reference_x))
                .abs()
                    <= tolerance
            });
        if let Some(index) = duplicate_index {
            if line.median_error < deduplicated[index].median_error {
                deduplicated[index] = line;
            }
        } else {
            deduplicated.push(line);
        }
    }
    deduplicated
}

fn detect_vertical_edge_columns(
    alignment_image: &GrayImage,
    optional_region: Option<(f64, f64)>,
) -> Vec<f64> {
    let (width, height) = alignment_image.dimensions();
    if width < 96 || height < 96 {
        return Vec::new();
    }

    // This is the transposed counterpart of the horizontal long-edge detector.
    // A vertical holder end or frame edge has horizontal-gradient support over
    // a long part of the frame, whereas individual character strokes are
    // rejected later by the robust line fit and cross-image consensus.
    let pixels = alignment_image.as_raw();
    let stride = width as usize;
    let region_start = optional_region
        .map(|(minimum, _)| (minimum.clamp(0.0, 1.0) * height as f64).round() as u32)
        .unwrap_or(1)
        .max(1)
        .min(height.saturating_sub(2));
    let region_end = optional_region
        .map(|(_, maximum)| (maximum.clamp(0.0, 1.0) * height as f64).round() as u32)
        .unwrap_or(height.saturating_sub(2))
        .max(region_start + 1)
        .min(height.saturating_sub(2));
    let interior_height = region_end.saturating_sub(region_start) as f64;
    if interior_height < 8.0 {
        return Vec::new();
    }
    let mut profile = vec![0.0f64; width as usize];
    for x in 1..width.saturating_sub(1) {
        let mut gradient_sum = 0.0f64;
        let mut supported = 0usize;
        for y in region_start..region_end {
            let left = pixels[y as usize * stride + (x as usize - 1)] as i32;
            let right = pixels[y as usize * stride + (x as usize + 1)] as i32;
            let gradient = (right - left).unsigned_abs() as f64 * 0.5;
            gradient_sum += gradient;
            if gradient >= 14.0 {
                supported += 1;
            }
        }
        let mean_gradient = gradient_sum / interior_height;
        let support_ratio = supported as f64 / interior_height;
        profile[x as usize] = mean_gradient * (0.35 + support_ratio * 0.65);
    }

    let peak = profile.iter().copied().fold(0.0f64, f64::max);
    if !peak.is_finite() || peak < 4.0 {
        return Vec::new();
    }
    let threshold = (peak * 0.10).max(2.5);
    let minimum_separation =
        ((width as f64 * FOCUS_VERTICAL_EDGE_MIN_SEPARATION_RATIO).round() as usize).max(16);
    let mut candidates = (1..width.saturating_sub(1))
        .filter(|&x| {
            let score = profile[x as usize];
            score >= threshold
                && score >= profile[x as usize - 1]
                && score >= profile[x as usize + 1]
        })
        .map(|x| (x, profile[x as usize]))
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|(_, left), (_, right)| right.total_cmp(left));

    let mut selected = Vec::with_capacity(FOCUS_VERTICAL_EDGE_MAX_COLUMNS);
    for (column, _) in candidates {
        if selected.iter().all(|selected_column: &u32| {
            (*selected_column as usize).abs_diff(column as usize) >= minimum_separation
        }) {
            selected.push(column);
            if selected.len() == FOCUS_VERTICAL_EDGE_MAX_COLUMNS {
                break;
            }
        }
    }
    selected.sort_unstable();
    selected
        .into_iter()
        .map(|column| column as f64 / width.max(1) as f64)
        .collect()
}

fn vertical_edge_line_samples(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    normalized_column: f64,
    optional_region: Option<(f64, f64)>,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let (width, height) = image.alignment_image.dimensions();
    if width < 96 || height < 96 || !normalized_column.is_finite() {
        return Vec::new();
    }

    let center_x = (normalized_column.clamp(0.02, 0.98) * width as f64).round() as i32;
    let search_radius = ((width as f64 * 0.025).round() as i32)
        .max(FOCUS_VERTICAL_EDGE_SEARCH_RADIUS)
        .min((width as i32 / 5).max(FOCUS_VERTICAL_EDGE_SEARCH_RADIUS));
    let y_start = optional_region
        .map(|(minimum, _)| (minimum.clamp(0.0, 1.0) * height as f64).round() as i32)
        .unwrap_or_else(|| (height as f64 * 0.04).round() as i32)
        .max(2);
    let y_end = optional_region
        .map(|(_, maximum)| (maximum.clamp(0.0, 1.0) * height as f64).round() as i32)
        .unwrap_or_else(|| (height as f64 * 0.96).round() as i32)
        .min(height as i32 - 3);
    if y_end <= y_start {
        return Vec::new();
    }
    let pixels = image.alignment_image.as_raw();
    let stride = width as usize;
    let x_start = (center_x - search_radius).max(2);
    let x_end = (center_x + search_radius).min(width as i32 - 3);
    let responses = (y_start..=y_end)
        .map(|y| {
            (x_start..=x_end)
                .map(|x| {
                    if !image.source_is_supported(Point2::new(
                        (x - 1) as f64 * image.scale_factor,
                        y as f64 * image.scale_factor,
                    )) || !image.source_is_supported(Point2::new(
                        (x + 1) as f64 * image.scale_factor,
                        y as f64 * image.scale_factor,
                    )) {
                        return (x, 0.0);
                    }
                    let left = pixels[y as usize * stride + (x as usize - 1)] as i32;
                    let right = pixels[y as usize * stride + (x as usize + 1)] as i32;
                    (x, (right - left).unsigned_abs() as f64 * 0.5)
                })
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .unwrap_or((center_x, 0.0))
        })
        .collect::<Vec<_>>();
    let maximum_gap = ((height as f64 * 0.01).round() as usize).max(2);
    let mut runs = Vec::<(usize, usize)>::new();
    let mut run_start = None;
    let mut last_supported = None;
    let mut gap = 0usize;
    for (index, (_, gradient)) in responses.iter().enumerate() {
        if *gradient >= FOCUS_VERTICAL_EDGE_MIN_GRADIENT {
            if run_start.is_none() {
                run_start = Some(index);
            }
            last_supported = Some(index);
            gap = 0;
        } else if let (Some(start), Some(last)) = (run_start, last_supported) {
            gap += 1;
            if gap > maximum_gap {
                runs.push((start, last));
                run_start = None;
                last_supported = None;
                gap = 0;
            }
        }
    }
    if let (Some(start), Some(last)) = (run_start, last_supported) {
        runs.push((start, last));
    }
    let Some((run_start, run_end)) = runs
        .into_iter()
        .max_by_key(|(start, end)| end.saturating_sub(*start) + 1)
    else {
        return Vec::new();
    };
    if run_end.saturating_sub(run_start) + 1
        < (height as f64 * FOCUS_VERTICAL_EDGE_MIN_SOURCE_SPAN_RATIO).round() as usize
    {
        return Vec::new();
    }

    let run_length = run_end.saturating_sub(run_start) + 1;
    let sample_count = FOCUS_VERTICAL_EDGE_SAMPLE_COUNT.min(run_length).max(1);
    let mut samples = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        let fraction = if sample_count == 1 {
            0.0
        } else {
            sample_index as f64 / (sample_count - 1) as f64
        };
        let response_index =
            run_start + (run_length.saturating_sub(1) as f64 * fraction).round() as usize;
        let y = y_start + response_index as i32;
        let (best_x, _) = responses[response_index];
        let source = Point2::new(
            best_x as f64 * image.scale_factor,
            y as f64 * image.scale_factor,
        );
        let Some(world) = transformed_point(homography, source) else {
            continue;
        };
        samples.push((source, world));
    }
    samples
}

fn fit_focus_vertical_edge_line(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    normalized_column: f64,
    optional_region: Option<(f64, f64)>,
) -> Option<FocusVerticalEdgeLine> {
    let samples = vertical_edge_line_samples(image, homography, normalized_column, optional_region);
    if samples.len() < FOCUS_VERTICAL_EDGE_MIN_SAMPLES {
        return None;
    }
    let source_min_y = samples
        .iter()
        .map(|(source, _)| source.y)
        .fold(f64::INFINITY, f64::min);
    let source_max_y = samples
        .iter()
        .map(|(source, _)| source.y)
        .fold(f64::NEG_INFINITY, f64::max);
    if source_max_y - source_min_y < image.height as f64 * FOCUS_VERTICAL_EDGE_MIN_SOURCE_SPAN_RATIO
    {
        return None;
    }

    let mut slopes = Vec::new();
    for (index, (_, first)) in samples.iter().enumerate() {
        for (_, second) in samples.iter().skip(index + 1) {
            let delta_y = second.y - first.y;
            if delta_y.abs() < image.height as f64 * 0.02 {
                continue;
            }
            let slope = (second.x - first.x) / delta_y;
            if slope.is_finite() && slope.abs() <= FOCUS_VERTICAL_EDGE_MAX_SLOPE_DELTA * 2.0 {
                slopes.push(slope);
            }
        }
    }
    let slope = median_value(&mut slopes)?;
    let mut intercepts = samples
        .iter()
        .map(|(_, world)| world.x - slope * world.y)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let intercept = median_value(&mut intercepts)?;
    let mut errors = samples
        .iter()
        .map(|(_, world)| (world.x - (slope * world.y + intercept)).abs())
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let median_error = median_value(&mut errors)?;
    let maximum_error =
        image.width.max(image.height) as f64 * FOCUS_VERTICAL_EDGE_MAX_LINE_RESIDUAL_RATIO;
    if !median_error.is_finite() || median_error > maximum_error {
        return None;
    }
    let mut source_columns = samples
        .iter()
        .map(|(source, _)| source.x / image.width.max(1) as f64)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let source_column = median_value(&mut source_columns)?;
    let mut world_ys = samples
        .iter()
        .map(|(_, world)| world.y)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let world_y_center = median_value(&mut world_ys)?;
    Some(FocusVerticalEdgeLine {
        image_id: image.id,
        source_column,
        world_y_center,
        slope,
        intercept,
        median_error,
    })
}

fn deduplicate_focus_vertical_edge_lines(
    lines: Vec<FocusVerticalEdgeLine>,
    coordinate_scale: f64,
) -> Vec<FocusVerticalEdgeLine> {
    if lines.len() < 2 {
        return lines;
    }
    let tolerance = coordinate_scale.max(1.0) * FOCUS_VERTICAL_EDGE_DEDUP_TOLERANCE_RATIO;
    let mut deduplicated = Vec::with_capacity(lines.len());
    for line in lines {
        let duplicate_index = deduplicated
            .iter()
            .position(|existing: &FocusVerticalEdgeLine| {
                if existing.image_id != line.image_id
                    || (existing.slope - line.slope).abs() > FOCUS_VERTICAL_EDGE_MAX_SLOPE_DELTA
                {
                    return false;
                }
                let reference_y = (existing.world_y_center + line.world_y_center) * 0.5;
                (focus_vertical_edge_line_x(existing, reference_y)
                    - focus_vertical_edge_line_x(&line, reference_y))
                .abs()
                    <= tolerance
            });
        if let Some(index) = duplicate_index {
            if line.median_error < deduplicated[index].median_error {
                deduplicated[index] = line;
            }
        } else {
            deduplicated.push(line);
        }
    }
    deduplicated
}

fn focus_vertical_edge_line_x(line: &FocusVerticalEdgeLine, reference_y: f64) -> f64 {
    line.slope * reference_y + line.intercept
}

fn focus_vertical_edge_line_clusters(
    lines: &[FocusVerticalEdgeLine],
    coordinate_scale: f64,
) -> Vec<Vec<usize>> {
    if lines.len() < FOCUS_VERTICAL_EDGE_MIN_CLUSTER_IMAGES {
        return Vec::new();
    }
    let mut center_ys = lines
        .iter()
        .map(|line| line.world_y_center)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let Some(reference_y) = median_value(&mut center_ys) else {
        return Vec::new();
    };
    let tolerance = coordinate_scale.max(1.0) * FOCUS_VERTICAL_EDGE_CLUSTER_TOLERANCE_RATIO;
    let mut ordered_indices = (0..lines.len()).collect::<Vec<_>>();
    ordered_indices.sort_unstable_by(|left, right| {
        focus_vertical_edge_line_x(&lines[*left], reference_y)
            .total_cmp(&focus_vertical_edge_line_x(&lines[*right], reference_y))
    });

    let mut clusters = Vec::<Vec<usize>>::new();
    for index in ordered_indices {
        let line = &lines[index];
        let line_x = focus_vertical_edge_line_x(line, reference_y);
        let mut best_cluster = None;
        let mut best_distance = f64::INFINITY;
        for (cluster_index, cluster) in clusters.iter().enumerate() {
            if cluster
                .iter()
                .any(|member| lines[*member].image_id == line.image_id)
            {
                continue;
            }
            let mut cluster_xs = cluster
                .iter()
                .map(|member| focus_vertical_edge_line_x(&lines[*member], reference_y))
                .collect::<Vec<_>>();
            let Some(cluster_x) = median_value(&mut cluster_xs) else {
                continue;
            };
            let mut cluster_slopes = cluster
                .iter()
                .map(|member| lines[*member].slope)
                .collect::<Vec<_>>();
            let Some(cluster_slope) = median_value(&mut cluster_slopes) else {
                continue;
            };
            let distance = (line_x - cluster_x).abs();
            if distance <= tolerance
                && (line.slope - cluster_slope).abs() <= FOCUS_VERTICAL_EDGE_MAX_SLOPE_DELTA
                && distance < best_distance
            {
                best_cluster = Some(cluster_index);
                best_distance = distance;
            }
        }
        if let Some(cluster_index) = best_cluster {
            clusters[cluster_index].push(index);
        } else {
            clusters.push(vec![index]);
        }
    }

    clusters
        .into_iter()
        .filter(|cluster| {
            cluster
                .iter()
                .map(|index| lines[*index].image_id)
                .collect::<HashSet<_>>()
                .len()
                >= FOCUS_VERTICAL_EDGE_MIN_CLUSTER_IMAGES
        })
        .collect()
}

fn focus_vertical_edge_correction(
    local_line: &FocusVerticalEdgeLine,
    target_slope: f64,
    target_intercept: f64,
) -> Option<Matrix3<f64>> {
    let shear = target_slope - local_line.slope;
    let translation = target_intercept - local_line.intercept;
    if !shear.is_finite()
        || !translation.is_finite()
        || shear.abs() > FOCUS_VERTICAL_EDGE_MAX_SLOPE_DELTA
    {
        return None;
    }
    Some(Matrix3::new(
        1.0,
        shear,
        translation,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ))
}

fn build_focus_vertical_edge_bands(
    images: &[ImageInfo],
    poses: &HashMap<usize, Matrix3<f64>>,
) -> Vec<FocusWarpBand> {
    build_focus_vertical_edge_bands_in_region(images, poses, true)
}

fn build_focus_vertical_edge_bands_in_region(
    images: &[ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    use_foreground_region: bool,
) -> Vec<FocusWarpBand> {
    let mut lines = Vec::new();
    for image in images {
        let Some(homography) = global_homographies.get(&image.id) else {
            continue;
        };
        for &normalized_column in &image.vertical_edge_columns {
            if let Some(line) = fit_focus_vertical_edge_line(
                image,
                homography,
                normalized_column,
                if use_foreground_region {
                    image.foreground_range
                } else {
                    None
                },
            ) {
                lines.push(line);
            }
        }
    }
    let coordinate_scale = images
        .iter()
        .map(|image| image.width.max(image.height) as f64)
        .fold(1.0, f64::max);
    let raw_line_count = lines.len();
    let lines = deduplicate_focus_vertical_edge_lines(lines, coordinate_scale);
    println!(
        "  - Vertical-edge fitted candidates: {raw_line_count} (after dedup: {})",
        lines.len()
    );
    if raw_line_count != lines.len() {
        println!(
            "  - Collapsed {} duplicate vertical-edge candidate line(s)",
            raw_line_count - lines.len()
        );
    }
    if lines.len() < FOCUS_VERTICAL_EDGE_MIN_CLUSTER_IMAGES {
        return Vec::new();
    }
    let clusters = focus_vertical_edge_line_clusters(&lines, coordinate_scale);
    if clusters.is_empty() {
        println!(
            "  - No repeated vertical-edge consensus found from {} detected edge line(s)",
            lines.len()
        );
        return Vec::new();
    }

    let reference_y = {
        let mut center_ys = lines
            .iter()
            .map(|line| line.world_y_center)
            .collect::<Vec<_>>();
        median_value(&mut center_ys).unwrap_or(0.0)
    };
    let mut bands = Vec::new();
    for cluster in clusters {
        let mut slopes = cluster
            .iter()
            .map(|index| lines[*index].slope)
            .collect::<Vec<_>>();
        let Some(target_slope) = median_value(&mut slopes) else {
            continue;
        };
        let mut target_xs = cluster
            .iter()
            .map(|index| focus_vertical_edge_line_x(&lines[*index], reference_y))
            .collect::<Vec<_>>();
        let Some(target_x_at_reference) = median_value(&mut target_xs) else {
            continue;
        };
        let target_intercept = target_x_at_reference - target_slope * reference_y;
        let mut homographies = HashMap::new();
        let mut source_ranges = HashMap::new();
        let mut source_x_ranges = HashMap::new();
        for &index in &cluster {
            let line = lines[index];
            let Some(global) = global_homographies.get(&line.image_id).copied() else {
                continue;
            };
            let Some(image) = images.iter().find(|image| image.id == line.image_id) else {
                continue;
            };
            let Some(correction) =
                focus_vertical_edge_correction(&line, target_slope, target_intercept)
            else {
                continue;
            };
            let corrected = correction * global;
            if !transform_is_stable_for_focus_stack(&corrected, image.dimensions()) {
                continue;
            }
            let minimum_source_x =
                (line.source_column - FOCUS_VERTICAL_EDGE_BAND_HALF_WIDTH_RATIO).max(0.0);
            let maximum_source_x =
                (line.source_column + FOCUS_VERTICAL_EDGE_BAND_HALF_WIDTH_RATIO).min(1.0);
            let displacement =
                focus_band_maximum_displacement(image, &global, &corrected, (0.0, 1.0));
            let maximum_allowed = image.width.max(image.height).max(1) as f64
                * FOCUS_VERTICAL_EDGE_MAX_DISPLACEMENT_RATIO;
            if !displacement.is_finite() || displacement > maximum_allowed {
                continue;
            }
            homographies.insert(line.image_id, corrected);
            source_ranges.insert(line.image_id, (0.0, 1.0));
            source_x_ranges.insert(line.image_id, (minimum_source_x, maximum_source_x));
        }
        if homographies.len() < FOCUS_VERTICAL_EDGE_MIN_CLUSTER_IMAGES {
            continue;
        }
        let average_error = cluster
            .iter()
            .map(|index| lines[*index].median_error)
            .sum::<f64>()
            / cluster.len().max(1) as f64;
        let maximum_consensus_error =
            FOCUS_VERTICAL_EDGE_MAX_CONSENSUS_ERROR_PX.max(coordinate_scale * 0.0005);
        if !average_error.is_finite() || average_error > maximum_consensus_error {
            println!(
                "  - Rejected vertical-edge consensus band: {} image(s), fit error {:.2}px exceeds {:.2}px",
                homographies.len(),
                average_error,
                maximum_consensus_error
            );
            continue;
        }
        println!(
            "  - Vertical-edge consensus band: {} image(s), world x {:.1}, fit error {:.2}px",
            homographies.len(),
            target_x_at_reference,
            average_error
        );
        bands.push(FocusWarpBand {
            homographies,
            source_ranges,
            source_x_ranges,
            relax_foreground_seam: true,
            foreground_only: false,
            physical_edge: true,
        });
    }
    bands
}

fn emit_match_progress<R: Runtime>(
    completed: &Mutex<usize>,
    total: usize,
    progress_step: usize,
    app_handle: &AppHandle<R>,
    progress_event: &str,
) {
    // Count and emit under the same short lock so parallel match workers cannot
    // deliver an older progress value after a newer one.
    let mut completed = completed
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *completed += 1;
    let current = *completed;
    if current != total && !current.is_multiple_of(progress_step) {
        return;
    }
    let overall_percentage = 30.0 + (current as f64 / total.max(1) as f64) * 24.0;
    let _ = app_handle.emit(
        progress_event,
        format!("Matching image overlaps {current} of {total} ({overall_percentage:.0}%)"),
    );
}

struct PairMatchProgress<'a, R: Runtime> {
    completed: &'a Mutex<usize>,
    total: usize,
    progress_step: usize,
    app_handle: &'a AppHandle<R>,
    progress_event: &'a str,
}

impl<R: Runtime> Drop for PairMatchProgress<'_, R> {
    fn drop(&mut self) {
        emit_match_progress(
            self.completed,
            self.total,
            self.progress_step,
            self.app_handle,
            self.progress_event,
        );
    }
}

struct PreparedStackSource {
    image: DynamicImage,
    source_footprint: Option<crate::lens_correction::StackLensFootprint>,
    focal_length_35mm: Option<f64>,
}

/// Photon exposure at the sensor, including the recorded ISO sensitivity.
/// This is a camera metadata calibration, not a frame histogram adjustment.
#[cfg(test)]
fn stack_raw_exposure_value(metadata: &HashMap<String, String>) -> Option<f64> {
    let number = |text: &str| {
        let text = text.trim().trim_start_matches("f/").trim_end_matches(" s");
        if let Some((a, b)) = text.split_once('/') {
            Some(a.parse::<f64>().ok()? / b.parse::<f64>().ok()?)
        } else {
            text.parse::<f64>().ok()
        }
    };
    let time = number(metadata.get("ExposureTime")?)?;
    let aperture = number(metadata.get("FNumber")?)?;
    let iso = number(
        metadata
            .get("ISOSpeedRatings")
            .or_else(|| metadata.get("PhotographicSensitivity"))
            .or_else(|| metadata.get("ISOSpeed"))?,
    )?;
    let exposure = time * iso / aperture.powi(2);
    (time > 0.0 && aperture > 0.0 && iso > 0.0 && exposure.is_finite()).then_some(exposure)
}

fn stack_raw_exposure_from_bytes(bytes: &[u8]) -> Option<f64> {
    let metadata = crate::exif_processing::read_raw_metadata(bytes)?;
    let time = metadata.exif.exposure_time?;
    let aperture = metadata.exif.fnumber?;
    if time.d == 0 || aperture.d == 0 {
        return None;
    }
    let iso = metadata
        .exif
        .iso_speed
        .or_else(|| metadata.exif.iso_speed_ratings.map(|iso| iso as u32))?;
    let time = f64::from(time.n) / f64::from(time.d);
    let aperture = f64::from(aperture.n) / f64::from(aperture.d);
    let exposure = time * f64::from(iso) / aperture.powi(2);
    (time > 0.0 && aperture > 0.0 && iso > 0 && exposure.is_finite()).then_some(exposure)
}

fn stack_raw_exposure_reference(paths: &[String]) -> Option<f64> {
    let mut exposures = Vec::with_capacity(paths.len());
    let mut camera = None;
    for path in paths {
        if !is_raw_file(path) {
            return None;
        }
        let bytes = read_file_mapped(Path::new(path)).ok()?;
        let metadata = crate::exif_processing::read_exif_data_from_bytes(path, &bytes);
        let identity = (
            metadata.get("Make")?.clone(),
            metadata.get("Model")?.clone(),
        );
        if camera.as_ref().is_some_and(|camera| camera != &identity) {
            return None;
        }
        camera = Some(identity);
        exposures.push(stack_raw_exposure_from_bytes(&bytes)?);
    }
    exposures.sort_by(f64::total_cmp);
    exposures.get(exposures.len() / 2).copied()
}

fn load_prepared_stack_source(
    filename: &str,
    settings: &AppSettings,
    raw_exposure_reference: Option<f64>,
) -> Result<PreparedStackSource, String> {
    let file_data = read_file_mapped(Path::new(filename))
        .map_err(|error| format!("Failed to read image {filename}: {error}"))?;
    let focal_length_35mm = crate::exif_processing::focal_length_35mm_from_bytes(&file_data);
    let mut image = crate::image_loader::load_base_image_from_bytes(
        &file_data, filename, false, settings, None,
    )
    .map_err(|error| format!("Failed to load image {filename}: {error}"))?;
    let mut source_footprint = None;
    if is_raw_file(filename) {
        let mut metadata = crate::exif_processing::read_exif_data_from_bytes(filename, &file_data);
        // RAW metadata extraction otherwise copies the physical focal length
        // into this field. Use the actual EXIF value to detect sensor crop mode.
        if let Some(equivalent) = focal_length_35mm {
            metadata.insert("FocalLengthIn35mmFilm".into(), equivalent.to_string());
        } else {
            metadata.remove("FocalLengthIn35mmFilm");
        }
        let calibration =
            crate::lens_correction::stack_lens_calibration(&metadata, image.dimensions());
        if let Some(reference) = raw_exposure_reference
            && let Some(exposure) = stack_raw_exposure_from_bytes(&file_data)
        {
            let gain = (reference / exposure) as f32;
            if gain.is_finite() && gain > 0.0 {
                if image.as_rgb32f().is_none() {
                    image = DynamicImage::ImageRgb32F(image.to_rgb32f());
                }
                image
                    .as_mut_rgb32f()
                    .unwrap()
                    .as_mut()
                    .par_iter_mut()
                    .for_each(|channel| *channel = (*channel * gain).max(0.0));
            }
        }
        if let Some(calibration) = calibration {
            if image.as_rgb32f().is_none() {
                image = DynamicImage::ImageRgb32F(image.to_rgb32f());
            }
            calibration.correct_linear_illumination(image.as_mut_rgb32f().unwrap());
        }
        apply_cpu_default_raw_processing(&mut image);
        if let Some(calibration) = calibration
            && let Some(corrected) = {
                // Developed RAW is already RGB32F. Borrow it so optical
                // preparation does not retain another full 45MP RGB buffer.
                let rgb = image
                    .as_rgb32f()
                    .map(Cow::Borrowed)
                    .unwrap_or_else(|| Cow::Owned(image.to_rgb32f()));
                calibration.correct(&rgb)
            }
        {
            println!(
                "  - Calibrated stack lens geometry: {} ({}x{} -> {}x{})",
                filename,
                image.width(),
                image.height(),
                corrected.width(),
                corrected.height()
            );
            source_footprint = Some(crate::lens_correction::StackLensFootprint {
                calibration,
                original: image.dimensions(),
                corrected: corrected.dimensions(),
            });
            image = DynamicImage::ImageRgb32F(corrected);
        }
    }
    Ok(PreparedStackSource {
        image,
        source_footprint,
        focal_length_35mm,
    })
}

fn normalize_stack_raw_display_exposure(rgb: &mut Rgb32FImage) {
    const TARGET_BRIGHT_PERCENTILE: f32 = 0.90;
    const MAX_DISPLAY_GAIN: f32 = 2.10;
    // The RAW developer deliberately leaves headroom, while image-stack
    // export freezes these samples directly as display-referred sRGB. Restore
    // a conservative display white from the bright end of the actual frame;
    // use a percentile rather than a maximum so lamps and specular acrylic do
    // not suppress the correction. This is bounded and can only brighten.
    let step = ((u64::from(rgb.width()) * u64::from(rgb.height()) / 24_000).max(1) as f64)
        .sqrt()
        .floor()
        .max(1.0) as usize;
    let mut luma_samples = Vec::with_capacity(24_000);
    for y in (0..rgb.height()).step_by(step) {
        for x in (0..rgb.width()).step_by(step) {
            let pixel = rgb.get_pixel(x, y);
            let luma = pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722;
            if luma.is_finite() && luma > 0.02 {
                luma_samples.push(luma);
            }
        }
    }
    if luma_samples.len() < 64 {
        return;
    }
    luma_samples.sort_by(f32::total_cmp);
    let high = luma_samples[((luma_samples.len() - 1) * 97) / 100];
    if !high.is_finite() || high <= 0.0 {
        return;
    }
    let gain = (TARGET_BRIGHT_PERCENTILE / high).clamp(1.0, MAX_DISPLAY_GAIN);
    if gain <= 1.005 {
        return;
    }
    rgb.as_mut()
        .par_iter_mut()
        .for_each(|channel| *channel = (*channel * gain).clamp(0.0, 1.0));
}

/// Return the robust bright-end luma of a developed RAW frame without
/// allocating a second image.  This is used only during focus-stack
/// preparation to derive one shared display gain for the whole selection.
/// Each frame already excludes the brightest 3% of pixels. Protect the
/// brightest retained channel across the batch: a median frame white clips
/// brighter scan regions and changes their RGB ratios before tone matching.
fn shared_stack_raw_display_gain(highs: &[f32]) -> Option<f32> {
    highs
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .max_by(f32::total_cmp)
        .map(|high| (0.90 / high).clamp(1.0, 2.10))
}

fn raw_display_channel_percentile(image: &DynamicImage) -> Option<f32> {
    let rgb = image.as_rgb32f()?;
    let step = ((u64::from(rgb.width()) * u64::from(rgb.height()) / 24_000).max(1) as f64)
        .sqrt()
        .floor()
        .max(1.0) as usize;
    let mut samples = Vec::with_capacity(24_000);
    for y in (0..rgb.height()).step_by(step) {
        for x in (0..rgb.width()).step_by(step) {
            let pixel = rgb.get_pixel(x, y);
            let luma = pixel[0].max(pixel[1]).max(pixel[2]);
            if luma.is_finite() && luma > 0.02 {
                samples.push(luma);
            }
        }
    }
    if samples.len() < 64 {
        return None;
    }
    samples.sort_unstable_by(f32::total_cmp);
    samples.get(((samples.len() - 1) * 97) / 100).copied()
}

fn prepare_focus_rescue_image(
    image: &ImageInfo,
    settings: &AppSettings,
    raw_exposure_reference: Option<f64>,
    brief_pairs: &[(nalgebra::Point2<i32>, nalgebra::Point2<i32>)],
) -> Result<ImageInfo, String> {
    let prepared = load_prepared_stack_source(&image.filename, settings, raw_exposure_reference)?;
    let (width, height) = prepared.image.dimensions();
    let (new_width, new_height, scale_factor) =
        processing::calculate_downscale_dimensions_capped(width, height, 2_400);
    let alignment_image = prepared
        .image
        .resize_exact(new_width, new_height, image::imageops::FilterType::Triangle)
        .to_luma8();
    let foreground_range = detect_foreground_range(&alignment_image);
    Ok(ImageInfo {
        id: image.id,
        filename: image.filename.clone(),
        width,
        height,
        features: find_alignment_features(
            &alignment_image,
            brief_pairs,
            1_600,
            true,
            prepared.focal_length_35mm,
        ),
        top_features: find_top_alignment_features(&alignment_image, brief_pairs, foreground_range),
        foreground_mask: build_foreground_mask(&alignment_image, foreground_range),
        horizontal_edge_rows: horizontal_edge_row_candidates(&alignment_image, foreground_range),
        vertical_edge_columns: detect_vertical_edge_columns(&alignment_image, foreground_range),
        alignment_image,
        full_image: None,
        scale_factor,
        focal_length_35mm: prepared.focal_length_35mm,
        source_footprint: prepared.source_footprint,
        overview_reference: false,
        foreground_range,
    })
}

fn largest_focus_match_component(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> HashSet<usize> {
    let motion_scale = focus_auto_order_motion_scale(images, matches);
    let mut adjacency = vec![Vec::new(); images.len()];
    // Direct map iteration is order safe here: the loop only inserts undirected
    // edges, and the connected components it feeds are the same set partition for
    // any edge order.  The winning component is chosen by a strict `>` over the
    // ascending root scan below, so no hash-seed order reaches the result
    // (需求 14.6).
    for (&(source, target), match_info) in matches {
        let Some(source_image) = images.get(source) else {
            continue;
        };
        let Some(target_image) = images.get(target) else {
            continue;
        };
        let bracket_boost =
            focus_graph_capture_sequence_boost(source_image, target_image, match_info);
        let weight =
            focus_auto_order_edge_weight(source_image, target_image, match_info, motion_scale)
                * focus_cycle_edge_factor(images, matches, source_image.id, target_image.id)
                * bracket_boost;
        if source != target && weight.is_finite() {
            adjacency[source].push(target);
            adjacency[target].push(source);
        }
    }
    let mut visited = HashSet::new();
    let mut largest = HashSet::new();
    for root in 0..images.len() {
        if !visited.insert(root) {
            continue;
        }
        let mut component = HashSet::from([root]);
        let mut pending = vec![root];
        while let Some(node) = pending.pop() {
            for &neighbour in &adjacency[node] {
                if visited.insert(neighbour) {
                    component.insert(neighbour);
                    pending.push(neighbour);
                }
            }
        }
        if component.len() > largest.len() {
            largest = component;
        }
    }
    largest
}

fn focus_graph_capture_sequence_boost(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
) -> f64 {
    let capture_gap = trailing_capture_number(&source.filename)
        .zip(trailing_capture_number(&target.filename))
        .map(|(left, right)| left.abs_diff(right));
    // A nearby filename is only a candidate-order hint.  It may strengthen a
    // geometrically verified local edge, but it cannot make a remote repeated
    // motif look like a focus bracket and it is never used by group
    // membership itself.
    if focus_match_is_local_bracket(source, target, match_info)
        && capture_gap.is_some_and(|gap| gap <= FOCUS_BRACKET_MAX_CAPTURE_GAP)
    {
        return FOCUS_CAPTURE_SEQUENCE_GRAPH_WEIGHT_BOOST;
    }
    if !match_info.sequence_bridge
        && !match_info.coarse_bridge
        && match_info.points.len() >= FOCUS_MODEL_MIN_INLIERS
        && capture_gap.is_some_and(|gap| gap <= 3)
    {
        FOCUS_CAPTURE_SEQUENCE_GRAPH_WEIGHT_BOOST
    } else {
        1.0
    }
}

fn focus_unstitched_sources_are_redundant(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    stitched_indices: &HashSet<usize>,
) -> bool {
    let mut filename_order = (0..images.len()).collect::<Vec<_>>();
    filename_order
        .sort_by(|&left, &right| natural_path_cmp(&images[left].filename, &images[right].filename));
    filename_order
        .iter()
        .enumerate()
        .filter(|(_, image_index)| !stitched_indices.contains(image_index))
        .all(|(position, &image_index)| {
            // Only discard a demonstrably weak interior frame. The verified
            // direct overlap across its two capture neighbours proves that no
            // unique spatial coverage is lost by omitting it.
            if images[image_index].features.len() > 64 {
                return false;
            }
            let previous = filename_order[..position]
                .iter()
                .rev()
                .copied()
                .find(|index| stitched_indices.contains(index));
            let next = filename_order[position + 1..]
                .iter()
                .copied()
                .find(|index| stitched_indices.contains(index));
            let (Some(previous), Some(next)) = (previous, next) else {
                return false;
            };
            let capture_span = trailing_capture_number(&images[previous].filename)
                .zip(trailing_capture_number(&images[next].filename))
                .map(|(left, right)| left.abs_diff(right));
            capture_span.is_some_and(|span| span <= 3)
                && matches.contains_key(&(previous.min(next), previous.max(next)))
        })
}

fn focus_overlap_is_verified(
    quality: (f64, f64, f64, usize),
    inlier_count: usize,
    median_error: f64,
    allow_low_texture_boundary: bool,
) -> bool {
    let (intensity_ncc, edge_ncc, edge_orientation, samples) = quality;
    let normal = intensity_ncc >= FOCUS_MATCH_MIN_INTENSITY_NCC
        && edge_ncc >= FOCUS_MATCH_MIN_EDGE_NCC
        && edge_orientation >= FOCUS_MATCH_MIN_EDGE_ORIENTATION;
    let low_texture_boundary = allow_low_texture_boundary
        && intensity_ncc >= 0.82
        && edge_ncc >= 0.30
        && edge_orientation >= 0.10
        && samples >= 1_000
        && inlier_count >= 12
        && median_error.is_finite()
        && median_error <= 2.5;
    normal || low_texture_boundary
}

fn focus_source_structure_plane(info: &ImageInfo) -> ImageInfo {
    let (width, height, scale_factor) = processing::calculate_downscale_dimensions_capped(
        info.width,
        info.height,
        info.alignment_image
            .width()
            .max(info.alignment_image.height())
            .min(600),
    );
    ImageInfo {
        id: info.id,
        filename: info.filename.clone(),
        width: info.width,
        height: info.height,
        scale_factor,
        alignment_image: imageproc::filter::gaussian_blur_f32(
            &image::imageops::resize(
                &info.alignment_image,
                width,
                height,
                image::imageops::FilterType::Triangle,
            ),
            1.0,
        ),
        full_image: None,
        focal_length_35mm: info.focal_length_35mm,
        source_footprint: info.source_footprint,
        features: Vec::new(),
        top_features: Vec::new(),
        overview_reference: false,
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

fn shifted_focus_structure_is_verified(
    source: &ImageInfo,
    target: &ImageInfo,
    homography: &Matrix3<f64>,
    points: &[(Point2<f64>, Point2<f64>)],
    log_match: bool,
) -> bool {
    if !focus_compatible_focal_length(source, target)
        || points.len() < 30
        || !focus_stack_motion_is_shifted_mosaic(points, source.dimensions())
        || !panorama_transform_is_stable(homography, source.dimensions())
        || panorama_spatial_support(points, source.dimensions(), target.dimensions())
            < FOCUS_PROJECTIVE_MIN_SPATIAL_SUPPORT
        || median_symmetric_error(homography, points) > 3.0
    {
        return false;
    }
    let quality = focus_overlap_quality(
        &focus_source_structure_plane(source),
        &focus_source_structure_plane(target),
        homography,
    );
    let verified = quality.is_some_and(|(intensity, edge, orientation, samples)| {
        intensity >= 0.65 && edge >= 0.30 && orientation >= 0.35 && samples >= 600
    });
    if log_match {
        println!(
            "  - Shifted focus structural validation: {:?} ({} distributed inliers, {})",
            quality,
            points.len(),
            if verified { "verified" } else { "rejected" }
        );
    }
    verified
}

fn station_relation_overlap_scale(
    transform: &Matrix3<f64>,
    points: &[(Point2<f64>, Point2<f64>)],
) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    let centre = points.iter().fold(Point2::origin(), |sum, point| {
        Point2::new(sum.x + point.0.x, sum.y + point.0.y)
    }) / points.len() as f64;
    let denominator =
        transform[(2, 0)] * centre.x + transform[(2, 1)] * centre.y + transform[(2, 2)];
    let determinant = transform.determinant() / denominator.powi(3);
    let scale = determinant.abs().sqrt();
    (denominator.is_finite()
        && denominator.abs() > f64::EPSILON
        && scale.is_finite()
        && scale > 0.0)
        .then_some(scale)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentMode {
    Auto,
    Perspective,
    Cylindrical,
    Spherical,
    Position,
    /// Alignment for a flat artwork photographed as a mostly orthogonal grid.
    /// Keep the per-pair model at similarity (translation + uniform scale +
    /// rotation) instead of allowing the panorama selector to overfit paper
    /// texture with a projective warp.
    FlatArtwork,
}

impl AlignmentMode {
    pub fn from_wire(value: &str) -> Self {
        match value {
            "perspective" => Self::Perspective,
            "cylindrical" => Self::Cylindrical,
            "spherical" => Self::Spherical,
            "position" => Self::Position,
            "flat-artwork" | "flat_artwork" | "artwork" => Self::FlatArtwork,
            _ => Self::Auto,
        }
    }

    fn projection_for(self, blend_mode: BlendMode) -> Projection {
        match self {
            Self::Cylindrical => Projection::Cylindrical,
            Self::Spherical => Projection::Spherical,
            Self::Perspective | Self::Position | Self::FlatArtwork => Projection::Planar,
            Self::Auto => match blend_mode {
                BlendMode::Panorama => Projection::Planar,
                BlendMode::FocusStack => Projection::Planar,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Panorama,
    FocusStack,
}

fn match_image_pair(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    projection: Projection,
    blend_mode: BlendMode,
    alignment_mode: AlignmentMode,
    stable_four_point_solver: bool,
    log_match: bool,
    allow_low_texture_boundary: bool,
) -> Option<MatchInfo> {
    match_image_pair_with_seed(
        source_image,
        target_image,
        projection,
        blend_mode,
        alignment_mode,
        stable_four_point_solver,
        log_match,
        allow_low_texture_boundary,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn match_image_pair_with_seed(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    projection: Projection,
    blend_mode: BlendMode,
    alignment_mode: AlignmentMode,
    stable_four_point_solver: bool,
    log_match: bool,
    allow_low_texture_boundary: bool,
    seeded_initial_matches: Option<Vec<Match>>,
) -> Option<MatchInfo> {
    let candidates = build_match_candidates(
        source_image,
        target_image,
        blend_mode,
        stable_four_point_solver,
        log_match,
        seeded_initial_matches,
    );
    let MatchCandidates {
        initial_matches,
        mixed_focal_pair,
        minimum_inliers,
        adjacent_same_focal_focus_pair,
    } = candidates;
    let features1 = &source_image.features;
    let features2 = &target_image.features;
    // Filename adjacency is only a candidate-order hint. It must not enable
    // a coarse transform during pair matching: with shuffled imports, an
    // arbitrary numeric suffix can otherwise turn two scan positions into a
    // false focus-layer edge before geometry has been established.
    let severe_focus_feature_imbalance = features1.len().min(features2.len()).saturating_mul(4)
        < features1.len().max(features2.len());
    if adjacent_same_focal_focus_pair
        && severe_focus_feature_imbalance
        && let Some((homography, points, score)) = {
            let source_focal = source_image
                .focal_length_35mm
                .or(target_image.focal_length_35mm)
                .unwrap_or(50.0);
            let target_focal = target_image
                .focal_length_35mm
                .or(source_image.focal_length_35mm)
                .unwrap_or(source_focal);
            estimate_mixed_focal_coarse_registration(
                source_image,
                target_image,
                source_focal,
                target_focal,
                true,
                log_match,
            )
        }
    {
        if log_match {
            println!(
                "  - Defocused adjacent bracket registration recovered {} points (NCC {:.3})",
                points.len(),
                score
            );
        }
        return Some(build_match_info(
            homography,
            points.clone(),
            points,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            true,
        ));
    }
    if initial_matches.len() < minimum_inliers {
        // A focus bracket can contain one deliberately defocused endpoint
        // whose local descriptors collapse even though the frame has the same
        // camera pose as its immediate neighbour. Photoshop can still align
        // these pairs from their global image structure. Give only adjacent,
        // same-focal captures that bounded fallback: the coarse registrar
        // validates luminance and edges before returning a transform, while
        // the filename/focal gates prevent it from connecting unrelated scan
        // tiles merely because they contain repeated artwork texture.
        if adjacent_same_focal_focus_pair
            && let Some((homography, points, score)) = {
                let source_focal = source_image
                    .focal_length_35mm
                    .or(target_image.focal_length_35mm)
                    .unwrap_or(50.0);
                let target_focal = target_image
                    .focal_length_35mm
                    .or(source_image.focal_length_35mm)
                    .unwrap_or(source_focal);
                estimate_mixed_focal_coarse_registration(
                    source_image,
                    target_image,
                    source_focal,
                    target_focal,
                    true,
                    log_match,
                )
            }
        {
            if log_match {
                println!(
                    "  - Adjacent focus-bracket image registration recovered {} points (NCC {:.3})",
                    points.len(),
                    score
                );
            }
            return Some(build_match_info(
                homography,
                points.clone(),
                points,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                true,
            ));
        }
        if log_match && mixed_focal_pair {
            println!(
                "  - Rejecting mixed-focal pair before geometry: {} descriptor candidates (need {})",
                initial_matches.len(),
                minimum_inliers
            );
        }
        return None;
    }

    let geometry = match match_geometry::verify_match_geometry(
        source_image,
        target_image,
        projection,
        blend_mode,
        alignment_mode,
        stable_four_point_solver,
        log_match,
        allow_low_texture_boundary,
        mixed_focal_pair,
        minimum_inliers,
        &initial_matches,
    ) {
        None => return None,
        Some(match_geometry::MatchGeometryResult::CoarseBridge { homography, points }) => {
            return Some(build_match_info(
                homography,
                points.clone(),
                points,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                true,
            ));
        }
        Some(match_geometry::MatchGeometryResult::Verified(geometry)) => geometry,
    };
    let (geometry, inlier_count) = match match_focus_rescue::apply_focus_rescue(
        source_image,
        target_image,
        geometry,
        mixed_focal_pair,
        blend_mode,
        stable_four_point_solver,
        allow_low_texture_boundary,
        log_match,
    ) {
        match_focus_rescue::FocusRescueResult::Rejected => return None,
        match_focus_rescue::FocusRescueResult::Rescued(match_info) => return Some(match_info),
        match_focus_rescue::FocusRescueResult::Accepted {
            geometry,
            inlier_count,
        } => (geometry, inlier_count),
    };
    let match_geometry::MatchGeometry {
        homography,
        inlier_points,
        projected_match_points,
        top_candidate_points,
        dense_focus_points,
        foreground_feature_points,
        ..
    } = geometry;
    if log_match {
        println!(
            "  - Good match found: '{}' <-> '{}' ({} inliers)",
            Path::new(&source_image.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            Path::new(&target_image.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            inlier_count
        );
        let reprojection_error = symmetric_reprojection_rmse(&homography, &inlier_points);
        println!(
            "  - Refined match: '{}' <-> '{}' ({} inliers, {:.3}px symmetric RMS)",
            Path::new(&source_image.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            Path::new(&target_image.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            inlier_count,
            reprojection_error
        );
    }
    Some(build_match_info(
        homography,
        inlier_points,
        projected_match_points,
        top_candidate_points,
        dense_focus_points,
        foreground_feature_points,
        false,
    ))
}

fn focus_alignment_support(image: &ImageInfo) -> usize {
    image.features.len().max(image.top_features.len())
}

fn focus_position_recovery_accepts(
    capture_gap: u64,
    same_focal: bool,
    scale: f64,
    center_motion: f64,
    overlap: f64,
    samples: usize,
    intensity_ncc: f64,
    edge_ncc: f64,
    edge_orientation: f64,
    orphan_attach: bool,
) -> bool {
    let scale_ok = if same_focal {
        (0.86..=1.14).contains(&scale)
    } else {
        (MIXED_FOCAL_SCALE_MIN_RATIO..=MIXED_FOCAL_SCALE_MAX_RATIO).contains(&scale)
    };
    // A small NCC patch is useful for a same-pose focus layer, but cannot
    // establish that two scan positions are neighbours. Position bridges are
    // retried with feature/RANSAC matching instead of this relaxed path.
    if !scale_ok || center_motion <= 0.05 || center_motion > 0.92 || overlap < 0.10 || samples < 300
    {
        return false;
    }
    if orphan_attach && capture_gap <= 2 && intensity_ncc >= 0.74 && edge_ncc >= 0.12 {
        return true;
    }
    if capture_gap == 1 && intensity_ncc >= 0.62 && edge_ncc >= 0.35 && overlap >= 0.20 {
        return true;
    }
    if intensity_ncc < 0.70 || edge_ncc < 0.14 || overlap < 0.12 || samples < 400 {
        return false;
    }
    edge_orientation >= 0.16
        || (capture_gap <= 2 && intensity_ncc >= 0.85 && edge_ncc >= 0.20)
        || (capture_gap == 1 && intensity_ncc >= 0.78 && edge_ncc >= 0.16)
}

fn focus_capture_stations(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Vec<Vec<usize>> {
    let mut dsu = Dsu::new(images.len());
    for (&(left, right), match_info) in sorted_pairs(matches) {
        if left >= images.len() || right >= images.len() {
            continue;
        }
        if focus_match_is_capture_station_link(images, left, right, match_info) {
            dsu.union(left, right);
        }
    }
    // Grouping by root in an ordered map makes the station order depend on the
    // representative image index instead of the hash seed (需求 14.6).
    let mut by_root = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..images.len() {
        by_root.entry(dsu.find(index)).or_default().push(index);
    }
    let mut stations = by_root.into_values().collect::<Vec<_>>();
    for station in &mut stations {
        station.sort_by(|&left, &right| {
            natural_path_cmp(&images[left].filename, &images[right].filename)
                .then_with(|| left.cmp(&right))
        });
    }
    stations.sort_by(|left, right| {
        let left_number = left
            .iter()
            .filter_map(|&index| trailing_capture_number(&images[index].filename))
            .min()
            .unwrap_or(u64::MAX);
        let right_number = right
            .iter()
            .filter_map(|&index| trailing_capture_number(&images[index].filename))
            .min()
            .unwrap_or(u64::MAX);
        // Sources without a trailing capture number all collapse to `u64::MAX`.
        // The members themselves are the stable secondary key (需求 14.6).
        left_number
            .cmp(&right_number)
            .then_with(|| left.first().cmp(&right.first()))
    });
    stations
}

fn attach_focus_bracket_orphans_to_pose_graph(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    ordered_indices: &mut Vec<usize>,
    global_homographies: &mut HashMap<usize, Matrix3<f64>>,
) -> usize {
    let mut attached = 0usize;
    let mut pending: Vec<usize> = (0..images.len())
        .filter(|index| !global_homographies.contains_key(index))
        .collect();
    pending.sort_by(|&left, &right| {
        natural_path_cmp(&images[left].filename, &images[right].filename)
            .then_with(|| left.cmp(&right))
    });
    while !pending.is_empty() {
        let mut progress = false;
        let mut next_pending = Vec::new();
        for orphan in pending {
            let mut best: Option<(usize, Matrix3<f64>, usize)> = None;
            for anchor in 0..images.len() {
                if anchor == orphan || !global_homographies.contains_key(&anchor) {
                    continue;
                }
                let key = (anchor.min(orphan), anchor.max(orphan));
                let Some(match_info) = matches.get(&key) else {
                    continue;
                };
                let (source, target) = if anchor < orphan {
                    (&images[anchor], &images[orphan])
                } else {
                    (&images[orphan], &images[anchor])
                };
                if !focus_match_is_local_bracket(source, target, match_info) {
                    continue;
                }
                let Some((_, anchor_to_orphan)) =
                    focus_stack_link_transform(matches, anchor, orphan)
                else {
                    continue;
                };
                let homography = global_homographies[&anchor] * anchor_to_orphan;
                let anchor_support = focus_alignment_support(&images[anchor]);
                if best.as_ref().is_none_or(|(_, _, best_support)| {
                    anchor_support > *best_support
                        || (anchor_support == *best_support && anchor < orphan)
                }) {
                    best = Some((anchor, homography, anchor_support));
                }
            }
            if let Some((_, homography, _)) = best {
                global_homographies.insert(orphan, homography);
                if !ordered_indices.contains(&orphan) {
                    ordered_indices.push(orphan);
                }
                attached += 1;
                progress = true;
            } else {
                next_pending.push(orphan);
            }
        }
        if !progress {
            break;
        }
        pending = next_pending;
    }
    attached
}

fn focus_station_representative_indices(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> HashSet<usize> {
    focus_capture_stations(images, matches)
        .into_iter()
        .map(|station| focus_station_representative(&station, images))
        .collect()
}

fn focus_stack_spine_edge(
    images: &[ImageInfo],
    left: usize,
    right: usize,
    match_info: &MatchInfo,
    representatives: &HashSet<usize>,
) -> bool {
    if match_info.sequence_bridge {
        return false;
    }
    if focus_match_is_local_bracket(&images[left], &images[right], match_info) {
        return true;
    }
    if !match_info.coarse_bridge {
        return match_info.inliers.max(match_info.points.len()) >= FOCUS_MODEL_MIN_INLIERS;
    }
    // Coarse NCC may recover another focus plane at one pose, but it is not
    // sufficient evidence for a scan-position edge. Filename-adjacent frames
    // can straddle a raster reset and share a repeated seal or figure. Keep
    // coarse position bridges representative-to-representative only; the
    // remaining position edges must come from feature/RANSAC evidence.
    // A multi-layer consensus bridge is still excluded from bundle
    // adjustment, but it is safe enough to seed the pose graph. The marker is
    // kept in `top_candidate_points`; ordinary coarse bridges leave that list
    // empty and remain representative-gated.
    let consensus_coarse = !match_info.top_candidate_points.is_empty();
    if !consensus_coarse && (!representatives.contains(&left) || !representatives.contains(&right))
    {
        return false;
    }
    let Some(motion) = focus_match_center_motion_ratio(&images[left], &images[right], match_info)
    else {
        return false;
    };
    if motion <= FOCUS_BRACKET_MAX_CENTER_MOTION_RATIO {
        return false;
    }
    let capture_gap = focus_capture_number_gap(&images[left], &images[right]).unwrap_or(u64::MAX);
    let same_focal = focus_compatible_focal_length(&images[left], &images[right]);
    let scale = match_info.homography[(0, 0)].abs();
    let overlap = panorama_transform_overlap_support(
        &match_info.homography,
        images[left].dimensions(),
        images[right].dimensions(),
    );
    if consensus_coarse {
        true
    } else {
        focus_overlap_quality(&images[left], &images[right], &match_info.homography).is_some_and(
            |(intensity_ncc, edge_ncc, edge_orientation, samples)| {
                focus_position_recovery_accepts(
                    capture_gap,
                    same_focal,
                    scale,
                    motion,
                    overlap,
                    samples,
                    intensity_ncc,
                    edge_ncc,
                    edge_orientation,
                    false,
                )
            },
        )
    }
}

fn focus_stack_spine_matches(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> HashMap<(usize, usize), MatchInfo> {
    if images.len() < SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES {
        return matches.clone();
    }
    let representatives = focus_station_representative_indices(images, matches);
    matches
        .iter()
        .filter(|(key, match_info)| {
            focus_stack_spine_edge(images, key.0, key.1, match_info, &representatives)
        })
        .map(|(key, match_info)| (*key, match_info.clone()))
        .collect()
}

fn focus_capture_geometry_diagnostics(
    images: &[ImageInfo],
    order: &[usize],
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> (usize, usize, f64, f64) {
    let mut valid_edges = 0usize;
    let mut total_edges = 0usize;
    let mut overlap_values = Vec::new();
    let mut scales = Vec::new();
    for image in images {
        let Some(homography) = homographies.get(&image.id) else {
            continue;
        };
        let (scale, _, anisotropy) = panorama_model_linear_characteristics(homography);
        if scale.is_finite() && anisotropy.is_finite() {
            scales.push(scale);
        }
    }
    for pair in order.windows(2) {
        let source = pair[0];
        let target = pair[1];
        total_edges += 1;
        let Some(source_global) = homographies.get(&images[source].id) else {
            continue;
        };
        let Some(target_global) = homographies.get(&images[target].id) else {
            continue;
        };
        let Some(source_to_target) = target_global
            .try_inverse()
            .map(|inverse| inverse * source_global)
        else {
            continue;
        };
        let overlap = panorama_transform_overlap_support(
            &source_to_target,
            images[source].dimensions(),
            images[target].dimensions(),
        );
        let Some((intensity_ncc, edge_ncc, edge_orientation, samples)) =
            focus_overlap_quality(&images[source], &images[target], &source_to_target)
        else {
            continue;
        };
        let same_focal = focus_compatible_focal_length(&images[source], &images[target]);
        let scale = source_to_target[(0, 0)].abs();
        let accepted = same_focal
            && scale.is_finite()
            && (0.82..=1.18).contains(&scale)
            && overlap >= 0.10
            && samples >= 120
            && intensity_ncc >= 0.45
            && edge_ncc >= -0.10
            && edge_orientation >= -0.10;
        if accepted {
            valid_edges += 1;
            overlap_values.push(overlap);
        } else {
            println!(
                "  - Focus geometry retry candidate rejected edge '{}'<->'{}': scale {:.3}, overlap {:.1}%, NCC {:.3}, edges {:.3}/{:.3}, samples {}",
                Path::new(&images[source].filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                Path::new(&images[target].filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                scale,
                overlap * 100.0,
                intensity_ncc,
                edge_ncc,
                edge_orientation,
                samples,
            );
        }
    }
    let median_overlap = median_value(&mut overlap_values).unwrap_or(0.0);
    let median_scale = median_value(&mut scales).unwrap_or(0.0);
    (valid_edges, total_edges, median_overlap, median_scale)
}

fn focus_capture_group_geometry_diagnostics(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    groups: &[FocusCaptureGroup],
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> (usize, usize, usize, usize, f64) {
    let mut group_for_image = vec![usize::MAX; images.len()];
    for (group_index, group) in groups.iter().enumerate() {
        for &image_index in &group.members {
            group_for_image[image_index] = group_index;
        }
    }
    // Both the station-pair order and the image-pair order inside a station pair
    // decide which candidate wins the `best_score` comparison that is reported in
    // the rejection diagnostics, so neither may follow the hash seed (需求 14.6).
    let mut relation_pairs = BTreeMap::<(usize, usize), Vec<(usize, usize)>>::new();
    for (source, target) in sorted_keys(matches) {
        let (Some(&source_group), Some(&target_group)) =
            (group_for_image.get(source), group_for_image.get(target))
        else {
            continue;
        };
        if source_group == usize::MAX || target_group == usize::MAX || source_group == target_group
        {
            continue;
        }
        relation_pairs
            .entry((
                source_group.min(target_group),
                source_group.max(target_group),
            ))
            .or_default()
            .push((source, target));
    }

    let candidate_relations = relation_pairs.len();
    let mut valid_relations = 0usize;
    let mut connectivity = Dsu::new(groups.len());
    let mut overlap_values = Vec::new();
    for ((left_group, right_group), image_pairs) in relation_pairs {
        let mut accepted_overlap = None::<f64>;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_summary = None;
        for (key_source, key_target) in image_pairs {
            // `relation_pairs` carries the `(min import index, max import index)`
            // orientation of the `matches` key, and everything below is
            // direction sensitive: `focus_overlap_quality` warps the source into
            // the target and correlates there, and the `scale` gate
            // `0.82..=1.18` is not symmetric around 1.0, so a relation measured
            // at 0.948 in one direction is measured at 1.055 in the other and a
            // near-threshold pair can be accepted one way and rejected the
            // other.  Measuring in the key direction therefore let a shuffled
            // import change which station relations pass this validation, which
            // in turn changed the accepted closure and every world pose.  Take
            // the same content derived direction the station solver takes
            // (需求 14.6).
            let (source, target, _) = canonical_match_direction(images, key_source, key_target);
            let (Some(source_global), Some(target_global)) = (
                homographies.get(&images[source].id),
                homographies.get(&images[target].id),
            ) else {
                continue;
            };
            let Some(source_to_target) = target_global
                .try_inverse()
                .map(|inverse| inverse * source_global)
            else {
                continue;
            };
            let overlap = panorama_transform_overlap_support(
                &source_to_target,
                images[source].dimensions(),
                images[target].dimensions(),
            );
            let Some((intensity_ncc, edge_ncc, edge_orientation, samples)) =
                focus_overlap_quality(&images[source], &images[target], &source_to_target)
            else {
                continue;
            };
            let scale = source_to_target[(0, 0)].abs();
            let accepted = focus_compatible_focal_length(&images[source], &images[target])
                && scale.is_finite()
                && (0.82..=1.18).contains(&scale)
                && overlap >= 0.10
                && samples >= 120
                && intensity_ncc >= 0.45
                && edge_ncc >= -0.10
                && edge_orientation >= -0.10;
            let score = intensity_ncc + edge_ncc.max(-0.10) + edge_orientation.max(-0.10);
            if score > best_score {
                best_score = score;
                best_summary = Some((
                    source,
                    target,
                    scale,
                    overlap,
                    intensity_ncc,
                    edge_ncc,
                    edge_orientation,
                    samples,
                ));
            }
            if accepted {
                accepted_overlap = Some(accepted_overlap.map_or(overlap, |best| best.max(overlap)));
            }
        }
        if let Some(overlap) = accepted_overlap {
            valid_relations += 1;
            connectivity.union(left_group, right_group);
            overlap_values.push(overlap);
        } else if let Some((
            source,
            target,
            scale,
            overlap,
            intensity_ncc,
            edge_ncc,
            edge_orientation,
            samples,
        )) = best_summary
        {
            println!(
                "  - Focus station relation rejected '{}'<->'{}': scale {:.3}, overlap {:.1}%, NCC {:.3}, edges {:.3}/{:.3}, samples {}",
                Path::new(&images[source].filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                Path::new(&images[target].filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                scale,
                overlap * 100.0,
                intensity_ncc,
                edge_ncc,
                edge_orientation,
                samples,
            );
        }
    }
    let mut component_sizes = HashMap::<usize, usize>::new();
    for group in 0..groups.len() {
        *component_sizes.entry(connectivity.find(group)).or_default() += 1;
    }
    let connected_groups = component_sizes.values().copied().max().unwrap_or(0);
    let median_overlap = median_value(&mut overlap_values).unwrap_or(0.0);
    (
        connected_groups,
        groups.len(),
        valid_relations,
        candidate_relations,
        median_overlap,
    )
}

fn focus_capture_group_geometry_passes(
    connected_groups: usize,
    total_groups: usize,
    valid_relations: usize,
    median_overlap: f64,
) -> bool {
    total_groups > 1
        && connected_groups == total_groups
        && valid_relations + 1 >= total_groups
        && median_overlap >= 0.25
}

/// Build capture-order candidates from solved station centres.  This is used
/// for long scans when the import order and filenames are arbitrary.  The
/// station grid is inferred from the measured x/y centres; filenames only
/// break exact content ties inside one station.
fn focus_geometry_capture_order(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<Vec<usize>> {
    let (groups, _) = focus_capture_groups(images, matches, homographies)?;
    if groups.len() < 2 {
        return None;
    }
    let centers = groups
        .iter()
        .map(|group| {
            let image = &images[group.anchor];
            let center = Point2::new(image.width as f64 * 0.5, image.height as f64 * 0.5);
            homographies
                .get(&image.id)
                .and_then(|pose| transformed_point(pose, center))
        })
        .collect::<Option<Vec<_>>>()?;

    let mut x_sorted = (0..groups.len()).collect::<Vec<_>>();
    x_sorted.sort_by(|left, right| {
        centers[*right]
            .x
            .total_cmp(&centers[*left].x)
            .then_with(|| centers[*left].y.total_cmp(&centers[*right].y))
    });
    let mut x_gaps = x_sorted
        .windows(2)
        .map(|pair| (centers[pair[0]].x - centers[pair[1]].x).abs())
        .filter(|gap| gap.is_finite() && *gap > 0.0)
        .collect::<Vec<_>>();
    x_gaps.sort_by(f64::total_cmp);
    let x_threshold = if x_gaps.len() >= 2 {
        let split = x_gaps
            .windows(2)
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                (right[1] / right[0].max(1.0)).total_cmp(&(left[1] / left[0].max(1.0)))
            })
            .map(|(index, _)| index);
        split
            .map(|index| (x_gaps[index] + x_gaps[index + 1]) * 0.5)
            .unwrap_or_else(|| x_gaps[x_gaps.len() / 2] * 2.0)
    } else {
        f64::INFINITY
    };
    let mut columns = Vec::<Vec<usize>>::new();
    for group_index in x_sorted.iter().copied() {
        let starts_new_column = columns.last().is_some_and(|column| {
            let previous = column.last().copied().unwrap_or(group_index);
            (centers[previous].x - centers[group_index].x).abs() > x_threshold
        });
        if starts_new_column {
            columns.push(Vec::new());
        }
        if columns.is_empty() {
            columns.push(Vec::new());
        }
        columns
            .last_mut()
            .expect("column was inserted")
            .push(group_index);
    }
    for column in &mut columns {
        column.sort_by(|left, right| {
            centers[*left]
                .y
                .total_cmp(&centers[*right].y)
                .then_with(|| centers[*right].x.total_cmp(&centers[*left].x))
        });
    }

    let mut group_orders = Vec::<Vec<usize>>::new();
    let mut append_columns = |reverse_columns: bool, reverse_rows: bool, alternate: bool| {
        let mut order = Vec::with_capacity(groups.len());
        let column_indices = if reverse_columns {
            (0..columns.len()).rev().collect::<Vec<_>>()
        } else {
            (0..columns.len()).collect::<Vec<_>>()
        };
        for (column_position, column_index) in column_indices.into_iter().enumerate() {
            let reverse = if alternate {
                reverse_rows ^ (column_position % 2 == 1)
            } else {
                reverse_rows
            };
            if reverse {
                order.extend(columns[column_index].iter().rev().copied());
            } else {
                order.extend(columns[column_index].iter().copied());
            }
        }
        group_orders.push(order);
    };
    append_columns(true, false, false);
    append_columns(true, true, false);
    append_columns(true, false, true);
    append_columns(true, true, true);
    append_columns(false, false, false);
    append_columns(false, true, false);

    // Keep the capture-order path as one candidate. It is useful when the
    // camera sequence is genuinely encoded in names, but it is scored by
    // measured station geometry below and therefore cannot fabricate a path
    // for renamed/shuffled imports.
    let mut filename_groups = (0..groups.len()).collect::<Vec<_>>();
    filename_groups.sort_by(|left, right| {
        let left_name = groups[*left]
            .members
            .iter()
            .map(|&index| images[index].filename.as_str())
            .min_by(|left, right| natural_path_cmp(left, right))
            .unwrap_or("");
        let right_name = groups[*right]
            .members
            .iter()
            .map(|&index| images[index].filename.as_str())
            .min_by(|left, right| natural_path_cmp(left, right))
            .unwrap_or("");
        natural_path_cmp(left_name, right_name)
    });
    group_orders.push(filename_groups);

    let mut nearest = vec![x_sorted.first().copied().unwrap_or(0)];
    let mut remaining = (0..groups.len()).collect::<HashSet<_>>();
    remaining.remove(&nearest[0]);
    while !remaining.is_empty() {
        let current = *nearest.last().expect("nearest path has a seed");
        let next = remaining
            .iter()
            .copied()
            .min_by(|left, right| {
                let left_distance = (centers[*left] - centers[current]).norm_squared();
                let right_distance = (centers[*right] - centers[current]).norm_squared();
                left_distance
                    .total_cmp(&right_distance)
                    .then_with(|| left.cmp(right))
            })
            .expect("remaining station exists");
        nearest.push(next);
        remaining.remove(&next);
    }
    group_orders.push(nearest);

    let mut best_order = None::<(f64, usize, f64, f64, Vec<usize>)>;
    for group_order in group_orders {
        let center_distance = group_order
            .windows(2)
            .map(|pair| (centers[pair[0]] - centers[pair[1]]).norm())
            .sum::<f64>();
        let mut expanded = Vec::with_capacity(images.len());
        for group_index in group_order {
            let mut members = groups[group_index].members.clone();
            members.sort_by(|left, right| {
                focus_image_direction_key(&images[*left])
                    .cmp(&focus_image_direction_key(&images[*right]))
            });
            expanded.extend(members);
        }
        if expanded.len() != images.len() {
            continue;
        }
        let (matched_edges, score) = focus_auto_order_path_score(
            &expanded,
            images,
            matches,
            focus_auto_order_motion_scale(images, matches),
        );
        let motion = focus_auto_order_path_motion_cost(&expanded, images, matches);
        if best_order.as_ref().is_none_or(
            |(best_distance, best_edges, best_score, best_motion, _)| {
                center_distance < *best_distance * 0.97
                    || (center_distance <= *best_distance * 1.03
                        && (score > *best_score
                            || (score.total_cmp(best_score).is_eq()
                                && (matched_edges > *best_edges
                                    || (matched_edges == *best_edges && motion < *best_motion)))))
            },
        ) {
            best_order = Some((center_distance, matched_edges, score, motion, expanded));
        }
    }
    best_order.map(|(_, _, _, _, order)| order)
}

fn focus_geometry_group_compositor_order(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    homographies: &HashMap<usize, Matrix3<f64>>,
    groups: &[FocusCaptureGroup],
) -> Vec<usize> {
    let mut image_to_group = vec![usize::MAX; images.len()];
    for (group_index, group) in groups.iter().enumerate() {
        for &image_index in &group.members {
            if image_index < image_to_group.len() {
                image_to_group[image_index] = group_index;
            }
        }
    }
    let mut order = Vec::with_capacity(groups.len());
    let mut seen = vec![false; groups.len()];
    if let Some(source_order) = focus_geometry_capture_order(images, matches, homographies) {
        for image_index in source_order {
            let Some(&group_index) = image_to_group.get(image_index) else {
                continue;
            };
            if group_index < groups.len() && !seen[group_index] {
                seen[group_index] = true;
                order.push(group_index);
            }
        }
    }
    // The geometry path should cover every group. Keep a deterministic
    // geometric fallback for a partially solved component rather than
    // reintroducing filename order into the seam accumulator.
    if order.len() != groups.len() {
        let mut remaining = (0..groups.len())
            .filter(|&index| !seen[index])
            .collect::<Vec<_>>();
        remaining.sort_by(|left, right| {
            let left_anchor = &images[groups[*left].anchor];
            let right_anchor = &images[groups[*right].anchor];
            let center = |image: &ImageInfo| {
                homographies
                    .get(&image.id)
                    .and_then(|pose| {
                        transformed_point(
                            pose,
                            Point2::new(image.width as f64 * 0.5, image.height as f64 * 0.5),
                        )
                    })
                    .unwrap_or(Point2::new(f64::INFINITY, f64::INFINITY))
            };
            let left_center = center(left_anchor);
            let right_center = center(right_anchor);
            left_center
                .y
                .total_cmp(&right_center.y)
                .then_with(|| left_center.x.total_cmp(&right_center.x))
        });
        order.extend(remaining);
    }
    order
}

/// Decide the order in which Capture_Station virtual tiles enter the default
/// layered virtual-tile compositor.
///
/// The compositor places the first tile as the base and resolves every later
/// tile against the canvas built so far, so this order is part of the rendered
/// result: change it and the output pixels change.  It therefore has to be a
/// function of the measured geometry only (需求 14.6).
///
/// The previous implementation collapsed the *source* stitching path
/// (`ordered_indices`) to station ids.  That path is scored against image
/// evidence, but its orientation and several of its tie-breaks are filename
/// derived — `canonical_focus_order_direction` decides which end of a candidate
/// path comes first by comparing the first and last filename with
/// `natural_path_cmp`, and the greedy/graph builders break equal weights on the
/// import index.  Renaming the sources or importing them in reverse could
/// therefore reverse the whole accumulation even though every measured match was
/// identical, which is exactly what 需求 1.2/1.3/5.8 forbid: filename order may
/// only produce *candidate* orders and break exact ties, never stand in for
/// geometry.
///
/// Stations are ordered by the production `StationTopology`: row index
/// ascending (top to bottom), then column index ascending (right to left,
/// because topology inference numbers columns by descending world `x`). This
/// makes the scan convention explicit at the seam accumulator instead of
/// approximating it with continuous projected centres.
///
/// Requirement 5.1 makes `(row, column)` unique for every inferred station, so
/// filenames, source paths and import ranks never participate. The station
/// index is consulted only as the deterministic final key for an exact
/// topology-coordinate tie, which can occur only in malformed or synthetic
/// input; production topology remains rename/import-order invariant.
fn focus_source_order_group_compositor_order(
    station_count: usize,
    station_topology: &topology::StationTopology,
) -> Vec<usize> {
    debug_assert_eq!(
        station_topology.row.len(),
        station_count,
        "topology must contain one row index per station"
    );
    debug_assert_eq!(
        station_topology.column.len(),
        station_count,
        "topology must contain one column index per station"
    );

    let topology_key = |station: usize| {
        (
            station_topology
                .row
                .get(station)
                .copied()
                .unwrap_or(u32::MAX),
            station_topology
                .column
                .get(station)
                .copied()
                .unwrap_or(u32::MAX),
        )
    };
    let mut order = (0..station_count).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        let left_topology = topology_key(left);
        let right_topology = topology_key(right);
        left_topology
            // Row indices increase from top to bottom. Column indices increase
            // from right to left, so their ascending lexicographic order is the
            // required scan convention inferred by Capture_Topology_Model.
            .cmp(&right_topology)
            // Requirement 5.1 makes topology coordinates unique in production.
            // Keep only this content-derived station index as a deterministic
            // final key for malformed/ambiguous exact topology ties; filenames,
            // import order and continuous world coordinates never participate.
            .then_with(|| left.cmp(&right))
    });
    order
}

fn focus_rescue_match_is_verified(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
) -> bool {
    if match_info.sequence_bridge
        || match_info.coarse_bridge
        || match_info.points.len() < FOCUS_MODEL_MIN_INLIERS
    {
        return false;
    }
    let scale = match_info.homography[(0, 0)].abs();
    let overlap = panorama_transform_overlap_support(
        &match_info.homography,
        source.dimensions(),
        target.dimensions(),
    );
    let Some((intensity_ncc, edge_ncc, edge_orientation, samples)) =
        focus_overlap_quality(source, target, &match_info.homography)
    else {
        return false;
    };
    scale.is_finite()
        && (0.65..=1.45).contains(&scale)
        && overlap >= 0.18
        && samples >= 180
        && intensity_ncc >= 0.45
        && edge_ncc >= -0.05
        && edge_orientation >= -0.05
}

fn focus_capture_geometry_passes(
    images: &[ImageInfo],
    valid_edges: usize,
    total_edges: usize,
    median_overlap: f64,
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> bool {
    if total_edges == 0
        // Filename neighbours include transitions between focus brackets and
        // camera positions. Keep this strict: a merely connected group graph
        // can still fold a long scroll into a visually invalid 2D collage.
        // Consecutive files can straddle capture stations and opposite focal
        // planes, so direct pixel correlation is not expected at every group
        // boundary. Group geometry has already required independent
        // multi-layer consensus and a connected robust closure solve.
        || valid_edges.saturating_mul(100) < total_edges.saturating_mul(85)
        || median_overlap < 0.25
    {
        return false;
    }
    let mut scales = images
        .iter()
        .filter_map(|image| {
            homographies
                .get(&image.id)
                .map(panorama_model_linear_characteristics)
                .map(|(scale, _, _)| scale)
                .filter(|scale| scale.is_finite())
        })
        .collect::<Vec<_>>();
    if scales.is_empty() {
        return false;
    }
    let min_scale = scales.iter().copied().fold(f64::INFINITY, f64::min);
    let max_scale = scales.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let Some(median_scale) = median_value(&mut scales) else {
        return false;
    };
    // Absolute scale depends on which source becomes the graph reference and
    // legitimately accumulates when a long scan is photographed from several
    // distances. Cross-station pixel agreement and graph connectivity are
    // validated separately. Keep this as an outlier guard, not as an
    // assumption that every camera station has nearly identical magnification.
    // A 25% scale outlier is large enough to indicate a folded or mixed-lens
    // chain rather than ordinary focus breathing. Long scans can accumulate
    // scale in both directions, so a low-side excursion is accepted when the
    // same scan also has corroborating high-side spread. A lone low-side
    // excursion remains rejected (the synthetic 0.78 drift case exercises
    // that guard).
    let minimum_allowed = median_scale * 0.75;
    let maximum_allowed = median_scale * 1.30;
    let one_sided_low_drift = min_scale < median_scale * 0.80 && max_scale < median_scale * 1.10;
    let scale_ok =
        min_scale >= minimum_allowed && max_scale <= maximum_allowed && !one_sided_low_drift;
    if !scale_ok {
        println!(
            "  - Focus geometry scale range rejected: {min_scale:.3}..{max_scale:.3} around median {median_scale:.3} (expected {minimum_allowed:.3}..{maximum_allowed:.3})"
        );
    }
    scale_ok
}

fn focus_station_representative(station: &[usize], images: &[ImageInfo]) -> usize {
    station
        .iter()
        .copied()
        .max_by_key(|&index| {
            (
                focus_alignment_support(&images[index]),
                std::cmp::Reverse(trailing_capture_number(&images[index].filename).unwrap_or(0)),
            )
        })
        .unwrap_or(station[0])
}

fn focus_indices_share_component(left: usize, right: usize, components: &[Vec<usize>]) -> bool {
    components
        .iter()
        .any(|component| component.contains(&left) && component.contains(&right))
}

fn insert_recovered_focus_match(
    images: &[ImageInfo],
    left_index: usize,
    right_index: usize,
    log_match: bool,
    orphan_attach: bool,
) -> Option<((usize, usize), MatchInfo)> {
    let (source, target, invert_for_storage) =
        canonical_match_direction(images, left_index, right_index);
    let mut match_info = recover_adjacent_focus_position_match(
        &images[source],
        &images[target],
        log_match,
        orphan_attach,
    )?;
    // Record the fit before it is rotated into the key orientation; see
    // `MatchInfo::canonical_homography` (需求 14.6).
    match_info.canonical_homography = Some(match_info.homography);
    if invert_for_storage {
        match_info.homography = match_info.homography.try_inverse()?;
        match_info.points = match_info
            .points
            .into_iter()
            .map(|(source, target)| (target, source))
            .collect();
        match_info.candidate_points = match_info
            .candidate_points
            .into_iter()
            .map(|(source, target)| (target, source))
            .collect();
    }
    Some((
        (left_index.min(right_index), left_index.max(right_index)),
        match_info,
    ))
}

fn attach_focus_orphan_components(
    images: &[ImageInfo],
    matches: &mut HashMap<(usize, usize), MatchInfo>,
) -> usize {
    let components = focus_overlap_components(images, matches);
    if components.len() <= 1 {
        return 0;
    }
    let main_index = components
        .iter()
        .enumerate()
        .max_by_key(|(_, component)| component.len())
        .map(|(index, _)| index)
        .unwrap_or(0);
    let main_component = &components[main_index];
    let mut added = 0usize;
    for (orphan_index, orphan) in components.iter().enumerate() {
        if orphan_index == main_index {
            continue;
        }
        let mut candidates = Vec::new();
        for &left in main_component {
            for &right in orphan {
                let Some(gap) = trailing_capture_number(&images[left].filename)
                    .zip(trailing_capture_number(&images[right].filename))
                    .map(|(left, right)| left.abs_diff(right))
                else {
                    continue;
                };
                if gap == 0 || gap > 12 {
                    continue;
                }
                candidates.push((
                    gap,
                    std::cmp::Reverse(
                        focus_alignment_support(&images[left])
                            + focus_alignment_support(&images[right]),
                    ),
                    left,
                    right,
                ));
            }
        }
        candidates.sort_by_key(|candidate| (candidate.0, candidate.1, candidate.2, candidate.3));
        for (_, _, left, right) in candidates.into_iter().take(16) {
            let key = (left.min(right), left.max(right));
            if matches.contains_key(&key) {
                continue;
            }
            let Some(recovered) = insert_recovered_focus_match(images, left, right, false, true)
            else {
                continue;
            };
            matches.insert(recovered.0, recovered.1);
            added += 1;
            break;
        }
    }
    added
}

fn recover_adjacent_focus_position_match(
    source: &ImageInfo,
    target: &ImageInfo,
    log_match: bool,
    orphan_attach: bool,
) -> Option<MatchInfo> {
    let capture_gap = trailing_capture_number(&source.filename)
        .zip(trailing_capture_number(&target.filename))
        .map(|(left, right)| left.abs_diff(right))?;
    if capture_gap == 0 || capture_gap > FOCUS_BRACKET_MAX_CAPTURE_GAP {
        return None;
    }
    let source_focal = source
        .focal_length_35mm
        .or(target.focal_length_35mm)
        .unwrap_or(50.0);
    let target_focal = target
        .focal_length_35mm
        .or(source.focal_length_35mm)
        .unwrap_or(source_focal);
    let focal_ratio = (source_focal / target_focal).max(target_focal / source_focal);
    if !focal_ratio.is_finite() || focal_ratio > 1.7 {
        return None;
    }
    let same_focal = focal_ratio <= 1.03;
    let (homography, points, score) = estimate_mixed_focal_coarse_registration(
        source,
        target,
        source_focal,
        target_focal,
        // Spatial neighbours can include opposite ends of a focus bracket.
        // Let the coarse search retain a blurred but structurally consistent
        // candidate; the independent gates below are still stricter than the
        // same-pose recovery profile.
        true,
        log_match,
    )?;
    let scale = homography[(0, 0)].abs();
    let center_motion = focus_match_center_motion_ratio(
        source,
        target,
        &MatchInfo {
            homography,
            inliers: points.len(),
            sequence_bridge: false,
            coarse_bridge: true,
            points: points.clone(),
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            canonical_homography: None,
            native_verified_points: Vec::new(),
        },
    )?;
    let overlap =
        panorama_transform_overlap_support(&homography, source.dimensions(), target.dimensions());
    let (intensity_ncc, edge_ncc, edge_orientation, samples) =
        focus_overlap_quality(source, target, &homography)?;
    // Unlike the same-pose fallback, this edge may join two panorama
    // positions. Require strong independent luminance, edge-magnitude, and
    // edge-orientation agreement over a substantial real overlap. Filename
    // adjacency merely selects a bounded candidate; it is never sufficient
    // evidence by itself.
    if !focus_position_recovery_accepts(
        capture_gap,
        same_focal,
        scale,
        center_motion,
        overlap,
        samples,
        intensity_ncc,
        edge_ncc,
        edge_orientation,
        orphan_attach,
    ) {
        if log_match {
            println!(
                "  - Rejecting focus-position recovery '{}' <-> '{}': scale {scale:.3}, intensity {intensity_ncc:.3}, edges {edge_ncc:.3}/{edge_orientation:.3}, motion {:.1}%, overlap {:.1}%, samples {samples}",
                Path::new(&source.filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                Path::new(&target.filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                center_motion * 100.0,
                overlap * 100.0,
            );
        }
        return None;
    }
    if log_match {
        println!(
            "  - Recovered adjacent focus-position edge: '{}' <-> '{}' (NCC {:.3}, edges {:.3}/{:.3}, motion {:.1}%, overlap {:.1}%)",
            Path::new(&source.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            Path::new(&target.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            score,
            edge_ncc,
            edge_orientation,
            center_motion * 100.0,
            overlap * 100.0,
        );
    }
    Some(MatchInfo {
        homography,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: true,
        points: points.clone(),
        candidate_points: points,
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
        native_verified_points: Vec::new(),
    })
}

fn focus_overlap_components(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Vec<Vec<usize>> {
    let mut dsu = Dsu::new(images.len());
    for (left, right) in sorted_keys(matches) {
        if left < images.len() && right < images.len() {
            dsu.union(left, right);
        }
    }
    // Ordered by representative index so that the component order does not
    // depend on the hash seed (需求 14.6).
    let mut by_root = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..images.len() {
        let root = dsu.find(index);
        by_root.entry(root).or_default().push(index);
    }
    let mut components = by_root.into_values().collect::<Vec<_>>();
    for component in &mut components {
        component.sort_by(|&left, &right| {
            natural_path_cmp(&images[left].filename, &images[right].filename)
                .then_with(|| left.cmp(&right))
        });
    }
    components.sort_by(|left, right| {
        natural_path_cmp(
            &images[*left.first().unwrap_or(&0)].filename,
            &images[*right.first().unwrap_or(&0)].filename,
        )
        // `natural_path_cmp` can rate two distinct names equal; the member index
        // is the stable secondary key (需求 14.6).
        .then_with(|| left.first().cmp(&right.first()))
    });
    components
}

#[derive(Clone)]
struct FocusCoarseComponentCandidate {
    left: usize,
    right: usize,
    match_info: MatchInfo,
    quality: f64,
}

fn coarse_focus_component_candidate(
    images: &[ImageInfo],
    left_index: usize,
    right_index: usize,
) -> Option<FocusCoarseComponentCandidate> {
    let capture_gap = focus_capture_number_gap(&images[left_index], &images[right_index])?;
    if capture_gap == 0 || capture_gap > 12 {
        return None;
    }
    let (source_index, target_index, invert_for_storage) =
        canonical_match_direction(images, left_index, right_index);
    let source = &images[source_index];
    let target = &images[target_index];
    let source_focal = source
        .focal_length_35mm
        .or(target.focal_length_35mm)
        .unwrap_or(50.0);
    let target_focal = target
        .focal_length_35mm
        .or(source.focal_length_35mm)
        .unwrap_or(source_focal);
    let focal_ratio = (source_focal / target_focal).max(target_focal / source_focal);
    if !focal_ratio.is_finite() || focal_ratio > 1.7 {
        return None;
    }
    let (homography, points, coarse_score) = estimate_mixed_focal_coarse_registration(
        source,
        target,
        source_focal,
        target_focal,
        true,
        false,
    )?;
    let provisional = MatchInfo {
        homography,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: true,
        points: points.clone(),
        candidate_points: points.clone(),
        top_candidate_points: points.clone(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
        native_verified_points: Vec::new(),
    };
    let scale = homography[(0, 0)].abs();
    let motion = focus_match_center_motion_ratio(source, target, &provisional)?;
    let overlap =
        panorama_transform_overlap_support(&homography, source.dimensions(), target.dimensions());
    let (intensity_ncc, edge_ncc, edge_orientation, samples) =
        focus_overlap_quality(source, target, &homography)?;
    let same_focal = focal_ratio <= 1.03;
    if !scale.is_finite()
        || (!same_focal
            && !(MIXED_FOCAL_SCALE_MIN_RATIO..=MIXED_FOCAL_SCALE_MAX_RATIO).contains(&scale))
        || (same_focal && !(0.86..=1.14).contains(&scale))
        || motion <= 0.025
        || motion > 0.92
        || overlap < 0.12
        || samples < 120
        || coarse_score < 0.55
        || intensity_ncc < 0.35
        || edge_ncc < -0.10
        || edge_orientation < -0.10
    {
        return None;
    }
    let quality = coarse_score * 0.40
        + intensity_ncc * 0.30
        + edge_ncc.max(0.0) * 0.20
        + edge_orientation.max(0.0) * 0.10;
    let mut match_info = provisional;
    // Record the fit before it is rotated into the key orientation; see
    // `MatchInfo::canonical_homography` (需求 14.6).
    match_info.canonical_homography = Some(match_info.homography);
    if invert_for_storage {
        match_info.homography = match_info.homography.try_inverse()?;
        match_info.points = match_info
            .points
            .into_iter()
            .map(|(source, target)| (target, source))
            .collect();
        match_info.candidate_points = match_info
            .candidate_points
            .into_iter()
            .map(|(source, target)| (target, source))
            .collect();
    }
    Some(FocusCoarseComponentCandidate {
        left: left_index.min(right_index),
        right: left_index.max(right_index),
        match_info,
        quality,
    })
}

fn focus_component_bridge_candidates(component: &[usize], images: &[ImageInfo]) -> Vec<usize> {
    let mut candidates = component.to_vec();
    candidates.sort_by_key(|&index| std::cmp::Reverse(focus_alignment_support(&images[index])));
    candidates.truncate(6);
    // Include both filename boundaries as candidate-order hints. Applying
    // `take(4)` after chaining the iterators only ever selected the first four
    // members, so the tail of a long scan component could never participate in
    // component recovery unless it also happened to be among the six sharpest
    // frames.
    for &index in component
        .iter()
        .take(2)
        .chain(component.iter().rev().take(2))
    {
        if !candidates.contains(&index) {
            candidates.push(index);
        }
    }
    candidates
}

fn recover_focus_component_edges(
    images: &[ImageInfo],
    matches: &mut HashMap<(usize, usize), MatchInfo>,
) -> usize {
    let mut total_added = 0usize;
    for pass in 0..3 {
        let components = focus_overlap_components(images, matches);
        if components.len() <= 1 {
            break;
        }
        let mut pending = Vec::new();
        for left_component_index in 0..components.len() {
            for right_component_index in left_component_index + 1..components.len() {
                let left_candidates =
                    focus_component_bridge_candidates(&components[left_component_index], images);
                let right_candidates =
                    focus_component_bridge_candidates(&components[right_component_index], images);
                let mut candidates = Vec::new();
                for &left in &left_candidates {
                    for &right in &right_candidates {
                        let key = (left.min(right), left.max(right));
                        if matches.contains_key(&key) {
                            continue;
                        }
                        if let Some(candidate) =
                            coarse_focus_component_candidate(images, left, right)
                        {
                            candidates.push(candidate);
                        }
                    }
                }
                if candidates.len() < 2 {
                    continue;
                }
                let dimensions = images[components[left_component_index][0]].dimensions();
                let disagreement_limit = dimensions.0.max(dimensions.1) as f64 * 0.08;
                let mut best: Option<(usize, f64, usize)> = None;
                for (candidate_index, candidate) in candidates.iter().enumerate() {
                    let mut support = 0usize;
                    let mut score = 0.0;
                    let mut left_sources = HashSet::new();
                    let mut right_sources = HashSet::new();
                    for other in &candidates {
                        if focus_transform_disagreement_px(
                            &candidate.match_info.homography,
                            &other.match_info.homography,
                            dimensions,
                        )
                        .is_some_and(|error| error <= disagreement_limit)
                        {
                            support += 1;
                            score += other.quality;
                            left_sources.insert(other.left);
                            right_sources.insert(other.right);
                        }
                    }
                    if left_sources.len() < 2 || right_sources.len() < 2 {
                        continue;
                    }
                    let should_replace =
                        best.as_ref().is_none_or(|(best_support, best_score, _)| {
                            support > *best_support
                                || (support == *best_support && score > *best_score)
                        });
                    if should_replace {
                        best = Some((support, score, candidate_index));
                    }
                }
                let Some((support, score, candidate_index)) = best else {
                    continue;
                };
                if support < 2 {
                    continue;
                }
                let candidate = &candidates[candidate_index];
                println!(
                    "  - Accepted consensus coarse bridge '{}' <-> '{}' ({support} agreeing focus layers, score {score:.3})",
                    Path::new(&images[candidate.left].filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    Path::new(&images[candidate.right].filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                );
                pending.push((
                    (candidate.left, candidate.right),
                    candidate.match_info.clone(),
                ));
            }
        }
        if pending.is_empty() {
            break;
        }
        let added = pending.len();
        matches.extend(pending);
        total_added += added;
        println!(
            "  - Added {added} consensus coarse bridge(s) on retry pass {}",
            pass + 1
        );
    }
    total_added
}

fn collect_dense_focus_region_points(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    if projection != Projection::Planar
        || source_image.alignment_image.width() < 64
        || source_image.alignment_image.height() < 64
        || target_image.alignment_image.width() < 64
        || target_image.alignment_image.height() < 64
    {
        return Vec::new();
    }

    let source_plane = LumaPlane::Gray(&source_image.alignment_image);
    let target_plane = LumaPlane::Gray(&target_image.alignment_image);
    let source_width = source_image.alignment_image.width() as f64;
    let source_height = source_image.alignment_image.height() as f64;
    let target_width = target_image.alignment_image.width() as i32;
    let target_height = target_image.alignment_image.height() as i32;
    let source_scale = source_image.scale_factor;
    let target_scale = target_image.scale_factor;
    let mut points = Vec::new();

    // Use a sparse grid over the complete source frame. This is the generic
    // residual-registration signal: ordinary paper texture and any foreground
    // object can contribute, while the later regional RANSAC decides whether a
    // consistent local model exists. The search radius is intentionally much
    // smaller than the previous foreground-only search so repeated strokes on
    // a long edge cannot jump to a different copy.
    // The grid is deliberately independent of colour, brightness, or a named
    // object. Extra samples near both frame edges let the generic regional
    // registration see structured layers near the frame boundary even when an
    // optional depth-mask detector does not fire.
    let mut source_y_fractions = vec![
        0.03, 0.08, 0.14, 0.20, 0.26, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 0.95,
    ];
    // A narrow depth layer is easy to miss when the fixed grid lands only on
    // its uniform interior. Add samples at both silhouettes and along the
    // interior whenever the capture contains a detected occlusion band. The
    // same path is used for every object; the detector only supplies optional
    // sampling locations and never changes the stitcher's geometry model.
    if let Some((minimum_y, maximum_y)) = source_image.foreground_range {
        let span = (maximum_y - minimum_y).max(0.0);
        source_y_fractions.extend([
            minimum_y,
            minimum_y + span * 0.25,
            minimum_y + span * 0.5,
            minimum_y + span * 0.75,
            maximum_y,
        ]);
        source_y_fractions.sort_unstable_by(f64::total_cmp);
        source_y_fractions.dedup_by(|left, right| (*left - *right).abs() < 0.005);
    }
    let mut sample_rows = source_y_fractions
        .into_iter()
        .map(|fraction| (fraction, 14))
        .collect::<Vec<_>>();
    for &edge_row in &source_image.horizontal_edge_rows {
        if edge_row.is_finite()
            && (0.02..=0.98).contains(&edge_row)
            && sample_rows
                .iter()
                .all(|(row, _)| (*row - edge_row).abs() >= 0.005)
        {
            sample_rows.push((edge_row, FOCUS_HORIZONTAL_EDGE_SEARCH_RADIUS));
        }
    }
    sample_rows.sort_unstable_by(|(left, _), (right, _)| left.total_cmp(right));
    let source_x_fractions = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
    let patch_radius = 8;
    for (source_y_fraction, search_radius) in sample_rows {
        for source_x_fraction in source_x_fractions {
            let source_x = (source_width * source_x_fraction).round() as i32;
            let source_y = (source_height * source_y_fraction).round() as i32;
            if gradient_patch_energy(&source_plane, source_x, source_y, patch_radius) < 1.5 {
                continue;
            }
            let source_full = Point2::new(
                source_x as f64 * source_scale,
                source_y as f64 * source_scale,
            );
            let predicted = *homography * Point3::new(source_full.x, source_full.y, 1.0);
            if predicted.z.abs() < 1e-8 {
                continue;
            }
            let predicted_target_full =
                Point2::new(predicted.x / predicted.z, predicted.y / predicted.z);
            let target_x = (predicted_target_full.x / target_scale).round() as i32;
            let target_y = (predicted_target_full.y / target_scale).round() as i32;
            if source_x < patch_radius
                || source_y < patch_radius
                || source_x + patch_radius >= source_width as i32
                || source_y + patch_radius >= source_height as i32
                || target_x < patch_radius + search_radius
                || target_y < patch_radius + search_radius
                || target_x + patch_radius + search_radius >= target_width
                || target_y + patch_radius + search_radius >= target_height
            {
                continue;
            }
            let Some((best_x, best_y, subpixel_x, subpixel_y)) = refine_foreground_patch_position(
                &source_plane,
                &target_plane,
                source_x,
                source_y,
                target_x,
                target_y,
                patch_radius,
                search_radius,
            ) else {
                continue;
            };
            let score = gradient_patch_ncc(
                &source_plane,
                &target_plane,
                source_x,
                source_y,
                best_x,
                best_y,
                patch_radius,
            );
            let luminance_score = patch_ncc(
                &source_plane,
                &target_plane,
                source_x,
                source_y,
                best_x,
                best_y,
                patch_radius,
            );
            if !score.is_finite()
                || score < FOCUS_DENSE_REGION_MIN_NCC
                || !luminance_score.is_finite()
                || luminance_score < 0.40
                || gradient_patch_energy(&target_plane, best_x, best_y, patch_radius) < 1.5
            {
                continue;
            }
            let target_full = Point2::new(
                (best_x as f64 + subpixel_x) * target_scale,
                (best_y as f64 + subpixel_y) * target_scale,
            );
            points.push((source_full, target_full));
        }
    }
    points
}

fn collect_foreground_feature_points(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let Some((source_minimum_y, source_maximum_y)) = source_image.foreground_range else {
        return Vec::new();
    };
    let Some((target_minimum_y, target_maximum_y)) = target_image.foreground_range else {
        return Vec::new();
    };
    if projection != Projection::Planar {
        return Vec::new();
    }

    let source_plane = LumaPlane::Gray(&source_image.alignment_image);
    let target_plane = LumaPlane::Gray(&target_image.alignment_image);
    let source_width = source_image.alignment_image.width() as i32;
    let source_height = source_image.alignment_image.height() as i32;
    let target_width = target_image.alignment_image.width() as i32;
    let target_height = target_image.alignment_image.height() as i32;
    let patch_radius = FOCUS_FOREGROUND_PATCH_RADIUS;
    let search_radius = FOCUS_FOREGROUND_SEARCH_RADIUS;
    let mut points = Vec::new();

    for feature in &source_image.top_features {
        let source_x = feature.keypoint.x as i32;
        let source_y = feature.keypoint.y as i32;
        if source_x < patch_radius
            || source_y < patch_radius
            || source_x + patch_radius >= source_width - 1
            || source_y + patch_radius >= source_height - 1
        {
            continue;
        }
        let source_y_fraction = source_y as f64 / source_height.max(1) as f64;
        if !(source_minimum_y..=source_maximum_y).contains(&source_y_fraction) {
            continue;
        }
        if gradient_patch_corner_energy(&source_plane, source_x, source_y, patch_radius)
            < FOCUS_FOREGROUND_MIN_CORNER_ENERGY
        {
            continue;
        }
        let source_full = Point2::new(
            source_x as f64 * source_image.scale_factor,
            source_y as f64 * source_image.scale_factor,
        );
        let predicted = *homography * Point3::new(source_full.x, source_full.y, 1.0);
        if predicted.z.abs() < 1e-8 {
            continue;
        }
        let predicted_target_full =
            Point2::new(predicted.x / predicted.z, predicted.y / predicted.z);
        let target_x = (predicted_target_full.x / target_image.scale_factor).round() as i32;
        let target_y = (predicted_target_full.y / target_image.scale_factor).round() as i32;
        if target_x < patch_radius + search_radius
            || target_y < patch_radius + search_radius
            || target_x + patch_radius + search_radius >= target_width
            || target_y + patch_radius + search_radius >= target_height
        {
            continue;
        }
        let Some((best_x, best_y, subpixel_x, subpixel_y)) = refine_foreground_patch_position(
            &source_plane,
            &target_plane,
            source_x,
            source_y,
            target_x,
            target_y,
            patch_radius,
            search_radius,
        ) else {
            continue;
        };
        let score = gradient_patch_ncc(
            &source_plane,
            &target_plane,
            source_x,
            source_y,
            best_x,
            best_y,
            patch_radius,
        );
        let target_y_fraction = (best_y as f64 + subpixel_y) / target_height.max(1) as f64;
        if !score.is_finite()
            || score < FOCUS_DENSE_REGION_MIN_NCC
            || !(target_minimum_y..=target_maximum_y).contains(&target_y_fraction)
            || target_plane.luma_at(best_x, best_y).unwrap_or(0.0) < 110.0
            || gradient_patch_energy(&target_plane, best_x, best_y, patch_radius)
                < FOCUS_FOREGROUND_MIN_GRADIENT_ENERGY
            || gradient_patch_corner_energy(&target_plane, best_x, best_y, patch_radius)
                < FOCUS_FOREGROUND_MIN_CORNER_ENERGY
        {
            continue;
        }
        if points.iter().any(|(previous_source, previous_target)| {
            (previous_source - source_full).norm() < 10.0
                || (previous_target
                    - Point2::new(
                        (best_x as f64 + subpixel_x) * target_image.scale_factor,
                        (best_y as f64 + subpixel_y) * target_image.scale_factor,
                    ))
                .norm()
                    < 10.0
        }) {
            continue;
        }
        points.push((
            source_full,
            Point2::new(
                (best_x as f64 + subpixel_x) * target_image.scale_factor,
                (best_y as f64 + subpixel_y) * target_image.scale_factor,
            ),
        ));
    }
    points
}

fn panorama_transform_is_stable(transform: &Matrix3<f64>, dimensions: (u32, u32)) -> bool {
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    if width <= 1.0 || height <= 1.0 || transform.try_inverse().is_none() {
        return false;
    }
    let corners = [
        Point2::new(0.0, 0.0),
        Point2::new(width, 0.0),
        Point2::new(width, height),
        Point2::new(0.0, height),
    ]
    .into_iter()
    .map(|point| transformed_point(transform, point))
    .collect::<Option<Vec<_>>>();
    let Some(corners) = corners else {
        return false;
    };

    let edge_scales = [
        (corners[1] - corners[0]).norm() / width.max(1.0),
        (corners[2] - corners[1]).norm() / height.max(1.0),
        (corners[2] - corners[3]).norm() / width.max(1.0),
        (corners[3] - corners[0]).norm() / height.max(1.0),
    ];
    if edge_scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale < 0.18 || *scale > 5.5)
    {
        return false;
    }
    let min_scale = edge_scales.iter().copied().fold(f64::INFINITY, f64::min);
    let max_scale = edge_scales
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !min_scale.is_finite() || max_scale / min_scale > 6.0 {
        return false;
    }

    let signed_double_area = corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(4)
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f64>();
    let area_ratio = signed_double_area / (2.0 * width * height);
    area_ratio.is_finite() && area_ratio.abs() >= 0.025 && area_ratio.abs() <= 30.0
}

fn panorama_model_fit(
    transform: Matrix3<f64>,
    points: &[(Point2<f64>, Point2<f64>)],
) -> Option<(Matrix3<f64>, Vec<usize>, f64)> {
    let inliers = symmetric_inlier_indices(&transform, points, PANORAMA_MODEL_INLIER_THRESHOLD);
    if inliers.len() < PANORAMA_MODEL_MIN_INLIERS {
        return None;
    }
    let inlier_points = inliers
        .iter()
        .map(|&index| points[index])
        .collect::<Vec<_>>();
    // One deterministic refit removes the small bias introduced when a model
    // was estimated from a mixed set of descriptor inliers.
    let refitted = if transform[(2, 0)].abs() < 1e-14 && transform[(2, 1)].abs() < 1e-14 {
        // Keep the model family (translation/similarity/affine) selected by the
        // caller; a generic projective refit here would reintroduce drift.
        transform
    } else {
        processing::compute_homography(&inlier_points).unwrap_or(transform)
    };
    let error = median_symmetric_error(&refitted, &inlier_points);
    error.is_finite().then_some((refitted, inliers, error))
}

fn panorama_model_linear_characteristics(transform: &Matrix3<f64>) -> (f64, f64, f64) {
    let a = transform[(0, 0)];
    let b = transform[(0, 1)];
    let c = transform[(1, 0)];
    let d = transform[(1, 1)];
    let scale_x = (a * a + c * c).sqrt();
    let scale_y = (b * b + d * d).sqrt();
    let scale = ((scale_x * scale_y).max(0.0)).sqrt();
    let rotation = 0.5 * (c - b).atan2(a + d);
    let anisotropy = if scale_x.max(scale_y) > f64::EPSILON {
        (scale_x - scale_y).abs() / scale_x.max(scale_y)
    } else {
        f64::INFINITY
    };
    (scale, rotation, anisotropy)
}

fn mixed_focal_scale_is_plausible(
    transform: &Matrix3<f64>,
    source_focal: f64,
    target_focal: f64,
) -> bool {
    if !source_focal.is_finite()
        || !target_focal.is_finite()
        || source_focal <= 0.0
        || target_focal <= 0.0
    {
        return true;
    }
    let expected = target_focal / source_focal;
    let (scale, _, anisotropy) = panorama_model_linear_characteristics(transform);
    if !expected.is_finite() || !scale.is_finite() || anisotropy > 0.35 {
        return false;
    }
    let ratio = scale / expected;
    ratio.is_finite()
        && (MIXED_FOCAL_SCALE_MIN_RATIO..=MIXED_FOCAL_SCALE_MAX_RATIO).contains(&ratio)
}

fn estimate_mixed_focal_coarse_registration(
    source: &ImageInfo,
    target: &ImageInfo,
    source_focal: f64,
    target_focal: f64,
    allow_adjacent_defocus: bool,
    log_match: bool,
) -> Option<(Matrix3<f64>, Vec<(Point2<f64>, Point2<f64>)>, f64)> {
    if !source_focal.is_finite()
        || !target_focal.is_finite()
        || source_focal <= 0.0
        || target_focal <= 0.0
    {
        return None;
    }
    let focal_scale = target_focal / source_focal;
    if !focal_scale.is_finite() || focal_scale <= 0.0 {
        return None;
    }

    // Work at a deliberately small common resolution. The source is resized
    // by the metadata-predicted focal ratio before the translation search, so
    // this is a real coarse registration rather than a same-size image hash.
    // It recovers a close-up inside a wider frame even when BRIEF selects a
    // repeated character as its only cross-lens consensus.
    let target_alignment_width = target.alignment_image.width().max(1) as f64;
    let target_alignment_height = target.alignment_image.height().max(1) as f64;
    let target_low_width = 128u32.min(target.alignment_image.width().max(32));
    let target_low_height = ((target_alignment_height / target_alignment_width)
        * target_low_width as f64)
        .round()
        .clamp(32.0, 128.0) as u32;
    let target_low = image::imageops::resize(
        &target.alignment_image,
        target_low_width,
        target_low_height,
        image::imageops::FilterType::Triangle,
    );
    let target_low_scale = target_low_width as f64 / target_alignment_width;
    // EXIF focal ratio is a useful prior, but a handheld capture can also
    // change camera distance while zooming. Search a bounded scale band around
    // that prior; otherwise a valid close-up in a wide overview is rejected by
    // the fixed-ratio translation search before dense validation sees it.
    let focal_ratio = (source_focal / target_focal).max(target_focal / source_focal);
    let scale_candidates = if focal_ratio <= 1.03 {
        // Same lens module: do not let EXIF noise expand the search to 1.08/1.20
        // scales that mimic a false lens switch on dark, low-texture artwork.
        [0.94_f64, 0.97, 1.0, 1.03, 1.06]
            .into_iter()
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .collect::<Vec<_>>()
    } else {
        [0.70, 0.82, 0.94, 1.0, 1.08, 1.20, 1.34]
            .into_iter()
            .map(|adjustment| focal_scale * adjustment)
            .filter(|scale| scale.is_finite() && *scale > 0.0)
            .collect::<Vec<_>>()
    };
    let mut best: Option<(i32, i32, f64, f64)> = None;
    let mut candidate_scores: Vec<(i32, i32, f64, f64)> = Vec::new();

    for scale in scale_candidates {
        let source_alignment_scale =
            scale * source.scale_factor / target.scale_factor.max(f64::EPSILON);
        let source_low_width =
            (source.alignment_image.width() as f64 * source_alignment_scale * target_low_scale)
                .round()
                .clamp(24.0, 360.0) as u32;
        let source_low_height =
            (source.alignment_image.height() as f64 * source_alignment_scale * target_low_scale)
                .round()
                .clamp(24.0, 280.0) as u32;
        let source_low = image::imageops::resize(
            &source.alignment_image,
            source_low_width,
            source_low_height,
            image::imageops::FilterType::Triangle,
        );
        let minimum_overlap_width = (source_low.width().min(target_low.width()) as f64 * 0.30)
            .round()
            .max(12.0) as u32;
        let minimum_overlap_height = (source_low.height().min(target_low.height()) as f64 * 0.30)
            .round()
            .max(12.0) as u32;
        let x_min = -(source_low.width() as i32 - minimum_overlap_width as i32);
        let x_max = target_low.width() as i32 - minimum_overlap_width as i32;
        let y_min = -(source_low.height() as i32 - minimum_overlap_height as i32);
        let y_max = target_low.height() as i32 - minimum_overlap_height as i32;

        for offset_y in y_min..=y_max {
            for offset_x in x_min..=x_max {
                let source_start_x = offset_x.max(0) as u32;
                let source_start_y = offset_y.max(0) as u32;
                let target_start_x = (-offset_x).max(0) as u32;
                let target_start_y = (-offset_y).max(0) as u32;
                if source_start_x >= source_low.width()
                    || source_start_y >= source_low.height()
                    || target_start_x >= target_low.width()
                    || target_start_y >= target_low.height()
                {
                    continue;
                }
                let overlap_width =
                    (source_low.width() - source_start_x).min(target_low.width() - target_start_x);
                let overlap_height = (source_low.height() - source_start_y)
                    .min(target_low.height() - target_start_y);
                if overlap_width < minimum_overlap_width || overlap_height < minimum_overlap_height
                {
                    continue;
                }
                let stride = 2u32;
                let mut values = Vec::new();
                let mut target_values = Vec::new();
                for y in (0..overlap_height).step_by(stride as usize) {
                    for x in (0..overlap_width).step_by(stride as usize) {
                        values.push(f64::from(
                            source_low.get_pixel(source_start_x + x, source_start_y + y)[0],
                        ));
                        target_values.push(f64::from(
                            target_low.get_pixel(target_start_x + x, target_start_y + y)[0],
                        ));
                    }
                }
                if values.len() < 120 {
                    continue;
                }
                let mean_source = values.iter().sum::<f64>() / values.len() as f64;
                let mean_target = target_values.iter().sum::<f64>() / values.len() as f64;
                let mut covariance = 0.0;
                let mut source_variance = 0.0;
                let mut target_variance = 0.0;
                for (source_value, target_value) in values.iter().zip(target_values.iter()) {
                    let source_delta = source_value - mean_source;
                    let target_delta = target_value - mean_target;
                    covariance += source_delta * target_delta;
                    source_variance += source_delta * source_delta;
                    target_variance += target_delta * target_delta;
                }
                if source_variance <= f64::EPSILON || target_variance <= f64::EPSILON {
                    continue;
                }
                let score = covariance / (source_variance * target_variance).sqrt();
                if !score.is_finite() {
                    continue;
                }
                candidate_scores.push((offset_x, offset_y, scale, score));
                if best
                    .as_ref()
                    .is_none_or(|(_, _, _, best_score)| score > *best_score)
                {
                    best = Some((offset_x, offset_y, scale, score));
                }
            }
        }
    }

    let (best_offset_x, best_offset_y, best_scale, best_score) = best?;
    let alternate = candidate_scores
        .iter()
        .filter(|(candidate_x, candidate_y, candidate_scale, _)| {
            (*candidate_x - best_offset_x).abs() >= 8
                || (*candidate_y - best_offset_y).abs() >= 8
                || (*candidate_scale - best_scale).abs() >= best_scale * 0.08
        })
        .map(|(_, _, _, candidate_score)| *candidate_score)
        .fold(f64::NEG_INFINITY, f64::max);
    if log_match {
        println!(
            "  - Mixed-focal coarse search best offset=({}, {}), NCC {:.3}, margin {:.3}",
            best_offset_x,
            best_offset_y,
            best_score,
            best_score - alternate
        );
    }
    if best_score < 0.40 {
        return None;
    }
    // The best low-resolution luminance peak is not necessarily the best
    // geometric peak: a repeated stroke can win the cheap search while a
    // weaker peak is the actual shared patch. Re-score a small set of
    // spatially separated peaks against the denser source/target images.
    let mut ranked_candidates = candidate_scores;
    ranked_candidates.sort_by(|left, right| right.3.total_cmp(&left.3));
    let mut selected = None;
    let mut best_overlap = None;
    let mut tried: Vec<(i32, i32, f64)> = Vec::new();
    for (offset_x, offset_y, scale, score) in ranked_candidates {
        if score < (best_score - 0.16).max(0.40) || tried.len() >= 16 {
            break;
        }
        if tried
            .iter()
            .any(|(previous_x, previous_y, previous_scale)| {
                (offset_x - *previous_x).abs() < 8
                    && (offset_y - *previous_y).abs() < 8
                    && (scale - *previous_scale).abs() < scale * 0.08
            })
        {
            continue;
        }
        tried.push((offset_x, offset_y, scale));
        // `offset_*` above is expressed as the crop displacement applied to
        // the scaled source. A positive source crop aligns with target zero,
        // so the source-to-target transform has the opposite translation.
        // Using the same sign validates a mirrored search pose and was the
        // reason genuine overview/close-up bridges never survived the dense
        // overlap check.
        let tx_alignment = -offset_x as f64 / target_low_scale;
        let ty_alignment = -offset_y as f64 / target_low_scale;
        let tx_full = tx_alignment * target.scale_factor;
        let ty_full = ty_alignment * target.scale_factor;
        let homography = Matrix3::new(scale, 0.0, tx_full, 0.0, scale, ty_full, 0.0, 0.0, 1.0);
        let Some((intensity_ncc, edge_ncc, edge_orientation, samples)) =
            focus_overlap_quality(source, target, &homography)
        else {
            continue;
        };
        let validation_score = intensity_ncc * 0.45 + edge_ncc * 0.35 + edge_orientation * 0.20;
        if best_overlap
            .as_ref()
            .is_none_or(|(_, _, _, _, _, best_validation)| validation_score > *best_validation)
        {
            best_overlap = Some((
                offset_x,
                offset_y,
                intensity_ncc,
                edge_ncc,
                edge_orientation,
                validation_score,
            ));
        }
        let structurally_verified = if allow_adjacent_defocus {
            // A heavily defocused bracket endpoint preserves coarse luminance
            // and edge magnitude but not reliable gradient orientation. This
            // relaxed profile is reachable only for adjacent, same-focal
            // captures at the caller and still requires both image channels.
            (score >= 0.60
                && intensity_ncc >= 0.30
                && edge_ncc >= -0.10
                && edge_orientation >= -0.10)
                || (intensity_ncc >= 0.60 && edge_ncc >= 0.20 && edge_orientation >= 0.0)
                || (intensity_ncc >= 0.85 && edge_ncc >= 0.05 && edge_orientation >= 0.10)
        } else {
            intensity_ncc >= MIXED_FOCAL_COARSE_MIN_INTENSITY_NCC
                && edge_ncc >= MIXED_FOCAL_COARSE_MIN_EDGE_NCC
                && edge_orientation >= MIXED_FOCAL_COARSE_MIN_EDGE_ORIENTATION
        };
        if !structurally_verified {
            continue;
        }
        if selected
            .as_ref()
            .is_none_or(|(_, _, _, _, _, _, _, _, _, best_validation)| {
                validation_score > *best_validation
            })
        {
            selected = Some((
                offset_x,
                offset_y,
                scale,
                score,
                homography,
                intensity_ncc,
                edge_ncc,
                edge_orientation,
                samples,
                validation_score,
            ));
        }
    }
    let Some((_, _, _, score, homography, intensity_ncc, edge_ncc, edge_orientation, samples, _)) =
        selected
    else {
        if log_match {
            if let Some((offset_x, offset_y, intensity_ncc, edge_ncc, edge_orientation, _)) =
                best_overlap
            {
                println!(
                    "  - Rejecting mixed-focal coarse bridge: best structural candidate at ({offset_x}, {offset_y}) had intensity NCC {intensity_ncc:.3}, edge NCC {edge_ncc:.3}, edge orientation {edge_orientation:.3}"
                );
            } else {
                println!(
                    "  - Rejecting mixed-focal coarse bridge: overlap structure is not verified"
                );
            }
        }
        return None;
    };
    if log_match {
        println!(
            "  - Mixed-focal coarse overlap validation: intensity NCC {:.3}, edge NCC {:.3}, edge orientation {:.3} ({} samples)",
            intensity_ncc, edge_ncc, edge_orientation, samples
        );
    }
    let mut points = Vec::new();
    for row in 1..=6 {
        for column in 1..=8 {
            let source_point = Point2::new(
                source.width as f64 * column as f64 / 9.0,
                source.height as f64 * row as f64 / 7.0,
            );
            let Some(target_point) = transformed_point(&homography, source_point) else {
                continue;
            };
            if target_point.x >= 0.0
                && target_point.y >= 0.0
                && target_point.x < target.width as f64
                && target_point.y < target.height as f64
            {
                points.push((source_point, target_point));
            }
        }
    }
    // A narrow same-focal scan strip can cover only a handful of the coarse
    // grid cells.  It is still independently validated by the dense
    // luminance/edge checks above; do not discard it solely because the
    // synthetic grid has fewer than the cross-lens minimum correspondences.
    let minimum_points = if allow_adjacent_defocus && focal_ratio <= 1.03 {
        // A very narrow but well-correlated bracket boundary may expose no
        // coarse grid cell after the transform is scaled back to the source
        // dimensions. The dense overlap validator remains the evidence gate;
        // these points are optional and only carry the recovered placement.
        0
    } else {
        MIXED_FOCAL_MIN_INLIERS
    };
    (points.len() >= minimum_points).then_some((homography, points, score))
}

/// Validate a metadata-scaled translation against the actual alignment images.
/// The low-resolution search above is intentionally cheap, but repeated
/// calligraphy can make a wrong translation look unique. Sampling the proposed
/// overlap at a denser grid and comparing both luminance and edge structure
/// catches that failure before synthetic points can be promoted to a match.
#[derive(Clone, Copy, Debug)]
struct FocusOverlapQuality {
    intensity_ncc: f64,
    edge_ncc: f64,
    mean_edge_orientation_cosine: f64,
    samples: usize,
    low_frequency_mean_relative_difference: f64,
    edge_strength_ratio: f64,
    median_edge_orientation_difference_degrees: f64,
    exposure_gain: f64,
    exposure_normalized_difference: f64,
    exposure_normalized_mean_relative_difference: f64,
}

impl FocusOverlapQuality {
    fn legacy_tuple(self) -> (f64, f64, f64, usize) {
        (
            self.intensity_ncc,
            self.edge_ncc,
            self.mean_edge_orientation_cosine,
            self.samples,
        )
    }
}

fn focus_overlap_quality_measurements(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
) -> Option<FocusOverlapQuality> {
    focus_overlap_quality_measurements_on_coverage(source, target, source_to_target, None)
}

/// [`focus_overlap_quality_measurements`] on the pixels both analysis planes
/// cover. `coverages` holds the `(source, target)` Coverage_Masks in
/// `alignment_image` pixels; a sample counts only when every pixel it reads,
/// directly or through the plane's blur, is covered in both (需求 6.1, 6.8:
/// uncovered payload is not overlap). `None` samples every in-bounds position.
fn focus_overlap_quality_measurements_on_coverage(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
    coverages: Option<(&stitching::CoverageMask, &stitching::CoverageMask)>,
) -> Option<FocusOverlapQuality> {
    let inverse = source_to_target.try_inverse()?;
    let target_width = target.alignment_image.width();
    let target_height = target.alignment_image.height();
    if target_width < 32 || target_height < 32 {
        return None;
    }
    if coverages.is_some_and(|(source_coverage, target_coverage)| {
        source_coverage.dimensions() != source.alignment_image.dimensions()
            || target_coverage.dimensions() != target.alignment_image.dimensions()
    }) {
        return None;
    }
    let sample_is_covered = |coverage: &stitching::CoverageMask, x: f64, y: f64| {
        let radius = VIRTUAL_TILE_QUALITY_COVERAGE_RADIUS_PX;
        virtual_tile_coverage_contains_rectangle(
            coverage,
            x - radius,
            y - radius,
            x + radius,
            y + radius,
        )
    };
    let columns = 64u32;
    let rows = 40u32;
    let mut source_values = Vec::new();
    let mut target_values = Vec::new();
    let mut source_edges = Vec::new();
    let mut target_edges = Vec::new();
    let mut edge_orientation_cosines = Vec::new();
    let mut edge_orientation_differences = Vec::new();
    for row in 2..rows.saturating_sub(2) {
        let target_y = (row as f64 + 0.5) * target_height as f64 / rows as f64;
        for column in 2..columns.saturating_sub(2) {
            let target_x = (column as f64 + 0.5) * target_width as f64 / columns as f64;
            let target_full = Point2::new(
                target_x * target.scale_factor,
                target_y * target.scale_factor,
            );
            let Some(source_full) = transformed_point(&inverse, target_full) else {
                continue;
            };
            let source_x = source_full.x / source.scale_factor.max(f64::EPSILON);
            let source_y = source_full.y / source.scale_factor.max(f64::EPSILON);
            if let Some((source_coverage, target_coverage)) = coverages
                && !(sample_is_covered(target_coverage, target_x, target_y)
                    && sample_is_covered(source_coverage, source_x, source_y))
            {
                continue;
            }
            let Some(target_value) =
                registration::sample_gray(&target.alignment_image, target_x, target_y)
            else {
                continue;
            };
            let Some(source_value) =
                registration::sample_gray(&source.alignment_image, source_x, source_y)
            else {
                continue;
            };
            let Some(target_dx_left) =
                registration::sample_gray(&target.alignment_image, target_x - 1.0, target_y)
            else {
                continue;
            };
            let Some(target_dx_right) =
                registration::sample_gray(&target.alignment_image, target_x + 1.0, target_y)
            else {
                continue;
            };
            let Some(target_dy_up) =
                registration::sample_gray(&target.alignment_image, target_x, target_y - 1.0)
            else {
                continue;
            };
            let Some(target_dy_down) =
                registration::sample_gray(&target.alignment_image, target_x, target_y + 1.0)
            else {
                continue;
            };
            let Some(source_dx_left) =
                registration::sample_gray(&source.alignment_image, source_x - 1.0, source_y)
            else {
                continue;
            };
            let Some(source_dx_right) =
                registration::sample_gray(&source.alignment_image, source_x + 1.0, source_y)
            else {
                continue;
            };
            let Some(source_dy_up) =
                registration::sample_gray(&source.alignment_image, source_x, source_y - 1.0)
            else {
                continue;
            };
            let Some(source_dy_down) =
                registration::sample_gray(&source.alignment_image, source_x, source_y + 1.0)
            else {
                continue;
            };
            // Gradients are covectors: compare them in the target pixel grid,
            // not in the two independently rotated/scaled source grids. The
            // patch matcher already applies this inverse-transpose Jacobian;
            // the dense acceptance check must use the same convention.
            let Some(gradient_transform) =
                virtual_tile_local_jacobian(source_to_target, source_full)
                    .and_then(|jacobian| jacobian.try_inverse())
                    .map(|inverse| inverse.transpose())
            else {
                continue;
            };
            let source_gradient = gradient_transform
                * nalgebra::Vector2::new(
                    source_dx_right - source_dx_left,
                    source_dy_down - source_dy_up,
                )
                * (target.scale_factor / source.scale_factor.max(f64::EPSILON));
            let source_dx = source_gradient.x;
            let source_dy = source_gradient.y;
            let target_dx = target_dx_right - target_dx_left;
            let target_dy = target_dy_down - target_dy_up;
            let source_edge = source_dx.hypot(source_dy);
            let target_edge = target_dx.hypot(target_dy);
            if source_edge.is_finite() && target_edge.is_finite() {
                source_edges.push(source_edge);
                target_edges.push(target_edge);
                if source_edge > 4.0 && target_edge > 4.0 {
                    let cosine = ((source_dx * target_dx + source_dy * target_dy)
                        / (source_edge * target_edge))
                        .clamp(-1.0, 1.0);
                    edge_orientation_cosines.push(cosine);
                    // Edge orientation is axial rather than directional: a
                    // 180-degree gradient reversal describes the same edge.
                    edge_orientation_differences.push(cosine.abs().acos().to_degrees());
                }
            }
            source_values.push(source_value);
            target_values.push(target_value);
        }
    }
    if source_values.len() < 120 || source_edges.len() != source_values.len() {
        return None;
    }
    let intensity_ncc = normalized_correlation(&source_values, &target_values)?;
    let edge_ncc = normalized_correlation(&source_edges, &target_edges)?;
    let mean_edge_orientation_cosine = if edge_orientation_cosines.is_empty() {
        0.0
    } else {
        edge_orientation_cosines.iter().sum::<f64>() / edge_orientation_cosines.len() as f64
    };
    let source_mean = source_values.iter().sum::<f64>() / source_values.len() as f64;
    let target_mean = target_values.iter().sum::<f64>() / target_values.len() as f64;
    // A mean is the zero-frequency component of the sampled overlap. Dividing
    // by the larger magnitude makes the comparison symmetric and bounded.
    let low_frequency_mean_relative_difference = (source_mean - target_mean).abs()
        / source_mean.abs().max(target_mean.abs()).max(f64::EPSILON);
    // Estimate a scalar exposure gain on alternating overlap samples and
    // validate it on the held-out samples. Unlike matching the two means,
    // this residual cannot be made zero for arbitrary different images.
    let mut exposure_ratios = source_values
        .iter()
        .zip(&target_values)
        .enumerate()
        .filter_map(|(index, (&source, &target))| {
            (index % 2 == 0 && source > 8.0 && target > 8.0).then_some(target / source)
        })
        .collect::<Vec<_>>();
    let exposure_gain = median_value(&mut exposure_ratios).unwrap_or(f64::NAN);
    let mut exposure_residual = 0.0;
    let mut exposure_magnitude = 0.0;
    let mut predicted_sum = 0.0;
    let mut held_out_target_sum = 0.0;
    for (index, (&source, &target)) in source_values.iter().zip(&target_values).enumerate() {
        if index % 2 == 1 {
            let predicted = source * exposure_gain;
            exposure_residual += (predicted - target).abs();
            exposure_magnitude += predicted.abs().max(target.abs());
            predicted_sum += predicted;
            held_out_target_sum += target;
        }
    }
    let exposure_normalized_difference = exposure_residual / exposure_magnitude.max(f64::EPSILON);
    let exposure_normalized_mean_relative_difference = (predicted_sum - held_out_target_sum).abs()
        / predicted_sum
            .abs()
            .max(held_out_target_sum.abs())
            .max(f64::EPSILON);
    let source_edge_mean = source_edges.iter().sum::<f64>() / source_edges.len() as f64;
    let target_edge_mean = target_edges.iter().sum::<f64>() / target_edges.len() as f64;
    let edge_strength_ratio = if target_edge_mean > f64::EPSILON {
        source_edge_mean / target_edge_mean
    } else if source_edge_mean <= f64::EPSILON {
        1.0
    } else {
        f64::INFINITY
    };
    let median_edge_orientation_difference_degrees =
        median_value(&mut edge_orientation_differences).unwrap_or(f64::INFINITY);
    Some(FocusOverlapQuality {
        intensity_ncc,
        edge_ncc,
        mean_edge_orientation_cosine,
        samples: source_values.len(),
        low_frequency_mean_relative_difference,
        edge_strength_ratio,
        median_edge_orientation_difference_degrees,
        exposure_gain,
        exposure_normalized_difference,
        exposure_normalized_mean_relative_difference,
    })
}

/// Geometry is checked before the tone harmonizer runs. A confirmed structural
/// overlap must therefore survive a moderate camera-exposure change. Keep the
/// ordinary raw photometric gate unless correlation, held-out compensated
/// pixels and compensated edge strength independently verify that change.
fn virtual_tile_photometric_measurements(quality: FocusOverlapQuality) -> (f64, f64) {
    let compensated_strength = quality.edge_strength_ratio * quality.exposure_gain;
    let exposure_structure_verified = (quality.intensity_ncc >= 0.95
        && quality.edge_ncc >= 0.65
        && quality.exposure_normalized_difference <= 0.20)
        || (quality.intensity_ncc >= 0.85
            && quality.edge_ncc >= 0.45
            && quality.exposure_normalized_difference <= 0.10);
    if quality.low_frequency_mean_relative_difference
        > STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE
        && exposure_structure_verified
        && (0.5..=2.0).contains(&quality.exposure_gain)
        && quality.exposure_normalized_mean_relative_difference
            <= STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE
        && (STATION_RELATION_EDGE_STRENGTH_RATIO_MIN..=STATION_RELATION_EDGE_STRENGTH_RATIO_MAX)
            .contains(&compensated_strength)
    {
        (
            quality.exposure_normalized_mean_relative_difference,
            compensated_strength,
        )
    } else {
        (
            quality.low_frequency_mean_relative_difference,
            quality.edge_strength_ratio,
        )
    }
}

/// Legacy overlap score used by non-Station_Relation paths. Station relation
/// acceptance reads the complete measurement structure above.
fn focus_overlap_quality(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
) -> Option<(f64, f64, f64, usize)> {
    focus_overlap_quality_measurements(source, target, source_to_target)
        .map(FocusOverlapQuality::legacy_tuple)
}

fn normalized_correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_variance = 0.0;
    let mut right_variance = 0.0;
    for (&left_value, &right_value) in left.iter().zip(right) {
        let left_delta = left_value - left_mean;
        let right_delta = right_value - right_mean;
        covariance += left_delta * right_delta;
        left_variance += left_delta * left_delta;
        right_variance += right_delta * right_delta;
    }
    (left_variance > f64::EPSILON && right_variance > f64::EPSILON)
        .then(|| covariance / (left_variance * right_variance).sqrt())
        .filter(|value| value.is_finite())
}

fn find_mixed_focal_similarity_ransac(
    points: &[(Point2<f64>, Point2<f64>)],
    source_focal: f64,
    target_focal: f64,
    inlier_threshold: f64,
    minimum_inliers: usize,
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    if points.len() < 2 || !inlier_threshold.is_finite() || inlier_threshold <= 0.0 {
        return None;
    }

    // See `find_homography_ransac_points_with_solver`: site constant mixed with
    // the content derived run seed, no time and no thread id (需求 14.6).
    let mut rng = StdRng::seed_from_u64(
        0xB4D4_5EED_5CA1_E001u64
            ^ (points.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ determinism::run_random_seed_mix(),
    );
    let all_indices: Vec<usize> = (0..points.len()).collect();
    let required_inliers = minimum_inliers.max(2);
    let mut best_transform = None;
    let mut best_inliers = Vec::new();
    let mut best_median = f64::INFINITY;

    for _ in 0..MIXED_FOCAL_SIMILARITY_RANSAC_ITERATIONS {
        let sample_indices: Vec<usize> = all_indices.sample(&mut rng, 2).copied().collect();
        if sample_indices.len() != 2 {
            continue;
        }
        let sample = sample_indices
            .iter()
            .map(|&index| points[index])
            .collect::<Vec<_>>();
        let source_span = (sample[1].0 - sample[0].0).norm();
        let target_span = (sample[1].1 - sample[0].1).norm();
        if source_span < 8.0 || target_span < 8.0 {
            continue;
        }
        let Some(transform) = estimate_similarity(&sample) else {
            continue;
        };
        if !mixed_focal_scale_is_plausible(&transform, source_focal, target_focal) {
            continue;
        }
        let inliers = symmetric_inlier_indices(&transform, points, inlier_threshold);
        if inliers.len() < required_inliers {
            continue;
        }
        let inlier_points = inliers
            .iter()
            .map(|&index| points[index])
            .collect::<Vec<_>>();
        let median = median_symmetric_error(&transform, &inlier_points);
        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len() && median < best_median)
        {
            best_transform = Some(transform);
            best_inliers = inliers;
            best_median = median;
        }
    }

    if best_inliers.len() < required_inliers {
        return None;
    }
    let mut transform = best_transform?;
    let mut inliers = best_inliers;
    for _ in 0..4 {
        let inlier_points = inliers
            .iter()
            .map(|&index| points[index])
            .collect::<Vec<_>>();
        let Some(refitted) = estimate_similarity(&inlier_points) else {
            break;
        };
        if !mixed_focal_scale_is_plausible(&refitted, source_focal, target_focal) {
            break;
        }
        let refitted_inliers = symmetric_inlier_indices(&refitted, points, inlier_threshold);
        if refitted_inliers.len() < required_inliers {
            break;
        }
        transform = refitted;
        if refitted_inliers == inliers {
            break;
        }
        inliers = refitted_inliers;
    }

    Some((transform, inliers))
}

fn select_large_panorama_transform(
    projective: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    source_dimensions: (u32, u32),
    log_selection: bool,
) -> Matrix3<f64> {
    if points.len() < PANORAMA_MODEL_MIN_INLIERS {
        return *projective;
    }

    let translation = panorama_model_fit(estimate_translation(points), points);
    let similarity = estimate_similarity(points).and_then(|model| {
        panorama_transform_is_stable(&model, source_dimensions)
            .then(|| panorama_model_fit(model, points))
            .flatten()
    });
    let affine = estimate_affine(points).and_then(|model| {
        panorama_transform_is_stable(&model, source_dimensions)
            .then(|| panorama_model_fit(model, points))
            .flatten()
    });
    let projective_fit = panorama_transform_is_stable(projective, source_dimensions)
        .then(|| panorama_model_fit(*projective, points))
        .flatten();

    let mut selected = translation
        .clone()
        .or_else(|| similarity.clone())
        .or_else(|| affine.clone())
        .or_else(|| projective_fit.clone());
    let mut selected_name = if translation.is_some() {
        "translation"
    } else if similarity.is_some() {
        "similarity"
    } else if affine.is_some() {
        "affine"
    } else {
        "projective"
    };

    // A pure translation is still the safest model for a conventional long
    // panorama.  Promote it only when the data demonstrates a real scale or
    // rotation change, or when the lower-DOF model cannot explain the overlap.
    if let (Some(current), Some(candidate)) = (selected.as_ref(), similarity.as_ref()) {
        let (scale, rotation, _) = panorama_model_linear_characteristics(&candidate.0);
        let current_error = current.2;
        let meaningful_motion = (scale - 1.0).abs() > PANORAMA_MODEL_SCALE_EPSILON
            || rotation.abs() > PANORAMA_MODEL_ROTATION_EPSILON;
        let materially_better = candidate.2 + PANORAMA_MODEL_MIN_ERROR_GAIN < current_error
            || candidate.2 <= current_error * PANORAMA_MODEL_RELATIVE_ERROR_GAIN;
        let sufficient_support = candidate.1.len() + 2 >= current.1.len();
        if meaningful_motion && materially_better && sufficient_support {
            selected = Some(candidate.clone());
            selected_name = "similarity";
        }
    }

    if let (Some(current), Some(candidate)) = (selected.as_ref(), affine.as_ref()) {
        let (_, _, anisotropy) = panorama_model_linear_characteristics(&candidate.0);
        let meaningful_deformation = anisotropy > 0.018
            || candidate.0[(2, 0)].abs() > 1e-8
            || candidate.0[(2, 1)].abs() > 1e-8;
        let materially_better = candidate.2 + PANORAMA_MODEL_MIN_ERROR_GAIN < current.2
            && candidate.2 <= current.2 * 0.92;
        if meaningful_deformation && materially_better && candidate.1.len() + 2 >= current.1.len() {
            selected = Some(candidate.clone());
            selected_name = "affine";
        }
    }

    if let (Some(current), Some(candidate)) = (selected.as_ref(), projective_fit.as_ref()) {
        let perspective_strength = projective[(2, 0)].abs() + projective[(2, 1)].abs();
        let materially_better = candidate.2 + 0.2 < current.2 && candidate.2 <= current.2 * 0.82;
        if perspective_strength > 1e-8
            && materially_better
            && candidate.1.len() + 3 >= current.1.len()
        {
            selected = Some(candidate.clone());
            selected_name = "projective";
        }
    }

    let selected = selected.map(|fit| fit.0).unwrap_or_else(|| {
        // Never hand an unbounded projective fit to the graph merely because
        // every scored model missed the minimum support. A finite translation
        // is the conservative fallback and cannot fold the output canvas.
        if panorama_transform_is_stable(projective, source_dimensions) {
            *projective
        } else {
            estimate_translation(points)
        }
    });
    if log_selection {
        let describe = |fit: &Option<(Matrix3<f64>, Vec<usize>, f64)>| {
            fit.as_ref()
                .map(|(_, inliers, error)| format!("{error:.3}px/{}", inliers.len()))
                .unwrap_or_else(|| "n/a".to_string())
        };
        println!(
            "  - Large-panorama alignment selected {selected_name}: translation {}, similarity {}, affine {}, projective {}",
            describe(&translation),
            describe(&similarity),
            describe(&affine),
            describe(&projective_fit),
        );
    }
    selected
}

#[tauri::command]
pub async fn stitch_panorama(
    paths: Vec<String>,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if paths.len() < 2 {
        return Err("Please select at least two images to stitch.".to_string());
    }

    let source_paths: Vec<String> = paths
        .iter()
        .map(|p| parse_virtual_path(p).0.to_string_lossy().into_owned())
        .collect();

    let panorama_result_handle = state.panorama_result.clone();

    let task = tokio::task::spawn_blocking(move || {
        let panorama_result = stitch_images_with_options(
            source_paths,
            app_handle.clone(),
            AlignmentMode::Auto,
            BlendMode::Panorama,
            "panorama-progress",
        );

        match panorama_result {
            Ok(outcome) => {
                let panorama_image = outcome.image;
                let _ = app_handle.emit("panorama-progress", "Creating preview...");

                let (w, h) = panorama_image.dimensions();
                let (new_w, new_h) = if w > h {
                    (800, (800.0 * h as f32 / w as f32).round() as u32)
                } else {
                    ((800.0 * w as f32 / h as f32).round() as u32, 800)
                };

                let preview_f32 =
                    crate::image_processing::downscale_f32_image(&panorama_image, new_w, new_h);

                let preview_u8 = preview_f32.to_rgb8();

                let mut buf = Cursor::new(Vec::new());

                if let Err(e) = preview_u8.write_to(&mut buf, ImageFormat::Png) {
                    return Err(format!("Failed to encode panorama preview: {}", e));
                }

                let base64_str = general_purpose::STANDARD.encode(buf.get_ref());
                let final_base64 = format!("data:image/png;base64,{}", base64_str);

                *panorama_result_handle.lock().unwrap() = Some(panorama_image);

                let _ = app_handle.emit(
                    "panorama-complete",
                    serde_json::json!({
                        "base64": final_base64,
                    }),
                );
                Ok(())
            }
            Err(e) => {
                let _ = app_handle.emit("panorama-error", e.clone());
                Err(e)
            }
        }
    });

    match task.await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(join_err) => Err(format!("Panorama task failed: {}", join_err)),
    }
}

#[tauri::command]
pub async fn save_panorama(
    first_path_str: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let panorama_result = state.panorama_result.lock().unwrap();
    let panorama_image = panorama_result.as_ref().ok_or_else(|| {
        "No panorama image found in memory to save. Please generate the panorama first.".to_string()
    })?;

    let (first_path, _) = parse_virtual_path(&first_path_str);
    let parent_dir = first_path
        .parent()
        .ok_or_else(|| "Could not determine parent directory of the first image.".to_string())?;
    let stem = first_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("panorama");

    let output_path = if panorama_image.color().has_alpha() {
        parent_dir.join(format!("{}_Pano.png", stem))
    } else if panorama_image.as_rgb32f().is_some() {
        parent_dir.join(format!("{}_Pano.tiff", stem))
    } else {
        parent_dir.join(format!("{}_Pano.png", stem))
    };

    if panorama_image.color().has_alpha() {
        DynamicImage::ImageRgba8(panorama_image.to_rgba8())
            .save(&output_path)
            .map_err(|e| format!("Failed to save panorama image: {}", e))?;
    } else if panorama_image.as_rgb32f().is_some() {
        panorama_image
            .save(&output_path)
            .map_err(|e| format!("Failed to save panorama image: {}", e))?;
    } else {
        DynamicImage::ImageRgb8(panorama_image.to_rgb8())
            .save(&output_path)
            .map_err(|e| format!("Failed to save panorama image: {}", e))?;
    }

    let (real_path, _) = crate::file_management::parse_virtual_path(&first_path_str);
    let _ =
        crate::exif_processing::write_rrexif_sidecar(&real_path.to_string_lossy(), &output_path);

    Ok(output_path.to_string_lossy().to_string())
}

/// SHA-256 over every byte of each selected Source_RAW, in the order the paths
/// were handed in.  `None` marks a source that could not be read.
///
/// Every file is opened read only with `File::open` and streamed (需求 4.8), so
/// a 50 MB RAW never lands in memory as a whole and no path in this function can
/// modify a source.  Hashing runs on the `rayon` pool; the result is a plain
/// per-path mapping, so the thread count cannot reach it.
fn focus_stack_source_digests(image_paths: &[String]) -> Vec<Option<[u8; 32]>> {
    image_paths
        .par_iter()
        .map(|path| {
            let (real_path, _) = crate::file_management::parse_virtual_path(path);
            match virtual_tile::source_file_sha256(&real_path) {
                Ok(digest) => Some(digest),
                Err(error) => {
                    println!("Focus stack: {error}");
                    None
                }
            }
        })
        .collect()
}

/// Lower hex of a source digest, or the empty string when the file could not be
/// read.  An empty string is what the Stack_Report records for "not measured",
/// which must stay distinguishable from a digest that is genuinely all zero.
fn source_digest_hex(digest: Option<[u8; 32]>) -> String {
    digest
        .map(|digest| virtual_tile::hex_digest(&digest))
        .unwrap_or_default()
}

/// The Source_RAW digests taken before the run, together with the three
/// Virtual_Tile `cache_key` ingredients they feed (需求 4.5).
struct FocusStackSourceDigests {
    /// Pipeline version, sorted absolute source paths, sorted source digests.
    inputs: virtual_tile::CacheKeyInputs,
    /// Lower hex digest per selected path, in selection order (需求 4.8 / 12.8).
    sha256_before: Vec<String>,
}

/// Hash every selected Source_RAW once and derive the run's `cache_key`
/// ingredients from the result.
///
/// A source that cannot be opened contributes an all-zero digest to the key
/// rather than dropping out of the set: the key must stay a pure function of the
/// selection, and an unreadable source fails the run later on its own terms.
/// `sha256_before` keeps that case empty instead, so the report never claims a
/// digest it did not measure.
///
/// The ingredients are sorted after hashing, so neither the import order nor the
/// completion order of the `rayon` tasks reaches the key.
fn focus_stack_cache_key_inputs(image_paths: &[String]) -> FocusStackSourceDigests {
    let digests = focus_stack_source_digests(image_paths);
    let sources = image_paths
        .iter()
        .zip(&digests)
        .map(|(path, digest)| (path.clone(), digest.unwrap_or([0u8; 32])))
        .collect::<Vec<_>>();
    // The proxy is part of the focus-stack analysis input even though its
    // pixels are generated inside the preparation workers.  Include the
    // versioned resize contract and the input-count budget in the cache
    // identity so a change to proxy dimensions/filtering cannot reuse a stale
    // Virtual_Tile entry.  Source dimensions remain content-bound through the
    // digest set below.
    FocusStackSourceDigests {
        inputs: virtual_tile::CacheKeyInputs::new(
            &focus_stack_cache_pipeline_version(image_paths.len()),
            &sources,
        ),
        sha256_before: digests.into_iter().map(source_digest_hex).collect(),
    }
}

fn focus_stack_cache_pipeline_version(input_count: usize) -> String {
    stack_pipeline::proxy::ProxyBudget::for_source(1, 1, input_count, None)
        .map(|budget| {
            format!(
                "{}|{}",
                crate::image_stack::STACK_PIPELINE_VERSION,
                budget.cache_key_fragment()
            )
        })
        .unwrap_or_else(|_| crate::image_stack::STACK_PIPELINE_VERSION.to_string())
}

pub(crate) fn stitch_images_with_options<R: Runtime>(
    image_paths: Vec<String>,
    app_handle: AppHandle<R>,
    alignment_mode: AlignmentMode,
    blend_mode: BlendMode,
    progress_event: &str,
) -> Result<StitchOutcome, String> {
    let run_generation = degradation::active_run_generation();
    let _run_scope = degradation::begin_run_scope_for_generation(run_generation).map_err(|_| {
        "image-stack run cancelled by a newer request before it started".to_string()
    })?;
    let image_paths = image_paths
        .into_iter()
        .filter(|path| !is_generated_stitch_output(path) && !is_auxiliary_stitch_file(path))
        .collect::<Vec<_>>();
    // A PSD in this product is one flattened scan tile, not one focus plane.
    // The default UI blend mode is FocusStack, but routing a complete PSD set
    // through the singleton focus fallback creates hard source-owner blocks.
    // Use the panorama seam compositor for this specific input shape while
    // keeping the historical focus path unchanged for RAW/JPEG stacks.
    let psd_source_scan = image_paths.len() > 1
        && image_paths.iter().all(is_psd_file)
        && blend_mode == BlendMode::FocusStack;
    let blend_mode = if psd_source_scan {
        BlendMode::Panorama
    } else {
        blend_mode
    };
    let alignment_mode = if psd_source_scan {
        AlignmentMode::Auto
    } else {
        alignment_mode
    };
    // Open a fresh degradation observation window for this run. Recording is
    // additive: no fallback decision below consults the ledger.
    degradation::reset_run_ledger();
    // The Intra_Station_Registrar records one entry per registered frame
    // (需求 2.9); open its window alongside the ledger so a second run in the
    // same process cannot inherit the previous run's frames.
    intra_station::reset_run_records();
    // The Focus_Fuser records one entry per Capture_Station (需求 3.5 / 3.8 / 3.9)
    // through the same run scoped sink.
    focus_fuser::reset_run_records();
    #[cfg(test)]
    stitching::reset_final_world_owner_oracle();
    // Residual_Warp_Model and Tone_Harmonizer report through run scoped sinks
    // as well, and the Tile_Compositor samples the recorded residual model, so
    // a later run must never see an earlier run's model (需求 8.1, 9.11).
    residual_warp::reset_run_records();
    tone::reset_run_records();
    // The accepted Station_Relations the Tone_Harmonizer samples from belong to
    // this run's station solve only (需求 9.1).
    tone::reset_run_topology();
    // Capture_Topology_Model observations are likewise scoped to this run so
    // the report receives exactly the topology used by the production station
    // relation candidate gate (需求 5.1 / 5.7).
    topology::reset_run_topology();
    // Install the RANSAC seed of this run before any matching starts (需求 14.6).
    // The focus stack path seeds from the Virtual_Tile `cache_key`, which binds
    // the pipeline version, the sorted source paths and the sorted per-source
    // content digests (需求 4.5), so the sampling sequence follows the input
    // content rather than the wall clock or the thread count.  The panorama path
    // keeps its site constants, and clearing here stops it from inheriting the
    // seed of a preceding focus stack run in the same process.
    // The same read-only pass also produces the `sha256_before` column of the
    // Stack_Report (需求 4.8 / 12.8), so the sources are hashed once per run.
    let mut source_sha256_before = Vec::<String>::new();
    if blend_mode == BlendMode::FocusStack {
        let digests = focus_stack_cache_key_inputs(&image_paths);
        determinism::set_run_random_seed(determinism::derive_run_seed_from_cache_key(
            &digests.inputs.digest(),
        ));
        source_sha256_before = digests.sha256_before;
    } else {
        determinism::clear_run_random_seed();
    }
    // Observation only: the Stack_Report records what a focus-stack run did and
    // is flushed on every terminating path (success, degraded output,
    // rejection, cancellation) by its drop guard, well inside the 30 s budget
    // of requirement 10.8.  No field written here feeds a decision yet.
    let stack_report = (blend_mode == BlendMode::FocusStack).then(|| {
        let recorder = stack_report::StackReportRecorder::new(
            crate::image_stack::STACK_PIPELINE_VERSION,
            stack_report::resolve_stack_report_dir(&app_handle),
        )
        // Requirement 4.8 / 12.8: whichever way this run ends — success,
        // failure or cancellation — the recorder's drop guard re-reads every
        // Source_RAW and records the digest next to the one taken above, so a
        // source modified behind the pipeline's back stays visible.  The
        // pipeline itself only ever opens a source read only, so the two
        // columns are expected to match.
        .with_source_digest_recheck(Box::new(|paths| {
            focus_stack_source_digests(paths)
                .into_iter()
                .map(source_digest_hex)
                .collect()
        }));
        recorder.update(|report| {
            report.input.source_count = image_paths.len();
            report.input.max_sources = MAX_STITCH_SOURCE_IMAGES;
            report.input.sources = image_paths
                .iter()
                .enumerate()
                .map(|(index, path)| stack_report::SourceRecord {
                    path: path.clone(),
                    sha256_before: source_sha256_before.get(index).cloned().unwrap_or_default(),
                    ..Default::default()
                })
                .collect();
            // Any return before the compositor finishes is a run that produced
            // no result; a successful path overwrites this below.
            report.degradation.result = stack_report::StackRunResult::Rejected;
            report.resources.worker_threads = rayon::current_num_threads();
            report.resources.random_seed_source = determinism::run_random_seed_source().to_string();
        });
        Arc::new(recorder)
    });
    // Sample process RSS for the complete focus-stack run.  No settings
    // source is exposed by the current runtime entry point, so the documented
    // default threshold is used and reported as such.
    let mut rss_sampler = if blend_mode == BlendMode::FocusStack {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        let physical_memory_bytes = system.total_memory();
        let threshold = resources::resolve_memory_threshold(physical_memory_bytes, None, false);
        let pid = sysinfo::get_current_pid().ok();
        let sampler = resources::RssSampler::start(threshold.bytes, move || {
            let Some(pid) = pid else {
                return 0;
            };
            let mut system = sysinfo::System::new();
            system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
            system.process(pid).map_or(0, |process| process.memory())
        });
        if let Some(recorder) = stack_report.as_ref() {
            recorder.update(|report| {
                report.resources.memory_threshold_bytes = threshold.bytes;
                report.resources.memory_threshold_source = threshold.source.as_str().to_string();
                report.resources.physical_memory_bytes = physical_memory_bytes;
            });
        }
        Some(FocusRssGuard {
            sampler: Some(sampler),
            threshold_bytes: threshold.bytes,
            recorder: stack_report.clone(),
        })
    } else {
        None
    };
    ensure_focus_memory_available(rss_sampler.as_ref())?;
    if image_paths.len() < 2 {
        degradation::record_run_degradation(
            degradation::INPUT_SOURCE_COUNT_OUT_OF_RANGE,
            serde_json::json!({
                "count": image_paths.len(),
                "minimum": 2,
                "maximum": MAX_STITCH_SOURCE_IMAGES,
            }),
        );
        return Err("At least two images are required for a panorama.".to_string());
    }
    if image_paths.len() > MAX_STITCH_SOURCE_IMAGES {
        degradation::record_run_degradation(
            degradation::INPUT_SOURCE_COUNT_OUT_OF_RANGE,
            serde_json::json!({
                "count": image_paths.len(),
                "minimum": 2,
                "maximum": MAX_STITCH_SOURCE_IMAGES,
            }),
        );
        return Err(format!(
            "Image stitching is limited to {MAX_STITCH_SOURCE_IMAGES} source images."
        ));
    }

    let _ = app_handle.emit(progress_event, "Starting image alignment process...");
    println!(
        "Starting panorama stitching process for {} images...",
        image_paths.len()
    );

    let focus_stack = blend_mode == BlendMode::FocusStack;
    let scalable_stack = stack_requires_bounded_memory(&image_paths);
    let mixed_focal_stack = !scalable_stack && selection_has_mixed_focal_lengths(&image_paths);
    let scale_robust_alignment = scalable_stack || mixed_focal_stack;
    let settings = load_settings_for_runtime(&app_handle).unwrap_or_default();
    // Diagnostics are opt-in through the persisted settings.  Creating the
    // directory up front keeps the recorder's per-station writes atomic while
    // leaving the default run allocation-free and pixel-identical.
    let mut diagnostics_recorder = if focus_stack {
        settings
            .stack_diagnostics
            .output_dir
            .as_deref()
            .map(str::trim)
            .filter(|directory| !directory.is_empty())
            .map(|directory| {
                let path = PathBuf::from(directory);
                match std::fs::create_dir_all(&path) {
                    Ok(()) => DiagnosticsRecorder::new(Some(path)),
                    Err(error) => {
                        degradation::record_run_degradation(
                            degradation::DIAGNOSTICS_WRITE_FAILED,
                            serde_json::json!({
                                "directory": directory,
                                "error": error.to_string(),
                            }),
                        );
                        DiagnosticsRecorder::new(None)
                    }
                }
            })
            .unwrap_or_else(|| DiagnosticsRecorder::new(None))
    } else {
        DiagnosticsRecorder::new(None)
    };
    // A focus stack must retain one shared display scale across its RAW
    // sources.  Per-frame percentile normalization makes the 97th percentile
    // of every frame identical even when that percentile belongs to artwork,
    // a holder, or a different amount of canvas.  The overlap solver then
    // sees a pre-distorted sequence and can only repair it with tile-sized
    // gains.  Collect the inexpensive low-resolution percentile while the
    // preparation workers already have each developed frame in memory; the
    // rendering closure below applies one robust gain to every RAW.
    let raw_exposure_reference = focus_stack
        .then(|| stack_raw_exposure_reference(&image_paths))
        .flatten();
    if let Some(reference) = raw_exposure_reference {
        println!("Shared RAW physical exposure reference: {reference:.6} (time × ISO / aperture²)");
    }
    let raw_display_highs = Mutex::new(Vec::<f32>::new());

    let start_time = Instant::now();
    let _ = app_handle.emit(progress_event, "Loading and preparing images...");
    let (alignment_max_dimension, alignment_max_features) =
        scalable_alignment_budget(image_paths.len());
    println!(
        "Loading and preparing images ({} mode)...",
        if scalable_stack {
            "bounded-memory"
        } else if mixed_focal_stack {
            "full-resolution, mixed-focal"
        } else {
            "full-resolution"
        }
    );
    let brief_pairs = processing::generate_brief_pairs();
    let prepared_count = Mutex::new(0usize);
    let prepare_images = || {
        image_paths
            .par_iter()
            .enumerate()
            .map(|(i, filename)| {
                if degradation::run_generation_changed_for(run_generation) {
                    return Err("image-stack run cancelled by a newer request".to_string());
                }
                println!("  - Processing '{}'", filename);
                let prepared_source =
                    load_prepared_stack_source(filename, &settings, raw_exposure_reference)?;
                if degradation::run_generation_changed_for(run_generation) {
                    return Err("image-stack run cancelled by a newer request".to_string());
                }
                let dynamic_image = prepared_source.image;
                let focal_length_35mm = prepared_source.focal_length_35mm;
                let source_footprint = prepared_source.source_footprint;
                if (focus_stack || alignment_mode == AlignmentMode::FlatArtwork)
                    && is_raw_file(filename)
                    && std::env::var_os("RAW_EDITOR_SKIP_STACK_RAW_EXPOSURE_NORMALIZATION")
                        .is_none()
                    && let Some(high) = raw_display_channel_percentile(&dynamic_image)
                {
                    raw_display_highs.lock().unwrap().push(high);
                }
                let (width, height) = dynamic_image.dimensions();
                let (new_width, new_height, scale_factor) = if scalable_stack {
                    processing::calculate_downscale_dimensions_capped(
                        width,
                        height,
                        alignment_max_dimension,
                    )
                } else {
                    processing::calculate_downscale_dimensions(width, height)
                };

                let alignment_image = if scalable_stack {
                    if (new_width, new_height) == (width, height) {
                        dynamic_image.to_luma8()
                    } else {
                        dynamic_image
                            .resize_exact(
                                new_width,
                                new_height,
                                image::imageops::FilterType::Triangle,
                            )
                            .to_luma8()
                    }
                } else {
                    let color_full_u8 = dynamic_image.to_rgb8();
                    let gray_full = image::imageops::colorops::grayscale(&color_full_u8);
                    image::imageops::resize(
                        &gray_full,
                        new_width,
                        new_height,
                        image::imageops::FilterType::Triangle,
                    )
                };

                let features = find_alignment_features(
                    &alignment_image,
                    &brief_pairs,
                    alignment_max_features,
                    scale_robust_alignment,
                    focal_length_35mm,
                );
                let foreground_range = if focus_stack {
                    detect_foreground_range(&alignment_image)
                } else {
                    None
                };
                let foreground_mask = if focus_stack {
                    build_foreground_mask(&alignment_image, foreground_range)
                } else {
                    None
                };
                let horizontal_edge_rows = if focus_stack {
                    horizontal_edge_row_candidates(&alignment_image, foreground_range)
                } else {
                    Vec::new()
                };
                let vertical_edge_columns = if focus_stack {
                    detect_vertical_edge_columns(&alignment_image, foreground_range)
                } else {
                    Vec::new()
                };
                let top_features = if focus_stack {
                    find_top_alignment_features(&alignment_image, &brief_pairs, foreground_range)
                } else {
                    Vec::new()
                };
                let full_image = (!scalable_stack).then(|| dynamic_image.to_rgb32f());
                println!("    Found {} features in '{}'", features.len(), filename);

                let mut prepared_count = prepared_count
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                *prepared_count += 1;
                let completed = *prepared_count;
                let overall_percentage = 5.0 + (completed as f64 / image_paths.len() as f64) * 24.0;
                let _ = app_handle.emit(
                    progress_event,
                    format!(
                        "Analyzing image {completed} of {} ({overall_percentage:.0}%): {}",
                        image_paths.len(),
                        Path::new(filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                    ),
                );
                drop(prepared_count);

                Ok(ImageInfo {
                    id: i,
                    filename: filename.to_string(),
                    width,
                    height,
                    alignment_image,
                    full_image,
                    scale_factor,
                    focal_length_35mm,
                    source_footprint: source_footprint,
                    overview_reference: false,
                    features,
                    top_features,
                    foreground_range,
                    foreground_mask,
                    horizontal_edge_rows,
                    vertical_edge_columns,
                })
            })
            .collect::<Vec<Result<ImageInfo, String>>>()
    };

    let image_data_results = if scalable_stack {
        let preparation_workers = scalable_preparation_worker_count(image_paths.len());
        println!("Preparing images with {preparation_workers} bounded worker(s)...");
        let pool = ThreadPoolBuilder::new()
            .num_threads(preparation_workers)
            .thread_name(|index| format!("image-stack-analysis-{index}"))
            .build()
            .map_err(|error| format!("Failed to start image-stack workers: {error}"))?;
        pool.install(prepare_images)
    } else {
        prepare_images()
    };
    ensure_focus_memory_available(rss_sampler.as_ref())?;

    // Preparation already finished for every source, so collect the whole batch
    // before propagating.  The caller still observes the first error exactly as
    // a `?` in the loop would, but the Stack_Report can now name which sources
    // decoded instead of reporting the entire selection as undecodable.
    let mut image_data = Vec::new();
    let mut preparation_error = None;
    for result in image_data_results {
        match result {
            Ok(image) => image_data.push(image),
            Err(error) => {
                if preparation_error.is_none() {
                    preparation_error = Some(error);
                }
            }
        }
    }
    // 需求 1.9: a Source_RAW that does not decode is excluded from every
    // Capture_Station and reported by absolute path with `source_decode_failed`,
    // while the remaining sources keep the grouping they would have had anyway.
    let undecodable_paths = undecodable_source_paths(&image_paths, &image_data);
    if let Some(recorder) = stack_report.as_ref() {
        let decoded_paths = image_data
            .iter()
            .map(|image| image.filename.as_str())
            .collect::<HashSet<_>>();
        let undecodable = undecodable_paths
            .iter()
            .map(|path| stack_report::ExcludedSourceRecord {
                path: path.clone(),
                reason: degradation::SOURCE_DECODE_FAILED.to_string(),
            })
            .collect::<Vec<_>>();
        recorder.update(|report| {
            for source in &mut report.input.sources {
                source.decoded = decoded_paths.contains(source.path.as_str());
            }
            report.grouping.undecodable = undecodable;

            // The alignment image is the single proxy consumed by the
            // feature, geometry and seam stages. Record its actual dimensions
            // and axis scales instead of assuming a fixed four-times resize.
            let proxy_budget = image_data.first().and_then(|image| {
                stack_pipeline::proxy::ProxyBudget::for_source(
                    image.width,
                    image.height,
                    image_paths.len(),
                    None,
                )
                .ok()
            });
            report.input.proxy_budget = proxy_budget.clone();
            report.input.proxy_images = image_data
                .iter()
                .filter_map(|image| {
                    let digest = source_sha256_before
                        .get(image.id)
                        .and_then(|hex_digest| hex::decode(hex_digest).ok())
                        .and_then(|bytes| <[u8; 32]>::try_from(bytes).ok())?;
                    let source = stack_pipeline::proxy::FullResSource::new(
                        PathBuf::from(&image.filename),
                        image.width,
                        image.height,
                        digest,
                        "source-full-pixel-v1",
                        stitching::ColorEncoding::DisplaySrgb,
                    )
                    .ok()?;
                    let budget = stack_pipeline::proxy::ProxyBudget::for_source(
                        image.width,
                        image.height,
                        image_paths.len(),
                        None,
                    )
                    .ok()?;
                    stack_pipeline::proxy::ProxyImage::from_source(&source, &budget).ok()
                })
                .collect();
        });
    }
    // The ledger entry is independent of the report recorder: 需求 1.9 asks for
    // one `source_decode_failed` record per excluded source, by absolute path.
    for path in &undecodable_paths {
        degradation::record_run_degradation(
            degradation::SOURCE_DECODE_FAILED,
            serde_json::json!({ "path": path }),
        );
    }
    if let Some(error) = preparation_error {
        // 需求 1.9: on the FocusStack path an undecodable Source_RAW is excluded
        // from every Capture_Station and the remaining sources keep the grouping
        // they would have had anyway, so the run continues on the decodable
        // subset.  This entry point is shared with the panorama path, which has
        // no Capture_Station layer to exclude a source from and no Stack_Report
        // to name it in: a panorama still aborts on the first failure.
        if !focus_stack {
            return Err(error);
        }
        // 需求 1.8: fewer than two decodable sources leaves nothing to group, so
        // the run is rejected with the input-range identifier rather than
        // degraded.
        if image_data.len() < 2 {
            degradation::record_run_degradation(
                degradation::INPUT_SOURCE_COUNT_OUT_OF_RANGE,
                serde_json::json!({
                    "count": image_data.len(),
                    "minimum": 2,
                    "maximum": MAX_STITCH_SOURCE_IMAGES,
                    "undecodable": undecodable_paths.len(),
                }),
            );
            return Err(format!(
                "Only {} of {} selected images could be decoded; at least two are required. First failure: {error}",
                image_data.len(),
                image_paths.len(),
            ));
        }
        println!(
            "  - Excluded {} undecodable source(s) from every capture station; continuing with {} of {}. First failure: {error}",
            undecodable_paths.len(),
            image_data.len(),
            image_paths.len(),
        );
        let _ = app_handle.emit(
            progress_event,
            format!(
                "Skipped {} undecodable image(s); continuing with {}.",
                undecodable_paths.len(),
                image_data.len(),
            ),
        );
    }

    if alignment_mode == AlignmentMode::FlatArtwork {
        filter_flat_artwork_blurry_sources(&mut image_data);
        if image_data.len() < 2 {
            return Err(
                "Flat-artwork blur filtering left fewer than two usable source images.".to_string(),
            );
        }
        // Matching and the panorama compositor use the prepared-vector index
        // as the pose-map key. Reassign it after filtering so removing a blurry
        // source cannot leave a hole between `ImageInfo::id` and that map.
        for (index, image) in image_data.iter_mut().enumerate() {
            image.id = index;
        }
    }

    let shared_raw_display_gain = if (focus_stack || alignment_mode == AlignmentMode::FlatArtwork)
        && std::env::var_os("RAW_EDITOR_SKIP_STACK_RAW_EXPOSURE_NORMALIZATION").is_none()
        && std::env::var_os("RAW_EDITOR_STACK_RAW_FIXED_GAIN").is_none()
        && std::env::var_os("RAW_EDITOR_DISABLE_SHARED_RAW_NORMALIZATION").is_none()
    {
        shared_stack_raw_display_gain(&raw_display_highs.into_inner().unwrap_or_default())
    } else {
        None
    };
    if let Some(gain) = shared_raw_display_gain {
        println!(
            "Shared RAW display normalization: gain={gain:.3} from {} frame percentiles",
            image_paths.len()
        );
    }

    println!(
        "Image loading and feature detection completed in {:.2?}\n",
        start_time.elapsed()
    );

    let start_time = Instant::now();
    let exhaustive_search = !scalable_stack
        || (blend_mode == BlendMode::FocusStack
            && image_data.len() <= FOCUS_AUTO_ORDER_EXHAUSTIVE_MAX_SOURCES)
        || (SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES..=SCALE_ROBUST_EXHAUSTIVE_MAX_SOURCES)
            .contains(&image_data.len());
    let matching_strategy = if exhaustive_search {
        "all pairwise"
    } else {
        "ordered-neighbor + visual retrieval"
    };
    let _ = app_handle.emit(
        progress_event,
        format!("Finding image matches ({matching_strategy})..."),
    );
    println!("Finding {matching_strategy} matches (in parallel)...");
    let projection = alignment_mode.projection_for(blend_mode);
    let mut pairwise_matches: HashMap<(usize, usize), MatchInfo> = HashMap::new();

    #[cfg(test)]
    let sequence_only = std::env::var_os("RAW_EDITOR_STACK_SEQUENCE_ONLY").is_some();
    #[cfg(not(test))]
    let sequence_only = false;

    let pairs_to_check = if sequence_only {
        Vec::new()
    } else {
        pairs_to_match_for_images(&image_data, blend_mode)
    };
    let matched_pair_count = Mutex::new(0);
    let pair_progress_step = (pairs_to_check.len() / 100).max(1);

    let match_results: Vec<Option<((usize, usize), MatchInfo)>> = pairs_to_check
        .par_iter()
        .map(|&(i, j)| {
            if degradation::run_generation_changed_for(run_generation) {
                return None;
            }
            let _progress = PairMatchProgress {
                completed: &matched_pair_count,
                total: pairs_to_check.len(),
                progress_step: pair_progress_step,
                app_handle: &app_handle,
                progress_event,
            };
            // Focus-stack matching is not perfectly symmetric: the ratio test and
            // local NCC search are evaluated from source to target. Use a stable
            // path-based direction, then convert back to the (i, j) key orientation.
            // Otherwise changing import order can produce a different transform graph
            // and expose defocus halos at object silhouettes. Keep panorama matching's
            // established direction unchanged.
            let (source_index, target_index, invert_for_storage) =
                if blend_mode == BlendMode::FocusStack {
                    canonical_match_direction(&image_data, i, j)
                } else {
                    (i, j, false)
                };
            let source_image = &image_data[source_index];
            let target_image = &image_data[target_index];
            let mut match_info = match_image_pair(
                source_image,
                target_image,
                projection,
                blend_mode,
                alignment_mode,
                scale_robust_alignment,
                true,
                false,
            )?;
            if blend_mode == BlendMode::FocusStack {
                // Record the fit before it is rotated into the key orientation;
                // see `MatchInfo::canonical_homography` (需求 14.6).
                match_info.canonical_homography = Some(match_info.homography);
            }
            if invert_for_storage {
                match_info.homography = match_info.homography.try_inverse()?;
                match_info.points = match_info
                    .points
                    .into_iter()
                    .map(|(source, target)| (target, source))
                    .collect();
                match_info.candidate_points = match_info
                    .candidate_points
                    .into_iter()
                    .map(|(source, target)| (target, source))
                    .collect();
                match_info.top_candidate_points = match_info
                    .top_candidate_points
                    .into_iter()
                    .map(|(source, target)| (target, source))
                    .collect();
                match_info.dense_focus_points = match_info
                    .dense_focus_points
                    .into_iter()
                    .map(|(source, target)| (target, source))
                    .collect();
                match_info.foreground_feature_points = match_info
                    .foreground_feature_points
                    .into_iter()
                    .map(|(source, target)| (target, source))
                    .collect();
            }
            Some(((i, j), match_info))
        })
        .collect();
    ensure_focus_memory_available(rss_sampler.as_ref())?;

    for result in match_results.into_iter().flatten() {
        pairwise_matches.insert(result.0, result.1);
    }
    if sequence_only {
        add_capture_sequence_bridges(&image_data, &mut pairwise_matches);
    }
    // Filename adjacency never creates or clears a focus-stack edge. The old
    // filename-window recovery pass (clear + retry on natural-sort neighbours)
    // was removed in favour of the geometry-driven retry below: a capture
    // sequence may reset to another raster row, and repeated seals/figures can
    // make such a false edge look convincing. Scan-position edges are therefore
    // never synthesized from filename order or a coarse NCC peak; if the full
    // feature/RANSAC matcher finds no evidence the stack stays disconnected and
    // is rejected instead of rendered as a duplicated collage.
    if scalable_stack && focus_stack && !sequence_only {
        // A disconnected focus stack gets a bounded, evidence-only retry. The
        // old retry examined only filename-window boundaries, which is exactly
        // where a raster reset can masquerade as a neighbouring tile. Search
        // the strongest frames in every component pair instead, and accept a
        // bridge only when the normal feature/RANSAC matcher returns a real
        // geometric match (never a synthetic/coarse bridge).
        let mut rescue_cache = HashMap::<usize, ImageInfo>::new();
        for pass in 0..3 {
            let components = focus_overlap_components(&image_data, &pairwise_matches);
            if components.len() <= 1 {
                break;
            }
            let mut rescue_pairs = Vec::<(usize, usize, usize)>::new();
            for left_component_index in 0..components.len() {
                for right_component_index in left_component_index + 1..components.len() {
                    let mut left_candidates = components[left_component_index].clone();
                    let mut right_candidates = components[right_component_index].clone();
                    left_candidates.sort_by_key(|&index| {
                        std::cmp::Reverse(focus_alignment_support(&image_data[index]))
                    });
                    right_candidates.sort_by_key(|&index| {
                        std::cmp::Reverse(focus_alignment_support(&image_data[index]))
                    });
                    left_candidates.truncate(6);
                    right_candidates.truncate(6);
                    for &index in components[left_component_index]
                        .iter()
                        .chain(components[left_component_index].iter().rev())
                        .take(4)
                    {
                        if !left_candidates.contains(&index) {
                            left_candidates.push(index);
                        }
                    }
                    for &index in components[right_component_index]
                        .iter()
                        .chain(components[right_component_index].iter().rev())
                        .take(4)
                    {
                        if !right_candidates.contains(&index) {
                            right_candidates.push(index);
                        }
                    }
                    for &left in &left_candidates {
                        for &right in &right_candidates {
                            let key = (left.min(right), left.max(right));
                            if !pairwise_matches.contains_key(&key) {
                                rescue_pairs.push((
                                    focus_alignment_support(&image_data[left])
                                        + focus_alignment_support(&image_data[right]),
                                    left,
                                    right,
                                ));
                            }
                        }
                    }
                }
            }
            rescue_pairs
                .sort_by_key(|(support, left, right)| (std::cmp::Reverse(*support), *left, *right));
            rescue_pairs.truncate(128);
            if rescue_pairs.is_empty() {
                break;
            }
            println!(
                "Retrying {} disconnected focus component pair(s) with strict feature geometry (pass {})...",
                rescue_pairs.len(),
                pass + 1,
            );
            let mut added = 0usize;
            for (_, first, second) in rescue_pairs {
                let (source_index, target_index, invert_for_storage) =
                    canonical_match_direction(&image_data, first, second);
                if !rescue_cache.contains_key(&source_index) {
                    rescue_cache.insert(
                        source_index,
                        prepare_focus_rescue_image(
                            &image_data[source_index],
                            &settings,
                            raw_exposure_reference,
                            &brief_pairs,
                        )?,
                    );
                }
                if !rescue_cache.contains_key(&target_index) {
                    rescue_cache.insert(
                        target_index,
                        prepare_focus_rescue_image(
                            &image_data[target_index],
                            &settings,
                            raw_exposure_reference,
                            &brief_pairs,
                        )?,
                    );
                }
                let source = rescue_cache
                    .get(&source_index)
                    .expect("rescue source cached");
                let target = rescue_cache
                    .get(&target_index)
                    .expect("rescue target cached");
                let Some(mut match_info) = match_image_pair(
                    source,
                    target,
                    projection,
                    blend_mode,
                    alignment_mode,
                    true,
                    true,
                    true,
                ) else {
                    continue;
                };
                if !focus_rescue_match_is_verified(source, target, &match_info) {
                    println!(
                        "  - Rejected rescue candidate '{}' <-> '{}' after geometry validation",
                        Path::new(&image_data[first].filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy(),
                        Path::new(&image_data[second].filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy(),
                    );
                    continue;
                }
                // Record the fit before it is rotated into the key orientation;
                // see `MatchInfo::canonical_homography` (需求 14.6).
                match_info.canonical_homography = Some(match_info.homography);
                if invert_for_storage {
                    match_info.homography = match_info
                        .homography
                        .try_inverse()
                        .ok_or_else(|| "Failed to invert a focus rescue transform".to_string())?;
                    match_info.points = match_info
                        .points
                        .into_iter()
                        .map(|(source, target)| (target, source))
                        .collect();
                    match_info.candidate_points = match_info
                        .candidate_points
                        .into_iter()
                        .map(|(source, target)| (target, source))
                        .collect();
                    match_info.top_candidate_points = match_info
                        .top_candidate_points
                        .into_iter()
                        .map(|(source, target)| (target, source))
                        .collect();
                    match_info.dense_focus_points = match_info
                        .dense_focus_points
                        .into_iter()
                        .map(|(source, target)| (target, source))
                        .collect();
                    match_info.foreground_feature_points = match_info
                        .foreground_feature_points
                        .into_iter()
                        .map(|(source, target)| (target, source))
                        .collect();
                }
                let key = (first.min(second), first.max(second));
                println!(
                    "  - Accepted strict focus rescue: '{}' <-> '{}' ({} inliers)",
                    Path::new(&image_data[first].filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    Path::new(&image_data[second].filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    match_info.points.len()
                );
                pairwise_matches.insert(key, match_info);
                added += 1;
            }
            if added == 0 {
                break;
            }
        }
        let consensus_bridges = recover_focus_component_edges(&image_data, &mut pairwise_matches);
        if consensus_bridges > 0 {
            println!(
                "  - Recovered {consensus_bridges} component bridge(s) from multi-layer coarse consensus"
            );
        }
    }
    // Never invent a focus-stack edge from capture order. A large artwork is
    // commonly photographed in several raster passes and focus brackets, so a
    // filename-adjacent pair can be a scan reset rather than a neighbouring
    // tile. Every global pose must be connected by verified image evidence.
    println!(
        "Pairwise matching completed in {:.2?}\n",
        start_time.elapsed()
    );

    if pairwise_matches.is_empty() {
        return Err(if scalable_stack && focus_stack {
            if exhaustive_search {
                "No overlap was found among the selected focus-stack images. Make sure the selection is one contiguous scene with sufficient overlap."
                    .to_string()
            } else {
                "No overlap was found among the automatic focus-stack candidate pairs. Large stacks use a bounded search; use frames with sufficient overlap or reduce the number of layers."
                    .to_string()
            }
        } else if scalable_stack {
            if exhaustive_search {
                "No overlap was found among the selected panorama images. Make sure the selection is one contiguous scene with sufficient overlap."
                    .to_string()
            } else {
                "No overlap was found between nearby images. For large panoramas, make sure consecutive images overlap."
                    .to_string()
            }
        } else {
            "No suitable matches found between any pair of images. Cannot create a panorama."
                .to_string()
        });
    }

    let start_time = Instant::now();
    let _ = app_handle.emit(progress_event, "Determining stitching order...");
    println!("Determining stitching order...");
    let compositor_choice = (blend_mode == BlendMode::FocusStack)
        .then(|| StackCompositorChoice::resolve(settings.stack_compositor.as_deref()));
    let defer_authoritative_station_solve =
        compositor_choice == Some(StackCompositorChoice::LayeredVirtualTile);
    let (ordered_indices, mut global_homographies) = if blend_mode == BlendMode::FocusStack {
        // The default two-stage path uses source images only to group stations
        // and establish each station's local tile geometry.  Its authoritative
        // requirement-6/7 report is produced later from the fused tiles, so an
        // early source-only support=1 graph must neither reject nor publish a
        // report that could overwrite the Virtual_Tile result.
        let mut closure = stack_report::ClosureReport::default();
        let mut station_relations = stack_report::StationRelationsReport::default();
        let mut stitching_order = build_focus_stack_stitching_order_with_report(
            &image_data,
            &pairwise_matches,
            &mut closure,
            &mut station_relations,
            !defer_authoritative_station_solve,
        );
        #[cfg(test)]
        if std::env::var_os("RAW_EDITOR_STACK_PAIR_HARNESS_STOP_AFTER_RELATIONS").is_some() {
            stitching_order =
                pair_harness_disconnected_focus_geometry(&image_data, &pairwise_matches);
        }
        let disconnected_component_count = station_relations.connectivity.components;
        if !defer_authoritative_station_solve {
            if let Some(recorder) = stack_report.as_ref() {
                recorder.update(|report| {
                    report.closure = closure;
                    report.station_relations = station_relations;
                });
            }
            if disconnected_component_count > 1 {
                return Err(geometry_disconnected_user_error(
                    disconnected_component_count,
                ));
            }
        }
        stitching_order
    } else {
        build_stitching_order(&image_data, &pairwise_matches)
    };
    if blend_mode == BlendMode::Panorama && alignment_mode == AlignmentMode::FlatArtwork {
        if let Some(refined) =
            refine_flat_artwork_tree_poses(&image_data, &pairwise_matches, &global_homographies)
        {
            global_homographies = refined;
        }
    }
    let focus_layer_warp = (blend_mode == BlendMode::FocusStack).then(|| {
        build_focus_layer_warp(
            &image_data,
            &pairwise_matches,
            &global_homographies,
            projection,
        )
    });
    // Flat-artwork panorama inputs can still contain focus brackets at each
    // camera position. Build the same station/group map so the detail-preserving
    // source-owner compositor fuses those layers instead of treating every RAW
    // as a new panorama tile.
    let focus_station_grouping_outcome = if blend_mode == BlendMode::FocusStack
        || (blend_mode == BlendMode::Panorama && alignment_mode == AlignmentMode::FlatArtwork)
    {
        focus_capture_groups_with_grouping(&image_data, &pairwise_matches, &global_homographies)
    } else {
        None
    };
    let station_topology = focus_station_grouping_outcome
        .as_ref()
        .and_then(|(groups, _, _)| {
            (groups.len() >= 2).then(|| {
                // For long scans this is the exact pre-pose topology already
                // used by `solve_focus_capture_group_poses`. Compact
                // multi-station runs do not enter that solver yet, so infer
                // here from their image-level world centres. This value is
                // shared by reporting and the default layered compositor.
                topology::run_topology_snapshot()
                    .filter(|topology| topology.row.len() == groups.len())
                    .unwrap_or_else(|| {
                        let inferred =
                            focus_station_topology(&image_data, groups, &global_homographies);
                        topology::record_run_topology(inferred.clone());
                        inferred
                    })
            })
        });
    if let (Some(recorder), Some((groups, _, grouping))) = (
        stack_report.as_ref(),
        focus_station_grouping_outcome.as_ref(),
    ) {
        // 需求 1.7: member paths, evidence kind, score, inlier count and inlier
        // spatial support of every Capture_Station.  Task 9.1 augments these
        // same production records with displacement-derived topology indices.
        let stations = groups
            .iter()
            .enumerate()
            .map(|(index, group)| {
                let evidence =
                    focus_station_binding_evidence(&group.members, &image_data, &pairwise_matches);
                let topology_ambiguous = station_topology
                    .as_ref()
                    .is_some_and(|topology| topology.ambiguous.binary_search(&index).is_ok());
                stack_report::StationRecord {
                    index,
                    row: station_topology
                        .as_ref()
                        .and_then(|topology| topology.row.get(index))
                        .map(|&row| row as usize),
                    column: station_topology
                        .as_ref()
                        .and_then(|topology| topology.column.get(index))
                        .map(|&column| column as usize),
                    topology_status: topology_ambiguous
                        .then(|| degradation::TOPOLOGY_INDEX_AMBIGUOUS.to_string()),
                    members: group
                        .members
                        .iter()
                        .filter_map(|&member| image_data.get(member))
                        .map(|image| image.filename.clone())
                        .collect(),
                    anchor: image_data
                        .get(group.anchor)
                        .map(|image| image.filename.clone())
                        .unwrap_or_default(),
                    evidence: evidence
                        .map(|evidence| stack_report::StationEvidenceRecord {
                            kind: StationEvidence::KIND.to_string(),
                            score: evidence.score,
                            inliers: evidence.inliers,
                            overlap_ncc: evidence.overlap_ncc,
                            spatial_support: evidence.spatial_support,
                            scale_ratio: evidence.scale_ratio,
                        })
                        .unwrap_or_default(),
                }
            })
            .collect::<Vec<_>>();
        // 需求 1.6: a source that satisfies the 需求 1.1 evidence with no other
        // source is reported by absolute path with its reason identifier, and the
        // remaining sources keep the grouping they would have had anyway.
        let isolated = excluded_source_records(
            &image_data,
            &grouping.isolated,
            degradation::GROUPING_NO_OVERLAP_EVIDENCE,
        );
        // 需求 1.4 / 1.10: every split position.
        let splits = grouping
            .splits
            .iter()
            .map(|split| stack_report::StationSplitRecord {
                station_index: split.component,
                position: split.position,
                reason: split.reason.to_string(),
                motion_ratio: split.motion_ratio,
            })
            .collect::<Vec<_>>();
        for split in &grouping.splits {
            degradation::record_run_degradation(
                split.reason,
                serde_json::json!({
                    "component": split.component,
                    "position": split.position,
                    "motion_ratio": split.motion_ratio,
                }),
            );
        }
        for record in &isolated {
            degradation::record_run_degradation(
                degradation::GROUPING_NO_OVERLAP_EVIDENCE,
                serde_json::json!({ "path": record.path }),
            );
        }
        let topology_report = station_topology.as_ref().map(|topology| {
            for &station_index in &topology.ambiguous {
                degradation::record_run_degradation(
                    degradation::TOPOLOGY_INDEX_AMBIGUOUS,
                    serde_json::json!({ "station_index": station_index }),
                );
            }
            if topology.truncated_count > 0 {
                degradation::record_run_degradation(
                    degradation::TOPOLOGY_CANDIDATES_TRUNCATED,
                    serde_json::json!({
                        "truncated_count": topology.truncated_count,
                        "truncation_min_score": topology.truncation_min_score,
                    }),
                );
            }
            stack_report::TopologyReport::from(topology)
        });
        recorder.update(|report| {
            report.grouping.station_count = stations.len();
            report.grouping.stations = stations;
            report.grouping.isolated = isolated;
            report.grouping.splits = splits;
            if let Some(topology) = topology_report {
                report.topology = topology;
            }
        });
    }
    #[cfg(test)]
    let source_raw_pair_diagnostics =
        if std::env::var_os("RAW_EDITOR_STACK_SOURCE_RAW_LAYER_DIAGNOSTICS").is_some() {
            focus_station_grouping_outcome
                .as_ref()
                .map(|(groups, _, _)| {
                    source_raw_layer_pair_diagnostics(&image_data, &pairwise_matches, groups)
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
    #[cfg(not(test))]
    let source_raw_pair_diagnostics = Vec::<stack_report::SourceRawPairDiagnosticRecord>::new();
    #[cfg(test)]
    let source_raw_direct_diagnostics =
        if std::env::var_os("RAW_EDITOR_STACK_SOURCE_RAW_DIRECT_DIAGNOSTICS").is_some() {
            focus_station_grouping_outcome
                .as_ref()
                .map(|(groups, _, _)| source_raw_direct_pair_diagnostics(&image_data, groups))
                .unwrap_or_default()
        } else {
            Vec::new()
        };
    #[cfg(not(test))]
    let source_raw_direct_diagnostics =
        Vec::<stack_report::SourceRawDirectPairDiagnosticRecord>::new();
    let has_isolated_focus_sources = focus_station_grouping_outcome
        .as_ref()
        .is_some_and(|(_, _, grouping)| !grouping.isolated.is_empty());
    let focus_capture_groups = focus_station_grouping_outcome.map(|(groups, _, _)| groups);
    let focus_capture_group_ids = focus_capture_groups.as_ref().map(|groups| {
        let mut ids = HashMap::new();
        for (group_index, group) in groups.iter().enumerate() {
            let group_id = u8::try_from(group_index + 1)
                .expect("focus stack source limit keeps group ids in u8");
            for &image_index in &group.members {
                ids.insert(image_data[image_index].id, group_id);
            }
        }
        ids
    });

    if ordered_indices.len() < 2 {
        return Err("Could not find a connected sequence of at least two images.".to_string());
    }

    let ordered_filenames: Vec<_> = ordered_indices
        .iter()
        .map(|&i| {
            Path::new(&image_data[i].filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string()
        })
        .collect();
    println!("Stitching order determined: {:?}", ordered_filenames);
    let _ = app_handle.emit(
        progress_event,
        format!("Stitching order: {}", ordered_filenames.join(" -> ")),
    );
    let ordered_paths = ordered_indices
        .iter()
        .map(|&index| image_data[index].filename.clone())
        .collect::<Vec<_>>();

    let mut retained_full_images = HashMap::new();
    for image in &mut image_data {
        if let Some(full_image) = image.full_image.take() {
            retained_full_images.insert(image.id, full_image);
        }
    }
    let mut stitched_images_info: Vec<&ImageInfo> =
        ordered_indices.iter().map(|&i| &image_data[i]).collect();
    if alignment_mode == AlignmentMode::FlatArtwork {
        // A planar artwork mosaic benefits from a spatial render walk: every
        // new source should overlap pixels already materialised whenever the
        // solved geometry permits it. A plain y/x sort can start an entire
        // row with disjoint rectangles, which leaves those first layers with
        // no overlap from which to estimate a low-frequency tone field.
        let render_order = flat_artwork_overlap_render_order(
            &image_data,
            &ordered_indices,
            &global_homographies,
            focus_capture_group_ids.as_ref(),
        );
        stitched_images_info = render_order.iter().map(|&i| &image_data[i]).collect();
    } else if let Some(group_ids) = focus_capture_group_ids.as_ref() {
        order_focus_render_sources_by_capture_group(&mut stitched_images_info, group_ids);
    }
    let unstitched_count = image_data.len() - stitched_images_info.len();
    if unstitched_count > 0 {
        let stitched_indices = ordered_indices.iter().copied().collect::<HashSet<_>>();
        let unstitched_filenames = image_data
            .iter()
            .enumerate()
            .filter(|(index, _)| !stitched_indices.contains(index))
            .map(|(_, image)| {
                Path::new(&image.filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();
        let warning_msg = format!(
            "{} image(s) could not be aligned with the selected set: {}.",
            unstitched_count,
            unstitched_filenames.join(", ")
        );
        println!("{}", warning_msg);
        let _ = app_handle.emit(progress_event, warning_msg);
        let redundant_focus_sources = scalable_stack
            && focus_stack
            && focus_unstitched_sources_are_redundant(
                &image_data,
                &pairwise_matches,
                &stitched_indices,
            );
        if redundant_focus_sources {
            let message = "Continuing without weak interior focus source(s); verified neighbouring captures preserve their coverage.";
            println!("{}", message);
            let _ = app_handle.emit(progress_event, message);
        }
        let search_description = if exhaustive_search {
            "the full pairwise search"
        } else {
            "the bounded candidate search"
        };
        if !redundant_focus_sources {
            return Err(if scalable_stack && focus_stack {
                format!(
                    "Focus-stack geometry was rejected after strict feature/RANSAC retries: {unstitched_count} image(s) have no verified overlap in {search_description}. No unsafe collage was rendered; select one contiguous scene or ensure each layer has enough visual overlap."
                )
            } else if scalable_stack {
                format!(
                    "Could not align all selected panorama images; {unstitched_count} image(s) have no verified overlap in {search_description}. The selection may contain multiple independent scenes. Select one contiguous scene or ensure consecutive images overlap."
                )
            } else {
                format!(
                    "Could not align all selected images; {unstitched_count} image(s) have no verified overlap. The selection may contain multiple independent scenes; select one contiguous scene and retry."
                )
            });
        }
    }
    println!(
        "Global homography calculation completed in {:.2?}\n",
        start_time.elapsed()
    );

    let (full_canvas_width, full_canvas_height) = if blend_mode == BlendMode::FocusStack {
        stitching::output_canvas_dimensions_with_focus_warp(
            &stitched_images_info,
            &global_homographies,
            projection,
            focus_layer_warp.as_ref(),
        )
    } else {
        stitching::output_canvas_dimensions(&stitched_images_info, &global_homographies, projection)
    };
    if full_canvas_width == 0 || full_canvas_height == 0 {
        return Err("The aligned panorama canvas is empty or invalid.".to_string());
    }
    // The acceptance harness can persist the expensive alignment result for
    // offline ROI diagnostics.  This is compiled only into test binaries so
    // the production path has no environment-controlled behavior.
    #[cfg(test)]
    if let Some(cache_path) = std::env::var_os("RAW_EDITOR_STACK_ALIGNMENT_CACHE") {
        write_alignment_cache(
            Path::new(&cache_path),
            &image_data,
            &ordered_indices,
            &pairwise_matches,
            &global_homographies,
            focus_layer_warp.as_ref(),
            focus_capture_groups.as_deref(),
            projection,
            full_canvas_width,
            full_canvas_height,
        )?;
    }
    #[cfg(test)]
    if std::env::var_os("RAW_EDITOR_STACK_ALIGNMENT_ONLY").is_some() {
        println!(
            "Alignment-only acceptance: skipping source rendering after {}x{} geometry validation",
            full_canvas_width, full_canvas_height
        );
        return Ok(StitchOutcome {
            image: DynamicImage::ImageRgb32F(Rgb32FImage::new(1, 1)),
            coverage: None,
            full_canvas_width,
            full_canvas_height,
            render_scale: 1.0,
            ordered_paths,
            stack_report_path: None,
        });
    }
    let render_scale = if blend_mode == BlendMode::Panorama {
        memory_safe_panorama_render_scale(full_canvas_width, full_canvas_height)
    } else {
        1.0
    };
    #[cfg(test)]
    let render_scale = std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_RENDER_SCALE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && (0.02..=1.0).contains(value))
        .unwrap_or(render_scale);
    let render_image_data = (render_scale < 1.0).then(|| {
        stitched_images_info
            .iter()
            .map(|image| scaled_render_image_info(image, render_scale))
            .collect::<Vec<_>>()
    });
    let render_images_info = render_image_data.as_ref().map_or_else(
        || stitched_images_info.clone(),
        |images| images.iter().collect::<Vec<_>>(),
    );
    #[cfg(test)]
    let mut render_images_info = render_images_info;
    #[cfg(test)]
    if std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_GROUP_REPRESENTATIVES").is_some()
        && let Some(group_ids) = focus_capture_group_ids.as_ref()
    {
        let mut seen = HashSet::new();
        render_images_info.retain(|image| {
            group_ids
                .get(&image.id)
                .is_some_and(|group| seen.insert(*group))
        });
        println!(
            "Acceptance geometry/illumination preview: rendering {} capture-group representative(s)",
            render_images_info.len()
        );
    }
    let scaled_global_homographies;
    let render_homographies = if render_scale < 1.0 {
        scaled_global_homographies =
            scaled_source_render_homographies(&global_homographies, render_scale);
        let (render_width, render_height) = stitching::output_canvas_dimensions(
            &render_images_info,
            &scaled_global_homographies,
            projection,
        );
        let render_percentage = render_scale * 100.0;
        let message = format!(
            "Large canvas {full_canvas_width}x{full_canvas_height}; pre-scaling sources and rendering a memory-safe {render_width}x{render_height} result ({render_percentage:.0}%)."
        );
        println!("{message}");
        let _ = app_handle.emit(progress_event, &message);
        &scaled_global_homographies
    } else {
        &global_homographies
    };

    let start_time = Instant::now();
    let _ = app_handle.emit(progress_event, "Warping and blending images...");
    println!("Warping and blending images with progressive optimal seams...");

    let mut load_render_image = |image: &ImageInfo| {
        ensure_focus_memory_available(rss_sampler.as_ref())?;
        let mut rendered = if let Some(full_image) = retained_full_images.remove(&image.id) {
            if full_image.dimensions() == image.dimensions() {
                Ok(full_image)
            } else {
                Ok(image::imageops::resize(
                    &full_image,
                    image.width,
                    image.height,
                    image::imageops::FilterType::Triangle,
                ))
            }
        } else {
            load_prepared_stack_source(&image.filename, &settings, raw_exposure_reference)
                .map(|source| source_to_render_rgb32f(source.image, image.width, image.height))
        }?;
        if is_raw_file(&image.filename) {
            if let Some(gain) = std::env::var("RAW_EDITOR_STACK_RAW_FIXED_GAIN")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)
            {
                rendered
                    .as_mut()
                    .par_iter_mut()
                    .for_each(|channel| *channel = (*channel * gain).clamp(0.0, 1.0));
            } else if let Some(gain) = shared_raw_display_gain {
                rendered
                    .as_mut()
                    .par_iter_mut()
                    .for_each(|channel| *channel = (*channel * gain).clamp(0.0, 1.0));
            } else if std::env::var_os("RAW_EDITOR_SKIP_STACK_RAW_EXPOSURE_NORMALIZATION").is_none()
            {
                normalize_stack_raw_display_exposure(&mut rendered);
            }
        }
        Ok(rendered)
    };
    let sequence_gap_aware = blend_mode == BlendMode::FocusStack
        && pairwise_matches
            .values()
            .any(|match_info| match_info.sequence_bridge);
    // Set once the Virtual_Tile stage replaces the source-stage station poses
    // (a rendering-prior repair or the authoritative station solve): the
    // stations then no longer sit where the source-stage canvas above was
    // measured.
    let mut station_poses_replaced = false;
    #[allow(clippy::type_complexity)]
    let mut quality_planes: Option<(
        GrayImage,
        Vec<u16>,
        Vec<f32>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        u64,
        (f64, f64),
    )> = None;
    let mut owner_reverse_lookup_failures = stack_report::OwnerReverseLookupFailures::default();
    let mut confidence_scores = stack_report::ConfidenceScoreDistribution {
        winner_histogram: vec![0; 21],
        runner_up_histogram: vec![0; 21],
        ..Default::default()
    };
    // The output Coverage_Mask the export writes as alpha (需求 10.6).
    let mut output_coverage: Option<GrayImage> = None;
    let mut quality_sources: Option<Vec<quality_gate_runner::QualitySource>> = None;
    let mut diagnostics_world_origin: Option<(f64, f64)> = None;
    let mut panorama = match blend_mode {
        // A flat artwork scan is a source mosaic, not a photographic panorama.
        // Keep one native source owner for each output pixel so the seam pass
        // cannot average displaced ink, weave, or brush texture. The existing
        // detail-preserving compositor uses its tiled path automatically for
        // this canvas size and only applies a low-frequency illumination field.
        BlendMode::Panorama if alignment_mode == AlignmentMode::FlatArtwork => {
            println!("Warping and blending flat artwork with detail-preserving ownership...");
            crate::panorama_utils::mosaic::detail_preserving_mosaic(
                &render_images_info,
                render_homographies,
                projection,
                focus_capture_group_ids.as_ref(),
                // FlatArtwork is a position mosaic even when grouping found a
                // single connected component.  Keep every source eligible for
                // a geometry-aware seam; otherwise the streaming compositor
                // treats later tiles as focus planes and leaves rectangular
                // source ownership blocks.
                true,
                app_handle.clone(),
                progress_event,
                &mut load_render_image,
            )
        }
        // A PSD set is a planar mosaic of already-fused stations. Average
        // nothing across overlaps: refine the poses natively, pick the tile
        // that delivers the most output-pixel detail per region and blend
        // only the low-frequency band across seams.
        BlendMode::Panorama if psd_source_scan && projection == Projection::Planar => {
            println!("Warping and blending PSD scan tiles with sharpness-aware ownership...");
            let mut load_scan_source = |image: &ImageInfo| {
                ensure_focus_memory_available(rss_sampler.as_ref())?;
                let prepared =
                    load_prepared_stack_source(&image.filename, &settings, raw_exposure_reference)?;
                let valid = crate::panorama_utils::scan_mosaic::opaque_source_mask(
                    &prepared.image,
                    image.width,
                    image.height,
                );
                let rgb = source_to_render_rgb32f(prepared.image, image.width, image.height);
                Ok(crate::panorama_utils::scan_mosaic::ScanSource { rgb, valid })
            };
            crate::panorama_utils::scan_mosaic::planar_scan_mosaic(
                &render_images_info,
                render_homographies,
                app_handle.clone(),
                progress_event,
                &mut load_scan_source,
            )
            .map(|composed| {
                // The refined poses and the detail-preserving output grid
                // define the delivered canvas.
                station_poses_replaced = true;
                output_coverage = Some(composed.coverage);
                composed.image
            })
        }
        BlendMode::Panorama => stitching::progressive_seam_stitcher(
            &render_images_info,
            render_homographies,
            projection,
            alignment_mode == AlignmentMode::FlatArtwork || psd_source_scan,
            app_handle.clone(),
            progress_event,
            &mut load_render_image,
        ),
        BlendMode::FocusStack => {
            // Collapse every evidence-backed focus bracket before composing
            // stations.  The virtual-tile path is deliberately selected by
            // the inferred groups, never by filename ranges or a fixture's
            // expected station count.
            //
            // 需求 15.2: the compositor comes from a setting, not from an
            // environment variable, and an unset setting is the default layered
            // path.  需求 15.10: every comparison path is therefore off unless a
            // developer opts in.
            let configured_compositor =
                compositor_choice.expect("focus-stack compositor choice resolved above");
            // A long moving scan can produce Capture_Stations that are (almost)
            // all single frames — grouping may still pair a near-duplicate
            // frame — or leave a source isolated from the station graph. There
            // is no local focus bracket to fuse at most positions in that
            // shape; treating every frame as a Virtual_Tile and then re-solving
            // their poses compounds projective closure drift and multiplies the
            // runtime, while omitting an isolated source loses a real input.
            // Keep the default layered compositor for multi-frame stations, but
            // route this source-backed shape through the proven global focus
            // compositor. An explicit persisted compositor still wins, so
            // comparison runs remain reproducible.
            let source_scan_legacy_fallback = settings.stack_compositor.is_none()
                && configured_compositor == StackCompositorChoice::LayeredVirtualTile
                && (has_isolated_focus_sources
                    || focus_capture_groups
                        .as_deref()
                        .is_some_and(capture_groups_form_singleton_scan));
            let compositor_choice = if source_scan_legacy_fallback {
                println!(
                    "  - Singleton Capture_Station scan detected; preserving source global poses with the legacy focus compositor"
                );
                StackCompositorChoice::LegacySingleLayerMosaic
            } else {
                configured_compositor
            };
            // The legacy compositor's shifted-mosaic path is the safe route for
            // a moving source scan.  In this shape the grouping graph may still
            // collapse the scan into one connected component, so the regular
            // compact-stack heuristic cannot infer the motion from the group
            // count alone.  Mark this explicitly as sequence-gap-aware so the
            // detail-preserving compositor keeps every source on its solved
            // global pose and avoids the strict compact-station rejection loop.
            let compositor_sequence_gap_aware = sequence_gap_aware || source_scan_legacy_fallback;
            println!(
                "  - Focus compositor sequence-gap aware: {}",
                compositor_sequence_gap_aware
            );
            if compositor_choice.is_comparison_path() {
                println!(
                    "  - Focus compositor comparison path selected: {}",
                    compositor_choice.as_identifier()
                );
            }
            let station_count = focus_capture_groups.as_ref().map(|groups| groups.len());
            let path_selection = select_and_record_run_path(
                station_count,
                compositor_choice,
                stack_report.as_deref(),
            );
            if !path_selection.use_virtual_tiles {
                stitching::focus_stack_stitcher(
                    &render_images_info,
                    render_homographies,
                    projection,
                    if render_scale < 1.0 {
                        // Test-only reduced renders validate global layout. Local
                        // focus bands are expressed in full-resolution coordinates
                        // and are exercised by the normal 1:1 acceptance render.
                        None
                    } else {
                        focus_layer_warp.as_ref()
                    },
                    focus_capture_group_ids.as_ref(),
                    compositor_sequence_gap_aware,
                    app_handle.clone(),
                    progress_event,
                    &mut load_render_image,
                )
            } else {
                let groups = focus_capture_groups.as_ref().expect("checked above");
                // The focus grouping indices refer to the source list, while
                // the compositor must use the render-scaled metadata.  Using
                // `image_data` here silently mixed full-resolution dimensions
                // with scaled homographies and produced oversized, distorted
                // virtual tiles in reduced renders.
                let render_by_id = render_images_info
                    .iter()
                    .map(|image| (image.id, *image))
                    .collect::<HashMap<_, _>>();
                let group_images = groups
                    .iter()
                    .map(|group| {
                        group
                            .members
                            .iter()
                            .filter_map(|&index| {
                                let source = image_data.get(index)?;
                                let image = render_by_id.get(&source.id).copied()?;
                                render_homographies.contains_key(&image.id).then_some(image)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let group_slices = group_images.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let groups_complete = group_images
                    .iter()
                    .zip(groups.iter())
                    .all(|(members, group)| members.len() == group.members.len());
                let tile_focus_warp = if render_scale < 1.0 {
                    None
                } else {
                    focus_layer_warp.as_ref()
                };
                // Fuse in a station's own anchor grid. Baking uncertain global
                // placement into these pixels made the subsequent station
                // solve try to undo a distorted image through a bounded repair.
                let anchor_frames = if compositor_choice
                    == StackCompositorChoice::LayeredVirtualTile
                    && projection == Projection::Planar
                {
                    Some(
                        focus_anchor_render_frames(
                            groups,
                            &image_data,
                            render_homographies,
                            tile_focus_warp,
                        )
                        .ok_or("cannot establish camera-station anchor coordinates")?,
                    )
                } else {
                    None
                };
                let anchor_layout = anchor_frames
                    .as_ref()
                    .map(|frames| frames.anchor_to_world.clone());
                let mut station_render_homographies = anchor_frames.as_ref().map_or_else(
                    || render_homographies.clone(),
                    |frames| frames.homographies.clone(),
                );
                let mut station_focus_warp = anchor_frames.as_ref().map_or_else(
                    || tile_focus_warp.cloned(),
                    |frames| frames.focus_warp.clone(),
                );
                let geometries = if groups_complete {
                    group_slices
                        .iter()
                        .map(|group| {
                            stitching::focus_stack_virtual_tile_geometry(
                                group,
                                &station_render_homographies,
                                projection,
                                station_focus_warp.as_ref(),
                            )
                        })
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| {
                            "one or more inferred groups have invalid bounds".to_string()
                        })
                } else {
                    Err("one or more inferred groups lack a rendered source".to_string())
                };
                if let Ok(mut geometries) = geometries {
                    // Feed the default layered compositor in the canonical
                    // row/column order inferred by Capture_Topology_Model.
                    // Continuous projected centres, source names and the
                    // candidate source path must not break topology ties or
                    // change seam accumulation (任务 9.10, 需求 14.6).
                    let station_topology = station_topology
                        .as_ref()
                        .expect("multi-station layered composition requires inferred topology");
                    let group_order =
                        focus_source_order_group_compositor_order(groups.len(), station_topology);
                    println!(
                        "  - Focus virtual-tile compositor order: {}",
                        group_order
                            .iter()
                            .map(|group_index| (group_index + 1).to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    println!(
                        "  - Composing {} focus virtual tile(s) from inferred capture groups",
                        geometries.len()
                    );

                    let mut tile_infos = Vec::with_capacity(group_order.len());
                    let mut tile_homographies = HashMap::with_capacity(group_order.len());
                    let mut tile_group_indices = HashMap::with_capacity(group_order.len());
                    for group_index in 0..groups.len() {
                        let geometry = geometries[group_index];
                        let tile_id = usize::MAX.saturating_sub(group_index);
                        let source_ids = group_slices[group_index]
                            .iter()
                            .map(|image| image.id.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        let placement = anchor_layout
                            .as_ref()
                            .map_or(geometry.tile_to_world, |anchors| {
                                anchors[group_index] * geometry.tile_to_world
                            });
                        tile_homographies.insert(tile_id, placement);
                        tile_group_indices.insert(tile_id, group_index);
                        tile_infos.push(ImageInfo {
                            id: tile_id,
                            filename: format!(
                                "virtual://focus-group-{}-{}",
                                group_index + 1,
                                source_ids
                            ),
                            width: geometry.width,
                            height: geometry.height,
                            // Replaced by the finest probe plane once rendered.
                            alignment_image: GrayImage::new(1, 1),
                            full_image: None,
                            scale_factor: 1.0,
                            focal_length_35mm: None,
                            source_footprint: None,
                            overview_reference: false,
                            features: Vec::new(),
                            top_features: Vec::new(),
                            foreground_range: None,
                            foreground_mask: None,
                            horizontal_edge_rows: Vec::new(),
                            vertical_edge_columns: Vec::new(),
                        });
                    }

                    // The production station-relation pass works on the actual
                    // focus-fused Virtual_Tiles, not on rectangular source
                    // buffers. Render one station at a time, keep only its
                    // area-averaged Coverage_Mask planes, and discard the
                    // fused pixels before moving to the next station. This
                    // preserves source ownership and does not increase the
                    // compositor's resident-tile set.
                    //
                    // Rendering priors may be repaired below, so every later
                    // render (including the compositor's) reads these owned
                    // copies rather than the source-stage poses.
                    let mut pyramid = VirtualTileMatchPyramid::new(
                        virtual_tile_pyramid_factors(
                            geometries
                                .iter()
                                .map(|geometry| geometry.width.max(geometry.height))
                                .min()
                                .unwrap_or(0),
                        ),
                        tile_infos.len(),
                    );
                    for (group_index, tile) in tile_infos.iter_mut().enumerate() {
                        let (gray, coverage) = render_virtual_tile_match_plane(
                            group_slices[group_index],
                            &station_render_homographies,
                            projection,
                            station_focus_warp.as_ref(),
                            group_index,
                            geometries[group_index],
                            app_handle.clone(),
                            progress_event,
                            &mut load_render_image,
                        )?;
                        install_virtual_tile_match_planes(
                            tile,
                            &mut pyramid,
                            group_index,
                            &gray,
                            &coverage,
                            &brief_pairs,
                        )?;
                    }
                    #[cfg(test)]
                    if std::env::var_os("RAW_EDITOR_STACK_STRUCTURAL_ORACLE").is_some() {
                        if tile_infos.len() != 2 {
                            return Err(format!(
                                "structural oracle requires exactly two Virtual_Tiles, got {}",
                                tile_infos.len()
                            ));
                        }
                        let record =
                            structural_oracle::run_structural_oracle(&tile_infos, &pyramid);
                        let report_dir = std::env::var_os("RAW_EDITOR_STACK_REPORT_DIR")
                            .map(PathBuf::from)
                            .ok_or_else(|| {
                                "RAW_EDITOR_STACK_REPORT_DIR is required by structural oracle"
                                    .to_string()
                            })?;
                        std::fs::write(
                            report_dir.join("structural-oracle.json"),
                            serde_json::to_vec_pretty(&record)
                                .map_err(|error| error.to_string())?,
                        )
                        .map_err(|error| error.to_string())?;
                        return Err(
                            "structural oracle harness stopped before production relation selection"
                                .to_string(),
                        );
                    }

                    let (mut tile_relations, mut tile_relation_candidates) =
                        virtual_tile_measure_prior_relations(
                            &tile_infos,
                            &pyramid,
                            &tile_homographies,
                            station_topology,
                            None,
                            false,
                        );

                    // A station whose rendering prior is grossly wrong cannot be
                    // reached by any bounded probe. Only candidates that join
                    // two different prior-consistent clusters get a prior-free
                    // covered-feature proposal; a proposal the coarse-to-fine
                    // probe verifies may move the unfixed cluster's prior, whose
                    // stations are then rendered once more and measured again
                    // around the repaired prior. A prior-free hypothesis enters
                    // the solver only after passing the complete shared final
                    // Station_Relation gate; weaker fits remain proposals only.
                    let mut prior_repair_records = Vec::new();
                    if tile_infos.len() >= 2 {
                        let authoritative_pairs = virtual_tile_authoritative_pairs(
                            &tile_relations,
                            &tile_infos,
                            pyramid.solver_coverages(),
                        );
                        let prior_free_pairs = virtual_tile_prior_free_candidate_pairs(
                            tile_infos.len(),
                            &authoritative_pairs,
                            station_topology,
                        );
                        let mut proposals = Vec::new();
                        for (left, right) in prior_free_pairs {
                            let mut feature_relation_passed = false;
                            if let Some((seed, feature_inliers)) = virtual_tile_prior_free_relation(
                                pyramid.feature_factors[left],
                                &tile_infos[left].features,
                                pyramid.feature_factors[right],
                                &tile_infos[right].features,
                                tile_infos[left].dimensions(),
                            ) {
                                let (verified, mut diagnostic) = virtual_tile_pyramid_station_match(
                                    left,
                                    right,
                                    &pyramid,
                                    &seed,
                                    VIRTUAL_TILE_SEED_PRIOR_FREE,
                                );
                                diagnostic.prior_free_inliers = feature_inliers;
                                let selection = select_virtual_tile_relation_hypothesis(
                                    &tile_infos,
                                    left,
                                    right,
                                    &verified,
                                    pyramid.solver_coverages(),
                                );
                                if let Some(selection) = selection.as_ref() {
                                    diagnostic.hypotheses = selection.outcomes.clone();
                                }
                                populate_station_relation_candidate_geometry(
                                    &mut diagnostic,
                                    &tile_infos,
                                    pyramid.solver_coverages(),
                                    &tile_homographies,
                                );
                                log_virtual_tile_candidate(&diagnostic, false);
                                if let Some(selection) =
                                    selection.filter(|selection| selection.passes)
                                {
                                    feature_relation_passed = true;
                                    // A BRIEF seed retains its established repair behaviour.
                                    tile_relations
                                        .entry((left, right))
                                        .or_default()
                                        .extend(verified.clone());
                                    proposals.push(VirtualTilePriorProposal {
                                        left,
                                        right,
                                        left_to_right: selection.relation.homography,
                                        inliers: selection.relation.inliers,
                                    });
                                } else if virtual_tile_prior_free_seed_fallback_allowed(&diagnostic)
                                {
                                    proposals.push(VirtualTilePriorProposal {
                                        left,
                                        right,
                                        left_to_right: seed,
                                        inliers: feature_inliers,
                                    });
                                }
                                replace_virtual_tile_candidate_record(
                                    &mut tile_relation_candidates,
                                    diagnostic,
                                );
                            }

                            // A prior-free feature seed is not guaranteed to find a
                            // basin when the station's rendering prior has drifted by
                            // an entire scan step. First reuse independently measured
                            // cross-Source_RAW relations to express a bounded local
                            // tile seed. This is still only a search basin: the
                            // Virtual_Tile matcher and shared final selector below
                            // remain the authority.
                            if !feature_relation_passed {
                                let source_seeds = match (
                                    tile_homographies.get(&tile_infos[left].id),
                                    tile_homographies.get(&tile_infos[right].id),
                                ) {
                                    (Some(left_tile_to_world), Some(right_tile_to_world)) => {
                                        virtual_tile_source_match_proposals(
                                            group_slices[left],
                                            group_slices[right],
                                            &pairwise_matches,
                                            &station_render_homographies,
                                            left_tile_to_world,
                                            right_tile_to_world,
                                            render_scale,
                                        )
                                    }
                                    _ => Vec::new(),
                                };
                                for source_seed in source_seeds {
                                    let (verified, mut diagnostic) =
                                        virtual_tile_pyramid_station_match(
                                            left,
                                            right,
                                            &pyramid,
                                            &source_seed.left_to_right,
                                            VIRTUAL_TILE_SEED_SOURCE_MATCH,
                                        );
                                    diagnostic.prior_free_inliers = source_seed.inliers;
                                    let selection = select_virtual_tile_relation_hypothesis(
                                        &tile_infos,
                                        left,
                                        right,
                                        &verified,
                                        pyramid.solver_coverages(),
                                    );
                                    if let Some(selection) = selection.as_ref() {
                                        diagnostic.hypotheses = selection.outcomes.clone();
                                    }
                                    populate_station_relation_candidate_geometry(
                                        &mut diagnostic,
                                        &tile_infos,
                                        pyramid.solver_coverages(),
                                        &tile_homographies,
                                    );
                                    log_virtual_tile_candidate(&diagnostic, false);
                                    replace_virtual_tile_candidate_record(
                                        &mut tile_relation_candidates,
                                        diagnostic,
                                    );
                                    let Some(selection) =
                                        selection.filter(|selection| selection.passes)
                                    else {
                                        continue;
                                    };
                                    feature_relation_passed = true;
                                    tile_relations
                                        .entry((left, right))
                                        .or_default()
                                        .extend(verified);
                                    proposals.push(VirtualTilePriorProposal {
                                        left,
                                        right,
                                        left_to_right: selection.relation.homography,
                                        inliers: selection.relation.inliers,
                                    });
                                    break;
                                }
                            }

                            // A prior-free feature or source-pair seed can still
                            // miss when the station's rendering prior has drifted
                            // by an entire scan step. In that case use the cached,
                            // Coverage_Mask-only ultra-coarse planes to propose a
                            // small set of bounded-similarity seeds. These are
                            // proposal basins only: every seed must still run the
                            // ordinary coarse-to-fine matcher and the shared final
                            // Station_Relation selector before it can enter either
                            // `tile_relations` or prior repair.
                            if !feature_relation_passed {
                                let global = pyramid
                                    .global_planes
                                    .get(left)
                                    .and_then(Option::as_ref)
                                    .zip(pyramid.global_planes.get(right).and_then(Option::as_ref))
                                    .map(|(left_plane, right_plane)| {
                                        virtual_tile_global_similarity_proposals(
                                            left_plane,
                                            right_plane,
                                        )
                                    })
                                    .unwrap_or_default();
                                let global_records = global
                                    .iter()
                                    .enumerate()
                                    .map(|(rank, proposal)| {
                                        stack_report::StationRelationGlobalProposalRecord {
                                            rank,
                                            score: proposal.score,
                                            overlap_samples: proposal.overlap_samples,
                                            left_factor: proposal.left_factor,
                                            right_factor: proposal.right_factor,
                                            translation_x_px: proposal.left_to_right[(0, 2)],
                                            translation_y_px: proposal.left_to_right[(1, 2)],
                                            scale: proposal.scale,
                                            rotation_degrees: proposal.rotation_degrees,
                                        }
                                    })
                                    .collect::<Vec<_>>();
                                for proposal in global {
                                    let (verified, mut diagnostic) =
                                        virtual_tile_pyramid_station_match(
                                            left,
                                            right,
                                            &pyramid,
                                            &proposal.left_to_right,
                                            VIRTUAL_TILE_SEED_GLOBAL_COVERAGE,
                                        );
                                    diagnostic.global_proposals = global_records.clone();
                                    let selection = select_virtual_tile_relation_hypothesis(
                                        &tile_infos,
                                        left,
                                        right,
                                        &verified,
                                        pyramid.solver_coverages(),
                                    );
                                    if let Some(selection) = selection.as_ref() {
                                        diagnostic.hypotheses = selection.outcomes.clone();
                                    }
                                    populate_station_relation_candidate_geometry(
                                        &mut diagnostic,
                                        &tile_infos,
                                        pyramid.solver_coverages(),
                                        &tile_homographies,
                                    );
                                    log_virtual_tile_candidate(&diagnostic, false);
                                    replace_virtual_tile_candidate_record(
                                        &mut tile_relation_candidates,
                                        diagnostic,
                                    );
                                    let Some(selection) =
                                        selection.filter(|selection| selection.passes)
                                    else {
                                        continue;
                                    };
                                    // The global score selected a basin only. The
                                    // measured hypothesis is the sole evidence that
                                    // is allowed to enter the authoritative graph.
                                    tile_relations
                                        .entry((left, right))
                                        .or_default()
                                        .extend(verified);
                                    proposals.push(VirtualTilePriorProposal {
                                        left,
                                        right,
                                        left_to_right: selection.relation.homography,
                                        inliers: selection.relation.inliers,
                                    });
                                    break;
                                }
                            }
                        }
                        let authoritative_pairs = virtual_tile_authoritative_pairs(
                            &tile_relations,
                            &tile_infos,
                            pyramid.solver_coverages(),
                        );
                        let station_tile_to_world = tile_infos
                            .iter()
                            .map(|tile| tile_homographies[&tile.id])
                            .collect::<Vec<_>>();
                        let repairs = plan_virtual_tile_prior_repairs(
                            tile_infos.len(),
                            &authoritative_pairs,
                            &proposals,
                            &station_tile_to_world,
                            station_topology,
                        );
                        let mut repaired = BTreeSet::new();
                        for repair in &repairs {
                            let station = repair.station;
                            // Evaluate the complete correction on copies before
                            // mutating the rendering priors.  A prior-free fit
                            // can be internally coherent while still implying
                            // an implausible world jump or scale change; such a
                            // proposal must never reach the renderer.
                            let mut candidate_homographies = station_render_homographies.clone();
                            let mut candidate_focus_warp = station_focus_warp.clone();
                            if !apply_virtual_tile_prior_repair(
                                group_slices[station],
                                &repair.world_correction,
                                &mut candidate_homographies,
                                candidate_focus_warp.as_mut(),
                            ) {
                                println!(
                                    "  - Virtual_Tile prior repair of station {station} rejected: a corrected source pose is not a valid planar pose"
                                );
                                continue;
                            }
                            let Some(candidate_geometry) =
                                stitching::focus_stack_virtual_tile_geometry(
                                    group_slices[station],
                                    &candidate_homographies,
                                    projection,
                                    candidate_focus_warp.as_ref(),
                                )
                            else {
                                println!(
                                    "  - Virtual_Tile prior repair of station {station} rejected: corrected geometry is invalid"
                                );
                                continue;
                            };
                            let old_center = transformed_point(
                                &station_tile_to_world[station],
                                Point2::new(
                                    f64::from(geometries[station].width) * 0.5,
                                    f64::from(geometries[station].height) * 0.5,
                                ),
                            );
                            let new_center = transformed_point(
                                &candidate_geometry.tile_to_world,
                                Point2::new(
                                    f64::from(candidate_geometry.width) * 0.5,
                                    f64::from(candidate_geometry.height) * 0.5,
                                ),
                            );
                            let center_shift_px = old_center
                                .zip(new_center)
                                .map_or(f64::INFINITY, |(old, new)| (new - old).norm());
                            let correction_scale_ratio = old_center
                                .and_then(|center| {
                                    virtual_tile_repair_local_scale_ratio(
                                        &repair.world_correction,
                                        center,
                                    )
                                })
                                .unwrap_or(f64::NAN);
                            if !virtual_tile_prior_repair_is_bounded(
                                center_shift_px,
                                correction_scale_ratio,
                            ) {
                                println!(
                                    "  - Virtual_Tile prior repair of station {station} rejected: correction is outside bounds (shift {center_shift_px:.1}px, scale {correction_scale_ratio:.4})"
                                );
                                continue;
                            }
                            if !apply_virtual_tile_prior_repair(
                                group_slices[station],
                                &repair.world_correction,
                                &mut station_render_homographies,
                                station_focus_warp.as_mut(),
                            ) {
                                println!(
                                    "  - Virtual_Tile prior repair of station {station} rejected: a corrected source pose is not a valid planar pose"
                                );
                                continue;
                            }
                            let Some(geometry) = stitching::focus_stack_virtual_tile_geometry(
                                group_slices[station],
                                &station_render_homographies,
                                projection,
                                station_focus_warp.as_ref(),
                            ) else {
                                return Err(format!(
                                    "Virtual tile station {station} has invalid bounds after its prior repair"
                                ));
                            };
                            println!(
                                "  - Virtual_Tile prior repair: station {station} moved {center_shift_px:.1}px (scale {correction_scale_ratio:.4}) onto station {} via prior-free relation {}->{} ({} verified inliers); rendering it once more",
                                repair.fixed_station,
                                repair.relation.0,
                                repair.relation.1,
                                repair.relation_inliers,
                            );
                            geometries[station] = geometry;
                            let tile = &mut tile_infos[station];
                            tile.width = geometry.width;
                            tile.height = geometry.height;
                            tile_homographies.insert(tile.id, geometry.tile_to_world);
                            let (gray, coverage) = render_virtual_tile_match_plane(
                                group_slices[station],
                                &station_render_homographies,
                                projection,
                                station_focus_warp.as_ref(),
                                station,
                                geometry,
                                app_handle.clone(),
                                progress_event,
                                &mut load_render_image,
                            )?;
                            install_virtual_tile_match_planes(
                                tile,
                                &mut pyramid,
                                station,
                                &gray,
                                &coverage,
                                &brief_pairs,
                            )?;
                            repaired.insert(station);
                            prior_repair_records.push(stack_report::StationPriorRepairRecord {
                                station,
                                fixed_station: repair.fixed_station,
                                relation_left: repair.relation.0,
                                relation_right: repair.relation.1,
                                relation_inliers: repair.relation_inliers,
                                center_shift_px,
                                correction_scale_ratio,
                            });
                        }
                        if !repaired.is_empty() {
                            station_poses_replaced = true;
                            let pairs = station_topology
                                .candidates
                                .iter()
                                .filter(|candidate| {
                                    repaired.contains(&candidate.left)
                                        || repaired.contains(&candidate.right)
                                })
                                .map(|candidate| {
                                    (
                                        candidate.left.min(candidate.right),
                                        candidate.left.max(candidate.right),
                                    )
                                })
                                .collect::<BTreeSet<_>>();
                            tile_relations.retain(|&(left, right), _| {
                                !pairs.contains(&(left.min(right), left.max(right)))
                            });
                            let (remeasured, remeasured_candidates) =
                                virtual_tile_measure_prior_relations(
                                    &tile_infos,
                                    &pyramid,
                                    &tile_homographies,
                                    station_topology,
                                    Some(&pairs),
                                    true,
                                );
                            tile_relations.extend(remeasured);
                            for record in remeasured_candidates {
                                replace_virtual_tile_candidate_record(
                                    &mut tile_relation_candidates,
                                    record,
                                );
                            }
                        }
                    }
                    if compositor_choice == StackCompositorChoice::LayeredVirtualTile {
                        // Authoritative stage 4 starts only now: each ImageInfo
                        // is one focus-fused Virtual_Tile and every feature was
                        // extracted through its Coverage_Mask.  The shared
                        // relation/tree/closure core knows this evidence kind,
                        // so requirement 6.2's single-Source_RAW discard does
                        // not apply and no support value is fabricated.
                        let mut closure = stack_report::ClosureReport::default();
                        let mut station_relations = stack_report::StationRelationsReport::default();
                        if tile_infos.len() >= 2 {
                            let selected_station_relations =
                                select_virtual_tile_relation_hypotheses(
                                    &tile_infos,
                                    &tile_relations,
                                    pyramid.solver_coverages(),
                                );
                            let mut solved = solve_virtual_tile_station_poses_with_report(
                                &tile_infos,
                                pyramid.solver_coverages(),
                                &selected_station_relations,
                                &tile_homographies,
                                station_topology,
                                &mut closure,
                                &mut station_relations,
                            );
                            // Calibration uses measured native correspondence evidence before
                            // the geometry-only photometric rejection removes a relation.
                            // Fitted colour still has its own ordinary-range/MAD/DeltaE gates.
                            let mut photometric_pairs = station_relations
                                .accepted
                                .iter()
                                .map(|r| (r.left, r.right))
                                .collect::<Vec<_>>();
                            photometric_pairs.extend(native_photometric_pairs(
                                &selected_station_relations,
                                &tile_infos,
                                pyramid.solver_coverages(),
                            ));
                            photometric_pairs.sort_unstable();
                            photometric_pairs.dedup();
                            #[cfg(test)]
                            if let Some(directory) =
                                std::env::var_os("RAW_EDITOR_NATIVE_PROOF_DIAGNOSTICS_DIR")
                            {
                                let records = photometric_pairs.iter().map(|&(left, right)| {
                                    serde_json::json!({"left":left,"right":right,"calibration_eligible":true,
                                        "native_round_trip_matches":selected_station_relations.get(&(left,right)).map(|r|r.native_verified_points.len()).unwrap_or(0)})
                                }).collect::<Vec<_>>();
                                std::fs::create_dir_all(&directory).unwrap();
                                std::fs::write(
                                    Path::new(&directory).join("native-photometric-relations.json"),
                                    serde_json::to_vec(&records).unwrap(),
                                )
                                .unwrap();
                            }
                            tone::record_native_photometric_topology(tone::ToneStationTopology {
                                seam_candidates: Vec::new(),
                                seam_candidate_transforms: Vec::new(),
                                stations: tile_infos
                                    .iter()
                                    .enumerate()
                                    .map(|(station, tile)| (tile.id, station))
                                    .collect(),
                                accepted: photometric_pairs
                                    .into_iter()
                                    .filter_map(|(left, right)| {
                                        Some((tile_infos.get(left)?.id, tile_infos.get(right)?.id))
                                    })
                                    .collect(),
                            });
                            if let Some(refined) = refine_virtual_tile_joint_geometry(
                                &tile_infos,
                                &selected_station_relations,
                                &solved,
                                &mut closure,
                                &mut station_relations,
                            ) {
                                solved = refined;
                                for rejected in &closure.rejected_nonlinear_constraints {
                                    if let Some(candidate) =
                                        tile_relation_candidates.iter_mut().find(|c| {
                                            (c.left, c.right) == (rejected.left, rejected.right)
                                        })
                                    {
                                        candidate.failure_stage =
                                            "native_joint_validation_p95".to_string();
                                    }
                                }
                            }
                            let source_station_members = group_slices
                                .iter()
                                .map(|members| {
                                    members
                                        .iter()
                                        .map(|image| absolute_source_path(&image.filename))
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>();
                            let accepted_pairs = station_relations
                                .accepted
                                .iter()
                                .map(|relation| (relation.left, relation.right))
                                .collect::<Vec<_>>();
                            let source_components = station_graph_components_from_members(
                                &source_station_members,
                                &accepted_pairs,
                            );
                            station_relations.connectivity.components = source_components.len();
                            station_relations.connectivity.component_member_counts =
                                source_components.iter().map(Vec::len).collect();
                            station_relations.connectivity.component_members = source_components;
                            let component_count = station_relations.connectivity.components;
                            // A locally measured adjacent pair can be removed
                            // by the strict global closure P95 gate while its
                            // own overlap remains valid for seam placement.
                            // Preserve that evidence for the compositor only;
                            // it must never become tone evidence or a pose
                            // constraint. The final Station_Relation gates
                            // remain unchanged here.
                            let authoritative_pairs = station_relations
                                .accepted
                                .iter()
                                .map(|relation| {
                                    (
                                        relation.left.min(relation.right),
                                        relation.left.max(relation.right),
                                    )
                                })
                                .collect::<BTreeSet<_>>();
                            // A visually overlapping pair can be rejected by
                            // the inferred row/column topology even when its
                            // local relation has enough inliers and spatial
                            // support to repair a known owner boundary.  Keep
                            // the normal adjacent-only rule, but let an
                            // explicit one-based pair request opt that pair
                            // into the seam-only evidence path.  This never
                            // changes the authoritative pose graph or the
                            // strict Station_Relation gates.
                            let explicitly_requested_seam_pairs =
                                std::env::var("RAW_EDITOR_STATION_LOCAL_RELATION_RESAMPLE_PAIR")
                                    .ok()
                                    .map(|value| {
                                        value
                                            .split(',')
                                            .filter_map(|part| {
                                                let (left, right) = part.trim().split_once('-')?;
                                                let left = left
                                                    .trim()
                                                    .parse::<usize>()
                                                    .ok()?
                                                    .checked_sub(1)?;
                                                let right = right
                                                    .trim()
                                                    .parse::<usize>()
                                                    .ok()?
                                                    .checked_sub(1)?;
                                                Some((left.min(right), left.max(right)))
                                            })
                                            .collect::<BTreeSet<_>>()
                                    })
                                    .unwrap_or_default();
                            let seam_refinement_station_pairs = tile_relation_candidates
                                .iter()
                                .filter(|candidate| {
                                    candidate.model_fitted
                                        && candidate.failure_stage == "fitted"
                                        && candidate.fitted_inliers >= STATION_RELATION_MIN_INLIERS
                                        && candidate.hull_support
                                            >= STATION_RELATION_MIN_SPATIAL_SUPPORT
                                        && candidate.hypotheses.iter().any(|hypothesis| {
                                            hypothesis.selected
                                                && hypothesis.rejection_reasons.is_empty()
                                        })
                                })
                                .filter_map(|candidate| {
                                    let left = candidate.left.min(candidate.right);
                                    let right = candidate.left.max(candidate.right);
                                    if left >= station_topology.row.len()
                                        || right >= station_topology.row.len()
                                        || left >= station_topology.column.len()
                                        || right >= station_topology.column.len()
                                    {
                                        return None;
                                    }
                                    let adjacent = (station_topology.row[left]
                                        == station_topology.row[right]
                                        && station_topology.column[left]
                                            .abs_diff(station_topology.column[right])
                                            == 1)
                                        || (station_topology.column[left]
                                            == station_topology.column[right]
                                            && station_topology.row[left]
                                                .abs_diff(station_topology.row[right])
                                                == 1);
                                    if !adjacent
                                        && !explicitly_requested_seam_pairs.contains(&(left, right))
                                    {
                                        return None;
                                    }
                                    Some((left, right))
                                })
                                .collect::<BTreeSet<_>>();
                            // Keep the owner-seam candidate list restricted to
                            // closure-rejected pairs, but retain locally fitted
                            // transforms for every adjacent pair.  A pair can
                            // be authoritative in the global pose graph and
                            // still need a small seam-only geometric
                            // refinement when the final-resolution owner edge
                            // crosses a painted stroke.  The compositor gates
                            // those authoritative refinements behind the
                            // explicit pair-scoped environment override, so
                            // this does not alter the accepted pose graph or
                            // widen the normal seam pass.
                            let seam_candidate_station_pairs = seam_refinement_station_pairs
                                .iter()
                                .copied()
                                .filter(|pair| !authoritative_pairs.contains(pair))
                                .collect::<BTreeSet<_>>();
                            let seam_candidates = seam_candidate_station_pairs
                                .iter()
                                .filter_map(|&(left, right)| {
                                    Some((tile_infos.get(left)?.id, tile_infos.get(right)?.id))
                                })
                                .collect::<Vec<_>>();
                            let seam_candidate_transforms = seam_refinement_station_pairs
                                .iter()
                                .filter_map(|&(left, right)| {
                                    let (hypotheses, reverse) = tile_relations
                                        .get(&(left, right))
                                        .map(|hypotheses| (hypotheses, false))
                                        .or_else(|| {
                                            tile_relations
                                                .get(&(right, left))
                                                .map(|hypotheses| (hypotheses, true))
                                        })?;
                                    let selection = select_virtual_tile_relation_hypothesis(
                                        &tile_infos,
                                        if reverse { right } else { left },
                                        if reverse { left } else { right },
                                        hypotheses,
                                        pyramid.solver_coverages(),
                                    )?;
                                    let homography = if reverse {
                                        selection.relation.homography.try_inverse()?
                                    } else {
                                        selection.relation.homography
                                    };
                                    let left_id = tile_infos.get(left)?.id;
                                    let right_id = tile_infos.get(right)?.id;
                                    Some((
                                        left_id,
                                        right_id,
                                        matrix3_row_major(&homography).map(f64::to_bits),
                                    ))
                                })
                                .collect::<Vec<_>>();
                            println!(
                                "  - Seam-only local transforms: {}",
                                seam_candidate_transforms.len()
                            );
                            // 需求 9.1: the Tone_Harmonizer samples only the
                            // overlaps of these accepted relations.
                            tone::record_run_topology(tone::ToneStationTopology {
                                stations: tile_infos
                                    .iter()
                                    .enumerate()
                                    .map(|(station, tile)| (tile.id, station))
                                    .collect(),
                                accepted: station_relations
                                    .accepted
                                    .iter()
                                    .filter_map(|relation| {
                                        Some((
                                            tile_infos.get(relation.left)?.id,
                                            tile_infos.get(relation.right)?.id,
                                        ))
                                    })
                                    .collect(),
                                seam_candidates,
                                seam_candidate_transforms,
                            });
                            station_relations.candidates = tile_relation_candidates.clone();
                            station_relations.source_raw_pair_diagnostics =
                                source_raw_pair_diagnostics.clone();
                            station_relations.source_raw_direct_diagnostics =
                                source_raw_direct_diagnostics.clone();
                            station_relations.summarize_candidate_measurements();
                            station_relations.prior_repairs = prior_repair_records.clone();
                            if let Some(recorder) = stack_report.as_ref() {
                                recorder.update(|report| {
                                    report.closure = closure.clone();
                                    report.station_relations = station_relations.clone();
                                });
                            }
                            #[cfg(test)]
                            if std::env::var_os(
                                "RAW_EDITOR_STACK_PAIR_HARNESS_STOP_AFTER_RELATIONS",
                            )
                            .is_some()
                            {
                                return Err("pair evidence harness stopped after Station_Relation"
                                    .to_string());
                            }
                            if component_count > 1 || solved.len() != tile_infos.len() {
                                return Err(geometry_disconnected_user_error(
                                    component_count.max(2),
                                ));
                            }
                            tile_homographies = solved;
                            if closure.geometry_model != "bounded_station_field" {
                                record_residual_warp_for_station_relations(
                                    &tile_infos,
                                    &tile_homographies,
                                    &selected_station_relations,
                                    &station_relations.accepted,
                                );
                            }
                            station_poses_replaced = true;
                            println!(
                                "  - Authoritative Virtual_Tile station solve accepted {} relation(s)",
                                station_relations.accepted.len()
                            );
                        }
                    } else if !tile_relations.is_empty() {
                        // Diagnostic comparison compositors retain their old
                        // image-space optimiser. Their authoritative source
                        // solve already ran before rendering.
                        let selected_tile_relations = select_virtual_tile_relation_hypotheses(
                            &tile_infos,
                            &tile_relations,
                            pyramid.solver_coverages(),
                        );
                        tile_homographies = optimize_focus_stack_global_homographies_with_reference(
                            &tile_infos,
                            &selected_tile_relations,
                            &tile_homographies,
                            0,
                        );
                    }

                    let tile_refs = group_order
                        .iter()
                        .map(|&group_index| &tile_infos[group_index])
                        .collect::<Vec<_>>();
                    let mut station_masks: Vec<Option<stitching::FocusStackTileMasks>> =
                        (0..tile_infos.len()).map(|_| None).collect();
                    let mut station_sampling_origins = vec![(0.0f64, 0.0f64); tile_infos.len()];
                    // The layered compositor asks for the same fused station
                    // during ownership, seam planning, and seam application.
                    // Keep a bounded set of pose-finalized virtual tiles so
                    // those passes do not rerun the full RAW station fusion.
                    // A miss deliberately keeps the old streaming behaviour
                    // once the pixel budget is reached.
                    let mut rendered_tile_cache = HashMap::<usize, Rgb32FImage>::new();
                    let mut rendered_tile_cache_pixels = 0u64;
                    // Native acceptance tiles are large enough that the
                    // reduced-render budget retains only the first few
                    // stations.  The seam pass then has to fuse the later
                    // stations again for every plan.  This run's native
                    // station union is about 550M pixels; retaining it costs
                    // roughly 6.6GB for RGB32F pixels and stays below the
                    // existing process memory gate together with the masks.
                    // Keep the smaller budget for reduced renders so their
                    // bounded-memory contract remains unchanged.
                    let rendered_tile_cache_budget = if render_scale >= 1.0 {
                        600_000_000
                    } else {
                        MAX_RETAINED_STACK_PIXELS
                    };
                    let mut load_tile = |tile: &ImageInfo| {
                        ensure_focus_memory_available(rss_sampler.as_ref())?;
                        let group_index =
                            tile_group_indices.get(&tile.id).copied().ok_or_else(|| {
                                format!("Missing virtual tile provenance {}", tile.id)
                            })?;
                        if let Some(cached) = rendered_tile_cache.get(&tile.id) {
                            return Ok(stitching::FocusTilePayload::from(cached.clone()));
                        }
                        let rendered = stitching::focus_stack_stitcher_unfilled(
                            group_slices[group_index],
                            &station_render_homographies,
                            projection,
                            station_focus_warp.as_ref(),
                            None,
                            group_index,
                            false,
                            app_handle.clone(),
                            progress_event,
                            &mut load_render_image,
                        )?;
                        let expected = geometries[group_index];
                        let (rendered_width, rendered_height) = rendered.image.dimensions();
                        if (rendered_width, rendered_height) != (expected.width, expected.height) {
                            return Err(format!(
                                "Virtual tile {} dimensions changed from {}x{} to {}x{}",
                                group_index,
                                expected.width,
                                expected.height,
                                rendered_width,
                                rendered_height
                            ));
                        }
                        // The render now also publishes the Coverage_Mask, the
                        // Ownership_Map and the Sharpness_Confidence it built at
                        // the focus selection's pixel write sites (需求 3.7 /
                        // 3.9 / 3.11 / 4.6).  The Virtual_Tile_Store (task 5.3)
                        // is what consumes the masks; the tile compositors below
                        // still take pixels only, so hand them the same pixel
                        // buffer as before.  The Focus_Fuser observations are
                        // attached to their Capture_Station here, because this is
                        // the first caller that knows which station was rendered
                        // (需求 3.5 / 3.8 / 3.9).
                        if let Some(masks) = rendered.masks.as_ref() {
                            station_masks[group_index] = Some(masks.clone());
                            station_sampling_origins[group_index] = rendered.sampling_origin;
                            let mut fusion = masks.fusion.clone();
                            fusion.station_index = group_index;
                            focus_fuser::record_run_station(fusion);

                            // Diagnostics are produced from the same masks and
                            // source metadata that drive the compositor.  The
                            // recorder is disabled by default, so these
                            // closures do not allocate or run during ordinary
                            // renders.
                            let members = group_slices[group_index]
                                .iter()
                                .map(|source| source.filename.clone())
                                .collect::<Vec<_>>();
                            diagnostics_recorder.record(
                                group_index,
                                DiagnosticItem::Members,
                                "json",
                                || serde_json::to_vec(&members).unwrap_or_default(),
                            );
                            let transforms = group_slices[group_index]
                                .iter()
                                .map(|source| {
                                    serde_json::json!({
                                        "path": source.filename,
                                        "width": source.width,
                                        "height": source.height,
                                    })
                                })
                                .collect::<Vec<_>>();
                            diagnostics_recorder.record(
                                group_index,
                                DiagnosticItem::FrameTransforms,
                                "json",
                                || serde_json::to_vec(&transforms).unwrap_or_default(),
                            );
                            diagnostics_recorder.record(
                                group_index,
                                DiagnosticItem::ResidualField,
                                "json",
                                || {
                                    serde_json::to_vec(&residual_warp::run_report_snapshot())
                                        .unwrap_or_default()
                                },
                            );
                            let ownership = masks.ownership.owners().to_vec();
                            diagnostics_recorder.record(
                                group_index,
                                DiagnosticItem::OwnershipMap,
                                "json",
                                || serde_json::to_vec(&ownership).unwrap_or_default(),
                            );
                            let confidence = masks.confidence.to_row_major();
                            diagnostics_recorder.record(
                                group_index,
                                DiagnosticItem::SharpnessConfidence,
                                "json",
                                || serde_json::to_vec(&confidence).unwrap_or_default(),
                            );
                            let coverage = masks.coverage.covered().to_vec();
                            diagnostics_recorder.record(
                                group_index,
                                DiagnosticItem::CoverageMask,
                                "json",
                                || serde_json::to_vec(&coverage).unwrap_or_default(),
                            );
                            diagnostics_recorder.record(
                                group_index,
                                DiagnosticItem::ToneField,
                                "json",
                                || {
                                    serde_json::to_vec(&tone::run_report_snapshot())
                                        .unwrap_or_default()
                                },
                            );
                        }
                        let source_samples = if compositor_choice
                            == StackCompositorChoice::LayeredVirtualTile
                        {
                            if let Some(masks) = rendered.masks.as_ref() {
                                let mut originals =
                                    Vec::with_capacity(masks.ownership.legend().len());
                                for path in masks.ownership.legend() {
                                    let source = group_slices[group_index]
                                        .iter()
                                        .find(|source| Path::new(&source.filename) == path)
                                        .ok_or_else(|| {
                                            format!("Missing original for owner {}", path.display())
                                        })?;
                                    let refinement = masks
                                        .source_refinements
                                        .get(&source.id)
                                        .filter(|refinement| refinement.source_sampling.is_some())
                                        .cloned()
                                        .ok_or_else(|| {
                                            format!(
                                                "Missing source sampling for {}",
                                                path.display()
                                            )
                                        })?;
                                    originals.push(stitching::FocusTileSource {
                                        image: load_render_image(source)?,
                                        refinement,
                                    });
                                }
                                Some(stitching::FocusTileSources {
                                    ownership: masks.ownership.clone(),
                                    origin: rendered.sampling_origin,
                                    originals,
                                })
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        let image = rendered.into_image();
                        let pixels =
                            u64::from(image.width()).saturating_mul(u64::from(image.height()));
                        if source_samples.is_none()
                            && rendered_tile_cache_pixels.saturating_add(pixels)
                                <= rendered_tile_cache_budget
                        {
                            rendered_tile_cache_pixels =
                                rendered_tile_cache_pixels.saturating_add(pixels);
                            rendered_tile_cache.insert(tile.id, image.clone());
                        }
                        Ok(stitching::FocusTilePayload {
                            image,
                            sources: source_samples,
                        })
                    };
                    if let Some(recorder) = stack_report.as_ref() {
                        let tiles = tile_infos
                            .iter()
                            .enumerate()
                            .map(|(group_index, tile)| stack_report::VirtualTileReport {
                                station_index: group_index,
                                width: tile.width,
                                height: tile.height,
                                provenance: group_slices[group_index]
                                    .iter()
                                    .map(|image| stack_report::ProvenanceRecord {
                                        path: image.filename.clone(),
                                        ..Default::default()
                                    })
                                    .collect(),
                                ..Default::default()
                            })
                            .collect::<Vec<_>>();
                        recorder.update(|report| report.virtual_tiles = tiles);
                    }
                    if let Some(recorder) = stack_report.as_ref() {
                        // 任务 13.4 / 需求 11.6: 0.0 unless a diagnostic session
                        // overrode it.
                        let sharpen = f64::from(stitching::layered_final_sharpen_amount());
                        recorder.update(|report| report.composition.final_sharpen_amount = sharpen);
                    }
                    let tile_result = match compositor_choice {
                        // Comparison: the streaming detail-preserving mosaic.
                        StackCompositorChoice::StreamingMosaic => {
                            let tile_groups = tile_infos
                                .iter()
                                .enumerate()
                                .map(|(index, tile)| {
                                    (tile.id, (index + 1).min(u8::MAX as usize) as u8)
                                })
                                .collect::<HashMap<_, _>>();
                            crate::panorama_utils::mosaic::detail_preserving_mosaic(
                                &tile_refs,
                                &tile_homographies,
                                Projection::Planar,
                                Some(&tile_groups),
                                false,
                                app_handle.clone(),
                                progress_event,
                                &mut |tile| load_tile(tile).map(|payload| payload.image),
                            )
                        }
                        // Comparison: the image-space minimum-cost seam with a
                        // low-frequency multiband transition.  It was the
                        // default until 任务 13.1; it is order dependent and it
                        // crops and sharpens the delivered canvas, which
                        // 需求 10.2 / 10.3 / 11.6 no longer allow.
                        StackCompositorChoice::ProgressiveSeamTile => {
                            stitching::progressive_seam_stitcher(
                                &tile_refs,
                                &tile_homographies,
                                Projection::Planar,
                                false,
                                app_handle.clone(),
                                progress_event,
                                &mut |tile| load_tile(tile).map(|payload| payload.image),
                            )
                        }
                        // Default (需求 15.1): hard source-interior ownership,
                        // the covered union as the canvas and no second-pass
                        // sharpen.
                        //
                        // `LegacySingleLayerMosaic` never reaches this match —
                        // it clears `use_virtual_tiles` above — so it is folded
                        // in here instead of panicking on an unreachable arm.
                        StackCompositorChoice::LayeredVirtualTile
                        | StackCompositorChoice::LegacySingleLayerMosaic => {
                            ensure_focus_memory_available(rss_sampler.as_ref())?;
                            // The ownership planes are retained for the Quality_Gate. A
                            // compositor error still reaches the fallback renderer below.
                            stitching::layered_virtual_tile_compositor_with_ownership(
                                &tile_refs,
                                &tile_homographies,
                                Projection::Planar,
                                app_handle.clone(),
                                progress_event,
                                &mut load_tile,
                            )
                            .map(|rendered| {
                                if let Some(recorder) = stack_report.as_ref() {
                                    // The layered compositor decodes one full-size
                                    // Virtual_Tile at a time; focus masks are retained
                                    // separately and contain no pixel buffer.
                                    recorder.record_max_resident_virtual_tiles(1);
                                }
                                let (width, height) = rendered.coverage.dimensions();
                                // The compositor names the owning Virtual_Tile.  Resolve that
                                // name back through the station's focus Ownership_Map so the
                                // quality gate samples the actual Source_RAW that supplied the
                                // pixel (output -> station -> virtual-tile coordinate -> raw
                                // ownership cell), rather than a synthetic virtual path.
                                let mut raw_sources = Vec::new();
                                let mut raw_owner_ids = vec![Vec::new(); group_slices.len()];
                                let mut next_owner = 1u16;
                                for station in 0..group_slices.len() {
                                    let station_to_world = tile_homographies
                                        .get(&tile_infos[station].id)
                                        .copied()
                                        .unwrap_or_else(Matrix3::identity);
                                    let station_origin = station_sampling_origins
                                        .get(station)
                                        .copied()
                                        .unwrap_or((0.0, 0.0));
                                    let station_crop = Matrix3::new(
                                        1.0,
                                        0.0,
                                        -station_origin.0,
                                        0.0,
                                        1.0,
                                        -station_origin.1,
                                        0.0,
                                        0.0,
                                        1.0,
                                    );
                                    // The focus renderer may reorder a compact station before
                                    // building its Ownership_Map (the sharpest valid frame is
                                    // the anchor).  Quality-Gate owner ids must follow that
                                    // published legend exactly; iterating the original grouping
                                    // order here would attribute pixels from, for example,
                                    // DSC_6483 to the DSC_6484 owner slot.  Keep a conservative
                                    // fallback for an unattributed station, but use the mask
                                    // legend whenever it exists.
                                    let ordered_sources = station_masks
                                        .get(station)
                                        .and_then(Option::as_ref)
                                        .map(|masks| {
                                            masks
                                                .ownership
                                                .legend()
                                                .iter()
                                                .filter_map(|path| {
                                                    group_slices[station]
                                                        .iter()
                                                        .find(|source| {
                                                            PathBuf::from(&source.filename) == *path
                                                        })
                                                })
                                                .collect::<Vec<_>>()
                                        })
                                        .filter(|sources| {
                                            sources.len() == group_slices[station].len()
                                        })
                                        .unwrap_or_else(|| group_slices[station].iter().collect());
                                    for (source_index, source) in ordered_sources.iter().enumerate() {
                                        let owner = next_owner;
                                        raw_owner_ids[station].push(owner);
                                        let native_displacement = station_masks
                                            .get(station)
                                            .and_then(Option::as_ref)
                                            .and_then(|masks| {
                                                masks
                                                    .native_displacements
                                                    .get(source_index)
                                                    .and_then(Option::as_ref)
                                                    .cloned()
                                            });
                                        // Keep station provenance even when no
                                        // native residual was needed. The
                                        // Quality_Gate reference must replay
                                        // the ordinary raw -> station-layer ->
                                        // final-tile sampling chain; using
                                        // `None` here would compare the output
                                        // against a sharper one-stage raw
                                        // sample and under-report gradient.
                                        let native_warp =
                                            station_render_homographies
                                                .get(&source.id)
                                                .copied()
                                                .map(|source_to_station| {
                                                    quality_gate_runner::NativeSourceWarp {
                                                        displacement: native_displacement,
                                                        source_to_station,
                                                        station_to_world,
                                                    }
                                                });
                                        raw_sources.push(quality_gate_runner::QualitySource {
                                            owner,
                                            path: source.filename.clone(),
                                            geometry: quality_gate::SourceGeometry {
                                                member_to_anchor: Matrix3::identity(),
                                                tile_to_world: station_to_world
                                                    * station_crop
                                                    * station_render_homographies
                                                        .get(&source.id)
                                                        .copied()
                                                        .unwrap_or_else(Matrix3::identity),
                                                station_id: station,
                                            },
                                            dimensions: (source.width, source.height),
                                            native_warp,
                                            native_registration: station_masks
                                                .get(station)
                                                .and_then(Option::as_ref)
                                                .and_then(|masks| {
                                                    masks.source_refinements.get(&source.id)
                                                })
                                                .and_then(|refinement| {
                                                    Some(quality_gate_runner::NativeReferenceMap {
                                                        world_to_station: station_to_world
                                                            .try_inverse()?,
                                                        station_to_world,
                                                        station_origin,
                                                        refinement: refinement.clone(),
                                                    })
                                                }),
                                        });
                                        next_owner = next_owner.saturating_add(1);
                                    }
                                }
                                let mut ownership = rendered.ownership.owners().to_vec();
                                let mut confidence = vec![f32::NAN; ownership.len()];
                                let mut textured = vec![0u8; ownership.len()];
                                // 0..=100 stores owner shortfall in hundredths;
                                // 255 marks a covered pixel whose reverse lookup
                                // could not establish a Source_RAW owner.
                                let mut owner_shortfall = vec![u8::MAX; ownership.len()];
                                let mut owner_disagreement = vec![u8::MAX; ownership.len()];
                                #[cfg(test)]
                                let mut owner_cell_lookup = std::env::var_os(
                                    "RAW_EDITOR_FOCUS_FINAL_ROI_MAPPING_DIR",
                                )
                                .map(|_| vec![None::<(usize, usize, u16)>; ownership.len()]);
                                // Report-only diagnostic plane: a non-transparent
                                // output sample whose exact bicubic footprint
                                // touched an uncovered station pixel.
                                let mut footprint_uncovered = vec![0u8; ownership.len()];
                                let mut unresolved_pixel_count = 0u64;
                                let reverse_residual = residual_warp::run_model_snapshot();
                                let covered_pixel = |index: usize| {
                                    rendered.coverage.covered().get(index).copied().unwrap_or(0)
                                        != 0
                                };
                                for (index, owner) in ownership.iter_mut().enumerate() {
                                    let compositor_owner = *owner;
                                    let tile_index = usize::from(compositor_owner).checked_sub(1);
                                    let Some(tile_index) = tile_index else {
                                        *owner = 0;
                                        if covered_pixel(index) {
                                            unresolved_pixel_count += 1;
                                            owner_reverse_lookup_failures.compositor_owner_zero +=
                                                1;
                                        }
                                        continue;
                                    };
                                    let Some(&station) = group_order.get(tile_index) else {
                                        *owner = 0;
                                        if covered_pixel(index) {
                                            unresolved_pixel_count += 1;
                                            owner_reverse_lookup_failures.station_missing += 1;
                                        }
                                        continue;
                                    };
                                    let Some(masks) =
                                        station_masks.get(station).and_then(Option::as_ref)
                                    else {
                                        *owner = 0;
                                        if covered_pixel(index) {
                                            unresolved_pixel_count += 1;
                                            owner_reverse_lookup_failures.station_missing += 1;
                                        }
                                        continue;
                                    };
                                    let station_to_world = tile_homographies
                                        .get(&tile_infos[station].id)
                                        .copied()
                                        .unwrap_or_else(Matrix3::identity);
                                    let Some(world_to_station) = station_to_world.try_inverse()
                                    else {
                                        *owner = 0;
                                        if covered_pixel(index) {
                                            unresolved_pixel_count += 1;
                                            owner_reverse_lookup_failures.transform_failed += 1;
                                        }
                                        continue;
                                    };
                                    let x = (index % width as usize) as f64;
                                    let y = (index / width as usize) as f64;
                                    let Some(local) = stitching::map_target_to_source_with_residual(
                                        &world_to_station,
                                        Point3::new(
                                            rendered.sampling_origin.0 + x,
                                            rendered.sampling_origin.1 + y,
                                            1.0,
                                        ),
                                        &tile_infos[station],
                                        projection,
                                        &reverse_residual,
                                    ) else {
                                        *owner = 0;
                                        if covered_pixel(index) {
                                            unresolved_pixel_count += 1;
                                            owner_reverse_lookup_failures.transform_failed += 1;
                                        }
                                        continue;
                                    };
                                    let (mask_width, mask_height) = masks.ownership.dimensions();
                                    let rounded_x = local.x.round();
                                    let rounded_y = local.y.round();
                                    let mut sampled = None;
                                    let mut rounded_has_owner = false;
                                    let mut footprint_has_uncovered = false;
                                    if rounded_x.is_finite()
                                        && rounded_y.is_finite()
                                        && mask_width > 0
                                        && mask_height > 0
                                    {
                                        // The compositor's bicubic sampler uses the
                                        // source coordinate's floor as its anchor;
                                        // rounding is only a diagnostic for whether
                                        // that nearest pixel carried an owner.
                                        let floor_x = local.x.floor() as i64;
                                        let floor_y = local.y.floor() as i64;
                                        let mut nearest_distance = f64::INFINITY;
                                        for dy in -1i64..=2 {
                                            for dx in -1i64..=2 {
                                                let Some(px) = floor_x.checked_add(dx) else {
                                                    continue;
                                                };
                                                let Some(py) = floor_y.checked_add(dy) else {
                                                    continue;
                                                };
                                                if px < 0
                                                    || py < 0
                                                    || px >= i64::from(mask_width)
                                                    || py >= i64::from(mask_height)
                                                {
                                                    continue;
                                                }
                                                let px = px as u32;
                                                let py = py as u32;
                                                if masks
                                                    .coverage
                                                    .covered()
                                                    .get(
                                                        py as usize * mask_width as usize
                                                            + px as usize,
                                                    )
                                                    .copied()
                                                    .unwrap_or(0)
                                                    == 0
                                                {
                                                    footprint_has_uncovered = true;
                                                }
                                                let raw_local = masks.ownership.owner_at(px, py);
                                                let Some(raw_owner) = raw_local
                                                    .checked_sub(1)
                                                    .and_then(|raw_index| {
                                                        raw_owner_ids.get(station).and_then(
                                                            |owners| {
                                                                owners
                                                                    .get(usize::from(raw_index))
                                                                    .copied()
                                                            },
                                                        )
                                                    })
                                                else {
                                                    continue;
                                                };
                                                if f64::from(px) == rounded_x
                                                    && f64::from(py) == rounded_y
                                                {
                                                    rounded_has_owner = true;
                                                }
                                                let distance = (f64::from(px) - local.x).mul_add(
                                                    f64::from(px) - local.x,
                                                    (f64::from(py) - local.y)
                                                        * (f64::from(py) - local.y),
                                                );
                                                if distance < nearest_distance {
                                                    nearest_distance = distance;
                                                    sampled = Some((px, py, raw_owner, raw_local));
                                                }
                                            }
                                        }
                                    }
                                    if footprint_has_uncovered
                                        && rendered
                                            .image
                                            .get_pixel(x as u32, y as u32)
                                            .0
                                            .iter()
                                            .any(|value| *value > 1e-6)
                                    {
                                        footprint_uncovered[index] = 1;
                                    }
                                    let Some((sx, sy, raw_owner, _raw_local)) = sampled else {
                                        *owner = 0;
                                        if covered_pixel(index) {
                                            unresolved_pixel_count += 1;
                                            if !rounded_x.is_finite() || !rounded_y.is_finite() {
                                                owner_reverse_lookup_failures.transform_failed += 1;
                                            } else if local.x < 0.0
                                                || local.y < 0.0
                                                || local.x >= f64::from(mask_width)
                                                || local.y >= f64::from(mask_height)
                                            {
                                                owner_reverse_lookup_failures.out_of_bounds += 1;
                                            } else {
                                                owner_reverse_lookup_failures
                                                    .raw_owner_unresolved += 1;
                                            }
                                        }
                                        continue;
                                    };
                                    if !rounded_has_owner {
                                        owner_reverse_lookup_failures.snapped_to_footprint += 1;
                                    }
                                    *owner = raw_owner;
                                    #[cfg(test)]
                                    if let Some(lookup) = owner_cell_lookup.as_mut() {
                                        let cell_size = masks.fusion.cell_size_px.max(1);
                                        let column = sx / cell_size;
                                        let row = sy / cell_size;
                                        let cell = (row * masks.fusion.grid.columns + column) as usize;
                                        lookup[index] = Some((station, cell, _raw_local));
                                    }
                                    confidence[index] = masks.confidence.value_at(sx, sy);
                                    textured[index] = masks
                                        .textured
                                        .get(sy as usize * mask_width as usize + sx as usize)
                                        .copied()
                                        .unwrap_or(0);
                                    let evidence = masks.sharpness_evidence.at_pixel(sx, sy);
                                    let shortfall = evidence.map_or(
                                        quality_gate_runner::OWNER_EVIDENCE_UNKNOWN,
                                        |evidence| {
                                            quality_gate_runner::quantize_hundredths(
                                                evidence.owner_shortfall(),
                                            )
                                        },
                                    );
                                    // An owned pixel without usable cell evidence
                                    // is an evidence gap, not a flat pixel (需求 11.12).
                                    if shortfall == quality_gate_runner::OWNER_EVIDENCE_UNKNOWN
                                        && covered_pixel(index)
                                    {
                                        unresolved_pixel_count += 1;
                                        owner_reverse_lookup_failures.evidence_missing += 1;
                                    }
                                    if let Some(evidence) = evidence {
                                        owner_shortfall[index] = shortfall;
                                        owner_disagreement[index] =
                                            quality_gate_runner::quantize_hundredths(
                                                evidence.disagreement,
                                            );
                                        quality_gate_runner::accumulate_confidence_score(
                                            &mut confidence_scores,
                                            rendered.coverage.covered()[index],
                                            confidence[index],
                                            textured[index],
                                            evidence,
                                        );
                                    }
                                }
                                #[cfg(test)]
                                if let (Some(directory), Some(lookup)) = (
                                    std::env::var_os("RAW_EDITOR_FOCUS_FINAL_ROI_MAPPING_DIR"),
                                    owner_cell_lookup.as_ref(),
                                ) {
                                    let directory = PathBuf::from(directory);
                                    std::fs::create_dir_all(&directory)
                                        .expect("create final ROI mapping directory");
                                    let coverage_image = GrayImage::from_raw(
                                        width,
                                        height,
                                        rendered.coverage.covered().to_vec(),
                                    )
                                    .expect("valid final ROI mapping coverage");
                                    let rois = quality_gate_runner::select_sparse_rois(
                                        &coverage_image,
                                        &ownership,
                                        rendered.sampling_origin,
                                    );
                                    let margin = quality_gate::QUALITY_ROI_MARGIN as u32;
                                    let constraints = rois
                                        .iter()
                                        .filter_map(|roi| {
                                            let left = roi.x.checked_sub(margin)?;
                                            let top = roi.y.checked_sub(margin)?;
                                            let right = roi.x.checked_add(roi.side)?.checked_add(margin)?;
                                            let bottom = roi.y.checked_add(roi.side)?.checked_add(margin)?;
                                            if right > width || bottom > height {
                                                return None;
                                            }
                                            let mut cells = std::collections::BTreeSet::new();
                                            for y in top..bottom {
                                                for x in left..right {
                                                    let item = lookup[(y * width + x) as usize]?;
                                                    cells.insert(item);
                                                }
                                            }
                                            let mut stations = cells.iter().map(|item| item.0);
                                            let station = stations.next()?;
                                            if stations.any(|candidate| candidate != station) {
                                                return None;
                                            }
                                            let mut local_owners = cells.iter().map(|item| item.2);
                                            let local_owner = local_owners.next()?;
                                            if local_owners.any(|candidate| candidate != local_owner) {
                                                return None;
                                            }
                                            Some(serde_json::json!({
                                                "x": roi.x,
                                                "y": roi.y,
                                                "world_origin": roi.world_origin,
                                                "station": station,
                                                "local_owner": local_owner,
                                                "source_owner": roi.owner,
                                                "cells": cells.into_iter().map(|item| item.1).collect::<Vec<_>>(),
                                            }))
                                        })
                                        .collect::<Vec<_>>();
                                    let path = directory.join("final-owner-roi-constraints.json");
                                    let file = std::fs::File::create(&path)
                                        .expect("create final ROI mapping file");
                                    serde_json::to_writer_pretty(
                                        std::io::BufWriter::new(file),
                                        &constraints,
                                    )
                                    .expect("write final ROI mapping file");
                                }
                                let mut brightness_ratio_sum = 0.0f64;
                                let mut brightness_ratio_count = 0u64;
                                for (index, &incomplete) in footprint_uncovered.iter().enumerate() {
                                    if incomplete == 0 {
                                        continue;
                                    }
                                    let x = index % width as usize;
                                    let y = index / width as usize;
                                    let center = rendered.image.get_pixel(x as u32, y as u32).0;
                                    let center_luma = f64::from(center[0]) * 0.299
                                        + f64::from(center[1]) * 0.587
                                        + f64::from(center[2]) * 0.114;
                                    if !center_luma.is_finite() || center_luma <= 1e-6 {
                                        continue;
                                    }
                                    let mut neighbour_sum = 0.0f64;
                                    let mut neighbour_count = 0u32;
                                    for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                                        let nx = x as i32 + dx;
                                        let ny = y as i32 + dy;
                                        if nx < 0
                                            || ny < 0
                                            || nx >= width as i32
                                            || ny >= height as i32
                                        {
                                            continue;
                                        }
                                        let neighbour_index =
                                            ny as usize * width as usize + nx as usize;
                                        if footprint_uncovered[neighbour_index] != 0
                                            || rendered.coverage.covered()[neighbour_index] == 0
                                        {
                                            continue;
                                        }
                                        let pixel =
                                            rendered.image.get_pixel(nx as u32, ny as u32).0;
                                        let luma = f64::from(pixel[0]) * 0.299
                                            + f64::from(pixel[1]) * 0.587
                                            + f64::from(pixel[2]) * 0.114;
                                        if luma.is_finite() && luma > 1e-6 {
                                            neighbour_sum += luma;
                                            neighbour_count += 1;
                                        }
                                    }
                                    if neighbour_count > 0 {
                                        brightness_ratio_sum += center_luma
                                            / (neighbour_sum / f64::from(neighbour_count));
                                        brightness_ratio_count += 1;
                                    }
                                }
                                owner_reverse_lookup_failures.footprint_uncovered_nontransparent =
                                    footprint_uncovered
                                        .iter()
                                        .filter(|&&value| value != 0)
                                        .count() as u64;
                                owner_reverse_lookup_failures.footprint_brightness_ratio =
                                    (brightness_ratio_count > 0).then_some(
                                        brightness_ratio_sum / brightness_ratio_count as f64,
                                    );
                                quality_sources = Some(raw_sources);
                                let coverage_plane = GrayImage::from_raw(
                                    width,
                                    height,
                                    rendered.coverage.covered().to_vec(),
                                )
                                .unwrap_or_else(|| GrayImage::new(width, height));
                                output_coverage = Some(coverage_plane.clone());
                                quality_planes = Some((
                                    coverage_plane,
                                    ownership,
                                    confidence,
                                    textured,
                                    owner_shortfall,
                                    owner_disagreement,
                                    unresolved_pixel_count,
                                    rendered.sampling_origin,
                                ));
                                rendered.image
                            })
                        }
                    };
                    match tile_result {
                        Ok(image) => Ok(image),
                        // 需求 10.11: an oversized canvas is a rejection, not a
                        // failure a fallback renderer may retry.
                        Err(error)
                            if crate::panorama_utils::stack_pipeline::compositor::is_canvas_rejection(
                                &error,
                            ) =>
                        {
                            Err(error)
                        }
                        Err(error) => {
                            let message = format!(
                                "Virtual-tile panorama failed ({error}); falling back to the existing focus renderer."
                            );
                            println!("  - {message}");
                            let _ = app_handle.emit(progress_event, &message);
                            if let Some(recorder) = stack_report.as_ref() {
                                recorder.set_selected_path(
                                    stack_report::SelectedPath::LegacySingleLayerMosaic,
                                );
                            }
                            stitching::focus_stack_stitcher(
                                &render_images_info,
                                render_homographies,
                                projection,
                                focus_layer_warp.as_ref(),
                                focus_capture_group_ids.as_ref(),
                                compositor_sequence_gap_aware,
                                app_handle.clone(),
                                progress_event,
                                &mut load_render_image,
                            )
                        }
                    }
                } else {
                    let message = format!(
                        "Virtual-tile geometry unavailable ({}); falling back to the existing focus renderer.",
                        geometries.unwrap_err()
                    );
                    println!("  - {message}");
                    let _ = app_handle.emit(progress_event, &message);
                    if let Some(recorder) = stack_report.as_ref() {
                        recorder
                            .set_selected_path(stack_report::SelectedPath::LegacySingleLayerMosaic);
                    }
                    stitching::focus_stack_stitcher(
                        &render_images_info,
                        render_homographies,
                        projection,
                        focus_layer_warp.as_ref(),
                        focus_capture_group_ids.as_ref(),
                        compositor_sequence_gap_aware,
                        app_handle.clone(),
                        progress_event,
                        &mut load_render_image,
                    )
                }
            }
        }
    }?;
    ensure_focus_memory_available(rss_sampler.as_ref())?;
    if let Some((
        coverage,
        mut ownership,
        mut confidence,
        mut textured,
        mut owner_shortfall,
        mut owner_disagreement,
        unresolved_pixel_count,
        world_origin,
    )) = quality_planes.take()
    {
        diagnostics_world_origin = Some(world_origin);
        let sources = quality_sources.take().unwrap_or_default();
        let mut quality_loader =
            |source: &quality_gate_runner::QualitySource| {
                load_prepared_stack_source(&source.path, &settings, raw_exposure_reference).map(
                    |prepared| {
                        let mut rendered = source_to_render_rgb32f(
                            prepared.image,
                            source.dimensions.0,
                            source.dimensions.1,
                        );
                        if is_raw_file(&source.path) {
                            if let Some(gain) = std::env::var("RAW_EDITOR_STACK_RAW_FIXED_GAIN")
                                .ok()
                                .and_then(|value| value.parse::<f32>().ok())
                                .filter(|value| value.is_finite() && *value > 0.0)
                            {
                                rendered.as_mut().par_iter_mut().for_each(|channel| {
                                    *channel = (*channel * gain).clamp(0.0, 1.0)
                                });
                            } else if let Some(gain) = shared_raw_display_gain {
                                rendered.as_mut().par_iter_mut().for_each(|channel| {
                                    *channel = (*channel * gain).clamp(0.0, 1.0)
                                });
                            } else if std::env::var_os(
                                "RAW_EDITOR_SKIP_STACK_RAW_EXPOSURE_NORMALIZATION",
                            )
                            .is_none()
                            {
                                normalize_stack_raw_display_exposure(&mut rendered);
                            }
                        }
                        rendered
                    },
                )
            };
        let residual = residual_warp::run_model_snapshot();
        let accepted_station_relations = tone::run_topology_snapshot()
            .map(|topology| {
                topology
                    .accepted
                    .iter()
                    .filter_map(|&(left_id, right_id)| {
                        let left = topology.station_of(left_id)?;
                        let right = topology.station_of(right_id)?;
                        (left != right).then_some((left.min(right), left.max(right)))
                    })
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        #[cfg(test)]
        if std::env::var_os("RAW_EDITOR_FOCUS_FINAL_WORLD_OWNER_ORACLE_DIR").is_some() {
            stitching::run_final_world_owner_oracle(
                &mut panorama,
                &coverage,
                &mut ownership,
                &mut confidence,
                &mut textured,
                &mut owner_shortfall,
                &mut owner_disagreement,
                world_origin,
                &sources,
                &residual,
                &mut quality_loader,
            )
            .unwrap_or_else(|error| panic!("final-world owner oracle failed: {error}"));
        }
        let mut report = {
            let quality_input = quality_gate_runner::QualityGateInput {
                output: &panorama,
                coverage: &coverage,
                ownership: &ownership,
                confidence: &confidence,
                textured: Some(&textured),
                owner_shortfall: Some(&owner_shortfall),
                owner_disagreement: Some(&owner_disagreement),
                unresolved_pixel_count,
                world_origin,
                sources: &sources,
                accepted_station_relations: &accepted_station_relations,
                residual: &residual,
                acceptance_render_scale: render_scale,
                final_sharpen_amount: stitching::layered_final_sharpen_amount() as f64,
            };
            let report = quality_gate_runner::run_quality_gate(&quality_input, &mut quality_loader);
            #[cfg(test)]
            acceptance_tests::capture_failed_native_boundaries(&quality_input, &report);
            report
        };
        #[cfg(test)]
        if std::env::var_os("RAW_EDITOR_FOCUS_BOUNDARY_DEBUG").is_some()
            && report.criteria.iter().any(|criterion| {
                criterion.name == "boundary_stroke_alignment"
                    && criterion.verdict == stack_report::QualityGateVerdict::Fail
            })
        {
            // Persist the raw-owner neighbourhood that the native boundary
            // criterion actually sees.  Station-level seam traces can contain
            // a third virtual tile, while this ledger is the authoritative
            // Source_RAW provenance used by the gate; keeping both views
            // separate is necessary to diagnose a displaced edge.
            let directory = std::env::var_os("RAW_EDITOR_FOCUS_BOUNDARY_DEBUG_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/private/tmp/raw-editor-boundary-debug"));
            let _ = std::fs::create_dir_all(&directory);
            let (region_left, region_top, region_width, region_height) =
                (6400u32, 6700u32, 500u32, 500u32);
            let clipped_left = region_left.min(panorama.width());
            let clipped_top = region_top.min(panorama.height());
            let clipped_right = region_left
                .saturating_add(region_width)
                .min(panorama.width());
            let clipped_bottom = region_top
                .saturating_add(region_height)
                .min(panorama.height());
            let mut owners = Vec::with_capacity(
                clipped_right
                    .saturating_sub(clipped_left)
                    .saturating_mul(clipped_bottom.saturating_sub(clipped_top))
                    as usize,
            );
            for y in clipped_top..clipped_bottom {
                for x in clipped_left..clipped_right {
                    owners.push(ownership[y as usize * panorama.width() as usize + x as usize]);
                }
            }
            let owner_sources = sources
                .iter()
                .map(|source| {
                    serde_json::json!({
                        "owner": source.owner,
                        "station": source.geometry.station_id,
                        "path": source.path,
                    })
                })
                .collect::<Vec<_>>();
            let _ = std::fs::write(
                directory.join("raw-owner-map.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "canvas": { "width": panorama.width(), "height": panorama.height() },
                    "region": { "left": clipped_left, "top": clipped_top, "width": clipped_right.saturating_sub(clipped_left), "height": clipped_bottom.saturating_sub(clipped_top) },
                    "owners": owners,
                    "owner_sources": owner_sources,
                    "world_origin": [world_origin.0, world_origin.1],
                }))
                .unwrap_or_default(),
            );
            if let Some(failure) = report
                .criteria
                .iter()
                .find(|criterion| criterion.name == "boundary_stroke_alignment")
                .and_then(|criterion| criterion.failed.first())
            {
                let center_x = (failure.world.x - world_origin.0).round() as i32;
                let center_y = (failure.world.y - world_origin.1).round() as i32;
                let mut neighbourhood = Vec::new();
                for y in center_y - 12..=center_y + 12 {
                    for x in center_x - 12..=center_x + 12 {
                        if x < 0
                            || y < 0
                            || x >= panorama.width() as i32
                            || y >= panorama.height() as i32
                        {
                            continue;
                        }
                        let index = y as usize * panorama.width() as usize + x as usize;
                        let pixel = panorama.get_pixel(x as u32, y as u32);
                        let luma = 0.2126 * f64::from(pixel[0])
                            + 0.7152 * f64::from(pixel[1])
                            + 0.0722 * f64::from(pixel[2]);
                        neighbourhood.push(serde_json::json!({
                            "x": x,
                            "y": y,
                            "owner": ownership[index],
                            "covered": coverage.as_raw().get(index).copied().unwrap_or(0),
                            "luma": luma,
                        }));
                    }
                }
                let _ = std::fs::write(
                    directory.join("failure-neighbourhood.json"),
                    serde_json::to_vec_pretty(&serde_json::json!({
                        "world": { "x": failure.world.x, "y": failure.world.y },
                        "output": { "x": center_x, "y": center_y },
                        "owner_path": failure.owner_path,
                        "pixels": neighbourhood,
                    }))
                    .unwrap_or_default(),
                );
            }
        }
        if std::env::var_os("RAW_EDITOR_FOCUS_BOUNDARY_REPAIR").is_some()
            && report.criteria.iter().any(|criterion| {
                criterion.name == "boundary_stroke_alignment"
                    && criterion.verdict == stack_report::QualityGateVerdict::Fail
            })
        {
            let repaired = quality_gate_runner::repair_failed_boundary_samples(
                &mut panorama,
                &coverage,
                &mut ownership,
                world_origin,
                &sources,
                &residual,
                &report,
                &mut quality_loader,
            )
            .unwrap_or_else(|error| panic!("boundary source resample repair failed: {error}"));
            if repaired != 0 {
                let quality_input = quality_gate_runner::QualityGateInput {
                    output: &panorama,
                    coverage: &coverage,
                    ownership: &ownership,
                    confidence: &confidence,
                    textured: Some(&textured),
                    owner_shortfall: Some(&owner_shortfall),
                    owner_disagreement: Some(&owner_disagreement),
                    unresolved_pixel_count,
                    world_origin,
                    sources: &sources,
                    accepted_station_relations: &accepted_station_relations,
                    residual: &residual,
                    acceptance_render_scale: render_scale,
                    final_sharpen_amount: stitching::layered_final_sharpen_amount() as f64,
                };
                report = quality_gate_runner::run_quality_gate(&quality_input, &mut quality_loader);
            }
        }
        quality_gate_runner::record_gate_outcome(&report);
        let effective_pixel_counts = report
            .criteria
            .iter()
            .find(|criterion| criterion.name == "effective_pixel_count")
            .and_then(|criterion| {
                Some((
                    criterion.effective_output_pixels?,
                    criterion.effective_projected_pixels?,
                ))
            });
        if let Some(recorder) = stack_report.as_ref() {
            recorder.update(|stack| {
                stack.quality_gate = report;
                stack.quality_gate.confidence_scores = confidence_scores.clone();
                stack.quality_gate.owner_reverse_lookup_failures =
                    owner_reverse_lookup_failures.clone();
                if let Some((opaque_pixels, union_projected_pixels)) = effective_pixel_counts {
                    stack.composition.opaque_pixels = opaque_pixels;
                    stack.composition.union_projected_pixels = union_projected_pixels;
                }
            });
        }
    } else if let Some(recorder) = stack_report.as_ref() {
        degradation::record_run_degradation(
            degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE,
            serde_json::json!({
                "verdict": "insufficient_evidence",
                "reason": "quality_planes_unavailable",
            }),
        );
        recorder.update(|stack| {
            stack.quality_gate.verdict = stack_report::QualityGateVerdict::InsufficientEvidence;
        });
    }
    if diagnostics_recorder.is_enabled() {
        let (width, height) = panorama.dimensions();
        let (origin_x, origin_y) = diagnostics_world_origin.unwrap_or((0.0, 0.0));
        diagnostics_recorder.record_crop(
            origin_x.round() as i64,
            origin_y.round() as i64,
            width,
            height,
        );
        if let Err(error) = diagnostics_recorder.finish() {
            eprintln!("{error}");
        }
    }
    if rss_sampler.as_mut().is_some_and(|guard| {
        guard
            .finish()
            .is_some_and(|sample| sample.threshold_exceeded)
    }) {
        return Err("memory threshold exceeded during focus-stack run".to_string());
    }
    // The source-stage canvas places every station at its source-stage pose.
    // Once the Virtual_Tile stage replaced those poses the composed canvas is
    // the full-resolution one (langyuan-10: a repaired station moved 3,606px
    // and the composed canvas, 8788x10949, outgrew it; wenyuan-10: the
    // authoritative solve alone, without a repair, composed 8816 columns).
    let (full_canvas_width, full_canvas_height) = if station_poses_replaced {
        let (width, height) = panorama.dimensions();
        (
            (f64::from(width) / render_scale).ceil() as u32,
            (f64::from(height) / render_scale).ceil() as u32,
        )
    } else {
        (full_canvas_width, full_canvas_height)
    };

    println!("Stitching completed in {:.2?}\n", start_time.elapsed());

    let _ = app_handle.emit(progress_event, "Finalizing image result...");

    // Only a mask of the delivered size can become the result alpha.
    let coverage = output_coverage.filter(|mask| mask.dimensions() == panorama.dimensions());

    if let Some(recorder) = stack_report.as_ref() {
        let (width, height) = panorama.dimensions();
        recorder.update(|report| {
            report.composition.canvas.width = u64::from(width);
            report.composition.canvas.height = u64::from(height);
            report.degradation.result = stack_report::StackRunResult::Success;
        });
        // Flush eagerly so the report exists before the result travels back to
        // the caller; the drop guard then becomes a no-op.
        let stack_report_path = recorder.write_once();

        return Ok(StitchOutcome {
            image: DynamicImage::ImageRgb32F(panorama),
            coverage,
            full_canvas_width,
            full_canvas_height,
            render_scale,
            ordered_paths,
            stack_report_path,
        });
    }

    Ok(StitchOutcome {
        image: DynamicImage::ImageRgb32F(panorama),
        coverage,
        full_canvas_width,
        full_canvas_height,
        render_scale,
        ordered_paths,
        stack_report_path: None,
    })
}

/// Persist the geometry needed to inspect a real focus stack without paying
/// the feature matching cost again.  The cache intentionally contains source
/// paths and dimensions alongside every matrix so it cannot silently be
/// applied to a different selection.  This helper is test-only; normal builds
/// never read an environment variable or write this file.
#[cfg(test)]
fn write_alignment_cache(
    path: &Path,
    images: &[ImageInfo],
    ordered_indices: &[usize],
    matches: &HashMap<(usize, usize), MatchInfo>,
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    focus_layer_warp: Option<&FocusLayerWarp>,
    capture_groups: Option<&[FocusCaptureGroup]>,
    projection: Projection,
    canvas_width: u32,
    canvas_height: u32,
) -> Result<(), String> {
    fn matrix_values(matrix: &Matrix3<f64>) -> [f64; 9] {
        [
            matrix[(0, 0)],
            matrix[(0, 1)],
            matrix[(0, 2)],
            matrix[(1, 0)],
            matrix[(1, 1)],
            matrix[(1, 2)],
            matrix[(2, 0)],
            matrix[(2, 1)],
            matrix[(2, 2)],
        ]
    }

    let source_records = images
        .iter()
        .map(|image| {
            serde_json::json!({
                "id": image.id,
                "filename": image.filename,
                "width": image.width,
                "height": image.height,
                "scale_factor": image.scale_factor,
                "focal_length_35mm": image.focal_length_35mm,
                "overview_reference": image.overview_reference,
            })
        })
        .collect::<Vec<_>>();
    let homographies = global_homographies
        .iter()
        .map(|(id, matrix)| (id.to_string(), serde_json::json!(matrix_values(matrix))))
        .collect::<serde_json::Map<_, _>>();
    let match_records = matches
        .iter()
        .filter_map(|(&(source_index, target_index), match_info)| {
            let source = images.get(source_index)?;
            let target = images.get(target_index)?;
            let direct = match_info.homography;
            let global_source = global_homographies.get(&source.id)?;
            let global_target = global_homographies.get(&target.id)?;
            let global_relative = global_target.try_inverse()? * global_source;
            let samples = [
                Point2::new(source.width as f64 * 0.2, source.height as f64 * 0.2),
                Point2::new(source.width as f64 * 0.8, source.height as f64 * 0.2),
                Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5),
                Point2::new(source.width as f64 * 0.2, source.height as f64 * 0.8),
                Point2::new(source.width as f64 * 0.8, source.height as f64 * 0.8),
            ];
            let mut disagreement = samples
                .into_iter()
                .filter_map(|point| {
                    let direct_point = transformed_point(&direct, point)?;
                    let global_point = transformed_point(&global_relative, point)?;
                    let distance = (direct_point - global_point).norm();
                    distance.is_finite().then_some(distance)
                })
                .collect::<Vec<_>>();
            let global_disagreement_px = median_value(&mut disagreement);
            Some(serde_json::json!({
                "source_index": source_index,
                "target_index": target_index,
                "source_id": source.id,
                "target_id": target.id,
                "inliers": match_info.inliers,
                "observations": match_info.points.len(),
                "sequence_bridge": match_info.sequence_bridge,
                "coarse_bridge": match_info.coarse_bridge,
                "local_bracket": focus_match_is_local_bracket(source, target, match_info),
                "center_motion_ratio": focus_match_center_motion_ratio(source, target, match_info),
                "overlap_support": panorama_transform_overlap_support(
                    &direct,
                    source.dimensions(),
                    target.dimensions(),
                ),
                "spatial_support": panorama_spatial_support(
                    &match_info.points,
                    source.dimensions(),
                    target.dimensions(),
                ),
                "direct_median_error_px": median_symmetric_error(&direct, &match_info.points),
                "global_disagreement_px": global_disagreement_px,
                "direct_homography": matrix_values(&direct),
                "global_relative_homography": matrix_values(&global_relative),
            }))
        })
        .collect::<Vec<_>>();
    let focus_bands = focus_layer_warp.map(|warp| {
        warp.bands
            .iter()
            .map(|band| {
                let homographies = band
                    .homographies
                    .iter()
                    .map(|(id, matrix)| (id.to_string(), serde_json::json!(matrix_values(matrix))))
                    .collect::<serde_json::Map<_, _>>();
                let source_ranges = band
                    .source_ranges
                    .iter()
                    .map(|(id, (start, end))| (id.to_string(), serde_json::json!([start, end])))
                    .collect::<serde_json::Map<_, _>>();
                let source_x_ranges = band
                    .source_x_ranges
                    .iter()
                    .map(|(id, (start, end))| (id.to_string(), serde_json::json!([start, end])))
                    .collect::<serde_json::Map<_, _>>();
                serde_json::json!({
                    "homographies": homographies,
                    "source_ranges": source_ranges,
                    "source_x_ranges": source_x_ranges,
                    "relax_foreground_seam": band.relax_foreground_seam,
                    "foreground_only": band.foreground_only,
                    "physical_edge": band.physical_edge,
                })
            })
            .collect::<Vec<_>>()
    });
    let capture_group_records = capture_groups.map(|groups| {
        groups
            .iter()
            .enumerate()
            .map(|(group_index, group)| {
                let anchor = &images[group.anchor];
                let anchor_center = Point2::new(
                    anchor.width as f64 * 0.5,
                    anchor.height as f64 * 0.5,
                );
                let anchor_world = global_homographies
                    .get(&anchor.id)
                    .and_then(|pose| transformed_point(pose, anchor_center));
                let mut drifts = group
                    .members
                    .iter()
                    .filter_map(|&member_index| {
                        let image = images.get(member_index)?;
                        let pose = global_homographies.get(&image.id)?;
                        let center = Point2::new(
                            image.width as f64 * 0.5,
                            image.height as f64 * 0.5,
                        );
                        let world = transformed_point(pose, center)?;
                        let anchor_world = anchor_world?;
                        let drift = (world - anchor_world).norm();
                        drift.is_finite().then_some(drift)
                    })
                    .collect::<Vec<_>>();
                drifts.sort_by(f64::total_cmp);
                let median_drift_px = median_value(&mut drifts);
                let maximum_drift_px = drifts.into_iter().fold(0.0, f64::max);
                serde_json::json!({
                    "group_index": group_index,
                    "anchor_id": anchor.id,
                    "member_ids": group.members.iter().filter_map(|&index| images.get(index).map(|image| image.id)).collect::<Vec<_>>(),
                    "member_count": group.members.len(),
                    "median_center_drift_px": median_drift_px,
                    "maximum_center_drift_px": maximum_drift_px,
                })
            })
            .collect::<Vec<_>>()
    });
    let projection_name = match projection {
        Projection::Planar => "planar",
        Projection::Cylindrical => "cylindrical",
        Projection::Spherical => "spherical",
    };
    let cache = serde_json::json!({
        "schema": 2,
        "projection": projection_name,
        "canvas": { "width": canvas_width, "height": canvas_height },
        "sources": source_records,
        "ordered_ids": ordered_indices.iter().map(|id| *id).collect::<Vec<_>>(),
        "global_homographies": homographies,
        "matches": match_records,
        "focus_warp": focus_bands,
        "capture_groups": capture_group_records,
    });
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create alignment cache directory: {error}"))?;
    }
    let bytes = serde_json::to_vec_pretty(&cache)
        .map_err(|error| format!("Failed to encode alignment cache: {error}"))?;
    std::fs::write(path, bytes).map_err(|error| {
        format!(
            "Failed to write alignment cache {}: {error}",
            path.display()
        )
    })?;
    println!(
        "Alignment cache written: {} ({} source poses, canvas {}x{})",
        path.display(),
        images.len(),
        canvas_width,
        canvas_height
    );
    Ok(())
}

fn is_generated_stitch_output(path: &str) -> bool {
    let Some(stem) = Path::new(path).file_stem().and_then(|stem| stem.to_str()) else {
        return false;
    };
    let stem = stem.to_ascii_lowercase();
    ["_focusstack", "_panorama", "_pano"]
        .into_iter()
        .any(|suffix| {
            let Some(marker) = stem.rfind(suffix) else {
                return false;
            };
            let trailing = &stem[marker + suffix.len()..];
            trailing.is_empty()
                || trailing
                    .chars()
                    .all(|character| character.is_ascii_digit() || matches!(character, '-' | '_'))
        })
}

fn is_auxiliary_stitch_file(path: &str) -> bool {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("._"))
}

fn estimate_translation(points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)]) -> Matrix3<f64> {
    let mut dx: Vec<f64> = points.iter().map(|(a, b)| b.x - a.x).collect();
    let mut dy: Vec<f64> = points.iter().map(|(a, b)| b.y - a.y).collect();
    // `partial_cmp(...).unwrap_or(Equal)` is not a valid sort order when a
    // degenerate homography produces NaN: NaN compares equal to every finite
    // value but finite values still compare against one another.  Use the
    // IEEE total order so a real mixed-focal stack cannot panic in the
    // geometry fallback.
    dx.sort_by(f64::total_cmp);
    dy.sort_by(f64::total_cmp);
    let median = |values: &[f64]| values[values.len() / 2];
    Matrix3::new(1.0, 0.0, median(&dx), 0.0, 1.0, median(&dy), 0.0, 0.0, 1.0)
}

fn estimate_similarity(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> Option<Matrix3<f64>> {
    if points.len() < 2 {
        return None;
    }
    let count = points.len() as f64;
    let source_center = points
        .iter()
        .fold(nalgebra::Vector2::zeros(), |sum, (source, _)| {
            sum + source.coords
        })
        / count;
    let target_center = points
        .iter()
        .fold(nalgebra::Vector2::zeros(), |sum, (_, target)| {
            sum + target.coords
        })
        / count;

    let mut denominator = 0.0;
    let mut dot = 0.0;
    let mut cross = 0.0;
    for (source, target) in points {
        let source_delta = source.coords - source_center;
        let target_delta = target.coords - target_center;
        denominator += source_delta.norm_squared();
        dot += source_delta.x * target_delta.x + source_delta.y * target_delta.y;
        cross += source_delta.x * target_delta.y - source_delta.y * target_delta.x;
    }
    if denominator <= f64::EPSILON {
        return None;
    }
    let a = dot / denominator;
    let b = cross / denominator;
    let translation_x = target_center.x - a * source_center.x + b * source_center.y;
    let translation_y = target_center.y - b * source_center.x - a * source_center.y;
    let transform = Matrix3::new(a, -b, translation_x, b, a, translation_y, 0.0, 0.0, 1.0);
    transform
        .iter()
        .all(|value| value.is_finite())
        .then_some(transform)
}

fn estimate_affine(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> Option<Matrix3<f64>> {
    if points.len() < 3 {
        return None;
    }
    let mut design_data = Vec::with_capacity(points.len() * 3);
    let mut target_x = Vec::with_capacity(points.len());
    let mut target_y = Vec::with_capacity(points.len());
    for (source, target) in points {
        design_data.extend_from_slice(&[source.x, source.y, 1.0]);
        target_x.push(target.x);
        target_y.push(target.y);
    }
    let design = nalgebra::DMatrix::from_row_slice(points.len(), 3, &design_data);
    let decomposition = design.svd(true, true);
    let x_solution = decomposition
        .solve(&nalgebra::DVector::from_vec(target_x), 1e-10)
        .ok()?;
    let y_solution = decomposition
        .solve(&nalgebra::DVector::from_vec(target_y), 1e-10)
        .ok()?;
    let transform = Matrix3::new(
        x_solution[0],
        x_solution[1],
        x_solution[2],
        y_solution[0],
        y_solution[1],
        y_solution[2],
        0.0,
        0.0,
        1.0,
    );
    transform
        .iter()
        .all(|value| value.is_finite())
        .then_some(transform)
}

fn transformed_point(
    transform: &Matrix3<f64>,
    point: nalgebra::Point2<f64>,
) -> Option<nalgebra::Point2<f64>> {
    let mapped = transform * nalgebra::Point3::new(point.x, point.y, 1.0);
    if mapped.z.abs() < 1e-8 {
        return None;
    }
    let mapped = nalgebra::Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
    (mapped.x.is_finite() && mapped.y.is_finite()).then_some(mapped)
}

fn homography_preserves_focus_orientation(
    transform: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> bool {
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    if width <= 1.0 || height <= 1.0 {
        return false;
    }
    let source_corners = [
        Point2::new(0.0, 0.0),
        Point2::new(width, 0.0),
        Point2::new(width, height),
        Point2::new(0.0, height),
    ];
    let mut mapped_corners = Vec::with_capacity(source_corners.len());
    let mut depth_sign = 0.0;
    for corner in source_corners {
        let mapped = transform * Point3::new(corner.x, corner.y, 1.0);
        if !mapped.iter().all(|value| value.is_finite()) || mapped.z.abs() < 1e-8 {
            return false;
        }
        let sign = mapped.z.signum();
        if depth_sign == 0.0 {
            depth_sign = sign;
        } else if sign != depth_sign {
            // A corner crossing the projective horizon can fold the image even
            // when the matrix is numerically invertible. Such a pose is never
            // a valid focus-stack registration.
            return false;
        }
        mapped_corners.push(Point2::new(mapped.x / mapped.z, mapped.y / mapped.z));
    }
    let signed_double_area = mapped_corners
        .iter()
        .zip(mapped_corners.iter().cycle().skip(1))
        .take(4)
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum::<f64>();
    signed_double_area.is_finite() && signed_double_area > 0.0
}

fn transform_is_stable_for_focus_stack(transform: &Matrix3<f64>, dimensions: (u32, u32)) -> bool {
    let (width, height) = (dimensions.0 as f64, dimensions.1 as f64);
    if width <= 1.0 || height <= 1.0 || transform.try_inverse().is_none() {
        return false;
    }
    let source_corners = [
        nalgebra::Point2::new(0.0, 0.0),
        nalgebra::Point2::new(width, 0.0),
        nalgebra::Point2::new(width, height),
        nalgebra::Point2::new(0.0, height),
    ];
    let Some(corners) = source_corners
        .into_iter()
        .map(|point| transformed_point(transform, point))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };

    let top = (corners[1] - corners[0]).norm() / width;
    let right = (corners[2] - corners[1]).norm() / height;
    let bottom = (corners[2] - corners[3]).norm() / width;
    let left = (corners[3] - corners[0]).norm() / height;
    let edge_scales = [top, right, bottom, left];
    if edge_scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale < 0.65 || *scale > 1.45)
    {
        return false;
    }
    let min_scale = edge_scales.iter().copied().fold(f64::INFINITY, f64::min);
    let max_scale = edge_scales
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if max_scale / min_scale > 1.18 {
        return false;
    }

    let top_vector = corners[1] - corners[0];
    let left_vector = corners[3] - corners[0];
    let orthogonality = top_vector.dot(&left_vector) / (top_vector.norm() * left_vector.norm());
    if !orthogonality.is_finite() || orthogonality.abs() > 0.22 {
        return false;
    }

    let signed_double_area = corners
        .iter()
        .zip(corners.iter().cycle().skip(1))
        .take(4)
        .map(|(a, b)| a.x * b.y - b.x * a.y)
        .sum::<f64>();
    let area_ratio = signed_double_area / (2.0 * width * height);
    area_ratio.is_finite() && (0.5..=2.0).contains(&area_ratio)
}

fn median_symmetric_error(
    transform: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> f64 {
    let Some(inverse) = transform.try_inverse() else {
        return f64::INFINITY;
    };
    let mut errors: Vec<f64> = points
        .iter()
        .map(|(source, target)| symmetric_point_error(transform, &inverse, *source, *target))
        .filter(|error| error.is_finite())
        .collect();
    if errors.is_empty() {
        return f64::INFINITY;
    }
    errors.sort_by(f64::total_cmp);
    errors[errors.len() / 2]
}

#[derive(Debug, Clone)]
struct RobustTransformFit {
    transform: Matrix3<f64>,
    inlier_indices: Vec<usize>,
    median_error: f64,
}

fn symmetric_inlier_indices(
    transform: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    threshold: f64,
) -> Vec<usize> {
    let Some(inverse) = transform.try_inverse() else {
        return Vec::new();
    };
    points
        .iter()
        .enumerate()
        .filter_map(|(index, (source, target))| {
            (symmetric_point_error(transform, &inverse, *source, *target) <= threshold)
                .then_some(index)
        })
        .collect()
}

fn robust_transform_fit<F>(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    sample_size: usize,
    seed: u64,
    estimator: F,
) -> Option<RobustTransformFit>
where
    F: Fn(&[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)]) -> Option<Matrix3<f64>>,
{
    robust_transform_fit_with_threshold(
        points,
        sample_size,
        seed,
        FOCUS_MODEL_INLIER_THRESHOLD,
        estimator,
    )
}

/// [`robust_transform_fit`] with a caller-chosen symmetric inlier threshold.
fn robust_transform_fit_with_threshold<F>(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    sample_size: usize,
    seed: u64,
    inlier_threshold: f64,
    estimator: F,
) -> Option<RobustTransformFit>
where
    F: Fn(&[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)]) -> Option<Matrix3<f64>>,
{
    if points.len() < sample_size {
        return None;
    }

    // `seed` is the caller's site constant; the run seed below is derived from
    // the input content (需求 14.6).
    let mut rng = StdRng::seed_from_u64(
        seed ^ (points.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ determinism::run_random_seed_mix(),
    );
    let all_indices: Vec<usize> = (0..points.len()).collect();
    let mut best_transform = None;
    let mut best_inliers = Vec::new();
    let mut best_median = f64::INFINITY;

    let iterations = if points.len() == sample_size {
        1
    } else {
        FOCUS_MODEL_RANSAC_ITERATIONS
    };
    for _ in 0..iterations {
        let sample_indices: Vec<usize> =
            all_indices.sample(&mut rng, sample_size).copied().collect();
        if sample_indices.len() != sample_size {
            continue;
        }
        let sample: Vec<_> = sample_indices.iter().map(|&index| points[index]).collect();
        let Some(transform) = estimator(&sample) else {
            continue;
        };
        let inlier_indices = symmetric_inlier_indices(&transform, points, inlier_threshold);
        if inlier_indices.len() < sample_size {
            continue;
        }
        let inlier_points: Vec<_> = inlier_indices.iter().map(|&index| points[index]).collect();
        let median = median_symmetric_error(&transform, &inlier_points);
        if inlier_indices.len() > best_inliers.len()
            || (inlier_indices.len() == best_inliers.len() && median < best_median)
        {
            best_transform = Some(transform);
            best_inliers = inlier_indices;
            best_median = median;
        }
    }

    let mut transform = best_transform?;
    let mut inlier_indices = best_inliers;
    for _ in 0..4 {
        let inlier_points: Vec<_> = inlier_indices.iter().map(|&index| points[index]).collect();
        let Some(refitted) = estimator(&inlier_points) else {
            break;
        };
        let refitted_inliers = symmetric_inlier_indices(&refitted, points, inlier_threshold);
        if refitted_inliers.len() < sample_size {
            break;
        }
        transform = refitted;
        if refitted_inliers == inlier_indices {
            break;
        }
        inlier_indices = refitted_inliers;
    }

    let inlier_points: Vec<_> = inlier_indices.iter().map(|&index| points[index]).collect();
    Some(RobustTransformFit {
        transform,
        inlier_indices,
        median_error: median_symmetric_error(&transform, &inlier_points),
    })
}

fn focus_fit_is_competitive(candidate: &RobustTransformFit, selected: &RobustTransformFit) -> bool {
    if candidate.inlier_indices.len() < FOCUS_MODEL_MIN_INLIERS {
        return false;
    }
    // A focus stack is normally captured from one stable camera pose. Prefer the
    // lower-error, lower-DOF model instead of allowing a slightly larger inlier
    // consensus to select an affine warp that bends the paper differently at each
    // edge. Extra support may win only when it does not materially worsen the fit.
    let materially_more_precise = candidate.median_error + 0.25 < selected.median_error
        && candidate.median_error <= selected.median_error * 0.90;
    let similar_precision_with_more_support = candidate.inlier_indices.len()
        >= selected.inlier_indices.len() + 4
        && candidate.median_error <= selected.median_error * 1.04;
    materially_more_precise || similar_precision_with_more_support
}

fn select_focus_stack_transform(
    projective: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    source_dimensions: (u32, u32),
    alignment_mode: AlignmentMode,
) -> Matrix3<f64> {
    // A scale change is a shifted mosaic even when the optical centres line up
    // perfectly.  Use the broader, bounded geometry guard for that case; the
    // normal focus-stack guard intentionally rejects large scale changes so a
    // bad repeated-stroke match cannot bend a fixed-camera stack.
    let shifted_mosaic = focus_stack_motion_is_shifted_mosaic(points, source_dimensions);
    let stable_transform = |transform: &Matrix3<f64>| {
        if shifted_mosaic {
            panorama_transform_is_stable(transform, source_dimensions)
        } else {
            transform_is_stable_for_focus_stack(transform, source_dimensions)
        }
    };
    let translation = estimate_translation(points);
    let translation_inliers =
        symmetric_inlier_indices(&translation, points, FOCUS_MODEL_INLIER_THRESHOLD);
    let translation_is_valid = translation_inliers.len() >= FOCUS_MODEL_MIN_INLIERS;
    let translation_points: Vec<_> = translation_inliers
        .iter()
        .map(|&index| points[index])
        .collect();
    let mut selected = RobustTransformFit {
        transform: translation,
        inlier_indices: if translation_is_valid {
            translation_inliers
        } else {
            Vec::new()
        },
        median_error: if translation_is_valid {
            median_symmetric_error(&translation, &translation_points)
        } else {
            f64::INFINITY
        },
    };
    let mut selected_name = "translation";

    let similarity = robust_transform_fit(points, 2, 0xA24B_AED4_963E_E407, estimate_similarity);
    if let Some(model) = similarity.as_ref()
        && stable_transform(&model.transform)
        && focus_fit_is_competitive(model, &selected)
    {
        selected = model.clone();
        selected_name = "similarity";
    }

    let affine = robust_transform_fit(points, 3, 0x9FB2_1C65_1E98_DF25, estimate_affine);
    if let Some(model) = affine.as_ref()
        && stable_transform(&model.transform)
        && focus_fit_is_competitive(model, &selected)
    {
        selected = model.clone();
        selected_name = "affine";
    }

    let projective_error = median_symmetric_error(projective, points);
    let explicit_projective = matches!(
        alignment_mode,
        AlignmentMode::Perspective | AlignmentMode::Cylindrical | AlignmentMode::Spherical
    );
    // Auto mode may still need a homography for a handheld view of a flat
    // artwork. Require the inliers to cover a substantial part of both
    // frames before accepting it; a compact repeated brush stroke must stay
    // on the lower-DOF model even if its projective fit has a tiny residual.
    let projective_spatial_support =
        panorama_spatial_support(points, source_dimensions, source_dimensions);
    let allow_projective =
        explicit_projective || projective_spatial_support >= FOCUS_PROJECTIVE_MIN_SPATIAL_SUPPORT;
    if allow_projective
        && stable_transform(projective)
        && projective_error <= FOCUS_MODEL_INLIER_THRESHOLD * 0.5
        && projective_error + 0.15 < selected.median_error
        && projective_error <= selected.median_error * 0.80
    {
        selected.transform = *projective;
        selected.inlier_indices = (0..points.len()).collect();
        selected.median_error = projective_error;
        selected_name = "projective";
    }

    let similarity_summary = similarity
        .as_ref()
        .map(|fit| format!("{:.3}px/{}", fit.median_error, fit.inlier_indices.len()))
        .unwrap_or_else(|| "n/a".to_string());
    let affine_summary = affine
        .as_ref()
        .map(|fit| format!("{:.3}px/{}", fit.median_error, fit.inlier_indices.len()))
        .unwrap_or_else(|| "n/a".to_string());
    let translation_summary = if translation_is_valid {
        format!(
            "{:.3}px/{}",
            median_symmetric_error(&translation, &translation_points),
            translation_points.len()
        )
    } else {
        format!("invalid/{}", translation_points.len())
    };
    println!(
        "  - Focus alignment selected {selected_name}: median symmetric error {:.3}px with {} inliers (translation {translation_summary}, similarity {similarity_summary}, affine {affine_summary}, projective {:.3}px/{}, shifted-mosaic {})",
        selected.median_error,
        selected.inlier_indices.len(),
        projective_error,
        points.len(),
        shifted_mosaic
    );
    selected.transform
}

fn focus_stack_motion_is_shifted_mosaic(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    source_dimensions: (u32, u32),
) -> bool {
    if points.len() < FOCUS_MODEL_MIN_INLIERS {
        return false;
    }
    let translation = estimate_translation(points);
    let motion = nalgebra::Vector2::new(translation[(0, 2)], translation[(1, 2)]).norm();
    let image_scale = source_dimensions.0.max(source_dimensions.1) as f64;
    let translation_shift = motion.is_finite()
        && image_scale > 1.0
        && motion > image_scale * FOCUS_SHIFTED_MOSAIC_MOTION_RATIO;
    let scale_or_rotation_shift = estimate_similarity(points)
        .map(|similarity| {
            let scale_x = similarity[(0, 0)].hypot(similarity[(1, 0)]);
            let scale_y = similarity[(0, 1)].hypot(similarity[(1, 1)]);
            let scale = (scale_x.max(0.0) * scale_y.max(0.0)).sqrt();
            let rotation = 0.5
                * (similarity[(1, 0)] - similarity[(0, 1)])
                    .atan2(similarity[(0, 0)] + similarity[(1, 1)]);
            (scale.is_finite() && (scale - 1.0).abs() > 0.10)
                || (rotation.is_finite() && rotation.abs() > 0.035)
        })
        .unwrap_or(false);
    translation_shift || scale_or_rotation_shift
}

fn refine_match_point_from_homography(
    image1: &ImageInfo,
    image2: &ImageInfo,
    keypoint1: KeyPoint,
    keypoint2: KeyPoint,
    projection: Projection,
    homography: &Matrix3<f64>,
    prefer_feature_center: bool,
    high_precision: bool,
) -> Option<(nalgebra::Point2<f64>, nalgebra::Point2<f64>)> {
    let source_full_x = (keypoint1.x as f64 * image1.scale_factor).round() as i32;
    let source_full_y = (keypoint1.y as f64 * image1.scale_factor).round() as i32;
    let fallback_target_full_x = (keypoint2.x as f64 * image2.scale_factor).round() as i32;
    let fallback_target_full_y = (keypoint2.y as f64 * image2.scale_factor).round() as i32;
    let (target_full_x, target_full_y) =
        if projection == Projection::Planar && !prefer_feature_center {
            let predicted =
                homography * nalgebra::Point3::new(source_full_x as f64, source_full_y as f64, 1.0);
            if predicted.z.abs() < 1e-8 {
                (fallback_target_full_x, fallback_target_full_y)
            } else {
                (
                    (predicted.x / predicted.z).round() as i32,
                    (predicted.y / predicted.z).round() as i32,
                )
            }
        } else {
            (fallback_target_full_x, fallback_target_full_y)
        };

    let (
        source_plane,
        target_plane,
        source_x,
        source_y,
        target_x,
        target_y,
        target_scale,
        patch_radius,
        search_radius,
    ) = if let (Some(source), Some(target)) = (&image1.full_image, &image2.full_image) {
        (
            LumaPlane::Rgb(source),
            LumaPlane::Rgb(target),
            source_full_x,
            source_full_y,
            target_full_x,
            target_full_y,
            1.0,
            if high_precision {
                FOCUS_MATCH_REFINE_PATCH_RADIUS
            } else {
                MATCH_REFINE_PATCH_RADIUS
            },
            if high_precision {
                FOCUS_MATCH_REFINE_SEARCH_RADIUS
            } else {
                MATCH_REFINE_SEARCH_RADIUS
            },
        )
    } else {
        (
            LumaPlane::Gray(&image1.alignment_image),
            LumaPlane::Gray(&image2.alignment_image),
            keypoint1.x as i32,
            keypoint1.y as i32,
            (target_full_x as f64 / image2.scale_factor).round() as i32,
            (target_full_y as f64 / image2.scale_factor).round() as i32,
            image2.scale_factor,
            if high_precision { 8 } else { 4 },
            if high_precision { 12 } else { 6 },
        )
    };

    let (best_x, best_y, subpixel_x, subpixel_y) = refine_patch_position(
        &source_plane,
        &target_plane,
        source_x,
        source_y,
        target_x,
        target_y,
        patch_radius,
        search_radius,
    )?;
    let source = project_point(
        image1,
        source_full_x as f64,
        source_full_y as f64,
        projection,
    )?;
    let target = project_point(
        image2,
        (best_x as f64 + subpixel_x) * target_scale,
        (best_y as f64 + subpixel_y) * target_scale,
        projection,
    )?;
    Some((source, target))
}

enum LumaPlane<'a> {
    Rgb(&'a Rgb32FImage),
    Gray(&'a GrayImage),
}

impl LumaPlane<'_> {
    fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Rgb(image) => image.dimensions(),
            Self::Gray(image) => image.dimensions(),
        }
    }

    fn luma_at(&self, x: i32, y: i32) -> Option<f64> {
        let (width, height) = self.dimensions();
        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
            return None;
        }
        match self {
            Self::Rgb(image) => {
                let pixel = image.get_pixel(x as u32, y as u32);
                Some(
                    (pixel[0] as f64 * 0.299)
                        + (pixel[1] as f64 * 0.587)
                        + (pixel[2] as f64 * 0.114),
                )
            }
            Self::Gray(image) => Some(image.get_pixel(x as u32, y as u32)[0] as f64),
        }
    }

    fn gradient_at(&self, x: i32, y: i32) -> Option<(f64, f64)> {
        let horizontal = self.luma_at(x + 1, y)? - self.luma_at(x - 1, y)?;
        let vertical = self.luma_at(x, y + 1)? - self.luma_at(x, y - 1)?;
        Some((horizontal, vertical))
    }
}

fn gradient_patch_energy(image: &LumaPlane<'_>, center_x: i32, center_y: i32, radius: i32) -> f64 {
    let mut energy = 0.0;
    let mut sample_count = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some((horizontal, vertical)) = image.gradient_at(center_x + dx, center_y + dy)
            else {
                return 0.0;
            };
            energy += horizontal * horizontal + vertical * vertical;
            sample_count += 1;
        }
    }
    if sample_count == 0 {
        0.0
    } else {
        (energy / sample_count as f64).sqrt()
    }
}

fn gradient_patch_corner_energy(
    image: &LumaPlane<'_>,
    center_x: i32,
    center_y: i32,
    radius: i32,
) -> f64 {
    let mut sum_horizontal_squared = 0.0;
    let mut sum_vertical_squared = 0.0;
    let mut sum_cross = 0.0;
    let mut sample_count = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some((horizontal, vertical)) = image.gradient_at(center_x + dx, center_y + dy)
            else {
                return 0.0;
            };
            sum_horizontal_squared += horizontal * horizontal;
            sum_vertical_squared += vertical * vertical;
            sum_cross += horizontal * vertical;
            sample_count += 1;
        }
    }
    if sample_count == 0 {
        return 0.0;
    }
    let sample_count = sample_count as f64;
    let horizontal_squared = sum_horizontal_squared / sample_count;
    let vertical_squared = sum_vertical_squared / sample_count;
    let cross = sum_cross / sample_count;
    (horizontal_squared * vertical_squared - cross * cross)
        .max(0.0)
        .sqrt()
}

fn gradient_patch_ncc(
    image1: &LumaPlane<'_>,
    image2: &LumaPlane<'_>,
    center1_x: i32,
    center1_y: i32,
    center2_x: i32,
    center2_y: i32,
    radius: i32,
) -> f64 {
    let mut sum1_horizontal = 0.0;
    let mut sum1_vertical = 0.0;
    let mut sum2_horizontal = 0.0;
    let mut sum2_vertical = 0.0;
    let mut sum_squared1 = 0.0;
    let mut sum_squared2 = 0.0;
    let mut sum_product = 0.0;
    let mut sample_count = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some((horizontal1, vertical1)) = image1.gradient_at(center1_x + dx, center1_y + dy)
            else {
                return f64::NEG_INFINITY;
            };
            let Some((horizontal2, vertical2)) = image2.gradient_at(center2_x + dx, center2_y + dy)
            else {
                return f64::NEG_INFINITY;
            };
            sum1_horizontal += horizontal1;
            sum1_vertical += vertical1;
            sum2_horizontal += horizontal2;
            sum2_vertical += vertical2;
            sum_squared1 += horizontal1 * horizontal1 + vertical1 * vertical1;
            sum_squared2 += horizontal2 * horizontal2 + vertical2 * vertical2;
            sum_product += horizontal1 * horizontal2 + vertical1 * vertical2;
            sample_count += 1;
        }
    }

    let sample_count = sample_count as f64;
    let covariance = sum_product
        - (sum1_horizontal * sum2_horizontal + sum1_vertical * sum2_vertical) / sample_count;
    let variance1 = sum_squared1
        - (sum1_horizontal * sum1_horizontal + sum1_vertical * sum1_vertical) / sample_count;
    let variance2 = sum_squared2
        - (sum2_horizontal * sum2_horizontal + sum2_vertical * sum2_vertical) / sample_count;
    if variance1 <= f64::EPSILON || variance2 <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        covariance / (variance1 * variance2).sqrt()
    }
}

#[allow(clippy::too_many_arguments)]
fn refine_foreground_patch_position(
    image1: &LumaPlane<'_>,
    image2: &LumaPlane<'_>,
    source_x: i32,
    source_y: i32,
    target_x: i32,
    target_y: i32,
    patch_radius: i32,
    search_radius: i32,
) -> Option<(i32, i32, f64, f64)> {
    let source_energy = gradient_patch_energy(image1, source_x, source_y, patch_radius);
    if source_energy < FOCUS_FOREGROUND_MIN_GRADIENT_ENERGY {
        return None;
    }

    let mut best_score = f64::NEG_INFINITY;
    let mut best_target = None;
    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let candidate_x = target_x + dx;
            let candidate_y = target_y + dy;
            let gradient_score = gradient_patch_ncc(
                image1,
                image2,
                source_x,
                source_y,
                candidate_x,
                candidate_y,
                patch_radius,
            );
            if !gradient_score.is_finite() {
                continue;
            }
            // Gradient correlation identifies stable depth-layer edges and
            // structure; a small luminance contribution keeps the position from
            // becoming arbitrary on a long, nearly uniform surface.
            let luminance_score = patch_ncc(
                image1,
                image2,
                source_x,
                source_y,
                candidate_x,
                candidate_y,
                patch_radius,
            );
            let score = if luminance_score.is_finite() {
                gradient_score * 0.75 + luminance_score * 0.25
            } else {
                gradient_score
            };
            if score > best_score {
                best_score = score;
                best_target = Some((candidate_x, candidate_y));
            }
        }
    }

    let (best_x, best_y) = best_target?;
    if !best_score.is_finite() {
        return None;
    }
    let sample = |x, y| {
        let gradient_score =
            gradient_patch_ncc(image1, image2, source_x, source_y, x, y, patch_radius);
        let luminance_score = patch_ncc(image1, image2, source_x, source_y, x, y, patch_radius);
        if luminance_score.is_finite() {
            gradient_score * 0.75 + luminance_score * 0.25
        } else {
            gradient_score
        }
    };
    let subpixel_offset = |negative: f64, center: f64, positive: f64| {
        if !negative.is_finite() || !center.is_finite() || !positive.is_finite() {
            return 0.0;
        }
        let denominator = negative - 2.0 * center + positive;
        if denominator.abs() < 1e-8 {
            0.0
        } else {
            (0.5 * (negative - positive) / denominator).clamp(-1.0, 1.0)
        }
    };
    let subpixel_x = subpixel_offset(
        sample(best_x - 1, best_y),
        best_score,
        sample(best_x + 1, best_y),
    );
    let subpixel_y = subpixel_offset(
        sample(best_x, best_y - 1),
        best_score,
        sample(best_x, best_y + 1),
    );
    Some((best_x, best_y, subpixel_x, subpixel_y))
}

#[allow(clippy::too_many_arguments)]
fn refine_patch_position(
    image1: &LumaPlane<'_>,
    image2: &LumaPlane<'_>,
    source_x: i32,
    source_y: i32,
    target_x: i32,
    target_y: i32,
    patch_radius: i32,
    search_radius: i32,
) -> Option<(i32, i32, f64, f64)> {
    let mut best_score = f64::NEG_INFINITY;
    let mut best_target = None;

    for dy in -search_radius..=search_radius {
        for dx in -search_radius..=search_radius {
            let candidate_x = target_x + dx;
            let candidate_y = target_y + dy;
            let score = patch_ncc(
                image1,
                image2,
                source_x,
                source_y,
                candidate_x,
                candidate_y,
                patch_radius,
            );
            if score > best_score {
                best_score = score;
                best_target = Some((candidate_x, candidate_y));
            }
        }
    }

    let (best_x, best_y) = best_target?;
    if !best_score.is_finite() {
        return None;
    }
    let subpixel_offset = |negative: f64, center: f64, positive: f64| {
        if !negative.is_finite() || !center.is_finite() || !positive.is_finite() {
            return 0.0;
        }
        let denominator = negative - 2.0 * center + positive;
        if denominator.abs() < 1e-8 {
            0.0
        } else {
            (0.5 * (negative - positive) / denominator).clamp(-1.0, 1.0)
        }
    };
    let sample = |x, y| patch_ncc(image1, image2, source_x, source_y, x, y, patch_radius);
    let subpixel_x = subpixel_offset(
        sample(best_x - 1, best_y),
        best_score,
        sample(best_x + 1, best_y),
    );
    let subpixel_y = subpixel_offset(
        sample(best_x, best_y - 1),
        best_score,
        sample(best_x, best_y + 1),
    );
    Some((best_x, best_y, subpixel_x, subpixel_y))
}

fn patch_ncc(
    image1: &LumaPlane<'_>,
    image2: &LumaPlane<'_>,
    center1_x: i32,
    center1_y: i32,
    center2_x: i32,
    center2_y: i32,
    radius: i32,
) -> f64 {
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum_squares1 = 0.0;
    let mut sum_squares2 = 0.0;
    let mut sum_products = 0.0;
    let mut sample_count = 0usize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some(value1) = image1.luma_at(center1_x + dx, center1_y + dy) else {
                return f64::NEG_INFINITY;
            };
            let Some(value2) = image2.luma_at(center2_x + dx, center2_y + dy) else {
                return f64::NEG_INFINITY;
            };
            sum1 += value1;
            sum2 += value2;
            sum_squares1 += value1 * value1;
            sum_squares2 += value2 * value2;
            sum_products += value1 * value2;
            sample_count += 1;
        }
    }

    let sample_count = sample_count as f64;
    let covariance = sum_products - sum1 * sum2 / sample_count;
    let variance1 = sum_squares1 - sum1 * sum1 / sample_count;
    let variance2 = sum_squares2 - sum2 * sum2 / sample_count;
    if variance1 <= f64::EPSILON || variance2 <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        covariance / (variance1 * variance2).sqrt()
    }
}

fn refine_homography_inliers(
    points: &mut Vec<(nalgebra::Point2<f64>, nalgebra::Point2<f64>)>,
    refinement_threshold: f64,
    minimum_inliers: usize,
) -> Option<Matrix3<f64>> {
    if points.len() < minimum_inliers {
        return None;
    }

    // Patch correlation can occasionally lock onto a nearby repeated stroke or
    // texture. Starting the least-squares refinement from every correlation lets
    // a coherent minority bend a projective model enough that none of those bad
    // points exceeds the later residual threshold. Re-run a deterministic robust
    // fit at full resolution before optimizing all remaining correspondences.
    let (_, robust_inlier_indices) =
        processing::find_homography_ransac_points_stable_with_min_inliers(
            points,
            refinement_threshold,
            minimum_inliers,
        )?;
    if robust_inlier_indices.len() < minimum_inliers {
        return None;
    }
    if robust_inlier_indices.len() < points.len() {
        *points = robust_inlier_indices
            .into_iter()
            .map(|index| points[index])
            .collect();
    }

    let mut homography = processing::compute_homography(points)?;
    for _ in 0..3 {
        let Some(inverse) = homography.try_inverse() else {
            break;
        };
        let refined_points: Vec<_> = points
            .iter()
            .copied()
            .filter(|(source, target)| {
                symmetric_point_error(&homography, &inverse, *source, *target)
                    <= refinement_threshold
            })
            .collect();
        if refined_points.len() < minimum_inliers || refined_points.len() == points.len() {
            break;
        }
        *points = refined_points;
        homography = processing::compute_homography(points)?;
    }
    Some(homography)
}

fn retain_model_inliers(
    points: &mut Vec<(nalgebra::Point2<f64>, nalgebra::Point2<f64>)>,
    transform: &Matrix3<f64>,
    threshold: f64,
    minimum_inliers: usize,
) -> bool {
    let Some(inverse) = transform.try_inverse() else {
        return false;
    };
    let filtered = points
        .iter()
        .copied()
        .filter(|(source, target)| {
            symmetric_point_error(transform, &inverse, *source, *target) <= threshold
        })
        .collect::<Vec<_>>();
    if filtered.len() < minimum_inliers {
        return false;
    }
    *points = filtered;
    true
}

fn symmetric_point_error(
    homography: &Matrix3<f64>,
    inverse: &Matrix3<f64>,
    source: nalgebra::Point2<f64>,
    target: nalgebra::Point2<f64>,
) -> f64 {
    let forward = homography * nalgebra::Point3::new(source.x, source.y, 1.0);
    let reverse = inverse * nalgebra::Point3::new(target.x, target.y, 1.0);
    if forward.z.abs() < 1e-8 || reverse.z.abs() < 1e-8 {
        return f64::INFINITY;
    }
    let forward_point = nalgebra::Point2::new(forward.x / forward.z, forward.y / forward.z);
    let reverse_point = nalgebra::Point2::new(reverse.x / reverse.z, reverse.y / reverse.z);
    ((forward_point - target).norm_squared() + (reverse_point - source).norm_squared()).sqrt()
}

fn symmetric_reprojection_rmse(
    homography: &Matrix3<f64>,
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
) -> f64 {
    let Some(inverse) = homography.try_inverse() else {
        return f64::INFINITY;
    };
    let sum_squared_error = points
        .iter()
        .filter_map(|(source, target)| {
            let forward = homography * nalgebra::Point3::new(source.x, source.y, 1.0);
            let reverse = inverse * nalgebra::Point3::new(target.x, target.y, 1.0);
            if forward.z.abs() < 1e-8 || reverse.z.abs() < 1e-8 {
                return None;
            }
            let forward_point = nalgebra::Point2::new(forward.x / forward.z, forward.y / forward.z);
            let reverse_point = nalgebra::Point2::new(reverse.x / reverse.z, reverse.y / reverse.z);
            Some((forward_point - target).norm_squared() + (reverse_point - source).norm_squared())
        })
        .sum::<f64>();
    if points.is_empty() {
        f64::INFINITY
    } else {
        (sum_squared_error / (points.len() as f64 * 2.0)).sqrt()
    }
}

struct Dsu {
    parent: Vec<usize>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Dsu {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, i: usize) -> usize {
        if self.parent[i] == i {
            i
        } else {
            self.parent[i] = self.find(self.parent[i]);
            self.parent[i]
        }
    }

    fn union(&mut self, i: usize, j: usize) {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            self.parent[root_i] = root_j;
        }
    }
}

fn build_stitching_order(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> (Vec<usize>, HashMap<usize, Matrix3<f64>>) {
    build_graph_stitching_order(images, matches, |source, target, match_info| {
        panorama_edge_weight(source, target, match_info)
    })
}

#[derive(Clone, Copy, Debug)]
struct FlatArtworkRenderRect {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
}

impl FlatArtworkRenderRect {
    fn width(self) -> f64 {
        (self.right - self.left).max(0.0)
    }

    fn height(self) -> f64 {
        (self.bottom - self.top).max(0.0)
    }

    fn area(self) -> f64 {
        self.width() * self.height()
    }

    fn center(self) -> Point2<f64> {
        Point2::new(
            (self.left + self.right) * 0.5,
            (self.top + self.bottom) * 0.5,
        )
    }

    fn overlap_area(self, other: Self) -> f64 {
        (self.right.min(other.right) - self.left.max(other.left)).max(0.0)
            * (self.bottom.min(other.bottom) - self.top.max(other.top)).max(0.0)
    }
}

fn flat_artwork_projected_rect(
    image: &ImageInfo,
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<FlatArtworkRenderRect> {
    let transform = homographies.get(&image.id)?;
    let corners = [
        Point2::new(0.0, 0.0),
        Point2::new(image.width as f64, 0.0),
        Point2::new(image.width as f64, image.height as f64),
        Point2::new(0.0, image.height as f64),
    ];
    let points = corners
        .into_iter()
        .map(|point| transformed_point(transform, point))
        .collect::<Option<Vec<_>>>()?;
    let left = points
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let right = points
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let top = points
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min);
    let bottom = points
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let rect = FlatArtworkRenderRect {
        left,
        top,
        right,
        bottom,
    };
    (left.is_finite()
        && top.is_finite()
        && right.is_finite()
        && bottom.is_finite()
        && rect.area() > 1.0)
        .then_some(rect)
}

/// Order flat-artwork sources as a centre-outward overlap walk.
///
/// The graph/MST order is a geometry solve order, not a compositor order. A
/// breadth-first traversal of that tree can visit a branch whose projected
/// rectangle is disjoint from the already rendered canvas. The detail-
/// preserving compositor then has no existing pixels from which to estimate
/// that source's low-frequency exposure, so the complete source rectangle
/// survives as a visible block. Start near the projected canvas centre, finish
/// the current focus group, and then choose the unrendered rectangle with the
/// largest overlap with the materialised union. If the solved graph contains a
/// real gap, fall back to the nearest projected centre while retaining a
/// deterministic seed order.
fn flat_artwork_overlap_render_order(
    images: &[ImageInfo],
    seed_order: &[usize],
    homographies: &HashMap<usize, Matrix3<f64>>,
    capture_group_ids: Option<&HashMap<usize, u8>>,
) -> Vec<usize> {
    if seed_order.len() < 2 {
        return seed_order.to_vec();
    }
    let mut rank = HashMap::with_capacity(seed_order.len());
    for (position, &index) in seed_order.iter().enumerate() {
        rank.insert(index, position);
    }
    let rects = seed_order
        .iter()
        .filter_map(|&index| {
            images
                .get(index)
                .and_then(|image| flat_artwork_projected_rect(image, homographies))
                .map(|rect| (index, rect))
        })
        .collect::<HashMap<_, _>>();
    if rects.len() < 2 {
        return seed_order.to_vec();
    }

    let mut center_x = rects
        .values()
        .map(|rect| rect.center().x)
        .collect::<Vec<_>>();
    let mut center_y = rects
        .values()
        .map(|rect| rect.center().y)
        .collect::<Vec<_>>();
    let median_x = median_value(&mut center_x).unwrap_or(0.0);
    let median_y = median_value(&mut center_y).unwrap_or(0.0);
    let initial = rects.keys().copied().min_by(|left, right| {
        let left_rect = rects[left];
        let right_rect = rects[right];
        let left_distance =
            (left_rect.center().x - median_x).hypot(left_rect.center().y - median_y);
        let right_distance =
            (right_rect.center().x - median_x).hypot(right_rect.center().y - median_y);
        left_distance
            .total_cmp(&right_distance)
            .then_with(|| rank[left].cmp(&rank[right]))
    });
    let Some(initial) = initial else {
        return seed_order.to_vec();
    };

    let group_for = |index: usize| {
        capture_group_ids
            .and_then(|groups| images.get(index).and_then(|image| groups.get(&image.id)))
            .copied()
    };
    let mut remaining = rects.keys().copied().collect::<HashSet<_>>();
    let mut ordered = Vec::with_capacity(seed_order.len());
    let mut rendered_rects = Vec::with_capacity(rects.len());
    let mut next = Some(initial);
    while let Some(index) = next {
        if !remaining.remove(&index) {
            break;
        }
        let rect = rects[&index];
        ordered.push(index);
        rendered_rects.push(rect);

        // Keep focus layers from one camera position contiguous. Their
        // rectangles are intentionally coincident, and this lets later layers
        // use the station anchor instead of being mistaken for new panorama
        // tiles.
        next = group_for(index).and_then(|group| {
            remaining
                .iter()
                .filter(|&&candidate| group_for(candidate) == Some(group))
                .min_by_key(|&&candidate| rank[&candidate])
                .copied()
        });
        if next.is_some() {
            continue;
        }
        let scene_diagonal = rendered_rects
            .iter()
            .flat_map(|rect| [rect.left, rect.right])
            .fold(
                (f64::INFINITY, f64::NEG_INFINITY),
                |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
            );
        let scene_vertical = rendered_rects
            .iter()
            .flat_map(|rect| [rect.top, rect.bottom])
            .fold(
                (f64::INFINITY, f64::NEG_INFINITY),
                |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
            );
        let scene_scale = (scene_diagonal.1 - scene_diagonal.0)
            .hypot(scene_vertical.1 - scene_vertical.0)
            .max(1.0);
        next = remaining.iter().copied().min_by(|left, right| {
            let score = |candidate: usize| {
                let candidate_rect = rects[&candidate];
                let candidate_area = candidate_rect.area().max(1.0);
                let maximum_overlap = rendered_rects
                    .iter()
                    .map(|rendered| candidate_rect.overlap_area(*rendered))
                    .fold(0.0, f64::max);
                let overlap_fraction = maximum_overlap / candidate_area;
                let candidate_center = candidate_rect.center();
                let nearest_distance = rendered_rects
                    .iter()
                    .map(|rendered| (candidate_center - rendered.center()).norm())
                    .fold(f64::INFINITY, f64::min)
                    / scene_scale;
                // A real overlap dominates proximity. The distance fallback
                // is only relevant when the graph has a genuine coverage gap.
                overlap_fraction * 100.0 - nearest_distance
            };
            score(*left)
                .total_cmp(&score(*right))
                .reverse()
                .then_with(|| rank[left].cmp(&rank[right]))
        });
    }

    // Keep malformed/unprojectable sources in their original solved order so
    // the caller still reports them instead of silently dropping coverage.
    let ordered_set = ordered.iter().copied().collect::<HashSet<_>>();
    ordered.extend(
        seed_order
            .iter()
            .copied()
            .filter(|index| !ordered_set.contains(index)),
    );
    ordered
}

fn trailing_capture_number(path: &str) -> Option<u64> {
    let stem = Path::new(path).file_stem()?.to_string_lossy();
    let digits = stem
        .chars()
        .rev()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty())
        .then(|| digits.chars().rev().collect::<String>())
        .and_then(|digits| digits.parse().ok())
}

fn ordered_horizontal_capture_shift(
    source: &ImageInfo,
    target: &ImageInfo,
) -> Option<(f64, f64, f64)> {
    const WIDTH: u32 = 320;
    let height = ((source.alignment_image.height() as f64
        / source.alignment_image.width().max(1) as f64)
        * WIDTH as f64)
        .round()
        .clamp(96.0, 320.0) as u32;
    let source_low = image::imageops::resize(
        &source.alignment_image,
        WIDTH,
        height,
        image::imageops::FilterType::Triangle,
    );
    let target_low = image::imageops::resize(
        &target.alignment_image,
        WIDTH,
        height,
        image::imageops::FilterType::Triangle,
    );
    let gradient = |image: &GrayImage| {
        let mut values = vec![0.0f64; (WIDTH * height) as usize];
        for y in 1..height.saturating_sub(1) {
            for x in 1..WIDTH - 1 {
                let gx = f64::from(image.get_pixel(x + 1, y)[0])
                    - f64::from(image.get_pixel(x - 1, y)[0]);
                let gy = f64::from(image.get_pixel(x, y + 1)[0])
                    - f64::from(image.get_pixel(x, y - 1)[0]);
                values[(y * WIDTH + x) as usize] = gx.hypot(gy);
            }
        }
        values
    };
    let source_gradient = gradient(&source_low);
    let target_gradient = gradient(&target_low);
    let mut best: Option<(f64, i32, i32)> = None;
    let maximum_x = (WIDTH as f64 * 0.75).round() as i32;
    let maximum_y = (height as f64 * 0.17).round() as i32;
    for dy in (-maximum_y..=maximum_y).step_by(3) {
        for dx in (0..=maximum_x).step_by(3) {
            let y_start = (-dy).max(0) as u32 + 6;
            let y_end = (height as i32 - dy.max(0)) as u32;
            if y_end <= y_start + 12 || WIDTH <= dx as u32 + 24 {
                continue;
            }
            let mut source_sum = 0.0;
            let mut target_sum = 0.0;
            let mut source_square = 0.0;
            let mut target_square = 0.0;
            let mut product = 0.0;
            let mut count = 0.0;
            for y in (y_start..y_end - 6).step_by(3) {
                for x in (6..WIDTH - dx as u32 - 6).step_by(3) {
                    let source_value = source_gradient[(y * WIDTH + x) as usize];
                    let target_y = (y as i32 + dy) as u32;
                    let target_value = target_gradient[(target_y * WIDTH + x + dx as u32) as usize];
                    source_sum += source_value;
                    target_sum += target_value;
                    source_square += source_value * source_value;
                    target_square += target_value * target_value;
                    product += source_value * target_value;
                    count += 1.0;
                }
            }
            if count < 300.0 {
                continue;
            }
            let covariance = product - source_sum * target_sum / count;
            let variance = (source_square - source_sum * source_sum / count)
                * (target_square - target_sum * target_sum / count);
            if variance <= f64::EPSILON {
                continue;
            }
            let score = covariance / variance.sqrt();
            if score.is_finite() && best.is_none_or(|candidate| score > candidate.0) {
                best = Some((score, dx, dy));
            }
        }
    }
    let (score, dx, dy) = best?;
    let scale = source.width as f64 / WIDTH as f64;
    Some((score, dx as f64 * scale, dy as f64 * scale))
}

fn synthetic_capture_sequence_bridge(
    source: &ImageInfo,
    target: &ImageInfo,
    motion_hint: Option<Matrix3<f64>>,
    capture_steps: u64,
    nominal_overlap_fraction: f64,
) -> Option<MatchInfo> {
    if source.width == 0 || source.height == 0 || target.width == 0 || target.height == 0 {
        return None;
    }
    let source_focal = source.focal_length_35mm.unwrap_or(50.0);
    let target_focal = target.focal_length_35mm.unwrap_or(source_focal);
    let scale = target_focal / source_focal;
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }

    // When consecutive captures have no literal overlap, their sequence still
    // supplies an ordering cue. Place the next frame immediately after the
    // previous frame with a conservative nominal overlap. This path is used
    // only for adjacent numbered captures when visual matching left a gap;
    // real visual matches always remain preferred and still drive the ordinary
    // graph.
    let has_lens_switch = image_pair_has_mixed_focal_lengths(source, target);
    let nominal_overlap = if has_lens_switch {
        // A lens-switch gap in this fixture has adjacent content but no
        // literal overlap. Leave only a small numerical guard so the
        // renderer's covered-rectangle crop does not split the sequence at
        // its one-cell ownership boundary; the bridge still carries no
        // measured points, so no duplicate artwork strip is treated as real
        // evidence.
        32.0
    } else {
        // This bridge remains an ordering prior with no measured points. The
        // caller distinguishes rapid focus captures at one camera pose from
        // the longer pause used to advance a dense artwork scan.
        f64::from(target.width.min(source.width)) * nominal_overlap_fraction.clamp(0.0, 1.0)
    };
    let (translation_x, translation_y) = if let Some(motion_hint) = motion_hint {
        // A missing capture number is often a missing scan row, not a
        // horizontal panorama step.  Extrapolate the measured centre motion
        // immediately before the gap and apply the EXIF focal scale around
        // the optical centre.  This keeps a 120 -> 90mm switch on the same
        // capture trajectory instead of manufacturing a side-by-side tile.
        let source_center = Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5);
        let target_center = Point2::new(target.width as f64 * 0.5, target.height as f64 * 0.5);
        let hinted_center = transformed_point(&motion_hint, source_center);
        let motion = hinted_center.map(|point| point - target_center);
        let steps = capture_steps.max(1) as f64;
        (
            target_center.x - scale * source_center.x
                + motion.map_or(0.0, |value| value.x * scale * steps),
            target_center.y - scale * source_center.y
                + motion.map_or(0.0, |value| value.y * scale * steps),
        )
    } else {
        (
            -(scale * f64::from(source.width) - nominal_overlap.max(1.0)),
            f64::from(target.height) * 0.5 - scale * f64::from(source.height) * 0.5,
        )
    };
    let homography = Matrix3::new(
        scale,
        0.0,
        translation_x,
        0.0,
        scale,
        translation_y,
        0.0,
        0.0,
        1.0,
    );
    let mut points = Vec::new();
    if nominal_overlap > 0.0 && !has_lens_switch {
        for row in 1..=5 {
            for column in 1..=8 {
                let source_point = Point2::new(
                    f64::from(source.width) * column as f64 / 9.0,
                    f64::from(source.height) * row as f64 / 6.0,
                );
                let Some(target_point) = transformed_point(&homography, source_point) else {
                    continue;
                };
                if target_point.x >= 0.0
                    && target_point.y >= 0.0
                    && target_point.x < f64::from(target.width)
                    && target_point.y < f64::from(target.height)
                {
                    points.push((source_point, target_point));
                }
            }
        }
    }
    // Sequence bridges deliberately carry no point observations. The
    // transform is still sufficient to place the next frame in sequence, but
    // the global optimizer must treat it as a prior rather than as measured
    // artwork geometry.
    let inliers = if points.is_empty() {
        SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION
    } else {
        points.len()
    };
    Some(MatchInfo {
        homography,
        inliers,
        sequence_bridge: true,
        coarse_bridge: false,
        candidate_points: points.clone(),
        points,
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
        native_verified_points: Vec::new(),
    })
}

fn synthetic_overview_overlay_bridge(source: &ImageInfo, target: &ImageInfo) -> Option<MatchInfo> {
    if source.width == 0 || source.height == 0 || target.width == 0 || target.height == 0 {
        return None;
    }
    let source_focal = source.focal_length_35mm.unwrap_or(50.0);
    let target_focal = target.focal_length_35mm.unwrap_or(source_focal);
    let scale = target_focal / source_focal;
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    // A wide overview supplied to repair a missing row is a scene reference,
    // not a new horizontal panorama tile. Keep its optical centre over the
    // neighbouring frame; the real detail layers then own the sharp pixels.
    let translation_x = f64::from(target.width) * 0.5 - scale * f64::from(source.width) * 0.5;
    let translation_y = f64::from(target.height) * 0.5 - scale * f64::from(source.height) * 0.5;
    Some(MatchInfo {
        homography: Matrix3::new(
            scale,
            0.0,
            translation_x,
            0.0,
            scale,
            translation_y,
            0.0,
            0.0,
            1.0,
        ),
        inliers: SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION,
        sequence_bridge: true,
        coarse_bridge: false,
        points: Vec::new(),
        candidate_points: Vec::new(),
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
        native_verified_points: Vec::new(),
    })
}

fn synthetic_overview_gap_bridges(
    left: &ImageInfo,
    overview: &ImageInfo,
    right: &ImageInfo,
    left_to_right: &Matrix3<f64>,
) -> Option<(MatchInfo, MatchInfo)> {
    if left.width == 0
        || left.height == 0
        || overview.width == 0
        || overview.height == 0
        || right.width == 0
        || right.height == 0
    {
        return None;
    }
    let left_focal = left.focal_length_35mm.unwrap_or(50.0);
    let overview_focal = overview.focal_length_35mm.unwrap_or(left_focal);
    let right_focal = right.focal_length_35mm.unwrap_or(overview_focal);
    if !left_focal.is_finite()
        || !overview_focal.is_finite()
        || !right_focal.is_finite()
        || left_focal <= 0.0
        || overview_focal <= 0.0
        || right_focal <= 0.0
    {
        return None;
    }

    let left_center = Point2::new(left.width as f64 * 0.5, left.height as f64 * 0.5);
    let overview_center = Point2::new(overview.width as f64 * 0.5, overview.height as f64 * 0.5);
    let right_center = Point2::new(right.width as f64 * 0.5, right.height as f64 * 0.5);
    let mapped_right_center = transformed_point(left_to_right, left_center)?;
    let total_center_motion = mapped_right_center - right_center;
    // A synthetic capture bridge may still be a nominal horizontal placement;
    // only split it when its displacement is a small, plausible continuation.
    // Large motion is precisely the case where the filename gap is a missing
    // row and the bridge contains no measured geometry. Fall back to the
    // bounded optical-centre overlay below instead of amplifying that guess.
    let maximum_plausible_motion = left
        .width
        .max(left.height)
        .max(right.width)
        .max(right.height) as f64
        * 0.35;
    if !total_center_motion.norm().is_finite()
        || total_center_motion.norm() > maximum_plausible_motion
    {
        return None;
    }
    let left_to_overview_scale = overview_focal / left_focal;
    let overview_to_right_scale = right_focal / overview_focal;
    if !left_to_overview_scale.is_finite()
        || !overview_to_right_scale.is_finite()
        || left_to_overview_scale <= 0.0
        || overview_to_right_scale <= 0.0
    {
        return None;
    }

    // The supplied overview is the missing middle row. Preserve the measured
    // left->right centre displacement, but split it at the optical midpoint
    // instead of centring the overview over both neighbours. This keeps the
    // wide frame in the actual gap while retaining its focal-length scale.
    let left_motion = total_center_motion * 0.5 / overview_to_right_scale;
    let right_motion = total_center_motion * 0.5;
    let left_to_overview = Matrix3::new(
        left_to_overview_scale,
        0.0,
        overview_center.x - left_to_overview_scale * left_center.x + left_motion.x,
        0.0,
        left_to_overview_scale,
        overview_center.y - left_to_overview_scale * left_center.y + left_motion.y,
        0.0,
        0.0,
        1.0,
    );
    let overview_to_right = Matrix3::new(
        overview_to_right_scale,
        0.0,
        right_center.x - overview_to_right_scale * overview_center.x + right_motion.x,
        0.0,
        overview_to_right_scale,
        right_center.y - overview_to_right_scale * overview_center.y + right_motion.y,
        0.0,
        0.0,
        1.0,
    );
    Some((
        MatchInfo {
            homography: left_to_overview,
            inliers: SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION,
            sequence_bridge: true,
            coarse_bridge: false,
            points: Vec::new(),
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            canonical_homography: None,
            native_verified_points: Vec::new(),
        },
        MatchInfo {
            homography: overview_to_right,
            inliers: SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION,
            sequence_bridge: true,
            coarse_bridge: false,
            points: Vec::new(),
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            canonical_homography: None,
            native_verified_points: Vec::new(),
        },
    ))
}

fn add_capture_sequence_bridges(
    images: &[ImageInfo],
    matches: &mut HashMap<(usize, usize), MatchInfo>,
) -> usize {
    let mut filename_order = (0..images.len()).collect::<Vec<_>>();
    filename_order
        .sort_by(|&left, &right| natural_path_cmp(&images[left].filename, &images[right].filename));
    let mut added = 0usize;
    let estimates = filename_order
        .windows(2)
        .filter_map(|window| {
            let [source_index, target_index] = window else {
                return None;
            };
            ordered_horizontal_capture_shift(&images[*source_index], &images[*target_index]).map(
                |estimate| {
                    (
                        (
                            (*source_index).min(*target_index),
                            (*source_index).max(*target_index),
                        ),
                        estimate,
                    )
                },
            )
        })
        .collect::<HashMap<_, _>>();
    let mut verified_steps = estimates
        .values()
        .filter_map(|&(score, dx, _)| (score >= 0.40 && dx > 1.0).then_some(dx))
        .collect::<Vec<_>>();
    let fallback_step = median_value(&mut verified_steps).unwrap_or_else(|| {
        images
            .first()
            .map_or(1.0, |image| image.width as f64 * 0.12)
    });
    for window in filename_order.windows(2) {
        let [source_index, target_index] = window else {
            continue;
        };
        let source = &images[*source_index];
        let target = &images[*target_index];
        let Some(source_number) = trailing_capture_number(&source.filename) else {
            continue;
        };
        let Some(target_number) = trailing_capture_number(&target.filename) else {
            continue;
        };
        if target_number <= source_number || target_number - source_number > 3 {
            continue;
        }
        let key = (
            (*source_index).min(*target_index),
            (*source_index).max(*target_index),
        );
        let replace_unreliable = matches.contains_key(&key);
        let capture_steps = target_number.saturating_sub(source_number);
        let (score, measured_dx, measured_dy) =
            estimates
                .get(&key)
                .copied()
                .unwrap_or((f64::NEG_INFINITY, 0.0, 0.0));
        let (dx, dy, used_measurement) = if score >= 0.40 && measured_dx > 1.0 {
            (measured_dx, measured_dy, true)
        } else {
            (fallback_step * capture_steps as f64, 0.0, false)
        };
        let mut match_info = MatchInfo {
            homography: Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0),
            inliers: SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION,
            sequence_bridge: true,
            coarse_bridge: false,
            points: Vec::new(),
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            canonical_homography: None,
            native_verified_points: Vec::new(),
        };
        if source_index > target_index {
            let Some(inverse) = match_info.homography.try_inverse() else {
                continue;
            };
            match_info.homography = inverse;
        }
        matches.insert(key, match_info);
        added += 1;
        println!(
            "  - {} ordered horizontal capture bridge: '{}' -> '{}' ({}, score {:.3}, dx {:.1}, dy {:.1})",
            if replace_unreliable {
                "Replaced with"
            } else {
                "Added"
            },
            Path::new(&source.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            Path::new(&target.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            if used_measurement {
                "measured"
            } else {
                "median fallback"
            },
            score,
            dx,
            dy,
        );
    }
    added
}

/// Insert a supplied wide/overview capture into a filename gap when the
/// capture path visibly changes camera modules but has no literal overlap.
/// This is the case where one missing row is covered by a wider frame: there
/// is no honest feature correspondence to feed the optimizer, but the capture
/// still belongs between the two neighbouring focal-length runs. The inserted
/// links remain priors and carry no artwork observations.
fn add_unmatched_overview_gap_bridges(
    images: &mut [ImageInfo],
    matches: &mut HashMap<(usize, usize), MatchInfo>,
) -> usize {
    if images.len() < 3 {
        return 0;
    }
    let mut filename_order = (0..images.len()).collect::<Vec<_>>();
    filename_order
        .sort_by(|&left, &right| natural_path_cmp(&images[left].filename, &images[right].filename));

    let has_verified_overlap = |left: usize, right: usize| {
        matches
            .get(&(left.min(right), left.max(right)))
            .is_some_and(|match_info| {
                !match_info.sequence_bridge
                    && !match_info.coarse_bridge
                    && match_info.points.len() >= MIXED_FOCAL_MIN_INLIERS
            })
    };
    let orphan_overviews = filename_order
        .iter()
        .copied()
        .filter(|&index| {
            images[index]
                .focal_length_35mm
                .is_some_and(|focal| focal.is_finite() && focal > 0.0)
                && !matches.iter().any(|(&(left, right), match_info)| {
                    (left == index || right == index)
                        && !match_info.sequence_bridge
                        && !match_info.coarse_bridge
                        && match_info.points.len() >= MIXED_FOCAL_MIN_INLIERS
                })
        })
        .collect::<Vec<_>>();
    if orphan_overviews.is_empty() {
        return 0;
    }

    let mut pending = Vec::new();
    for window in filename_order.windows(2) {
        let [left, right] = window else {
            continue;
        };
        if has_verified_overlap(*left, *right) {
            continue;
        }
        let (Some(left_focal), Some(right_focal)) = (
            images[*left].focal_length_35mm,
            images[*right].focal_length_35mm,
        ) else {
            continue;
        };
        let focal_ratio = (left_focal / right_focal).max(right_focal / left_focal);
        if !focal_ratio.is_finite() || focal_ratio < 1.25 {
            continue;
        }
        let minimum_neighbour_focal = left_focal.min(right_focal);
        let Some(overview) = orphan_overviews.iter().copied().find(|&candidate| {
            candidate != *left
                && candidate != *right
                && images[candidate]
                    .focal_length_35mm
                    .is_some_and(|focal| focal < minimum_neighbour_focal * 0.82)
        }) else {
            continue;
        };
        images[overview].overview_reference = true;
        pending.push((*left, overview, *right));
        break;
    }

    let mut added = 0;
    for (left, overview, right) in pending {
        // The generic filename-gap pass may already have inserted a direct
        // synthetic bridge for `left -> right` (for example 1038 -> 1041).
        // Once a real wide frame is available for that gap, keeping both
        // paths makes the order solver ambiguous: it can skip the overview
        // or route through it twice. Remove only that synthetic edge; a real
        // measured match is never replaced here.
        let direct_key = (left.min(right), left.max(right));
        let direct_transform = matches
            .get(&direct_key)
            .filter(|match_info| match_info.sequence_bridge)
            .and_then(|_| focus_stack_link_transform(matches, left, right))
            .map(|(_, transform)| transform);
        if matches
            .get(&direct_key)
            .is_some_and(|match_info| match_info.sequence_bridge)
        {
            matches.remove(&direct_key);
        }
        let Some((left_bridge, right_bridge)) = direct_transform
            .and_then(|transform| {
                synthetic_overview_gap_bridges(
                    &images[left],
                    &images[overview],
                    &images[right],
                    &transform,
                )
            })
            .or_else(|| {
                Some((
                    synthetic_overview_overlay_bridge(&images[left], &images[overview])?,
                    synthetic_overview_overlay_bridge(&images[overview], &images[right])?,
                ))
            })
        else {
            continue;
        };
        for (source, target, mut bridge) in [
            (left, overview, left_bridge),
            (overview, right, right_bridge),
        ] {
            let key = (source.min(target), source.max(target));
            if matches.contains_key(&key) {
                continue;
            }
            // MatchInfo keys are stored with the lower array index first, but
            // the synthetic homography is constructed in capture direction.
            // 0981 sorts before 1038 while this first bridge is 1038 -> 0981;
            // normalize the matrix to the key direction or the reader will
            // apply the reciprocal focal scale and shrink the overview.
            if source > target {
                let Some(inverse) = bridge.homography.try_inverse() else {
                    continue;
                };
                bridge.homography = inverse;
            }
            matches.insert(key, bridge);
            added += 1;
        }
    }
    added
}

fn focus_filename_order_with_overview_gaps(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Vec<usize> {
    let mut base_order = (0..images.len()).collect::<Vec<_>>();
    base_order
        .sort_by(|&left, &right| natural_path_cmp(&images[left].filename, &images[right].filename));
    let mut insertions = Vec::new();
    let mut removed = HashSet::new();
    for window in base_order.windows(2) {
        let [left, right] = window else {
            continue;
        };
        let Some(overview) = (0..images.len()).find(|candidate| {
            if *candidate == *left || *candidate == *right {
                return false;
            }
            if !images[*candidate].overview_reference {
                return false;
            }
            let left_bridge = matches
                .get(&((*left).min(*candidate), (*left).max(*candidate)))
                .is_some_and(|match_info| match_info.sequence_bridge);
            let right_bridge = matches
                .get(&((*candidate).min(*right), (*candidate).max(*right)))
                .is_some_and(|match_info| match_info.sequence_bridge);
            left_bridge && right_bridge
        }) else {
            continue;
        };
        insertions.push((*left, overview, *right));
        removed.insert(overview);
    }
    let mut ordered = Vec::with_capacity(images.len());
    for index in base_order {
        if removed.contains(&index) {
            continue;
        }
        ordered.push(index);
        if let Some((_, overview, _)) = insertions.iter().find(|(left, _, _)| *left == index) {
            ordered.push(*overview);
        }
    }
    ordered
}

fn panorama_edge_weight(source: &ImageInfo, target: &ImageInfo, match_info: &MatchInfo) -> f64 {
    if match_info.points.len() < 4 || match_info.candidate_points.len() < 4 {
        // Unit/test callers may provide only an inlier count. Keep that API
        // behaviour deterministic while real matches use the stronger quality
        // terms below.
        return match_info.inliers as f64;
    }
    let candidate_count = match_info
        .candidate_points
        .len()
        .max(match_info.points.len());
    let inlier_ratio = match_info.points.len() as f64 / candidate_count as f64;
    let error = median_symmetric_error(&match_info.homography, &match_info.points);
    let precision = if error.is_finite() {
        1.0 / (1.0 + error / FULL_RES_RANSAC_INLIER_THRESHOLD)
    } else {
        0.0
    };
    let spatial =
        panorama_spatial_support(&match_info.points, source.dimensions(), target.dimensions());
    let overlap = panorama_transform_overlap_support(
        &match_info.homography,
        source.dimensions(),
        target.dimensions(),
    );
    // A repeated brush pattern can produce many descriptor inliers in a tiny
    // region.  Inlier count alone then makes that false edge the MST backbone.
    // Keep the count dominant, but make broad, low-error, bidirectionally
    // visible overlaps win over compact coincidences.
    (match_info.inliers as f64)
        * (0.35 + 0.65 * inlier_ratio.clamp(0.0, 1.0))
        * (0.35 + 0.65 * precision.clamp(0.0, 1.0))
        * (0.25 + 0.75 * spatial.clamp(0.0, 1.0))
        * (0.25 + 0.75 * overlap.clamp(0.0, 1.0))
}

fn panorama_spatial_support(
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
) -> f64 {
    if points.len() < 4 {
        return 0.0;
    }
    let side_support = |side: usize, dimensions: (u32, u32)| {
        let width = dimensions.0.max(1) as f64;
        let height = dimensions.1.max(1) as f64;
        let mut xs = Vec::with_capacity(points.len());
        let mut ys = Vec::with_capacity(points.len());
        let mut occupied = HashSet::new();
        for &(source, target) in points {
            let point = if side == 0 { source } else { target };
            if !point.x.is_finite() || !point.y.is_finite() {
                continue;
            }
            xs.push(point.x);
            ys.push(point.y);
            let cell_x = ((point.x / width).clamp(0.0, 0.999_999) * 4.0).floor() as u8;
            let cell_y = ((point.y / height).clamp(0.0, 0.999_999) * 4.0).floor() as u8;
            occupied.insert((cell_x, cell_y));
        }
        if xs.len() < 4 {
            return 0.0;
        }
        xs.sort_unstable_by(f64::total_cmp);
        ys.sort_unstable_by(f64::total_cmp);
        let lower = ((xs.len() - 1) as f64 * 0.05).round() as usize;
        let upper = ((xs.len() - 1) as f64 * 0.95).round() as usize;
        let y_lower = ((ys.len() - 1) as f64 * 0.05).round() as usize;
        let y_upper = ((ys.len() - 1) as f64 * 0.95).round() as usize;
        let area = ((xs[upper] - xs[lower]) / width).max(0.0)
            * ((ys[y_upper] - ys[y_lower]) / height).max(0.0);
        let occupancy = occupied.len() as f64 / 16.0;
        (area.sqrt() * 0.7 + occupancy * 0.3).clamp(0.0, 1.0)
    };
    side_support(0, source_dimensions)
        .min(side_support(1, target_dimensions))
        .clamp(0.0, 1.0)
}

fn panorama_transform_overlap_support(
    transform: &Matrix3<f64>,
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
) -> f64 {
    let Some(inverse) = transform.try_inverse() else {
        return 0.0;
    };
    let inside = |mapped: Point3<f64>, dimensions: (u32, u32)| {
        mapped.z.abs() >= 1e-8 && {
            let x = mapped.x / mapped.z;
            let y = mapped.y / mapped.z;
            x >= 0.0 && y >= 0.0 && x < dimensions.0 as f64 && y < dimensions.1 as f64
        }
    };
    let mut forward = 0usize;
    let mut reverse = 0usize;
    for row in 0..=4 {
        for column in 0..=4 {
            let source = Point3::new(
                source_dimensions.0 as f64 * column as f64 / 4.0,
                source_dimensions.1 as f64 * row as f64 / 4.0,
                1.0,
            );
            if inside(*transform * source, target_dimensions) {
                forward += 1;
            }
            let target = Point3::new(
                target_dimensions.0 as f64 * column as f64 / 4.0,
                target_dimensions.1 as f64 * row as f64 / 4.0,
                1.0,
            );
            if inside(inverse * target, source_dimensions) {
                reverse += 1;
            }
        }
    }
    (forward.min(reverse) as f64 / 25.0).clamp(0.0, 1.0)
}

#[cfg(test)]
fn pair_harness_disconnected_focus_geometry(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> (Vec<usize>, HashMap<usize, Matrix3<f64>>) {
    let mut adjacency = vec![Vec::<usize>::new(); images.len()];
    for &(left, right) in matches.keys() {
        if left < images.len() && right < images.len() && left != right {
            adjacency[left].push(right);
            adjacency[right].push(left);
        }
    }
    for neighbours in &mut adjacency {
        neighbours.sort_by(|&left, &right| images[left].filename.cmp(&images[right].filename));
        neighbours.dedup();
    }
    let mut roots = (0..images.len()).collect::<Vec<_>>();
    roots.sort_by(|&left, &right| images[left].filename.cmp(&images[right].filename));
    let separation = f64::from(
        images
            .iter()
            .map(|image| image.width.max(image.height))
            .max()
            .unwrap_or(1),
    ) * 2.5;
    let mut order = Vec::with_capacity(images.len());
    let mut homographies = HashMap::with_capacity(images.len());
    let mut visited = vec![false; images.len()];
    let mut component = 0usize;
    for root in roots {
        if visited[root] {
            continue;
        }
        let origin = virtual_tile_translation(component as f64 * separation, 0.0);
        component += 1;
        visited[root] = true;
        let mut pending = VecDeque::from([(root, origin)]);
        while let Some((current, current_to_world)) = pending.pop_front() {
            order.push(current);
            homographies.insert(images[current].id, current_to_world);
            for &next in &adjacency[current] {
                if visited[next] {
                    continue;
                }
                let next_to_current = if let Some(relation) = matches.get(&(next, current)) {
                    relation.homography
                } else if let Some(relation) = matches.get(&(current, next)) {
                    let Some(inverse) = relation.homography.try_inverse() else {
                        continue;
                    };
                    inverse
                } else {
                    continue;
                };
                visited[next] = true;
                pending.push_back((next, current_to_world * next_to_current));
            }
        }
    }
    println!(
        "  - Pair harness retained {} disconnected source component(s) as finite rendering priors",
        component
    );
    (order, homographies)
}

fn build_graph_stitching_order<F>(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    edge_weight: F,
) -> (Vec<usize>, HashMap<usize, Matrix3<f64>>)
where
    F: Fn(&ImageInfo, &ImageInfo, &MatchInfo) -> f64,
{
    if images.is_empty() {
        return (vec![], HashMap::new());
    }
    let n = images.len();
    if n < 2 {
        let mut homographies = HashMap::new();
        if n == 1 {
            homographies.insert(images[0].id, Matrix3::identity());
        }
        return ((0..n).collect(), homographies);
    }

    let mut edges = matches
        .iter()
        .filter_map(|(&(i, j), match_info)| {
            let source = images.get(i)?;
            let target = images.get(j)?;
            let weight = edge_weight(source, target, match_info);
            if !weight.is_finite() {
                println!(
                    "  - Ignoring non-finite overlap edge '{}' <-> '{}'",
                    source.filename, target.filename
                );
                return None;
            }
            Some((weight, i, j))
        })
        .collect::<Vec<_>>();
    edges.sort_by(|left, right| {
        let left_names = {
            let first = images[left.1].filename.as_str();
            let second = images[left.2].filename.as_str();
            if first <= second {
                (first, second)
            } else {
                (second, first)
            }
        };
        let right_names = {
            let first = images[right.1].filename.as_str();
            let second = images[right.2].filename.as_str();
            if first <= second {
                (first, second)
            } else {
                (second, first)
            }
        };
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left_names.cmp(&right_names))
            // Duplicate paths in the selection would leave the name comparison
            // equal; the source index pair is the stable secondary key (需求 14.6).
            .then_with(|| (left.1, left.2).cmp(&(right.1, right.2)))
    });
    let mut mst_adj: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut dsu = Dsu::new(n);
    let mut num_edges = 0;

    for &(_, i, j) in &edges {
        if dsu.find(i) != dsu.find(j) {
            dsu.union(i, j);
            mst_adj.entry(i).or_default().push(j);
            mst_adj.entry(j).or_default().push(i);
            num_edges += 1;
            if num_edges == n - 1 {
                break;
            }
        }
    }

    // Kruskal produces a forest when the source selection contains separate
    // scenes.  The previous code chose a start node across the whole forest,
    // so a tiny two-image component could win merely because it had a low
    // degree.  Traverse the largest connected component deterministically.
    let mut component_visited = HashSet::new();
    let mut components = Vec::<Vec<usize>>::new();
    for root in 0..n {
        if component_visited.contains(&root) || !mst_adj.contains_key(&root) {
            continue;
        }
        let mut component = Vec::new();
        let mut pending = vec![root];
        component_visited.insert(root);
        while let Some(node) = pending.pop() {
            component.push(node);
            if let Some(neighbours) = mst_adj.get(&node) {
                for &neighbour in neighbours {
                    if component_visited.insert(neighbour) {
                        pending.push(neighbour);
                    }
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components.sort_by(|left, right| {
        right.len().cmp(&left.len()).then_with(|| {
            let left_name = left
                .iter()
                .map(|&index| images[index].filename.as_str())
                .min()
                .unwrap_or("");
            let right_name = right
                .iter()
                .map(|&index| images[index].filename.as_str())
                .min()
                .unwrap_or("");
            natural_path_cmp(left_name, right_name)
        })
    });
    if components.len() > 1 {
        let summary = components
            .iter()
            .map(|component| {
                let first = component
                    .iter()
                    .map(|&index| {
                        Path::new(&images[index].filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                    })
                    .min()
                    .unwrap_or_default();
                let last = component
                    .iter()
                    .map(|&index| {
                        Path::new(&images[index].filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                    })
                    .max()
                    .unwrap_or_default();
                format!("{} [{first}..{last}]", component.len())
            })
            .collect::<Vec<_>>();
        println!(
            "  - Verified overlap graph components: {}",
            summary.join(", ")
        );
    }
    let selected_component = components.first();
    let start_node = selected_component
        .into_iter()
        .flat_map(|component| component.iter().copied())
        .min_by(|&left, &right| {
            mst_adj
                .get(&left)
                .map_or(usize::MAX, |neighbors| neighbors.len())
                .cmp(
                    &mst_adj
                        .get(&right)
                        .map_or(usize::MAX, |neighbors| neighbors.len()),
                )
                .then_with(|| images[left].filename.cmp(&images[right].filename))
        })
        // `components` is empty only when `mst_adj` is empty, so this arm always
        // yields 0 in practice.  Take the lowest key rather than an arbitrary one
        // so that no hash-seed dependent value can ever reach the pose graph
        // (需求 14.6).
        .unwrap_or_else(|| mst_adj.keys().copied().min().unwrap_or(0));

    let mut ordered_indices = Vec::new();
    let mut global_homographies = HashMap::new();
    let mut q = VecDeque::new();
    let mut visited = HashSet::new();

    q.push_back((start_node, Matrix3::identity()));
    visited.insert(start_node);

    while let Some((u, h_u_global)) = q.pop_front() {
        ordered_indices.push(u);
        global_homographies.insert(u, h_u_global);

        if let Some(neighbors) = mst_adj.get(&u) {
            for &v in neighbors {
                if !visited.contains(&v) {
                    visited.insert(v);

                    let h_vu = if let Some(m) = matches.get(&(v, u)) {
                        m.homography
                    } else if let Some(m) = matches.get(&(u, v)) {
                        m.homography
                            .try_inverse()
                            .expect("Failed to invert homography for MST edge")
                    } else {
                        panic!("Match not found for MST edge between {} and {}", u, v);
                    };

                    let h_v_global = h_u_global * h_vu;
                    q.push_back((v, h_v_global));
                }
            }
        }
    }

    (ordered_indices, global_homographies)
}

/// Close the translation drift of the maximum-support panorama tree using all
/// measured flat-artwork overlaps. The MST is a stable initialization, but a
/// long raster can otherwise accumulate a small error at every step and make a
/// later source-sized exposure block look like a seam. Keep the measured
/// similarity scale/rotation untouched; solve only a bounded world translation
/// correction from the redundant pair relations.
fn refine_flat_artwork_tree_poses(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    base_poses: &HashMap<usize, Matrix3<f64>>,
) -> Option<HashMap<usize, Matrix3<f64>>> {
    if base_poses.len() < 3 || matches.len() <= base_poses.len().saturating_sub(1) {
        return None;
    }
    let mut constraints = Vec::<(usize, usize, [f64; 2], f64)>::new();
    let mut connectivity = Dsu::new(images.len());
    for (&(left, right), relation) in matches {
        let (Some(left_pose), Some(right_pose)) = (
            base_poses.get(&images[left].id),
            base_poses.get(&images[right].id),
        ) else {
            continue;
        };
        if left == right || relation.points.len() < 8 {
            continue;
        }
        let mut residuals = Vec::new();
        for &(left_point, right_point) in &relation.points {
            let Some(left_world) = transformed_point(left_pose, left_point) else {
                continue;
            };
            let Some(right_world) = transformed_point(right_pose, right_point) else {
                continue;
            };
            let delta = left_world - right_world;
            if delta.x.is_finite() && delta.y.is_finite() {
                residuals.push(delta);
            }
        }
        if residuals.len() < 8 {
            continue;
        }
        let mut dx = residuals.iter().map(|point| point.x).collect::<Vec<_>>();
        let mut dy = residuals.iter().map(|point| point.y).collect::<Vec<_>>();
        let (Some(dx), Some(dy)) = (median_value(&mut dx), median_value(&mut dy)) else {
            continue;
        };
        let median_error = median_symmetric_error(&relation.homography, &relation.points);
        let weight = (relation.points.len() as f64).sqrt() / (1.0 + median_error.max(0.0) / 4.0);
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        connectivity.union(left, right);
        constraints.push((left, right, [dx, dy], weight));
    }
    if constraints.len() <= base_poses.len().saturating_sub(1) {
        return None;
    }
    let reference = base_poses
        .keys()
        .copied()
        .min_by(|left, right| images[*left].filename.cmp(&images[*right].filename))?;
    let reference_root = connectivity.find(reference);
    if !base_poses
        .keys()
        .all(|index| connectivity.find(*index) == reference_root)
    {
        return None;
    }
    let solve = solve_robust_closure_translations(images.len(), reference, &constraints)?;
    if !solve.converged {
        return None;
    }
    let mut corrected = base_poses.clone();
    let mut max_correction: f64 = 0.0;
    for (&index, pose) in base_poses {
        let correction = solve.corrections.get(index).copied().unwrap_or([0.0, 0.0]);
        let magnitude = correction[0].hypot(correction[1]);
        // A correction larger than one source frame is evidence that the
        // redundant graph is mixing independent scenes; retain the measured
        // tree in that case instead of making a plausible-looking collage.
        if !magnitude.is_finite()
            || magnitude > images[index].width.max(images[index].height) as f64
        {
            return None;
        }
        max_correction = max_correction.max(magnitude);
        let translation = Matrix3::new(
            1.0,
            0.0,
            correction[0],
            0.0,
            1.0,
            correction[1],
            0.0,
            0.0,
            1.0,
        );
        corrected.insert(index, translation * pose);
    }
    let mut solved_residuals = constraints
        .iter()
        .map(|&(left, right, observed, _)| {
            (solve.corrections[right][0] - solve.corrections[left][0] - observed[0])
                .hypot(solve.corrections[right][1] - solve.corrections[left][1] - observed[1])
        })
        .collect::<Vec<_>>();
    let median_residual = median_value(&mut solved_residuals).unwrap_or(f64::INFINITY);
    println!(
        "  - Flat-artwork pose closure accepted: {} redundant relation(s), median residual {:.2}px, maximum translation correction {:.1}px",
        constraints.len(),
        median_residual,
        max_correction,
    );
    Some(corrected)
}

/// The selected sources that produced no decoded image (需求 1.9).
///
/// Returned in selection order and by absolute path, which is what the
/// Stack_Report and the degradation ledger record.  Excluding them from every
/// Capture_Station is not a separate step: the grouper only ever sees the decoded
/// images, so a source named here cannot reach it.
fn undecodable_source_paths(selection: &[String], decoded: &[ImageInfo]) -> Vec<String> {
    selection
        .iter()
        .filter(|path| !decoded.iter().any(|image| &image.filename == *path))
        .cloned()
        .collect()
}

/// The Stack_Report records of the sources one grouping excluded (需求 1.6 / 1.9).
///
/// Each record names the absolute path of the excluded Source_RAW and the stable
/// reason identifier it was excluded for, and nothing else: the exclusion itself
/// has already happened by the time this runs.
fn excluded_source_records(
    images: &[ImageInfo],
    indices: &[usize],
    reason: &'static str,
) -> Vec<stack_report::ExcludedSourceRecord> {
    indices
        .iter()
        .filter_map(|&image_index| images.get(image_index))
        .map(|image| stack_report::ExcludedSourceRecord {
            path: image.filename.clone(),
            reason: reason.to_string(),
        })
        .collect()
}

struct FocusAnchorRenderFrames {
    homographies: HashMap<usize, Matrix3<f64>>,
    focus_warp: Option<FocusLayerWarp>,
    anchor_to_world: Vec<Matrix3<f64>>,
}

fn focus_anchor_render_frames(
    groups: &[FocusCaptureGroup],
    sources: &[ImageInfo],
    global: &HashMap<usize, Matrix3<f64>>,
    warp: Option<&FocusLayerWarp>,
) -> Option<FocusAnchorRenderFrames> {
    let reference_anchor = sources.get(groups.first()?.anchor)?;
    let world_to_reference = global.get(&reference_anchor.id)?.try_inverse()?;
    let mut local = HashMap::new();
    let mut local_warp = warp.cloned();
    let mut placements = Vec::with_capacity(groups.len());
    for group in groups {
        let anchor = sources.get(group.anchor)?;
        let anchor_global = global.get(&anchor.id)?;
        let world_to_anchor = anchor_global.try_inverse()?;
        placements.push(normalized_homography(
            &(world_to_reference * anchor_global),
        )?);
        for &member in &group.members {
            let source = sources.get(member)?;
            local.insert(
                source.id,
                normalized_homography(&(world_to_anchor * global.get(&source.id)?))?,
            );
            if let Some(warp) = local_warp.as_mut() {
                for band in &mut warp.bands {
                    if let Some(pose) = band.homographies.get_mut(&source.id) {
                        *pose = normalized_homography(&(world_to_anchor * *pose))?;
                    }
                }
            }
        }
    }
    Some(FocusAnchorRenderFrames {
        homographies: local,
        focus_warp: local_warp,
        anchor_to_world: placements,
    })
}

const VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS: u32 = 20;

/// Convert a rendered Virtual_Tile to its native-pixel station-matching
/// representation.
///
/// Uncovered RGB values are never read. The deterministic zero outside the
/// mask prevents transparent-corner payloads from changing detection, while
/// the final erosion keeps both FAST and BRIEF support wholly inside genuine
/// Coverage_Mask pixels (需求 6.1). Production matches the area-averaged planes
/// of [`install_virtual_tile_match_planes`]; this native form backs the tests.
#[cfg(test)]
fn virtual_tile_covered_features(
    pixels: &Rgb32FImage,
    coverage: &stitching::CoverageMask,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
) -> (GrayImage, Vec<Feature>) {
    let (width, height) = pixels.dimensions();
    if coverage.dimensions() != (width, height) {
        return (GrayImage::new(width, height), Vec::new());
    }
    let gray = virtual_tile_covered_luminance(pixels, coverage);
    let radius = VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS;
    let features = processing::find_features(&gray, brief_pairs)
        .into_iter()
        .filter(|feature| {
            let x = feature.keypoint.x;
            let y = feature.keypoint.y;
            x >= radius
                && y >= radius
                && x.saturating_add(radius) < width
                && y.saturating_add(radius) < height
                && (y - radius..=y + radius).all(|sample_y| {
                    (x - radius..=x + radius)
                        .all(|sample_x| coverage.is_covered(sample_x, sample_y))
                })
        })
        .collect();
    (gray, features)
}

const VIRTUAL_TILE_GUIDED_PATCH_RADIUS: i32 = 6;
const VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC: f64 = 0.55;
const VIRTUAL_TILE_GUIDED_MIN_GRADIENT_NCC: f64 = 0.35;
const VIRTUAL_TILE_GUIDED_MIN_PEAK_MARGIN: f64 = 0.025;
const VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES: f64 = 15.0;
const VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE: u32 = 64;
const VIRTUAL_TILE_GUIDED_RATIO: f64 = 0.90;
const VIRTUAL_TILE_GUIDED_MIN_SEARCH_RADIUS_PX: f64 = 12.0;
const VIRTUAL_TILE_GUIDED_MAX_SEARCH_RADIUS_PX: f64 = 96.0;
const VIRTUAL_TILE_GUIDED_GRID_COLUMNS: usize = 6;
const VIRTUAL_TILE_GUIDED_GRID_ROWS: usize = 6;
const VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA: usize = 2;
const VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS: usize = 12;

#[derive(Clone, Copy, Debug)]
struct VirtualTileStructuralScore {
    patch_ncc: f64,
    gradient_ncc: f64,
    orientation_difference_degrees: f64,
}

impl VirtualTileStructuralScore {
    fn rank(self, descriptor_distance: u32) -> f64 {
        self.patch_ncc + 0.5 * self.gradient_ncc
            - f64::from(descriptor_distance)
                / f64::from(VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE.max(1))
                * 0.05
    }
}

#[derive(Clone, Copy, Debug)]
struct VirtualTileGuidedFeatureMatch {
    target_index: usize,
    rank: f64,
    orientation_difference_degrees: f64,
}

#[cfg(test)]
struct VirtualTileStationMatchOutcome {
    relations: HashMap<(usize, usize), MatchInfo>,
    diagnostics: Vec<stack_report::StationRelationCandidateRecord>,
}

#[cfg(test)]
fn virtual_tile_unseeded_station_match(
    left: usize,
    right: usize,
    tiles: &[ImageInfo],
) -> Option<MatchInfo> {
    let descriptor_matches =
        processing::match_features(&tiles[left].features, &tiles[right].features);
    let left_keypoints = tiles[left]
        .features
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let right_keypoints = tiles[right]
        .features
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let (homography, inliers) =
        processing::find_homography_ransac(&descriptor_matches, &left_keypoints, &right_keypoints)?;
    let points = inliers
        .iter()
        .map(|matched| {
            let source = left_keypoints[matched.index1];
            let target = right_keypoints[matched.index2];
            (
                Point2::new(source.x as f64, source.y as f64),
                Point2::new(target.x as f64, target.y as f64),
            )
        })
        .collect::<Vec<_>>();
    Some(MatchInfo {
        homography,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: false,
        points,
        candidate_points: Vec::new(),
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        // Key-direction Virtual_Tile fit; see `MatchInfo::canonical_homography`.
        canonical_homography: None,
        native_verified_points: Vec::new(),
    })
}

fn virtual_tile_point_is_covered(coverage: &stitching::CoverageMask, point: Point2<f64>) -> bool {
    let (width, height) = coverage.dimensions();
    point.x.is_finite()
        && point.y.is_finite()
        && point.x >= 0.0
        && point.y >= 0.0
        && point.x < width as f64
        && point.y < height as f64
        && coverage.is_covered(point.x.round() as u32, point.y.round() as u32)
}

fn virtual_tile_local_jacobian(
    transform: &Matrix3<f64>,
    point: Point2<f64>,
) -> Option<nalgebra::Matrix2<f64>> {
    let center = transformed_point(transform, point)?;
    let horizontal = transformed_point(transform, point + nalgebra::Vector2::new(1.0, 0.0))?;
    let vertical = transformed_point(transform, point + nalgebra::Vector2::new(0.0, 1.0))?;
    let jacobian = nalgebra::Matrix2::new(
        horizontal.x - center.x,
        vertical.x - center.x,
        horizontal.y - center.y,
        vertical.y - center.y,
    );
    (jacobian.iter().all(|value| value.is_finite()) && jacobian.determinant().abs() > 1e-8)
        .then_some(jacobian)
}

fn virtual_tile_gradient(image: &GrayImage, point: Point2<f64>) -> Option<nalgebra::Vector2<f64>> {
    let gx = registration::sample_gray(image, point.x + 1.0, point.y)?
        - registration::sample_gray(image, point.x - 1.0, point.y)?;
    let gy = registration::sample_gray(image, point.x, point.y + 1.0)?
        - registration::sample_gray(image, point.x, point.y - 1.0)?;
    let gradient = nalgebra::Vector2::new(gx, gy);
    gradient
        .iter()
        .all(|value| value.is_finite())
        .then_some(gradient)
}

/// A projective repair's scale belongs at the station, not at the unrelated
/// world origin. Reading its normalized linear block made the bound depend on
/// canvas translation and rejected a near-unit correction as a 1.5x jump.
fn virtual_tile_repair_local_scale_ratio(
    transform: &Matrix3<f64>,
    center: Point2<f64>,
) -> Option<f64> {
    let ratio = virtual_tile_local_jacobian(transform, center)?
        .determinant()
        .abs()
        .sqrt();
    (ratio.is_finite() && ratio > 0.0).then_some(ratio)
}

fn axial_orientation_difference_degrees(
    left: nalgebra::Vector2<f64>,
    right: nalgebra::Vector2<f64>,
) -> Option<f64> {
    let denominator = left.norm() * right.norm();
    if !denominator.is_finite() || denominator <= 1e-8 {
        return None;
    }
    let cosine = (left.dot(&right) / denominator).abs().clamp(0.0, 1.0);
    Some(cosine.acos().to_degrees())
}

fn virtual_tile_gradient_magnitude_ncc(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    if let Some(correlation) = normalized_correlation(left, right) {
        return Some(correlation);
    }
    // A constant-strength edge has zero variance but still carries orientation
    // and location evidence. Treat equal non-zero magnitudes as perfect NCC;
    // unequal or flat gradients remain non-matches.
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let scale = left_mean.abs().max(right_mean.abs());
    (scale > 1e-8).then(|| (1.0 - (left_mean - right_mean).abs() / scale).clamp(-1.0, 1.0))
}

#[allow(clippy::too_many_arguments)]
fn virtual_tile_structural_score_points(
    left: &GrayImage,
    right: &GrayImage,
    left_coverage: &stitching::CoverageMask,
    right_coverage: &stitching::CoverageMask,
    center: Point2<f64>,
    right_center: Point2<f64>,
    left_to_right: &Matrix3<f64>,
) -> Option<VirtualTileStructuralScore> {
    let projected_center = transformed_point(left_to_right, center)?;
    let residual = right_center - projected_center;
    let jacobian = virtual_tile_local_jacobian(left_to_right, center)?;
    let gradient_transform = jacobian.try_inverse()?.transpose();
    let radius = VIRTUAL_TILE_GUIDED_PATCH_RADIUS;
    let capacity = ((radius * 2 + 1) * (radius * 2 + 1)) as usize;
    let mut left_values = Vec::with_capacity(capacity);
    let mut right_values = Vec::with_capacity(capacity);
    let mut left_gradient_magnitudes = Vec::with_capacity(capacity);
    let mut right_gradient_magnitudes = Vec::with_capacity(capacity);
    let mut orientation_differences = Vec::with_capacity(capacity);

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let source = center + nalgebra::Vector2::new(f64::from(dx), f64::from(dy));
            let target = transformed_point(left_to_right, source)? + residual;
            if !virtual_tile_point_is_covered(left_coverage, source)
                || !virtual_tile_point_is_covered(right_coverage, target)
            {
                return None;
            }
            left_values.push(registration::sample_gray(left, source.x, source.y)?);
            right_values.push(registration::sample_gray(right, target.x, target.y)?);

            let left_gradient = virtual_tile_gradient(left, source)?;
            let right_gradient = virtual_tile_gradient(right, target)?;
            let transformed_gradient = gradient_transform * left_gradient;
            left_gradient_magnitudes.push(transformed_gradient.norm());
            right_gradient_magnitudes.push(right_gradient.norm());
            if transformed_gradient.norm().min(right_gradient.norm()) >= 4.0
                && let Some(difference) =
                    axial_orientation_difference_degrees(transformed_gradient, right_gradient)
            {
                orientation_differences.push(difference);
            }
        }
    }

    let patch_ncc = normalized_correlation(&left_values, &right_values)?;
    let gradient_ncc =
        virtual_tile_gradient_magnitude_ncc(&left_gradient_magnitudes, &right_gradient_magnitudes)?;
    let orientation_difference_degrees = median_value(&mut orientation_differences)?;
    Some(VirtualTileStructuralScore {
        patch_ncc,
        gradient_ncc,
        orientation_difference_degrees,
    })
}

#[allow(clippy::too_many_arguments)]
fn virtual_tile_structural_score(
    left: &GrayImage,
    right: &GrayImage,
    left_coverage: &stitching::CoverageMask,
    right_coverage: &stitching::CoverageMask,
    left_point: KeyPoint,
    right_point: KeyPoint,
    left_to_right: &Matrix3<f64>,
) -> Option<VirtualTileStructuralScore> {
    virtual_tile_structural_score_points(
        left,
        right,
        left_coverage,
        right_coverage,
        Point2::new(f64::from(left_point.x), f64::from(left_point.y)),
        Point2::new(f64::from(right_point.x), f64::from(right_point.y)),
        left_to_right,
    )
}

fn virtual_tile_descriptor_distance(left: &Descriptor, right: &Descriptor) -> u32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (left ^ right).count_ones())
        .sum()
}

/// Search radius in the Virtual_Tile analysis plane.
///
/// The source-stage station geometry supplies a measured uncertainty of at
/// most `FOCUS_MODEL_INLIER_THRESHOLD`; three such sigmas cover its current
/// geometric uncertainty. Sparse instruments add two standard sampling errors:
/// `sqrt(overlap_area / count) / sqrt(count)`. The result is converted through
/// the actual analysis scale and bounded to eight percent of the shorter tile
/// side (and 96 analysis pixels) so a prior can never turn into an unrestricted
/// repeated-texture search.
fn virtual_tile_guided_search_radius(
    left: &ImageInfo,
    right: &ImageInfo,
    predicted_overlap_area: f64,
    instrument_count: usize,
) -> f64 {
    let analysis_scale = left.scale_factor.max(right.scale_factor).max(1.0);
    let count = instrument_count.max(1) as f64;
    let sampling_error = (predicted_overlap_area.max(1.0) / count).sqrt() / count.sqrt();
    let measured_geometry_error = 3.0 * FOCUS_MODEL_INLIER_THRESHOLD / analysis_scale;
    let dimension_cap = 0.08
        * f64::from(
            left.width
                .min(left.height)
                .min(right.width.min(right.height)),
        );
    (measured_geometry_error + 2.0 * sampling_error).clamp(
        VIRTUAL_TILE_GUIDED_MIN_SEARCH_RADIUS_PX,
        VIRTUAL_TILE_GUIDED_MAX_SEARCH_RADIUS_PX
            .min(dimension_cap.max(VIRTUAL_TILE_GUIDED_MIN_SEARCH_RADIUS_PX)),
    )
}

fn virtual_tile_feature_buckets(
    features: &[Feature],
    eligible: &[bool],
    cell_size: f64,
) -> BTreeMap<(i32, i32), Vec<usize>> {
    let mut buckets = BTreeMap::<(i32, i32), Vec<usize>>::new();
    for (index, feature) in features.iter().enumerate() {
        if !eligible.get(index).copied().unwrap_or(false) {
            continue;
        }
        let key = (
            (f64::from(feature.keypoint.x) / cell_size).floor() as i32,
            (f64::from(feature.keypoint.y) / cell_size).floor() as i32,
        );
        buckets.entry(key).or_default().push(index);
    }
    buckets
}

#[allow(clippy::too_many_arguments)]
fn virtual_tile_best_guided_feature(
    source_index: usize,
    source_features: &[Feature],
    target_features: &[Feature],
    target_buckets: &BTreeMap<(i32, i32), Vec<usize>>,
    source_gray: &GrayImage,
    target_gray: &GrayImage,
    source_coverage: &stitching::CoverageMask,
    target_coverage: &stitching::CoverageMask,
    source_to_target: &Matrix3<f64>,
    search_radius: f64,
    orientation_observations: &mut Vec<f64>,
) -> Option<VirtualTileGuidedFeatureMatch> {
    let source = source_features.get(source_index)?;
    let projected = transformed_point(
        source_to_target,
        Point2::new(f64::from(source.keypoint.x), f64::from(source.keypoint.y)),
    )?;
    let center = (
        (projected.x / search_radius).floor() as i32,
        (projected.y / search_radius).floor() as i32,
    );
    let mut candidates = Vec::<(f64, u32, usize, f64)>::new();
    for bucket_y in center.1 - 1..=center.1 + 1 {
        for bucket_x in center.0 - 1..=center.0 + 1 {
            for &target_index in target_buckets
                .get(&(bucket_x, bucket_y))
                .into_iter()
                .flatten()
            {
                let target = &target_features[target_index];
                let displacement = Point2::new(
                    f64::from(target.keypoint.x) - projected.x,
                    f64::from(target.keypoint.y) - projected.y,
                );
                if displacement.coords.norm() > search_radius {
                    continue;
                }
                let descriptor_distance =
                    virtual_tile_descriptor_distance(&source.descriptor, &target.descriptor);
                if descriptor_distance > VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE {
                    continue;
                }
                let Some(structure) = virtual_tile_structural_score(
                    source_gray,
                    target_gray,
                    source_coverage,
                    target_coverage,
                    source.keypoint,
                    target.keypoint,
                    source_to_target,
                ) else {
                    continue;
                };
                // The rough station pose is only a search prior. Its local
                // Jacobian may have the wrong rotation, so pre-fit orientation
                // is recorded for diagnostics but cannot reject a measured
                // correspondence. Patch/gradient structure remains evidence.
                orientation_observations.push(structure.orientation_difference_degrees);
                if structure.patch_ncc < VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC
                    || structure.gradient_ncc < VIRTUAL_TILE_GUIDED_MIN_GRADIENT_NCC
                {
                    continue;
                }
                candidates.push((
                    structure.rank(descriptor_distance),
                    descriptor_distance,
                    target_index,
                    structure.orientation_difference_degrees,
                ));
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| {
                let left_point = target_features[left.2].keypoint;
                let right_point = target_features[right.2].keypoint;
                (left_point.y, left_point.x, left.2).cmp(&(right_point.y, right_point.x, right.2))
            })
    });
    let &(best_rank, best_distance, best_index, orientation_difference_degrees) =
        candidates.first()?;
    if let Some(&(second_rank, second_distance, _, _)) = candidates.get(1)
        && (second_distance == 0
            || f64::from(best_distance) / f64::from(second_distance) >= VIRTUAL_TILE_GUIDED_RATIO
            || best_rank - second_rank < VIRTUAL_TILE_GUIDED_MIN_PEAK_MARGIN)
    {
        return None;
    }
    Some(VirtualTileGuidedFeatureMatch {
        target_index: best_index,
        rank: best_rank,
        orientation_difference_degrees,
    })
}

fn virtual_tile_predicted_overlap_bounds(
    left_to_right: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
) -> Option<(f64, f64, f64, f64)> {
    let right_to_left = left_to_right.try_inverse()?;
    let footprint = [
        Point2::new(0.0, 0.0),
        Point2::new(f64::from(right_dimensions.0), 0.0),
        Point2::new(f64::from(right_dimensions.0), f64::from(right_dimensions.1)),
        Point2::new(0.0, f64::from(right_dimensions.1)),
    ]
    .into_iter()
    .map(|point| transformed_point(&right_to_left, point))
    .collect::<Option<Vec<_>>>()?;
    let clipped = clip_polygon_to_rectangle(&footprint, left_dimensions);
    if clipped.len() < 3 {
        return None;
    }
    let min_x = clipped
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let max_x = clipped
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = clipped
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min);
    let max_y = clipped
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max);
    (min_x.is_finite()
        && max_x.is_finite()
        && min_y.is_finite()
        && max_y.is_finite()
        && max_x > min_x
        && max_y > min_y)
        .then_some((min_x, min_y, max_x, max_y))
}

fn virtual_tile_overlap_grid_cell(
    point: Point2<f64>,
    bounds: (f64, f64, f64, f64),
) -> (usize, usize) {
    let (min_x, min_y, max_x, max_y) = bounds;
    let x = (((point.x - min_x) / (max_x - min_x)) * VIRTUAL_TILE_GUIDED_GRID_COLUMNS as f64)
        .floor()
        .clamp(0.0, (VIRTUAL_TILE_GUIDED_GRID_COLUMNS - 1) as f64) as usize;
    let y = (((point.y - min_y) / (max_y - min_y)) * VIRTUAL_TILE_GUIDED_GRID_ROWS as f64)
        .floor()
        .clamp(0.0, (VIRTUAL_TILE_GUIDED_GRID_ROWS - 1) as f64) as usize;
    (x, y)
}

#[allow(dead_code)]
fn virtual_tile_descriptor_guided_station_match(
    left: usize,
    right: usize,
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
) -> (
    Option<MatchInfo>,
    stack_report::StationRelationCandidateRecord,
) {
    let mut diagnostic = stack_report::StationRelationCandidateRecord {
        left,
        right,
        left_tile: Default::default(),
        right_tile: Default::default(),
        predicted_overlap_area_px: 0.0,
        model_fitted: false,
        evidence_kind: stack_report::StationRelationEvidenceKind::VirtualTile,
        grid_probe_count: 0,
        measurable_patch_count: 0,
        bidirectional_match_count: 0,
        occupied_grid_cells: 0,
        search_radius_px: 0.0,
        residual_dx_px: Default::default(),
        residual_dy_px: Default::default(),
        residual_magnitude_px: Default::default(),
        max_residual_translation_consensus: 0,
        residual_translation: Default::default(),
        residual_affine: Default::default(),
        residual_projective: Default::default(),
        residual_model: "not_run".to_string(),
        fitted_inliers: 0,
        hull_support: 0.0,
        median_prefit_structural_orientation_difference_degrees: None,
        median_fitted_orientation_difference_degrees: None,
        failure_stage: "missing_input".to_string(),
        pyramid_factors: Vec::new(),
        measurement_factor: 1,
        seed: VIRTUAL_TILE_SEED_RENDERING_PRIOR.to_string(),
        prior_free_inliers: 0,
        global_proposals: Vec::new(),
        after_prior_repair: false,
        hypotheses: Vec::new(),
    };
    let Some(left_tile) = tiles.get(left) else {
        return (None, diagnostic);
    };
    let Some(right_tile) = tiles.get(right) else {
        return (None, diagnostic);
    };
    let Some(left_coverage) = coverages.get(left) else {
        return (None, diagnostic);
    };
    let Some(right_coverage) = coverages.get(right) else {
        return (None, diagnostic);
    };
    diagnostic.failure_stage = "missing_initial_pose".to_string();
    let Some(left_to_world) = initial_homographies.get(&left_tile.id) else {
        return (None, diagnostic);
    };
    let Some(right_to_world) = initial_homographies.get(&right_tile.id) else {
        return (None, diagnostic);
    };
    let Some(world_to_right) = right_to_world.try_inverse() else {
        return (None, diagnostic);
    };
    let left_to_right = world_to_right * left_to_world;
    let Some(right_to_left) = left_to_right.try_inverse() else {
        return (None, diagnostic);
    };
    diagnostic.failure_stage = "no_predicted_overlap".to_string();
    let Some(overlap_bounds) = virtual_tile_predicted_overlap_bounds(
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
    ) else {
        return (None, diagnostic);
    };
    let Some(predicted_overlap_area) = station_relation_overlap_area(
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
        true,
    ) else {
        return (None, diagnostic);
    };
    diagnostic.predicted_overlap_area_px = predicted_overlap_area;

    let left_eligible = left_tile
        .features
        .iter()
        .map(|feature| {
            let point = Point2::new(f64::from(feature.keypoint.x), f64::from(feature.keypoint.y));
            virtual_tile_point_is_covered(left_coverage, point)
                && transformed_point(&left_to_right, point)
                    .is_some_and(|mapped| virtual_tile_point_is_covered(right_coverage, mapped))
        })
        .collect::<Vec<_>>();
    let right_eligible = right_tile
        .features
        .iter()
        .map(|feature| {
            let point = Point2::new(f64::from(feature.keypoint.x), f64::from(feature.keypoint.y));
            virtual_tile_point_is_covered(right_coverage, point)
                && transformed_point(&right_to_left, point)
                    .is_some_and(|mapped| virtual_tile_point_is_covered(left_coverage, mapped))
        })
        .collect::<Vec<_>>();
    let instrument_count = left_eligible
        .iter()
        .filter(|&&eligible| eligible)
        .count()
        .min(right_eligible.iter().filter(|&&eligible| eligible).count());
    diagnostic.grid_probe_count = instrument_count;
    diagnostic.failure_stage = "insufficient_jointly_covered_probes".to_string();
    if instrument_count < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }

    let search_radius = virtual_tile_guided_search_radius(
        left_tile,
        right_tile,
        predicted_overlap_area,
        instrument_count,
    );
    diagnostic.search_radius_px = search_radius;
    let left_buckets =
        virtual_tile_feature_buckets(&left_tile.features, &left_eligible, search_radius);
    let right_buckets =
        virtual_tile_feature_buckets(&right_tile.features, &right_eligible, search_radius);
    let mut by_cell = BTreeMap::<(usize, usize), Vec<(f64, usize, usize, f64)>>::new();
    let mut orientation_observations = Vec::new();
    for (left_index, &eligible) in left_eligible.iter().enumerate() {
        if !eligible {
            continue;
        }
        let Some(forward) = virtual_tile_best_guided_feature(
            left_index,
            &left_tile.features,
            &right_tile.features,
            &right_buckets,
            &left_tile.alignment_image,
            &right_tile.alignment_image,
            left_coverage,
            right_coverage,
            &left_to_right,
            search_radius,
            &mut orientation_observations,
        ) else {
            continue;
        };
        let mut reverse_orientations = Vec::new();
        let reverse = virtual_tile_best_guided_feature(
            forward.target_index,
            &right_tile.features,
            &left_tile.features,
            &left_buckets,
            &right_tile.alignment_image,
            &left_tile.alignment_image,
            right_coverage,
            left_coverage,
            &right_to_left,
            search_radius,
            &mut reverse_orientations,
        );
        orientation_observations.extend(reverse_orientations);
        if reverse.is_none_or(|reverse| reverse.target_index != left_index) {
            continue;
        }
        let source = left_tile.features[left_index].keypoint;
        let cell = virtual_tile_overlap_grid_cell(
            Point2::new(f64::from(source.x), f64::from(source.y)),
            overlap_bounds,
        );
        by_cell.entry(cell).or_default().push((
            forward.rank,
            left_index,
            forward.target_index,
            forward.orientation_difference_degrees,
        ));
    }
    diagnostic.median_prefit_structural_orientation_difference_degrees =
        median_value(&mut orientation_observations);
    diagnostic.bidirectional_match_count = by_cell.values().map(Vec::len).sum();
    diagnostic.occupied_grid_cells = by_cell.len();
    diagnostic.failure_stage = "insufficient_bidirectional_match_count".to_string();
    if diagnostic.bidirectional_match_count < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    diagnostic.failure_stage = "insufficient_grid_coverage".to_string();
    if diagnostic.occupied_grid_cells < VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS {
        return (None, diagnostic);
    }

    for proposals in by_cell.values_mut() {
        proposals.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| {
                    let left_source = left_tile.features[left.1].keypoint;
                    let right_source = left_tile.features[right.1].keypoint;
                    (left_source.y, left_source.x).cmp(&(right_source.y, right_source.x))
                })
                .then_with(|| {
                    let left_target = right_tile.features[left.2].keypoint;
                    let right_target = right_tile.features[right.2].keypoint;
                    (left_target.y, left_target.x).cmp(&(right_target.y, right_target.x))
                })
        });
        proposals.truncate(VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA);
    }
    // Round-robin by quota rank gives RANSAC one observation from every cell
    // before any repeated structure receives its second slot.
    let mut selected = Vec::new();
    for rank in 0..VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA {
        for proposals in by_cell.values() {
            if let Some(proposal) = proposals.get(rank) {
                selected.push(*proposal);
            }
        }
    }
    diagnostic.failure_stage = "insufficient_matches_after_cell_quota".to_string();
    if selected.len() < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    let candidate_points = selected
        .iter()
        .map(|&(_, left_index, right_index, _)| {
            let source = left_tile.features[left_index].keypoint;
            let target = right_tile.features[right_index].keypoint;
            (
                Point2::new(f64::from(source.x), f64::from(source.y)),
                Point2::new(f64::from(target.x), f64::from(target.y)),
            )
        })
        .collect::<Vec<_>>();
    diagnostic.hull_support = station_relation_spatial_support(
        &candidate_points,
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
    );
    diagnostic.failure_stage = "insufficient_prefit_hull_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        return (None, diagnostic);
    }

    // The initial pose ends here. RANSAC and its refit receive only measured
    // fused-pixel correspondences, so the prior can neither become an inlier
    // nor contribute acceptance evidence.
    diagnostic.failure_stage = "ransac_no_fit".to_string();
    let Some((seed_fit, seed_inliers)) =
        processing::find_homography_ransac_points_stable_with_min_inliers(
            &candidate_points,
            STATION_RELATION_MAX_MEDIAN_ERROR_PX,
            STATION_RELATION_MIN_INLIERS,
        )
    else {
        return (None, diagnostic);
    };
    diagnostic.model_fitted = true;
    let seed_points = seed_inliers
        .iter()
        .map(|&index| candidate_points[index])
        .collect::<Vec<_>>();
    let refined = processing::compute_homography(&seed_points).unwrap_or(seed_fit);
    let refined_inliers = symmetric_inlier_indices(
        &refined,
        &candidate_points,
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
    );
    diagnostic.fitted_inliers = refined_inliers.len();
    diagnostic.failure_stage = "insufficient_fitted_inliers".to_string();
    if refined_inliers.len() < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    let points = refined_inliers
        .iter()
        .map(|&index| candidate_points[index])
        .collect::<Vec<_>>();
    diagnostic.hull_support = station_relation_spatial_support(
        &points,
        &refined,
        left_tile.dimensions(),
        right_tile.dimensions(),
    );
    diagnostic.failure_stage = "insufficient_fitted_hull_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        return (None, diagnostic);
    }

    // Orientation becomes an acceptance gate only after fused-pixel
    // correspondences have produced a measured homography. Re-measure every
    // accepted inlier patch with the fitted local Jacobian; the rough prior is
    // deliberately absent from this decision.
    let mut fitted_orientation_observations = refined_inliers
        .iter()
        .filter_map(|&index| {
            let &(_, left_index, right_index, _) = selected.get(index)?;
            virtual_tile_structural_score(
                &left_tile.alignment_image,
                &right_tile.alignment_image,
                left_coverage,
                right_coverage,
                left_tile.features[left_index].keypoint,
                right_tile.features[right_index].keypoint,
                &refined,
            )
            .map(|score| score.orientation_difference_degrees)
        })
        .collect::<Vec<_>>();
    diagnostic.median_fitted_orientation_difference_degrees =
        median_value(&mut fitted_orientation_observations);
    diagnostic.failure_stage = "fitted_orientation_mismatch".to_string();
    if diagnostic
        .median_fitted_orientation_difference_degrees
        .is_some_and(|difference| difference > VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES)
    {
        return (None, diagnostic);
    }

    diagnostic.failure_stage = "fitted".to_string();
    (
        Some(MatchInfo {
            homography: refined,
            inliers: points.len(),
            sequence_bridge: false,
            coarse_bridge: false,
            points,
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            // Key-direction Virtual_Tile fit; see `MatchInfo::canonical_homography`.
            canonical_homography: None,
            native_verified_points: Vec::new(),
        }),
        diagnostic,
    )
}

const VIRTUAL_TILE_DIRECT_PATCH_RADIUS: i32 = 6;
const VIRTUAL_TILE_DIRECT_GRID_SPACING_PX: f64 = 24.0;
const VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES: usize = 1_024;
const VIRTUAL_TILE_DIRECT_MAX_GRID_AXIS: usize = 64;
const VIRTUAL_TILE_DIRECT_MAX_SEARCH_RADIUS_PX: f64 = 16.0;
const VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX: f64 = 1.0;
const VIRTUAL_TILE_DIRECT_PATCH_GATES: registration::PatchMatchGates =
    registration::PatchMatchGates {
        min_correlation: VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC,
        min_peak_margin: VIRTUAL_TILE_GUIDED_MIN_PEAK_MARGIN,
        min_source_variance: 2.0,
        min_target_variance: 1.0,
    };

fn virtual_tile_direct_grid_points(bounds: (f64, f64, f64, f64)) -> Vec<Point2<f64>> {
    let (min_x, min_y, max_x, max_y) = bounds;
    let width = max_x - min_x;
    let height = max_y - min_y;
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        return Vec::new();
    }
    let area = width * height;
    let desired = (area
        / (VIRTUAL_TILE_DIRECT_GRID_SPACING_PX * VIRTUAL_TILE_DIRECT_GRID_SPACING_PX))
        .ceil()
        .clamp(
            STATION_RELATION_MIN_INLIERS as f64,
            VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES as f64,
        ) as usize;
    let aspect = width / height;
    let mut columns = ((desired as f64 * aspect).sqrt().round() as usize)
        .clamp(1, VIRTUAL_TILE_DIRECT_MAX_GRID_AXIS);
    let mut rows = desired
        .div_ceil(columns)
        .clamp(1, VIRTUAL_TILE_DIRECT_MAX_GRID_AXIS);
    while columns * rows > VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES {
        if columns >= rows && columns > 1 {
            columns -= 1;
        } else if rows > 1 {
            rows -= 1;
        } else {
            break;
        }
    }
    let mut points = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        let y = min_y + (row as f64 + 0.5) * height / rows as f64;
        for column in 0..columns {
            let x = min_x + (column as f64 + 0.5) * width / columns as f64;
            points.push(Point2::new(x, y));
        }
    }
    points
}

fn virtual_tile_coverage_contains_rectangle(
    coverage: &stitching::CoverageMask,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
) -> bool {
    let (width, height) = coverage.dimensions();
    if ![min_x, min_y, max_x, max_y]
        .iter()
        .all(|value| value.is_finite())
        || min_x < 0.0
        || min_y < 0.0
        || max_x >= f64::from(width.saturating_sub(1))
        || max_y >= f64::from(height.saturating_sub(1))
        || max_x < min_x
        || max_y < min_y
    {
        return false;
    }
    let start_x = min_x.floor() as u32;
    let end_x = max_x.ceil() as u32;
    let start_y = min_y.floor() as u32;
    let end_y = max_y.ceil() as u32;
    (start_y..=end_y).all(|y| (start_x..=end_x).all(|x| coverage.is_covered(x, y)))
}

fn virtual_tile_direct_patch_support_is_covered(
    source_coverage: &stitching::CoverageMask,
    target_coverage: &stitching::CoverageMask,
    source_to_target: &Matrix3<f64>,
    center: Point2<f64>,
    patch_radius: i32,
    search_radius: i32,
) -> bool {
    let Some(jacobian) = virtual_tile_local_jacobian(source_to_target, center) else {
        return false;
    };
    let scale = (jacobian.column(0).norm() * jacobian.column(1).norm()).sqrt();
    if !(0.15..6.0).contains(&scale) {
        return false;
    }
    let spacing = scale.recip().max(1.0);
    // Include one pixel for gradient sampling and another for bilinear access.
    let source_extent = f64::from(patch_radius) * spacing + 2.0;
    if !virtual_tile_coverage_contains_rectangle(
        source_coverage,
        center.x - source_extent,
        center.y - source_extent,
        center.x + source_extent,
        center.y + source_extent,
    ) {
        return false;
    }
    let corners = [
        Point2::new(center.x - source_extent, center.y - source_extent),
        Point2::new(center.x + source_extent, center.y - source_extent),
        Point2::new(center.x + source_extent, center.y + source_extent),
        Point2::new(center.x - source_extent, center.y + source_extent),
    ];
    let Some(projected) = corners
        .into_iter()
        .map(|point| transformed_point(source_to_target, point))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let padding = f64::from(search_radius) + 2.0;
    let min_x = projected
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min)
        - padding;
    let max_x = projected
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max)
        + padding;
    let min_y = projected
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min)
        - padding;
    let max_y = projected
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max)
        + padding;
    virtual_tile_coverage_contains_rectangle(target_coverage, min_x, min_y, max_x, max_y)
}

fn virtual_tile_round_trip_is_consistent(source: Point2<f64>, returned: Point2<f64>) -> bool {
    let distance = (returned - source).norm();
    distance.is_finite() && distance <= VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX
}

/// Invert the *measured* local forward map, not merely the rough station prior.
///
/// `probe_warped_patch` returns `target = H_initial(source) + residual` in the
/// target tile's coordinates. The corresponding local forward homography is
/// therefore `T(residual) * H_initial`; its inverse is the only transform whose
/// source/target convention maps that measured target back to the original
/// source. Using `H_initial.inverse()` here incorrectly asks the reverse probe
/// to rediscover the already measured residual and made valid pairs depend on
/// which station happened to be the source (任务 10 checkpoint).
fn virtual_tile_measured_reverse_transform(
    initial_source_to_target: &Matrix3<f64>,
    source: Point2<f64>,
    measured_target: Point2<f64>,
) -> Option<Matrix3<f64>> {
    let predicted_target = transformed_point(initial_source_to_target, source)?;
    let residual = measured_target - predicted_target;
    if !residual.iter().all(|value| value.is_finite()) {
        return None;
    }
    let residual_translation =
        Matrix3::new(1.0, 0.0, residual.x, 0.0, 1.0, residual.y, 0.0, 0.0, 1.0);
    (residual_translation * initial_source_to_target).try_inverse()
}

const VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX: f64 = 8.0;
const VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX: f64 = 24.0;
const VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MIN: f64 = 0.98;
const VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MAX: f64 = 1.02;

#[derive(Debug, Clone)]
struct VirtualTileResidualFit {
    measured_homography: Matrix3<f64>,
    inlier_indices: Vec<usize>,
    median_error_px: f64,
    spatial_support: f64,
    model: &'static str,
}

fn residual_distribution(mut values: Vec<f64>) -> stack_report::ResidualDistributionRecord {
    values.retain(|value| value.is_finite());
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return Default::default();
    }
    let at = |quantile: f64| {
        let index = ((values.len() - 1) as f64 * quantile).round() as usize;
        values[index]
    };
    stack_report::ResidualDistributionRecord {
        count: values.len(),
        min: values[0],
        p10: at(0.10),
        median: at(0.50),
        p90: at(0.90),
        max: values[values.len() - 1],
    }
}

fn virtual_tile_measured_residuals(
    points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
) -> Vec<(usize, nalgebra::Vector2<f64>)> {
    points
        .iter()
        .enumerate()
        .filter_map(|(index, (source, target))| {
            let predicted = transformed_point(initial, *source)?;
            let residual = target - predicted;
            residual
                .iter()
                .all(|value| value.is_finite())
                .then_some((index, residual))
        })
        .collect()
}

fn virtual_tile_residual_candidate(
    correction: &Matrix3<f64>,
    initial: &Matrix3<f64>,
    measured_points: &[(Point2<f64>, Point2<f64>)],
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    model: &'static str,
    inlier_threshold_px: f64,
) -> Option<VirtualTileResidualFit> {
    // Both matrices map left -> right, so a correction measured in right-tile
    // coordinates composes on the left: H_measured = H_residual * H_initial.
    let measured_homography = correction * initial;
    let inlier_indices =
        symmetric_inlier_indices(&measured_homography, measured_points, inlier_threshold_px);
    let inlier_points = inlier_indices
        .iter()
        .map(|&index| measured_points[index])
        .collect::<Vec<_>>();
    let median_error_px = median_symmetric_error(&measured_homography, &inlier_points);
    let spatial_support = station_relation_spatial_support(
        &inlier_points,
        &measured_homography,
        left_dimensions,
        right_dimensions,
    );
    Some(VirtualTileResidualFit {
        measured_homography,
        inlier_indices,
        median_error_px,
        spatial_support,
        model,
    })
}

fn virtual_tile_residual_fit_is_better(
    candidate: &VirtualTileResidualFit,
    current: &VirtualTileResidualFit,
) -> bool {
    candidate.inlier_indices.len() > current.inlier_indices.len()
        || (candidate.inlier_indices.len() == current.inlier_indices.len()
            && (candidate.median_error_px < current.median_error_px
                || (candidate.median_error_px == current.median_error_px
                    && candidate
                        .measured_homography
                        .iter()
                        .map(|value| value.to_bits())
                        .cmp(
                            current
                                .measured_homography
                                .iter()
                                .map(|value| value.to_bits()),
                        )
                        .is_lt())))
}

/// Whole left tile as the bounded-correction evidence region.
#[cfg(test)]
fn virtual_tile_full_tile_bounds(left_dimensions: (u32, u32)) -> (f64, f64, f64, f64) {
    (
        0.0,
        0.0,
        f64::from(left_dimensions.0),
        f64::from(left_dimensions.1),
    )
}

/// The correction may move the evidence region's corners by at most
/// `VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX`, differ from its centre's
/// translation by at most `VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX`,
/// and keep its scale within the residual scale range. `evidence_bounds` is
/// `(min_x, min_y, max_x, max_y)` in left-tile pixels.
fn virtual_tile_residual_correction_is_bounded(
    correction: &Matrix3<f64>,
    initial: &Matrix3<f64>,
    evidence_bounds: (f64, f64, f64, f64),
) -> bool {
    let (min_x, min_y, max_x, max_y) = evidence_bounds;
    let source_samples = [
        Point2::new(min_x, min_y),
        Point2::new(max_x, min_y),
        Point2::new(max_x, max_y),
        Point2::new(min_x, max_y),
        Point2::new((min_x + max_x) * 0.5, (min_y + max_y) * 0.5),
    ];
    let Some(displacements) = source_samples
        .into_iter()
        .map(|source| {
            let predicted = transformed_point(initial, source)?;
            let corrected = transformed_point(correction, predicted)?;
            Some(corrected - predicted)
        })
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let center = displacements[4];
    // The correction's scale is its local one where the evidence lies. The
    // upper-left block of a projective correction is not a Jacobian anywhere
    // but at the origin: at 1/8 on langyuan-10 it read 1.035 for a correction
    // whose local scale over its 900px evidence region stayed near 1.0.
    let local_scales = source_samples
        .into_iter()
        .map(|source| {
            let predicted = transformed_point(initial, source)?;
            let jacobian = virtual_tile_local_jacobian(correction, predicted)?;
            Some(jacobian.determinant().abs().sqrt())
        })
        .collect::<Option<Vec<_>>>();
    displacements.iter().all(|displacement| {
        displacement.norm() <= VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX
            && (*displacement - center).norm()
                <= VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX
    }) && local_scales.is_some_and(|scales| {
        scales.iter().all(|scale| {
            (VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MIN..=VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MAX)
                .contains(scale)
        })
    })
}

fn virtual_tile_robust_translation_residual(
    measured_points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    inlier_threshold_px: f64,
) -> Option<VirtualTileResidualFit> {
    let residuals = virtual_tile_measured_residuals(measured_points, initial);
    let mut hypotheses = residuals
        .iter()
        .map(|(_, residual)| *residual)
        .collect::<Vec<_>>();
    hypotheses.sort_by(|left, right| {
        left.x
            .total_cmp(&right.x)
            .then_with(|| left.y.total_cmp(&right.y))
    });
    hypotheses.dedup_by(|left, right| left.x == right.x && left.y == right.y);

    let mut best = None::<VirtualTileResidualFit>;
    for residual in hypotheses {
        let correction = Matrix3::new(1.0, 0.0, residual.x, 0.0, 1.0, residual.y, 0.0, 0.0, 1.0);
        let candidate = virtual_tile_residual_candidate(
            &correction,
            initial,
            measured_points,
            left_dimensions,
            right_dimensions,
            "translation",
            inlier_threshold_px,
        )?;
        if best
            .as_ref()
            .is_none_or(|current| virtual_tile_residual_fit_is_better(&candidate, current))
        {
            best = Some(candidate);
        }
    }

    let best = best?;
    let best_indices = best.inlier_indices.iter().copied().collect::<HashSet<_>>();
    let mut dx = residuals
        .iter()
        .filter(|(index, _)| best_indices.contains(index))
        .map(|(_, residual)| residual.x)
        .collect::<Vec<_>>();
    let mut dy = residuals
        .iter()
        .filter(|(index, _)| best_indices.contains(index))
        .map(|(_, residual)| residual.y)
        .collect::<Vec<_>>();
    dx.sort_by(f64::total_cmp);
    dy.sort_by(f64::total_cmp);
    if dx.is_empty() || dy.is_empty() {
        return Some(best);
    }
    let correction = Matrix3::new(
        1.0,
        0.0,
        dx[dx.len() / 2],
        0.0,
        1.0,
        dy[dy.len() / 2],
        0.0,
        0.0,
        1.0,
    );
    let refined = virtual_tile_residual_candidate(
        &correction,
        initial,
        measured_points,
        left_dimensions,
        right_dimensions,
        "translation",
        inlier_threshold_px,
    )?;
    if virtual_tile_residual_fit_is_better(&refined, &best) {
        Some(refined)
    } else {
        Some(best)
    }
}

#[allow(clippy::too_many_arguments)]
fn virtual_tile_refit_residual_affine(
    seed: Matrix3<f64>,
    residual_points: &[(Point2<f64>, Point2<f64>)],
    measured_points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    inlier_threshold_px: f64,
    evidence_bounds: (f64, f64, f64, f64),
) -> Option<VirtualTileResidualFit> {
    let mut correction = seed;
    let mut fit = virtual_tile_residual_candidate(
        &correction,
        initial,
        measured_points,
        left_dimensions,
        right_dimensions,
        "affine",
        inlier_threshold_px,
    )?;
    for _ in 0..2 {
        let points = fit
            .inlier_indices
            .iter()
            .map(|&index| residual_points[index])
            .collect::<Vec<_>>();
        let Some(refitted) = estimate_affine(&points) else {
            break;
        };
        if !virtual_tile_residual_correction_is_bounded(&refitted, initial, evidence_bounds) {
            break;
        }
        correction = refitted;
        fit = virtual_tile_residual_candidate(
            &correction,
            initial,
            measured_points,
            left_dimensions,
            right_dimensions,
            "affine",
            inlier_threshold_px,
        )?;
    }
    virtual_tile_residual_correction_is_bounded(&correction, initial, evidence_bounds)
        .then_some(fit)
}

/// How the residual model chooses among its passing stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VirtualTileResidualSelection {
    /// Stop at the lowest-DOF stage that passes (single-level rule).
    LowestComplexity,
    /// Evaluate every stage and keep the passing one with the most measured
    /// inliers; on an inlier tie a strictly lower median error wins, otherwise
    /// the lower-DOF stage stays. A coarse-to-fine level must carry
    /// the seed's rotation, scale and perspective error forward: a translation
    /// that explains only the overlap's central band would leave the finer level
    /// a correction it cannot bound.
    MostInliers,
}

#[derive(Debug)]
struct VirtualTileResidualModelOutcome {
    fit: Option<VirtualTileResidualFit>,
    max_translation_consensus: usize,
    translation: stack_report::ResidualModelStageRecord,
    affine: stack_report::ResidualModelStageRecord,
    projective: stack_report::ResidualModelStageRecord,
}

fn residual_model_stage(
    fit: Option<&VirtualTileResidualFit>,
    bounded_correction: Option<bool>,
) -> stack_report::ResidualModelStageRecord {
    let candidate_inliers = fit.map_or(0, |fit| fit.inlier_indices.len());
    let final_support = fit.map_or(0.0, |fit| fit.spatial_support);
    let rejection_reason = if bounded_correction == Some(false) {
        "correction_out_of_bounds"
    } else if fit.is_none() {
        "no_fit"
    } else if candidate_inliers < STATION_RELATION_MIN_INLIERS {
        "insufficient_inliers"
    } else if final_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        "insufficient_support"
    } else {
        "accepted"
    };
    stack_report::ResidualModelStageRecord {
        candidate_inliers,
        final_support,
        bounded_correction,
        rejection_reason: rejection_reason.to_string(),
    }
}

fn residual_model_not_run_stage() -> stack_report::ResidualModelStageRecord {
    stack_report::ResidualModelStageRecord {
        rejection_reason: "not_run".to_string(),
        ..Default::default()
    }
}

/// Residual model fit whose inlier tolerance is the requirement-6.3 world
/// pixel bound itself; the points are native tile pixels and the whole left
/// tile bounds the correction (the single-level rule).
#[cfg(test)]
fn virtual_tile_fit_measured_residual_model(
    measured_points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
) -> VirtualTileResidualModelOutcome {
    virtual_tile_fit_measured_residual_model_with_threshold(
        measured_points,
        initial,
        left_dimensions,
        right_dimensions,
        STATION_RELATION_MAX_MEDIAN_ERROR_PX,
        virtual_tile_full_tile_bounds(left_dimensions),
        VirtualTileResidualSelection::LowestComplexity,
        StationOverlapMeasure::Rectangles,
    )
}

/// Re-measure a stage fit's requirement-6.4 support over `overlap`. The fits
/// themselves rank hypotheses by inliers and error only, so the (costlier)
/// covered overlap is measured once per stage rather than per hypothesis.
fn virtual_tile_fit_with_overlap_support(
    mut fit: VirtualTileResidualFit,
    measured_points: &[(Point2<f64>, Point2<f64>)],
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    overlap: StationOverlapMeasure<'_>,
) -> VirtualTileResidualFit {
    if matches!(overlap, StationOverlapMeasure::Covered { .. }) {
        let inlier_points = fit
            .inlier_indices
            .iter()
            .map(|&index| measured_points[index])
            .collect::<Vec<_>>();
        fit.spatial_support = station_relation_spatial_support_with(
            &inlier_points,
            &fit.measured_homography,
            left_dimensions,
            right_dimensions,
            overlap,
        );
    }
    fit
}

/// Residual model fit in the pixels of one probe level. `inlier_threshold_px`
/// is a symmetric error in those pixels, so the finest pyramid level passes the
/// requirement-6.3 bound divided by its area-averaging factor. The correction
/// bounds are evaluated over `evidence_bounds` (left-tile pixels), the region
/// the probes actually measured; each stage's support over `overlap`.
#[allow(clippy::too_many_arguments)]
fn virtual_tile_fit_measured_residual_model_with_threshold(
    measured_points: &[(Point2<f64>, Point2<f64>)],
    initial: &Matrix3<f64>,
    left_dimensions: (u32, u32),
    right_dimensions: (u32, u32),
    inlier_threshold_px: f64,
    evidence_bounds: (f64, f64, f64, f64),
    selection: VirtualTileResidualSelection,
    overlap: StationOverlapMeasure<'_>,
) -> VirtualTileResidualModelOutcome {
    let with_support = |fit: VirtualTileResidualFit| {
        virtual_tile_fit_with_overlap_support(
            fit,
            measured_points,
            left_dimensions,
            right_dimensions,
            overlap,
        )
    };
    let translation = virtual_tile_robust_translation_residual(
        measured_points,
        initial,
        left_dimensions,
        right_dimensions,
        inlier_threshold_px,
    )
    .map(with_support);
    let max_translation_consensus = translation
        .as_ref()
        .map_or(0, |fit| fit.inlier_indices.len());
    let translation_correction = translation.as_ref().and_then(|fit| {
        initial
            .try_inverse()
            .map(|initial_inverse| fit.measured_homography * initial_inverse)
    });
    let translation_bounded = translation_correction.as_ref().map(|correction| {
        virtual_tile_residual_correction_is_bounded(correction, initial, evidence_bounds)
    });
    let translation_stage = residual_model_stage(translation.as_ref(), translation_bounded);
    if selection == VirtualTileResidualSelection::LowestComplexity
        && translation_stage.rejection_reason == "accepted"
    {
        return VirtualTileResidualModelOutcome {
            fit: translation,
            max_translation_consensus,
            translation: translation_stage,
            affine: residual_model_not_run_stage(),
            projective: residual_model_not_run_stage(),
        };
    }

    // Escalate model complexity only when the lower-DOF correction cannot pass
    // the unchanged inlier/support/bounds gates (or, for a coarse-to-fine
    // level, evaluate every stage). The fit uses predicted target points as
    // coordinates, but every accepted inlier is recomputed from the original
    // measured source/target pair under H_residual * H_initial.
    let mut indexed_residual_points = measured_points
        .iter()
        .enumerate()
        .filter_map(|(index, (source, target))| {
            transformed_point(initial, *source).map(|predicted| (index, predicted, *target))
        })
        .collect::<Vec<_>>();
    indexed_residual_points.sort_by(|left, right| {
        left.1
            .y
            .total_cmp(&right.1.y)
            .then_with(|| left.1.x.total_cmp(&right.1.x))
            .then_with(|| left.2.y.total_cmp(&right.2.y))
            .then_with(|| left.2.x.total_cmp(&right.2.x))
    });
    let sorted_pairs = indexed_residual_points
        .iter()
        .map(|(_, predicted, target)| (*predicted, *target))
        .collect::<Vec<_>>();
    let residual_points = measured_points
        .iter()
        .map(|(source, target)| {
            (
                transformed_point(initial, *source).unwrap_or(*source),
                *target,
            )
        })
        .collect::<Vec<_>>();

    // The single-level rule seeds the affine with the source-stage tolerance;
    // a pyramid level seeds it at its own precision so its spurious peaks
    // cannot pull the seed out of bounds.
    let affine_seed_threshold = match selection {
        VirtualTileResidualSelection::LowestComplexity => FOCUS_MODEL_INLIER_THRESHOLD,
        VirtualTileResidualSelection::MostInliers => inlier_threshold_px,
    };
    let affine_seed = robust_transform_fit_with_threshold(
        &sorted_pairs,
        3,
        0xD1EC_7A11_AFF1_4E01,
        affine_seed_threshold,
        estimate_affine,
    );
    let affine_seed_bounded = affine_seed.as_ref().map(|seed| {
        virtual_tile_residual_correction_is_bounded(&seed.transform, initial, evidence_bounds)
    });
    let affine = affine_seed
        .filter(|_| affine_seed_bounded == Some(true))
        .and_then(|seed| {
            virtual_tile_refit_residual_affine(
                seed.transform,
                &residual_points,
                measured_points,
                initial,
                left_dimensions,
                right_dimensions,
                inlier_threshold_px,
                evidence_bounds,
            )
        })
        .map(with_support);
    let affine_bounded = affine
        .as_ref()
        .map(|fit| {
            initial.try_inverse().is_some_and(|initial_inverse| {
                let correction = fit.measured_homography * initial_inverse;
                virtual_tile_residual_correction_is_bounded(&correction, initial, evidence_bounds)
            })
        })
        .or(affine_seed_bounded);
    let affine_stage = residual_model_stage(affine.as_ref(), affine_bounded);
    if selection == VirtualTileResidualSelection::LowestComplexity
        && affine_stage.rejection_reason == "accepted"
    {
        return VirtualTileResidualModelOutcome {
            fit: affine,
            max_translation_consensus,
            translation: translation_stage,
            affine: affine_stage,
            projective: residual_model_not_run_stage(),
        };
    }

    let projective_seed = processing::find_homography_ransac_points_stable_with_min_inliers(
        &sorted_pairs,
        inlier_threshold_px,
        STATION_RELATION_MIN_INLIERS,
    );
    let projective_correction = projective_seed.as_ref().map(|(seed, seed_inliers)| {
        let seed_points = seed_inliers
            .iter()
            .map(|&index| sorted_pairs[index])
            .collect::<Vec<_>>();
        processing::compute_homography(&seed_points).unwrap_or(*seed)
    });
    let projective_bounded = projective_correction.as_ref().map(|correction| {
        virtual_tile_residual_correction_is_bounded(correction, initial, evidence_bounds)
    });
    let projective = projective_correction
        .filter(|_| projective_bounded == Some(true))
        .and_then(|correction| {
            virtual_tile_residual_candidate(
                &correction,
                initial,
                measured_points,
                left_dimensions,
                right_dimensions,
                "small_projective",
                inlier_threshold_px,
            )
        })
        .map(with_support);
    let projective_stage = residual_model_stage(projective.as_ref(), projective_bounded);
    let fit = match selection {
        VirtualTileResidualSelection::LowestComplexity => (projective_stage.rejection_reason
            == "accepted")
            .then_some(projective)
            .flatten(),
        VirtualTileResidualSelection::MostInliers => {
            // Lowest DOF first: a later stage replaces it with strictly more
            // measured inliers, or with as many and a strictly lower median
            // error. Under a coarse tolerance a translation and an affine can
            // explain the same points; only the lower error shows which one
            // carries the seed's rotation and scale remainder.
            let mut chosen = None::<VirtualTileResidualFit>;
            for (fit, stage) in [
                (translation, &translation_stage),
                (affine, &affine_stage),
                (projective, &projective_stage),
            ] {
                let Some(fit) = fit.filter(|_| stage.rejection_reason == "accepted") else {
                    continue;
                };
                if chosen.as_ref().is_none_or(|current| {
                    fit.inlier_indices.len() > current.inlier_indices.len()
                        || (fit.inlier_indices.len() == current.inlier_indices.len()
                            && fit.median_error_px < current.median_error_px)
                }) {
                    chosen = Some(fit);
                }
            }
            chosen
        }
    };
    VirtualTileResidualModelOutcome {
        fit,
        max_translation_consensus,
        translation: translation_stage,
        affine: affine_stage,
        projective: projective_stage,
    }
}

/// Bounded bidirectional grid probe of one tile pair in the pixels of the
/// supplied planes. `inlier_threshold_px` is the symmetric residual-model
/// tolerance in those pixels (see [`virtual_tile_pyramid_station_match`]).
#[allow(clippy::too_many_arguments)]
fn virtual_tile_direct_station_match(
    left: usize,
    right: usize,
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    inlier_threshold_px: f64,
) -> (
    Option<MatchInfo>,
    stack_report::StationRelationCandidateRecord,
) {
    virtual_tile_direct_station_match_with_structure(
        left,
        right,
        tiles,
        coverages,
        initial_homographies,
        inlier_threshold_px,
        None,
    )
}

/// [`virtual_tile_direct_station_match`] whose structural edge-orientation
/// measurements (prefit diagnostic and fitted gate) read `structure_planes`
/// `(left, right)` instead of the probe planes. The patch and gradient NCC
/// filters always read the probe planes. `None` measures on the probe planes.
#[allow(clippy::too_many_arguments)]
fn virtual_tile_direct_station_match_with_structure(
    left: usize,
    right: usize,
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    inlier_threshold_px: f64,
    structure_planes: Option<(&GrayImage, &GrayImage)>,
) -> (
    Option<MatchInfo>,
    stack_report::StationRelationCandidateRecord,
) {
    let mut diagnostic = stack_report::StationRelationCandidateRecord {
        left,
        right,
        left_tile: Default::default(),
        right_tile: Default::default(),
        predicted_overlap_area_px: 0.0,
        model_fitted: false,
        evidence_kind: stack_report::StationRelationEvidenceKind::VirtualTile,
        grid_probe_count: 0,
        measurable_patch_count: 0,
        bidirectional_match_count: 0,
        occupied_grid_cells: 0,
        search_radius_px: 0.0,
        residual_dx_px: Default::default(),
        residual_dy_px: Default::default(),
        residual_magnitude_px: Default::default(),
        max_residual_translation_consensus: 0,
        residual_translation: Default::default(),
        residual_affine: Default::default(),
        residual_projective: Default::default(),
        residual_model: "not_run".to_string(),
        fitted_inliers: 0,
        hull_support: 0.0,
        median_prefit_structural_orientation_difference_degrees: None,
        median_fitted_orientation_difference_degrees: None,
        failure_stage: "missing_input".to_string(),
        pyramid_factors: Vec::new(),
        measurement_factor: 1,
        seed: VIRTUAL_TILE_SEED_RENDERING_PRIOR.to_string(),
        prior_free_inliers: 0,
        global_proposals: Vec::new(),
        after_prior_repair: false,
        hypotheses: Vec::new(),
    };
    // Keep the full-resolution Virtual_Tile station probe available for normal
    // builds, while allowing test fixtures to isolate the descriptor/coarse
    // geometry path. The helper is hard-wired to `false` outside `cfg(test)`,
    // so a production environment variable cannot weaken station evidence.
    if registration::native_patch_refinement_disabled_for_tests() {
        diagnostic.failure_stage = "native_patch_refinement_disabled".to_string();
        return (None, diagnostic);
    }
    let (Some(left_tile), Some(right_tile), Some(left_coverage), Some(right_coverage)) = (
        tiles.get(left),
        tiles.get(right),
        coverages.get(left),
        coverages.get(right),
    ) else {
        return (None, diagnostic);
    };
    diagnostic.failure_stage = "missing_initial_pose".to_string();
    let (Some(left_to_world), Some(right_to_world)) = (
        initial_homographies.get(&left_tile.id),
        initial_homographies.get(&right_tile.id),
    ) else {
        return (None, diagnostic);
    };
    let Some(world_to_right) = right_to_world.try_inverse() else {
        return (None, diagnostic);
    };
    let left_to_right = world_to_right * left_to_world;
    if left_to_right.try_inverse().is_none() {
        return (None, diagnostic);
    }
    let (left_structure, right_structure) =
        structure_planes.unwrap_or((&left_tile.alignment_image, &right_tile.alignment_image));
    diagnostic.failure_stage = "no_predicted_overlap".to_string();
    let Some(overlap_bounds) = virtual_tile_predicted_overlap_bounds(
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
    ) else {
        return (None, diagnostic);
    };
    let Some(predicted_overlap_area) = station_relation_overlap_area(
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
        true,
    ) else {
        return (None, diagnostic);
    };
    diagnostic.predicted_overlap_area_px = predicted_overlap_area;
    let grid = virtual_tile_direct_grid_points(overlap_bounds);
    diagnostic.grid_probe_count = grid.len();
    diagnostic.failure_stage = "insufficient_grid_probes".to_string();
    if grid.len() < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    let search_radius = virtual_tile_guided_search_radius(
        left_tile,
        right_tile,
        predicted_overlap_area,
        grid.len(),
    )
    .min(VIRTUAL_TILE_DIRECT_MAX_SEARCH_RADIUS_PX)
    .ceil() as i32;
    diagnostic.search_radius_px = f64::from(search_radius);

    let mut by_cell = BTreeMap::<(usize, usize), Vec<(f64, Point2<f64>, Point2<f64>, f64)>>::new();
    let mut prefit_orientations = Vec::new();
    for source in grid {
        if !virtual_tile_direct_patch_support_is_covered(
            left_coverage,
            right_coverage,
            &left_to_right,
            source,
            VIRTUAL_TILE_DIRECT_PATCH_RADIUS,
            search_radius,
        ) {
            continue;
        }
        let forward_probe = registration::probe_warped_patch(
            &left_tile.alignment_image,
            &right_tile.alignment_image,
            &left_to_right,
            source,
            VIRTUAL_TILE_DIRECT_PATCH_RADIUS,
            search_radius,
            VIRTUAL_TILE_DIRECT_PATCH_GATES,
        );
        if forward_probe.best_correlation.is_some() {
            diagnostic.measurable_patch_count += 1;
        }
        let Some(forward) = forward_probe.matched else {
            continue;
        };
        let Some(structure) = virtual_tile_structural_score_points(
            &left_tile.alignment_image,
            &right_tile.alignment_image,
            left_coverage,
            right_coverage,
            source,
            forward.target,
            &left_to_right,
        ) else {
            continue;
        };
        let prefit_orientation = if structure_planes.is_some() {
            virtual_tile_structural_score_points(
                left_structure,
                right_structure,
                left_coverage,
                right_coverage,
                source,
                forward.target,
                &left_to_right,
            )
            .map(|score| score.orientation_difference_degrees)
        } else {
            Some(structure.orientation_difference_degrees)
        };
        if let Some(orientation) = prefit_orientation {
            prefit_orientations.push(orientation);
        }
        if structure.patch_ncc < VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC
            || structure.gradient_ncc < VIRTUAL_TILE_GUIDED_MIN_GRADIENT_NCC
        {
            continue;
        }
        let Some(measured_right_to_left) =
            virtual_tile_measured_reverse_transform(&left_to_right, source, forward.target)
        else {
            continue;
        };
        if !virtual_tile_direct_patch_support_is_covered(
            right_coverage,
            left_coverage,
            &measured_right_to_left,
            forward.target,
            VIRTUAL_TILE_DIRECT_PATCH_RADIUS,
            search_radius,
        ) {
            continue;
        }
        let reverse_probe = registration::probe_warped_patch(
            &right_tile.alignment_image,
            &left_tile.alignment_image,
            &measured_right_to_left,
            forward.target,
            VIRTUAL_TILE_DIRECT_PATCH_RADIUS,
            search_radius,
            VIRTUAL_TILE_DIRECT_PATCH_GATES,
        );
        let Some(reverse) = reverse_probe.matched else {
            continue;
        };
        if !virtual_tile_round_trip_is_consistent(source, reverse.target) {
            continue;
        }
        diagnostic.bidirectional_match_count += 1;
        let cell = virtual_tile_overlap_grid_cell(source, overlap_bounds);
        let rank = structure.patch_ncc + 0.5 * structure.gradient_ncc
            - 0.05 * (reverse.target - source).norm();
        by_cell.entry(cell).or_default().push((
            rank,
            source,
            forward.target,
            structure.orientation_difference_degrees,
        ));
    }
    diagnostic.median_prefit_structural_orientation_difference_degrees =
        median_value(&mut prefit_orientations);
    diagnostic.occupied_grid_cells = by_cell.len();
    diagnostic.failure_stage = "insufficient_measurable_patches".to_string();
    if diagnostic.measurable_patch_count < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    diagnostic.failure_stage = "insufficient_bidirectional_matches".to_string();
    if diagnostic.bidirectional_match_count < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    diagnostic.failure_stage = "insufficient_grid_coverage".to_string();
    if diagnostic.occupied_grid_cells < VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS {
        return (None, diagnostic);
    }

    // Rank locally, but do not discard candidates before finding a globally
    // coherent residual model. Repeated artwork can give every cell a strong
    // yet different local peak; truncating each cell first made those modes
    // indistinguishable to RANSAC. The strict measured model selects one mode,
    // then the per-cell quota only limits its already verified inliers.
    for proposals in by_cell.values_mut() {
        proposals.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.y.total_cmp(&right.1.y))
                .then_with(|| left.1.x.total_cmp(&right.1.x))
                .then_with(|| left.2.y.total_cmp(&right.2.y))
                .then_with(|| left.2.x.total_cmp(&right.2.x))
        });
    }
    let maximum_cell_candidates = by_cell.values().map(Vec::len).max().unwrap_or(0);
    let mut all_selected = Vec::with_capacity(diagnostic.bidirectional_match_count);
    for rank in 0..maximum_cell_candidates {
        for (&cell, proposals) in &by_cell {
            if let Some(proposal) = proposals.get(rank) {
                all_selected.push((cell, *proposal));
            }
        }
    }
    let candidate_points = all_selected
        .iter()
        .map(|(_, (_, source, target, _))| (*source, *target))
        .collect::<Vec<_>>();
    // Requirement 6.4 on the pixels both tiles cover, in this plane's pixels.
    let overlap_measure = StationOverlapMeasure::Covered {
        source: left_coverage,
        target: right_coverage,
        factor: 1,
    };
    diagnostic.hull_support = station_relation_spatial_support_with(
        &candidate_points,
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
        overlap_measure,
    );
    diagnostic.failure_stage = "insufficient_prefit_hull_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        return (None, diagnostic);
    }

    let measured_residuals = virtual_tile_measured_residuals(&candidate_points, &left_to_right);
    diagnostic.residual_dx_px = residual_distribution(
        measured_residuals
            .iter()
            .map(|(_, residual)| residual.x)
            .collect(),
    );
    diagnostic.residual_dy_px = residual_distribution(
        measured_residuals
            .iter()
            .map(|(_, residual)| residual.y)
            .collect(),
    );
    diagnostic.residual_magnitude_px = residual_distribution(
        measured_residuals
            .iter()
            .map(|(_, residual)| residual.norm())
            .collect(),
    );

    diagnostic.failure_stage = "residual_model_no_fit".to_string();
    // Bound the correction where the probes measured it: the extent of the
    // bidirectional matches inside the predicted overlap. The far corners of a
    // tile that overlaps along one strip would turn any rotation or perspective
    // remainder of the seed into an apparent large correction there.
    let evidence_bounds = candidate_points.iter().fold(
        (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ),
        |(min_x, min_y, max_x, max_y), (source, _)| {
            (
                min_x.min(source.x),
                min_y.min(source.y),
                max_x.max(source.x),
                max_y.max(source.y),
            )
        },
    );
    let model = virtual_tile_fit_measured_residual_model_with_threshold(
        &candidate_points,
        &left_to_right,
        left_tile.dimensions(),
        right_tile.dimensions(),
        inlier_threshold_px,
        evidence_bounds,
        VirtualTileResidualSelection::MostInliers,
        overlap_measure,
    );
    diagnostic.max_residual_translation_consensus = model.max_translation_consensus;
    diagnostic.residual_translation = model.translation.clone();
    diagnostic.residual_affine = model.affine.clone();
    diagnostic.residual_projective = model.projective.clone();
    let Some(fit) = model.fit else {
        return (None, diagnostic);
    };
    diagnostic.model_fitted = true;
    diagnostic.residual_model = fit.model.to_string();
    let refined = fit.measured_homography;

    let fitted_indices = fit.inlier_indices.into_iter().collect::<HashSet<_>>();
    let mut fitted_by_cell =
        BTreeMap::<(usize, usize), Vec<(f64, Point2<f64>, Point2<f64>, f64)>>::new();
    let mut fitted_proposals = Vec::with_capacity(fitted_indices.len());
    for (index, (cell, proposal)) in all_selected.into_iter().enumerate() {
        if fitted_indices.contains(&index) {
            fitted_by_cell.entry(cell).or_default().push(proposal);
            fitted_proposals.push(proposal);
        }
    }
    for proposals in fitted_by_cell.values_mut() {
        proposals.truncate(VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA);
    }
    let mut selected = Vec::new();
    for rank in 0..VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA {
        for proposals in fitted_by_cell.values() {
            if let Some(proposal) = proposals.get(rank) {
                selected.push(*proposal);
            }
        }
    }
    // Requirement 6.4 measures the convex hull of the *inliers*. The per-cell
    // quota only limits how strongly a textured cell weighs in the pose graph,
    // so it must not shrink that hull: keep the fitted inliers that are hull
    // vertices on either station plane as well.
    let hull_vertex = |vertices: &[Point2<f64>], point: Point2<f64>| {
        vertices
            .iter()
            .any(|vertex| vertex.x == point.x && vertex.y == point.y)
    };
    let source_hull =
        convex_hull_vertices(fitted_proposals.iter().map(|&(_, source, _, _)| source));
    let target_hull =
        convex_hull_vertices(fitted_proposals.iter().map(|&(_, _, target, _)| target));
    for &proposal in &fitted_proposals {
        let (_, source, target, _) = proposal;
        let already = selected
            .iter()
            .any(|&(_, selected_source, selected_target, _)| {
                selected_source == source && selected_target == target
            });
        if !already && (hull_vertex(&source_hull, source) || hull_vertex(&target_hull, target)) {
            selected.push(proposal);
        }
    }
    diagnostic.failure_stage = "insufficient_matches_after_model_quota".to_string();
    if selected.len() < STATION_RELATION_MIN_INLIERS {
        return (None, diagnostic);
    }
    let points = selected
        .iter()
        .map(|&(_, source, target, _)| (source, target))
        .collect::<Vec<_>>();
    diagnostic.fitted_inliers = points.len();
    diagnostic.hull_support = station_relation_spatial_support_with(
        &points,
        &refined,
        left_tile.dimensions(),
        right_tile.dimensions(),
        overlap_measure,
    );
    diagnostic.failure_stage = "insufficient_fitted_hull_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT {
        return (None, diagnostic);
    }

    let mut fitted_orientations = selected
        .iter()
        .filter_map(|&(_, source, target, _)| {
            virtual_tile_structural_score_points(
                left_structure,
                right_structure,
                left_coverage,
                right_coverage,
                source,
                target,
                &refined,
            )
            .map(|score| score.orientation_difference_degrees)
        })
        .collect::<Vec<_>>();
    diagnostic.median_fitted_orientation_difference_degrees =
        median_value(&mut fitted_orientations);
    diagnostic.failure_stage = "fitted_orientation_mismatch".to_string();
    if diagnostic
        .median_fitted_orientation_difference_degrees
        .is_some_and(|difference| difference > VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES)
    {
        return (None, diagnostic);
    }

    diagnostic.failure_stage = "fitted".to_string();
    (
        Some(MatchInfo {
            homography: refined,
            inliers: points.len(),
            sequence_bridge: false,
            coarse_bridge: false,
            points,
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            // Key-direction Virtual_Tile fit; see `MatchInfo::canonical_homography`.
            canonical_homography: None,
            native_verified_points: Vec::new(),
        }),
        diagnostic,
    )
}

// ---------------------------------------------------------------------------
// Coarse-to-fine Virtual_Tile station matching (任务 10 checkpoint)
//
// `langyuan-10` measured two independent failures of a native-pixel probe
// centred on the rendering prior (the source-level world poses the tiles are
// rendered with):
//
// * the fine texture of two focus-fused stations is not comparable. Seeded by
//   an independent SIFT relation (median error ~1.6px), 13px and 33px native
//   windows still gave only 58–92 of ~900 bidirectional matches, with a 32–35°
//   structural orientation difference (spurious peaks). The same 13-sample
//   window on 4×4 area averages gave 408–605 with 13–22°, and 0→2 was
//   accepted from the rough prior alone at 1/8 and 1/4;
// * the prior itself can be grossly wrong: a false source-level coarse bridge
//   put one station ~3,400px away at the wrong scale, where no bounded search
//   can reach it. Covered-pixel FAST/BRIEF at analysis resolution found its
//   true neighbour (253–260 RANSAC inliers) without any prior.
//
// The authoritative relation is therefore always a bounded probe around the
// rendering prior, run coarse to fine on area-averaged planes. A prior-free
// feature relation normally proposes a repair of a whole cluster's rendering
// prior, after which the repaired stations are rendered once more and measured
// again around the new prior. It may enter the Station_Relation graph directly
// only when its independently polished hypothesis passes the complete shared
// covered-overlap gate.
// ---------------------------------------------------------------------------

/// Stable seed identifiers of a Virtual_Tile relation measurement.
const VIRTUAL_TILE_SEED_RENDERING_PRIOR: &str = "rendering_prior";
const VIRTUAL_TILE_SEED_PRIOR_FREE: &str = "prior_free_features";
/// Smallest native-pixel extent one direct-probe window must span before two
/// focus-fused stations present comparable structure (measured above; the
/// intra-station registrar needed 51px windows for the same reason, 任务 7.5).
const VIRTUAL_TILE_MIN_MATCH_WINDOW_NATIVE_PX: f64 = 48.0;
/// Smallest long side on which a pyramid level still carries a probe grid.
const VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE: u32 = 384;
/// Symmetric residual-model tolerance of the octaves coarser than the finest,
/// in their own pixels.
const VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX: f64 = 1.5;
/// Covered-feature budget and support scales of the prior-free proposal,
/// identical to the scalable source analysis path.
const VIRTUAL_TILE_PRIOR_FREE_MAX_FEATURES: usize = 1_600;
const VIRTUAL_TILE_PRIOR_FREE_SCALES: [f32; 3] = [0.38, 0.5, std::f32::consts::FRAC_1_SQRT_2];

/// One area-averaged level of a Virtual_Tile's covered luminance plane.
///
/// A level pixel is covered only when every native pixel of its block is, so
/// uncovered payload never reaches a probe (需求 6.1). Level pixel `p` is the
/// mean of the native block `[f·p, f·p + f)`, whose centre is
/// `f·p + (f − 1) / 2`; [`Self::level_to_native`] is exactly that map.
#[derive(Clone)]
struct VirtualTileMatchPlane {
    factor: u32,
    gray: GrayImage,
    coverage: stitching::CoverageMask,
}

impl VirtualTileMatchPlane {
    fn from_native(
        gray: &GrayImage,
        coverage: &stitching::CoverageMask,
        factor: u32,
    ) -> Option<Self> {
        let factor = factor.max(1);
        let (width, height) = gray.dimensions();
        if coverage.dimensions() != (width, height) {
            return None;
        }
        if factor == 1 {
            return Some(Self {
                factor,
                gray: gray.clone(),
                coverage: coverage.clone(),
            });
        }
        let level_width = width / factor;
        let level_height = height / factor;
        if level_width == 0 || level_height == 0 {
            return None;
        }
        let area = f64::from(factor * factor);
        let row_len = level_width as usize;
        let mut values = vec![0u8; row_len * level_height as usize];
        let mut covered = vec![0u8; row_len * level_height as usize];
        // Every output element is written by exactly one task and no value is
        // reduced across tasks, so the result is independent of scheduling.
        values
            .par_chunks_mut(row_len)
            .zip(covered.par_chunks_mut(row_len))
            .enumerate()
            .for_each(|(y, (value_row, covered_row))| {
                let y = y as u32;
                for x in 0..level_width {
                    let mut sum = 0u32;
                    let mut all_covered = true;
                    'block: for dy in 0..factor {
                        for dx in 0..factor {
                            let (sample_x, sample_y) = (x * factor + dx, y * factor + dy);
                            if !coverage.is_covered(sample_x, sample_y) {
                                all_covered = false;
                                break 'block;
                            }
                            sum += u32::from(gray.get_pixel(sample_x, sample_y)[0]);
                        }
                    }
                    if all_covered {
                        value_row[x as usize] = (f64::from(sum) / area).round() as u8;
                        covered_row[x as usize] = u8::MAX;
                    }
                }
            });
        Some(Self {
            factor,
            gray: GrayImage::from_raw(level_width, level_height, values)?,
            coverage: stitching::CoverageMask::from_bytes(level_width, level_height, covered)
                .ok()?,
        })
    }

    fn level_to_native(factor: u32) -> Matrix3<f64> {
        let scale = f64::from(factor.max(1));
        let offset = (scale - 1.0) * 0.5;
        Matrix3::new(scale, 0.0, offset, 0.0, scale, offset, 0.0, 0.0, 1.0)
    }

    fn native_to_level(factor: u32) -> Matrix3<f64> {
        let scale = f64::from(factor.max(1));
        let offset = (scale - 1.0) * 0.5;
        Matrix3::new(
            1.0 / scale,
            0.0,
            -offset / scale,
            0.0,
            1.0 / scale,
            -offset / scale,
            0.0,
            0.0,
            1.0,
        )
    }
}

/// Partially covered ultra-coarse luminance plane used only to propose a
/// similarity basin. Unlike `VirtualTileMatchPlane`, a cell averages every
/// covered native sample in its block and never reads an uncovered payload.
#[derive(Clone)]
struct VirtualTileGlobalSearchPlane {
    factor: u32,
    width: u32,
    height: u32,
    luminance: Vec<f64>,
    gradient: Vec<f64>,
    covered_count: Vec<u32>,
}

impl VirtualTileGlobalSearchPlane {
    fn from_native(gray: &GrayImage, coverage: &stitching::CoverageMask) -> Option<Self> {
        Self::from_native_target(gray, coverage, 160)
    }

    fn from_native_target(
        gray: &GrayImage,
        coverage: &stitching::CoverageMask,
        target_long_side: u32,
    ) -> Option<Self> {
        let (native_width, native_height) = gray.dimensions();
        if coverage.dimensions() != (native_width, native_height)
            || native_width == 0
            || native_height == 0
            || target_long_side == 0
        {
            return None;
        }
        let factor = native_width
            .max(native_height)
            .div_ceil(target_long_side)
            .max(1);
        let width = native_width.div_ceil(factor);
        let height = native_height.div_ceil(factor);
        let mut luminance = vec![0.0; width as usize * height as usize];
        let mut covered_count = vec![0u32; luminance.len()];
        for level_y in 0..height {
            for level_x in 0..width {
                let mut sum = 0u64;
                let mut count = 0u32;
                let max_y = ((level_y + 1) * factor).min(native_height);
                let max_x = ((level_x + 1) * factor).min(native_width);
                for y in level_y * factor..max_y {
                    for x in level_x * factor..max_x {
                        if coverage.is_covered(x, y) {
                            // Coverage is checked before the pixel is read: an
                            // uncovered payload cannot affect any proposal bit.
                            sum += u64::from(gray.get_pixel(x, y)[0]);
                            count += 1;
                        }
                    }
                }
                let index = (level_y * width + level_x) as usize;
                covered_count[index] = count;
                if count > 0 {
                    luminance[index] = sum as f64 / f64::from(count);
                }
            }
        }
        let mut gradient = vec![0.0; luminance.len()];
        if width >= 3 && height >= 3 {
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    let at = |x: u32, y: u32| (y * width + x) as usize;
                    let support = [
                        at(x, y),
                        at(x - 1, y),
                        at(x + 1, y),
                        at(x, y - 1),
                        at(x, y + 1),
                    ];
                    if support.iter().all(|&index| covered_count[index] > 0) {
                        let gx = 0.5 * (luminance[at(x + 1, y)] - luminance[at(x - 1, y)]);
                        let gy = 0.5 * (luminance[at(x, y + 1)] - luminance[at(x, y - 1)]);
                        gradient[at(x, y)] = gx.hypot(gy);
                    }
                }
            }
        }
        Some(Self {
            factor,
            width,
            height,
            luminance,
            gradient,
            covered_count,
        })
    }

    fn index(&self, x: i32, y: i32) -> Option<usize> {
        (x >= 0 && y >= 0 && x < self.width as i32 && y < self.height as i32)
            .then(|| y as usize * self.width as usize + x as usize)
    }

    fn native_to_level(&self) -> Matrix3<f64> {
        VirtualTileMatchPlane::native_to_level(self.factor)
    }

    fn level_to_native(&self) -> Matrix3<f64> {
        VirtualTileMatchPlane::level_to_native(self.factor)
    }
}

const VIRTUAL_TILE_SEED_SOURCE_MATCH: &str = "source_pairwise_match";
const VIRTUAL_TILE_SEED_GLOBAL_COVERAGE: &str = "global_coverage_similarity";
const VIRTUAL_TILE_SOURCE_MATCH_MAX_SEEDS: usize = 4;
const VIRTUAL_TILE_GLOBAL_TOP_K: usize = 8;
const VIRTUAL_TILE_GLOBAL_MAX_SAMPLES: usize = 512;
const VIRTUAL_TILE_GLOBAL_MIN_OVERLAP_SAMPLES: usize = 32;

#[derive(Clone, Copy, Debug)]
struct VirtualTileSourceMatchProposal {
    left_to_right: Matrix3<f64>,
    inliers: usize,
    left_source: usize,
    right_source: usize,
}

#[derive(Clone, Copy, Debug)]
struct VirtualTileGlobalProposal {
    left_to_right: Matrix3<f64>,
    score: f64,
    overlap_samples: usize,
    left_factor: u32,
    right_factor: u32,
    scale: f64,
    rotation_degrees: f64,
}

#[derive(Clone, Copy, Debug)]
struct VirtualTileGlobalPeak {
    transform: Matrix3<f64>,
    score: f64,
    overlap_samples: usize,
    translation: (f64, f64),
    scale: f64,
    rotation_degrees: f64,
}

fn virtual_tile_translation(tx: f64, ty: f64) -> Matrix3<f64> {
    Matrix3::new(1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0)
}

/// Sparse masked zero-mean luminance/gradient NCC. This score only ranks
/// proposal basins; it is never consulted by a Station_Relation gate.
fn virtual_tile_masked_similarity_score(
    left: &VirtualTileGlobalSearchPlane,
    right: &VirtualTileGlobalSearchPlane,
    transform: &Matrix3<f64>,
    samples: &[(u32, u32)],
) -> Option<(f64, usize)> {
    let mut left_values = Vec::new();
    let mut right_values = Vec::new();
    let mut left_gradients = Vec::new();
    let mut right_gradients = Vec::new();
    for &(x, y) in samples {
        let source_index = (y * left.width + x) as usize;
        let mapped = transformed_point(transform, Point2::new(f64::from(x), f64::from(y)))?;
        let (target_x, target_y) = (mapped.x.round() as i32, mapped.y.round() as i32);
        let Some(target_index) = right.index(target_x, target_y) else {
            continue;
        };
        if right.covered_count[target_index] == 0 {
            continue;
        }
        left_values.push(left.luminance[source_index]);
        right_values.push(right.luminance[target_index]);
        if left.gradient[source_index] > 0.0 && right.gradient[target_index] > 0.0 {
            left_gradients.push(left.gradient[source_index]);
            right_gradients.push(right.gradient[target_index]);
        }
    }
    let minimum_overlap = VIRTUAL_TILE_GLOBAL_MIN_OVERLAP_SAMPLES.max(samples.len().div_ceil(5));
    if left_values.len() < minimum_overlap {
        return None;
    }
    let luminance = virtual_tile_ncc(&left_values, &right_values)?;
    let score = virtual_tile_ncc(&left_gradients, &right_gradients)
        .map_or(luminance, |gradient| 0.7 * luminance + 0.3 * gradient);
    score.is_finite().then_some((score, left_values.len()))
}

fn virtual_tile_global_peak_order(
    left: &VirtualTileGlobalPeak,
    right: &VirtualTileGlobalPeak,
) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| right.overlap_samples.cmp(&left.overlap_samples))
        .then_with(|| left.scale.total_cmp(&right.scale))
        .then_with(|| left.rotation_degrees.total_cmp(&right.rotation_degrees))
        .then_with(|| left.translation.1.total_cmp(&right.translation.1))
        .then_with(|| left.translation.0.total_cmp(&right.translation.0))
}

fn virtual_tile_global_select_peaks(
    mut candidates: Vec<VirtualTileGlobalPeak>,
    limit: usize,
    translation_separation: f64,
) -> Vec<VirtualTileGlobalPeak> {
    candidates.sort_by(virtual_tile_global_peak_order);
    let mut selected = Vec::<VirtualTileGlobalPeak>::new();
    for candidate in candidates {
        if selected.iter().any(|existing| {
            (candidate.translation.0 - existing.translation.0)
                .hypot(candidate.translation.1 - existing.translation.1)
                < translation_separation
                && (candidate.scale - existing.scale).abs() < 0.0125
                && (candidate.rotation_degrees - existing.rotation_degrees).abs() < 2.5
        }) {
            continue;
        }
        selected.push(candidate);
        if selected.len() == limit {
            break;
        }
    }
    selected
}

/// Coverage-aware, full-translation, bounded-similarity, multi-peak search.
/// Returned matrices are native `left -> right` seeds for the existing dense
/// pyramid path; none is relation evidence by itself.
fn virtual_tile_global_similarity_proposals(
    left: &VirtualTileGlobalSearchPlane,
    right: &VirtualTileGlobalSearchPlane,
) -> Vec<VirtualTileGlobalProposal> {
    let covered = (0..left.height)
        .flat_map(|y| (0..left.width).map(move |x| (x, y)))
        .filter(|&(x, y)| left.covered_count[(y * left.width + x) as usize] > 0)
        .collect::<Vec<_>>();
    if covered.len() < VIRTUAL_TILE_GLOBAL_MIN_OVERLAP_SAMPLES {
        return Vec::new();
    }
    let sample_stride = covered
        .len()
        .div_ceil(VIRTUAL_TILE_GLOBAL_MAX_SAMPLES)
        .max(1);
    let samples = covered
        .into_iter()
        .step_by(sample_stride)
        .take(VIRTUAL_TILE_GLOBAL_MAX_SAMPLES)
        .collect::<Vec<_>>();
    let mut coarse = Vec::new();
    for scale in [0.95f64, 1.0, 1.05] {
        for rotation_degrees in [-10.0f64, 0.0, 10.0] {
            let theta = rotation_degrees.to_radians();
            let (sin, cos) = theta.sin_cos();
            let level_scale = scale * f64::from(left.factor) / f64::from(right.factor);
            let linear = Matrix3::new(
                level_scale * cos,
                -level_scale * sin,
                0.0,
                level_scale * sin,
                level_scale * cos,
                0.0,
                0.0,
                0.0,
                1.0,
            );
            let corners = [
                Point2::new(0.0, 0.0),
                Point2::new(f64::from(left.width.saturating_sub(1)), 0.0),
                Point2::new(0.0, f64::from(left.height.saturating_sub(1))),
                Point2::new(
                    f64::from(left.width.saturating_sub(1)),
                    f64::from(left.height.saturating_sub(1)),
                ),
            ]
            .into_iter()
            .filter_map(|point| transformed_point(&linear, point))
            .collect::<Vec<_>>();
            let min_x = corners
                .iter()
                .map(|point| point.x)
                .fold(f64::INFINITY, f64::min);
            let max_x = corners
                .iter()
                .map(|point| point.x)
                .fold(f64::NEG_INFINITY, f64::max);
            let min_y = corners
                .iter()
                .map(|point| point.y)
                .fold(f64::INFINITY, f64::min);
            let max_y = corners
                .iter()
                .map(|point| point.y)
                .fold(f64::NEG_INFINITY, f64::max);
            let tx_min = (-max_x).floor() as i32;
            let tx_max = (f64::from(right.width.saturating_sub(1)) - min_x).ceil() as i32;
            let ty_min = (-max_y).floor() as i32;
            let ty_max = (f64::from(right.height.saturating_sub(1)) - min_y).ceil() as i32;
            let translation_step = if scale == 1.0 && rotation_degrees == 0.0 {
                1usize
            } else {
                4usize
            };
            for ty in (ty_min..=ty_max).step_by(translation_step) {
                for tx in (tx_min..=tx_max).step_by(translation_step) {
                    let transform = virtual_tile_translation(f64::from(tx), f64::from(ty)) * linear;
                    if let Some((score, overlap_samples)) =
                        virtual_tile_masked_similarity_score(left, right, &transform, &samples)
                    {
                        coarse.push(VirtualTileGlobalPeak {
                            transform,
                            score,
                            overlap_samples,
                            translation: (f64::from(tx), f64::from(ty)),
                            scale,
                            rotation_degrees,
                        });
                    }
                }
            }
        }
    }
    let coarse = virtual_tile_global_select_peaks(coarse, 32, 10.0);
    let mut refined = Vec::new();
    for peak in coarse {
        for scale in [peak.scale - 0.0125, peak.scale, peak.scale + 0.0125]
            .into_iter()
            .map(|scale| scale.clamp(0.95, 1.05))
        {
            for rotation_degrees in [
                peak.rotation_degrees - 2.5,
                peak.rotation_degrees,
                peak.rotation_degrees + 2.5,
            ]
            .into_iter()
            .map(|rotation| rotation.clamp(-10.0, 10.0))
            {
                let theta = rotation_degrees.to_radians();
                let (sin, cos) = theta.sin_cos();
                let level_scale = scale * f64::from(left.factor) / f64::from(right.factor);
                let linear = Matrix3::new(
                    level_scale * cos,
                    -level_scale * sin,
                    0.0,
                    level_scale * sin,
                    level_scale * cos,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                );
                for dy in -3..=3 {
                    for dx in -3..=3 {
                        let tx = peak.translation.0 + f64::from(dx);
                        let ty = peak.translation.1 + f64::from(dy);
                        let transform = virtual_tile_translation(tx, ty) * linear;
                        if let Some((score, overlap_samples)) =
                            virtual_tile_masked_similarity_score(left, right, &transform, &samples)
                        {
                            refined.push(VirtualTileGlobalPeak {
                                transform,
                                score,
                                overlap_samples,
                                translation: (tx, ty),
                                scale,
                                rotation_degrees,
                            });
                        }
                    }
                }
            }
        }
    }
    virtual_tile_global_select_peaks(refined, VIRTUAL_TILE_GLOBAL_TOP_K, 12.0)
        .into_iter()
        .filter_map(|peak| {
            let native = normalized_homography(
                &(right.level_to_native() * peak.transform * left.native_to_level()),
            )?;
            Some(VirtualTileGlobalProposal {
                left_to_right: native,
                score: peak.score,
                overlap_samples: peak.overlap_samples,
                left_factor: left.factor,
                right_factor: right.factor,
                scale: peak.scale,
                rotation_degrees: peak.rotation_degrees,
            })
        })
        .collect()
}

/// Coarse-to-fine area-averaging factors, coarsest first.
///
/// The finest level makes the 13-sample probe window span at least
/// [`VIRTUAL_TILE_MIN_MATCH_WINDOW_NATIVE_PX`], but never shrinks a tile below
/// [`VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE`]. Coarser octaves are added while
/// they still fit: each doubles how far the ±16px search and the bounded
/// residual correction reach, so a prior that is off by a rotation or a
/// perspective remainder is corrected in steps that stay inside the bounds.
/// Small (synthetic) tiles therefore keep a single native level.
fn virtual_tile_pyramid_factors(shortest_long_side: u32) -> Vec<u32> {
    let window = f64::from(2 * VIRTUAL_TILE_DIRECT_PATCH_RADIUS + 1);
    let wanted = (VIRTUAL_TILE_MIN_MATCH_WINDOW_NATIVE_PX / window)
        .ceil()
        .max(1.0) as u32;
    let largest = (shortest_long_side / VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE).max(1);
    let mut factors = vec![wanted.min(largest).max(1)];
    while factors[0].saturating_mul(2) <= largest {
        factors.insert(0, factors[0] * 2);
    }
    factors
}

/// Probe-level metadata for one station: the bounded matcher treats a level as
/// its own pixel grid, so the plane is exposed with `scale_factor = 1`.
fn virtual_tile_plane_image_info(tile: &ImageInfo, plane: &VirtualTileMatchPlane) -> ImageInfo {
    ImageInfo {
        id: tile.id,
        filename: tile.filename.clone(),
        width: plane.gray.width(),
        height: plane.gray.height(),
        alignment_image: plane.gray.clone(),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: None,
        source_footprint: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

/// Covered luminance of a rendered Virtual_Tile; uncovered payload is zero.
fn virtual_tile_covered_luminance(
    pixels: &Rgb32FImage,
    coverage: &stitching::CoverageMask,
) -> GrayImage {
    let (width, height) = pixels.dimensions();
    if coverage.dimensions() != (width, height) {
        return GrayImage::new(width, height);
    }
    GrayImage::from_fn(width, height, |x, y| {
        if !coverage.is_covered(x, y) {
            return image::Luma([0]);
        }
        let pixel = pixels.get_pixel(x, y);
        let luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
        image::Luma([(luminance.clamp(0.0, 1.0) * 255.0).round() as u8])
    })
}

/// FAST/BRIEF features of a covered plane, keeping only keypoints whose
/// [`VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS`] window is covered, the same rule
/// as the native covered-feature path. (Scaling that window by
/// `1 / support_scale` discarded two thirds of the `langyuan-10` 1→2 inliers:
/// the overlaps are strips along tile borders.) Uncovered plane pixels are
/// zero, so a descriptor can only see a flat border, never uncovered payload.
fn virtual_tile_plane_features(
    plane: &VirtualTileMatchPlane,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
) -> Vec<Feature> {
    let (width, height) = plane.gray.dimensions();
    processing::find_features_multiscale(
        &plane.gray,
        brief_pairs,
        VIRTUAL_TILE_PRIOR_FREE_MAX_FEATURES,
        &VIRTUAL_TILE_PRIOR_FREE_SCALES,
    )
    .into_iter()
    .filter(|feature| {
        let radius = VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS;
        let (x, y) = (feature.keypoint.x, feature.keypoint.y);
        x >= radius
            && y >= radius
            && x.saturating_add(radius) < width
            && y.saturating_add(radius) < height
            && (y - radius..=y + radius).all(|sample_y| {
                (x - radius..=x + radius)
                    .all(|sample_x| plane.coverage.is_covered(sample_x, sample_y))
            })
    })
    .collect()
}

/// Station planes of every probe level, `[level][station]`, coarsest first.
struct VirtualTileMatchPyramid {
    factors: Vec<u32>,
    levels: Vec<Vec<ImageInfo>>,
    coverages: Vec<Vec<stitching::CoverageMask>>,
    /// Native tile dimensions per station.
    native_dimensions: Vec<(u32, u32)>,
    /// Feature proposal grids are independent of the bounded probe grids.
    feature_factors: Vec<u32>,
    /// Native structural images for the final, one-native-pixel round trip.
    native_refinement: Vec<Option<(GrayImage, stitching::CoverageMask)>>,
    /// Coverage_Mask of each station's relation-quality plane, in the pixels
    /// of the `alignment_image` [`install_virtual_tile_match_planes`] installs.
    quality_coverages: Vec<stitching::CoverageMask>,
    /// Partially covered ultra-coarse planes cached once per rendered station.
    global_planes: Vec<Option<VirtualTileGlobalSearchPlane>>,
}

impl VirtualTileMatchPyramid {
    fn new(factors: Vec<u32>, station_count: usize) -> Self {
        let empty_coverage = || {
            stitching::CoverageMask::from_bytes(1, 1, vec![0])
                .expect("a one-pixel empty Coverage_Mask is valid")
        };
        let empty_info = || ImageInfo {
            id: usize::MAX,
            filename: String::new(),
            width: 1,
            height: 1,
            alignment_image: GrayImage::new(1, 1),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            source_footprint: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        };
        let level_count = factors.len();
        Self {
            factors,
            levels: (0..level_count)
                .map(|_| (0..station_count).map(|_| empty_info()).collect())
                .collect(),
            coverages: (0..level_count)
                .map(|_| (0..station_count).map(|_| empty_coverage()).collect())
                .collect(),
            native_dimensions: vec![(1, 1); station_count],
            feature_factors: vec![1; station_count],
            native_refinement: vec![None; station_count],
            quality_coverages: (0..station_count).map(|_| empty_coverage()).collect(),
            global_planes: vec![None; station_count],
        }
    }

    /// The masks the station solver re-checks each relation on: the
    /// relation-quality planes (requirement 6.8) and the finest match planes
    /// (requirement 6.4's overlap area).
    fn solver_coverages(&self) -> VirtualTileSolverCoverages<'_> {
        VirtualTileSolverCoverages {
            quality: &self.quality_coverages,
            overlap: self.coverages.last().map_or(&[][..], Vec::as_slice),
            overlap_factor: self.factors.last().copied().unwrap_or(1),
        }
    }

    /// Replace one station's planes and return them, coarsest first.
    fn set_station(
        &mut self,
        station: usize,
        tile: &ImageInfo,
        gray: &GrayImage,
        coverage: &stitching::CoverageMask,
    ) -> Option<Vec<VirtualTileMatchPlane>> {
        let mut planes = Vec::with_capacity(self.factors.len());
        self.native_dimensions[station] = gray.dimensions();
        self.global_planes[station] = VirtualTileGlobalSearchPlane::from_native(gray, coverage);
        let native_sigma = 1.0;
        #[cfg(test)]
        let native_sigma = std::env::var("RAW_EDITOR_TILE_REPLAY_NATIVE_SIGMA")
            .ok()
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(native_sigma);
        self.native_refinement[station] = (gray.width().max(gray.height()) >= 4096
            && self.factors.last().is_some_and(|factor| *factor > 1))
        .then(|| {
            (
                imageproc::filter::gaussian_blur_f32(gray, native_sigma),
                coverage.clone(),
            )
        });
        for (level, &factor) in self.factors.iter().enumerate() {
            let plane = VirtualTileMatchPlane::from_native(gray, coverage, factor)?;
            self.levels[level][station] = virtual_tile_plane_image_info(tile, &plane);
            self.coverages[level][station] = plane.coverage.clone();
            planes.push(plane);
        }
        (!planes.is_empty()).then_some(planes)
    }
}

/// Octaves above the finest one at which the station solver compares edges.
const VIRTUAL_TILE_RELATION_QUALITY_OCTAVES: u32 = 2;
/// Gaussian σ, in plane pixels, of the structural relation-quality plane.
const VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA: f32 = 1.0;
/// Half-width, in relation-quality plane pixels, of the neighbourhood that must
/// be covered around a requirement-6.8 sample: the ±1px gradient taps with
/// bilinear access read 2 pixels around it and the σ = 1 blur (imageproc
/// kernel radius ⌈2σ⌉) 2 more. On langyuan-10 the uncovered payload inside
/// that reach alone moved the verified 1→2 relation's edge-strength ratio
/// from 1.01 to 1.40 and its low-frequency difference from 0.002 to 0.13.
const VIRTUAL_TILE_QUALITY_COVERAGE_RADIUS_PX: f64 = 4.0;

/// The plane on which the station solver re-checks requirement 6.8.
///
/// Its `focus_overlap_quality_measurements` compares ±1-pixel gradients at a
/// sparse grid. On langyuan-10 the verified 0→2 and 1→2 relations read a
/// median edge-orientation difference of 21–38° on the unfiltered 1/4–1/16
/// planes — the level of a 40px-shifted wrong relation (38–46°), because single
/// pixel gradients follow fine silk texture. Two octaves above the finest level
/// and after a σ = 1 blur they read 4.6–5.9°, against 27–28° for the shifted
/// relation. A single matching level still needs a separate structural quality
/// plane: the probe pyramid's minimum size must not switch verification back
/// to fine silk texture. Keep at least 64 pixels on its shorter side so the
/// dense grid, gradient taps and covered blur neighbourhood remain measurable.
fn virtual_tile_relation_quality_plane(
    planes: &[VirtualTileMatchPlane],
    native: &GrayImage,
    coverage: &stitching::CoverageMask,
) -> (GrayImage, u32) {
    let Some(finest) = planes.last() else {
        return (GrayImage::new(1, 1), 1);
    };
    if planes.len() < 2 {
        // The probe's factor can be truncated to fit a small preview. It is
        // not a structural scale: multiplying that factor by four gave only
        // ~100 pixels across real previews and followed illumination gradients
        // instead of artwork edges. Choose the quality grid from native pixels.
        let wanted = native
            .width()
            .max(native.height())
            .div_ceil(600)
            .max(1)
            .next_power_of_two()
            .max(1 << VIRTUAL_TILE_RELATION_QUALITY_OCTAVES);
        let factor = wanted.min((native.width().min(native.height()) / 64).max(1));
        let structural = VirtualTileMatchPlane::from_native(native, coverage, factor)
            .expect("a validated probe plane has valid grayscale and coverage dimensions");
        return (
            imageproc::filter::gaussian_blur_f32(
                &structural.gray,
                VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
            ),
            factor,
        );
    }
    let wanted = finest
        .factor
        .saturating_mul(1 << VIRTUAL_TILE_RELATION_QUALITY_OCTAVES);
    let plane = planes
        .iter()
        .filter(|plane| plane.factor <= wanted)
        .max_by_key(|plane| plane.factor)
        .unwrap_or(finest);
    (
        imageproc::filter::gaussian_blur_f32(&plane.gray, VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA),
        plane.factor,
    )
}

/// A coarse pyramid failure may reflect the level's sparse residual model,
/// while a finer level can still measure the same prior.  Keep this retry
/// list deliberately narrow: a missing overlap or missing texture is not
/// evidence that becomes stronger at a finer level.
fn virtual_tile_coarse_failure_can_retry(failure_stage: &str) -> bool {
    matches!(
        failure_stage,
        // At the coarsest octave a narrow, but valid, overlap can contain
        // fewer than the final minimum number of grid probes.  That is a
        // sampling-density failure of this octave, not evidence that the
        // station pair is unrelated.  Retry at the next finer bounded octave
        // and let the unchanged final polish/Station_Relation gates decide.
        "insufficient_grid_probes"
            | "insufficient_measurable_patches"
            | "insufficient_bidirectional_matches"
            | "insufficient_grid_coverage"
            | "residual_model_no_fit"
            | "insufficient_matches_after_model_quota"
            | "insufficient_fitted_hull_support"
    )
}

/// Probe one pair coarse to fine around `seed_left_to_right` (native tile
/// pixels). Every octave but the finest runs the bounded bidirectional probe
/// (edge orientation read on the octave's σ = 1 structural plane) and re-seeds
/// the next with its measured model (tolerance
/// [`VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX`] in its own pixels); the
/// finest octave polishes that model with model-warped templates
/// ([`virtual_tile_polished_station_match`]). A single native level (small
/// tiles) keeps the bounded probe with the requirement-6.3 tolerance. The
/// relation is expressed in native tile pixels; the diagnostic describes the
/// last stage that ran.
fn virtual_tile_pyramid_station_match(
    left: usize,
    right: usize,
    pyramid: &VirtualTileMatchPyramid,
    seed_left_to_right: &Matrix3<f64>,
    seed_kind: &'static str,
) -> (Vec<MatchInfo>, stack_report::StationRelationCandidateRecord) {
    let empty_diagnostic = || {
        let (_, mut diagnostic) = virtual_tile_direct_station_match(
            left,
            right,
            &[],
            &[],
            &HashMap::new(),
            STATION_RELATION_MAX_MEDIAN_ERROR_PX,
        );
        diagnostic.pyramid_factors = pyramid.factors.clone();
        diagnostic.seed = seed_kind.to_string();
        diagnostic
    };
    let level_count = pyramid.factors.len();
    if level_count == 0
        || left >= pyramid.native_dimensions.len()
        || right >= pyramid.native_dimensions.len()
    {
        return (Vec::new(), empty_diagnostic());
    }
    let single_level = level_count == 1;
    let bounded_levels = if single_level { 1 } else { level_count - 1 };
    let mut seed = *seed_left_to_right;
    let mut diagnostic = empty_diagnostic();
    for level in 0..bounded_levels {
        let factor = pyramid.factors[level];
        let to_native = VirtualTileMatchPlane::level_to_native(factor);
        let from_native = VirtualTileMatchPlane::native_to_level(factor);
        let level_tiles = &pyramid.levels[level];
        let Some(level_seed_inverse) = (from_native * seed * to_native).try_inverse() else {
            diagnostic.failure_stage = "non_invertible_seed".to_string();
            return (Vec::new(), diagnostic);
        };
        let initial = HashMap::from([
            (level_tiles[left].id, Matrix3::identity()),
            (level_tiles[right].id, level_seed_inverse),
        ]);
        let inlier_threshold_px = if single_level {
            STATION_RELATION_MAX_MEDIAN_ERROR_PX
        } else {
            VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX
        };
        // A coarse octave reads edge orientation on its structural (σ = 1)
        // plane, the rule of the relation-quality plane: on langyuan-10 ±1px
        // gradients of the unfiltered 1/8 planes read 13.4–15.5° for verified
        // relations (the prior-free 1→2 proposal failed the 15° gate at 15.5°),
        // the structural planes 5.1–6.0°, and a 2-level-pixel shift 27–36°.
        let structure = (!single_level).then(|| {
            (
                imageproc::filter::gaussian_blur_f32(
                    &level_tiles[left].alignment_image,
                    VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
                ),
                imageproc::filter::gaussian_blur_f32(
                    &level_tiles[right].alignment_image,
                    VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA,
                ),
            )
        });
        let (relation, level_diagnostic) = virtual_tile_direct_station_match_with_structure(
            left,
            right,
            level_tiles,
            &pyramid.coverages[level],
            &initial,
            inlier_threshold_px,
            structure
                .as_ref()
                .map(|(left_plane, right_plane)| (left_plane, right_plane)),
        );
        diagnostic = level_diagnostic;
        diagnostic.pyramid_factors = pyramid.factors.clone();
        diagnostic.measurement_factor = factor;
        diagnostic.seed = seed_kind.to_string();
        let Some(relation) = relation else {
            // A later octave has a smaller residual in its own pixels and can
            // recover a model that was under-supported at this coarse level.
            // Keep the original seed until a measured relation is available;
            // no prior is ever promoted to acceptance evidence.
            if !single_level && virtual_tile_coarse_failure_can_retry(&diagnostic.failure_stage) {
                if level + 1 < bounded_levels {
                    continue;
                }
                // The last bounded octave has no finer bounded probe left.
                // Its residual-model/support failure is still a recoverable
                // measurement failure: the finest polish has independent
                // patch, reverse-consistency, hull, scale, and orientation
                // gates and must get the chance to verify the original prior.
                // Do not return the coarse diagnostic here, because that
                // would make the finest evidence unreachable for exactly the
                // sparse, shifted overlaps this retry is meant to recover.
                break;
            }
            return (Vec::new(), diagnostic);
        };
        let Some(native) = normalized_homography(&(to_native * relation.homography * from_native))
        else {
            diagnostic.failure_stage = "non_finite_native_relation".to_string();
            return (Vec::new(), diagnostic);
        };
        seed = native;
        if single_level && factor == 1 {
            let points = relation
                .points
                .iter()
                .filter_map(|(source, target)| {
                    transformed_point(&to_native, *source)
                        .zip(transformed_point(&to_native, *target))
                })
                .collect::<Vec<_>>();
            return (
                vec![MatchInfo {
                    homography: native,
                    inliers: points.len(),
                    points,
                    // Key-direction fit; see `MatchInfo::canonical_homography`.
                    canonical_homography: None,
                    native_verified_points: Vec::new(),
                    ..relation
                }],
                diagnostic,
            );
        }
    }
    let finest = level_count - 1;
    let factor = pyramid.factors[finest];
    diagnostic.measurement_factor = factor;
    let relation = virtual_tile_polished_station_match(
        &pyramid.levels[finest][left],
        &pyramid.coverages[finest][left],
        &pyramid.levels[finest][right],
        &pyramid.coverages[finest][right],
        factor,
        &seed,
        (
            pyramid.native_dimensions[left],
            pyramid.native_dimensions[right],
        ),
        &mut diagnostic,
    );
    let relation = relation.into_iter().filter_map(|relation| {
        match (
            &pyramid.native_refinement[left],
            &pyramid.native_refinement[right],
        ) {
            (Some(left_native), Some(right_native)) => virtual_tile_native_station_polish(
                left_native,
                right_native,
                &relation,
                &mut diagnostic,
            ),
            _ => Some(relation),
        }
    });
    (relation.collect(), diagnostic)
}

/// Coarse pixels establish a search model. Final camera geometry is measured
/// on native pixels: a one-level-pixel reverse check at factor four otherwise
/// admits four-native-pixel errors and biases long projective tree chains.
fn native_station_affine_candidate(
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    source: &stitching::CoverageMask,
    target: &stitching::CoverageMask,
) -> Option<Matrix3<f64>> {
    if points.len() < STATION_RELATION_MIN_INLIERS {
        return None;
    }
    robust_native_affine(points).filter(|affine| {
        median_symmetric_error(affine, points) <= STATION_RELATION_MAX_MEDIAN_ERROR_PX
            && homography_preserves_focus_orientation(affine, source_dimensions)
            && (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
                .contains(&homography_scale_ratio(affine).unwrap_or(f64::NAN))
            && station_relation_spatial_support_with(
                points,
                affine,
                source_dimensions,
                target_dimensions,
                StationOverlapMeasure::Covered {
                    source,
                    target,
                    factor: 1,
                },
            ) >= STATION_RELATION_MIN_SPATIAL_SUPPORT
    })
}

/// Downweight native point outliers without changing the observations used by
/// the acceptance gates. Keep the model with the lowest symmetric median.
fn robust_native_affine(points: &[(Point2<f64>, Point2<f64>)]) -> Option<Matrix3<f64>> {
    let mut best = estimate_affine(points)?;
    let mut best_error = median_symmetric_error(&best, points);
    let mut current = best;
    for _ in 0..30 {
        let inverse = current.try_inverse()?;
        let errors = points
            .iter()
            .map(|&(a, b)| {
                Some(
                    ((transformed_point(&current, a)? - b).norm_squared()
                        + (transformed_point(&inverse, b)? - a).norm_squared())
                    .sqrt(),
                )
            })
            .collect::<Option<Vec<_>>>()?;
        let mut design = nalgebra::DMatrix::zeros(points.len(), 3);
        let mut targets = nalgebra::DMatrix::zeros(points.len(), 2);
        for (index, (&(a, b), error)) in points.iter().zip(&errors).enumerate() {
            let weight = (1.0 / error.max(1e-9)).min(1.0).sqrt();
            design[(index, 0)] = a.x * weight;
            design[(index, 1)] = a.y * weight;
            design[(index, 2)] = weight;
            targets[(index, 0)] = b.x * weight;
            targets[(index, 1)] = b.y * weight;
        }
        let solution = design.svd(true, true).solve(&targets, 1e-10).ok()?;
        current = Matrix3::new(
            solution[(0, 0)],
            solution[(1, 0)],
            solution[(2, 0)],
            solution[(0, 1)],
            solution[(1, 1)],
            solution[(2, 1)],
            0.0,
            0.0,
            1.0,
        );
        let error = median_symmetric_error(&current, points);
        if error < best_error {
            best = current;
            best_error = error;
        }
    }
    Some(best)
}

/// Use only the perspective needed by the native evidence. A noisy projective
/// fit can predict large corner motion outside the measured overlap even when
/// its inlier error is small. Every proposal is checked on the full retained
/// point set and against the existing coverage, scale and orientation gates.
fn native_station_regularized_candidate(
    points: &[(Point2<f64>, Point2<f64>)],
    projective: &Matrix3<f64>,
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    source: &stitching::CoverageMask,
    target: &stitching::CoverageMask,
) -> Option<Matrix3<f64>> {
    if points.len() < STATION_RELATION_MIN_INLIERS {
        return None;
    }
    let affine = robust_native_affine(points)?;
    let projective = projective / projective[(2, 2)];
    for step in 0..=100 {
        let fraction = step as f64 / 100.0;
        let candidate = affine * (1.0 - fraction) + projective * fraction;
        if median_symmetric_error(&candidate, points) <= STATION_RELATION_MAX_MEDIAN_ERROR_PX
            && homography_preserves_focus_orientation(&candidate, source_dimensions)
            && (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
                .contains(&homography_scale_ratio(&candidate).unwrap_or(f64::NAN))
            && station_relation_spatial_support_with(
                points,
                &candidate,
                source_dimensions,
                target_dimensions,
                StationOverlapMeasure::Covered {
                    source,
                    target,
                    factor: 1,
                },
            ) >= STATION_RELATION_MIN_SPATIAL_SUPPORT
        {
            return Some(candidate);
        }
    }
    None
}

fn virtual_tile_native_station_polish(
    left: &(GrayImage, stitching::CoverageMask),
    right: &(GrayImage, stitching::CoverageMask),
    seed: &MatchInfo,
    diagnostic: &mut stack_report::StationRelationCandidateRecord,
) -> Option<MatchInfo> {
    let gates = crate::panorama_utils::registration::PatchMatchGates::LEGACY;
    #[cfg(test)]
    let gates = crate::panorama_utils::registration::PatchMatchGates {
        min_correlation: std::env::var("RAW_EDITOR_TILE_REPLAY_NATIVE_CORRELATION")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
            .unwrap_or(gates.min_correlation),
        min_peak_margin: std::env::var("RAW_EDITOR_TILE_REPLAY_NATIVE_PEAK_MARGIN")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(gates.min_peak_margin),
        ..gates
    };
    let inverse = seed.homography.try_inverse()?;
    let step = (seed.points.len() / 1000).max(1);
    let supported = |mask: &stitching::CoverageMask, point: Point2<f64>| {
        let (w, h) = mask.dimensions();
        let radius = 40i32;
        let (x, y) = (point.x.round() as i32, point.y.round() as i32);
        x >= radius
            && y >= radius
            && x + radius < w as i32
            && y + radius < h as i32
            && (-radius..=radius).step_by(8).all(|dy| {
                (-radius..=radius)
                    .step_by(8)
                    .all(|dx| mask.is_covered((x + dx) as u32, (y + dy) as u32))
            })
    };
    let native_verified_points = seed
        .points
        .par_iter()
        .enumerate()
        .filter(|(index, _)| index % step == 0)
        .filter_map(|(_, &(source, target))| {
            if !supported(&left.1, source) || !supported(&right.1, target) {
                return None;
            }
            let measured = residual_warp::verify_round_trip_match_with_gates(
                &left.0,
                &right.0,
                &seed.homography,
                &inverse,
                source,
                31,
                gates,
            )?;
            supported(&right.1, measured.forward).then_some((
                source,
                measured.forward,
                measured.round_trip_error_px,
            ))
        })
        .collect::<Vec<_>>();
    let points = native_verified_points
        .iter()
        .map(|&(a, b, _)| (a, b))
        .collect::<Vec<_>>();
    diagnostic.failure_stage = "insufficient_native_round_trip_matches".to_string();
    diagnostic.bidirectional_match_count = points.len();
    diagnostic.measurement_factor = 1;
    if points.len() < STATION_RELATION_MIN_INLIERS {
        return None;
    }
    let (homography, indices) = virtual_tile_polish_fit(
        &points,
        VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX,
        left.0.dimensions(),
    )?;
    let native_verified_points = indices
        .into_iter()
        .map(|index| native_verified_points[index])
        .collect::<Vec<_>>();
    let points = native_verified_points
        .iter()
        .map(|&(a, b, _)| (a, b))
        .collect::<Vec<_>>();
    // Prefer fewer camera parameters only on the same verified native points
    // and with the existing median, scale, orientation and support gates.
    let regularized = native_station_regularized_candidate(
        &points,
        &homography,
        left.0.dimensions(),
        right.0.dimensions(),
        &left.1,
        &right.1,
    );
    let homography = regularized.unwrap_or(homography);
    diagnostic.fitted_inliers = points.len();
    diagnostic.hull_support = station_relation_spatial_support_with(
        &points,
        &homography,
        left.0.dimensions(),
        right.0.dimensions(),
        StationOverlapMeasure::Covered {
            source: &left.1,
            target: &right.1,
            factor: 1,
        },
    );
    diagnostic.residual_magnitude_px = residual_distribution(match homography.try_inverse() {
        Some(inverse) => points
            .iter()
            .map(|(a, b)| symmetric_point_error(&homography, &inverse, *a, *b))
            .collect(),
        None => Vec::new(),
    });
    diagnostic.failure_stage = "insufficient_native_polished_support".to_string();
    if diagnostic.hull_support < STATION_RELATION_MIN_SPATIAL_SUPPORT
        || median_symmetric_error(&homography, &points) > STATION_RELATION_MAX_MEDIAN_ERROR_PX
    {
        return None;
    }
    diagnostic.residual_model = if homography[(2, 0)] == 0.0 && homography[(2, 1)] == 0.0 {
        "native_polished_affine"
    } else if regularized.is_some() {
        "native_polished_regularized_projective"
    } else {
        "native_polished_projective"
    }
    .to_string();
    diagnostic.failure_stage = "fitted".to_string();
    Some(MatchInfo {
        homography,
        inliers: points.len(),
        points,
        canonical_homography: None,
        native_verified_points,
        ..seed.clone()
    })
}

// ---------------------------------------------------------------------------
// Finest-octave polish
//
// The bounded 13-sample probe is a good *seed* but a poor final instrument on
// the finest octave: on langyuan-10 only 23–30% of its 1/4 bidirectional
// matches agreed with an independent relation, and the ones that did covered
// 16–19% of the 0→2 overlap. Matching 15×15 templates of the left plane against
// the right plane resampled through the current relation (so the template and
// the window share one geometry) gave 846–1011 inliers at a 0.9–1.1px one-way
// median, and 0→2 reached 0.22–0.28 support at a 1.2–1.9px one-way median once
// the inlier tolerance admitted the overlap's non-homographic remainder.
// Requirement 6.3 bounds the inliers' *median*; the solver re-checks it.
// ---------------------------------------------------------------------------

const VIRTUAL_TILE_POLISH_TEMPLATE_HALF: i32 = 7;
const VIRTUAL_TILE_POLISH_GRID_STEP_PX: f64 = 16.0;
/// Search radii of the two polish passes, in finest-plane pixels. The first
/// absorbs what the coarser octaves left; the second re-matches around it.
const VIRTUAL_TILE_POLISH_SEARCH_PASSES: [i32; 2] = [8, 4];
const VIRTUAL_TILE_POLISH_MIN_NCC: f64 = 0.6;
const VIRTUAL_TILE_POLISH_MIN_PEAK_MARGIN: f64 = 0.03;
/// Smallest template standard deviation in 8-bit levels.
const VIRTUAL_TILE_POLISH_MIN_TEMPLATE_STD: f64 = 3.0;
/// Symmetric inlier tolerance of the polished relation in native pixels: both
/// one-way errors within the requirement-6.3 3.0 world pixels. (Twice that let
/// the refits drift to a looser model: 0→2 read a 3.43px symmetric median,
/// against 2.09px at 0.29 support with this tolerance.)
const VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX: f64 =
    std::f64::consts::SQRT_2 * STATION_RELATION_MAX_MEDIAN_ERROR_PX;

#[derive(Default)]
struct VirtualTilePolishCounts {
    probes: usize,
    textured: usize,
    matched: usize,
}

/// Zero-mean normalised cross-correlation of two equally long sample runs.
fn virtual_tile_ncc(left: &[f64], right: &[f64]) -> Option<f64> {
    let count = left.len() as f64;
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / count;
    let right_mean = right.iter().sum::<f64>() / count;
    let mut product = 0.0;
    let mut left_energy = 0.0;
    let mut right_energy = 0.0;
    for (&l, &r) in left.iter().zip(right) {
        let (l, r) = (l - left_mean, r - right_mean);
        product += l * r;
        left_energy += l * l;
        right_energy += r * r;
    }
    let denominator = (left_energy * right_energy).sqrt();
    (denominator > f64::EPSILON).then(|| product / denominator)
}

/// Best integer offset of `template` inside a square `window` of side
/// `2·(half + search) + 1`, its NCC, and the best NCC at least two pixels away.
#[allow(clippy::type_complexity)]
fn virtual_tile_best_window_offset(
    template: &[f64],
    window: &[f64],
    half: i32,
    search: i32,
) -> Option<((i32, i32), f64, f64, Vec<f64>)> {
    let side = (2 * (half + search) + 1) as usize;
    let template_side = (2 * half + 1) as usize;
    let offsets = (2 * search + 1) as usize;
    let mut scores = vec![-1.0; offsets * offsets];
    let mut sample = vec![0.0; template.len()];
    for oy in 0..offsets {
        for ox in 0..offsets {
            for ty in 0..template_side {
                let row = (oy + ty) * side + ox;
                sample[ty * template_side..(ty + 1) * template_side]
                    .copy_from_slice(&window[row..row + template_side]);
            }
            if let Some(score) = virtual_tile_ncc(template, &sample) {
                scores[oy * offsets + ox] = score;
            }
        }
    }
    let (best_index, &best) =
        scores
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })?;
    let (bx, by) = ((best_index % offsets) as i32, (best_index / offsets) as i32);
    let second = scores
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            let (x, y) = ((index % offsets) as i32, (index / offsets) as i32);
            (x - bx).abs() >= 2 || (y - by).abs() >= 2
        })
        .map(|(_, &score)| score)
        .fold(-1.0f64, f64::max);
    Some(((bx - search, by - search), best, second, scores))
}

/// Model-warped template correspondences `(left plane, right plane)`.
#[allow(clippy::too_many_arguments)]
fn virtual_tile_polish_correspondences(
    left: &GrayImage,
    left_coverage: &stitching::CoverageMask,
    right: &GrayImage,
    right_coverage: &stitching::CoverageMask,
    left_to_right: &Matrix3<f64>,
    bounds: (f64, f64, f64, f64),
    search: i32,
    counts: &mut VirtualTilePolishCounts,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let half = VIRTUAL_TILE_POLISH_TEMPLATE_HALF;
    #[cfg(test)]
    let half = std::env::var("RAW_EDITOR_TILE_REPLAY_TEMPLATE_HALF")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(half);
    let reach = half + search;
    let (min_x, min_y, max_x, max_y) = bounds;
    let mut centers = Vec::new();
    let mut y = (min_y + f64::from(reach)).ceil();
    while y <= max_y - f64::from(reach) {
        let mut x = (min_x + f64::from(reach)).ceil();
        while x <= max_x - f64::from(reach) {
            centers.push((x as i32, y as i32));
            x += VIRTUAL_TILE_POLISH_GRID_STEP_PX;
        }
        y += VIRTUAL_TILE_POLISH_GRID_STEP_PX;
    }
    counts.probes += centers.len();
    let right_sample = |point: Point2<f64>| -> Option<f64> {
        let (x0, y0) = (point.x.floor(), point.y.floor());
        if x0 < 0.0 || y0 < 0.0 {
            return None;
        }
        let (x0, y0) = (x0 as u32, y0 as u32);
        [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)]
            .iter()
            .all(|&(x, y)| right_coverage.is_covered(x, y))
            .then(|| registration::sample_gray(right, point.x, point.y))
            .flatten()
    };
    // One task per centre; results are collected in centre order.
    let results = centers
        .par_iter()
        .map(|&(cx, cy)| {
            let mut template = Vec::with_capacity(((2 * half + 1) * (2 * half + 1)) as usize);
            for dy in -half..=half {
                for dx in -half..=half {
                    let (x, y) = (cx + dx, cy + dy);
                    if x < 0 || y < 0 || !left_coverage.is_covered(x as u32, y as u32) {
                        return (false, None);
                    }
                    template.push(f64::from(left.get_pixel(x as u32, y as u32)[0]));
                }
            }
            let mean = template.iter().sum::<f64>() / template.len() as f64;
            let std = (template.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / template.len() as f64)
                .sqrt();
            if std < VIRTUAL_TILE_POLISH_MIN_TEMPLATE_STD {
                return (false, None);
            }
            let side = (2 * reach + 1) as usize;
            let mut window = Vec::with_capacity(side * side);
            for dy in -reach..=reach {
                for dx in -reach..=reach {
                    let point = Point2::new(f64::from(cx + dx), f64::from(cy + dy));
                    let Some(value) =
                        transformed_point(left_to_right, point).and_then(right_sample)
                    else {
                        return (true, None);
                    };
                    window.push(value);
                }
            }
            let Some(((ox, oy), best, second, scores)) =
                virtual_tile_best_window_offset(&template, &window, half, search)
            else {
                return (true, None);
            };
            if best < VIRTUAL_TILE_POLISH_MIN_NCC
                || best - second < VIRTUAL_TILE_POLISH_MIN_PEAK_MARGIN
                || ox.abs() >= search
                || oy.abs() >= search
            {
                return (true, None);
            }
            // Reverse: the matched warped-right content must find its way back
            // to the template centre in the left plane.
            let reverse_template = {
                let mut values = Vec::with_capacity(template.len());
                for dy in -half..=half {
                    for dx in -half..=half {
                        let row = (oy + dy + reach) as usize;
                        let column = (ox + dx + reach) as usize;
                        values.push(window[row * side + column]);
                    }
                }
                values
            };
            let back_search = 2;
            let back_reach = half + back_search;
            let mut back_window = Vec::with_capacity(((2 * back_reach + 1).pow(2)) as usize);
            for dy in -back_reach..=back_reach {
                for dx in -back_reach..=back_reach {
                    // The reverse template should land back at the original
                    // source centre.  Searching around the forward match
                    // (`cx + ox`) made the reverse offset equal to `-ox`,
                    // although `back_search` is intentionally only two
                    // pixels wide.  A valid forward correction of five
                    // pixels was therefore rejected before fitting.  Keep
                    // the reverse search in the source coordinate frame and
                    // require its offset to return to that centre.
                    let (x, y) = (cx + dx, cy + dy);
                    if x < 0 || y < 0 || !left_coverage.is_covered(x as u32, y as u32) {
                        return (true, None);
                    }
                    back_window.push(f64::from(left.get_pixel(x as u32, y as u32)[0]));
                }
            }
            let Some(((bx, by), _, _, _)) =
                virtual_tile_best_window_offset(&reverse_template, &back_window, half, back_search)
            else {
                return (true, None);
            };
            if f64::from(bx).hypot(f64::from(by)) > 1.0 {
                return (true, None);
            }
            // Separable parabolic sub-pixel peak on the forward score surface.
            let offsets = (2 * search + 1) as usize;
            let at =
                |x: i32, y: i32| scores[(y + search) as usize * offsets + (x + search) as usize];
            let parabola = |minus: f64, center: f64, plus: f64| {
                let denominator = minus - 2.0 * center + plus;
                if denominator.abs() > f64::EPSILON {
                    (0.5 * (minus - plus) / denominator).clamp(-0.5, 0.5)
                } else {
                    0.0
                }
            };
            let sx = parabola(at(ox - 1, oy), best, at(ox + 1, oy));
            let sy = parabola(at(ox, oy - 1), best, at(ox, oy + 1));
            let warped = Point2::new(f64::from(cx + ox) + sx, f64::from(cy + oy) + sy);
            let Some(target) = transformed_point(left_to_right, warped) else {
                return (true, None);
            };
            (
                true,
                Some((Point2::new(f64::from(cx), f64::from(cy)), target)),
            )
        })
        .collect::<Vec<_>>();
    counts.textured += results.iter().filter(|(textured, _)| *textured).count();
    let matches = results
        .into_iter()
        .filter_map(|(_, correspondence)| correspondence)
        .collect::<Vec<_>>();
    counts.matched += matches.len();
    matches
}

/// Robust homography over polished correspondences: RANSAC with a per-direction
/// tolerance of `threshold / √2`, then least-squares refits on the symmetric
/// inlier set.
fn virtual_tile_polish_fit(
    correspondences: &[(Point2<f64>, Point2<f64>)],
    threshold: f64,
    left_dimensions: (u32, u32),
) -> Option<(Matrix3<f64>, Vec<usize>)> {
    let (seed, mut inliers) = processing::find_homography_ransac_points_stable_with_min_inliers(
        correspondences,
        threshold / std::f64::consts::SQRT_2,
        STATION_RELATION_MIN_INLIERS,
    )?;
    let mut homography = seed;
    for _ in 0..3 {
        let points = inliers
            .iter()
            .map(|&index| correspondences[index])
            .collect::<Vec<_>>();
        let Some(refitted) = processing::compute_homography(&points) else {
            break;
        };
        let refitted_inliers = symmetric_inlier_indices(&refitted, correspondences, threshold);
        if refitted_inliers.len() < inliers.len() {
            break;
        }
        let converged = refitted_inliers == inliers;
        homography = refitted;
        inliers = refitted_inliers;
        if converged {
            break;
        }
    }
    let homography = normalized_homography(&homography)?;
    (inliers.len() >= STATION_RELATION_MIN_INLIERS
        && homography_preserves_focus_orientation(&homography, left_dimensions))
    .then_some((homography, inliers))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum VirtualTilePolishModel {
    Similarity,
    Affine,
    Projective,
}

impl VirtualTilePolishModel {
    fn name(self) -> &'static str {
        match self {
            Self::Similarity => "polished_similarity",
            Self::Affine => "polished_affine",
            Self::Projective => "polished_projective",
        }
    }

    fn fit(
        self,
        correspondences: &[(Point2<f64>, Point2<f64>)],
        threshold: f64,
        left_dimensions: (u32, u32),
    ) -> Option<(Matrix3<f64>, Vec<usize>)> {
        if self == Self::Projective {
            // Preserve the retained project's exact projective RANSAC/refit
            // path. Multi-hypothesis polish adds lower-DOF competitors; it
            // must not perturb the projective evidence that already passed.
            return virtual_tile_polish_fit(correspondences, threshold, left_dimensions);
        }
        let fit = match self {
            Self::Similarity => robust_transform_fit_with_threshold(
                correspondences,
                2,
                0x504f_4c49_5348_5349,
                threshold,
                estimate_similarity,
            ),
            Self::Affine => robust_transform_fit_with_threshold(
                correspondences,
                3,
                0x504f_4c49_5348_4146,
                threshold,
                estimate_affine,
            ),
            Self::Projective => unreachable!("handled above"),
        }?;
        let homography = normalized_homography(&fit.transform)?;
        (fit.inlier_indices.len() >= STATION_RELATION_MIN_INLIERS
            && homography.try_inverse().is_some()
            && homography_preserves_focus_orientation(&homography, left_dimensions))
        .then_some((homography, fit.inlier_indices))
    }
}

#[derive(Clone)]
struct VirtualTilePolishHypothesis {
    model: VirtualTilePolishModel,
    homography: Matrix3<f64>,
    inliers: Vec<usize>,
    correspondences: Vec<(Point2<f64>, Point2<f64>)>,
}

fn virtual_tile_hypotheses_are_equivalent(
    left: &Matrix3<f64>,
    right: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> bool {
    focus_transform_disagreement_px(left, right, dimensions)
        .is_some_and(|difference| difference <= 1e-6)
}

/// Polish a pyramid relation on the finest planes. Every independently
/// measurable similarity, affine, and projective model survives to the shared
/// Station_Relation gate; this stage does not let one model's scale or hull
/// result suppress another model before dense covered-overlap measurement.
#[allow(clippy::too_many_arguments)]
fn virtual_tile_polished_station_match(
    left_plane: &ImageInfo,
    left_coverage: &stitching::CoverageMask,
    right_plane: &ImageInfo,
    right_coverage: &stitching::CoverageMask,
    factor: u32,
    seed_native: &Matrix3<f64>,
    native_dimensions: ((u32, u32), (u32, u32)),
    diagnostic: &mut stack_report::StationRelationCandidateRecord,
) -> Vec<MatchInfo> {
    let to_native = VirtualTileMatchPlane::level_to_native(factor);
    let from_native = VirtualTileMatchPlane::native_to_level(factor);
    let threshold = VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX / f64::from(factor.max(1));
    let Some(seed) = normalized_homography(&(from_native * seed_native * to_native)) else {
        diagnostic.failure_stage = "non_invertible_seed".to_string();
        return Vec::new();
    };
    let mut hypotheses = Vec::<VirtualTilePolishHypothesis>::new();
    let mut representative_counts = VirtualTilePolishCounts::default();

    for (pass_index, search) in VIRTUAL_TILE_POLISH_SEARCH_PASSES.into_iter().enumerate() {
        let seeds = if pass_index == 0 {
            vec![(None, seed)]
        } else {
            hypotheses
                .iter()
                .map(|hypothesis| (Some(hypothesis.model), hypothesis.homography))
                .collect::<Vec<_>>()
        };
        if seeds.is_empty() {
            break;
        }
        let mut next = Vec::new();
        for (restricted_model, relation) in seeds {
            diagnostic.failure_stage = "no_predicted_overlap".to_string();
            let Some(bounds) = virtual_tile_predicted_overlap_bounds(
                &relation,
                left_plane.dimensions(),
                right_plane.dimensions(),
            ) else {
                continue;
            };
            diagnostic.predicted_overlap_area_px = diagnostic.predicted_overlap_area_px.max(
                station_relation_overlap_area(
                    &relation,
                    left_plane.dimensions(),
                    right_plane.dimensions(),
                    true,
                )
                .unwrap_or(0.0),
            );
            let mut counts = VirtualTilePolishCounts::default();
            let correspondences = virtual_tile_polish_correspondences(
                &left_plane.alignment_image,
                left_coverage,
                &right_plane.alignment_image,
                right_coverage,
                &relation,
                bounds,
                search,
                &mut counts,
            );
            if counts.matched > representative_counts.matched {
                representative_counts = counts;
            }
            if correspondences.len() < STATION_RELATION_MIN_INLIERS {
                continue;
            }
            for model in [
                VirtualTilePolishModel::Similarity,
                VirtualTilePolishModel::Affine,
                VirtualTilePolishModel::Projective,
            ] {
                if restricted_model.is_some_and(|restricted| restricted != model) {
                    continue;
                }
                let Some((homography, inliers)) =
                    model.fit(&correspondences, threshold, left_plane.dimensions())
                else {
                    continue;
                };
                next.push(VirtualTilePolishHypothesis {
                    model,
                    homography,
                    inliers,
                    correspondences: correspondences.clone(),
                });
            }
        }
        if pass_index == 0 {
            hypotheses = next;
        } else {
            // A first-pass lower-DOF model can cover the complete overlap even
            // when its own narrower second-pass search locks onto only one
            // repeated-texture band. Keep both independently measured fits;
            // the shared final gate, not polish iteration order, decides.
            hypotheses.extend(next);
        }
    }

    diagnostic.grid_probe_count = representative_counts.probes;
    diagnostic.measurable_patch_count = representative_counts.textured;
    diagnostic.bidirectional_match_count = representative_counts.matched;
    diagnostic.search_radius_px =
        f64::from(*VIRTUAL_TILE_POLISH_SEARCH_PASSES.last().unwrap_or(&0));
    diagnostic.failure_stage = "polished_fit_failed".to_string();

    let (left_native, right_native) = native_dimensions;
    let mut measured = hypotheses
        .into_iter()
        .filter_map(|hypothesis| {
            let native = normalized_homography(&(to_native * hypothesis.homography * from_native))?;
            let points = hypothesis
                .inliers
                .iter()
                .filter_map(|&index| {
                    let (source, target) = hypothesis.correspondences[index];
                    transformed_point(&to_native, source).zip(transformed_point(&to_native, target))
                })
                .collect::<Vec<_>>();
            (points.len() >= STATION_RELATION_MIN_INLIERS).then_some((
                hypothesis.model,
                native,
                points,
            ))
        })
        .collect::<Vec<_>>();
    measured.sort_by(|left, right| left.0.cmp(&right.0));
    let mut deduplicated = Vec::<(
        VirtualTilePolishModel,
        Matrix3<f64>,
        Vec<(Point2<f64>, Point2<f64>)>,
    )>::new();
    for hypothesis in measured {
        if deduplicated.iter().any(|existing| {
            virtual_tile_hypotheses_are_equivalent(&existing.1, &hypothesis.1, left_native)
        }) {
            continue;
        }
        deduplicated.push(hypothesis);
    }
    if deduplicated.is_empty() {
        return Vec::new();
    }

    let overlap = StationOverlapMeasure::Covered {
        source: left_coverage,
        target: right_coverage,
        factor,
    };
    let representative = deduplicated
        .iter()
        .max_by(|left, right| {
            left.2
                .len()
                .cmp(&right.2.len())
                .then_with(|| right.0.cmp(&left.0))
        })
        .expect("a non-empty hypothesis set has a representative");
    diagnostic.model_fitted = true;
    diagnostic.residual_model = "polished_multi_hypothesis".to_string();
    diagnostic.fitted_inliers = representative.2.len();
    diagnostic.hull_support = station_relation_spatial_support_with(
        &representative.2,
        &representative.1,
        left_native,
        right_native,
        overlap,
    );
    diagnostic.residual_magnitude_px =
        residual_distribution(match representative.1.try_inverse() {
            Some(inverse) => representative
                .2
                .iter()
                .map(|(source, target)| {
                    symmetric_point_error(&representative.1, &inverse, *source, *target)
                })
                .collect(),
            None => Vec::new(),
        });
    diagnostic.failure_stage = "fitted".to_string();

    deduplicated
        .into_iter()
        .map(|(model, homography, points)| {
            let _model_name = model.name();
            MatchInfo {
                homography,
                inliers: points.len(),
                sequence_bridge: false,
                coarse_bridge: false,
                points,
                candidate_points: Vec::new(),
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
                canonical_homography: None,
                native_verified_points: Vec::new(),
            }
        })
        .collect()
}

fn normalized_homography(transform: &Matrix3<f64>) -> Option<Matrix3<f64>> {
    let scale = transform[(2, 2)];
    if !scale.is_finite() || scale.abs() <= f64::EPSILON {
        return None;
    }
    let normalized = transform / scale;
    normalized
        .iter()
        .all(|value| value.is_finite())
        .then_some(normalized)
}

/// Rendering-prior relation `left tile -> right tile` in native tile pixels.
fn virtual_tile_prior_relation(
    left: &ImageInfo,
    right: &ImageInfo,
    tile_to_world: &HashMap<usize, Matrix3<f64>>,
) -> Option<Matrix3<f64>> {
    let left_to_world = tile_to_world.get(&left.id)?;
    let world_to_right = tile_to_world.get(&right.id)?.try_inverse()?;
    normalized_homography(&(world_to_right * left_to_world))
}

/// Lift independently measured Source_RAW relations into the two fused
/// station tiles' local pixel coordinates.  These transforms are search seeds
/// only: the Virtual_Tile pyramid still measures bidirectionally and the
/// shared Station_Relation selector remains the sole acceptance authority.
///
/// A source match is stored in `(min source id, max source id)` direction.  The
/// composition below maps a left-tile pixel to the corresponding right-tile
/// pixel through the measured source relation and the current source/world
/// poses.  Keeping this conversion explicit avoids treating a world-space
/// station prior as if it were already a local tile transform when a station
/// crop or pose has moved.
fn virtual_tile_source_match_proposals(
    left_group: &[&ImageInfo],
    right_group: &[&ImageInfo],
    pairwise_matches: &HashMap<(usize, usize), MatchInfo>,
    source_to_world: &HashMap<usize, Matrix3<f64>>,
    left_tile_to_world: &Matrix3<f64>,
    right_tile_to_world: &Matrix3<f64>,
    source_render_scale: f64,
) -> Vec<VirtualTileSourceMatchProposal> {
    if !source_render_scale.is_finite() || source_render_scale <= 0.0 {
        return Vec::new();
    }
    let Some(right_world_to_tile) = right_tile_to_world.try_inverse() else {
        return Vec::new();
    };
    // `pairwise_matches` are measured in full-resolution Source_RAW pixels,
    // while `source_to_world` is built from `scaled_source_render_homographies`
    // and therefore consumes render-scaled source pixels. Move the measured
    // relation into that same coordinate system before composing it with the
    // station/world poses. At 1:1 this is the identity.
    let source_scale = Matrix3::new(
        source_render_scale,
        0.0,
        0.0,
        0.0,
        source_render_scale,
        0.0,
        0.0,
        0.0,
        1.0,
    );
    let Some(source_scale_inverse) = source_scale.try_inverse() else {
        return Vec::new();
    };
    let mut proposals = Vec::new();
    for left in left_group {
        let Some(left_world_to_source) = source_to_world
            .get(&left.id)
            .and_then(|matrix| matrix.try_inverse())
        else {
            continue;
        };
        for right in right_group {
            let Some(right_world) = source_to_world.get(&right.id) else {
                continue;
            };
            let key = (left.id.min(right.id), left.id.max(right.id));
            let Some(measurement) = pairwise_matches.get(&key) else {
                continue;
            };
            // Filename/sequence and an ordinary coarse retrieval edge are
            // ordering or basin hints, not independent pixel geometry. A
            // mixed-focal coarse edge with top-candidate consensus is the one
            // exception: it already carries multiple image-derived candidate
            // correspondences, so it may seed a search basin. The downstream
            // Virtual_Tile measurement and selector still gate the result.
            if measurement.sequence_bridge
                || (measurement.coarse_bridge && measurement.top_candidate_points.is_empty())
            {
                continue;
            }
            let Some(source_to_target) = (if left.id <= right.id {
                Some(measurement.homography)
            } else {
                measurement.homography.try_inverse()
            }) else {
                continue;
            };
            let source_to_target = source_scale * source_to_target * source_scale_inverse;
            let Some(left_to_right) = normalized_homography(
                &(right_world_to_tile
                    * *right_world
                    * source_to_target
                    * left_world_to_source
                    * *left_tile_to_world),
            ) else {
                continue;
            };
            if left_to_right.try_inverse().is_none() {
                continue;
            }
            proposals.push(VirtualTileSourceMatchProposal {
                left_to_right,
                inliers: measurement.inliers.max(measurement.points.len()),
                left_source: left.id,
                right_source: right.id,
            });
        }
    }
    proposals.sort_by(|left, right| {
        right
            .inliers
            .cmp(&left.inliers)
            .then_with(|| left.left_source.cmp(&right.left_source))
            .then_with(|| left.right_source.cmp(&right.right_source))
    });
    proposals.truncate(VIRTUAL_TILE_SOURCE_MATCH_MAX_SEEDS);
    proposals
}

/// Prior-free covered-feature relation between two finest planes, in native
/// tile pixels, with its RANSAC inlier count. It starts as a proposal; only a
/// subsequent independently polished hypothesis that passes the complete
/// shared Station_Relation gate may become authoritative.
fn virtual_tile_prior_free_relation(
    left_factor: u32,
    left_features: &[Feature],
    right_factor: u32,
    right_features: &[Feature],
    left_native_dimensions: (u32, u32),
) -> Option<(Matrix3<f64>, usize)> {
    let matches = processing::match_features(left_features, right_features);
    let correspondences = matches
        .iter()
        .map(|matched| {
            let left = left_features[matched.index1].keypoint;
            let right = right_features[matched.index2].keypoint;
            (
                Point2::new(f64::from(left.x), f64::from(left.y)),
                Point2::new(f64::from(right.x), f64::from(right.y)),
            )
        })
        .collect::<Vec<_>>();
    let (homography, inliers) = processing::find_homography_ransac_points_stable_with_min_inliers(
        &correspondences,
        5.0,
        STATION_RELATION_MIN_INLIERS,
    )?;
    if inliers.len() < STATION_RELATION_MIN_INLIERS {
        return None;
    }
    let native = normalized_homography(
        &(VirtualTileMatchPlane::level_to_native(right_factor)
            * homography
            * VirtualTileMatchPlane::native_to_level(left_factor)),
    )?;
    (native.try_inverse().is_some()
        && homography_preserves_focus_orientation(&native, left_native_dimensions))
    .then_some((native, inliers.len()))
}

/// A covered-feature seed can still repair a rendering prior when the
/// subsequent finest-plane polish fitted a model but its inlier hull missed
/// the fixed spatial-support gate. The seed remains a proposal only: the
/// repaired station is rendered again and the ordinary bounded matcher must
/// accept the resulting relation before it enters the pose solver.
fn virtual_tile_prior_free_seed_fallback_conditions(
    model_fitted: bool,
    failure_stage: &str,
    fitted_inliers: usize,
    prior_free_inliers: usize,
) -> bool {
    model_fitted
        && failure_stage == "insufficient_polished_support"
        && fitted_inliers >= STATION_RELATION_MIN_INLIERS
        // A covered-feature seed is only a repair proposal when it carries at
        // least as much support as the model that the finest-plane polish
        // already fitted.  A smaller seed can move an entire station cluster
        // on a repeated texture even when the polished model has hundreds of
        // mutually consistent matches but fails only its spatial-support gate.
        && prior_free_inliers >= fitted_inliers
}

fn virtual_tile_prior_free_seed_fallback_allowed(
    diagnostic: &stack_report::StationRelationCandidateRecord,
) -> bool {
    virtual_tile_prior_free_seed_fallback_conditions(
        diagnostic.model_fitted,
        &diagnostic.failure_stage,
        diagnostic.fitted_inliers,
        diagnostic.prior_free_inliers,
    )
}

fn log_virtual_tile_candidate(
    diagnostic: &stack_report::StationRelationCandidateRecord,
    accepted: bool,
) {
    println!(
        "  - Virtual_Tile candidate {}->{}: seed={} level=1/{} of {:?} accepted={} prior_free_inliers={} repaired={} grid_probes={} measurable={} bidirectional={} cells={} radius={:.1}px residual_mag=[{:.2},{:.2},{:.2}] translation={}/{:.4}/{:?}/{} affine={}/{:.4}/{:?}/{} projective={}/{:.4}/{:?}/{} residual_model={} fit_inliers={} support={:.4} prefit_structural_orientation={} fitted_orientation={} stage={}",
        diagnostic.left,
        diagnostic.right,
        diagnostic.seed,
        diagnostic.measurement_factor,
        diagnostic.pyramid_factors,
        accepted,
        diagnostic.prior_free_inliers,
        diagnostic.after_prior_repair,
        diagnostic.grid_probe_count,
        diagnostic.measurable_patch_count,
        diagnostic.bidirectional_match_count,
        diagnostic.occupied_grid_cells,
        diagnostic.search_radius_px,
        diagnostic.residual_magnitude_px.p10,
        diagnostic.residual_magnitude_px.median,
        diagnostic.residual_magnitude_px.p90,
        diagnostic.residual_translation.candidate_inliers,
        diagnostic.residual_translation.final_support,
        diagnostic.residual_translation.bounded_correction,
        diagnostic.residual_translation.rejection_reason,
        diagnostic.residual_affine.candidate_inliers,
        diagnostic.residual_affine.final_support,
        diagnostic.residual_affine.bounded_correction,
        diagnostic.residual_affine.rejection_reason,
        diagnostic.residual_projective.candidate_inliers,
        diagnostic.residual_projective.final_support,
        diagnostic.residual_projective.bounded_correction,
        diagnostic.residual_projective.rejection_reason,
        diagnostic.residual_model,
        diagnostic.fitted_inliers,
        diagnostic.hull_support,
        diagnostic
            .median_prefit_structural_orientation_difference_degrees
            .map_or_else(|| "n/a".to_string(), |value| format!("{value:.2}deg")),
        diagnostic
            .median_fitted_orientation_difference_degrees
            .map_or_else(|| "n/a".to_string(), |value| format!("{value:.2}deg")),
        diagnostic.failure_stage,
    );
}

type VirtualTileRelationHypotheses = HashMap<(usize, usize), Vec<MatchInfo>>;

fn station_relation_tile_geometry_record(
    tile: &ImageInfo,
    coverage: Option<&stitching::CoverageMask>,
    factor: u32,
    tile_to_world: Option<&Matrix3<f64>>,
) -> stack_report::StationRelationTileGeometryRecord {
    let factor = factor.max(1);
    let effective_bounds = coverage
        .and_then(stitching::CoverageMask::covered_bounds)
        .map(|(x, y, width, height)| {
            let native_x = x.saturating_mul(factor).min(tile.width);
            let native_y = y.saturating_mul(factor).min(tile.height);
            let native_right = x
                .saturating_add(width)
                .saturating_mul(factor)
                .min(tile.width);
            let native_bottom = y
                .saturating_add(height)
                .saturating_mul(factor)
                .min(tile.height);
            [
                native_x,
                native_y,
                native_right.saturating_sub(native_x),
                native_bottom.saturating_sub(native_y),
            ]
        });
    stack_report::StationRelationTileGeometryRecord {
        dimensions: [tile.width, tile.height],
        effective_bounds,
        tile_to_world: tile_to_world.map(matrix3_row_major),
    }
}

fn populate_station_relation_candidate_geometry(
    diagnostic: &mut stack_report::StationRelationCandidateRecord,
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
    tile_to_world: &HashMap<usize, Matrix3<f64>>,
) {
    let factor = coverages.overlap_factor;
    if let Some(tile) = tiles.get(diagnostic.left) {
        diagnostic.left_tile = station_relation_tile_geometry_record(
            tile,
            coverages.overlap.get(diagnostic.left),
            factor,
            tile_to_world.get(&tile.id),
        );
    }
    if let Some(tile) = tiles.get(diagnostic.right) {
        diagnostic.right_tile = station_relation_tile_geometry_record(
            tile,
            coverages.overlap.get(diagnostic.right),
            factor,
            tile_to_world.get(&tile.id),
        );
    }
}

/// Authoritative relations: bounded coarse-to-fine hypotheses around the
/// rendering prior for every topology candidate (or for `only` when given).
/// Returns `(relations, diagnostics)`, diagnostics in candidate order.
fn virtual_tile_measure_prior_relations(
    tiles: &[ImageInfo],
    pyramid: &VirtualTileMatchPyramid,
    tile_to_world: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
    only: Option<&BTreeSet<(usize, usize)>>,
    after_prior_repair: bool,
) -> (
    VirtualTileRelationHypotheses,
    Vec<stack_report::StationRelationCandidateRecord>,
) {
    let mut relations = VirtualTileRelationHypotheses::new();
    let mut diagnostics = Vec::with_capacity(station_topology.candidates.len());
    for candidate in &station_topology.candidates {
        let (left, right) = (candidate.left, candidate.right);
        if only.is_some_and(|pairs| !pairs.contains(&(left.min(right), left.max(right)))) {
            continue;
        }
        let prior = tiles
            .get(left)
            .zip(tiles.get(right))
            .and_then(|(left_tile, right_tile)| {
                virtual_tile_prior_relation(left_tile, right_tile, tile_to_world)
            });
        let (relation, mut diagnostic) = match prior {
            Some(prior) => virtual_tile_pyramid_station_match(
                left,
                right,
                pyramid,
                &prior,
                VIRTUAL_TILE_SEED_RENDERING_PRIOR,
            ),
            None => {
                let (_, mut diagnostic) = virtual_tile_direct_station_match(
                    left,
                    right,
                    &[],
                    &[],
                    &HashMap::new(),
                    STATION_RELATION_MAX_MEDIAN_ERROR_PX,
                );
                diagnostic.failure_stage = "missing_initial_pose".to_string();
                (Vec::new(), diagnostic)
            }
        };
        diagnostic.after_prior_repair = after_prior_repair;
        if let Some(selection) = select_virtual_tile_relation_hypothesis(
            tiles,
            left,
            right,
            &relation,
            pyramid.solver_coverages(),
        ) {
            diagnostic.hypotheses = selection.outcomes;
        }
        populate_station_relation_candidate_geometry(
            &mut diagnostic,
            tiles,
            pyramid.solver_coverages(),
            tile_to_world,
        );
        log_virtual_tile_candidate(&diagnostic, !relation.is_empty());
        if !relation.is_empty() {
            relations.insert((left, right), relation);
        }
        diagnostics.push(diagnostic);
    }
    (relations, diagnostics)
}

/// A densely verified prior-free relation between two stations whose rendering
/// priors disagree with it. It may only repair a prior; it is never a
/// Station_Relation.
#[derive(Clone, Copy, Debug)]
struct VirtualTilePriorProposal {
    left: usize,
    right: usize,
    /// Measured `left tile -> right tile` relation in native tile pixels.
    left_to_right: Matrix3<f64>,
    inliers: usize,
}

/// One cluster member's rendering-prior repair: `world_correction` maps the
/// station's current world placement onto the placement the proposal measured
/// relative to an already consistent station.
#[derive(Clone, Copy, Debug, PartialEq)]
struct VirtualTilePriorRepair {
    station: usize,
    fixed_station: usize,
    relation: (usize, usize),
    relation_inliers: usize,
    world_correction: Matrix3<f64>,
}

/// A rendering-prior repair may move a station only within the same bounded
/// geometry envelope used by closure corrections and the station scale gate.
/// Keep this decision pure so production and property tests share the exact
/// finite-value and inclusive-boundary semantics.
fn virtual_tile_prior_repair_is_bounded(center_shift_px: f64, correction_scale_ratio: f64) -> bool {
    center_shift_px.is_finite()
        && center_shift_px >= 0.0
        && center_shift_px <= CLOSURE_MAX_CORNER_CORRECTION_PX
        && correction_scale_ratio.is_finite()
        && (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
            .contains(&correction_scale_ratio)
}

fn virtual_tile_hypothesis_model_rank(transform: &Matrix3<f64>) -> u8 {
    let Some(transform) = normalized_homography(transform) else {
        return u8::MAX;
    };
    if transform[(2, 0)].abs() > 1e-12 || transform[(2, 1)].abs() > 1e-12 {
        return 2;
    }
    let first = nalgebra::Vector2::new(transform[(0, 0)], transform[(1, 0)]);
    let second = nalgebra::Vector2::new(transform[(0, 1)], transform[(1, 1)]);
    let scale = first.norm().max(second.norm()).max(f64::EPSILON);
    if first.dot(&second).abs() <= 1e-10 * scale * scale
        && (first.norm() - second.norm()).abs() <= 1e-10 * scale
    {
        0
    } else {
        1
    }
}

fn virtual_tile_hypothesis_matrix_key(transform: &Matrix3<f64>) -> [u64; 9] {
    let mut key = [0; 9];
    for (slot, value) in key.iter_mut().zip(transform.iter()) {
        *slot = value.to_bits();
    }
    key
}

#[derive(Clone)]
struct MeasuredVirtualTileHypothesis {
    relation: MatchInfo,
    canonical_source: usize,
    canonical_target: usize,
    canonical_homography: Matrix3<f64>,
    correspondence_count: usize,
    correspondence_sample: Vec<stack_report::StationRelationCorrespondenceRecord>,
    holdout_fit_count: usize,
    holdout_count: usize,
    holdout_median_error_px: Option<f64>,
    holdout_spatial_support: Option<f64>,
    holdout_transform: Option<Matrix3<f64>>,
    measured: MeasuredStationRelationCandidate,
    rejection_count: usize,
    model_rank: u8,
}

#[derive(Clone)]
struct VirtualTileHypothesisSelection {
    relation: MatchInfo,
    passes: bool,
    outcomes: Vec<stack_report::StationRelationHypothesisRecord>,
}

/// Compare the independent checkerboard holdout before the in-sample quality
/// score.  A holdout is usable only when it has the same minimum support and
/// spatial coverage required by the Station_Relation gate; otherwise it must
/// not make an otherwise measurable hypothesis win by reporting a synthetic
/// zero/`None` value.  The gate rejection count remains the first key, so this
/// is a deterministic choice among hypotheses with the same acceptance status
/// and never relaxes a Station_Relation threshold.
fn virtual_tile_hypothesis_holdout_key(
    hypothesis: &MeasuredVirtualTileHypothesis,
) -> (bool, f64, f64) {
    let usable = hypothesis.holdout_fit_count >= STATION_RELATION_MIN_INLIERS
        && hypothesis.holdout_count >= STATION_RELATION_MIN_INLIERS
        && hypothesis.holdout_median_error_px.is_some_and(|error| {
            error.is_finite() && error <= STATION_RELATION_MAX_MEDIAN_ERROR_PX
        })
        && hypothesis.holdout_spatial_support.is_some_and(|support| {
            support.is_finite() && support >= STATION_RELATION_MIN_SPATIAL_SUPPORT
        });
    if usable {
        (
            true,
            hypothesis
                .holdout_median_error_px
                .expect("usable holdout has a median error"),
            hypothesis
                .holdout_spatial_support
                .expect("usable holdout has spatial support"),
        )
    } else {
        (false, f64::INFINITY, f64::NEG_INFINITY)
    }
}

#[allow(clippy::type_complexity)]
fn virtual_tile_hypothesis_holdout(
    model_rank: u8,
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    overlap_measure: Option<StationOverlapMeasure<'_>>,
) -> (usize, usize, Option<f64>, Option<f64>, Option<Matrix3<f64>>) {
    let cell = f64::from(source_dimensions.0.max(source_dimensions.1).max(1)) / 12.0;
    let mut fit = Vec::new();
    let mut holdout = Vec::new();
    for &point in points {
        let column = (point.0.x / cell).floor() as i64;
        let row = (point.0.y / cell).floor() as i64;
        if (column + row).rem_euclid(2) == 0 {
            fit.push(point);
        } else {
            holdout.push(point);
        }
    }
    let model = match model_rank {
        0 => VirtualTilePolishModel::Similarity,
        1 => VirtualTilePolishModel::Affine,
        2 => VirtualTilePolishModel::Projective,
        _ => return (fit.len(), holdout.len(), None, None, None),
    };
    let transform = model
        .fit(
            &fit,
            VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX,
            source_dimensions,
        )
        .map(|(transform, _)| transform);
    let median_error_px = transform
        .as_ref()
        .map(|transform| median_symmetric_error(transform, &holdout))
        .filter(|error| error.is_finite());
    let spatial_support = transform.as_ref().and_then(|transform| {
        overlap_measure.map(|measure| {
            station_relation_spatial_support_with(
                &holdout,
                transform,
                source_dimensions,
                target_dimensions,
                measure,
            )
        })
    });
    (
        fit.len(),
        holdout.len(),
        median_error_px,
        spatial_support,
        transform,
    )
}

/// Measure, canonicalize, deduplicate, and rank one pair's independent model
/// hypotheses. Fully passing hypotheses always outrank failing hypotheses;
/// ties use the complete deterministic quality tuple and then intrinsic model
/// complexity/matrix bits, never input order. Both provisional clustering and
/// the final solver call this exact selector.
fn select_virtual_tile_relation_hypothesis(
    tiles: &[ImageInfo],
    left: usize,
    right: usize,
    hypotheses: &[MatchInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> Option<VirtualTileHypothesisSelection> {
    let mut measured = Vec::<MeasuredVirtualTileHypothesis>::new();
    for relation in hypotheses {
        if left == right
            || relation.sequence_bridge
            || relation.points.len() < FOCUS_MODEL_MIN_INLIERS
        {
            continue;
        }
        let canonical = match canonical_relation_view(tiles, left, right, relation) {
            Ok(view) => view,
            Err(CanonicalRelationViewError::NonInvertible) => continue,
            Err(CanonicalRelationViewError::IndexOutOfBounds {
                left,
                right,
                image_count,
            }) => {
                panic!("canonical relation pair ({left}, {right}) is outside {image_count} tiles")
            }
        };
        let points: Cow<'_, [(Point2<f64>, Point2<f64>)]> = if canonical.inverted {
            Cow::Owned(
                relation
                    .points
                    .iter()
                    .map(|&(source, target)| (target, source))
                    .collect(),
            )
        } else {
            Cow::Borrowed(&relation.points)
        };
        let quality_coverages = coverages
            .quality
            .get(canonical.source_index)
            .zip(coverages.quality.get(canonical.target_index));
        let dense_measurements = focus_overlap_quality_measurements_on_coverage(
            &tiles[canonical.source_index],
            &tiles[canonical.target_index],
            &canonical.homography,
            quality_coverages,
        );
        let overlap_measure =
            coverages.overlap_measure(canonical.source_index, canonical.target_index);
        let Some(candidate) = measure_station_relation_candidate(
            relation.inliers.max(points.len()),
            &canonical.homography,
            &points,
            tiles[canonical.source_index].dimensions(),
            tiles[canonical.target_index].dimensions(),
            overlap_measure,
            dense_measurements,
            canonical.inverted,
        ) else {
            continue;
        };
        let correspondence_count = points.len();
        let sample_count = correspondence_count.min(64);
        let correspondence_sample = (0..sample_count)
            .map(|sample| {
                let index = if sample_count <= 1 {
                    0
                } else {
                    sample * (correspondence_count - 1) / (sample_count - 1)
                };
                let (source, target) = points[index];
                let local_scale = virtual_tile_local_jacobian(&canonical.homography, source)
                    .map(|jacobian| jacobian.determinant().abs().sqrt());
                stack_report::StationRelationCorrespondenceRecord {
                    source: [source.x, source.y],
                    target: [target.x, target.y],
                    local_scale,
                }
            })
            .collect();
        let (
            holdout_fit_count,
            holdout_count,
            holdout_median_error_px,
            holdout_spatial_support,
            holdout_transform,
        ) = virtual_tile_hypothesis_holdout(
            virtual_tile_hypothesis_model_rank(&canonical.homography),
            &points,
            tiles[canonical.source_index].dimensions(),
            tiles[canonical.target_index].dimensions(),
            overlap_measure,
        );
        measured.push(MeasuredVirtualTileHypothesis {
            relation: relation.clone(),
            canonical_source: canonical.source_index,
            canonical_target: canonical.target_index,
            canonical_homography: canonical.homography,
            correspondence_count,
            correspondence_sample,
            holdout_fit_count,
            holdout_count,
            holdout_median_error_px,
            holdout_spatial_support,
            holdout_transform,
            measured: candidate,
            rejection_count: station_relation_rejection_reasons(&candidate.measurements).len(),
            model_rank: virtual_tile_hypothesis_model_rank(&canonical.homography),
        });
    }
    measured.sort_by(|left, right| {
        left.rejection_count
            .cmp(&right.rejection_count)
            .then_with(|| {
                let left_holdout = virtual_tile_hypothesis_holdout_key(left);
                let right_holdout = virtual_tile_hypothesis_holdout_key(right);
                right_holdout
                    .0
                    .cmp(&left_holdout.0)
                    .then_with(|| left_holdout.1.total_cmp(&right_holdout.1))
                    .then_with(|| right_holdout.2.total_cmp(&left_holdout.2))
            })
            .then_with(|| right.measured.quality.total_cmp(&left.measured.quality))
            .then_with(|| {
                right
                    .measured
                    .measurements
                    .spatial_support
                    .total_cmp(&left.measured.measurements.spatial_support)
            })
            .then_with(|| {
                right
                    .measured
                    .measurements
                    .inliers
                    .cmp(&left.measured.measurements.inliers)
            })
            .then_with(|| {
                left.measured
                    .measurements
                    .median_error_px
                    .total_cmp(&right.measured.measurements.median_error_px)
            })
            .then_with(|| left.model_rank.cmp(&right.model_rank))
            .then_with(|| {
                virtual_tile_hypothesis_matrix_key(&left.canonical_homography).cmp(
                    &virtual_tile_hypothesis_matrix_key(&right.canonical_homography),
                )
            })
    });
    let mut deduplicated = Vec::<MeasuredVirtualTileHypothesis>::new();
    for hypothesis in measured {
        if deduplicated.iter().any(|existing| {
            virtual_tile_hypotheses_are_equivalent(
                &existing.canonical_homography,
                &hypothesis.canonical_homography,
                tiles[left].dimensions(),
            )
        }) {
            continue;
        }
        deduplicated.push(hypothesis);
    }
    let outcomes = deduplicated
        .iter()
        .enumerate()
        .map(|(index, hypothesis)| {
            let measurements = hypothesis.measured.measurements;
            stack_report::StationRelationHypothesisRecord {
                model: match hypothesis.model_rank {
                    0 => "similarity",
                    1 => "affine",
                    2 => "projective",
                    _ => "unknown",
                }
                .to_string(),
                selected: index == 0,
                inliers: measurements.inliers,
                median_error_px: measurements.median_error_px,
                scale_ratio: measurements.scale_ratio,
                spatial_support: measurements.spatial_support,
                low_frequency_mean_relative_difference: measurements
                    .low_frequency_mean_relative_difference,
                edge_strength_ratio: measurements.edge_strength_ratio,
                median_edge_orientation_difference_degrees: measurements
                    .median_edge_orientation_difference_degrees,
                preserves_convex_orientation: measurements.preserves_convex_orientation,
                rejection_reasons: station_relation_rejection_reasons(&measurements)
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                canonical_source: hypothesis.canonical_source,
                canonical_target: hypothesis.canonical_target,
                source_to_target_transform: matrix3_row_major(&hypothesis.canonical_homography),
                correspondence_count: hypothesis.correspondence_count,
                correspondence_sample: hypothesis.correspondence_sample.clone(),
                holdout_fit_count: hypothesis.holdout_fit_count,
                holdout_count: hypothesis.holdout_count,
                holdout_median_error_px: hypothesis.holdout_median_error_px,
                holdout_spatial_support: hypothesis.holdout_spatial_support,
                holdout_transform: hypothesis.holdout_transform.as_ref().map(matrix3_row_major),
            }
        })
        .collect::<Vec<_>>();
    deduplicated
        .first()
        .map(|hypothesis| VirtualTileHypothesisSelection {
            relation: hypothesis.relation.clone(),
            passes: hypothesis.rejection_count == 0,
            outcomes,
        })
}

fn select_virtual_tile_relation_hypotheses(
    tiles: &[ImageInfo],
    relations: &VirtualTileRelationHypotheses,
    coverages: VirtualTileSolverCoverages<'_>,
) -> HashMap<(usize, usize), MatchInfo> {
    let mut selected = HashMap::new();
    let mut pairs = relations.iter().collect::<Vec<_>>();
    pairs.sort_by_key(|(pair, _)| **pair);
    for (&pair, hypotheses) in pairs {
        if let Some(selection) =
            select_virtual_tile_relation_hypothesis(tiles, pair.0, pair.1, hypotheses, coverages)
        {
            selected.insert(pair, selection.relation);
        } else if let Some(relation) = hypotheses.first() {
            // Preserve one singular measured hypothesis so the final solver can
            // report the pair once as a canonical inversion failure.
            selected.insert(pair, relation.clone());
        }
    }
    selected
}

/// Apply the complete canonical Station_Relation measurement and rejection
/// gate before a prior-seeded Virtual_Tile relation may merge repair clusters.
/// This is intentionally side-effect free: final report counts remain owned by
/// the solver, so provisional filtering cannot double-count a rejection.
fn virtual_tile_provisional_authoritative_candidate(
    tiles: &[ImageInfo],
    left: usize,
    right: usize,
    hypotheses: &[MatchInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> bool {
    select_virtual_tile_relation_hypothesis(tiles, left, right, hypotheses, coverages)
        .is_some_and(|selection| selection.passes)
}

fn virtual_tile_relation_measurements(
    left: usize,
    right: usize,
    relation: &MatchInfo,
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> Option<StationRelationMeasurements> {
    if left == right || relation.sequence_bridge || relation.points.len() < FOCUS_MODEL_MIN_INLIERS
    {
        return None;
    }
    let canonical = canonical_relation_view(tiles, left, right, relation).ok()?;
    let points: Cow<'_, [(Point2<f64>, Point2<f64>)]> = if canonical.inverted {
        Cow::Owned(
            relation
                .points
                .iter()
                .map(|&(source, target)| (target, source))
                .collect(),
        )
    } else {
        Cow::Borrowed(&relation.points)
    };
    let quality_coverages = coverages
        .quality
        .get(canonical.source_index)
        .zip(coverages.quality.get(canonical.target_index));
    let dense_measurements = focus_overlap_quality_measurements_on_coverage(
        &tiles[canonical.source_index],
        &tiles[canonical.target_index],
        &canonical.homography,
        quality_coverages,
    );
    let overlap_measure = coverages.overlap_measure(canonical.source_index, canonical.target_index);
    let candidate = measure_station_relation_candidate(
        relation.inliers.max(points.len()),
        &canonical.homography,
        &points,
        tiles[canonical.source_index].dimensions(),
        tiles[canonical.target_index].dimensions(),
        overlap_measure,
        dense_measurements,
        canonical.inverted,
    )?;
    Some(candidate.measurements)
}

/// Geometry keeps all of its original rejection gates. Radiometry may fit an
/// exposure difference, but only on independently verified native image pairs
/// whose geometry, support and edge measurements passed those same gates.
fn native_photometric_measurements_are_usable(measurements: &StationRelationMeasurements) -> bool {
    measurements
        .low_frequency_mean_relative_difference
        .is_finite()
        && station_relation_rejection_reasons(measurements)
            .iter()
            .all(|&reason| reason == degradation::STATION_RELATION_PHOTOMETRIC_MISMATCH)
}

fn native_photometric_pairs(
    relations: &HashMap<(usize, usize), MatchInfo>,
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> Vec<(usize, usize)> {
    let mut pairs = relations
        .iter()
        .filter_map(|(&(left, right), relation)| {
            if relation.sequence_bridge
                || relation.native_verified_points.len() < STATION_RELATION_MIN_INLIERS
            {
                return None;
            }
            let measurements =
                virtual_tile_relation_measurements(left, right, relation, tiles, coverages)?;
            native_photometric_measurements_are_usable(&measurements).then_some((left, right))
        })
        .collect::<Vec<_>>();
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

fn virtual_tile_authoritative_pairs(
    relations: &VirtualTileRelationHypotheses,
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
) -> Vec<(usize, usize)> {
    let mut pairs = relations
        .iter()
        .filter_map(|(&(left, right), hypotheses)| {
            virtual_tile_provisional_authoritative_candidate(
                tiles, left, right, hypotheses, coverages,
            )
            .then_some((left, right))
        })
        .collect::<Vec<_>>();
    pairs.sort_unstable();
    pairs
}

/// Topology pairs that remain eligible for an actual prior-free proposal after
/// authoritative relations form their prior-consistent clusters.
fn virtual_tile_prior_free_candidate_pairs(
    station_count: usize,
    authoritative_pairs: &[(usize, usize)],
    station_topology: &topology::StationTopology,
) -> Vec<(usize, usize)> {
    let mut clusters = Dsu::new(station_count);
    for &(left, right) in authoritative_pairs {
        if left < station_count && right < station_count && left != right {
            clusters.union(left, right);
        }
    }
    station_topology
        .candidates
        .iter()
        .filter_map(|candidate| {
            let (left, right) = (candidate.left, candidate.right);
            (left < station_count
                && right < station_count
                && clusters.find(left) != clusters.find(right))
            .then_some((left, right))
        })
        .collect()
}

/// Plan rendering-prior repairs from prior-free proposals.
///
/// Stations joined by authoritative (prior-seeded) relations form clusters
/// whose priors agree. The largest cluster is the reference (ties: smallest
/// canonical row/column key). Each proposal joining a fixed cluster to an
/// unfixed one — strongest first, ties by canonical pair key — moves the whole
/// unfixed cluster so that the proposal's measured relation holds; a cluster is
/// repaired at most once and a station therefore re-rendered at most once.
/// Stations no proposal reaches keep their prior (and stay disconnected).
fn plan_virtual_tile_prior_repairs(
    station_count: usize,
    authoritative_pairs: &[(usize, usize)],
    proposals: &[VirtualTilePriorProposal],
    tile_to_world: &[Matrix3<f64>],
    station_topology: &topology::StationTopology,
) -> Vec<VirtualTilePriorRepair> {
    if station_count < 2 || tile_to_world.len() != station_count {
        return Vec::new();
    }
    let mut clusters = Dsu::new(station_count);
    for &(left, right) in authoritative_pairs {
        if left < station_count && right < station_count && left != right {
            clusters.union(left, right);
        }
    }
    let mut members = BTreeMap::<usize, Vec<usize>>::new();
    for station in 0..station_count {
        members
            .entry(clusters.find(station))
            .or_default()
            .push(station);
    }
    if members.len() < 2 {
        return Vec::new();
    }
    let cluster_key = |stations: &[usize]| {
        stations
            .iter()
            .map(|&station| station_tree_position_key(station_topology, station))
            .min()
    };
    let Some(reference) = members
        .iter()
        .max_by(|(_, left), (_, right)| {
            left.len()
                .cmp(&right.len())
                .then_with(|| cluster_key(right).cmp(&cluster_key(left)))
        })
        .map(|(&root, _)| root)
    else {
        return Vec::new();
    };
    let mut corrections = BTreeMap::from([(reference, Matrix3::identity())]);
    let mut remaining = proposals
        .iter()
        .copied()
        .filter(|proposal| {
            proposal.left < station_count
                && proposal.right < station_count
                && proposal.left != proposal.right
        })
        .collect::<Vec<_>>();
    let mut repairs = Vec::new();
    loop {
        let next = remaining
            .iter()
            .enumerate()
            .filter(|(_, proposal)| {
                corrections.contains_key(&clusters.find(proposal.left))
                    != corrections.contains_key(&clusters.find(proposal.right))
            })
            .max_by(|(_, left), (_, right)| {
                left.inliers.cmp(&right.inliers).then_with(|| {
                    station_tree_edge_key(station_topology, right.left, right.right).cmp(
                        &station_tree_edge_key(station_topology, left.left, left.right),
                    )
                })
            })
            .map(|(index, _)| index);
        let Some(index) = next else {
            break;
        };
        let proposal = remaining.remove(index);
        let left_root = clusters.find(proposal.left);
        let right_root = clusters.find(proposal.right);
        // tile pixel q of the unfixed station lands where the fixed station's
        // (already corrected) placement puts the measured corresponding pixel.
        let (fixed, unfixed, unfixed_root, fixed_to_unfixed_tile) =
            if corrections.contains_key(&left_root) {
                (
                    proposal.left,
                    proposal.right,
                    right_root,
                    proposal.left_to_right.try_inverse(),
                )
            } else {
                (
                    proposal.right,
                    proposal.left,
                    left_root,
                    Some(proposal.left_to_right),
                )
            };
        let Some(unfixed_to_fixed) = fixed_to_unfixed_tile else {
            continue;
        };
        let fixed_correction = corrections[&clusters.find(fixed)];
        let Some(unfixed_world_to_tile) = tile_to_world[unfixed].try_inverse() else {
            continue;
        };
        let Some(correction) = normalized_homography(
            &(fixed_correction * tile_to_world[fixed] * unfixed_to_fixed * unfixed_world_to_tile),
        ) else {
            continue;
        };
        if correction.try_inverse().is_none()
            || !virtual_tile_prior_repair_is_bounded(
                0.0,
                homography_scale_ratio(&correction).unwrap_or(f64::NAN),
            )
        {
            // An inadmissible strongest proposal must not mark this cluster as
            // repaired and suppress a weaker proposal that stays inside the
            // unchanged repair envelope. The render-time check below still
            // measures the actual centre shift before mutating source poses.
            continue;
        }
        corrections.insert(unfixed_root, correction);
        for &station in &members[&unfixed_root] {
            repairs.push(VirtualTilePriorRepair {
                station,
                fixed_station: fixed,
                relation: (proposal.left, proposal.right),
                relation_inliers: proposal.inliers,
                world_correction: correction,
            });
        }
    }
    repairs.sort_by_key(|repair| repair.station);
    repairs
}

/// Render one Capture_Station for station matching only and return its covered
/// luminance plane and Coverage_Mask; the fused colour pixels are dropped.
#[allow(clippy::too_many_arguments)]
fn render_virtual_tile_match_plane<R: Runtime, F>(
    group: &[&ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    station_index: usize,
    expected: stitching::FocusVirtualTileGeometry,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<(GrayImage, stitching::CoverageMask), String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let rendered = stitching::focus_stack_stitcher_unfilled(
        group,
        homographies,
        projection,
        focus_warp,
        None,
        station_index,
        false,
        app_handle,
        progress_event,
        load_image,
    )?;
    let rendered_dimensions = rendered.image.dimensions();
    if rendered_dimensions != (expected.width, expected.height) {
        return Err(format!(
            "Virtual tile station {station_index} geometry/render mismatch: expected {}x{} at world origin ({:.3},{:.3}), rendered {}x{}",
            expected.width,
            expected.height,
            expected.tile_to_world[(0, 2)],
            expected.tile_to_world[(1, 2)],
            rendered_dimensions.0,
            rendered_dimensions.1,
        ));
    }
    let coverage = match rendered.masks.as_ref() {
        Some(masks) => masks.coverage.clone(),
        None => stitching::CoverageMask::from_bytes(
            expected.width,
            expected.height,
            vec![0; expected.width as usize * expected.height as usize],
        )?,
    };
    let gray = virtual_tile_covered_luminance(&rendered.image, &coverage);
    Ok((gray, coverage))
}

/// Install one station's probe planes. The tile keeps the structural
/// relation-quality plane as its analysis image (`scale_factor` = that plane's
/// factor, the source `ImageInfo` convention) and independent covered features.
/// Small preview probe grids must not downsample feature descriptors twice.
/// The feature factor is explicit; nothing reads keypoints in quality-plane
/// coordinates or promotes a feature fit to solver evidence.
fn install_virtual_tile_match_planes(
    tile: &mut ImageInfo,
    pyramid: &mut VirtualTileMatchPyramid,
    station: usize,
    gray: &GrayImage,
    coverage: &stitching::CoverageMask,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
) -> Result<(), String> {
    let planes = pyramid
        .set_station(station, tile, gray, coverage)
        .ok_or_else(|| format!("Virtual tile station {station} has no usable matching plane"))?;
    let finest = planes
        .last()
        .expect("set_station returns at least one plane");
    let feature_factor = finest
        .factor
        .min((gray.width().max(gray.height()) / 1200).max(1));
    let feature_plane = VirtualTileMatchPlane::from_native(gray, coverage, feature_factor)
        .ok_or_else(|| format!("Virtual tile station {station} has no feature proposal plane"))?;
    tile.features = virtual_tile_plane_features(&feature_plane, brief_pairs);
    pyramid.feature_factors[station] = feature_factor;
    let (quality, factor) = virtual_tile_relation_quality_plane(&planes, gray, coverage);
    #[cfg(test)]
    if let Some(directory) = std::env::var_os("RAW_EDITOR_VIRTUAL_TILE_DIAGNOSTICS_DIR") {
        let directory = PathBuf::from(directory);
        std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        gray.save(directory.join(format!("station-{station:03}.gray.png")))
            .map_err(|error| error.to_string())?;
        quality
            .save(directory.join(format!("station-{station:03}.quality.png")))
            .map_err(|error| error.to_string())?;
        let (w, h) = coverage.dimensions();
        GrayImage::from_raw(w, h, coverage.covered().to_vec())
            .ok_or("invalid diagnostic coverage")?
            .save(directory.join(format!("station-{station:03}.coverage.png")))
            .map_err(|error| error.to_string())?;
    }
    let quality_coverage = planes
        .iter()
        .find(|plane| plane.factor == factor)
        .map(|plane| plane.coverage.clone())
        .or_else(|| {
            VirtualTileMatchPlane::from_native(gray, coverage, factor).map(|plane| plane.coverage)
        })
        .ok_or_else(|| format!("Virtual tile station {station} has no relation-quality plane"))?;
    pyramid.quality_coverages[station] = quality_coverage;
    tile.scale_factor = f64::from(factor);
    tile.alignment_image = quality;
    Ok(())
}

/// Move one station's rendering prior: every member source pose (and its
/// focus-band poses) is composed with `world_correction`. Returns false, and
/// changes nothing, when a corrected pose would not be a valid planar pose.
fn apply_virtual_tile_prior_repair(
    group: &[&ImageInfo],
    world_correction: &Matrix3<f64>,
    homographies: &mut HashMap<usize, Matrix3<f64>>,
    focus_warp: Option<&mut FocusLayerWarp>,
) -> bool {
    let mut corrected = Vec::with_capacity(group.len());
    for image in group {
        let Some(pose) = homographies.get(&image.id) else {
            return false;
        };
        let Some(pose) = normalized_homography(&(world_correction * pose)) else {
            return false;
        };
        if pose.try_inverse().is_none()
            || !homography_preserves_focus_orientation(&pose, image.dimensions())
        {
            return false;
        }
        corrected.push((image.id, pose));
    }
    for (id, pose) in corrected {
        homographies.insert(id, pose);
    }
    if let Some(warp) = focus_warp {
        for band in &mut warp.bands {
            for image in group {
                if let Some(pose) = band.homographies.get_mut(&image.id)
                    && let Some(corrected) = normalized_homography(&(world_correction * *pose))
                {
                    *pose = corrected;
                }
            }
        }
    }
    true
}

fn replace_virtual_tile_candidate_record(
    records: &mut Vec<stack_report::StationRelationCandidateRecord>,
    record: stack_report::StationRelationCandidateRecord,
) {
    let key = (record.left.min(record.right), record.left.max(record.right));
    match records.iter_mut().find(|existing| {
        (
            existing.left.min(existing.right),
            existing.left.max(existing.right),
        ) == key
    }) {
        Some(existing) => *existing = record,
        None => records.push(record),
    }
}

/// Fit station relations directly from focus-fused Virtual_Tiles. The topology
/// controls deterministic attempt order and budget. Initial poses select the
/// jointly covered grid and predict bounded target search centres only; every
/// returned point and fitted model is measured directly from fused tile pixels,
/// coarse to fine on area-averaged planes of the supplied native planes.
/// Production drives the same measurement through
/// [`virtual_tile_measure_prior_relations`] with its per-station render loop.
#[cfg(test)]
fn virtual_tile_station_match_outcome(
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
) -> VirtualTileStationMatchOutcome {
    let shortest_long_side = tiles
        .iter()
        .map(|tile| {
            tile.alignment_image
                .width()
                .max(tile.alignment_image.height())
        })
        .min()
        .unwrap_or(0);
    let mut pyramid = VirtualTileMatchPyramid::new(
        virtual_tile_pyramid_factors(shortest_long_side),
        tiles.len(),
    );
    for (station, tile) in tiles.iter().enumerate() {
        if let Some(coverage) = coverages.get(station) {
            pyramid.set_station(station, tile, &tile.alignment_image, coverage);
        }
    }
    let (hypotheses, diagnostics) = virtual_tile_measure_prior_relations(
        tiles,
        &pyramid,
        initial_homographies,
        station_topology,
        None,
        false,
    );
    let relations = hypotheses
        .into_iter()
        .filter_map(|(pair, hypotheses)| {
            hypotheses
                .into_iter()
                .next()
                .map(|relation| (pair, relation))
        })
        .collect();
    VirtualTileStationMatchOutcome {
        relations,
        diagnostics,
    }
}

#[cfg(test)]
fn virtual_tile_station_matches(
    tiles: &[ImageInfo],
    coverages: &[stitching::CoverageMask],
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
) -> HashMap<(usize, usize), MatchInfo> {
    virtual_tile_station_match_outcome(tiles, coverages, initial_homographies, station_topology)
        .relations
}

fn single_layer_station_evidence_discard(
    independent_support: usize,
    left_station: usize,
    right_station: usize,
) -> Option<serde_json::Value> {
    (independent_support < 2).then(|| {
        serde_json::json!({
            "count": 1,
            "stations": [left_station, right_station],
            "independent_support": independent_support,
        })
    })
}

/// Total, content-only ordering key of a station relation candidate.
///
/// Two focus layer relations of the same station pair can measure the same
/// quality and the same support; the bit pattern of the relation they propose is
/// then the only thing left that still comes from the measurement rather than
/// from the import order (需求 14.6).
fn focus_group_candidate_key(candidate: &FocusGroupEdgeCandidate) -> [u64; 9] {
    let mut key = [0u64; 9];
    for (slot, value) in key.iter_mut().zip(candidate.transform.iter()) {
        *slot = value.to_bits();
    }
    key
}

fn focus_transform_disagreement_px(
    left: &Matrix3<f64>,
    right: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> Option<f64> {
    let (width, height) = dimensions;
    let samples = [
        Point2::new(width as f64 * 0.2, height as f64 * 0.2),
        Point2::new(width as f64 * 0.8, height as f64 * 0.2),
        Point2::new(width as f64 * 0.5, height as f64 * 0.5),
        Point2::new(width as f64 * 0.2, height as f64 * 0.8),
        Point2::new(width as f64 * 0.8, height as f64 * 0.8),
    ];
    let mut errors = samples
        .into_iter()
        .filter_map(|point| {
            let left = transformed_point(left, point)?;
            let right = transformed_point(right, point)?;
            let error = (left - right).norm();
            error.is_finite().then_some(error)
        })
        .collect::<Vec<_>>();
    median_value(&mut errors)
}

#[derive(Clone)]
struct SourceRawLayerDiagnosticModel {
    left_source: usize,
    right_source: usize,
    record_index: usize,
    transform: Matrix3<f64>,
    spatial_support: f64,
}

fn matrix3_row_major(transform: &Matrix3<f64>) -> [f64; 9] {
    [
        transform[(0, 0)],
        transform[(0, 1)],
        transform[(0, 2)],
        transform[(1, 0)],
        transform[(1, 1)],
        transform[(1, 2)],
        transform[(2, 0)],
        transform[(2, 1)],
        transform[(2, 2)],
    ]
}

/// Select the largest set of cluster observations that shares no Source_RAW
/// on either station. Thus many patches, alternate fits, or matches from one
/// particularly sharp RAW can contribute only one unit of independent support.
fn source_raw_independent_observations(
    models: &[SourceRawLayerDiagnosticModel],
    cluster: &[usize],
) -> Vec<usize> {
    fn augment(
        left: usize,
        adjacency: &BTreeMap<usize, Vec<(usize, usize)>>,
        seen_right: &mut HashSet<usize>,
        right_matches: &mut BTreeMap<usize, (usize, usize)>,
    ) -> bool {
        for &(right, model_index) in adjacency.get(&left).into_iter().flatten() {
            if !seen_right.insert(right) {
                continue;
            }
            let previous = right_matches.get(&right).copied();
            if previous.is_none_or(|(previous_left, _)| {
                augment(previous_left, adjacency, seen_right, right_matches)
            }) {
                right_matches.insert(right, (left, model_index));
                return true;
            }
        }
        false
    }

    let mut adjacency = BTreeMap::<usize, Vec<(usize, usize)>>::new();
    for &model_index in cluster {
        let model = &models[model_index];
        adjacency
            .entry(model.left_source)
            .or_default()
            .push((model.right_source, model_index));
    }
    for edges in adjacency.values_mut() {
        edges.sort_unstable();
    }
    let mut right_matches = BTreeMap::<usize, (usize, usize)>::new();
    for &left in adjacency.keys() {
        augment(left, &adjacency, &mut HashSet::new(), &mut right_matches);
    }
    let mut selected = right_matches
        .into_values()
        .map(|(_, model_index)| model_index)
        .collect::<Vec<_>>();
    selected.sort_unstable();
    selected
}

fn source_raw_consensus_clusters(
    models: &[SourceRawLayerDiagnosticModel],
    layer_records: &mut [stack_report::SourceRawLayerModelRecord],
    dimensions: (u32, u32),
) -> Vec<stack_report::SourceRawConsensusClusterRecord> {
    let mut remaining = (0..models.len()).collect::<BTreeSet<_>>();
    let mut clusters = Vec::new();
    while !remaining.is_empty() {
        let mut best: Option<(usize, usize, f64, [u64; 9], usize, Vec<usize>, Vec<usize>)> = None;
        for &representative in &remaining {
            let members = remaining
                .iter()
                .copied()
                .filter(|&other| {
                    focus_transform_disagreement_px(
                        &models[representative].transform,
                        &models[other].transform,
                        dimensions,
                    )
                    .is_some_and(|error| error <= STATION_RELATION_MAX_MEDIAN_ERROR_PX)
                })
                .collect::<Vec<_>>();
            let independent = source_raw_independent_observations(models, &members);
            let mut disagreements = members
                .iter()
                .filter_map(|&other| {
                    focus_transform_disagreement_px(
                        &models[representative].transform,
                        &models[other].transform,
                        dimensions,
                    )
                })
                .collect::<Vec<_>>();
            let median_disagreement = median_value(&mut disagreements).unwrap_or(f64::INFINITY);
            let matrix_key = virtual_tile_hypothesis_matrix_key(&models[representative].transform);
            let rank = (
                independent.len(),
                members.len(),
                median_disagreement,
                matrix_key,
                representative,
                members,
                independent,
            );
            let replace = best.as_ref().is_none_or(|current| {
                rank.0 > current.0
                    || (rank.0 == current.0 && rank.1 > current.1)
                    || (rank.0 == current.0
                        && rank.1 == current.1
                        && rank.2.total_cmp(&current.2).is_lt())
                    || (rank.0 == current.0
                        && rank.1 == current.1
                        && rank.2.total_cmp(&current.2).is_eq()
                        && rank.3 < current.3)
            });
            if replace {
                best = Some(rank);
            }
        }
        let Some((support, _, median_disagreement, _, representative, members, independent)) = best
        else {
            break;
        };
        if support < 2 {
            break;
        }
        let cluster_index = clusters.len();
        for &model_index in &members {
            layer_records[models[model_index].record_index].consensus_cluster = Some(cluster_index);
            remaining.remove(&model_index);
        }
        let representative_model = &models[representative];
        let scale_ratio =
            homography_scale_ratio(&representative_model.transform).unwrap_or(f64::INFINITY);
        let spatial_support_potential = independent
            .iter()
            .map(|&index| models[index].spatial_support)
            .fold(0.0, f64::max);
        let supporting_layer_pairs = independent
            .iter()
            .map(|&index| {
                let record = &layer_records[models[index].record_index];
                [record.left_source.clone(), record.right_source.clone()]
            })
            .collect();
        clusters.push(stack_report::SourceRawConsensusClusterRecord {
            cluster: cluster_index,
            observation_count: members.len(),
            independent_support: support,
            supporting_layer_pairs,
            median_disagreement_px: median_disagreement,
            scale_ratio,
            spatial_support_potential,
            scale_compliant: (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
                .contains(&scale_ratio),
            spatially_broad: spatial_support_potential >= STATION_RELATION_MIN_SPATIAL_SUPPORT,
            anchor_transform: matrix3_row_major(&representative_model.transform),
        });
    }
    clusters
}

const SOURCE_RAW_CORRELATION_TARGET_LONG_SIDE: u32 = 64;

/// A coverage-safe coarse pyramid built from exactly one Source_RAW. It uses
/// the generic luminance/gradient correlation plane implementation, but never
/// reads a fused VirtualTile, station prior, or another RAW layer.
#[derive(Clone)]
struct SourceRawCorrelationPyramid {
    coarse: VirtualTileGlobalSearchPlane,
}

impl SourceRawCorrelationPyramid {
    fn from_native(gray: &GrayImage, coverage: &stitching::CoverageMask) -> Option<Self> {
        Some(Self {
            coarse: VirtualTileGlobalSearchPlane::from_native_target(
                gray,
                coverage,
                SOURCE_RAW_CORRELATION_TARGET_LONG_SIDE,
            )?,
        })
    }

    fn from_image(image: &ImageInfo) -> Option<Self> {
        let (width, height) = image.alignment_image.dimensions();
        let coverage = stitching::CoverageMask::from_bytes(
            width,
            height,
            vec![u8::MAX; width as usize * height as usize],
        )
        .ok()?;
        Self::from_native(&image.alignment_image, &coverage)
    }
}

#[derive(Clone, Copy, Debug)]
struct SourceRawCorrelationProposal {
    proposal: VirtualTileGlobalProposal,
    second_peak_margin: f64,
}

fn source_raw_correlation_proposals(
    left: &SourceRawCorrelationPyramid,
    right: &SourceRawCorrelationPyramid,
) -> Vec<SourceRawCorrelationProposal> {
    let proposals = virtual_tile_global_similarity_proposals(&left.coarse, &right.coarse);
    proposals
        .iter()
        .enumerate()
        .map(|(index, &proposal)| {
            let second_score = proposals.get(index + 1).map(|other| other.score);
            SourceRawCorrelationProposal {
                proposal,
                second_peak_margin: second_score.map_or(0.0, |score| proposal.score - score),
            }
        })
        .collect()
}

const SOURCE_RAW_DIRECT_TARGET_LONG_SIDES: [u32; 4] = [64, 128, 256, 512];
const SOURCE_RAW_DIRECT_GRID: (usize, usize) = (16, 12);
const SOURCE_RAW_DIRECT_MAX_SEEDS: usize = VIRTUAL_TILE_GLOBAL_TOP_K;
const SOURCE_RAW_DIRECT_PATCH_RADIUS: i32 = 4;
const SOURCE_RAW_DIRECT_SEARCH_RADIUS: i32 = 3;
const SOURCE_RAW_DIRECT_PATCH_GATES: registration::PatchMatchGates =
    registration::PatchMatchGates {
        min_correlation: 0.72,
        min_peak_margin: 0.01,
        min_source_variance: 2.0,
        min_target_variance: 1.0,
    };

#[derive(Clone)]
struct SourceRawDirectLevel {
    plane: VirtualTileMatchPlane,
    gx: Vec<f32>,
    gy: Vec<f32>,
}

impl SourceRawDirectLevel {
    fn new(plane: VirtualTileMatchPlane) -> Self {
        let (width, height) = plane.gray.dimensions();
        let mut gx = vec![0.0; width as usize * height as usize];
        let mut gy = vec![0.0; gx.len()];
        if width >= 3 && height >= 3 {
            for y in 1..height - 1 {
                for x in 1..width - 1 {
                    if [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                        .into_iter()
                        .all(|(x, y)| plane.coverage.is_covered(x, y))
                    {
                        let index = (y * width + x) as usize;
                        gx[index] = 0.5
                            * (f32::from(plane.gray.get_pixel(x + 1, y)[0])
                                - f32::from(plane.gray.get_pixel(x - 1, y)[0]));
                        gy[index] = 0.5
                            * (f32::from(plane.gray.get_pixel(x, y + 1)[0])
                                - f32::from(plane.gray.get_pixel(x, y - 1)[0]));
                    }
                }
            }
        }
        Self { plane, gx, gy }
    }

    fn gradient(&self, point: Point2<f64>) -> Option<nalgebra::Vector2<f64>> {
        let (width, height) = self.plane.gray.dimensions();
        let x = point.x.round() as i32;
        let y = point.y.round() as i32;
        if x <= 0 || y <= 0 || x >= width as i32 - 1 || y >= height as i32 - 1 {
            return None;
        }
        let index = y as usize * width as usize + x as usize;
        let gradient = nalgebra::Vector2::new(f64::from(self.gx[index]), f64::from(self.gy[index]));
        (gradient.norm() > 1e-6).then_some(gradient)
    }
}

#[derive(Clone)]
struct SourceRawDirectPyramid {
    levels: Vec<SourceRawDirectLevel>,
}

impl SourceRawDirectPyramid {
    fn from_native(gray: &GrayImage, coverage: &stitching::CoverageMask) -> Option<Self> {
        let long_side = gray.width().max(gray.height());
        if coverage.dimensions() != gray.dimensions() {
            return None;
        }
        // Sanitize before even the factor-1 level is cloned. This makes every
        // stored luma/gradient bit independent of uncovered payload, not only
        // every objective sample.
        let safe_gray = GrayImage::from_fn(gray.width(), gray.height(), |x, y| {
            if coverage.is_covered(x, y) {
                *gray.get_pixel(x, y)
            } else {
                image::Luma([0])
            }
        });
        let mut factors = SOURCE_RAW_DIRECT_TARGET_LONG_SIDES
            .into_iter()
            .map(|target| long_side.div_ceil(target).max(1))
            .collect::<Vec<_>>();
        factors.sort_unstable_by(|left, right| right.cmp(left));
        factors.dedup();
        let levels = factors
            .into_iter()
            .filter_map(|factor| VirtualTileMatchPlane::from_native(&safe_gray, coverage, factor))
            .map(SourceRawDirectLevel::new)
            .collect::<Vec<_>>();
        (!levels.is_empty()).then_some(Self { levels })
    }

    fn from_image(image: &ImageInfo) -> Option<Self> {
        let (width, height) = image.alignment_image.dimensions();
        let coverage = stitching::CoverageMask::from_bytes(
            width,
            height,
            vec![u8::MAX; width as usize * height as usize],
        )
        .ok()?;
        Self::from_native(&image.alignment_image, &coverage)
    }

    fn working_set_bytes(&self) -> usize {
        self.levels
            .iter()
            .map(|level| {
                let pixels = level.plane.gray.width() as usize * level.plane.gray.height() as usize;
                pixels * (1 + 1 + 4 + 4)
            })
            .sum()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SourceRawDirectLoss {
    loss: f64,
    gain: f64,
    bias: f64,
    samples: usize,
    cells: usize,
}

fn source_raw_direct_level_transform(
    native: &Matrix3<f64>,
    left_factor: u32,
    right_factor: u32,
) -> Matrix3<f64> {
    VirtualTileMatchPlane::native_to_level(right_factor)
        * native
        * VirtualTileMatchPlane::level_to_native(left_factor)
}

fn source_raw_direct_native_transform(
    level: &Matrix3<f64>,
    left_factor: u32,
    right_factor: u32,
) -> Matrix3<f64> {
    VirtualTileMatchPlane::level_to_native(right_factor)
        * level
        * VirtualTileMatchPlane::native_to_level(left_factor)
}

fn source_raw_direct_raw_transform(
    analysis: &Matrix3<f64>,
    left_scale: f64,
    right_scale: f64,
) -> Matrix3<f64> {
    let left_to_raw = Matrix3::new(left_scale, 0.0, 0.0, 0.0, left_scale, 0.0, 0.0, 0.0, 1.0);
    let right_to_raw = Matrix3::new(right_scale, 0.0, 0.0, 0.0, right_scale, 0.0, 0.0, 0.0, 1.0);
    right_to_raw * analysis * left_to_raw.try_inverse().unwrap_or_else(Matrix3::identity)
}

fn source_raw_direct_huber(value: f64, delta: f64) -> f64 {
    let absolute = value.abs();
    if absolute <= delta {
        0.5 * value * value
    } else {
        delta * (absolute - 0.5 * delta)
    }
}

/// Coverage-safe F-Direct objective. Gain and bias are solved independently in
/// each spatial cell before robust intensity and normalized-gradient-vector
/// residuals are accumulated. No descriptor or correspondence enters here.
fn source_raw_direct_objective(
    left: &SourceRawDirectLevel,
    right: &SourceRawDirectLevel,
    transform: &Matrix3<f64>,
    fold: usize,
    holdout: bool,
) -> Option<SourceRawDirectLoss> {
    let (width, height) = left.plane.gray.dimensions();
    let mut cells = BTreeMap::<(usize, usize), Vec<(f64, f64, f64)>>::new();
    let (columns, rows) = (32usize, 24usize);
    for row in 1..rows - 1 {
        let y = (row as f64 + 0.5) * f64::from(height) / rows as f64;
        for column in 1..columns - 1 {
            let cell = (
                column * SOURCE_RAW_DIRECT_GRID.0 / columns,
                row * SOURCE_RAW_DIRECT_GRID.1 / rows,
            );
            let is_holdout = (cell.0 + cell.1) % 2 != fold % 2;
            if is_holdout != holdout {
                continue;
            }
            let x = (column as f64 + 0.5) * f64::from(width) / columns as f64;
            let source = Point2::new(x, y);
            let Some(target) = transformed_point(transform, source) else {
                continue;
            };
            if !virtual_tile_point_is_covered(&left.plane.coverage, source)
                || !virtual_tile_point_is_covered(&right.plane.coverage, target)
            {
                continue;
            }
            let Some(source_value) =
                registration::sample_gray(&left.plane.gray, source.x, source.y)
            else {
                continue;
            };
            let Some(target_value) =
                registration::sample_gray(&right.plane.gray, target.x, target.y)
            else {
                continue;
            };
            let Some(source_gradient) = left.gradient(source) else {
                continue;
            };
            let Some(target_gradient) = right.gradient(target) else {
                continue;
            };
            let Some(jacobian) = virtual_tile_local_jacobian(transform, source) else {
                continue;
            };
            let Some(inverse) = jacobian.try_inverse() else {
                continue;
            };
            let warped_gradient = inverse.transpose() * source_gradient;
            let denominator = warped_gradient.norm() * target_gradient.norm();
            if denominator <= 1e-8 {
                continue;
            }
            let ngf = 1.0
                - (warped_gradient.dot(&target_gradient) / denominator)
                    .abs()
                    .clamp(0.0, 1.0);
            cells
                .entry(cell)
                .or_default()
                .push((source_value, target_value, ngf));
        }
    }
    let mut total_loss = 0.0;
    let mut total_samples = 0usize;
    let mut gains = Vec::new();
    let mut biases = Vec::new();
    for values in cells.values().filter(|values| values.len() >= 4) {
        let mut weights = vec![1.0; values.len()];
        let mut gain = 1.0;
        let mut bias = 0.0;
        for _ in 0..4 {
            let sw = weights.iter().sum::<f64>();
            let sx = values
                .iter()
                .zip(&weights)
                .map(|((source, _, _), weight)| source * weight)
                .sum::<f64>();
            let sy = values
                .iter()
                .zip(&weights)
                .map(|((_, target, _), weight)| target * weight)
                .sum::<f64>();
            let sxx = values
                .iter()
                .zip(&weights)
                .map(|((source, _, _), weight)| source * source * weight)
                .sum::<f64>();
            let sxy = values
                .iter()
                .zip(&weights)
                .map(|((source, target, _), weight)| source * target * weight)
                .sum::<f64>();
            let denominator = sw * sxx - sx * sx;
            if denominator.abs() > 1e-8 {
                gain = (sw * sxy - sx * sy) / denominator;
                bias = (sy - gain * sx) / sw;
            }
            for (weight, (source, target, _)) in weights.iter_mut().zip(values) {
                let residual = (gain * source + bias - target) / 24.0;
                *weight = if residual.abs() <= 1.5 {
                    1.0
                } else {
                    1.5 / residual.abs()
                };
            }
        }
        if !gain.is_finite() || !bias.is_finite() {
            continue;
        }
        gains.push(gain);
        biases.push(bias);
        for &(source, target, ngf) in values {
            let intensity = (gain * source + bias - target) / 24.0;
            total_loss += 0.65 * source_raw_direct_huber(intensity, 1.5)
                + 0.35 * source_raw_direct_huber(ngf * 3.0, 1.0);
            total_samples += 1;
        }
    }
    if total_samples < 24 {
        return None;
    }
    Some(SourceRawDirectLoss {
        loss: total_loss / total_samples as f64,
        gain: gains.iter().sum::<f64>() / gains.len().max(1) as f64,
        bias: biases.iter().sum::<f64>() / biases.len().max(1) as f64,
        samples: total_samples,
        cells: gains.len(),
    })
}

fn source_raw_direct_symmetric_objective(
    left: &SourceRawDirectLevel,
    right: &SourceRawDirectLevel,
    transform: &Matrix3<f64>,
    fold: usize,
    holdout: bool,
) -> Option<SourceRawDirectLoss> {
    let forward = source_raw_direct_objective(left, right, transform, fold, holdout)?;
    let reverse_transform = transform.try_inverse()?;
    let reverse = source_raw_direct_objective(right, left, &reverse_transform, fold, holdout)?;
    Some(SourceRawDirectLoss {
        loss: 0.5 * (forward.loss + reverse.loss),
        samples: forward.samples.min(reverse.samples),
        cells: forward.cells.min(reverse.cells),
        gain: forward.gain,
        bias: forward.bias,
    })
}

#[derive(Clone)]
struct SourceRawDirectFold {
    transform: Matrix3<f64>,
    model: &'static str,
    iterations: usize,
    fit: SourceRawDirectLoss,
    holdout: SourceRawDirectLoss,
    factors: Vec<u32>,
    seed_rank: usize,
}

fn source_raw_direct_geometry_valid(transform: &Matrix3<f64>, dimensions: (u32, u32)) -> bool {
    transform.try_inverse().is_some()
        && homography_scale_ratio(transform).is_some_and(|scale| {
            (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX).contains(&scale)
        })
        && homography_preserves_focus_orientation(transform, dimensions)
}

fn source_raw_direct_adjustments(
    model: &'static str,
    translation_step: f64,
    mapped_center: Point2<f64>,
) -> Vec<Matrix3<f64>> {
    let around = |linear: Matrix3<f64>| {
        virtual_tile_translation(mapped_center.x, mapped_center.y)
            * linear
            * virtual_tile_translation(-mapped_center.x, -mapped_center.y)
    };
    let mut adjustments = vec![
        virtual_tile_translation(translation_step, 0.0),
        virtual_tile_translation(-translation_step, 0.0),
        virtual_tile_translation(0.0, translation_step),
        virtual_tile_translation(0.0, -translation_step),
    ];
    let angle = (translation_step * 0.12).min(1.0).to_radians();
    let scale = (translation_step * 0.0015).min(0.01);
    for sign in [-1.0, 1.0] {
        let (sin, cos) = (sign * angle).sin_cos();
        adjustments.push(around(Matrix3::new(
            cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0,
        )));
        adjustments.push(around(Matrix3::new(
            1.0 + sign * scale,
            0.0,
            0.0,
            0.0,
            1.0 + sign * scale,
            0.0,
            0.0,
            0.0,
            1.0,
        )));
    }
    if matches!(model, "affine" | "projective") {
        let shear = (translation_step * 0.0008).min(0.004);
        for sign in [-1.0, 1.0] {
            adjustments.push(around(Matrix3::new(
                1.0 + sign * shear,
                sign * shear,
                0.0,
                0.0,
                1.0 - sign * shear,
                0.0,
                0.0,
                0.0,
                1.0,
            )));
        }
    }
    if model == "projective" {
        let projective = (translation_step * 0.000_002).min(0.000_01);
        for sign in [-1.0, 1.0] {
            adjustments.push(Matrix3::new(
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                sign * projective,
                0.0,
                1.0,
            ));
            adjustments.push(Matrix3::new(
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                sign * projective,
                1.0,
            ));
        }
    }
    adjustments
}

fn source_raw_direct_optimize_stage(
    left: &SourceRawDirectLevel,
    right: &SourceRawDirectLevel,
    initial: Matrix3<f64>,
    fold: usize,
    model: &'static str,
) -> Option<(Matrix3<f64>, SourceRawDirectLoss, usize)> {
    let mut transform = initial;
    let mut best = source_raw_direct_symmetric_objective(left, right, &transform, fold, false)?;
    let center = Point2::new(
        f64::from(left.plane.gray.width()) * 0.5,
        f64::from(left.plane.gray.height()) * 0.5,
    );
    let mapped_center = transformed_point(&transform, center)?;
    let mut iterations = 0usize;
    for step in [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625] {
        for _ in 0..4 {
            iterations += 1;
            let mut improved = false;
            for adjustment in source_raw_direct_adjustments(model, step, mapped_center) {
                let Some(candidate) = normalized_homography(&(adjustment * transform)) else {
                    continue;
                };
                if !source_raw_direct_geometry_valid(&candidate, left.plane.gray.dimensions()) {
                    continue;
                }
                let Some(loss) =
                    source_raw_direct_symmetric_objective(left, right, &candidate, fold, false)
                else {
                    continue;
                };
                if loss.loss.total_cmp(&best.loss).is_lt() {
                    transform = candidate;
                    best = loss;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }
    Some((transform, best, iterations.min(20)))
}

fn source_raw_direct_refine(
    left: &SourceRawDirectPyramid,
    right: &SourceRawDirectPyramid,
    seed: Matrix3<f64>,
    seed_rank: usize,
    fold: usize,
) -> Option<SourceRawDirectFold> {
    let mut native = seed;
    let mut model = "similarity";
    let mut iterations = 0usize;
    let mut final_fit = None;
    let mut final_holdout = None;
    let mut factors = Vec::new();
    for (left_level, right_level) in left.levels.iter().zip(&right.levels) {
        factors.push(left_level.plane.factor.max(right_level.plane.factor));
        let mut level_transform = source_raw_direct_level_transform(
            &native,
            left_level.plane.factor,
            right_level.plane.factor,
        );
        let (similarity, fit, count) = source_raw_direct_optimize_stage(
            left_level,
            right_level,
            level_transform,
            fold,
            "similarity",
        )?;
        level_transform = similarity;
        iterations += count;
        let mut holdout = source_raw_direct_symmetric_objective(
            left_level,
            right_level,
            &level_transform,
            fold,
            true,
        )?;
        model = "similarity";
        for next_model in ["affine", "projective"] {
            let Some((candidate, candidate_fit, count)) = source_raw_direct_optimize_stage(
                left_level,
                right_level,
                level_transform,
                fold,
                next_model,
            ) else {
                continue;
            };
            let Some(candidate_holdout) = source_raw_direct_symmetric_objective(
                left_level,
                right_level,
                &candidate,
                fold,
                true,
            ) else {
                continue;
            };
            if candidate_holdout.loss < holdout.loss * 0.90
                && source_raw_direct_geometry_valid(&candidate, left_level.plane.gray.dimensions())
            {
                level_transform = candidate;
                holdout = candidate_holdout;
                model = next_model;
                iterations += count;
                final_fit = Some(candidate_fit);
            }
        }
        native = source_raw_direct_native_transform(
            &level_transform,
            left_level.plane.factor,
            right_level.plane.factor,
        );
        final_fit = final_fit.or(Some(fit));
        final_holdout = Some(holdout);
    }
    Some(SourceRawDirectFold {
        transform: native,
        model,
        iterations,
        fit: final_fit?,
        holdout: final_holdout?,
        factors,
        seed_rank,
    })
}

fn source_raw_direct_grid_centers(
    dimensions: (u32, u32),
    fold: usize,
) -> Vec<((usize, usize), Point2<f64>)> {
    let (columns, rows) = SOURCE_RAW_DIRECT_GRID;
    (0..rows)
        .flat_map(|row| {
            (0..columns).filter_map(move |column| {
                ((column + row) % 2 != fold % 2).then(|| {
                    (
                        (column, row),
                        Point2::new(
                            (column as f64 + 0.5) * f64::from(dimensions.0) / columns as f64,
                            (row as f64 + 0.5) * f64::from(dimensions.1) / rows as f64,
                        ),
                    )
                })
            })
        })
        .collect()
}

#[derive(Clone)]
struct SourceRawDirectObservation {
    record: stack_report::SourceRawDirectObservationRecord,
    left_source: usize,
    right_source: usize,
    transform: Matrix3<f64>,
    reverse_transform: Matrix3<f64>,
    anchor_points: Vec<(Point2<f64>, Point2<f64>)>,
}

fn source_raw_direct_finite_diagnostic(value: f64) -> f64 {
    if value.is_finite() { value } else { f64::MAX }
}

fn source_raw_direct_closure(
    forward: &Matrix3<f64>,
    reverse: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> Option<f64> {
    let samples = [
        Point2::new(0.0, 0.0),
        Point2::new(f64::from(dimensions.0), 0.0),
        Point2::new(0.0, f64::from(dimensions.1)),
        Point2::new(f64::from(dimensions.0), f64::from(dimensions.1)),
        Point2::new(f64::from(dimensions.0) * 0.5, f64::from(dimensions.1) * 0.5),
    ];
    samples
        .into_iter()
        .map(|point| {
            transformed_point(forward, point)
                .and_then(|mapped| transformed_point(reverse, mapped))
                .map(|returned| (returned - point).norm())
        })
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max_by(f64::total_cmp)
}

fn source_raw_direct_patch_fingerprint(image: &GrayImage, center: Point2<f64>) -> Option<u32> {
    let mut values = Vec::with_capacity(25);
    for dy in [-4.0, -2.0, 0.0, 2.0, 4.0] {
        for dx in [-4.0, -2.0, 0.0, 2.0, 4.0] {
            values.push(registration::sample_gray(
                image,
                center.x + dx,
                center.y + dy,
            )?);
        }
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    if variance < 2.0 {
        return None;
    }
    Some(
        values
            .into_iter()
            .enumerate()
            .fold(0u32, |bits, (index, value)| {
                bits | (u32::from(value >= mean) << index)
            }),
    )
}

fn source_raw_direct_localize_holdout(
    left: &ImageInfo,
    right: &ImageInfo,
    left_pyramid: &SourceRawDirectPyramid,
    right_pyramid: &SourceRawDirectPyramid,
    forward: &Matrix3<f64>,
    reverse: &Matrix3<f64>,
    fold: usize,
) -> Vec<stack_report::SourceRawDirectPointRecord> {
    let Some(left_level) = left_pyramid.levels.last() else {
        return Vec::new();
    };
    let Some(right_level) = right_pyramid.levels.last() else {
        return Vec::new();
    };
    let level_forward = source_raw_direct_level_transform(
        forward,
        left_level.plane.factor,
        right_level.plane.factor,
    );
    let level_reverse = source_raw_direct_level_transform(
        reverse,
        right_level.plane.factor,
        left_level.plane.factor,
    );
    let left_level_to_native = VirtualTileMatchPlane::level_to_native(left_level.plane.factor);
    let right_level_to_native = VirtualTileMatchPlane::level_to_native(right_level.plane.factor);
    let mut fingerprints = HashSet::new();
    source_raw_direct_grid_centers(left_level.plane.gray.dimensions(), fold)
        .into_iter()
        .filter_map(|(cell, source)| {
            let fingerprint = source_raw_direct_patch_fingerprint(&left_level.plane.gray, source)?;
            if !virtual_tile_direct_patch_support_is_covered(
                &left_level.plane.coverage,
                &right_level.plane.coverage,
                &level_forward,
                source,
                SOURCE_RAW_DIRECT_PATCH_RADIUS,
                SOURCE_RAW_DIRECT_SEARCH_RADIUS,
            ) {
                return None;
            }
            let forward_probe = registration::probe_warped_patch(
                &left_level.plane.gray,
                &right_level.plane.gray,
                &level_forward,
                source,
                SOURCE_RAW_DIRECT_PATCH_RADIUS,
                SOURCE_RAW_DIRECT_SEARCH_RADIUS,
                SOURCE_RAW_DIRECT_PATCH_GATES,
            );
            let measured = forward_probe.matched?;
            if !virtual_tile_direct_patch_support_is_covered(
                &right_level.plane.coverage,
                &left_level.plane.coverage,
                &level_reverse,
                measured.target,
                SOURCE_RAW_DIRECT_PATCH_RADIUS,
                SOURCE_RAW_DIRECT_SEARCH_RADIUS,
            ) {
                return None;
            }
            let reverse_probe = registration::probe_warped_patch(
                &right_level.plane.gray,
                &left_level.plane.gray,
                &level_reverse,
                measured.target,
                SOURCE_RAW_DIRECT_PATCH_RADIUS,
                SOURCE_RAW_DIRECT_SEARCH_RADIUS,
                SOURCE_RAW_DIRECT_PATCH_GATES,
            );
            let returned = reverse_probe.matched?.target;
            let source_native = transformed_point(&left_level_to_native, source)?;
            let target_native = transformed_point(&right_level_to_native, measured.target)?;
            let returned_native = transformed_point(&left_level_to_native, returned)?;
            let source_raw = source_native * left.scale_factor;
            let target_raw = target_native * right.scale_factor;
            let returned_raw = returned_native * left.scale_factor;
            let round_trip = (returned_raw - source_raw).norm();
            (round_trip <= VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX).then_some((
                fingerprint,
                stack_report::SourceRawDirectPointRecord {
                    cell: [cell.0, cell.1],
                    source: [source_raw.x, source_raw.y],
                    target: [target_raw.x, target_raw.y],
                    reverse_source: [returned_raw.x, returned_raw.y],
                    correlation: measured.correlation,
                    peak_margin: forward_probe.peak_margin.unwrap_or_default(),
                    round_trip_error_px: round_trip,
                },
            ))
        })
        .filter_map(|(fingerprint, record)| fingerprints.insert(fingerprint).then_some(record))
        .collect()
}

#[derive(Clone)]
struct SourceRawDirectBasinSelection {
    forward: Vec<SourceRawDirectFold>,
    reverse: Vec<SourceRawDirectFold>,
    points: Vec<Vec<stack_report::SourceRawDirectPointRecord>>,
    closures: Vec<f64>,
    forward_fold_disagreement: f64,
    reverse_fold_disagreement: f64,
    coarse_closure: f64,
    minimum_points: usize,
    union_points: usize,
    minimum_spatial_support: f64,
    covered_overlap: usize,
    maximum_holdout_loss: f64,
}

fn source_raw_direct_basin_is_better(
    candidate: &SourceRawDirectBasinSelection,
    current: &SourceRawDirectBasinSelection,
) -> bool {
    let reciprocal = |selection: &SourceRawDirectBasinSelection| {
        selection.closures.iter().copied().fold(0.0, f64::max)
            <= VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX
            && selection.forward_fold_disagreement <= STATION_RELATION_MAX_MEDIAN_ERROR_PX
            && selection.reverse_fold_disagreement <= STATION_RELATION_MAX_MEDIAN_ERROR_PX
    };
    let candidate_reciprocal = reciprocal(candidate);
    let current_reciprocal = reciprocal(current);
    if candidate_reciprocal != current_reciprocal {
        return candidate_reciprocal;
    }
    if candidate.minimum_points != current.minimum_points {
        return candidate.minimum_points > current.minimum_points;
    }
    if candidate.union_points != current.union_points {
        return candidate.union_points > current.union_points;
    }
    if candidate
        .minimum_spatial_support
        .total_cmp(&current.minimum_spatial_support)
        .is_ne()
    {
        return candidate.minimum_spatial_support > current.minimum_spatial_support;
    }
    let candidate_closure = candidate.closures.iter().copied().fold(0.0, f64::max);
    let current_closure = current.closures.iter().copied().fold(0.0, f64::max);
    if candidate_closure.total_cmp(&current_closure).is_ne() {
        return candidate_closure < current_closure;
    }
    let candidate_disagreement = candidate
        .forward_fold_disagreement
        .max(candidate.reverse_fold_disagreement);
    let current_disagreement = current
        .forward_fold_disagreement
        .max(current.reverse_fold_disagreement);
    if candidate_disagreement
        .total_cmp(&current_disagreement)
        .is_ne()
    {
        return candidate_disagreement < current_disagreement;
    }
    if candidate.covered_overlap != current.covered_overlap {
        return candidate.covered_overlap > current.covered_overlap;
    }
    if candidate
        .coarse_closure
        .total_cmp(&current.coarse_closure)
        .is_ne()
    {
        return candidate.coarse_closure < current.coarse_closure;
    }
    if candidate
        .maximum_holdout_loss
        .total_cmp(&current.maximum_holdout_loss)
        .is_ne()
    {
        return candidate.maximum_holdout_loss < current.maximum_holdout_loss;
    }
    let candidate_key = (
        candidate.forward[0].seed_rank,
        candidate.reverse[0].seed_rank,
        virtual_tile_hypothesis_matrix_key(&candidate.forward[0].transform),
        virtual_tile_hypothesis_matrix_key(&candidate.reverse[0].transform),
    );
    let current_key = (
        current.forward[0].seed_rank,
        current.reverse[0].seed_rank,
        virtual_tile_hypothesis_matrix_key(&current.forward[0].transform),
        virtual_tile_hypothesis_matrix_key(&current.reverse[0].transform),
    );
    candidate_key < current_key
}

/// Select one reciprocal basin from independently generated and independently
/// refined forward/reverse peaks. The accepted station relation is never read:
/// ranking uses only transform composition, covered overlap, spatially spread
/// measured hold-out localizations, fold stability, and direct image losses.
fn source_raw_direct_select_basin(
    left: &ImageInfo,
    right: &ImageInfo,
    left_pyramid: &SourceRawDirectPyramid,
    right_pyramid: &SourceRawDirectPyramid,
    forward_seeds: &[SourceRawCorrelationProposal],
    reverse_seeds: &[SourceRawCorrelationProposal],
) -> Option<SourceRawDirectBasinSelection> {
    let forward_seeds = &forward_seeds[..forward_seeds.len().min(SOURCE_RAW_DIRECT_MAX_SEEDS)];
    let reverse_seeds = &reverse_seeds[..reverse_seeds.len().min(SOURCE_RAW_DIRECT_MAX_SEEDS)];
    let forward_folds = (0..2)
        .map(|fold| {
            forward_seeds
                .iter()
                .enumerate()
                .map(|(rank, seed)| {
                    source_raw_direct_refine(
                        left_pyramid,
                        right_pyramid,
                        seed.proposal.left_to_right,
                        rank,
                        fold,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let reverse_folds = (0..2)
        .map(|fold| {
            reverse_seeds
                .iter()
                .enumerate()
                .map(|(rank, seed)| {
                    source_raw_direct_refine(
                        right_pyramid,
                        left_pyramid,
                        seed.proposal.left_to_right,
                        rank,
                        fold,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut best = None;
    for (forward_rank, forward_seed) in forward_seeds.iter().enumerate() {
        for (reverse_rank, reverse_seed) in reverse_seeds.iter().enumerate() {
            let forward = (0..2)
                .map(|fold| forward_folds[fold][forward_rank].clone())
                .collect::<Option<Vec<_>>>();
            let reverse = (0..2)
                .map(|fold| reverse_folds[fold][reverse_rank].clone())
                .collect::<Option<Vec<_>>>();
            let (Some(forward), Some(reverse)) = (forward, reverse) else {
                continue;
            };
            let coarse_forward = source_raw_direct_raw_transform(
                &forward_seed.proposal.left_to_right,
                left.scale_factor,
                right.scale_factor,
            );
            let coarse_reverse = source_raw_direct_raw_transform(
                &reverse_seed.proposal.left_to_right,
                right.scale_factor,
                left.scale_factor,
            );
            let Some(coarse_closure) =
                source_raw_direct_closure(&coarse_forward, &coarse_reverse, left.dimensions())
            else {
                continue;
            };
            let mut points = Vec::with_capacity(2);
            let mut closures = Vec::with_capacity(2);
            let mut spatial_support = Vec::with_capacity(2);
            for fold in 0..2 {
                let raw_forward = source_raw_direct_raw_transform(
                    &forward[fold].transform,
                    left.scale_factor,
                    right.scale_factor,
                );
                let raw_reverse = source_raw_direct_raw_transform(
                    &reverse[fold].transform,
                    right.scale_factor,
                    left.scale_factor,
                );
                let Some(closure) =
                    source_raw_direct_closure(&raw_forward, &raw_reverse, left.dimensions())
                else {
                    points.clear();
                    break;
                };
                let measured = source_raw_direct_localize_holdout(
                    left,
                    right,
                    left_pyramid,
                    right_pyramid,
                    &forward[fold].transform,
                    &reverse[fold].transform,
                    fold,
                );
                let measured_pairs = measured
                    .iter()
                    .map(|point| {
                        (
                            Point2::new(point.source[0], point.source[1]),
                            Point2::new(point.target[0], point.target[1]),
                        )
                    })
                    .collect::<Vec<_>>();
                spatial_support.push(station_relation_spatial_support(
                    &measured_pairs,
                    &raw_forward,
                    left.dimensions(),
                    right.dimensions(),
                ));
                closures.push(closure);
                points.push(measured);
            }
            if points.len() != 2 {
                continue;
            }
            let forward_raw = forward
                .iter()
                .map(|fold| {
                    source_raw_direct_raw_transform(
                        &fold.transform,
                        left.scale_factor,
                        right.scale_factor,
                    )
                })
                .collect::<Vec<_>>();
            let reverse_raw = reverse
                .iter()
                .map(|fold| {
                    source_raw_direct_raw_transform(
                        &fold.transform,
                        right.scale_factor,
                        left.scale_factor,
                    )
                })
                .collect::<Vec<_>>();
            let Some(forward_fold_disagreement) = focus_transform_disagreement_px(
                &forward_raw[0],
                &forward_raw[1],
                left.dimensions(),
            ) else {
                continue;
            };
            let Some(reverse_fold_disagreement) = focus_transform_disagreement_px(
                &reverse_raw[0],
                &reverse_raw[1],
                right.dimensions(),
            ) else {
                continue;
            };
            let union_points = points
                .iter()
                .flat_map(|fold| fold.iter().map(|point| point.cell))
                .collect::<BTreeSet<_>>()
                .len();
            let selection = SourceRawDirectBasinSelection {
                minimum_points: points.iter().map(Vec::len).min().unwrap_or_default(),
                union_points,
                minimum_spatial_support: spatial_support.into_iter().fold(1.0, f64::min),
                covered_overlap: forward_seed
                    .proposal
                    .overlap_samples
                    .min(reverse_seed.proposal.overlap_samples),
                maximum_holdout_loss: forward
                    .iter()
                    .chain(&reverse)
                    .map(|fold| fold.holdout.loss)
                    .fold(0.0, f64::max),
                forward,
                reverse,
                points,
                closures,
                forward_fold_disagreement,
                reverse_fold_disagreement,
                coarse_closure,
            };
            if best
                .as_ref()
                .is_none_or(|current| source_raw_direct_basin_is_better(&selection, current))
            {
                best = Some(selection);
            }
        }
    }
    best
}

fn source_raw_direct_observation(
    images: &[ImageInfo],
    left_source: usize,
    right_source: usize,
    left_group: &FocusCaptureGroup,
    right_group: &FocusCaptureGroup,
) -> Option<SourceRawDirectObservation> {
    let left = &images[left_source];
    let right = &images[right_source];
    let left_pyramid = SourceRawDirectPyramid::from_image(left)?;
    let right_pyramid = SourceRawDirectPyramid::from_image(right)?;
    let left_coarse = SourceRawCorrelationPyramid::from_image(left)?;
    let right_coarse = SourceRawCorrelationPyramid::from_image(right)?;
    let forward_seeds = source_raw_correlation_proposals(&left_coarse, &right_coarse);
    let reverse_seeds = source_raw_correlation_proposals(&right_coarse, &left_coarse);
    if forward_seeds.is_empty() || reverse_seeds.is_empty() {
        return None;
    }
    let selection = source_raw_direct_select_basin(
        left,
        right,
        &left_pyramid,
        &right_pyramid,
        &forward_seeds,
        &reverse_seeds,
    )?;
    let mut fold_records = Vec::new();
    let mut fold_transforms = Vec::new();
    let mut all_points = Vec::new();
    let mut reverse_transforms = Vec::new();
    for fold in 0..2 {
        let forward = &selection.forward[fold];
        let reverse = &selection.reverse[fold];
        let raw_forward = source_raw_direct_raw_transform(
            &forward.transform,
            left.scale_factor,
            right.scale_factor,
        );
        let raw_reverse = source_raw_direct_raw_transform(
            &reverse.transform,
            right.scale_factor,
            left.scale_factor,
        );
        let closure = selection.closures[fold];
        let points = selection.points[fold].clone();
        fold_records.push(stack_report::SourceRawDirectFoldRecord {
            fold,
            seed_rank: forward.seed_rank,
            reverse_seed_rank: reverse.seed_rank,
            seed_pair_coarse_closure_px: selection.coarse_closure,
            seed_provenance: "reciprocal_coarse_multi_peak_fold_consistent".to_string(),
            pyramid_factors: forward.factors.clone(),
            iterations: forward.iterations + reverse.iterations,
            fit_cell_count: forward.fit.cells.min(reverse.fit.cells),
            holdout_cell_count: forward.holdout.cells.min(reverse.holdout.cells),
            fit_loss: forward.fit.loss.max(reverse.fit.loss),
            holdout_loss: forward.holdout.loss.max(reverse.holdout.loss),
            gain: forward.holdout.gain,
            bias: forward.holdout.bias,
            model: format!("{}+independent_reverse_{}", forward.model, reverse.model),
            raw_transform: matrix3_row_major(&raw_forward),
            reverse_raw_transform: matrix3_row_major(&raw_reverse),
            bidirectional_closure_px: closure,
            points: points.clone(),
        });
        all_points.extend(points);
        fold_transforms.push(raw_forward);
        reverse_transforms.push(raw_reverse);
    }
    let fold_disagreement = selection.forward_fold_disagreement;
    let representative = if fold_records[1].holdout_loss < fold_records[0].holdout_loss {
        1
    } else {
        0
    };
    let raw_transform = fold_transforms[representative];
    let reverse_transform = reverse_transforms[representative];
    let closure = fold_records
        .iter()
        .map(|record| record.bidirectional_closure_px)
        .max_by(f64::total_cmp)?;
    all_points.sort_by(|left, right| {
        left.cell
            .cmp(&right.cell)
            .then_with(|| left.source[1].total_cmp(&right.source[1]))
            .then_with(|| left.source[0].total_cmp(&right.source[0]))
    });
    all_points.dedup_by_key(|point| point.cell);
    let raw_points = all_points
        .iter()
        .map(|point| {
            (
                Point2::new(point.source[0], point.source[1]),
                Point2::new(point.target[0], point.target[1]),
            )
        })
        .collect::<Vec<_>>();
    let left_local = left_group.local_to_anchor.get(&left_source)?;
    let right_local = right_group.local_to_anchor.get(&right_source)?;
    let anchor_transform = right_local * raw_transform * left_local.try_inverse()?;
    let anchor_reverse = left_local * reverse_transform * right_local.try_inverse()?;
    let anchor_points = raw_points
        .iter()
        .filter_map(|&(source, target)| {
            transformed_point(left_local, source).zip(transformed_point(right_local, target))
        })
        .collect::<Vec<_>>();
    let median_error_px = median_symmetric_error(&anchor_transform, &anchor_points);
    let scale_ratio = homography_scale_ratio(&anchor_transform).unwrap_or(f64::INFINITY);
    let spatial_support = station_relation_spatial_support(
        &anchor_points,
        &anchor_transform,
        images[left_group.anchor].dimensions(),
        images[right_group.anchor].dimensions(),
    );
    let preserves_convex_orientation = homography_preserves_focus_orientation(
        &anchor_transform,
        images[left_group.anchor].dimensions(),
    );
    let dense = focus_overlap_quality_measurements(left, right, &raw_transform);
    let measured = measure_station_relation_candidate(
        raw_points.len(),
        &raw_transform,
        &raw_points,
        left.dimensions(),
        right.dimensions(),
        Some(StationOverlapMeasure::Rectangles),
        dense,
        false,
    );
    let mut rejection_reasons = measured
        .as_ref()
        .map(|measured| {
            station_relation_rejection_reasons(&measured.measurements)
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec!["direct_measurement_unavailable".to_string()]);
    if fold_disagreement > STATION_RELATION_MAX_MEDIAN_ERROR_PX {
        rejection_reasons.push("direct_fold_disagreement".to_string());
    }
    if closure > VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX {
        rejection_reasons.push("direct_bidirectional_closure".to_string());
    }
    rejection_reasons.sort();
    rejection_reasons.dedup();
    let record = stack_report::SourceRawDirectObservationRecord {
        left_source: absolute_source_path(&left.filename),
        right_source: absolute_source_path(&right.filename),
        folds: fold_records,
        model: if all_points.is_empty() {
            "none".to_string()
        } else {
            "similarity_affine_projective_heldout".to_string()
        },
        holdout_points: anchor_points.len(),
        median_error_px: source_raw_direct_finite_diagnostic(median_error_px),
        scale_ratio: source_raw_direct_finite_diagnostic(scale_ratio),
        spatial_support: source_raw_direct_finite_diagnostic(spatial_support),
        preserves_convex_orientation,
        fold_disagreement_px: source_raw_direct_finite_diagnostic(fold_disagreement),
        bidirectional_closure_px: source_raw_direct_finite_diagnostic(closure),
        anchor_transform: matrix3_row_major(&anchor_transform),
        rejection_reasons,
    };
    Some(SourceRawDirectObservation {
        record,
        left_source,
        right_source,
        transform: anchor_transform,
        reverse_transform: anchor_reverse,
        anchor_points,
    })
}

fn source_raw_direct_consensus_clusters(
    images: &[ImageInfo],
    left_group: &FocusCaptureGroup,
    right_group: &FocusCaptureGroup,
    observations: &[SourceRawDirectObservation],
) -> Vec<stack_report::SourceRawDirectConsensusRecord> {
    let valid = observations
        .iter()
        .enumerate()
        .filter(|(_, observation)| observation.record.rejection_reasons.is_empty())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut clusters = Vec::new();
    let mut remaining = valid.into_iter().collect::<BTreeSet<_>>();
    while let Some(&representative) = remaining.first() {
        let members = remaining
            .iter()
            .copied()
            .filter(|&index| {
                focus_transform_disagreement_px(
                    &observations[representative].transform,
                    &observations[index].transform,
                    images[left_group.anchor].dimensions(),
                )
                .is_some_and(|error| error <= STATION_RELATION_MAX_MEDIAN_ERROR_PX)
            })
            .collect::<Vec<_>>();
        let support_models = members
            .iter()
            .enumerate()
            .map(|(record_index, &index)| SourceRawLayerDiagnosticModel {
                left_source: observations[index].left_source,
                right_source: observations[index].right_source,
                record_index,
                transform: observations[index].transform,
                spatial_support: observations[index].record.spatial_support,
            })
            .collect::<Vec<_>>();
        let selected_local = source_raw_independent_observations(
            &support_models,
            &(0..support_models.len()).collect::<Vec<_>>(),
        );
        let selected = selected_local
            .iter()
            .map(|&index| members[index])
            .collect::<Vec<_>>();
        for member in &members {
            remaining.remove(member);
        }
        if selected.len() < 2 {
            continue;
        }
        let representative = selected[0];
        let transform = observations[representative].transform;
        let reverse = observations[representative].reverse_transform;
        let points = selected
            .iter()
            .flat_map(|&index| observations[index].anchor_points.iter().copied())
            .collect::<Vec<_>>();
        let mut disagreements = selected
            .iter()
            .filter_map(|&index| {
                focus_transform_disagreement_px(
                    &transform,
                    &observations[index].transform,
                    images[left_group.anchor].dimensions(),
                )
            })
            .collect::<Vec<_>>();
        let canonical_disagreement_px = median_value(&mut disagreements).unwrap_or(f64::INFINITY);
        let closure =
            source_raw_direct_closure(&transform, &reverse, images[left_group.anchor].dimensions())
                .unwrap_or(f64::INFINITY);
        let dense = focus_overlap_quality_measurements(
            &images[left_group.anchor],
            &images[right_group.anchor],
            &transform,
        );
        let measured = measure_station_relation_candidate(
            points.len(),
            &transform,
            &points,
            images[left_group.anchor].dimensions(),
            images[right_group.anchor].dimensions(),
            Some(StationOverlapMeasure::Rectangles),
            dense,
            false,
        );
        let mut reasons = measured
            .as_ref()
            .map(|measured| {
                station_relation_rejection_reasons(&measured.measurements)
                    .into_iter()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec!["direct_station_measurement_unavailable".to_string()]);
        if canonical_disagreement_px > STATION_RELATION_MAX_MEDIAN_ERROR_PX {
            reasons.push("direct_canonical_disagreement".to_string());
        }
        if closure > VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX {
            reasons.push("direct_bidirectional_closure".to_string());
        }
        reasons.sort();
        reasons.dedup();
        let measurements = measured.map(|measured| measured.measurements);
        clusters.push(stack_report::SourceRawDirectConsensusRecord {
            cluster: clusters.len(),
            observation_count: members.len(),
            independent_support: selected.len(),
            supporting_layer_pairs: selected
                .iter()
                .map(|&index| {
                    [
                        observations[index].record.left_source.clone(),
                        observations[index].record.right_source.clone(),
                    ]
                })
                .collect(),
            canonical_disagreement_px: source_raw_direct_finite_diagnostic(
                canonical_disagreement_px,
            ),
            anchor_transform: matrix3_row_major(&transform),
            station_inliers: measurements.as_ref().map_or(0, |value| value.inliers),
            station_median_error_px: source_raw_direct_finite_diagnostic(
                measurements
                    .as_ref()
                    .map_or(f64::MAX, |value| value.median_error_px),
            ),
            station_scale_ratio: source_raw_direct_finite_diagnostic(
                measurements
                    .as_ref()
                    .map_or(f64::MAX, |value| value.scale_ratio),
            ),
            station_spatial_support: source_raw_direct_finite_diagnostic(
                measurements
                    .as_ref()
                    .map_or(0.0, |value| value.spatial_support),
            ),
            station_preserves_convex_orientation: measurements
                .as_ref()
                .is_some_and(|value| value.preserves_convex_orientation),
            station_bidirectional_closure_px: source_raw_direct_finite_diagnostic(closure),
            station_rejection_reasons: reasons,
        });
    }
    clusters
}

fn source_raw_direct_pair_diagnostics(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
) -> Vec<stack_report::SourceRawDirectPairDiagnosticRecord> {
    let mut diagnostics = Vec::new();
    for left_station in 0..groups.len() {
        for right_station in left_station + 1..groups.len() {
            let start = std::time::Instant::now();
            let left_group = &groups[left_station];
            let right_group = &groups[right_station];
            let mut observations = Vec::new();
            let mut peak_working_set = 0usize;
            for &left_source in &left_group.members {
                for &right_source in &right_group.members {
                    if let (Some(left), Some(right)) = (
                        SourceRawDirectPyramid::from_image(&images[left_source]),
                        SourceRawDirectPyramid::from_image(&images[right_source]),
                    ) {
                        peak_working_set = peak_working_set
                            .max(left.working_set_bytes() + right.working_set_bytes());
                    }
                    if let Some(observation) = source_raw_direct_observation(
                        images,
                        left_source,
                        right_source,
                        left_group,
                        right_group,
                    ) {
                        observations.push(observation);
                    }
                }
            }
            observations
                .sort_by_key(|observation| (observation.left_source, observation.right_source));
            let consensus_clusters = source_raw_direct_consensus_clusters(
                images,
                left_group,
                right_group,
                &observations,
            );
            diagnostics.push(stack_report::SourceRawDirectPairDiagnosticRecord {
                left: left_station,
                right: right_station,
                observations: observations
                    .into_iter()
                    .map(|observation| observation.record)
                    .collect(),
                consensus_clusters,
                elapsed_ms: start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
                peak_working_set_bytes: peak_working_set as u64,
            });
        }
    }
    diagnostics
}

/// Convert one correlation peak into deterministic, one-to-one RAW descriptor
/// correspondences. Geometry only narrows each descriptor search window; the
/// unchanged descriptor distance and ratio limits prevent a peak from creating
/// support by itself.
fn source_raw_correlation_guided_matches(
    source: &ImageInfo,
    target: &ImageInfo,
    proposal: &VirtualTileGlobalProposal,
) -> Vec<Match> {
    let search_radius = f64::from(proposal.left_factor.max(proposal.right_factor)) * 2.0;
    let mut candidates = Vec::<(u32, u64, usize, usize)>::new();
    for (source_index, source_feature) in source.features.iter().enumerate() {
        let Some(predicted) = transformed_point(
            &proposal.left_to_right,
            Point2::new(
                f64::from(source_feature.keypoint.x),
                f64::from(source_feature.keypoint.y),
            ),
        ) else {
            continue;
        };
        let mut nearby = target
            .features
            .iter()
            .enumerate()
            .filter_map(|(target_index, target_feature)| {
                let dx = f64::from(target_feature.keypoint.x) - predicted.x;
                let dy = f64::from(target_feature.keypoint.y) - predicted.y;
                let residual_squared = dx * dx + dy * dy;
                (residual_squared <= search_radius * search_radius).then(|| {
                    (
                        virtual_tile_descriptor_distance(
                            &source_feature.descriptor,
                            &target_feature.descriptor,
                        ),
                        residual_squared,
                        target_index,
                    )
                })
            })
            .collect::<Vec<_>>();
        nearby.sort_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });
        let Some(&(best_distance, residual_squared, target_index)) = nearby.first() else {
            continue;
        };
        let second_distance = nearby.get(1).map_or(u32::MAX, |entry| entry.0);
        if best_distance > VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE
            || f64::from(best_distance)
                >= f64::from(second_distance) * SCALABLE_MATCH_RATIO_THRESHOLD as f64
        {
            continue;
        }
        candidates.push((
            best_distance,
            residual_squared.to_bits(),
            source_index,
            target_index,
        ));
    }
    candidates.sort_unstable();
    let mut used_source = HashSet::new();
    let mut used_target = HashSet::new();
    candidates
        .into_iter()
        .filter_map(|(_, _, index1, index2)| {
            (used_source.insert(index1) && used_target.insert(index2))
                .then_some(Match { index1, index2 })
        })
        .collect()
}

fn source_raw_seed_points(
    source: &ImageInfo,
    target: &ImageInfo,
    matches: &[Match],
) -> Vec<(Point2<f64>, Point2<f64>)> {
    matches
        .iter()
        .filter_map(|matched| {
            let source_point = source.features.get(matched.index1)?.keypoint;
            let target_point = target.features.get(matched.index2)?.keypoint;
            project_point(
                source,
                f64::from(source_point.x) * source.scale_factor,
                f64::from(source_point.y) * source.scale_factor,
                Projection::Planar,
            )
            .zip(project_point(
                target,
                f64::from(target_point.x) * target.scale_factor,
                f64::from(target_point.y) * target.scale_factor,
                Projection::Planar,
            ))
        })
        .collect()
}

fn source_raw_normalize_validated_match(
    relation: &MatchInfo,
    match_direction_reversed: bool,
) -> Option<(Matrix3<f64>, Vec<(Point2<f64>, Point2<f64>)>, usize)> {
    if match_direction_reversed {
        relation.homography.try_inverse().map(|transform| {
            (
                transform,
                relation
                    .points
                    .iter()
                    .map(|&(source, target)| (target, source))
                    .collect(),
                relation.inliers,
            )
        })
    } else {
        Some((
            relation.homography,
            relation.points.clone(),
            relation.inliers,
        ))
    }
}

#[derive(Default)]
struct SourceRawMatcherAttrition {
    left_descriptor_count: usize,
    right_descriptor_count: usize,
    match_direction_reversed: bool,
    descriptor_retrieval_count: usize,
    ratio_filter_count: usize,
    mutual_filter_count: usize,
    ransac_model_fitted: bool,
    ransac_inliers: usize,
    ransac_anchor_transform: Option<[f64; 9]>,
    canonical_conversion_succeeded: bool,
    proposal_candidate_count: usize,
    proposal_score: Option<f64>,
    proposal_match_count: usize,
    proposal_match: Option<MatchInfo>,
    correlation_proposals: Vec<stack_report::SourceRawCorrelationProposalRecord>,
}

/// Independent per-RAW spatial proposal recall. Descriptor correspondences are
/// grouped only by their observed displacement on this RAW pair; no fused tile,
/// station prior, filename, or proposal from another layer is visible here.
/// Returned subsets are merely seeds for the unchanged stable RANSAC and full
/// image validation in `match_image_pair_with_seed`.
fn source_raw_descriptor_proposals(
    matches: &[Match],
    points: &[(Point2<f64>, Point2<f64>)],
    dimensions: (u32, u32),
) -> Vec<(f64, Vec<Match>)> {
    if matches.len() != points.len() || matches.len() < SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION {
        return Vec::new();
    }
    let bin_size = f64::from(dimensions.0.max(dimensions.1).max(1)) * 0.01;
    let mut bins = BTreeMap::<(i32, i32), Vec<usize>>::new();
    for (index, (source, target)) in points.iter().enumerate() {
        let displacement = target - source;
        if displacement.x.is_finite() && displacement.y.is_finite() {
            bins.entry((
                (displacement.x / bin_size).floor() as i32,
                (displacement.y / bin_size).floor() as i32,
            ))
            .or_default()
            .push(index);
        }
    }
    let mut candidates = bins
        .keys()
        .map(|&(bin_x, bin_y)| {
            let mut indices = Vec::new();
            for y in bin_y - 1..=bin_y + 1 {
                for x in bin_x - 1..=bin_x + 1 {
                    indices.extend(bins.get(&(x, y)).into_iter().flatten().copied());
                }
            }
            indices.sort_unstable();
            indices.dedup();
            ((bin_x, bin_y), indices)
        })
        .filter(|(_, indices)| indices.len() >= SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION)
        .collect::<Vec<_>>();
    candidates.sort_by(|(left_bin, left), (right_bin, right)| {
        right
            .len()
            .cmp(&left.len())
            .then_with(|| left_bin.cmp(right_bin))
    });
    let mut seen = HashSet::new();
    candidates
        .into_iter()
        .filter_map(|(_, indices)| {
            let proposal = indices
                .into_iter()
                .map(|index| matches[index])
                .collect::<Vec<_>>();
            let provenance = proposal
                .iter()
                .map(|matched| (matched.index1, matched.index2))
                .collect::<Vec<_>>();
            seen.insert(provenance)
                .then_some((proposal.len() as f64 / matches.len() as f64, proposal))
        })
        .take(8)
        .collect()
}

fn source_raw_record_correlation_oracle(
    source: &ImageInfo,
    target: &ImageInfo,
    seed_threshold: f64,
    attrition: &mut SourceRawMatcherAttrition,
) {
    let proposals = SourceRawCorrelationPyramid::from_image(source)
        .zip(SourceRawCorrelationPyramid::from_image(target))
        .map(|(source_pyramid, target_pyramid)| {
            source_raw_correlation_proposals(&source_pyramid, &target_pyramid)
        })
        .unwrap_or_default();
    attrition.proposal_candidate_count = proposals.len();
    for (rank, correlation) in proposals.into_iter().enumerate() {
        let proposal = correlation.proposal;
        let guided = source_raw_correlation_guided_matches(source, target, &proposal);
        let seed_points = source_raw_seed_points(source, target, &guided);
        let stable_inliers = (seed_points.len() >= SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION)
            .then(|| {
                processing::find_homography_ransac_points_stable_with_min_inliers(
                    &seed_points,
                    seed_threshold,
                    SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION,
                )
            })
            .flatten()
            .map_or(0, |(_, inliers)| inliers.len());
        let validation = (stable_inliers >= SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION)
            .then(|| {
                match_image_pair_with_seed(
                    source,
                    target,
                    Projection::Planar,
                    BlendMode::FocusStack,
                    AlignmentMode::Auto,
                    true,
                    false,
                    false,
                    Some(guided.clone()),
                )
            })
            .flatten();
        let authoritative = validation.as_ref().filter(|relation| {
            !relation.sequence_bridge
                && !relation.coarse_bridge
                && relation.points.len() >= FOCUS_MODEL_MIN_INLIERS
        });
        let validation_stage = if guided.len() < SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION {
            "guided_feature_matching"
        } else if stable_inliers < SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION {
            "stable_ransac"
        } else if validation.is_none() {
            "original_full_resolution_validation"
        } else if authoritative.is_none() {
            "original_full_resolution_non_authoritative"
        } else {
            "accepted"
        };
        attrition
            .correlation_proposals
            .push(stack_report::SourceRawCorrelationProposalRecord {
                rank,
                score: proposal.score,
                second_peak_margin: correlation.second_peak_margin,
                overlap_samples: proposal.overlap_samples,
                left_factor: proposal.left_factor,
                right_factor: proposal.right_factor,
                translation_x_px: proposal.left_to_right[(0, 2)],
                translation_y_px: proposal.left_to_right[(1, 2)],
                scale: proposal.scale,
                rotation_degrees: proposal.rotation_degrees,
                guided_match_count: guided.len(),
                stable_ransac_inliers: stable_inliers,
                full_resolution_validated: authoritative.is_some(),
                validation_stage: validation_stage.to_string(),
                match_source: absolute_source_path(&source.filename),
                match_target: absolute_source_path(&target.filename),
                source_to_target_transform: matrix3_row_major(&proposal.left_to_right),
            });
        if attrition.proposal_match.is_none()
            && let Some(validated) = authoritative.cloned()
        {
            attrition.proposal_score = Some(proposal.score);
            attrition.proposal_match_count = guided.len();
            attrition.proposal_match = Some(validated);
        }
    }
}

/// Replays only the immutable descriptor and provisional geometry stages used
/// by the stable RAW-pair matcher. The final validation result still comes from
/// `matches`, so this oracle cannot create or promote a relation.
fn source_raw_matcher_attrition(
    images: &[ImageInfo],
    left_source: usize,
    right_source: usize,
    left_group: &FocusCaptureGroup,
    right_group: &FocusCaptureGroup,
) -> SourceRawMatcherAttrition {
    let left_image = &images[left_source];
    let right_image = &images[right_source];
    let (match_source, match_target, reversed) =
        canonical_match_direction(images, left_source, right_source);
    let source = &images[match_source];
    let target = &images[match_target];
    let descriptor = processing::match_features_with_ratio_attrition(
        &source.features,
        &target.features,
        SCALABLE_MATCH_RATIO_THRESHOLD,
    );
    let mut attrition = SourceRawMatcherAttrition {
        left_descriptor_count: left_image.features.len(),
        right_descriptor_count: right_image.features.len(),
        match_direction_reversed: reversed,
        descriptor_retrieval_count: descriptor.retrieval_count,
        ratio_filter_count: descriptor.ratio_filter_count,
        mutual_filter_count: descriptor.mutual_matches.len(),
        ..Default::default()
    };
    let source_points = source
        .features
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let target_points = target
        .features
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let projected = descriptor
        .mutual_matches
        .iter()
        .filter_map(|matched| {
            let source_point = source_points.get(matched.index1)?;
            let target_point = target_points.get(matched.index2)?;
            project_point(
                source,
                source_point.x as f64 * source.scale_factor,
                source_point.y as f64 * source.scale_factor,
                Projection::Planar,
            )
            .zip(project_point(
                target,
                target_point.x as f64 * target.scale_factor,
                target_point.y as f64 * target.scale_factor,
                Projection::Planar,
            ))
        })
        .collect::<Vec<_>>();
    let seed_threshold =
        FULL_RES_RANSAC_INLIER_THRESHOLD.max(source.scale_factor.max(target.scale_factor) * 3.5);
    let stable = (projected.len() >= SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION)
        .then(|| {
            processing::find_homography_ransac_points_stable_with_min_inliers(
                &projected,
                seed_threshold,
                SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION,
            )
        })
        .flatten();
    source_raw_record_correlation_oracle(source, target, seed_threshold, &mut attrition);
    let Some((match_transform, inliers)) = stable else {
        return attrition;
    };
    attrition.ransac_model_fitted = true;
    attrition.ransac_inliers = inliers.len();
    let left_to_right = if reversed {
        match_transform.try_inverse()
    } else {
        Some(match_transform)
    };
    let anchor_transform = left_to_right
        .zip(left_group.local_to_anchor.get(&left_source))
        .zip(right_group.local_to_anchor.get(&right_source))
        .and_then(|((left_to_right, left_local), right_local)| {
            left_local
                .try_inverse()
                .map(|left_inverse| right_local * left_to_right * left_inverse)
        });
    if let Some(anchor_transform) = anchor_transform {
        attrition.canonical_conversion_succeeded = true;
        attrition.ransac_anchor_transform = Some(matrix3_row_major(&anchor_transform));
    }
    attrition
}

/// Experiment E's read-only layer oracle. It consumes each Source_RAW's own
/// luminance/gradient pyramid, immutable descriptors, already fitted RAW-pair
/// models, and each station's verified local-to-anchor maps. No result is
/// inserted into the Virtual_Tile graph or prior repair.
fn source_raw_layer_pair_diagnostics(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    groups: &[FocusCaptureGroup],
) -> Vec<stack_report::SourceRawPairDiagnosticRecord> {
    let mut diagnostics = Vec::new();
    for left_station in 0..groups.len() {
        for right_station in left_station + 1..groups.len() {
            let left_group = &groups[left_station];
            let right_group = &groups[right_station];
            let mut layer_records = Vec::new();
            let mut models = Vec::new();
            for &left_source in &left_group.members {
                for &right_source in &right_group.members {
                    let record_index = layer_records.len();
                    let attrition = source_raw_matcher_attrition(
                        images,
                        left_source,
                        right_source,
                        left_group,
                        right_group,
                    );
                    let key = (left_source.min(right_source), left_source.max(right_source));
                    let fitted = matches.get(&key).filter(|relation| {
                        !relation.sequence_bridge
                            && !relation.coarse_bridge
                            && relation.points.len() >= FOCUS_MODEL_MIN_INLIERS
                    });
                    let full_resolution_validated =
                        fitted.is_some() || attrition.proposal_match.is_some();
                    let failure_stage = if attrition.left_descriptor_count == 0
                        || attrition.right_descriptor_count == 0
                    {
                        "descriptor_extraction"
                    } else if attrition.descriptor_retrieval_count == 0 {
                        "descriptor_retrieval"
                    } else if fitted.is_some() {
                        "accepted"
                    } else if attrition.proposal_match.is_some() {
                        "accepted_by_raw_correlation_oracle"
                    } else if !attrition.correlation_proposals.is_empty() {
                        "raw_correlation_oracle_no_full_validation"
                    } else if attrition.mutual_filter_count
                        < SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION
                    {
                        "ratio_mutual_filtering"
                    } else if !attrition.ransac_model_fitted {
                        "model_ransac"
                    } else if attrition.ransac_model_fitted
                        && !attrition.canonical_conversion_succeeded
                    {
                        "canonical_local_to_anchor"
                    } else if !full_resolution_validated {
                        "full_resolution_validation"
                    } else {
                        "accepted"
                    };
                    let mut record = stack_report::SourceRawLayerModelRecord {
                        left_source: absolute_source_path(&images[left_source].filename),
                        right_source: absolute_source_path(&images[right_source].filename),
                        left_descriptor_count: attrition.left_descriptor_count,
                        right_descriptor_count: attrition.right_descriptor_count,
                        match_direction_reversed: attrition.match_direction_reversed,
                        descriptor_retrieval_count: attrition.descriptor_retrieval_count,
                        ratio_filter_count: attrition.ratio_filter_count,
                        mutual_filter_count: attrition.mutual_filter_count,
                        ransac_model_fitted: attrition.ransac_model_fitted,
                        ransac_inliers: attrition.ransac_inliers,
                        ransac_anchor_transform: attrition.ransac_anchor_transform,
                        canonical_conversion_succeeded: attrition.canonical_conversion_succeeded,
                        full_resolution_validated,
                        failure_stage: failure_stage.to_string(),
                        model_provenance: if fitted.is_some() {
                            "stable_matcher".to_string()
                        } else if attrition.proposal_match.is_some() {
                            "per_raw_coarse_luminance_gradient_correlation".to_string()
                        } else {
                            "stable_matcher".to_string()
                        },
                        proposal_seeded: attrition.proposal_match.is_some(),
                        proposal_score: attrition.proposal_score,
                        proposal_candidate_count: attrition.proposal_candidate_count,
                        proposal_match_count: attrition.proposal_match_count,
                        proposal_ransac_inliers: attrition
                            .correlation_proposals
                            .iter()
                            .filter(|proposal| proposal.full_resolution_validated)
                            .map(|proposal| proposal.stable_ransac_inliers)
                            .next()
                            .unwrap_or_default(),
                        correlation_proposals: attrition.correlation_proposals.clone(),
                        ..Default::default()
                    };
                    let normalized = if let Some(relation) = fitted {
                        if left_source == key.0 {
                            Some((
                                relation.homography,
                                relation.points.clone(),
                                relation.inliers,
                            ))
                        } else {
                            relation.homography.try_inverse().map(|transform| {
                                (
                                    transform,
                                    relation
                                        .points
                                        .iter()
                                        .map(|&(source, target)| (target, source))
                                        .collect(),
                                    relation.inliers,
                                )
                            })
                        }
                    } else if let Some(relation) = attrition.proposal_match.as_ref() {
                        source_raw_normalize_validated_match(
                            relation,
                            attrition.match_direction_reversed,
                        )
                    } else {
                        None
                    };
                    if let Some((left_to_right, raw_points, relation_inliers)) = normalized {
                        let left_local = left_group.local_to_anchor.get(&left_source);
                        let right_local = right_group.local_to_anchor.get(&right_source);
                        if let Some((left_local, left_local_inverse, right_local)) = left_local
                            .and_then(|transform| {
                                transform.try_inverse().map(|inverse| (transform, inverse))
                            })
                            .zip(right_local)
                            .map(|((left, inverse), right)| (left, inverse, right))
                        {
                            let transform = right_local * left_to_right * left_local_inverse;
                            let anchor_points = raw_points
                                .iter()
                                .filter_map(|&(left, right)| {
                                    transformed_point(left_local, left)
                                        .zip(transformed_point(right_local, right))
                                })
                                .collect::<Vec<_>>();
                            let median_error_px =
                                median_symmetric_error(&transform, &anchor_points);
                            let scale_ratio = homography_scale_ratio(&transform);
                            if anchor_points.len() >= FOCUS_MODEL_MIN_INLIERS
                                && median_error_px.is_finite()
                                && scale_ratio.is_some_and(f64::is_finite)
                            {
                                let spatial_support = station_relation_spatial_support(
                                    &anchor_points,
                                    &transform,
                                    images[left_group.anchor].dimensions(),
                                    images[right_group.anchor].dimensions(),
                                );
                                record.model_fitted = true;
                                record.canonical_conversion_succeeded = true;
                                if record.ransac_anchor_transform.is_none() {
                                    record.ransac_anchor_transform =
                                        Some(matrix3_row_major(&transform));
                                }
                                record.inliers = relation_inliers.max(anchor_points.len());
                                record.median_error_px = median_error_px;
                                record.scale_ratio = scale_ratio.unwrap_or_default();
                                record.spatial_support = spatial_support;
                                record.anchor_transform = matrix3_row_major(&transform);
                                models.push(SourceRawLayerDiagnosticModel {
                                    left_source,
                                    right_source,
                                    record_index,
                                    transform,
                                    spatial_support,
                                });
                            }
                        }
                    }
                    layer_records.push(record);
                }
            }
            let dimensions = images[left_group.anchor].dimensions();
            let consensus_clusters =
                source_raw_consensus_clusters(&models, &mut layer_records, dimensions);
            diagnostics.push(stack_report::SourceRawPairDiagnosticRecord {
                left: left_station,
                right: right_station,
                layer_models: layer_records,
                consensus_clusters,
            });
        }
    }
    diagnostics
}

/// Median Sharpness_Score over the valid pixels of one Source_RAW (需求 2.1).
///
/// Measured on the already prepared analysis image, resampled to the
/// Intra_Station analysis budget of 需求 2.2 so the number is a property of the
/// photograph rather than of whichever alignment budget the run happened to
/// pick.  The score is still the unbounded acutance ratio; task 7.6 normalises
/// it monotonically, which cannot change which frame is sharpest.
fn focus_frame_median_sharpness(image: &ImageInfo) -> f64 {
    let analysis = &image.alignment_image;
    let (width, height) = (analysis.width(), analysis.height());
    let spacing = intra_station::analysis_probe_spacing(width.max(height));
    intra_station::median_valid_sharpness(width, height, |x, y| {
        crate::panorama_utils::mosaic::gray_analysis_sharpness(analysis, x, y, spacing)
    })
}

/// Drop only clear outliers in sharpness for a flat-artwork scan. A low score
/// alone is not enough to reject a mostly blank paper tile, so require the
/// frame to be both in the bottom fifth of the selection and less than 55% of
/// the robust median. This keeps low-texture but usable edge coverage while
/// removing visibly defocused frames that would soften an otherwise sharp
/// mosaic.
fn filter_flat_artwork_blurry_sources(images: &mut Vec<ImageInfo>) {
    if images.len() < 4 {
        return;
    }
    let mut scores = images
        .iter()
        .map(|image| (focus_frame_median_sharpness(image), image.filename.clone()))
        .collect::<Vec<_>>();
    scores.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let median = scores[scores.len() / 2].0;
    if !median.is_finite() || median <= 0.0 {
        return;
    }
    let bottom_fifth_index = ((scores.len() - 1) * 20) / 100;
    let bottom_fifth = scores[bottom_fifth_index].0;
    let threshold = median * 0.55;
    let rejected = scores
        .iter()
        .filter(|(score, _)| *score <= bottom_fifth && *score < threshold)
        .map(|(_, filename)| filename.clone())
        .collect::<HashSet<_>>();
    if rejected.is_empty() || rejected.len() >= images.len().saturating_sub(2) {
        return;
    }
    for image in images
        .iter()
        .filter(|image| rejected.contains(&image.filename))
    {
        let score = focus_frame_median_sharpness(image);
        println!(
            "  - Flat-artwork blur filter: excluding '{}' (median sharpness {:.4}, selection median {:.4})",
            image.filename, score, median
        );
    }
    images.retain(|image| !rejected.contains(&image.filename));
    println!(
        "  - Flat-artwork blur filter retained {} of {} source images",
        images.len(),
        images.len() + rejected.len()
    );
}

fn absolute_source_path(path: &str) -> String {
    let path = Path::new(path);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(path)
    };
    absolute.to_string_lossy().into_owned()
}

/// Refine native camera stations jointly, then validate the exact bounded
/// sampling field on points excluded from both global and local fitting.
/// Rejecting a relation never removes a source or disconnects the graph.
fn refine_virtual_tile_joint_geometry(
    tiles: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial: &HashMap<usize, Matrix3<f64>>,
    closure: &mut stack_report::ClosureReport,
    report: &mut stack_report::StationRelationsReport,
) -> Option<HashMap<usize, Matrix3<f64>>> {
    if tiles.len() < 3
        || report.accepted.len() < tiles.len()
        || closure.status == stack_report::ClosureStatus::Converged
    {
        return None;
    }
    let poses = tiles
        .iter()
        .map(|tile| initial.get(&tile.id).copied())
        .collect::<Option<Vec<_>>>()?;
    let dimensions = tiles.iter().map(ImageInfo::dimensions).collect::<Vec<_>>();
    let mut pairs = report
        .accepted
        .iter()
        .map(|r| (r.left.min(r.right), r.left.max(r.right)))
        .collect::<Vec<_>>();
    pairs.sort_unstable();
    pairs.dedup();
    // Coarse matches have no native reverse-check evidence. They retain the
    // established matrix-only path, including all small property fixtures.
    if pairs.iter().any(|pair| {
        matches
            .get(pair)
            .is_none_or(|m| m.native_verified_points.len() < STATION_RELATION_MIN_INLIERS)
    }) {
        return None;
    }
    let mut limits = vec![CLOSURE_MAX_CORNER_CORRECTION_PX; tiles.len()];
    for &(left, right) in &pairs {
        let relation = matches.get(&(left, right))?;
        for (station, source_side) in [(left, true), (right, false)] {
            let polygon = station_relation_overlap_polygon(
                &relation.homography,
                dimensions[left],
                dimensions[right],
                source_side,
            )?;
            let short = projected_polygon_short_side(&polygon, &poses[station])?;
            limits[station] =
                limits[station].min(short * CLOSURE_OVERLAP_SHORT_SIDE_CORRECTION_RATIO);
        }
    }
    let mut rejected = Vec::new();
    #[cfg(test)]
    if let Some(directory) = std::env::var_os("RAW_EDITOR_NATIVE_PROOF_DIAGNOSTICS_DIR")
        .or_else(|| std::env::var_os("RAW_EDITOR_VIRTUAL_TILE_DIAGNOSTICS_DIR"))
    {
        let records = report.accepted.iter().map(|r| {
            let key=(r.left.min(r.right),r.left.max(r.right));
            let relation=&matches[&key];
            serde_json::json!({"left":key.0,"right":key.1,"authoritative":true,
                "inliers":relation.inliers,"error":r.median_error_px,"scale":r.scale_ratio,
                "diagnostic":{"hull_support":r.spatial_support},
                "homography":relation.homography.as_slice(),
                "points":relation.points.iter().map(|(a,b)|[a.x,a.y,b.x,b.y]).collect::<Vec<_>>(),
                "native_verified_points":relation.native_verified_points.iter().map(|(a,b,e)|[a.x,a.y,b.x,b.y,*e]).collect::<Vec<_>>()})
        }).collect::<Vec<_>>();
        let snapshot = serde_json::json!({"poses":poses.iter().map(|h|(0..3).map(|r|(0..3).map(|c|h[(r,c)]).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>(),"dimensions":dimensions,"limits":limits,"records":records});
        let path = Path::new(&directory).join("joint-production-input.json");
        std::fs::create_dir_all(&directory).unwrap();
        if let Err(error) = std::fs::write(&path, serde_json::to_vec(&snapshot).unwrap()) {
            eprintln!(
                "Could not save native joint diagnostic {}: {error}",
                path.display()
            );
        }
    }
    // Use the complete native graph once to initialize the projective part.
    // Re-solving this gauge after every rejected local constraint can drift
    // through weakly observed corners. The final field is fitted and checked
    // only on retained relations; this proposal never establishes acceptance.
    let proposal_relations = pairs
        .iter()
        .map(|&(left, right)| joint_warp::Relation {
            left,
            right,
            points: matches[&(left, right)].points.clone(),
        })
        .collect::<Vec<_>>();
    let scale = dimensions.iter().map(|&(w, h)| w.max(h)).max()? as f64;
    let proposal = joint_warp::projective_proposal(&poses, &proposal_relations, 0, scale)?;
    for attempt in 0..8 {
        let relations = pairs
            .iter()
            .map(|&(left, right)| joint_warp::Relation {
                left,
                right,
                points: matches[&(left, right)].points.clone(),
            })
            .collect::<Vec<_>>();
        let fitted = joint_warp::fit(&proposal, &dimensions, &relations, 0)?;
        let mut verified = vec![Vec::new(); tiles.len()];
        for &(left, right) in &pairs {
            for &(a, b, error) in &matches[&(left, right)].native_verified_points {
                verified[left].push(joint_warp::VerifiedSource {
                    point: a,
                    round_trip_error_px: error,
                });
                verified[right].push(joint_warp::VerifiedSource {
                    point: b,
                    round_trip_error_px: error,
                });
            }
        }
        let mut bounded = joint_warp::materialize(
            &proposal,
            &fitted,
            &dimensions,
            &poses,
            &limits,
            &relations,
            &verified,
        )?;
        let record = |v: &joint_warp::PairValidation| stack_report::ClosurePairValidationRecord {
            left: v.left,
            right: v.right,
            count: v.count,
            median_px: v.median,
            p95_px: v.p95,
        };
        let failed = bounded
            .validation
            .iter()
            .filter(|v| !v.p95.is_finite() || v.p95 > CLOSURE_DIRECT_PAIR_P95_LIMIT_PX)
            .collect::<Vec<_>>();
        if failed.is_empty() {
            // A nonlinear correction does not waive the station-pose scale
            // or orientation gates used by the matrix-only solver.
            let station_groups = (0..tiles.len())
                .map(|station| FocusCaptureGroup {
                    members: vec![station],
                    anchor: station,
                    local_to_anchor: HashMap::from([(station, Matrix3::identity())]),
                })
                .collect::<Vec<_>>();
            if bounded
                .poses
                .iter()
                .zip(&dimensions)
                .any(|(pose, &size)| !homography_preserves_focus_orientation(pose, size))
                || !station_pose_candidate_has_bounded_local_scale(
                    tiles,
                    &station_groups,
                    &bounded.poses,
                    "native_joint_field",
                )
            {
                return None;
            }
            let distribution = residual_distribution(bounded.validation_errors.clone());
            let mut errors = bounded.validation_errors.clone();
            errors.sort_by(f64::total_cmp);
            closure.global_before_residual_median_px = Some(closure.residual_median_px);
            closure.global_before_residual_p95_px = Some(closure.residual_p95_px);
            closure.geometry_model = "bounded_station_field".to_string();
            closure.nonlinear_initialization_constraints = proposal_relations.len();
            closure.nonlinear_refit_passes = attempt + 1;
            closure.nonlinear_validation = bounded.validation.iter().map(record).collect();
            closure.rejected_nonlinear_constraints = rejected;
            closure.residual_median_px = distribution.median;
            closure.residual_p95_px = errors[(errors.len() * 95).div_ceil(100) - 1];
            closure.max_direct_pair_p95_px =
                bounded.validation.iter().map(|v| v.p95).fold(0.0, f64::max);
            closure.max_corner_correction_px = bounded.maximum_corner_motion;
            closure.status = stack_report::ClosureStatus::Converged;
            closure.fallback_reason = None;
            closure.offending_pair = None;
            closure.participating_constraints = pairs.len();
            closure.constraint_weights = bounded
                .validation
                .iter()
                .map(|v| stack_report::ClosureConstraintWeightRecord {
                    left: v.left,
                    right: v.right,
                    residual_px: v.median,
                    weight: 1.0,
                })
                .collect();
            report.accepted.retain(|r| {
                pairs
                    .binary_search(&(r.left.min(r.right), r.left.max(r.right)))
                    .is_ok()
            });
            report.rejected_count += closure.rejected_nonlinear_constraints.len();
            *report
                .rejected_by_reason
                .entry("native_joint_validation_p95".to_string())
                .or_default() += closure.rejected_nonlinear_constraints.len() as u64;
            for (station, tile) in tiles.iter().enumerate() {
                bounded.residual.register_station_alias(tile.id, station);
            }
            residual_warp::record_run_model(&bounded.residual);
            println!(
                "  - Bounded native station field: {} relations, independent P95 {:.3}px, maximum corner {:.3}px",
                pairs.len(),
                closure.max_direct_pair_p95_px,
                bounded.maximum_corner_motion
            );
            return Some(
                tiles
                    .iter()
                    .zip(bounded.poses)
                    .map(|(tile, pose)| (tile.id, pose))
                    .collect(),
            );
        }
        let failed_pairs = failed
            .iter()
            .map(|v| (v.left, v.right))
            .collect::<BTreeSet<_>>();
        let retained = pairs
            .iter()
            .copied()
            .filter(|pair| !failed_pairs.contains(pair))
            .collect::<Vec<_>>();
        let mut graph = Dsu::new(tiles.len());
        for &(left, right) in &retained {
            graph.union(left, right);
        }
        let root = graph.find(0);
        if retained.len() < tiles.len()
            || (0..tiles.len()).any(|station| graph.find(station) != root)
        {
            return None;
        }
        rejected.extend(failed.into_iter().map(record));
        pairs = retained;
    }
    None
}

fn focus_auto_order_motion_scale(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> f64 {
    let mut minimum_motion_by_image = vec![f64::INFINITY; images.len()];
    // Direct map iteration is order safe here: the loop only takes a per-image
    // minimum and the result is the median of a `total_cmp`-sorted vector, both of
    // which are independent of the visiting order (需求 14.6).
    for (&(source_index, target_index), match_info) in matches {
        let Some(source) = images.get(source_index) else {
            continue;
        };
        let Some(target) = images.get(target_index) else {
            continue;
        };
        let Some(motion) = focus_match_center_motion_ratio(source, target, match_info) else {
            continue;
        };
        if motion <= FOCUS_AUTO_ORDER_MOTION_SCALE_FLOOR * 0.1 {
            continue;
        }
        minimum_motion_by_image[source_index] = minimum_motion_by_image[source_index].min(motion);
        minimum_motion_by_image[target_index] = minimum_motion_by_image[target_index].min(motion);
    }
    let mut finite_minimums = minimum_motion_by_image
        .into_iter()
        .filter(|motion| motion.is_finite())
        .collect::<Vec<_>>();
    median_value(&mut finite_minimums).unwrap_or(0.0)
}

fn focus_auto_order_edge_weight(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
    motion_scale: f64,
) -> f64 {
    let support = (match_info.inliers.max(1) as f64).sqrt();
    let precision = if match_info.points.len() >= 4 {
        let error = median_symmetric_error(&match_info.homography, &match_info.points);
        if error.is_finite() {
            1.0 / (1.0 + error / FOCUS_MODEL_INLIER_THRESHOLD)
        } else {
            0.0
        }
    } else {
        1.0
    };
    let motion_factor = if motion_scale > FOCUS_AUTO_ORDER_MOTION_SCALE_FLOOR {
        let motion =
            focus_match_center_motion_ratio(source, target, match_info).unwrap_or(f64::INFINITY);
        if motion.is_finite() {
            1.0 / (1.0 + (motion / motion_scale).powf(FOCUS_AUTO_ORDER_MOTION_EXPONENT))
        } else {
            0.0
        }
    } else {
        1.0
    };
    support * precision * motion_factor
}

fn focus_stack_link_transform(
    matches: &HashMap<(usize, usize), MatchInfo>,
    from: usize,
    to: usize,
) -> Option<(usize, Matrix3<f64>)> {
    if let Some(match_info) = matches.get(&(from, to)) {
        return Some((match_info.inliers, match_info.homography));
    }
    matches.get(&(to, from)).and_then(|match_info| {
        match_info
            .homography
            .try_inverse()
            .map(|transform| (match_info.inliers, transform))
    })
}

fn focus_cycle_edge_factor(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    source_index: usize,
    target_index: usize,
) -> f64 {
    let Some((_, direct)) = focus_stack_link_transform(matches, source_index, target_index) else {
        return 0.0;
    };
    let Some(source) = images.get(source_index) else {
        return 0.0;
    };
    let Some(target) = images.get(target_index) else {
        return 0.0;
    };
    let samples = [
        Point2::new(source.width as f64 * 0.2, source.height as f64 * 0.2),
        Point2::new(source.width as f64 * 0.8, source.height as f64 * 0.2),
        Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5),
        Point2::new(source.width as f64 * 0.2, source.height as f64 * 0.8),
        Point2::new(source.width as f64 * 0.8, source.height as f64 * 0.8),
    ];
    let normalization = target.width.max(target.height).max(1) as f64;
    let mut consistent_weight = 0.0;
    let mut conflicting_weight = 0.0;
    for middle_index in 0..images.len() {
        if middle_index == source_index || middle_index == target_index {
            continue;
        }
        let Some((source_support, source_to_middle)) =
            focus_stack_link_transform(matches, source_index, middle_index)
        else {
            continue;
        };
        let Some((target_support, target_to_middle)) =
            focus_stack_link_transform(matches, target_index, middle_index)
        else {
            continue;
        };
        let Some(middle_to_target) = target_to_middle.try_inverse() else {
            continue;
        };
        let via_middle = middle_to_target * source_to_middle;
        let mut errors = samples
            .iter()
            .filter_map(|&point| {
                let direct_point = transformed_point(&direct, point)?;
                let indirect_point = transformed_point(&via_middle, point)?;
                let error = (direct_point - indirect_point).norm() / normalization;
                error.is_finite().then_some(error)
            })
            .collect::<Vec<_>>();
        let Some(error) = median_value(&mut errors) else {
            continue;
        };
        let weight = source_support.min(target_support).max(1) as f64;
        if error <= 0.025 {
            consistent_weight += weight.sqrt();
        } else if error >= 0.075 {
            conflicting_weight += weight.sqrt();
        }
    }
    if consistent_weight == 0.0 && conflicting_weight == 0.0 {
        return 1.0;
    }
    ((consistent_weight + 4.0) / (conflicting_weight + 4.0)).clamp(0.15, 3.0)
}

fn focus_auto_order_edge_weight_between(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    left: usize,
    right: usize,
    motion_scale: f64,
) -> Option<f64> {
    if let Some(match_info) = matches.get(&(left, right)) {
        return Some(focus_auto_order_edge_weight(
            &images[left],
            &images[right],
            match_info,
            motion_scale,
        ));
    }
    matches.get(&(right, left)).map(|match_info| {
        focus_auto_order_edge_weight(&images[right], &images[left], match_info, motion_scale)
    })
}

fn focus_auto_order_path_score(
    order: &[usize],
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    motion_scale: f64,
) -> (usize, f64) {
    let mut matched_edges = 0usize;
    let mut score = 0.0;
    for pair in order.windows(2) {
        if let Some(edge_weight) =
            focus_auto_order_edge_weight_between(images, matches, pair[0], pair[1], motion_scale)
        {
            matched_edges += 1;
            score += edge_weight;
        } else {
            score -= FOCUS_AUTO_ORDER_MISSING_EDGE_PENALTY;
        }
    }
    (matched_edges, score)
}

fn focus_auto_order_path_motion_cost(
    order: &[usize],
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> f64 {
    order
        .windows(2)
        .filter_map(|pair| {
            if let Some(match_info) = matches.get(&(pair[0], pair[1])) {
                focus_match_center_motion_ratio(&images[pair[0]], &images[pair[1]], match_info)
            } else {
                matches.get(&(pair[1], pair[0])).and_then(|match_info| {
                    focus_match_center_motion_ratio(&images[pair[1]], &images[pair[0]], match_info)
                })
            }
        })
        .sum()
}

fn focus_auto_order_path_max_motion(
    order: &[usize],
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> f64 {
    let mut maximum = 0.0f64;
    for pair in order.windows(2) {
        let Some(motion) = matches
            .get(&(pair[0], pair[1]))
            .and_then(|match_info| {
                focus_match_center_motion_ratio(&images[pair[0]], &images[pair[1]], match_info)
            })
            .or_else(|| {
                matches.get(&(pair[1], pair[0])).and_then(|match_info| {
                    focus_match_center_motion_ratio(&images[pair[1]], &images[pair[0]], match_info)
                })
            })
        else {
            return f64::INFINITY;
        };
        maximum = maximum.max(motion);
    }
    maximum
}

fn canonical_focus_order_direction(order: &[usize], images: &[ImageInfo]) -> Vec<usize> {
    if order.len() < 2 {
        return order.to_vec();
    }
    let first = order[0];
    let last = *order
        .last()
        .expect("an order with two images has a last image");
    let direction = natural_path_cmp(&images[first].filename, &images[last].filename)
        .then_with(|| first.cmp(&last));
    if direction == Ordering::Greater {
        order.iter().rev().copied().collect()
    } else {
        order.to_vec()
    }
}

fn focus_auto_order_adjacency(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    motion_scale: f64,
) -> Vec<Vec<(usize, f64)>> {
    let mut adjacency = vec![Vec::new(); images.len()];
    // Direct map iteration is order safe here: every neighbour list is sorted
    // below by `(weight desc, natural path, index)`, which is a total order, so the
    // insertion order cannot survive into the traversal (需求 14.6).
    for (&(left, right), match_info) in matches {
        if left >= images.len() || right >= images.len() || left == right {
            continue;
        }
        let weight =
            focus_auto_order_edge_weight(&images[left], &images[right], match_info, motion_scale);
        if !weight.is_finite() {
            continue;
        }
        adjacency[left].push((right, weight));
        adjacency[right].push((left, weight));
    }
    for neighbors in &mut adjacency {
        neighbors.sort_by(|(left_index, left_weight), (right_index, right_weight)| {
            right_weight
                .total_cmp(left_weight)
                .then_with(|| {
                    natural_path_cmp(
                        &images[*left_index].filename,
                        &images[*right_index].filename,
                    )
                })
                .then_with(|| left_index.cmp(right_index))
        });
    }
    adjacency
}

fn focus_auto_order_greedy_path(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    motion_scale: f64,
) -> Vec<usize> {
    if images.len() < 2 {
        return (0..images.len()).collect();
    }
    let adjacency = focus_auto_order_adjacency(images, matches, motion_scale);
    let mut best_order = Vec::new();
    let mut best_matched_edges = 0usize;
    let mut best_score = f64::NEG_INFINITY;

    for start in 0..images.len() {
        let mut order = vec![start];
        let mut used = vec![false; images.len()];
        used[start] = true;
        while order.len() < images.len() {
            let current = *order
                .last()
                .expect("a greedy path always has a current image");
            let next = adjacency[current]
                .iter()
                .filter(|(neighbor, _)| !used[*neighbor])
                .max_by(|(left, left_weight), (right, right_weight)| {
                    let left_future = adjacency[*left]
                        .iter()
                        .filter(|(neighbor, _)| !used[*neighbor] && *neighbor != current)
                        .map(|(_, weight)| *weight)
                        .max_by(f64::total_cmp)
                        .unwrap_or(0.0);
                    let right_future = adjacency[*right]
                        .iter()
                        .filter(|(neighbor, _)| !used[*neighbor] && *neighbor != current)
                        .map(|(_, weight)| *weight)
                        .max_by(f64::total_cmp)
                        .unwrap_or(0.0);
                    (left_weight + left_future * 0.25)
                        .total_cmp(&(right_weight + right_future * 0.25))
                        .then_with(|| {
                            natural_path_cmp(&images[*left].filename, &images[*right].filename)
                                .reverse()
                        })
                })
                .map(|(neighbor, _)| *neighbor)
                .or_else(|| {
                    (0..images.len())
                        .filter(|neighbor| !used[*neighbor])
                        .min_by(|left, right| {
                            natural_path_cmp(&images[*left].filename, &images[*right].filename)
                                .then_with(|| left.cmp(right))
                        })
                });
            let Some(next) = next else {
                break;
            };
            used[next] = true;
            order.push(next);
        }

        let (matched_edges, score) =
            focus_auto_order_path_score(&order, images, matches, motion_scale);
        let motion_cost = focus_auto_order_path_motion_cost(&order, images, matches);
        let best_motion_cost = focus_auto_order_path_motion_cost(&best_order, images, matches);
        if best_order.is_empty()
            || matched_edges > best_matched_edges
            || (matched_edges == best_matched_edges
                && (motion_cost < best_motion_cost
                    || (motion_cost == best_motion_cost && score > best_score)))
        {
            best_order = order;
            best_matched_edges = matched_edges;
            best_score = score;
        }
    }
    best_order
}

fn build_focus_sequence_homographies(
    order: &[usize],
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    motion_scale: f64,
) -> Option<HashMap<usize, Matrix3<f64>>> {
    let first = *order.first()?;
    let mut global_homographies = HashMap::new();
    global_homographies.insert(first, Matrix3::identity());

    for (position, &current) in order.iter().enumerate().skip(1) {
        let mut best_link: Option<(f64, f64, usize, f64, usize, Matrix3<f64>)> = None;
        // Preserve the capture path. Searching every earlier frame lets a
        // repeated character/seal beat the real neighbour and teleports the
        // next tile to a distant row. Only the immediate predecessor and one
        // frame of slack for a deleted/missing capture may supply the link.
        for previous_position in (position.saturating_sub(2)..position).rev() {
            let previous = order[previous_position];
            let Some((_, current_to_previous)) =
                focus_stack_link_transform(matches, current, previous)
            else {
                continue;
            };
            let Some(motion_ratio) = focus_sequence_link_motion_ratio(
                &images[current],
                &images[previous],
                &current_to_previous,
            ) else {
                continue;
            };
            let is_sequence_bridge = matches
                .get(&(current, previous))
                .or_else(|| matches.get(&(previous, current)))
                .is_some_and(|match_info| match_info.sequence_bridge);
            if !is_sequence_bridge && motion_ratio > FOCUS_SEQUENCE_MAX_LINK_MOTION_RATIO {
                continue;
            }
            let Some(edge_weight) = focus_auto_order_edge_weight_between(
                images,
                matches,
                current,
                previous,
                motion_scale,
            ) else {
                continue;
            };
            let gap = position - previous_position;
            let score = edge_weight - gap.saturating_sub(1) as f64 * FOCUS_AUTO_ORDER_GAP_PENALTY;
            let should_replace = best_link.as_ref().is_none_or(
                |(best_score, best_weight, best_gap, best_motion, _, _)| {
                    score > *best_score
                        || (score == *best_score
                            && (edge_weight > *best_weight
                                || (edge_weight == *best_weight
                                    && (gap < *best_gap
                                        || (gap == *best_gap && motion_ratio < *best_motion)))))
                },
            );
            if should_replace {
                best_link = Some((
                    score,
                    edge_weight,
                    gap,
                    motion_ratio,
                    previous,
                    current_to_previous,
                ));
            }
        }
        let (_, _, _, _, previous, current_to_previous) = best_link?;
        let previous_global = global_homographies.get(&previous).copied()?;
        global_homographies.insert(current, previous_global * current_to_previous);
    }
    Some(global_homographies)
}

fn focus_sequence_link_motion_ratio(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
) -> Option<f64> {
    let source_center = Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5);
    let target_center = Point2::new(target.width as f64 * 0.5, target.height as f64 * 0.5);
    let mapped_center = transformed_point(source_to_target, source_center)?;
    let width = target.width.max(1) as f64;
    let height = target.height.max(1) as f64;
    Some(
        ((mapped_center.x - target_center.x) / width)
            .hypot((mapped_center.y - target_center.y) / height),
    )
}

fn focus_match_graph_is_dense_continuous_scan(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> bool {
    if images.len() < 9 {
        return false;
    }
    let possible_edges = images.len().saturating_mul(images.len() - 1) / 2;
    if possible_edges == 0 {
        return false;
    }
    let mut degrees = vec![0usize; images.len()];
    let verified_edges = matches
        .iter()
        .filter(|((source, target), match_info)| {
            let verified = *source < images.len()
                && *target < images.len()
                && source != target
                && !match_info.sequence_bridge
                && !match_info.coarse_bridge
                && match_info.points.len() >= FOCUS_MODEL_MIN_INLIERS;
            if verified {
                degrees[*source] += 1;
                degrees[*target] += 1;
            }
            verified
        })
        .count();
    let minimum_degree = images.len().div_ceil(4);
    verified_edges.saturating_mul(100) >= possible_edges.saturating_mul(45)
        && degrees.iter().all(|degree| *degree >= minimum_degree)
}

#[cfg(test)]
fn build_focus_stack_stitching_order(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> (Vec<usize>, HashMap<usize, Matrix3<f64>>) {
    let mut closure_report = stack_report::ClosureReport::default();
    let mut station_relations_report = stack_report::StationRelationsReport::default();
    build_focus_stack_stitching_order_with_report(
        images,
        matches,
        &mut closure_report,
        &mut station_relations_report,
        true,
    )
}

fn build_focus_stack_stitching_order_with_report(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
    solve_source_station_geometry: bool,
) -> (Vec<usize>, HashMap<usize, Matrix3<f64>>) {
    if images.len() < 2 {
        let mut homographies = HashMap::new();
        if let Some(image) = images.first() {
            homographies.insert(image.id, Matrix3::identity());
        }
        return ((0..images.len()).collect(), homographies);
    }

    // Focus stacking still has an order-sensitive ownership pass, but that
    // order must come from the image evidence rather than from upload order.
    // Build a maximum-confidence overlap graph, preferring strong, precise,
    // spatially-near matches. Compare several deterministic paths against the
    // graph: upload order, natural filename order, a graph traversal, and a
    // greedy visual path. This lets a camera sequence encoded in filenames win
    // when it is genuinely supported by image evidence, while still working
    // when filenames carry no useful order.
    let motion_scale = focus_auto_order_motion_scale(images, matches);
    let spine_matches = focus_stack_spine_matches(images, matches);
    if spine_matches.len() < matches.len() {
        println!(
            "  - Focus stack pose spine uses {}/{} pairwise edges (station representatives + brackets)",
            spine_matches.len(),
            matches.len()
        );
    }
    let (mut graph_order, mut graph_homographies) =
        build_graph_stitching_order(images, &spine_matches, |source, target, match_info| {
            let bracket_boost = if focus_match_is_local_bracket(source, target, match_info) {
                FOCUS_BRACKET_GRAPH_WEIGHT_BOOST
            } else {
                1.0
            };
            focus_auto_order_edge_weight(source, target, match_info, motion_scale)
                * focus_cycle_edge_factor(images, matches, source.id, target.id)
                * focus_graph_capture_sequence_boost(source, target, match_info)
                * bracket_boost
        });
    if graph_order.len() != images.len() {
        let attached = attach_focus_bracket_orphans_to_pose_graph(
            images,
            matches,
            &mut graph_order,
            &mut graph_homographies,
        );
        if attached > 0 {
            println!("  - Attached {attached} same-pose focus layer(s) to the verified pose graph");
        }
    }
    let filename_order = focus_filename_order_with_overview_gaps(images, matches);
    let input_order =
        canonical_focus_order_direction(&(0..images.len()).collect::<Vec<_>>(), images);
    let filename_order = canonical_focus_order_direction(&filename_order, images);
    let graph_order = canonical_focus_order_direction(&graph_order, images);
    let greedy_order = canonical_focus_order_direction(
        &focus_auto_order_greedy_path(images, matches, motion_scale),
        images,
    );
    let candidate_orders = vec![
        ("input", input_order, 0u8),
        ("filename", filename_order.clone(), 3u8),
        ("graph", graph_order.clone(), 1u8),
        ("visual", greedy_order, 2u8),
    ];

    #[cfg(test)]
    if std::env::var_os("RAW_EDITOR_STACK_FORCE_SEQUENCE_GEOMETRY").is_some() {
        let mut sequence_matches = matches.clone();
        let added = add_capture_sequence_bridges(images, &mut sequence_matches);
        if let Some(sequence_homographies) = build_focus_sequence_homographies(
            &filename_order,
            images,
            &sequence_matches,
            motion_scale,
        ) {
            let (valid_edges, total_edges, median_overlap, median_scale) =
                focus_capture_geometry_diagnostics(images, &filename_order, &sequence_homographies);
            println!(
                "  - Capture-sequence geometry experiment: {valid_edges}/{total_edges} adjacent edges, {added} synthetic gaps, median overlap {:.1}%, median scale {:.3}",
                median_overlap * 100.0,
                median_scale,
            );
            return (filename_order, sequence_homographies);
        }
        println!("  - Capture-sequence geometry experiment could not link every source");
    }
    let mut candidate_summaries = Vec::new();
    let mut selected = candidate_orders
        .into_iter()
        .filter(|(_, order, _)| order.len() == images.len())
        .map(|(name, order, priority)| {
            let (matched_edges, score) =
                focus_auto_order_path_score(&order, images, matches, motion_scale);
            let maximum_motion = focus_auto_order_path_max_motion(&order, images, matches);
            let motion_cost = focus_auto_order_path_motion_cost(&order, images, matches);
            candidate_summaries.push(format!(
                "{name}={matched_edges}/{}:{score:.2}/motion{motion_cost:.3}/max{maximum_motion:.3}",
                images.len().saturating_sub(1),
            ));
            (
                name,
                order,
                matched_edges,
                score,
                motion_cost,
                maximum_motion,
                priority,
            )
        })
        .max_by(|left, right| {
            // Synthetic filename bridges are only a fallback for an actual
            // gap. They must never outrank a visual path merely because they
            // make the edge count look complete: that creates a plausible
            // filename order with incorrect canvas poses and rectangular
            // duplicate tiles. Prefer the path's evidence score first, then
            // its real-edge coverage and motion stability.
            left.3
                .total_cmp(&right.3)
                .then_with(|| left.2
                .cmp(&right.2)
                // A direct match across several captured positions can have
                // many inliers, but letting it become one edge of the path
                // makes the ownership pass jump over the intermediate focus
                // layers. Prefer the path with the smallest single jump
                // before comparing its total travel distance.
                .then_with(|| right.5.total_cmp(&left.5))
                .then_with(|| right.4.total_cmp(&left.4))
                .then_with(|| left.6.cmp(&right.6)))
        });
    let Some((selected_name, selected_order, matched_edges, selected_score, _, _, _)) =
        selected.take()
    else {
        return (graph_order, graph_homographies);
    };

    // Rendering ownership may stay in the deterministic capture order, but
    // geometry must come from the verified maximum-confidence overlap graph.
    // Chaining filename neighbours accumulates camera-pass resets into the
    // diagonal snake/collage failure seen on large focus-stack mosaics.
    // If a demonstrably weak frame is absent from the graph, filter it out of
    // the capture order instead of falling back to graph traversal; otherwise
    // one skipped frame makes hundreds of valid layers render in tree order.
    let graph_sources = graph_order.iter().copied().collect::<HashSet<_>>();
    let connected_filename_order = filename_order
        .iter()
        .copied()
        .filter(|index| graph_sources.contains(index))
        .collect::<Vec<_>>();
    // Focus ownership is order-sensitive. Keep captures that were made next
    // to each other adjacent in the renderer even when a greedy visual path
    // can collect more graph edges by jumping between distant parts of the
    // scroll. Geometry still comes exclusively from the verified graph.
    let large_scan = images.len() >= SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES;
    let dense_continuous_scan = focus_match_graph_is_dense_continuous_scan(images, matches);
    if graph_order.len() != images.len() {
        let render_order = if large_scan || dense_continuous_scan {
            connected_filename_order
        } else {
            graph_order
        };
        return (render_order, graph_homographies);
    }
    let render_order = if large_scan || dense_continuous_scan {
        filename_order
    } else {
        selected_order
    };
    let reference_index = graph_order[0];
    let optimized_homographies = optimize_focus_stack_global_homographies_with_reference(
        images,
        matches,
        &graph_homographies,
        reference_index,
    );
    // Bracket locking belongs to the station model, not to the capture-order
    // heuristic. Infer Capture_Stations from the locked intra-station poses on
    // every default focus run; raw image count must never bypass the strict
    // Station_Pose_Solver gates (requirements 6, 7 and 15.1).
    let locked_homographies =
        lock_focus_local_bracket_poses(images, matches, &optimized_homographies);
    let inferred_station_count = focus_capture_groups(images, matches, &locked_homographies)
        .map(|(groups, _)| groups.len())
        .unwrap_or_default();
    let solve_station_geometry = solve_source_station_geometry && inferred_station_count >= 2;
    let optimized_or_locked_homographies = if solve_station_geometry {
        let mut translation_geometry_verified = false;
        solve_focus_capture_group_poses_with_report(
            images,
            matches,
            &locked_homographies,
            reference_index,
            &mut translation_geometry_verified,
            closure_report,
            station_relations_report,
        )
    } else if inferred_station_count >= 2 {
        // Default two-stage Virtual_Tile flow: preserve the image-level solve
        // only as local station geometry.  Bracket locking keeps every source
        // in its station anchor coordinates; inter-station placement is
        // replaced by the authoritative fused-tile solve after rendering.
        locked_homographies.clone()
    } else {
        // A single Capture_Station has no inter-station relation or closure to
        // solve. Keep compact single-station rendering on its established
        // image-level geometry rather than applying bracket-lock side effects.
        optimized_homographies.clone()
    };
    if station_relations_report.connectivity.components > 1 {
        return (Vec::new(), HashMap::new());
    }
    let geometry_render_order = if large_scan {
        focus_geometry_capture_order(images, matches, &optimized_or_locked_homographies)
            .filter(|order| order.len() == images.len())
            .unwrap_or_else(|| render_order.clone())
    } else {
        render_order.clone()
    };
    let (valid_geometry_edges, total_geometry_edges, median_overlap, median_scale) =
        focus_capture_geometry_diagnostics(
            images,
            &geometry_render_order,
            &optimized_or_locked_homographies,
        );
    println!(
        "  - Focus geometry validation: {valid_geometry_edges}/{total_geometry_edges} adjacent edges, median overlap {:.1}%, median scale {:.3}",
        median_overlap * 100.0,
        median_scale,
    );
    let (
        connected_group_count,
        total_group_count,
        valid_group_relations,
        candidate_group_relations,
        median_group_overlap,
    ) = focus_capture_groups(images, matches, &optimized_or_locked_homographies)
        .map(|(groups, _)| {
            focus_capture_group_geometry_diagnostics(
                images,
                matches,
                &groups,
                &optimized_or_locked_homographies,
            )
        })
        .unwrap_or((0, 0, 0, 0, 0.0));
    if solve_station_geometry {
        println!(
            "  - Focus station validation: {connected_group_count}/{total_group_count} connected groups, {valid_group_relations}/{candidate_group_relations} verified relations, median overlap {:.1}%",
            median_group_overlap * 100.0,
        );
    }
    let global_homographies = if !solve_source_station_geometry
        || !large_scan
        || (focus_capture_geometry_passes(
            images,
            valid_geometry_edges,
            total_geometry_edges,
            median_overlap,
            &optimized_or_locked_homographies,
        ) && (!solve_source_station_geometry
            || focus_capture_group_geometry_passes(
                connected_group_count,
                total_group_count,
                valid_group_relations,
                median_group_overlap,
            ))) {
        optimized_or_locked_homographies
    } else {
        // The long-scan refinement can be pulled by a repeated character or
        // seal. Retry the same verified graph without the capture-group
        // translation solve before rejecting the stack outright.
        println!(
            "  - Focus geometry rejected ({valid_geometry_edges}/{total_geometry_edges} continuous edges); retrying the ungrouped global pose"
        );
        let (retry_valid_edges, retry_total_edges, retry_overlap, retry_scale) =
            focus_capture_geometry_diagnostics(
                images,
                &geometry_render_order,
                &optimized_homographies,
            );
        println!(
            "  - Focus geometry retry: {retry_valid_edges}/{retry_total_edges} adjacent edges, median overlap {:.1}%, median scale {:.3}",
            retry_overlap * 100.0,
            retry_scale,
        );
        let (
            retry_connected_groups,
            retry_total_groups,
            retry_valid_relations,
            retry_candidate_relations,
            retry_group_overlap,
        ) = focus_capture_groups(images, matches, &optimized_homographies)
            .map(|(groups, _)| {
                focus_capture_group_geometry_diagnostics(
                    images,
                    matches,
                    &groups,
                    &optimized_homographies,
                )
            })
            .unwrap_or((0, 0, 0, 0, 0.0));
        println!(
            "  - Focus station retry: {retry_connected_groups}/{retry_total_groups} connected groups, {retry_valid_relations}/{retry_candidate_relations} verified relations, median overlap {:.1}%",
            retry_group_overlap * 100.0,
        );
        if focus_capture_geometry_passes(
            images,
            retry_valid_edges,
            retry_total_edges,
            retry_overlap,
            &optimized_homographies,
        ) && (!solve_source_station_geometry
            || focus_capture_group_geometry_passes(
                retry_connected_groups,
                retry_total_groups,
                retry_valid_relations,
                retry_group_overlap,
            ))
        {
            optimized_homographies
        } else {
            println!(
                "  - Focus geometry rejected after retry; refusing to render a discontinuous focus stack"
            );
            return (Vec::new(), HashMap::new());
        }
    };
    println!(
        "Focus auto-order diagnostic selected {selected_name} path with {matched_edges}/{} adjacent matches and score {selected_score:.2}; rendering uses stable capture order (candidates: {}; local motion scale {:.3}%)",
        images.len().saturating_sub(1),
        candidate_summaries.join(", "),
        motion_scale * 100.0,
    );
    if dense_continuous_scan {
        println!(
            "  - Dense continuous focus scan: preserving filename capture order for detail ownership"
        );
    }
    (geometry_render_order, global_homographies)
}

type FocusPose = [f64; 8];

#[derive(Clone, Copy)]
struct FocusGlobalObservation {
    source_index: usize,
    target_index: usize,
    source: Point2<f64>,
    target: Point2<f64>,
    weight: f64,
}

fn focus_pose_matrix(pose: &FocusPose) -> Matrix3<f64> {
    Matrix3::new(
        pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], pose[7], 1.0,
    )
}

fn focus_matrix_pose(matrix: &Matrix3<f64>) -> Option<FocusPose> {
    let normalization = matrix[(2, 2)];
    if !normalization.is_finite() || normalization.abs() < 1e-8 {
        return None;
    }
    let pose = [
        matrix[(0, 0)] / normalization,
        matrix[(0, 1)] / normalization,
        matrix[(0, 2)] / normalization,
        matrix[(1, 0)] / normalization,
        matrix[(1, 1)] / normalization,
        matrix[(1, 2)] / normalization,
        matrix[(2, 0)] / normalization,
        matrix[(2, 1)] / normalization,
    ];
    pose.iter().all(|value| value.is_finite()).then_some(pose)
}

fn focus_local_norm_to_full(image: &ImageInfo, coordinate_scale: f64) -> Matrix3<f64> {
    Matrix3::new(
        coordinate_scale,
        0.0,
        image.width as f64 * 0.5,
        0.0,
        coordinate_scale,
        image.height as f64 * 0.5,
        0.0,
        0.0,
        1.0,
    )
}

fn focus_world_full_to_norm(reference: &ImageInfo, coordinate_scale: f64) -> Matrix3<f64> {
    Matrix3::new(
        coordinate_scale.recip(),
        0.0,
        -(reference.width as f64 * 0.5) / coordinate_scale,
        0.0,
        coordinate_scale.recip(),
        -(reference.height as f64 * 0.5) / coordinate_scale,
        0.0,
        0.0,
        1.0,
    )
}

fn focus_world_norm_to_full(reference: &ImageInfo, coordinate_scale: f64) -> Matrix3<f64> {
    focus_local_norm_to_full(reference, coordinate_scale)
}

fn focus_normalized_point(
    image: &ImageInfo,
    point: Point2<f64>,
    coordinate_scale: f64,
) -> Point2<f64> {
    Point2::new(
        (point.x - image.width as f64 * 0.5) / coordinate_scale,
        (point.y - image.height as f64 * 0.5) / coordinate_scale,
    )
}

fn focus_pose_to_full_homography(
    pose: &FocusPose,
    image: &ImageInfo,
    reference: &ImageInfo,
    coordinate_scale: f64,
) -> Option<Matrix3<f64>> {
    let local_full_to_norm = Matrix3::new(
        coordinate_scale.recip(),
        0.0,
        -(image.width as f64 * 0.5) / coordinate_scale,
        0.0,
        coordinate_scale.recip(),
        -(image.height as f64 * 0.5) / coordinate_scale,
        0.0,
        0.0,
        1.0,
    );
    let full = focus_world_norm_to_full(reference, coordinate_scale)
        * focus_pose_matrix(pose)
        * local_full_to_norm;
    focus_matrix_pose(&full).map(|normalized| focus_pose_matrix(&normalized))
}

fn focus_homography_to_pose(
    homography: &Matrix3<f64>,
    image: &ImageInfo,
    reference: &ImageInfo,
    coordinate_scale: f64,
) -> Option<FocusPose> {
    let normalized = focus_world_full_to_norm(reference, coordinate_scale)
        * homography
        * focus_local_norm_to_full(image, coordinate_scale);
    focus_matrix_pose(&normalized)
}

fn focus_pose_projection(
    pose: &FocusPose,
    point: Point2<f64>,
) -> Option<(Point2<f64>, [f64; 8], [f64; 8])> {
    let x = point.x;
    let y = point.y;
    let numerator_x = pose[0] * x + pose[1] * y + pose[2];
    let numerator_y = pose[3] * x + pose[4] * y + pose[5];
    let denominator = pose[6] * x + pose[7] * y + 1.0;
    if !denominator.is_finite() || denominator.abs() < 1e-8 {
        return None;
    }
    let inverse_denominator = denominator.recip();
    let projected = Point2::new(
        numerator_x * inverse_denominator,
        numerator_y * inverse_denominator,
    );
    if !projected.x.is_finite() || !projected.y.is_finite() {
        return None;
    }
    let x_jacobian = [
        x * inverse_denominator,
        y * inverse_denominator,
        inverse_denominator,
        0.0,
        0.0,
        0.0,
        -projected.x * x * inverse_denominator,
        -projected.x * y * inverse_denominator,
    ];
    let y_jacobian = [
        0.0,
        0.0,
        0.0,
        x * inverse_denominator,
        y * inverse_denominator,
        inverse_denominator,
        -projected.y * x * inverse_denominator,
        -projected.y * y * inverse_denominator,
    ];
    Some((projected, x_jacobian, y_jacobian))
}

fn focus_match_points_for_region(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    match_info: &MatchInfo,
    normalized_y_range: (f64, f64),
) -> Vec<(Point2<f64>, Point2<f64>)> {
    if match_info.coarse_bridge {
        return Vec::new();
    }
    let Some(source_foreground_range) = source_image.foreground_range else {
        return Vec::new();
    };
    let Some(target_foreground_range) = target_image.foreground_range else {
        return Vec::new();
    };
    let mut selected_points = match_info.dense_focus_points.clone();
    selected_points.extend(match_info.foreground_feature_points.iter().copied());
    if selected_points.len() < FOCUS_LOCAL_MODEL_MIN_INLIERS {
        selected_points.extend(match_info.top_candidate_points.iter().copied());
    }
    let maximum_residual =
        source_image.width.max(source_image.height) as f64 * FOCUS_LOCAL_MATCH_RESIDUAL_RATIO;
    let filtered = selected_points
        .iter()
        .copied()
        .filter(|(source, target)| {
            let mapped = match_info.homography * Point3::new(source.x, source.y, 1.0);
            if mapped.z.abs() < 1e-8 {
                return false;
            }
            let predicted = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
            if (predicted - *target).norm() > maximum_residual {
                return false;
            }
            let source_y = source.y / source_image.height.max(1) as f64;
            let target_y = target.y / target_image.height.max(1) as f64;
            source_y >= normalized_y_range.0
                && source_y <= normalized_y_range.1
                && target_y >= normalized_y_range.0
                && target_y <= normalized_y_range.1
                && source_y >= source_foreground_range.0
                && source_y <= source_foreground_range.1
                && target_y >= target_foreground_range.0
                && target_y <= target_foreground_range.1
        })
        .collect::<Vec<_>>();
    if filtered.len() < FOCUS_LOCAL_MODEL_MIN_INLIERS {
        return Vec::new();
    }

    // A narrow depth layer can provide only a few long, mostly one-dimensional edges.
    // The general panorama homography solver intentionally requires 15 inliers,
    // which rejects otherwise useful local models from a narrow foreground band.
    // Fit an affine model here with the focus-specific minimum instead; the
    // stability check below still prevents a sparse or degenerate fit from being
    // applied to the rendered layer.
    let Some(regional_fit) =
        robust_transform_fit(&filtered, 3, 0x5EED_7A11_4F52_9B31, estimate_affine)
    else {
        return Vec::new();
    };
    let RobustTransformFit {
        transform: regional_homography,
        inlier_indices,
        median_error: _,
    } = regional_fit;
    let inlier_points = inlier_indices
        .iter()
        .map(|&index| filtered[index])
        .collect::<Vec<_>>();
    if inlier_indices.len() < FOCUS_LOCAL_MODEL_MIN_INLIERS
        || !transform_is_stable_for_focus_stack(&regional_homography, source_image.dimensions())
        || !focus_regional_support_is_diverse(
            &inlier_points,
            source_image.dimensions(),
            target_image.dimensions(),
        )
    {
        return Vec::new();
    }

    let maximum_displacement = [0.0, 0.5, 1.0]
        .into_iter()
        .flat_map(|x| {
            [normalized_y_range.0, normalized_y_range.1]
                .into_iter()
                .map(move |y| {
                    Point2::new(
                        source_image.width as f64 * x,
                        source_image.height as f64 * y,
                    )
                })
        })
        .filter_map(|point| {
            let local = regional_homography * nalgebra::Point3::new(point.x, point.y, 1.0);
            let global = match_info.homography * nalgebra::Point3::new(point.x, point.y, 1.0);
            if local.z.abs() < 1e-8 || global.z.abs() < 1e-8 {
                return None;
            }
            Some(
                (Point2::new(local.x / local.z, local.y / local.z)
                    - Point2::new(global.x / global.z, global.y / global.z))
                .norm(),
            )
        })
        .fold(0.0, f64::max);
    if maximum_displacement
        <= source_image.width.max(source_image.height) as f64
            * FOCUS_LOCAL_MODEL_MAX_DISPLACEMENT_RATIO
    {
        inlier_indices
            .into_iter()
            .map(|index| filtered[index])
            .collect()
    } else {
        Vec::new()
    }
}

fn focus_regional_support_is_diverse(
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
) -> bool {
    if points.len() < FOCUS_LOCAL_MODEL_MIN_INLIERS {
        return false;
    }
    let source_min_x = points
        .iter()
        .map(|(source, _)| source.x)
        .fold(f64::INFINITY, f64::min);
    let source_max_x = points
        .iter()
        .map(|(source, _)| source.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let source_min_y = points
        .iter()
        .map(|(source, _)| source.y)
        .fold(f64::INFINITY, f64::min);
    let source_max_y = points
        .iter()
        .map(|(source, _)| source.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let target_min_x = points
        .iter()
        .map(|(_, target)| target.x)
        .fold(f64::INFINITY, f64::min);
    let target_max_x = points
        .iter()
        .map(|(_, target)| target.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let target_min_y = points
        .iter()
        .map(|(_, target)| target.y)
        .fold(f64::INFINITY, f64::min);
    let target_max_y = points
        .iter()
        .map(|(_, target)| target.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let source_width = source_dimensions.0.max(1) as f64;
    let source_height = source_dimensions.1.max(1) as f64;
    let target_width = target_dimensions.0.max(1) as f64;
    let target_height = target_dimensions.1.max(1) as f64;
    let source_span_x = (source_max_x - source_min_x) / source_width;
    let source_span_y = (source_max_y - source_min_y) / source_height;
    let target_span_x = (target_max_x - target_min_x) / target_width;
    let target_span_y = (target_max_y - target_min_y) / target_height;
    let source_area = source_span_x * source_span_y;
    let target_area = target_span_x * target_span_y;

    // A local model needs support in both axes. The area fallback permits a
    // long, thin object such as a mat edge, but a single screw/corner cluster
    // cannot satisfy it and therefore cannot extrapolate a warp over the band.
    let source_supported = (source_span_x >= 0.10 && source_span_y >= 0.035)
        || source_area >= 0.006
        // A long, well-supported edge is intentionally one-dimensional. It
        // constrains the vertical translation/shear that causes a stepped
        // holder or paper boundary, while the global pose prior still keeps
        // the under-constrained horizontal parameters stable.
        || (source_span_x >= 0.40 && source_span_y >= 0.001);
    let target_supported = (target_span_x >= 0.10 && target_span_y >= 0.035)
        || target_area >= 0.006
        || (target_span_x >= 0.40 && target_span_y >= 0.001);
    source_supported && target_supported
}

fn focus_global_observations(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    coordinate_scale: f64,
    normalized_y_range: Option<(f64, f64)>,
) -> Vec<FocusGlobalObservation> {
    focus_global_observations_with_mode(images, matches, coordinate_scale, normalized_y_range, true)
}

fn focus_global_observations_for_generic_region(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    coordinate_scale: f64,
    normalized_y_range: (f64, f64),
) -> Vec<FocusGlobalObservation> {
    focus_global_observations_with_mode(
        images,
        matches,
        coordinate_scale,
        Some(normalized_y_range),
        false,
    )
}

fn focus_global_observations_with_mode(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    coordinate_scale: f64,
    normalized_y_range: Option<(f64, f64)>,
    use_foreground_region_matches: bool,
) -> Vec<FocusGlobalObservation> {
    let mut observations = Vec::new();
    // The observation vector defines the summation order of the robust cost and
    // of the normal equations in the global pose solve.  Iterating the match map
    // directly made that order depend on the per-process hash seed, which
    // changed the solved poses in the last mantissa bits and therefore the
    // rendered canvas.  Walk the pairs in ascending key order instead (需求 14.6).
    for (&(source_index, target_index), match_info) in sorted_pairs(matches) {
        if source_index >= images.len() || target_index >= images.len() {
            continue;
        }
        // An overview/reference frame is permitted to fill an explicitly
        // missing row, but it is not an identity-bearing layer for the focus
        // pose solve. Its wide, earlier capture often contains repeated seals
        // and rails that can pull the close-up sequence to a false scale.
        if images[source_index].overview_reference || images[target_index].overview_reference {
            continue;
        }
        if match_info.sequence_bridge || match_info.coarse_bridge {
            // A gap bridge supplies capture order and a conservative initial
            // placement only. A coarse lens-switch candidate has synthetic
            // grid points as well. Neither is an observed artwork
            // correspondence, so allowing either into the global solve would
            // turn a nominal seam into a fake reprojection constraint.
            continue;
        }
        // The match has already passed full-resolution model, spatial-support
        // and overlap validation. Keep long-range closures in the global solve
        // as well: vertical links and cross-column links are often far apart
        // in capture order, and filtering them by filename gap leaves a snake
        // scan anchored to one drifting spanning tree.
        let distant_edge = source_index.abs_diff(target_index) > 1;
        let selected_points = normalized_y_range
            .map(|range| {
                if use_foreground_region_matches {
                    focus_match_points_for_region(
                        &images[source_index],
                        &images[target_index],
                        match_info,
                        range,
                    )
                } else {
                    focus_match_points_for_generic_region(
                        match_info,
                        &images[source_index],
                        &images[target_index],
                        range,
                    )
                }
            })
            .unwrap_or_else(|| match_info.points.clone());
        let maximum_points = if distant_edge {
            FOCUS_GLOBAL_MAX_POINTS_PER_EDGE.min(64)
        } else {
            FOCUS_GLOBAL_MAX_POINTS_PER_EDGE
        };
        let observation_weight = if focus_match_is_local_bracket(
            &images[source_index],
            &images[target_index],
            match_info,
        ) {
            FOCUS_BRACKET_OBSERVATION_WEIGHT
        } else {
            1.0
        };
        let point_count = selected_points.len().min(maximum_points);
        for sample_index in 0..point_count {
            let point_index = if point_count == selected_points.len() {
                sample_index
            } else {
                sample_index * (selected_points.len() - 1) / (point_count - 1).max(1)
            };
            let (source, target) = selected_points[point_index];
            if source.x.is_finite()
                && source.y.is_finite()
                && target.x.is_finite()
                && target.y.is_finite()
            {
                let source =
                    focus_normalized_point(&images[source_index], source, coordinate_scale);
                let target =
                    focus_normalized_point(&images[target_index], target, coordinate_scale);
                observations.push(FocusGlobalObservation {
                    source_index,
                    target_index,
                    source,
                    target,
                    weight: observation_weight,
                });
            }
        }
    }
    observations
}

fn focus_match_points_for_generic_region(
    match_info: &MatchInfo,
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    normalized_y_range: (f64, f64),
) -> Vec<(Point2<f64>, Point2<f64>)> {
    if match_info.coarse_bridge {
        return Vec::new();
    }
    // Ordinary keypoint inliers provide the reliable identity signal, while the
    // dense patch matches carry the sub-pixel residual that a broad homography
    // cannot explain. Keep both for generic regions; restricting this to
    // keypoints made the "generic" bands numerically different but visually
    // inert on texture-poor or near-field parts of a scan.
    let mut candidates = match_info.points.clone();
    candidates.extend(match_info.dense_focus_points.iter().copied());
    let maximum_residual = source_image.width.max(source_image.height).max(1) as f64
        * FOCUS_LOCAL_MATCH_RESIDUAL_RATIO;
    candidates
        .into_iter()
        .filter(|(source, target)| {
            if !source.x.is_finite()
                || !source.y.is_finite()
                || !target.x.is_finite()
                || !target.y.is_finite()
            {
                return false;
            }
            let mapped = match_info.homography * Point3::new(source.x, source.y, 1.0);
            if mapped.z.abs() < 1e-8 {
                return false;
            }
            let predicted = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
            if (predicted - *target).norm() > maximum_residual {
                return false;
            }
            let source_y = source.y / source_image.height.max(1) as f64;
            let target_y = target.y / target_image.height.max(1) as f64;
            // A moving mosaic can move the same physical strip to a different
            // normalized y coordinate in the adjacent frame. Associate a
            // correspondence with a band when either endpoint lies in it;
            // requiring both endpoints silently dropped exactly those residuals.
            (normalized_y_range.0..=normalized_y_range.1).contains(&source_y)
                || (normalized_y_range.0..=normalized_y_range.1).contains(&target_y)
        })
        .collect()
}

fn focus_global_poses_are_valid(
    poses: &[FocusPose],
    images: &[ImageInfo],
    reference: &ImageInfo,
    coordinate_scale: f64,
) -> bool {
    if poses.len() != images.len()
        || poses.iter().any(|pose| {
            pose.iter()
                .any(|value| !value.is_finite() || value.abs() > 100.0)
        })
    {
        return false;
    }
    poses.iter().zip(images).all(|(pose, image)| {
        let corners = [
            Point2::new(0.0, 0.0),
            Point2::new(image.width as f64, 0.0),
            Point2::new(image.width as f64, image.height as f64),
            Point2::new(0.0, image.height as f64),
        ];
        let normalized_corners = corners
            .into_iter()
            .map(|corner| focus_normalized_point(image, corner, coordinate_scale));
        if normalized_corners
            .filter_map(|corner| focus_pose_projection(pose, corner))
            .count()
            != 4
        {
            return false;
        }
        let Some(full) = focus_pose_to_full_homography(pose, image, reference, coordinate_scale)
        else {
            return false;
        };
        full.try_inverse().is_some()
            && homography_preserves_focus_orientation(&full, (image.width, image.height))
    })
}

fn focus_global_robust_cost(
    poses: &[FocusPose],
    initial_poses: &[FocusPose],
    observations: &[FocusGlobalObservation],
    reference_index: usize,
) -> f64 {
    let mut cost = 0.0;
    for observation in observations {
        let Some((source, _, _)) =
            focus_pose_projection(&poses[observation.source_index], observation.source)
        else {
            return f64::INFINITY;
        };
        let Some((target, _, _)) =
            focus_pose_projection(&poses[observation.target_index], observation.target)
        else {
            return f64::INFINITY;
        };
        let residual = source - target;
        let magnitude = residual.norm();
        if !magnitude.is_finite() {
            return f64::INFINITY;
        }
        let robust_cost = if magnitude <= FOCUS_GLOBAL_HUBER_THRESHOLD {
            0.5 * magnitude * magnitude
        } else {
            FOCUS_GLOBAL_HUBER_THRESHOLD * (magnitude - 0.5 * FOCUS_GLOBAL_HUBER_THRESHOLD)
        };
        cost += observation.weight * robust_cost;
    }
    for (image_index, pose) in poses.iter().enumerate() {
        if image_index == reference_index {
            continue;
        }
        for parameter in 0..8 {
            let prior_weight = if parameter >= 6 {
                FOCUS_GLOBAL_PROJECTIVE_PRIOR_WEIGHT
            } else {
                FOCUS_GLOBAL_PRIOR_WEIGHT
            };
            let difference = pose[parameter] - initial_poses[image_index][parameter];
            cost += 0.5 * prior_weight * difference * difference;
        }
    }
    cost
}

fn focus_global_normal_equations(
    poses: &[FocusPose],
    initial_poses: &[FocusPose],
    observations: &[FocusGlobalObservation],
    reference_index: usize,
) -> (nalgebra::DMatrix<f64>, nalgebra::DVector<f64>) {
    let variable_count = poses.len().saturating_sub(1) * 8;
    let mut normal = nalgebra::DMatrix::zeros(variable_count, variable_count);
    let mut gradient = nalgebra::DVector::zeros(variable_count);

    for observation in observations {
        let Some((source, source_x_jacobian, source_y_jacobian)) =
            focus_pose_projection(&poses[observation.source_index], observation.source)
        else {
            continue;
        };
        let Some((target, target_x_jacobian, target_y_jacobian)) =
            focus_pose_projection(&poses[observation.target_index], observation.target)
        else {
            continue;
        };
        let residual = source - target;
        let magnitude = residual.norm();
        if !magnitude.is_finite() {
            continue;
        }
        let robust_weight = observation.weight
            * if magnitude <= FOCUS_GLOBAL_HUBER_THRESHOLD {
                1.0
            } else {
                FOCUS_GLOBAL_HUBER_THRESHOLD / magnitude
            };

        for (residual_component, source_jacobian, target_jacobian) in [
            (residual.x, source_x_jacobian, target_x_jacobian),
            (residual.y, source_y_jacobian, target_y_jacobian),
        ] {
            let mut jacobian_entries = Vec::with_capacity(16);
            if observation.source_index != reference_index {
                let base = if observation.source_index < reference_index {
                    observation.source_index * 8
                } else {
                    (observation.source_index - 1) * 8
                };
                for (parameter, &value) in source_jacobian.iter().enumerate() {
                    jacobian_entries.push((base + parameter, value));
                }
            }
            if observation.target_index != reference_index {
                let base = if observation.target_index < reference_index {
                    observation.target_index * 8
                } else {
                    (observation.target_index - 1) * 8
                };
                for (parameter, &value) in target_jacobian.iter().enumerate() {
                    jacobian_entries.push((base + parameter, -value));
                }
            }
            for &(column, jacobian) in &jacobian_entries {
                gradient[column] += robust_weight * jacobian * residual_component;
                for &(row, other_jacobian) in &jacobian_entries {
                    normal[(column, row)] += robust_weight * jacobian * other_jacobian;
                }
            }
        }
    }

    for (image_index, pose) in poses.iter().enumerate() {
        if image_index == reference_index {
            continue;
        }
        let base = if image_index < reference_index {
            image_index * 8
        } else {
            (image_index - 1) * 8
        };
        for parameter in 0..8 {
            let prior_weight = if parameter >= 6 {
                FOCUS_GLOBAL_PROJECTIVE_PRIOR_WEIGHT
            } else {
                FOCUS_GLOBAL_PRIOR_WEIGHT
            };
            normal[(base + parameter, base + parameter)] += prior_weight + FOCUS_GLOBAL_DAMPING;
            gradient[base + parameter] +=
                prior_weight * (pose[parameter] - initial_poses[image_index][parameter]);
        }
    }
    (normal, gradient)
}

fn optimize_focus_stack_global_homographies_with_reference(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
) -> HashMap<usize, Matrix3<f64>> {
    if images.iter().any(|image| image.overview_reference) {
        // The wide frame is a bounded coverage source, not a registration
        // anchor. It has no verified identity correspondences with the
        // close-up sequence, so letting bundle adjustment see the disconnected
        // components would invent a scale for the whole stack.
        println!(
            "  - Skipping global pose refinement: overview reference has no verified identity overlap"
        );
        return initial_homographies.clone();
    }
    optimize_focus_stack_global_homographies_in_region_mode_with_reference(
        images,
        matches,
        initial_homographies,
        None,
        false,
        reference_index,
    )
}

fn optimize_focus_stack_global_homographies_in_region(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    normalized_y_range: Option<(f64, f64)>,
) -> HashMap<usize, Matrix3<f64>> {
    optimize_focus_stack_global_homographies_in_region_mode_with_reference(
        images,
        matches,
        initial_homographies,
        normalized_y_range,
        true,
        0,
    )
}

fn optimize_focus_stack_global_homographies_in_generic_region(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    normalized_y_range: (f64, f64),
) -> HashMap<usize, Matrix3<f64>> {
    optimize_focus_stack_global_homographies_in_region_mode_with_reference(
        images,
        matches,
        initial_homographies,
        Some(normalized_y_range),
        false,
        0,
    )
}

fn optimize_focus_stack_global_homographies_in_region_mode_with_reference(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    normalized_y_range: Option<(f64, f64)>,
    use_foreground_region_matches: bool,
    reference_index: usize,
) -> HashMap<usize, Matrix3<f64>> {
    if images.len() < 2 || reference_index >= images.len() {
        return initial_homographies.clone();
    }
    let mixed_focal_stack = images_have_mixed_focal_lengths(images);
    let reference = &images[reference_index];
    let coordinate_scale = images
        .iter()
        .map(|image| image.width.max(image.height) as f64)
        .fold(1.0, f64::max);
    let mut initial_poses = Vec::with_capacity(images.len());
    for image in images {
        let Some(homography) = initial_homographies.get(&image.id) else {
            return initial_homographies.clone();
        };
        let Some(pose) = focus_homography_to_pose(homography, image, reference, coordinate_scale)
        else {
            return initial_homographies.clone();
        };
        initial_poses.push(pose);
    }
    initial_poses[reference_index] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let initial_pose_maxima = (0..8)
        .map(|parameter| {
            initial_poses
                .iter()
                .map(|pose| pose[parameter].abs())
                .fold(0.0, f64::max)
        })
        .collect::<Vec<_>>();
    println!(
        "  - Focus global initial pose maxima: {:?}",
        initial_pose_maxima
    );

    let observations = if use_foreground_region_matches {
        focus_global_observations(images, matches, coordinate_scale, normalized_y_range)
    } else if let Some(range) = normalized_y_range {
        focus_global_observations_for_generic_region(images, matches, coordinate_scale, range)
    } else {
        focus_global_observations(images, matches, coordinate_scale, None)
    };
    if mixed_focal_stack {
        println!(
            "  - Focus global registration for mixed-focal stack: solving validated same-/cross-lens closures"
        );
    }
    let minimum_observations = if normalized_y_range.is_some() {
        if use_foreground_region_matches {
            FOCUS_LOCAL_MODEL_MIN_INLIERS
        } else {
            FOCUS_MODEL_MIN_INLIERS
        }
    } else {
        processing::MIN_INLIERS_FOR_CONNECTION
    };
    if observations.len() < minimum_observations {
        return initial_homographies.clone();
    }
    let mut poses = initial_poses.clone();
    let mut current_cost =
        focus_global_robust_cost(&poses, &initial_poses, &observations, reference_index);
    if !current_cost.is_finite() {
        return initial_homographies.clone();
    }
    let initial_poses_valid =
        focus_global_poses_are_valid(&poses, images, reference, coordinate_scale);
    println!("  - Focus global registration initial geometry valid: {initial_poses_valid}");
    if !initial_poses_valid {
        // Do not let the optimizer turn a bad pairwise registration into a
        // plausible-looking mirrored canvas. The original homographies are the
        // safer fallback; pairwise model validation remains responsible for
        // finding a usable non-mirrored path on the next stage.
        return initial_homographies.clone();
    }
    let initial_cost = current_cost;
    let mut accepted_iterations = 0;

    for _ in 0..FOCUS_GLOBAL_MAX_ITERATIONS {
        let (normal, gradient) =
            focus_global_normal_equations(&poses, &initial_poses, &observations, reference_index);
        let right_hand_side = -gradient;
        let Some(delta) = normal.lu().solve(&right_hand_side) else {
            break;
        };
        let delta_norm = delta.norm();
        if !delta_norm.is_finite() || delta_norm < 1e-9 {
            break;
        }
        let max_parameter_delta = delta
            .as_slice()
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        println!(
            "  - Focus global pose step: norm {:.6}, max {:.6}",
            delta_norm, max_parameter_delta
        );

        let mut accepted = None;
        for step in [1.0, 0.5, 0.25, 0.125, 0.0625] {
            let mut candidate = poses.clone();
            for (image_index, pose) in candidate.iter_mut().enumerate() {
                if image_index == reference_index {
                    continue;
                }
                let base = if image_index < reference_index {
                    image_index * 8
                } else {
                    (image_index - 1) * 8
                };
                for parameter in 0..8 {
                    pose[parameter] += step * delta[base + parameter];
                }
            }
            if candidate
                .iter()
                .enumerate()
                .filter(|(image_index, _)| *image_index != reference_index)
                .any(|(image_index, pose)| {
                    pose.iter().enumerate().any(|(parameter, value)| {
                        let limit = match parameter {
                            2 | 5 => FOCUS_GLOBAL_MAX_TRANSLATION_ADJUSTMENT,
                            6 | 7 => FOCUS_GLOBAL_MAX_PROJECTIVE_ADJUSTMENT,
                            _ => FOCUS_GLOBAL_MAX_LINEAR_ADJUSTMENT,
                        };
                        (value - initial_poses[image_index][parameter]).abs() > limit
                    })
                })
            {
                continue;
            }
            if !focus_global_poses_are_valid(&candidate, images, reference, coordinate_scale) {
                continue;
            }
            let candidate_cost = focus_global_robust_cost(
                &candidate,
                &initial_poses,
                &observations,
                reference_index,
            );
            if candidate_cost.is_finite() && candidate_cost + 1e-12 < current_cost {
                accepted = Some((candidate, candidate_cost));
                break;
            }
        }
        let Some((candidate, candidate_cost)) = accepted else {
            break;
        };
        poses = candidate;
        current_cost = candidate_cost;
        accepted_iterations += 1;
    }

    if accepted_iterations == 0 || current_cost >= initial_cost {
        return initial_homographies.clone();
    }

    let mut optimized = HashMap::new();
    for (image_index, image) in images.iter().enumerate() {
        let Some(homography) =
            focus_pose_to_full_homography(&poses[image_index], image, reference, coordinate_scale)
        else {
            return initial_homographies.clone();
        };
        optimized.insert(image.id, homography);
    }
    println!(
        "  - Focus global registration used {} observations over {} edges: robust cost {:.6} -> {:.6} in {} iteration(s)",
        observations.len(),
        matches
            .values()
            .filter(|match_info| !match_info.points.is_empty())
            .count(),
        initial_cost,
        current_cost,
        accepted_iterations,
    );
    optimized
}

fn build_focus_layer_warp(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> FocusLayerWarp {
    let mut bands = Vec::new();

    // A real capture can contain small depth/lens residuals even on the artwork
    // plane. Estimate overlapping local models from the ordinary inlier matches,
    // so the focus stack remains useful when there is no bright foreground object
    // at all. Each model is bounded against the global pose before it can reach
    // the renderer; sparse/unstable regional fits simply fall back to global.
    for &(minimum_source_y, maximum_source_y) in &FOCUS_GENERIC_BAND_RANGES {
        let optimized = optimize_focus_stack_global_homographies_in_generic_region(
            images,
            matches,
            global_homographies,
            (minimum_source_y, maximum_source_y),
        );
        let (homographies, changed) = constrain_focus_band_homographies(
            images,
            global_homographies,
            &optimized,
            (minimum_source_y, maximum_source_y),
            FOCUS_GENERIC_BAND_MAX_DISPLACEMENT_RATIO,
        );
        if !changed {
            continue;
        }
        let source_ranges: HashMap<usize, (f64, f64)> = images
            .iter()
            .map(|image| (image.id, (minimum_source_y, maximum_source_y)))
            .collect();
        bands.push(FocusWarpBand {
            homographies,
            source_ranges,
            source_x_ranges: HashMap::new(),
            relax_foreground_seam: false,
            foreground_only: false,
            physical_edge: false,
        });
    }

    // Long, content-independent physical boundaries are stronger geometric
    // evidence than repeated characters in a narrow local band. Build a band
    // only when the same edge is observed in multiple source images; a rail or
    // other one-off object therefore cannot force a warp by itself.
    bands.extend(build_focus_horizontal_edge_bands(
        images,
        global_homographies,
    ));
    bands.extend(build_focus_vertical_edge_bands(images, global_homographies));

    // A detected near-field/occlusion layer gets a narrower model only when the
    // source frame actually contains it. This is deliberately an optional depth
    // layer, not a rule for white bars or any other named object.
    let minimum_source_y = 0.0;
    let maximum_source_y = FOCUS_FOREGROUND_SCAN_MAX_Y;
    let mut foreground_homographies = optimize_focus_stack_global_homographies_in_region(
        images,
        matches,
        global_homographies,
        Some((minimum_source_y, maximum_source_y)),
    );
    align_focus_foreground_edges(images, &mut foreground_homographies);
    let (foreground_homographies, _) = constrain_focus_band_homographies(
        images,
        global_homographies,
        &foreground_homographies,
        (minimum_source_y, maximum_source_y),
        FOCUS_DEPTH_LAYER_MAX_DISPLACEMENT_RATIO,
    );
    let source_ranges: HashMap<usize, (f64, f64)> = images
        .iter()
        .filter_map(|image| {
            let (minimum, maximum) = image.foreground_range?;
            let minimum = minimum.max(minimum_source_y);
            let maximum = maximum.min(maximum_source_y);
            (minimum < maximum).then_some((image.id, (minimum, maximum)))
        })
        .collect();
    if !source_ranges.is_empty() {
        bands.push(FocusWarpBand {
            homographies: foreground_homographies,
            source_ranges,
            source_x_ranges: HashMap::new(),
            relax_foreground_seam: false,
            foreground_only: true,
            physical_edge: false,
        });
    }
    let shifted_mosaic =
        focus_stack_homographies_have_large_shift(images, global_homographies, projection);
    if shifted_mosaic {
        let before = bands.len();
        bands.retain(|band| band.foreground_only || band.physical_edge);
        println!(
            "  - Shifted focus mosaic: retained {} foreground/physical-edge local warp band(s), filtered {} generic paper-plane band(s)",
            bands.len(),
            before.saturating_sub(bands.len())
        );
    }
    FocusLayerWarp { bands }
}

fn focus_stack_homographies_have_large_shift(
    images: &[ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
) -> bool {
    let Some(first) = images.first() else {
        return false;
    };
    let Some(first_homography) = global_homographies.get(&first.id) else {
        return false;
    };
    let Some(reference_center) =
        mapped_image_center_for_focus_warp(first, first_homography, projection)
    else {
        return false;
    };
    let reference_width = first.width.max(1) as f64;
    let reference_height = first.height.max(1) as f64;
    images.iter().skip(1).any(|image| {
        global_homographies
            .get(&image.id)
            .and_then(|homography| {
                mapped_image_center_for_focus_warp(image, homography, projection)
            })
            .is_some_and(|center| {
                (center.x - reference_center.x).abs() > reference_width * 0.08
                    || (center.y - reference_center.y).abs() > reference_height * 0.08
            })
    })
}

fn mapped_image_center_for_focus_warp(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
) -> Option<Point2<f64>> {
    let center = project_point(
        image,
        image.width as f64 * 0.5,
        image.height as f64 * 0.5,
        projection,
    )?;
    let mapped = homography * Point3::new(center.x, center.y, 1.0);
    if mapped.z.abs() < 1e-8 {
        return None;
    }
    let point = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
    (point.x.is_finite() && point.y.is_finite()).then_some(point)
}

fn constrain_focus_band_homographies(
    images: &[ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    optimized_homographies: &HashMap<usize, Matrix3<f64>>,
    normalized_source_y_range: (f64, f64),
    maximum_displacement_ratio: f64,
) -> (HashMap<usize, Matrix3<f64>>, bool) {
    let mut constrained = HashMap::new();
    let mut changed = false;
    for image in images {
        let Some(global) = global_homographies.get(&image.id).copied() else {
            continue;
        };
        let candidate = optimized_homographies
            .get(&image.id)
            .copied()
            .unwrap_or(global);
        let displacement =
            focus_band_maximum_displacement(image, &global, &candidate, normalized_source_y_range);
        let maximum_allowed =
            image.width.max(image.height).max(1) as f64 * maximum_displacement_ratio;
        let use_candidate = transform_is_stable_for_focus_stack(&candidate, image.dimensions())
            && displacement.is_finite()
            && displacement <= maximum_allowed;
        let selected = if use_candidate { candidate } else { global };
        if use_candidate && displacement > 0.25 {
            changed = true;
        }
        constrained.insert(image.id, selected);
    }
    (constrained, changed)
}

fn focus_band_maximum_displacement(
    image: &ImageInfo,
    global: &Matrix3<f64>,
    candidate: &Matrix3<f64>,
    normalized_source_y_range: (f64, f64),
) -> f64 {
    let source_x_fractions = [0.0, 0.25, 0.5, 0.75, 1.0];
    let source_y_fractions = [
        normalized_source_y_range.0,
        (normalized_source_y_range.0 + normalized_source_y_range.1) * 0.5,
        normalized_source_y_range.1,
    ];
    source_x_fractions
        .into_iter()
        .flat_map(|x_fraction| {
            source_y_fractions.into_iter().map(move |y_fraction| {
                Point2::new(
                    image.width as f64 * x_fraction,
                    image.height as f64 * y_fraction,
                )
            })
        })
        .filter_map(|point| {
            let global_point = transformed_point(global, point)?;
            let candidate_point = transformed_point(candidate, point)?;
            Some((candidate_point - global_point).norm())
        })
        .fold(0.0, f64::max)
}

#[derive(Clone, Copy)]
struct FocusForegroundEdgeSample {
    source: Point2<f64>,
    world: Point2<f64>,
    top_edge: bool,
}

fn foreground_edge_samples(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
) -> Vec<FocusForegroundEdgeSample> {
    let Some((minimum_y, maximum_y)) = image.foreground_range else {
        return Vec::new();
    };
    let width = image.alignment_image.width();
    let height = image.alignment_image.height();
    if width < 64 || height < 128 || minimum_y >= maximum_y {
        return Vec::new();
    }

    let scan_start = ((minimum_y * height as f64).floor() as i32).max(0);
    let scan_end = ((maximum_y * height as f64).ceil() as i32).min(height.saturating_sub(1) as i32);
    let minimum_run = ((height as f64 * 0.018).round() as i32).max(8);
    let maximum_gap = ((height as f64 * 0.006).round() as i32).max(2);
    let column_step = ((width as f64 / 64.0).round() as u32).max(1);
    let source_scale = image.scale_factor;
    let mut samples = Vec::new();

    for x in (0..width).step_by(column_step as usize) {
        let mut runs = Vec::new();
        let mut run_start = None;
        let mut last_occupied = None;
        for y in scan_start..=scan_end {
            let occupied =
                image.alignment_image.get_pixel(x, y as u32)[0] >= FOCUS_FOREGROUND_LUMA_THRESHOLD;
            if occupied {
                if run_start.is_none() {
                    run_start = Some(y);
                }
                last_occupied = Some(y);
            } else if let (Some(start), Some(last)) = (run_start, last_occupied) {
                if y - last > maximum_gap {
                    runs.push((start, last));
                    run_start = None;
                    last_occupied = None;
                }
            }
        }
        if let (Some(start), Some(last)) = (run_start, last_occupied) {
            runs.push((start, last));
        }
        let Some((start, end)) = runs
            .into_iter()
            .max_by_key(|(run_start, run_end)| run_end - run_start)
            .filter(|(run_start, run_end)| run_end - run_start + 1 >= minimum_run)
        else {
            continue;
        };

        for (y, top_edge) in [(start, true), (end, false)] {
            let source = Point2::new(x as f64 * source_scale, y as f64 * source_scale);
            let Some(world) = transformed_point(homography, source) else {
                continue;
            };
            samples.push(FocusForegroundEdgeSample {
                source,
                world,
                top_edge,
            });
        }
    }
    samples
}

fn median_value(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

fn fit_foreground_edge_line(samples: &[FocusForegroundEdgeSample]) -> Option<(f64, f64)> {
    if samples.len() < 2 {
        return None;
    }
    let mut slopes = Vec::new();
    for (index, first) in samples.iter().enumerate() {
        for second in samples.iter().skip(index + 1) {
            let delta_x = second.world.x - first.world.x;
            if delta_x.abs() < 1.0 {
                continue;
            }
            let slope = (second.world.y - first.world.y) / delta_x;
            if slope.is_finite() && slope.abs() < 0.5 {
                slopes.push(slope);
            }
        }
    }
    let slope = median_value(&mut slopes)?;
    let mut intercepts = samples
        .iter()
        .map(|sample| sample.world.y - slope * sample.world.x)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let intercept = median_value(&mut intercepts)?;
    Some((slope, intercept))
}

fn estimate_vertical_foreground_correction(
    samples: &[FocusForegroundEdgeSample],
    top_line: (f64, f64),
    bottom_line: (f64, f64),
) -> Option<Matrix3<f64>> {
    if samples.len() < 6 {
        return None;
    }
    // Fit the two silhouettes independently before solving the correction. A
    // single IRLS fit over all pixels can compromise both edges when one side
    // contains a highlight or a gap; matching the two robust edge lines keeps
    // the layer thickness stable across the whole overlap.
    let top_samples = samples
        .iter()
        .copied()
        .filter(|sample| sample.top_edge)
        .collect::<Vec<_>>();
    let bottom_samples = samples
        .iter()
        .copied()
        .filter(|sample| !sample.top_edge)
        .collect::<Vec<_>>();
    let local_top_line = fit_foreground_edge_line(&top_samples)?;
    let local_bottom_line = fit_foreground_edge_line(&bottom_samples)?;
    let mut reference_x = samples
        .iter()
        .map(|sample| sample.world.x)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let reference_x = median_value(&mut reference_x)?;
    let local_top_y = local_top_line.0 * reference_x + local_top_line.1;
    let local_bottom_y = local_bottom_line.0 * reference_x + local_bottom_line.1;
    let target_top_y = top_line.0 * reference_x + top_line.1;
    let target_bottom_y = bottom_line.0 * reference_x + bottom_line.1;
    let local_separation = local_bottom_y - local_top_y;
    let target_separation = target_bottom_y - target_top_y;
    if !local_separation.is_finite()
        || !target_separation.is_finite()
        || local_separation.abs() < 8.0
    {
        return None;
    }
    let vertical_scale = target_separation / local_separation;
    let shear = top_line.0 - vertical_scale * local_top_line.0;
    let translation = target_top_y - shear * reference_x - vertical_scale * local_top_y;
    let correction = Matrix3::new(
        1.0,
        0.0,
        0.0,
        shear,
        vertical_scale,
        translation,
        0.0,
        0.0,
        1.0,
    );
    if !correction.iter().all(|value| value.is_finite()) {
        return None;
    }

    // The edge fit is allowed to correct the near-field holder, but should not
    // be able to turn a bad edge detection into a second panorama warp.
    let vertical_scale = correction[(1, 1)];
    let shear = correction[(1, 0)];
    if !(0.75..=1.25).contains(&vertical_scale) || shear.abs() > 0.08 {
        return None;
    }
    Some(correction)
}

fn align_focus_foreground_edges(
    images: &[ImageInfo],
    homographies: &mut HashMap<usize, Matrix3<f64>>,
) {
    let mut top_samples = Vec::new();
    let mut bottom_samples = Vec::new();
    let mut samples_by_image = HashMap::<usize, Vec<FocusForegroundEdgeSample>>::new();
    for image in images {
        let Some(homography) = homographies.get(&image.id) else {
            continue;
        };
        let samples = foreground_edge_samples(image, homography);
        top_samples.extend(samples.iter().copied().filter(|sample| sample.top_edge));
        bottom_samples.extend(samples.iter().copied().filter(|sample| !sample.top_edge));
        samples_by_image.insert(image.id, samples);
    }
    let Some(top_line) = fit_foreground_edge_line(&top_samples) else {
        return;
    };
    let Some(bottom_line) = fit_foreground_edge_line(&bottom_samples) else {
        return;
    };

    for (image_id, samples) in samples_by_image {
        let Some(base_homography) = homographies.get(&image_id).copied() else {
            continue;
        };
        let Some(image_info) = images.iter().find(|image| image.id == image_id) else {
            continue;
        };
        let Some(correction) =
            estimate_vertical_foreground_correction(&samples, top_line, bottom_line)
        else {
            continue;
        };
        let corrected = correction * base_homography;
        if !transform_is_stable_for_focus_stack(&corrected, image_info.dimensions()) {
            continue;
        }
        let displacement = samples
            .iter()
            .filter_map(|sample| {
                let baseline = transformed_point(&base_homography, sample.source)?;
                let adjusted = transformed_point(&corrected, sample.source)?;
                Some((adjusted - baseline).norm())
            })
            .fold(0.0, f64::max);
        if displacement.is_finite() && displacement <= 240.0 {
            homographies.insert(image_id, corrected);
        }
    }
}

fn focus_horizontal_edge_line_y(line: &FocusHorizontalEdgeLine, reference_x: f64) -> f64 {
    line.slope * reference_x + line.intercept
}

fn focus_horizontal_edge_line_clusters(
    lines: &[FocusHorizontalEdgeLine],
    coordinate_scale: f64,
) -> Vec<Vec<usize>> {
    if lines.len() < FOCUS_HORIZONTAL_EDGE_MIN_CLUSTER_IMAGES {
        return Vec::new();
    }
    let mut center_xs = lines
        .iter()
        .map(|line| line.world_x_center)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let Some(reference_x) = median_value(&mut center_xs) else {
        return Vec::new();
    };
    let tolerance = coordinate_scale.max(1.0) * FOCUS_HORIZONTAL_EDGE_CLUSTER_TOLERANCE_RATIO;
    let mut ordered_indices = (0..lines.len()).collect::<Vec<_>>();
    ordered_indices.sort_unstable_by(|left, right| {
        focus_horizontal_edge_line_y(&lines[*left], reference_x)
            .total_cmp(&focus_horizontal_edge_line_y(&lines[*right], reference_x))
    });

    let mut clusters = Vec::<Vec<usize>>::new();
    for index in ordered_indices {
        let line = &lines[index];
        let line_y = focus_horizontal_edge_line_y(line, reference_x);
        let mut best_cluster = None;
        let mut best_distance = f64::INFINITY;
        for (cluster_index, cluster) in clusters.iter().enumerate() {
            if cluster
                .iter()
                .any(|member| lines[*member].image_id == line.image_id)
            {
                continue;
            }
            let mut cluster_ys = cluster
                .iter()
                .map(|member| focus_horizontal_edge_line_y(&lines[*member], reference_x))
                .collect::<Vec<_>>();
            let Some(cluster_y) = median_value(&mut cluster_ys) else {
                continue;
            };
            let mut cluster_slopes = cluster
                .iter()
                .map(|member| lines[*member].slope)
                .collect::<Vec<_>>();
            let Some(cluster_slope) = median_value(&mut cluster_slopes) else {
                continue;
            };
            let distance = (line_y - cluster_y).abs();
            if distance <= tolerance
                && (line.slope - cluster_slope).abs() <= FOCUS_HORIZONTAL_EDGE_MAX_SLOPE_DELTA
                && distance < best_distance
            {
                best_cluster = Some(cluster_index);
                best_distance = distance;
            }
        }
        if let Some(cluster_index) = best_cluster {
            clusters[cluster_index].push(index);
        } else {
            clusters.push(vec![index]);
        }
    }

    clusters
        .into_iter()
        .filter(|cluster| {
            cluster
                .iter()
                .map(|index| lines[*index].image_id)
                .collect::<HashSet<_>>()
                .len()
                >= FOCUS_HORIZONTAL_EDGE_MIN_CLUSTER_IMAGES
        })
        .collect()
}

fn focus_horizontal_edge_correction(
    local_line: &FocusHorizontalEdgeLine,
    target_slope: f64,
    target_intercept: f64,
) -> Option<Matrix3<f64>> {
    let shear = target_slope - local_line.slope;
    let translation = target_intercept - local_line.intercept;
    if !shear.is_finite()
        || !translation.is_finite()
        || shear.abs() > FOCUS_HORIZONTAL_EDGE_MAX_SLOPE_DELTA
    {
        return None;
    }
    Some(Matrix3::new(
        1.0,
        0.0,
        0.0,
        shear,
        1.0,
        translation,
        0.0,
        0.0,
        1.0,
    ))
}

fn focus_horizontal_cluster_geometry(
    lines: &[FocusHorizontalEdgeLine],
    cluster: &[usize],
    reference_x: f64,
) -> Option<(f64, f64, f64)> {
    let mut slopes = cluster
        .iter()
        .map(|index| lines[*index].slope)
        .collect::<Vec<_>>();
    let target_slope = median_value(&mut slopes)?;
    let mut target_ys = cluster
        .iter()
        .map(|index| focus_horizontal_edge_line_y(&lines[*index], reference_x))
        .collect::<Vec<_>>();
    let target_y_at_reference = median_value(&mut target_ys)?;
    let target_intercept = target_y_at_reference - target_slope * reference_x;
    Some((target_slope, target_intercept, target_y_at_reference))
}

fn build_focus_horizontal_edge_bands(
    images: &[ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Vec<FocusWarpBand> {
    let mut lines = Vec::new();
    for image in images {
        let Some(homography) = global_homographies.get(&image.id) else {
            continue;
        };
        for &normalized_row in &image.horizontal_edge_rows {
            if let Some(line) = fit_focus_horizontal_edge_line(image, homography, normalized_row) {
                lines.push(line);
            }
        }
    }
    let coordinate_scale = images
        .iter()
        .map(|image| image.width.max(image.height) as f64)
        .fold(1.0, f64::max);
    let raw_line_count = lines.len();
    let lines = deduplicate_focus_horizontal_edge_lines(lines, coordinate_scale);
    if raw_line_count != lines.len() {
        println!(
            "  - Collapsed {} duplicate long-edge candidate line(s)",
            raw_line_count - lines.len()
        );
    }
    if lines.len() < FOCUS_HORIZONTAL_EDGE_MIN_CLUSTER_IMAGES {
        return Vec::new();
    }
    let clusters = focus_horizontal_edge_line_clusters(&lines, coordinate_scale);
    if clusters.is_empty() {
        println!(
            "  - No repeated long-edge consensus found from {} detected edge line(s)",
            lines.len()
        );
        return Vec::new();
    }

    let reference_x = {
        let mut center_xs = lines
            .iter()
            .map(|line| line.world_x_center)
            .collect::<Vec<_>>();
        median_value(&mut center_xs).unwrap_or(0.0)
    };
    let mut bands = Vec::new();
    for cluster in &clusters {
        let Some((target_slope, target_intercept, target_y_at_reference)) =
            focus_horizontal_cluster_geometry(&lines, cluster, reference_x)
        else {
            continue;
        };
        let mut homographies = HashMap::new();
        let mut source_ranges = HashMap::new();
        for &index in cluster {
            let line = lines[index];
            let Some(global) = global_homographies.get(&line.image_id).copied() else {
                continue;
            };
            let Some(image) = images.iter().find(|image| image.id == line.image_id) else {
                continue;
            };
            let Some(correction) =
                focus_horizontal_edge_correction(&line, target_slope, target_intercept)
            else {
                continue;
            };
            let corrected = correction * global;
            if !transform_is_stable_for_focus_stack(&corrected, image.dimensions()) {
                continue;
            }
            let minimum_source_y =
                (line.source_row - FOCUS_HORIZONTAL_EDGE_BAND_HALF_HEIGHT_RATIO).max(0.0);
            let maximum_source_y =
                (line.source_row + FOCUS_HORIZONTAL_EDGE_BAND_HALF_HEIGHT_RATIO).min(1.0);
            let displacement = focus_band_maximum_displacement(
                image,
                &global,
                &corrected,
                (minimum_source_y, maximum_source_y),
            );
            let maximum_allowed = image.width.max(image.height).max(1) as f64
                * FOCUS_HORIZONTAL_EDGE_MAX_DISPLACEMENT_RATIO;
            if !displacement.is_finite() || displacement > maximum_allowed {
                continue;
            }
            homographies.insert(line.image_id, corrected);
            source_ranges.insert(line.image_id, (minimum_source_y, maximum_source_y));
        }
        if homographies.len() < FOCUS_HORIZONTAL_EDGE_MIN_CLUSTER_IMAGES {
            continue;
        }
        let average_error = cluster
            .iter()
            .map(|index| lines[*index].median_error)
            .sum::<f64>()
            / cluster.len().max(1) as f64;
        let maximum_consensus_error =
            FOCUS_HORIZONTAL_EDGE_MAX_CONSENSUS_ERROR_PX.max(coordinate_scale * 0.0005);
        if !average_error.is_finite() || average_error > maximum_consensus_error {
            println!(
                "  - Rejected long-edge consensus band: {} image(s), fit error {:.2}px exceeds {:.2}px",
                homographies.len(),
                average_error,
                maximum_consensus_error
            );
            continue;
        }
        println!(
            "  - Long-edge consensus band: {} image(s), world y {:.1}, fit error {:.2}px",
            homographies.len(),
            target_y_at_reference,
            average_error
        );
        bands.push(FocusWarpBand {
            homographies,
            source_ranges,
            source_x_ranges: HashMap::new(),
            relax_foreground_seam: false,
            foreground_only: false,
            physical_edge: true,
        });
    }

    bands
}

/// Test-only access to the determinism critical station solver internals.
///
/// Property 90 (逐字节可复现, 需求 14.6) is implemented in
/// `panorama_utils::stack_pipeline::properties`, a sibling module that cannot
/// see the private items of this module.  The wrappers below contain no logic
/// of their own: they only widen visibility inside `cfg(test)` so the property
/// test drives the real production functions — the ones tasks 3.1 to 3.4
/// changed — instead of a re-implementation that could drift away from them.
#[cfg(test)]
pub(crate) mod determinism_test_access;

/// Test-only access to the Station_Grouper internals of 需求 1.1 to 1.10.
///
/// Properties 1, 3, 4 and 5 live in `panorama_utils::stack_pipeline::properties`,
/// a sibling module that cannot see the private items of this module.  As with
/// [`determinism_test_access`] the wrappers below carry no logic: they widen
/// visibility inside `cfg(test)` so the property tests drive
/// [`focus_station_link_evidence`], [`focus_station_grouping`],
/// [`split_focus_local_component_by_motion`] and [`focus_frame_median_sharpness`]
/// themselves rather than a re-implementation of them.
#[cfg(test)]
pub(crate) mod grouping_test_access;

/// Test-only access to the production Station_Relation evidence and acceptance
/// decisions used by Properties 28 and 29. These wrappers deliberately contain
/// no thresholds or alternate implementations: they expose the real decisions
/// while keeping the production types private.
#[cfg(test)]
pub(crate) mod station_relation_test_access;

#[cfg(test)]
mod alignment_tests;

#[cfg(test)]
pub(crate) mod acceptance_tests;

#[cfg(test)]
#[path = "panorama_reference_acceptance.rs"]
mod reference_acceptance;

#[cfg(test)]
#[path = "panorama_acceptance_metrics.rs"]
mod acceptance_metrics;

#[cfg(test)]
#[path = "panorama_virtual_tile_tests.rs"]
mod virtual_tile_pyramid_tests;

#[cfg(test)]
#[path = "panorama_structural_oracle.rs"]
mod structural_oracle;
