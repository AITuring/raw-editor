#[cfg(test)]
use crate::app_settings::AppSettings;
use crate::app_settings::load_settings_for_runtime;
use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;
use base64::{Engine as _, engine::general_purpose};
use image::ImageFormat;
#[cfg(test)]
use image::RgbImage;
use image::{DynamicImage, GenericImageView, GrayImage, Rgb32FImage};
use nalgebra::{Matrix3, Point2, Point3};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Runtime};

use crate::formats::{is_psd_file, is_raw_file};
use crate::panorama_utils::registration;
use crate::panorama_utils::stack_pipeline;
use crate::panorama_utils::stack_pipeline::compositor::{
    StackCompositorChoice, select_and_record_run_path,
};
use crate::panorama_utils::stack_pipeline::degradation;
use crate::panorama_utils::stack_pipeline::determinism;
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
#[cfg(test)]
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

mod run_control;

use run_control::{
    FocusRssGuard, PairMatchProgress, ensure_focus_memory_available, focus_stack_cache_key_inputs,
    focus_stack_source_digests, memory_safe_panorama_render_scale, scalable_alignment_budget,
    scalable_preparation_worker_count, source_digest_hex, stack_requires_bounded_memory,
};
#[cfg(test)]
use run_control::{
    MAX_IN_MEMORY_PANORAMA_PIXELS, MAX_SCALABLE_PREPARATION_WORKERS,
    PREPARATION_RAM_PER_WORKER_BYTES, bounded_preparation_worker_count,
    focus_stack_cache_pipeline_version,
};

mod options;

pub(crate) use options::StitchOutcome;
pub use options::{AlignmentMode, BlendMode};

mod source_prep;

use source_prep::{
    absolute_source_path, excluded_source_records, image_pair_has_mixed_focal_lengths,
    images_have_mixed_focal_lengths, is_auxiliary_stitch_file, is_generated_stitch_output,
    load_prepared_stack_source, normalize_stack_raw_display_exposure, prepare_focus_rescue_image,
    raw_display_channel_percentile, scaled_render_image_info, scaled_source_render_homographies,
    selection_has_mixed_focal_lengths, shared_stack_raw_display_gain, source_to_render_rgb32f,
    stack_raw_exposure_reference, undecodable_source_paths,
};
#[cfg(test)]
use source_prep::{
    focal_lengths_span_multiple_lenses, resize_rgb8_area_to_rgb32f, scaled_homographies,
    stack_raw_exposure_value,
};

mod transform_math;

use transform_math::{
    Dsu, RobustTransformFit, estimate_affine, estimate_similarity, estimate_translation,
    focus_transform_disagreement_px, homography_preserves_focus_orientation, matrix3_row_major,
    median_symmetric_error, median_value, normalized_homography, refine_homography_inliers,
    retain_model_inliers, robust_transform_fit, robust_transform_fit_with_threshold,
    symmetric_inlier_indices, symmetric_point_error, symmetric_reprojection_rmse,
    transform_is_stable_for_focus_stack, transformed_point,
};

mod transform_selection;

#[cfg(test)]
use transform_selection::focus_fit_is_competitive;
use transform_selection::{
    find_mixed_focal_similarity_ransac, focus_stack_motion_is_shifted_mosaic,
    mixed_focal_scale_is_plausible, panorama_model_fit, panorama_model_linear_characteristics,
    panorama_transform_is_stable, select_focus_stack_transform, select_large_panorama_transform,
};

mod patch_refine;

#[cfg(test)]
use patch_refine::patch_ncc;
#[cfg(test)]
use patch_refine::{
    FOCUS_FOREGROUND_PATCH_RADIUS, LumaPlane, gradient_patch_corner_energy, refine_patch_position,
};
use patch_refine::{
    collect_dense_focus_region_points, collect_foreground_feature_points,
    refine_match_point_from_homography,
};

mod overlap_quality;

#[cfg(test)]
use overlap_quality::VIRTUAL_TILE_QUALITY_COVERAGE_RADIUS_PX;
use overlap_quality::{
    FocusOverlapQuality, focus_overlap_is_verified, focus_overlap_quality,
    focus_overlap_quality_measurements, focus_overlap_quality_measurements_on_coverage,
    normalized_correlation, panorama_spatial_support, panorama_transform_overlap_support,
    shifted_focus_structure_is_verified, station_relation_overlap_scale,
    virtual_tile_photometric_measurements,
};

mod pair_matching;

use pair_matching::{
    estimate_mixed_focal_coarse_registration, match_image_pair, match_image_pair_with_seed,
};

mod vertical_edge_bands;

#[cfg(test)]
use vertical_edge_bands::{
    FocusVerticalEdgeLine, build_focus_vertical_edge_bands_in_region,
    deduplicate_focus_vertical_edge_lines, fit_focus_vertical_edge_line,
    focus_vertical_edge_line_clusters, vertical_edge_line_samples,
};
use vertical_edge_bands::{build_focus_vertical_edge_bands, detect_vertical_edge_columns};

mod horizontal_edge_bands;

#[cfg(test)]
use horizontal_edge_bands::{
    FocusHorizontalEdgeLine, deduplicate_focus_horizontal_edge_lines,
    focus_horizontal_edge_line_clusters, horizontal_edge_line_samples,
};
use horizontal_edge_bands::{
    build_focus_horizontal_edge_bands, fit_focus_horizontal_edge_line,
    horizontal_edge_row_candidates,
};

mod focus_capture_spine;

#[cfg(test)]
use focus_capture_spine::{FOCUS_CAPTURE_SEQUENCE_GRAPH_WEIGHT_BOOST, focus_capture_stations};
use focus_capture_spine::{
    attach_focus_bracket_orphans_to_pose_graph, focus_alignment_support,
    focus_graph_capture_sequence_boost, focus_position_recovery_accepts, focus_stack_spine_matches,
    focus_unstitched_sources_are_redundant,
};

mod focus_capture_geometry;

use focus_capture_geometry::{
    focus_capture_geometry_diagnostics, focus_capture_geometry_passes,
    focus_capture_group_geometry_diagnostics, focus_capture_group_geometry_passes,
    focus_geometry_capture_order, focus_source_order_group_compositor_order,
};

mod focus_component_recovery;

#[cfg(test)]
use focus_component_recovery::focus_component_bridge_candidates;
use focus_component_recovery::{
    focus_overlap_components, focus_rescue_match_is_verified, recover_focus_component_edges,
};

mod capture_sequence_bridges;

use capture_sequence_bridges::{
    add_capture_sequence_bridges, focus_filename_order_with_overview_gaps, trailing_capture_number,
};

mod stitching_order;

#[cfg(test)]
use stitching_order::pair_harness_disconnected_focus_geometry;
use stitching_order::{build_graph_stitching_order, build_stitching_order};

mod flat_artwork;

#[cfg(test)]
use flat_artwork::flat_artwork_projected_rect;
use flat_artwork::{
    filter_flat_artwork_blurry_sources, flat_artwork_overlap_render_order,
    focus_frame_median_sharpness, refine_flat_artwork_tree_poses,
};

mod focus_auto_order;

#[cfg(test)]
use focus_auto_order::build_focus_sequence_homographies;
use focus_auto_order::{
    canonical_focus_order_direction, focus_auto_order_edge_weight, focus_auto_order_greedy_path,
    focus_auto_order_motion_scale, focus_auto_order_path_max_motion,
    focus_auto_order_path_motion_cost, focus_auto_order_path_score, focus_cycle_edge_factor,
    focus_match_graph_is_dense_continuous_scan, focus_stack_link_transform,
};

mod focus_stack_order;

#[cfg(test)]
use focus_stack_order::build_focus_stack_stitching_order;
use focus_stack_order::build_focus_stack_stitching_order_with_report;

mod focus_global_observations;

#[cfg(test)]
use focus_global_observations::focus_match_points_for_region;
use focus_global_observations::{
    FocusGlobalObservation, FocusPose, focus_global_observations,
    focus_global_observations_for_generic_region, focus_homography_to_pose, focus_normalized_point,
    focus_pose_projection, focus_pose_to_full_homography,
};

mod focus_global_solver;

use focus_global_solver::{
    optimize_focus_stack_global_homographies_in_generic_region,
    optimize_focus_stack_global_homographies_in_region,
    optimize_focus_stack_global_homographies_with_reference,
};

mod focus_layer_warp;

use focus_layer_warp::{
    build_focus_layer_warp, focus_anchor_render_frames, focus_band_maximum_displacement,
};

mod virtual_tile_structure;

#[cfg(test)]
use virtual_tile_structure::virtual_tile_covered_features;
use virtual_tile_structure::{
    VIRTUAL_TILE_FEATURE_COVERAGE_RADIUS, virtual_tile_local_jacobian,
    virtual_tile_overlap_grid_cell, virtual_tile_point_is_covered,
    virtual_tile_predicted_overlap_bounds, virtual_tile_repair_local_scale_ratio,
    virtual_tile_structural_score, virtual_tile_structural_score_points,
};

mod virtual_tile_guided_match;

use virtual_tile_guided_match::{
    VIRTUAL_TILE_GUIDED_MAX_DESCRIPTOR_DISTANCE, VIRTUAL_TILE_GUIDED_MAX_LOCAL_ORIENTATION_DEGREES,
    VIRTUAL_TILE_GUIDED_MIN_GRADIENT_NCC, VIRTUAL_TILE_GUIDED_MIN_OCCUPIED_CELLS,
    VIRTUAL_TILE_GUIDED_MIN_PATCH_NCC, VIRTUAL_TILE_GUIDED_MIN_PEAK_MARGIN,
    VIRTUAL_TILE_GUIDED_PER_CELL_QUOTA, virtual_tile_descriptor_distance,
    virtual_tile_guided_search_radius,
};
#[cfg(test)]
use virtual_tile_guided_match::{
    VirtualTileStationMatchOutcome, virtual_tile_unseeded_station_match,
};

mod virtual_tile_residual;

#[cfg(test)]
use virtual_tile_residual::{
    VIRTUAL_TILE_DIRECT_MAX_NON_TRANSLATION_RESIDUAL_PX,
    VIRTUAL_TILE_DIRECT_MAX_RESIDUAL_DISPLACEMENT_PX, VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MAX,
    VIRTUAL_TILE_DIRECT_RESIDUAL_SCALE_MIN, VirtualTileResidualModelOutcome,
    virtual_tile_fit_measured_residual_model, virtual_tile_full_tile_bounds,
    virtual_tile_residual_correction_is_bounded,
};
use virtual_tile_residual::{
    VirtualTileResidualSelection, residual_distribution,
    virtual_tile_fit_measured_residual_model_with_threshold, virtual_tile_measured_residuals,
};

mod virtual_tile_direct_match;

#[cfg(test)]
use virtual_tile_direct_match::{
    VIRTUAL_TILE_DIRECT_MAX_GRID_PROBES, virtual_tile_measured_reverse_transform,
    virtual_tile_round_trip_is_consistent,
};
use virtual_tile_direct_match::{
    VIRTUAL_TILE_DIRECT_PATCH_RADIUS, VIRTUAL_TILE_DIRECT_ROUND_TRIP_TOLERANCE_PX,
    virtual_tile_coverage_contains_rectangle, virtual_tile_direct_patch_support_is_covered,
    virtual_tile_direct_station_match, virtual_tile_direct_station_match_with_structure,
};

const FULL_RES_RANSAC_INLIER_THRESHOLD: f64 = 12.0;
const FULL_RES_REFINEMENT_THRESHOLD: f64 = 2.5;
const FOCUS_MODEL_INLIER_THRESHOLD: f64 = 6.0;
const FOCUS_MODEL_MIN_INLIERS: usize = 8;
// A handheld angle change over a planar artwork can leave a real projective
// residual even when the optical centre barely moves.  Do not force that
// residual through an affine model: the resulting edge error is large enough
// to make every focus layer look soft.  Requiring broad support keeps a small
// repeated stroke from enabling a projective warp by itself.
const FOCUS_PROJECTIVE_MIN_SPATIAL_SUPPORT: f64 = 0.22;
const FOCUS_LOCAL_MODEL_MIN_INLIERS: usize = 6;
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
const CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX: f64 = 2.0;
const CLOSURE_DIRECT_PAIR_P95_LIMIT_PX: f64 = 3.0;
const CLOSURE_MAX_CORNER_CORRECTION_PX: f64 = 256.0;
const CLOSURE_OVERLAP_SHORT_SIDE_CORRECTION_RATIO: f64 = 0.10;
const CLOSURE_RELATIVE_RESIDUAL_DECREASE: f64 = 0.0001;
const CLOSURE_MAX_ITERATIONS: usize = 100;
const FOCUS_GLOBAL_MAX_LINEAR_ADJUSTMENT: f64 = 0.12;
const FOCUS_GLOBAL_MAX_TRANSLATION_ADJUSTMENT: f64 = 0.18;
const FOCUS_GLOBAL_MAX_PROJECTIVE_ADJUSTMENT: f64 = 0.02;
// A random file-picker order must not become the focus-stack layer order. For
// a medium-sized stack, inspect every pair so the capture path can be rebuilt
// from image evidence rather than from filenames or import order.
const FOCUS_AUTO_ORDER_EXHAUSTIVE_MAX_SOURCES: usize = 64;
const SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES: usize = 65;
const SCALE_ROBUST_EXHAUSTIVE_MAX_SOURCES: usize = 96;
pub(crate) const FOCUS_FOREGROUND_LUMA_THRESHOLD: u8 = 150;
const FOCUS_FOREGROUND_MIN_BRIGHT_FRACTION: f64 = 0.35;
const FOCUS_FOREGROUND_SCAN_MAX_Y: f64 = 0.45;
const FOCUS_FOREGROUND_MIN_HEIGHT_RATIO: f64 = 0.025;
const FOCUS_HORIZONTAL_EDGE_SEARCH_RADIUS: i32 = 32;
const FOCUS_GROUP_ROLL_MIN_LINES: usize = 2;
const FOCUS_GROUP_ROLL_MAX_SLOPE_MAD: f64 = 0.018;
const FOCUS_GROUP_ROLL_MAX_CORRECTION_RADIANS: f64 = 0.045;
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
const SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION: usize = 10;
const PANORAMA_MODEL_INLIER_THRESHOLD: f64 = 8.0;
const PANORAMA_MODEL_MIN_INLIERS: usize = 8;
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
const MAX_STITCH_SOURCE_IMAGES: usize = 500;
const MAX_RETAINED_STACK_PIXELS: u64 = 120_000_000;

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

#[derive(Clone)]
struct SourceRawLayerDiagnosticModel {
    left_source: usize,
    right_source: usize,
    record_index: usize,
    transform: Matrix3<f64>,
    spatial_support: f64,
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
