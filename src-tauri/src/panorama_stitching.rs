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
#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Runtime};

use crate::formats::{is_psd_file, is_raw_file};
#[cfg(test)]
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
use crate::panorama_utils::stack_pipeline::quality_gate;
use crate::panorama_utils::stack_pipeline::quality_gate_runner;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use crate::panorama_utils::stack_pipeline::residual_warp;
use crate::panorama_utils::stack_pipeline::resources;
use crate::panorama_utils::stack_pipeline::tone;
use crate::panorama_utils::stack_pipeline::topology;
#[cfg(test)]
use crate::panorama_utils::stack_pipeline::virtual_tile;
use crate::panorama_utils::stitching::Projection;
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

mod virtual_tile_planes;

#[cfg(test)]
use virtual_tile_planes::virtual_tile_covered_luminance;
#[cfg(test)]
use virtual_tile_planes::{
    VIRTUAL_TILE_MIN_MATCH_WINDOW_NATIVE_PX, VIRTUAL_TILE_PYRAMID_MIN_LEVEL_LONG_SIDE,
    VIRTUAL_TILE_RELATION_QUALITY_OCTAVES, virtual_tile_plane_image_info,
    virtual_tile_relation_quality_plane,
};
use virtual_tile_planes::{
    VIRTUAL_TILE_RELATION_QUALITY_BLUR_SIGMA, VIRTUAL_TILE_SEED_PRIOR_FREE,
    VIRTUAL_TILE_SEED_RENDERING_PRIOR, VirtualTileGlobalSearchPlane, VirtualTileMatchPlane,
    VirtualTileMatchPyramid, install_virtual_tile_match_planes, render_virtual_tile_match_plane,
    virtual_tile_pyramid_factors,
};

mod virtual_tile_global_search;

#[cfg(test)]
use virtual_tile_global_search::{
    VIRTUAL_TILE_GLOBAL_MAX_SAMPLES, VIRTUAL_TILE_GLOBAL_MIN_OVERLAP_SAMPLES,
    virtual_tile_masked_similarity_score, virtual_tile_prior_free_seed_fallback_conditions,
};
use virtual_tile_global_search::{
    VIRTUAL_TILE_GLOBAL_TOP_K, VIRTUAL_TILE_SEED_GLOBAL_COVERAGE, VIRTUAL_TILE_SEED_SOURCE_MATCH,
    VirtualTileGlobalProposal, virtual_tile_global_similarity_proposals,
    virtual_tile_prior_free_relation, virtual_tile_prior_free_seed_fallback_allowed,
    virtual_tile_source_match_proposals, virtual_tile_translation,
};

mod virtual_tile_pyramid_match;

use virtual_tile_pyramid_match::virtual_tile_pyramid_station_match;
#[cfg(test)]
use virtual_tile_pyramid_match::{
    VIRTUAL_TILE_PYRAMID_COARSE_INLIER_THRESHOLD_PX, native_station_affine_candidate,
    native_station_regularized_candidate, robust_native_affine,
    virtual_tile_coarse_failure_can_retry,
};

mod virtual_tile_polish;

use virtual_tile_polish::{
    VIRTUAL_TILE_POLISH_INLIER_THRESHOLD_NATIVE_PX, VirtualTilePolishModel,
    virtual_tile_hypotheses_are_equivalent, virtual_tile_ncc, virtual_tile_polish_fit,
    virtual_tile_polished_station_match,
};
#[cfg(test)]
use virtual_tile_polish::{
    VirtualTilePolishCounts, virtual_tile_best_window_offset, virtual_tile_polish_correspondences,
};

mod virtual_tile_relations;

use virtual_tile_relations::{
    VirtualTileRelationHypotheses, focus_group_candidate_key, log_virtual_tile_candidate,
    native_photometric_pairs, populate_station_relation_candidate_geometry,
    replace_virtual_tile_candidate_record, single_layer_station_evidence_discard,
    virtual_tile_authoritative_pairs, virtual_tile_measure_prior_relations,
    virtual_tile_prior_free_candidate_pairs,
};
#[cfg(test)]
use virtual_tile_relations::{
    native_photometric_measurements_are_usable, virtual_tile_station_match_outcome,
    virtual_tile_station_matches,
};

mod virtual_tile_hypothesis;

#[cfg(test)]
use virtual_tile_hypothesis::{
    MeasuredVirtualTileHypothesis, virtual_tile_hypothesis_holdout_key,
    virtual_tile_hypothesis_model_rank,
};
use virtual_tile_hypothesis::{
    select_virtual_tile_relation_hypotheses, select_virtual_tile_relation_hypothesis,
    virtual_tile_hypothesis_matrix_key, virtual_tile_provisional_authoritative_candidate,
};

mod virtual_tile_geometry_repair;

#[cfg(test)]
use virtual_tile_geometry_repair::VirtualTilePriorRepair;
use virtual_tile_geometry_repair::{
    VirtualTilePriorProposal, apply_virtual_tile_prior_repair, plan_virtual_tile_prior_repairs,
    refine_virtual_tile_joint_geometry, virtual_tile_prior_repair_is_bounded,
};

mod source_raw_direct;

use source_raw_direct::{
    SOURCE_RAW_DIRECT_MAX_SEEDS, SOURCE_RAW_DIRECT_PATCH_GATES, SOURCE_RAW_DIRECT_PATCH_RADIUS,
    SOURCE_RAW_DIRECT_SEARCH_RADIUS, SourceRawCorrelationProposal, SourceRawCorrelationPyramid,
    SourceRawDirectFold, SourceRawDirectPyramid, source_raw_correlation_proposals,
    source_raw_direct_grid_centers, source_raw_direct_level_transform,
    source_raw_direct_raw_transform, source_raw_direct_refine,
};

mod source_raw_direct_basin;

#[cfg(test)]
use source_raw_direct_basin::source_raw_direct_localize_holdout;
use source_raw_direct_basin::{source_raw_direct_closure, source_raw_direct_select_basin};

mod source_raw_direct_pairs;

#[cfg(test)]
use source_raw_direct_pairs::source_raw_direct_pair_diagnostics;

mod source_raw_layer_diagnostics;

#[cfg(test)]
use source_raw_layer_diagnostics::source_raw_layer_pair_diagnostics;
use source_raw_layer_diagnostics::{
    SourceRawLayerDiagnosticModel, source_raw_independent_observations,
};
#[cfg(test)]
use source_raw_layer_diagnostics::{
    source_raw_consensus_clusters, source_raw_correlation_guided_matches,
    source_raw_descriptor_proposals, source_raw_normalize_validated_match,
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
