#[cfg(test)]
use crate::app_settings::AppSettings;
use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;
use base64::{Engine as _, engine::general_purpose};
use image::ImageFormat;
#[cfg(test)]
use image::RgbImage;
use image::{DynamicImage, GenericImageView};
#[cfg(test)]
use image::{GrayImage, Rgb32FImage};
#[cfg(test)]
use nalgebra::{Matrix3, Point2, Point3};
#[cfg(test)]
use rayon::ThreadPoolBuilder;
#[cfg(test)]
use rayon::prelude::*;
#[cfg(test)]
use std::collections::BTreeMap;
#[cfg(test)]
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Cursor;
#[cfg(test)]
use std::path::Path;
#[cfg(test)]
use std::time::Instant;
use tauri::Emitter;

#[cfg(test)]
use crate::panorama_utils::registration;
#[cfg(test)]
use crate::panorama_utils::stack_pipeline::degradation;
#[cfg(test)]
use crate::panorama_utils::stack_pipeline::determinism;
#[cfg(test)]
use crate::panorama_utils::stack_pipeline::quality_gate;
#[cfg(test)]
use crate::panorama_utils::stack_pipeline::quality_gate_runner;
#[cfg(test)]
use crate::panorama_utils::stack_pipeline::report as stack_report;
#[cfg(test)]
use crate::panorama_utils::stack_pipeline::residual_warp;
#[cfg(test)]
use crate::panorama_utils::stack_pipeline::topology;
#[cfg(test)]
use crate::panorama_utils::stack_pipeline::virtual_tile;
#[cfg(test)]
use crate::panorama_utils::stitching::Projection;
#[cfg(test)]
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

mod stitch_images;

pub(crate) use stitch_images::stitch_images_with_options;

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
