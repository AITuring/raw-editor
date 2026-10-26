use crate::app_settings::{AppSettings, load_settings_for_runtime};
use crate::app_state::AppState;
use crate::file_management::{parse_virtual_path, read_file_mapped};
use base64::{Engine as _, engine::general_purpose};
use image::ImageFormat;
use image::{ColorType, DynamicImage, GenericImageView, GrayImage, Rgb32FImage, RgbImage};
use nalgebra::{Matrix2, Matrix3, Point2, Point3};
use rand::prelude::*;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Runtime};

use crate::formats::is_raw_file;
use crate::image_processing::apply_cpu_default_raw_processing;
use crate::panorama_utils::registration;
use crate::panorama_utils::stack_pipeline::compositor::{
    StackCompositorChoice, select_and_record_run_path,
};
use crate::panorama_utils::stack_pipeline::degradation;
use crate::panorama_utils::stack_pipeline::determinism;
use crate::panorama_utils::stack_pipeline::determinism::{sorted_keys, sorted_pairs};
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
use crate::panorama_utils::stack_pipeline::virtual_tile;
use crate::panorama_utils::stitching::{Projection, project_point};
use crate::panorama_utils::{processing, stitching};

#[path = "panorama_station_solver.rs"]
mod station_solver;
pub(crate) use station_solver::*;

#[path = "panorama_focus_warp.rs"]
mod focus_warp;
pub(crate) use focus_warp::*;

#[path = "panorama_focus_warp_render.rs"]
mod focus_warp_render;
#[allow(unused_imports)]
use focus_warp_render::*;

#[path = "panorama_shared_helpers.rs"]
mod shared_helpers;
#[allow(unused_imports)]
use shared_helpers::*;

#[path = "panorama_overlap_geometry.rs"]
mod overlap_geometry;
#[allow(unused_imports)]
use overlap_geometry::*;

#[path = "panorama_source_preparation.rs"]
mod source_preparation;
#[allow(unused_imports)]
use source_preparation::*;

#[path = "panorama_focus_grouping.rs"]
mod focus_grouping;
#[allow(unused_imports)]
use focus_grouping::*;

#[path = "panorama_feature_matching.rs"]
mod feature_matching;
use feature_matching::*;

#[path = "panorama_orchestration.rs"]
mod orchestration;
pub(crate) use orchestration::*;

#[path = "panorama_virtual_tile_solver.rs"]
mod virtual_tile_solver;
use virtual_tile_solver::*;

#[path = "panorama_focus_pose_solver.rs"]
mod focus_pose_solver;
use focus_pose_solver::*;

#[path = "panorama_focus_order.rs"]
mod focus_order;
use focus_order::*;

#[path = "panorama_focus_global_optimizer.rs"]
mod focus_global_optimizer;
use focus_global_optimizer::*;

#[path = "panorama_execution.rs"]
mod execution;
pub(crate) use execution::stitch_images_with_options;
#[allow(unused_imports)]
use execution::{is_auxiliary_stitch_file, is_generated_stitch_output};

#[tauri::command]
pub async fn stitch_panorama(
    paths: Vec<String>,
    app_handle: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    execution::stitch_panorama(paths, app_handle, state).await
}

#[tauri::command]
pub async fn save_panorama(
    first_path_str: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    execution::save_panorama(first_path_str, state).await
}

pub use crate::panorama_types::{
    BRIEF_DESCRIPTOR_SIZE, Descriptor, Feature, ImageInfo, KeyPoint, Match, MatchInfo,
};
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

/// Compare artwork structure rather than single-pixel silk texture. This is
/// only a second check of a feature-supported shifted scan, never a source of
/// synthetic correspondences or a replacement for station-relation gates.
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

fn canonical_match_direction(
    images: &[ImageInfo],
    first: usize,
    second: usize,
) -> (usize, usize, bool) {
    let first_has_features =
        !images[first].features.is_empty() || !images[first].top_features.is_empty();
    let second_has_features =
        !images[second].features.is_empty() || !images[second].top_features.is_empty();
    let content_order = first_has_features && second_has_features;
    let first_before_second = if content_order {
        focus_image_direction_key(&images[first]) <= focus_image_direction_key(&images[second])
    } else {
        images[first].filename <= images[second].filename
    };
    if first_before_second {
        (first, second, false)
    } else {
        (second, first, true)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CanonicalRelationView {
    source_index: usize,
    target_index: usize,
    inverted: bool,
    homography: Matrix3<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CanonicalRelationViewError {
    IndexOutOfBounds {
        left: usize,
        right: usize,
        image_count: usize,
    },
    NonInvertible,
}

fn canonical_relation_view(
    images: &[ImageInfo],
    left: usize,
    right: usize,
    relation: &MatchInfo,
) -> Result<CanonicalRelationView, CanonicalRelationViewError> {
    if left >= images.len() || right >= images.len() {
        return Err(CanonicalRelationViewError::IndexOutOfBounds {
            left,
            right,
            image_count: images.len(),
        });
    }
    let (source_index, target_index, inverted) = canonical_match_direction(images, left, right);
    let homography = match relation.canonical_homography {
        Some(canonical) => canonical,
        None if inverted => relation
            .homography
            .try_inverse()
            .ok_or(CanonicalRelationViewError::NonInvertible)?,
        None => relation.homography,
    };
    if homography.try_inverse().is_none() {
        return Err(CanonicalRelationViewError::NonInvertible);
    }
    Ok(CanonicalRelationView {
        source_index,
        target_index,
        inverted,
        homography,
    })
}

/// Return a deterministic direction key from image content rather than its
/// path or import position. Matching is mildly asymmetric because descriptor
/// ratio tests and patch refinement use the source image as the reference; a
/// path-based direction therefore made renamed/shuffled imports choose a
/// different transform. Synthetic unit-test images have no features and keep
/// the filename fallback above.
fn focus_image_direction_key(image: &ImageInfo) -> (u64, usize, u32, u32) {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    image.width.hash(&mut hasher);
    image.height.hash(&mut hasher);
    image.features.len().hash(&mut hasher);
    image.top_features.len().hash(&mut hasher);
    for feature in image
        .features
        .iter()
        .chain(image.top_features.iter())
        .take(16)
    {
        feature.keypoint.x.hash(&mut hasher);
        feature.keypoint.y.hash(&mut hasher);
        feature.descriptor.hash(&mut hasher);
        feature.support_scale.to_bits().hash(&mut hasher);
    }
    (
        hasher.finish(),
        image.features.len().max(image.top_features.len()),
        image.width,
        image.height,
    )
}

/// Singular values `(σ1, σ2)`, `σ1 ≥ σ2`, of the upper-left 2×2 block of a
/// homography.
///
/// The block is read after normalising the matrix by `h[(2, 2)]`: a homography
/// is only defined up to an overall scale, so `2 · I` and `I` describe the same
/// mapping and must report the same singular values.
fn homography_linear_singular_values(transform: &Matrix3<f64>) -> Option<(f64, f64)> {
    let normaliser = transform[(2, 2)];
    let scale = if normaliser.is_finite() && normaliser.abs() > f64::EPSILON {
        1.0 / normaliser
    } else {
        1.0
    };
    let linear = Matrix2::new(
        transform[(0, 0)] * scale,
        transform[(0, 1)] * scale,
        transform[(1, 0)] * scale,
        transform[(1, 1)] * scale,
    );
    if linear.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let values = linear.singular_values();
    let (first, second) = (values[0], values[1]);
    (first.is_finite() && second.is_finite() && second >= 0.0)
        .then_some((first.max(second), first.min(second)))
}

/// Scale ratio of a homography: the geometric mean `sqrt(σ1 · σ2)` of the
/// singular values of its linear part.
///
/// One measurement, two thresholds: 需求 1.1 requires `[0.98, 1.02]` between two
/// members of one Capture_Station, 需求 6.3 requires `[0.95, 1.05]` for an
/// inter-station relation. Callers own the threshold; this function owns the
/// measurement so both read the same number.
fn homography_scale_ratio(transform: &Matrix3<f64>) -> Option<f64> {
    let (first, second) = homography_linear_singular_values(transform)?;
    let ratio = (first * second).sqrt();
    (ratio.is_finite() && ratio > 0.0).then_some(ratio)
}

/// Projective scale belongs to the measured overlap, not the coordinate
/// origin's linear block. Recentring either tile must not change this gate.
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

/// Sample the valid station-tile rectangle and return its Local_Scale range.
///
/// For `H: tile -> world`, Local_Scale is `sqrt(abs(det J_H))`.  The analytic
/// quotient-rule Jacobian keeps this deterministic and measures projective
/// variation without constraining any of H's eight free parameters (需求 6.6).
fn station_pose_local_scale_range(
    transform: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> Option<(f64, f64)> {
    if dimensions.0 == 0 || dimensions.1 == 0 || transform.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0f64;
    let sample_denominator = (STATION_POSE_LOCAL_SCALE_SAMPLES_PER_AXIS - 1) as f64;
    let maximum_x = f64::from(dimensions.0.saturating_sub(1));
    let maximum_y = f64::from(dimensions.1.saturating_sub(1));
    for sample_y in 0..STATION_POSE_LOCAL_SCALE_SAMPLES_PER_AXIS {
        let y = maximum_y * sample_y as f64 / sample_denominator;
        for sample_x in 0..STATION_POSE_LOCAL_SCALE_SAMPLES_PER_AXIS {
            let x = maximum_x * sample_x as f64 / sample_denominator;
            let denominator = transform[(2, 0)] * x + transform[(2, 1)] * y + transform[(2, 2)];
            if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
                return None;
            }
            let world_x_numerator =
                transform[(0, 0)] * x + transform[(0, 1)] * y + transform[(0, 2)];
            let world_y_numerator =
                transform[(1, 0)] * x + transform[(1, 1)] * y + transform[(1, 2)];
            let denominator_squared = denominator * denominator;
            let dx_dx = (transform[(0, 0)] * denominator - world_x_numerator * transform[(2, 0)])
                / denominator_squared;
            let dx_dy = (transform[(0, 1)] * denominator - world_x_numerator * transform[(2, 1)])
                / denominator_squared;
            let dy_dx = (transform[(1, 0)] * denominator - world_y_numerator * transform[(2, 0)])
                / denominator_squared;
            let dy_dy = (transform[(1, 1)] * denominator - world_y_numerator * transform[(2, 1)])
                / denominator_squared;
            let local_scale = (dx_dx * dy_dy - dx_dy * dy_dx).abs().sqrt();
            if !local_scale.is_finite() || local_scale <= 0.0 {
                return None;
            }
            minimum = minimum.min(local_scale);
            maximum = maximum.max(local_scale);
        }
    }
    (minimum.is_finite() && maximum.is_finite() && minimum > 0.0).then_some((minimum, maximum))
}

fn match_has_spatial_support(
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
) -> bool {
    if points.len() < PANORAMA_MODEL_MIN_INLIERS {
        return false;
    }
    let spread = |side: usize, dimensions: (u32, u32)| {
        let mut xs = points
            .iter()
            .map(
                |(source, target)| {
                    if side == 0 { source.x } else { target.x }
                },
            )
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let mut ys = points
            .iter()
            .map(
                |(source, target)| {
                    if side == 0 { source.y } else { target.y }
                },
            )
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if xs.len() < PANORAMA_MODEL_MIN_INLIERS || ys.len() < PANORAMA_MODEL_MIN_INLIERS {
            return (0.0, 0.0, 0.0);
        }
        xs.sort_unstable_by(f64::total_cmp);
        ys.sort_unstable_by(f64::total_cmp);
        let lower = ((xs.len() as f64 - 1.0) * 0.05).round() as usize;
        let upper = ((xs.len() as f64 - 1.0) * 0.95).round() as usize;
        let y_lower = ((ys.len() as f64 - 1.0) * 0.05).round() as usize;
        let y_upper = ((ys.len() as f64 - 1.0) * 0.95).round() as usize;
        let span_x = (xs[upper] - xs[lower]) / dimensions.0.max(1) as f64;
        let span_y = (ys[y_upper] - ys[y_lower]) / dimensions.1.max(1) as f64;
        (span_x, span_y, span_x.max(0.0) * span_y.max(0.0))
    };
    let (source_x, source_y, source_area) = spread(0, source_dimensions);
    let (target_x, target_y, target_area) = spread(1, target_dimensions);
    let source_long = source_x.max(source_y);
    let target_long = target_x.max(target_y);
    let source_short = source_x.min(source_y);
    let target_short = target_x.min(target_y);
    // A valid tile overlap normally covers a broad strip.  Permit a very thin
    // strip (for a scan along a long edge), but reject compact repeated-stroke
    // clusters that otherwise satisfy the minimum inlier count.
    source_long >= 0.12
        && target_long >= 0.12
        && source_short >= 0.006
        && target_short >= 0.006
        && source_area >= 0.0012
        && target_area >= 0.0012
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentMode {
    Auto,
    Perspective,
    Cylindrical,
    Spherical,
    Position,
}

impl AlignmentMode {
    pub fn from_wire(value: &str) -> Self {
        match value {
            "perspective" => Self::Perspective,
            "cylindrical" => Self::Cylindrical,
            "spherical" => Self::Spherical,
            "position" => Self::Position,
            _ => Self::Auto,
        }
    }

    fn projection_for(self, blend_mode: BlendMode) -> Projection {
        match self {
            Self::Cylindrical => Projection::Cylindrical,
            Self::Spherical => Projection::Spherical,
            Self::Perspective | Self::Position => Projection::Planar,
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

/// Test-only access to the determinism critical station solver internals.
///
/// Property 90 (逐字节可复现, 需求 14.6) is implemented in
/// `panorama_utils::stack_pipeline::properties`, a sibling module that cannot
/// see the private items of this module.  The wrappers below contain no logic
/// of their own: they only widen visibility inside `cfg(test)` so the property
/// test drives the real production functions — the ones tasks 3.1 to 3.4
/// changed — instead of a re-implementation that could drift away from them.
#[cfg(test)]
pub(crate) mod determinism_test_access {
    use super::*;

    /// One capture station in a shape a sibling module can hold.
    ///
    /// [`FocusCaptureGroup`] itself stays private; this carries the same data
    /// with the visibility the property test needs.
    pub(crate) struct StationView {
        pub(crate) members: Vec<usize>,
        pub(crate) anchor: usize,
        pub(crate) local_to_anchor: HashMap<usize, Matrix3<f64>>,
    }

    fn into_views(groups: Vec<FocusCaptureGroup>) -> Vec<StationView> {
        groups
            .into_iter()
            .map(|group| StationView {
                members: group.members,
                anchor: group.anchor,
                local_to_anchor: group.local_to_anchor,
            })
            .collect()
    }

    fn from_views(stations: &[StationView]) -> Vec<FocusCaptureGroup> {
        stations
            .iter()
            .map(|station| FocusCaptureGroup {
                members: station.members.clone(),
                anchor: station.anchor,
                local_to_anchor: station.local_to_anchor.clone(),
            })
            .collect()
    }

    /// [`focus_capture_groups`] — station membership and station order.
    pub(crate) fn capture_stations(
        images: &[ImageInfo],
        matches: &HashMap<(usize, usize), MatchInfo>,
        locked_homographies: &HashMap<usize, Matrix3<f64>>,
    ) -> Option<Vec<StationView>> {
        focus_capture_groups(images, matches, locked_homographies)
            .map(|(groups, _)| into_views(groups))
    }

    /// [`canonical_match_direction`] — pair orientation derived from feature
    /// content whenever both synthetic images carry content signatures.
    pub(crate) fn canonical_pair_direction(
        images: &[ImageInfo],
        first: usize,
        second: usize,
    ) -> (usize, usize, bool) {
        canonical_match_direction(images, first, second)
    }

    /// Full station-pose test outcome, including the connectivity report that
    /// distinguishes a valid solved graph from the explicit requirements
    /// 6.7/12.4 disconnected rejection.
    pub(crate) struct StationPoseSolveView {
        pub(crate) poses: HashMap<usize, Matrix3<f64>>,
        pub(crate) translation_geometry_verified: bool,
        pub(crate) connectivity: stack_report::ConnectivityReport,
    }

    /// [`solve_focus_capture_group_poses_with_report`] — the station-level pose
    /// solve together with its machine-readable connectivity outcome.
    pub(crate) fn solve_station_poses_with_report(
        images: &[ImageInfo],
        matches: &HashMap<(usize, usize), MatchInfo>,
        locked_homographies: &HashMap<usize, Matrix3<f64>>,
        reference_index: usize,
    ) -> StationPoseSolveView {
        let mut translation_geometry_verified = false;
        let mut closure_report = stack_report::ClosureReport::default();
        let mut station_relations_report = stack_report::StationRelationsReport::default();
        let poses = solve_focus_capture_group_poses_with_report(
            images,
            matches,
            locked_homographies,
            reference_index,
            &mut translation_geometry_verified,
            &mut closure_report,
            &mut station_relations_report,
        );
        StationPoseSolveView {
            poses,
            translation_geometry_verified,
            connectivity: station_relations_report.connectivity,
        }
    }

    /// [`solve_focus_capture_group_poses`] — the station level pose solve.
    ///
    /// Returns the solved per-source world homographies together with the
    /// `translation_geometry_verified` flag the caller receives out-of-band.
    pub(crate) fn solve_station_poses(
        images: &[ImageInfo],
        matches: &HashMap<(usize, usize), MatchInfo>,
        locked_homographies: &HashMap<usize, Matrix3<f64>>,
        reference_index: usize,
    ) -> (HashMap<usize, Matrix3<f64>>, bool) {
        let outcome =
            solve_station_poses_with_report(images, matches, locked_homographies, reference_index);
        (outcome.poses, outcome.translation_geometry_verified)
    }

    /// [`focus_source_order_group_compositor_order`] — the seam order derived
    /// from the production Capture_Topology_Model row/column indices.
    pub(crate) fn station_compositor_order(
        images: &[ImageInfo],
        stations: &[StationView],
        station_homographies: &HashMap<usize, Matrix3<f64>>,
    ) -> Vec<usize> {
        let mut groups = from_views(stations);
        let _ = canonicalize_station_geometry_anchors(images, &mut groups, station_homographies);
        let station_topology = focus_station_topology(images, &groups, station_homographies);
        focus_source_order_group_compositor_order(groups.len(), &station_topology)
    }
}

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
pub(crate) mod grouping_test_access {
    use super::*;

    /// The four 需求 1.1 measurements of one pairwise relation plus the verdict.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) struct EvidenceView {
        pub(crate) inliers: usize,
        pub(crate) overlap_ncc: f64,
        pub(crate) overlap_ncc_measured: bool,
        pub(crate) spatial_support: f64,
        pub(crate) scale_ratio: f64,
        pub(crate) score: f64,
        /// [`StationEvidence::accepts`] — the membership decision itself.
        pub(crate) accepts: bool,
    }

    /// One recorded Capture_Station split (需求 1.4 / 1.10).
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) struct SplitView {
        pub(crate) component: usize,
        pub(crate) position: usize,
        pub(crate) motion_ratio: f64,
        pub(crate) reason: &'static str,
    }

    impl From<&StationSplit> for SplitView {
        fn from(split: &StationSplit) -> Self {
            Self {
                component: split.component,
                position: split.position,
                motion_ratio: split.motion_ratio,
                reason: split.reason,
            }
        }
    }

    /// [`StationGroupingOutcome`] in a shape a sibling module can hold.
    #[derive(Clone, Debug, Default)]
    pub(crate) struct GroupingView {
        pub(crate) components: Vec<Vec<usize>>,
        pub(crate) isolated: Vec<usize>,
        pub(crate) splits: Vec<SplitView>,
    }

    /// The 需求 1.1 / 1.4 / 1.5 thresholds, so a property test can state the
    /// requirement's own numbers *and* assert the production constants are them.
    pub(crate) const MIN_INLIER_MATCHES: usize = STATION_MIN_INLIER_MATCHES;
    pub(crate) const MIN_OVERLAP_NCC: f64 = STATION_MIN_OVERLAP_NCC;
    pub(crate) const MIN_INLIER_SPATIAL_SUPPORT: f64 = STATION_MIN_INLIER_SPATIAL_SUPPORT;
    pub(crate) const SCALE_RATIO_MIN: f64 = STATION_SCALE_RATIO_MIN;
    pub(crate) const SCALE_RATIO_MAX: f64 = STATION_SCALE_RATIO_MAX;
    pub(crate) const MAX_ACCUMULATED_CENTER_MOTION_RATIO: f64 =
        STATION_MAX_ACCUMULATED_CENTER_MOTION_RATIO;
    pub(crate) const MAX_STATION_MEMBERS: usize = FOCUS_BRACKET_MAX_COMPONENT_SOURCES;

    /// [`focus_station_link_evidence`] — the measured 需求 1.1 evidence of one
    /// pair, `None` when the pair carries no overlap evidence at all.
    pub(crate) fn station_link_evidence(
        images: &[ImageInfo],
        first: usize,
        second: usize,
        match_info: &MatchInfo,
    ) -> Option<EvidenceView> {
        focus_station_link_evidence(images, first, second, match_info).map(|evidence| {
            EvidenceView {
                inliers: evidence.inliers,
                overlap_ncc: evidence.overlap_ncc,
                overlap_ncc_measured: evidence.overlap_ncc_measured,
                spatial_support: evidence.spatial_support,
                scale_ratio: evidence.scale_ratio,
                score: evidence.score,
                accepts: evidence.accepts(),
            }
        })
    }

    /// [`focus_match_is_capture_station_link`] — the membership decision the
    /// grouper's union-find is driven by.
    pub(crate) fn is_capture_station_link(
        images: &[ImageInfo],
        first: usize,
        second: usize,
        match_info: &MatchInfo,
    ) -> bool {
        focus_match_is_capture_station_link(images, first, second, match_info)
    }

    /// [`focus_station_grouping`] — membership, isolated sources and splits.
    pub(crate) fn station_grouping(
        images: &[ImageInfo],
        matches: &HashMap<(usize, usize), MatchInfo>,
    ) -> GroupingView {
        let outcome = focus_station_grouping(images, matches);
        GroupingView {
            components: outcome.components,
            isolated: outcome.isolated,
            splits: outcome.splits.iter().map(SplitView::from).collect(),
        }
    }

    /// [`split_focus_local_component_by_motion`] — the accumulated displacement
    /// split of one verified-link component (需求 1.4 / 1.5 / 1.10).
    pub(crate) fn split_component_by_motion(
        component: Vec<usize>,
        images: &[ImageInfo],
        matches: &HashMap<(usize, usize), MatchInfo>,
    ) -> (Vec<Vec<usize>>, Vec<SplitView>) {
        let mut splits = Vec::new();
        let groups =
            split_focus_local_component_by_motion(component, 0, images, matches, &mut splits);
        (groups, splits.iter().map(SplitView::from).collect())
    }

    /// [`focus_frame_median_sharpness`] — the 需求 2.1 anchor measurement.
    pub(crate) fn frame_median_sharpness(image: &ImageInfo) -> f64 {
        focus_frame_median_sharpness(image)
    }

    /// [`excluded_source_records`] — the 需求 1.6 / 1.9 report records.
    pub(crate) fn excluded_sources(
        images: &[ImageInfo],
        indices: &[usize],
        reason: &'static str,
    ) -> Vec<stack_report::ExcludedSourceRecord> {
        excluded_source_records(images, indices, reason)
    }

    /// [`undecodable_source_paths`] — the 需求 1.9 exclusion itself.
    pub(crate) fn undecodable_sources(selection: &[String], decoded: &[ImageInfo]) -> Vec<String> {
        undecodable_source_paths(selection, decoded)
    }
}

/// Test-only access to the production Station_Relation evidence and acceptance
/// decisions used by Properties 28 and 29. These wrappers deliberately contain
/// no thresholds or alternate implementations: they expose the real decisions
/// while keeping the production types private.
#[cfg(test)]
pub(crate) mod station_relation_test_access {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct MeasurementsView {
        pub(crate) inliers: usize,
        pub(crate) median_error_px: f64,
        pub(crate) scale_ratio: f64,
        pub(crate) spatial_support: f64,
        pub(crate) low_frequency_mean_relative_difference: f64,
        pub(crate) edge_strength_ratio: f64,
        pub(crate) median_edge_orientation_difference_degrees: f64,
        pub(crate) preserves_convex_orientation: bool,
    }

    pub(crate) fn covered_features(
        pixels: &Rgb32FImage,
        coverage: &stitching::CoverageMask,
        brief_pairs: &[(Point2<i32>, Point2<i32>)],
    ) -> (GrayImage, Vec<Feature>) {
        virtual_tile_covered_features(pixels, coverage, brief_pairs)
    }

    pub(crate) fn station_matches(tiles: &[ImageInfo]) -> HashMap<(usize, usize), MatchInfo> {
        all_image_pairs(tiles.len())
            .into_iter()
            .filter_map(|(left, right)| {
                virtual_tile_unseeded_station_match(left, right, tiles)
                    .map(|relation| ((left, right), relation))
            })
            .collect()
    }

    pub(crate) fn guided_station_matches(
        tiles: &[ImageInfo],
        coverages: &[stitching::CoverageMask],
        initial_homographies: &HashMap<usize, Matrix3<f64>>,
        station_topology: &topology::StationTopology,
    ) -> HashMap<(usize, usize), MatchInfo> {
        virtual_tile_station_matches(tiles, coverages, initial_homographies, station_topology)
    }

    pub(crate) fn single_source_discard(
        independent_support: usize,
        left_station: usize,
        right_station: usize,
    ) -> Option<serde_json::Value> {
        single_layer_station_evidence_discard(independent_support, left_station, right_station)
    }

    pub(crate) fn rejection_reasons(measurements: MeasurementsView) -> Vec<&'static str> {
        station_relation_rejection_reasons(&StationRelationMeasurements {
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
        })
    }

    fn production_measurements(measurements: MeasurementsView) -> StationRelationMeasurements {
        StationRelationMeasurements {
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
        }
    }

    pub(crate) struct RelationDecisionBatchView {
        pub(crate) report: stack_report::StationRelationsReport,
        pub(crate) pose_writes: Vec<(usize, usize)>,
    }

    /// Drive the exact production reject/report gate for a batch of candidate
    /// adjacencies. `pose_writes` models the immediately following pose-graph
    /// insertion and therefore contains accepted candidates only.
    pub(crate) fn relation_decision_batch(
        candidates: &[(usize, usize, MeasurementsView)],
    ) -> RelationDecisionBatchView {
        let mut report = stack_report::StationRelationsReport::default();
        let mut pose_writes = Vec::new();
        for &(left, right, measurements) in candidates {
            if station_relation_is_accepted_with_report(
                left,
                right,
                0.0,
                &production_measurements(measurements),
                &mut report,
            ) {
                pose_writes.push((left, right));
            }
        }
        RelationDecisionBatchView {
            report,
            pose_writes,
        }
    }

    fn synthetic_image(id: usize, path: String, dimensions: (u32, u32)) -> ImageInfo {
        ImageInfo {
            id,
            filename: path,
            width: dimensions.0,
            height: dimensions.1,
            alignment_image: GrayImage::new(dimensions.0.max(1), dimensions.1.max(1)),
            full_image: None,
            scale_factor: 1.0,
            focal_length_35mm: None,
            overview_reference: false,
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
            horizontal_edge_rows: Vec::new(),
            vertical_edge_columns: Vec::new(),
        }
    }

    fn singleton_station_inputs(
        dimensions: (u32, u32),
        station_count: usize,
    ) -> (Vec<ImageInfo>, Vec<FocusCaptureGroup>) {
        let images = (0..station_count)
            .map(|station| {
                synthetic_image(
                    station,
                    format!("/synthetic/property/station-{station}.raw"),
                    dimensions,
                )
            })
            .collect::<Vec<_>>();
        let groups = (0..station_count)
            .map(|station| FocusCaptureGroup {
                members: vec![station],
                anchor: station,
                local_to_anchor: HashMap::from([(station, Matrix3::identity())]),
            })
            .collect::<Vec<_>>();
        (images, groups)
    }

    pub(crate) struct PoseCandidateView {
        pub(crate) selected: Matrix3<f64>,
        pub(crate) minimum_local_scale: Option<f64>,
        pub(crate) maximum_local_scale: Option<f64>,
        pub(crate) rejection_reason: Option<&'static str>,
    }

    /// Exercise the production Local_Scale publication gate for one station.
    pub(crate) fn select_pose_candidate(
        candidate: Matrix3<f64>,
        spanning_tree_seed: Matrix3<f64>,
        dimensions: (u32, u32),
    ) -> PoseCandidateView {
        let (images, groups) = singleton_station_inputs(dimensions, 1);
        let accepted = station_pose_local_scale_range(&candidate, dimensions).is_some_and(
            |(minimum, maximum)| maximum / minimum <= STATION_POSE_MAX_LOCAL_SCALE_RATIO,
        );
        let selected = select_station_pose_candidate(
            &images,
            &groups,
            &[candidate],
            &[spanning_tree_seed],
            "property_31",
        )[0];
        let range = station_pose_local_scale_range(&candidate, dimensions);
        PoseCandidateView {
            selected,
            minimum_local_scale: range.map(|range| range.0),
            maximum_local_scale: range.map(|range| range.1),
            rejection_reason: (!accepted).then_some(degradation::STATION_POSE_LOCAL_SCALE_EXCEEDED),
        }
    }

    pub(crate) struct StationGraphOutcomeView {
        pub(crate) report: stack_report::StationRelationsReport,
        pub(crate) rejection_reason: Option<&'static str>,
        pub(crate) rejection_detail: Option<serde_json::Value>,
        pub(crate) user_error: Option<String>,
    }

    /// Run the production component canonicalisation/report boundary on an
    /// arbitrary station graph supplied by a property test.
    pub(crate) fn station_graph_outcome(
        station_member_paths: &[Vec<String>],
        relation_pairs: &[(usize, usize)],
    ) -> StationGraphOutcomeView {
        let mut images = Vec::new();
        let mut groups = Vec::new();
        for members in station_member_paths {
            let first = images.len();
            let mut member_indices = Vec::new();
            for path in members {
                let image_index = images.len();
                images.push(synthetic_image(image_index, path.clone(), (64, 64)));
                member_indices.push(image_index);
            }
            groups.push(FocusCaptureGroup {
                members: member_indices,
                anchor: first,
                local_to_anchor: HashMap::from([(first, Matrix3::identity())]),
            });
        }
        let mut report = stack_report::StationRelationsReport::default();
        let rejection_detail =
            station_graph_connectivity_rejection(&images, &groups, relation_pairs, &mut report);
        let rejection_reason = rejection_detail
            .as_ref()
            .map(|_| degradation::GEOMETRY_DISCONNECTED);
        let user_error = rejection_detail
            .as_ref()
            .map(|_| geometry_disconnected_user_error(report.connectivity.components));
        StationGraphOutcomeView {
            report,
            rejection_reason,
            rejection_detail,
            user_error,
        }
    }

    /// Production Kruskal selection, returned as canonical station pairs so
    /// relation input order is not observable to the property test.
    pub(crate) fn maximum_inlier_tree(
        station_count: usize,
        rows: &[u32],
        columns: &[u32],
        relations: &[(usize, usize, usize, f64)],
    ) -> Vec<(usize, usize)> {
        let station_topology = topology::StationTopology {
            row: rows.to_vec(),
            column: columns.to_vec(),
            ..topology::StationTopology::default()
        };
        let production_relations = relations
            .iter()
            .map(|&(left, right, inliers, _score)| (left, right, inliers))
            .collect::<Vec<_>>();
        maximum_inlier_spanning_tree_indices(
            station_count,
            &station_topology,
            &production_relations,
        )
        .into_iter()
        .map(|index| {
            let (left, right, _, _) = relations[index];
            (left.min(right), left.max(right))
        })
        .collect()
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct ClosureRelationView {
        pub(crate) score: f64,
        pub(crate) left: usize,
        pub(crate) right: usize,
        pub(crate) left_to_right: Matrix3<f64>,
        pub(crate) independent_support: usize,
        pub(crate) median_error_px: f64,
    }

    pub(crate) struct ClosureRunView {
        pub(crate) poses: Vec<Matrix3<f64>>,
        pub(crate) report: stack_report::ClosureReport,
    }

    fn production_group_edges(
        relations: &[ClosureRelationView],
    ) -> Vec<(f64, usize, usize, Matrix3<f64>, usize, f64)> {
        relations
            .iter()
            .map(|relation| {
                (
                    relation.score,
                    relation.left,
                    relation.right,
                    relation.left_to_right,
                    relation.independent_support,
                    relation.median_error_px,
                )
            })
            .collect()
    }

    /// Exact production M-estimator access for Property 35.
    pub(crate) fn closure_weight(residual_px: f64) -> f64 {
        closure_residual_weight(residual_px)
    }

    /// Exact production stopping predicate access for Property 37.
    pub(crate) fn closure_converged(previous: f64, current: f64) -> bool {
        closure_relative_residual_decrease_converged(previous, current)
    }

    pub(crate) struct ClosureSolveView {
        pub(crate) corrections: Vec<[f64; 2]>,
        pub(crate) iterations: usize,
        pub(crate) converged: bool,
    }

    /// Exercise the production robust translation solver without image setup.
    pub(crate) fn closure_solve(
        station_count: usize,
        reference_station: usize,
        constraints: &[(usize, usize, [f64; 2], f64)],
    ) -> Option<ClosureSolveView> {
        solve_robust_closure_translations(station_count, reference_station, constraints).map(
            |solve| ClosureSolveView {
                corrections: solve.corrections,
                iterations: solve.iterations,
                converged: solve.converged,
            },
        )
    }

    pub(crate) struct ClosureClampView {
        pub(crate) corrections: Vec<[f64; 2]>,
        pub(crate) maximum_corner_displacement_px: f64,
        pub(crate) clamped_stations: usize,
    }

    /// Exercise the production four-corner correction clamp for Property 36.
    pub(crate) fn clamp_closure_corrections(
        dimensions: (u32, u32),
        base_poses: &[Matrix3<f64>],
        relations: &[ClosureRelationView],
        corrections: &[[f64; 2]],
    ) -> Option<ClosureClampView> {
        let (images, groups) = singleton_station_inputs(dimensions, base_poses.len());
        let group_edges = production_group_edges(relations);
        let mut corrections = corrections.to_vec();
        let (maximum_corner_displacement_px, clamped_stations) =
            clamp_station_translation_corrections(
                &images,
                &groups,
                &group_edges,
                base_poses,
                &mut corrections,
            )?;
        Some(ClosureClampView {
            corrections,
            maximum_corner_displacement_px,
            clamped_stations,
        })
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum ClosureAcceptanceView {
        Accepted,
        NoConstraints,
        UnreliableResidual,
        UnreliableIterations,
        UnreliablePairP95(usize, usize),
    }

    /// Exact production acceptance gate access for Properties 38 and 39.
    pub(crate) fn closure_acceptance_view(
        station_count: usize,
        accepted_relation_count: usize,
        converged: bool,
        iterations: usize,
        residual_median_px: f64,
        direct_pairs: &[(usize, usize, f64)],
    ) -> ClosureAcceptanceView {
        let direct_pairs = direct_pairs
            .iter()
            .map(|&(left, right, p95_px)| ClosureDirectPairP95 {
                left,
                right,
                p95_px,
            })
            .collect::<Vec<_>>();
        match closure_acceptance(
            station_count,
            accepted_relation_count,
            converged,
            iterations,
            residual_median_px,
            &direct_pairs,
        ) {
            ClosureAcceptance::Accepted => ClosureAcceptanceView::Accepted,
            ClosureAcceptance::NoConstraints => ClosureAcceptanceView::NoConstraints,
            ClosureAcceptance::UnreliableResidual => ClosureAcceptanceView::UnreliableResidual,
            ClosureAcceptance::UnreliableIterations => ClosureAcceptanceView::UnreliableIterations,
            ClosureAcceptance::UnreliablePairP95(pair) => {
                ClosureAcceptanceView::UnreliablePairP95(pair.left, pair.right)
            }
        }
    }

    /// Run the actual closure/report boundary with one singleton source per
    /// station; no image payload is decoded or mocked.
    pub(crate) fn closure_run(
        dimensions: (u32, u32),
        rows: &[u32],
        columns: &[u32],
        base_poses: &[Matrix3<f64>],
        relations: &[ClosureRelationView],
    ) -> ClosureRunView {
        let (images, groups) = singleton_station_inputs(dimensions, base_poses.len());
        let group_edges = production_group_edges(relations);
        let station_topology = topology::StationTopology {
            row: rows.to_vec(),
            column: columns.to_vec(),
            ..topology::StationTopology::default()
        };
        let mut report = stack_report::ClosureReport::default();
        let poses = refine_focus_group_projective_tree_poses_with_report(
            &images,
            &groups,
            &group_edges,
            base_poses,
            0,
            f64::from(dimensions.0.max(dimensions.1)),
            &station_topology,
            2,
            &mut report,
        )
        .expect("valid property closure fixture returns a pose vector");
        ClosureRunView { poses, report }
    }
}

#[cfg(test)]
#[path = "panorama_stitching_alignment_tests.rs"]
mod alignment_tests;

#[cfg(test)]
#[path = "panorama_stitching_acceptance_tests.rs"]
mod acceptance_tests;

#[cfg(test)]
#[path = "panorama_reference_acceptance.rs"]
mod reference_acceptance;

#[cfg(test)]
#[path = "panorama_virtual_tile_tests.rs"]
mod virtual_tile_pyramid_tests;
