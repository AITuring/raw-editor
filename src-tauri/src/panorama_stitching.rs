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

#[path = "panorama_feature_matching.rs"]
mod feature_matching;
use feature_matching::*;

#[path = "panorama_orchestration.rs"]
mod orchestration;
pub(crate) use orchestration::*;

#[path = "panorama_virtual_tile_solver.rs"]
mod virtual_tile_solver;
use virtual_tile_solver::*;

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

struct PreparedStackSource {
    image: DynamicImage,
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
            image = DynamicImage::ImageRgb32F(corrected);
        }
    }
    Ok(PreparedStackSource {
        image,
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

fn focus_capture_number_gap(source: &ImageInfo, target: &ImageInfo) -> Option<u64> {
    trailing_capture_number(&source.filename)
        .zip(trailing_capture_number(&target.filename))
        .map(|(left, right)| left.abs_diff(right))
}

fn focus_compatible_focal_length(source: &ImageInfo, target: &ImageInfo) -> bool {
    match (source.focal_length_35mm, target.focal_length_35mm) {
        (Some(left), Some(right))
            if left.is_finite() && right.is_finite() && left > 0.0 && right > 0.0 =>
        {
            (left / right).max(right / left) <= 1.03
        }
        _ => true,
    }
}

fn focus_adjacent_same_focal_pair(source: &ImageInfo, target: &ImageInfo) -> bool {
    focus_capture_number_gap(source, target).is_some_and(|gap| gap == 1)
        && focus_compatible_focal_length(source, target)
}

fn recover_adjacent_focus_bracket_match(
    source: &ImageInfo,
    target: &ImageInfo,
    log_match: bool,
) -> Option<MatchInfo> {
    let adjacent_capture = trailing_capture_number(&source.filename)
        .zip(trailing_capture_number(&target.filename))
        .is_some_and(|(left, right)| left.abs_diff(right) == 1);
    if !adjacent_capture {
        return None;
    }
    let compatible_focal_length = source
        .focal_length_35mm
        .zip(target.focal_length_35mm)
        .map(|(left, right)| {
            left.is_finite()
                && right.is_finite()
                && left > 0.0
                && right > 0.0
                && (left / right).max(right / left) <= 1.03
        })
        // Missing EXIF must not disable focus-bracket recovery. The strict
        // same-pose checks below remain the authority.
        .unwrap_or(true);
    if !compatible_focal_length {
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
    let (homography, points, score) = estimate_mixed_focal_coarse_registration(
        source,
        target,
        source_focal,
        target_focal,
        true,
        log_match,
    )?;
    // This fallback is only allowed to attach another focal plane at the same
    // camera pose. A spatial scan step or repeated motif must continue through
    // the ordinary feature/cycle-verified panorama graph.
    let scale = homography[(0, 0)].abs();
    let source_center = Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5);
    let target_center = Point2::new(target.width as f64 * 0.5, target.height as f64 * 0.5);
    let mapped_center = transformed_point(&homography, source_center)?;
    let normalization = source
        .width
        .max(source.height)
        .max(target.width.max(target.height))
        .max(1) as f64;
    let center_motion = (mapped_center - target_center).norm() / normalization;
    let overlap =
        panorama_transform_overlap_support(&homography, source.dimensions(), target.dimensions());
    // The overlap estimator deliberately ignores an outer validation margin;
    // identical full-frame images therefore report about 0.64, not 1.0.
    if !(0.88..=1.12).contains(&scale) || center_motion > 0.12 || overlap < 0.60 {
        if log_match {
            println!(
                "  - Rejecting adjacent focus-layer recovery: scale {scale:.3}, motion {:.3}%, overlap {:.1}%",
                center_motion * 100.0,
                overlap * 100.0,
            );
        }
        return None;
    }
    if log_match {
        println!(
            "  - Recovered same-pose adjacent focus layer: '{}' <-> '{}' (NCC {:.3}, motion {:.3}%, overlap {:.1}%)",
            Path::new(&source.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            Path::new(&target.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            score,
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
    })
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
    FocusStackSourceDigests {
        inputs: virtual_tile::CacheKeyInputs::new(
            crate::image_stack::STACK_PIPELINE_VERSION,
            &sources,
        ),
        sha256_before: digests.into_iter().map(source_digest_hex).collect(),
    }
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
                if focus_stack
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

    let shared_raw_display_gain = if focus_stack
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
    let (ordered_indices, global_homographies) = if blend_mode == BlendMode::FocusStack {
        // The default two-stage path uses source images only to group stations
        // and establish each station's local tile geometry.  Its authoritative
        // requirement-6/7 report is produced later from the fused tiles, so an
        // early source-only support=1 graph must neither reject nor publish a
        // report that could overwrite the Virtual_Tile result.
        let mut closure = stack_report::ClosureReport::default();
        let mut station_relations = stack_report::StationRelationsReport::default();
        let stitching_order = build_focus_stack_stitching_order_with_report(
            &image_data,
            &pairwise_matches,
            &mut closure,
            &mut station_relations,
            !defer_authoritative_station_solve,
        );
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
    let focus_layer_warp = (blend_mode == BlendMode::FocusStack).then(|| {
        build_focus_layer_warp(
            &image_data,
            &pairwise_matches,
            &global_homographies,
            projection,
        )
    });
    let focus_station_grouping_outcome = if blend_mode == BlendMode::FocusStack {
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
    if let Some(group_ids) = focus_capture_group_ids.as_ref() {
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
    let panorama = match blend_mode {
        BlendMode::Panorama => stitching::progressive_seam_stitcher(
            &render_images_info,
            render_homographies,
            projection,
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
            let compositor_choice =
                compositor_choice.expect("focus-stack compositor choice resolved above");
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
                    sequence_gap_aware,
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
                    // covered-feature proposal. The bounded pixel probe fits
                    // an independent relation before any solver admission.
                    // Anchor-grid tiles can use that measured relation directly;
                    // globally warped comparison tiles instead repair their
                    // prior and render again. Feature proposals never enter the
                    // solver themselves.
                    let mut prior_repair_records = Vec::new();
                    if tile_infos.len() >= 2 {
                        let authoritative_pairs = virtual_tile_authoritative_pairs(
                            &tile_relations,
                            &tile_infos,
                            pyramid.solver_coverages(),
                        );
                        let mut clusters = Dsu::new(tile_infos.len());
                        for &(left, right) in &authoritative_pairs {
                            clusters.union(left, right);
                        }
                        let mut proposals = Vec::new();
                        for candidate in &station_topology.candidates {
                            let (left, right) = (candidate.left, candidate.right);
                            if left >= tile_infos.len()
                                || right >= tile_infos.len()
                                || clusters.find(left) == clusters.find(right)
                            {
                                continue;
                            }
                            let Some((seed, feature_inliers)) = virtual_tile_prior_free_relation(
                                pyramid.feature_factors[left],
                                &tile_infos[left].features,
                                pyramid.feature_factors[right],
                                &tile_infos[right].features,
                                tile_infos[left].dimensions(),
                            ) else {
                                continue;
                            };
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
                            if let Some(selection) = selection.filter(|selection| selection.passes)
                            {
                                if anchor_layout.is_some() {
                                    clusters.union(left, right);
                                    tile_relations
                                        .entry((left, right))
                                        .or_default()
                                        .extend(verified);
                                } else {
                                    proposals.push(VirtualTilePriorProposal {
                                        left,
                                        right,
                                        left_to_right: selection.relation.homography,
                                        inliers: selection.relation.inliers,
                                    });
                                }
                            } else if anchor_layout.is_none()
                                && virtual_tile_prior_free_seed_fallback_allowed(&diagnostic)
                            {
                                // Keep this relation out of the authoritative graph. It only
                                // moves the unfixed cluster close enough for a fresh Virtual_Tile
                                // render, after which the regular measurement pass verifies it.
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
                        let authoritative_pairs = virtual_tile_authoritative_pairs(
                            &tile_relations,
                            &tile_infos,
                            pyramid.solver_coverages(),
                        );
                        let station_tile_to_world = tile_infos
                            .iter()
                            .map(|tile| tile_homographies[&tile.id])
                            .collect::<Vec<_>>();
                        let station_world_centres = geometries
                            .iter()
                            .zip(&station_tile_to_world)
                            .filter_map(|(geometry, tile_to_world)| {
                                transformed_point(
                                    tile_to_world,
                                    Point2::new(
                                        f64::from(geometry.width) * 0.5,
                                        f64::from(geometry.height) * 0.5,
                                    ),
                                )
                            })
                            .collect::<Vec<_>>();
                        let repairs = plan_virtual_tile_prior_repairs_with_centres(
                            tile_infos.len(),
                            &authoritative_pairs,
                            &proposals,
                            &station_tile_to_world,
                            &station_world_centres,
                            station_topology,
                        );
                        let mut repaired = BTreeSet::new();
                        let mut repair_clusters =
                            BTreeMap::<usize, Vec<&VirtualTilePriorRepair>>::new();
                        for repair in &repairs {
                            repair_clusters
                                .entry(repair.cluster_root)
                                .or_default()
                                .push(repair);
                        }
                        for cluster_repairs in repair_clusters.values() {
                            // Preflight every member on one copy. No rendering
                            // prior is committed unless the entire cluster has
                            // valid poses, geometry, centre shift and scale.
                            let mut candidate_homographies = station_render_homographies.clone();
                            let mut candidate_focus_warp = station_focus_warp.clone();
                            let mut candidates = Vec::with_capacity(cluster_repairs.len());
                            let mut rejected = None;
                            for &repair in cluster_repairs {
                                let station = repair.station;
                                if !apply_virtual_tile_prior_repair(
                                    group_slices[station],
                                    &repair.world_correction,
                                    &mut candidate_homographies,
                                    candidate_focus_warp.as_mut(),
                                ) {
                                    rejected = Some((
                                        station,
                                        "a corrected source pose is not a valid planar pose",
                                    ));
                                    break;
                                }
                                let Some(candidate_geometry) =
                                    stitching::focus_stack_virtual_tile_geometry(
                                        group_slices[station],
                                        &candidate_homographies,
                                        projection,
                                        candidate_focus_warp.as_ref(),
                                    )
                                else {
                                    rejected = Some((station, "corrected geometry is invalid"));
                                    break;
                                };
                                let Some((center_shift_px, correction_scale_ratio)) =
                                    virtual_tile_prior_repair_measurements(
                                        &repair.world_correction,
                                        station_world_centres[station],
                                    )
                                else {
                                    rejected =
                                        Some((station, "correction measurements are invalid"));
                                    break;
                                };
                                if !virtual_tile_prior_repair_is_bounded(
                                    center_shift_px,
                                    correction_scale_ratio,
                                ) {
                                    rejected = Some((station, "correction is outside bounds"));
                                    break;
                                }
                                candidates.push((
                                    repair,
                                    candidate_geometry,
                                    center_shift_px,
                                    correction_scale_ratio,
                                ));
                            }
                            if let Some((station, reason)) = rejected {
                                println!(
                                    "  - Virtual_Tile prior repair cluster containing station {station} rejected atomically: {reason}"
                                );
                                continue;
                            }
                            station_render_homographies = candidate_homographies;
                            station_focus_warp = candidate_focus_warp;
                            for (repair, geometry, center_shift_px, correction_scale_ratio) in
                                candidates
                            {
                                let station = repair.station;
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
                            let solved = solve_virtual_tile_station_hypotheses_with_report(
                                &tile_infos,
                                pyramid.solver_coverages(),
                                &tile_relations,
                                &tile_homographies,
                                station_topology,
                                &mut closure,
                                &mut station_relations,
                            );
                            let component_count = station_relations.connectivity.components;
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
                            });
                            station_relations.candidates = tile_relation_candidates.clone();
                            station_relations.summarize_candidate_measurements();
                            station_relations.prior_repairs = prior_repair_records.clone();
                            if let Some(recorder) = stack_report.as_ref() {
                                recorder.update(|report| {
                                    report.closure = closure.clone();
                                    report.station_relations = station_relations.clone();
                                });
                            }
                            if component_count > 1 || solved.len() != tile_infos.len() {
                                return Err(geometry_disconnected_user_error(
                                    component_count.max(2),
                                ));
                            }
                            tile_homographies = solved;
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
                    let mut load_tile = |tile: &ImageInfo| {
                        ensure_focus_memory_available(rss_sampler.as_ref())?;
                        let group_index =
                            tile_group_indices.get(&tile.id).copied().ok_or_else(|| {
                                format!("Missing virtual tile provenance {}", tile.id)
                            })?;
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
                        Ok(rendered.into_image())
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
                                &mut load_tile,
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
                                app_handle.clone(),
                                progress_event,
                                &mut load_tile,
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
                                    for source in group_slices[station] {
                                        let owner = next_owner;
                                        raw_owner_ids[station].push(owner);
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
                                                    sampled = Some((px, py, raw_owner));
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
                                    let Some((sx, sy, raw_owner)) = sampled else {
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
                                sequence_gap_aware,
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
                        sequence_gap_aware,
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
        ownership,
        confidence,
        textured,
        owner_shortfall,
        owner_disagreement,
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
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
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
        if let Some(recorder) = stack_report.as_ref() {
            recorder.update(|stack| {
                stack.quality_gate = report;
                stack.quality_gate.confidence_scores = confidence_scores.clone();
                stack.quality_gate.owner_reverse_lookup_failures =
                    owner_reverse_lookup_failures.clone();
            });
        }
    } else if let Some(recorder) = stack_report.as_ref() {
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

fn focus_capture_groups(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<(Vec<FocusCaptureGroup>, Vec<usize>)> {
    focus_capture_groups_with_grouping(images, matches, locked_homographies)
        .map(|(groups, group_for_image, _)| (groups, group_for_image))
}

/// [`focus_capture_groups`] together with everything the Stack_Report needs to
/// carry about the grouping decision (需求 1.4 / 1.6 / 1.7 / 1.10).
fn focus_capture_groups_with_grouping(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<(Vec<FocusCaptureGroup>, Vec<usize>, StationGroupingOutcome)> {
    // Direct local links already provide the strongest station membership
    // evidence. Keep those components intact; solved world centres may only
    // select among attachments backed by same-pose evidence. The previous implementation
    // sorted each connected component by filename before splitting it; a
    // renamed or shuffled import therefore changed the station count even
    // though every geometric match was identical.
    let grouping = focus_station_grouping(images, matches);
    let mut components = grouping.components.clone();
    let mut assigned = components
        .iter()
        .flat_map(|component| component.iter().copied())
        .collect::<HashSet<_>>();
    // 需求 1.6: a source with neither same-pose nor verified scan overlap is
    // isolated. It joins no Capture_Station at all — not even
    // by its solved world centre, which is only as trustworthy as the pose the
    // missing evidence could not confirm — while every other source keeps the
    // grouping it would have had anyway.  A single member left over by the
    // accumulated-motion split is *not* isolated: it had verified evidence, so
    // it still becomes a station of its own (需求 1.5).
    let isolated = grouping.isolated.iter().copied().collect::<HashSet<_>>();
    for image_index in 0..images.len() {
        if assigned.contains(&image_index) || isolated.contains(&image_index) {
            continue;
        }
        components.push(vec![image_index]);
        assigned.insert(image_index);
    }

    let world_center = |image_index: usize| {
        locked_homographies
            .get(&images[image_index].id)
            .and_then(|pose| {
                transformed_point(
                    pose,
                    Point2::new(
                        images[image_index].width as f64 * 0.5,
                        images[image_index].height as f64 * 0.5,
                    ),
                )
            })
    };
    let component_center = |component: &[usize]| {
        let centers = component
            .iter()
            .filter_map(|&index| world_center(index))
            .collect::<Vec<_>>();
        if centers.is_empty() {
            return None;
        }
        let x = centers.iter().map(|point| point.x).sum::<f64>() / centers.len() as f64;
        let y = centers.iter().map(|point| point.y).sum::<f64>() / centers.len() as f64;
        Some(Point2::new(x, y))
    };

    // Absorb a singleton into the nearest established station only with
    // same-pose evidence and a centre within the bracket-sized radius. A coarse-only
    // edge is an explicit boundary: it can suggest scan order, but cannot
    // make two focus layers compete for ownership.
    for singleton_index in (0..components.len())
        .filter(|&index| components[index].len() == 1)
        .collect::<Vec<_>>()
    {
        let image_index = components[singleton_index][0];
        let Some(center) = world_center(image_index) else {
            continue;
        };
        let mut best = None::<(f64, usize)>;
        for (candidate_index, candidate) in components.iter().enumerate() {
            if candidate_index == singleton_index
                || candidate.is_empty()
                || candidate.len() >= FOCUS_BRACKET_MAX_COMPONENT_SOURCES
                || !focus_compatible_focal_length(&images[image_index], &images[candidate[0]])
            {
                continue;
            }
            // A world-centre prior is not camera-pose evidence. In particular,
            // a connected single-frame scan station must not be folded back
            // into a neighbouring focus bracket just because their priors drift.
            let same_pose_evidence = candidate.iter().any(|&member| {
                let key = (image_index.min(member), image_index.max(member));
                matches.get(&key).is_some_and(|edge| {
                    focus_match_is_capture_station_link(images, key.0, key.1, edge)
                })
            });
            if !same_pose_evidence {
                continue;
            }
            let blocked = candidate.iter().any(|&member| {
                matches
                    .get(&(image_index.min(member), image_index.max(member)))
                    .is_some_and(|edge| edge.coarse_bridge)
            });
            if blocked {
                continue;
            }
            let Some(candidate_center) = component_center(candidate) else {
                continue;
            };
            let scale = images[image_index]
                .width
                .max(images[image_index].height)
                .max(images[candidate[0]].width.max(images[candidate[0]].height))
                .max(1) as f64;
            let distance = (center - candidate_center).norm() / scale;
            if distance <= FOCUS_BRACKET_INTERNAL_MOTION_RATIO
                && best.is_none_or(|(best_distance, _)| distance < best_distance)
            {
                best = Some((distance, candidate_index));
            }
        }
        if let Some((_, candidate_index)) = best {
            let singleton = std::mem::take(&mut components[singleton_index]);
            components[candidate_index].extend(singleton);
        }
    }

    // Establish a deterministic geometric order for diagnostics and the
    // station pose graph.  Content-derived keys only break exact-centre ties;
    // filenames and import indices do not participate.
    for component in &mut components {
        component.sort_by(|left, right| {
            world_center(*left)
                .zip(world_center(*right))
                .map(|(left_center, right_center)| {
                    left_center
                        .y
                        .total_cmp(&right_center.y)
                        .then_with(|| left_center.x.total_cmp(&right_center.x))
                })
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    focus_image_direction_key(&images[*left])
                        .cmp(&focus_image_direction_key(&images[*right]))
                })
        });
    }
    components.retain(|component| !component.is_empty());
    components.sort_by(|left, right| {
        component_center(left)
            .zip(component_center(right))
            .map(|(left_center, right_center)| {
                left_center
                    .y
                    .total_cmp(&right_center.y)
                    .then_with(|| left_center.x.total_cmp(&right_center.x))
            })
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut group_for_image = vec![usize::MAX; images.len()];
    let mut groups = Vec::with_capacity(components.len());
    for (group_index, members) in components.into_iter().enumerate() {
        // 需求 2.1: the anchor frame is the member whose valid pixels have the
        // highest median Sharpness_Score, with near-equal scores resolved by
        // ascending absolute path.  The previous `members[0]` took the first
        // frame in world-centre order, which is a geometric accident of the
        // bracket: every other frame is then registered into the coordinate
        // system of a possibly defocused reference, and the station's own
        // representative frame (its dimensions, focal length and edge lines
        // feed the station pose graph) carried the least detail of the bracket.
        let anchor_candidates = members
            .iter()
            .map(|&image_index| intra_station::AnchorCandidate {
                path: images[image_index].filename.clone(),
                median_sharpness: focus_frame_median_sharpness(&images[image_index]),
            })
            .collect::<Vec<_>>();
        let anchor = intra_station::select_anchor(&anchor_candidates)
            .map(|position| members[position])
            .unwrap_or(members[0]);
        let anchor_global = locked_homographies.get(&images[anchor].id)?;
        let anchor_inverse = anchor_global.try_inverse()?;
        let mut local_to_anchor = HashMap::new();
        for &image_index in &members {
            let image_global = locked_homographies.get(&images[image_index].id)?;
            local_to_anchor.insert(image_index, anchor_inverse * image_global);
            group_for_image[image_index] = group_index;
        }
        groups.push(FocusCaptureGroup {
            members,
            anchor,
            local_to_anchor,
        });
    }
    Some((groups, group_for_image, grouping))
}

fn canonicalize_station_geometry_anchors(
    images: &[ImageInfo],
    groups: &mut [FocusCaptureGroup],
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<()> {
    let pose_order = |left: usize, right: usize| {
        let left_pose = locked_homographies.get(&images[left].id);
        let right_pose = locked_homographies.get(&images[right].id);
        (0..9)
            .map(|index| {
                left_pose
                    .map(|pose| pose[index])
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&right_pose.map(|pose| pose[index]).unwrap_or(f64::INFINITY))
            })
            .find(|order| !order.is_eq())
            .unwrap_or(std::cmp::Ordering::Equal)
    };

    for group in groups {
        let geometry_anchor = group.members.iter().copied().min_by(|&left, &right| {
            focus_image_direction_key(&images[left])
                .cmp(&focus_image_direction_key(&images[right]))
                .then_with(|| pose_order(left, right))
        })?;
        let anchor_inverse = locked_homographies
            .get(&images[geometry_anchor].id)?
            .try_inverse()?;
        let mut local_to_anchor = HashMap::new();
        for &image_index in &group.members {
            let image_global = locked_homographies.get(&images[image_index].id)?;
            local_to_anchor.insert(image_index, anchor_inverse * image_global);
        }
        group.anchor = geometry_anchor;
        group.local_to_anchor = local_to_anchor;
    }
    Some(())
}

fn order_focus_render_sources_by_capture_group(
    images: &mut [&ImageInfo],
    capture_group_ids: &HashMap<usize, u8>,
) {
    images.sort_by(|left, right| {
        capture_group_ids
            .get(&left.id)
            .copied()
            .unwrap_or(u8::MAX)
            .cmp(&capture_group_ids.get(&right.id).copied().unwrap_or(u8::MAX))
            .then_with(|| focus_image_direction_key(left).cmp(&focus_image_direction_key(right)))
            .then_with(|| natural_path_cmp(&left.filename, &right.filename))
    });
}

fn solve_focus_group_translation_poses(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    _locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_group: usize,
    coordinate_scale: f64,
) -> Option<Vec<Matrix3<f64>>> {
    if groups.len() < 2 || reference_group >= groups.len() {
        return None;
    }
    let Some(reference_focal) = images[groups[reference_group].anchor].focal_length_35mm else {
        println!(
            "  - Focus capture-group translation geometry unavailable: missing reference focal length"
        );
        return None;
    };
    if groups.iter().any(|group| {
        images[group.anchor].focal_length_35mm.is_some_and(|focal| {
            let ratio = (focal / reference_focal).max(reference_focal / focal);
            !ratio.is_finite() || ratio > 1.03
        })
    }) {
        println!(
            "  - Focus capture-group translation geometry unavailable: mixed focal-length groups"
        );
        return None;
    }

    let mut consensus_connectivity = Dsu::new(groups.len());
    // Only multi-layer Consensus_Feature evidence may connect stations.
    // A single Source_RAW observation is supplemental evidence, never a
    // station relation on its own (需求 6.2).
    for &(_, left, right, _, support, _) in group_edges {
        if support >= 2 {
            consensus_connectivity.union(left, right);
        }
    }
    let mut consensus_sizes = HashMap::<usize, usize>::new();
    for group in 0..groups.len() {
        *consensus_sizes
            .entry(consensus_connectivity.find(group))
            .or_default() += 1;
    }
    let consensus_covered = consensus_sizes.values().copied().max().unwrap_or(0);
    // The checks below still require every group to be connected and every
    // translation constraint to pass robust closure validation.  Requiring
    // an 85% multi-layer core here made the solver fall back to a projective
    // spanning tree precisely on long, low-texture scans where scale drift
    // is most damaging.
    if consensus_covered < groups.len() {
        println!(
            "  - Focus capture-group translation geometry unavailable: multi-layer graph covers only {consensus_covered}/{} groups",
            groups.len()
        );
        return None;
    }

    let constraints = group_edges
        .iter()
        .filter_map(
            |&(score, left, right, left_to_right, support, median_error)| {
                if support < 2 || left >= groups.len() || right >= groups.len() {
                    return None;
                }
                let left_image = &images[groups[left].anchor];
                let left_center = Point2::new(
                    left_image.width as f64 * 0.5,
                    left_image.height as f64 * 0.5,
                );
                let right_point = transformed_point(&left_to_right, left_center)?;
                let observed = left_center - right_point;
                let weight = support as f64 * score.max(1e-8).sqrt()
                    / (1.0 + median_error / STATION_RELATION_MAX_MEDIAN_ERROR_PX);
                (observed.x.is_finite()
                    && observed.y.is_finite()
                    && weight.is_finite()
                    && weight > 0.0)
                    .then_some((left, right, [observed.x, observed.y], weight))
            },
        )
        .collect::<Vec<_>>();
    if constraints.len() + 1 < groups.len() {
        println!(
            "  - Focus capture-group translation geometry unavailable: {} relations for {} groups",
            constraints.len(),
            groups.len()
        );
        return None;
    }
    let mut connectivity = Dsu::new(groups.len());
    for &(left, right, _, _) in &constraints {
        connectivity.union(left, right);
    }
    let root = connectivity.find(reference_group);
    let disconnected = (0..groups.len())
        .filter(|&group| connectivity.find(group) != root)
        .collect::<Vec<_>>();
    if !disconnected.is_empty() {
        let names = disconnected
            .iter()
            .map(|&group| {
                Path::new(&images[groups[group].anchor].filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();
        println!(
            "  - Focus capture-group translation geometry unavailable: disconnected group anchors {:?}",
            names
        );
        return None;
    }

    let solve = solve_robust_closure_translations(groups.len(), reference_group, &constraints)?;
    let positions = solve.corrections;
    let iterations = solve.iterations;
    let mut residuals = constraints
        .iter()
        .map(|&(left, right, observed, _)| {
            (positions[right][0] - positions[left][0] - observed[0])
                .hypot(positions[right][1] - positions[left][1] - observed[1])
        })
        .collect::<Vec<_>>();
    let Some(median_residual) = median_value(&mut residuals) else {
        println!(
            "  - Focus capture-group translation geometry unavailable: no finite closure residuals"
        );
        return None;
    };
    let robust_limit = coordinate_scale * 0.04;
    if !median_residual.is_finite() || median_residual > robust_limit {
        println!(
            "  - Focus capture-group translation geometry rejected: median closure error {:.2}px exceeds {:.2}px",
            median_residual, robust_limit
        );
        return None;
    }
    println!(
        "  - Focus capture-group translation geometry seeded from {} relation(s), multi-layer core {consensus_covered}/{}, {iterations} iteration(s), median closure error {:.2}px",
        constraints.len(),
        groups.len(),
        median_residual
    );
    Some(
        positions
            .into_iter()
            .map(|position| {
                Matrix3::new(1.0, 0.0, position[0], 0.0, 1.0, position[1], 0.0, 0.0, 1.0)
            })
            .collect(),
    )
}

/// Refine the maximum-support group spanning-tree poses with all available
/// group relations.  The spanning tree is still used as a stable projective
/// initialisation (it preserves focal scale/roll), but every additional edge
/// contributes a robust translation closure residual.  This prevents one
/// tree edge from carrying the accumulated drift of a long snake scan while
/// retaining the projective geometry measured by the image-level matcher.
///
/// We intentionally solve only a world-space translation correction here. A
/// full projective bundle adjustment at station level is under-constrained on
/// flat artwork and can be pulled by repeated calligraphy. The measured
/// homography remains the base pose; the closure correction is bounded and
/// M-estimator weighted, so an occasional repeated-content edge is strongly
/// downweighted without changing the camera scale or local focus ownership.
#[cfg(test)]
fn refine_focus_group_projective_tree_poses(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    base_poses: &[Matrix3<f64>],
    reference_group: usize,
    coordinate_scale: f64,
) -> Option<Vec<Matrix3<f64>>> {
    let mut closure_report = stack_report::ClosureReport::default();
    let station_topology = topology::StationTopology {
        row: vec![0; groups.len()],
        column: (0..groups.len() as u32).collect(),
        ..topology::StationTopology::default()
    };
    refine_focus_group_projective_tree_poses_with_report(
        images,
        groups,
        group_edges,
        base_poses,
        reference_group,
        coordinate_scale,
        &station_topology,
        2,
        &mut closure_report,
    )
}

fn refine_focus_group_projective_tree_poses_with_report(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    base_poses: &[Matrix3<f64>],
    reference_group: usize,
    _coordinate_scale: f64,
    station_topology: &topology::StationTopology,
    minimum_independent_support: usize,
    closure_report: &mut stack_report::ClosureReport,
) -> Option<Vec<Matrix3<f64>>> {
    closure_report.participating_constraints = group_edges.len();
    closure_report.constraint_weights.clear();
    closure_report.status = stack_report::ClosureStatus::NotRun;
    closure_report.iterations = 0;
    closure_report.residual_median_px = 0.0;
    closure_report.residual_p95_px = 0.0;
    closure_report.max_direct_pair_p95_px = 0.0;
    closure_report.max_corner_correction_px = 0.0;
    closure_report.clamped_stations = 0;
    closure_report.fallback_reason = None;
    closure_report.offending_pair = None;
    if groups.len() < 2 || base_poses.len() != groups.len() || reference_group >= groups.len() {
        return None;
    }

    // Requirement 7.10: a connected graph with no edge beyond its spanning
    // tree has no cycle constraint. Do not enter either joint solver; returning
    // this clone preserves every f64 bit of the tree seed.
    if group_edges.len() <= groups.len().saturating_sub(1) {
        closure_report.status = stack_report::ClosureStatus::NoConstraints;
        closure_report.fallback_reason = Some(degradation::CLOSURE_NO_CONSTRAINTS.to_string());
        degradation::record_run_degradation(
            degradation::CLOSURE_NO_CONSTRAINTS,
            serde_json::json!({
                "accepted_relations": group_edges.len(),
                "station_count": groups.len(),
            }),
        );
        return Some(base_poses.to_vec());
    }

    let mut constraints = Vec::<(usize, usize, [f64; 2], f64)>::new();
    for &(score, left_group, right_group, left_to_right, support, median_error) in group_edges {
        if support < minimum_independent_support
            || left_group >= groups.len()
            || right_group >= groups.len()
            || left_group == right_group
        {
            continue;
        }
        let left_image = &images[groups[left_group].anchor];
        let dimensions = left_image.dimensions();
        let samples = [
            Point2::new(dimensions.0 as f64 * 0.2, dimensions.1 as f64 * 0.2),
            Point2::new(dimensions.0 as f64 * 0.8, dimensions.1 as f64 * 0.2),
            Point2::new(dimensions.0 as f64 * 0.5, dimensions.1 as f64 * 0.5),
            Point2::new(dimensions.0 as f64 * 0.2, dimensions.1 as f64 * 0.8),
            Point2::new(dimensions.0 as f64 * 0.8, dimensions.1 as f64 * 0.8),
        ];
        let mut dx = Vec::new();
        let mut dy = Vec::new();
        for point in samples {
            let Some(left_world) = transformed_point(&base_poses[left_group], point) else {
                continue;
            };
            let Some(right_local) = transformed_point(&left_to_right, point) else {
                continue;
            };
            let Some(right_world) = transformed_point(&base_poses[right_group], right_local) else {
                continue;
            };
            let delta = left_world - right_world;
            if delta.x.is_finite() && delta.y.is_finite() {
                dx.push(delta.x);
                dy.push(delta.y);
            }
        }
        let (Some(dx), Some(dy)) = (median_value(&mut dx), median_value(&mut dy)) else {
            continue;
        };
        let precision = 1.0 / (1.0 + median_error / STATION_RELATION_MAX_MEDIAN_ERROR_PX);
        let weight = support as f64 * score.max(1e-8).sqrt() * precision;
        if weight.is_finite() && weight > 0.0 {
            constraints.push((left_group, right_group, [dx, dy], weight));
        }
    }

    let constraints_are_connected = if constraints.len() + 1 >= groups.len() {
        let mut connectivity = Dsu::new(groups.len());
        for &(left, right, _, _) in &constraints {
            connectivity.union(left, right);
        }
        let reference_root = connectivity.find(reference_group);
        (0..groups.len()).all(|group| connectivity.find(group) == reference_root)
    } else {
        false
    };
    if !constraints_are_connected {
        closure_report.status = stack_report::ClosureStatus::Unreliable;
        closure_report.fallback_reason =
            Some(degradation::CLOSURE_UNRELIABLE_ITERATIONS.to_string());
        degradation::record_run_degradation(
            degradation::CLOSURE_UNRELIABLE_ITERATIONS,
            serde_json::json!({
                "iterations": 0,
                "converged": false,
                "accepted_relations": group_edges.len(),
                "usable_constraints": constraints.len(),
            }),
        );
        return Some(base_poses.to_vec());
    }

    let Some(mut solve) =
        solve_robust_closure_translations(groups.len(), reference_group, &constraints)
    else {
        closure_report.status = stack_report::ClosureStatus::Unreliable;
        closure_report.fallback_reason =
            Some(degradation::CLOSURE_UNRELIABLE_ITERATIONS.to_string());
        degradation::record_run_degradation(
            degradation::CLOSURE_UNRELIABLE_ITERATIONS,
            serde_json::json!({ "iterations": 0, "converged": false }),
        );
        return Some(base_poses.to_vec());
    };
    let iterations = solve.iterations;
    closure_report.iterations = iterations;
    let Some((maximum_correction, clamped_stations)) = clamp_station_translation_corrections(
        images,
        groups,
        group_edges,
        base_poses,
        &mut solve.corrections,
    ) else {
        closure_report.status = stack_report::ClosureStatus::Unreliable;
        closure_report.fallback_reason =
            Some(degradation::CLOSURE_UNRELIABLE_ITERATIONS.to_string());
        degradation::record_run_degradation(
            degradation::CLOSURE_UNRELIABLE_ITERATIONS,
            serde_json::json!({
                "iterations": solve.iterations,
                "converged": false,
                "invalid_correction": true,
            }),
        );
        return Some(base_poses.to_vec());
    };

    closure_report.constraint_weights = constraints
        .iter()
        .map(|&(left, right, observed, _)| {
            let residual = (solve.corrections[right][0] - solve.corrections[left][0] - observed[0])
                .hypot(solve.corrections[right][1] - solve.corrections[left][1] - observed[1]);
            stack_report::ClosureConstraintWeightRecord {
                left,
                right,
                residual_px: residual,
                weight: closure_residual_weight(residual),
            }
        })
        .collect();
    let mut residuals = closure_report
        .constraint_weights
        .iter()
        .map(|relation| relation.residual_px)
        .collect::<Vec<_>>();
    let (median_residual, residual_p95) =
        closure_residual_summary(&mut residuals).unwrap_or((f64::INFINITY, f64::INFINITY));
    closure_report.residual_median_px = median_residual;
    closure_report.residual_p95_px = residual_p95;

    let preliminary = closure_acceptance(
        groups.len(),
        group_edges.len(),
        solve.converged,
        solve.iterations,
        median_residual,
        &[],
    );
    if preliminary != ClosureAcceptance::Accepted {
        let reason = match preliminary {
            ClosureAcceptance::UnreliableResidual => degradation::CLOSURE_UNRELIABLE_RESIDUAL,
            ClosureAcceptance::UnreliableIterations => degradation::CLOSURE_UNRELIABLE_ITERATIONS,
            ClosureAcceptance::NoConstraints => degradation::CLOSURE_NO_CONSTRAINTS,
            ClosureAcceptance::Accepted | ClosureAcceptance::UnreliablePairP95(_) => unreachable!(),
        };
        closure_report.status = stack_report::ClosureStatus::Unreliable;
        closure_report.fallback_reason = Some(reason.to_string());
        degradation::record_run_degradation(
            reason,
            serde_json::json!({
                "iterations": solve.iterations,
                "converged": solve.converged,
                "residual_median_px": median_residual,
                "residual_limit_px": CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX,
            }),
        );
        return Some(base_poses.to_vec());
    }

    let mut refined = Vec::with_capacity(groups.len());
    for (group_index, pose) in base_poses.iter().enumerate() {
        let correction = solve.corrections[group_index];
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
        let corrected = translation * pose;
        if corrected.try_inverse().is_none()
            || !homography_preserves_focus_orientation(
                &corrected,
                images[groups[group_index].anchor].dimensions(),
            )
        {
            closure_report.status = stack_report::ClosureStatus::Unreliable;
            closure_report.fallback_reason =
                Some(degradation::CLOSURE_UNRELIABLE_PAIR_P95.to_string());
            degradation::record_run_degradation(
                degradation::CLOSURE_UNRELIABLE_PAIR_P95,
                serde_json::json!({
                    "station": group_index,
                    "invalid_corrected_pose": true,
                }),
            );
            return Some(base_poses.to_vec());
        }
        refined.push(corrected);
    }

    let direct_pairs = closure_direct_pair_p95s(images, groups, group_edges, &refined);
    closure_report.max_direct_pair_p95_px = direct_pairs
        .iter()
        .map(|pair| pair.p95_px)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    match closure_acceptance(
        groups.len(),
        group_edges.len(),
        solve.converged,
        solve.iterations,
        median_residual,
        &direct_pairs,
    ) {
        ClosureAcceptance::Accepted => {}
        ClosureAcceptance::UnreliablePairP95(pair) => {
            let offending = closure_offending_pair_record(pair, station_topology);
            closure_report.status = stack_report::ClosureStatus::Unreliable;
            closure_report.fallback_reason =
                Some(degradation::CLOSURE_UNRELIABLE_PAIR_P95.to_string());
            closure_report.offending_pair = Some(offending);
            closure_report.max_corner_correction_px = 0.0;
            closure_report.clamped_stations = 0;
            degradation::record_run_degradation(
                degradation::CLOSURE_UNRELIABLE_PAIR_P95,
                serde_json::json!({
                    "left": offending.left,
                    "left_row": offending.left_row,
                    "left_column": offending.left_column,
                    "right": offending.right,
                    "right_row": offending.right_row,
                    "right_column": offending.right_column,
                    "p95_px": offending.p95_px,
                    "limit_px": CLOSURE_DIRECT_PAIR_P95_LIMIT_PX,
                }),
            );
            return Some(base_poses.to_vec());
        }
        ClosureAcceptance::NoConstraints
        | ClosureAcceptance::UnreliableResidual
        | ClosureAcceptance::UnreliableIterations => unreachable!(),
    }

    closure_report.status = stack_report::ClosureStatus::Converged;
    closure_report.max_corner_correction_px = maximum_correction;
    closure_report.clamped_stations = clamped_stations;
    if clamped_stations > 0 {
        degradation::record_run_degradation(
            degradation::CLOSURE_CORRECTION_CLAMPED,
            serde_json::json!({
                "clamped_stations": clamped_stations,
                "max_corner_correction_px": maximum_correction,
            }),
        );
    }
    println!(
        "  - Focus capture-group closure accepted: {} relations, {} iteration(s), median/P95 residual {:.2}/{:.2}px, maximum direct-pair P95 {:.2}px, maximum corner correction {:.2}px, {} clamped station(s)",
        constraints.len(),
        solve.iterations,
        median_residual,
        residual_p95,
        closure_report.max_direct_pair_p95_px,
        maximum_correction,
        clamped_stations,
    );
    Some(refined)
}

fn correct_focus_group_roll_from_long_edges(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_poses: &mut [Matrix3<f64>],
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_group: usize,
) -> usize {
    if groups.len() != group_poses.len() || reference_group >= groups.len() {
        return 0;
    }
    let mut group_angles = vec![None; groups.len()];
    for (group_index, group) in groups.iter().enumerate() {
        let image = &images[group.anchor];
        let pose = &group_poses[group_index];
        let mut slopes = image
            .horizontal_edge_rows
            .iter()
            .filter_map(|&row| fit_focus_horizontal_edge_line(image, pose, row))
            .map(|line| line.slope)
            .filter(|slope| slope.is_finite())
            .collect::<Vec<_>>();
        if slopes.len() < FOCUS_GROUP_ROLL_MIN_LINES {
            continue;
        }
        let Some(slope) = median_value(&mut slopes) else {
            continue;
        };
        let mut deviations = slopes
            .iter()
            .map(|value| (value - slope).abs())
            .collect::<Vec<_>>();
        let Some(mad) = median_value(&mut deviations) else {
            continue;
        };
        if mad > FOCUS_GROUP_ROLL_MAX_SLOPE_MAD {
            continue;
        }
        group_angles[group_index] = Some(slope.atan());
    }
    let mut reliable_angles = group_angles.iter().flatten().copied().collect::<Vec<_>>();
    let target_angle = median_value(&mut reliable_angles);
    let reference_pose_angle = locked_homographies
        .get(&images[groups[reference_group].anchor].id)
        .map(|pose| pose[(1, 0)].atan2(pose[(0, 0)]));

    let mut corrected_count = 0usize;
    let mut edge_corrected_count = 0usize;
    for (group_index, angle) in group_angles.into_iter().enumerate() {
        let edge_correction = angle
            .zip(target_angle)
            .map(|(angle, target)| target - angle);
        let pose_correction = reference_pose_angle.and_then(|reference| {
            locked_homographies
                .get(&images[groups[group_index].anchor].id)
                .map(|pose| pose[(1, 0)].atan2(pose[(0, 0)]) - reference)
        });
        let Some(raw_correction) = edge_correction.or(pose_correction) else {
            continue;
        };
        let correction_angle = raw_correction.clamp(
            -FOCUS_GROUP_ROLL_MAX_CORRECTION_RADIANS,
            FOCUS_GROUP_ROLL_MAX_CORRECTION_RADIANS,
        );
        if correction_angle.abs() < 1e-5 {
            continue;
        }
        let image = &images[groups[group_index].anchor];
        let source_center = Point2::new(image.width as f64 * 0.5, image.height as f64 * 0.5);
        let Some(world_center) = transformed_point(&group_poses[group_index], source_center) else {
            continue;
        };
        let (sin, cos) = correction_angle.sin_cos();
        let rotate_about_center = Matrix3::new(
            cos,
            -sin,
            world_center.x - cos * world_center.x + sin * world_center.y,
            sin,
            cos,
            world_center.y - sin * world_center.x - cos * world_center.y,
            0.0,
            0.0,
            1.0,
        );
        let corrected = rotate_about_center * group_poses[group_index];
        if transform_is_stable_for_focus_stack(&corrected, image.dimensions()) {
            group_poses[group_index] = corrected;
            corrected_count += 1;
            edge_corrected_count += usize::from(edge_correction.is_some());
        }
    }
    if corrected_count > 0 {
        println!(
            "  - Focus capture-group roll correction: {corrected_count}/{} groups ({edge_corrected_count} from long edges, {} pose-only), long-edge target {}, maximum correction {:.3} degrees",
            groups.len(),
            corrected_count.saturating_sub(edge_corrected_count),
            target_angle
                .map(|angle| format!("{:.3} degrees", angle.to_degrees()))
                .unwrap_or_else(|| "unavailable".to_string()),
            FOCUS_GROUP_ROLL_MAX_CORRECTION_RADIANS.to_degrees(),
        );
    }
    corrected_count
}

fn focus_station_topology(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    station_homographies: &HashMap<usize, Matrix3<f64>>,
) -> topology::StationTopology {
    let observations = groups
        .iter()
        .enumerate()
        .map(|(station_index, group)| {
            let anchor = &images[group.anchor];
            let pose = station_homographies.get(&anchor.id);
            let center = pose.and_then(|pose| {
                transformed_point(
                    pose,
                    Point2::new(anchor.width as f64 * 0.5, anchor.height as f64 * 0.5),
                )
            });
            let effective_width = pose.and_then(|pose| {
                let corners = [
                    Point2::new(0.0, 0.0),
                    Point2::new(anchor.width as f64, 0.0),
                    Point2::new(0.0, anchor.height as f64),
                    Point2::new(anchor.width as f64, anchor.height as f64),
                ];
                let projected = corners
                    .into_iter()
                    .map(|corner| transformed_point(pose, corner))
                    .collect::<Option<Vec<_>>>()?;
                let min_x = projected
                    .iter()
                    .map(|point| point.x)
                    .fold(f64::INFINITY, f64::min);
                let max_x = projected
                    .iter()
                    .map(|point| point.x)
                    .fold(f64::NEG_INFINITY, f64::max);
                let width = max_x - min_x;
                (width.is_finite() && width > 0.0).then_some(width)
            });
            topology::StationObservation {
                station_index,
                world_x: center.map(|point| point.x),
                world_y: center.map(|point| point.y),
                effective_width,
                filename: anchor.filename.clone(),
            }
        })
        .collect::<Vec<_>>();

    // Candidate scores are the transform-estimated overlap area divided by one
    // station's effective area. Keep the matrix outside StationTopology so the
    // topology result remains pose-free (requirements 5.4 and 5.5).
    let mut overlap_scores = BTreeMap::new();
    for left in 0..groups.len() {
        let left_anchor = &images[groups[left].anchor];
        let Some(left_to_world) = station_homographies.get(&left_anchor.id) else {
            continue;
        };
        for right in (left + 1)..groups.len() {
            let right_anchor = &images[groups[right].anchor];
            let Some(world_to_right) = station_homographies
                .get(&right_anchor.id)
                .and_then(|pose| pose.try_inverse())
            else {
                continue;
            };
            let left_to_right = world_to_right * left_to_world;
            overlap_scores.insert(
                (left, right),
                panorama_transform_overlap_support(
                    &left_to_right,
                    left_anchor.dimensions(),
                    right_anchor.dimensions(),
                ),
            );
        }
    }

    topology::infer_station_topology(&observations, &overlap_scores)
}

fn station_pose_candidate_has_bounded_local_scale(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    poses: &[Matrix3<f64>],
    stage: &str,
) -> bool {
    if poses.len() != groups.len() {
        return false;
    }
    let mut accepted = true;
    for (station_index, (group, pose)) in groups.iter().zip(poses).enumerate() {
        let anchor = &images[group.anchor];
        let range = station_pose_local_scale_range(pose, anchor.dimensions());
        let ratio = range.map(|(minimum, maximum)| maximum / minimum);
        if ratio.is_some_and(|ratio| ratio <= STATION_POSE_MAX_LOCAL_SCALE_RATIO) {
            continue;
        }
        accepted = false;
        degradation::record_run_degradation(
            degradation::STATION_POSE_LOCAL_SCALE_EXCEEDED,
            serde_json::json!({
                "station_index": station_index,
                "anchor_path": absolute_source_path(&anchor.filename),
                "stage": stage,
                "minimum_local_scale": range.map(|range| range.0),
                "maximum_local_scale": range.map(|range| range.1),
                "local_scale_ratio": ratio,
                "maximum_ratio": STATION_POSE_MAX_LOCAL_SCALE_RATIO,
            }),
        );
        println!(
            "  - Station pose candidate rejected at station {station_index}: Local_Scale ratio {} exceeds {:.2}; retaining spanning-tree seed",
            ratio
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "unmeasurable".to_string()),
            STATION_POSE_MAX_LOCAL_SCALE_RATIO,
        );
    }
    accepted
}

/// Keep a general projective candidate exactly as solved when its sampled
/// Local_Scale is bounded; otherwise preserve every bit of the spanning-tree
/// seed (requirement 6.6).
fn select_station_pose_candidate(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    candidate: &[Matrix3<f64>],
    spanning_tree_seed: &[Matrix3<f64>],
    stage: &str,
) -> Vec<Matrix3<f64>> {
    if station_pose_candidate_has_bounded_local_scale(images, groups, candidate, stage) {
        candidate.to_vec()
    } else {
        spanning_tree_seed.to_vec()
    }
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

fn station_graph_components_from_members(
    station_members: &[Vec<String>],
    relation_pairs: &[(usize, usize)],
) -> Vec<Vec<String>> {
    let mut connectivity = Dsu::new(station_members.len());
    for &(left, right) in relation_pairs {
        if left < station_members.len() && right < station_members.len() {
            connectivity.union(left, right);
        }
    }
    let mut stations_by_root = BTreeMap::<usize, Vec<usize>>::new();
    for station_index in 0..station_members.len() {
        let root = connectivity.find(station_index);
        stations_by_root
            .entry(root)
            .or_default()
            .push(station_index);
    }
    stations_by_root
        .into_values()
        .map(|station_indices| {
            let mut members = station_indices
                .into_iter()
                .flat_map(|station_index| station_members[station_index].iter().cloned())
                .collect::<Vec<_>>();
            members.sort();
            members
        })
        .collect()
}

fn station_graph_components(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    relation_pairs: &[(usize, usize)],
) -> Vec<Vec<String>> {
    let station_members = groups
        .iter()
        .map(|group| {
            group
                .members
                .iter()
                .map(|&image_index| absolute_source_path(&images[image_index].filename))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    station_graph_components_from_members(&station_members, relation_pairs)
}

/// Populate the canonical connectivity report and return the exact rejected
/// diagnostic payload when the accepted Station_Relation graph is disconnected.
fn station_graph_connectivity_rejection(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    relation_pairs: &[(usize, usize)],
    report: &mut stack_report::StationRelationsReport,
) -> Option<serde_json::Value> {
    let components = station_graph_components(images, groups, relation_pairs);
    report.connectivity.components = components.len();
    report.connectivity.component_member_counts = components.iter().map(Vec::len).collect();
    report.connectivity.component_members = components.clone();
    (components.len() > 1).then(|| {
        let component_details = components
            .iter()
            .map(|members| {
                serde_json::json!({
                    "member_count": members.len(),
                    "members": members,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "component_count": components.len(),
            "components": component_details,
            "actions": ["按连续场景重新分组", "补拍增加重叠"],
        })
    })
}

fn geometry_disconnected_user_error(component_count: usize) -> String {
    format!(
        "机位几何图包含 {component_count} 个不连通分量，无法导出完整拼图。可执行操作：“按连续场景重新分组”或“补拍增加重叠”。"
    )
}

#[cfg(test)]
fn solve_focus_capture_group_poses(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
    translation_geometry_verified: &mut bool,
) -> HashMap<usize, Matrix3<f64>> {
    let mut closure_report = stack_report::ClosureReport::default();
    let mut station_relations_report = stack_report::StationRelationsReport::default();
    solve_focus_capture_group_poses_with_report(
        images,
        matches,
        locked_homographies,
        reference_index,
        translation_geometry_verified,
        &mut closure_report,
        &mut station_relations_report,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StationRelationEvidenceKind {
    SourceRaw,
    VirtualTile,
}

impl StationRelationEvidenceKind {
    fn minimum_independent_support(self) -> usize {
        match self {
            Self::SourceRaw => 2,
            // One fused station tile is already the requirement-6.1 primary
            // observation.  This is not a fabricated second source: the
            // evidence kind explicitly exempts it from requirement 6.2.
            Self::VirtualTile => 1,
        }
    }
}

fn solve_focus_capture_group_poses_with_report(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
    translation_geometry_verified: &mut bool,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
) -> HashMap<usize, Matrix3<f64>> {
    solve_station_poses_for_evidence_with_report(
        images,
        matches,
        locked_homographies,
        reference_index,
        translation_geometry_verified,
        closure_report,
        station_relations_report,
        StationRelationEvidenceKind::SourceRaw,
        None,
        None,
    )
}

/// Coverage_Masks the station solver reads for Virtual_Tile evidence: every
/// re-check it makes on a tile pair looks only at pixels both tiles cover
/// (需求 6.1).
fn solve_virtual_tile_station_hypotheses_with_report(
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
    hypotheses: &VirtualTileRelationHypotheses,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
) -> HashMap<usize, Matrix3<f64>> {
    let selected = select_virtual_tile_relation_hypotheses(tiles, hypotheses, coverages);
    solve_virtual_tile_station_poses_with_report(
        tiles,
        coverages,
        &selected,
        initial_homographies,
        station_topology,
        closure_report,
        station_relations_report,
    )
}

fn solve_virtual_tile_station_poses_with_report(
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
) -> HashMap<usize, Matrix3<f64>> {
    let mut verified = false;
    solve_station_poses_for_evidence_with_report(
        tiles,
        matches,
        initial_homographies,
        0,
        &mut verified,
        closure_report,
        station_relations_report,
        StationRelationEvidenceKind::VirtualTile,
        Some(station_topology),
        Some(coverages),
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_station_poses_for_evidence_with_report(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
    translation_geometry_verified: &mut bool,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
    evidence_kind: StationRelationEvidenceKind,
    supplied_topology: Option<&topology::StationTopology>,
    coverages: Option<VirtualTileSolverCoverages<'_>>,
) -> HashMap<usize, Matrix3<f64>> {
    *translation_geometry_verified = false;
    *station_relations_report = stack_report::StationRelationsReport::default();
    let (mut groups, group_for_image) = match evidence_kind {
        StationRelationEvidenceKind::SourceRaw => {
            let Some(groups) = focus_capture_groups(images, matches, locked_homographies) else {
                return locked_homographies.clone();
            };
            groups
        }
        StationRelationEvidenceKind::VirtualTile => {
            let groups = (0..images.len())
                .map(|index| FocusCaptureGroup {
                    members: vec![index],
                    anchor: index,
                    local_to_anchor: HashMap::from([(index, Matrix3::identity())]),
                })
                .collect::<Vec<_>>();
            (groups, (0..images.len()).collect::<Vec<_>>())
        }
    };
    // The Focus_Fuser anchor follows requirement 2.1 and may legitimately use
    // the source path for a near-sharpness tie. Station geometry cannot inherit
    // that rename-sensitive coordinate choice (requirements 1.3/5.8), so the
    // pose graph uses an equivalent content/geometry-derived representative.
    if canonicalize_station_geometry_anchors(images, &mut groups, locked_homographies).is_none() {
        return locked_homographies.clone();
    }
    if groups.len() < 2 || reference_index >= images.len() {
        return locked_homographies.clone();
    }
    // 需求 1.6: an isolated source belongs to no Capture_Station, so it cannot
    // anchor the station pose graph.  Keep the image-level geometry instead of
    // rooting the tree at a station that does not exist.
    if group_for_image[reference_index] == usize::MAX {
        return locked_homographies.clone();
    }
    // Task 9.1 is part of the production station-relation path: infer the
    // displacement-only topology before collecting relation evidence, retain
    // it for Stack_Report, and use it below as the candidate eligibility gate.
    // Matrix-valued poses are reduced to centres/widths before entering the
    // topology module (需求 5.2 / 5.4).
    let station_topology = supplied_topology
        .cloned()
        .unwrap_or_else(|| focus_station_topology(images, &groups, locked_homographies));
    if evidence_kind == StationRelationEvidenceKind::SourceRaw {
        topology::record_run_topology(station_topology.clone());
    }
    // An isolated source keeps the image-level pose the station solve cannot
    // refine, so the returned map still covers the whole selection.
    let carry_unstationed_poses = |solved: &mut HashMap<usize, Matrix3<f64>>| {
        for (image_index, image) in images.iter().enumerate() {
            if group_for_image[image_index] != usize::MAX {
                continue;
            }
            if let Some(&pose) = locked_homographies.get(&image.id) {
                solved.insert(image.id, pose);
            }
        }
    };

    #[cfg(test)]
    if std::env::var_os("RAW_EDITOR_STACK_GROUP_DIAGNOSTICS").is_some() {
        println!("  - Focus capture groups in capture order:");
        for (group_index, group) in groups.iter().enumerate() {
            let names = group
                .members
                .iter()
                .map(|&image_index| {
                    Path::new(&images[image_index].filename)
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect::<Vec<_>>()
                .join(",");
            println!("    group {group_index}: {names}");
        }
    }

    // Station-pair candidates are keyed by a group pair and later compared by
    // score; both the candidate order inside a pair and the order of the pairs
    // themselves must not depend on the hash seed (需求 14.6).  The map is one
    // entry per station pair, so an ordered container costs nothing here.
    let mut candidates = BTreeMap::<(usize, usize), Vec<FocusGroupEdgeCandidate>>::new();
    let mut non_invertible_candidate_pairs = BTreeSet::<(usize, usize)>::new();
    for (&(source_index, target_index), match_info) in sorted_pairs(matches) {
        if source_index >= images.len()
            || target_index >= images.len()
            || match_info.sequence_bridge
        {
            continue;
        }
        // A single coarse correlation peak is not geometry evidence.  A
        // component-recovery edge, however, is inserted only after at least
        // two different focus layers on both sides agree on the same planar
        // transform.  Preserve that independent support in the station-level
        // solver instead of using the edge merely to make the image graph
        // connected and then falling back to a repeated-texture image pose.
        let consensus_coarse =
            match_info.coarse_bridge && !match_info.top_candidate_points.is_empty();
        if (match_info.coarse_bridge && !consensus_coarse)
            || (!consensus_coarse && match_info.points.len() < FOCUS_MODEL_MIN_INLIERS)
        {
            continue;
        }
        // 需求 1.6: an isolated source is a member of no Capture_Station, so it
        // contributes no station relation either.
        if group_for_image[source_index] == usize::MAX
            || group_for_image[target_index] == usize::MAX
        {
            continue;
        }
        if group_for_image[source_index] == group_for_image[target_index] {
            continue;
        }
        if !station_topology
            .pair_is_eligible(group_for_image[source_index], group_for_image[target_index])
        {
            continue;
        }
        let canonical =
            match canonical_relation_view(images, source_index, target_index, match_info) {
                Ok(view) => view,
                Err(CanonicalRelationViewError::NonInvertible) => {
                    let source_group = group_for_image[source_index];
                    let target_group = group_for_image[target_index];
                    non_invertible_candidate_pairs.insert((
                        source_group.min(target_group),
                        source_group.max(target_group),
                    ));
                    continue;
                }
                Err(CanonicalRelationViewError::IndexOutOfBounds {
                    left,
                    right,
                    image_count,
                }) => panic!(
                    "canonical relation pair ({left}, {right}) is outside {image_count} images"
                ),
            };
        let quality_source = canonical.source_index;
        let quality_target = canonical.target_index;
        let quality_inverted = canonical.inverted;
        let quality_homography = canonical.homography;
        let source_group = group_for_image[quality_source];
        let target_group = group_for_image[quality_target];
        let quality_points: Cow<'_, [(Point2<f64>, Point2<f64>)]> = if quality_inverted {
            Cow::Owned(
                match_info
                    .points
                    .iter()
                    .map(|&(source, target)| (target, source))
                    .collect(),
            )
        } else {
            Cow::Borrowed(&match_info.points)
        };
        let Some(source_local) = groups[source_group].local_to_anchor.get(&quality_source) else {
            continue;
        };
        let Some(source_local_inverse) = (*source_local).try_inverse() else {
            continue;
        };
        let Some(target_local) = groups[target_group].local_to_anchor.get(&quality_target) else {
            continue;
        };
        let source_anchor_to_target_anchor =
            target_local * quality_homography * source_local_inverse;
        let (key, transform) = if source_group < target_group {
            ((source_group, target_group), source_anchor_to_target_anchor)
        } else {
            let Some(inverse) = source_anchor_to_target_anchor.try_inverse() else {
                continue;
            };
            ((target_group, source_group), inverse)
        };
        // Acceptance evidence belongs to the immutable measured source pair,
        // not to whichever near-tied focus frames happen to become station
        // anchors. Requirement 2.1 permits a path tie-break for that anchor;
        // requirements 1.3/5.8 therefore require these measurements to remain
        // in the canonical content-derived source direction. The composed
        // anchor transform and anchor points remain the pose-graph payload.
        let measurement_source_dimensions = images[quality_source].dimensions();
        let measurement_target_dimensions = images[quality_target].dimensions();
        // Virtual_Tile evidence is re-checked only where both tiles are
        // covered; a missing mask leaves the relation unmeasured (rejected).
        let quality_coverage_pair = match coverages {
            None => Some(None),
            Some(coverages) => coverages
                .quality
                .get(quality_source)
                .zip(coverages.quality.get(quality_target))
                .map(Some),
        };
        // Requirement 6.4's overlap area: the frames' geometric intersection
        // for source images, the doubly covered pixels for Virtual_Tiles.
        let overlap_measure = match coverages {
            None => Some(StationOverlapMeasure::Rectangles),
            Some(coverages) => coverages.overlap_measure(quality_source, quality_target),
        };
        let dense_measurements = quality_coverage_pair.and_then(|coverages| {
            focus_overlap_quality_measurements_on_coverage(
                &images[quality_source],
                &images[quality_target],
                &quality_homography,
                coverages,
            )
        });
        let Some(measured) = measure_station_relation_candidate(
            match_info.inliers.max(quality_points.len()),
            &quality_homography,
            &quality_points,
            measurement_source_dimensions,
            measurement_target_dimensions,
            overlap_measure,
            dense_measurements,
        ) else {
            continue;
        };
        let measurements = measured.measurements;
        candidates
            .entry(key)
            .or_default()
            .push(FocusGroupEdgeCandidate {
                transform,
                quality: measured.quality,
                independent_support: if consensus_coarse { 2 } else { 1 },
                inlier_count: measurements.inliers,
                median_error_px: measurements.median_error_px,
                scale_ratio: measurements.scale_ratio,
                spatial_support: measurements.spatial_support,
                low_frequency_mean_relative_difference: measurements
                    .low_frequency_mean_relative_difference,
                edge_strength_ratio: measurements.edge_strength_ratio,
                median_edge_orientation_difference_degrees: measurements
                    .median_edge_orientation_difference_degrees,
                preserves_convex_orientation: measurements.preserves_convex_orientation,
            });
    }

    let final_non_invertible_pairs = non_invertible_candidate_pairs
        .into_iter()
        .filter(|pair| !candidates.contains_key(pair))
        .collect::<BTreeSet<_>>();
    for &(left, right) in &final_non_invertible_pairs {
        station_relations_report.record_non_invertible_canonical_pair(left, right);
    }

    // One station pair collects one candidate per accepted focus layer
    // relation, and the consensus search below reads that vector in order: it
    // accumulates `supporting_quality` along it and keeps the first candidate on
    // a score tie.  Pushing in `matches` key order would make both inherit the
    // import positions the keys are built from, so re-order every candidate
    // list by its own measured content (需求 14.6).
    for pair_candidates in candidates.values_mut() {
        pair_candidates.sort_by(|left, right| {
            right
                .quality
                .total_cmp(&left.quality)
                .then_with(|| right.independent_support.cmp(&left.independent_support))
                .then_with(|| {
                    focus_group_candidate_key(left).cmp(&focus_group_candidate_key(right))
                })
        });
    }

    let coordinate_scale = images
        .iter()
        .map(|image| image.width.max(image.height) as f64)
        .fold(1.0, f64::max);
    let consensus_threshold = STATION_RELATION_MAX_MEDIAN_ERROR_PX;
    let mut group_edges = Vec::<(f64, usize, usize, Matrix3<f64>, usize, f64)>::new();
    let mut station_relation_inlier_counts = BTreeMap::<(usize, usize), usize>::new();
    let mut evaluated_candidate_pairs = final_non_invertible_pairs
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut consensus_pair_count = 0usize;
    for ((left_group, right_group), pair_candidates) in candidates {
        evaluated_candidate_pairs.insert((left_group, right_group));
        let dimensions = images[groups[left_group].anchor].dimensions();
        let mut best: Option<(f64, FocusGroupEdgeCandidate, usize, f64, usize)> = None;
        for candidate in &pair_candidates {
            let mut supporting_errors = Vec::new();
            let mut supporting_quality = 0.0;
            let mut independent_support = 0usize;
            let mut inlier_count = 0usize;
            for other in &pair_candidates {
                let Some(error) = focus_transform_disagreement_px(
                    &candidate.transform,
                    &other.transform,
                    dimensions,
                ) else {
                    continue;
                };
                if error <= consensus_threshold {
                    supporting_errors.push(error);
                    supporting_quality += other.quality;
                    independent_support += other.independent_support;
                    inlier_count = inlier_count.saturating_add(other.inlier_count);
                }
            }
            let support = independent_support;
            let discard = (evidence_kind == StationRelationEvidenceKind::SourceRaw)
                .then(|| single_layer_station_evidence_discard(support, left_group, right_group))
                .flatten();
            if discard.is_none() {
                let consensus_error =
                    median_value(&mut supporting_errors).unwrap_or(consensus_threshold);
                let score = (support * support) as f64 * supporting_quality
                    / (1.0 + consensus_error / consensus_threshold.max(1.0));
                if best
                    .as_ref()
                    .is_none_or(|(best_score, _, _, _, _)| score > *best_score)
                {
                    best = Some((score, *candidate, support, consensus_error, inlier_count));
                }
                continue;
            }
            degradation::record_run_degradation(
                degradation::STATION_RELATION_SINGLE_LAYER_EVIDENCE_DISCARDED,
                discard.expect("source evidence discard was checked above"),
            );
        }
        let Some((score, candidate, support, consensus_error, inlier_count)) = best else {
            if evidence_kind == StationRelationEvidenceKind::VirtualTile {
                record_station_relation_no_fit(left_group, right_group, station_relations_report);
            }
            continue;
        };
        let measurements = StationRelationMeasurements {
            inliers: inlier_count,
            median_error_px: candidate.median_error_px,
            scale_ratio: candidate.scale_ratio,
            spatial_support: candidate.spatial_support,
            low_frequency_mean_relative_difference: candidate
                .low_frequency_mean_relative_difference,
            edge_strength_ratio: candidate.edge_strength_ratio,
            median_edge_orientation_difference_degrees: candidate
                .median_edge_orientation_difference_degrees,
            preserves_convex_orientation: candidate.preserves_convex_orientation,
        };
        if !station_relation_is_accepted_with_report(
            left_group,
            right_group,
            consensus_error,
            &measurements,
            station_relations_report,
        ) {
            continue;
        }
        consensus_pair_count += 1;
        station_relations_report
            .accepted
            .push(stack_report::AcceptedStationRelationRecord {
                left: left_group,
                right: right_group,
                evidence_kind: match evidence_kind {
                    StationRelationEvidenceKind::SourceRaw => {
                        stack_report::StationRelationEvidenceKind::SourceRawConsensus
                    }
                    StationRelationEvidenceKind::VirtualTile => {
                        stack_report::StationRelationEvidenceKind::VirtualTile
                    }
                },
                inliers: measurements.inliers,
                spatial_support: measurements.spatial_support,
                scale_ratio: measurements.scale_ratio,
                median_error_px: measurements.median_error_px,
                independent_support: support,
            });
        station_relation_inlier_counts.insert((left_group, right_group), inlier_count);
        group_edges.push((
            score,
            left_group,
            right_group,
            candidate.transform,
            support,
            measurements.median_error_px,
        ));
    }
    if evidence_kind == StationRelationEvidenceKind::VirtualTile {
        for candidate in &station_topology.candidates {
            let pair = (
                candidate.left.min(candidate.right),
                candidate.left.max(candidate.right),
            );
            if pair.0 >= groups.len()
                || pair.1 >= groups.len()
                || !evaluated_candidate_pairs.insert(pair)
            {
                continue;
            }
            record_station_relation_no_fit(pair.0, pair.1, station_relations_report);
        }
    }
    closure_report.participating_constraints = group_edges.len();
    closure_report.constraint_weights.clear();

    // Requirements 6.7 / 12.4: an accepted Station_Relation graph must cover
    // every Capture_Station.  Image-level locked poses are diagnostics and a
    // spanning-tree seed, not permission to render independent components into
    // one apparently complete mosaic.
    let relation_pairs = group_edges
        .iter()
        .map(|edge| (edge.1, edge.2))
        .collect::<Vec<_>>();
    if let Some(rejection) = station_graph_connectivity_rejection(
        images,
        &groups,
        &relation_pairs,
        station_relations_report,
    ) {
        let component_count = station_relations_report.connectivity.components;
        degradation::record_run_degradation(degradation::GEOMETRY_DISCONNECTED, rejection);
        println!(
            "  - Focus station geometry rejected: {component_count} disconnected component(s)"
        );
        return HashMap::new();
    }

    let reference_group = group_for_image[reference_index];
    #[cfg(test)]
    if std::env::var_os("RAW_EDITOR_STACK_GROUP_DIAGNOSTICS").is_some() {
        let mut diagnostics = group_edges.clone();
        diagnostics.sort_by_key(|edge| (edge.1, edge.2));
        println!("  - Focus capture-group relations:");
        for &(_, left, right, transform, support, median_error) in &diagnostics {
            let image = &images[groups[left].anchor];
            let center = Point2::new(image.width as f64 * 0.5, image.height as f64 * 0.5);
            let mapped = transformed_point(&transform, center).unwrap_or(center);
            println!(
                "    {left}->{right}: gap={} support={support} error={median_error:.1}px center=({:.1},{:.1})",
                right.saturating_sub(left),
                center.x - mapped.x,
                center.y - mapped.y,
            );
        }
    }
    // Requirement 7.1: accepted-relation inlier count is the only spanning-tree
    // weight. Composite quality continues to inform the joint optimizer, but it
    // cannot change the projective seed. Equal inlier counts use the inferred
    // station row/column coordinates as the complete deterministic tie-break.
    let tree_relations = group_edges
        .iter()
        .map(|&(_, left, right, transform, _, _)| {
            let key = (left.min(right), left.max(right));
            let inlier_count = station_relation_inlier_counts
                .get(&key)
                .copied()
                .unwrap_or_default();
            (
                left,
                right,
                transform.try_inverse().is_some().then_some(inlier_count),
            )
        })
        .collect::<Vec<_>>();
    let selectable_relations = tree_relations
        .iter()
        .filter_map(|&(left, right, inlier_count)| {
            inlier_count.map(|inlier_count| (left, right, inlier_count))
        })
        .collect::<Vec<_>>();
    let selected_tree_indices = maximum_inlier_spanning_tree_indices(
        groups.len(),
        &station_topology,
        &selectable_relations,
    );
    let selectable_group_edge_indices = tree_relations
        .iter()
        .enumerate()
        .filter_map(|(index, relation)| relation.2.map(|_| index))
        .collect::<Vec<_>>();

    let mut group_adjacency = HashMap::<usize, Vec<(usize, Matrix3<f64>)>>::new();
    let mut tree_edges = 0usize;
    for selectable_index in selected_tree_indices {
        let group_edge_index = selectable_group_edge_indices[selectable_index];
        let (_, left_group, right_group, left_to_right, _, _) = group_edges[group_edge_index];
        let right_to_left = left_to_right
            .try_inverse()
            .expect("tree candidates were filtered to invertible transforms");
        group_adjacency
            .entry(left_group)
            .or_default()
            .push((right_group, right_to_left));
        group_adjacency
            .entry(right_group)
            .or_default()
            .push((left_group, left_to_right));
        tree_edges += 1;
    }
    closure_report.spanning_tree_edges = tree_edges;
    let mut group_poses = vec![None; groups.len()];
    group_poses[reference_group] = locked_homographies
        .get(&images[groups[reference_group].anchor].id)
        .copied();
    let mut pending = VecDeque::from([reference_group]);
    while let Some(current) = pending.pop_front() {
        let Some(current_pose) = group_poses[current] else {
            continue;
        };
        for &(neighbor, neighbor_to_current) in group_adjacency.get(&current).into_iter().flatten()
        {
            if group_poses[neighbor].is_some() {
                continue;
            }
            group_poses[neighbor] = Some(current_pose * neighbor_to_current);
            pending.push_back(neighbor);
        }
    }
    if tree_edges + 1 == groups.len() && group_poses.iter().all(Option::is_some) {
        let base_group_poses = group_poses
            .iter()
            .map(|pose| pose.expect("connected group tree must provide every pose"))
            .collect::<Vec<_>>();
        // The maximum-support tree keeps each station's full 8-DOF measured
        // homography. Task 9.9 integrates the 9.6-9.8 robust closure path here
        // as the only joint update, so no earlier projective solve can bypass
        // its convergence, median-residual, and direct-pair P95 acceptance
        // gates.
        // Keep the maximum-support tree as the projective initialisation, but
        // let every reliable closure edge vote on a bounded world translation
        // correction. This is the generic path for both horizontal/vertical
        // scan closures and snake cross-column links; no filename topology is
        // assumed here.
        let refined_group_poses = refine_focus_group_projective_tree_poses_with_report(
            images,
            &groups,
            &group_edges,
            &base_group_poses,
            reference_group,
            coordinate_scale,
            &station_topology,
            evidence_kind.minimum_independent_support(),
            closure_report,
        )
        .unwrap_or_else(|| base_group_poses.clone());
        let selected_group_poses = select_station_pose_candidate(
            images,
            &groups,
            &refined_group_poses,
            &base_group_poses,
            "closure_refinement",
        );
        let build_solved = |poses: &[Matrix3<f64>]| {
            let mut solved = HashMap::new();
            for (group_index, group) in groups.iter().enumerate() {
                let group_pose = poses[group_index];
                for &image_index in &group.members {
                    let replacement = group_pose * group.local_to_anchor[&image_index];
                    if replacement.try_inverse().is_none()
                        || !homography_preserves_focus_orientation(
                            &replacement,
                            images[image_index].dimensions(),
                        )
                    {
                        return None;
                    }
                    solved.insert(images[image_index].id, replacement);
                }
            }
            carry_unstationed_poses(&mut solved);
            Some(solved)
        };
        let solved =
            build_solved(&selected_group_poses).or_else(|| build_solved(&base_group_poses));
        let Some(solved) = solved else {
            println!(
                "  - Focus capture-group planar geometry produced an unstable pose; keeping image-level geometry"
            );
            return locked_homographies.clone();
        };
        println!(
            "  - Focus capture-group planar geometry seeded from {tree_edges} pixel-verified relation(s)"
        );
        *translation_geometry_verified = true;
        return solved;
    }
    if let Some(mut translation_poses) = solve_focus_group_translation_poses(
        images,
        &groups,
        &group_edges,
        locked_homographies,
        reference_group,
        coordinate_scale,
    ) {
        // The translation solve has already required a connected group graph
        // and robust closure. Do not subsequently "refine" it with every
        // long-range planar relation: repeated figures and calligraphy can
        // agree as homographies yet pull a valid scan position by hundreds of
        // pixels. Keep the solved camera positions and only retain the local
        // focus-plane transform inside each capture group.
        correct_focus_group_roll_from_long_edges(
            images,
            &groups,
            &mut translation_poses,
            locked_homographies,
            reference_group,
        );
        let mut solved = HashMap::new();
        for (group_index, group) in groups.iter().enumerate() {
            let group_pose = translation_poses[group_index];
            for &image_index in &group.members {
                let replacement = group_pose * group.local_to_anchor[&image_index];
                if replacement.try_inverse().is_none()
                    || !homography_preserves_focus_orientation(
                        &replacement,
                        images[image_index].dimensions(),
                    )
                {
                    println!(
                        "  - Focus capture-group translation geometry produced an unstable pose for '{}'",
                        images[image_index].filename
                    );
                    return locked_homographies.clone();
                }
                solved.insert(images[image_index].id, replacement);
            }
        }
        carry_unstationed_poses(&mut solved);
        println!(
            "  - Focus capture-group translation geometry accepted for {} groups without projective re-refinement",
            groups.len()
        );
        *translation_geometry_verified = true;
        return solved;
    }
    let translation_geometry_seeded = false;
    let use_consensus_group_poses = translation_geometry_seeded
        || (tree_edges + 1 == groups.len()
            && group_poses.iter().all(Option::is_some)
            && group_poses.iter().all(|pose| {
                pose.is_some_and(|pose| {
                    pose.try_inverse().is_some()
                        && homography_preserves_focus_orientation(
                            &pose,
                            images[groups[reference_group].anchor].dimensions(),
                        )
                })
            }));
    if use_consensus_group_poses {
        println!(
            "  - Focus capture-group geometry seeded from {tree_edges} multi-member consensus relation(s)"
        );
    }
    let mut constraints = Vec::<(usize, usize, [f64; 2], f64, usize)>::new();
    for &(score, left_group, right_group, left_to_right, support, median_error) in &group_edges {
        // One repeated character can produce one excellent-looking edge. A
        // group may move only when two independent member pairs agree on the
        // same anchor-to-anchor transform.
        if support < 2 {
            continue;
        }
        let left_global = if use_consensus_group_poses {
            group_poses[left_group].unwrap()
        } else {
            let Some(pose) = locked_homographies
                .get(&images[groups[left_group].anchor].id)
                .copied()
            else {
                continue;
            };
            pose
        };
        let right_global = if use_consensus_group_poses {
            group_poses[right_group].unwrap()
        } else {
            let Some(pose) = locked_homographies
                .get(&images[groups[right_group].anchor].id)
                .copied()
            else {
                continue;
            };
            pose
        };
        let dimensions = images[groups[left_group].anchor].dimensions();
        let samples = [
            Point2::new(dimensions.0 as f64 * 0.2, dimensions.1 as f64 * 0.2),
            Point2::new(dimensions.0 as f64 * 0.8, dimensions.1 as f64 * 0.2),
            Point2::new(dimensions.0 as f64 * 0.5, dimensions.1 as f64 * 0.5),
            Point2::new(dimensions.0 as f64 * 0.2, dimensions.1 as f64 * 0.8),
            Point2::new(dimensions.0 as f64 * 0.8, dimensions.1 as f64 * 0.8),
        ];
        let mut dx = Vec::new();
        let mut dy = Vec::new();
        for point in samples {
            let Some(left_world) = transformed_point(&left_global, point) else {
                continue;
            };
            let Some(right_local) = transformed_point(&left_to_right, point) else {
                continue;
            };
            let Some(right_world) = transformed_point(&right_global, right_local) else {
                continue;
            };
            let delta = right_world - left_world;
            if delta.x.is_finite() && delta.y.is_finite() {
                dx.push(delta.x);
                dy.push(delta.y);
            }
        }
        let (Some(dx), Some(dy)) = (median_value(&mut dx), median_value(&mut dy)) else {
            continue;
        };
        let weight =
            (support as f64) * score.sqrt() / (1.0 + median_error / consensus_threshold.max(1.0));
        if weight.is_finite() && weight > 0.0 {
            constraints.push((left_group, right_group, [dx, dy], weight, support));
        }
    }
    if constraints.is_empty() {
        println!(
            "  - Focus capture-group translation solve: no multi-edge constraints; keeping image-level geometry"
        );
        return locked_homographies.clone();
    }

    let closure_constraints = constraints
        .iter()
        .map(|&(left, right, observed, base_weight, _)| {
            (left, right, [-observed[0], -observed[1]], base_weight)
        })
        .collect::<Vec<_>>();
    let Some(mut closure_solve) =
        solve_robust_closure_translations(groups.len(), reference_group, &closure_constraints)
    else {
        return locked_homographies.clone();
    };
    let iterations = closure_solve.iterations;
    let corrections = &mut closure_solve.corrections;
    let initial_group_poses = groups
        .iter()
        .enumerate()
        .map(|(group_index, group)| {
            if use_consensus_group_poses {
                group_poses[group_index]
            } else {
                locked_homographies.get(&images[group.anchor].id).copied()
            }
        })
        .collect::<Option<Vec<_>>>();
    let Some(initial_group_poses) = initial_group_poses else {
        return locked_homographies.clone();
    };
    let Some((maximum_corner_correction, clamped_stations)) = clamp_station_translation_corrections(
        images,
        &groups,
        &group_edges,
        &initial_group_poses,
        corrections,
    ) else {
        return locked_homographies.clone();
    };
    if clamped_stations > 0 {
        degradation::record_run_degradation(
            degradation::CLOSURE_CORRECTION_CLAMPED,
            serde_json::json!({
                "clamped_stations": clamped_stations,
                "max_corner_correction_px": maximum_corner_correction,
            }),
        );
    }

    let mut solved = HashMap::new();
    for (group_index, group) in groups.iter().enumerate() {
        let original_group_pose = initial_group_poses[group_index];
        let correction = corrections[group_index];
        let group_pose = Matrix3::new(
            1.0,
            0.0,
            correction[0],
            0.0,
            1.0,
            correction[1],
            0.0,
            0.0,
            1.0,
        ) * original_group_pose;
        for &image_index in &group.members {
            let replacement = group_pose * group.local_to_anchor[&image_index];
            if replacement.try_inverse().is_none()
                || !homography_preserves_focus_orientation(
                    &replacement,
                    images[image_index].dimensions(),
                )
            {
                println!(
                    "  - Focus capture-group translation solve produced an unstable pose for '{}'",
                    images[image_index].filename
                );
                return locked_homographies.clone();
            }
            solved.insert(images[image_index].id, replacement);
        }
    }
    carry_unstationed_poses(&mut solved);
    println!(
        "  - Focus capture-group translation solve: {} groups, {} candidate group pairs, {} consensus pairs, {} accepted constraints; {iterations} iteration(s), maximum corner correction {:.2}px, {} clamped station(s)",
        groups.len(),
        group_edges.len(),
        consensus_pair_count,
        constraints.len(),
        maximum_corner_correction,
        clamped_stations,
    );
    solved
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
