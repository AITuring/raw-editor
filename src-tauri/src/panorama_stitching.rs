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
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Cursor;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use tauri::{AppHandle, Emitter, Runtime};

use crate::formats::is_raw_file;
use crate::image_processing::apply_cpu_default_raw_processing;
use crate::panorama_utils::registration;
use crate::panorama_utils::stitching::{Projection, project_point};
use crate::panorama_utils::{processing, stitching};

pub const BRIEF_DESCRIPTOR_SIZE: usize = 256;
pub type Descriptor = [u8; BRIEF_DESCRIPTOR_SIZE / 8];
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
// Repeated calligraphy produces convincing but wrong long-range matches. The
// global solve should close only a small temporal neighbourhood; long edges
// remain useful for order discovery, not for moving every later pose.
const FOCUS_GLOBAL_MAX_SEQUENCE_GAP: usize = 1;
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
const FOCUS_BRACKET_MIN_OVERLAP_SUPPORT: f64 = 0.60;
const FOCUS_BRACKET_MAX_CAPTURE_GAP: u64 = 6;
const FOCUS_BRACKET_MAX_COMPONENT_SOURCES: usize = 8;
const FOCUS_BRACKET_GRAPH_WEIGHT_BOOST: f64 = 24.0;
// A long scan of repeated calligraphy can contain a visually convincing edge
// between distant characters. Once two nearby capture numbers have a verified
// overlap, keep that camera-sequence evidence ahead of a remote lookalike when
// constructing the initial pose tree. Remote edges still connect separate
// capture runs, but cannot replace an already verified local backbone.
const FOCUS_CAPTURE_SEQUENCE_GRAPH_WEIGHT_BOOST: f64 = 128.0;
const FOCUS_BRACKET_OBSERVATION_WEIGHT: f64 = 16.0;
const FOCUS_GROUP_CONSENSUS_MAX_ERROR_RATIO: f64 = 0.006;
const FOCUS_GROUP_SINGLE_EDGE_WEIGHT_FACTOR: f64 = 0.02;
const FOCUS_GROUP_MAX_CENTER_CORRECTION_RATIO: f64 = 0.50;
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

#[derive(Debug, Clone, Copy)]
pub struct KeyPoint {
    pub x: u32,
    pub y: u32,
}

#[derive(Clone)]
pub struct Feature {
    pub keypoint: KeyPoint,
    pub descriptor: Descriptor,
    /// Pyramid level used to build the descriptor. The keypoint is mapped
    /// back to the input image, but the support scale is retained so a
    /// different focal module can be matched against a physically compatible
    /// BRIEF neighbourhood instead of an arbitrary level.
    pub support_scale: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Match {
    pub index1: usize,
    pub index2: usize,
}

pub struct ImageInfo {
    pub id: usize,
    pub filename: String,
    pub width: u32,
    pub height: u32,
    pub alignment_image: GrayImage,
    pub full_image: Option<Rgb32FImage>,
    pub scale_factor: f64,
    /// 35mm-equivalent focal length when available in source metadata.  This
    /// is an alignment hint; it does not alter exported metadata or pixels.
    pub focal_length_35mm: Option<f64>,
    /// A supplied wide frame used to bridge a missing focal-length row is a
    /// coverage fallback, never the preferred sharp source over a close-up.
    pub overview_reference: bool,
    pub features: Vec<Feature>,
    pub top_features: Vec<Feature>,
    pub foreground_range: Option<(f64, f64)>,
    pub foreground_mask: Option<GrayImage>,
    pub horizontal_edge_rows: Vec<f64>,
    pub vertical_edge_columns: Vec<f64>,
}

impl ImageInfo {
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

#[derive(Clone)]
pub struct MatchInfo {
    pub homography: Matrix3<f64>,
    pub inliers: usize,
    /// A filename-sequence bridge is an ordering cue, not a visual
    /// correspondence. It may connect adjacent captures after a lens switch
    /// left no literal overlap, but it must never enter the pixel-level global
    /// registration solve as if its points were measured from the artwork.
    pub sequence_bridge: bool,
    /// A metadata-scaled image-correlation candidate is an ordering cue only
    /// until its overlap passes the full-resolution validation. Its grid
    /// points are synthetic and must not pull the artwork pose optimizer.
    pub coarse_bridge: bool,
    pub points: Vec<(Point2<f64>, Point2<f64>)>,
    pub candidate_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub top_candidate_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub dense_focus_points: Vec<(Point2<f64>, Point2<f64>)>,
    pub foreground_feature_points: Vec<(Point2<f64>, Point2<f64>)>,
}

#[derive(Clone)]
pub(crate) struct FocusWarpBand {
    pub(crate) homographies: HashMap<usize, Matrix3<f64>>,
    pub(crate) source_ranges: HashMap<usize, (f64, f64)>,
    pub(crate) source_x_ranges: HashMap<usize, (f64, f64)>,
    // A repeated, content-independent long edge may safely change ownership
    // at its narrow silhouette. Generic/depth bands keep the stricter
    // foreground ownership rule so a repeated stroke cannot be reintroduced.
    pub(crate) relax_foreground_seam: bool,
    // A depth-layer correction is only valid for pixels that belong to the
    // detected layer. Keeping this bit on the band prevents a near-field
    // correction from bending the continuous paper plane beside it.
    pub(crate) foreground_only: bool,
    // Long-edge consensus is a separate, content-independent geometry cue.
    // It may be retained in a shifted mosaic to keep a physical frame/rail
    // continuous, while generic regional focus fits remain disabled there.
    pub(crate) physical_edge: bool,
}

#[derive(Clone)]
pub(crate) struct FocusLayerWarp {
    pub(crate) bands: Vec<FocusWarpBand>,
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
    pub full_canvas_width: u32,
    pub full_canvas_height: u32,
    pub render_scale: f64,
    pub ordered_paths: Vec<String>,
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

fn pairs_to_match(image_count: usize) -> Vec<(usize, usize)> {
    let neighbor_window = if image_count > SCALABLE_STACK_THRESHOLD {
        LARGE_STACK_NEIGHBOR_WINDOW
    } else {
        image_count.saturating_sub(1)
    };
    (0..image_count)
        .flat_map(|first| {
            let end = image_count.min(first.saturating_add(neighbor_window + 1));
            (first + 1..end).map(move |second| (first, second))
        })
        .collect()
}

fn all_image_pairs(image_count: usize) -> Vec<(usize, usize)> {
    (0..image_count)
        .flat_map(|first| (first + 1..image_count).map(move |second| (first, second)))
        .collect()
}

fn natural_path_cmp(left: &str, right: &str) -> Ordering {
    let left = Path::new(left)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(left)
        .as_bytes();
    let right = Path::new(right)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(right)
        .as_bytes();
    let mut left_index = 0usize;
    let mut right_index = 0usize;

    while left_index < left.len() && right_index < right.len() {
        if left[left_index].is_ascii_digit() && right[right_index].is_ascii_digit() {
            let left_start = left_index;
            let right_start = right_index;
            while left_index < left.len() && left[left_index].is_ascii_digit() {
                left_index += 1;
            }
            while right_index < right.len() && right[right_index].is_ascii_digit() {
                right_index += 1;
            }
            let left_digits = &left[left_start..left_index];
            let right_digits = &right[right_start..right_index];
            let left_trimmed = left_digits
                .iter()
                .position(|digit| *digit != b'0')
                .map_or(&left_digits[left_digits.len()..], |index| {
                    &left_digits[index..]
                });
            let right_trimmed = right_digits
                .iter()
                .position(|digit| *digit != b'0')
                .map_or(&right_digits[right_digits.len()..], |index| {
                    &right_digits[index..]
                });
            let number_order = left_trimmed
                .len()
                .cmp(&right_trimmed.len())
                .then_with(|| left_trimmed.cmp(right_trimmed))
                .then_with(|| left_digits.len().cmp(&right_digits.len()));
            if number_order != Ordering::Equal {
                return number_order;
            }
        } else {
            let byte_order = left[left_index]
                .to_ascii_lowercase()
                .cmp(&right[right_index].to_ascii_lowercase());
            if byte_order != Ordering::Equal {
                return byte_order;
            }
            left_index += 1;
            right_index += 1;
        }
    }

    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}

fn add_neighbor_pairs(
    order: &[usize],
    neighbor_window: usize,
    unique_pairs: &mut HashSet<(usize, usize)>,
) {
    for (position, &first) in order.iter().enumerate() {
        let end = order
            .len()
            .min(position.saturating_add(neighbor_window + 1));
        for &second in &order[position + 1..end] {
            unique_pairs.insert((first.min(second), first.max(second)));
        }
    }
}

fn pairs_to_match_for_images(images: &[ImageInfo], blend_mode: BlendMode) -> Vec<(usize, usize)> {
    if images.len() <= SCALABLE_STACK_THRESHOLD {
        return pairs_to_match(images.len());
    }
    // The 65–96 image range is common for high-resolution artwork scans and is
    // still small enough for an exhaustive bounded-resolution pass.  A lens
    // switch can move the only useful bridge dozens of filenames away, so a
    // neighbour-only graph is too brittle here.  Larger selections retain the
    // bounded visual-retrieval path below.
    if (SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES..=SCALE_ROBUST_EXHAUSTIVE_MAX_SOURCES)
        .contains(&images.len())
    {
        return all_image_pairs(images.len());
    }
    if blend_mode == BlendMode::FocusStack
        && images.len() <= FOCUS_AUTO_ORDER_EXHAUSTIVE_MAX_SOURCES
    {
        return all_image_pairs(images.len());
    }

    let input_order = (0..images.len()).collect::<Vec<_>>();
    let mut filename_order = input_order.clone();
    filename_order.sort_by(|&left, &right| {
        natural_path_cmp(&images[left].filename, &images[right].filename)
            .then_with(|| left.cmp(&right))
    });

    let mut unique_pairs = HashSet::new();
    add_neighbor_pairs(&input_order, LARGE_STACK_NEIGHBOR_WINDOW, &mut unique_pairs);
    add_neighbor_pairs(
        &filename_order,
        LARGE_STACK_NEIGHBOR_WINDOW,
        &mut unique_pairs,
    );
    let mut pairs = unique_pairs.into_iter().collect::<Vec<_>>();
    pairs.sort_unstable();
    augment_large_stack_candidate_pairs(images, &mut pairs);
    pairs
}

fn retrieval_feature_subset(features: &[Feature], limit: usize) -> Vec<Feature> {
    if features.len() <= limit {
        return features.to_vec();
    }
    // Feature vectors are ordered by detector strength.  Evenly sampling the
    // vector retains strong corners throughout the image instead of taking only
    // the first (often high-contrast) region.
    (0..limit)
        .map(|slot| {
            let index = slot
                .saturating_mul(features.len())
                .checked_div(limit)
                .unwrap_or(0)
                .min(features.len().saturating_sub(1));
            features[index].clone()
        })
        .collect()
}

/// Add a bounded set of visually likely pairs to the normal input/filename
/// neighbour graph.  File-picker order is often random, and a focal-length
/// switch can put the first useful bridge well outside the neighbour window.
/// The retrieval pass only counts compact descriptor matches; full-resolution
/// geometric verification still decides whether a pair is accepted.
fn augment_large_stack_candidate_pairs(images: &[ImageInfo], pairs: &mut Vec<(usize, usize)>) {
    if images.len() <= SCALABLE_STACK_THRESHOLD || images.len() < 2 {
        return;
    }
    let compact_features = images
        .iter()
        .map(|image| retrieval_feature_subset(&image.features, LARGE_STACK_RETRIEVAL_FEATURES))
        .collect::<Vec<_>>();
    if compact_features.iter().all(Vec::is_empty) {
        return;
    }

    let existing = pairs.iter().copied().collect::<HashSet<_>>();
    let scores = all_image_pairs(images.len())
        .into_par_iter()
        .filter_map(|(left, right)| {
            let score = processing::count_mutual_descriptor_matches_with_ratio(
                &compact_features[left],
                &compact_features[right],
                SCALABLE_MATCH_RATIO_THRESHOLD,
            );
            (score >= LARGE_STACK_RETRIEVAL_MIN_MATCHES).then_some((score, left, right))
        })
        .collect::<Vec<_>>();
    if scores.is_empty() {
        return;
    }

    let mut by_image = vec![Vec::<(usize, usize)>::new(); images.len()];
    for (score, left, right) in scores {
        by_image[left].push((score, right));
        by_image[right].push((score, left));
    }
    let mut additions = HashSet::new();
    for (image_index, neighbours) in by_image.iter_mut().enumerate() {
        neighbours.sort_by(|(left_score, left_index), (right_score, right_index)| {
            right_score
                .cmp(left_score)
                .then_with(|| {
                    natural_path_cmp(
                        &images[*left_index].filename,
                        &images[*right_index].filename,
                    )
                })
                .then_with(|| left_index.cmp(right_index))
        });
        for &(_, neighbour) in neighbours.iter().take(LARGE_STACK_RETRIEVAL_NEIGHBORS) {
            let pair = (image_index.min(neighbour), image_index.max(neighbour));
            if !existing.contains(&pair) {
                additions.insert(pair);
            }
        }
    }
    pairs.extend(additions);
    pairs.sort_unstable();
    pairs.dedup();
}

fn find_alignment_features(
    alignment_image: &GrayImage,
    brief_pairs: &[(nalgebra::Point2<i32>, nalgebra::Point2<i32>)],
    max_features: usize,
    scalable_stack: bool,
    focal_length_35mm: Option<f64>,
) -> Vec<Feature> {
    // A fixed-size BRIEF patch is not scale invariant.  Large phone stacks can
    // switch lenses midway through a capture (for example 35mm -> 85mm), so
    // keep descriptors from a small image pyramid on the bounded-memory path.
    // The native level remains dominant; the extra levels only provide a bridge
    // when the same detail is rendered at a different magnification.
    let mut features = if scalable_stack {
        let mut scales = vec![0.38, 0.5, std::f32::consts::FRAC_1_SQRT_2];
        // A metadata-guided level places the two camera modules on a common
        // support scale.  The generic levels remain in place for cropped or
        // non-EXIF sources.
        if let Some(focal) = focal_length_35mm.filter(|value| value.is_finite() && *value > 0.0) {
            let metadata_scale = (50.0 / focal).clamp(0.28, 2.5) as f32;
            if scales
                .iter()
                .all(|existing| (*existing - metadata_scale).abs() >= 0.04)
            {
                scales.push(metadata_scale);
            }
        }
        processing::find_features_multiscale(alignment_image, brief_pairs, max_features, &scales)
    } else {
        processing::find_features(alignment_image, brief_pairs)
    };
    if scalable_stack && features.len() < SCALABLE_LOW_TEXTURE_FEATURE_TARGET {
        let normalized = processing::normalize_grayscale(alignment_image);
        let fallback = processing::find_features_tuned(
            &normalized,
            brief_pairs,
            SCALABLE_LOW_TEXTURE_FAST_THRESHOLD,
            SCALABLE_LOW_TEXTURE_NMS_RADIUS,
        );
        if fallback.len() > features.len() {
            // Keep the pyramid descriptors even when the normalized native
            // pass finds more corners. Replacing the vector here used to erase
            // the only scale bridge on a low-texture frame—the exact case in
            // which the 35mm/85mm pair is hardest to register.
            let remaining = max_features.saturating_sub(features.len());
            features.extend(fallback.into_iter().take(remaining));
        }
    }
    features.truncate(max_features);
    features
}

fn detect_foreground_range(alignment_image: &GrayImage) -> Option<(f64, f64)> {
    let width = alignment_image.width();
    let height = alignment_image.height();
    if width < 64 || height < 128 {
        return None;
    }

    let scan_end = ((height as f64 * FOCUS_FOREGROUND_SCAN_MAX_Y).round() as u32)
        .clamp(1, height.saturating_sub(1));
    let minimum_bright_pixels = (width as f64 * FOCUS_FOREGROUND_MIN_BRIGHT_FRACTION).ceil() as u32;
    let bright_rows = (0..scan_end)
        .map(|y| {
            let bright_pixels = alignment_image
                .rows()
                .nth(y as usize)
                .into_iter()
                .flatten()
                .filter(|pixel| pixel[0] >= FOCUS_FOREGROUND_LUMA_THRESHOLD)
                .count() as u32;
            bright_pixels >= minimum_bright_pixels
        })
        .collect::<Vec<_>>();

    // This is only a conservative seed for an optional depth mask. The main
    // focus registration does not depend on it and samples the complete frame.
    let max_gap = ((height as f64 * 0.012).round() as usize).max(2);
    let mut first = None;
    let mut last = None;
    let mut gap = 0usize;
    for (row, is_bright) in bright_rows.iter().copied().enumerate() {
        if is_bright {
            if first.is_none() {
                first = Some(row);
            }
            last = Some(row);
            gap = 0;
        } else if first.is_some() {
            gap += 1;
            if gap > max_gap {
                break;
            }
        }
    }
    let (first, last) = first.zip(last)?;
    let run_height = last.saturating_sub(first) + 1;
    if (run_height as f64) < height as f64 * FOCUS_FOREGROUND_MIN_HEIGHT_RATIO {
        return None;
    }

    let padding = (height as f64 * 0.012).max(3.0);
    let minimum = ((first as f64 - padding) / height as f64).clamp(0.0, 1.0);
    let maximum = (((last + 1) as f64 + padding) / height as f64).clamp(0.0, 1.0);
    Some((minimum, maximum))
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
    global_homographies: &HashMap<usize, Matrix3<f64>>,
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
                image.foreground_range,
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

fn dilate_binary_mask(
    mask: &GrayImage,
    horizontal_radius: usize,
    vertical_radius: usize,
) -> GrayImage {
    let (width, height) = mask.dimensions();
    if width == 0 || height == 0 {
        return mask.clone();
    }
    let width = width as usize;
    let height = height as usize;
    let source = mask.as_raw();
    let mut horizontal = vec![0u8; width * height];
    let mut output = vec![0u8; width * height];

    for y in 0..height {
        let row_start = y * width;
        let mut prefix = vec![0u32; width + 1];
        for x in 0..width {
            prefix[x + 1] = prefix[x] + u32::from(source[row_start + x] > 0);
        }
        for x in 0..width {
            let start = x.saturating_sub(horizontal_radius);
            let end = (x + horizontal_radius).min(width - 1);
            if prefix[end + 1] > prefix[start] {
                horizontal[row_start + x] = 255;
            }
        }
    }

    for x in 0..width {
        let mut prefix = vec![0u32; height + 1];
        for y in 0..height {
            prefix[y + 1] = prefix[y] + u32::from(horizontal[y * width + x] > 0);
        }
        for y in 0..height {
            let start = y.saturating_sub(vertical_radius);
            let end = (y + vertical_radius).min(height - 1);
            if prefix[end + 1] > prefix[start] {
                output[y * width + x] = 255;
            }
        }
    }

    GrayImage::from_raw(width as u32, height as u32, output)
        .expect("foreground mask dimensions must match")
}

fn build_foreground_mask(
    alignment_image: &GrayImage,
    foreground_range: Option<(f64, f64)>,
) -> Option<GrayImage> {
    let (minimum_y, maximum_y) = foreground_range?;
    let (width, height) = alignment_image.dimensions();
    if width == 0 || height == 0 || minimum_y >= maximum_y {
        return None;
    }
    let base = GrayImage::from_fn(width, height, |x, y| {
        let normalized_y = y as f64 / height.max(1) as f64;
        let value = alignment_image.get_pixel(x, y)[0];
        image::Luma([
            if (minimum_y..=maximum_y).contains(&normalized_y)
                && value >= FOCUS_FOREGROUND_LUMA_THRESHOLD
            {
                255
            } else {
                0
            },
        ])
    });
    // Include the narrow shadow and metallic lip around the bright core so
    // ownership remains stable at its silhouette. This is a mask-space margin;
    // it does not alter the generic registration path.
    let horizontal_radius = ((width as f64 * 0.012).round() as usize).clamp(2, 32);
    let vertical_radius = ((height as f64 * 0.025).round() as usize).clamp(3, 48);
    Some(dilate_binary_mask(
        &base,
        horizontal_radius,
        vertical_radius,
    ))
}

fn focus_alignment_keypoint_is_foreground(image: &ImageInfo, keypoint: KeyPoint) -> bool {
    let Some((minimum_y, maximum_y)) = image.foreground_range else {
        return false;
    };
    let (width, height) = image.alignment_image.dimensions();
    if keypoint.x >= width || keypoint.y >= height {
        return false;
    }
    let normalized_y = keypoint.y as f64 / height.max(1) as f64;
    if !(minimum_y..=maximum_y).contains(&normalized_y) {
        return false;
    }
    image.foreground_mask.as_ref().map_or_else(
        || {
            image.alignment_image.get_pixel(keypoint.x, keypoint.y)[0]
                >= FOCUS_FOREGROUND_LUMA_THRESHOLD
        },
        |mask| mask.get_pixel(keypoint.x, keypoint.y)[0] > 0,
    )
}

fn find_top_alignment_features(
    alignment_image: &GrayImage,
    brief_pairs: &[(nalgebra::Point2<i32>, nalgebra::Point2<i32>)],
    foreground_range: Option<(f64, f64)>,
) -> Vec<Feature> {
    let Some((minimum_y, maximum_y)) = foreground_range else {
        return Vec::new();
    };
    if alignment_image.height() < 128 || minimum_y >= maximum_y {
        return Vec::new();
    }
    let top_start = (alignment_image.height() as f64 * minimum_y).round() as u32;
    let top_end = (alignment_image.height() as f64 * maximum_y)
        .round()
        .max((top_start + 64) as f64)
        .min(alignment_image.height() as f64) as u32;
    let top_region = image::imageops::crop_imm(
        alignment_image,
        0,
        top_start,
        alignment_image.width(),
        top_end.saturating_sub(top_start),
    )
    .to_image();
    let mut features = processing::find_features_tuned(
        &top_region,
        brief_pairs,
        SCALABLE_LOW_TEXTURE_FAST_THRESHOLD,
        (SCALABLE_LOW_TEXTURE_NMS_RADIUS * 0.75).max(6.0),
    );
    let normalized_region = processing::normalize_grayscale(&top_region);
    features.extend(processing::find_features_tuned(
        &normalized_region,
        brief_pairs,
        5,
        8.0,
    ));
    for feature in &mut features {
        feature.keypoint.y = feature.keypoint.y.saturating_add(top_start);
    }
    features.retain(|feature| {
        let y = feature.keypoint.y as f64 / alignment_image.height().max(1) as f64;
        (minimum_y..=maximum_y).contains(&y)
            && alignment_image
                .get_pixel(feature.keypoint.x, feature.keypoint.y)
                .0[0]
                >= FOCUS_FOREGROUND_LUMA_THRESHOLD
    });
    features
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
    focal_length_35mm: Option<f64>,
}

fn load_prepared_stack_source(
    filename: &str,
    settings: &AppSettings,
) -> Result<PreparedStackSource, String> {
    let file_data = read_file_mapped(Path::new(filename))
        .map_err(|error| format!("Failed to read image {filename}: {error}"))?;
    let focal_length_35mm = crate::exif_processing::focal_length_35mm_from_bytes(&file_data);
    let mut image = crate::image_loader::load_base_image_from_bytes(
        &file_data, filename, false, settings, None,
    )
    .map_err(|error| format!("Failed to load image {filename}: {error}"))?;
    if is_raw_file(filename) {
        apply_cpu_default_raw_processing(&mut image);
    }
    Ok(PreparedStackSource {
        image,
        focal_length_35mm,
    })
}

fn prepare_focus_rescue_image(
    image: &ImageInfo,
    settings: &AppSettings,
    brief_pairs: &[(nalgebra::Point2<i32>, nalgebra::Point2<i32>)],
) -> Result<ImageInfo, String> {
    let prepared = load_prepared_stack_source(&image.filename, settings)?;
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
    if focus_match_is_local_bracket(source, target, match_info) {
        return FOCUS_CAPTURE_SEQUENCE_GRAPH_WEIGHT_BOOST;
    }
    let capture_gap = trailing_capture_number(&source.filename)
        .zip(trailing_capture_number(&target.filename))
        .map(|(left, right)| left.abs_diff(right));
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

fn canonical_match_direction(
    images: &[ImageInfo],
    first: usize,
    second: usize,
) -> (usize, usize, bool) {
    if images[first].filename <= images[second].filename {
        (first, second, false)
    } else {
        (second, first, true)
    }
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
    let mixed_focal_pair = image_pair_has_mixed_focal_lengths(source_image, target_image);
    let features1 = &source_image.features;
    let features2 = &target_image.features;
    let minimum_inliers = if stable_four_point_solver {
        SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION
    } else {
        processing::MIN_INLIERS_FOR_CONNECTION
    };
    let initial_matches = if stable_four_point_solver {
        let ratio_threshold = if mixed_focal_pair {
            MIXED_FOCAL_MATCH_RATIO_THRESHOLD
        } else {
            SCALABLE_MATCH_RATIO_THRESHOLD
        };
        let mut matches = if mixed_focal_pair {
            match (
                source_image.focal_length_35mm,
                target_image.focal_length_35mm,
            ) {
                (Some(source_focal), Some(target_focal)) if source_focal > 0.0 => {
                    processing::match_features_with_ratio_and_scale(
                        features1,
                        features2,
                        MIXED_FOCAL_EMERGENCY_MATCH_RATIO_THRESHOLD,
                        (source_focal / target_focal) as f32,
                        1.65,
                    )
                }
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        if matches.len() < minimum_inliers {
            matches = processing::match_features_with_ratio(features1, features2, ratio_threshold);
        }
        if mixed_focal_pair && matches.len() < minimum_inliers {
            matches = processing::match_features_with_ratio(
                features1,
                features2,
                MIXED_FOCAL_EMERGENCY_MATCH_RATIO_THRESHOLD,
            );
            if log_match {
                println!(
                    "  - Mixed-focal descriptor fallback produced {} candidates",
                    matches.len()
                );
            }
        }
        matches
    } else {
        processing::match_features(features1, features2)
    };
    let adjacent_same_focal_focus_pair = blend_mode == BlendMode::FocusStack
        && focus_adjacent_same_focal_pair(source_image, target_image);
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
        return Some(MatchInfo {
            homography,
            inliers: points.len(),
            sequence_bridge: false,
            coarse_bridge: true,
            points: points.clone(),
            candidate_points: points,
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
        });
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
            return Some(MatchInfo {
                homography,
                inliers: points.len(),
                sequence_bridge: false,
                coarse_bridge: true,
                points: points.clone(),
                candidate_points: points,
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
            });
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

    let keypoints1 = features1
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let keypoints2 = features2
        .iter()
        .map(|feature| feature.keypoint)
        .collect::<Vec<_>>();
    let projected_points1 = keypoints1
        .iter()
        .map(|point| {
            project_point(
                source_image,
                point.x as f64 * source_image.scale_factor,
                point.y as f64 * source_image.scale_factor,
                projection,
            )
            .expect("projection should produce finite feature coordinates")
        })
        .collect::<Vec<_>>();
    let projected_points2 = keypoints2
        .iter()
        .map(|point| {
            project_point(
                target_image,
                point.x as f64 * target_image.scale_factor,
                point.y as f64 * target_image.scale_factor,
                projection,
            )
            .expect("projection should produce finite feature coordinates")
        })
        .collect::<Vec<_>>();
    let projected_match_points = initial_matches
        .iter()
        .map(|matched| {
            (
                projected_points1[matched.index1],
                projected_points2[matched.index2],
            )
        })
        .collect::<Vec<_>>();
    let background_match_indices = if blend_mode == BlendMode::FocusStack
        && (source_image.foreground_range.is_some() || target_image.foreground_range.is_some())
    {
        initial_matches
            .iter()
            .enumerate()
            .filter_map(|(index, matched)| {
                (!focus_alignment_keypoint_is_foreground(source_image, keypoints1[matched.index1])
                    && !focus_alignment_keypoint_is_foreground(
                        target_image,
                        keypoints2[matched.index2],
                    ))
                .then_some(index)
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    // A depth-discontinuous object can dominate a local descriptor match (the
    // bright holder in this fixture is one example). Solve the global camera
    // pose from the paper/background whenever there is enough background
    // support, then let the optional regional model handle the object itself.
    // If a pair contains too little background, retain the old all-point solver
    // so unusual stacks do not become disconnected merely because an object
    // detector fired.
    let solver_indices = if background_match_indices.len() >= minimum_inliers {
        background_match_indices
    } else {
        (0..projected_match_points.len()).collect::<Vec<_>>()
    };
    let solver_points = solver_indices
        .iter()
        .map(|&index| projected_match_points[index])
        .collect::<Vec<_>>();
    let top_matches =
        processing::match_features(&source_image.top_features, &target_image.top_features);
    let top_candidate_points = top_matches
        .iter()
        .filter_map(|matched| {
            let source = source_image.top_features.get(matched.index1)?.keypoint;
            let target = target_image.top_features.get(matched.index2)?.keypoint;
            Some((
                project_point(
                    source_image,
                    source.x as f64 * source_image.scale_factor,
                    source.y as f64 * source_image.scale_factor,
                    projection,
                )?,
                project_point(
                    target_image,
                    target.x as f64 * target_image.scale_factor,
                    target.y as f64 * target_image.scale_factor,
                    projection,
                )?,
            ))
        })
        .collect::<Vec<_>>();
    // The detector works in analysis pixels. The 12px native-image gate is
    // smaller than two quantized detector pixels on 200MP phone photographs.
    let seed_threshold = FULL_RES_RANSAC_INLIER_THRESHOLD
        .max(source_image.scale_factor.max(target_image.scale_factor) * 3.5);
    let mut solver_indices_used = solver_indices.clone();
    let ransac_result = if stable_four_point_solver {
        processing::find_homography_ransac_points_stable_with_min_inliers(
            &solver_points,
            seed_threshold,
            minimum_inliers,
        )
    } else {
        processing::find_homography_ransac_points(&solver_points, seed_threshold)
    };
    let mut mixed_focal_fallback_used = false;
    let ransac_result = ransac_result.or_else(|| {
        if !(mixed_focal_pair && projection == Projection::Planar) {
            return None;
        }
        let (Some(source_focal), Some(target_focal)) = (
            source_image.focal_length_35mm,
            target_image.focal_length_35mm,
        ) else {
            return None;
        };
        let fallback = find_mixed_focal_similarity_ransac(
            &solver_points,
            source_focal,
            target_focal,
            MIXED_FOCAL_SIMILARITY_RANSAC_THRESHOLD,
            minimum_inliers,
        )
        .or_else(|| {
            // Foreground masking is normally valuable for focus stacks, but a
            // lens switch can leave the paper/object boundary outside one
            // mask. Retry the complete candidate set before declaring the
            // focal-length component disconnected.
            if solver_indices.len() == projected_match_points.len() {
                return None;
            }
            solver_indices_used = (0..projected_match_points.len()).collect();
            find_mixed_focal_similarity_ransac(
                &projected_match_points,
                source_focal,
                target_focal,
                MIXED_FOCAL_SIMILARITY_RANSAC_THRESHOLD,
                minimum_inliers,
            )
        });
        if log_match && fallback.is_some() {
            mixed_focal_fallback_used = true;
            println!(
                "  - Mixed-focal similarity fallback recovered a geometric bridge ({} inliers)",
                fallback.as_ref().map_or(0, |(_, indices)| indices.len())
            );
        }
        fallback
    });
    let Some((projected_homography, projected_inlier_indices)) = ransac_result else {
        if mixed_focal_pair
            && projection == Projection::Planar
            && let (Some(source_focal), Some(target_focal)) = (
                source_image.focal_length_35mm,
                target_image.focal_length_35mm,
            )
            && let Some((coarse_homography, coarse_points, coarse_score)) =
                estimate_mixed_focal_coarse_registration(
                    source_image,
                    target_image,
                    source_focal,
                    target_focal,
                    false,
                    log_match,
                )
        {
            if log_match {
                println!(
                    "  - Mixed-focal coarse image registration recovered {} points (NCC {:.3})",
                    coarse_points.len(),
                    coarse_score
                );
            }
            return Some(MatchInfo {
                homography: coarse_homography,
                inliers: coarse_points.len(),
                sequence_bridge: false,
                coarse_bridge: true,
                points: coarse_points.clone(),
                candidate_points: coarse_points,
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
            });
        }
        if log_match && mixed_focal_pair {
            println!(
                "  - Rejecting mixed-focal pair after RANSAC: {} descriptor candidates",
                initial_matches.len()
            );
        }
        return None;
    };
    if mixed_focal_pair
        && projection == Projection::Planar
        && let (Some(source_focal), Some(target_focal)) = (
            source_image.focal_length_35mm,
            target_image.focal_length_35mm,
        )
        && !mixed_focal_scale_is_plausible(&projected_homography, source_focal, target_focal)
    {
        return None;
    }
    let projected_inlier_indices = projected_inlier_indices
        .into_iter()
        .map(|index| solver_indices_used[index])
        .collect::<Vec<_>>();
    if mixed_focal_pair
        && projection == Projection::Planar
        && (source_image.foreground_range.is_some() || target_image.foreground_range.is_some())
    {
        let foreground_support = projected_inlier_indices
            .iter()
            .filter(|&&index| {
                let matched = initial_matches[index];
                focus_alignment_keypoint_is_foreground(source_image, keypoints1[matched.index1])
                    || focus_alignment_keypoint_is_foreground(
                        target_image,
                        keypoints2[matched.index2],
                    )
            })
            .count();
        if log_match {
            println!(
                "  - Mixed-focal foreground support: {foreground_support}/{} inliers",
                projected_inlier_indices.len()
            );
        }
        // A scale-correct homography supported only by a repeated wall weave
        // is still a false bridge between independent phone captures. When
        // foreground masks exist, require a few identity-bearing points too.
        if foreground_support < minimum_inliers.min(8) {
            let descriptor_pose_verified = focus_overlap_quality(
                source_image,
                target_image,
                &projected_homography,
            )
            .is_some_and(|(intensity_ncc, edge_ncc, edge_orientation, samples)| {
                if log_match {
                    println!(
                        "  - Rail-supported mixed-focal pose validation: intensity NCC {:.3}, edge NCC {:.3}, edge orientation {:.3} ({} samples)",
                        intensity_ncc, edge_ncc, edge_orientation, samples
                    );
                }
                intensity_ncc >= MIXED_FOCAL_COARSE_MIN_INTENSITY_NCC
                    && edge_ncc >= MIXED_FOCAL_COARSE_MIN_EDGE_NCC
                    && edge_orientation >= MIXED_FOCAL_COARSE_MIN_EDGE_ORIENTATION
            });
            if !descriptor_pose_verified {
                // The descriptor consensus may have landed entirely on the
                // display rail even when a genuine cross-lens artwork overlap
                // is present elsewhere. Give the denser image-level validator
                // one chance; it will reject a repeated rail-only pose by its
                // full-overlap structure score.
                if let (Some(source_focal), Some(target_focal)) = (
                    source_image.focal_length_35mm,
                    target_image.focal_length_35mm,
                ) && let Some((coarse_homography, coarse_points, coarse_score)) =
                    estimate_mixed_focal_coarse_registration(
                        source_image,
                        target_image,
                        source_focal,
                        target_focal,
                        false,
                        log_match,
                    )
                {
                    if log_match {
                        println!(
                            "  - Mixed-focal coarse image registration recovered {} points (NCC {:.3})",
                            coarse_points.len(),
                            coarse_score
                        );
                    }
                    return Some(MatchInfo {
                        homography: coarse_homography,
                        inliers: coarse_points.len(),
                        sequence_bridge: false,
                        coarse_bridge: true,
                        points: coarse_points.clone(),
                        candidate_points: coarse_points,
                        top_candidate_points: Vec::new(),
                        dense_focus_points: Vec::new(),
                        foreground_feature_points: Vec::new(),
                    });
                }
                return None;
            }
        }
    }
    let (dense_focus_points, foreground_feature_points) = if blend_mode == BlendMode::FocusStack {
        let dense_focus_points = collect_dense_focus_region_points(
            source_image,
            target_image,
            &projected_homography,
            projection,
        );
        let mut foreground_feature_points = collect_foreground_feature_points(
            source_image,
            target_image,
            &projected_homography,
            projection,
        );
        // Descriptor matches are sparse but carry a stronger identity signal
        // than a patch search on a long edge. Keep them alongside the strictly
        // corner-filtered matches; the regional residual/RANSAC checks below
        // decide whether a point is geometrically usable.
        foreground_feature_points.extend(top_candidate_points.iter().copied());
        (dense_focus_points, foreground_feature_points)
    } else {
        (Vec::new(), Vec::new())
    };
    let mut inlier_points = projected_inlier_indices
        .iter()
        .map(|&index| {
            let matched = initial_matches[index];
            if mixed_focal_pair && projection == Projection::Planar {
                // Refine the complete patch under the scale-aware model below.
                // Returning the descriptor point here is only a fallback for a
                // low-texture patch; a fixed-size native NCC patch compares the
                // wrong physical support after a 35mm/85mm module switch.
                projected_match_points[index]
            } else {
                refine_match_point_from_homography(
                    source_image,
                    target_image,
                    keypoints1[matched.index1],
                    keypoints2[matched.index2],
                    projection,
                    &projected_homography,
                    false,
                    blend_mode == BlendMode::FocusStack,
                )
                .unwrap_or(projected_match_points[index])
            }
        })
        .collect::<Vec<_>>();
    if mixed_focal_pair && projection == Projection::Planar {
        // The descriptors live in the reduced alignment images while the RANSAC
        // model above lives in source pixels. Convert the model back into the
        // descriptor coordinate system, warp every patch sample, then convert
        // the refined target observations to source pixels. This removes the
        // several-pixel quantisation error that otherwise becomes a visible
        // soft edge at 85mm/35mm overlaps.
        let source_scale = source_image.scale_factor.max(1e-9);
        let target_scale = target_image.scale_factor.max(1e-9);
        let to_target_alignment = Matrix3::new(
            target_scale.recip(),
            0.0,
            0.0,
            0.0,
            target_scale.recip(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let from_source_alignment = Matrix3::new(
            source_scale,
            0.0,
            0.0,
            0.0,
            source_scale,
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let alignment_homography =
            to_target_alignment * projected_homography * from_source_alignment;
        let mut refined_points = Vec::new();
        for &index in &projected_inlier_indices {
            let matched = initial_matches[index];
            let center = Point2::new(
                keypoints1[matched.index1].x as f64,
                keypoints1[matched.index1].y as f64,
            );
            if let Some(refined) = registration::refine_warped_patch(
                &source_image.alignment_image,
                &target_image.alignment_image,
                &alignment_homography,
                center,
                7,
                4,
            ) {
                let source = projected_match_points[index].0;
                let target = Point2::new(
                    refined.target.x * target_scale,
                    refined.target.y * target_scale,
                );
                if target.x.is_finite() && target.y.is_finite() {
                    refined_points.push((source, target));
                }
            }
        }
        if refined_points.len() >= minimum_inliers
            && match_has_spatial_support(
                &refined_points,
                source_image.dimensions(),
                target_image.dimensions(),
            )
        {
            println!(
                "  - Cross-scale warped-patch refinement retained {}/{} observations",
                refined_points.len(),
                projected_inlier_indices.len()
            );
            inlier_points = refined_points;
        }
    }
    let model_refinement_threshold = if blend_mode == BlendMode::FocusStack {
        FOCUS_MODEL_INLIER_THRESHOLD
    } else {
        FULL_RES_REFINEMENT_THRESHOLD
    };
    let alignment_quantization = source_image.scale_factor.max(target_image.scale_factor) * 1.5;
    let refinement_threshold = if source_image.full_image.is_some()
        && target_image.full_image.is_some()
        && !mixed_focal_pair
    {
        model_refinement_threshold
    } else {
        model_refinement_threshold
            .max(alignment_quantization)
            .max(if mixed_focal_pair {
                PANORAMA_MODEL_INLIER_THRESHOLD
            } else {
                0.0
            })
    };
    let refined_homography = if mixed_focal_fallback_used {
        // The two-point scale-constrained fallback has already performed its
        // own robust inlier/refit loop. Re-running the four-point homography
        // refit on only a dozen cross-scale observations can reject the valid
        // bridge because the sparse close-up has different lens distortion.
        projected_homography
    } else {
        refine_homography_inliers(&mut inlier_points, refinement_threshold, minimum_inliers)?
    };
    if stable_four_point_solver
        && !match_has_spatial_support(
            &inlier_points,
            source_image.dimensions(),
            target_image.dimensions(),
        )
    {
        // Descriptor consensus concentrated on one repeated face/stroke is not
        // enough to place a large tile. Reject it before it can become a strong
        // graph edge and pull an otherwise coherent mosaic into a false overlap.
        return None;
    }
    let homography = if alignment_mode == AlignmentMode::Position {
        estimate_translation(&inlier_points)
    } else if mixed_focal_fallback_used {
        // The fallback is already a scale-constrained similarity fit. Running
        // the generic focus model selector on a sparse lens-switch bridge can
        // turn its valid scale into an arbitrary affine shear, so preserve the
        // model that passed the focal-length plausibility gate.
        projected_homography
    } else if blend_mode == BlendMode::FocusStack {
        select_focus_stack_transform(
            &refined_homography,
            &inlier_points,
            source_image.dimensions(),
            alignment_mode,
        )
    } else if stable_four_point_solver && alignment_mode == AlignmentMode::Auto {
        select_large_panorama_transform(
            &refined_homography,
            &inlier_points,
            source_image.dimensions(),
            log_match,
        )
    } else {
        refined_homography
    };
    if blend_mode == BlendMode::FocusStack
        && !mixed_focal_fallback_used
        && !retain_model_inliers(
            &mut inlier_points,
            &homography,
            FOCUS_MODEL_INLIER_THRESHOLD,
            minimum_inliers,
        )
    {
        // The projective RANSAC fit is only an intermediate consensus. A
        // lower-DOF focus-stack model must get its own residual-consistent
        // observations; otherwise the global pose solve receives points that
        // support a different warp and can introduce a soft/double edge.
        if log_match {
            println!(
                "  - Rejecting focus match after selected-model validation: fewer than {} consistent observations",
                minimum_inliers
            );
        }
        return None;
    }
    let inlier_count = inlier_points.len();
    let median_error = median_symmetric_error(&homography, &inlier_points);
    let focus_overlap_quality = (blend_mode == BlendMode::FocusStack && stable_four_point_solver)
        .then(|| focus_overlap_quality(source_image, target_image, &homography))
        .flatten();
    let focus_overlap_verified = focus_overlap_quality.is_some_and(|quality| {
        focus_overlap_is_verified(
            quality,
            inlier_count,
            median_error,
            allow_low_texture_boundary,
        )
    });
    if log_match
        && let Some((intensity_ncc, edge_ncc, edge_orientation, samples)) = focus_overlap_quality
    {
        println!(
            "  - Focus overlap validation: intensity NCC {:.3}, edge NCC {:.3}, edge orientation {:.3} ({} samples, {})",
            intensity_ncc,
            edge_ncc,
            edge_orientation,
            samples,
            if focus_overlap_verified {
                "verified"
            } else {
                "rejected"
            }
        );
    }
    if mixed_focal_pair
        && blend_mode == BlendMode::FocusStack
        && (inlier_count < MIXED_FOCAL_MIN_INLIERS
            || !median_error.is_finite()
            || median_error > MIXED_FOCAL_MAX_MEDIAN_ERROR
            || (stable_four_point_solver && !focus_overlap_verified))
    {
        if mixed_focal_fallback_used && focus_overlap_verified {
            // Detector coordinates are heavily quantized when a small
            // overview is matched to a long-lens frame. The focal-constrained
            // similarity can therefore miss the full-resolution point-error
            // cutoff while the actual warped luminance and edge structure are
            // clearly the same region. Preserve that image-verified pose as a
            // graph-only bridge; its noisy points must not enter bundle
            // adjustment.
            if log_match {
                println!(
                    "  - Accepting image-verified mixed-focal graph bridge despite {:.3}px quantized point error",
                    median_error
                );
            }
            return Some(MatchInfo {
                homography,
                inliers: inlier_count,
                sequence_bridge: false,
                coarse_bridge: true,
                points: inlier_points.clone(),
                candidate_points: projected_match_points,
                top_candidate_points,
                dense_focus_points,
                foreground_feature_points,
            });
        }
        if let (Some(source_focal), Some(target_focal)) = (
            source_image.focal_length_35mm,
            target_image.focal_length_35mm,
        ) && let Some((coarse_homography, coarse_points, coarse_score)) =
            estimate_mixed_focal_coarse_registration(
                source_image,
                target_image,
                source_focal,
                target_focal,
                false,
                log_match,
            )
        {
            if log_match {
                println!(
                    "  - Mixed-focal coarse image registration recovered {} points (NCC {:.3})",
                    coarse_points.len(),
                    coarse_score
                );
            }
            return Some(MatchInfo {
                homography: coarse_homography,
                inliers: coarse_points.len(),
                sequence_bridge: false,
                coarse_bridge: true,
                points: coarse_points.clone(),
                candidate_points: coarse_points,
                top_candidate_points: Vec::new(),
                dense_focus_points: Vec::new(),
                foreground_feature_points: Vec::new(),
            });
        }
        if log_match {
            println!(
                "  - Rejecting weak mixed-focal bridge: {} inliers, {:.3}px median symmetric error (need at least {} and <= {:.2}px)",
                inlier_count, median_error, MIXED_FOCAL_MIN_INLIERS, MIXED_FOCAL_MAX_MEDIAN_ERROR
            );
        }
        return None;
    }
    if blend_mode == BlendMode::FocusStack && stable_four_point_solver && !focus_overlap_verified {
        if focus_adjacent_same_focal_pair(source_image, target_image)
            && let Some(match_info) =
                recover_adjacent_focus_bracket_match(source_image, target_image, log_match)
        {
            if log_match {
                println!(
                    "  - Focus overlap check failed; recovered same-pose adjacent bracket instead"
                );
            }
            return Some(match_info);
        }
        if log_match {
            println!(
                "  - Rejecting focus match: the proposed overlap does not preserve the source structure"
            );
        }
        return None;
    }
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
    Some(MatchInfo {
        homography,
        inliers: inlier_count,
        sequence_bridge: false,
        coarse_bridge: false,
        points: inlier_points,
        candidate_points: projected_match_points,
        top_candidate_points,
        dense_focus_points,
        foreground_feature_points,
    })
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
    for (&(left, right), match_info) in matches {
        if left >= images.len() || right >= images.len() {
            continue;
        }
        if focus_match_is_local_bracket(&images[left], &images[right], match_info) {
            dsu.union(left, right);
        }
    }
    let mut by_root = HashMap::<usize, Vec<usize>>::new();
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
        left_number.cmp(&right_number)
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
        return match_info.points.len() >= FOCUS_MODEL_MIN_INLIERS;
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
        || valid_edges.saturating_mul(100) < total_edges.saturating_mul(95)
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
    min_scale >= 0.82 && max_scale <= 1.18 && median_value(&mut scales).is_some()
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

fn focus_stack_adjacent_match_is_unreliable(
    images: &[ImageInfo],
    left: usize,
    right: usize,
    match_info: &MatchInfo,
) -> bool {
    if match_info.sequence_bridge {
        return false;
    }
    let Some(gap) = trailing_capture_number(&images[left].filename)
        .zip(trailing_capture_number(&images[right].filename))
        .map(|(left_number, right_number)| left_number.abs_diff(right_number))
    else {
        return false;
    };
    if gap == 0 || gap > FOCUS_BRACKET_MAX_CAPTURE_GAP {
        return false;
    }
    if focus_match_is_local_bracket(&images[left], &images[right], match_info) {
        return false;
    }
    if match_info.coarse_bridge || match_info.inliers < SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION {
        return true;
    }
    let Some(motion) = focus_match_center_motion_ratio(&images[left], &images[right], match_info)
    else {
        return true;
    };
    if gap <= 2 && motion <= 0.10 {
        return match_info.inliers < FOCUS_MODEL_MIN_INLIERS;
    }
    if motion <= 0.08 {
        return false;
    }
    focus_overlap_quality(&images[left], &images[right], &match_info.homography).is_none_or(
        |(intensity_ncc, edge_ncc, edge_orientation, samples)| {
            intensity_ncc < 0.72 || edge_ncc < 0.20 || edge_orientation < 0.12 || samples < 400
        },
    )
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

fn bridge_focus_capture_station_chain(
    images: &[ImageInfo],
    matches: &mut HashMap<(usize, usize), MatchInfo>,
) -> usize {
    let stations = focus_capture_stations(images, matches);
    if stations.len() < 2 {
        return 0;
    }
    let mut added = 0usize;
    for window in stations.windows(2) {
        let left_rep = focus_station_representative(&window[0], images);
        let right_rep = focus_station_representative(&window[1], images);
        let key = (left_rep.min(right_rep), left_rep.max(right_rep));
        if matches.contains_key(&key) {
            continue;
        }
        let components = focus_overlap_components(images, matches);
        if focus_indices_share_component(left_rep, right_rep, &components) {
            continue;
        }
        if let Some(recovered) =
            insert_recovered_focus_match(images, left_rep, right_rep, false, false)
        {
            matches.insert(recovered.0, recovered.1);
            added += 1;
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
    })
}

fn focus_overlap_components(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Vec<Vec<usize>> {
    let mut dsu = Dsu::new(images.len());
    for &(left, right) in matches.keys() {
        if left < images.len() && right < images.len() {
            dsu.union(left, right);
        }
    }
    let mut by_root = HashMap::<usize, Vec<usize>>::new();
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
    // Include both filename boundaries.  Applying `take(4)` after chaining
    // the iterators only ever selected the first four members, so the tail of
    // a long scan component (for example DSC_3720 before DSC_3721) could never
    // participate in component recovery unless it also happened to be among
    // the six sharpest frames.
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
fn focus_overlap_quality(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
) -> Option<(f64, f64, f64, usize)> {
    let inverse = source_to_target.try_inverse()?;
    let target_width = target.alignment_image.width();
    let target_height = target.alignment_image.height();
    if target_width < 32 || target_height < 32 {
        return None;
    }
    let columns = 64u32;
    let rows = 40u32;
    let mut source_values = Vec::new();
    let mut target_values = Vec::new();
    let mut source_edges = Vec::new();
    let mut target_edges = Vec::new();
    let mut edge_orientations = Vec::new();
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
            let source_dx = source_dx_right - source_dx_left;
            let source_dy = source_dy_down - source_dy_up;
            let target_dx = target_dx_right - target_dx_left;
            let target_dy = target_dy_down - target_dy_up;
            let source_edge = source_dx.hypot(source_dy);
            let target_edge = target_dx.hypot(target_dy);
            if source_edge.is_finite() && target_edge.is_finite() {
                source_edges.push(source_edge);
                target_edges.push(target_edge);
                if source_edge > 4.0 && target_edge > 4.0 {
                    edge_orientations.push(
                        ((source_dx * target_dx + source_dy * target_dy)
                            / (source_edge * target_edge))
                            .clamp(-1.0, 1.0),
                    );
                }
            }
            source_values.push(source_value);
            target_values.push(target_value);
        }
    }
    if source_values.len() < 120 || source_edges.len() != source_values.len() {
        return None;
    }
    let ncc = normalized_correlation(&source_values, &target_values)?;
    let edge_ncc = normalized_correlation(&source_edges, &target_edges)?;
    let orientation = if edge_orientations.is_empty() {
        0.0
    } else {
        edge_orientations.iter().sum::<f64>() / edge_orientations.len() as f64
    };
    Some((ncc, edge_ncc, orientation, source_values.len()))
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

    let mut rng = StdRng::seed_from_u64(
        0xB4D4_5EED_5CA1_E001u64 ^ (points.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93),
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

pub(crate) fn stitch_images_with_options<R: Runtime>(
    image_paths: Vec<String>,
    app_handle: AppHandle<R>,
    alignment_mode: AlignmentMode,
    blend_mode: BlendMode,
    progress_event: &str,
) -> Result<StitchOutcome, String> {
    let image_paths = image_paths
        .into_iter()
        .filter(|path| !is_generated_stitch_output(path) && !is_auxiliary_stitch_file(path))
        .collect::<Vec<_>>();
    if image_paths.len() < 2 {
        return Err("At least two images are required for a panorama.".to_string());
    }
    if image_paths.len() > MAX_STITCH_SOURCE_IMAGES {
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
                println!("  - Processing '{}'", filename);
                let prepared_source = load_prepared_stack_source(filename, &settings)?;
                let dynamic_image = prepared_source.image;
                let focal_length_35mm = prepared_source.focal_length_35mm;
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

    let mut image_data = Vec::new();
    for result in image_data_results {
        image_data.push(result?);
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

    for result in match_results.into_iter().flatten() {
        pairwise_matches.insert(result.0, result.1);
    }
    if sequence_only {
        add_capture_sequence_bridges(&image_data, &mut pairwise_matches);
    }
    if focus_stack && !sequence_only {
        // Ordinary BRIEF/RANSAC matching intentionally rejects ambiguous
        // repeated artwork texture. That is correct for panorama-position
        // edges, but it also strands deliberately defocused or low-texture
        // members of an otherwise valid focus bracket. Retry only missing,
        // filename-adjacent pairs and accept them only when dense correlation
        // proves that they occupy the same camera pose.
        let mut filename_order = (0..image_data.len()).collect::<Vec<_>>();
        filename_order.sort_by(|&left, &right| {
            natural_path_cmp(&image_data[left].filename, &image_data[right].filename)
                .then_with(|| left.cmp(&right))
        });
        let mut cleared_adjacent = 0usize;
        for pair in filename_order.windows(2) {
            let left = pair[0].min(pair[1]);
            let right = pair[0].max(pair[1]);
            let key = (left, right);
            let Some(match_info) = pairwise_matches.get(&key) else {
                continue;
            };
            let should_clear = if focus_match_is_local_bracket(
                &image_data[left],
                &image_data[right],
                match_info,
            ) {
                false
            } else {
                focus_stack_adjacent_match_is_unreliable(&image_data, left, right, match_info)
            };
            if should_clear {
                pairwise_matches.remove(&key);
                cleared_adjacent += 1;
            }
        }
        if cleared_adjacent > 0 {
            println!(
                "  - Clearing {cleared_adjacent} adjacent focus-stack edge(s) for recovery retry"
            );
        }
        let recovery_pairs = filename_order
            .windows(2)
            .filter_map(|pair| {
                let left = pair[0].min(pair[1]);
                let right = pair[0].max(pair[1]);
                (!pairwise_matches.contains_key(&(left, right))).then_some((left, right))
            })
            .collect::<Vec<_>>();
        let recovered = recovery_pairs
            .par_iter()
            .filter_map(|&(left, right)| {
                let (source, target, invert_for_storage) =
                    canonical_match_direction(&image_data, left, right);
                let mut match_info = recover_adjacent_focus_bracket_match(
                    &image_data[source],
                    &image_data[target],
                    true,
                )?;
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
                Some(((left, right), match_info))
            })
            .collect::<Vec<_>>();
        if !recovered.is_empty() {
            println!(
                "  - Recovered {} same-pose low-texture focus-layer edge(s)",
                recovered.len()
            );
            pairwise_matches.extend(recovered);
        }
        // Do not synthesize scan-position edges from filename adjacency or a
        // coarse NCC peak. A capture sequence may reset to another raster row,
        // and repeated seals/figures can make that false edge look convincing.
        // Position bridges are retried below with the full feature/RANSAC
        // matcher; if that evidence is absent the stack remains disconnected
        // and is rejected instead of rendered as a duplicated collage.
    }
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
    let (ordered_indices, global_homographies) = if blend_mode == BlendMode::FocusStack {
        build_focus_stack_stitching_order(&image_data, &pairwise_matches)
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
    let focus_capture_group_ids = if blend_mode == BlendMode::FocusStack {
        focus_capture_groups(&image_data, &pairwise_matches, &global_homographies).map(
            |(groups, _)| {
                let mut ids = HashMap::new();
                for (group_index, group) in groups.iter().enumerate() {
                    let group_id = u8::try_from(group_index + 1)
                        .expect("focus stack source limit keeps group ids in u8");
                    for &image_index in &group.members {
                        ids.insert(image_data[image_index].id, group_id);
                    }
                }
                ids
            },
        )
    } else {
        None
    };

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
    let stitched_images_info: Vec<&ImageInfo> =
        ordered_indices.iter().map(|&i| &image_data[i]).collect();
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
            full_canvas_width,
            full_canvas_height,
            render_scale: 1.0,
            ordered_paths,
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
        if let Some(full_image) = retained_full_images.remove(&image.id) {
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
            load_prepared_stack_source(&image.filename, &settings)
                .map(|source| source_to_render_rgb32f(source.image, image.width, image.height))
        }
    };
    let sequence_gap_aware = blend_mode == BlendMode::FocusStack
        && pairwise_matches
            .values()
            .any(|match_info| match_info.sequence_bridge);
    let panorama = match blend_mode {
        BlendMode::Panorama => stitching::progressive_seam_stitcher(
            &render_images_info,
            render_homographies,
            projection,
            app_handle.clone(),
            progress_event,
            &mut load_render_image,
        ),
        BlendMode::FocusStack => stitching::focus_stack_stitcher(
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
        ),
    }?;

    println!("Stitching completed in {:.2?}\n", start_time.elapsed());

    let _ = app_handle.emit(progress_event, "Finalizing image result...");

    Ok(StitchOutcome {
        image: DynamicImage::ImageRgb32F(panorama),
        full_canvas_width,
        full_canvas_height,
        render_scale,
        ordered_paths,
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
    dx.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
    if points.len() < sample_size {
        return None;
    }

    let mut rng =
        StdRng::seed_from_u64(seed ^ (points.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
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
        let inlier_indices =
            symmetric_inlier_indices(&transform, points, FOCUS_MODEL_INLIER_THRESHOLD);
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
        let refitted_inliers =
            symmetric_inlier_indices(&refitted, points, FOCUS_MODEL_INLIER_THRESHOLD);
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
    // A moving focus stack is still a flat artwork scan. In Auto mode a
    // projective fit can explain a handful of repeated brush strokes while
    // bending the entire paper plane; that error compounds over dozens of
    // frames and becomes the rectangular/stepped output seen in v33. Keep
    // projective geometry for an explicit user-selected perspective mode only.
    let allow_projective = explicit_projective;
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
        .unwrap_or_else(|| mst_adj.keys().next().copied().unwrap_or(0));

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

fn focus_match_center_motion_ratio(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
) -> Option<f64> {
    let source_center = Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5);
    let target_center = Point2::new(target.width as f64 * 0.5, target.height as f64 * 0.5);
    let mapped_center = transformed_point(&match_info.homography, source_center)?;
    let scale = source
        .width
        .max(source.height)
        .max(target.width.max(target.height))
        .max(1) as f64;
    let displacement = (mapped_center - target_center).norm();
    (displacement.is_finite()).then_some(displacement / scale)
}

fn focus_match_is_local_bracket(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
) -> bool {
    if match_info.sequence_bridge {
        return false;
    }
    let compatible_focal_length = source
        .focal_length_35mm
        .zip(target.focal_length_35mm)
        .map(|(left, right)| {
            let ratio = (left / right.max(1e-8)).max(right / left.max(1e-8));
            ratio.is_finite() && ratio <= 1.10
        })
        .unwrap_or(true);
    let compatible_capture_distance = trailing_capture_number(&source.filename)
        .zip(trailing_capture_number(&target.filename))
        .map(|(left, right)| left.abs_diff(right) <= FOCUS_BRACKET_MAX_CAPTURE_GAP)
        .unwrap_or(true);
    if match_info.coarse_bridge {
        let scale = match_info.homography[(0, 0)].abs();
        return compatible_focal_length
            && compatible_capture_distance
            && match_info.points.len() >= SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION
            && (0.88..=1.12).contains(&scale)
            && focus_match_center_motion_ratio(source, target, match_info)
                .is_some_and(|motion| motion <= 0.12)
            && panorama_transform_overlap_support(
                &match_info.homography,
                source.dimensions(),
                target.dimensions(),
            ) >= 0.60;
    }
    if match_info.points.len() < FOCUS_MODEL_MIN_INLIERS {
        return false;
    }
    compatible_focal_length
        && compatible_capture_distance
        && focus_match_center_motion_ratio(source, target, match_info)
            .is_some_and(|motion| motion <= FOCUS_BRACKET_MAX_CENTER_MOTION_RATIO)
        && panorama_transform_overlap_support(
            &match_info.homography,
            source.dimensions(),
            target.dimensions(),
        ) >= FOCUS_BRACKET_MIN_OVERLAP_SUPPORT
}

fn focus_local_bracket_components(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Vec<Vec<usize>> {
    let mut dsu = Dsu::new(images.len());
    for (&(source_index, target_index), match_info) in matches {
        let (Some(source), Some(target)) = (images.get(source_index), images.get(target_index))
        else {
            continue;
        };
        if focus_match_is_local_bracket(source, target, match_info) {
            dsu.union(source_index, target_index);
        }
    }
    let mut components = HashMap::<usize, Vec<usize>>::new();
    for image_index in 0..images.len() {
        let root = dsu.find(image_index);
        components.entry(root).or_default().push(image_index);
    }
    let mut components = components
        .into_values()
        .filter(|component| {
            component.len() >= 2 && component.len() <= FOCUS_BRACKET_MAX_COMPONENT_SOURCES
        })
        .collect::<Vec<_>>();
    for component in &mut components {
        component.sort_by(|left, right| {
            natural_path_cmp(&images[*left].filename, &images[*right].filename)
        });
    }
    components.sort_by(|left, right| {
        natural_path_cmp(&images[left[0]].filename, &images[right[0]].filename)
    });
    components
}

fn lock_focus_local_bracket_poses(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    global_homographies: &HashMap<usize, Matrix3<f64>>,
) -> HashMap<usize, Matrix3<f64>> {
    let components = focus_local_bracket_components(images, matches);
    if components.is_empty() {
        return global_homographies.clone();
    }
    let mut locked = global_homographies.clone();
    let mut locked_sources = 0usize;
    let mut maximum_center_shift = 0.0f64;

    for component in &components {
        let members = component.iter().copied().collect::<HashSet<_>>();
        let mut edges = matches
            .iter()
            .filter_map(|(&(source_index, target_index), match_info)| {
                if !members.contains(&source_index)
                    || !members.contains(&target_index)
                    || !focus_match_is_local_bracket(
                        &images[source_index],
                        &images[target_index],
                        match_info,
                    )
                {
                    return None;
                }
                let error = median_symmetric_error(&match_info.homography, &match_info.points);
                let precision = if error.is_finite() {
                    1.0 / (1.0 + error / FOCUS_MODEL_INLIER_THRESHOLD)
                } else {
                    0.0
                };
                Some((
                    match_info.inliers.max(match_info.points.len()) as f64 * precision,
                    source_index,
                    target_index,
                    match_info.homography,
                ))
            })
            .collect::<Vec<_>>();
        edges.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| natural_path_cmp(&images[left.1].filename, &images[right.1].filename))
                .then_with(|| natural_path_cmp(&images[left.2].filename, &images[right.2].filename))
        });

        let mut support = HashMap::<usize, f64>::new();
        for &(weight, source_index, target_index, _) in &edges {
            *support.entry(source_index).or_default() += weight;
            *support.entry(target_index).or_default() += weight;
        }
        let Some(anchor) = component.iter().copied().min_by(|left, right| {
            support
                .get(right)
                .copied()
                .unwrap_or(0.0)
                .total_cmp(&support.get(left).copied().unwrap_or(0.0))
                .then_with(|| natural_path_cmp(&images[*left].filename, &images[*right].filename))
        }) else {
            continue;
        };
        let Some(anchor_global) = global_homographies.get(&images[anchor].id).copied() else {
            continue;
        };

        let mut local_dsu = Dsu::new(images.len());
        let mut adjacency = HashMap::<usize, Vec<(usize, Matrix3<f64>)>>::new();
        for &(_, source_index, target_index, source_to_target) in &edges {
            if local_dsu.find(source_index) == local_dsu.find(target_index) {
                continue;
            }
            let Some(target_to_source) = source_to_target.try_inverse() else {
                continue;
            };
            local_dsu.union(source_index, target_index);
            adjacency
                .entry(source_index)
                .or_default()
                .push((target_index, target_to_source));
            adjacency
                .entry(target_index)
                .or_default()
                .push((source_index, source_to_target));
        }

        let mut relative_to_anchor = HashMap::new();
        relative_to_anchor.insert(anchor, Matrix3::identity());
        let mut pending = VecDeque::from([anchor]);
        while let Some(current) = pending.pop_front() {
            let current_to_anchor = relative_to_anchor[&current];
            for &(neighbor, neighbor_to_current) in adjacency.get(&current).into_iter().flatten() {
                if relative_to_anchor.contains_key(&neighbor) {
                    continue;
                }
                relative_to_anchor.insert(neighbor, current_to_anchor * neighbor_to_current);
                pending.push_back(neighbor);
            }
        }
        if relative_to_anchor.len() != component.len() {
            continue;
        }

        for &image_index in component {
            let replacement = anchor_global * relative_to_anchor[&image_index];
            if !transform_is_stable_for_focus_stack(&replacement, images[image_index].dimensions())
            {
                continue;
            }
            if let Some(previous) = global_homographies.get(&images[image_index].id) {
                let center = Point2::new(
                    images[image_index].width as f64 * 0.5,
                    images[image_index].height as f64 * 0.5,
                );
                if let (Some(before), Some(after)) = (
                    transformed_point(previous, center),
                    transformed_point(&replacement, center),
                ) {
                    maximum_center_shift = maximum_center_shift.max((after - before).norm());
                }
            }
            locked.insert(images[image_index].id, replacement);
            locked_sources += 1;
        }
    }
    println!(
        "  - Locked {} local focus-bracket component(s), {} source poses; maximum center correction {:.2}px",
        components.len(),
        locked_sources,
        maximum_center_shift
    );
    locked
}

struct FocusCaptureGroup {
    members: Vec<usize>,
    anchor: usize,
    local_to_anchor: HashMap<usize, Matrix3<f64>>,
}

#[derive(Clone, Copy)]
struct FocusGroupEdgeCandidate {
    transform: Matrix3<f64>,
    quality: f64,
    independent_support: usize,
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

fn focus_capture_groups(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<(Vec<FocusCaptureGroup>, Vec<usize>)> {
    let mut components = focus_local_bracket_components(images, matches);
    let mut assigned = vec![false; images.len()];
    for component in &components {
        for &image_index in component {
            assigned[image_index] = true;
        }
    }
    for (image_index, is_assigned) in assigned.into_iter().enumerate() {
        if !is_assigned {
            components.push(vec![image_index]);
        }
    }
    components.sort_by(|left, right| {
        natural_path_cmp(&images[left[0]].filename, &images[right[0]].filename)
    });

    let mut group_for_image = vec![usize::MAX; images.len()];
    let mut groups = Vec::with_capacity(components.len());
    for (group_index, members) in components.into_iter().enumerate() {
        let anchor = members[0];
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
    Some((groups, group_for_image))
}

fn solve_focus_group_translation_poses(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
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
    if consensus_covered.saturating_mul(100) < groups.len().saturating_mul(90) {
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
                if support == 0 || left >= groups.len() || right >= groups.len() {
                    return None;
                }
                let left_image = &images[groups[left].anchor];
                let left_center = Point2::new(
                    left_image.width as f64 * 0.5,
                    left_image.height as f64 * 0.5,
                );
                let right_point = transformed_point(&left_to_right, left_center)?;
                let observed = left_center - right_point;
                let support_weight = if support >= 2 {
                    support as f64
                } else {
                    FOCUS_GROUP_SINGLE_EDGE_WEIGHT_FACTOR
                };
                let weight = support_weight * score.max(1e-8).sqrt()
                    / (1.0
                        + median_error
                            / (coordinate_scale * FOCUS_GROUP_CONSENSUS_MAX_ERROR_RATIO));
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

    let variable_count = groups.len().saturating_sub(1) * 2;
    let variable_index = |group: usize, axis: usize| {
        let compact = if group < reference_group {
            group
        } else {
            group - 1
        };
        compact * 2 + axis
    };
    let mut positions = vec![[0.0f64; 2]; groups.len()];
    let robust_limit = coordinate_scale * 0.04;
    for _ in 0..8 {
        let mut normal = nalgebra::DMatrix::<f64>::zeros(variable_count, variable_count);
        let mut right_hand_side = nalgebra::DVector::<f64>::zeros(variable_count);
        for &(left, right, observed, base_weight) in &constraints {
            let residual = [
                positions[right][0] - positions[left][0] - observed[0],
                positions[right][1] - positions[left][1] - observed[1],
            ];
            let magnitude = residual[0].hypot(residual[1]);
            let weight = base_weight
                * if magnitude <= robust_limit {
                    1.0
                } else {
                    robust_limit / magnitude.max(1e-8)
                };
            for axis in 0..2 {
                let left_variable = (left != reference_group).then(|| variable_index(left, axis));
                let right_variable =
                    (right != reference_group).then(|| variable_index(right, axis));
                if let Some(variable) = left_variable {
                    normal[(variable, variable)] += weight;
                    right_hand_side[variable] -= weight * observed[axis];
                }
                if let Some(variable) = right_variable {
                    normal[(variable, variable)] += weight;
                    right_hand_side[variable] += weight * observed[axis];
                }
                if let (Some(left_variable), Some(right_variable)) = (left_variable, right_variable)
                {
                    normal[(left_variable, right_variable)] -= weight;
                    normal[(right_variable, left_variable)] -= weight;
                }
            }
        }
        for variable in 0..variable_count {
            normal[(variable, variable)] += FOCUS_GLOBAL_DAMPING;
        }
        let Some(solution) = normal.lu().solve(&right_hand_side) else {
            println!(
                "  - Focus capture-group translation geometry unavailable: singular relation graph"
            );
            return None;
        };
        let mut maximum_change = 0.0f64;
        for group in 0..groups.len() {
            if group == reference_group {
                continue;
            }
            let next = [
                solution[variable_index(group, 0)],
                solution[variable_index(group, 1)],
            ];
            maximum_change = maximum_change
                .max((next[0] - positions[group][0]).hypot(next[1] - positions[group][1]));
            positions[group] = next;
        }
        if maximum_change < 0.01 {
            break;
        }
    }
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
    if !median_residual.is_finite() || median_residual > robust_limit {
        println!(
            "  - Focus capture-group translation geometry rejected: median closure error {:.2}px exceeds {:.2}px",
            median_residual, robust_limit
        );
        return None;
    }
    println!(
        "  - Focus capture-group translation geometry seeded from {} relation(s), multi-layer core {consensus_covered}/{}, median closure error {:.2}px",
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

fn solve_focus_capture_group_poses(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
) -> HashMap<usize, Matrix3<f64>> {
    let Some((groups, group_for_image)) =
        focus_capture_groups(images, matches, locked_homographies)
    else {
        return locked_homographies.clone();
    };
    if groups.len() < 2 || reference_index >= images.len() {
        return locked_homographies.clone();
    }

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

    let mut candidates = HashMap::<(usize, usize), Vec<FocusGroupEdgeCandidate>>::new();
    for (&(source_index, target_index), match_info) in matches {
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
        let source_group = group_for_image[source_index];
        let target_group = group_for_image[target_index];
        if source_group == target_group {
            continue;
        }
        let Some(source_local_inverse) = groups[source_group]
            .local_to_anchor
            .get(&source_index)
            .and_then(|matrix| (*matrix).try_inverse())
        else {
            continue;
        };
        let Some(target_local) = groups[target_group].local_to_anchor.get(&target_index) else {
            continue;
        };
        let source_anchor_to_target_anchor =
            target_local * match_info.homography * source_local_inverse;
        let (key, transform) = if source_group < target_group {
            ((source_group, target_group), source_anchor_to_target_anchor)
        } else {
            let Some(inverse) = source_anchor_to_target_anchor.try_inverse() else {
                continue;
            };
            ((target_group, source_group), inverse)
        };
        let error = median_symmetric_error(&match_info.homography, &match_info.points);
        let precision = if error.is_finite() {
            1.0 / (1.0 + error / FOCUS_MODEL_INLIER_THRESHOLD)
        } else {
            0.0
        };
        let spatial = panorama_spatial_support(
            &match_info.points,
            images[source_index].dimensions(),
            images[target_index].dimensions(),
        );
        let overlap = panorama_transform_overlap_support(
            &match_info.homography,
            images[source_index].dimensions(),
            images[target_index].dimensions(),
        );
        let quality = (match_info.inliers.max(match_info.points.len()) as f64).sqrt()
            * precision
            * (0.25 + 0.75 * spatial)
            * (0.25 + 0.75 * overlap);
        if quality.is_finite() && quality > 0.0 {
            candidates
                .entry(key)
                .or_default()
                .push(FocusGroupEdgeCandidate {
                    transform,
                    quality,
                    independent_support: if consensus_coarse { 2 } else { 1 },
                });
        }
    }

    let coordinate_scale = images
        .iter()
        .map(|image| image.width.max(image.height) as f64)
        .fold(1.0, f64::max);
    let consensus_threshold = coordinate_scale * FOCUS_GROUP_CONSENSUS_MAX_ERROR_RATIO;
    let mut group_edges = Vec::<(f64, usize, usize, Matrix3<f64>, usize, f64)>::new();
    let mut consensus_pair_count = 0usize;
    for ((left_group, right_group), pair_candidates) in candidates {
        let dimensions = images[groups[left_group].anchor].dimensions();
        let mut best: Option<(f64, Matrix3<f64>, usize, f64)> = None;
        for candidate in &pair_candidates {
            let mut supporting_errors = Vec::new();
            let mut supporting_quality = 0.0;
            let mut independent_support = 0usize;
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
                }
            }
            let support = independent_support;
            if support == 0 {
                continue;
            }
            let median_error = median_value(&mut supporting_errors).unwrap_or(consensus_threshold);
            let support_factor = if support >= 2 {
                (support * support) as f64
            } else {
                FOCUS_GROUP_SINGLE_EDGE_WEIGHT_FACTOR
            };
            let score = support_factor * supporting_quality
                / (1.0 + median_error / consensus_threshold.max(1.0));
            if best
                .as_ref()
                .is_none_or(|(best_score, _, _, _)| score > *best_score)
            {
                best = Some((score, candidate.transform, support, median_error));
            }
        }
        let Some((score, transform, support, median_error)) = best else {
            continue;
        };
        if support >= 2 {
            consensus_pair_count += 1;
        }
        group_edges.push((
            score,
            left_group,
            right_group,
            transform,
            support,
            median_error,
        ));
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
    // The image-level pose graph can be pulled toward a repeated character.
    // Once multiple independent bracket members agree on a group relation,
    // use those full planar relations (not only their translation residuals)
    // to seed the camera-group geometry. A maximum-confidence tree avoids
    // averaging incompatible loop edges, and an incomplete tree falls back to
    // the established image-level poses.
    let mut consensus_edges = group_edges
        .iter()
        .filter(|edge| edge.4 >= 2)
        .copied()
        .collect::<Vec<_>>();
    consensus_edges.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut group_dsu = Dsu::new(groups.len());
    let mut group_adjacency = HashMap::<usize, Vec<(usize, Matrix3<f64>)>>::new();
    let mut tree_edges = 0usize;
    for &(_, left_group, right_group, left_to_right, _, _) in &consensus_edges {
        if group_dsu.find(left_group) == group_dsu.find(right_group) {
            continue;
        }
        let Some(right_to_left) = left_to_right.try_inverse() else {
            continue;
        };
        group_dsu.union(left_group, right_group);
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
    let translation_geometry_seeded = if let Some(translation_poses) =
        solve_focus_group_translation_poses(
            images,
            &groups,
            &group_edges,
            reference_group,
            coordinate_scale,
        ) {
        group_poses = translation_poses.into_iter().map(Some).collect();
        true
    } else {
        false
    };
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
    let maximum_constraint = coordinate_scale * 0.08;
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
        if dx.hypot(dy) > maximum_constraint {
            continue;
        }
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

    let variable_count = groups.len().saturating_sub(1) * 2;
    let variable_index = |group: usize, axis: usize| {
        let compact = if group < reference_group {
            group
        } else {
            group - 1
        };
        compact * 2 + axis
    };
    let mut corrections = vec![[0.0f64; 2]; groups.len()];
    for _ in 0..6 {
        let mut normal = nalgebra::DMatrix::<f64>::zeros(variable_count, variable_count);
        let mut right_hand_side = nalgebra::DVector::<f64>::zeros(variable_count);
        for &(left_group, right_group, observed, base_weight, _) in &constraints {
            let residual = [
                corrections[left_group][0] - corrections[right_group][0] - observed[0],
                corrections[left_group][1] - corrections[right_group][1] - observed[1],
            ];
            let magnitude = residual[0].hypot(residual[1]);
            let robust = if magnitude <= consensus_threshold {
                1.0
            } else {
                consensus_threshold / magnitude.max(1e-8)
            };
            let weight = base_weight * robust;
            for axis in 0..2 {
                let left_variable =
                    (left_group != reference_group).then(|| variable_index(left_group, axis));
                let right_variable =
                    (right_group != reference_group).then(|| variable_index(right_group, axis));
                if let Some(left) = left_variable {
                    normal[(left, left)] += weight;
                    right_hand_side[left] += weight * observed[axis];
                }
                if let Some(right) = right_variable {
                    normal[(right, right)] += weight;
                    right_hand_side[right] -= weight * observed[axis];
                }
                if let (Some(left), Some(right)) = (left_variable, right_variable) {
                    normal[(left, right)] -= weight;
                    normal[(right, left)] -= weight;
                }
            }
        }
        for variable in 0..variable_count {
            normal[(variable, variable)] += FOCUS_GLOBAL_DAMPING;
        }
        let Some(solution) = normal.lu().solve(&right_hand_side) else {
            return locked_homographies.clone();
        };
        let mut maximum_change = 0.0f64;
        for (group_index, correction) in corrections.iter_mut().enumerate() {
            if group_index == reference_group {
                continue;
            }
            let next = [
                solution[variable_index(group_index, 0)],
                solution[variable_index(group_index, 1)],
            ];
            maximum_change =
                maximum_change.max((next[0] - correction[0]).hypot(next[1] - correction[1]));
            *correction = next;
        }
        if maximum_change < 0.01 {
            break;
        }
    }

    let maximum_center_correction = corrections
        .iter()
        .map(|correction| correction[0].hypot(correction[1]))
        .fold(0.0, f64::max);
    let maximum_allowed = coordinate_scale * FOCUS_GROUP_MAX_CENTER_CORRECTION_RATIO;
    if !maximum_center_correction.is_finite() || maximum_center_correction > maximum_allowed {
        println!(
            "  - Focus capture-group translation solve rejected: {:.1}px maximum correction exceeds {:.1}px",
            maximum_center_correction, maximum_allowed
        );
        return locked_homographies.clone();
    }

    let mut solved = HashMap::new();
    for (group_index, group) in groups.iter().enumerate() {
        let original_group_pose = if use_consensus_group_poses {
            group_poses[group_index].unwrap()
        } else {
            let Some(pose) = locked_homographies.get(&images[group.anchor].id).copied() else {
                return locked_homographies.clone();
            };
            pose
        };
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
    println!(
        "  - Focus capture-group translation solve: {} groups, {} candidate group pairs, {} consensus pairs, {} accepted constraints; maximum center correction {:.2}px",
        groups.len(),
        group_edges.len(),
        consensus_pair_count,
        constraints.len(),
        maximum_center_correction,
    );
    solved
}

fn focus_auto_order_motion_scale(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> f64 {
    let mut minimum_motion_by_image = vec![f64::INFINITY; images.len()];
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

fn build_focus_stack_stitching_order(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
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
    // Preserve the established geometry and evidence-selected order for normal
    // focus stacks.  Bracket locking and camera-group translation were added
    // specifically for long scans whose repeated calligraphy otherwise lets
    // the pose graph drift; applying them to compact stacks can manufacture
    // multiple displaced frames from ordinary focus breathing.
    let optimized_or_locked_homographies = if large_scan {
        let locked_homographies =
            lock_focus_local_bracket_poses(images, matches, &optimized_homographies);
        solve_focus_capture_group_poses(images, matches, &locked_homographies, reference_index)
    } else {
        optimized_homographies.clone()
    };
    let (valid_geometry_edges, total_geometry_edges, median_overlap, median_scale) =
        focus_capture_geometry_diagnostics(
            images,
            &render_order,
            &optimized_or_locked_homographies,
        );
    println!(
        "  - Focus geometry validation: {valid_geometry_edges}/{total_geometry_edges} adjacent edges, median overlap {:.1}%, median scale {:.3}",
        median_overlap * 100.0,
        median_scale,
    );
    let global_homographies = if !large_scan
        || focus_capture_geometry_passes(
            images,
            valid_geometry_edges,
            total_geometry_edges,
            median_overlap,
            &optimized_or_locked_homographies,
        ) {
        optimized_or_locked_homographies
    } else {
        // The long-scan refinement can be pulled by a repeated character or
        // seal. Retry the same verified graph without the capture-group
        // translation solve before rejecting the stack outright.
        println!(
            "  - Focus geometry rejected ({valid_geometry_edges}/{total_geometry_edges} continuous edges); retrying the ungrouped global pose"
        );
        let (retry_valid_edges, retry_total_edges, retry_overlap, retry_scale) =
            focus_capture_geometry_diagnostics(images, &render_order, &optimized_homographies);
        println!(
            "  - Focus geometry retry: {retry_valid_edges}/{retry_total_edges} adjacent edges, median overlap {:.1}%, median scale {:.3}",
            retry_overlap * 100.0,
            retry_scale,
        );
        if focus_capture_geometry_passes(
            images,
            retry_valid_edges,
            retry_total_edges,
            retry_overlap,
            &optimized_homographies,
        ) {
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
    (render_order, global_homographies)
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
    for (&(source_index, target_index), match_info) in matches {
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
        let distant_edge = source_index.abs_diff(target_index) > FOCUS_GLOBAL_MAX_SEQUENCE_GAP;
        let mixed_focal_edge =
            image_pair_has_mixed_focal_lengths(&images[source_index], &images[target_index]);
        if distant_edge && !mixed_focal_edge {
            // Long same-lens edges are especially ambiguous on repeated
            // calligraphy. Cross-lens edges are different: focus brackets and
            // overview/detail captures can be far apart in filename order,
            // and excluding all of them leaves a whole lens group hanging
            // from whichever single edge won the spanning tree. Those edges
            // have already passed focal-scale and full-overlap validation, so
            // let their aggregate consensus close the pose graph.
            continue;
        }
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
        // Only local cross-lens edges survive the sequence-neighbourhood
        // filter above. They are the measured bridge between focal modules;
        // distant repeated-stroke matches are still excluded so a lens switch
        // cannot move an entire capture run to another scene region.
        println!(
            "  - Focus global registration for mixed-focal stack: solving local same-/cross-lens edges"
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

#[cfg(test)]
mod alignment_tests {
    use super::*;
    use nalgebra::Point2;

    fn test_image(id: usize, filename: &str) -> ImageInfo {
        ImageInfo {
            id,
            filename: filename.to_string(),
            width: 1,
            height: 1,
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
        }
    }

    fn focus_test_image(id: usize, filename: &str) -> ImageInfo {
        let mut image = test_image(id, filename);
        image.width = 1_000;
        image.height = 1_000;
        image
    }

    #[test]
    fn focus_adjacent_same_focal_pair_allows_missing_exif() {
        let mut left = focus_test_image(0, "DSC_3743.png");
        left.focal_length_35mm = Some(50.0);
        let mut right = focus_test_image(1, "DSC_3744.png");
        right.focal_length_35mm = None;
        assert!(focus_adjacent_same_focal_pair(&left, &right));
        let mut mismatched = focus_test_image(2, "DSC_3745.png");
        mismatched.focal_length_35mm = Some(85.0);
        assert!(!focus_compatible_focal_length(&left, &mismatched));
    }

    #[test]
    fn narrow_same_focal_scan_boundary_rejects_low_texture_overlap() {
        assert!(!focus_position_recovery_accepts(
            1, true, 1.0, 0.045, 0.12, 120, 0.35, -0.05, -0.05, false,
        ));
        assert!(!focus_position_recovery_accepts(
            2, true, 1.0, 0.045, 0.12, 120, 0.35, -0.05, -0.05, false,
        ));
        assert!(!focus_position_recovery_accepts(
            1, false, 1.0, 0.20, 0.12, 120, 0.35, -0.05, -0.05, false,
        ));
    }

    #[test]
    fn focus_geometry_guard_rejects_incomplete_or_drifting_scan() {
        let images = (0..4)
            .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
            .collect::<Vec<_>>();
        let stable = (0..images.len())
            .map(|index| (images[index].id, Matrix3::identity()))
            .collect::<HashMap<_, _>>();
        assert!(focus_capture_geometry_passes(&images, 3, 3, 0.64, &stable));
        assert!(!focus_capture_geometry_passes(&images, 2, 3, 0.64, &stable));

        let drifting = HashMap::from([
            (0, Matrix3::identity()),
            (
                1,
                Matrix3::new(0.78, 0.0, 0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0),
            ),
            (2, Matrix3::identity()),
            (3, Matrix3::identity()),
        ]);
        assert!(!focus_capture_geometry_passes(
            &images, 3, 3, 0.64, &drifting
        ));
    }

    fn identity_match(inliers: usize) -> MatchInfo {
        MatchInfo {
            homography: Matrix3::identity(),
            inliers,
            sequence_bridge: false,
            coarse_bridge: false,
            points: Vec::new(),
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
        }
    }

    #[test]
    fn low_texture_overlap_requires_boundary_rescue_context() {
        let zhang_haohao_boundary = (0.886, 0.323, 0.134, 1_313);
        assert!(!focus_overlap_is_verified(
            zhang_haohao_boundary,
            13,
            1.887,
            false
        ));
        assert!(focus_overlap_is_verified(
            zhang_haohao_boundary,
            13,
            1.887,
            true
        ));
        assert!(!focus_overlap_is_verified(
            (0.886, 0.323, 0.099, 1_313),
            13,
            1.887,
            true
        ));
        assert!(!focus_overlap_is_verified(
            zhang_haohao_boundary,
            11,
            1.887,
            true
        ));
    }

    #[test]
    fn capture_sequence_edges_outrank_remote_calligraphy_lookalikes() {
        let first = focus_test_image(0, "DSC_1000.NEF");
        let adjacent = focus_test_image(1, "DSC_1001.NEF");
        let remote = focus_test_image(2, "DSC_1200.NEF");
        let local_match = observed_translation_match(180.0, 0.0, 16);
        let remote_match = observed_translation_match(20.0, 0.0, 300);

        assert_eq!(
            focus_graph_capture_sequence_boost(&first, &adjacent, &local_match),
            FOCUS_CAPTURE_SEQUENCE_GRAPH_WEIGHT_BOOST
        );
        assert_eq!(
            focus_graph_capture_sequence_boost(&first, &remote, &remote_match),
            1.0
        );
    }

    #[test]
    fn weak_interior_focus_frame_can_be_skipped_only_across_verified_overlap() {
        let images = vec![
            focus_test_image(0, "DSC_2740.NEF"),
            focus_test_image(1, "DSC_2741.NEF"),
            focus_test_image(2, "DSC_2742.NEF"),
        ];
        let stitched = HashSet::from([0, 2]);
        let verified_bypass = HashMap::from([((0, 2), observed_translation_match(20.0, 0.0, 24))]);

        assert!(focus_unstitched_sources_are_redundant(
            &images,
            &verified_bypass,
            &stitched
        ));
        assert!(!focus_unstitched_sources_are_redundant(
            &images,
            &HashMap::new(),
            &stitched
        ));
    }

    fn translation_match(dx: f64, dy: f64, inliers: usize) -> MatchInfo {
        MatchInfo {
            homography: Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0),
            inliers,
            sequence_bridge: false,
            coarse_bridge: false,
            points: Vec::new(),
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
        }
    }

    fn observed_translation_match(dx: f64, dy: f64, inliers: usize) -> MatchInfo {
        let points = (0..4)
            .flat_map(|row| {
                (0..4).map(move |column| {
                    let source =
                        Point2::new(140.0 + column as f64 * 220.0, 140.0 + row as f64 * 220.0);
                    (source, source + nalgebra::Vector2::new(dx, dy))
                })
            })
            .collect::<Vec<_>>();
        MatchInfo {
            homography: Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0),
            inliers,
            sequence_bridge: false,
            coarse_bridge: false,
            candidate_points: points.clone(),
            points,
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
        }
    }

    #[test]
    fn canonical_match_direction_is_independent_of_import_order() {
        let reversed = vec![test_image(0, "z.jpg"), test_image(1, "a.jpg")];
        let (source, target, invert_for_storage) = canonical_match_direction(&reversed, 0, 1);

        assert_eq!(reversed[source].filename, "a.jpg");
        assert_eq!(reversed[target].filename, "z.jpg");
        assert!(invert_for_storage);

        let forward = vec![test_image(0, "a.jpg"), test_image(1, "z.jpg")];
        let (source, target, invert_for_storage) = canonical_match_direction(&forward, 0, 1);

        assert_eq!((source, target), (0, 1));
        assert_eq!(forward[source].filename, "a.jpg");
        assert_eq!(forward[target].filename, "z.jpg");
        assert!(!invert_for_storage);
    }

    #[test]
    fn large_stack_matching_scales_with_neighbors_instead_of_all_pairs() {
        let small_pairs = pairs_to_match(SCALABLE_STACK_THRESHOLD);
        assert_eq!(
            small_pairs.len(),
            SCALABLE_STACK_THRESHOLD * (SCALABLE_STACK_THRESHOLD - 1) / 2
        );

        let large_pairs = pairs_to_match(200);
        assert!(large_pairs.len() <= 200 * LARGE_STACK_NEIGHBOR_WINDOW);
        assert!(large_pairs.len() * 10 < 200 * 199 / 2);
        for index in 0..199 {
            assert!(large_pairs.contains(&(index, index + 1)));
        }
    }

    #[test]
    fn large_stack_candidates_include_natural_filename_neighbors() {
        let images = (0..31)
            .map(|index| {
                let number = (index * 13) % 31;
                test_image(index, &format!("tile-{number}.jpg"))
            })
            .collect::<Vec<_>>();
        let pairs = pairs_to_match_for_images(&images, BlendMode::Panorama);
        let index_for_number = |number: usize| {
            images
                .iter()
                .position(|image| image.filename == format!("tile-{number}.jpg"))
                .expect("generated filename should exist")
        };

        for number in 0..30 {
            let left = index_for_number(number);
            let right = index_for_number(number + 1);
            assert!(pairs.contains(&(left.min(right), left.max(right))));
        }
        assert!(pairs.len() <= images.len() * LARGE_STACK_NEIGHBOR_WINDOW * 2);
    }

    #[test]
    fn medium_focus_stack_candidates_are_independent_of_upload_order() {
        let images = (0..45)
            .map(|index| test_image(index, &format!("upload-random-{index}.jpg")))
            .collect::<Vec<_>>();
        let pairs = pairs_to_match_for_images(&images, BlendMode::FocusStack);

        assert_eq!(
            pairs.len(),
            45 * 44 / 2,
            "a medium focus stack must inspect every possible overlap before ordering"
        );
        assert!(pairs.contains(&(0, 44)));
        assert!(pairs.contains(&(7, 31)));
    }

    #[test]
    fn medium_panorama_candidates_cover_a_possible_lens_switch() {
        let images = (0..69)
            .map(|index| test_image(index, &format!("frame-{index}.jpg")))
            .collect::<Vec<_>>();
        let pairs = pairs_to_match_for_images(&images, BlendMode::Panorama);

        assert_eq!(pairs.len(), 69 * 68 / 2);
        assert!(pairs.contains(&(0, 68)));
    }

    #[test]
    fn natural_path_order_compares_numeric_filename_runs() {
        let mut paths = ["tile-10.jpg", "tile-2.jpg", "tile-001.jpg", "tile-1.jpg"];
        paths.sort_by(|left, right| natural_path_cmp(left, right));

        assert_eq!(
            paths,
            ["tile-1.jpg", "tile-001.jpg", "tile-2.jpg", "tile-10.jpg"]
        );
    }

    #[test]
    fn focus_stack_geometry_uses_overlap_evidence_and_rendering_uses_capture_order() {
        let images = vec![
            focus_test_image(0, "random-c.jpg"),
            focus_test_image(1, "random-a.jpg"),
            focus_test_image(2, "random-d.jpg"),
            focus_test_image(3, "random-b.jpg"),
        ];
        let matches = HashMap::from([
            ((0, 2), translation_match(100.0, 0.0, 30)),
            ((2, 3), translation_match(100.0, 0.0, 35)),
            ((3, 1), translation_match(100.0, 0.0, 25)),
            // A long overlap can have more descriptors than its immediate
            // neighbor. Motion-aware graph weights must still prefer the
            // local capture path.
            ((0, 3), translation_match(200.0, 0.0, 400)),
            ((0, 1), translation_match(300.0, 0.0, 350)),
        ]);

        let (order, homographies) = build_focus_stack_stitching_order(&images, &matches);

        assert_eq!(order.len(), images.len());
        assert_eq!(
            order.iter().copied().collect::<HashSet<_>>().len(),
            images.len()
        );
        assert!(order.windows(2).all(|pair| {
            matches.contains_key(&(pair[0], pair[1])) || matches.contains_key(&(pair[1], pair[0]))
        }));
        assert_eq!(homographies.len(), images.len());
        assert!(
            homographies
                .values()
                .any(|matrix| *matrix == Matrix3::identity())
        );
    }

    #[test]
    fn dense_focus_overlap_graph_identifies_a_continuous_scan() {
        let images = (0..12)
            .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
            .collect::<Vec<_>>();
        let mut matches = HashMap::new();
        for source in 0..images.len() {
            for target in source + 1..images.len() {
                matches.insert(
                    (source, target),
                    observed_translation_match(0.0, (target - source) as f64 * 12.0, 32),
                );
            }
        }

        assert!(focus_match_graph_is_dense_continuous_scan(
            &images, &matches
        ));
    }

    #[test]
    fn sparse_focus_overlap_graph_keeps_evidence_selected_render_order() {
        let images = (0..12)
            .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
            .collect::<Vec<_>>();
        let matches = (0..images.len() - 1)
            .map(|source| {
                (
                    (source, source + 1),
                    observed_translation_match(0.0, 12.0, 32),
                )
            })
            .collect::<HashMap<_, _>>();

        assert!(!focus_match_graph_is_dense_continuous_scan(
            &images, &matches
        ));
    }

    #[test]
    fn large_focus_scan_keeps_capture_order_when_one_weak_frame_is_disconnected() {
        let images = (0..65)
            .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
            .collect::<Vec<_>>();
        let skipped = 32usize;
        let retained = (0..images.len())
            .filter(|index| *index != skipped)
            .collect::<Vec<_>>();
        let matches = retained
            .windows(2)
            .map(|pair| {
                (
                    (pair[0].min(pair[1]), pair[0].max(pair[1])),
                    translation_match(20.0, 0.0, 32),
                )
            })
            .collect::<HashMap<_, _>>();

        let (order, _) = build_focus_stack_stitching_order(&images, &matches);

        assert_eq!(order, retained);
    }

    #[test]
    fn local_focus_bracket_lock_restores_the_direct_relative_pose() {
        let mut images = vec![
            focus_test_image(0, "DSC_1001.NEF"),
            focus_test_image(1, "DSC_1002.NEF"),
        ];
        for image in &mut images {
            image.focal_length_35mm = Some(90.0);
        }
        let matches = HashMap::from([((0, 1), observed_translation_match(10.0, 4.0, 32))]);
        let global = HashMap::from([
            (0, Matrix3::identity()),
            (
                1,
                Matrix3::new(1.0, 0.0, 90.0, 0.0, 1.0, 40.0, 0.0, 0.0, 1.0),
            ),
        ]);

        let locked = lock_focus_local_bracket_poses(&images, &matches, &global);
        let relative = locked[&1]
            .try_inverse()
            .expect("locked target pose should be invertible")
            * locked[&0];

        assert!((relative[(0, 2)] - 10.0).abs() < 1e-9);
        assert!((relative[(1, 2)] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn focus_capture_group_consensus_rejects_a_single_conflicting_edge() {
        let mut images = vec![
            focus_test_image(0, "DSC_1001.NEF"),
            focus_test_image(1, "DSC_1002.NEF"),
            focus_test_image(2, "DSC_1010.NEF"),
            focus_test_image(3, "DSC_1011.NEF"),
        ];
        for image in &mut images {
            image.focal_length_35mm = Some(90.0);
        }
        let matches = HashMap::from([
            ((0, 1), observed_translation_match(5.0, 0.0, 40)),
            ((2, 3), observed_translation_match(5.0, 0.0, 40)),
            ((0, 2), observed_translation_match(100.0, 0.0, 32)),
            ((1, 3), observed_translation_match(100.0, 0.0, 28)),
            // Repeated content proposes a different group placement, but it
            // has no independent agreement from another member pair.
            ((0, 3), observed_translation_match(400.0, 0.0, 80)),
        ]);
        let translation = |x: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let locked = HashMap::from([
            (0, translation(0.0)),
            (1, translation(-5.0)),
            (2, translation(-160.0)),
            (3, translation(-165.0)),
        ]);

        let solved = solve_focus_capture_group_poses(&images, &matches, &locked, 0);

        assert!((solved[&2][(0, 2)] + 100.0).abs() < 1e-3);
        assert!((solved[&3][(0, 2)] + 105.0).abs() < 1e-3);
    }

    #[test]
    fn focus_capture_group_solver_keeps_multi_layer_coarse_consensus() {
        let mut images = vec![
            focus_test_image(0, "DSC_1001.NEF"),
            focus_test_image(1, "DSC_1002.NEF"),
            focus_test_image(2, "DSC_1010.NEF"),
            focus_test_image(3, "DSC_1011.NEF"),
        ];
        for image in &mut images {
            image.focal_length_35mm = Some(90.0);
        }
        let mut consensus_bridge = observed_translation_match(240.0, 20.0, 12);
        consensus_bridge.coarse_bridge = true;
        // Component recovery uses this non-empty field only after different
        // focus layers on both sides agreed on the coarse transform.
        consensus_bridge.top_candidate_points = consensus_bridge.points.clone();
        let matches = HashMap::from([
            ((0, 1), observed_translation_match(5.0, 0.0, 40)),
            ((2, 3), observed_translation_match(5.0, 0.0, 40)),
            ((0, 2), consensus_bridge),
        ]);
        let translation = |x: f64, y: f64| Matrix3::new(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0);
        let locked = HashMap::from([
            (0, translation(0.0, 0.0)),
            (1, translation(-5.0, 0.0)),
            (2, translation(-60.0, 0.0)),
            (3, translation(-65.0, 0.0)),
        ]);

        let solved = solve_focus_capture_group_poses(&images, &matches, &locked, 0);

        assert!((solved[&2][(0, 2)] + 240.0).abs() < 1e-3);
        assert!((solved[&2][(1, 2)] + 20.0).abs() < 1e-3);
        assert!((solved[&3][(0, 2)] + 245.0).abs() < 1e-3);
    }

    #[test]
    fn focus_component_bridge_candidates_keep_both_sequence_boundaries() {
        let images = (0..12)
            .map(|index| focus_test_image(index, &format!("DSC_{:04}.NEF", 1000 + index)))
            .collect::<Vec<_>>();
        let component = (0..images.len()).collect::<Vec<_>>();

        let candidates = focus_component_bridge_candidates(&component, &images);

        assert!(candidates.contains(&0));
        assert!(candidates.contains(&1));
        assert!(candidates.contains(&10));
        assert!(candidates.contains(&11));
    }

    #[test]
    fn focus_stack_motion_detects_a_scan_without_reclassifying_a_fixed_stack() {
        let points = (0..3)
            .flat_map(|row| {
                (0..3).map(move |column| {
                    let source = Point2::new(
                        1_000.0 + column as f64 * 2_000.0,
                        900.0 + row as f64 * 2_000.0,
                    );
                    (source, source + nalgebra::Vector2::new(4.0, 2.0))
                })
            })
            .collect::<Vec<_>>();
        assert!(!focus_stack_motion_is_shifted_mosaic(
            &points,
            (9_504, 6_336)
        ));

        let shifted = points
            .iter()
            .map(|(source, target)| (*source, *target + nalgebra::Vector2::new(300.0, 0.0)))
            .collect::<Vec<_>>();
        assert!(focus_stack_motion_is_shifted_mosaic(
            &shifted,
            (9_504, 6_336)
        ));

        let scale_only = points
            .iter()
            .map(|(source, _)| {
                let centered = *source - Point2::new(4_752.0, 3_168.0);
                (*source, Point2::new(4_752.0, 3_168.0) + centered * 1.24)
            })
            .collect::<Vec<_>>();
        assert!(focus_stack_motion_is_shifted_mosaic(
            &scale_only,
            (9_504, 6_336)
        ));
    }

    #[test]
    fn large_stack_alignment_budget_tightens_as_source_count_grows() {
        assert_eq!(scalable_alignment_budget(64), (2_400, 1_600));
        assert_eq!(scalable_alignment_budget(128), (1_800, 1_100));
        assert_eq!(scalable_alignment_budget(200), (1_536, 800));
    }

    #[test]
    fn mixed_focal_detection_distinguishes_lens_changes_from_rounding_noise() {
        assert!(focal_lengths_span_multiple_lenses([35.0, 85.0]));
        assert!(!focal_lengths_span_multiple_lenses([34.0, 35.0, 36.0]));
        assert!(!focal_lengths_span_multiple_lenses([f64::NAN, -1.0, 85.0]));
    }

    #[test]
    fn scalable_preparation_workers_are_bounded_by_images_cpu_and_memory() {
        let abundant_memory = 32 * PREPARATION_RAM_PER_WORKER_BYTES;

        assert_eq!(
            bounded_preparation_worker_count(185, 10, abundant_memory),
            MAX_SCALABLE_PREPARATION_WORKERS
        );
        assert_eq!(bounded_preparation_worker_count(185, 4, abundant_memory), 4);
        assert_eq!(
            bounded_preparation_worker_count(185, 10, 2 * PREPARATION_RAM_PER_WORKER_BYTES),
            2
        );
        assert_eq!(bounded_preparation_worker_count(3, 10, abundant_memory), 3);
        assert_eq!(bounded_preparation_worker_count(185, 10, 0), 1);
    }

    #[test]
    fn oversized_panorama_canvas_is_scaled_to_the_memory_budget() {
        assert_eq!(memory_safe_panorama_render_scale(12_000, 8_000), 1.0);
        let scale = memory_safe_panorama_render_scale(248_296, 9_843);
        let scaled_pixels = (248_296.0 * scale).ceil() as u64 * (9_843.0 * scale).ceil() as u64;

        assert!(scale < 0.32);
        assert!(scaled_pixels <= MAX_IN_MEMORY_PANORAMA_PIXELS + 100_000);
    }

    #[test]
    fn render_scale_is_applied_after_the_global_image_transform() {
        let source = HashMap::from([(
            7,
            Matrix3::new(1.0, 0.0, 400.0, 0.0, 1.0, -80.0, 0.0, 0.0, 1.0),
        )]);
        let scaled = scaled_homographies(&source, 0.25);
        let transform = scaled.get(&7).expect("scaled transform should exist");

        assert_eq!(transform[(0, 0)], 0.25);
        assert_eq!(transform[(1, 1)], 0.25);
        assert_eq!(transform[(0, 2)], 100.0);
        assert_eq!(transform[(1, 2)], -20.0);
    }

    #[test]
    fn scaled_render_sources_preserve_the_output_coordinate_system() {
        let source = HashMap::from([(
            7,
            Matrix3::new(1.0, 0.0, 400.0, 0.0, 1.0, -80.0, 0.0, 0.0, 1.0),
        )]);
        let scale = 0.25;
        let scaled = scaled_source_render_homographies(&source, scale);
        let source_point = nalgebra::Point3::new(800.0, 200.0, 1.0);
        let expected = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0)
            * source[&7]
            * source_point;
        let actual =
            scaled[&7] * nalgebra::Point3::new(source_point.x * scale, source_point.y * scale, 1.0);

        assert!((actual.x - expected.x).abs() < 1e-9);
        assert!((actual.y - expected.y).abs() < 1e-9);
        assert!((actual.z - expected.z).abs() < 1e-9);

        let mut image = test_image(7, "large.jpg");
        image.width = 4_672;
        image.height = 7_008;
        let rendered = scaled_render_image_info(&image, scale);
        assert_eq!(rendered.dimensions(), (1_168, 1_752));
        assert!(rendered.alignment_image.is_empty());
        assert!(rendered.features.is_empty());
    }

    #[test]
    fn parallel_area_resize_preserves_constant_rgb8_sources() {
        let source = RgbImage::from_pixel(7, 5, image::Rgb([51, 102, 204]));
        let resized = resize_rgb8_area_to_rgb32f(&source, 3, 2);

        assert_eq!(resized.dimensions(), (3, 2));
        for pixel in resized.pixels() {
            assert!((pixel[0] - 0.2).abs() < 1e-6);
            assert!((pixel[1] - 0.4).abs() < 1e-6);
            assert!((pixel[2] - 0.8).abs() < 1e-6);
        }
    }

    #[test]
    fn panorama_refinement_rejects_multi_pixel_correspondence_errors() {
        let expected = Matrix3::new(1.0, 0.0, 4_273.25, 0.0, 1.0, 71.75, 0.0, 0.0, 1.0);
        let mut points = Vec::new();
        for row in 0..4 {
            for column in 0..6 {
                let source = Point2::new(
                    480.0 + column as f64 * 1_350.0,
                    420.0 + row as f64 * 1_420.0,
                );
                let mut target = transformed_point(&expected, source).unwrap();
                target.x += ((row * 7 + column * 5) as f64).sin() * 0.18;
                target.y += ((row * 3 + column * 11) as f64).cos() * 0.18;
                points.push((source, target));
            }
        }
        for index in 0..8 {
            let source = Point2::new(700.0 + index as f64 * 780.0, 850.0 + index as f64 * 510.0);
            let mut target = transformed_point(&expected, source).unwrap();
            target.x += 5.5;
            target.y -= 4.5;
            points.push((source, target));
        }

        let refined = refine_homography_inliers(
            &mut points,
            FULL_RES_REFINEMENT_THRESHOLD,
            processing::MIN_INLIERS_FOR_CONNECTION,
        )
        .expect("the accurate correspondence grid should remain connected");

        assert_eq!(points.len(), 24);
        assert!(symmetric_reprojection_rmse(&refined, &points) < 0.35);
    }

    #[test]
    fn allocation_free_patch_ncc_preserves_correlation_range() {
        let source = GrayImage::from_fn(9, 9, |x, y| {
            image::Luma([((x * 17 + y * 29 + x * y * 3) % 255) as u8])
        });
        let inverted =
            GrayImage::from_fn(9, 9, |x, y| image::Luma([255 - source.get_pixel(x, y)[0]]));
        let source_plane = LumaPlane::Gray(&source);
        let inverted_plane = LumaPlane::Gray(&inverted);

        assert!(patch_ncc(&source_plane, &source_plane, 4, 4, 4, 4, 3) > 0.999_999);
        assert!(patch_ncc(&source_plane, &inverted_plane, 4, 4, 4, 4, 3) < -0.999_999);
    }

    #[test]
    fn stitching_order_uses_stable_filenames_for_equivalent_graphs() {
        let first_images = vec![
            test_image(0, "c.jpg"),
            test_image(1, "a.jpg"),
            test_image(2, "b.jpg"),
        ];
        let first_matches =
            HashMap::from([((1, 2), identity_match(30)), ((0, 2), identity_match(20))]);
        let (first_order, _) = build_stitching_order(&first_images, &first_matches);

        let second_images = vec![
            test_image(0, "b.jpg"),
            test_image(1, "c.jpg"),
            test_image(2, "a.jpg"),
        ];
        let second_matches =
            HashMap::from([((0, 2), identity_match(30)), ((0, 1), identity_match(20))]);
        let (second_order, _) = build_stitching_order(&second_images, &second_matches);

        let first_names = first_order
            .iter()
            .map(|&index| first_images[index].filename.as_str())
            .collect::<Vec<_>>();
        let second_names = second_order
            .iter()
            .map(|&index| second_images[index].filename.as_str())
            .collect::<Vec<_>>();

        assert_eq!(first_names, vec!["a.jpg", "b.jpg", "c.jpg"]);
        assert_eq!(second_names, first_names);
    }

    #[test]
    fn disconnected_stitch_graph_selects_the_largest_component() {
        let images = (0..7)
            .map(|index| test_image(index, &format!("tile-{index}.jpg")))
            .collect::<Vec<_>>();
        let matches = HashMap::from([
            ((0, 1), identity_match(80)),
            // The four-image component must win even though every one of its
            // edges has fewer inliers than the isolated two-image component.
            ((2, 3), identity_match(30)),
            ((3, 4), identity_match(30)),
            ((4, 5), identity_match(30)),
        ]);

        let (order, homographies) = build_stitching_order(&images, &matches);

        assert_eq!(order.len(), 4);
        assert_eq!(
            order.iter().copied().collect::<HashSet<_>>(),
            HashSet::from([2, 3, 4, 5])
        );
        assert_eq!(homographies.len(), 4);
    }

    #[test]
    fn robust_affine_fit_rejects_false_correspondences() {
        let expected = Matrix3::new(1.012, -0.008, 420.0, 0.006, 0.994, -730.0, 0.0, 0.0, 1.0);
        let mut points = Vec::new();
        for row in 0..5 {
            for column in 0..6 {
                let source = Point2::new(
                    300.0 + column as f64 * 1_420.0,
                    240.0 + row as f64 * 1_080.0,
                );
                let mut target = transformed_point(&expected, source).unwrap();
                target.x += ((row * 7 + column * 3) as f64).sin() * 0.18;
                target.y += ((row * 5 + column * 11) as f64).cos() * 0.18;
                points.push((source, target));
            }
        }
        for index in 0..12 {
            points.push((
                Point2::new(index as f64 * 613.0, index as f64 * 277.0),
                Point2::new(8_000.0 - index as f64 * 193.0, 900.0 + index as f64 * 421.0),
            ));
        }

        let fit = robust_transform_fit(&points, 3, 42, estimate_affine)
            .expect("the inlier grid should produce an affine consensus");
        assert!(fit.inlier_indices.len() >= 29);
        assert!(fit.median_error < 0.5);
        let probe = Point2::new(4_500.0, 2_900.0);
        let expected_probe = transformed_point(&expected, probe).unwrap();
        let actual_probe = transformed_point(&fit.transform, probe).unwrap();
        assert!((actual_probe - expected_probe).norm() < 0.5);
    }

    #[test]
    fn large_panorama_auto_alignment_preserves_real_focal_scale() {
        let expected = Matrix3::new(2.35, -0.08, 410.0, 0.08, 2.35, -270.0, 0.0, 0.0, 1.0);
        let points = (0..5)
            .flat_map(|row| {
                (0..6).map(move |column| {
                    let source =
                        Point2::new(100.0 + column as f64 * 140.0, 80.0 + row as f64 * 120.0);
                    (source, transformed_point(&expected, source).unwrap())
                })
            })
            .collect::<Vec<_>>();
        let projective = processing::compute_homography(&points).unwrap();

        let selected = select_large_panorama_transform(&projective, &points, (1_000, 800), false);
        let (scale, _, _) = panorama_model_linear_characteristics(&selected);

        assert!((scale - 2.351).abs() < 0.02, "scale={scale}");
        assert!(median_symmetric_error(&selected, &points) < 0.05);
    }

    #[test]
    fn large_panorama_auto_alignment_keeps_true_translation_stable() {
        let expected = Matrix3::new(1.0, 0.0, 330.0, 0.0, 1.0, -140.0, 0.0, 0.0, 1.0);
        let points = (0..5)
            .flat_map(|row| {
                (0..6).map(move |column| {
                    let source =
                        Point2::new(100.0 + column as f64 * 140.0, 80.0 + row as f64 * 120.0);
                    (source, transformed_point(&expected, source).unwrap())
                })
            })
            .collect::<Vec<_>>();
        let projective = processing::compute_homography(&points).unwrap();

        let selected = select_large_panorama_transform(&projective, &points, (1_000, 800), false);

        assert_eq!(selected, expected);
    }

    #[test]
    fn focus_alignment_accepts_a_real_lens_scale_change() {
        let expected = Matrix3::new(2.35, 0.0, -640.0, 0.0, 2.35, -510.0, 0.0, 0.0, 1.0);
        let points = (0..5)
            .flat_map(|row| {
                (0..6).map(move |column| {
                    let source =
                        Point2::new(100.0 + column as f64 * 140.0, 80.0 + row as f64 * 120.0);
                    (source, transformed_point(&expected, source).unwrap())
                })
            })
            .collect::<Vec<_>>();
        let projective = processing::compute_homography(&points).unwrap();

        let selected =
            select_focus_stack_transform(&projective, &points, (1_000, 800), AlignmentMode::Auto);
        let (scale, _, _) = panorama_model_linear_characteristics(&selected);

        assert!((scale - 2.35).abs() < 0.03, "scale={scale}");
    }

    #[test]
    fn focus_alignment_auto_accepts_broad_handheld_perspective() {
        // A small planar tilt is common when a focus stack is shot by hand.
        // The projective residual is real even though the camera centre does
        // not move enough to classify the stack as a shifted mosaic.
        let expected = Matrix3::new(
            1.0, 0.002, 3.0, -0.001, 1.0, -2.0, 0.000_018, -0.000_012, 1.0,
        );
        let points = (0..6)
            .flat_map(|row| {
                (0..7).map(move |column| {
                    let source =
                        Point2::new(70.0 + column as f64 * 140.0, 60.0 + row as f64 * 110.0);
                    (source, transformed_point(&expected, source).unwrap())
                })
            })
            .collect::<Vec<_>>();
        let projective = processing::compute_homography(&points).unwrap();
        let selected =
            select_focus_stack_transform(&projective, &points, (1_000, 800), AlignmentMode::Auto);
        let error = median_symmetric_error(&selected, &points);
        assert!(error < 0.05, "selected model error={error}");
        assert!(selected[(2, 0)].abs() > 1e-6 || selected[(2, 1)].abs() > 1e-6);
    }

    #[test]
    fn focus_alignment_auto_rejects_compact_projective_stroke_support() {
        // Four or five keypoints on one repeated calligraphic stroke can fit a
        // projective model by accident.  The broad support gate must keep that
        // local warp from bending the whole focus layer.
        let expected = Matrix3::new(1.0, 0.003, 0.0, -0.002, 1.0, 0.0, 0.000_12, -0.000_09, 1.0);
        let points = (0..3)
            .flat_map(|row| {
                (0..4).map(move |column| {
                    let source =
                        Point2::new(470.0 + column as f64 * 18.0, 370.0 + row as f64 * 16.0);
                    (source, transformed_point(&expected, source).unwrap())
                })
            })
            .collect::<Vec<_>>();
        let projective = processing::compute_homography(&points).unwrap();
        let selected =
            select_focus_stack_transform(&projective, &points, (1_000, 800), AlignmentMode::Auto);
        assert!(
            selected[(2, 0)].abs() < 1e-8 && selected[(2, 1)].abs() < 1e-8,
            "compact support unexpectedly selected projective model: {:?}",
            selected
        );
    }

    #[test]
    fn mixed_focal_registration_rejects_a_repeated_texture_bridge() {
        let identity = Matrix3::identity();
        assert!(!mixed_focal_scale_is_plausible(&identity, 35.0, 85.0));
        let expected = Matrix3::new(
            85.0 / 35.0,
            0.0,
            -640.0,
            0.0,
            85.0 / 35.0,
            -510.0,
            0.0,
            0.0,
            1.0,
        );
        assert!(mixed_focal_scale_is_plausible(&expected, 35.0, 85.0));
    }

    #[test]
    fn focus_alignment_does_not_trade_fit_precision_for_a_few_more_inliers() {
        let selected = RobustTransformFit {
            transform: Matrix3::identity(),
            inlier_indices: (0..22).collect(),
            median_error: 2.0,
        };
        let overfit = RobustTransformFit {
            transform: Matrix3::identity(),
            inlier_indices: (0..28).collect(),
            median_error: 2.45,
        };
        let precise = RobustTransformFit {
            transform: Matrix3::identity(),
            inlier_indices: (0..18).collect(),
            median_error: 1.6,
        };

        assert!(!focus_fit_is_competitive(&overfit, &selected));
        assert!(focus_fit_is_competitive(&precise, &selected));
    }

    #[test]
    fn focus_stack_stability_rejects_extrapolated_projective_warp() {
        let stable = Matrix3::new(1.01, -0.01, 180.0, 0.01, 1.01, -90.0, 0.0, 0.0, 1.0);
        let unstable = Matrix3::new(1.0, 0.0, 180.0, 0.0, 1.0, -90.0, 0.000_18, -0.000_12, 1.0);
        assert!(transform_is_stable_for_focus_stack(&stable, (9_504, 6_336)));
        assert!(!transform_is_stable_for_focus_stack(
            &unstable,
            (9_504, 6_336)
        ));
    }

    #[test]
    fn focus_registration_rejects_mirrored_image_orientation() {
        let identity = Matrix3::identity();
        let mirrored = Matrix3::new(-1.0, 0.0, 9_504.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);

        assert!(homography_preserves_focus_orientation(
            &identity,
            (9_504, 6_336)
        ));
        assert!(!homography_preserves_focus_orientation(
            &mirrored,
            (9_504, 6_336)
        ));
    }

    #[test]
    fn generic_focus_registration_samples_frames_without_a_detected_depth_layer() {
        let source = GrayImage::from_fn(320, 240, |x, y| {
            image::Luma([((x * 17 + y * 29 + x * y * 3) % 255) as u8])
        });
        let target = GrayImage::from_fn(320, 240, |x, y| {
            image::Luma([source.get_pixel(x.saturating_sub(5), y)[0]])
        });
        let mut source_info = test_image(0, "source.jpg");
        source_info.width = 320;
        source_info.height = 240;
        source_info.alignment_image = source;
        let mut target_info = test_image(1, "target.jpg");
        target_info.width = 320;
        target_info.height = 240;
        target_info.alignment_image = target;

        let points = collect_dense_focus_region_points(
            &source_info,
            &target_info,
            &Matrix3::new(1.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            Projection::Planar,
        );

        assert!(source_info.foreground_range.is_none());
        assert!(target_info.foreground_range.is_none());
        assert!(points.len() >= 8);
    }

    #[test]
    fn long_edge_consensus_groups_only_distinct_images() {
        let lines = vec![
            FocusHorizontalEdgeLine {
                image_id: 0,
                source_row: 0.20,
                world_x_center: 500.0,
                slope: 0.01,
                intercept: 995.0,
                median_error: 2.0,
            },
            FocusHorizontalEdgeLine {
                image_id: 1,
                source_row: 0.18,
                world_x_center: 700.0,
                slope: 0.011,
                intercept: 994.0,
                median_error: 2.0,
            },
            FocusHorizontalEdgeLine {
                image_id: 2,
                source_row: 0.22,
                world_x_center: 900.0,
                slope: 0.009,
                intercept: 996.0,
                median_error: 2.0,
            },
            FocusHorizontalEdgeLine {
                image_id: 0,
                source_row: 0.40,
                world_x_center: 500.0,
                slope: 0.01,
                intercept: 2_000.0,
                median_error: 2.0,
            },
        ];

        let clusters = focus_horizontal_edge_line_clusters(&lines, 9_504.0);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn long_edge_duplicate_candidates_keep_the_best_fit_per_image() {
        let lines = vec![
            FocusHorizontalEdgeLine {
                image_id: 0,
                source_row: 0.20,
                world_x_center: 500.0,
                slope: 0.01,
                intercept: 995.0,
                median_error: 4.0,
            },
            FocusHorizontalEdgeLine {
                image_id: 0,
                source_row: 0.201,
                world_x_center: 500.0,
                slope: 0.0105,
                intercept: 995.2,
                median_error: 1.5,
            },
            FocusHorizontalEdgeLine {
                image_id: 0,
                source_row: 0.31,
                world_x_center: 500.0,
                slope: 0.01,
                intercept: 1_500.0,
                median_error: 2.0,
            },
        ];

        let deduplicated = deduplicate_focus_horizontal_edge_lines(lines, 9_504.0);
        assert_eq!(deduplicated.len(), 2);
        assert!(
            deduplicated
                .iter()
                .any(|line| (line.median_error - 1.5).abs() < f64::EPSILON)
        );
        assert!(
            deduplicated
                .iter()
                .any(|line| (line.source_row - 0.31).abs() < f64::EPSILON)
        );
    }

    #[test]
    fn vertical_long_edge_consensus_groups_only_distinct_images() {
        let lines = vec![
            FocusVerticalEdgeLine {
                image_id: 0,
                source_column: 0.20,
                world_y_center: 500.0,
                slope: 0.01,
                intercept: 995.0,
                median_error: 2.0,
            },
            FocusVerticalEdgeLine {
                image_id: 1,
                source_column: 0.18,
                world_y_center: 700.0,
                slope: 0.011,
                intercept: 994.0,
                median_error: 2.0,
            },
            FocusVerticalEdgeLine {
                image_id: 2,
                source_column: 0.22,
                world_y_center: 900.0,
                slope: 0.009,
                intercept: 996.0,
                median_error: 2.0,
            },
            FocusVerticalEdgeLine {
                image_id: 0,
                source_column: 0.40,
                world_y_center: 500.0,
                slope: 0.01,
                intercept: 2_000.0,
                median_error: 2.0,
            },
        ];

        let clusters = focus_vertical_edge_line_clusters(&lines, 9_504.0);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn vertical_edge_duplicate_candidates_keep_the_best_fit_per_image() {
        let lines = vec![
            FocusVerticalEdgeLine {
                image_id: 0,
                source_column: 0.20,
                world_y_center: 500.0,
                slope: 0.01,
                intercept: 995.0,
                median_error: 4.0,
            },
            FocusVerticalEdgeLine {
                image_id: 0,
                source_column: 0.201,
                world_y_center: 500.0,
                slope: 0.0105,
                intercept: 995.2,
                median_error: 1.5,
            },
            FocusVerticalEdgeLine {
                image_id: 0,
                source_column: 0.31,
                world_y_center: 500.0,
                slope: 0.01,
                intercept: 1_500.0,
                median_error: 2.0,
            },
        ];

        let deduplicated = deduplicate_focus_vertical_edge_lines(lines, 9_504.0);
        assert_eq!(deduplicated.len(), 2);
        assert!(
            deduplicated
                .iter()
                .any(|line| (line.median_error - 1.5).abs() < f64::EPSILON)
        );
        assert!(
            deduplicated
                .iter()
                .any(|line| (line.source_column - 0.31).abs() < f64::EPSILON)
        );
    }

    #[test]
    fn generated_stack_outputs_and_macos_sidecars_are_not_reused_as_sources() {
        assert!(is_generated_stitch_output("/tmp/DSC08897_FocusStack1.jpg"));
        assert!(is_generated_stitch_output(
            "/tmp/scan_Panorama_20260905.jpg"
        ));
        assert!(is_generated_stitch_output("/tmp/scan_Pano.png"));
        assert!(!is_generated_stitch_output("/tmp/DSC08897.jpg"));
        assert!(!is_generated_stitch_output("/tmp/focusstack_reference.jpg"));
        assert!(is_auxiliary_stitch_file("/tmp/._DSC08897.jpg"));
        assert!(!is_auxiliary_stitch_file("/tmp/DSC08897.jpg"));
    }
}

#[cfg(test)]
mod acceptance_tests {
    use super::*;
    use image::{ImageFormat, Rgb, RgbImage};
    use std::fs;
    use std::path::PathBuf;

    fn ordered_panorama_fixture_paths() -> Vec<PathBuf> {
        let fixture_root = std::env::var("RAW_EDITOR_ORDERED_PANORAMA_DIR")
            .map(PathBuf::from)
            .expect("set RAW_EDITOR_ORDERED_PANORAMA_DIR to the source-image directory");
        let mut paths = fs::read_dir(&fixture_root)
            .expect("ordered panorama fixture directory must be readable")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                !is_auxiliary_stitch_file(&path.to_string_lossy())
                    && !is_generated_stitch_output(&path.to_string_lossy())
                    && path
                        .extension()
                        .and_then(|extension| extension.to_str())
                        .is_some_and(|extension| {
                            matches!(
                                extension.to_ascii_lowercase().as_str(),
                                "jpg" | "jpeg" | "png" | "nef"
                            )
                        })
            })
            .collect::<Vec<_>>();
        paths.sort_by(|left, right| {
            natural_path_cmp(&left.to_string_lossy(), &right.to_string_lossy())
        });
        if std::env::var_os("RAW_EDITOR_ORDERED_PANORAMA_REVERSE_INPUT").is_some() {
            paths.reverse();
        }
        assert!(paths.len() >= 2, "fixture must contain at least two images");
        paths
    }

    #[test]
    #[ignore = "requires an external ordered panorama fixture directory"]
    fn real_ordered_panorama_pair_diagnostics() {
        let alignment_name = std::env::var("RAW_EDITOR_ORDERED_PANORAMA_ALIGNMENT")
            .unwrap_or_else(|_| "auto".to_string());
        let alignment_mode = AlignmentMode::from_wire(&alignment_name.to_ascii_lowercase());
        let blend_mode = match std::env::var("RAW_EDITOR_ORDERED_PANORAMA_BLEND")
            .unwrap_or_else(|_| "panorama".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "focus" => BlendMode::FocusStack,
            _ => BlendMode::Panorama,
        };
        let paths = ordered_panorama_fixture_paths();

        let (max_dimension, max_features) = scalable_alignment_budget(paths.len());
        let brief_pairs = processing::generate_brief_pairs();
        let images = paths
            .par_iter()
            .enumerate()
            .map(|(id, path)| {
                let source = image::open(path).expect("fixture image must decode");
                let (width, height) = source.dimensions();
                let focal_length_35mm = fs::read(path)
                    .ok()
                    .and_then(|bytes| crate::exif_processing::focal_length_35mm_from_bytes(&bytes));
                let (new_width, new_height, scale_factor) =
                    processing::calculate_downscale_dimensions_capped(width, height, max_dimension);
                let alignment_image = source
                    .resize_exact(new_width, new_height, image::imageops::FilterType::Triangle)
                    .to_luma8();
                let features = find_alignment_features(
                    &alignment_image,
                    &brief_pairs,
                    max_features,
                    true,
                    focal_length_35mm,
                );
                let foreground_range = detect_foreground_range(&alignment_image);
                let top_features =
                    find_top_alignment_features(&alignment_image, &brief_pairs, foreground_range);
                let foreground_mask = build_foreground_mask(&alignment_image, foreground_range);
                let horizontal_edge_rows =
                    horizontal_edge_row_candidates(&alignment_image, foreground_range);
                let vertical_edge_columns =
                    detect_vertical_edge_columns(&alignment_image, foreground_range);
                ImageInfo {
                    id,
                    filename: path.to_string_lossy().into_owned(),
                    width,
                    height,
                    alignment_image,
                    full_image: None,
                    scale_factor,
                    focal_length_35mm,
                    overview_reference: false,
                    features,
                    top_features,
                    foreground_range,
                    foreground_mask,
                    horizontal_edge_rows,
                    vertical_edge_columns,
                }
            })
            .collect::<Vec<_>>();
        println!(
            "focus top feature counts: min={} max={} total={}",
            images
                .iter()
                .map(|image| image.top_features.len())
                .min()
                .unwrap_or(0),
            images
                .iter()
                .map(|image| image.top_features.len())
                .max()
                .unwrap_or(0),
            images
                .iter()
                .map(|image| image.top_features.len())
                .sum::<usize>(),
        );
        println!(
            "focus foreground ranges: {}",
            images
                .iter()
                .map(|image| {
                    let name = Path::new(&image.filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy();
                    match image.foreground_range {
                        Some((minimum, maximum)) => format!("{name}=({minimum:.3},{maximum:.3})"),
                        None => format!("{name}=none"),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        );
        println!(
            "focus foreground corner energy: {}",
            images
                .iter()
                .filter_map(|image| {
                    if image.top_features.is_empty() {
                        return None;
                    }
                    let width = image.alignment_image.width() as i32;
                    let height = image.alignment_image.height() as i32;
                    let mut values = image
                        .top_features
                        .iter()
                        .filter_map(|feature| {
                            let x = feature.keypoint.x as i32;
                            let y = feature.keypoint.y as i32;
                            (x >= FOCUS_FOREGROUND_PATCH_RADIUS
                                && y >= FOCUS_FOREGROUND_PATCH_RADIUS
                                && x + FOCUS_FOREGROUND_PATCH_RADIUS < width
                                && y + FOCUS_FOREGROUND_PATCH_RADIUS < height)
                                .then(|| {
                                    gradient_patch_corner_energy(
                                        &LumaPlane::Gray(&image.alignment_image),
                                        x,
                                        y,
                                        FOCUS_FOREGROUND_PATCH_RADIUS,
                                    )
                                })
                        })
                        .filter(|value| value.is_finite())
                        .collect::<Vec<_>>();
                    values.sort_unstable_by(f64::total_cmp);
                    let percentile = |fraction: f64| {
                        values
                            .get(((values.len().saturating_sub(1)) as f64 * fraction) as usize)
                            .copied()
                            .unwrap_or(0.0)
                    };
                    Some(format!(
                        "{} n={} p50={:.2} p75={:.2} p90={:.2} max={:.2}",
                        Path::new(&image.filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy(),
                        values.len(),
                        percentile(0.50),
                        percentile(0.75),
                        percentile(0.90),
                        percentile(1.0),
                    ))
                })
                .collect::<Vec<_>>()
                .join(" | ")
        );
        let candidate_pairs = pairs_to_match_for_images(&images, blend_mode);
        let matches = candidate_pairs
            .par_iter()
            .filter_map(|&(source, target)| {
                let (source_index, target_index, invert_for_storage) =
                    if blend_mode == BlendMode::FocusStack {
                        canonical_match_direction(&images, source, target)
                    } else {
                        (source, target, false)
                    };
                let mut match_info = match_image_pair(
                    &images[source_index],
                    &images[target_index],
                    Projection::Planar,
                    blend_mode,
                    alignment_mode,
                    true,
                    false,
                    false,
                )?;
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
                Some(((source, target), match_info))
            })
            .collect::<HashMap<_, _>>();
        for (&(source, target), match_info) in &matches {
            let Some(source_range) = images[source].foreground_range else {
                continue;
            };
            let Some(target_range) = images[target].foreground_range else {
                continue;
            };
            let active_top_count = match_info
                .top_candidate_points
                .iter()
                .filter(|(source_point, target_point)| {
                    let source_y = source_point.y / images[source].height as f64;
                    let target_y = target_point.y / images[target].height as f64;
                    (source_range.0..=source_range.1).contains(&source_y)
                        && (target_range.0..=target_range.1).contains(&target_y)
                })
                .count();
            let active_dense_count = match_info
                .dense_focus_points
                .iter()
                .filter(|(source_point, target_point)| {
                    let source_y = source_point.y / images[source].height as f64;
                    let target_y = target_point.y / images[target].height as f64;
                    (source_range.0..=source_range.1).contains(&source_y)
                        && (target_range.0..=target_range.1).contains(&target_y)
                })
                .count();
            if active_top_count >= FOCUS_MODEL_MIN_INLIERS
                || active_dense_count >= FOCUS_MODEL_MIN_INLIERS
            {
                println!(
                    "focus active foreground pair {}<->{}: top={} dense={} local={}",
                    Path::new(&images[source].filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    Path::new(&images[target].filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    active_top_count,
                    active_dense_count,
                    focus_match_points_for_region(
                        &images[source],
                        &images[target],
                        match_info,
                        (0.04, 0.34),
                    )
                    .len(),
                );
            }
        }
        let (connected_order, homographies) = if blend_mode == BlendMode::FocusStack {
            build_focus_stack_stitching_order(&images, &matches)
        } else {
            build_stitching_order(&images, &matches)
        };
        if blend_mode == BlendMode::FocusStack {
            let mut band_counts = [0usize; 6];
            let mut band_error_sums = [0.0f64; 6];
            let mut band_error_counts = [0usize; 6];
            for (&(source, target), match_info) in &matches {
                let Some(source_homography) = homographies.get(&images[source].id) else {
                    continue;
                };
                let Some(target_homography) = homographies.get(&images[target].id) else {
                    continue;
                };
                for &(source_point, target_point) in &match_info.points {
                    let source_normalized_y = source_point.y / images[source].height as f64;
                    let target_normalized_y = target_point.y / images[target].height as f64;
                    let band = (((source_normalized_y + target_normalized_y) * 0.5) * 6.0)
                        .floor()
                        .clamp(0.0, 5.0) as usize;
                    band_counts[band] += 1;
                    let source_world = source_homography
                        * nalgebra::Point3::new(source_point.x, source_point.y, 1.0);
                    let target_world = target_homography
                        * nalgebra::Point3::new(target_point.x, target_point.y, 1.0);
                    if source_world.z.abs() >= 1e-8 && target_world.z.abs() >= 1e-8 {
                        let source_world = nalgebra::Point2::new(
                            source_world.x / source_world.z,
                            source_world.y / source_world.z,
                        );
                        let target_world = nalgebra::Point2::new(
                            target_world.x / target_world.z,
                            target_world.y / target_world.z,
                        );
                        band_error_sums[band] += (source_world - target_world).norm();
                        band_error_counts[band] += 1;
                    }
                }
            }
            println!(
                "focus residual bands: counts={band_counts:?} mean_px={:?}",
                band_error_sums
                    .iter()
                    .zip(band_error_counts.iter())
                    .map(|(sum, count)| {
                        if *count == 0 {
                            0.0
                        } else {
                            *sum / *count as f64
                        }
                    })
                    .collect::<Vec<_>>()
            );
            let mut top_candidate_count = 0usize;
            let mut top_local_count = 0usize;
            for (&(source, target), match_info) in &matches {
                let candidates = if !match_info.top_candidate_points.is_empty() {
                    &match_info.top_candidate_points
                } else if match_info.candidate_points.is_empty() {
                    &match_info.points
                } else {
                    &match_info.candidate_points
                };
                top_candidate_count += candidates
                    .iter()
                    .filter(|(source_point, target_point)| {
                        let source_y = source_point.y / images[source].height as f64;
                        let target_y = target_point.y / images[target].height as f64;
                        (0.12..=0.28).contains(&source_y) && (0.12..=0.28).contains(&target_y)
                    })
                    .count();
                top_local_count += focus_match_points_for_region(
                    &images[source],
                    &images[target],
                    match_info,
                    (0.12, 0.28),
                )
                .len();
            }
            println!("focus top candidate/local points: {top_candidate_count}/{top_local_count}");
            let mut dense_offsets = [Vec::<(f64, f64)>::new(), Vec::new(), Vec::new()];
            for (&(source, target), match_info) in &matches {
                if source.abs_diff(target) > 4 {
                    continue;
                }
                let source_image = &images[source];
                let target_image = &images[target];
                for (band_index, y_fraction) in [0.12, 0.20, 0.28].into_iter().enumerate() {
                    for x_fraction in [0.1, 0.3, 0.5, 0.7, 0.9] {
                        let source_x = (source_image.width as f64 * x_fraction).round() as i32;
                        let source_y = (source_image.height as f64 * y_fraction).round() as i32;
                        let predicted = match_info.homography
                            * nalgebra::Point3::new(source_x as f64, source_y as f64, 1.0);
                        if predicted.z.abs() < 1e-8 {
                            continue;
                        }
                        let target_x = (predicted.x / predicted.z).round() as i32;
                        let target_y = (predicted.y / predicted.z).round() as i32;
                        if target_x < 12
                            || target_y < 12
                            || target_x + 12 >= target_image.width as i32
                            || target_y + 12 >= target_image.height as i32
                        {
                            continue;
                        }
                        let Some((best_x, best_y, subpixel_x, subpixel_y)) = refine_patch_position(
                            &LumaPlane::Gray(&source_image.alignment_image),
                            &LumaPlane::Gray(&target_image.alignment_image),
                            source_x,
                            source_y,
                            target_x,
                            target_y,
                            10,
                            20,
                        ) else {
                            continue;
                        };
                        dense_offsets[band_index].push((
                            best_x as f64 + subpixel_x - predicted.x / predicted.z,
                            best_y as f64 + subpixel_y - predicted.y / predicted.z,
                        ));
                    }
                }
            }
            println!(
                "focus dense top offsets: {:?}",
                dense_offsets
                    .iter()
                    .map(|offsets| {
                        let (sum_x, sum_y) = offsets
                            .iter()
                            .fold((0.0, 0.0), |(sum_x, sum_y), (x, y)| (sum_x + x, sum_y + y));
                        if offsets.is_empty() {
                            (0, 0.0, 0.0)
                        } else {
                            (
                                offsets.len(),
                                sum_x / offsets.len() as f64,
                                sum_y / offsets.len() as f64,
                            )
                        }
                    })
                    .collect::<Vec<_>>()
            );
            let top_homographies = optimize_focus_stack_global_homographies_in_region(
                &images,
                &matches,
                &homographies,
                Some((0.12, 0.28)),
            );
            let mut maximum_top_delta = (0.0f64, String::new(), 0.0f64, 0.0f64);
            for &index in connected_order.iter().take(6) {
                let image = &images[index];
                let point =
                    nalgebra::Point3::new(image.width as f64 * 0.5, image.height as f64 * 0.2, 1.0);
                let global = homographies[&image.id] * point;
                let top = top_homographies[&image.id] * point;
                println!(
                    "focus top model {}: delta=({:.1},{:.1})",
                    Path::new(&image.filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    top.x / top.z - global.x / global.z,
                    top.y / top.z - global.y / global.z,
                );
            }
            for image in &images {
                let point =
                    nalgebra::Point3::new(image.width as f64 * 0.5, image.height as f64 * 0.2, 1.0);
                let global = homographies[&image.id] * point;
                let top = top_homographies[&image.id] * point;
                let delta_x = top.x / top.z - global.x / global.z;
                let delta_y = top.y / top.z - global.y / global.z;
                let magnitude = (delta_x * delta_x + delta_y * delta_y).sqrt();
                if magnitude > maximum_top_delta.0 {
                    maximum_top_delta = (
                        magnitude,
                        Path::new(&image.filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned(),
                        delta_x,
                        delta_y,
                    );
                }
            }
            println!("focus top model maximum delta: {maximum_top_delta:?}");
            for (&(source, target), match_info) in &matches {
                let top_count = match_info
                    .points
                    .iter()
                    .filter(|(source_point, target_point)| {
                        source_point.y < images[source].height as f64 * 0.28
                            || target_point.y < images[target].height as f64 * 0.28
                    })
                    .count();
                if top_count >= 8 && (source.abs_diff(target) <= 4 || source == 0) {
                    println!(
                        "focus top-points {}<->{}: {}/{}",
                        Path::new(&images[source].filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy(),
                        Path::new(&images[target].filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy(),
                        top_count,
                        match_info.points.len(),
                    );
                }
            }
            for (position, &index) in connected_order.iter().enumerate() {
                let image = &images[index];
                let homography = homographies
                    .get(&image.id)
                    .expect("focus diagnostic image must have a transform");
                let center = homography
                    * nalgebra::Point3::new(
                        image.width as f64 * 0.5,
                        image.height as f64 * 0.5,
                        1.0,
                    );
                let top_left = homography * nalgebra::Point3::new(0.0, 0.0, 1.0);
                let top_right = homography * nalgebra::Point3::new(image.width as f64, 0.0, 1.0);
                let top_left =
                    nalgebra::Point2::new(top_left.x / top_left.z, top_left.y / top_left.z);
                let top_right =
                    nalgebra::Point2::new(top_right.x / top_right.z, top_right.y / top_right.z);
                println!(
                    "focus frame {position:02} {}: center=({:.1},{:.1}) top_scale={:.6}",
                    Path::new(&image.filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    center.x / center.z,
                    center.y / center.z,
                    (top_right - top_left).norm() / image.width as f64,
                );
            }
        }
        let minimum_features = images
            .iter()
            .map(|image| image.features.len())
            .min()
            .unwrap_or(0);
        let maximum_features = images
            .iter()
            .map(|image| image.features.len())
            .max()
            .unwrap_or(0);
        let mut minimum_x = f64::INFINITY;
        let mut maximum_x = f64::NEG_INFINITY;
        let mut minimum_y = f64::INFINITY;
        let mut maximum_y = f64::NEG_INFINITY;
        for image in &images {
            let homography = homographies
                .get(&image.id)
                .expect("every connected fixture image should have a transform");
            for (x, y) in [
                (0.0, 0.0),
                (image.width as f64, 0.0),
                (image.width as f64, image.height as f64),
                (0.0, image.height as f64),
            ] {
                let mapped = homography * nalgebra::Point3::new(x, y, 1.0);
                assert!(mapped.z.abs() >= 1e-8, "fixture corner must remain finite");
                let mapped_x = mapped.x / mapped.z;
                let mapped_y = mapped.y / mapped.z;
                minimum_x = minimum_x.min(mapped_x);
                maximum_x = maximum_x.max(mapped_x);
                minimum_y = minimum_y.min(mapped_y);
                maximum_y = maximum_y.max(mapped_y);
            }
        }
        let output_width = (maximum_x + (-minimum_x).ceil()).ceil().max(1.0) as u64;
        let output_height = (maximum_y + (-minimum_y).ceil()).ceil().max(1.0) as u64;
        let output_pixels = output_width.saturating_mul(output_height);
        let rgb32f_gib = output_pixels.saturating_mul(12) as f64 / 1024_f64.powi(3);
        let safe_scale =
            memory_safe_panorama_render_scale(output_width as u32, output_height as u32);
        let safe_homographies = scaled_homographies(&homographies, safe_scale);
        let image_refs = connected_order
            .iter()
            .map(|&index| &images[index])
            .collect::<Vec<_>>();
        let (safe_width, safe_height) = stitching::output_canvas_dimensions(
            &image_refs,
            &safe_homographies,
            Projection::Planar,
        );
        println!(
            "ordered {} diagnostics ({alignment_name}): {} images, {} candidate pairs, {} matched pairs, {} connected images, features {}..{}, canvas {}x{} ({} pixels, {:.2} GiB RGB32F), safe {}x{} ({:.1}%)",
            if blend_mode == BlendMode::FocusStack {
                "focus-stack"
            } else {
                "panorama"
            },
            images.len(),
            candidate_pairs.len(),
            matches.len(),
            connected_order.len(),
            minimum_features,
            maximum_features,
            output_width,
            output_height,
            output_pixels,
            rgb32f_gib,
            safe_width,
            safe_height,
            safe_scale * 100.0,
        );
        assert_eq!(connected_order.len(), images.len());
        assert_eq!(homographies.len(), images.len());
    }

    #[test]
    #[ignore = "requires an external ordered panorama fixture directory and renders a large result"]
    fn real_ordered_panorama_full_render() {
        let paths = ordered_panorama_fixture_paths()
            .into_iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let alignment_name = std::env::var("RAW_EDITOR_ORDERED_PANORAMA_ALIGNMENT")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase();
        let alignment_mode = AlignmentMode::from_wire(&alignment_name);
        let blend_mode = match std::env::var("RAW_EDITOR_ORDERED_PANORAMA_BLEND")
            .unwrap_or_else(|_| "panorama".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "focus" => BlendMode::FocusStack,
            _ => BlendMode::Panorama,
        };
        let app = tauri::test::mock_app();
        crate::sidecar_storage::initialize(
            PathBuf::from("/private/tmp/raw-editor-ordered-panorama-sidecars").as_path(),
        )
        .expect("test sidecar storage should initialize once");

        let started = Instant::now();
        let outcome = stitch_images_with_options(
            paths,
            app.handle().clone(),
            alignment_mode,
            blend_mode,
            "test-image-stack-progress",
        )
        .expect("the complete ordered panorama fixture should align and render");
        let (rendered_width, rendered_height) = outcome.image.dimensions();
        let full_canvas_width = outcome.full_canvas_width;
        let full_canvas_height = outcome.full_canvas_height;
        let render_scale = outcome.render_scale;
        let rendered_pixels = u64::from(rendered_width) * u64::from(rendered_height);
        assert!(rendered_width > 0 && rendered_height > 0);
        // Panorama renders are memory-bounded by the production path. Focus stacks
        // intentionally keep the source resolution so the exported TIFF does not
        // lose detail; their large canvas is therefore expected in this visual
        // acceptance fixture too.
        if blend_mode == BlendMode::Panorama {
            assert!(rendered_pixels <= MAX_IN_MEMORY_PANORAMA_PIXELS + 1_000_000);
        }
        assert!(outcome.full_canvas_width >= rendered_width);
        assert!(outcome.full_canvas_height >= rendered_height);

        let preview_scale = (4_000.0 / f64::from(rendered_width.max(rendered_height))).min(1.0);
        let preview_width = (f64::from(rendered_width) * preview_scale).round().max(1.0) as u32;
        let preview_height = (f64::from(rendered_height) * preview_scale)
            .round()
            .max(1.0) as u32;
        let preview = crate::image_processing::downscale_f32_image(
            &outcome.image,
            preview_width,
            preview_height,
        );
        let preview_path = std::env::var_os("RAW_EDITOR_ORDERED_PANORAMA_PREVIEW")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from("/private/tmp/raw-editor-ordered-panorama-preview.jpg")
            });
        preview
            .save_with_format(&preview_path, ImageFormat::Jpeg)
            .expect("ordered panorama preview should be writable");
        let full_output_path =
            std::env::var_os("RAW_EDITOR_ORDERED_PANORAMA_OUTPUT").map(PathBuf::from);
        if let Some(output_path) = full_output_path.as_ref() {
            let canonical = crate::image_stack::canonicalize_image_stack_result(outcome.image);
            crate::image_stack::write_srgb_jpeg(&canonical, output_path)
                .expect("full-resolution ordered panorama JPEG should be writable");
        }
        println!(
            "ordered panorama full render ({alignment_name}): {}x{} from full canvas {}x{} at {:.1}% in {:.2?}\npreview: {}{}",
            rendered_width,
            rendered_height,
            full_canvas_width,
            full_canvas_height,
            render_scale * 100.0,
            started.elapsed(),
            preview_path.display(),
            full_output_path
                .as_ref()
                .map(|path| format!("\nfull: {}", path.display()))
                .unwrap_or_default(),
        );
    }

    #[test]
    #[ignore = "temporary fixture diagnostics"]
    fn temporary_pair_probe() {
        let root = std::env::var("RAW_EDITOR_ORDERED_PANORAMA_DIR").expect("fixture dir");
        crate::sidecar_storage::initialize(
            PathBuf::from("/private/tmp/raw-editor-temporary-pair-probe-sidecars").as_path(),
        )
        .expect("temporary probe sidecar storage should initialize");
        let mut paths = fs::read_dir(&root)
            .expect("fixture dir")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("jpg"))
            })
            .collect::<Vec<_>>();
        paths.sort_by(|a, b| natural_path_cmp(&a.to_string_lossy(), &b.to_string_lossy()));
        let (max_dimension, max_features) = scalable_alignment_budget(paths.len());
        let pairs = processing::generate_brief_pairs();
        let images = paths
            .iter()
            .enumerate()
            .map(|(id, path)| {
                let bytes = fs::read(path).unwrap();
                let focal = crate::exif_processing::focal_length_35mm_from_bytes(&bytes);
                let source = crate::image_loader::load_base_image_from_bytes(
                    &bytes,
                    &path.to_string_lossy(),
                    false,
                    &AppSettings::default(),
                    None,
                )
                .unwrap();
                let (w, h, sf) = processing::calculate_downscale_dimensions_capped(
                    source.width(),
                    source.height(),
                    max_dimension,
                );
                let alignment = source
                    .resize_exact(w, h, image::imageops::FilterType::Triangle)
                    .to_luma8();
                let features =
                    find_alignment_features(&alignment, &pairs, max_features, true, focal);
                ImageInfo {
                    id,
                    filename: path.to_string_lossy().into_owned(),
                    width: source.width(),
                    height: source.height(),
                    alignment_image: alignment,
                    full_image: Some(source.to_rgb32f()),
                    scale_factor: sf,
                    focal_length_35mm: focal,
                    overview_reference: false,
                    features,
                    top_features: Vec::new(),
                    foreground_range: None,
                    foreground_mask: None,
                    horizontal_edge_rows: Vec::new(),
                    vertical_edge_columns: Vec::new(),
                }
            })
            .collect::<Vec<_>>();
        for i in 0..images.len() {
            for j in i + 1..images.len() {
                let a = &images[i];
                let b = &images[j];
                let raw = processing::match_features_with_ratio(
                    &a.features,
                    &b.features,
                    SCALABLE_MATCH_RATIO_THRESHOLD,
                );
                if raw.len() < 5 {
                    continue;
                }
                let kp1 = a.features.iter().map(|f| f.keypoint).collect::<Vec<_>>();
                let kp2 = b.features.iter().map(|f| f.keypoint).collect::<Vec<_>>();
                let pts = raw
                    .iter()
                    .map(|m| {
                        (
                            Point2::new(
                                kp1[m.index1].x as f64 * a.scale_factor,
                                kp1[m.index1].y as f64 * a.scale_factor,
                            ),
                            Point2::new(
                                kp2[m.index2].x as f64 * b.scale_factor,
                                kp2[m.index2].y as f64 * b.scale_factor,
                            ),
                        )
                    })
                    .collect::<Vec<_>>();
                let r = processing::find_homography_ransac_points_stable(
                    &pts,
                    FULL_RES_RANSAC_INLIER_THRESHOLD,
                );
                let (n, spread) = r
                    .as_ref()
                    .map(|(_, idx)| {
                        let p = idx.iter().map(|&k| pts[k]).collect::<Vec<_>>();
                        (
                            idx.len(),
                            match_has_spatial_support(&p, a.dimensions(), b.dimensions()),
                        )
                    })
                    .unwrap_or((0, false));
                if n >= 5 {
                    let accepted = match_image_pair(
                        a,
                        b,
                        Projection::Planar,
                        BlendMode::Panorama,
                        AlignmentMode::Auto,
                        true,
                        false,
                        false,
                    )
                    .is_some();
                    println!(
                        "PAIR {} {} focal={:?}/{:?} raw={} inliers={} spatial={} accepted={}",
                        Path::new(&a.filename)
                            .file_name()
                            .unwrap()
                            .to_string_lossy(),
                        Path::new(&b.filename)
                            .file_name()
                            .unwrap()
                            .to_string_lossy(),
                        a.focal_length_35mm,
                        b.focal_length_35mm,
                        raw.len(),
                        n,
                        spread,
                        accepted
                    );
                }
            }
        }
    }

    fn synthetic_texture_pixel(x: u32, y: u32) -> Rgb<u8> {
        let mut hash = x.wrapping_mul(0x9E37_79B9) ^ y.wrapping_mul(0x85EB_CA6B);
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x7FEB_352D);
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(0x846C_A68B);
        hash ^= hash >> 16;
        let value = hash as u8;
        Rgb([
            value,
            value.rotate_left(3) ^ ((x.wrapping_add(y) * 13) as u8),
            value.rotate_right(2) ^ ((x.wrapping_mul(3).wrapping_add(y * 5)) as u8),
        ])
    }

    #[test]
    #[ignore = "manual deterministic 200-image scalable stitching smoke test"]
    fn synthetic_two_hundred_image_scalable_path() {
        const WIDTH: u32 = 320;
        const HEIGHT: u32 = 240;
        const HORIZONTAL_STEP: u32 = 2;

        let fixture_dir = tempfile::tempdir().expect("temporary stack fixture should be writable");
        let mut paths = Vec::with_capacity(MAX_STITCH_SOURCE_IMAGES);
        for index in 0..MAX_STITCH_SOURCE_IMAGES {
            let horizontal_offset = index as u32 * HORIZONTAL_STEP;
            let image = RgbImage::from_fn(WIDTH, HEIGHT, |x, y| {
                synthetic_texture_pixel(x + horizontal_offset, y)
            });
            let path = fixture_dir.path().join(format!("tile-{index:03}.png"));
            image
                .save_with_format(&path, ImageFormat::Png)
                .expect("synthetic stack tile should be writable");
            paths.push(path.to_string_lossy().into_owned());
        }

        crate::sidecar_storage::initialize(&fixture_dir.path().join("sidecars"))
            .expect("test sidecar storage should initialize once");
        let app = tauri::test::mock_app();
        let started = Instant::now();
        let result = stitch_images_with_options(
            paths,
            app.handle().clone(),
            AlignmentMode::Position,
            BlendMode::Panorama,
            "test-image-stack-progress",
        )
        .expect("the 200 overlapping tiles should stitch through the scalable path");
        let image = result.image;

        let expected_width = WIDTH + (MAX_STITCH_SOURCE_IMAGES as u32 - 1) * HORIZONTAL_STEP;
        assert!(image.width().abs_diff(expected_width) <= 2);
        assert!(image.height().abs_diff(HEIGHT) <= 2);
        println!(
            "synthetic 200-image stack: {}x{} in {:.2?}",
            image.width(),
            image.height(),
            started.elapsed()
        );
    }

    #[test]
    #[ignore = "requires the real three-image fixture and writes a large temporary output"]
    fn real_three_image_panorama_fixture() {
        let fixture_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../src/assets/test")
            .canonicalize()
            .expect("test fixture directory must exist");
        let paths = ["DSC01721_1.tif", "DSC01700.tif", "DSC01728.tif"]
            .iter()
            .map(|name| fixture_root.join(name).to_string_lossy().into_owned())
            .collect();
        let alignment_name = std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_ALIGNMENT")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase();
        let alignment_mode = AlignmentMode::from_wire(&alignment_name);
        let app = tauri::test::mock_app();
        crate::sidecar_storage::initialize(
            PathBuf::from("/private/tmp/raw-editor-stack-sidecars").as_path(),
        )
        .expect("test sidecar storage should initialize once");
        let result = stitch_images_with_options(
            paths,
            app.handle().clone(),
            alignment_mode,
            BlendMode::Panorama,
            "test-image-stack-progress",
        )
        .expect("the three real overlapping images should stitch");
        let image = result.image;

        assert!(image.width() > 0 && image.height() > 0);
        let output_path = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(format!(
                    "/private/tmp/raw-editor-three-image-panorama-{alignment_name}.tiff"
                ))
            });
        image
            .save_with_format(&output_path, ImageFormat::Tiff)
            .expect("full-resolution TIFF result should be writable");

        let preview = crate::image_processing::downscale_f32_image(&image, 1800, 1800);
        let preview_path = output_path.with_extension("jpg");
        preview
            .save_with_format(&preview_path, ImageFormat::Jpeg)
            .expect("panorama preview should be writable");
        println!(
            "three-image panorama result: {}x{}\nfull: {}\npreview: {}",
            image.width(),
            image.height(),
            output_path.display(),
            preview_path.display()
        );
    }

    #[test]
    #[ignore = "requires user-provided focus-stack paths and writes a large temporary output"]
    fn real_focus_stack_fixture_from_env() {
        let encoded_paths = std::env::var_os("RAW_EDITOR_FOCUS_STACK_PATHS")
            .expect("RAW_EDITOR_FOCUS_STACK_PATHS must contain platform-separated image paths");
        let paths: Vec<String> = std::env::split_paths(&encoded_paths)
            .map(|path| path.to_string_lossy().into_owned())
            .collect();
        assert!(
            paths.len() >= 2,
            "at least two focus-stack paths are required"
        );

        let alignment_name = std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_ALIGNMENT")
            .unwrap_or_else(|_| "auto".to_string())
            .to_ascii_lowercase();
        let alignment_mode = AlignmentMode::from_wire(&alignment_name);
        let blend_mode = match std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_BLEND")
            .unwrap_or_else(|_| "focus-stack".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "panorama" => BlendMode::Panorama,
            _ => BlendMode::FocusStack,
        };
        let app = tauri::test::mock_app();
        crate::sidecar_storage::initialize(
            PathBuf::from("/private/tmp/raw-editor-stack-sidecars").as_path(),
        )
        .expect("test sidecar storage should initialize once");
        let result = stitch_images_with_options(
            paths,
            app.handle().clone(),
            alignment_mode,
            blend_mode,
            "test-image-stack-progress",
        )
        .expect("the provided focus-stack images should align and blend");

        let output_path = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_OUTPUT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(format!(
                    "/private/tmp/raw-editor-focus-stack-{alignment_name}.tiff"
                ))
            });
        let canonical = crate::image_stack::canonicalize_image_stack_result(result.image);
        let preview_only = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_PREVIEW_ONLY").is_some();
        if !preview_only {
            crate::image_stack::write_srgb_tiff(&canonical, &output_path)
                .expect("full-resolution color-managed TIFF result should be writable");
        }
        let preview_edge = std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_PREVIEW_EDGE")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|&value| value >= 256)
            .unwrap_or(1800);
        let preview = canonical.resize(
            preview_edge,
            preview_edge,
            image::imageops::FilterType::Lanczos3,
        );
        let preview_path = output_path.with_extension("preview.jpg");
        preview
            .save_with_format(&preview_path, ImageFormat::Jpeg)
            .expect("focus-stack preview should be writable");
        if std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_CROPS").is_some() {
            let (width, height) = canonical.dimensions();
            let crop_width = (width / 3).max(1);
            let crop_height = (height / 2).max(1);
            for (name, x, y) in [
                ("top-left", 0, 0),
                ("top-center", width.saturating_sub(crop_width) / 2, 0),
                ("top-right", width.saturating_sub(crop_width), 0),
                ("bottom-left", 0, height.saturating_sub(crop_height)),
                (
                    "bottom-center",
                    width.saturating_sub(crop_width) / 2,
                    height.saturating_sub(crop_height),
                ),
                (
                    "bottom-right",
                    width.saturating_sub(crop_width),
                    height.saturating_sub(crop_height),
                ),
            ] {
                let crop = canonical
                    .crop_imm(x, y, crop_width.min(width - x), crop_height.min(height - y))
                    .resize(
                        preview_edge,
                        preview_edge,
                        image::imageops::FilterType::Lanczos3,
                    );
                crop.save_with_format(
                    output_path.with_file_name(format!(
                        "{}.{}.jpg",
                        output_path
                            .file_stem()
                            .and_then(|stem| stem.to_str())
                            .unwrap_or("focus-stack"),
                        name
                    )),
                    ImageFormat::Jpeg,
                )
                .expect("focus-stack crop preview should be writable");
            }
        }
        let app_jpeg_path = output_path.with_extension("app.jpg");
        if !preview_only {
            crate::image_stack::write_srgb_jpeg(&canonical, &app_jpeg_path)
                .expect("full-resolution application JPEG should be writable");
        }

        if !preview_only
            && let Some(reference_path) =
                std::env::var_os("RAW_EDITOR_FOCUS_STACK_REFERENCE_JPEG").map(PathBuf::from)
        {
            let actual = fs::read(&app_jpeg_path)
                .expect("read the full-resolution application JPEG for comparison");
            let reference =
                fs::read(&reference_path).expect("read the user-provided Photoshop reference JPEG");
            assert!(
                actual == reference,
                "application JPEG does not match the Photoshop reference byte for byte: {} ({} bytes) vs {} ({} bytes)",
                app_jpeg_path.display(),
                actual.len(),
                reference_path.display(),
                reference.len()
            );
            println!(
                "Photoshop parity: byte-identical to {}",
                reference_path.display()
            );
        }
        println!(
            "focus-stack result: {}x{}\nfull: {}\napp JPEG: {}\npreview: {}",
            canonical.width(),
            canonical.height(),
            output_path.display(),
            app_jpeg_path.display(),
            preview_path.display()
        );
    }

    #[test]
    #[ignore = "reads an alignment cache emitted by real_focus_stack_fixture_from_env"]
    fn real_focus_stack_alignment_cache_contract() {
        let cache_path = std::env::var_os("RAW_EDITOR_STACK_ALIGNMENT_CACHE")
            .map(PathBuf::from)
            .expect("RAW_EDITOR_STACK_ALIGNMENT_CACHE must point to an alignment cache");
        let bytes = fs::read(&cache_path).expect("alignment cache should be readable");
        let cache: serde_json::Value =
            serde_json::from_slice(&bytes).expect("alignment cache should be valid JSON");
        assert_eq!(
            cache.get("schema").and_then(|value| value.as_u64()),
            Some(2)
        );
        let sources = cache
            .get("sources")
            .and_then(|value| value.as_array())
            .expect("alignment cache should contain source records");
        assert!(sources.len() >= 2, "alignment cache should contain a stack");
        let homographies = cache
            .get("global_homographies")
            .and_then(|value| value.as_object())
            .expect("alignment cache should contain global homographies");
        for source in sources {
            let id = source
                .get("id")
                .and_then(|value| value.as_u64())
                .expect("source record should have an id");
            let matrix = homographies
                .get(&id.to_string())
                .and_then(|value| value.as_array())
                .expect("every source should have a global homography");
            assert_eq!(matrix.len(), 9, "homography {id} should have nine values");
        }
        let matches = cache
            .get("matches")
            .and_then(|value| value.as_array())
            .expect("alignment cache should contain pair diagnostics");
        assert!(
            !matches.is_empty(),
            "alignment cache should retain at least one verified pair"
        );
        for pair in matches {
            assert!(
                pair.get("source_id")
                    .and_then(|value| value.as_u64())
                    .is_some()
                    && pair
                        .get("target_id")
                        .and_then(|value| value.as_u64())
                        .is_some(),
                "pair diagnostics should identify both sources"
            );
            assert_eq!(
                pair.get("direct_homography")
                    .and_then(|value| value.as_array())
                    .map(Vec::len),
                Some(9),
                "pair diagnostics should retain the measured transform"
            );
            assert!(
                pair.get("global_disagreement_px")
                    .and_then(|value| value.as_f64())
                    .is_some(),
                "pair diagnostics should report pose disagreement"
            );
        }
        let canvas = cache
            .get("canvas")
            .and_then(|value| value.as_object())
            .expect("alignment cache should contain canvas dimensions");
        assert!(
            canvas
                .get("width")
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
                > 0
        );
        assert!(
            canvas
                .get("height")
                .and_then(|value| value.as_u64())
                .unwrap_or(0)
                > 0
        );
        println!(
            "alignment cache contract passed: {} sources, {} bytes, {}",
            sources.len(),
            bytes.len(),
            cache_path.display()
        );
    }
}
