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
use crate::panorama_utils::stitching::{Projection, project_point};
use crate::panorama_utils::{processing, stitching};

pub const BRIEF_DESCRIPTOR_SIZE: usize = 256;
pub type Descriptor = [u8; BRIEF_DESCRIPTOR_SIZE / 8];
const FULL_RES_RANSAC_INLIER_THRESHOLD: f64 = 12.0;
const FULL_RES_REFINEMENT_THRESHOLD: f64 = 2.5;
const MATCH_REFINE_PATCH_RADIUS: i32 = 14;
const MATCH_REFINE_SEARCH_RADIUS: i32 = 14;
const FOCUS_MODEL_INLIER_THRESHOLD: f64 = 6.0;
const FOCUS_MODEL_RANSAC_ITERATIONS: usize = 1_500;
const FOCUS_MODEL_MIN_INLIERS: usize = 8;
const FOCUS_LOCAL_MODEL_MIN_INLIERS: usize = 6;
const FOCUS_SHIFTED_MOSAIC_MOTION_RATIO: f64 = 0.015;
const FOCUS_SEQUENCE_LINK_WINDOW: usize = 2;
const FOCUS_SEQUENCE_GAP_PENALTY: f64 = 96.0;
const FOCUS_GLOBAL_MAX_POINTS_PER_EDGE: usize = 256;
const FOCUS_GLOBAL_MAX_ITERATIONS: usize = 12;
const FOCUS_GLOBAL_HUBER_THRESHOLD: f64 = 0.0012;
const FOCUS_GLOBAL_PRIOR_WEIGHT: f64 = 0.01;
const FOCUS_GLOBAL_PROJECTIVE_PRIOR_WEIGHT: f64 = 0.5;
const FOCUS_GLOBAL_DAMPING: f64 = 1e-6;
const FOCUS_GLOBAL_MAX_LINEAR_ADJUSTMENT: f64 = 0.12;
const FOCUS_GLOBAL_MAX_TRANSLATION_ADJUSTMENT: f64 = 0.18;
const FOCUS_GLOBAL_MAX_PROJECTIVE_ADJUSTMENT: f64 = 0.02;
const FOCUS_LOCAL_MODEL_MAX_DISPLACEMENT_RATIO: f64 = 0.08;
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
const MAX_SCALABLE_PREPARATION_WORKERS: usize = 6;
const PREPARATION_RAM_PER_WORKER_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_IN_MEMORY_PANORAMA_PIXELS: u64 = 240_000_000;
const MAX_STITCH_SOURCE_IMAGES: usize = 200;

#[derive(Debug, Clone, Copy)]
pub struct KeyPoint {
    pub x: u32,
    pub y: u32,
}

pub struct Feature {
    pub keypoint: KeyPoint,
    pub descriptor: Descriptor,
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
    pub features: Vec<Feature>,
    pub top_features: Vec<Feature>,
    pub foreground_range: Option<(f64, f64)>,
    pub foreground_mask: Option<GrayImage>,
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
}

#[derive(Clone)]
pub(crate) struct FocusLayerWarp {
    pub(crate) bands: Vec<FocusWarpBand>,
}

pub(crate) struct StitchOutcome {
    pub image: DynamicImage,
    pub full_canvas_width: u32,
    pub full_canvas_height: u32,
    pub render_scale: f64,
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
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: image.foreground_range,
        foreground_mask: image.foreground_mask.clone(),
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

fn pairs_to_match_for_images(images: &[ImageInfo]) -> Vec<(usize, usize)> {
    if images.len() <= SCALABLE_STACK_THRESHOLD {
        return pairs_to_match(images.len());
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
    pairs
}

fn find_alignment_features(
    alignment_image: &GrayImage,
    brief_pairs: &[(nalgebra::Point2<i32>, nalgebra::Point2<i32>)],
    max_features: usize,
    scalable_stack: bool,
) -> Vec<Feature> {
    let mut features = processing::find_features(alignment_image, brief_pairs);
    if scalable_stack && features.len() < SCALABLE_LOW_TEXTURE_FEATURE_TARGET {
        let normalized = processing::normalize_grayscale(alignment_image);
        let fallback = processing::find_features_tuned(
            &normalized,
            brief_pairs,
            SCALABLE_LOW_TEXTURE_FAST_THRESHOLD,
            SCALABLE_LOW_TEXTURE_NMS_RADIUS,
        );
        if fallback.len() > features.len() {
            features = fallback;
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

fn load_prepared_stack_source(
    filename: &str,
    settings: &AppSettings,
) -> Result<DynamicImage, String> {
    let file_data = read_file_mapped(Path::new(filename))
        .map_err(|error| format!("Failed to read image {filename}: {error}"))?;
    let mut image = crate::image_loader::load_base_image_from_bytes(
        &file_data, filename, false, settings, None,
    )
    .map_err(|error| format!("Failed to load image {filename}: {error}"))?;
    if is_raw_file(filename) {
        apply_cpu_default_raw_processing(&mut image);
    }
    Ok(image)
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

fn match_image_pair(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    projection: Projection,
    blend_mode: BlendMode,
    alignment_mode: AlignmentMode,
    stable_four_point_solver: bool,
    log_match: bool,
) -> Option<MatchInfo> {
    let features1 = &source_image.features;
    let features2 = &target_image.features;
    let initial_matches = processing::match_features(features1, features2);
    if initial_matches.len() < processing::MIN_INLIERS_FOR_CONNECTION {
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
    let solver_indices = if background_match_indices.len() >= processing::MIN_INLIERS_FOR_CONNECTION
    {
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
    let (projected_homography, projected_inlier_indices) = if stable_four_point_solver {
        processing::find_homography_ransac_points_stable(
            &solver_points,
            FULL_RES_RANSAC_INLIER_THRESHOLD,
        )
    } else {
        processing::find_homography_ransac_points(&solver_points, FULL_RES_RANSAC_INLIER_THRESHOLD)
    }?;
    let projected_inlier_indices = projected_inlier_indices
        .into_iter()
        .map(|index| solver_indices[index])
        .collect::<Vec<_>>();
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
            refine_match_point_from_homography(
                source_image,
                target_image,
                keypoints1[matched.index1],
                keypoints2[matched.index2],
                projection,
                &projected_homography,
                false,
            )
            .unwrap_or(projected_match_points[index])
        })
        .collect::<Vec<_>>();
    let model_refinement_threshold = if blend_mode == BlendMode::FocusStack {
        FOCUS_MODEL_INLIER_THRESHOLD
    } else {
        FULL_RES_REFINEMENT_THRESHOLD
    };
    let refinement_threshold =
        if source_image.full_image.is_some() && target_image.full_image.is_some() {
            model_refinement_threshold
        } else {
            model_refinement_threshold
                .max(source_image.scale_factor.max(target_image.scale_factor) * 1.5)
        };
    let refined_homography = refine_homography_inliers(&mut inlier_points, refinement_threshold)?;
    let inlier_count = inlier_points.len();
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
        let reprojection_error = symmetric_reprojection_rmse(&refined_homography, &inlier_points);
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
    let homography = if alignment_mode == AlignmentMode::Position {
        estimate_translation(&inlier_points)
    } else if blend_mode == BlendMode::FocusStack {
        select_focus_stack_transform(
            &refined_homography,
            &inlier_points,
            source_image.dimensions(),
            alignment_mode,
        )
    } else if stable_four_point_solver && alignment_mode == AlignmentMode::Auto {
        select_large_panorama_transform(&inlier_points, log_match)
    } else {
        refined_homography
    };
    Some(MatchInfo {
        homography,
        inliers: inlier_count,
        points: inlier_points,
        candidate_points: projected_match_points,
        top_candidate_points,
        dense_focus_points,
        foreground_feature_points,
    })
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
    let source_x_fractions = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
    let patch_radius = 8;
    let search_radius = 14;
    for source_y_fraction in source_y_fractions {
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

fn select_large_panorama_transform(
    points: &[(nalgebra::Point2<f64>, nalgebra::Point2<f64>)],
    log_selection: bool,
) -> Matrix3<f64> {
    // A tiny scale/rotation bias compounds catastrophically across a 100-200
    // image chain. Large automatic mosaics are normally captured from one
    // camera angle, so keep their global pose translation-stable. Users who
    // intentionally changed viewpoint can still select Perspective explicitly.
    let selected = estimate_translation(points);
    if log_selection {
        println!(
            "  - Large-panorama alignment selected translation-stable: median symmetric error {:.3}px with {} inliers",
            median_symmetric_error(&selected, points),
            points.len(),
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
    let panorama_image = state
        .panorama_result
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| {
            "No panorama image found in memory to save. It might have already been saved."
                .to_string()
        })?;

    let (first_path, _) = parse_virtual_path(&first_path_str);
    let parent_dir = first_path
        .parent()
        .ok_or_else(|| "Could not determine parent directory of the first image.".to_string())?;
    let stem = first_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("panorama");

    let (output_filename, image_to_save): (String, DynamicImage) =
        if panorama_image.color().has_alpha() {
            (
                format!("{}_Pano.png", stem),
                DynamicImage::ImageRgba8(panorama_image.to_rgba8()),
            )
        } else if panorama_image.as_rgb32f().is_some() {
            (format!("{}_Pano.tiff", stem), panorama_image)
        } else {
            (
                format!("{}_Pano.png", stem),
                DynamicImage::ImageRgb8(panorama_image.to_rgb8()),
            )
        };

    let output_path = parent_dir.join(output_filename);

    image_to_save
        .save(&output_path)
        .map_err(|e| format!("Failed to save panorama image: {}", e))?;

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

    let settings = load_settings_for_runtime(&app_handle).unwrap_or_default();

    let start_time = Instant::now();
    let _ = app_handle.emit(progress_event, "Loading and preparing images...");
    let focus_stack = blend_mode == BlendMode::FocusStack;
    let scalable_stack = image_paths.len() > SCALABLE_STACK_THRESHOLD;
    let (alignment_max_dimension, alignment_max_features) =
        scalable_alignment_budget(image_paths.len());
    println!(
        "Loading and preparing images ({} mode)...",
        if scalable_stack {
            "bounded-memory"
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
                let dynamic_image = load_prepared_stack_source(filename, &settings)?;
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
                    scalable_stack,
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
                    features,
                    top_features,
                    foreground_range,
                    foreground_mask,
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
    let _ = app_handle.emit(progress_event, "Finding image matches...");
    println!(
        "Finding {} matches (in parallel)...",
        if scalable_stack {
            "ordered-neighbor"
        } else {
            "all pairwise"
        }
    );
    let projection = alignment_mode.projection_for(blend_mode);
    let mut pairwise_matches: HashMap<(usize, usize), MatchInfo> = HashMap::new();

    let pairs_to_check = pairs_to_match_for_images(&image_data);
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
                scalable_stack,
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
    println!(
        "Pairwise matching completed in {:.2?}\n",
        start_time.elapsed()
    );

    if pairwise_matches.is_empty() {
        return Err(if scalable_stack {
            "No overlap was found between nearby images. For large stacks, arrange the source layers in shooting order and make sure consecutive images overlap."
                .to_string()
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
    let focus_layer_warp = (blend_mode == BlendMode::FocusStack)
        .then(|| build_focus_layer_warp(&image_data, &pairwise_matches, &global_homographies));

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
        let warning_msg = format!(
            "{} image(s) could not be aligned with the selected set.",
            unstitched_count
        );
        println!("{}", warning_msg);
        let _ = app_handle.emit(progress_event, warning_msg);
        return Err(if scalable_stack {
            format!(
                "Could not align all selected images; {unstitched_count} image(s) were not connected to nearby layers. Reorder the sources into shooting order and retry."
            )
        } else {
            format!(
                "Could not align all selected images; {unstitched_count} image(s) were not connected."
            )
        });
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
    let render_scale = if blend_mode == BlendMode::Panorama {
        memory_safe_panorama_render_scale(full_canvas_width, full_canvas_height)
    } else {
        1.0
    };
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
                .map(|source| source_to_render_rgb32f(source, image.width, image.height))
        }
    };
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
            focus_layer_warp.as_ref(),
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
    })
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
        && transform_is_stable_for_focus_stack(&model.transform, source_dimensions)
        && focus_fit_is_competitive(model, &selected)
    {
        selected = model.clone();
        selected_name = "similarity";
    }

    let affine = robust_transform_fit(points, 3, 0x9FB2_1C65_1E98_DF25, estimate_affine);
    if let Some(model) = affine.as_ref()
        && transform_is_stable_for_focus_stack(&model.transform, source_dimensions)
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
    // A moving focus stack is also a planar scan/mosaic. Auto may use the
    // projective fit for that case, while a genuinely fixed-camera stack keeps
    // the lower-DOF model that is safer against defocus-driven false matches.
    let shifted_mosaic = focus_stack_motion_is_shifted_mosaic(points, source_dimensions);
    let allow_projective = explicit_projective || shifted_mosaic;
    if allow_projective
        && transform_is_stable_for_focus_stack(projective, source_dimensions)
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
    motion.is_finite()
        && image_scale > 1.0
        && motion > image_scale * FOCUS_SHIFTED_MOSAIC_MOTION_RATIO
}

fn refine_match_point_from_homography(
    image1: &ImageInfo,
    image2: &ImageInfo,
    keypoint1: KeyPoint,
    keypoint2: KeyPoint,
    projection: Projection,
    homography: &Matrix3<f64>,
    prefer_feature_center: bool,
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
            MATCH_REFINE_PATCH_RADIUS,
            MATCH_REFINE_SEARCH_RADIUS,
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
            4,
            6,
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
) -> Option<Matrix3<f64>> {
    if points.len() < processing::MIN_INLIERS_FOR_CONNECTION {
        return None;
    }

    // Patch correlation can occasionally lock onto a nearby repeated stroke or
    // texture. Starting the least-squares refinement from every correlation lets
    // a coherent minority bend a projective model enough that none of those bad
    // points exceeds the later residual threshold. Re-run a deterministic robust
    // fit at full resolution before optimizing all remaining correspondences.
    let (_, robust_inlier_indices) =
        processing::find_homography_ransac_points_stable(points, refinement_threshold)?;
    if robust_inlier_indices.len() < processing::MIN_INLIERS_FOR_CONNECTION {
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
        if refined_points.len() < processing::MIN_INLIERS_FOR_CONNECTION
            || refined_points.len() == points.len()
        {
            break;
        }
        *points = refined_points;
        homography = processing::compute_homography(points)?;
    }
    Some(homography)
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
    if images.is_empty() {
        return (vec![], HashMap::new());
    }
    let n = images.len();
    if n < 2 {
        let mut homographies = HashMap::new();
        if n == 1 {
            homographies.insert(0, Matrix3::identity());
        }
        return ((0..n).collect(), homographies);
    }

    let mut edges = Vec::new();
    for (&(i, j), m) in matches {
        edges.push((m.inliers, i, j));
    }
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
            .cmp(&left.0)
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

    let start_node = (0..n)
        .filter(|i| mst_adj.contains_key(i))
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

    // Focus stacking is a source-order operation: later layers must compete
    // with the pixels already selected from earlier layers. For a moving scan,
    // using the panorama MST here can reorder frames around high-texture areas
    // and then compound unrelated transforms. Keep the user/import order and
    // choose a strong nearby link for each frame instead.
    let ordered_indices = (0..images.len()).collect::<Vec<_>>();
    let mut global_homographies = HashMap::new();
    global_homographies.insert(ordered_indices[0], Matrix3::identity());

    for position in 1..ordered_indices.len() {
        let current = ordered_indices[position];
        let preferred_gap = position.min(FOCUS_SEQUENCE_LINK_WINDOW);
        let mut best_link: Option<(f64, usize, usize, Matrix3<f64>)> = None;

        for gap in 1..=preferred_gap {
            let previous = ordered_indices[position - gap];
            let Some((inliers, current_to_previous)) =
                focus_stack_link_transform(matches, current, previous)
            else {
                continue;
            };
            let score =
                inliers as f64 - (gap.saturating_sub(1) as f64) * FOCUS_SEQUENCE_GAP_PENALTY;
            let should_replace =
                best_link
                    .as_ref()
                    .is_none_or(|(best_score, best_inliers, best_gap, _)| {
                        score > *best_score
                            || (score == *best_score
                                && (inliers > *best_inliers
                                    || (inliers == *best_inliers && gap < *best_gap)))
                    });
            if should_replace {
                best_link = Some((score, inliers, gap, current_to_previous));
            }
        }

        // The normal scalable matcher considers a four-frame window. If the
        // preferred one/two-frame path has a missing edge, use the remaining
        // nearby links before falling back to the generic graph order.
        if best_link.is_none() {
            for gap in (preferred_gap + 1)..=position.min(LARGE_STACK_NEIGHBOR_WINDOW) {
                let previous = ordered_indices[position - gap];
                let Some((inliers, current_to_previous)) =
                    focus_stack_link_transform(matches, current, previous)
                else {
                    continue;
                };
                let score =
                    inliers as f64 - (gap.saturating_sub(1) as f64) * FOCUS_SEQUENCE_GAP_PENALTY;
                let should_replace =
                    best_link
                        .as_ref()
                        .is_none_or(|(best_score, best_inliers, best_gap, _)| {
                            score > *best_score
                                || (score == *best_score
                                    && (inliers > *best_inliers
                                        || (inliers == *best_inliers && gap < *best_gap)))
                        });
                if should_replace {
                    best_link = Some((score, inliers, gap, current_to_previous));
                }
            }
        }

        let Some((_, _, gap, current_to_previous)) = best_link else {
            return build_stitching_order(images, matches);
        };
        let previous = ordered_indices[position - gap];
        let Some(previous_global) = global_homographies.get(&previous).copied() else {
            return build_stitching_order(images, matches);
        };
        global_homographies.insert(current, previous_global * current_to_previous);
    }

    let global_homographies =
        optimize_focus_stack_global_homographies(images, matches, &global_homographies);
    (ordered_indices, global_homographies)
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

type FocusPose = [f64; 8];

#[derive(Clone, Copy)]
struct FocusGlobalObservation {
    source_index: usize,
    target_index: usize,
    source: Point2<f64>,
    target: Point2<f64>,
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
    let source_supported =
        (source_span_x >= 0.10 && source_span_y >= 0.035) || source_area >= 0.006;
    let target_supported =
        (target_span_x >= 0.10 && target_span_y >= 0.035) || target_area >= 0.006;
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
        let point_count = selected_points.len().min(FOCUS_GLOBAL_MAX_POINTS_PER_EDGE);
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
    })
}

fn focus_global_robust_cost(
    poses: &[FocusPose],
    initial_poses: &[FocusPose],
    observations: &[FocusGlobalObservation],
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
        cost += if magnitude <= FOCUS_GLOBAL_HUBER_THRESHOLD {
            0.5 * magnitude * magnitude
        } else {
            FOCUS_GLOBAL_HUBER_THRESHOLD * (magnitude - 0.5 * FOCUS_GLOBAL_HUBER_THRESHOLD)
        };
    }
    for (image_index, pose) in poses.iter().enumerate().skip(1) {
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
        let robust_weight = if magnitude <= FOCUS_GLOBAL_HUBER_THRESHOLD {
            1.0
        } else {
            FOCUS_GLOBAL_HUBER_THRESHOLD / magnitude
        };

        for (residual_component, source_jacobian, target_jacobian) in [
            (residual.x, source_x_jacobian, target_x_jacobian),
            (residual.y, source_y_jacobian, target_y_jacobian),
        ] {
            let mut jacobian_entries = Vec::with_capacity(16);
            if observation.source_index > 0 {
                let base = (observation.source_index - 1) * 8;
                for parameter in 0..8 {
                    jacobian_entries.push((base + parameter, source_jacobian[parameter]));
                }
            }
            if observation.target_index > 0 {
                let base = (observation.target_index - 1) * 8;
                for parameter in 0..8 {
                    jacobian_entries.push((base + parameter, -target_jacobian[parameter]));
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

    for (image_index, pose) in poses.iter().enumerate().skip(1) {
        let base = (image_index - 1) * 8;
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

fn optimize_focus_stack_global_homographies(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
) -> HashMap<usize, Matrix3<f64>> {
    optimize_focus_stack_global_homographies_in_region_mode(
        images,
        matches,
        initial_homographies,
        None,
        false,
    )
}

fn optimize_focus_stack_global_homographies_in_region(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    normalized_y_range: Option<(f64, f64)>,
) -> HashMap<usize, Matrix3<f64>> {
    optimize_focus_stack_global_homographies_in_region_mode(
        images,
        matches,
        initial_homographies,
        normalized_y_range,
        true,
    )
}

fn optimize_focus_stack_global_homographies_in_generic_region(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    normalized_y_range: (f64, f64),
) -> HashMap<usize, Matrix3<f64>> {
    optimize_focus_stack_global_homographies_in_region_mode(
        images,
        matches,
        initial_homographies,
        Some(normalized_y_range),
        false,
    )
}

fn optimize_focus_stack_global_homographies_in_region_mode(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    normalized_y_range: Option<(f64, f64)>,
    use_foreground_region_matches: bool,
) -> HashMap<usize, Matrix3<f64>> {
    if images.len() < 2 {
        return initial_homographies.clone();
    }
    let Some(reference) = images.first() else {
        return initial_homographies.clone();
    };
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
    initial_poses[0] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
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
    let mut current_cost = focus_global_robust_cost(&poses, &initial_poses, &observations);
    if !current_cost.is_finite() {
        return initial_homographies.clone();
    }
    let initial_poses_valid =
        focus_global_poses_are_valid(&poses, images, reference, coordinate_scale);
    println!("  - Focus global registration initial geometry valid: {initial_poses_valid}");
    let initial_cost = current_cost;
    let mut accepted_iterations = 0;

    for _ in 0..FOCUS_GLOBAL_MAX_ITERATIONS {
        let (normal, gradient) =
            focus_global_normal_equations(&poses, &initial_poses, &observations);
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
            for (image_index, pose) in candidate.iter_mut().enumerate().skip(1) {
                let base = (image_index - 1) * 8;
                for parameter in 0..8 {
                    pose[parameter] += step * delta[base + parameter];
                }
            }
            if candidate
                .iter()
                .enumerate()
                .skip(1)
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
            let candidate_cost =
                focus_global_robust_cost(&candidate, &initial_poses, &observations);
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
        });
    }

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
        });
    }
    FocusLayerWarp { bands }
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
        let use_candidate = candidate.try_inverse().is_some()
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
        let Some(correction) =
            estimate_vertical_foreground_correction(&samples, top_line, bottom_line)
        else {
            continue;
        };
        let corrected = correction * base_homography;
        if corrected.try_inverse().is_none() {
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
            features: Vec::new(),
            top_features: Vec::new(),
            foreground_range: None,
            foreground_mask: None,
        }
    }

    fn identity_match(inliers: usize) -> MatchInfo {
        MatchInfo {
            homography: Matrix3::identity(),
            inliers,
            points: Vec::new(),
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
        }
    }

    fn translation_match(dx: f64, dy: f64, inliers: usize) -> MatchInfo {
        MatchInfo {
            homography: Matrix3::new(1.0, 0.0, dx, 0.0, 1.0, dy, 0.0, 0.0, 1.0),
            inliers,
            points: Vec::new(),
            candidate_points: Vec::new(),
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
        let pairs = pairs_to_match_for_images(&images);
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
    fn natural_path_order_compares_numeric_filename_runs() {
        let mut paths = ["tile-10.jpg", "tile-2.jpg", "tile-001.jpg", "tile-1.jpg"];
        paths.sort_by(|left, right| natural_path_cmp(left, right));

        assert_eq!(
            paths,
            ["tile-1.jpg", "tile-001.jpg", "tile-2.jpg", "tile-10.jpg"]
        );
    }

    #[test]
    fn focus_stack_order_preserves_input_sequence_over_a_high_inlier_mst() {
        let images = vec![
            test_image(0, "DSC08854.jpg"),
            test_image(1, "DSC08855.jpg"),
            test_image(2, "DSC08856.jpg"),
            test_image(3, "DSC08857.jpg"),
        ];
        let matches = HashMap::from([
            ((0, 1), translation_match(100.0, 0.0, 30)),
            ((1, 2), translation_match(100.0, 0.0, 35)),
            ((2, 3), translation_match(100.0, 0.0, 25)),
            // The generic panorama MST would prefer these long links and can
            // traverse the source layers in a non-capture order.
            ((0, 2), translation_match(200.0, 0.0, 400)),
            ((0, 3), translation_match(300.0, 0.0, 350)),
        ]);

        let (order, homographies) = build_focus_stack_stitching_order(&images, &matches);

        assert_eq!(order, vec![0, 1, 2, 3]);
        assert_eq!(homographies[&0], Matrix3::identity());
        assert_eq!(homographies[&1][(0, 2)], -100.0);
        assert_eq!(homographies[&2][(0, 2)], -200.0);
        assert_eq!(homographies[&3][(0, 2)], -300.0);
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
    }

    #[test]
    fn large_stack_alignment_budget_tightens_as_source_count_grows() {
        assert_eq!(scalable_alignment_budget(64), (2_400, 1_600));
        assert_eq!(scalable_alignment_budget(128), (1_800, 1_100));
        assert_eq!(scalable_alignment_budget(200), (1_536, 800));
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

        let refined = refine_homography_inliers(&mut points, FULL_RES_REFINEMENT_THRESHOLD)
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
                                "jpg" | "jpeg" | "png"
                            )
                        })
            })
            .collect::<Vec<_>>();
        paths.sort_by(|left, right| {
            natural_path_cmp(&left.to_string_lossy(), &right.to_string_lossy())
        });
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
                let (new_width, new_height, scale_factor) =
                    processing::calculate_downscale_dimensions_capped(width, height, max_dimension);
                let alignment_image = source
                    .resize_exact(new_width, new_height, image::imageops::FilterType::Triangle)
                    .to_luma8();
                let features =
                    find_alignment_features(&alignment_image, &brief_pairs, max_features, true);
                let foreground_range = detect_foreground_range(&alignment_image);
                let top_features =
                    find_top_alignment_features(&alignment_image, &brief_pairs, foreground_range);
                let foreground_mask = build_foreground_mask(&alignment_image, foreground_range);
                ImageInfo {
                    id,
                    filename: path.to_string_lossy().into_owned(),
                    width,
                    height,
                    alignment_image,
                    full_image: None,
                    scale_factor,
                    features,
                    top_features,
                    foreground_range,
                    foreground_mask,
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
        let candidate_pairs = pairs_to_match_for_images(&images);
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
        assert!(rendered_pixels <= MAX_IN_MEMORY_PANORAMA_PIXELS + 1_000_000);
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
        let app = tauri::test::mock_app();
        crate::sidecar_storage::initialize(
            PathBuf::from("/private/tmp/raw-editor-stack-sidecars").as_path(),
        )
        .expect("test sidecar storage should initialize once");
        let result = stitch_images_with_options(
            paths,
            app.handle().clone(),
            alignment_mode,
            BlendMode::FocusStack,
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
        crate::image_stack::write_srgb_tiff(&canonical, &output_path)
            .expect("full-resolution color-managed TIFF result should be writable");
        let preview = canonical.resize(1800, 1800, image::imageops::FilterType::Lanczos3);
        let preview_path = output_path.with_extension("preview.jpg");
        preview
            .save_with_format(&preview_path, ImageFormat::Jpeg)
            .expect("focus-stack preview should be writable");
        let app_jpeg_path = output_path.with_extension("app.jpg");
        crate::image_stack::write_srgb_jpeg(&canonical, &app_jpeg_path)
            .expect("full-resolution application JPEG should be writable");

        if let Some(reference_path) =
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
}
