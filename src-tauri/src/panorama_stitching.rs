use crate::app_settings::{AppSettings, load_settings_for_runtime};
use crate::app_state::AppState;
use crate::file_management::{parse_virtual_path, read_file_mapped};
use base64::{Engine as _, engine::general_purpose};
use image::ImageFormat;
use image::{DynamicImage, GenericImageView, GrayImage, Rgb32FImage};
use nalgebra::Matrix3;
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
const FULL_RES_REFINEMENT_THRESHOLD: f64 = 8.0;
const MATCH_REFINE_PATCH_RADIUS: i32 = 6;
const MATCH_REFINE_SEARCH_RADIUS: i32 = 10;
const FOCUS_MODEL_INLIER_THRESHOLD: f64 = 6.0;
const FOCUS_MODEL_RANSAC_ITERATIONS: usize = 1_500;
const FOCUS_MODEL_MIN_INLIERS: usize = 8;
const SCALABLE_STACK_THRESHOLD: usize = 30;
const LARGE_STACK_NEIGHBOR_WINDOW: usize = 4;
const SCALABLE_LOW_TEXTURE_FEATURE_TARGET: usize = 96;
const SCALABLE_LOW_TEXTURE_FAST_THRESHOLD: u8 = 7;
const SCALABLE_LOW_TEXTURE_NMS_RADIUS: f32 = 10.0;
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

fn memory_safe_panorama_render_scale(width: u32, height: u32) -> f64 {
    let pixels = u64::from(width).saturating_mul(u64::from(height));
    if pixels <= MAX_IN_MEMORY_PANORAMA_PIXELS || pixels == 0 {
        1.0
    } else {
        (MAX_IN_MEMORY_PANORAMA_PIXELS as f64 / pixels as f64).sqrt()
    }
}

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
    let (projected_homography, projected_inlier_indices) = if stable_four_point_solver {
        processing::find_homography_ransac_points_stable(
            &projected_match_points,
            FULL_RES_RANSAC_INLIER_THRESHOLD,
        )
    } else {
        processing::find_homography_ransac_points(
            &projected_match_points,
            FULL_RES_RANSAC_INLIER_THRESHOLD,
        )
    }?;
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
                blend_mode == BlendMode::FocusStack,
            )
            .unwrap_or(projected_match_points[index])
        })
        .collect::<Vec<_>>();
    let refinement_threshold =
        if source_image.full_image.is_some() && target_image.full_image.is_some() {
            FULL_RES_REFINEMENT_THRESHOLD
        } else {
            FULL_RES_REFINEMENT_THRESHOLD
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
    })
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
                })
            })
            .collect::<Vec<Result<ImageInfo, String>>>()
    };

    let image_data_results = if scalable_stack {
        let available_threads = std::thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(1);
        let pool = ThreadPoolBuilder::new()
            .num_threads(available_threads.clamp(1, 3))
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
    let (ordered_indices, global_homographies) =
        build_stitching_order(&image_data, &pairwise_matches);

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

    let (full_canvas_width, full_canvas_height) = stitching::output_canvas_dimensions(
        &stitched_images_info,
        &global_homographies,
        projection,
    );
    if full_canvas_width == 0 || full_canvas_height == 0 {
        return Err("The aligned panorama canvas is empty or invalid.".to_string());
    }
    let render_scale = if blend_mode == BlendMode::Panorama {
        memory_safe_panorama_render_scale(full_canvas_width, full_canvas_height)
    } else {
        1.0
    };
    let scaled_global_homographies;
    let render_homographies = if render_scale < 1.0 {
        scaled_global_homographies = scaled_homographies(&global_homographies, render_scale);
        let (render_width, render_height) = stitching::output_canvas_dimensions(
            &stitched_images_info,
            &scaled_global_homographies,
            projection,
        );
        let render_percentage = render_scale * 100.0;
        let message = format!(
            "Large canvas {full_canvas_width}x{full_canvas_height}; rendering a memory-safe {render_width}x{render_height} result ({render_percentage:.0}%)."
        );
        println!("{message}");
        let _ = app_handle.emit(progress_event, &message);
        &scaled_global_homographies
    } else {
        &global_homographies
    };

    let start_time = Instant::now();
    let _ = app_handle.emit(progress_event, "Warping and blending images...");
    println!("Warping and blending full-resolution images with progressive optimal seams...");

    let mut load_full_image = |image: &ImageInfo| {
        if let Some(full_image) = retained_full_images.remove(&image.id) {
            Ok(full_image)
        } else {
            load_prepared_stack_source(&image.filename, &settings).map(|source| source.to_rgb32f())
        }
    };
    let panorama = match blend_mode {
        BlendMode::Panorama => stitching::progressive_seam_stitcher(
            &stitched_images_info,
            render_homographies,
            projection,
            app_handle.clone(),
            progress_event,
            &mut load_full_image,
        ),
        BlendMode::FocusStack => stitching::focus_stack_stitcher(
            &stitched_images_info,
            render_homographies,
            projection,
            app_handle.clone(),
            progress_event,
            &mut load_full_image,
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
    let retains_consensus = candidate.inlier_indices.len() + 2 >= selected.inlier_indices.len();
    let materially_more_precise = candidate.median_error + 0.35 < selected.median_error
        && candidate.median_error <= selected.median_error * 0.82;
    candidate.inlier_indices.len() > selected.inlier_indices.len()
        || (retains_consensus && materially_more_precise)
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
    let translation_points: Vec<_> = translation_inliers
        .iter()
        .map(|&index| points[index])
        .collect();
    let mut selected = RobustTransformFit {
        transform: translation,
        inlier_indices: translation_inliers,
        median_error: median_symmetric_error(&translation, &translation_points),
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
    let allow_projective = matches!(
        alignment_mode,
        AlignmentMode::Perspective | AlignmentMode::Cylindrical | AlignmentMode::Spherical
    );
    if allow_projective
        && transform_is_stable_for_focus_stack(projective, source_dimensions)
        && projective_error + 0.75 < selected.median_error
        && projective_error <= selected.median_error * 0.72
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
    println!(
        "  - Focus alignment selected {selected_name}: median symmetric error {:.3}px with {} inliers (similarity {similarity_summary}, affine {affine_summary}, projective {:.3}px/{})",
        selected.median_error,
        selected.inlier_indices.len(),
        projective_error,
        points.len()
    );
    selected.transform
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
    let mut values1 = Vec::with_capacity(((radius * 2 + 1) * (radius * 2 + 1)) as usize);
    let mut values2 = Vec::with_capacity(values1.capacity());
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let Some(value1) = image1.luma_at(center1_x + dx, center1_y + dy) else {
                return f64::NEG_INFINITY;
            };
            let Some(value2) = image2.luma_at(center2_x + dx, center2_y + dy) else {
                return f64::NEG_INFINITY;
            };
            values1.push(value1);
            values2.push(value2);
        }
    }

    let mean1 = values1.iter().sum::<f64>() / values1.len() as f64;
    let mean2 = values2.iter().sum::<f64>() / values2.len() as f64;
    let mut covariance = 0.0;
    let mut variance1 = 0.0;
    let mut variance2 = 0.0;
    for (value1, value2) in values1.iter().zip(values2.iter()) {
        let centered1 = value1 - mean1;
        let centered2 = value2 - mean2;
        covariance += centered1 * centered2;
        variance1 += centered1 * centered1;
        variance2 += centered2 * centered2;
    }
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
        }
    }

    fn identity_match(inliers: usize) -> MatchInfo {
        MatchInfo {
            homography: Matrix3::identity(),
            inliers,
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
    fn large_stack_alignment_budget_tightens_as_source_count_grows() {
        assert_eq!(scalable_alignment_budget(64), (2_400, 1_600));
        assert_eq!(scalable_alignment_budget(128), (1_800, 1_100));
        assert_eq!(scalable_alignment_budget(200), (1_536, 800));
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
    fn focus_stack_stability_rejects_extrapolated_projective_warp() {
        let stable = Matrix3::new(1.01, -0.01, 180.0, 0.01, 1.01, -90.0, 0.0, 0.0, 1.0);
        let unstable = Matrix3::new(1.0, 0.0, 180.0, 0.0, 1.0, -90.0, 0.000_18, -0.000_12, 1.0);
        assert!(transform_is_stable_for_focus_stack(&stable, (9_504, 6_336)));
        assert!(!transform_is_stable_for_focus_stack(
            &unstable,
            (9_504, 6_336)
        ));
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
                path.extension()
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
                ImageInfo {
                    id,
                    filename: path.to_string_lossy().into_owned(),
                    width,
                    height,
                    alignment_image,
                    full_image: None,
                    scale_factor,
                    features,
                }
            })
            .collect::<Vec<_>>();

        let candidate_pairs = pairs_to_match_for_images(&images);
        let matches = candidate_pairs
            .par_iter()
            .filter_map(|&(source, target)| {
                match_image_pair(
                    &images[source],
                    &images[target],
                    Projection::Planar,
                    BlendMode::Panorama,
                    alignment_mode,
                    true,
                    false,
                )
                .map(|match_info| ((source, target), match_info))
            })
            .collect::<HashMap<_, _>>();
        let (connected_order, homographies) = build_stitching_order(&images, &matches);
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
            "ordered panorama diagnostics ({alignment_name}): {} images, {} candidate pairs, {} matched pairs, {} connected images, features {}..{}, canvas {}x{} ({} pixels, {:.2} GiB RGB32F), safe {}x{} ({:.1}%)",
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
            BlendMode::Panorama,
            "test-image-stack-progress",
        )
        .expect("the complete ordered panorama fixture should align and render");
        let (rendered_width, rendered_height) = outcome.image.dimensions();
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
        println!(
            "ordered panorama full render ({alignment_name}): {}x{} from full canvas {}x{} at {:.1}% in {:.2?}\npreview: {}",
            rendered_width,
            rendered_height,
            outcome.full_canvas_width,
            outcome.full_canvas_height,
            outcome.render_scale * 100.0,
            started.elapsed(),
            preview_path.display(),
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
