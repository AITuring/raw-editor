use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
#[cfg(not(target_os = "android"))]
use std::io::{BufWriter, Seek, SeekFrom};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use image::codecs::{jpeg::JpegEncoder, png::PngEncoder, tiff::TiffEncoder};
use image::{
    DynamicImage, GenericImageView, GrayImage, ImageBuffer, ImageEncoder, ImageFormat, Luma,
    RgbaImage, imageops,
};
#[cfg(not(target_os = "android"))]
use image::{Pixel, Rgba};
use jxl_encoder::{
    ImageMetadata as JxlImageMetadata, LosslessConfig, LossyConfig, PixelLayout,
    api::{calibrated_jxl_quality, quality_to_distance},
};
#[cfg(not(target_os = "android"))]
use little_exif::{
    endian::Endian as ExifEndian, ifd::ExifTagGroup, metadata::Metadata as ExifMetadata,
};
#[cfg(not(target_os = "android"))]
use mozjpeg::{
    ColorSpace as MozjpegColorSpace, Compress as MozjpegCompressor, Marker as MozjpegMarker,
};
#[cfg(not(target_os = "android"))]
use png::{BitDepth, ColorType as PngColorType, Encoder as PngStreamEncoder, Info as PngInfo};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::Emitter;
use tauri::Manager;
use tempfile::NamedTempFile;
#[cfg(not(target_os = "android"))]
use tiff::Directory as TiffDirectory;
#[cfg(not(target_os = "android"))]
use tiff::encoder::{
    DirectoryEncoder as TiffDirectoryEncoder, TiffEncoder as StreamingTiffEncoder,
    TiffKindStandard, colortype::RGB16,
};
#[cfg(not(target_os = "android"))]
use tiff::tags::{Tag as TiffTag, Type as TiffType};
#[cfg(not(target_os = "android"))]
use zenresize::{Filter as StreamingResizeFilter, PixelDescriptor, ResizeConfig, StreamingResize};

use crate::AppState;
use crate::app_state::SharedMaskBitmap;
use crate::exif_processing;
use crate::file_management::{
    generate_filename_from_template, parse_virtual_path, read_file_mapped,
};
use crate::formats::is_raw_file;
#[cfg(not(target_os = "android"))]
use crate::gpu_processing::process_and_stream_rgba_rows;
use crate::gpu_processing::reclaim_gpu_resources_after_export;
use crate::image_loader::{
    composite_patches_on_image, load_and_composite, load_base_image_from_bytes,
};
use crate::image_processing::{
    AllAdjustments, Crop, GeometryWarpRows, GpuContext, RenderRequest, downscale_f32_image,
    get_all_adjustments_from_json, get_geometry_params_from_json, get_or_init_gpu_context,
    is_geometry_identity, process_and_get_dynamic_image, resolve_tonemapper_override_from_handle,
};
use crate::lut_processing::{
    Lut, convert_image_to_cube_lut, generate_identity_lut_image, get_or_load_lut,
};
use crate::mask_generation::{MaskDefinition, generate_mask_bitmap};
use crate::render_strategy::RenderTier;

use crate::cache_utils::{calculate_full_job_hash, calculate_transform_hash};
use crate::color_management::srgb_v4_profile;
use crate::{
    apply_all_transformations, generate_transformed_preview, get_cached_or_generate_mask,
    hydrate_adjustments, load_settings, resolve_warped_image_for_masks,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub enum ResizeMode {
    LongEdge,
    ShortEdge,
    Width,
    Height,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ResizeOptions {
    pub mode: ResizeMode,
    pub value: u32,
    pub dont_enlarge: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ExportSettings {
    pub jpeg_quality: u8,
    pub resize: Option<ResizeOptions>,
    pub keep_metadata: bool,
    #[serde(default)]
    pub metadata_overrides: Option<exif_processing::ExportMetadataOverrides>,
    #[serde(default)]
    pub preserve_timestamps: bool,
    pub strip_gps: bool,
    #[serde(default = "default_embed_color_profile")]
    pub embed_color_profile: bool,
    pub filename_template: Option<String>,
    pub watermark: Option<WatermarkSettings>,
    #[serde(default)]
    pub export_masks: bool,
    #[serde(default)]
    pub preserve_folders: bool,
}

const fn default_embed_color_profile() -> bool {
    true
}

#[derive(Clone)]
pub(crate) enum ExportAdjustmentsMode {
    UseSidecars {
        active_path: Option<String>,
        active_adjustments: Option<Value>,
    },
    GlobalOverride(Value),
}

impl ExportAdjustmentsMode {
    fn normalize_active_path(self) -> Self {
        match self {
            Self::UseSidecars {
                active_path,
                active_adjustments,
            } => Self::UseSidecars {
                active_path: active_path.map(|path| {
                    path.rsplit_once("?vc=")
                        .map(|(physical_path, _)| physical_path.to_string())
                        .unwrap_or(path)
                }),
                active_adjustments,
            },
            Self::GlobalOverride(adjustments) => Self::GlobalOverride(adjustments),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub enum WatermarkAnchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct WatermarkSettings {
    pub path: String,
    pub anchor: WatermarkAnchor,
    pub scale: f32,
    pub spacing: f32,
    pub opacity: f32,
}

pub(crate) const DEFAULT_WATERMARK_PATH: &str = "builtin://default-watermark";
const DEFAULT_WATERMARK_BYTES: &[u8] = include_bytes!("../../src/assets/default-watermark.png");
const MAX_WATERMARK_SOURCE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_WATERMARK_SOURCE_PIXELS: u64 = 100_000_000;
const MAX_WATERMARK_SOURCE_EDGE: u32 = 32_768;
const STORED_WATERMARK_EDGE: u32 = 4_096;

struct PreparedWatermark {
    image: RgbaImage,
    x: i64,
    y: i64,
}

fn load_watermark_image(path: &str) -> Result<DynamicImage, String> {
    if path == DEFAULT_WATERMARK_PATH {
        return image::load_from_memory_with_format(DEFAULT_WATERMARK_BYTES, ImageFormat::Png)
            .map_err(|error| format!("Failed to decode the built-in watermark image: {error}"));
    }

    image::open(path).map_err(|error| format!("Failed to open watermark image: {error}"))
}

fn safe_watermark_stem(source_path: &Path) -> String {
    let stem = source_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("watermark");
    let sanitized: String = stem
        .chars()
        .filter_map(|character| {
            if character.is_alphanumeric() || matches!(character, '-' | '_') {
                Some(character)
            } else if character.is_whitespace() {
                Some('-')
            } else {
                None
            }
        })
        .take(48)
        .collect();

    if sanitized.is_empty() {
        "watermark".to_string()
    } else {
        sanitized
    }
}

fn import_watermark_image_impl(
    source_path: &Path,
    watermark_directory: &Path,
) -> Result<PathBuf, String> {
    let metadata = fs::metadata(source_path)
        .map_err(|error| format!("Failed to inspect the selected watermark image: {error}"))?;
    if !metadata.is_file() {
        return Err("The selected watermark path is not a file.".to_string());
    }
    if metadata.len() > MAX_WATERMARK_SOURCE_BYTES {
        return Err("The selected watermark image exceeds the 64 MiB input limit.".to_string());
    }

    let source_file = fs::File::open(source_path)
        .map_err(|error| format!("Failed to open the selected watermark image: {error}"))?;
    let mut source_bytes = Vec::new();
    source_file
        .take(MAX_WATERMARK_SOURCE_BYTES + 1)
        .read_to_end(&mut source_bytes)
        .map_err(|error| format!("Failed to read the selected watermark image: {error}"))?;
    if source_bytes.len() as u64 > MAX_WATERMARK_SOURCE_BYTES {
        return Err("The selected watermark image exceeds the 64 MiB input limit.".to_string());
    }

    let (source_width, source_height) = image::ImageReader::new(Cursor::new(&source_bytes))
        .with_guessed_format()
        .map_err(|error| format!("Failed to detect the selected watermark format: {error}"))?
        .into_dimensions()
        .map_err(|error| format!("Failed to read the selected watermark dimensions: {error}"))?;
    let source_pixels = u64::from(source_width)
        .checked_mul(u64::from(source_height))
        .ok_or_else(|| {
            "The selected watermark dimensions overflow the supported range.".to_string()
        })?;
    if source_pixels > MAX_WATERMARK_SOURCE_PIXELS
        || source_width > MAX_WATERMARK_SOURCE_EDGE
        || source_height > MAX_WATERMARK_SOURCE_EDGE
    {
        return Err("The selected watermark dimensions exceed the supported limit.".to_string());
    }

    let source_image = image::load_from_memory(&source_bytes)
        .map_err(|error| format!("Failed to decode the selected watermark image: {error}"))?;
    let stored_image = if source_width.max(source_height) > STORED_WATERMARK_EDGE {
        source_image.resize(
            STORED_WATERMARK_EDGE,
            STORED_WATERMARK_EDGE,
            imageops::FilterType::Lanczos3,
        )
    } else {
        source_image
    };

    let mut encoded_png = Vec::new();
    stored_image
        .write_to(&mut Cursor::new(&mut encoded_png), ImageFormat::Png)
        .map_err(|error| format!("Failed to encode the selected watermark image: {error}"))?;

    fs::create_dir_all(watermark_directory)
        .map_err(|error| format!("Failed to create the watermark library: {error}"))?;
    let content_hash = blake3::hash(&encoded_png).to_hex().to_string();
    let destination = watermark_directory.join(format!(
        "{}-{}.png",
        safe_watermark_stem(source_path),
        content_hash
    ));

    let mut temporary = NamedTempFile::new_in(watermark_directory)
        .map_err(|error| format!("Failed to create a temporary watermark asset: {error}"))?;
    temporary
        .write_all(&encoded_png)
        .map_err(|error| format!("Failed to write the watermark asset: {error}"))?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| format!("Failed to sync the watermark asset: {error}"))?;
    match temporary.persist_noclobber(&destination) {
        Ok(_) => Ok(destination),
        Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = fs::read(&destination).map_err(|read_error| {
                format!("Failed to verify the existing watermark asset: {read_error}")
            })?;
            if blake3::hash(&existing).to_hex().to_string() == content_hash {
                Ok(destination)
            } else {
                Err("The stored watermark asset failed its content-hash check.".to_string())
            }
        }
        Err(error) => Err(format!(
            "Failed to publish the watermark asset: {}",
            error.error
        )),
    }
}

#[tauri::command]
pub async fn import_watermark_image(
    path: String,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    let source_path = fs::canonicalize(PathBuf::from(path))
        .map_err(|error| format!("Failed to resolve the selected watermark image: {error}"))?;
    if !app_handle.asset_protocol_scope().is_allowed(&source_path) {
        return Err("Select the watermark image with the application file picker.".to_string());
    }
    let watermark_directory = app_handle
        .path()
        .app_data_dir()
        .map_err(|error| error.to_string())?
        .join("watermarks");

    tauri::async_runtime::spawn_blocking(move || {
        import_watermark_image_impl(&source_path, &watermark_directory)
            .map(|stored_path| stored_path.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| format!("Watermark import task failed: {error}"))?
}

fn prepare_watermark(
    base_w: u32,
    base_h: u32,
    watermark_settings: &WatermarkSettings,
) -> Result<Option<PreparedWatermark>, String> {
    let watermark_img = load_watermark_image(&watermark_settings.path)?;

    let base_min_dim = base_w.min(base_h) as f32;

    let watermark_scale_factor =
        (base_min_dim * (watermark_settings.scale / 100.0)) / watermark_img.width().max(1) as f32;
    let new_wm_w = (watermark_img.width() as f32 * watermark_scale_factor).round() as u32;
    let new_wm_h = (watermark_img.height() as f32 * watermark_scale_factor).round() as u32;

    if new_wm_w == 0 || new_wm_h == 0 {
        return Ok(None);
    }

    let scaled_watermark =
        watermark_img.resize_exact(new_wm_w, new_wm_h, image::imageops::FilterType::Lanczos3);
    let mut scaled_watermark_rgba = scaled_watermark.to_rgba8();

    let opacity_factor = (watermark_settings.opacity / 100.0).clamp(0.0, 1.0);
    for pixel in scaled_watermark_rgba.pixels_mut() {
        pixel[3] = (pixel[3] as f32 * opacity_factor) as u8;
    }
    let spacing_pixels = (base_min_dim * (watermark_settings.spacing / 100.0)) as i64;
    let (wm_w, wm_h) = scaled_watermark_rgba.dimensions();

    let x = match watermark_settings.anchor {
        WatermarkAnchor::TopLeft | WatermarkAnchor::CenterLeft | WatermarkAnchor::BottomLeft => {
            spacing_pixels
        }
        WatermarkAnchor::TopCenter | WatermarkAnchor::Center | WatermarkAnchor::BottomCenter => {
            (base_w as i64 - wm_w as i64) / 2
        }
        WatermarkAnchor::TopRight | WatermarkAnchor::CenterRight | WatermarkAnchor::BottomRight => {
            base_w as i64 - wm_w as i64 - spacing_pixels
        }
    };

    let y = match watermark_settings.anchor {
        WatermarkAnchor::TopLeft | WatermarkAnchor::TopCenter | WatermarkAnchor::TopRight => {
            spacing_pixels
        }
        WatermarkAnchor::CenterLeft | WatermarkAnchor::Center | WatermarkAnchor::CenterRight => {
            (base_h as i64 - wm_h as i64) / 2
        }
        WatermarkAnchor::BottomLeft
        | WatermarkAnchor::BottomCenter
        | WatermarkAnchor::BottomRight => base_h as i64 - wm_h as i64 - spacing_pixels,
    };

    Ok(Some(PreparedWatermark {
        image: scaled_watermark_rgba,
        x,
        y,
    }))
}

fn apply_watermark(
    base_image: &mut DynamicImage,
    watermark_settings: &WatermarkSettings,
) -> Result<(), String> {
    let (base_w, base_h) = base_image.dimensions();
    if let Some(watermark) = prepare_watermark(base_w, base_h, watermark_settings)? {
        image::imageops::overlay(base_image, &watermark.image, watermark.x, watermark.y);
    }

    Ok(())
}

pub(crate) fn calculate_resize_target(
    current_w: u32,
    current_h: u32,
    resize_opts: &ResizeOptions,
) -> (u32, u32) {
    if resize_opts.dont_enlarge {
        let exceeds = match resize_opts.mode {
            ResizeMode::LongEdge => current_w.max(current_h) > resize_opts.value,
            ResizeMode::ShortEdge => current_w.min(current_h) > resize_opts.value,
            ResizeMode::Width => current_w > resize_opts.value,
            ResizeMode::Height => current_h > resize_opts.value,
        };
        if !exceeds {
            return (current_w, current_h);
        }
    }

    let fix_width = match resize_opts.mode {
        ResizeMode::LongEdge => current_w >= current_h,
        ResizeMode::ShortEdge => current_w <= current_h,
        ResizeMode::Width => true,
        ResizeMode::Height => false,
    };

    let value = resize_opts.value;
    if fix_width {
        let h = (value as f32 * (current_h as f32 / current_w as f32)).round() as u32;
        (value, h)
    } else {
        let w = (value as f32 * (current_w as f32 / current_h as f32)).round() as u32;
        (w, value)
    }
}

fn relative_dir_is_safe(rel_dir: &Path) -> bool {
    rel_dir.components().all(|component| {
        matches!(
            component,
            std::path::Component::Normal(_) | std::path::Component::CurDir
        )
    })
}

#[cfg(windows)]
fn component_matches(left: std::path::Component<'_>, right: std::path::Component<'_>) -> bool {
    left.as_os_str()
        .to_string_lossy()
        .eq_ignore_ascii_case(&right.as_os_str().to_string_lossy())
}

#[cfg(not(windows))]
fn component_matches(left: std::path::Component<'_>, right: std::path::Component<'_>) -> bool {
    left == right
}

fn strip_prefix_preserving_source_case(source_path: &Path, base_path: &Path) -> Option<PathBuf> {
    let source_components: Vec<_> = source_path.components().collect();
    let base_components: Vec<_> = base_path.components().collect();

    if base_components.len() > source_components.len() {
        return None;
    }

    if !source_components
        .iter()
        .zip(base_components.iter())
        .all(|(source, base)| component_matches(*source, *base))
    {
        return None;
    }

    Some(source_components[base_components.len()..].iter().collect())
}

fn relative_export_dir_for_preserved_folders(
    source_path: &Path,
    base_origin_folders: &[String],
) -> Option<PathBuf> {
    base_origin_folders
        .iter()
        .filter_map(|base| {
            let base_path = Path::new(base);
            strip_prefix_preserving_source_case(source_path, base_path)
                .map(|rel_path| (base_path.components().count(), rel_path))
        })
        .max_by_key(|(component_count, _)| *component_count)
        .and_then(|(_, rel_path)| {
            let rel_dir = rel_path.parent().unwrap_or_else(|| Path::new(""));
            if relative_dir_is_safe(rel_dir) {
                Some(rel_dir.to_path_buf())
            } else {
                None
            }
        })
}

pub(crate) fn apply_export_resize_and_watermark(
    mut image: DynamicImage,
    export_settings: &ExportSettings,
) -> Result<DynamicImage, String> {
    if let Some(resize_opts) = &export_settings.resize {
        let (current_w, current_h) = image.dimensions();
        let (target_w, target_h) = calculate_resize_target(current_w, current_h, resize_opts);

        if target_w != current_w || target_h != current_h {
            image = image.resize(target_w, target_h, imageops::FilterType::Lanczos3);
        }
    }

    if let Some(watermark_settings) = &export_settings.watermark {
        apply_watermark(&mut image, watermark_settings)?;
    }
    Ok(image)
}

fn ensure_export_not_cancelled(cancellation_token: &AtomicBool) -> Result<(), String> {
    if cancellation_token.load(Ordering::SeqCst) {
        Err("Export cancelled".to_string())
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportCancellationRequest {
    Requested,
    AlreadyRequested,
    NoActiveTask,
}

#[derive(Debug, PartialEq, Eq)]
struct BatchExportSummary {
    completed: usize,
    errors: Vec<String>,
}

fn summarize_batch_export_results(
    results: impl IntoIterator<Item = Result<(), String>>,
) -> BatchExportSummary {
    let mut completed = 0;
    let mut errors = Vec::new();
    for result in results {
        completed += 1;
        if let Err(error) = result {
            errors.push(error);
        }
    }
    BatchExportSummary { completed, errors }
}

struct ExportTaskGuard {
    task_token: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    cancellation_token: Arc<AtomicBool>,
    app_handle: Option<tauri::AppHandle>,
}

impl ExportTaskGuard {
    fn new(
        task_token: Arc<Mutex<Option<Arc<AtomicBool>>>>,
        cancellation_token: Arc<AtomicBool>,
    ) -> Self {
        Self {
            task_token,
            cancellation_token,
            app_handle: None,
        }
    }

    fn with_app_handle(
        task_token: Arc<Mutex<Option<Arc<AtomicBool>>>>,
        cancellation_token: Arc<AtomicBool>,
        app_handle: tauri::AppHandle,
    ) -> Self {
        let mut guard = Self::new(task_token, cancellation_token);
        guard.app_handle = Some(app_handle);
        guard
    }
}

fn register_export_task(
    task_token: &Mutex<Option<Arc<AtomicBool>>>,
) -> Result<Arc<AtomicBool>, String> {
    let mut active_token = task_token.lock().unwrap();
    if active_token.is_some() {
        return Err("An export is already in progress.".to_string());
    }

    let cancellation_token = Arc::new(AtomicBool::new(false));
    *active_token = Some(Arc::clone(&cancellation_token));
    Ok(cancellation_token)
}

fn request_export_cancellation<F>(
    task_token: &Mutex<Option<Arc<AtomicBool>>>,
    on_requested: F,
) -> ExportCancellationRequest
where
    F: FnOnce(),
{
    let active_token = task_token.lock().unwrap();
    let Some(cancellation_token) = active_token.as_ref() else {
        return ExportCancellationRequest::NoActiveTask;
    };

    if cancellation_token.swap(true, Ordering::SeqCst) {
        ExportCancellationRequest::AlreadyRequested
    } else {
        on_requested();
        ExportCancellationRequest::Requested
    }
}

fn finish_export_task<F>(
    task_token: &Mutex<Option<Arc<AtomicBool>>>,
    cancellation_token: &Arc<AtomicBool>,
    on_finish: F,
) -> bool
where
    F: FnOnce(bool),
{
    let mut active_token = task_token.lock().unwrap();
    let Some(current_token) = active_token.as_ref() else {
        return false;
    };
    if !Arc::ptr_eq(current_token, cancellation_token) {
        return false;
    }

    let cancelled = cancellation_token.load(Ordering::SeqCst);
    *active_token = None;

    on_finish(cancelled);
    true
}

impl Drop for ExportTaskGuard {
    fn drop(&mut self) {
        let app_handle = self.app_handle.clone();
        let _ = finish_export_task(
            &self.task_token,
            &self.cancellation_token,
            |cancelled| match (cancelled, app_handle) {
                (true, Some(app_handle)) => {
                    let _ = app_handle.emit("export-cancelled", ());
                }
                (false, Some(app_handle)) => {
                    let _ = app_handle.emit("export-error", "Export task terminated unexpectedly");
                }
                _ => {}
            },
        );
    }
}

enum PreparedExportSource<'a> {
    Materialized(Cow<'a, DynamicImage>),
    StreamedGeometry(Box<GeometryWarpRows<'a>>),
}

impl PreparedExportSource<'_> {
    fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Materialized(image) => image.dimensions(),
            Self::StreamedGeometry(rows) => rows.dimensions(),
        }
    }

    fn gpu_input(&self) -> &dyn crate::gpu_processing::GpuInputSource {
        match self {
            Self::Materialized(image) => image.as_ref(),
            Self::StreamedGeometry(rows) => rows.as_ref(),
        }
    }
}

struct PreparedExportRender<'a> {
    source: PreparedExportSource<'a>,
    mask_bitmaps: Vec<SharedMaskBitmap>,
    adjustments: AllAdjustments,
    lut: Option<Arc<Lut>>,
    unique_hash: u64,
}

fn can_stream_geometry_output(
    adjustments: &Value,
    geometry_params: &crate::image_processing::GeometryParams,
    masks: &[MaskDefinition],
) -> bool {
    !is_geometry_identity(geometry_params)
        && adjustments["rotation"]
            .as_f64()
            .unwrap_or(0.0)
            .rem_euclid(360.0)
            == 0.0
        && !adjustments["lensBlurEnabled"].as_bool().unwrap_or(false)
        && !masks.iter().any(MaskDefinition::requires_warped_image)
}

#[allow(clippy::too_many_arguments)]
fn prepare_export_render<'a>(
    path: &str,
    base_image: &'a DynamicImage,
    js_adjustments: &Value,
    state: &tauri::State<AppState>,
    is_raw: bool,
    debug_tag: &str,
    app_handle: &tauri::AppHandle,
) -> PreparedExportRender<'a> {
    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();
    let geometry_params = get_geometry_params_from_json(js_adjustments);
    let can_stream_geometry =
        can_stream_geometry_output(js_adjustments, &geometry_params, &mask_definitions);

    let orientation_steps = js_adjustments["orientationSteps"].as_u64().unwrap_or(0) as u8;
    let flip_horizontal = js_adjustments["flipHorizontal"].as_bool().unwrap_or(false);
    let flip_vertical = js_adjustments["flipVertical"].as_bool().unwrap_or(false);
    let streamed_geometry = can_stream_geometry.then(|| {
        GeometryWarpRows::try_new_transformed_borrowed(
            base_image,
            geometry_params,
            orientation_steps,
            flip_horizontal,
            flip_vertical,
            &js_adjustments["crop"],
        )
    });
    let (source, unscaled_crop_offset) = match streamed_geometry.flatten() {
        Some(rows) => {
            let saved_bytes = rows.materialized_output_bytes();
            let checkpoint_bytes = rows.checkpoint_bytes();
            let band_scratch_bytes =
                rows.max_band_scratch_bytes(crate::render_strategy::GPU_INPUT_UPLOAD_BAND_ROWS);
            let crop_offset = rows.crop_offset();
            log::info!(
                "[{}] streaming transformed geometry bands directly to GPU input (avoiding {} B RGBA32F output; checkpoints={} B, transpose_scratch<={} B)",
                debug_tag,
                saved_bytes,
                checkpoint_bytes,
                band_scratch_bytes,
            );
            (
                PreparedExportSource::StreamedGeometry(Box::new(rows)),
                crop_offset,
            )
        }
        None => {
            let (image, crop_offset) =
                apply_all_transformations(Cow::Borrowed(base_image), js_adjustments);
            (PreparedExportSource::Materialized(image), crop_offset)
        }
    };
    let (img_w, img_h) = source.dimensions();
    log::info!(
        "[{}] tier={} source={}x{}",
        debug_tag,
        RenderTier::FullResolutionExport.as_str(),
        img_w,
        img_h
    );

    let warped_image = resolve_warped_image_for_masks(state, js_adjustments, &mask_definitions);
    let mask_bitmaps: Vec<SharedMaskBitmap> = mask_definitions
        .iter()
        .filter_map(|def| {
            generate_mask_bitmap(
                def,
                img_w,
                img_h,
                1.0,
                unscaled_crop_offset,
                warped_image.as_deref(),
            )
            .map(Arc::new)
        })
        .collect();

    let tm_override = resolve_tonemapper_override_from_handle(app_handle, is_raw);
    let mut all_adjustments = get_all_adjustments_from_json(js_adjustments, is_raw, tm_override);
    all_adjustments.global.show_clipping = 0;

    let lut_path = js_adjustments["lutPath"].as_str();
    let lut = lut_path.and_then(|p| get_or_load_lut(state, p).ok());

    let unique_hash = calculate_full_job_hash(path, js_adjustments);

    PreparedExportRender {
        source,
        mask_bitmaps,
        adjustments: all_adjustments,
        lut,
        unique_hash,
    }
}

#[allow(clippy::too_many_arguments)]
fn process_image_for_export_pipeline(
    path: &str,
    base_image: &DynamicImage,
    js_adjustments: &Value,
    context: &GpuContext,
    state: &tauri::State<AppState>,
    is_raw: bool,
    debug_tag: &str,
    app_handle: &tauri::AppHandle,
) -> Result<DynamicImage, String> {
    let prepared = prepare_export_render(
        path,
        base_image,
        js_adjustments,
        state,
        is_raw,
        debug_tag,
        app_handle,
    );

    process_and_get_dynamic_image(
        context,
        state,
        prepared.source.gpu_input(),
        prepared.unique_hash,
        RenderRequest {
            adjustments: prepared.adjustments,
            mask_bitmaps: &prepared.mask_bitmaps,
            lut: prepared.lut,
            roi: None,
        },
        debug_tag,
    )
}

fn set_timestamps_from_exif(src: &Path, dst: &Path) {
    let capture_dt = exif_processing::get_creation_date_from_path(src);
    let ft = filetime::FileTime::from_unix_time(
        capture_dt.timestamp(),
        capture_dt.timestamp_subsec_nanos(),
    );
    if let Err(e) = filetime::set_file_times(dst, ft, ft) {
        log::warn!("Could not set timestamps on '{}': {}", dst.display(), e);
    }
}

fn save_image_with_metadata(
    image: &DynamicImage,
    output_path: &std::path::Path,
    source_path_str: &str,
    export_settings: &ExportSettings,
    cancellation_token: &AtomicBool,
) -> Result<(), String> {
    ensure_export_not_cancelled(cancellation_token)?;
    let extension = output_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    #[cfg(not(target_os = "android"))]
    if extension == "webp" {
        return save_webp_with_bounded_output(
            image,
            output_path,
            export_settings.jpeg_quality,
            cancellation_token,
        );
    }

    let mut image_bytes = encode_image_to_bytes_with_profile(
        image,
        &extension,
        export_settings.jpeg_quality,
        export_settings.embed_color_profile,
    )?;
    ensure_export_not_cancelled(cancellation_token)?;

    exif_processing::write_image_with_metadata(
        &mut image_bytes,
        source_path_str,
        &extension,
        export_settings.keep_metadata,
        export_settings.strip_gps,
        export_settings.metadata_overrides.as_ref(),
    )?;
    ensure_export_not_cancelled(cancellation_token)?;

    #[cfg(target_os = "android")]
    {
        let file_name = output_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "Missing Android export file name".to_string())?;
        crate::android_integration::save_image_bytes_to_android_gallery(
            file_name,
            mime_type_for_extension(&extension),
            &image_bytes,
        )?;
    }

    #[cfg(not(target_os = "android"))]
    fs::write(output_path, image_bytes).map_err(|e| e.to_string())?;

    Ok(())
}

fn supports_streaming_export(output_format: &str, _export_settings: &ExportSettings) -> bool {
    #[cfg(target_os = "android")]
    {
        let _ = (output_format, _export_settings);
        false
    }

    #[cfg(not(target_os = "android"))]
    {
        matches!(
            output_format.to_ascii_lowercase().as_str(),
            "jpg" | "jpeg" | "png" | "tif" | "tiff"
        )
    }
}

#[cfg(not(target_os = "android"))]
fn create_temporary_export(output_path: &Path) -> Result<NamedTempFile, String> {
    let output_parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    NamedTempFile::new_in(output_parent).map_err(|error| {
        format!(
            "Failed to create temporary export beside '{}': {error}",
            output_path.display()
        )
    })
}

#[cfg(not(target_os = "android"))]
fn preserve_existing_export_permissions(
    temporary: &NamedTempFile,
    output_path: &Path,
) -> Result<(), String> {
    if let Ok(metadata) = fs::metadata(output_path) {
        temporary
            .as_file()
            .set_permissions(metadata.permissions())
            .map_err(|error| {
                format!(
                    "Failed to preserve permissions for '{}': {error}",
                    output_path.display()
                )
            })?;
    }
    Ok(())
}

#[cfg(not(target_os = "android"))]
const WEBP_OUTPUT_COPY_BUFFER_BYTES: usize = 64 * 1024;

#[cfg(not(target_os = "android"))]
struct WebpPictureGuard(libwebp_sys::WebPPicture);

#[cfg(not(target_os = "android"))]
impl Drop for WebpPictureGuard {
    fn drop(&mut self) {
        // SAFETY: WebPPicture::new initialized the value and this guard owns
        // the sole call that releases its libwebp allocations.
        unsafe { libwebp_sys::WebPPictureFree(&mut self.0) };
    }
}

#[cfg(not(target_os = "android"))]
struct WebpFileWriterContext {
    output: *mut fs::File,
    cancellation_token: *const AtomicBool,
    cancelled: bool,
    first_error: Option<String>,
}

#[cfg(not(target_os = "android"))]
unsafe extern "C" fn write_webp_bytes_to_file(
    data: *const u8,
    data_size: usize,
    picture: *const libwebp_sys::WebPPicture,
) -> std::ffi::c_int {
    let completed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if picture.is_null() {
            return false;
        }

        // SAFETY: encode_webp_to_file sets custom_ptr to this stack context
        // immediately before the synchronous WebPEncode call and clears it
        // before returning.
        let context_ptr = unsafe { (*picture).custom_ptr.cast::<WebpFileWriterContext>() };
        if context_ptr.is_null() {
            return false;
        }
        // SAFETY: the pointer remains valid and exclusively used by libwebp
        // for the duration of the synchronous encoder call.
        let context = unsafe { &mut *context_ptr };
        if data_size == 0 {
            return true;
        }
        if data.is_null() || context.output.is_null() {
            context
                .first_error
                .get_or_insert_with(|| "libwebp supplied an invalid output buffer".to_string());
            return false;
        }

        // SAFETY: libwebp guarantees data points to data_size readable bytes
        // for the duration of this callback.
        let bytes = unsafe { std::slice::from_raw_parts(data, data_size) };
        // SAFETY: output points to the uniquely borrowed File held by the
        // callback context for the duration of WebPEncode.
        let output = unsafe { &mut *context.output };
        if let Err(error) = output.write_all(bytes) {
            context
                .first_error
                .get_or_insert_with(|| format!("Failed to write WebP output: {error}"));
            return false;
        }
        true
    }))
    .unwrap_or(false);

    i32::from(completed)
}

#[cfg(not(target_os = "android"))]
unsafe extern "C" fn check_webp_encoding_progress(
    _percent: std::ffi::c_int,
    picture: *const libwebp_sys::WebPPicture,
) -> std::ffi::c_int {
    let should_continue = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if picture.is_null() {
            return false;
        }
        // SAFETY: encode_webp_to_file installs the same live callback context
        // in user_data for the duration of the synchronous WebPEncode call.
        let context_ptr = unsafe { (*picture).user_data.cast::<WebpFileWriterContext>() };
        if context_ptr.is_null() {
            return false;
        }
        // SAFETY: libwebp invokes the progress hook synchronously and the
        // context remains valid until WebPEncode returns.
        let context = unsafe { &mut *context_ptr };
        if context.cancellation_token.is_null() {
            return true;
        }
        // SAFETY: the caller-owned cancellation token outlives WebPEncode and
        // AtomicBool permits concurrent reads from the export worker.
        if unsafe { &*context.cancellation_token }.load(Ordering::SeqCst) {
            context.cancelled = true;
            return false;
        }
        true
    }))
    .unwrap_or(false);

    i32::from(should_continue)
}

#[cfg(not(target_os = "android"))]
fn encode_webp_to_file(
    image: &DynamicImage,
    jpeg_quality: u8,
    output: &mut fs::File,
    cancellation_token: &AtomicBool,
) -> Result<(), String> {
    const WEBP_MAX_DIMENSION: u32 = 16_383;

    let (pixels, width, height, bytes_per_pixel) = match image {
        DynamicImage::ImageRgb8(image) => (
            image.as_raw().as_slice(),
            image.width(),
            image.height(),
            3_usize,
        ),
        DynamicImage::ImageRgba8(image) => (
            image.as_raw().as_slice(),
            image.width(),
            image.height(),
            4_usize,
        ),
        _ => return Err("Failed to create WebP encoder".to_string()),
    };
    if width == 0 || height == 0 || width > WEBP_MAX_DIMENSION || height > WEBP_MAX_DIMENSION {
        return Err(format!(
            "WebP dimensions {width}x{height} exceed the format limit of {WEBP_MAX_DIMENSION} pixels per axis"
        ));
    }
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
        .ok_or_else(|| "WebP input dimensions exceed the addressable buffer size".to_string())?;
    if pixels.len() != expected_len {
        return Err(format!(
            "WebP input has {} bytes; expected {expected_len}",
            pixels.len()
        ));
    }
    let stride = (width as usize)
        .checked_mul(bytes_per_pixel)
        .and_then(|value| i32::try_from(value).ok())
        .ok_or_else(|| "WebP row stride exceeds the encoder limit".to_string())?;

    let mut config = libwebp_sys::WebPConfig::new()
        .map_err(|_| "Failed to initialize the WebP encoder configuration".to_string())?;
    config.lossless = 0;
    config.alpha_compression = 1;
    config.quality = f32::from(jpeg_quality);
    // SAFETY: config was initialized by libwebp and remains valid here.
    if unsafe { libwebp_sys::WebPValidateConfig(&config) } == 0 {
        return Err("Invalid WebP encoder configuration".to_string());
    }

    let picture = libwebp_sys::WebPPicture::new()
        .map_err(|_| "Failed to initialize the WebP picture".to_string())?;
    let mut picture = WebpPictureGuard(picture);
    // Lossy WebP ultimately encodes YUV. Import directly into libwebp's YUVA
    // picture instead of retaining a second full ARGB frame until encoding.
    picture.0.use_argb = 0;
    picture.0.width = i32::try_from(width).map_err(|error| error.to_string())?;
    picture.0.height = i32::try_from(height).map_err(|error| error.to_string())?;
    // SAFETY: the validated input slice contains exactly height rows of the
    // declared stride and stays alive until the synchronous import completes.
    let imported = unsafe {
        if bytes_per_pixel == 4 {
            libwebp_sys::WebPPictureImportRGBA(&mut picture.0, pixels.as_ptr(), stride)
        } else {
            libwebp_sys::WebPPictureImportRGB(&mut picture.0, pixels.as_ptr(), stride)
        }
    };
    if imported == 0 {
        return Err(format!(
            "Failed to import WebP pixels: {:?}",
            picture.0.error_code
        ));
    }

    let mut writer_context = WebpFileWriterContext {
        output,
        cancellation_token,
        cancelled: false,
        first_error: None,
    };
    picture.0.writer = Some(write_webp_bytes_to_file);
    picture.0.custom_ptr = std::ptr::from_mut(&mut writer_context).cast();
    picture.0.progress_hook = Some(check_webp_encoding_progress);
    picture.0.user_data = std::ptr::from_mut(&mut writer_context).cast();
    // SAFETY: config and picture are initialized; all pixel data, callback
    // state and output storage outlive this synchronous call.
    let status = unsafe { libwebp_sys::WebPEncode(&config, &mut picture.0) };
    picture.0.writer = None;
    picture.0.custom_ptr = std::ptr::null_mut();
    picture.0.progress_hook = None;
    picture.0.user_data = std::ptr::null_mut();

    if status == 0 {
        if writer_context.cancelled {
            return Err("Export cancelled".to_string());
        }
        return Err(writer_context.first_error.unwrap_or_else(|| {
            format!("Failed to encode WebP image: {:?}", picture.0.error_code)
        }));
    }
    if let Some(error) = writer_context.first_error {
        return Err(error);
    }
    Ok(())
}

#[cfg(not(target_os = "android"))]
fn inspect_webp_file<R: Read + Seek>(
    source: &mut R,
    cancellation_token: &AtomicBool,
) -> Result<(u64, bool), String> {
    ensure_export_not_cancelled(cancellation_token)?;
    let source_len = source
        .seek(SeekFrom::End(0))
        .map_err(|error| format!("Failed to inspect WebP output length: {error}"))?;
    source
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("Failed to rewind WebP output: {error}"))?;

    let mut riff_header = [0_u8; 12];
    source
        .read_exact(&mut riff_header)
        .map_err(|_| "WebP encoder returned an invalid RIFF container".to_string())?;
    if &riff_header[..4] != b"RIFF" || &riff_header[8..] != b"WEBP" {
        return Err("WebP encoder returned an invalid RIFF container".to_string());
    }
    let declared_len = u64::from(u32::from_le_bytes(
        riff_header[4..8]
            .try_into()
            .map_err(|_| "Invalid WebP RIFF length".to_string())?,
    )) + 8;
    if declared_len != source_len {
        return Err(format!(
            "WebP RIFF length {declared_len} does not match file length {source_len}"
        ));
    }

    let mut offset = 12_u64;
    let mut found_vp8x = false;
    while offset + 8 <= source_len {
        ensure_export_not_cancelled(cancellation_token)?;
        let mut chunk_header = [0_u8; 8];
        source
            .read_exact(&mut chunk_header)
            .map_err(|error| format!("Failed to read WebP chunk header: {error}"))?;
        let chunk_len = u64::from(u32::from_le_bytes(
            chunk_header[4..8]
                .try_into()
                .map_err(|_| "Invalid WebP chunk length".to_string())?,
        ));
        let padded_len = chunk_len + chunk_len % 2;
        let chunk_end = offset
            .checked_add(8)
            .and_then(|value| value.checked_add(padded_len))
            .ok_or_else(|| "WebP chunk size overflow".to_string())?;
        if chunk_end > source_len {
            return Err("WebP chunk extends past the RIFF container".to_string());
        }
        if &chunk_header[..4] == b"VP8X" {
            if chunk_len != 10 {
                return Err("WebP VP8X chunk has an invalid length".to_string());
            }
            found_vp8x = true;
        }
        source
            .seek(SeekFrom::Current(padded_len as i64))
            .map_err(|error| format!("Failed to seek over WebP chunk: {error}"))?;
        offset = chunk_end;
    }
    if offset != source_len {
        return Err("WebP container has trailing partial chunk data".to_string());
    }
    Ok((source_len, found_vp8x))
}

#[cfg(not(target_os = "android"))]
fn write_webp_chunk_to_writer<W: Write>(
    output: &mut W,
    fourcc: &[u8; 4],
    payload: &[u8],
) -> Result<u64, String> {
    let payload_len =
        u32::try_from(payload.len()).map_err(|_| "WebP chunk is too large for RIFF".to_string())?;
    output
        .write_all(fourcc)
        .and_then(|_| output.write_all(&payload_len.to_le_bytes()))
        .and_then(|_| output.write_all(payload))
        .map_err(|error| format!("Failed to write WebP chunk: {error}"))?;
    if !payload.len().is_multiple_of(2) {
        output
            .write_all(&[0])
            .map_err(|error| format!("Failed to pad WebP chunk: {error}"))?;
    }
    Ok(8 + payload.len() as u64 + (payload.len() % 2) as u64)
}

#[cfg(not(target_os = "android"))]
fn copy_webp_bytes_bounded<R: Read, W: Write>(
    source: &mut R,
    output: &mut W,
    mut remaining: u64,
    buffer: &mut [u8; WEBP_OUTPUT_COPY_BUFFER_BYTES],
    cancellation_token: &AtomicBool,
) -> Result<(), String> {
    while remaining > 0 {
        ensure_export_not_cancelled(cancellation_token)?;
        let chunk_len = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|error| error.to_string())?;
        source
            .read_exact(&mut buffer[..chunk_len])
            .map_err(|error| format!("Failed to read encoded WebP data: {error}"))?;
        output
            .write_all(&buffer[..chunk_len])
            .map_err(|error| format!("Failed to rewrite encoded WebP data: {error}"))?;
        remaining -= chunk_len as u64;
    }
    Ok(())
}

#[cfg(not(target_os = "android"))]
fn rewrite_webp_icc_bounded<R: Read + Seek, W: Write + Seek>(
    source: &mut R,
    output: &mut W,
    icc_profile: &[u8],
    width: u32,
    height: u32,
    has_alpha: bool,
    cancellation_token: &AtomicBool,
) -> Result<u64, String> {
    if width == 0 || height == 0 || width > 0x01_00_00_00 || height > 0x01_00_00_00 {
        return Err("WebP dimensions cannot be represented by a VP8X header".to_string());
    }
    let (source_len, found_vp8x) = inspect_webp_file(source, cancellation_token)?;
    source
        .seek(SeekFrom::Start(12))
        .map_err(|error| format!("Failed to rewind WebP chunks: {error}"))?;
    output
        .seek(SeekFrom::Start(0))
        .map_err(|error| format!("Failed to initialize rewritten WebP output: {error}"))?;
    output
        .write_all(b"RIFF\0\0\0\0WEBP")
        .map_err(|error| format!("Failed to write WebP RIFF header: {error}"))?;
    let mut output_len = 12_u64;
    let mut wrote_icc = false;
    if !found_vp8x {
        let mut vp8x = [0_u8; 10];
        vp8x[0] = 0x20 | if has_alpha { 0x10 } else { 0 };
        vp8x[4..7].copy_from_slice(&(width - 1).to_le_bytes()[..3]);
        vp8x[7..10].copy_from_slice(&(height - 1).to_le_bytes()[..3]);
        output_len += write_webp_chunk_to_writer(output, b"VP8X", &vp8x)?;
        output_len += write_webp_chunk_to_writer(output, b"ICCP", icc_profile)?;
        wrote_icc = true;
    }

    let mut offset = 12_u64;
    let mut copy_buffer = [0_u8; WEBP_OUTPUT_COPY_BUFFER_BYTES];
    while offset + 8 <= source_len {
        ensure_export_not_cancelled(cancellation_token)?;
        let mut chunk_header = [0_u8; 8];
        source
            .read_exact(&mut chunk_header)
            .map_err(|error| format!("Failed to read WebP chunk header: {error}"))?;
        let fourcc: [u8; 4] = chunk_header[..4]
            .try_into()
            .map_err(|_| "Invalid WebP chunk identifier".to_string())?;
        let chunk_len = u64::from(u32::from_le_bytes(
            chunk_header[4..8]
                .try_into()
                .map_err(|_| "Invalid WebP chunk length".to_string())?,
        ));
        let padded_len = chunk_len + chunk_len % 2;

        if &fourcc == b"ICCP" {
            source
                .seek(SeekFrom::Current(padded_len as i64))
                .map_err(|error| format!("Failed to skip existing WebP ICC chunk: {error}"))?;
        } else if &fourcc == b"VP8X" {
            let mut payload = [0_u8; 10];
            source
                .read_exact(&mut payload)
                .map_err(|error| format!("Failed to read WebP VP8X chunk: {error}"))?;
            payload[0] |= 0x20;
            output_len += write_webp_chunk_to_writer(output, b"VP8X", &payload)?;
            if !wrote_icc {
                output_len += write_webp_chunk_to_writer(output, b"ICCP", icc_profile)?;
                wrote_icc = true;
            }
        } else {
            output
                .write_all(&chunk_header)
                .map_err(|error| format!("Failed to write WebP chunk header: {error}"))?;
            copy_webp_bytes_bounded(
                source,
                output,
                padded_len,
                &mut copy_buffer,
                cancellation_token,
            )?;
            output_len = output_len
                .checked_add(8 + padded_len)
                .ok_or_else(|| "WebP output size overflow".to_string())?;
        }
        offset += 8 + padded_len;
    }
    if !wrote_icc {
        return Err("Failed to insert the WebP ICC profile".to_string());
    }

    let riff_size = u32::try_from(
        output_len
            .checked_sub(8)
            .ok_or_else(|| "WebP output size underflow".to_string())?,
    )
    .map_err(|_| "WebP output is too large for RIFF".to_string())?;
    output
        .seek(SeekFrom::Start(4))
        .and_then(|_| output.write_all(&riff_size.to_le_bytes()))
        .and_then(|_| output.seek(SeekFrom::Start(output_len)).map(|_| ()))
        .map_err(|error| format!("Failed to finalize WebP RIFF output: {error}"))?;
    ensure_export_not_cancelled(cancellation_token)?;
    Ok(output_len)
}

#[cfg(not(target_os = "android"))]
fn save_webp_with_bounded_output(
    image: &DynamicImage,
    output_path: &Path,
    jpeg_quality: u8,
    cancellation_token: &AtomicBool,
) -> Result<(), String> {
    ensure_export_not_cancelled(cancellation_token)?;
    let mut encoded_temporary = create_temporary_export(output_path)?;
    encode_webp_to_file(
        image,
        jpeg_quality,
        encoded_temporary.as_file_mut(),
        cancellation_token,
    )?;
    encoded_temporary
        .as_file_mut()
        .flush()
        .map_err(|error| format!("Failed to flush encoded WebP output: {error}"))?;
    ensure_export_not_cancelled(cancellation_token)?;

    let mut final_temporary = create_temporary_export(output_path)?;
    rewrite_webp_icc_bounded(
        encoded_temporary.as_file_mut(),
        final_temporary.as_file_mut(),
        srgb_v4_profile(),
        image.width(),
        image.height(),
        image.color().has_alpha(),
        cancellation_token,
    )?;
    final_temporary
        .as_file_mut()
        .flush()
        .map_err(|error| format!("Failed to flush rewritten WebP output: {error}"))?;
    ensure_export_not_cancelled(cancellation_token)?;
    drop(encoded_temporary);
    preserve_existing_export_permissions(&final_temporary, output_path)?;
    final_temporary.persist(output_path).map_err(|error| {
        format!(
            "Failed to atomically publish WebP export '{}': {}",
            output_path.display(),
            error.error
        )
    })?;
    Ok(())
}

#[cfg(not(target_os = "android"))]
fn validate_streamed_rgba_row(
    row: &[u8],
    width: u32,
    height: u32,
    written_rows: u32,
) -> Result<(), String> {
    if written_rows >= height {
        return Err(format!(
            "Streaming encoder received more than the declared {height} rows"
        ));
    }
    let expected = (width as usize)
        .checked_mul(4)
        .ok_or_else(|| format!("Streaming width {width} exceeds the RGBA row size limit"))?;
    if row.len() != expected {
        return Err(format!(
            "Streamed row {} has {} bytes; expected {} RGBA bytes",
            written_rows,
            row.len(),
            expected
        ));
    }
    Ok(())
}

#[cfg(not(target_os = "android"))]
impl PreparedWatermark {
    fn row_intersection(&self, base_width: u32, row_y: u32) -> Option<(u32, u32, u32)> {
        let watermark_y = i64::from(row_y) - self.y;
        if watermark_y < 0 || watermark_y >= i64::from(self.image.height()) {
            return None;
        }

        let start_x = self.x.max(0);
        let end_x = (self.x + i64::from(self.image.width())).min(i64::from(base_width));
        if start_x >= end_x {
            return None;
        }

        Some((start_x as u32, end_x as u32, watermark_y as u32))
    }

    fn blend_row(&self, base_width: u32, row_y: u32, rgba_row: &mut [u8]) {
        let Some((start_x, end_x, watermark_y)) = self.row_intersection(base_width, row_y) else {
            return;
        };

        for base_x in start_x..end_x {
            let watermark_x = (i64::from(base_x) - self.x) as u32;
            let byte_offset = base_x as usize * 4;
            let mut base = Rgba([
                rgba_row[byte_offset],
                rgba_row[byte_offset + 1],
                rgba_row[byte_offset + 2],
                rgba_row[byte_offset + 3],
            ]);
            base.blend(self.image.get_pixel(watermark_x, watermark_y));
            rgba_row[byte_offset..byte_offset + 4].copy_from_slice(&base.0);
        }
    }
}

#[cfg(not(target_os = "android"))]
struct StreamingRowOutput<'a> {
    width: u32,
    height: u32,
    written_rows: u32,
    watermark: Option<PreparedWatermark>,
    scratch: Vec<u8>,
    sink: &'a mut dyn FnMut(&[u8]) -> Result<(), String>,
}

#[cfg(not(target_os = "android"))]
impl<'a> StreamingRowOutput<'a> {
    fn new(
        width: u32,
        height: u32,
        watermark: Option<PreparedWatermark>,
        sink: &'a mut dyn FnMut(&[u8]) -> Result<(), String>,
    ) -> Result<Self, String> {
        let row_bytes = (width as usize)
            .checked_mul(4)
            .ok_or_else(|| format!("Streaming output width {width} exceeds the row size limit"))?;
        let mut scratch = Vec::new();
        if watermark.is_some() {
            scratch
                .try_reserve_exact(row_bytes)
                .map_err(|error| format!("Failed to reserve the watermark row buffer: {error}"))?;
        }
        Ok(Self {
            width,
            height,
            written_rows: 0,
            watermark,
            scratch,
            sink,
        })
    }

    fn emit(&mut self, rgba_row: &[u8]) -> Result<(), String> {
        validate_streamed_rgba_row(rgba_row, self.width, self.height, self.written_rows)?;
        let row_intersects_watermark = self
            .watermark
            .as_ref()
            .and_then(|watermark| watermark.row_intersection(self.width, self.written_rows))
            .is_some();

        if row_intersects_watermark {
            self.scratch.clear();
            self.scratch.extend_from_slice(rgba_row);
            if let Some(watermark) = &self.watermark {
                watermark.blend_row(self.width, self.written_rows, &mut self.scratch);
            }
            (self.sink)(&self.scratch)?;
        } else {
            (self.sink)(rgba_row)?;
        }
        self.written_rows += 1;
        Ok(())
    }

    fn finish(&self) -> Result<(), String> {
        if self.written_rows != self.height {
            return Err(format!(
                "Streaming transform produced {} rows; expected {}",
                self.written_rows, self.height
            ));
        }
        Ok(())
    }
}

#[cfg(not(target_os = "android"))]
#[allow(clippy::too_many_arguments)]
fn transform_streaming_rgba_rows<F>(
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
    watermark_settings: Option<&WatermarkSettings>,
    output_sink: &mut dyn FnMut(&[u8]) -> Result<(), String>,
    render_rows: F,
) -> Result<(), String>
where
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String>,
{
    if source_width == 0 || source_height == 0 || target_width == 0 || target_height == 0 {
        return Err(format!(
            "Streaming transform requires non-zero dimensions; source={}x{}, target={}x{}",
            source_width, source_height, target_width, target_height
        ));
    }

    let watermark = watermark_settings
        .map(|settings| prepare_watermark(target_width, target_height, settings))
        .transpose()?
        .flatten();
    let mut output = StreamingRowOutput::new(target_width, target_height, watermark, output_sink)?;
    let mut source_rows = 0_u32;

    if (source_width, source_height) == (target_width, target_height) {
        let mut input_sink = |row: &[u8]| -> Result<(), String> {
            validate_streamed_rgba_row(row, source_width, source_height, source_rows)?;
            source_rows += 1;
            output.emit(row)
        };
        render_rows(&mut input_sink)?;
    } else {
        // The legacy full-frame path used encoded-space Lanczos3. Keep that
        // semantic while switching storage to zenresize's bounded row ring.
        let resize_config =
            ResizeConfig::builder(source_width, source_height, target_width, target_height)
                .filter(StreamingResizeFilter::Lanczos)
                .format(PixelDescriptor::RGBA8_SRGB)
                .srgb()
                .build();
        let mut resizer = StreamingResize::new(&resize_config);
        let mut input_sink = |row: &[u8]| -> Result<(), String> {
            validate_streamed_rgba_row(row, source_width, source_height, source_rows)?;
            resizer.push_row(row).map_err(|error| {
                format!("Failed to push source row {source_rows} into streaming resize: {error}")
            })?;
            source_rows += 1;
            while let Some(resized_row) = resizer.next_output_row() {
                output.emit(resized_row)?;
            }
            Ok(())
        };
        render_rows(&mut input_sink)?;
        if source_rows != source_height {
            return Err(format!(
                "Streaming transform received {source_rows} source rows; expected {source_height}"
            ));
        }

        resizer.finish();
        while let Some(resized_row) = resizer.next_output_row() {
            output.emit(resized_row)?;
        }
        if !resizer.is_complete() {
            return Err("Streaming resize did not produce a complete output frame".to_string());
        }
    }

    if source_rows != source_height {
        return Err(format!(
            "Streaming transform received {source_rows} source rows; expected {source_height}"
        ));
    }
    output.finish()
}

#[cfg(not(target_os = "android"))]
pub(crate) fn encode_streaming_jpeg<F>(
    output: &mut fs::File,
    width: u32,
    height: u32,
    jpeg_quality: u8,
    embed_color_profile: bool,
    export_exif: Option<&[u8]>,
    render_rows: F,
) -> Result<(), String>
where
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String>,
{
    if width > u32::from(u16::MAX) || height > u32::from(u16::MAX) {
        return Err(format!(
            "JPEG dimensions {width}x{height} exceed the format limit of 65535 pixels per axis"
        ));
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        let mut compressor = MozjpegCompressor::new(MozjpegColorSpace::JCS_RGB);
        compressor.set_fastest_defaults();
        compressor.set_size(width as usize, height as usize);
        compressor.set_quality(f32::from(jpeg_quality.clamp(1, 100)));
        compressor.set_chroma_sampling_pixel_sizes((1, 1), (1, 1));
        compressor.set_optimize_coding(false);

        let mut encoder = compressor
            .start_compress(output)
            .map_err(|error| format!("Failed to configure streaming JPEG encoder: {error}"))?;
        const ICC_MARKER_OVERHEAD: usize = 14;
        const MAX_MARKER_PAYLOAD: usize = 65_533;
        if embed_color_profile {
            let icc_profile = srgb_v4_profile();
            // mozjpeg 0.10.13 numbers helper-generated ICC chunks from zero, while
            // the JPEG ICC convention and image decoders require one-based indices.
            let icc_chunk_size = MAX_MARKER_PAYLOAD - ICC_MARKER_OVERHEAD;
            let icc_chunk_count = icc_profile.len().div_ceil(icc_chunk_size);
            let icc_chunk_count = u8::try_from(icc_chunk_count)
                .map_err(|_| "Bundled ICC profile requires too many JPEG markers".to_string())?;
            for (chunk_index, chunk) in icc_profile.chunks(icc_chunk_size).enumerate() {
                let mut marker = Vec::with_capacity(ICC_MARKER_OVERHEAD + chunk.len());
                marker.extend_from_slice(b"ICC_PROFILE\0");
                marker.push(
                    u8::try_from(chunk_index + 1)
                        .map_err(|_| "Bundled ICC profile chunk index overflow".to_string())?,
                );
                marker.push(icc_chunk_count);
                marker.extend_from_slice(chunk);
                encoder.write_marker(MozjpegMarker::APP(2), &marker);
            }
        }
        if let Some(tiff_payload) = export_exif {
            const EXIF_HEADER: &[u8] = b"Exif\0\0";
            if tiff_payload.len() > MAX_MARKER_PAYLOAD - EXIF_HEADER.len() {
                return Err(format!(
                    "Encoded EXIF metadata exceeds the JPEG APP1 limit ({} bytes)",
                    tiff_payload.len()
                ));
            }
            let mut marker = Vec::with_capacity(EXIF_HEADER.len() + tiff_payload.len());
            marker.extend_from_slice(EXIF_HEADER);
            marker.extend_from_slice(tiff_payload);
            encoder.write_marker(MozjpegMarker::APP(1), &marker);
        }
        let mut written_rows = 0_u32;
        let mut rgb_row = vec![0_u8; width as usize * 3];
        {
            let mut sink = |rgba_row: &[u8]| -> Result<(), String> {
                validate_streamed_rgba_row(rgba_row, width, height, written_rows)?;
                for (rgba, rgb) in rgba_row.chunks_exact(4).zip(rgb_row.chunks_exact_mut(3)) {
                    rgb.copy_from_slice(&rgba[..3]);
                }
                encoder.write_scanlines(&rgb_row).map_err(|error| {
                    format!("Failed to stream JPEG row {written_rows}: {error}")
                })?;
                written_rows += 1;
                Ok(())
            };
            render_rows(&mut sink)?;
        }
        if written_rows != height {
            return Err(format!(
                "Streaming JPEG received {written_rows} rows; expected {height}"
            ));
        }
        encoder
            .finish()
            .map(|_| ())
            .map_err(|error| format!("Failed to finish streaming JPEG: {error}"))
    }))
    .map_err(|_| "Streaming JPEG encoder aborted while processing the image".to_string())?
}

#[cfg(not(target_os = "android"))]
fn encode_streaming_png<F>(
    output: &mut fs::File,
    width: u32,
    height: u32,
    embed_color_profile: bool,
    export_exif: Option<&[u8]>,
    render_rows: F,
) -> Result<(), String>
where
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String>,
{
    let mut info = PngInfo::with_size(width, height);
    info.color_type = PngColorType::Rgba;
    info.bit_depth = BitDepth::Eight;
    info.icc_profile = embed_color_profile.then(|| Cow::Owned(srgb_v4_profile().to_vec()));
    info.exif_metadata = export_exif.map(|metadata| Cow::Owned(metadata.to_vec()));
    let writer = BufWriter::new(output);
    let encoder = PngStreamEncoder::with_info(writer, info)
        .map_err(|error| format!("Failed to configure streaming PNG encoder: {error}"))?;
    let mut png_writer = encoder
        .write_header()
        .map_err(|error| format!("Failed to write streaming PNG header: {error}"))?;
    let mut stream = png_writer
        .stream_writer_with_size(64 * 1024)
        .map_err(|error| format!("Failed to start streaming PNG encoder: {error}"))?;
    let mut written_rows = 0_u32;
    {
        let mut sink = |rgba_row: &[u8]| -> Result<(), String> {
            validate_streamed_rgba_row(rgba_row, width, height, written_rows)?;
            stream
                .write_all(rgba_row)
                .map_err(|error| format!("Failed to stream PNG row {written_rows}: {error}"))?;
            written_rows += 1;
            Ok(())
        };
        render_rows(&mut sink)?;
    }
    if written_rows != height {
        return Err(format!(
            "Streaming PNG received {written_rows} rows; expected {height}"
        ));
    }
    stream
        .finish()
        .map_err(|error| format!("Failed to finish streaming PNG data: {error}"))?;
    png_writer
        .finish()
        .map_err(|error| format!("Failed to finish streaming PNG: {error}"))
}

#[cfg(not(target_os = "android"))]
fn has_writable_tiff_metadata_group(metadata: &ExifMetadata, group: ExifTagGroup) -> bool {
    metadata
        .into_iter()
        .any(|tag| tag.get_group() == group && tag.is_writable())
}

#[cfg(not(target_os = "android"))]
fn write_tiff_metadata_group<W: Write + Seek>(
    directory: &mut TiffDirectoryEncoder<'_, W, TiffKindStandard>,
    metadata: &ExifMetadata,
    group: ExifTagGroup,
) -> Result<(), String> {
    let endian = if cfg!(target_endian = "little") {
        ExifEndian::Little
    } else {
        ExifEndian::Big
    };
    let mut entries = TiffDirectory::empty();

    for tag in metadata
        .into_iter()
        .filter(|tag| tag.get_group() == group && tag.is_writable())
    {
        let format = tag.format();
        let data_type = TiffType::from_u16(format.as_u16()).ok_or_else(|| {
            format!(
                "Unsupported TIFF metadata type {} for tag 0x{:04x}",
                format.as_u16(),
                tag.as_u16()
            )
        })?;
        let expected_bytes = usize::try_from(tag.number_of_components())
            .ok()
            .and_then(|components| components.checked_mul(format.bytes_per_component() as usize))
            .ok_or_else(|| {
                format!(
                    "TIFF metadata tag 0x{:04x} exceeds the component size limit",
                    tag.as_u16()
                )
            })?;
        if expected_bytes == 0 {
            return Err(format!(
                "TIFF metadata tag 0x{:04x} has no components",
                tag.as_u16()
            ));
        }

        let mut value = tag.value_as_u8_vec(&endian);
        if value.len() > expected_bytes {
            return Err(format!(
                "TIFF metadata tag 0x{:04x} encoded {} bytes; expected at most {expected_bytes}",
                tag.as_u16(),
                value.len()
            ));
        }
        value.resize(expected_bytes, 0);

        let entry = directory
            .write_entry_bytes(data_type, &value)
            .map_err(|error| {
                format!(
                    "Failed to encode TIFF metadata tag 0x{:04x}: {error}",
                    tag.as_u16()
                )
            })?;
        entries.extend([(TiffTag::from_u16_exhaustive(tag.as_u16()), entry)]);
    }

    directory.extend_from(&entries);
    Ok(())
}

#[cfg(not(target_os = "android"))]
fn encode_streaming_tiff<F>(
    output: &mut fs::File,
    width: u32,
    height: u32,
    embed_color_profile: bool,
    export_metadata: Option<&ExifMetadata>,
    render_rows: F,
) -> Result<(), String>
where
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String>,
{
    const ROWS_PER_STRIP: u32 = 64;

    let mut encoder = StreamingTiffEncoder::new(output)
        .map_err(|error| format!("Failed to start streaming TIFF encoder: {error}"))?;

    let interoperability_directory = match export_metadata {
        Some(metadata) if has_writable_tiff_metadata_group(metadata, ExifTagGroup::INTEROP) => {
            let mut directory = encoder.extra_directory().map_err(|error| {
                format!("Failed to start TIFF interoperability metadata: {error}")
            })?;
            write_tiff_metadata_group(&mut directory, metadata, ExifTagGroup::INTEROP)?;
            Some(directory.finish_with_offsets().map_err(|error| {
                format!("Failed to finish TIFF interoperability metadata: {error}")
            })?)
        }
        _ => None,
    };

    let exif_directory = match export_metadata {
        Some(metadata) if has_writable_tiff_metadata_group(metadata, ExifTagGroup::EXIF) => {
            let mut directory = encoder
                .extra_directory()
                .map_err(|error| format!("Failed to start TIFF EXIF metadata: {error}"))?;
            write_tiff_metadata_group(&mut directory, metadata, ExifTagGroup::EXIF)?;
            if let Some(interoperability_directory) = interoperability_directory {
                directory
                    .write_tag(TiffTag::Unknown(0xa005), interoperability_directory.offset)
                    .map_err(|error| {
                        format!("Failed to link TIFF interoperability metadata: {error}")
                    })?;
            }
            Some(
                directory
                    .finish_with_offsets()
                    .map_err(|error| format!("Failed to finish TIFF EXIF metadata: {error}"))?,
            )
        }
        _ => None,
    };

    let gps_directory = match export_metadata {
        Some(metadata) if has_writable_tiff_metadata_group(metadata, ExifTagGroup::GPS) => {
            let mut directory = encoder
                .extra_directory()
                .map_err(|error| format!("Failed to start TIFF GPS metadata: {error}"))?;
            write_tiff_metadata_group(&mut directory, metadata, ExifTagGroup::GPS)?;
            Some(
                directory
                    .finish_with_offsets()
                    .map_err(|error| format!("Failed to finish TIFF GPS metadata: {error}"))?,
            )
        }
        _ => None,
    };

    let mut image = encoder
        .new_image::<RGB16>(width, height)
        .map_err(|error| format!("Failed to configure streaming TIFF image: {error}"))?;
    if let Some(metadata) = export_metadata
        && has_writable_tiff_metadata_group(metadata, ExifTagGroup::GENERIC)
    {
        write_tiff_metadata_group(image.encoder(), metadata, ExifTagGroup::GENERIC)?;
    }
    if let Some(exif_directory) = exif_directory {
        image
            .encoder()
            .write_tag(TiffTag::ExifDirectory, exif_directory.offset)
            .map_err(|error| format!("Failed to link TIFF EXIF metadata: {error}"))?;
    }
    if let Some(gps_directory) = gps_directory {
        image
            .encoder()
            .write_tag(TiffTag::GpsDirectory, gps_directory.offset)
            .map_err(|error| format!("Failed to link TIFF GPS metadata: {error}"))?;
    }
    if embed_color_profile {
        image
            .encoder()
            .write_tag(TiffTag::IccProfile, srgb_v4_profile())
            .map_err(|error| format!("Failed to attach TIFF ICC profile: {error}"))?;
    }
    image
        .rows_per_strip(ROWS_PER_STRIP.min(height).max(1))
        .map_err(|error| format!("Failed to configure TIFF strips: {error}"))?;

    let strip_capacity = (width as usize)
        .checked_mul(3)
        .and_then(|samples_per_row| samples_per_row.checked_mul(ROWS_PER_STRIP as usize))
        .ok_or_else(|| format!("TIFF width {width} exceeds the strip size limit"))?;
    let mut strip = Vec::<u16>::new();
    strip
        .try_reserve_exact(strip_capacity)
        .map_err(|error| format!("Failed to reserve the TIFF strip buffer: {error}"))?;
    let mut written_rows = 0_u32;
    {
        let mut sink = |rgba_row: &[u8]| -> Result<(), String> {
            validate_streamed_rgba_row(rgba_row, width, height, written_rows)?;
            for pixel in rgba_row.chunks_exact(4) {
                strip.push(u16::from(pixel[0]) * 257);
                strip.push(u16::from(pixel[1]) * 257);
                strip.push(u16::from(pixel[2]) * 257);
            }
            written_rows += 1;

            let expected_samples = image.next_strip_sample_count() as usize;
            if strip.len() == expected_samples {
                image
                    .write_strip(&strip)
                    .map_err(|error| format!("Failed to write TIFF strip: {error}"))?;
                strip.clear();
            } else if strip.len() > expected_samples {
                return Err(format!(
                    "Streaming TIFF strip exceeded its expected {} samples",
                    expected_samples
                ));
            }
            Ok(())
        };
        render_rows(&mut sink)?;
    }
    if written_rows != height || !strip.is_empty() || image.next_strip_sample_count() != 0 {
        return Err(format!(
            "Streaming TIFF received {written_rows} complete rows; expected {height}"
        ));
    }
    image
        .finish()
        .map_err(|error| format!("Failed to finish streaming TIFF: {error}"))
}

#[cfg(not(target_os = "android"))]
pub(crate) fn encode_rgb16_tiff_with_metadata(
    output: &mut fs::File,
    width: u32,
    height: u32,
    rgb16: &[u16],
    embed_color_profile: bool,
    export_metadata: Option<&ExifMetadata>,
) -> Result<(), String> {
    let expected_samples = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            "The RGB16 TIFF dimensions exceed the addressable sample count".to_string()
        })?;
    if rgb16.len() != expected_samples {
        return Err(format!(
            "RGB16 TIFF received {} samples; expected {expected_samples}",
            rgb16.len()
        ));
    }

    let mut encoder = StreamingTiffEncoder::new(output)
        .map_err(|error| format!("Failed to start RGB16 TIFF encoder: {error}"))?;
    let interoperability_directory = match export_metadata {
        Some(metadata) if has_writable_tiff_metadata_group(metadata, ExifTagGroup::INTEROP) => {
            let mut directory = encoder.extra_directory().map_err(|error| {
                format!("Failed to start TIFF interoperability metadata: {error}")
            })?;
            write_tiff_metadata_group(&mut directory, metadata, ExifTagGroup::INTEROP)?;
            Some(directory.finish_with_offsets().map_err(|error| {
                format!("Failed to finish TIFF interoperability metadata: {error}")
            })?)
        }
        _ => None,
    };
    let exif_directory = match export_metadata {
        Some(metadata) if has_writable_tiff_metadata_group(metadata, ExifTagGroup::EXIF) => {
            let mut directory = encoder
                .extra_directory()
                .map_err(|error| format!("Failed to start TIFF EXIF metadata: {error}"))?;
            write_tiff_metadata_group(&mut directory, metadata, ExifTagGroup::EXIF)?;
            if let Some(interoperability_directory) = interoperability_directory {
                directory
                    .write_tag(TiffTag::Unknown(0xa005), interoperability_directory.offset)
                    .map_err(|error| {
                        format!("Failed to link TIFF interoperability metadata: {error}")
                    })?;
            }
            Some(
                directory
                    .finish_with_offsets()
                    .map_err(|error| format!("Failed to finish TIFF EXIF metadata: {error}"))?,
            )
        }
        _ => None,
    };
    let gps_directory = match export_metadata {
        Some(metadata) if has_writable_tiff_metadata_group(metadata, ExifTagGroup::GPS) => {
            let mut directory = encoder
                .extra_directory()
                .map_err(|error| format!("Failed to start TIFF GPS metadata: {error}"))?;
            write_tiff_metadata_group(&mut directory, metadata, ExifTagGroup::GPS)?;
            Some(
                directory
                    .finish_with_offsets()
                    .map_err(|error| format!("Failed to finish TIFF GPS metadata: {error}"))?,
            )
        }
        _ => None,
    };

    let mut image = encoder
        .new_image::<RGB16>(width, height)
        .map_err(|error| format!("Failed to configure RGB16 TIFF image: {error}"))?;
    if let Some(metadata) = export_metadata
        && has_writable_tiff_metadata_group(metadata, ExifTagGroup::GENERIC)
    {
        write_tiff_metadata_group(image.encoder(), metadata, ExifTagGroup::GENERIC)?;
    }
    if let Some(exif_directory) = exif_directory {
        image
            .encoder()
            .write_tag(TiffTag::ExifDirectory, exif_directory.offset)
            .map_err(|error| format!("Failed to link TIFF EXIF metadata: {error}"))?;
    }
    if let Some(gps_directory) = gps_directory {
        image
            .encoder()
            .write_tag(TiffTag::GpsDirectory, gps_directory.offset)
            .map_err(|error| format!("Failed to link TIFF GPS metadata: {error}"))?;
    }
    if embed_color_profile {
        image
            .encoder()
            .write_tag(TiffTag::IccProfile, srgb_v4_profile())
            .map_err(|error| format!("Failed to attach TIFF ICC profile: {error}"))?;
    }
    image
        .write_data(rgb16)
        .map_err(|error| format!("Failed to encode RGB16 TIFF pixels: {error}"))
}

#[cfg(not(target_os = "android"))]
#[derive(Clone, Copy)]
enum StreamingExportMetadata<'a> {
    None,
    ExifTiffPayload(&'a [u8]),
    TiffDirectories(&'a ExifMetadata),
}

#[cfg(not(target_os = "android"))]
#[allow(clippy::too_many_arguments)]
fn encode_streaming_rgba_rows<F>(
    output: &mut fs::File,
    width: u32,
    height: u32,
    output_format: &str,
    jpeg_quality: u8,
    embed_color_profile: bool,
    export_metadata: StreamingExportMetadata<'_>,
    render_rows: F,
) -> Result<(), String>
where
    F: FnOnce(&mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String>,
{
    match output_format.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => {
            let exif = match export_metadata {
                StreamingExportMetadata::ExifTiffPayload(payload) => Some(payload),
                _ => None,
            };
            encode_streaming_jpeg(
                output,
                width,
                height,
                jpeg_quality,
                embed_color_profile,
                exif,
                render_rows,
            )
        }
        "png" => {
            let exif = match export_metadata {
                StreamingExportMetadata::ExifTiffPayload(payload) => Some(payload),
                _ => None,
            };
            encode_streaming_png(
                output,
                width,
                height,
                embed_color_profile,
                exif,
                render_rows,
            )
        }
        "tif" | "tiff" => {
            let metadata = match export_metadata {
                StreamingExportMetadata::TiffDirectories(metadata) => Some(metadata),
                _ => None,
            };
            encode_streaming_tiff(
                output,
                width,
                height,
                embed_color_profile,
                metadata,
                render_rows,
            )
        }
        _ => Err(format!(
            "Unsupported streaming export format: {output_format}"
        )),
    }
}

#[cfg(not(target_os = "android"))]
#[allow(clippy::too_many_arguments)]
fn process_and_save_streaming_export(
    path: &str,
    base_image: &DynamicImage,
    js_adjustments: &Value,
    output_path: &Path,
    output_format: &str,
    export_settings: &ExportSettings,
    context: &GpuContext,
    state: &tauri::State<AppState>,
    is_raw: bool,
    app_handle: &tauri::AppHandle,
    cancellation_token: &AtomicBool,
) -> Result<(), String> {
    let prepared = prepare_export_render(
        path,
        base_image,
        js_adjustments,
        state,
        is_raw,
        "process_and_save_streaming_export",
        app_handle,
    );
    let (source_width, source_height) = prepared.source.dimensions();
    let (target_width, target_height) = export_settings
        .resize
        .as_ref()
        .map(|resize| calculate_resize_target(source_width, source_height, resize))
        .unwrap_or((source_width, source_height));
    let is_tiff_output = matches!(output_format.to_ascii_lowercase().as_str(), "tif" | "tiff");
    let export_exif = if is_tiff_output {
        None
    } else {
        exif_processing::export_metadata_tiff_payload(
            path,
            output_format,
            export_settings.keep_metadata,
            export_settings.strip_gps,
            export_settings.metadata_overrides.as_ref(),
        )?
    };
    let export_tiff_metadata = if is_tiff_output {
        exif_processing::export_metadata_for_streaming_tiff(
            path,
            export_settings.keep_metadata,
            export_settings.strip_gps,
            export_settings.metadata_overrides.as_ref(),
        )?
    } else {
        None
    };
    let streaming_metadata = if let Some(metadata) = export_tiff_metadata.as_ref() {
        StreamingExportMetadata::TiffDirectories(metadata)
    } else if let Some(payload) = export_exif.as_deref() {
        StreamingExportMetadata::ExifTiffPayload(payload)
    } else {
        StreamingExportMetadata::None
    };
    let mut temporary = create_temporary_export(output_path)?;

    let PreparedExportRender {
        source,
        mask_bitmaps,
        adjustments,
        lut,
        unique_hash,
    } = prepared;
    encode_streaming_rgba_rows(
        temporary.as_file_mut(),
        target_width,
        target_height,
        output_format,
        export_settings.jpeg_quality,
        export_settings.embed_color_profile,
        streaming_metadata,
        |encoder_sink| {
            let mut checked_encoder_sink = |row: &[u8]| -> Result<(), String> {
                ensure_export_not_cancelled(cancellation_token)?;
                encoder_sink(row)
            };
            transform_streaming_rgba_rows(
                source_width,
                source_height,
                target_width,
                target_height,
                export_settings.watermark.as_ref(),
                &mut checked_encoder_sink,
                |transform_sink| {
                    let mut checked_sink = |row: &[u8]| -> Result<(), String> {
                        ensure_export_not_cancelled(cancellation_token)?;
                        transform_sink(row)
                    };
                    let rendered_dimensions = process_and_stream_rgba_rows(
                        context,
                        state,
                        source.gpu_input(),
                        unique_hash,
                        RenderRequest {
                            adjustments,
                            mask_bitmaps: &mask_bitmaps,
                            lut,
                            roi: None,
                        },
                        "process_and_save_streaming_export",
                        &mut checked_sink,
                    )?;
                    if rendered_dimensions != (source_width, source_height) {
                        return Err(format!(
                            "Streamed GPU dimensions {:?} do not match source dimensions {}x{}",
                            rendered_dimensions, source_width, source_height
                        ));
                    }
                    Ok(())
                },
            )
        },
    )?;
    temporary
        .as_file_mut()
        .flush()
        .map_err(|error| format!("Failed to flush temporary export: {error}"))?;
    ensure_export_not_cancelled(cancellation_token)?;
    ensure_export_not_cancelled(cancellation_token)?;
    preserve_existing_export_permissions(&temporary, output_path)?;

    temporary.persist(output_path).map_err(|error| {
        format!(
            "Failed to atomically publish streamed export '{}': {}",
            output_path.display(),
            error.error
        )
    })?;
    Ok(())
}

#[cfg(target_os = "android")]
pub fn mime_type_for_extension(extension: &str) -> &'static str {
    match extension {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        "tif" | "tiff" => "image/tiff",
        "jxl" => "image/jxl",
        _ => "application/octet-stream",
    }
}

#[allow(clippy::too_many_arguments)]
fn process_image_for_export(
    path: &str,
    base_image: &DynamicImage,
    js_adjustments: &Value,
    export_settings: &ExportSettings,
    context: &GpuContext,
    state: &tauri::State<AppState>,
    is_raw: bool,
    app_handle: &tauri::AppHandle,
) -> Result<DynamicImage, String> {
    let processed_image = process_image_for_export_pipeline(
        path,
        base_image,
        js_adjustments,
        context,
        state,
        is_raw,
        "process_image_for_export",
        app_handle,
    )?;

    apply_export_resize_and_watermark(processed_image, export_settings)
}

fn build_single_mask_adjustments(all: &AllAdjustments, mask_index: usize) -> AllAdjustments {
    let mut single = AllAdjustments {
        global: all.global,
        mask_adjustments: all.mask_adjustments,
        mask_count: 1,
        tile_offset_x: all.tile_offset_x,
        tile_offset_y: all.tile_offset_y,
        mask_atlas_cols: all.mask_atlas_cols,
    };
    single.mask_adjustments[0] = all.mask_adjustments[mask_index];
    for i in 1..single.mask_adjustments.len() {
        single.mask_adjustments[i] = Default::default();
    }
    single
}

fn encode_grayscale_to_png(bitmap: &GrayImage) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    bitmap
        .write_to(&mut cursor, ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(buf)
}

#[cfg(test)]
pub(crate) fn encode_image_to_bytes(
    image: &DynamicImage,
    output_format: &str,
    jpeg_quality: u8,
) -> Result<Vec<u8>, String> {
    encode_image_to_bytes_with_profile(image, output_format, jpeg_quality, true)
}

fn encode_image_to_bytes_with_profile(
    image: &DynamicImage,
    output_format: &str,
    jpeg_quality: u8,
    embed_color_profile: bool,
) -> Result<Vec<u8>, String> {
    let mut image_bytes = Vec::new();
    let mut cursor = Cursor::new(&mut image_bytes);

    match output_format.to_lowercase().as_str() {
        "jxl" => {
            let (width, height) = image.dimensions();
            let has_alpha = image.color().has_alpha();
            let metadata = if embed_color_profile {
                JxlImageMetadata::new().with_icc_profile(srgb_v4_profile())
            } else {
                JxlImageMetadata::new()
            };

            let jxl_data = if jpeg_quality == 100 {
                if has_alpha {
                    let rgba = image.to_rgba8();
                    LosslessConfig::new()
                        .encode_request(width, height, PixelLayout::Rgba8)
                        .with_metadata(&metadata)
                        .encode(rgba.as_raw())
                        .map_err(|e| format!("Failed to encode lossless JXL: {}", e))?
                } else {
                    let rgb = image.to_rgb8();
                    LosslessConfig::new()
                        .encode_request(width, height, PixelLayout::Rgb8)
                        .with_metadata(&metadata)
                        .encode(rgb.as_raw())
                        .map_err(|e| format!("Failed to encode lossless JXL: {}", e))?
                }
            } else {
                let jxl_quality = calibrated_jxl_quality(jpeg_quality as f32);
                let distance = quality_to_distance(jxl_quality);

                if has_alpha {
                    let rgba = image.to_rgba8();
                    LossyConfig::new(distance)
                        .encode_request(width, height, PixelLayout::Rgba8)
                        .with_metadata(&metadata)
                        .encode(rgba.as_raw())
                        .map_err(|e| format!("Failed to encode lossy JXL: {}", e))?
                } else {
                    let rgb = image.to_rgb8();
                    LossyConfig::new(distance)
                        .encode_request(width, height, PixelLayout::Rgb8)
                        .with_metadata(&metadata)
                        .encode(rgb.as_raw())
                        .map_err(|e| format!("Failed to encode lossy JXL: {}", e))?
                }
            };

            return Ok(jxl_data);
        }
        "webp" => {
            let encoder = webp::Encoder::from_image(image)
                .map_err(|_| "Failed to create WebP encoder".to_string())?;
            let webp_mem = encoder.encode(jpeg_quality as f32);
            return if embed_color_profile {
                embed_icc_in_webp(
                    webp_mem.as_ref(),
                    srgb_v4_profile(),
                    image.width(),
                    image.height(),
                    image.color().has_alpha(),
                )
            } else {
                Ok(webp_mem.to_vec())
            };
        }
        "jpg" | "jpeg" => {
            let rgb_image = image.to_rgb8();
            let mut encoder = JpegEncoder::new_with_quality(&mut cursor, jpeg_quality);
            if embed_color_profile {
                encoder
                    .set_icc_profile(srgb_v4_profile().to_vec())
                    .map_err(|e| format!("Failed to attach JPEG ICC profile: {e}"))?;
            }
            rgb_image
                .write_with_encoder(encoder)
                .map_err(|e| e.to_string())?;
        }
        "png" => {
            let image_to_encode = if image.as_rgb32f().is_some() {
                DynamicImage::ImageRgb16(image.to_rgb16())
            } else {
                image.clone()
            };

            let mut encoder = PngEncoder::new(&mut cursor);
            if embed_color_profile {
                encoder
                    .set_icc_profile(srgb_v4_profile().to_vec())
                    .map_err(|e| format!("Failed to attach PNG ICC profile: {e}"))?;
            }
            image_to_encode
                .write_with_encoder(encoder)
                .map_err(|e| e.to_string())?;
        }
        "tiff" => {
            let mut encoder = TiffEncoder::new(&mut cursor);
            if embed_color_profile {
                encoder
                    .set_icc_profile(srgb_v4_profile().to_vec())
                    .map_err(|e| format!("Failed to attach TIFF ICC profile: {e}"))?;
            }
            DynamicImage::ImageRgb16(image.to_rgb16())
                .write_with_encoder(encoder)
                .map_err(|e| e.to_string())?;
        }
        "avif" => {
            image
                .write_to(&mut cursor, image::ImageFormat::Avif)
                .map_err(|e| e.to_string())?;
        }
        _ => return Err(format!("Unsupported file format: {}", output_format)),
    };
    Ok(image_bytes)
}

fn write_webp_chunk(output: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    output.extend_from_slice(fourcc);
    output.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    output.extend_from_slice(payload);
    if !payload.len().is_multiple_of(2) {
        output.push(0);
    }
}

fn embed_icc_in_webp(
    encoded: &[u8],
    icc_profile: &[u8],
    width: u32,
    height: u32,
    has_alpha: bool,
) -> Result<Vec<u8>, String> {
    if encoded.len() < 12 || &encoded[0..4] != b"RIFF" || &encoded[8..12] != b"WEBP" {
        return Err("WebP encoder returned an invalid RIFF container".to_string());
    }
    if width == 0 || height == 0 || width > 0x01_00_00_00 || height > 0x01_00_00_00 {
        return Err("WebP dimensions cannot be represented by a VP8X header".to_string());
    }

    let mut output = Vec::with_capacity(encoded.len() + icc_profile.len() + 32);
    output.extend_from_slice(b"RIFF\0\0\0\0WEBP");

    let mut offset = 12;
    let mut wrote_icc = false;
    let mut found_vp8x = false;
    while offset + 8 <= encoded.len() {
        let fourcc: [u8; 4] = encoded[offset..offset + 4]
            .try_into()
            .map_err(|_| "Invalid WebP chunk identifier".to_string())?;
        let chunk_len = u32::from_le_bytes(
            encoded[offset + 4..offset + 8]
                .try_into()
                .map_err(|_| "Invalid WebP chunk length".to_string())?,
        ) as usize;
        let padded_len = chunk_len + (chunk_len % 2);
        let chunk_end = offset
            .checked_add(8 + padded_len)
            .ok_or_else(|| "WebP chunk size overflow".to_string())?;
        if chunk_end > encoded.len() {
            return Err("WebP chunk extends past the RIFF container".to_string());
        }

        if &fourcc == b"ICCP" {
            if found_vp8x && !wrote_icc {
                write_webp_chunk(&mut output, b"ICCP", icc_profile);
                wrote_icc = true;
            }
        } else if &fourcc == b"VP8X" {
            if chunk_len != 10 {
                return Err("WebP VP8X chunk has an invalid length".to_string());
            }
            let mut payload = encoded[offset + 8..offset + 18].to_vec();
            payload[0] |= 0x20;
            write_webp_chunk(&mut output, b"VP8X", &payload);
            found_vp8x = true;
            if !wrote_icc {
                write_webp_chunk(&mut output, b"ICCP", icc_profile);
                wrote_icc = true;
            }
        } else {
            output.extend_from_slice(&encoded[offset..chunk_end]);
        }
        offset = chunk_end;
    }

    if offset != encoded.len() {
        return Err("WebP container has trailing partial chunk data".to_string());
    }

    if !found_vp8x {
        let image_chunks = output.split_off(12);
        let mut vp8x = [0u8; 10];
        vp8x[0] = 0x20 | if has_alpha { 0x10 } else { 0 };
        let width_minus_one = width - 1;
        let height_minus_one = height - 1;
        vp8x[4..7].copy_from_slice(&width_minus_one.to_le_bytes()[..3]);
        vp8x[7..10].copy_from_slice(&height_minus_one.to_le_bytes()[..3]);
        write_webp_chunk(&mut output, b"VP8X", &vp8x);
        write_webp_chunk(&mut output, b"ICCP", icc_profile);
        output.extend_from_slice(&image_chunks);
    } else if !wrote_icc {
        return Err("Failed to insert the WebP ICC profile".to_string());
    }

    let riff_size = u32::try_from(output.len().saturating_sub(8))
        .map_err(|_| "WebP output is too large for RIFF".to_string())?;
    output[4..8].copy_from_slice(&riff_size.to_le_bytes());
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn export_masks_for_image(
    base_image: &DynamicImage,
    js_adjustments: &Value,
    export_settings: &ExportSettings,
    output_path_obj: &std::path::Path,
    source_path_str: &str,
    context: &Arc<GpuContext>,
    state: &tauri::State<AppState>,
    is_raw: bool,
    app_handle: &tauri::AppHandle,
    cancellation_token: &AtomicBool,
) -> Result<(), String> {
    ensure_export_not_cancelled(cancellation_token)?;
    let (transformed_image, unscaled_crop_offset) =
        apply_all_transformations(Cow::Borrowed(base_image), js_adjustments);
    ensure_export_not_cancelled(cancellation_token)?;
    let (img_w, img_h) = transformed_image.dimensions();
    let mask_definitions: Vec<MaskDefinition> = js_adjustments
        .get("masks")
        .and_then(|m| serde_json::from_value(m.clone()).ok())
        .unwrap_or_default();

    let warped_image = resolve_warped_image_for_masks(state, js_adjustments, &mask_definitions);
    let mut mask_bitmaps = Vec::with_capacity(mask_definitions.len());
    for definition in &mask_definitions {
        ensure_export_not_cancelled(cancellation_token)?;
        if let Some(bitmap) = generate_mask_bitmap(
            definition,
            img_w,
            img_h,
            1.0,
            unscaled_crop_offset,
            warped_image.as_deref(),
        ) {
            mask_bitmaps.push(Arc::new(bitmap));
        }
        ensure_export_not_cancelled(cancellation_token)?;
    }

    if !mask_bitmaps.is_empty() {
        let tm_override = resolve_tonemapper_override_from_handle(app_handle, is_raw);
        let all_adjustments = get_all_adjustments_from_json(js_adjustments, is_raw, tm_override);
        let lut_path = js_adjustments["lutPath"].as_str();
        let lut = lut_path.and_then(|p| get_or_load_lut(state, p).ok());
        let unique_hash = calculate_full_job_hash(source_path_str, js_adjustments);
        let output_dir = output_path_obj.parent().unwrap_or(output_path_obj);
        let stem = output_path_obj
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("export");
        let extension = output_path_obj
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("jpg");

        for (i, _) in mask_bitmaps.iter().enumerate() {
            ensure_export_not_cancelled(cancellation_token)?;
            let single_adjustments = build_single_mask_adjustments(&all_adjustments, i);
            let full_white_mask = ImageBuffer::from_fn(img_w, img_h, |_, _| Luma([255u8]));
            let single_bitmaps: Vec<SharedMaskBitmap> = vec![Arc::new(full_white_mask)];

            let processed = process_and_get_dynamic_image(
                context,
                state,
                transformed_image.as_ref(),
                unique_hash,
                RenderRequest {
                    adjustments: single_adjustments,
                    mask_bitmaps: &single_bitmaps,
                    lut: lut.clone(),
                    roi: None,
                },
                "export_mask_image",
            )?;
            ensure_export_not_cancelled(cancellation_token)?;

            let with_options = apply_export_resize_and_watermark(processed, export_settings)?;
            let (out_w, out_h) = with_options.dimensions();

            let alpha_resized = imageops::resize(
                mask_bitmaps[i].as_ref(),
                out_w,
                out_h,
                imageops::FilterType::Lanczos3,
            );
            ensure_export_not_cancelled(cancellation_token)?;

            let mask_image_path =
                output_dir.join(format!("{}_mask_{}_image.{}", stem, i, extension));
            let mask_alpha_path = output_dir.join(format!("{}_mask_{}_alpha.png", stem, i));

            save_image_with_metadata(
                &with_options,
                &mask_image_path,
                source_path_str,
                export_settings,
                cancellation_token,
            )?;
            ensure_export_not_cancelled(cancellation_token)?;

            if export_settings.preserve_timestamps {
                set_timestamps_from_exif(Path::new(source_path_str), &mask_image_path);
            }

            let alpha_bytes = encode_grayscale_to_png(&alpha_resized)?;
            ensure_export_not_cancelled(cancellation_token)?;
            #[cfg(target_os = "android")]
            {
                let file_name = mask_alpha_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| "Missing Android mask export file name".to_string())?;
                crate::android_integration::save_image_bytes_to_android_gallery(
                    file_name,
                    "image/png",
                    &alpha_bytes,
                )?;
            }

            #[cfg(not(target_os = "android"))]
            fs::write(&mask_alpha_path, alpha_bytes).map_err(|e| e.to_string())?;
            ensure_export_not_cancelled(cancellation_token)?;
        }
    }
    Ok(())
}

fn export_adjustments_as_lut(
    js_adjustments: &Value,
    source_path_str: &str,
    context: &Arc<GpuContext>,
    state: &tauri::State<AppState>,
    app_handle: &tauri::AppHandle,
    cancellation_token: &AtomicBool,
) -> Result<Vec<u8>, String> {
    ensure_export_not_cancelled(cancellation_token)?;
    let lut_size = 33;
    let identity_image = generate_identity_lut_image(lut_size);

    let tm_override = resolve_tonemapper_override_from_handle(app_handle, false);
    let mut all_adjustments = get_all_adjustments_from_json(js_adjustments, false, tm_override);

    all_adjustments.global.show_clipping = 0;
    all_adjustments.global.vignette_amount = 0.0;
    all_adjustments.global.grain_amount = 0.0;
    all_adjustments.global.sharpness = 0.0;
    all_adjustments.global.clarity = 0.0;
    all_adjustments.global.dehaze = 0.0;
    all_adjustments.global.structure = 0.0;
    all_adjustments.global.centré = 0.0;
    all_adjustments.global.glow_amount = 0.0;
    all_adjustments.global.halation_amount = 0.0;
    all_adjustments.global.flare_amount = 0.0;
    all_adjustments.global.luma_noise_reduction = 0.0;
    all_adjustments.global.color_noise_reduction = 0.0;
    all_adjustments.global.chromatic_aberration_red_cyan = 0.0;
    all_adjustments.global.chromatic_aberration_blue_yellow = 0.0;

    let lut_path = js_adjustments["lutPath"].as_str();
    let lut = lut_path.and_then(|p| get_or_load_lut(state, p).ok());
    let unique_hash = calculate_full_job_hash(source_path_str, js_adjustments);

    let processed_lut = process_and_get_dynamic_image(
        context,
        state,
        &identity_image,
        unique_hash,
        RenderRequest {
            adjustments: all_adjustments,
            mask_bitmaps: &[],
            lut,
            roi: None,
        },
        "export_lut",
    )?;
    ensure_export_not_cancelled(cancellation_token)?;

    let cube_lut = convert_image_to_cube_lut(&processed_lut, lut_size)?;
    ensure_export_not_cancelled(cancellation_token)?;
    Ok(cube_lut)
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn export_images_impl(
    paths: Vec<String>,
    output_folder_or_file: String,
    is_explicit_file_path: bool,
    base_origin_folders: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    adjustments_mode: ExportAdjustmentsMode,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
    completion_tx: Option<tokio::sync::oneshot::Sender<Result<(), usize>>>,
) -> Result<(), String> {
    // The editor can select a virtual copy (`?vc=N`), while each worker compares the
    // physical source path. Normalize once so the visible edit is always the one exported.
    let adjustments_mode = adjustments_mode.normalize_active_path();
    let cancellation_token = register_export_task(&state.export_task_token)?;
    let task_guard = ExportTaskGuard::with_app_handle(
        Arc::clone(&state.export_task_token),
        Arc::clone(&cancellation_token),
        app_handle.clone(),
    );
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;

    if cancellation_token.load(Ordering::SeqCst) {
        return Ok(());
    }

    let context = match get_or_init_gpu_context(&state, &app_handle) {
        Ok(context) => context,
        Err(_) if cancellation_token.load(Ordering::SeqCst) => return Ok(()),
        Err(error) => return Err(error),
    };

    if cancellation_token.load(Ordering::SeqCst) {
        return Ok(());
    }

    let context = Arc::new(context);
    let progress_counter = Arc::new(AtomicUsize::new(0));

    let available_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut sys = sysinfo::System::new();
    sys.refresh_memory();

    let available_ram_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let ram_based_limit = (available_ram_gb / 4.0).floor() as usize;

    let num_threads = if paths.len() == 1 {
        1
    } else {
        available_cores.min(ram_based_limit).clamp(1, 4)
    };

    log::info!(
        "Batch Export: {} cores, {:.1} GB free RAM -> {} threads",
        available_cores,
        available_ram_gb,
        num_threads
    );

    let _export_task = tokio::spawn(async move {
        let _task_guard = task_guard;
        let output_folder_path = std::path::Path::new(&output_folder_or_file);
        let total_paths = paths.len();
        let settings = load_settings(app_handle.clone()).unwrap_or_default();

        let mut base_path_counts: HashMap<String, usize> = HashMap::new();
        let mut export_items = Vec::with_capacity(total_paths);

        for (i, path_str) in paths.into_iter().enumerate() {
            let (source_path, _) = parse_virtual_path(&path_str);
            let source_str = source_path.to_string_lossy().to_string();
            let count = base_path_counts.entry(source_str.clone()).or_insert(0);
            *count += 1;

            let mut explicit_vc = None;
            if let Some(idx) = path_str.rfind("vc=") {
                let id_str = path_str[idx + 3..].split('&').next().unwrap_or("");
                if let Ok(id) = id_str.parse::<u32>() {
                    explicit_vc = Some(id);
                }
            }
            if explicit_vc.is_none() {
                let lower = path_str.to_lowercase();
                if let Some(idx) = lower.rfind("_vc") {
                    let id_str: String = lower[idx + 3..]
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if let Ok(id) = id_str.parse::<u32>() {
                        explicit_vc = Some(id);
                    }
                }
            }
            export_items.push((i, path_str, *count, explicit_vc));
        }

        let semaphore = Arc::new(tokio::sync::Semaphore::new(num_threads));
        let mut join_handles = Vec::new();

        for (global_index, image_path_str, appearance_count, explicit_vc) in export_items {
            if cancellation_token.load(Ordering::SeqCst) {
                break;
            }
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            if cancellation_token.load(Ordering::SeqCst) {
                drop(permit);
                break;
            }

            let app_handle_clone = app_handle.clone();
            let context_clone = Arc::clone(&context);
            let progress_counter_clone = Arc::clone(&progress_counter);
            let output_folder_path = output_folder_path.to_path_buf();
            let base_origin_folders = base_origin_folders.clone();
            let export_settings = export_settings.clone();
            let output_format = output_format.clone();
            let settings = settings.clone();
            let cancellation_token_clone = Arc::clone(&cancellation_token);
            let adjustments_mode = adjustments_mode.clone();

            let handle = tokio::task::spawn_blocking(move || {
                ensure_export_not_cancelled(&cancellation_token_clone)?;

                let state = app_handle_clone.state::<AppState>();
                let (source_path, sidecar_path) = parse_virtual_path(&image_path_str);
                let source_path_str = source_path.to_string_lossy().to_string();

                let is_current_edit = match &adjustments_mode {
                    ExportAdjustmentsMode::UseSidecars { active_path, .. } => {
                        Some(&source_path_str) == active_path.as_ref()
                    }
                    ExportAdjustmentsMode::GlobalOverride(_) => false,
                };

                let mut js_adjustments = match &adjustments_mode {
                    ExportAdjustmentsMode::UseSidecars {
                        active_adjustments, ..
                    } => {
                        if is_current_edit {
                            if let Some(adj) = active_adjustments {
                                adj.clone()
                            } else {
                                crate::exif_processing::load_sidecar(&sidecar_path).adjustments
                            }
                        } else {
                            crate::exif_processing::load_sidecar(&sidecar_path).adjustments
                        }
                    }
                    ExportAdjustmentsMode::GlobalOverride(adj) => adj.clone(),
                };

                hydrate_adjustments(&state, &mut js_adjustments);
                let is_raw = is_raw_file(&source_path_str);
                let original_path = std::path::Path::new(&source_path_str);
                let file_date = exif_processing::get_creation_date_from_path(original_path);

                let filename_template = export_settings
                    .filename_template
                    .as_deref()
                    .unwrap_or("{original_filename}_edited");

                let mut new_stem = generate_filename_from_template(
                    filename_template,
                    original_path,
                    global_index + 1,
                    total_paths,
                    &file_date,
                );

                if let Some(vc_id) = explicit_vc {
                    new_stem = format!("{}_VC{:02}", new_stem, vc_id);
                } else if appearance_count > 1 {
                    new_stem = format!("{}_VC{:02}", new_stem, appearance_count - 1);
                }

                let new_filename = format!("{}.{}", new_stem, output_format);
                let output_path = if is_explicit_file_path && total_paths == 1 {
                    output_folder_path
                } else if export_settings.preserve_folders {
                    if let Some(rel_dir) = relative_export_dir_for_preserved_folders(
                        source_path.as_path(),
                        &base_origin_folders,
                    ) {
                        let full_dir = output_folder_path.join(rel_dir);
                        if let Err(e) = std::fs::create_dir_all(&full_dir) {
                            log::warn!("Failed to create export subdirectory: {}", e);
                        }
                        full_dir.join(&new_filename)
                    } else {
                        output_folder_path.join(&new_filename)
                    }
                } else {
                    output_folder_path.join(&new_filename)
                };

                let extension = output_format.to_lowercase();

                let result: Result<(), String> = (|| {
                    if extension == "cube" {
                        let cube_bytes = export_adjustments_as_lut(
                            &js_adjustments,
                            &source_path_str,
                            &context_clone,
                            &state,
                            &app_handle_clone,
                            &cancellation_token_clone,
                        )?;
                        ensure_export_not_cancelled(&cancellation_token_clone)?;
                        #[cfg(target_os = "android")]
                        {
                            let file_name = output_path
                                .file_name()
                                .and_then(|name| name.to_str())
                                .ok_or_else(|| "Missing Android LUT file name".to_string())?;
                            crate::android_integration::save_file_bytes_to_android_downloads(
                                file_name,
                                "application/octet-stream",
                                &cube_bytes,
                            )?;
                        }
                        #[cfg(not(target_os = "android"))]
                        fs::write(&output_path, cube_bytes).map_err(|e| e.to_string())?;
                        ensure_export_not_cancelled(&cancellation_token_clone)?;
                        return Ok(());
                    }

                    let base_image = if is_current_edit {
                        match crate::get_original_image(&state) {
                            Ok((orig_data_arc, _)) => {
                                composite_patches_on_image(&orig_data_arc, &js_adjustments)
                                    .map_err(|e| format!("Failed to composite AI patches: {}", e))?
                            }
                            Err(_) => {
                                let bytes =
                                    fs::read(&source_path_str).map_err(|e| e.to_string())?;
                                load_and_composite(
                                    &bytes,
                                    &source_path_str,
                                    &js_adjustments,
                                    false,
                                    &settings,
                                    None,
                                )
                                .map_err(|e| format!("Failed to load fallback image: {}", e))?
                            }
                        }
                    } else {
                        match read_file_mapped(Path::new(&source_path_str)) {
                            Ok(mmap) => load_and_composite(
                                &mmap,
                                &source_path_str,
                                &js_adjustments,
                                false,
                                &settings,
                                None,
                            )
                            .map_err(|e| format!("Failed to load from mmap: {}", e))?,
                            Err(_) => {
                                let bytes =
                                    fs::read(&source_path_str).map_err(|e| e.to_string())?;
                                load_and_composite(
                                    &bytes,
                                    &source_path_str,
                                    &js_adjustments,
                                    false,
                                    &settings,
                                    None,
                                )
                                .map_err(|e| format!("Failed to load from bytes: {}", e))?
                            }
                        }
                    };
                    ensure_export_not_cancelled(&cancellation_token_clone)?;

                    let mut main_export_adjustments = js_adjustments.clone();
                    if export_settings.export_masks
                        && let Some(obj) = main_export_adjustments.as_object_mut()
                    {
                        obj.insert("masks".to_string(), serde_json::json!([]));
                    }

                    if supports_streaming_export(&extension, &export_settings) {
                        #[cfg(not(target_os = "android"))]
                        process_and_save_streaming_export(
                            &source_path_str,
                            &base_image,
                            &main_export_adjustments,
                            &output_path,
                            &extension,
                            &export_settings,
                            &context_clone,
                            &state,
                            is_raw,
                            &app_handle_clone,
                            &cancellation_token_clone,
                        )?;

                        #[cfg(target_os = "android")]
                        unreachable!("Android exports do not select the filesystem streaming path");
                    } else {
                        let final_image = process_image_for_export(
                            &source_path_str,
                            &base_image,
                            &main_export_adjustments,
                            &export_settings,
                            &context_clone,
                            &state,
                            is_raw,
                            &app_handle_clone,
                        )?;
                        ensure_export_not_cancelled(&cancellation_token_clone)?;
                        save_image_with_metadata(
                            &final_image,
                            &output_path,
                            &source_path_str,
                            &export_settings,
                            &cancellation_token_clone,
                        )?;
                    }
                    ensure_export_not_cancelled(&cancellation_token_clone)?;

                    if export_settings.preserve_timestamps {
                        set_timestamps_from_exif(Path::new(&source_path_str), &output_path);
                    }
                    ensure_export_not_cancelled(&cancellation_token_clone)?;

                    if export_settings.export_masks {
                        export_masks_for_image(
                            &base_image,
                            &js_adjustments,
                            &export_settings,
                            &output_path,
                            &source_path_str,
                            &context_clone,
                            &state,
                            is_raw,
                            &app_handle_clone,
                            &cancellation_token_clone,
                        )?;
                    }

                    Ok(())
                })();

                if !cancellation_token_clone.load(Ordering::SeqCst) {
                    let current_progress =
                        progress_counter_clone.fetch_add(1, Ordering::SeqCst) + 1;
                    let _ = app_handle_clone.emit(
                        "batch-export-progress",
                        serde_json::json!({
                            "current": current_progress,
                            "total": total_paths,
                            "path": &image_path_str
                        }),
                    );
                }

                drop(permit);
                if cancellation_token_clone.load(Ordering::SeqCst) {
                    Err("Export cancelled".to_string())
                } else {
                    result
                }
            });

            join_handles.push(handle);
        }

        let mut results = Vec::new();
        for handle in join_handles {
            match handle.await {
                Ok(res) => results.push(res),
                Err(e) => results.push(Err(format!("Thread crashed: {}", e))),
            }
        }

        let export_state = app_handle.state::<AppState>();
        reclaim_gpu_resources_after_export(&context, export_state.inner());

        let summary = summarize_batch_export_results(results);
        debug_assert_eq!(summary.completed, total_paths);
        let errors = summary.errors;
        let error_count = errors.len();
        let finalized = finish_export_task(
            &export_state.export_task_token,
            &cancellation_token,
            |cancelled| {
                if cancelled {
                    log::info!("Batch export cancelled and worker cleanup completed");
                    let _ = app_handle.emit("export-cancelled", ());
                    return;
                }

                for error in &errors {
                    log::error!("Export error: {}", error);
                    if total_paths == 1 {
                        let _ = app_handle.emit("export-error", error.clone());
                    }
                }

                if error_count > 0 && total_paths > 1 {
                    let _ = app_handle.emit(
                        "export-error",
                        format!("{error_count} of {total_paths} exports failed"),
                    );
                } else if error_count == 0 {
                    let _ = app_handle.emit(
                        "batch-export-progress",
                        serde_json::json!({ "current": total_paths, "total": total_paths, "path": "" }),
                    );
                    let _ = app_handle.emit("export-complete", ());
                }
            },
        );

        if !finalized {
            log::warn!("Ignoring terminal events from a stale export task");
        }

        if let Some(tx) = completion_tx {
            if error_count > 0 {
                let _ = tx.send(Err(error_count));
            } else {
                let _ = tx.send(Ok(()));
            }
        }
    });

    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn export_images(
    paths: Vec<String>,
    output_folder_or_file: String,
    is_explicit_file_path: bool,
    base_origin_folders: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    current_edit_path: Option<String>,
    current_edit_adjustments: Option<Value>,
    wait_for_completion: Option<bool>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let (completion_tx, completion_rx) = if wait_for_completion.unwrap_or(false) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    export_images_impl(
        paths,
        output_folder_or_file,
        is_explicit_file_path,
        base_origin_folders,
        export_settings,
        output_format,
        ExportAdjustmentsMode::UseSidecars {
            active_path: current_edit_path,
            active_adjustments: current_edit_adjustments,
        },
        state,
        app_handle,
        completion_tx,
    )
    .await?;

    match completion_rx {
        Some(rx) => match rx.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error_count)) => Err(format!("Export completed with {error_count} error(s).")),
            Err(_) => Err("Export task ended before reporting completion.".to_string()),
        },
        None => Ok(()),
    }
}

pub async fn run_headless_export(
    session: crate::launch_request::HeadlessExportSession,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    println!("Starting headless export...");
    let state = app_handle.state::<crate::AppState>();

    let source_path = std::path::Path::new(&session.source);
    if !source_path.exists() {
        return Err(format!("Source path does not exist: {}", session.source));
    }

    let mut paths = Vec::new();
    if source_path.is_dir() {
        let images = crate::file_management::list_images_recursive(
            session.source.clone(),
            app_handle.clone(),
        )?;
        paths = images.into_iter().map(|img| img.path).collect();
    } else {
        paths.push(session.source.clone());
    }

    if paths.is_empty() {
        return Err("No supported images found at the source path.".to_string());
    }

    let output_path = std::path::Path::new(&session.output);
    let is_explicit_file_path =
        paths.len() == 1 && output_path.extension().is_some() && !output_path.is_dir();

    if is_explicit_file_path {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create output parent directory: {}", e))?;
        }
    } else {
        std::fs::create_dir_all(&session.output)
            .map_err(|e| format!("Failed to create output directory: {}", e))?;
    }

    println!("Found {} images to export. Processing...", paths.len());

    let export_settings = ExportSettings {
        jpeg_quality: session.quality,
        resize: None,
        keep_metadata: session.keep_metadata,
        metadata_overrides: None,
        preserve_timestamps: true,
        strip_gps: false,
        embed_color_profile: true,
        filename_template: None,
        watermark: None,
        export_masks: false,
        preserve_folders: true,
    };

    let mut custom_adjustments = None;
    if let Some(adj_path) = &session.adjustments_override {
        let content = std::fs::read_to_string(adj_path)
            .map_err(|e| format!("Failed to read adjustments file: {}", e))?;
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse adjustments JSON: {}", e))?;
        custom_adjustments = Some(json);
        println!(
            "Loaded custom adjustments to override sidecars from: {}",
            adj_path
        );
    }

    let (tx, rx) = tokio::sync::oneshot::channel();

    let mode = if let Some(adj) = custom_adjustments {
        ExportAdjustmentsMode::GlobalOverride(adj)
    } else {
        ExportAdjustmentsMode::UseSidecars {
            active_path: None,
            active_adjustments: None,
        }
    };

    export_images_impl(
        paths,
        session.output,
        is_explicit_file_path,
        vec![session.source],
        export_settings,
        session.format,
        mode,
        state.clone(),
        app_handle.clone(),
        Some(tx),
    )
    .await?;

    match rx.await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(errors)) => Err(format!("Export completed with {} errors.", errors)),
        Err(_) => Err("Export task panicked or was cancelled.".to_string()),
    }
}

#[tauri::command]
pub fn cancel_export(
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    match request_export_cancellation(&state.export_task_token, || {
        let _ = app_handle.emit("export-cancelling", ());
    }) {
        ExportCancellationRequest::Requested => {
            log::info!("Export cancellation requested; workers will stop at the next checkpoint");
        }
        ExportCancellationRequest::AlreadyRequested => {
            log::info!("Export cancellation was already requested");
        }
        ExportCancellationRequest::NoActiveTask => {
            return Err("No export task is currently running.".to_string());
        }
    }
    Ok(())
}

#[tauri::command]
pub async fn estimate_export_sizes(
    paths: Vec<String>,
    export_settings: ExportSettings,
    output_format: String,
    current_edit_path: Option<String>,
    current_edit_adjustments: Option<Value>,
    state: tauri::State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<usize, String> {
    if output_format.to_lowercase() == "cube" {
        return Ok(1_050_000 * paths.len());
    }

    if paths.is_empty() {
        return Ok(0);
    }

    let first_path = &paths[0];
    let (source_path, sidecar_path) = parse_virtual_path(first_path);
    let source_path_str = source_path.to_string_lossy().to_string();

    let context = get_or_init_gpu_context(&state, &app_handle)?;
    let is_current_edit = Some(&source_path_str) == current_edit_path.as_ref();
    let is_raw = is_raw_file(&source_path_str);
    let settings = load_settings(app_handle.clone()).unwrap_or_default();

    let single_image_extrapolated_size: usize = if is_current_edit
        && current_edit_adjustments.is_some()
    {
        let loaded_image = state
            .original_image
            .lock()
            .unwrap()
            .clone()
            .ok_or("No original image loaded")?;
        let mut adjustments_clone = current_edit_adjustments.clone().unwrap();
        hydrate_adjustments(&state, &mut adjustments_clone);

        let new_transform_hash = calculate_transform_hash(&adjustments_clone);
        let cached_preview_lock = state.cached_preview.lock().unwrap();
        let preview_dim = settings.editor_preview_resolution.unwrap_or(1920);

        let (preview_image, scale, unscaled_crop_offset) = if let Some(cached) =
            &*cached_preview_lock
        {
            if cached.transform_hash == new_transform_hash && cached.preview_dim == preview_dim {
                let img = Arc::clone(&cached.image);
                let s = cached.scale;
                let offset = cached.unscaled_crop_offset;
                drop(cached_preview_lock);
                (img, s, offset)
            } else {
                drop(cached_preview_lock);
                generate_transformed_preview(
                    &state,
                    &loaded_image,
                    &adjustments_clone,
                    preview_dim,
                )?
            }
        } else {
            drop(cached_preview_lock);
            generate_transformed_preview(&state, &loaded_image, &adjustments_clone, preview_dim)?
        };

        let (img_w, img_h) = preview_image.dimensions();
        let mask_definitions: Vec<MaskDefinition> = adjustments_clone
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();

        let scaled_crop_offset = (
            unscaled_crop_offset.0 * scale,
            unscaled_crop_offset.1 * scale,
        );

        let mask_bitmaps: Vec<SharedMaskBitmap> = mask_definitions
            .iter()
            .filter_map(|def| {
                get_cached_or_generate_mask(
                    &state,
                    def,
                    img_w,
                    img_h,
                    scale,
                    scaled_crop_offset,
                    &adjustments_clone,
                )
            })
            .collect();

        let tm_override = resolve_tonemapper_override_from_handle(&app_handle, is_raw);
        let mut all_adjustments =
            get_all_adjustments_from_json(&adjustments_clone, is_raw, tm_override);
        all_adjustments.global.show_clipping = 0;

        let lut = adjustments_clone["lutPath"]
            .as_str()
            .and_then(|p| get_or_load_lut(&state, p).ok());
        let unique_hash =
            calculate_full_job_hash(&loaded_image.path, &adjustments_clone).wrapping_add(1);

        let processed_preview = process_and_get_dynamic_image(
            &context,
            &state,
            &preview_image,
            unique_hash,
            RenderRequest {
                adjustments: all_adjustments,
                mask_bitmaps: &mask_bitmaps,
                lut,
                roi: None,
            },
            "estimate_export_size",
        )?;

        let preview_bytes = encode_image_to_bytes_with_profile(
            &processed_preview,
            &output_format,
            export_settings.jpeg_quality,
            export_settings.embed_color_profile,
        )?;
        let preview_byte_size = preview_bytes.len();

        let (transformed_full_res, _) =
            apply_all_transformations(&loaded_image.image, &adjustments_clone);
        let (full_w, full_h) = transformed_full_res.dimensions();

        let (final_full_w, final_full_h) = if let Some(resize_opts) = &export_settings.resize {
            calculate_resize_target(full_w, full_h, resize_opts)
        } else {
            (full_w, full_h)
        };

        let (processed_preview_w, processed_preview_h) = processed_preview.dimensions();
        let pixel_ratio = if processed_preview_w > 0 && processed_preview_h > 0 {
            (final_full_w as f64 * final_full_h as f64)
                / (processed_preview_w as f64 * processed_preview_h as f64)
        } else {
            1.0
        };

        (preview_byte_size as f64 * pixel_ratio) as usize
    } else {
        let metadata = crate::exif_processing::load_sidecar(&sidecar_path);
        let mut js_adjustments = metadata.adjustments;

        const ESTIMATE_DIM: u32 = 1280;

        let file_slice: Vec<u8>;
        let mmap_guard;
        let file_data: &[u8] = match read_file_mapped(Path::new(&source_path_str)) {
            Ok(mmap) => {
                mmap_guard = Some(mmap);
                mmap_guard.as_ref().unwrap()
            }
            Err(_) => {
                file_slice = fs::read(&source_path_str).map_err(|io_err| io_err.to_string())?;
                &file_slice
            }
        };

        let original_image =
            load_base_image_from_bytes(file_data, &source_path_str, true, &settings, None)
                .map_err(|e| e.to_string())?;

        let raw_scale_factor = if is_raw {
            crate::raw_processing::get_fast_demosaic_scale_factor(
                file_data,
                original_image.width(),
                original_image.height(),
            )
        } else {
            1.0
        };

        if let Some(crop_val) = js_adjustments.get_mut("crop")
            && let Ok(c) = serde_json::from_value::<Crop>(crop_val.clone())
        {
            *crop_val = serde_json::to_value(Crop {
                x: c.x * raw_scale_factor as f64,
                y: c.y * raw_scale_factor as f64,
                width: c.width * raw_scale_factor as f64,
                height: c.height * raw_scale_factor as f64,
            })
            .unwrap_or(serde_json::Value::Null);
        }

        let (transformed_shrunk_res, unscaled_crop_offset) =
            apply_all_transformations(Cow::Borrowed(&original_image), &js_adjustments);
        let (shrunk_w, shrunk_h) = transformed_shrunk_res.dimensions();

        let preview_base = if shrunk_w > ESTIMATE_DIM || shrunk_h > ESTIMATE_DIM {
            downscale_f32_image(transformed_shrunk_res.as_ref(), ESTIMATE_DIM, ESTIMATE_DIM)
        } else {
            transformed_shrunk_res.into_owned()
        };

        let (preview_w, preview_h) = preview_base.dimensions();
        let gpu_scale = if shrunk_w > 0 {
            preview_w as f32 / shrunk_w as f32
        } else {
            1.0
        };
        let total_scale = gpu_scale * raw_scale_factor;

        let mask_definitions: Vec<MaskDefinition> = js_adjustments
            .get("masks")
            .and_then(|m| serde_json::from_value(m.clone()).ok())
            .unwrap_or_default();
        let scaled_crop_offset = (
            unscaled_crop_offset.0 * gpu_scale,
            unscaled_crop_offset.1 * gpu_scale,
        );

        let mask_bitmaps: Vec<SharedMaskBitmap> = mask_definitions
            .iter()
            .filter_map(|def| {
                get_cached_or_generate_mask(
                    &state,
                    def,
                    preview_w,
                    preview_h,
                    total_scale,
                    scaled_crop_offset,
                    &js_adjustments,
                )
            })
            .collect();

        let tm_override = resolve_tonemapper_override_from_handle(&app_handle, is_raw);
        let mut all_adjustments =
            get_all_adjustments_from_json(&js_adjustments, is_raw, tm_override);
        all_adjustments.global.show_clipping = 0;

        let lut = js_adjustments["lutPath"]
            .as_str()
            .and_then(|p| get_or_load_lut(&state, p).ok());
        let unique_hash =
            calculate_full_job_hash(&source_path_str, &js_adjustments).wrapping_add(1);

        let processed_preview = process_and_get_dynamic_image(
            &context,
            &state,
            &preview_base,
            unique_hash,
            RenderRequest {
                adjustments: all_adjustments,
                mask_bitmaps: &mask_bitmaps,
                lut,
                roi: None,
            },
            "estimate_batch_export_size",
        )?;

        let preview_bytes = encode_image_to_bytes_with_profile(
            &processed_preview,
            &output_format,
            export_settings.jpeg_quality,
            export_settings.embed_color_profile,
        )?;
        let single_image_estimated_size = preview_bytes.len();

        let full_w = (shrunk_w as f32 / raw_scale_factor).round() as u32;
        let full_h = (shrunk_h as f32 / raw_scale_factor).round() as u32;

        let (final_full_w, final_full_h) = if let Some(resize_opts) = &export_settings.resize {
            calculate_resize_target(full_w, full_h, resize_opts)
        } else {
            (full_w, full_h)
        };

        let (processed_preview_w, processed_preview_h) = processed_preview.dimensions();
        let pixel_ratio = if processed_preview_w > 0 && processed_preview_h > 0 {
            (final_full_w as f64 * final_full_h as f64)
                / (processed_preview_w as f64 * processed_preview_h as f64)
        } else {
            1.0
        };

        (single_image_estimated_size as f64 * pixel_ratio) as usize
    };

    Ok(single_image_extrapolated_size * paths.len())
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek, SeekFrom};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use crate::color_management::validate_icc_profile;
    use image::codecs::{jpeg::JpegDecoder, png::PngDecoder, tiff::TiffDecoder, webp::WebPDecoder};
    use image::{ImageDecoder, Rgb, RgbImage};
    use jxl_oxide::integration::JxlDecoder;
    use little_exif::exif_tag::ExifTag;
    use little_exif::filetype::FileExtension;
    use little_exif::metadata::Metadata;
    use little_exif::rational::uR64;

    use super::*;

    fn fixture_image(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
            Rgb([
                ((x * 17 + y * 3) % 256) as u8,
                ((x * 5 + y * 13) % 256) as u8,
                ((x * 11 + y * 7) % 256) as u8,
            ])
        }))
    }

    #[test]
    fn export_adjustment_mode_normalizes_virtual_copy_paths() {
        let mode = ExportAdjustmentsMode::UseSidecars {
            active_path: Some("/photos/source.jpg?vc=3".to_string()),
            active_adjustments: Some(serde_json::json!({ "exposure": 1.0 })),
        }
        .normalize_active_path();

        match mode {
            ExportAdjustmentsMode::UseSidecars {
                active_path,
                active_adjustments,
            } => {
                assert_eq!(active_path.as_deref(), Some("/photos/source.jpg"));
                assert_eq!(
                    active_adjustments,
                    Some(serde_json::json!({ "exposure": 1.0 }))
                );
            }
            ExportAdjustmentsMode::GlobalOverride(_) => panic!("unexpected global override"),
        }
    }

    #[test]
    fn geometry_output_streaming_supports_orthogonal_post_transforms_only() {
        let adjustments = serde_json::json!({
            "transformVertical": 18.0,
            "crop": { "unit": "px", "x": 4.0, "y": 3.0, "width": 80.0, "height": 60.0 }
        });
        let params = get_geometry_params_from_json(&adjustments);
        assert!(can_stream_geometry_output(&adjustments, &params, &[]));

        for compatible in [
            serde_json::json!({ "transformVertical": 18.0, "orientationSteps": 1 }),
            serde_json::json!({ "transformVertical": 18.0, "orientationSteps": 2, "flipHorizontal": true }),
            serde_json::json!({ "transformVertical": 18.0, "orientationSteps": 3, "flipVertical": true }),
            serde_json::json!({ "transformVertical": 18.0, "flipHorizontal": true, "flipVertical": true }),
        ] {
            let params = get_geometry_params_from_json(&compatible);
            assert!(can_stream_geometry_output(&compatible, &params, &[]));
        }

        for incompatible in [
            serde_json::json!({ "transformVertical": 18.0, "rotation": 0.5 }),
            serde_json::json!({ "transformVertical": 18.0, "lensBlurEnabled": true }),
        ] {
            let params = get_geometry_params_from_json(&incompatible);
            assert!(!can_stream_geometry_output(&incompatible, &params, &[]));
        }

        let color_mask: MaskDefinition = serde_json::from_value(serde_json::json!({
            "id": "color-mask",
            "name": "Color",
            "visible": true,
            "invert": false,
            "opacity": 100.0,
            "adjustments": null,
            "subMasks": [{
                "id": "color",
                "type": "color",
                "visible": true,
                "invert": false,
                "opacity": 100.0,
                "mode": "additive",
                "parameters": {}
            }]
        }))
        .expect("color mask fixture");
        assert!(!can_stream_geometry_output(
            &adjustments,
            &params,
            &[color_mask]
        ));
    }

    fn encode_streaming_fixture(
        image: &DynamicImage,
        output_format: &str,
        jpeg_quality: u8,
    ) -> Result<Vec<u8>, String> {
        encode_streaming_fixture_with_metadata(
            image,
            output_format,
            jpeg_quality,
            StreamingExportMetadata::None,
        )
    }

    fn encode_streaming_fixture_with_exif(
        image: &DynamicImage,
        output_format: &str,
        jpeg_quality: u8,
        export_exif: Option<&[u8]>,
    ) -> Result<Vec<u8>, String> {
        encode_streaming_fixture_with_metadata(
            image,
            output_format,
            jpeg_quality,
            export_exif.map_or(StreamingExportMetadata::None, |payload| {
                StreamingExportMetadata::ExifTiffPayload(payload)
            }),
        )
    }

    fn encode_streaming_fixture_with_tiff_metadata(
        image: &DynamicImage,
        export_metadata: Option<&Metadata>,
    ) -> Result<Vec<u8>, String> {
        encode_streaming_fixture_with_metadata(
            image,
            "tiff",
            100,
            export_metadata.map_or(StreamingExportMetadata::None, |metadata| {
                StreamingExportMetadata::TiffDirectories(metadata)
            }),
        )
    }

    fn encode_streaming_fixture_with_metadata(
        image: &DynamicImage,
        output_format: &str,
        jpeg_quality: u8,
        export_metadata: StreamingExportMetadata<'_>,
    ) -> Result<Vec<u8>, String> {
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let row_bytes = width as usize * 4;
        let mut file = tempfile::tempfile().map_err(|error| error.to_string())?;
        encode_streaming_rgba_rows(
            &mut file,
            width,
            height,
            output_format,
            jpeg_quality,
            true,
            export_metadata,
            |sink| {
                for row in rgba.as_raw().chunks_exact(row_bytes) {
                    sink(row)?;
                }
                Ok(())
            },
        )?;
        file.seek(SeekFrom::Start(0))
            .map_err(|error| error.to_string())?;
        let mut encoded = Vec::new();
        file.read_to_end(&mut encoded)
            .map_err(|error| error.to_string())?;
        Ok(encoded)
    }

    fn collect_streaming_transform(
        image: &DynamicImage,
        settings: &ExportSettings,
    ) -> Result<RgbaImage, String> {
        let rgba = image.to_rgba8();
        let (source_width, source_height) = rgba.dimensions();
        let (target_width, target_height) = settings
            .resize
            .as_ref()
            .map(|resize| calculate_resize_target(source_width, source_height, resize))
            .unwrap_or((source_width, source_height));
        let source_row_bytes = source_width as usize * 4;
        let mut transformed =
            Vec::with_capacity(target_width as usize * target_height as usize * 4);
        let mut output_sink = |row: &[u8]| -> Result<(), String> {
            transformed.extend_from_slice(row);
            Ok(())
        };
        transform_streaming_rgba_rows(
            source_width,
            source_height,
            target_width,
            target_height,
            settings.watermark.as_ref(),
            &mut output_sink,
            |source_sink| {
                for row in rgba.as_raw().chunks_exact(source_row_bytes) {
                    source_sink(row)?;
                }
                Ok(())
            },
        )?;

        RgbaImage::from_raw(target_width, target_height, transformed)
            .ok_or_else(|| "Streaming transform returned an invalid RGBA buffer".to_string())
    }

    fn assert_decoder_contract<D: ImageDecoder>(
        mut decoder: D,
        expected: (u32, u32),
        expect_exact_profile_bytes: bool,
    ) {
        assert_eq!(decoder.dimensions(), expected);
        let profile = decoder
            .icc_profile()
            .expect("read embedded ICC")
            .expect("export must contain an ICC profile");
        validate_icc_profile(&profile).expect("exported ICC profile must be valid RGB ICC");

        if expect_exact_profile_bytes {
            assert_eq!(profile, srgb_v4_profile());
        }
    }

    fn mean_rgb_abs_error(reference: &RgbImage, candidate: &RgbImage) -> f64 {
        assert_eq!(reference.dimensions(), candidate.dimensions());
        let total_delta: u64 = reference
            .pixels()
            .zip(candidate.pixels())
            .map(|(reference, candidate)| {
                (0..3)
                    .map(|channel| u64::from(reference[channel].abs_diff(candidate[channel])))
                    .sum::<u64>()
            })
            .sum();
        total_delta as f64 / (reference.width() as f64 * reference.height() as f64 * 3.0)
    }

    #[test]
    fn supported_color_managed_exports_round_trip_dimensions_and_icc() {
        let image = fixture_image(37, 23);

        let jpeg = encode_image_to_bytes(&image, "jpeg", 92).expect("encode JPEG");
        assert_decoder_contract(
            JpegDecoder::new(Cursor::new(jpeg)).expect("decode JPEG"),
            (37, 23),
            true,
        );

        let png = encode_image_to_bytes(&image, "png", 100).expect("encode PNG");
        assert_decoder_contract(
            PngDecoder::new(Cursor::new(png)).expect("decode PNG"),
            (37, 23),
            true,
        );

        let tiff = encode_image_to_bytes(&image, "tiff", 100).expect("encode TIFF");
        assert_decoder_contract(
            TiffDecoder::new(Cursor::new(tiff)).expect("decode TIFF"),
            (37, 23),
            true,
        );

        let webp = encode_image_to_bytes(&image, "webp", 90).expect("encode WebP");
        assert_decoder_contract(
            WebPDecoder::new(Cursor::new(webp)).expect("decode WebP"),
            (37, 23),
            true,
        );

        let jxl = encode_image_to_bytes(&image, "jxl", 90).expect("encode JPEG XL");
        assert_decoder_contract(
            JxlDecoder::new(Cursor::new(jxl)).expect("decode JPEG XL"),
            (37, 23),
            false,
        );
    }

    #[test]
    fn bounded_webp_output_matches_the_existing_encoder_bytes() {
        let rgb = fixture_image(37, 23);
        let rgba = DynamicImage::ImageRgba8(RgbaImage::from_fn(37, 23, |x, y| {
            Rgba([
                ((x * 17 + y * 3) % 256) as u8,
                ((x * 5 + y * 13) % 256) as u8,
                ((x * 11 + y * 7) % 256) as u8,
                ((x * 19 + y * 23) % 256) as u8,
            ])
        }));
        let directory = tempfile::tempdir().expect("temporary WebP directory");
        let cancellation_token = AtomicBool::new(false);

        for (index, image) in [rgb, rgba].iter().enumerate() {
            for quality in [50, 75, 90, 100] {
                let expected =
                    encode_image_to_bytes(image, "webp", quality).unwrap_or_else(|error| {
                        panic!("encode reference WebP {index} Q{quality}: {error}")
                    });
                let output_path = directory
                    .path()
                    .join(format!("bounded-{index}-q{quality}.webp"));
                save_webp_with_bounded_output(image, &output_path, quality, &cancellation_token)
                    .unwrap_or_else(|error| {
                        panic!("write bounded WebP {index} Q{quality}: {error}")
                    });
                let actual = fs::read(&output_path).expect("read bounded WebP");

                assert_eq!(actual, expected);
                assert_decoder_contract(
                    WebPDecoder::new(Cursor::new(actual)).expect("decode bounded WebP"),
                    image.dimensions(),
                    true,
                );
            }
        }
    }

    #[test]
    fn bounded_webp_icc_rewrite_replaces_existing_profile_and_rejects_bad_input() {
        let image = fixture_image(37, 23);
        let encoded = encode_image_to_bytes(&image, "webp", 90).expect("encode ICC WebP");
        let cancellation_token = AtomicBool::new(false);
        let mut source = Cursor::new(encoded.clone());
        let mut output = Cursor::new(Vec::new());
        let output_len = rewrite_webp_icc_bounded(
            &mut source,
            &mut output,
            srgb_v4_profile(),
            image.width(),
            image.height(),
            false,
            &cancellation_token,
        )
        .expect("rewrite existing WebP ICC");
        assert_eq!(output_len as usize, encoded.len());
        assert_eq!(output.into_inner(), encoded);
        assert_eq!(WEBP_OUTPUT_COPY_BUFFER_BYTES, 65_536);

        let mut truncated = source.into_inner();
        truncated.pop();
        let error = rewrite_webp_icc_bounded(
            &mut Cursor::new(truncated),
            &mut Cursor::new(Vec::new()),
            srgb_v4_profile(),
            image.width(),
            image.height(),
            false,
            &cancellation_token,
        )
        .expect_err("truncated WebP must fail");
        assert!(error.contains("RIFF length"), "unexpected error: {error}");

        let cancelled = AtomicBool::new(true);
        let error = rewrite_webp_icc_bounded(
            &mut Cursor::new(encoded),
            &mut Cursor::new(Vec::new()),
            srgb_v4_profile(),
            image.width(),
            image.height(),
            false,
            &cancelled,
        )
        .expect_err("cancelled WebP rewrite must fail");
        assert_eq!(error, "Export cancelled");
    }

    #[test]
    fn streaming_color_managed_exports_round_trip_dimensions_and_icc() {
        let image = fixture_image(37, 23);
        let expected_rgba = image.to_rgba8();

        let jpeg = encode_streaming_fixture(&image, "jpeg", 92).expect("stream JPEG");
        assert_decoder_contract(
            JpegDecoder::new(Cursor::new(&jpeg)).expect("decode streaming JPEG"),
            (37, 23),
            true,
        );

        let png = encode_streaming_fixture(&image, "png", 92).expect("stream PNG");
        assert_decoder_contract(
            PngDecoder::new(Cursor::new(&png)).expect("decode streaming PNG"),
            (37, 23),
            true,
        );
        let decoded_png = image::load_from_memory_with_format(&png, ImageFormat::Png)
            .expect("load streaming PNG pixels")
            .to_rgba8();
        assert_eq!(decoded_png, expected_rgba);

        let tiff = encode_streaming_fixture(&image, "tiff", 92).expect("stream TIFF");
        assert_decoder_contract(
            TiffDecoder::new(Cursor::new(&tiff)).expect("decode streaming TIFF"),
            (37, 23),
            true,
        );
        let decoded_tiff = image::load_from_memory_with_format(&tiff, ImageFormat::Tiff)
            .expect("load streaming TIFF pixels")
            .to_rgba8();
        assert_eq!(decoded_tiff, expected_rgba);
    }

    #[test]
    fn streaming_jpeg_png_and_tiff_embed_bounded_exif_and_strip_gps() {
        fn read_export_exif(encoded: &[u8]) -> exif::Exif {
            exif::Reader::new()
                .read_from_container(&mut Cursor::new(encoded))
                .expect("read streamed export EXIF")
        }

        fn ascii_tag(exif_data: &exif::Exif, tag: exif::Tag) -> Option<String> {
            let field = exif_data.get_field(tag, exif::In::PRIMARY)?;
            let exif::Value::Ascii(values) = &field.value else {
                return None;
            };
            Some(
                values
                    .iter()
                    .map(|value| {
                        String::from_utf8_lossy(value)
                            .trim_end_matches('\0')
                            .to_string()
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
            )
        }

        let image = fixture_image(37, 23);
        let mut source = encode_image_to_bytes(&image, "jpeg", 92).expect("encode EXIF source");
        let mut source_metadata = Metadata::new();
        source_metadata.set_tag(ExifTag::Make("Bounded Camera".to_string()));
        source_metadata.set_tag(ExifTag::Model("Synthetic 60MP".to_string()));
        source_metadata.set_tag(ExifTag::Artist("Original Artist".to_string()));
        source_metadata.set_tag(ExifTag::ImageDescription(
            "Museum export fixture".to_string(),
        ));
        source_metadata.set_tag(ExifTag::DateTimeOriginal("2026:08:10 12:34:56".to_string()));
        source_metadata.set_tag(ExifTag::GPSLatitudeRef("N".to_string()));
        source_metadata.set_tag(ExifTag::GPSLatitude(vec![
            uR64 {
                nominator: 31,
                denominator: 1,
            },
            uR64 {
                nominator: 14,
                denominator: 1,
            },
            uR64 {
                nominator: 15,
                denominator: 1,
            },
        ]));
        source_metadata
            .write_to_vec(&mut source, FileExtension::JPEG)
            .expect("attach source EXIF");

        let directory = tempfile::tempdir().expect("create metadata test directory");
        let source_path = directory.path().join("source.jpg");
        std::fs::write(&source_path, source).expect("write EXIF source");
        let source_path = source_path.to_string_lossy();

        let retained =
            exif_processing::export_metadata_tiff_payload(&source_path, "jpeg", true, false, None)
                .expect("build retained EXIF")
                .expect("retained EXIF must exist");
        let jpeg = encode_streaming_fixture_with_exif(&image, "jpeg", 92, Some(&retained))
            .expect("stream JPEG with EXIF");
        assert_decoder_contract(
            JpegDecoder::new(Cursor::new(&jpeg)).expect("decode metadata JPEG"),
            (37, 23),
            true,
        );
        let jpeg_exif = read_export_exif(&jpeg);
        assert_eq!(
            ascii_tag(&jpeg_exif, exif::Tag::Make).as_deref(),
            Some("Bounded Camera")
        );
        assert_eq!(
            ascii_tag(&jpeg_exif, exif::Tag::ImageDescription).as_deref(),
            Some("Museum export fixture")
        );
        assert!(
            jpeg_exif
                .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
                .is_some(),
            "JPEG must retain GPS when stripGps is false"
        );

        let cleared_metadata = exif_processing::export_metadata_tiff_payload(
            &source_path,
            "jpeg",
            true,
            false,
            Some(&exif_processing::ExportMetadataOverrides {
                artist: Some(String::new()),
                contact: None,
                copyright: None,
                description: None,
            }),
        )
        .expect("build EXIF with an explicit cleared artist")
        .expect("cleared EXIF must still contain retained metadata");
        let cleared_jpeg =
            encode_streaming_fixture_with_exif(&image, "jpeg", 92, Some(&cleared_metadata))
                .expect("stream JPEG with cleared artist");
        assert!(
            read_export_exif(&cleared_jpeg)
                .get_field(exif::Tag::Artist, exif::In::PRIMARY)
                .is_none(),
            "an empty editable artist field must remove the retained source artist"
        );

        let stripped =
            exif_processing::export_metadata_tiff_payload(&source_path, "png", true, true, None)
                .expect("build stripped EXIF")
                .expect("stripped EXIF must exist");
        let png = encode_streaming_fixture_with_exif(&image, "png", 92, Some(&stripped))
            .expect("stream PNG with EXIF");
        assert_decoder_contract(
            PngDecoder::new(Cursor::new(&png)).expect("decode metadata PNG"),
            (37, 23),
            true,
        );
        let png_exif = read_export_exif(&png);
        assert_eq!(
            ascii_tag(&png_exif, exif::Tag::Make).as_deref(),
            Some("Bounded Camera")
        );
        assert_eq!(
            ascii_tag(&png_exif, exif::Tag::Software).as_deref(),
            Some("RAW Editor")
        );
        assert!(
            png_exif
                .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
                .is_none(),
            "PNG must remove GPS when stripGps is true"
        );

        let retained_tiff_metadata =
            exif_processing::export_metadata_for_streaming_tiff(&source_path, true, false, None)
                .expect("build retained TIFF metadata")
                .expect("retained TIFF metadata must exist");
        let tiff =
            encode_streaming_fixture_with_tiff_metadata(&image, Some(&retained_tiff_metadata))
                .expect("stream TIFF with metadata");
        assert_decoder_contract(
            TiffDecoder::new(Cursor::new(&tiff)).expect("decode metadata TIFF"),
            (37, 23),
            true,
        );
        assert_eq!(
            image::load_from_memory_with_format(&tiff, ImageFormat::Tiff)
                .expect("load metadata TIFF pixels")
                .to_rgba8(),
            image.to_rgba8()
        );
        let tiff_exif = read_export_exif(&tiff);
        assert_eq!(
            ascii_tag(&tiff_exif, exif::Tag::Make).as_deref(),
            Some("Bounded Camera")
        );
        assert_eq!(
            ascii_tag(&tiff_exif, exif::Tag::Software).as_deref(),
            Some("RAW Editor")
        );
        assert_eq!(
            ascii_tag(&tiff_exif, exif::Tag::DateTimeOriginal).as_deref(),
            Some("2026:08:10 12:34:56")
        );
        assert!(
            tiff_exif
                .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
                .is_some(),
            "TIFF must retain GPS when stripGps is false"
        );

        let tiff_source_path = directory.path().join("source.tiff");
        std::fs::write(&tiff_source_path, &tiff).expect("write metadata TIFF source");
        let copied_tiff_metadata = exif_processing::export_metadata_for_streaming_tiff(
            &tiff_source_path.to_string_lossy(),
            true,
            false,
            None,
        )
        .expect("copy TIFF source metadata")
        .expect("copied TIFF metadata must exist");
        let copied_tiff =
            encode_streaming_fixture_with_tiff_metadata(&image, Some(&copied_tiff_metadata))
                .expect("stream TIFF copied from TIFF metadata");
        let copied_tiff_exif = read_export_exif(&copied_tiff);
        assert_eq!(
            ascii_tag(&copied_tiff_exif, exif::Tag::Make).as_deref(),
            Some("Bounded Camera")
        );
        assert!(
            copied_tiff_exif
                .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
                .is_some(),
            "TIFF source metadata must remain readable and copyable"
        );

        let stripped_tiff_metadata =
            exif_processing::export_metadata_for_streaming_tiff(&source_path, true, true, None)
                .expect("build stripped TIFF metadata")
                .expect("stripped TIFF metadata must exist");
        let stripped_tiff =
            encode_streaming_fixture_with_tiff_metadata(&image, Some(&stripped_tiff_metadata))
                .expect("stream TIFF without GPS");
        let stripped_tiff_exif = read_export_exif(&stripped_tiff);
        assert_eq!(
            ascii_tag(&stripped_tiff_exif, exif::Tag::Make).as_deref(),
            Some("Bounded Camera")
        );
        assert!(
            stripped_tiff_exif
                .get_field(exif::Tag::GPSLatitude, exif::In::PRIMARY)
                .is_none(),
            "TIFF must remove GPS when stripGps is true"
        );
    }

    #[test]
    fn streaming_jpeg_preserves_the_existing_quality_scale() {
        let image = fixture_image(257, 173);
        let reference = image.to_rgb8();
        let mut previous_streamed_error = f64::INFINITY;
        let mut previous_streamed_bytes = 0_usize;

        for quality in [50, 75, 92] {
            let legacy = encode_image_to_bytes(&image, "jpeg", quality)
                .unwrap_or_else(|error| panic!("encode legacy JPEG Q{quality}: {error}"));
            let streamed = encode_streaming_fixture(&image, "jpeg", quality)
                .unwrap_or_else(|error| panic!("stream JPEG Q{quality}: {error}"));
            let legacy_decoded = image::load_from_memory_with_format(&legacy, ImageFormat::Jpeg)
                .unwrap_or_else(|error| panic!("decode legacy JPEG Q{quality}: {error}"))
                .to_rgb8();
            let streamed_decoded =
                image::load_from_memory_with_format(&streamed, ImageFormat::Jpeg)
                    .unwrap_or_else(|error| panic!("decode streaming JPEG Q{quality}: {error}"))
                    .to_rgb8();
            let legacy_error = mean_rgb_abs_error(&reference, &legacy_decoded);
            let streamed_error = mean_rgb_abs_error(&reference, &streamed_decoded);
            let size_delta = streamed.len().abs_diff(legacy.len());
            let allowed_size_delta = (legacy.len() / 20).max(512);

            assert!(
                streamed_error <= legacy_error * 1.05 + 0.15,
                "streamed JPEG Q{quality} mean RGB error {streamed_error:.3} drifted from legacy {legacy_error:.3}"
            );
            assert!(
                size_delta <= allowed_size_delta,
                "streamed JPEG Q{quality} size {} drifted from legacy {}",
                streamed.len(),
                legacy.len()
            );
            assert!(
                streamed_error < previous_streamed_error,
                "streamed JPEG error did not improve at Q{quality}"
            );
            assert!(
                streamed.len() > previous_streamed_bytes,
                "streamed JPEG size did not increase at Q{quality}"
            );
            previous_streamed_error = streamed_error;
            previous_streamed_bytes = streamed.len();
        }
    }

    #[test]
    fn streaming_export_requires_exactly_one_complete_frame() {
        let mut file = tempfile::tempfile().expect("temporary output");
        let error = encode_streaming_rgba_rows(
            &mut file,
            4,
            3,
            "png",
            90,
            true,
            StreamingExportMetadata::None,
            |sink| {
                sink(&[0; 4 * 4])?;
                sink(&[0; 4 * 4])?;
                Ok(())
            },
        )
        .expect_err("missing final row must fail");
        assert!(error.contains("expected 3"), "unexpected error: {error}");

        let mut overflow_file = tempfile::tempfile().expect("temporary overflow output");
        let error = encode_streaming_rgba_rows(
            &mut overflow_file,
            4,
            1,
            "png",
            90,
            true,
            StreamingExportMetadata::None,
            |sink| {
                sink(&[0; 4 * 4])?;
                sink(&[0; 4 * 4])?;
                Ok(())
            },
        )
        .expect_err("extra row must fail");
        assert!(
            error.contains("more than the declared 1 rows"),
            "unexpected error: {error}"
        );

        let mut oversized_jpeg = tempfile::tempfile().expect("temporary oversized JPEG");
        let error = encode_streaming_rgba_rows(
            &mut oversized_jpeg,
            u32::from(u16::MAX) + 1,
            1,
            "jpeg",
            90,
            true,
            StreamingExportMetadata::None,
            |_| Ok(()),
        )
        .expect_err("oversized JPEG dimensions must fail before encoding");
        assert!(error.contains("65535"), "unexpected error: {error}");

        let settings = ExportSettings {
            jpeg_quality: 90,
            resize: None,
            keep_metadata: false,
            metadata_overrides: None,
            preserve_timestamps: false,
            strip_gps: false,
            embed_color_profile: true,
            filename_template: None,
            watermark: None,
            export_masks: false,
            preserve_folders: false,
        };
        assert!(supports_streaming_export("jpeg", &settings));
        assert!(supports_streaming_export("png", &settings));
        assert!(supports_streaming_export("tiff", &settings));
        assert!(!supports_streaming_export("webp", &settings));

        let mut resized = settings.clone();
        resized.resize = Some(ResizeOptions {
            mode: ResizeMode::LongEdge,
            value: 2_048,
            dont_enlarge: true,
        });
        assert!(supports_streaming_export("png", &resized));

        let mut watermarked = resized;
        watermarked.watermark = Some(WatermarkSettings {
            path: "unused-routing-fixture.png".to_string(),
            anchor: WatermarkAnchor::BottomRight,
            scale: 10.0,
            spacing: 5.0,
            opacity: 50.0,
        });
        assert!(supports_streaming_export("jpeg", &watermarked));
    }

    #[test]
    fn streaming_resize_matches_the_bounded_batch_reference() {
        let image = fixture_image(37, 23);
        let settings = ExportSettings {
            jpeg_quality: 92,
            resize: Some(ResizeOptions {
                mode: ResizeMode::Width,
                value: 19,
                dont_enlarge: false,
            }),
            keep_metadata: false,
            metadata_overrides: None,
            preserve_timestamps: false,
            strip_gps: false,
            embed_color_profile: true,
            filename_template: None,
            watermark: None,
            export_masks: false,
            preserve_folders: false,
        };
        let streamed = collect_streaming_transform(&image, &settings).expect("stream resize");
        let source = image.to_rgba8();
        let (target_width, target_height) = streamed.dimensions();
        let config =
            ResizeConfig::builder(source.width(), source.height(), target_width, target_height)
                .filter(StreamingResizeFilter::Lanczos)
                .format(PixelDescriptor::RGBA8_SRGB)
                .srgb()
                .build();
        let expected = zenresize::Resizer::new(&config).resize(source.as_raw());

        assert_eq!(streamed.as_raw(), &expected);
        assert_eq!(streamed.dimensions(), (19, 12));

        let legacy = image
            .resize(19, 12, imageops::FilterType::Lanczos3)
            .to_rgba8();
        let mut max_channel_delta = 0_u8;
        let mut total_channel_delta = 0_u64;
        for (streamed, legacy) in streamed.pixels().zip(legacy.pixels()) {
            for channel in 0..3 {
                let delta = streamed[channel].abs_diff(legacy[channel]);
                max_channel_delta = max_channel_delta.max(delta);
                total_channel_delta += u64::from(delta);
            }
        }
        let mean_channel_delta = total_channel_delta as f64 / (19 * 12 * 3) as f64;
        assert!(
            mean_channel_delta <= 0.5,
            "mean channel delta {mean_channel_delta:.3} exceeded the migration bound"
        );
        assert!(
            max_channel_delta <= 2,
            "max channel delta {max_channel_delta} exceeded the migration bound"
        );
    }

    #[test]
    fn imported_watermark_is_normalized_and_deduplicated_in_private_storage() {
        let directory = tempfile::tempdir().expect("watermark import directory");
        let source_path = directory.path().join("My Watermark.png");
        let source = RgbaImage::from_fn(11, 7, |x, y| {
            Rgba([
                (x * 17) as u8,
                (y * 29) as u8,
                ((x + y) * 13) as u8,
                (80 + x * 7 + y * 3) as u8,
            ])
        });
        source.save(&source_path).expect("save source watermark");

        let watermark_directory = directory.path().join("stored-watermarks");
        let first = import_watermark_image_impl(&source_path, &watermark_directory)
            .expect("import watermark");
        let second = import_watermark_image_impl(&source_path, &watermark_directory)
            .expect("deduplicate watermark");

        assert_eq!(first, second);
        assert_eq!(first.parent(), Some(watermark_directory.as_path()));
        assert!(
            first
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("My-Watermark-") && name.ends_with(".png"))
        );
        assert_eq!(
            image::image_dimensions(first).expect("stored dimensions"),
            (11, 7)
        );
    }

    #[test]
    fn bundled_default_watermark_matches_the_reference_asset() {
        let watermark = load_watermark_image(DEFAULT_WATERMARK_PATH)
            .expect("decode the bundled default watermark");
        assert_eq!(watermark.dimensions(), (1908, 1462));

        let prepared = prepare_watermark(
            1000,
            800,
            &WatermarkSettings {
                path: DEFAULT_WATERMARK_PATH.to_string(),
                anchor: WatermarkAnchor::Center,
                scale: 10.0,
                spacing: 5.0,
                opacity: 80.0,
            },
        )
        .expect("prepare the bundled default watermark")
        .expect("default watermark remains visible");

        assert_eq!(prepared.image.width(), 80);
        assert_eq!(prepared.x, 460);
        assert_eq!(prepared.y, (800 - i64::from(prepared.image.height())) / 2);
    }

    #[test]
    fn streaming_watermark_matches_the_existing_full_frame_blend() {
        let image = fixture_image(37, 23);
        let directory = tempfile::tempdir().expect("watermark directory");
        let watermark_path = directory.path().join("watermark.png");
        let watermark = RgbaImage::from_fn(7, 3, |x, y| {
            Rgba([
                (x * 31) as u8,
                (y * 67) as u8,
                ((x + y) * 23) as u8,
                (80 + x * 17 + y * 11) as u8,
            ])
        });
        watermark.save(&watermark_path).expect("save watermark");
        let settings = ExportSettings {
            jpeg_quality: 92,
            resize: None,
            keep_metadata: false,
            metadata_overrides: None,
            preserve_timestamps: false,
            strip_gps: false,
            embed_color_profile: true,
            filename_template: None,
            watermark: Some(WatermarkSettings {
                path: watermark_path.to_string_lossy().into_owned(),
                anchor: WatermarkAnchor::BottomRight,
                scale: 24.0,
                spacing: 7.0,
                opacity: 63.0,
            }),
            export_masks: false,
            preserve_folders: false,
        };

        let streamed = collect_streaming_transform(&image, &settings).expect("stream watermark");
        let expected = apply_export_resize_and_watermark(image, &settings)
            .expect("apply existing watermark")
            .to_rgba8();
        assert_eq!(streamed, expected);
    }

    #[cfg(unix)]
    #[test]
    fn streamed_export_temp_preserves_existing_file_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temporary export directory");
        let output_path = directory.path().join("existing.png");
        fs::write(&output_path, b"old").expect("create existing export");
        fs::set_permissions(&output_path, fs::Permissions::from_mode(0o640))
            .expect("set existing permissions");

        let temporary = create_temporary_export(&output_path).expect("create adjacent temporary");
        preserve_existing_export_permissions(&temporary, &output_path)
            .expect("preserve existing permissions");
        let mode = temporary
            .as_file()
            .metadata()
            .expect("temporary metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o640);
    }

    #[cfg(unix)]
    #[test]
    fn bounded_webp_publish_preserves_permissions_and_cancel_keeps_target() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temporary WebP export directory");
        let output_path = directory.path().join("existing.webp");
        fs::write(&output_path, b"old").expect("create existing WebP export");
        fs::set_permissions(&output_path, fs::Permissions::from_mode(0o640))
            .expect("set existing permissions");

        let image = fixture_image(37, 23);
        save_webp_with_bounded_output(&image, &output_path, 90, &AtomicBool::new(false))
            .expect("publish bounded WebP");
        let published = fs::read(&output_path).expect("read published WebP");
        let mode = fs::metadata(&output_path)
            .expect("published metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(mode, 0o640);

        let error = save_webp_with_bounded_output(&image, &output_path, 90, &AtomicBool::new(true))
            .expect_err("cancelled WebP publish must fail");
        assert_eq!(error, "Export cancelled");
        assert_eq!(fs::read(&output_path).expect("reread WebP"), published);

        let mut callback_output = tempfile::tempfile().expect("callback output");
        let callback_token = AtomicBool::new(true);
        let mut callback_context = WebpFileWriterContext {
            output: &mut callback_output,
            cancellation_token: &callback_token,
            cancelled: false,
            first_error: None,
        };
        let picture = libwebp_sys::WebPPicture::new().expect("initialize callback picture");
        let mut picture = WebpPictureGuard(picture);
        picture.0.user_data = std::ptr::from_mut(&mut callback_context).cast();
        // SAFETY: the initialized picture points to the live callback context.
        let should_continue = unsafe { check_webp_encoding_progress(1, &picture.0) };
        assert_eq!(should_continue, 0);
        assert!(callback_context.cancelled);
    }

    #[test]
    #[ignore = "manual deterministic large-image encoder benchmark"]
    fn synthetic_60mp_webp_output_harness() {
        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let mode =
            std::env::var("RAW_EDITOR_BENCH_WEBP_MODE").unwrap_or_else(|_| "file".to_string());
        assert!(width > 0 && height > 0);

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(0));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let Ok(pid) = sysinfo::get_current_pid() else {
                return;
            };
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });

        let image = DynamicImage::ImageRgba8(RgbaImage::from_fn(width, height, |x, y| {
            Rgba([
                x.wrapping_mul(17).wrapping_add(y * 3) as u8,
                x.wrapping_mul(5).wrapping_add(y * 13) as u8,
                x.wrapping_mul(11).wrapping_add(y * 7) as u8,
                255,
            ])
        }));
        let benchmark_directory = tempfile::tempdir().expect("temporary WebP benchmark directory");
        let output_path = benchmark_directory.path().join("benchmark.webp");
        let started = Instant::now();
        let mut retained_memory = None;
        let output_bytes = match mode.as_str() {
            "memory" => {
                let encoded =
                    encode_image_to_bytes(&image, "webp", 90).expect("encode WebP in memory");
                let output_bytes = encoded.len() as u64;
                retained_memory = Some(encoded);
                output_bytes
            }
            "file" => {
                save_webp_with_bounded_output(&image, &output_path, 90, &AtomicBool::new(false))
                    .expect("encode WebP to bounded files");
                fs::metadata(&output_path)
                    .expect("benchmark WebP metadata")
                    .len()
            }
            _ => panic!("unsupported WebP benchmark mode: {mode}"),
        };
        let elapsed = started.elapsed();
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join RSS sampler");

        println!(
            "{{\"format\":\"webp\",\"mode\":\"{}\",\"width\":{},\"height\":{},\"elapsedMs\":{},\"outputBytes\":{},\"peakRssBytes\":{},\"inputRgbaBytes\":{}}}",
            mode,
            width,
            height,
            elapsed.as_millis(),
            output_bytes,
            peak_rss.load(Ordering::Relaxed),
            u64::from(width) * u64::from(height) * 4,
        );

        std::hint::black_box(retained_memory);
        std::hint::black_box(benchmark_directory);
    }

    #[test]
    #[ignore = "manual deterministic large-image encoder benchmark"]
    fn synthetic_60mp_streaming_export_harness() {
        use crate::render_strategy::StreamingExportBufferPlan;

        let width = std::env::var("RAW_EDITOR_BENCH_WIDTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(9_504_u32);
        let height = std::env::var("RAW_EDITOR_BENCH_HEIGHT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6_336_u32);
        let output_format =
            std::env::var("RAW_EDITOR_BENCH_FORMAT").unwrap_or_else(|_| "png".to_string());
        let resize_long_edge = std::env::var("RAW_EDITOR_BENCH_RESIZE_LONG_EDGE")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .filter(|value| *value > 0);
        let resize_options = resize_long_edge.map(|value| ResizeOptions {
            mode: ResizeMode::LongEdge,
            value,
            dont_enlarge: true,
        });
        let (target_width, target_height) = resize_options
            .as_ref()
            .map(|resize| calculate_resize_target(width, height, resize))
            .unwrap_or((width, height));
        let watermark_enabled = std::env::var("RAW_EDITOR_BENCH_WATERMARK")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes"));
        let metadata_enabled = std::env::var("RAW_EDITOR_BENCH_METADATA")
            .is_ok_and(|value| matches!(value.as_str(), "1" | "true" | "yes"))
            && matches!(
                output_format.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "tif" | "tiff"
            );
        let benchmark_metadata = metadata_enabled.then(|| {
            let mut metadata = Metadata::new();
            metadata.set_tag(ExifTag::Make("Synthetic benchmark".to_string()));
            metadata.set_tag(ExifTag::Model("Bounded EXIF".to_string()));
            metadata.set_tag(ExifTag::Software("RAW Editor".to_string()));
            metadata
        });
        let is_tiff_output = matches!(output_format.to_ascii_lowercase().as_str(), "tif" | "tiff");
        let export_exif = benchmark_metadata
            .as_ref()
            .filter(|_| !is_tiff_output)
            .map(|metadata| {
                let encoded = metadata
                    .as_u8_vec(FileExtension::JPEG)
                    .expect("encode benchmark EXIF");
                assert_eq!(&encoded[..2], b"\xff\xe1");
                assert_eq!(&encoded[4..10], b"Exif\0\0");
                encoded[10..].to_vec()
            });
        let watermark_directory = watermark_enabled
            .then(|| tempfile::tempdir().expect("temporary benchmark watermark directory"));
        let watermark_settings = watermark_directory.as_ref().map(|directory| {
            let path = directory.path().join("benchmark-watermark.png");
            RgbaImage::from_fn(512, 128, |x, y| {
                Rgba([
                    x.wrapping_mul(29).wrapping_add(y * 3) as u8,
                    x.wrapping_mul(7).wrapping_add(y * 17) as u8,
                    x.wrapping_mul(13).wrapping_add(y * 11) as u8,
                    (96 + (x + y) % 160) as u8,
                ])
            })
            .save(&path)
            .expect("write benchmark watermark");
            WatermarkSettings {
                path: path.to_string_lossy().into_owned(),
                anchor: WatermarkAnchor::BottomRight,
                scale: 12.0,
                spacing: 5.0,
                opacity: 75.0,
            }
        });
        assert!(width > 0 && height > 0);

        let running = Arc::new(AtomicBool::new(true));
        let peak_rss = Arc::new(AtomicU64::new(0));
        let sampler_running = Arc::clone(&running);
        let sampler_peak = Arc::clone(&peak_rss);
        let sampler = std::thread::spawn(move || {
            let Ok(pid) = sysinfo::get_current_pid() else {
                return;
            };
            let mut system = sysinfo::System::new();
            while sampler_running.load(Ordering::Relaxed) {
                system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
                if let Some(process) = system.process(pid) {
                    sampler_peak.fetch_max(process.memory(), Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });

        let mut output = tempfile::tempfile().expect("temporary benchmark output");
        let buffer_plan = StreamingExportBufferPlan::new(width, height);
        let row_bytes = width as usize * 4;
        let mut band = vec![0_u8; buffer_plan.band_rgba_bytes()];
        let started = Instant::now();
        encode_streaming_rgba_rows(
            &mut output,
            target_width,
            target_height,
            &output_format,
            92,
            true,
            if let Some(metadata) = benchmark_metadata.as_ref().filter(|_| is_tiff_output) {
                StreamingExportMetadata::TiffDirectories(metadata)
            } else if let Some(payload) = export_exif.as_deref() {
                StreamingExportMetadata::ExifTiffPayload(payload)
            } else {
                StreamingExportMetadata::None
            },
            |encoder_sink| {
                transform_streaming_rgba_rows(
                    width,
                    height,
                    target_width,
                    target_height,
                    watermark_settings.as_ref(),
                    encoder_sink,
                    |source_sink| {
                        for band_y in (0..height).step_by(buffer_plan.band_rows as usize) {
                            let rows = buffer_plan.band_rows.min(height - band_y);
                            let active_band = &mut band[..rows as usize * row_bytes];
                            for (local_y, row) in
                                active_band.chunks_exact_mut(row_bytes).enumerate()
                            {
                                let y = band_y + local_y as u32;
                                for (x, pixel) in row.chunks_exact_mut(4).enumerate() {
                                    let x = x as u32;
                                    pixel[0] = x.wrapping_mul(17).wrapping_add(y * 3) as u8;
                                    pixel[1] = x.wrapping_mul(5).wrapping_add(y * 13) as u8;
                                    pixel[2] = x.wrapping_mul(11).wrapping_add(y * 7) as u8;
                                    pixel[3] = 255;
                                }
                            }
                            for row in active_band.chunks_exact(row_bytes) {
                                source_sink(row)?;
                            }
                        }
                        Ok(())
                    },
                )
            },
        )
        .expect("run synthetic streaming export");
        let elapsed = started.elapsed();
        running.store(false, Ordering::Relaxed);
        sampler.join().expect("join RSS sampler");
        let output_bytes = output.metadata().expect("benchmark output metadata").len();

        println!(
            "{{\"format\":\"{}\",\"width\":{},\"height\":{},\"targetWidth\":{},\"targetHeight\":{},\"watermark\":{},\"metadata\":{},\"elapsedMs\":{},\"outputBytes\":{},\"peakRssBytes\":{},\"producerBandBytes\":{},\"legacyFullFrameBytes\":{}}}",
            output_format,
            width,
            height,
            target_width,
            target_height,
            watermark_enabled,
            metadata_enabled,
            elapsed.as_millis(),
            output_bytes,
            peak_rss.load(Ordering::Relaxed),
            band.len(),
            buffer_plan.legacy_full_rgba_bytes()
        );
    }

    #[test]
    fn no_resize_export_keeps_the_full_input_dimensions() {
        let image = fixture_image(2049, 1367);
        let settings = ExportSettings {
            jpeg_quality: 90,
            resize: None,
            keep_metadata: false,
            metadata_overrides: None,
            preserve_timestamps: false,
            strip_gps: false,
            embed_color_profile: true,
            filename_template: None,
            watermark: None,
            export_masks: false,
            preserve_folders: false,
        };

        let processed =
            apply_export_resize_and_watermark(image, &settings).expect("apply export options");
        assert_eq!(processed.dimensions(), (2049, 1367));

        let bytes = encode_image_to_bytes(&processed, "png", 100).expect("encode full image");
        let decoder = PngDecoder::new(Cursor::new(bytes)).expect("decode full image");
        assert_eq!(decoder.dimensions(), (2049, 1367));
    }

    #[test]
    fn batch_summary_retains_later_results_after_an_item_failure() {
        let summary = summarize_batch_export_results([
            Ok(()),
            Err("second.jpg: simulated encoder failure".to_string()),
            Ok(()),
            Err("fourth.jpg: simulated disk failure".to_string()),
            Ok(()),
        ]);

        assert_eq!(summary.completed, 5);
        assert_eq!(summary.errors.len(), 2);
        assert!(summary.errors[0].contains("second.jpg"));
        assert!(summary.errors[1].contains("fourth.jpg"));
    }
}
