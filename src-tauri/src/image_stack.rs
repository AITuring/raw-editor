use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;
use crate::panorama_stitching::{AlignmentMode, BlendMode, stitch_images_with_options};
use image::codecs::jpeg::JpegEncoder;
use image::codecs::tiff::{TiffDecoder, TiffEncoder};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, ImageDecoder, ImageEncoder, RgbImage};
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use tauri::{AppHandle, Emitter, Manager};
use tempfile::NamedTempFile;
use uuid::Uuid;

// Keep the UI proxy comfortably inside desktop WebView image-decoding budgets.
// The full-resolution result remains in AppState and is used for TIFF export.
const PREVIEW_MAX_LONG_SIDE: u32 = 4_096;
const PREVIEW_MAX_PIXELS: u64 = 12_000_000;
const PREVIEW_JPEG_QUALITY: u8 = 92;
const DETAIL_PREVIEW_MAX_LONG_SIDE: u32 = 8_192;
const DETAIL_PREVIEW_MAX_PIXELS: u64 = 32_000_000;
const DETAIL_PREVIEW_JPEG_QUALITY: u8 = 98;

fn resolve_blend_mode(value: &str) -> BlendMode {
    match value {
        "focus" => BlendMode::FocusStack,
        _ => BlendMode::Panorama,
    }
}

fn bounded_preview_dimensions(
    width: u32,
    height: u32,
    max_long_side: u32,
    max_pixels: u64,
) -> (u32, u32) {
    if width == 0 || height == 0 {
        return (0, 0);
    }
    let long_side_scale = max_long_side as f64 / width.max(height) as f64;
    let pixel_scale = (max_pixels as f64 / (width as u64 * height as u64) as f64).sqrt();
    let scale = 1.0_f64.min(long_side_scale).min(pixel_scale);
    (
        (width as f64 * scale).round().max(1.0) as u32,
        (height as f64 * scale).round().max(1.0) as u32,
    )
}

fn preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    bounded_preview_dimensions(width, height, PREVIEW_MAX_LONG_SIDE, PREVIEW_MAX_PIXELS)
}

fn detail_preview_dimensions(width: u32, height: u32) -> (u32, u32) {
    bounded_preview_dimensions(
        width,
        height,
        DETAIL_PREVIEW_MAX_LONG_SIDE,
        DETAIL_PREVIEW_MAX_PIXELS,
    )
}

/// Image stacking works on display-referred samples after RAW development. Freeze
/// those samples once at 16-bit precision so preview and export cannot diverge by
/// independently interpreting a floating-point TIFF as linear RGB.
pub(crate) fn canonicalize_image_stack_result(image: DynamicImage) -> DynamicImage {
    DynamicImage::ImageRgb16(image.to_rgb16())
}

fn encode_srgb_tiff<W: Write + Seek>(image: &DynamicImage, writer: W) -> Result<(), String> {
    let mut encoder = TiffEncoder::new(writer);
    encoder
        .set_icc_profile(crate::color_management::srgb_v4_profile().to_vec())
        .map_err(|error| format!("Failed to attach the image-stack TIFF color profile: {error}"))?;
    image
        .write_with_encoder(encoder)
        .map_err(|error| format!("Failed to encode image-stack TIFF: {error}"))
}

pub(crate) fn write_srgb_tiff(image: &DynamicImage, output_path: &Path) -> Result<(), String> {
    let dimensions = image.dimensions();
    if dimensions.0 == 0 || dimensions.1 == 0 {
        return Err("The image-stack result is empty and cannot be saved.".to_string());
    }

    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| {
        format!(
            "Failed to create the image-stack output folder '{}': {error}",
            parent.display()
        )
    })?;

    let mut temporary = NamedTempFile::new_in(parent).map_err(|error| {
        format!(
            "Failed to create a temporary image-stack file beside '{}': {error}",
            output_path.display()
        )
    })?;
    {
        let mut writer = BufWriter::new(temporary.as_file_mut());
        encode_srgb_tiff(image, &mut writer)?;
        writer
            .flush()
            .map_err(|error| format!("Failed to finish writing the image-stack TIFF: {error}"))?;
    }
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| format!("Failed to sync the image-stack TIFF to disk: {error}"))?;

    let temporary_size = temporary
        .as_file()
        .metadata()
        .map_err(|error| format!("Failed to inspect the saved image-stack TIFF: {error}"))?
        .len();
    if temporary_size == 0 {
        return Err("The encoded image-stack TIFF is empty.".to_string());
    }

    let mut decoder = TiffDecoder::new(BufReader::new(
        File::open(temporary.path())
            .map_err(|error| format!("Failed to reopen the saved image-stack TIFF: {error}"))?,
    ))
    .map_err(|error| format!("Failed to validate the saved image-stack TIFF: {error}"))?;
    if decoder.dimensions() != dimensions {
        return Err(format!(
            "Saved image-stack dimensions do not match the result (expected {}×{}, found {}×{}).",
            dimensions.0,
            dimensions.1,
            decoder.dimensions().0,
            decoder.dimensions().1
        ));
    }
    let saved_profile = decoder
        .icc_profile()
        .map_err(|error| format!("Failed to validate the image-stack color profile: {error}"))?;
    if saved_profile.as_deref() != Some(crate::color_management::srgb_v4_profile()) {
        return Err("The saved image-stack TIFF is missing its sRGB color profile.".to_string());
    }
    drop(decoder);

    let persisted = temporary.persist(output_path).map_err(|error| {
        format!(
            "Failed to finalize the image-stack TIFF '{}': {}",
            output_path.display(),
            error.error
        )
    })?;
    persisted
        .sync_all()
        .map_err(|error| format!("Failed to finalize the image-stack TIFF on disk: {error}"))?;

    #[cfg(unix)]
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("Failed to sync the image-stack output folder: {error}"))?;

    Ok(())
}

fn resolve_image_stack_output_path(
    first_path: &Path,
    blend_mode: &str,
    output_path_str: Option<&str>,
) -> Result<PathBuf, String> {
    if let Some(requested_path) = output_path_str.filter(|path| !path.trim().is_empty()) {
        let mut output_path = PathBuf::from(requested_path);
        let extension = output_path
            .extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase);
        if !matches!(extension.as_deref(), Some("tif" | "tiff")) {
            output_path.set_extension("tiff");
        }
        return Ok(output_path);
    }

    let parent = first_path
        .parent()
        .ok_or_else(|| "Could not determine the source image folder.".to_string())?;
    let stem = first_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("image");
    let suffix = if blend_mode == "focus" {
        "FocusStack"
    } else {
        "Panorama"
    };
    Ok(parent.join(format!("{stem}_{suffix}.tiff")))
}

struct PreviewFiles {
    detail_height: u32,
    detail_path: String,
    detail_width: u32,
    interaction_height: u32,
    interaction_path: String,
    interaction_width: u32,
}

fn encode_preview_jpeg(image: &RgbImage, path: &Path, quality: u8) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|error| format!("Failed to create image-stack preview: {error}"))?;
    let mut writer = BufWriter::new(file);
    let mut encoder = JpegEncoder::new_with_quality(&mut writer, quality);
    encoder
        .set_icc_profile(crate::color_management::srgb_v4_profile().to_vec())
        .map_err(|error| {
            format!("Failed to attach the image-stack preview color profile: {error}")
        })?;
    encoder
        .encode_image(image)
        .map_err(|error| format!("Failed to encode image-stack preview: {error}"))?;
    drop(encoder);
    writer
        .flush()
        .map_err(|error| format!("Failed to finish image-stack preview: {error}"))
}

fn write_preview_files(
    image: &DynamicImage,
    app_handle: &AppHandle,
) -> Result<PreviewFiles, String> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err("The image result is empty.".to_string());
    }
    let (interaction_width, interaction_height) = preview_dimensions(width, height);
    let (detail_width, detail_height) = detail_preview_dimensions(width, height);
    let full_rgb = image.to_rgb8();
    let detail_rgb = if (detail_width, detail_height) == (width, height) {
        full_rgb
    } else {
        image::imageops::resize(&full_rgb, detail_width, detail_height, FilterType::Lanczos3)
    };

    let preview_dir = app_handle
        .path()
        .app_cache_dir()
        .map_err(|error| format!("Failed to resolve image-stack preview cache: {error}"))?
        .join("image-stack-previews");
    fs::create_dir_all(&preview_dir)
        .map_err(|error| format!("Failed to create image-stack preview cache: {error}"))?;
    let preview_id = Uuid::new_v4();
    let detail_path = preview_dir.join(format!("{preview_id}-detail.jpg"));
    encode_preview_jpeg(&detail_rgb, &detail_path, DETAIL_PREVIEW_JPEG_QUALITY)?;

    let interaction_path =
        if (interaction_width, interaction_height) == (detail_width, detail_height) {
            detail_path.clone()
        } else {
            let interaction_rgb = image::imageops::resize(
                &detail_rgb,
                interaction_width,
                interaction_height,
                FilterType::Lanczos3,
            );
            let path = preview_dir.join(format!("{preview_id}-interaction.jpg"));
            encode_preview_jpeg(&interaction_rgb, &path, PREVIEW_JPEG_QUALITY)?;
            path
        };

    if let Ok(entries) = fs::read_dir(&preview_dir) {
        for entry in entries.flatten() {
            let stale_path = entry.path();
            if stale_path != interaction_path && stale_path != detail_path && stale_path.is_file() {
                let _ = fs::remove_file(stale_path);
            }
        }
    }

    Ok(PreviewFiles {
        detail_height,
        detail_path: detail_path.to_string_lossy().into_owned(),
        detail_width,
        interaction_height,
        interaction_path: interaction_path.to_string_lossy().into_owned(),
        interaction_width,
    })
}

#[tauri::command]
pub async fn process_image_stack(
    paths: Vec<String>,
    blend_mode: String,
    alignment_mode: String,
    request_id: String,
    app_handle: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    if paths.len() < 2 {
        return Err("Please select at least two images.".to_string());
    }
    if paths.len() > 30 {
        return Err("Image stack is limited to 30 source images.".to_string());
    }
    if request_id.trim().is_empty() {
        return Err("Image-stack request ID is missing.".to_string());
    }

    let source_paths: Vec<String> = paths
        .iter()
        .map(|path| parse_virtual_path(path).0.to_string_lossy().into_owned())
        .collect();
    let selected_blend_mode = resolve_blend_mode(&blend_mode);
    let selected_alignment_mode = AlignmentMode::from_wire(&alignment_mode);
    let result_handle = state.image_stack_result.clone();
    let generation_handle = state.image_stack_generation.clone();
    let generation = generation_handle.fetch_add(1, Ordering::SeqCst) + 1;
    *result_handle.lock().unwrap() = None;

    let task = tokio::task::spawn_blocking(move || {
        let result = stitch_images_with_options(
            source_paths,
            app_handle.clone(),
            selected_alignment_mode,
            selected_blend_mode,
            "image-stack-progress",
        );

        match result {
            Ok(image) => {
                if generation_handle.load(Ordering::SeqCst) != generation {
                    return Ok(());
                }

                let image = canonicalize_image_stack_result(image);
                let _ = app_handle.emit("image-stack-progress", "Creating preview…");
                let previews = write_preview_files(&image, &app_handle)?;
                if generation_handle.load(Ordering::SeqCst) != generation {
                    let _ = fs::remove_file(&previews.interaction_path);
                    let _ = fs::remove_file(&previews.detail_path);
                    return Ok(());
                }

                let result_id = Uuid::new_v4().to_string();
                let (source_width, source_height) = image.dimensions();
                {
                    let mut stored_result = result_handle.lock().unwrap();
                    if generation_handle.load(Ordering::SeqCst) != generation {
                        let _ = fs::remove_file(&previews.interaction_path);
                        let _ = fs::remove_file(&previews.detail_path);
                        return Ok(());
                    }
                    *stored_result = Some((result_id.clone(), image));
                }
                let _ = app_handle.emit(
                    "image-stack-complete",
                    serde_json::json!({
                        "previewPath": previews.interaction_path,
                        "previewWidth": previews.interaction_width,
                        "previewHeight": previews.interaction_height,
                        "detailPreviewPath": previews.detail_path,
                        "detailPreviewWidth": previews.detail_width,
                        "detailPreviewHeight": previews.detail_height,
                        "sourceWidth": source_width,
                        "sourceHeight": source_height,
                        "requestId": request_id,
                        "resultId": result_id,
                    }),
                );
                Ok(())
            }
            Err(error) => {
                if generation_handle.load(Ordering::SeqCst) == generation {
                    let _ = app_handle.emit(
                        "image-stack-error",
                        serde_json::json!({
                            "message": error,
                            "requestId": request_id,
                        }),
                    );
                }
                Err(error)
            }
        }
    });

    match task.await {
        Ok(result) => result,
        Err(error) => Err(format!("Image stack task failed: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::io::{BufReader, Cursor};
    use std::path::Path;

    use image::codecs::tiff::TiffDecoder;
    use image::{DynamicImage, ImageDecoder, Rgb, Rgb32FImage};

    use super::{
        DETAIL_PREVIEW_MAX_LONG_SIDE, DETAIL_PREVIEW_MAX_PIXELS, PREVIEW_MAX_LONG_SIDE,
        PREVIEW_MAX_PIXELS, canonicalize_image_stack_result, detail_preview_dimensions,
        encode_srgb_tiff, preview_dimensions, resolve_image_stack_output_path, write_srgb_tiff,
    };

    #[test]
    fn preview_dimensions_keep_images_within_the_quality_budget() {
        assert_eq!(preview_dimensions(4_000, 3_000), (4_000, 3_000));

        let (wide_width, wide_height) = preview_dimensions(20_000, 2_000);
        assert_eq!(wide_width, PREVIEW_MAX_LONG_SIDE);
        assert_eq!(wide_height, 410);

        let (large_width, large_height) = preview_dimensions(8_256, 5_504);
        assert!(large_width <= PREVIEW_MAX_LONG_SIDE);
        assert!(large_width as u64 * large_height as u64 <= PREVIEW_MAX_PIXELS + 10_000);

        let (portrait_width, portrait_height) = preview_dimensions(9_305, 12_618);
        assert!(portrait_height <= PREVIEW_MAX_LONG_SIDE);
        assert!(portrait_width as u64 * portrait_height as u64 <= PREVIEW_MAX_PIXELS + 10_000);

        let (detail_width, detail_height) = detail_preview_dimensions(9_305, 12_618);
        assert!(detail_height <= DETAIL_PREVIEW_MAX_LONG_SIDE);
        assert!(detail_width as u64 * detail_height as u64 <= DETAIL_PREVIEW_MAX_PIXELS + 10_000);
        assert!(detail_width > portrait_width);
        assert!(detail_height > portrait_height);
    }

    #[test]
    fn canonical_stack_result_is_display_encoded_rgb16() {
        let source =
            DynamicImage::ImageRgb32F(Rgb32FImage::from_pixel(2, 1, Rgb([0.25, 0.5, 0.75])));
        let canonical = canonicalize_image_stack_result(source);
        let pixels = canonical.as_rgb16().expect("canonical RGB16 result");

        assert_eq!(pixels.dimensions(), (2, 1));
        assert!((pixels.get_pixel(0, 0)[0] as i32 - 16_384).abs() <= 1);
        assert!((pixels.get_pixel(0, 0)[1] as i32 - 32_768).abs() <= 1);
        assert!((pixels.get_pixel(0, 0)[2] as i32 - 49_151).abs() <= 1);
    }

    #[test]
    fn exported_tiff_preserves_canonical_pixels_and_srgb_profile() {
        let source = DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(3, 2, |x, y| {
            Rgb([
                (x as f32 + 1.0) / 4.0,
                (y as f32 + 1.0) / 3.0,
                (x as f32 + y as f32 + 1.0) / 6.0,
            ])
        }));
        let canonical = canonicalize_image_stack_result(source);
        let expected = canonical.to_rgb16();
        let mut encoded = Cursor::new(Vec::new());
        encode_srgb_tiff(&canonical, &mut encoded).expect("encode canonical TIFF");

        encoded.set_position(0);
        let mut metadata_decoder = TiffDecoder::new(&mut encoded).expect("decode TIFF metadata");
        let profile = metadata_decoder
            .icc_profile()
            .expect("read TIFF ICC")
            .expect("embedded TIFF ICC");
        assert_eq!(profile, crate::color_management::srgb_v4_profile());

        encoded.set_position(0);
        let decoded = DynamicImage::from_decoder(
            TiffDecoder::new(&mut encoded).expect("decode canonical TIFF"),
        )
        .expect("read canonical TIFF pixels")
        .to_rgb16();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn atomic_tiff_save_replaces_stale_output_with_a_valid_image() {
        let directory = tempfile::tempdir().expect("temporary image-stack output directory");
        let output_path = directory.path().join("stack-result.tiff");
        fs::write(&output_path, b"stale").expect("seed stale output");
        let source = DynamicImage::ImageRgb32F(Rgb32FImage::from_fn(4, 3, |x, y| {
            Rgb([
                (x as f32 + 1.0) / 5.0,
                (y as f32 + 1.0) / 4.0,
                (x as f32 + y as f32 + 1.0) / 8.0,
            ])
        }));
        let canonical = canonicalize_image_stack_result(source);

        write_srgb_tiff(&canonical, &output_path).expect("atomic image-stack TIFF save");
        assert!(
            fs::metadata(&output_path)
                .expect("saved TIFF metadata")
                .len()
                > 5
        );

        let mut decoder = TiffDecoder::new(BufReader::new(
            File::open(&output_path).expect("open persisted image-stack TIFF"),
        ))
        .expect("decode persisted image-stack TIFF");
        assert_eq!(decoder.dimensions(), (4, 3));
        assert_eq!(
            decoder
                .icc_profile()
                .expect("read persisted TIFF ICC")
                .as_deref(),
            Some(crate::color_management::srgb_v4_profile())
        );
    }

    #[test]
    fn save_path_prefers_the_requested_destination_and_normalizes_the_extension() {
        let source_path = Path::new("/photos/source.jpg");
        assert_eq!(
            resolve_image_stack_output_path(source_path, "focus", Some("/exports/custom-stack"))
                .expect("explicit save destination"),
            Path::new("/exports/custom-stack.tiff")
        );
        assert_eq!(
            resolve_image_stack_output_path(source_path, "focus", None)
                .expect("fallback save destination"),
            Path::new("/photos/source_FocusStack.tiff")
        );
    }
}

#[tauri::command]
pub async fn save_image_stack(
    first_path_str: String,
    blend_mode: String,
    result_id: String,
    output_path_str: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let (first_path, _) = parse_virtual_path(&first_path_str);
    let output_path =
        resolve_image_stack_output_path(&first_path, &blend_mode, output_path_str.as_deref())?;
    let output_path_for_task = output_path.clone();
    let sidecar_source = first_path.to_string_lossy().into_owned();
    let result_handle = state.image_stack_result.clone();

    tokio::task::spawn_blocking(move || {
        let mut result = result_handle
            .lock()
            .map_err(|_| "The image-stack result store is unavailable.".to_string())?;
        let (stored_result_id, image) = result
            .as_ref()
            .ok_or_else(|| "No image-stack result is available to save.".to_string())?;
        if stored_result_id != &result_id {
            return Err(
                "The visible image-stack preview is no longer the current result. Please realign before saving."
                    .to_string(),
            );
        }

        write_srgb_tiff(image, &output_path_for_task)?;
        *result = None;
        drop(result);
        let _ = crate::exif_processing::write_rrexif_sidecar(
            &sidecar_source,
            &output_path_for_task,
        );
        Ok(output_path_for_task.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| format!("Image-stack save task failed: {error}"))?
}
