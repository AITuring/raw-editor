//! Batch geometry review previews.
//!
//! Prepares each source once and renders before/after previews with optional
//! guides for the review modal.

use crate::apply_all_transformations;
use crate::exif_processing;
use crate::file_management::parse_virtual_path;
use crate::image_processing::downscale_f32_image;
use base64::Engine as _;
use base64::engine::general_purpose;
use image::DynamicImage;
use image::codecs::jpeg::JpegEncoder;
use imageproc::drawing::draw_line_segment_mut;
use serde_json::Value;
use std::borrow::Cow;
use std::fs;
use std::sync::Arc;

use super::{
    BatchGeometryPreviewItem, CONTENT_ORIENTATION_ANALYSIS_EDGE, ContentOrientationDecision,
    analyze_batch_content_orientation, build_batch_geometry_adjustments,
    find_batch_distortion_params, has_usable_distortion_params,
    load_batch_content_orientation_preview,
};

const BATCH_GEOMETRY_PREVIEW_EDGE: u32 = 1600;

const BATCH_GEOMETRY_PREVIEW_JPEG_QUALITY: u8 = 90;

#[derive(Debug)]
pub(super) struct BatchGeometryPreviewImages {
    pub(super) before_preview: String,
    pub(super) after_preview: String,
}

fn encode_batch_geometry_preview(image: &DynamicImage) -> Result<String, String> {
    let mut bytes = Vec::new();
    JpegEncoder::new_with_quality(&mut bytes, BATCH_GEOMETRY_PREVIEW_JPEG_QUALITY)
        .encode_image(&image.to_rgb8())
        .map_err(|error| format!("Failed to encode batch correction preview: {error}"))?;

    Ok(format!(
        "data:image/jpeg;base64,{}",
        general_purpose::STANDARD.encode(bytes)
    ))
}

/// Draws the guide before applying the transform so the corrected preview
/// shows the same lines after distortion/orientation processing. This makes a
/// subtle lens profile visible without changing the exported pixels.
fn overlay_batch_geometry_grid(image: &DynamicImage) -> DynamicImage {
    let mut visualization = image.to_rgba8();
    let width = visualization.width().saturating_sub(1) as f32;
    let height = visualization.height().saturating_sub(1) as f32;
    let minor_color = image::Rgba([255, 255, 255, 92]);
    let major_color = image::Rgba([92, 224, 255, 176]);

    for index in 1..10 {
        let fraction = index as f32 / 10.0;
        let color = if index == 5 { major_color } else { minor_color };
        let x = width * fraction;
        let y = height * fraction;
        draw_line_segment_mut(&mut visualization, (x, 0.0), (x, height), color);
        draw_line_segment_mut(&mut visualization, (0.0, y), (width, y), color);
    }

    DynamicImage::ImageRgba8(visualization)
}

fn render_batch_geometry_preview_images(
    preview: &DynamicImage,
    adjustments: &Value,
    show_guides: bool,
) -> Result<BatchGeometryPreviewImages, String> {
    let preview_source = if show_guides {
        overlay_batch_geometry_grid(preview)
    } else {
        preview.clone()
    };
    let before_preview = encode_batch_geometry_preview(&preview_source)?;
    let after_preview = apply_all_transformations(Cow::Borrowed(&preview_source), adjustments)
        .0
        .into_owned();

    Ok(BatchGeometryPreviewImages {
        before_preview,
        after_preview: encode_batch_geometry_preview(&after_preview)?,
    })
}

pub(super) struct PreparedBatchGeometrySource {
    pub(super) content_orientation: ContentOrientationDecision,
    pub(super) distortion_params: Option<Value>,
    pub(super) orientation_steps: u8,
    pub(super) preview_images: Option<BatchGeometryPreviewImages>,
    pub(super) source_path: String,
}

/// Does the per-source disk IO and CPU work away from the Tauri async executor.
/// The outer command awaits one source at a time because the shared CLIP session
/// serializes inference anyway; this bounds decoded-image and RAW-byte memory.
pub(super) fn prepare_batch_geometry_source(
    image_path: String,
    apply_distortion: bool,
    preview_settings: Option<crate::app_settings::AppSettings>,
    clip_models: Option<Arc<crate::ai_processing::ClipModels>>,
    lens_database: Option<Arc<crate::lens_correction::LensDatabase>>,
    orientation_override: Option<u8>,
    include_preview: bool,
    show_guides: bool,
) -> Result<PreparedBatchGeometrySource, String> {
    let (source_path, _) = parse_virtual_path(&image_path);
    if !source_path.is_file() {
        return Err(format!(
            "Selected source image does not exist: {}",
            source_path.display()
        ));
    }

    let source_path_string = source_path.to_string_lossy().into_owned();
    let file_bytes = fs::read(&source_path)
        .map_err(|error| format!("Failed to read '{}': {error}", source_path.display()))?;
    let metadata = exif_processing::read_exif_data_from_bytes(&source_path_string, &file_bytes);
    let needs_source_preview = include_preview || clip_models.is_some();
    let source_preview = if needs_source_preview {
        let settings = preview_settings.as_ref().ok_or_else(|| {
            "Batch correction preview settings were unavailable while loading the source."
                .to_string()
        })?;
        Some(load_batch_content_orientation_preview(
            &source_path,
            &file_bytes,
            settings,
            BATCH_GEOMETRY_PREVIEW_EDGE,
        )?)
    } else {
        None
    };
    let content_orientation = match (source_preview.as_ref(), clip_models.as_ref()) {
        (Some(preview), Some(clip_models)) => {
            let analysis_preview = downscale_f32_image(
                preview,
                CONTENT_ORIENTATION_ANALYSIS_EDGE,
                CONTENT_ORIENTATION_ANALYSIS_EDGE,
            );
            analyze_batch_content_orientation(&analysis_preview, &source_path, clip_models)?
        }
        _ => ContentOrientationDecision::preserve_current(),
    };
    let distortion_params = apply_distortion
        .then(|| find_batch_distortion_params(lens_database.as_deref(), &metadata))
        .flatten();
    let orientation_steps = orientation_override
        .unwrap_or(content_orientation.orientation_steps)
        .min(3);
    let preview_images = if include_preview {
        let preview = source_preview
            .as_ref()
            .ok_or_else(|| "Batch correction could not prepare a source preview.".to_string())?;
        let adjustments = build_batch_geometry_adjustments(
            orientation_steps,
            distortion_params.clone(),
            apply_distortion,
        );
        Some(render_batch_geometry_preview_images(
            preview,
            &adjustments,
            show_guides,
        )?)
    } else {
        None
    };

    Ok(PreparedBatchGeometrySource {
        content_orientation,
        distortion_params,
        orientation_steps,
        preview_images,
        source_path: source_path_string,
    })
}

pub(super) fn batch_geometry_preview_item_from_prepared(
    prepared: PreparedBatchGeometrySource,
) -> Result<BatchGeometryPreviewItem, String> {
    let preview_images = prepared.preview_images.ok_or_else(|| {
        "Batch correction did not generate a visual preview for this source.".to_string()
    })?;
    let distortion_applied = prepared
        .distortion_params
        .as_ref()
        .is_some_and(has_usable_distortion_params);

    Ok(BatchGeometryPreviewItem {
        source_path: prepared.source_path,
        before_preview: preview_images.before_preview,
        after_preview: preview_images.after_preview,
        suggested_orientation_steps: prepared.content_orientation.orientation_steps,
        orientation_steps: prepared.orientation_steps,
        auto_corrected: prepared.content_orientation.corrected,
        content_confidence: prepared.content_orientation.confidence,
        distortion_applied,
    })
}
