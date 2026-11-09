use crate::app_state::AppState;
use crate::export_processing::ExportSettings;
use crate::file_management::parse_virtual_path;
use crate::panorama_stitching::{AlignmentMode, BlendMode, stitch_images_with_options};
use crate::panorama_utils::stack_pipeline::degradation::{DegradationManager, OutcomeDecision};
use image::GenericImageView;
#[cfg(test)]
use image::RgbImage;
#[cfg(test)]
use image::codecs::jpeg::JpegEncoder;
#[cfg(test)]
use image::imageops::FilterType;
use std::fs;
use std::sync::atomic::Ordering;
use tauri::{AppHandle, Emitter};
use uuid::Uuid;

mod output_format;

pub use output_format::OutputDowngradeNotice;
use output_format::{ImageStackOutputFormat, OutputFidelity};

mod preview_files;

use preview_files::write_preview_files;
#[cfg(test)]
use preview_files::{
    DETAIL_PREVIEW_JPEG_QUALITY, DETAIL_PREVIEW_MAX_LONG_SIDE, DETAIL_PREVIEW_MAX_PIXELS,
    PREVIEW_JPEG_QUALITY, PREVIEW_MAX_LONG_SIDE, PREVIEW_MAX_PIXELS, derive_preview_images,
    detail_preview_dimensions, preview_dimensions, preview_file_names,
};

mod output_encoding;

pub(crate) use output_encoding::canonicalize_image_stack_result_with_coverage;
use output_encoding::encode_image_stack_file;
#[cfg(test)]
use output_encoding::encode_srgb_image_stack;

mod output_validation;

#[cfg(test)]
use output_validation::SavedOutputMetadata;
use output_validation::{read_image_stack_output_metadata, validate_image_stack_output};

mod output_writer;

use output_writer::{
    ImageStackReportContext, resolve_image_stack_output_path, save_image_stack_output_for_result,
};
#[cfg(test)]
use output_writer::{write_image_stack_output_for_run, write_image_stack_output_with_settings};

#[cfg(test)]
mod test_output;

#[cfg(test)]
pub(crate) use test_output::{canonicalize_image_stack_result, write_srgb_jpeg, write_srgb_tiff};
#[cfg(test)]
use test_output::{
    default_image_stack_export_settings, encode_srgb_tiff, write_image_stack_output,
};

// Large artwork scans commonly contain several focus distances at every camera
// position. The stitching backend keeps preparation and candidate matching
// bounded, so reject only genuinely exceptional selections rather than forcing
// a 382-frame scan to be split into independently warped mosaics.
const IMAGE_STACK_MAX_SOURCES: usize = 500;
const IMAGE_STACK_PIPELINE_VERSION: &str = "image-stack-2026.09.16.1";
// Identifies the layered Stack_Pipeline itself rather than the frontend/backend
// handshake.  It is one of the three ingredients of the Virtual_Tile cache key
// (source paths, source SHA-256 set, pipeline version), so *any* change to a
// grouping, registration, fusion, geometry, tone or composition stage must bump
// it or a stale cache entry will be replayed.
pub(crate) const STACK_PIPELINE_VERSION: &str = "stack-2026.10.28.1";

fn validate_image_stack_source_count(count: usize) -> Result<(), String> {
    if count < 2 {
        return Err("Please select at least two images.".to_string());
    }
    if count > IMAGE_STACK_MAX_SOURCES {
        return Err(format!(
            "Image stack is limited to {IMAGE_STACK_MAX_SOURCES} source images."
        ));
    }
    Ok(())
}

fn validate_image_stack_pipeline_version(value: &str) -> Result<(), String> {
    if value == IMAGE_STACK_PIPELINE_VERSION {
        return Ok(());
    }

    Err(format!(
        "The image-stack backend is out of date (expected {IMAGE_STACK_PIPELINE_VERSION}, received {}). Fully quit and restart RAW Editor before stacking again.",
        if value.trim().is_empty() {
            "no version"
        } else {
            value
        }
    ))
}

fn resolve_blend_mode(value: &str) -> BlendMode {
    match value {
        "focus" => BlendMode::FocusStack,
        _ => BlendMode::Panorama,
    }
}

#[tauri::command]
pub async fn process_image_stack(
    paths: Vec<String>,
    blend_mode: String,
    alignment_mode: String,
    pipeline_version: String,
    request_id: String,
    app_handle: AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    validate_image_stack_source_count(paths.len())?;
    if request_id.trim().is_empty() {
        return Err("Image-stack request ID is missing.".to_string());
    }
    validate_image_stack_pipeline_version(&pipeline_version)?;

    let source_paths: Vec<String> = paths
        .iter()
        .map(|path| parse_virtual_path(path).0.to_string_lossy().into_owned())
        .collect();
    let selected_blend_mode = resolve_blend_mode(&blend_mode);
    let selected_alignment_mode = AlignmentMode::from_wire(&alignment_mode);
    let result_handle = state.image_stack_result.clone();
    let generation_handle = state.image_stack_generation.clone();
    let generation = generation_handle.fetch_add(1, Ordering::SeqCst) + 1;
    crate::panorama_utils::stack_pipeline::degradation::set_active_run_generation(generation);
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
            Ok(outcome) => {
                if generation_handle.load(Ordering::SeqCst) != generation {
                    return Ok(());
                }

                let full_canvas_width = outcome.full_canvas_width;
                let full_canvas_height = outcome.full_canvas_height;
                let render_scale = outcome.render_scale;
                // The Coverage_Mask becomes alpha; the colour channels are the
                // opaque canonical ones either way (需求 10.6).
                let image = canonicalize_image_stack_result_with_coverage(
                    outcome.image,
                    outcome.coverage.as_ref(),
                );
                let _ = app_handle.emit("image-stack-progress", "Creating preview…");
                // One identifier names the stored result and its previews (需求 10.7).
                let result_id = Uuid::new_v4().to_string();
                if let Some(report_path) = outcome.stack_report_path.as_ref()
                    && let Err(error) =
                        crate::panorama_utils::stack_pipeline::report::assign_stack_report_result_id(
                            report_path,
                            &result_id,
                        )
                {
                    eprintln!(
                        "Image-stack result {result_id} could not be linked to its Stack_Report: {error}"
                    );
                }
                let previews = write_preview_files(&image, &result_id, &app_handle)?;
                if generation_handle.load(Ordering::SeqCst) != generation {
                    let _ = fs::remove_file(&previews.interaction_path);
                    let _ = fs::remove_file(&previews.detail_path);
                    return Ok(());
                }

                // This is the one authoritative run-outcome decision. The
                // report has already folded the same ledger on the stitching
                // terminal path; carrying the snapshot with the image prevents
                // a later save from consulting another run's global ledger.
                let degradation_ledger =
                    crate::panorama_utils::stack_pipeline::degradation::run_ledger_snapshot();
                let soft_regions = soft_region_notice(&degradation_ledger);
                let skipped_sources = skipped_sources_notice(&degradation_ledger);
                let outcome_decision = DegradationManager::new(&degradation_ledger).decision();
                if !outcome_decision.permits_final_output() {
                    // The composed pixels remain available only as diagnostic
                    // previews. They are intentionally not inserted into the
                    // exportable result store, and no cleanup targets either
                    // preview or the Stack_Report.
                    let message = match outcome_decision {
                        OutcomeDecision::Cancelled => {
                            "Image-stack processing was cancelled; diagnostic output was retained."
                                .to_string()
                        }
                        OutcomeDecision::Rejected => {
                            let detail = degradation_ledger
                                .quality_gate_rejection_detail()
                                .map(|detail| format!(" Quality_Gate: {detail}."))
                                .unwrap_or_default();
                            format!(
                                "The image-stack result was rejected; diagnostic output was retained.{detail}"
                            )
                        }
                        OutcomeDecision::Success | OutcomeDecision::Degraded => unreachable!(),
                    };
                    let _ = app_handle.emit(
                        "image-stack-error",
                        serde_json::json!({
                            "message": message,
                            "requestId": request_id,
                            "result": outcome_decision.as_identifier(),
                            "previewPath": previews.interaction_path,
                            "detailPreviewPath": previews.detail_path,
                        }),
                    );
                    return Err(message);
                }

                let (source_width, source_height) = image.dimensions();
                {
                    let mut stored_result = result_handle.lock().unwrap();
                    if generation_handle.load(Ordering::SeqCst) != generation {
                        let _ = fs::remove_file(&previews.interaction_path);
                        let _ = fs::remove_file(&previews.detail_path);
                        return Ok(());
                    }
                    *stored_result = Some((
                        result_id.clone(),
                        image,
                        degradation_ledger,
                        outcome.stack_report_path.clone(),
                    ));
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
                        "fullCanvasWidth": full_canvas_width,
                        "fullCanvasHeight": full_canvas_height,
                        "renderScale": render_scale,
                        "orderedPaths": outcome.ordered_paths,
                        "pipelineVersion": IMAGE_STACK_PIPELINE_VERSION,
                        "requestId": request_id,
                        "resultId": result_id,
                        "result": outcome_decision.as_identifier(),
                        "softRegions": soft_regions,
                        "skippedSources": skipped_sources,
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
mod tests;

/// Paths of the sources a stack went on without: rendered images among the
/// RAWs, and frames of a scan that aligned with none of the others or were shot
/// far more obliquely than the rest.
fn skipped_sources_notice(
    ledger: &crate::panorama_utils::stack_pipeline::degradation::DegradationLedger,
) -> serde_json::Value {
    use crate::panorama_utils::stack_pipeline::degradation::{
        SCAN_OBLIQUE_SOURCES_SKIPPED, SCAN_UNALIGNED_SOURCES_SKIPPED,
        STACK_RENDERED_SOURCES_SKIPPED,
    };
    let paths = ledger
        .entries()
        .iter()
        .filter(|entry| {
            entry.reason == SCAN_UNALIGNED_SOURCES_SKIPPED
                || entry.reason == SCAN_OBLIQUE_SOURCES_SKIPPED
                || entry.reason == STACK_RENDERED_SOURCES_SKIPPED
        })
        .filter_map(|entry| {
            entry
                .detail
                .get("paths")
                .and_then(serde_json::Value::as_array)
        })
        .flatten()
        .cloned()
        .collect::<Vec<_>>();
    serde_json::Value::Array(paths)
}

/// The soft regions a focus scan reported for this run (`scan_soft_region`),
/// for the result view: their boxes in result pixels, how many lack a sharp
/// source, and the overview they are circled on. `Null` when there are none.
fn soft_region_notice(
    ledger: &crate::panorama_utils::stack_pipeline::degradation::DegradationLedger,
) -> serde_json::Value {
    use crate::panorama_utils::stack_pipeline::degradation::SCAN_SOFT_REGION;
    let Some(entry) = ledger
        .entries()
        .iter()
        .rev()
        .find(|entry| entry.reason == SCAN_SOFT_REGION)
    else {
        return serde_json::Value::Null;
    };
    let detail = &entry.detail;
    let regions = detail
        .get("regions")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    serde_json::json!({
        "count": regions.len(),
        "sourcesSoft": detail.get("sources_soft").cloned().unwrap_or(serde_json::Value::Null),
        "outputSize": detail.get("output_size").cloned().unwrap_or(serde_json::Value::Null),
        "overviewPath": detail.get("overview").cloned().unwrap_or(serde_json::Value::Null),
        "regions": regions,
    })
}

/// 需求 10.10: the notices of saving the current result in `output_format`
/// with `bit_depth` (fewer than 16 bits per channel, or transparency the
/// format cannot keep), returned before anything is written so the caller can
/// show them ahead of `save_image_stack`.
#[tauri::command]
pub async fn image_stack_output_notices(
    output_format: String,
    bit_depth: u8,
    result_id: String,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<OutputDowngradeNotice>, String> {
    let output_format = ImageStackOutputFormat::from_wire(&output_format)?;
    let result = state
        .image_stack_result
        .lock()
        .map_err(|_| "The image-stack result store is unavailable.".to_string())?;
    let (stored_result_id, image, _, _) = result
        .as_ref()
        .ok_or_else(|| "No image-stack result is available to save.".to_string())?;
    if stored_result_id != &result_id {
        return Err(
            "The visible image-stack preview is no longer the current result. Please realign before saving."
                .to_string(),
        );
    }
    Ok(OutputFidelity::of(image, output_format, bit_depth).downgrades(output_format))
}

#[tauri::command]
pub async fn save_image_stack(
    first_path_str: String,
    blend_mode: String,
    output_format: String,
    export_settings: ExportSettings,
    result_id: String,
    output_path_str: Option<String>,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let (first_path, _) = parse_virtual_path(&first_path_str);
    let output_format = ImageStackOutputFormat::from_wire(&output_format)?;
    let output_path = resolve_image_stack_output_path(
        &first_path,
        &blend_mode,
        output_path_str.as_deref(),
        output_format,
    )?;
    let output_path_for_task = output_path.clone();
    let sidecar_source = first_path.to_string_lossy().into_owned();
    let result_handle = state.image_stack_result.clone();

    tokio::task::spawn_blocking(move || {
        let result = result_handle
            .lock()
            .map_err(|_| "The image-stack result store is unavailable.".to_string())?;
        let (stored_result_id, image, degradation_ledger, stack_report_path) = result
            .as_ref()
            .ok_or_else(|| "No image-stack result is available to save.".to_string())?;
        if stored_result_id != &result_id {
            return Err(
                "The visible image-stack preview is no longer the current result. Please realign before saving."
                    .to_string(),
            );
        }

        save_image_stack_output_for_result(
            image,
            &output_path_for_task,
            output_format,
            &export_settings,
            ImageStackReportContext {
                source_path: &sidecar_source,
                degradation_ledger,
                result_id: &result_id,
                stack_report_path: stack_report_path.as_deref(),
            },
        )?;
        drop(result);
        // Keep the canonical result cached so the same stack can be exported again in
        // another format or to another destination without running the expensive alignment
        // pass a second time. The encoded file is the export contract. Copying the source
        // sidecar here would silently restore EXIF/GPS fields that the user explicitly
        // removed in the dialog.
        Ok(output_path_for_task.to_string_lossy().into_owned())
    })
    .await
    .map_err(|error| format!("Image-stack save task failed: {error}"))?
}
