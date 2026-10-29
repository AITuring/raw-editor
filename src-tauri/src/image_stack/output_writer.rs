//! Publishing image-stack outputs.
//!
//! Writes the result beside the destination, validates it, publishes it through
//! the degradation ledger, records the stack report and resolves the default
//! output path.

use crate::export_processing::ExportSettings;
use crate::panorama_utils::stack_pipeline::degradation::DegradationLedger;
use crate::panorama_utils::stack_pipeline::degradation::DegradationManager;
use crate::panorama_utils::stack_pipeline::degradation::OutputPublication;
use crate::panorama_utils::stack_pipeline::degradation::StandardFinalOutputFileSystem;
use image::DynamicImage;
use image::GenericImageView;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use tempfile::NamedTempFile;

use super::{
    ImageStackOutputFormat, encode_image_stack_file, read_image_stack_output_metadata,
    validate_image_stack_output,
};

pub(super) fn write_image_stack_output_for_run(
    image: &DynamicImage,
    output_path: &Path,
    output_format: ImageStackOutputFormat,
    export_settings: &ExportSettings,
    source_path: &str,
    degradation_ledger: &DegradationLedger,
) -> Result<(), String> {
    let transformed_image =
        if export_settings.resize.is_some() || export_settings.watermark.is_some() {
            Some(crate::export_processing::apply_export_resize_and_watermark(
                image.clone(),
                export_settings,
            )?)
        } else {
            None
        };
    let image = transformed_image.as_ref().unwrap_or(image);
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
    encode_image_stack_file(
        image,
        temporary.as_file_mut(),
        output_format,
        export_settings,
        source_path,
    )?;
    temporary.as_file().sync_all().map_err(|error| {
        format!(
            "Failed to sync the image-stack {} to disk: {error}",
            output_format.label()
        )
    })?;

    let temporary_size = temporary
        .as_file()
        .metadata()
        .map_err(|error| {
            format!(
                "Failed to inspect the saved image-stack {}: {error}",
                output_format.label()
            )
        })?
        .len();
    if temporary_size == 0 {
        return Err(format!(
            "The encoded image-stack {} is empty.",
            output_format.label()
        ));
    }

    validate_image_stack_output(
        temporary.path(),
        dimensions,
        output_format,
        crate::export_processing::effective_export_bit_depth(
            output_format.canonical_extension(),
            export_settings.bit_depth,
        ),
        image.color().has_alpha(),
        export_settings.embed_color_profile,
    )?;

    let temporary_path = temporary.into_temp_path();
    let mut file_system = StandardFinalOutputFileSystem;
    match DegradationManager::new(degradation_ledger).publish_staged_output(
        &mut file_system,
        temporary_path.as_ref(),
        output_path,
    )? {
        OutputPublication::Published(_) => Ok(()),
        OutputPublication::Rejected => {
            let detail = degradation_ledger
                .quality_gate_rejection_detail()
                .map(|detail| format!(" Quality_Gate: {detail}."))
                .unwrap_or_default();
            Err(format!(
                "The image-stack result was rejected; the diagnostic preview and Stack_Report were retained.{detail}"
            ))
        }
        OutputPublication::Cancelled => Err(
            "The image-stack export was cancelled; the diagnostic preview and Stack_Report were retained."
                .to_string(),
        ),
    }
}

/// The save command and export/report integration tests share this boundary.
/// A failed image write returns before touching the report. Once publication
/// succeeds, report failures are diagnostic and do not undo the saved image.
pub(super) struct ImageStackReportContext<'a> {
    pub(super) source_path: &'a str,
    pub(super) degradation_ledger: &'a DegradationLedger,
    pub(super) result_id: &'a str,
    pub(super) stack_report_path: Option<&'a Path>,
}

pub(super) fn save_image_stack_output_for_result(
    image: &DynamicImage,
    output_path: &Path,
    output_format: ImageStackOutputFormat,
    export_settings: &ExportSettings,
    context: ImageStackReportContext<'_>,
) -> Result<(), String> {
    write_image_stack_output_for_run(
        image,
        output_path,
        output_format,
        export_settings,
        context.source_path,
        context.degradation_ledger,
    )?;
    if let Some(report_path) = context.stack_report_path {
        match read_image_stack_output_metadata(output_path, output_format) {
            Ok(metadata) => {
                if let Err(error) =
                    crate::panorama_utils::stack_pipeline::report::update_stack_report_output(
                        report_path,
                        context.result_id,
                        metadata.format,
                        metadata.bit_depth,
                        metadata.alpha_preserved,
                        metadata.icc,
                        output_path,
                    )
                {
                    eprintln!(
                        "Image-stack result {} was saved, but its Stack_Report could not be updated: {error}",
                        context.result_id
                    );
                }
            }
            Err(error) => eprintln!(
                "Image-stack result {} was saved, but its output metadata could not be read back: {error}",
                context.result_id
            ),
        }
    }
    Ok(())
}

pub(super) fn write_image_stack_output_with_settings(
    image: &DynamicImage,
    output_path: &Path,
    output_format: ImageStackOutputFormat,
    export_settings: &ExportSettings,
    source_path: &str,
) -> Result<(), String> {
    write_image_stack_output_for_run(
        image,
        output_path,
        output_format,
        export_settings,
        source_path,
        &DegradationLedger::new(),
    )
}

pub(super) fn resolve_image_stack_output_path(
    first_path: &Path,
    blend_mode: &str,
    output_path_str: Option<&str>,
    output_format: ImageStackOutputFormat,
) -> Result<PathBuf, String> {
    if let Some(requested_path) = output_path_str.filter(|path| !path.trim().is_empty()) {
        let mut output_path = PathBuf::from(requested_path);
        let path_format = output_path
            .extension()
            .and_then(|value| value.to_str())
            .and_then(ImageStackOutputFormat::from_extension);
        if path_format != Some(output_format) {
            output_path.set_extension(output_format.canonical_extension());
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
    Ok(parent.join(format!(
        "{stem}_{suffix}.{}",
        output_format.canonical_extension()
    )))
}
