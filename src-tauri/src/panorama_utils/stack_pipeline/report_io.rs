//! Atomic Stack_Report file publication and post-output metadata updates.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::value::RawValue;

use super::report_document::StackReport;
use super::report_schema_core::STACK_REPORT_DIR_NAME;
use super::report_schema_metrics::OutputReport;

/// Resolve `<app_cache_dir>/stack-reports` without forcing the caller to
/// import `tauri::Manager`.
pub(crate) fn resolve_stack_report_dir<R: tauri::Runtime>(
    app_handle: &tauri::AppHandle<R>,
) -> Option<PathBuf> {
    use tauri::Manager;

    if let Some(directory) = std::env::var_os("RAW_EDITOR_STACK_REPORT_DIR") {
        return Some(PathBuf::from(directory));
    }

    app_handle
        .path()
        .app_cache_dir()
        .ok()
        .map(|directory| directory.join(STACK_REPORT_DIR_NAME))
}
/// Serialize `report` into `directory` using a temporary file plus rename so a
/// partially written document is never observable.
pub(crate) fn write_stack_report(
    report: &StackReport,
    directory: &Path,
) -> Result<PathBuf, String> {
    let run_id = if report.run_id.is_empty() {
        "unknown".to_string()
    } else {
        report.run_id.clone()
    };
    let final_path = directory.join(format!("stack-report-{run_id}.json"));
    write_stack_report_at(report, &final_path)
}

fn write_stack_report_at(report: &StackReport, final_path: &Path) -> Result<PathBuf, String> {
    let json = report.to_json_string()?;
    write_stack_report_json_at(&json, final_path)
}

fn write_stack_report_json_at(json: &str, final_path: &Path) -> Result<PathBuf, String> {
    let directory = final_path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(directory).map_err(|error| {
        format!(
            "Failed to create the stack report directory {}: {error}",
            directory.display()
        )
    })?;
    let run_id = final_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("stack-report");
    let temporary_path = directory.join(format!(".{run_id}.tmp"));
    fs::write(&temporary_path, json.as_bytes()).map_err(|error| {
        format!(
            "Failed to write the stack report {}: {error}",
            temporary_path.display()
        )
    })?;
    fs::rename(&temporary_path, final_path).map_err(|error| {
        let _ = fs::remove_file(&temporary_path);
        format!(
            "Failed to publish the stack report {}: {error}",
            final_path.display()
        )
    })?;
    Ok(final_path.to_path_buf())
}

fn rewrite_stack_report<F>(report_path: &Path, apply: F) -> Result<(), String>
where
    F: FnOnce(&mut OutputReport) -> Result<(), String>,
{
    let json = fs::read_to_string(report_path).map_err(|error| {
        format!(
            "Failed to read the Stack_Report {}: {error}",
            report_path.display()
        )
    })?;
    // Never parse unrelated geometry/quality evidence through f64 and write
    // it back during an export update: the decimal token can change by an ULP.
    // RawValue also preserves evidence added by newer report producers.
    type RawFields = BTreeMap<String, Box<RawValue>>;
    let mut report: RawFields = serde_json::from_str(&json).map_err(|error| {
        format!(
            "Failed to parse the Stack_Report {}: {error}",
            report_path.display()
        )
    })?;
    let raw_output = report
        .remove("output")
        .ok_or_else(|| "Stack_Report is missing its output record".to_string())?;
    let mut output_fields: RawFields = serde_json::from_str(raw_output.get())
        .map_err(|error| format!("Failed to read raw Stack_Report output fields: {error}"))?;
    let mut output: OutputReport = serde_json::from_str(raw_output.get())
        .map_err(|error| format!("Failed to parse Stack_Report output: {error}"))?;
    let fields_of = |output: &OutputReport| -> Result<RawFields, String> {
        let json = serde_json::to_string(output)
            .map_err(|error| format!("Failed to serialize Stack_Report output: {error}"))?;
        serde_json::from_str(&json)
            .map_err(|error| format!("Failed to read serialized output fields: {error}"))
    };
    let before = fields_of(&output)?;
    apply(&mut output)?;
    for (key, value) in fields_of(&output)? {
        if !output_fields.contains_key(&key)
            || before.get(&key).map(|old| old.get()) != Some(value.get())
        {
            output_fields.insert(key, value);
        }
    }
    // Keep unknown output fields and unchanged values (including preview
    // evidence) verbatim too; only changed/new fields are serialized.
    report.insert(
        "output".into(),
        serde_json::value::to_raw_value(&output_fields)
            .map_err(|error| format!("Failed to serialize output field update: {error}"))?,
    );
    let rewritten = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("Failed to serialize raw Stack_Report fields: {error}"))?;
    write_stack_report_json_at(&rewritten, report_path)?;
    Ok(())
}

/// Attach the result identifier to the report written by the matching stack
/// run.  This is done before the result enters the export store, so a later
/// save can update only the report belonging to the visible result.
pub(crate) fn assign_stack_report_result_id(
    report_path: &Path,
    result_id: &str,
) -> Result<(), String> {
    rewrite_stack_report(report_path, |output| {
        if !output.result_id.is_empty() && output.result_id != result_id {
            return Err(format!(
                "Stack_Report result ID '{}' does not match '{}'.",
                output.result_id, result_id
            ));
        }
        output.result_id = result_id.to_string();
        Ok(())
    })
}

/// Record the metadata observed by reopening a successfully published output.
/// The rewrite uses the same temporary-file-plus-rename boundary as the initial
/// report, so readers never observe a partially updated document.
pub(crate) fn update_stack_report_output(
    report_path: &Path,
    result_id: &str,
    format: String,
    bit_depth: u8,
    alpha_preserved: bool,
    icc: String,
    output_path: &Path,
    dimensions: (u32, u32),
    tiff_extra_samples: Option<Vec<u16>>,
) -> Result<(), String> {
    rewrite_stack_report(report_path, |output| {
        if output.result_id != result_id {
            return Err(format!(
                "Stack_Report result ID '{}' does not match '{}'.",
                output.result_id, result_id
            ));
        }
        output.written = true;
        output.format = format;
        output.path = output_path.to_string_lossy().into_owned();
        output.width = dimensions.0;
        output.height = dimensions.1;
        output.bit_depth = bit_depth;
        output.alpha_preserved = alpha_preserved;
        output.icc = icc;
        output.tiff_extra_samples = tiff_extra_samples;
        output.artifact_validation = super::OutputArtifactValidation::Pass;
        output.artifact_validation_error = None;
        Ok(())
    })
}

/// Keep an acceptance export failure in its run report without presenting it
/// as a published artifact. Production save failures leave earlier successful
/// exports unchanged; the fixture calls this only for its one attempted output.
#[cfg(test)]
pub(crate) fn record_stack_report_artifact_failure(
    report_path: &Path,
    result_id: &str,
    error: &str,
) -> Result<(), String> {
    rewrite_stack_report(report_path, |output| {
        if output.result_id != result_id {
            return Err("Stack_Report artifact failure result ID does not match".into());
        }
        output.written = false;
        output.artifact_validation = super::OutputArtifactValidation::Fail;
        output.artifact_validation_error = Some(error.to_string());
        Ok(())
    })
}

#[cfg(test)]
#[path = "report_output_rewrite_tests.rs"]
mod output_rewrite_tests;
