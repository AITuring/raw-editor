//! Atomic Stack_Report file publication and post-output metadata updates.

use std::fs;
use std::path::{Path, PathBuf};

use super::report_document::StackReport;
use super::report_schema_core::STACK_REPORT_DIR_NAME;

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
    F: FnOnce(&mut StackReport) -> Result<(), String>,
{
    let json = fs::read_to_string(report_path).map_err(|error| {
        format!(
            "Failed to read the Stack_Report {}: {error}",
            report_path.display()
        )
    })?;
    let mut report: StackReport = serde_json::from_str(&json).map_err(|error| {
        format!(
            "Failed to parse the Stack_Report {}: {error}",
            report_path.display()
        )
    })?;
    apply(&mut report)?;
    write_stack_report_at(&report, report_path)?;
    Ok(())
}

/// Attach the result identifier to the report written by the matching stack
/// run.  This is done before the result enters the export store, so a later
/// save can update only the report belonging to the visible result.
pub(crate) fn assign_stack_report_result_id(
    report_path: &Path,
    result_id: &str,
) -> Result<(), String> {
    rewrite_stack_report(report_path, |report| {
        if !report.output.result_id.is_empty() && report.output.result_id != result_id {
            return Err(format!(
                "Stack_Report result ID '{}' does not match '{}'.",
                report.output.result_id, result_id
            ));
        }
        report.output.result_id = result_id.to_string();
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
) -> Result<(), String> {
    rewrite_stack_report(report_path, |report| {
        if report.output.result_id != result_id {
            return Err(format!(
                "Stack_Report result ID '{}' does not match '{}'.",
                report.output.result_id, result_id
            ));
        }
        report.output.written = true;
        report.output.format = format;
        report.output.path = output_path.to_string_lossy().into_owned();
        report.output.bit_depth = bit_depth;
        report.output.alpha_preserved = alpha_preserved;
        report.output.icc = icc;
        Ok(())
    })
}
