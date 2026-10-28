//! Export task registration, cancellation and progress events.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use tauri::Emitter;

pub(super) fn ensure_export_not_cancelled(cancellation_token: &AtomicBool) -> Result<(), String> {
    if cancellation_token.load(Ordering::SeqCst) {
        Err("Export cancelled".to_string())
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum ExportCancellationRequest {
    Requested,
    AlreadyRequested,
    NoActiveTask,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct BatchExportSummary {
    pub(super) completed: usize,
    pub(super) errors: Vec<String>,
}

#[derive(Clone, Copy)]
pub(super) struct ExportTaskEvents {
    pub(super) cancelled: &'static str,
    pub(super) complete: &'static str,
    pub(super) error: &'static str,
    pub(super) progress: &'static str,
}

#[derive(Clone, Copy)]
pub(super) enum ExportTaskEventScope {
    Standard,
    BatchGeometry,
}

impl ExportTaskEventScope {
    pub(super) const fn events(self) -> ExportTaskEvents {
        match self {
            Self::Standard => ExportTaskEvents {
                cancelled: "export-cancelled",
                complete: "export-complete",
                error: "export-error",
                progress: "batch-export-progress",
            },
            Self::BatchGeometry => ExportTaskEvents {
                cancelled: "batch-geometry-cancelled",
                complete: "batch-geometry-complete",
                error: "batch-geometry-error",
                progress: "batch-geometry-progress",
            },
        }
    }
}

pub(super) fn summarize_batch_export_results(
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

pub(super) struct ExportTaskGuard {
    pub(super) task_token: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    pub(super) cancellation_token: Arc<AtomicBool>,
    pub(super) app_handle: Option<tauri::AppHandle>,
    pub(super) events: ExportTaskEvents,
}

impl ExportTaskGuard {
    pub(super) fn new(
        task_token: Arc<Mutex<Option<Arc<AtomicBool>>>>,
        cancellation_token: Arc<AtomicBool>,
        events: ExportTaskEvents,
    ) -> Self {
        Self {
            task_token,
            cancellation_token,
            app_handle: None,
            events,
        }
    }

    pub(super) fn with_app_handle(
        task_token: Arc<Mutex<Option<Arc<AtomicBool>>>>,
        cancellation_token: Arc<AtomicBool>,
        app_handle: tauri::AppHandle,
        events: ExportTaskEvents,
    ) -> Self {
        let mut guard = Self::new(task_token, cancellation_token, events);
        guard.app_handle = Some(app_handle);
        guard
    }
}

pub(super) fn register_export_task(
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

pub(super) fn request_export_cancellation<F>(
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

pub(super) fn finish_export_task<F>(
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
        let events = self.events;
        let _ = finish_export_task(
            &self.task_token,
            &self.cancellation_token,
            |cancelled| match (cancelled, app_handle) {
                (true, Some(app_handle)) => {
                    let _ = app_handle.emit(events.cancelled, ());
                }
                (false, Some(app_handle)) => {
                    let _ = app_handle.emit(events.error, "Export task terminated unexpectedly");
                }
                _ => {}
            },
        );
    }
}
