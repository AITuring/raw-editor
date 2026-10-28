//! Run-scoped resource guards, progress events and focus-stack cache keys.
//!
//! The RSS guard and memory check stop a focus-stack run at stage boundaries,
//! the worker and render-scale budgets bound memory, and the cache key inputs
//! hash every Source_RAW together with the pipeline version.

use crate::panorama_utils::stack_pipeline;
use crate::panorama_utils::stack_pipeline::degradation;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use crate::panorama_utils::stack_pipeline::resources;
use crate::panorama_utils::stack_pipeline::virtual_tile;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;
use tauri::AppHandle;
use tauri::Emitter;
use tauri::Runtime;

use super::{MAX_RETAINED_STACK_PIXELS, SCALABLE_STACK_THRESHOLD};

/// Owns the run-scoped RSS sampler and publishes its final sample even when a
/// focus-stack stage returns early.  The sampler only requests cancellation;
/// the caller checks it at stage boundaries and returns before writing output.
pub(super) struct FocusRssGuard {
    pub(super) sampler: Option<resources::RssSampler>,
    pub(super) threshold_bytes: u64,
    pub(super) recorder: Option<Arc<stack_report::StackReportRecorder>>,
}

impl FocusRssGuard {
    pub(super) fn cancelled(&self) -> bool {
        self.sampler
            .as_ref()
            .is_some_and(resources::RssSampler::cancelled)
    }

    pub(super) fn finish(&mut self) -> Option<resources::RssSample> {
        let sample = self
            .sampler
            .take()
            .map(|sampler| sampler.stop(self.threshold_bytes))?;
        if let Some(recorder) = self.recorder.as_ref() {
            recorder.update(|report| {
                report.resources.peak_rss_bytes = sample.peak_rss_bytes;
                report.resources.rss_sample_count = sample.sample_count;
                report.resources.memory_threshold_exceeded = sample.threshold_exceeded;
            });
        }
        if sample.threshold_exceeded {
            degradation::record_run_degradation(
                degradation::MEMORY_THRESHOLD_EXCEEDED,
                serde_json::json!({
                    "threshold_bytes": self.threshold_bytes,
                    "peak_rss_bytes": sample.peak_rss_bytes,
                    "sample_count": sample.sample_count,
                }),
            );
        }
        Some(sample)
    }
}

impl Drop for FocusRssGuard {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

pub(super) fn ensure_focus_memory_available(sampler: Option<&FocusRssGuard>) -> Result<(), String> {
    if degradation::run_generation_changed() {
        degradation::record_run_degradation(
            degradation::RUN_CANCELLED_BY_USER,
            serde_json::json!({"reason": "newer_image_stack_generation"}),
        );
        return Err("image-stack run cancelled by a newer request".to_string());
    }
    if sampler.is_some_and(FocusRssGuard::cancelled) {
        return Err("memory threshold exceeded during focus-stack run".to_string());
    }
    Ok(())
}

pub(super) const MAX_SCALABLE_PREPARATION_WORKERS: usize = 6;

pub(super) const PREPARATION_RAM_PER_WORKER_BYTES: u64 = 1024 * 1024 * 1024;

pub(super) const MAX_IN_MEMORY_PANORAMA_PIXELS: u64 = 240_000_000;

pub(super) fn stack_requires_bounded_memory(image_paths: &[String]) -> bool {
    if image_paths.len() > SCALABLE_STACK_THRESHOLD {
        return true;
    }
    let mut pixels = 0u64;
    for path in image_paths {
        // Unknown/RAW dimensions use the bounded path as well. A count-only
        // limit lets even two 200MP phone frames retain several GiB of floats.
        let Ok((width, height)) = image::image_dimensions(path) else {
            return true;
        };
        pixels = pixels.saturating_add(u64::from(width) * u64::from(height));
        if pixels > MAX_RETAINED_STACK_PIXELS {
            return true;
        }
    }
    false
}

pub(super) fn scalable_alignment_budget(image_count: usize) -> (u32, usize) {
    if image_count <= 64 {
        (2_400, 1_600)
    } else if image_count <= 128 {
        (1_800, 1_100)
    } else {
        (1_536, 800)
    }
}

pub(super) fn bounded_preparation_worker_count(
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

pub(super) fn scalable_preparation_worker_count(image_count: usize) -> usize {
    let available_threads = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1);
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    bounded_preparation_worker_count(image_count, available_threads, system.available_memory())
}

pub(super) fn memory_safe_panorama_render_scale(width: u32, height: u32) -> f64 {
    let pixels = u64::from(width).saturating_mul(u64::from(height));
    if pixels <= MAX_IN_MEMORY_PANORAMA_PIXELS || pixels == 0 {
        1.0
    } else {
        (MAX_IN_MEMORY_PANORAMA_PIXELS as f64 / pixels as f64).sqrt()
    }
}

pub(super) fn emit_match_progress<R: Runtime>(
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

pub(super) struct PairMatchProgress<'a, R: Runtime> {
    pub(super) completed: &'a Mutex<usize>,
    pub(super) total: usize,
    pub(super) progress_step: usize,
    pub(super) app_handle: &'a AppHandle<R>,
    pub(super) progress_event: &'a str,
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

/// SHA-256 over every byte of each selected Source_RAW, in the order the paths
/// were handed in.  `None` marks a source that could not be read.
///
/// Every file is opened read only with `File::open` and streamed (需求 4.8), so
/// a 50 MB RAW never lands in memory as a whole and no path in this function can
/// modify a source.  Hashing runs on the `rayon` pool; the result is a plain
/// per-path mapping, so the thread count cannot reach it.
pub(super) fn focus_stack_source_digests(image_paths: &[String]) -> Vec<Option<[u8; 32]>> {
    image_paths
        .par_iter()
        .map(|path| {
            let (real_path, _) = crate::file_management::parse_virtual_path(path);
            match virtual_tile::source_file_sha256(&real_path) {
                Ok(digest) => Some(digest),
                Err(error) => {
                    println!("Focus stack: {error}");
                    None
                }
            }
        })
        .collect()
}

/// Lower hex of a source digest, or the empty string when the file could not be
/// read.  An empty string is what the Stack_Report records for "not measured",
/// which must stay distinguishable from a digest that is genuinely all zero.
pub(super) fn source_digest_hex(digest: Option<[u8; 32]>) -> String {
    digest
        .map(|digest| virtual_tile::hex_digest(&digest))
        .unwrap_or_default()
}

/// The Source_RAW digests taken before the run, together with the three
/// Virtual_Tile `cache_key` ingredients they feed (需求 4.5).
pub(super) struct FocusStackSourceDigests {
    /// Pipeline version, sorted absolute source paths, sorted source digests.
    pub(super) inputs: virtual_tile::CacheKeyInputs,
    /// Lower hex digest per selected path, in selection order (需求 4.8 / 12.8).
    pub(super) sha256_before: Vec<String>,
}

/// Hash every selected Source_RAW once and derive the run's `cache_key`
/// ingredients from the result.
///
/// A source that cannot be opened contributes an all-zero digest to the key
/// rather than dropping out of the set: the key must stay a pure function of the
/// selection, and an unreadable source fails the run later on its own terms.
/// `sha256_before` keeps that case empty instead, so the report never claims a
/// digest it did not measure.
///
/// The ingredients are sorted after hashing, so neither the import order nor the
/// completion order of the `rayon` tasks reaches the key.
pub(super) fn focus_stack_cache_key_inputs(image_paths: &[String]) -> FocusStackSourceDigests {
    let digests = focus_stack_source_digests(image_paths);
    let sources = image_paths
        .iter()
        .zip(&digests)
        .map(|(path, digest)| (path.clone(), digest.unwrap_or([0u8; 32])))
        .collect::<Vec<_>>();
    // The proxy is part of the focus-stack analysis input even though its
    // pixels are generated inside the preparation workers.  Include the
    // versioned resize contract and the input-count budget in the cache
    // identity so a change to proxy dimensions/filtering cannot reuse a stale
    // Virtual_Tile entry.  Source dimensions remain content-bound through the
    // digest set below.
    FocusStackSourceDigests {
        inputs: virtual_tile::CacheKeyInputs::new(
            &focus_stack_cache_pipeline_version(image_paths.len()),
            &sources,
        ),
        sha256_before: digests.into_iter().map(source_digest_hex).collect(),
    }
}

pub(super) fn focus_stack_cache_pipeline_version(input_count: usize) -> String {
    stack_pipeline::proxy::ProxyBudget::for_source(1, 1, input_count, None)
        .map(|budget| {
            format!(
                "{}|{}",
                crate::image_stack::STACK_PIPELINE_VERSION,
                budget.cache_key_fragment()
            )
        })
        .unwrap_or_else(|_| crate::image_stack::STACK_PIPELINE_VERSION.to_string())
}
