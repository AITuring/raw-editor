//! Run-scoped orchestration state and resource policy for panorama/focus-stack runs.
//!
//! The production stitcher keeps the matching, station/virtual-tile solving,
//! and rendering stages in `panorama_stitching.rs`.  This module owns the
//! small set of values that connect those stages: run cancellation/RSS
//! sampling, memory-bounded preparation policy, and the output envelope.
//! Keeping these policies together makes the top-level entry point easier to
//! read without changing any algorithm or public API.

use super::ImageInfo;
use super::{degradation, resources, stack_report};
use image::{DynamicImage, GrayImage};
use nalgebra::Matrix3;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Owns the run-scoped RSS sampler and publishes its final sample even when a
/// focus-stack stage returns early. The sampler only requests cancellation;
/// the caller checks it at stage boundaries and returns before writing output.
pub(crate) struct FocusRssGuard {
    pub(crate) sampler: Option<resources::RssSampler>,
    pub(crate) threshold_bytes: u64,
    pub(crate) recorder: Option<Arc<stack_report::StackReportRecorder>>,
}

impl FocusRssGuard {
    pub(crate) fn cancelled(&self) -> bool {
        self.sampler
            .as_ref()
            .is_some_and(resources::RssSampler::cancelled)
    }

    pub(crate) fn finish(&mut self) -> Option<resources::RssSample> {
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

pub(crate) fn ensure_focus_memory_available(sampler: Option<&FocusRssGuard>) -> Result<(), String> {
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

pub(crate) const SCALABLE_STACK_THRESHOLD: usize = 30;
pub(crate) const MAX_SCALABLE_PREPARATION_WORKERS: usize = 6;
pub(crate) const PREPARATION_RAM_PER_WORKER_BYTES: u64 = 1024 * 1024 * 1024;
pub(crate) const MAX_IN_MEMORY_PANORAMA_PIXELS: u64 = 240_000_000;
pub(crate) const MAX_STITCH_SOURCE_IMAGES: usize = 500;
pub(crate) const MAX_RETAINED_STACK_PIXELS: u64 = 120_000_000;

pub(crate) fn stack_requires_bounded_memory(image_paths: &[String]) -> bool {
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

pub(crate) struct StitchOutcome {
    pub image: DynamicImage,
    /// The layered path's output Coverage_Mask, the same size as `image`;
    /// the export writes it as alpha (需求 10.3 / 10.6).
    pub coverage: Option<GrayImage>,
    pub full_canvas_width: u32,
    pub full_canvas_height: u32,
    pub render_scale: f64,
    pub ordered_paths: Vec<String>,
    /// The report written for this focus-stack run. Image-stack export fills
    /// in the result identifier and output metadata after the file is
    /// published, while keeping the report tied to the same run.
    pub stack_report_path: Option<PathBuf>,
}

pub(crate) fn scalable_alignment_budget(image_count: usize) -> (u32, usize) {
    if image_count <= 64 {
        (2_400, 1_600)
    } else if image_count <= 128 {
        (1_800, 1_100)
    } else {
        (1_536, 800)
    }
}

pub(crate) fn bounded_preparation_worker_count(
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

pub(crate) fn scalable_preparation_worker_count(image_count: usize) -> usize {
    let available_threads = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1);
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    bounded_preparation_worker_count(image_count, available_threads, system.available_memory())
}

pub(crate) fn memory_safe_panorama_render_scale(width: u32, height: u32) -> f64 {
    let pixels = u64::from(width).saturating_mul(u64::from(height));
    if pixels <= MAX_IN_MEMORY_PANORAMA_PIXELS || pixels == 0 {
        1.0
    } else {
        (MAX_IN_MEMORY_PANORAMA_PIXELS as f64 / pixels as f64).sqrt()
    }
}

#[cfg(test)]
pub(crate) fn scaled_homographies(
    homographies: &HashMap<usize, Matrix3<f64>>,
    scale: f64,
) -> HashMap<usize, Matrix3<f64>> {
    let scale_matrix = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
    homographies
        .iter()
        .map(|(&id, homography)| (id, scale_matrix * homography))
        .collect()
}

pub(crate) fn scaled_source_render_homographies(
    homographies: &HashMap<usize, Matrix3<f64>>,
    scale: f64,
) -> HashMap<usize, Matrix3<f64>> {
    let output_scale = Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0);
    let source_scale_inverse = Matrix3::new(
        scale.recip(),
        0.0,
        0.0,
        0.0,
        scale.recip(),
        0.0,
        0.0,
        0.0,
        1.0,
    );
    homographies
        .iter()
        .map(|(&id, homography)| (id, output_scale * homography * source_scale_inverse))
        .collect()
}

pub(crate) fn scaled_render_image_info(image: &ImageInfo, scale: f64) -> ImageInfo {
    ImageInfo {
        id: image.id,
        filename: image.filename.clone(),
        width: (f64::from(image.width) * scale).round().max(1.0) as u32,
        height: (f64::from(image.height) * scale).round().max(1.0) as u32,
        alignment_image: image::GrayImage::new(0, 0),
        full_image: None,
        scale_factor: image.scale_factor,
        focal_length_35mm: image.focal_length_35mm,
        overview_reference: image.overview_reference,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: image.foreground_range,
        foreground_mask: image.foreground_mask.clone(),
        horizontal_edge_rows: image.horizontal_edge_rows.clone(),
        vertical_edge_columns: image.vertical_edge_columns.clone(),
    }
}
