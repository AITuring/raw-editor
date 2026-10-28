//! Content orientation analysis for batch geometry review.
//!
//! Scores the four upright candidates of a preview with the content model and
//! accepts a rotation only with clear margins.

use crate::formats::is_raw_file;
use crate::image_loader::load_base_image_from_bytes_without_exif_persistence;
use crate::image_processing::downscale_f32_image;
use image::DynamicImage;
use std::borrow::Cow;
use std::path::Path;

pub(super) const CONTENT_ORIENTATION_ANALYSIS_EDGE: u32 = 640;

const CONTENT_ORIENTATION_MIN_UPRIGHT_SCORE: f32 = 0.62;

const CONTENT_ORIENTATION_MIN_CURRENT_MARGIN: f32 = 0.14;

const CONTENT_ORIENTATION_MIN_RUNNER_UP_MARGIN: f32 = 0.08;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ContentOrientationDecision {
    pub(super) orientation_steps: u8,
    pub(super) corrected: bool,
    pub(super) confidence: f32,
}

impl ContentOrientationDecision {
    pub(super) const fn preserve_current() -> Self {
        Self {
            orientation_steps: 0,
            corrected: false,
            confidence: 0.0,
        }
    }
}

/// Applies a content model's four upright scores conservatively. Index zero is the
/// image as decoded by the normal loader (including its EXIF handling); indexes
/// one through three are the same preview rotated by 90, 180, and 270 degrees.
///
/// EXIF is therefore only a starting point. A non-zero rotation must be both
/// convincingly upright and materially better than the starting image before the
/// batch exporter is allowed to change a source's physical direction.
pub(super) fn decide_content_orientation(scores: [f32; 4]) -> ContentOrientationDecision {
    let mut best_index = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    let mut runner_up_score = f32::NEG_INFINITY;

    for (index, raw_score) in scores.into_iter().enumerate() {
        let score = if raw_score.is_finite() {
            raw_score
        } else {
            f32::NEG_INFINITY
        };
        if score > best_score {
            runner_up_score = best_score;
            best_score = score;
            best_index = index;
        } else if score > runner_up_score {
            runner_up_score = score;
        }
    }

    let current_score = if scores[0].is_finite() {
        scores[0]
    } else {
        f32::NEG_INFINITY
    };
    let is_confident_non_zero_rotation = best_index != 0
        && best_score >= CONTENT_ORIENTATION_MIN_UPRIGHT_SCORE
        && best_score - current_score >= CONTENT_ORIENTATION_MIN_CURRENT_MARGIN
        && best_score - runner_up_score >= CONTENT_ORIENTATION_MIN_RUNNER_UP_MARGIN;

    if is_confident_non_zero_rotation {
        ContentOrientationDecision {
            orientation_steps: best_index as u8,
            corrected: true,
            confidence: best_score.clamp(0.0, 1.0),
        }
    } else {
        ContentOrientationDecision {
            orientation_steps: 0,
            corrected: false,
            confidence: 0.0,
        }
    }
}

pub(super) fn load_batch_content_orientation_preview(
    source_path: &Path,
    source_bytes: &[u8],
    settings: &crate::app_settings::AppSettings,
    target_edge: u32,
) -> Result<DynamicImage, String> {
    let source_path_string = source_path.to_string_lossy();
    let preview = if is_raw_file(source_path) {
        crate::file_management::try_load_embedded_raw_preview(source_path, target_edge)
            .or_else(|| {
                load_base_image_from_bytes_without_exif_persistence(
                    source_bytes,
                    &source_path_string,
                    true,
                    settings,
                    None,
                )
                .ok()
            })
            .ok_or_else(|| {
                format!(
                    "Failed to load a content-analysis preview for '{}'.",
                    source_path.display()
                )
            })?
    } else {
        crate::image_loader::load_image_with_orientation(source_bytes, None).map_err(|error| {
            format!(
                "Failed to load a content-analysis preview for '{}': {error}",
                source_path.display()
            )
        })?
    };

    Ok(downscale_f32_image(&preview, target_edge, target_edge))
}

pub(super) fn analyze_batch_content_orientation(
    preview: &DynamicImage,
    source_path: &Path,
    clip_models: &crate::ai_processing::ClipModels,
) -> Result<ContentOrientationDecision, String> {
    let mut scores = [0.0f32; 4];

    for (orientation_steps, score) in scores.iter_mut().enumerate() {
        let candidate = match orientation_steps {
            0 => Cow::Borrowed(preview),
            1 => Cow::Owned(preview.rotate90()),
            2 => Cow::Owned(preview.rotate180()),
            3 => Cow::Owned(preview.rotate270()),
            _ => unreachable!(),
        };
        *score = crate::tagging::score_content_orientation_with_clip(
            candidate.as_ref(),
            &clip_models.model,
            &clip_models.tokenizer,
        )
        .map_err(|error| {
            format!(
                "Content orientation analysis failed for '{}': {error}",
                source_path.display()
            )
        })?;
    }

    Ok(decide_content_orientation(scores))
}
