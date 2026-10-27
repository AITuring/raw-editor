//! Deterministic station-to-station seam contract for task 21.6.
//!
//! This module is an offline seam solver boundary.  It uses the existing
//! bounded binary min-cut implementation and is not wired into the default
//! compositor until station Coverage/Ownership provenance is available at this
//! stage of the production pipeline.

use std::time::Instant;

use super::super::seam_cut::cut_grid;

pub(crate) const MIN_OVERLAP_WIDTH_PX: u32 = 16;
const MIN_EVIDENCE_SAMPLES: usize = 16;
const MAX_GRAPH_CELLS: usize = 512 * 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StationSeamSolver {
    GraphCut,
    CenterlineFallback,
    NotRun,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StationSeamInput {
    pub width: usize,
    pub height: usize,
    pub base_rgb: Vec<[f32; 3]>,
    pub candidate_rgb: Vec<[f32; 3]>,
    pub base_coverage: Vec<bool>,
    pub candidate_coverage: Vec<bool>,
    pub overlap_width_px: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StationSeamResult {
    pub labels: Vec<u8>,
    pub solver: StationSeamSolver,
    pub elapsed_ms: u64,
    pub fallback_reason: Option<String>,
}

fn luma(pixel: [f32; 3]) -> f32 {
    pixel[0] * 0.299 + pixel[1] * 0.587 + pixel[2] * 0.114
}

fn gradient(pixels: &[[f32; 3]], width: usize, height: usize, x: usize, y: usize) -> f32 {
    let left = pixels[y * width + x.saturating_sub(1)];
    let right = pixels[y * width + (x + 1).min(width - 1)];
    let top = pixels[y.saturating_sub(1) * width + x];
    let bottom = pixels[(y + 1).min(height - 1) * width + x];
    ((luma(right) - luma(left)).abs() + (luma(bottom) - luma(top)).abs()) * 0.5
}

fn centerline_labels(width: usize, height: usize, input: &StationSeamInput) -> Vec<u8> {
    let mut labels = vec![0; width * height];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if input.candidate_coverage[index] && !input.base_coverage[index] {
                labels[index] = 255;
            } else if input.base_coverage[index]
                && input.candidate_coverage[index]
                && x >= width / 2
            {
                labels[index] = 255;
            }
        }
    }
    labels
}

fn validate(input: &StationSeamInput) -> Result<usize, String> {
    if input.width == 0 || input.height == 0 {
        return Err("station seam dimensions must be non-zero".to_string());
    }
    let count = input
        .width
        .checked_mul(input.height)
        .ok_or_else(|| "station seam dimensions overflow".to_string())?;
    if input.base_rgb.len() != count
        || input.candidate_rgb.len() != count
        || input.base_coverage.len() != count
        || input.candidate_coverage.len() != count
    {
        return Err("station seam planes have inconsistent dimensions".to_string());
    }
    if input
        .base_rgb
        .iter()
        .chain(input.candidate_rgb.iter())
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err("station seam pixels must be finite".to_string());
    }
    Ok(count)
}

/// Solve a Coverage-constrained station seam.  The returned byte is 255 for
/// candidate ownership and 0 for base ownership. Pixels covered by neither
/// source remain 0 and never participate in the graph.
pub(crate) fn solve_station_seam(input: &StationSeamInput) -> Result<StationSeamResult, String> {
    let count = validate(input)?;
    let started = Instant::now();
    let overlap_samples = input
        .base_coverage
        .iter()
        .zip(&input.candidate_coverage)
        .filter(|(base, candidate)| **base && **candidate)
        .count();
    let fallback_reason = if input.overlap_width_px < MIN_OVERLAP_WIDTH_PX {
        Some("overlap_too_narrow".to_string())
    } else if overlap_samples < MIN_EVIDENCE_SAMPLES {
        Some("insufficient_overlap_evidence".to_string())
    } else if count > MAX_GRAPH_CELLS {
        Some("graph_budget_exceeded".to_string())
    } else {
        None
    };
    if let Some(reason) = fallback_reason {
        return Ok(StationSeamResult {
            labels: centerline_labels(input.width, input.height, input),
            solver: StationSeamSolver::CenterlineFallback,
            elapsed_ms: started.elapsed().as_millis() as u64,
            fallback_reason: Some(reason),
        });
    }

    let mut preference = vec![0.0; count];
    let mut disagreement = vec![0.0; count];
    let mut fixed = vec![0i8; count];
    for y in 0..input.height {
        for x in 0..input.width {
            let index = y * input.width + x;
            let base_valid = input.base_coverage[index];
            let candidate_valid = input.candidate_coverage[index];
            fixed[index] = match (base_valid, candidate_valid) {
                (true, false) => -1,
                (false, true) => 1,
                (false, false) => -1,
                (true, true) => 0,
            };
            let base_gradient = gradient(&input.base_rgb, input.width, input.height, x, y);
            let candidate_gradient =
                gradient(&input.candidate_rgb, input.width, input.height, x, y);
            let colour_delta = input.base_rgb[index]
                .iter()
                .zip(input.candidate_rgb[index].iter())
                .map(|(base, candidate)| f64::from((base - candidate).abs()))
                .sum::<f64>() as f32;
            // Terminal preference favours the source with the calmer local
            // structure; pairwise disagreement keeps the seam away from
            // high-contrast mismatch paths.
            preference[index] = f64::from(candidate_gradient - base_gradient);
            disagreement[index] =
                f64::from(colour_delta + (candidate_gradient - base_gradient).abs());
        }
    }
    let labels = cut_grid(
        input.width,
        input.height,
        &preference,
        &disagreement,
        &fixed,
    );
    Ok(StationSeamResult {
        labels,
        solver: StationSeamSolver::GraphCut,
        elapsed_ms: started.elapsed().as_millis() as u64,
        fallback_reason: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coverage(width: usize, height: usize) -> Vec<bool> {
        vec![true; width * height]
    }

    #[test]
    fn narrow_overlap_uses_deterministic_centerline_fallback() {
        let input = StationSeamInput {
            width: 8,
            height: 4,
            base_rgb: vec![[0.4; 3]; 32],
            candidate_rgb: vec![[0.6; 3]; 32],
            base_coverage: coverage(8, 4),
            candidate_coverage: coverage(8, 4),
            overlap_width_px: 4,
        };
        let result = solve_station_seam(&input).expect("seam result");
        assert_eq!(result.solver, StationSeamSolver::CenterlineFallback);
        assert_eq!(
            result.fallback_reason.as_deref(),
            Some("overlap_too_narrow")
        );
        assert_eq!(result.labels[0], 0);
        assert_eq!(result.labels[7], 255);
    }

    #[test]
    fn graphcut_respects_coverage_and_is_repeatable() {
        let input = StationSeamInput {
            width: 8,
            height: 8,
            base_rgb: vec![[0.4; 3]; 64],
            candidate_rgb: vec![[0.6; 3]; 64],
            base_coverage: coverage(8, 8),
            candidate_coverage: coverage(8, 8),
            overlap_width_px: 32,
        };
        let first = solve_station_seam(&input).expect("seam result");
        let second = solve_station_seam(&input).expect("seam result");
        assert_eq!(first.solver, StationSeamSolver::GraphCut);
        assert_eq!(first.labels, second.labels);
        assert!(first.labels.iter().all(|&label| label == 0 || label == 255));
    }

    #[test]
    fn uncovered_pixels_never_become_candidate_owned() {
        let input = StationSeamInput {
            width: 8,
            height: 8,
            base_rgb: vec![[0.4; 3]; 64],
            candidate_rgb: vec![[0.6; 3]; 64],
            base_coverage: vec![false; 64],
            candidate_coverage: vec![false; 64],
            overlap_width_px: 32,
        };
        let result = solve_station_seam(&input).expect("seam result");
        assert!(result.labels.iter().all(|&label| label == 0));
        assert_eq!(result.solver, StationSeamSolver::CenterlineFallback);
    }
}
