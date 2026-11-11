//! Sparse ROI selection and confidence / effective-pixel coverage.
//!
//! Deterministic sparse ROIs per side, the confidence coverage statistics and
//! the streaming effective-pixel coverage of the final owner planes.

use super::super::degradation;
use super::super::quality_gate;
use super::super::quality_gate::EffectivePixelCoverage;
use super::super::quality_gate::QualityRoi;
use image::GrayImage;
use nalgebra::Point2;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy)]
pub(super) struct ConfidenceCoverageStats {
    pub(super) textured_pixels: u64,
    pub(super) flat_pixels: u64,
    pub(super) textured_low: u64,
    pub(super) flat_low: u64,
    pub(super) textured_unknown: u64,
    pub(super) flat_unknown: u64,
}

/// Keep the observation-only gate bounded on the full-resolution acceptance
/// renders. The exact region labelling and distance transform are useful for
/// ordinary unit-sized fixtures, but a 9k by 11k canvas would allocate and
/// scan several gigabytes just to choose 512px ROIs. A deterministic lattice
/// still checks ownership and coverage over every pixel in a candidate ROI;
/// effective coverage and boundary tracing use separate streaming passes.
pub(crate) fn select_sparse_rois(
    coverage: &GrayImage,
    ownership: &[u16],
    origin: (f64, f64),
) -> Vec<QualityRoi> {
    select_sparse_rois_with_evidence(coverage, ownership, ownership, origin)
}

/// Select Quality_Gate windows from a station-level evidence plane while
/// retaining the raw owner at the window origin for Source_RAW provenance.
///
/// Focus ownership can switch between members of the same Capture_Station at
/// a much finer granularity than the station's delivered geometry.  In that
/// case the raw owner plane has no 64px interior even though the station plane
/// is continuous.  The station evidence plane is only used for geometric
/// eligibility; the returned ROI keeps the raw owner so source sampling and
/// report provenance still use the existing Source_RAW path.
pub(crate) fn select_sparse_rois_with_evidence(
    coverage: &GrayImage,
    ownership: &[u16],
    evidence_ownership: &[u16],
    origin: (f64, f64),
) -> Vec<QualityRoi> {
    select_sparse_rois_with_evidence_side(
        coverage,
        ownership,
        evidence_ownership,
        origin,
        quality_gate::QUALITY_ROI_SIDE,
    )
}

/// Select station evidence windows with a bounded primary side.  The public
/// selector above keeps the historical 512px contract used by unit tests and
/// ordinary source-owner gates; station-level acceptance can opt into the
/// smaller 256px window because its evidence plane already collapses the raw
/// focus members to one station identity.
pub(crate) fn select_sparse_rois_with_evidence_side(
    coverage: &GrayImage,
    ownership: &[u16],
    evidence_ownership: &[u16],
    origin: (f64, f64),
    primary_side: u32,
) -> Vec<QualityRoi> {
    // Preserve the required 512px evidence window whenever the owner plane
    // contains enough such interiors.  Focus ownership is cell based, so a
    // valid run can have many sharp, source-owned regions smaller than 512px
    // even while the whole-image owner coverage passes.  Returning zero ROIs
    // in that case makes gradient/MTF a geometry-evidence failure rather than
    // a measurement of the delivered pixels.  The 256px fallback keeps the
    // same 16px owner/coverage ring and unchanged metric thresholds; it only
    // restores the minimum eight measurable interiors required by the gate.
    let primary = sparse_rois_for_side(
        coverage,
        ownership,
        evidence_ownership,
        origin,
        primary_side,
    );
    if primary.len() >= quality_gate::QUALITY_MIN_MEASURABLE {
        return primary;
    }
    let mut best = primary;
    // A 256px core is the first fallback. Highly interleaved focus owners
    // can still leave fewer than eight 256px interiors; continue at 128px so
    // the dense gradient criterion has auditable samples without changing
    // its 0.95 threshold. The 128px minimum remains large enough for the
    // 13px acutance support and the fixed 16px owner ring. Real focus stacks
    // can still have owner runs shorter than 128px after the final virtual
    // tile warp, so keep a 64px evidence window as the last bounded fallback.
    // It is still larger than the acutance support plus both owner margins;
    // the quality thresholds and the per-pixel owner checks remain unchanged.
    for side in [
        quality_gate::QUALITY_ROI_SIDE / 2,
        quality_gate::QUALITY_ROI_SIDE / 4,
        quality_gate::QUALITY_ROI_SIDE / 8,
    ] {
        if side < 2 {
            continue;
        }
        let fallback = sparse_rois_for_side(coverage, ownership, evidence_ownership, origin, side);
        if fallback.len() > best.len() {
            best = fallback;
        }
        if best.len() >= quality_gate::QUALITY_MIN_MEASURABLE {
            break;
        }
    }
    best
}

pub(super) fn sparse_rois_for_side(
    coverage: &GrayImage,
    ownership: &[u16],
    evidence_ownership: &[u16],
    origin: (f64, f64),
    side: u32,
) -> Vec<QualityRoi> {
    let (width, height) = coverage.dimensions();
    if side == 0
        || width < side
        || height < side
        || ownership.len() != coverage.as_raw().len()
        || evidence_ownership.len() != coverage.as_raw().len()
    {
        return Vec::new();
    }
    let step = (side as usize / 2).max(1);
    let margin = quality_gate::QUALITY_ROI_MARGIN as u32;

    // Build compact horizontal runs once, then answer each candidate's
    // core+margin query by checking one interval per row.  The previous
    // implementation scanned every pixel in every candidate window; on a
    // 7.6k x 3.1k scale=0.5 canvas that repeated tens of millions of pixel
    // reads before Quality_Gate had even selected its first ROI.
    let mut row_runs: Vec<Vec<(u32, u32, u16)>> = (0..height).map(|_| Vec::new()).collect();
    for y in 0..height {
        let mut run_start = 0u32;
        let mut run_label = 0u16;
        for x in 0..=width {
            let label = if x < width {
                let index = (y * width + x) as usize;
                (coverage.as_raw()[index] != 0)
                    .then_some(evidence_ownership[index])
                    .unwrap_or(0)
            } else {
                0
            };
            if label == run_label {
                continue;
            }
            if run_label != 0 {
                row_runs[y as usize].push((run_start, x, run_label));
            }
            run_start = x;
            run_label = label;
        }
    }
    let row_has_run = |runs: &[(u32, u32, u16)], label: u16, left: u32, right: u32| {
        runs.iter()
            .any(|&(start, end, run_label)| run_label == label && start <= left && right <= end)
    };

    let mut result = Vec::new();
    for y in (0..=height - side).step_by(step) {
        for x in (0..=width - side).step_by(step) {
            let index = (y * width + x) as usize;
            let owner = ownership[index];
            let evidence_owner = evidence_ownership[index];
            if owner == 0 || evidence_owner == 0 || coverage.get_pixel(x, y)[0] == 0 {
                continue;
            }
            // A candidate must have a 16px owner/coverage margin on all four
            // sides. Checking this narrow ring is equivalent to the exact
            // distance-transform predicate used by the small-plane selector,
            // while keeping the large-plane path bounded.
            let margin_ok = x >= margin
                && y >= margin
                && x + side + margin <= width
                && y + side + margin <= height
                && (y - margin..y + side + margin).all(|row| {
                    row_has_run(
                        &row_runs[row as usize],
                        evidence_owner,
                        x - margin,
                        x + side + margin,
                    )
                });
            if !margin_ok {
                continue;
            }
            result.push(QualityRoi {
                x,
                y,
                side,
                region_label: result.len() as u32 + 1,
                owner,
                world_origin: (origin.0 + f64::from(x), origin.1 + f64::from(y)),
                boundary_clearance: f64::from(side),
            });
        }
    }
    if result.len() <= quality_gate::QUALITY_ROI_MAX_COUNT {
        return result;
    }
    // Scan the whole canvas before capping. Returning the first 256 lattice
    // candidates only measures the upper part of a native panorama. Keep one
    // eligible candidate per source owner, then spread the remaining samples
    // over the complete deterministic lattice.
    let mut first_per_owner = BTreeMap::new();
    for (index, roi) in result.iter().enumerate() {
        first_per_owner.entry(roi.owner).or_insert(index);
    }
    let mut selected = first_per_owner.into_values().collect::<BTreeSet<_>>();
    if selected.len() > quality_gate::QUALITY_ROI_MAX_COUNT {
        return Vec::new();
    }
    let remaining = quality_gate::QUALITY_ROI_MAX_COUNT.saturating_sub(selected.len());
    for slot in 0..remaining {
        selected.insert(slot * (result.len() - 1) / remaining.saturating_sub(1).max(1));
    }
    for index in 0..result.len() {
        if selected.len() >= quality_gate::QUALITY_ROI_MAX_COUNT {
            break;
        }
        selected.insert(index);
    }
    selected
        .into_iter()
        .take(quality_gate::QUALITY_ROI_MAX_COUNT)
        .map(|index| result[index].clone())
        .collect()
}

pub(super) fn confidence_coverage_stats(
    coverage: &GrayImage,
    confidence: &[f32],
    textured: &[u8],
) -> Option<ConfidenceCoverageStats> {
    let (width, height) = coverage.dimensions();
    if textured.len() != confidence.len() || confidence.len() != coverage.as_raw().len() {
        return None;
    }
    let mut stats = ConfidenceCoverageStats {
        textured_pixels: 0,
        flat_pixels: 0,
        textured_low: 0,
        flat_low: 0,
        textured_unknown: 0,
        flat_unknown: 0,
    };
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if coverage.as_raw()[index] == 0 {
                continue;
            }
            let is_textured = textured[index] != 0;
            if is_textured {
                stats.textured_pixels = stats.textured_pixels.saturating_add(1);
            } else {
                stats.flat_pixels = stats.flat_pixels.saturating_add(1);
            }
            if !confidence[index].is_finite() {
                if is_textured {
                    stats.textured_unknown = stats.textured_unknown.saturating_add(1);
                } else {
                    stats.flat_unknown = stats.flat_unknown.saturating_add(1);
                }
            } else if confidence[index] < 0.05 {
                if is_textured {
                    stats.textured_low = stats.textured_low.saturating_add(1);
                } else {
                    stats.flat_low = stats.flat_low.saturating_add(1);
                }
            }
        }
    }
    Some(stats)
}

/// Accumulate one output pixel while the compositor maps it back to its
/// ownership cell. Row-major calls preserve the diagnostic sum order without
/// retaining three additional full-resolution score/candidate planes.
pub(crate) fn accumulate_confidence_score(
    result: &mut crate::panorama_utils::stack_pipeline::report::ConfidenceScoreDistribution,
    covered: u8,
    confidence: f32,
    textured: u8,
    evidence: crate::panorama_utils::stack_pipeline::focus_fuser::SharpnessCellEvidence,
) {
    if covered == 0 || textured == 0 || !confidence.is_finite() || confidence >= 0.05 {
        return;
    }
    result.textured_low_count += 1;
    result.winner_sum += f64::from(evidence.winner_score);
    result.runner_up_sum += f64::from(evidence.runner_up_score);
    let wb = (f64::from(evidence.winner_score).clamp(0.0, 1.0) * 20.0).floor() as usize;
    let rb = (f64::from(evidence.runner_up_score).clamp(0.0, 1.0) * 20.0).floor() as usize;
    result.winner_histogram.resize(21, 0);
    result.runner_up_histogram.resize(21, 0);
    result.winner_histogram[wb.min(20)] += 1;
    result.runner_up_histogram[rb.min(20)] += 1;
    if evidence.winner_score < 0.10 {
        result.winner_below_0_10 += 1;
    } else if evidence.winner_score >= 0.10 && evidence.runner_up_score >= 0.10 {
        result.both_scores_at_least_0_10 += 1;
    } else if evidence.candidate_count <= 1 {
        result.single_candidate_count += 1;
    }
}

/// Count the unique projected source coverage one scanline at a time. This is
/// equivalent to `quality_gate::effective_pixel_coverage`, but never allocates
/// a canvas-sized union bitmap (which is prohibitive for the acceptance
/// renders). Only the current row's sorted intervals are resident.
pub(super) fn effective_pixel_coverage_streaming(
    coverage: &GrayImage,
    source_quadrilaterals: &[[Point2<f64>; 4]],
    canvas_world_origin: (f64, f64),
) -> Result<EffectivePixelCoverage, &'static str> {
    let (width, height) = coverage.dimensions();
    if width == 0 || height == 0 || source_quadrilaterals.is_empty() {
        return Err(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE);
    }
    let output_nontransparent = coverage
        .as_raw()
        .iter()
        .filter(|&&value| value != 0)
        .count() as u64;
    let mut projected_union = 0u64;
    for y in 0..height {
        let scan_y = canvas_world_origin.1 + f64::from(y) + 0.5;
        let mut intervals = Vec::<(f64, f64)>::new();
        for quad in source_quadrilaterals {
            if !quad
                .iter()
                .all(|point| point.x.is_finite() && point.y.is_finite())
            {
                return Err(degradation::DIAGNOSTICS_ROI_INVALID);
            }
            let mut intersections = [0.0f64; 4];
            let mut count = 0usize;
            for edge in 0..4 {
                let a = quad[edge];
                let b = quad[(edge + 1) % 4];
                if ((a.y <= scan_y && scan_y < b.y) || (b.y <= scan_y && scan_y < a.y))
                    && count < intersections.len()
                {
                    intersections[count] = a.x + (scan_y - a.y) * (b.x - a.x) / (b.y - a.y);
                    count += 1;
                }
            }
            if count >= 2 {
                intersections[..count].sort_by(f64::total_cmp);
                for pair in intersections[..count].chunks_exact(2) {
                    let start = (pair[0] - canvas_world_origin.0 - 0.5)
                        .ceil()
                        .max(0.0)
                        .min(f64::from(width)) as u32;
                    let end = (pair[1] - canvas_world_origin.0 - 0.5)
                        .ceil()
                        .max(0.0)
                        .min(f64::from(width)) as u32;
                    if end > start {
                        intervals.push((f64::from(start), f64::from(end)));
                    }
                }
            }
        }
        intervals.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
        });
        let mut union = 0u64;
        let mut active: Option<(f64, f64)> = None;
        for (start, end) in intervals {
            let Some((active_start, active_end)) = active else {
                active = Some((start, end));
                continue;
            };
            if start > active_end {
                union = union.saturating_add((active_end - active_start).max(0.0) as u64);
                active = Some((start, end));
            } else if end > active_end {
                active = Some((active_start, end));
            }
        }
        if let Some((active_start, active_end)) = active {
            union = union.saturating_add((active_end - active_start).max(0.0) as u64);
        }
        projected_union = projected_union.saturating_add(union);
    }
    if projected_union == 0 {
        return Err(degradation::QUALITY_GATE_INSUFFICIENT_EVIDENCE);
    }
    Ok(EffectivePixelCoverage {
        output_nontransparent,
        projected_union,
        ratio: output_nontransparent as f64 / projected_union as f64,
    })
}
