//! Test-only closed-stroke correspondence between station pairs.
//!
//! Research diagnostics that pair stroke features across a seam and estimate a
//! holdout translation; production seams never read them.

use super::super::coordinates::Projection;
#[cfg(test)]
use super::super::stack_pipeline::residual_warp;
use crate::panorama_utils::shared::ImageInfo;
use image::Rgb32FImage;
use nalgebra::Matrix3;

use super::{
    SeamOrientation, StationSeamPlan, station_current_cross_at, station_seam_luma,
    station_seam_sample, station_seam_world_gradient,
};

#[cfg(test)]
pub(super) const STROKE_MATCH_CONTRAST_MIN: f64 = 0.15;

#[cfg(test)]
pub(super) const STROKE_MATCH_SEARCH_RADIUS_PX: i32 = 32;

#[cfg(test)]
pub(super) const STROKE_MATCH_SEAM_RADIUS_PX: i32 = 48;

#[cfg(test)]
pub(super) const STROKE_MATCH_ORIENTATION_MAX_DEGREES: f64 = 10.0;

#[cfg(test)]
pub(super) const STROKE_MATCH_CURVATURE_MAX: f64 = 0.20;

#[cfg(test)]
pub(super) const STROKE_MATCH_PATCH_RMS_MAX: f64 = 0.10;

#[cfg(test)]
pub(super) const STROKE_MATCH_UNIQUENESS_RATIO_MAX: f64 = 0.80;

#[cfg(test)]
pub(super) const STROKE_MATCH_UNIQUENESS_MARGIN_MIN: f64 = 0.02;

#[cfg(test)]
pub(super) const STROKE_MATCH_CLOSURE_MAX_PX: f64 = 1.0;

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) struct StationStrokeView<'a> {
    pub(super) image: &'a ImageInfo,
    pub(super) pixels: &'a Rgb32FImage,
    pub(super) inverse: &'a Matrix3<f64>,
    pub(super) projection: Projection,
    pub(super) offset_x: f64,
    pub(super) offset_y: f64,
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) struct StationStrokeFeature {
    pub(super) point: (u32, u32),
    pub(super) gradient: (f64, f64),
    pub(super) strength: f64,
    pub(super) curvature: [f64; 2],
    pub(super) descriptor: Vec<f64>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ClosedStrokeMatch {
    pub(super) source: (u32, u32),
    pub(super) target: (u32, u32),
    pub(super) displacement: [f64; 2],
    pub(super) score: f64,
    pub(super) reverse_score: f64,
    pub(super) closure_px: f64,
    pub(super) source_strength: f64,
    pub(super) target_strength: f64,
}

#[cfg(test)]
#[derive(Debug, Default)]
pub(super) struct StrokeMatchDiagnostic {
    pub(super) matches: Vec<ClosedStrokeMatch>,
    pub(super) attrition: std::collections::BTreeMap<&'static str, usize>,
}

#[cfg(test)]
pub(super) fn stroke_increment(
    attrition: &mut std::collections::BTreeMap<&'static str, usize>,
    reason: &'static str,
) {
    *attrition.entry(reason).or_default() += 1;
}

#[cfg(test)]
pub(super) fn stroke_point(
    base: (u32, u32),
    direction: (f64, f64),
    distance: f64,
) -> Option<(u32, u32)> {
    let x = f64::from(base.0) + direction.0 * distance;
    let y = f64::from(base.1) + direction.1 * distance;
    (x >= 1.0 && y >= 1.0 && x <= f64::from(u32::MAX) && y <= f64::from(u32::MAX))
        .then_some((x.round() as u32, y.round() as u32))
}

#[cfg(test)]
pub(super) fn stroke_luma(view: StationStrokeView<'_>, point: (u32, u32)) -> Option<f64> {
    station_seam_sample(
        view.image,
        view.pixels,
        view.inverse,
        view.projection,
        &residual_warp::ResidualWarp::new(),
        point.0,
        point.1,
        view.offset_x,
        view.offset_y,
    )
    .map(|sample| station_seam_luma(sample.0))
}

#[cfg(test)]
pub(super) fn stroke_gradient(
    view: StationStrokeView<'_>,
    point: (u32, u32),
) -> Option<(f64, f64)> {
    station_seam_world_gradient(
        view.image,
        view.pixels,
        view.inverse,
        view.projection,
        &residual_warp::ResidualWarp::new(),
        point.0,
        point.1,
        view.offset_x,
        view.offset_y,
    )
}

#[cfg(test)]
pub(super) fn stroke_is_local_peak(
    view: StationStrokeView<'_>,
    point: (u32, u32),
    gradient: (f64, f64),
) -> bool {
    let strength = gradient.0.hypot(gradient.1);
    if strength <= 0.0 {
        return false;
    }
    let normal = (gradient.0 / strength, gradient.1 / strength);
    for direction in [-1.0, 1.0] {
        let Some(neighbour) = stroke_point(point, normal, direction) else {
            continue;
        };
        let Some(neighbour_gradient) = stroke_gradient(view, neighbour) else {
            continue;
        };
        if neighbour_gradient.0.hypot(neighbour_gradient.1) > strength {
            return false;
        }
    }
    true
}

#[cfg(test)]
pub(super) fn stroke_feature(
    view: StationStrokeView<'_>,
    point: (u32, u32),
) -> Option<StationStrokeFeature> {
    let gradient = stroke_gradient(view, point)?;
    let strength = gradient.0.hypot(gradient.1);
    if strength < STROKE_MATCH_CONTRAST_MIN || !stroke_is_local_peak(view, point, gradient) {
        return None;
    }
    let normal = (gradient.0 / strength, gradient.1 / strength);
    let tangent = (-normal.1, normal.0);
    let mut descriptor = Vec::with_capacity(16);
    for scale in [2.0, 4.0, 8.0, 16.0] {
        let normal_low = stroke_luma(view, stroke_point(point, normal, -scale)?)?;
        let normal_high = stroke_luma(view, stroke_point(point, normal, scale)?)?;
        let tangent_low = stroke_luma(view, stroke_point(point, tangent, -scale)?)?;
        let tangent_high = stroke_luma(view, stroke_point(point, tangent, scale)?)?;
        descriptor.extend([
            normal_high - normal_low,
            tangent_high - tangent_low,
            (normal_high + normal_low) * 0.5,
            (tangent_high + tangent_low) * 0.5,
        ]);
    }
    let oriented = |distance: f64| {
        let sample = stroke_gradient(view, stroke_point(point, tangent, distance)?)?;
        let magnitude = sample.0.hypot(sample.1);
        (magnitude >= STROKE_MATCH_CONTRAST_MIN * 0.5)
            .then_some((normal.0 * sample.1 - normal.1 * sample.0) / magnitude)
    };
    let curvature = [oriented(-8.0).unwrap_or(0.0), oriented(8.0).unwrap_or(0.0)];
    Some(StationStrokeFeature {
        point,
        gradient,
        strength,
        curvature,
        descriptor,
    })
}

#[cfg(test)]
pub(super) fn stroke_feature_score(
    source: &StationStrokeFeature,
    target: &StationStrokeFeature,
) -> Result<f64, &'static str> {
    let cosine = (source.gradient.0 * target.gradient.0 + source.gradient.1 * target.gradient.1)
        / (source.strength * target.strength);
    if cosine < STROKE_MATCH_ORIENTATION_MAX_DEGREES.to_radians().cos() {
        return Err("orientation_or_polarity");
    }
    if source
        .curvature
        .iter()
        .zip(target.curvature)
        .any(|(left, right)| (left - right).abs() > STROKE_MATCH_CURVATURE_MAX)
    {
        return Err("curvature");
    }
    let patch_rms = (source
        .descriptor
        .iter()
        .zip(&target.descriptor)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        / source.descriptor.len().max(1) as f64)
        .sqrt();
    if patch_rms > STROKE_MATCH_PATCH_RMS_MAX {
        return Err("patch");
    }
    Ok(patch_rms
        + (1.0 - cosine)
        + source
            .curvature
            .iter()
            .zip(target.curvature)
            .map(|(left, right)| (left - right).abs())
            .sum::<f64>()
            * 0.25)
}

#[cfg(test)]
pub(super) fn stroke_nearest_unique(
    source: &StationStrokeFeature,
    target_view: StationStrokeView<'_>,
) -> Result<(StationStrokeFeature, f64), &'static str> {
    let mut candidates = Vec::new();
    let mut saw_orientation = false;
    let mut saw_curvature = false;
    let mut saw_patch = false;
    for dy in -STROKE_MATCH_SEARCH_RADIUS_PX..=STROKE_MATCH_SEARCH_RADIUS_PX {
        for dx in -STROKE_MATCH_SEARCH_RADIUS_PX..=STROKE_MATCH_SEARCH_RADIUS_PX {
            let Some(x) = source.point.0.checked_add_signed(dx) else {
                continue;
            };
            let Some(y) = source.point.1.checked_add_signed(dy) else {
                continue;
            };
            let Some(target) = stroke_feature(target_view, (x, y)) else {
                continue;
            };
            match stroke_feature_score(source, &target) {
                Ok(score) => candidates.push((score, target)),
                Err("orientation_or_polarity") => saw_orientation = true,
                Err("curvature") => saw_curvature = true,
                Err("patch") => saw_patch = true,
                Err(_) => {}
            }
        }
    }
    candidates.sort_by(|left, right| {
        left.0
            .total_cmp(&right.0)
            .then_with(|| left.1.point.1.cmp(&right.1.point.1))
            .then_with(|| left.1.point.0.cmp(&right.1.point.0))
    });
    let Some((best_score, best)) = candidates.first().cloned() else {
        return Err(if saw_patch {
            "patch"
        } else if saw_curvature {
            "curvature"
        } else if saw_orientation {
            "orientation_or_polarity"
        } else {
            "no_target_edge"
        });
    };
    if let Some((second_score, _)) = candidates.iter().find(|(_, candidate)| {
        let normal = (
            source.gradient.0 / source.strength,
            source.gradient.1 / source.strength,
        );
        let dx = f64::from(candidate.point.0) - f64::from(best.point.0);
        let dy = f64::from(candidate.point.1) - f64::from(best.point.1);
        (dx * normal.0 + dy * normal.1).abs() >= 3.0
    }) && !(best_score <= second_score * STROKE_MATCH_UNIQUENESS_RATIO_MAX
        && best_score + STROKE_MATCH_UNIQUENESS_MARGIN_MIN <= *second_score)
    {
        return Err("not_unique");
    }
    Ok((best, best_score))
}

#[cfg(test)]
pub(super) fn closed_station_stroke_match(
    source: StationStrokeFeature,
    source_view: StationStrokeView<'_>,
    target_view: StationStrokeView<'_>,
) -> Result<ClosedStrokeMatch, &'static str> {
    let (target, score) = stroke_nearest_unique(&source, target_view)?;
    let (reverse, reverse_score) =
        stroke_nearest_unique(&target, source_view).map_err(|_| "reverse_not_unique")?;
    let closure_px = (f64::from(reverse.point.0) - f64::from(source.point.0))
        .hypot(f64::from(reverse.point.1) - f64::from(source.point.1));
    if closure_px > STROKE_MATCH_CLOSURE_MAX_PX {
        return Err("closure");
    }
    Ok(ClosedStrokeMatch {
        source: source.point,
        target: target.point,
        displacement: [
            f64::from(target.point.0) - f64::from(source.point.0),
            f64::from(target.point.1) - f64::from(source.point.1),
        ],
        score,
        reverse_score,
        closure_px,
        source_strength: source.strength,
        target_strength: target.strength,
    })
}

#[cfg(test)]
pub(super) fn station_stroke_correspondences(
    plan: &StationSeamPlan,
    source_view: StationStrokeView<'_>,
    target_view: StationStrokeView<'_>,
) -> StrokeMatchDiagnostic {
    const ALONG_CELL_PX: u32 = 64;
    const CANDIDATES_PER_CELL: usize = 4;
    const CANDIDATE_DEDUP_PX: f64 = 12.0;
    let mut diagnostic = StrokeMatchDiagnostic::default();
    let mut accepted = std::collections::BTreeMap::<(u32, u32), ClosedStrokeMatch>::new();
    let mut along = plan.along_min;
    while along <= plan.along_max {
        let along_end = along.saturating_add(ALONG_CELL_PX - 1).min(plan.along_max);
        let middle = along + (along_end - along) / 2;
        let seam_cross = station_current_cross_at(plan, middle).round() as i32;
        let mut features = Vec::new();
        for along_point in along..=along_end {
            for cross in
                seam_cross - STROKE_MATCH_SEAM_RADIUS_PX..=seam_cross + STROKE_MATCH_SEAM_RADIUS_PX
            {
                if cross < 1 {
                    continue;
                }
                let point = match plan.orientation {
                    SeamOrientation::Vertical => (cross as u32, along_point),
                    SeamOrientation::Horizontal => (along_point, cross as u32),
                };
                if let Some(feature) = stroke_feature(source_view, point) {
                    features.push(feature);
                }
            }
        }
        features.sort_by(|left, right| {
            right
                .strength
                .total_cmp(&left.strength)
                .then_with(|| left.point.1.cmp(&right.point.1))
                .then_with(|| left.point.0.cmp(&right.point.0))
        });
        let mut seeds = Vec::<StationStrokeFeature>::new();
        for feature in features {
            if seeds.iter().any(|seed| {
                (f64::from(seed.point.0) - f64::from(feature.point.0))
                    .hypot(f64::from(seed.point.1) - f64::from(feature.point.1))
                    < CANDIDATE_DEDUP_PX
            }) {
                continue;
            }
            seeds.push(feature);
            if seeds.len() == CANDIDATES_PER_CELL {
                break;
            }
        }
        if seeds.is_empty() {
            stroke_increment(&mut diagnostic.attrition, "source_contrast_or_coverage");
        }
        for source in seeds {
            stroke_increment(&mut diagnostic.attrition, "source_candidates");
            match closed_station_stroke_match(source, source_view, target_view) {
                Ok(candidate) => {
                    let midpoint = (
                        (candidate.source.0 / 2 + candidate.target.0 / 2) / ALONG_CELL_PX,
                        (candidate.source.1 / 2 + candidate.target.1 / 2) / ALONG_CELL_PX,
                    );
                    if accepted
                        .get(&midpoint)
                        .is_none_or(|current| candidate.score < current.score)
                    {
                        accepted.insert(midpoint, candidate);
                    } else {
                        stroke_increment(&mut diagnostic.attrition, "spatial_dedup");
                    }
                }
                Err(reason) => stroke_increment(&mut diagnostic.attrition, reason),
            }
        }
        if along_end == plan.along_max {
            break;
        }
        along = along_end + 1;
    }
    diagnostic.matches = accepted.into_values().collect();
    diagnostic.matches.sort_by(|left, right| {
        left.source
            .1
            .cmp(&right.source.1)
            .then_with(|| left.source.0.cmp(&right.source.0))
    });
    stroke_increment(&mut diagnostic.attrition, "accepted");
    diagnostic
        .attrition
        .insert("accepted", diagnostic.matches.len());
    diagnostic
}

#[cfg(test)]
pub(super) fn stroke_translation_holdout(matches: &[ClosedStrokeMatch]) -> serde_json::Value {
    let fit = matches
        .iter()
        .filter(|item| ((item.source.0 / 128) + (item.source.1 / 128)) % 2 == 0)
        .collect::<Vec<_>>();
    let holdout = matches
        .iter()
        .filter(|item| ((item.source.0 / 128) + (item.source.1 / 128)) % 2 == 1)
        .collect::<Vec<_>>();
    if fit.len() < 4 || holdout.len() < 4 {
        return serde_json::json!({
            "model": "translation",
            "applied": false,
            "reason": "insufficient_fit_holdout",
            "fit_count": fit.len(),
            "holdout_count": holdout.len(),
        });
    }
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let translation = [
        median(&mut fit.iter().map(|item| item.displacement[0]).collect()),
        median(&mut fit.iter().map(|item| item.displacement[1]).collect()),
    ];
    let before = holdout
        .iter()
        .map(|item| item.displacement[0].hypot(item.displacement[1]))
        .collect::<Vec<_>>();
    let after = holdout
        .iter()
        .map(|item| {
            (item.displacement[0] - translation[0]).hypot(item.displacement[1] - translation[1])
        })
        .collect::<Vec<_>>();
    let distribution = |values: &[f64]| {
        if values.is_empty() {
            return f64::NAN;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[(sorted.len() * 95).div_ceil(100).saturating_sub(1)]
    };
    let before_p95 = distribution(&before);
    let after_p95 = distribution(&after);
    serde_json::json!({
        "model": "translation",
        "applied": false,
        "reason": if after_p95 < before_p95 { "diagnostic_only" } else { "held_out_p95_not_lower" },
        "fit_count": fit.len(),
        "holdout_count": holdout.len(),
        "translation": translation,
        "scale": 1.0,
        "held_out_p95_before": before_p95,
        "held_out_p95_after": after_p95,
    })
}
