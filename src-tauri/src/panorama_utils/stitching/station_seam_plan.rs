//! Station seam planning between neighbouring Virtual_Tiles.
//!
//! Builds one seam plan per accepted station pair from final-world samples and
//! the shared seam cost, and locates the current owner transitions it replaces.

use super::super::coordinates::Projection;
use super::super::render_geometry::transformed_image_region;
use super::super::stack_pipeline::compositor::seam_candidate_cost_with_alignment;
use super::super::stack_pipeline::degradation;
use super::super::stack_pipeline::residual_warp;
#[cfg(test)]
use super::super::stack_pipeline::tone;
use crate::panorama_utils::shared::ImageInfo;
use nalgebra::Matrix3;
use std::collections::HashMap;

#[cfg(test)]
use super::write_station_seam_diagnostics;
use super::{
    FocusTilePayload, NO_OWNER, SeamOrientation, StationSeamPlan,
    projected_coverage_boundary_distance, seam_candidate_relation_for_pair, source_to_world_point,
    station_seam_edge_misalignment, station_seam_edge_strength_weight,
    station_seam_foreground_penalty, station_seam_local_tone, station_seam_pair_sample,
    station_seam_world_gradient,
};

/// Solve one deterministic minimum-cost seam for each accepted station pair.
/// This is intentionally an analysis-grid operation; applying the plan later
/// resamples only pixels that actually change owner.
pub(super) fn build_station_seam_plans<F, T>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    accepted_relations: &[(usize, usize)],
    offset_x: f64,
    offset_y: f64,
    out_width: u32,
    out_height: u32,
    residual: &residual_warp::ResidualWarp,
    current_owners: &[u16],
    load_image: &mut F,
) -> Result<Vec<StationSeamPlan>, String>
where
    F: FnMut(&ImageInfo) -> Result<T, String>,
    T: Into<FocusTilePayload>,
{
    const MAX_SEAM_GRID_DIMENSION: u32 = 768;
    const NARROW_OVERLAP_PX: u32 = 32;
    let edge_strength_weight = station_seam_edge_strength_weight();
    let mut relation_ids = accepted_relations
        .iter()
        .filter_map(|&(left, right)| (left != right).then_some((left.min(right), left.max(right))))
        .collect::<Vec<_>>();
    relation_ids.sort_unstable();
    relation_ids.dedup();

    let mut plans = Vec::new();
    for (left_id, right_id) in relation_ids {
        let Some(left_index) = images.iter().position(|image| image.id == left_id) else {
            continue;
        };
        let Some(right_index) = images.iter().position(|image| image.id == right_id) else {
            continue;
        };
        let left = images[left_index];
        let right = images[right_index];
        let Some(left_h) = global_homographies.get(&left.id) else {
            continue;
        };
        let Some(right_h) = global_homographies.get(&right.id) else {
            continue;
        };
        let Some(left_bounds) = transformed_image_region(
            left, left_h, projection, offset_x, offset_y, out_width, out_height,
        ) else {
            continue;
        };
        let Some(right_bounds) = transformed_image_region(
            right, right_h, projection, offset_x, offset_y, out_width, out_height,
        ) else {
            continue;
        };
        let min_x = left_bounds.0.max(right_bounds.0);
        let max_x = left_bounds.1.min(right_bounds.1);
        let min_y = left_bounds.2.max(right_bounds.2);
        let max_y = left_bounds.3.min(right_bounds.3);
        if min_x > max_x || min_y > max_y {
            continue;
        }
        let overlap_width = max_x - min_x + 1;
        let overlap_height = max_y - min_y + 1;
        let orientation = if overlap_width <= overlap_height {
            SeamOrientation::Vertical
        } else {
            SeamOrientation::Horizontal
        };
        let (along_min, along_max, cross_min, cross_max, cross_span) = match orientation {
            SeamOrientation::Vertical => (min_y, max_y, min_x, max_x, overlap_width),
            SeamOrientation::Horizontal => (min_x, max_x, min_y, max_y, overlap_height),
        };

        let left_pixels = load_image(left)?.into().image;
        let right_pixels = load_image(right)?.into().image;
        let left_inverse = left_h.try_inverse().ok_or_else(|| {
            format!(
                "Station seam pose for '{}' is not invertible",
                left.filename
            )
        })?;
        let right_inverse = right_h.try_inverse().ok_or_else(|| {
            format!(
                "Station seam pose for '{}' is not invertible",
                right.filename
            )
        })?;

        let Some(first_center) = source_to_world_point(
            left,
            left_h,
            projection,
            left.width as f64 * 0.5,
            left.height as f64 * 0.5,
        ) else {
            continue;
        };
        let Some(second_center) = source_to_world_point(
            right,
            right_h,
            projection,
            right.width as f64 * 0.5,
            right.height as f64 * 0.5,
        ) else {
            continue;
        };
        let first_on_low_side = match orientation {
            SeamOrientation::Vertical => first_center.x <= second_center.x,
            SeamOrientation::Horizontal => first_center.y <= second_center.y,
        };
        let owner_pair = (left_index + 1, right_index + 1);
        let local_relation = seam_candidate_relation_for_pair(left.id, right.id);
        let local_tone = local_relation.as_ref().and_then(|relation| {
            station_seam_local_tone(
                left,
                &left_pixels,
                &left_inverse,
                right,
                &right_pixels,
                &right_inverse,
                relation,
                projection,
                residual,
                min_x,
                max_x,
                min_y,
                max_y,
                offset_x,
                offset_y,
            )
        });

        if cross_span < NARROW_OVERLAP_PX {
            let center = cross_min as f64 + (cross_span.saturating_sub(1) as f64) * 0.5;
            degradation::record_run_degradation(
                degradation::COMPOSITION_NARROW_OVERLAP,
                serde_json::json!({
                    "stage": "layered_tile_compositor",
                    "left": left.id,
                    "right": right.id,
                    "overlap_width_px": overlap_width,
                    "overlap_height_px": overlap_height,
                    "seam": "coverage_centerline",
                }),
            );
            plans.push(StationSeamPlan {
                first_owner: owner_pair.0 as u16,
                second_owner: owner_pair.1 as u16,
                min_x,
                max_x,
                min_y,
                max_y,
                along_min,
                along_max,
                orientation,
                cross_samples: vec![center, center],
                current_cross_samples: Vec::new(),
                first_on_low_side,
                local_relation,
                local_tone,
            });
            continue;
        }

        let step = overlap_width
            .max(overlap_height)
            .div_ceil(MAX_SEAM_GRID_DIMENSION)
            .max(1);
        let along_count = (along_max - along_min).div_ceil(step) as usize + 1;
        let cross_count = cross_span.div_ceil(step) as usize + 1;
        let coordinate = |minimum: u32, maximum: u32, index: usize| {
            minimum.saturating_add(index as u32 * step).min(maximum)
        };
        // The existing hard station boundary is the locality prior for this
        // refinement.  Without a distance regularizer, a low-colour paper
        // strip at the far side of a multi-thousand-pixel overlap can replace
        // millions of otherwise valid owner pixels merely to avoid one local
        // stroke. A missing transition leaves the row unrestricted because
        // there is no current boundary to preserve there.
        let mut current_crosses = Vec::with_capacity(along_count);
        // A long overlap may contain more than one transition between the
        // same two virtual-tile owners. Keep every candidate so the final
        // full-resolution seam can anchor to the transition nearest its
        // planned path. Following only the previous row can drift onto a
        // separate owner island and then trigger the 32px locality guard.
        let mut current_transition_candidates = Vec::with_capacity(along_count);
        let mut previous_cross = None::<u32>;
        for along_index in 0..along_count {
            let along = coordinate(along_min, along_max, along_index);
            let mut previous_owner = None;
            let mut transitions = Vec::new();
            for cross in cross_min..=cross_max {
                let (x, y) = match orientation {
                    SeamOrientation::Vertical => (cross, along),
                    SeamOrientation::Horizontal => (along, cross),
                };
                let owner = current_owners
                    .get(y as usize * out_width as usize + x as usize)
                    .copied()
                    .unwrap_or(NO_OWNER);
                if let Some(prior) = previous_owner
                    && owner != prior
                    && [owner, prior].contains(&(owner_pair.0 as u16))
                    && [owner, prior].contains(&(owner_pair.1 as u16))
                {
                    transitions.push(cross);
                }
                previous_owner = Some(owner);
            }
            let target = previous_cross.unwrap_or(cross_min + (cross_max - cross_min) / 2);
            let selected = transitions
                .iter()
                .copied()
                .min_by_key(|cross| cross.abs_diff(target));
            if let Some(cross) = selected {
                previous_cross = Some(cross);
            }
            current_transition_candidates.push(transitions);
            current_crosses.push(selected);
        }
        // Evaluate the same content/coverage objective at both the analysis
        // grid and the final-resolution local refinement below.  The coarse
        // grid keeps the global solve bounded, while the full-resolution
        // pass prevents a narrow painted edge from falling between two grid
        // columns (the station 7/10 failure was exactly this case).
        let candidate_cost = |along: u32, cross: u32, current_seam: Option<u32>| {
            // A locally fitted relation is only admitted for an explicitly
            // measured closure defect. Its old owner boundary is still a
            // useful prior, but a 16px penalty makes the optimizer cling to
            // a boundary that cuts through a single long brush stroke. Give
            // the local solve room to route around that stroke while keeping
            // the application-side evidence relocation cap intact.
            // A rejected closure pair may carry a real foreground displacement
            // across several hundred output pixels.  Keeping the local prior
            // at the ordinary 64px scale makes the dynamic path pay more for
            // leaving the old owner boundary than for cutting through that
            // displaced contour.  The application pass still limits the
            // rewrite to the explicitly measured local band below; this only
            // lets the path reach that evidence-backed band.
            let current_seam_distance_scale = if local_relation.is_some() {
                256.0
            } else {
                16.0
            };
            let current_seam_penalty = current_seam.map_or(0.0, |current| {
                f64::from(current.abs_diff(cross)) / current_seam_distance_scale
            });
            let (x, y) = match orientation {
                SeamOrientation::Vertical => (cross, along),
                SeamOrientation::Horizontal => (along, cross),
            };
            let Some(((left_pixel, left_source), (right_pixel, right_source))) =
                station_seam_pair_sample(
                    left,
                    &left_pixels,
                    &left_inverse,
                    right,
                    &right_pixels,
                    &right_inverse,
                    local_relation.as_ref(),
                    projection,
                    residual,
                    x,
                    y,
                    offset_x,
                    offset_y,
                )
            else {
                return None;
            };
            let disagreement = left_pixel
                .0
                .iter()
                .zip(right_pixel.0)
                .map(|(left, right)| f64::from((*left - right).abs()).powi(2))
                .sum::<f64>()
                .sqrt();
            let left_boundary =
                projected_coverage_boundary_distance(left, left_h, projection, left_source)
                    .unwrap_or(0.0);
            let right_boundary =
                projected_coverage_boundary_distance(right, right_h, projection, right_source)
                    .unwrap_or(0.0);
            let (edge_alignment, edge_strength) = match (
                station_seam_world_gradient(
                    left,
                    &left_pixels,
                    &left_inverse,
                    projection,
                    residual,
                    x,
                    y,
                    offset_x,
                    offset_y,
                ),
                station_seam_world_gradient(
                    right,
                    &right_pixels,
                    &right_inverse,
                    projection,
                    residual,
                    x,
                    y,
                    offset_x,
                    offset_y,
                ),
            ) {
                (Some(left_gradient), Some(right_gradient)) => (
                    station_seam_edge_misalignment(left_gradient, right_gradient),
                    left_gradient
                        .0
                        .hypot(left_gradient.1)
                        .max(right_gradient.0.hypot(right_gradient.1)),
                ),
                _ => (1.0, 0.0),
            };
            // Prefer blank/aligned overlap over a stroke crossing.  A local
            // seam-only relation is used precisely for a visible closure
            // defect, so cutting a strong painted edge is more damaging than
            // a small paper-colour difference: it splits one physical stroke
            // between two independently rendered owners and creates the
            // double contour visible in the native crops. Keep the ordinary
            // accepted-pair weight unchanged; only the explicitly measured
            // local relation gets the stronger stroke-avoidance prior.
            let stroke_strength_weight = if local_relation.is_some() {
                edge_strength_weight.max(12.0)
            } else {
                edge_strength_weight
            };
            let stroke_cost = edge_alignment
                + stroke_strength_weight * edge_strength.min(1.0)
                + station_seam_foreground_penalty(
                    local_relation.as_ref(),
                    left,
                    left_source,
                    &left_pixel,
                    right,
                    right_source,
                    &right_pixel,
                    edge_strength,
                )
                + current_seam_penalty;
            Some(seam_candidate_cost_with_alignment(
                disagreement,
                left_boundary.min(right_boundary),
                stroke_cost,
            ))
        };
        let mut costs = vec![f64::INFINITY; along_count * cross_count];
        for along_index in 0..along_count {
            let along = coordinate(along_min, along_max, along_index);
            for cross_index in 0..cross_count {
                let cross = coordinate(cross_min, cross_max, cross_index);
                costs[along_index * cross_count + cross_index] =
                    candidate_cost(along, cross, current_crosses[along_index])
                        .unwrap_or(f64::INFINITY);
            }
        }
        let mut best_run = None::<(usize, usize)>;
        let mut run_start = None;
        for along_index in 0..=along_count {
            let finite = along_index < along_count
                && costs[along_index * cross_count..(along_index + 1) * cross_count]
                    .iter()
                    .any(|cost| cost.is_finite());
            match (run_start, finite) {
                (None, true) => run_start = Some(along_index),
                (Some(start), false) => {
                    let end = along_index - 1;
                    if best_run
                        .is_none_or(|(best_start, best_end)| end - start > best_end - best_start)
                    {
                        best_run = Some((start, end));
                    }
                    run_start = None;
                }
                _ => {}
            }
        }
        let Some((finite_start, finite_end)) = best_run else {
            continue;
        };
        let finite_along_count = finite_end - finite_start + 1;
        let finite_costs = &costs[finite_start * cross_count..(finite_end + 1) * cross_count];
        let Some((path, total_cost)) =
            crate::panorama_utils::stack_pipeline::compositor::minimum_vertical_seam(
                finite_costs,
                cross_count,
                finite_along_count,
            )
        else {
            #[cfg(test)]
            {
                let center = f64::from(cross_min + (cross_max - cross_min) / 2);
                let diagnostic_plan = StationSeamPlan {
                    first_owner: owner_pair.0 as u16,
                    second_owner: owner_pair.1 as u16,
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    along_min,
                    along_max,
                    orientation,
                    cross_samples: vec![center, center],
                    current_cross_samples: Vec::new(),
                    first_on_low_side,
                    local_relation: local_relation.clone(),
                    local_tone: local_tone.clone(),
                };
                let topology = tone::run_topology_snapshot().unwrap_or_default();
                write_station_seam_diagnostics(
                    &diagnostic_plan,
                    topology.station_of(left.id).unwrap_or(left_index),
                    topology.station_of(right.id).unwrap_or(right_index),
                    left,
                    right,
                    &left_pixels,
                    &right_pixels,
                    left_h,
                    right_h,
                    &left_inverse,
                    &right_inverse,
                    projection,
                    offset_x,
                    offset_y,
                    residual,
                    current_owners,
                    out_width,
                    "invalid_cost_grid",
                )?;
            }
            continue;
        };
        let path_is_valid = total_cost.is_finite();
        let coarse_cross_samples = path
            .into_iter()
            .map(|cross| coordinate(cross_min, cross_max, cross) as f64)
            .collect::<Vec<_>>();
        let finite_along_min = coordinate(along_min, along_max, finite_start);
        let finite_along_max = coordinate(along_min, along_max, finite_end);
        // Refine every along-axis pixel around the coarse path.  Searching a
        // bounded two-cell neighbourhood keeps the existing locality prior
        // and seam objective intact, but restores pixel-level placement for a
        // stroke that is narrower than the analysis-grid step.
        let mut cross_samples = Vec::new();
        let mut current_cross_samples = Vec::new();
        if path_is_valid {
            let refinement_radius = step.saturating_mul(2).max(2);
            let sample_count = finite_along_max.saturating_sub(finite_along_min) as usize + 1;
            cross_samples.reserve(sample_count);
            current_cross_samples.reserve(sample_count);
            for along in finite_along_min..=finite_along_max {
                let coarse_cross = station_seam_series_at(
                    &coarse_cross_samples,
                    finite_along_min,
                    finite_along_max,
                    along,
                )
                .round()
                .clamp(f64::from(cross_min), f64::from(cross_max))
                    as u32;
                let current_index = ((along.saturating_sub(along_min)) / step)
                    .min(along_count.saturating_sub(1) as u32)
                    as usize;
                let current_seam = current_transition_candidates
                    .get(current_index)
                    .and_then(|candidates| {
                        candidates
                            .iter()
                            .min_by_key(|cross| cross.abs_diff(coarse_cross))
                            .copied()
                    })
                    .or_else(|| current_crosses.get(current_index).copied().flatten());
                let search_min = coarse_cross
                    .saturating_sub(refinement_radius)
                    .max(cross_min);
                let search_max = coarse_cross
                    .saturating_add(refinement_radius)
                    .min(cross_max);
                // Preserve the one-cell-per-row continuity contract of the
                // coarse dynamic-programming path. A full-resolution search
                // may detour around a narrow stroke, but it cannot create a
                // two-pixel endpoint spike or a new hard ownership island.
                let neighbour_crosses = [
                    along.checked_sub(1).map(|value| {
                        station_seam_series_at(
                            &coarse_cross_samples,
                            finite_along_min,
                            finite_along_max,
                            value,
                        )
                        .round() as u32
                    }),
                    along.checked_add(1).map(|value| {
                        station_seam_series_at(
                            &coarse_cross_samples,
                            finite_along_min,
                            finite_along_max,
                            value.min(finite_along_max),
                        )
                        .round() as u32
                    }),
                ];
                let neighbour_min = neighbour_crosses
                    .iter()
                    .flatten()
                    .copied()
                    .min()
                    .unwrap_or(coarse_cross)
                    .saturating_sub(1)
                    .max(cross_min);
                let neighbour_max = neighbour_crosses
                    .iter()
                    .flatten()
                    .copied()
                    .max()
                    .unwrap_or(coarse_cross)
                    .saturating_add(1)
                    .min(cross_max);
                let search_min = search_min.max(neighbour_min);
                let search_max = search_max.min(neighbour_max);
                // Keep the coarse solution as the reference candidate.  The
                // refinement is allowed to improve a sampled location, but
                // it must not replace a valid stroke-avoiding coarse choice
                // merely because one full-resolution pixel has a marginally
                // lower colour cost.
                let mut best = (
                    coarse_cross,
                    candidate_cost(along, coarse_cross, current_seam).unwrap_or(f64::INFINITY),
                );
                for cross in search_min..=search_max {
                    let Some(cost) = candidate_cost(along, cross, current_seam) else {
                        continue;
                    };
                    if cost < best.1
                        || (cost.total_cmp(&best.1).is_eq()
                            && cross.abs_diff(coarse_cross) < best.0.abs_diff(coarse_cross))
                    {
                        best = (cross, cost);
                    }
                }
                cross_samples.push(f64::from(best.0));
                current_cross_samples.push(
                    current_seam
                        .map(f64::from)
                        .unwrap_or_else(|| f64::from(best.0)),
                );
            }
        } else {
            cross_samples = coarse_cross_samples;
            current_cross_samples = (finite_start..=finite_end)
                .enumerate()
                .map(|(offset, index)| {
                    current_crosses[index]
                        .map(f64::from)
                        .unwrap_or(cross_samples[offset])
                })
                .collect::<Vec<_>>();
        }
        let plan = StationSeamPlan {
            first_owner: owner_pair.0 as u16,
            second_owner: owner_pair.1 as u16,
            min_x,
            max_x,
            min_y,
            max_y,
            along_min: finite_along_min,
            along_max: finite_along_max,
            orientation,
            cross_samples,
            current_cross_samples,
            first_on_low_side,
            local_relation,
            local_tone,
        };
        #[cfg(test)]
        {
            let topology = tone::run_topology_snapshot().unwrap_or_default();
            write_station_seam_diagnostics(
                &plan,
                topology.station_of(left.id).unwrap_or(left_index),
                topology.station_of(right.id).unwrap_or(right_index),
                left,
                right,
                &left_pixels,
                &right_pixels,
                left_h,
                right_h,
                &left_inverse,
                &right_inverse,
                projection,
                offset_x,
                offset_y,
                residual,
                current_owners,
                out_width,
                if path_is_valid {
                    "minimum_cost_path"
                } else {
                    "no_finite_full_path"
                },
            )?;
        }
        if !path_is_valid {
            continue;
        }
        plans.push(plan);
    }
    Ok(plans)
}

pub(super) fn station_seam_series_at(
    samples: &[f64],
    along_min: u32,
    along_max: u32,
    along: u32,
) -> f64 {
    if samples.len() < 2 || along_max <= along_min {
        return samples.first().copied().unwrap_or_default();
    }
    let position = ((along.saturating_sub(along_min)) as f64 / (along_max - along_min) as f64
        * (samples.len() - 1) as f64)
        .clamp(0.0, (samples.len() - 1) as f64);
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(samples.len() - 1);
    let fraction = position - lower as f64;
    samples[lower] * (1.0 - fraction) + samples[upper] * fraction
}

pub(super) fn station_seam_cross_at(plan: &StationSeamPlan, along: u32) -> f64 {
    station_seam_series_at(&plan.cross_samples, plan.along_min, plan.along_max, along)
}

pub(super) fn station_current_cross_at(plan: &StationSeamPlan, along: u32) -> f64 {
    if plan.current_cross_samples.is_empty() {
        return station_seam_cross_at(plan, along);
    }
    station_seam_series_at(
        &plan.current_cross_samples,
        plan.along_min,
        plan.along_max,
        along,
    )
}

/// Read the owner transition from the full-resolution ledger for one overlap
/// row/column. A coarse seam row can contain several disconnected islands of
/// the same two owners, so the sampled transition is not always the boundary
/// that the final owner map actually contains. Select the real transition
/// nearest the planned seam while retaining the existing locality prior.
pub(super) fn station_current_transition_candidates(
    plan: &StationSeamPlan,
    source_owner_ids: &[u16],
    out_width: u32,
    along: u32,
) -> Vec<u32> {
    let (cross_min, cross_max) = match plan.orientation {
        SeamOrientation::Vertical => (plan.min_x, plan.max_x),
        SeamOrientation::Horizontal => (plan.min_y, plan.max_y),
    };
    let mut previous = None;
    let mut transitions = Vec::new();
    for cross in cross_min..=cross_max {
        let (x, y) = match plan.orientation {
            SeamOrientation::Vertical => (cross, along),
            SeamOrientation::Horizontal => (along, cross),
        };
        let index = y as usize * out_width as usize + x as usize;
        let owner = source_owner_ids.get(index).copied().unwrap_or(NO_OWNER);
        if let Some(prior) = previous
            && owner != prior
            && [owner, prior].contains(&plan.first_owner)
            && [owner, prior].contains(&plan.second_owner)
        {
            transitions.push(cross);
        }
        previous = Some(owner);
    }
    transitions
}

pub(super) fn station_transition_first_on_low_side(
    plan: &StationSeamPlan,
    source_owner_ids: &[u16],
    out_width: u32,
    along: u32,
    transition: u32,
) -> Option<bool> {
    let (cross_min, cross_max) = match plan.orientation {
        SeamOrientation::Vertical => (plan.min_x, plan.max_x),
        SeamOrientation::Horizontal => (plan.min_y, plan.max_y),
    };
    if transition <= cross_min || transition > cross_max {
        return None;
    }
    let owner_at = |cross: u32| {
        let (x, y) = match plan.orientation {
            SeamOrientation::Vertical => (cross, along),
            SeamOrientation::Horizontal => (along, cross),
        };
        source_owner_ids
            .get(y as usize * out_width as usize + x as usize)
            .copied()
            .unwrap_or(NO_OWNER)
    };
    match (owner_at(transition - 1), owner_at(transition)) {
        (first, second) if first == plan.first_owner && second == plan.second_owner => Some(true),
        (first, second) if first == plan.second_owner && second == plan.first_owner => Some(false),
        _ => None,
    }
}

/// Infer the side occupied by the second owner when the full-resolution owner
/// ledger has no direct first/second transition on this scanline.  A third
/// station can sit between the pair and hide that transition, but its pixels
/// do not erase the second owner's local evidence.  Only accept a clear
/// majority in the bounded relation band; an ambiguous island must remain
/// untouched rather than becoming a reason to rewrite both sides.
pub(super) fn station_current_second_on_low_side(
    plan: &StationSeamPlan,
    source_owner_ids: &[u16],
    out_width: u32,
    along: u32,
    target: f64,
    band_px: f64,
) -> Option<bool> {
    if !band_px.is_finite() || band_px <= 0.0 {
        return None;
    }
    let (cross_min, cross_max) = match plan.orientation {
        SeamOrientation::Vertical => (plan.min_x, plan.max_x),
        SeamOrientation::Horizontal => (plan.min_y, plan.max_y),
    };
    let radius = band_px.ceil().min(u32::MAX as f64) as u32;
    let target_cross = target.round().clamp(0.0, u32::MAX as f64) as u32;
    let scan_min = target_cross.saturating_sub(radius).max(cross_min);
    let scan_max = target_cross.saturating_add(radius).min(cross_max);
    let mut low_count = 0usize;
    let mut high_count = 0usize;
    for cross in scan_min..=scan_max {
        let (x, y) = match plan.orientation {
            SeamOrientation::Vertical => (cross, along),
            SeamOrientation::Horizontal => (along, cross),
        };
        let index = y as usize * out_width as usize + x as usize;
        if source_owner_ids.get(index).copied() != Some(plan.second_owner) {
            continue;
        }
        if f64::from(cross) <= target {
            low_count += 1;
        } else {
            high_count += 1;
        }
    }
    let total = low_count + high_count;
    if total < 16 {
        return None;
    }
    let dominant = low_count.max(high_count);
    if dominant as f64 / (total as f64) < 0.65 {
        return None;
    }
    Some(low_count > high_count)
}
