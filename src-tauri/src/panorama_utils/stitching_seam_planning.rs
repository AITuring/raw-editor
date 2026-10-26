use super::*;

/// Solve one deterministic minimum-cost seam for each accepted station pair.
/// This is intentionally an analysis-grid operation; applying the plan later
/// resamples only pixels that actually change owner.
pub(crate) fn build_station_seam_plans<F>(
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
) -> Result<(Vec<StationSeamPlan>, StationSeamPlanningStats), String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    pub(crate) const MAX_SEAM_GRID_DIMENSION: u32 = 768;
    pub(crate) const NARROW_OVERLAP_PX: u32 = 32;
    pub(crate) const EDGE_STRENGTH_WEIGHT: f64 = 2.0;
    let transition_index =
        StationOwnerTransitionIndex::build(current_owners, out_width, out_height);
    let mut stats = StationSeamPlanningStats {
        index_owner_reads: transition_index.owner_reads,
        indexed_transitions: transition_index.indexed_transitions,
        ..StationSeamPlanningStats::default()
    };
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

        let left_pixels = load_image(left)?;
        stats.endpoint_tile_loads += 1;
        let right_pixels = load_image(right)?;
        stats.endpoint_tile_loads += 1;
        stats.max_resident_endpoint_tiles = stats.max_resident_endpoint_tiles.max(2);
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

        let Some(_) = source_to_world_point(
            left,
            left_h,
            projection,
            left.width as f64 * 0.5,
            left.height as f64 * 0.5,
        ) else {
            continue;
        };
        let Some(_) = source_to_world_point(
            right,
            right_h,
            projection,
            right.width as f64 * 0.5,
            right.height as f64 * 0.5,
        ) else {
            continue;
        };
        let owner_pair = (left_index + 1, right_index + 1);

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
            let mut plan = StationSeamPlan {
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
                replacement_samples: Vec::new(),
            };
            stats.relation_query_visits += materialize_station_seam_samples(
                &mut plan,
                &transition_index,
                current_owners,
                out_width,
                left,
                &left_pixels,
                &left_inverse,
                right,
                &right_pixels,
                &right_inverse,
                projection,
                residual,
                offset_x,
                offset_y,
            );
            stats.materialized_samples += plan.replacement_samples.len();
            plans.push(plan);
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
            let (indexed, visits) = transition_index.query(
                owner_pair.0 as u16,
                owner_pair.1 as u16,
                orientation,
                along,
                cross_min,
                cross_max,
            );
            stats.relation_query_visits += visits;
            let transitions = indexed
                .iter()
                .map(|transition| transition.cross)
                .collect::<Vec<_>>();
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
            let current_seam_penalty =
                current_seam.map_or(0.0, |current| f64::from(current.abs_diff(cross)) / 16.0);
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
                    None,
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
            // Prefer blank/aligned overlap over a stroke crossing.  The
            // bounded strength term distinguishes two equally aligned
            // candidates while the mismatch term rejects displaced edges.
            let stroke_cost = edge_alignment
                + EDGE_STRENGTH_WEIGHT * edge_strength.min(1.0)
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
            super::super::stack_pipeline::compositor::minimum_vertical_seam(
                finite_costs,
                cross_count,
                finite_along_count,
            )
        else {
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
        if path_is_valid {
            let refinement_radius = step.saturating_mul(2).max(2);
            let sample_count = finite_along_max.saturating_sub(finite_along_min) as usize + 1;
            cross_samples.reserve(sample_count);
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
            }
        } else {
            cross_samples = coarse_cross_samples;
        }
        let mut plan = StationSeamPlan {
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
            replacement_samples: Vec::new(),
        };
        if !path_is_valid {
            continue;
        }
        stats.relation_query_visits += materialize_station_seam_samples(
            &mut plan,
            &transition_index,
            current_owners,
            out_width,
            left,
            &left_pixels,
            &left_inverse,
            right,
            &right_pixels,
            &right_inverse,
            projection,
            residual,
            offset_x,
            offset_y,
        );
        stats.materialized_samples += plan.replacement_samples.len();
        plans.push(plan);
    }
    Ok((plans, stats))
}

pub(crate) fn station_seam_series_at(
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

pub(crate) fn station_seam_cross_at(plan: &StationSeamPlan, along: u32) -> f64 {
    station_seam_series_at(&plan.cross_samples, plan.along_min, plan.along_max, along)
}

/// Read the nearest owner transition that can affect one overlap row/column.
/// Production discovery is restricted to the exact ±32px application band;
/// one preceding owner read is enough to detect a transition at its low edge.
pub(crate) fn station_current_transition_nearest(
    plan: &StationSeamPlan,
    source_owner_ids: &[u16],
    out_width: u32,
    along: u32,
    target: f64,
    maximum_displacement: Option<f64>,
) -> (Option<(u32, bool)>, usize) {
    if !target.is_finite() {
        return (None, 0);
    }
    let (cross_min, cross_max) = match plan.orientation {
        SeamOrientation::Vertical => (plan.min_x, plan.max_x),
        SeamOrientation::Horizontal => (plan.min_y, plan.max_y),
    };
    let (transition_min, transition_max) =
        maximum_displacement.map_or((cross_min, cross_max), |maximum_displacement| {
            (
                (target - maximum_displacement)
                    .ceil()
                    .max(f64::from(cross_min)) as u32,
                (target + maximum_displacement)
                    .floor()
                    .min(f64::from(cross_max)) as u32,
            )
        });
    if transition_min > transition_max {
        return (None, 0);
    }
    let scan_min = transition_min.saturating_sub(1).max(cross_min);
    let mut previous = None;
    let mut transitions = Vec::new();
    let mut owner_reads = 0;
    for cross in scan_min..=transition_max {
        let (x, y) = match plan.orientation {
            SeamOrientation::Vertical => (cross, along),
            SeamOrientation::Horizontal => (along, cross),
        };
        let index = y as usize * out_width as usize + x as usize;
        let owner = source_owner_ids.get(index).copied().unwrap_or(NO_OWNER);
        owner_reads += 1;
        if cross >= transition_min
            && let Some(prior) = previous
            && owner != prior
            && [owner, prior].contains(&plan.first_owner)
            && [owner, prior].contains(&plan.second_owner)
        {
            transitions.push((cross, prior == plan.first_owner));
        }
        previous = Some(owner);
    }
    (
        transitions
            .into_iter()
            .min_by_key(|(cross, _)| cross.abs_diff(target.round() as u32)),
        owner_reads,
    )
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StationSeamApplyStats {
    pub(crate) changed_pixels: usize,
    pub(crate) discovery_owner_reads: usize,
    pub(crate) rewrite_visits: usize,
}

impl StationSeamApplyStats {
    #[cfg(test)]
    pub(crate) fn total_owner_reads(self) -> usize {
        self.discovery_owner_reads + self.rewrite_visits
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum StationSeamTraversal {
    Corridor,
    #[cfg(test)]
    LegacyFullOverlap,
}

pub(crate) fn apply_station_seam_plans_with_traversal<F>(
    plans: &[StationSeamPlan],
    panorama: &mut Rgb32FImage,
    owner_map: &mut GrayImage,
    source_owner_ids: &mut [u16],
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
    residual: &residual_warp::ResidualWarp,
    traversal: StationSeamTraversal,
    load_image: &mut F,
) -> Result<StationSeamApplyStats, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let mut stats = StationSeamApplyStats::default();
    for plan in plans {
        let Some(first_index) = plan.first_owner.checked_sub(1).map(usize::from) else {
            continue;
        };
        let Some(second_index) = plan.second_owner.checked_sub(1).map(usize::from) else {
            continue;
        };
        let Some(first) = images.get(first_index).copied() else {
            continue;
        };
        let Some(second) = images.get(second_index).copied() else {
            continue;
        };
        let Some(first_h) = global_homographies.get(&first.id) else {
            continue;
        };
        let Some(second_h) = global_homographies.get(&second.id) else {
            continue;
        };
        let first_inverse = first_h.try_inverse().ok_or_else(|| {
            format!(
                "Station seam pose for '{}' is not invertible",
                first.filename
            )
        })?;
        let second_inverse = second_h.try_inverse().ok_or_else(|| {
            format!(
                "Station seam pose for '{}' is not invertible",
                second.filename
            )
        })?;
        let first_pixels = load_image(first)?;
        let second_pixels = load_image(second)?;
        let first_owner = plan.first_owner;
        let second_owner = plan.second_owner;
        let (cross_min, cross_max) = match plan.orientation {
            SeamOrientation::Vertical => (plan.min_x, plan.max_x),
            SeamOrientation::Horizontal => (plan.min_y, plan.max_y),
        };
        // The current owner transition is a property of one overlap
        // scanline/column, not of each pixel on that scanline. Production
        // discovery reads only transitions close enough to pass the exact
        // 32px displacement gate; the legacy traversal remains a full-scan
        // oracle for byte-equivalence tests.
        let mut current_seam_by_along = Vec::with_capacity(
            plan.along_max
                .saturating_sub(plan.along_min)
                .saturating_add(1) as usize,
        );
        for along in plan.along_min..=plan.along_max {
            let seam = station_seam_cross_at(plan, along);
            let maximum_displacement = match traversal {
                StationSeamTraversal::Corridor => Some(32.0),
                #[cfg(test)]
                StationSeamTraversal::LegacyFullOverlap => None,
            };
            let (current_transition, owner_reads) = station_current_transition_nearest(
                plan,
                source_owner_ids,
                panorama.width(),
                along,
                seam,
                maximum_displacement,
            );
            stats.discovery_owner_reads += owner_reads;
            current_seam_by_along.push(current_transition);
        }
        for along in plan.along_min..=plan.along_max {
            let seam = station_seam_cross_at(plan, along);
            let Some((current_transition, first_on_low_side)) = current_seam_by_along
                .get(along.saturating_sub(plan.along_min) as usize)
                .copied()
                .flatten()
            else {
                continue;
            };
            let current_seam = f64::from(current_transition);
            // A station seam is a local boundary refinement, not permission
            // to relabel a disconnected owner island. Visit only the integer
            // strip between old and new boundaries, never the full overlap.
            if (seam - current_seam).abs() > 32.0 {
                continue;
            }
            let corridor_min = seam.min(current_seam).floor().max(f64::from(cross_min)) as u32;
            let corridor_max = seam.max(current_seam).ceil().min(f64::from(cross_max)) as u32;
            let (visit_min, visit_max) = match traversal {
                StationSeamTraversal::Corridor => (corridor_min, corridor_max),
                #[cfg(test)]
                StationSeamTraversal::LegacyFullOverlap => (cross_min, cross_max),
            };
            for cross in visit_min..=visit_max {
                stats.rewrite_visits += 1;
                if cross < corridor_min || cross > corridor_max {
                    continue;
                }
                let (x, y) = match plan.orientation {
                    SeamOrientation::Vertical => (cross, along),
                    SeamOrientation::Horizontal => (along, cross),
                };
                let index = y as usize * panorama.width() as usize + x as usize;
                let current_owner = *source_owner_ids.get(index).unwrap_or(&NO_OWNER);
                if current_owner != first_owner && current_owner != second_owner {
                    continue;
                }
                let cross = f64::from(cross);
                let first_side = if first_on_low_side {
                    cross <= seam
                } else {
                    cross >= seam
                };
                let desired_owner = if first_side {
                    first_owner
                } else {
                    second_owner
                };
                if desired_owner == current_owner {
                    continue;
                }
                let sampled = if desired_owner == first_owner {
                    station_seam_sample(
                        first,
                        &first_pixels,
                        &first_inverse,
                        projection,
                        residual,
                        x,
                        y,
                        offset_x,
                        offset_y,
                    )
                } else {
                    station_seam_sample(
                        second,
                        &second_pixels,
                        &second_inverse,
                        projection,
                        residual,
                        x,
                        y,
                        offset_x,
                        offset_y,
                    )
                };
                if let Some((pixel, _)) = sampled {
                    let start = index * 3;
                    panorama.as_mut()[start..start + 3].copy_from_slice(&pixel.0);
                    source_owner_ids[index] = desired_owner;
                    owner_map.as_mut()[index] = desired_owner.min(u16::from(u8::MAX)) as u8;
                    stats.changed_pixels += 1;
                }
            }
        }
    }
    Ok(stats)
}

pub(crate) fn apply_station_seam_plans(
    plans: &[StationSeamPlan],
    panorama: &mut Rgb32FImage,
    owner_map: &mut GrayImage,
    source_owner_ids: &mut [u16],
) -> Result<StationSeamApplyStats, String> {
    let mut stats = StationSeamApplyStats::default();
    let width = panorama.width();
    for plan in plans {
        for sample in &plan.replacement_samples {
            stats.rewrite_visits += 1;
            if !sample.covered
                || sample.replacement_owner != plan.first_owner
                    && sample.replacement_owner != plan.second_owner
            {
                continue;
            }
            let index = sample.y as usize * width as usize + sample.x as usize;
            if source_owner_ids.get(index).copied() != Some(sample.expected_owner) {
                continue;
            }
            let Some(pixel_start) = index.checked_mul(3) else {
                continue;
            };
            let Some(pixel_end) = pixel_start.checked_add(3) else {
                continue;
            };
            let Some(pixel_target) = panorama.as_mut().get_mut(pixel_start..pixel_end) else {
                continue;
            };
            let Some(owner_target) = source_owner_ids.get_mut(index) else {
                continue;
            };
            let Some(map_target) = owner_map.as_mut().get_mut(index) else {
                continue;
            };
            pixel_target.copy_from_slice(&sample.pixel.0);
            *owner_target = sample.replacement_owner;
            *map_target = sample.replacement_owner.min(u16::from(u8::MAX)) as u8;
            stats.changed_pixels += 1;
        }
    }
    Ok(stats)
}

#[cfg(test)]
pub(crate) fn apply_station_seam_plans_legacy_full_scan<F>(
    plans: &[StationSeamPlan],
    panorama: &mut Rgb32FImage,
    owner_map: &mut GrayImage,
    source_owner_ids: &mut [u16],
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
    residual: &residual_warp::ResidualWarp,
    load_image: &mut F,
) -> Result<StationSeamApplyStats, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    apply_station_seam_plans_with_traversal(
        plans,
        panorama,
        owner_map,
        source_owner_ids,
        images,
        global_homographies,
        projection,
        offset_x,
        offset_y,
        residual,
        StationSeamTraversal::LegacyFullOverlap,
        load_image,
    )
}
