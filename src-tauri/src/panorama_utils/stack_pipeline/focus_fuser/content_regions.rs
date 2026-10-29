//! Test-only content-region ownership diagnostics.
//!
//! Candidate content regions from exact source masks and a deterministic
//! reduced branch-and-bound selection; production ownership never reads them.

#[cfg(test)]
use super::super::quality_gate;
#[cfg(test)]
use std::collections::VecDeque;

use super::OwnershipGridGeometry;

/// Number of independent content-derived owner regions requested from one
/// station. This is a cardinality target for generic flat components, not a
/// Quality Gate ROI list.
#[cfg(test)]
pub(crate) const CONTENT_OWNER_REGION_TARGET: usize = 8;

/// Exact source-covered flat evidence required by the production ROI and ring.
#[cfg(test)]
pub(crate) const CONTENT_OWNER_EVIDENCE_SPAN_PX: u32 =
    quality_gate::QUALITY_ROI_SIDE + quality_gate::QUALITY_ROI_MARGIN as u32 * 2;

/// A protected component must be wide enough to contain the exact 512px ROI,
/// its 16px same-owner ring, and every possible phase of the production 256px
/// sampling lattice. Consequently a translated station still contains a full
/// lattice-aligned 544px footprint without knowing final-canvas coordinates.
#[cfg(test)]
pub(crate) const CONTENT_OWNER_REGION_SPAN_PX: u32 = quality_gate::QUALITY_ROI_SIDE
    + quality_gate::QUALITY_ROI_MARGIN as u32 * 2
    + quality_gate::QUALITY_ROI_SIDE / 2
    - 1;

/// Current-run content evidence for one source on the ownership grid.
///
/// `fully_covered` is stricter than the normal centre-sample coverage used by
/// graph cut: every source pixel in the cell exists. `flat` and `mean_luma`
/// come from this source's pixels before ownership is solved. No final labels,
/// fixture identity, or Quality Gate enumeration enters this structure.
#[cfg(test)]
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ContentEvidence {
    pub fully_covered: Vec<bool>,
    pub flat: Vec<bool>,
    pub mean_luma: Vec<f64>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ContentRegionCandidate {
    pub(super) owner: u16,
    pub(super) cells: Vec<usize>,
    pub(super) origin: (usize, usize),
    pub(super) data_cost: f64,
}

#[cfg(test)]
pub(super) fn content_region_candidates(
    geometry: OwnershipGridGeometry,
    owner: u16,
    sharpness: &[f64],
    data_costs: &[f64],
    evidence: &ContentEvidence,
) -> Vec<ContentRegionCandidate> {
    let count = geometry.cell_count();
    if evidence.fully_covered.len() != count
        || evidence.flat.len() != count
        || evidence.mean_luma.len() != count
        || sharpness.len() != count
        || data_costs.len() != count
    {
        return Vec::new();
    }
    let eligible = (0..count)
        .map(|cell| {
            evidence.fully_covered[cell]
                && evidence.flat[cell]
                && evidence.mean_luma[cell].is_finite()
                && sharpness[cell].is_finite()
                && data_costs[cell].is_finite()
        })
        .collect::<Vec<_>>();
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    let mut component = vec![usize::MAX; count];
    let mut component_sizes = Vec::new();
    let mut component_bounds = Vec::new();
    for seed in 0..count {
        if !eligible[seed] || component[seed] != usize::MAX {
            continue;
        }
        let id = component_sizes.len();
        let mut size = 0usize;
        let mut bounds = (
            seed % columns,
            seed / columns,
            seed % columns,
            seed / columns,
        );
        let mut queue = VecDeque::from([seed]);
        component[seed] = id;
        while let Some(cell) = queue.pop_front() {
            size += 1;
            let x = cell % columns;
            let y = cell / columns;
            bounds.0 = bounds.0.min(x);
            bounds.1 = bounds.1.min(y);
            bounds.2 = bounds.2.max(x);
            bounds.3 = bounds.3.max(y);
            for neighbour in [
                (x > 0).then(|| cell - 1),
                (x + 1 < columns).then(|| cell + 1),
                (y > 0).then(|| cell - columns),
                (y + 1 < rows).then(|| cell + columns),
            ]
            .into_iter()
            .flatten()
            {
                if eligible[neighbour] && component[neighbour] == usize::MAX {
                    component[neighbour] = id;
                    queue.push_back(neighbour);
                }
            }
        }
        component_sizes.push(size);
        component_bounds.push(bounds);
    }

    let evidence_side =
        CONTENT_OWNER_EVIDENCE_SPAN_PX.div_ceil(geometry.cell_size_px.max(1)) as usize;
    let protected_side =
        CONTENT_OWNER_REGION_SPAN_PX.div_ceil(geometry.cell_size_px.max(1)) as usize;
    let evidence_cells = evidence_side.saturating_mul(evidence_side);
    let protected_cells = protected_side.saturating_mul(protected_side);
    if evidence_side == 0
        || protected_side > columns
        || protected_side > rows
        || evidence_side > protected_side
    {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    let candidate_step =
        (quality_gate::QUALITY_ROI_SIDE / 2).div_ceil(geometry.cell_size_px.max(1)) as usize;
    // Scan each content-derived component from its own local origin, preserving
    // translation determinism. No previous accepted ROI coordinates enter.
    for (id, &(minimum_x, minimum_y, maximum_x, maximum_y)) in component_bounds.iter().enumerate() {
        if component_sizes[id] < evidence_cells
            || maximum_x + 1 < minimum_x + evidence_side
            || maximum_y + 1 < minimum_y + evidence_side
        {
            continue;
        }
        let last_top = (maximum_y + 1 - evidence_side).min(rows - protected_side);
        let last_left = (maximum_x + 1 - evidence_side).min(columns - protected_side);
        if minimum_y > last_top || minimum_x > last_left {
            continue;
        }
        for top in (minimum_y..=last_top).step_by(candidate_step.max(1)) {
            for left in (minimum_x..=last_left).step_by(candidate_step.max(1)) {
                let first = top * columns + left;
                if component[first] != id {
                    continue;
                }
                let mut cells = Vec::with_capacity(protected_cells);
                let mut luma_sum = 0.0;
                let mut luma_square_sum = 0.0;
                let mut total_cost = 0.0;
                let mut valid = true;
                for row in top..top + protected_side {
                    for column in left..left + protected_side {
                        let cell = row * columns + column;
                        // The exact 544px evidence core is one flat connected
                        // component. The phase extension only needs exact source
                        // coverage and a finite current data cost.
                        if row < top + evidence_side && column < left + evidence_side {
                            if component[cell] != id {
                                valid = false;
                                break;
                            }
                            luma_sum += evidence.mean_luma[cell];
                            luma_square_sum += evidence.mean_luma[cell].powi(2);
                        } else if !evidence.fully_covered[cell] || !data_costs[cell].is_finite() {
                            valid = false;
                            break;
                        }
                        total_cost += data_costs[cell];
                        cells.push(cell);
                    }
                    if !valid {
                        break;
                    }
                }
                // Aggregate the same low-frequency cell means instead of imposing
                // a range bound (which is strictly stronger than the frozen
                // standard-deviation metric and rejects valid slow paper shading).
                let mean = luma_sum / evidence_cells as f64;
                let standard_deviation = (luma_square_sum / evidence_cells as f64 - mean.powi(2))
                    .max(0.0)
                    .sqrt();
                if !valid || standard_deviation > quality_gate::FLAT_LOW_FREQUENCY_STD_MAX {
                    continue;
                }
                candidates.push(ContentRegionCandidate {
                    owner,
                    cells,
                    origin: (left, top),
                    data_cost: total_cost / protected_cells as f64,
                });
            }
        }
    }
    candidates
}

#[cfg(test)]
pub(super) fn regions_overlap(
    left: &ContentRegionCandidate,
    right: &ContentRegionCandidate,
) -> bool {
    let right_cells = right
        .cells
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    left.cells.iter().any(|cell| right_cells.contains(cell))
}

/// Deterministic reduced branch-and-bound over genuine component candidates.
/// The candidate pool is bounded before search, while every returned region is
/// still an explicit source-covered component footprint. The objective is
/// cardinality first, then current-run data cost and row-major provenance.
#[cfg(test)]
pub(super) fn select_content_regions(
    mut candidates: Vec<ContentRegionCandidate>,
    target: usize,
) -> Vec<ContentRegionCandidate> {
    const REDUCED_OPTION_LIMIT: usize = 24;
    candidates.sort_by(|left, right| {
        left.data_cost
            .total_cmp(&right.data_cost)
            .then_with(|| left.origin.1.cmp(&right.origin.1))
            .then_with(|| left.origin.0.cmp(&right.origin.0))
            .then_with(|| left.owner.cmp(&right.owner))
    });
    // Preserve low-cost spatial diversity before the exact reduced search;
    // otherwise many overlapping translations of one component can consume
    // the whole option bound and counterfeit an impossible cardinality.
    let mut reduced = Vec::new();
    for candidate in candidates {
        if reduced.len() == REDUCED_OPTION_LIMIT {
            break;
        }
        if reduced
            .iter()
            .all(|other| !regions_overlap(&candidate, other))
        {
            reduced.push(candidate);
        }
    }
    let candidates = reduced;
    let mut best: Vec<usize> = Vec::new();
    let mut best_cost = f64::INFINITY;
    fn visit(
        candidates: &[ContentRegionCandidate],
        target: usize,
        cursor: usize,
        selected: &mut Vec<usize>,
        cost: f64,
        best: &mut Vec<usize>,
        best_cost: &mut f64,
    ) {
        let better = selected.len() > best.len()
            || (selected.len() == best.len()
                && (cost < *best_cost
                    || (cost == *best_cost && selected.as_slice() < best.as_slice())));
        if better {
            *best = selected.clone();
            *best_cost = cost;
        }
        if selected.len() == target
            || cursor == candidates.len()
            || selected.len() + candidates.len() - cursor < best.len()
        {
            return;
        }
        for index in cursor..candidates.len() {
            if selected
                .iter()
                .any(|&other| regions_overlap(&candidates[index], &candidates[other]))
            {
                continue;
            }
            selected.push(index);
            visit(
                candidates,
                target,
                index + 1,
                selected,
                cost + candidates[index].data_cost,
                best,
                best_cost,
            );
            selected.pop();
        }
    }
    visit(
        &candidates,
        target,
        0,
        &mut Vec::new(),
        0.0,
        &mut best,
        &mut best_cost,
    );
    best.into_iter()
        .map(|index| candidates[index].clone())
        .collect()
}
