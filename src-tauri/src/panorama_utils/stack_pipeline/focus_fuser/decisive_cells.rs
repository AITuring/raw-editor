//! Decisive-cell rules of the station fusion.
//!
//! Connected regions in which a materially sharper, consistent candidate must
//! replace the current owner, at cell, region and Quality_Gate ROI scale.

use super::super::quality_gate;
use std::collections::VecDeque;

use super::{CandidateCells, OwnershipGridGeometry, mismatch_penalty};

/// Region-level sharpness constraint for the cells outside protected flat
/// components. A candidate can override graph cut only when at least four
/// 4-connected cells all exceed the existing owner's unchanged 5% shortfall
/// limit and none carries the registration mismatch veto. Isolated cell-wise
/// winners therefore remain under coherent regularisation.
pub(super) fn coherent_decisive_cells(
    geometry: OwnershipGridGeometry,
    owner_sharpness: &[f64],
    candidate: &CandidateCells,
) -> Vec<bool> {
    const MINIMUM_COHERENT_CELLS: usize = 4;
    let count = geometry.cell_count();
    if owner_sharpness.len() != count
        || candidate.sharpness.len() != count
        || candidate.covered.len() != count
        || candidate.disagreement.len() != count
    {
        return vec![false; count];
    }
    let eligible = (0..count)
        .map(|cell| {
            if !candidate.covered[cell] || candidate.sharpness[cell] <= 0.0 {
                return false;
            }
            let shortfall =
                (candidate.sharpness[cell] - owner_sharpness[cell]) / candidate.sharpness[cell];
            shortfall > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                && mismatch_penalty(
                    candidate.disagreement[cell],
                    candidate.sharpness[cell],
                    owner_sharpness[cell],
                ) == 0.0
        })
        .collect::<Vec<_>>();
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    let mut decisive = vec![false; count];
    let mut visited = vec![false; count];
    for seed in 0..count {
        if !eligible[seed] || visited[seed] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([seed]);
        visited[seed] = true;
        while let Some(cell) = queue.pop_front() {
            component.push(cell);
            let x = cell % columns;
            let y = cell / columns;
            for neighbour in [
                (x > 0).then(|| cell - 1),
                (x + 1 < columns).then(|| cell + 1),
                (y > 0).then(|| cell - columns),
                (y + 1 < rows).then(|| cell + columns),
            ]
            .into_iter()
            .flatten()
            {
                if eligible[neighbour] && !visited[neighbour] {
                    visited[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
        if component.len() >= MINIMUM_COHERENT_CELLS {
            for cell in component {
                decisive[cell] = true;
            }
        }
    }
    decisive
}

/// Region-level owner move used by the production fold.  The Quality Gate
/// samples a 512px core with a 16px ring; the derived span below also leaves
/// room for the half-core phase of that lattice.  A candidate may move a block
/// only when its measured sharpness wins every covered cell in the block and
/// every one of those cells passes the unchanged shortfall and mismatch rules.
/// This keeps the data-term evidence decisive without turning the output into
/// a salt-and-pepper owner map that contains no auditable 512px ROI.
pub(super) fn region_decisive_cells(
    geometry: OwnershipGridGeometry,
    owner_sharpness: &[f64],
    candidate: &CandidateCells,
) -> Vec<bool> {
    const REGION_SIDE_PX: u32 = quality_gate::QUALITY_ROI_SIDE
        + (quality_gate::QUALITY_ROI_MARGIN as u32 * 2)
        + quality_gate::QUALITY_ROI_SIDE / 2
        - 1;
    const MIN_COVERAGE: f64 = 0.98;

    let count = geometry.cell_count();
    if owner_sharpness.len() != count
        || candidate.sharpness.len() != count
        || candidate.covered.len() != count
        || candidate.disagreement.len() != count
    {
        return vec![false; count];
    }
    let block_side = REGION_SIDE_PX.div_ceil(geometry.cell_size_px.max(1)) as usize;
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    if block_side == 0 || columns == 0 || rows == 0 {
        return vec![false; count];
    }
    let mut decisive = vec![false; count];
    for top in (0..rows).step_by(block_side) {
        let bottom = (top + block_side).min(rows);
        for left in (0..columns).step_by(block_side) {
            let right = (left + block_side).min(columns);
            let block_cells = (right - left) * (bottom - top);
            let mut covered = 0usize;
            let mut eligible = 0usize;
            for row in top..bottom {
                for column in left..right {
                    let cell = row * columns + column;
                    if !candidate.covered[cell] {
                        continue;
                    }
                    covered += 1;
                    let sharpness = candidate.sharpness[cell];
                    let owner = owner_sharpness[cell];
                    let shortfall = if sharpness > 0.0 {
                        (sharpness - owner) / sharpness
                    } else {
                        0.0
                    };
                    if shortfall > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                        && mismatch_penalty(candidate.disagreement[cell], sharpness, owner) == 0.0
                    {
                        eligible += 1;
                    }
                }
            }
            let coverage_ratio = covered as f64 / block_cells.max(1) as f64;
            if covered > 0 && coverage_ratio >= MIN_COVERAGE && eligible == covered {
                for row in top..bottom {
                    for column in left..right {
                        let cell = row * columns + column;
                        if candidate.covered[cell] {
                            decisive[cell] = true;
                        }
                    }
                }
            }
        }
    }
    decisive
}

/// Mark candidate labels that can support an auditable Quality_Gate region.
///
/// The graph cut is allowed to keep a smooth owner when the local data term is
/// only slightly better.  That is useful for avoiding seams, but the old
/// fold-time override only admitted a candidate that beat the current owner by
/// more than the five-percent gate margin.  A later candidate could therefore
/// be measurably within the gate everywhere in a 544px window and still never
/// form a single-owner ROI.  Here the comparison is made against the best
/// Sharpness_Score observed for the cell, which is the same winner used by the
/// owner shortfall evidence.  A candidate is eligible when it stays within the
/// existing limit, passes the unchanged mismatch veto, and covers every cell
/// of a full ROI core plus ring.  Prefix sums keep the sliding-window search
/// linear in the ownership grid rather than scanning each 544px region again.
pub(super) fn quality_region_decisive_cells(
    geometry: OwnershipGridGeometry,
    owner_sharpness: &[f64],
    best_sharpness: &[f64],
    candidate: &CandidateCells,
) -> Vec<bool> {
    let count = geometry.cell_count();
    if owner_sharpness.len() != count
        || best_sharpness.len() != count
        || candidate.sharpness.len() != count
        || candidate.covered.len() != count
        || candidate.disagreement.len() != count
    {
        return vec![false; count];
    }
    let cell_size = geometry.cell_size_px.max(1);
    let region_px = quality_gate::QUALITY_ROI_SIDE + quality_gate::QUALITY_ROI_MARGIN as u32 * 2;
    let span = region_px.div_ceil(cell_size) as usize;
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    if span == 0 || columns < span || rows < span {
        return vec![false; count];
    }

    let debug = std::env::var_os("RAW_EDITOR_FOCUS_OWNERSHIP_DEBUG").is_some();
    let mut covered_cells = 0usize;
    let mut shortfall_cells = 0usize;
    let mut mismatch_cells = 0usize;
    let mut safe_cells_direct = 0usize;
    let mut invalid = vec![0u32; (columns + 1) * (rows + 1)];
    let stride = columns + 1;
    for row in 0..rows {
        let mut row_sum = 0u32;
        for column in 0..columns {
            let cell = row * columns + column;
            let best = best_sharpness[cell];
            let candidate_score = candidate.sharpness[cell];
            let shortfall = if best.is_finite() && best > 0.0 && candidate_score.is_finite() {
                ((best - candidate_score) / best).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let mismatch = mismatch_penalty(
                candidate.disagreement[cell],
                candidate_score,
                owner_sharpness[cell],
            );
            if candidate.covered[cell] {
                covered_cells += 1;
                if shortfall > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64 {
                    shortfall_cells += 1;
                }
                if mismatch != 0.0 {
                    mismatch_cells += 1;
                }
            }
            let safe = candidate.covered[cell]
                && candidate_score > 0.0
                && candidate_score >= owner_sharpness[cell]
                && shortfall <= quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                && mismatch == 0.0;
            safe_cells_direct += usize::from(safe);
            row_sum += u32::from(!safe);
            let index = (row + 1) * stride + column + 1;
            invalid[index] = invalid[row * stride + column + 1] + row_sum;
        }
    }
    let window_invalid = |left: usize, top: usize| -> u32 {
        let right = left + span;
        let bottom = top + span;
        invalid[bottom * stride + right] + invalid[top * stride + left]
            - invalid[top * stride + right]
            - invalid[bottom * stride + left]
    };
    // Accumulate the union of all valid windows with a 2-D difference plane.
    // Marking every 34×34 cell immediately for every sliding window turns a
    // broad station into O(number_of_windows × window_area); the difference
    // plane keeps the same union semantics in O(number_of_cells).
    let mut valid_windows = 0usize;
    let mut window_hits = vec![0i32; (columns + 1) * (rows + 1)];
    for top in 0..=rows - span {
        for left in 0..=columns - span {
            if window_invalid(left, top) != 0 {
                continue;
            }
            valid_windows += 1;
            let right = left + span;
            let bottom = top + span;
            window_hits[top * stride + left] += 1;
            window_hits[top * stride + right] -= 1;
            window_hits[bottom * stride + left] -= 1;
            window_hits[bottom * stride + right] += 1;
        }
    }
    let mut decisive = vec![false; count];
    for row in 0..=rows {
        for column in 0..=columns {
            let index = row * stride + column;
            if row > 0 {
                window_hits[index] += window_hits[(row - 1) * stride + column];
            }
            if column > 0 {
                window_hits[index] += window_hits[row * stride + column - 1];
            }
            if row > 0 && column > 0 {
                window_hits[index] -= window_hits[(row - 1) * stride + column - 1];
            }
            if row < rows && column < columns {
                decisive[row * columns + column] = window_hits[index] > 0;
            }
        }
    }
    if debug {
        println!(
            "    - Focus quality-ROI scan: grid={}x{}, span={} cells, covered_cells={}, shortfall_cells={}, mismatch_cells={}, safe_cells={}, valid_windows={}",
            columns,
            rows,
            span,
            covered_cells,
            shortfall_cells,
            mismatch_cells,
            safe_cells_direct,
            valid_windows
        );
    }
    decisive
}

/// A cell-level Quality_Gate correction for a candidate that is materially
/// sharper than the current owner and remains within the same fixed shortfall
/// limit as the best observed source.  The Potts term is still the normal
/// graph-cut regulariser; this narrow correction only prevents it from
/// retaining a known gate-failing owner when the candidate has no registration
/// mismatch veto.  It is intentionally cell-local because the final gate
/// counts every covered textured pixel, while the ownership contract already
/// writes at ownership-cell resolution.
pub(super) fn quality_gate_decisive_cells(
    geometry: OwnershipGridGeometry,
    owner_sharpness: &[f64],
    best_sharpness: &[f64],
    candidate: &CandidateCells,
) -> Vec<bool> {
    let count = geometry.cell_count();
    if owner_sharpness.len() != count
        || best_sharpness.len() != count
        || candidate.sharpness.len() != count
        || candidate.covered.len() != count
        || candidate.disagreement.len() != count
    {
        return vec![false; count];
    }
    (0..count)
        .map(|cell| {
            if !candidate.covered[cell]
                || candidate.sharpness[cell] <= owner_sharpness[cell]
                || candidate.sharpness[cell] <= 0.0
                || !candidate.sharpness[cell].is_finite()
                || !best_sharpness[cell].is_finite()
                || best_sharpness[cell] <= 0.0
            {
                return false;
            }
            let candidate_advantage =
                (candidate.sharpness[cell] - owner_sharpness[cell]) / candidate.sharpness[cell];
            let candidate_shortfall =
                (best_sharpness[cell] - candidate.sharpness[cell]) / best_sharpness[cell];
            candidate_advantage > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                && candidate_shortfall <= quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
                && mismatch_penalty(
                    candidate.disagreement[cell],
                    candidate.sharpness[cell],
                    owner_sharpness[cell],
                ) == 0.0
        })
        .collect()
}

/// Symmetric coherent protection: graph cut may not replace a connected region
/// whose current real source is already sharper than this candidate by more
/// than the unchanged shortfall limit. This prevents broad smoothness from
/// undoing an earlier data-derived owner while leaving small/ambiguous cells to
/// the normal regularised solve.
#[cfg(test)]
pub(super) fn coherent_base_cells(
    geometry: OwnershipGridGeometry,
    owner_sharpness: &[f64],
    candidate: &CandidateCells,
) -> Vec<bool> {
    const MINIMUM_COHERENT_CELLS: usize = 4;
    let count = geometry.cell_count();
    if owner_sharpness.len() != count
        || candidate.sharpness.len() != count
        || candidate.covered.len() != count
    {
        return vec![false; count];
    }
    let eligible = (0..count)
        .map(|cell| {
            candidate.covered[cell]
                && owner_sharpness[cell] > 0.0
                && (owner_sharpness[cell] - candidate.sharpness[cell]) / owner_sharpness[cell]
                    > quality_gate::OWNER_SHARPNESS_SHORTFALL_MAX as f64
        })
        .collect::<Vec<_>>();
    let columns = geometry.columns as usize;
    let rows = geometry.rows as usize;
    let mut protected = vec![false; count];
    let mut visited = vec![false; count];
    for seed in 0..count {
        if !eligible[seed] || visited[seed] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::from([seed]);
        visited[seed] = true;
        while let Some(cell) = queue.pop_front() {
            component.push(cell);
            let x = cell % columns;
            let y = cell / columns;
            for neighbour in [
                (x > 0).then(|| cell - 1),
                (x + 1 < columns).then(|| cell + 1),
                (y > 0).then(|| cell - columns),
                (y + 1 < rows).then(|| cell + columns),
            ]
            .into_iter()
            .flatten()
            {
                if eligible[neighbour] && !visited[neighbour] {
                    visited[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
        if component.len() >= MINIMUM_COHERENT_CELLS {
            for cell in component {
                protected[cell] = true;
            }
        }
    }
    protected
}
