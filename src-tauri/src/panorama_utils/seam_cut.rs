//! Bounded binary min-cut for focus-source ownership, without a native CV
//! dependency. Blocking-flow traversal uses an explicit path, not recursion.

#[derive(Clone, Copy)]
struct Edge {
    to: usize,
    reverse: usize,
    capacity: f64,
}

struct Graph {
    edges: Vec<Vec<Edge>>,
}

/// Deterministic work meter of a bounded min-cut (需求 3.5).
///
/// One unit is one residual edge examined by the level BFS or one advance /
/// retreat step of the blocking-flow walk. The count is a pure function of the
/// graph, so a budget expressed in these units gives the same decision on any
/// thread count, machine load or clock speed — unlike a wall-clock deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CutWork {
    pub used: u64,
    pub limit: u64,
}

impl CutWork {
    /// A meter that never runs out; used by callers without a budget.
    pub(crate) fn unlimited() -> Self {
        Self {
            used: 0,
            limit: u64::MAX,
        }
    }

    pub(crate) fn with_limit(limit: u64) -> Self {
        Self { used: 0, limit }
    }

    /// True once more work was charged than the limit allows.
    pub(crate) fn exhausted(&self) -> bool {
        self.used > self.limit
    }

    /// Charge `units`; `false` means the budget is now exceeded.
    #[inline]
    fn charge(&mut self, units: u64) -> bool {
        self.used = self.used.saturating_add(units);
        self.used <= self.limit
    }
}

pub(super) const OWNERSHIP_PAIRWISE_BASE: f64 = 0.35;
pub(super) const OWNERSHIP_PAIRWISE_DISAGREEMENT_WEIGHT: f64 = 12.0;

/// Smoothness weight of one 4-neighbour edge of the ownership grid: the cost
/// charged when the two cells end up with different owners.
///
/// Published so a multi-label caller can evaluate the very same energy
/// [`cut_grid`] minimises instead of re-deriving the formula. The arithmetic is
/// unchanged and [`cut_grid`] is its only user inside this module, so the
/// min-cut semantics are untouched.
pub(super) fn pairwise_weight(disagreement_a: f64, disagreement_b: f64) -> f64 {
    OWNERSHIP_PAIRWISE_BASE
        + (disagreement_a + disagreement_b) * OWNERSHIP_PAIRWISE_DISAGREEMENT_WEIGHT
}

impl Graph {
    /// Same blocking-flow traversal with the unsaturated path prefix retained.
    /// Restarting at the source after every augmentation repeatedly walks a
    /// long prefix on native grids. Resume at its first saturated edge instead.
    fn source_side_reusing_path(mut self, source: usize, sink: usize) -> Vec<bool> {
        loop {
            let mut levels = self.levels(source);
            if levels[sink] < 0 {
                return levels.into_iter().map(|level| level >= 0).collect();
            }
            let mut next = vec![0usize; self.edges.len()];
            let mut path = Vec::<(usize, usize)>::new();
            let mut u = source;
            loop {
                if u == sink {
                    let flow = path
                        .iter()
                        .map(|&(node, edge)| self.edges[node][edge].capacity)
                        .fold(f64::INFINITY, f64::min);
                    for &(node, index) in &path {
                        let edge = self.edges[node][index];
                        self.edges[node][index].capacity -= flow;
                        self.edges[edge.to][edge.reverse].capacity += flow;
                    }
                    let saturated = path
                        .iter()
                        .position(|&(node, index)| self.edges[node][index].capacity <= 1e-9)
                        .unwrap();
                    u = path[saturated].0;
                    path.truncate(saturated);
                    next[u] += 1;
                    continue;
                }
                while next[u] < self.edges[u].len() {
                    let edge = self.edges[u][next[u]];
                    if edge.capacity > 1e-9 && levels[edge.to] == levels[u] + 1 {
                        break;
                    }
                    next[u] += 1;
                }
                if next[u] == self.edges[u].len() {
                    levels[u] = -1;
                    let Some((parent, _)) = path.pop() else {
                        break;
                    };
                    u = parent;
                    next[u] += 1;
                } else {
                    path.push((u, next[u]));
                    u = self.edges[u][next[u]].to;
                }
            }
        }
    }

    fn connect(&mut self, a: usize, b: usize, forward: f64, reverse: f64) {
        let ai = self.edges[a].len();
        let bi = self.edges[b].len();
        self.edges[a].push(Edge {
            to: b,
            reverse: bi,
            capacity: forward,
        });
        self.edges[b].push(Edge {
            to: a,
            reverse: ai,
            capacity: reverse,
        });
    }

    fn levels(&self, source: usize) -> Vec<i32> {
        let mut levels = vec![-1; self.edges.len()];
        let mut queue = Vec::with_capacity(self.edges.len());
        levels[source] = 0;
        queue.push(source);
        let mut head = 0;
        while head < queue.len() {
            let u = queue[head];
            head += 1;
            for e in &self.edges[u] {
                if e.capacity > 1e-9 && levels[e.to] < 0 {
                    levels[e.to] = levels[u] + 1;
                    queue.push(e.to);
                }
            }
        }
        levels
    }

    /// [`Self::levels`] that charges one unit per residual edge examined.
    fn levels_metered(&self, source: usize, work: &mut CutWork) -> Option<Vec<i32>> {
        let mut levels = vec![-1; self.edges.len()];
        let mut queue = Vec::with_capacity(self.edges.len());
        levels[source] = 0;
        queue.push(source);
        let mut head = 0;
        while head < queue.len() {
            let u = queue[head];
            head += 1;
            if !work.charge(self.edges[u].len() as u64) {
                return None;
            }
            for e in &self.edges[u] {
                if e.capacity > 1e-9 && levels[e.to] < 0 {
                    levels[e.to] = levels[u] + 1;
                    queue.push(e.to);
                }
            }
        }
        Some(levels)
    }

    #[allow(dead_code)]
    fn source_side(self, source: usize, sink: usize) -> Vec<bool> {
        self.source_side_metered(source, sink, &mut CutWork::unlimited())
            .expect("an unlimited meter never runs out")
    }

    /// Dinic blocking flow under a deterministic work budget. `None` means the
    /// budget ran out before the minimum cut was found. With an unlimited meter
    /// the control flow, and so the cut, is identical to the unmetered solver.
    fn source_side_metered(
        mut self,
        source: usize,
        sink: usize,
        work: &mut CutWork,
    ) -> Option<Vec<bool>> {
        loop {
            let mut levels = self.levels_metered(source, work)?;
            if levels[sink] < 0 {
                return Some(levels.into_iter().map(|level| level >= 0).collect());
            }
            let mut next = vec![0usize; self.edges.len()];
            loop {
                let mut path = Vec::<(usize, usize)>::new();
                let mut u = source;
                while u != sink {
                    while next[u] < self.edges[u].len() {
                        if !work.charge(1) {
                            return None;
                        }
                        let edge = self.edges[u][next[u]];
                        if edge.capacity > 1e-9 && levels[edge.to] == levels[u] + 1 {
                            break;
                        }
                        next[u] += 1;
                    }
                    if !work.charge(1) {
                        return None;
                    }
                    if next[u] == self.edges[u].len() {
                        levels[u] = -1;
                        let Some((parent, _)) = path.pop() else {
                            break;
                        };
                        u = parent;
                        next[u] += 1;
                    } else {
                        path.push((u, next[u]));
                        u = self.edges[u][next[u]].to;
                    }
                }
                if u != sink {
                    break;
                }
                let flow = path
                    .iter()
                    .map(|&(u, i)| self.edges[u][i].capacity)
                    .fold(f64::INFINITY, f64::min);
                if !work.charge(path.len() as u64) {
                    return None;
                }
                for (u, i) in path {
                    let edge = self.edges[u][i];
                    self.edges[u][i].capacity -= flow;
                    self.edges[edge.to][edge.reverse].capacity += flow;
                }
            }
        }
    }
}

/// Binary min-cut over a `width × height` 4-connected grid.
///
/// - `preference[i]` is `cost(base) − cost(candidate)`; the split into the two
///   terminal capacities is the standard reparametrisation, so only the
///   difference matters.
/// - `disagreement[i]` feeds [`pairwise_weight`] on every 4-neighbour edge.
/// - `fixed[i]`: `1` pins the cell to the candidate, `-1` pins it to the base,
///   `0` leaves it to the cut.
/// - the returned byte is `255` when the cell took the **candidate** and `0`
///   when it kept the **base**.
pub(super) fn cut_grid(
    width: usize,
    height: usize,
    preference: &[f64],
    disagreement: &[f64],
    fixed: &[i8],
) -> Vec<u8> {
    let count = width.saturating_mul(height);
    let edge_weights = vec![(None, None); count];
    cut_grid_weighted(
        width,
        height,
        preference,
        disagreement,
        fixed,
        &edge_weights,
    )
}

/// Binary cut with an optional per-cell horizontal/vertical edge weight. The
/// entry at `i` contains the optional weights for the right and lower edges of
/// cell `i`; `None`
/// retains the normal disagreement-derived weight. This keeps the original
/// API and tie behaviour while allowing a correct alpha-expansion reparametr-
/// isation for edges whose two current owners differ.
pub(super) fn cut_grid_weighted(
    width: usize,
    height: usize,
    preference: &[f64],
    disagreement: &[f64],
    fixed: &[i8],
    edge_weights: &[(Option<f64>, Option<f64>)],
) -> Vec<u8> {
    cut_grid_weighted_metered(
        width,
        height,
        preference,
        disagreement,
        fixed,
        edge_weights,
        &mut CutWork::unlimited(),
    )
    .expect("an unlimited meter never runs out")
}

/// [`cut_grid_weighted`] under a deterministic work budget (需求 3.5).
///
/// `None` means `work` ran out before the minimum cut was found; `work.used`
/// then holds the charged units. Within the budget the result is bit-identical
/// to [`cut_grid_weighted`], because the meter only observes the traversal.
pub(super) fn cut_grid_weighted_metered(
    width: usize,
    height: usize,
    preference: &[f64],
    disagreement: &[f64],
    fixed: &[i8],
    edge_weights: &[(Option<f64>, Option<f64>)],
    work: &mut CutWork,
) -> Option<Vec<u8>> {
    let count = width * height;
    assert_eq!(preference.len(), count);
    assert_eq!(disagreement.len(), count);
    assert_eq!(fixed.len(), count);
    assert_eq!(edge_weights.len(), count);
    let source = count;
    let sink = count + 1;
    let mut graph = Graph {
        edges: (0..count + 2).map(|_| Vec::with_capacity(6)).collect(),
    };
    for i in 0..count {
        let (base_cost, candidate_cost) = match fixed[i] {
            1 => (1e6, 0.0),
            -1 => (0.0, 1e6),
            _ => (preference[i].max(0.0), (-preference[i]).max(0.0)),
        };
        graph.connect(source, i, base_cost, 0.0);
        graph.connect(i, sink, candidate_cost, 0.0);
        let x = i % width;
        let y = i / width;
        for neighbour in [
            (x + 1 < width).then(|| i + 1),
            (y + 1 < height).then(|| i + width),
        ]
        .into_iter()
        .flatten()
        {
            let explicit = if neighbour == i + 1 {
                edge_weights[i].0
            } else {
                edge_weights[i].1
            };
            let weight = explicit
                .unwrap_or_else(|| pairwise_weight(disagreement[i], disagreement[neighbour]));
            graph.connect(i, neighbour, weight, weight);
        }
    }
    Some(
        graph
            .source_side_metered(source, sink, work)?
            .into_iter()
            .take(count)
            .map(|new| u8::from(new) * 255)
            .collect(),
    )
}

/// Exact fixed-node elimination for the large native ownership revisit.
/// Fixed/free edges become unary costs; fixed/fixed edges are constants.
/// Removing constants preserves the minimum cut and its minimal source side.
pub(super) fn cut_grid_weighted_compact_fixed(
    width: usize,
    height: usize,
    preference: &[f64],
    disagreement: &[f64],
    fixed: &[i8],
    edge_weights: &[(Option<f64>, Option<f64>)],
) -> Vec<u8> {
    let count = width * height;
    assert_eq!(preference.len(), count);
    assert_eq!(disagreement.len(), count);
    assert_eq!(fixed.len(), count);
    assert_eq!(edge_weights.len(), count);
    let mut nodes = vec![None; count];
    let mut free_count = 0;
    for (i, &pin) in fixed.iter().enumerate() {
        if pin == 0 {
            nodes[i] = Some(free_count);
            free_count += 1;
        }
    }
    let mut unary = preference.to_vec();
    for i in 0..count {
        for (other, horizontal) in [
            ((i % width + 1 < width).then_some(i + 1), true),
            ((i / width + 1 < height).then_some(i + width), false),
        ] {
            let Some(other) = other else {
                continue;
            };
            let weight = if horizontal {
                edge_weights[i].0
            } else {
                edge_weights[i].1
            }
            .unwrap_or_else(|| pairwise_weight(disagreement[i], disagreement[other]));
            match (nodes[i], nodes[other]) {
                (Some(_), None) => unary[i] += weight * f64::from(fixed[other]),
                (None, Some(_)) => unary[other] += weight * f64::from(fixed[i]),
                _ => {}
            }
        }
    }
    let source = free_count;
    let sink = free_count + 1;
    let mut graph = Graph {
        edges: (0..free_count + 2).map(|_| Vec::with_capacity(6)).collect(),
    };
    for i in 0..count {
        let Some(node) = nodes[i] else {
            continue;
        };
        graph.connect(source, node, unary[i].max(0.0), 0.0);
        graph.connect(node, sink, (-unary[i]).max(0.0), 0.0);
        for (other, horizontal) in [
            ((i % width + 1 < width).then_some(i + 1), true),
            ((i / width + 1 < height).then_some(i + width), false),
        ] {
            let Some(other) = other.and_then(|i| nodes[i].map(|node| (i, node))) else {
                continue;
            };
            let weight = if horizontal {
                edge_weights[i].0
            } else {
                edge_weights[i].1
            }
            .unwrap_or_else(|| pairwise_weight(disagreement[i], disagreement[other.0]));
            if weight > 0.0 {
                graph.connect(node, other.1, weight, weight);
            }
        }
    }
    let source_side = graph.source_side_reusing_path(source, sink);
    (0..count)
        .map(|i| match nodes[i] {
            Some(node) => u8::from(source_side[node]) * 255,
            None => u8::from(fixed[i] > 0) * 255,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_cut_matches_exhaustive_minimum_and_obeys_coverage_seeds() {
        let preference: [f64; 6] = [0.4, -0.8, 0.2, -0.5, 1.2, -0.7];
        let disagreement: [f64; 6] = [0.01, 0.02, 0.01, 0.03, 0.02, 0.01];
        let fixed = [1, 0, -1, 0, 0, 0];
        let energy = |mask: usize| {
            let mut cost = 0.0f64;
            for i in 0..6 {
                let new = mask & (1 << i) != 0;
                if (fixed[i] == 1 && !new) || (fixed[i] == -1 && new) {
                    return f64::INFINITY;
                }
                cost += if new {
                    (-preference[i]).max(0.0)
                } else {
                    preference[i].max(0.0)
                };
                for j in [(i % 3 < 2).then(|| i + 1), (i < 3).then(|| i + 3)]
                    .into_iter()
                    .flatten()
                {
                    if new != (mask & (1 << j) != 0) {
                        cost += OWNERSHIP_PAIRWISE_BASE
                            + (disagreement[i] + disagreement[j])
                                * OWNERSHIP_PAIRWISE_DISAGREEMENT_WEIGHT;
                    }
                }
            }
            cost
        };
        let labels = cut_grid(3, 2, &preference, &disagreement, &fixed);
        // The metered cut is the same cut; a budget below its work fails.
        let edges = vec![(None, None); 6];
        let mut work = CutWork::unlimited();
        let metered =
            cut_grid_weighted_metered(3, 2, &preference, &disagreement, &fixed, &edges, &mut work);
        assert_eq!(metered.as_deref(), Some(labels.as_slice()));
        assert!(work.used > 0);
        let mut short = CutWork::with_limit(work.used - 1);
        assert!(
            cut_grid_weighted_metered(3, 2, &preference, &disagreement, &fixed, &edges, &mut short)
                .is_none()
        );
        assert!(short.exhausted());
        let mut exact = CutWork::with_limit(work.used);
        assert_eq!(
            cut_grid_weighted_metered(3, 2, &preference, &disagreement, &fixed, &edges, &mut exact)
                .as_deref(),
            Some(labels.as_slice())
        );
        assert_eq!(exact.used, work.used);
        let selected = labels
            .iter()
            .enumerate()
            .fold(0usize, |mask, (i, &v)| mask | (usize::from(v > 0) << i));
        let optimal = (0..64).map(energy).fold(f64::INFINITY, f64::min);
        assert!((energy(selected) - optimal).abs() < 1e-9);
        assert_eq!(
            cut_grid_weighted_compact_fixed(
                3,
                2,
                &preference,
                &disagreement,
                &fixed,
                &[(None, None); 6]
            ),
            labels
        );
    }

    #[test]
    fn fixed_node_elimination_preserves_cut_labels_including_ties() {
        let mut state = 17u64;
        for case in 0..256 {
            let mut draw = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 32) as u32
            };
            let preference = (0..12)
                .map(|_| {
                    if case % 8 == 0 {
                        0.0
                    } else {
                        (draw() % 9) as f64 - 4.0
                    }
                })
                .collect::<Vec<_>>();
            let fixed = (0..12).map(|_| (draw() % 3) as i8 - 1).collect::<Vec<_>>();
            let edges = (0..12)
                .map(|_| (Some((draw() % 5) as f64), Some((draw() % 5) as f64)))
                .collect::<Vec<_>>();
            let old = cut_grid_weighted(4, 3, &preference, &[0.0; 12], &fixed, &edges);
            let compact =
                cut_grid_weighted_compact_fixed(4, 3, &preference, &[0.0; 12], &fixed, &edges);
            assert_eq!(
                compact, old,
                "fixed reduction changed optimum/tie at case {case}"
            );
        }
    }
}
