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

const OWNERSHIP_PAIRWISE_BASE: f64 = 0.35;
const OWNERSHIP_PAIRWISE_DISAGREEMENT_WEIGHT: f64 = 12.0;

impl Graph {
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

    fn source_side(mut self, source: usize, sink: usize) -> Vec<bool> {
        loop {
            let mut levels = self.levels(source);
            if levels[sink] < 0 {
                return levels.into_iter().map(|level| level >= 0).collect();
            }
            let mut next = vec![0usize; self.edges.len()];
            loop {
                let mut path = Vec::<(usize, usize)>::new();
                let mut u = source;
                while u != sink {
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
                if u != sink {
                    break;
                }
                let flow = path
                    .iter()
                    .map(|&(u, i)| self.edges[u][i].capacity)
                    .fold(f64::INFINITY, f64::min);
                for (u, i) in path {
                    let edge = self.edges[u][i];
                    self.edges[u][i].capacity -= flow;
                    self.edges[edge.to][edge.reverse].capacity += flow;
                }
            }
        }
    }
}

pub(super) fn cut_grid(
    width: usize,
    height: usize,
    preference: &[f64],
    disagreement: &[f64],
    fixed: &[i8],
) -> Vec<u8> {
    let count = width * height;
    assert_eq!(preference.len(), count);
    assert_eq!(disagreement.len(), count);
    assert_eq!(fixed.len(), count);
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
            let weight = OWNERSHIP_PAIRWISE_BASE
                + (disagreement[i] + disagreement[neighbour])
                    * OWNERSHIP_PAIRWISE_DISAGREEMENT_WEIGHT;
            graph.connect(i, neighbour, weight, weight);
        }
    }
    graph
        .source_side(source, sink)
        .into_iter()
        .take(count)
        .map(|new| u8::from(new) * 255)
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
        let selected = labels
            .iter()
            .enumerate()
            .fold(0usize, |mask, (i, &v)| mask | (usize::from(v > 0) << i));
        let optimal = (0..64).map(energy).fold(f64::INFINITY, f64::min);
        assert!((energy(selected) - optimal).abs() < 1e-9);
    }
}
