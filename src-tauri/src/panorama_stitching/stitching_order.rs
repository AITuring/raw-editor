//! Panorama stitching order.
//!
//! Edge weights from match support and the graph order shared by the panorama
//! and focus-stack paths.

use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::shared::MatchInfo;
use nalgebra::Matrix3;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::path::Path;

#[cfg(test)]
use super::virtual_tile_translation;
use super::{
    Dsu, FULL_RES_RANSAC_INLIER_THRESHOLD, median_symmetric_error, natural_path_cmp,
    panorama_spatial_support, panorama_transform_overlap_support,
};

pub(super) fn build_stitching_order(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> (Vec<usize>, HashMap<usize, Matrix3<f64>>) {
    build_graph_stitching_order(images, matches, |source, target, match_info| {
        panorama_edge_weight(source, target, match_info)
    })
}

pub(super) fn panorama_edge_weight(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
) -> f64 {
    if match_info.points.len() < 4 || match_info.candidate_points.len() < 4 {
        // Unit/test callers may provide only an inlier count. Keep that API
        // behaviour deterministic while real matches use the stronger quality
        // terms below.
        return match_info.inliers as f64;
    }
    let candidate_count = match_info
        .candidate_points
        .len()
        .max(match_info.points.len());
    let inlier_ratio = match_info.points.len() as f64 / candidate_count as f64;
    let error = median_symmetric_error(&match_info.homography, &match_info.points);
    let precision = if error.is_finite() {
        1.0 / (1.0 + error / FULL_RES_RANSAC_INLIER_THRESHOLD)
    } else {
        0.0
    };
    let spatial =
        panorama_spatial_support(&match_info.points, source.dimensions(), target.dimensions());
    let overlap = panorama_transform_overlap_support(
        &match_info.homography,
        source.dimensions(),
        target.dimensions(),
    );
    // A repeated brush pattern can produce many descriptor inliers in a tiny
    // region.  Inlier count alone then makes that false edge the MST backbone.
    // Keep the count dominant, but make broad, low-error, bidirectionally
    // visible overlaps win over compact coincidences.
    (match_info.inliers as f64)
        * (0.35 + 0.65 * inlier_ratio.clamp(0.0, 1.0))
        * (0.35 + 0.65 * precision.clamp(0.0, 1.0))
        * (0.25 + 0.75 * spatial.clamp(0.0, 1.0))
        * (0.25 + 0.75 * overlap.clamp(0.0, 1.0))
}

#[cfg(test)]
pub(super) fn pair_harness_disconnected_focus_geometry(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> (Vec<usize>, HashMap<usize, Matrix3<f64>>) {
    let mut adjacency = vec![Vec::<usize>::new(); images.len()];
    for &(left, right) in matches.keys() {
        if left < images.len() && right < images.len() && left != right {
            adjacency[left].push(right);
            adjacency[right].push(left);
        }
    }
    for neighbours in &mut adjacency {
        neighbours.sort_by(|&left, &right| images[left].filename.cmp(&images[right].filename));
        neighbours.dedup();
    }
    let mut roots = (0..images.len()).collect::<Vec<_>>();
    roots.sort_by(|&left, &right| images[left].filename.cmp(&images[right].filename));
    let separation = f64::from(
        images
            .iter()
            .map(|image| image.width.max(image.height))
            .max()
            .unwrap_or(1),
    ) * 2.5;
    let mut order = Vec::with_capacity(images.len());
    let mut homographies = HashMap::with_capacity(images.len());
    let mut visited = vec![false; images.len()];
    let mut component = 0usize;
    for root in roots {
        if visited[root] {
            continue;
        }
        let origin = virtual_tile_translation(component as f64 * separation, 0.0);
        component += 1;
        visited[root] = true;
        let mut pending = VecDeque::from([(root, origin)]);
        while let Some((current, current_to_world)) = pending.pop_front() {
            order.push(current);
            homographies.insert(images[current].id, current_to_world);
            for &next in &adjacency[current] {
                if visited[next] {
                    continue;
                }
                let next_to_current = if let Some(relation) = matches.get(&(next, current)) {
                    relation.homography
                } else if let Some(relation) = matches.get(&(current, next)) {
                    let Some(inverse) = relation.homography.try_inverse() else {
                        continue;
                    };
                    inverse
                } else {
                    continue;
                };
                visited[next] = true;
                pending.push_back((next, current_to_world * next_to_current));
            }
        }
    }
    println!(
        "  - Pair harness retained {} disconnected source component(s) as finite rendering priors",
        component
    );
    (order, homographies)
}

pub(super) fn build_graph_stitching_order<F>(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    edge_weight: F,
) -> (Vec<usize>, HashMap<usize, Matrix3<f64>>)
where
    F: Fn(&ImageInfo, &ImageInfo, &MatchInfo) -> f64,
{
    if images.is_empty() {
        return (vec![], HashMap::new());
    }
    let n = images.len();
    if n < 2 {
        let mut homographies = HashMap::new();
        if n == 1 {
            homographies.insert(images[0].id, Matrix3::identity());
        }
        return ((0..n).collect(), homographies);
    }

    let mut edges = matches
        .iter()
        .filter_map(|(&(i, j), match_info)| {
            let source = images.get(i)?;
            let target = images.get(j)?;
            let weight = edge_weight(source, target, match_info);
            if !weight.is_finite() {
                println!(
                    "  - Ignoring non-finite overlap edge '{}' <-> '{}'",
                    source.filename, target.filename
                );
                return None;
            }
            Some((weight, i, j))
        })
        .collect::<Vec<_>>();
    edges.sort_by(|left, right| {
        let left_names = {
            let first = images[left.1].filename.as_str();
            let second = images[left.2].filename.as_str();
            if first <= second {
                (first, second)
            } else {
                (second, first)
            }
        };
        let right_names = {
            let first = images[right.1].filename.as_str();
            let second = images[right.2].filename.as_str();
            if first <= second {
                (first, second)
            } else {
                (second, first)
            }
        };
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left_names.cmp(&right_names))
            // Duplicate paths in the selection would leave the name comparison
            // equal; the source index pair is the stable secondary key (需求 14.6).
            .then_with(|| (left.1, left.2).cmp(&(right.1, right.2)))
    });
    let mut mst_adj: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut dsu = Dsu::new(n);
    let mut num_edges = 0;

    for &(_, i, j) in &edges {
        if dsu.find(i) != dsu.find(j) {
            dsu.union(i, j);
            mst_adj.entry(i).or_default().push(j);
            mst_adj.entry(j).or_default().push(i);
            num_edges += 1;
            if num_edges == n - 1 {
                break;
            }
        }
    }

    // Kruskal produces a forest when the source selection contains separate
    // scenes.  The previous code chose a start node across the whole forest,
    // so a tiny two-image component could win merely because it had a low
    // degree.  Traverse the largest connected component deterministically.
    let mut component_visited = HashSet::new();
    let mut components = Vec::<Vec<usize>>::new();
    for root in 0..n {
        if component_visited.contains(&root) || !mst_adj.contains_key(&root) {
            continue;
        }
        let mut component = Vec::new();
        let mut pending = vec![root];
        component_visited.insert(root);
        while let Some(node) = pending.pop() {
            component.push(node);
            if let Some(neighbours) = mst_adj.get(&node) {
                for &neighbour in neighbours {
                    if component_visited.insert(neighbour) {
                        pending.push(neighbour);
                    }
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components.sort_by(|left, right| {
        right.len().cmp(&left.len()).then_with(|| {
            let left_name = left
                .iter()
                .map(|&index| images[index].filename.as_str())
                .min()
                .unwrap_or("");
            let right_name = right
                .iter()
                .map(|&index| images[index].filename.as_str())
                .min()
                .unwrap_or("");
            natural_path_cmp(left_name, right_name)
        })
    });
    if components.len() > 1 {
        let summary = components
            .iter()
            .map(|component| {
                let first = component
                    .iter()
                    .map(|&index| {
                        Path::new(&images[index].filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                    })
                    .min()
                    .unwrap_or_default();
                let last = component
                    .iter()
                    .map(|&index| {
                        Path::new(&images[index].filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                    })
                    .max()
                    .unwrap_or_default();
                format!("{} [{first}..{last}]", component.len())
            })
            .collect::<Vec<_>>();
        println!(
            "  - Verified overlap graph components: {}",
            summary.join(", ")
        );
    }
    let selected_component = components.first();
    let start_node = selected_component
        .into_iter()
        .flat_map(|component| component.iter().copied())
        .min_by(|&left, &right| {
            mst_adj
                .get(&left)
                .map_or(usize::MAX, |neighbors| neighbors.len())
                .cmp(
                    &mst_adj
                        .get(&right)
                        .map_or(usize::MAX, |neighbors| neighbors.len()),
                )
                .then_with(|| images[left].filename.cmp(&images[right].filename))
        })
        // `components` is empty only when `mst_adj` is empty, so this arm always
        // yields 0 in practice.  Take the lowest key rather than an arbitrary one
        // so that no hash-seed dependent value can ever reach the pose graph
        // (需求 14.6).
        .unwrap_or_else(|| mst_adj.keys().copied().min().unwrap_or(0));

    let mut ordered_indices = Vec::new();
    let mut global_homographies = HashMap::new();
    let mut q = VecDeque::new();
    let mut visited = HashSet::new();

    q.push_back((start_node, Matrix3::identity()));
    visited.insert(start_node);

    while let Some((u, h_u_global)) = q.pop_front() {
        ordered_indices.push(u);
        global_homographies.insert(u, h_u_global);

        if let Some(neighbors) = mst_adj.get(&u) {
            for &v in neighbors {
                if !visited.contains(&v) {
                    visited.insert(v);

                    let h_vu = if let Some(m) = matches.get(&(v, u)) {
                        m.homography
                    } else if let Some(m) = matches.get(&(u, v)) {
                        m.homography
                            .try_inverse()
                            .expect("Failed to invert homography for MST edge")
                    } else {
                        panic!("Match not found for MST edge between {} and {}", u, v);
                    };

                    let h_v_global = h_u_global * h_vu;
                    q.push_back((v, h_v_global));
                }
            }
        }
    }

    (ordered_indices, global_homographies)
}
