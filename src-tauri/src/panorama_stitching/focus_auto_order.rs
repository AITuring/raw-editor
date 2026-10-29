//! Automatic focus-stack ordering.
//!
//! Motion- and quality-based edge weights, path scores, the greedy path, the
//! sequence homographies and the dense continuous-scan check.

use nalgebra::Matrix3;
use nalgebra::Point2;
use std::cmp::Ordering;
use std::collections::HashMap;

use super::{
    FOCUS_MODEL_INLIER_THRESHOLD, FOCUS_MODEL_MIN_INLIERS, ImageInfo, MatchInfo,
    focus_match_center_motion_ratio, median_symmetric_error, median_value, natural_path_cmp,
    transformed_point,
};

// A real neighbouring capture can move by almost one frame width when the
// camera advances to the next tile. A descriptor match that moves the frame
// farther than this is almost certainly a repeated character/seal match.
pub(super) const FOCUS_SEQUENCE_MAX_LINK_MOTION_RATIO: f64 = 1.35;

pub(super) const FOCUS_AUTO_ORDER_MOTION_SCALE_FLOOR: f64 = 0.01;

pub(super) const FOCUS_AUTO_ORDER_MOTION_EXPONENT: f64 = 4.0;

pub(super) const FOCUS_AUTO_ORDER_GAP_PENALTY: f64 = 4.0;

pub(super) const FOCUS_AUTO_ORDER_MISSING_EDGE_PENALTY: f64 = 32.0;

pub(super) fn focus_auto_order_motion_scale(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> f64 {
    let mut minimum_motion_by_image = vec![f64::INFINITY; images.len()];
    // Direct map iteration is order safe here: the loop only takes a per-image
    // minimum and the result is the median of a `total_cmp`-sorted vector, both of
    // which are independent of the visiting order (需求 14.6).
    for (&(source_index, target_index), match_info) in matches {
        let Some(source) = images.get(source_index) else {
            continue;
        };
        let Some(target) = images.get(target_index) else {
            continue;
        };
        let Some(motion) = focus_match_center_motion_ratio(source, target, match_info) else {
            continue;
        };
        if motion <= FOCUS_AUTO_ORDER_MOTION_SCALE_FLOOR * 0.1 {
            continue;
        }
        minimum_motion_by_image[source_index] = minimum_motion_by_image[source_index].min(motion);
        minimum_motion_by_image[target_index] = minimum_motion_by_image[target_index].min(motion);
    }
    let mut finite_minimums = minimum_motion_by_image
        .into_iter()
        .filter(|motion| motion.is_finite())
        .collect::<Vec<_>>();
    median_value(&mut finite_minimums).unwrap_or(0.0)
}

pub(super) fn focus_auto_order_edge_weight(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
    motion_scale: f64,
) -> f64 {
    let support = (match_info.inliers.max(1) as f64).sqrt();
    let precision = if match_info.points.len() >= 4 {
        let error = median_symmetric_error(&match_info.homography, &match_info.points);
        if error.is_finite() {
            1.0 / (1.0 + error / FOCUS_MODEL_INLIER_THRESHOLD)
        } else {
            0.0
        }
    } else {
        1.0
    };
    let motion_factor = if motion_scale > FOCUS_AUTO_ORDER_MOTION_SCALE_FLOOR {
        let motion =
            focus_match_center_motion_ratio(source, target, match_info).unwrap_or(f64::INFINITY);
        if motion.is_finite() {
            1.0 / (1.0 + (motion / motion_scale).powf(FOCUS_AUTO_ORDER_MOTION_EXPONENT))
        } else {
            0.0
        }
    } else {
        1.0
    };
    support * precision * motion_factor
}

pub(super) fn focus_stack_link_transform(
    matches: &HashMap<(usize, usize), MatchInfo>,
    from: usize,
    to: usize,
) -> Option<(usize, Matrix3<f64>)> {
    if let Some(match_info) = matches.get(&(from, to)) {
        return Some((match_info.inliers, match_info.homography));
    }
    matches.get(&(to, from)).and_then(|match_info| {
        match_info
            .homography
            .try_inverse()
            .map(|transform| (match_info.inliers, transform))
    })
}

pub(super) fn focus_cycle_edge_factor(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    source_index: usize,
    target_index: usize,
) -> f64 {
    let Some((_, direct)) = focus_stack_link_transform(matches, source_index, target_index) else {
        return 0.0;
    };
    let Some(source) = images.get(source_index) else {
        return 0.0;
    };
    let Some(target) = images.get(target_index) else {
        return 0.0;
    };
    let samples = [
        Point2::new(source.width as f64 * 0.2, source.height as f64 * 0.2),
        Point2::new(source.width as f64 * 0.8, source.height as f64 * 0.2),
        Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5),
        Point2::new(source.width as f64 * 0.2, source.height as f64 * 0.8),
        Point2::new(source.width as f64 * 0.8, source.height as f64 * 0.8),
    ];
    let normalization = target.width.max(target.height).max(1) as f64;
    let mut consistent_weight = 0.0;
    let mut conflicting_weight = 0.0;
    for middle_index in 0..images.len() {
        if middle_index == source_index || middle_index == target_index {
            continue;
        }
        let Some((source_support, source_to_middle)) =
            focus_stack_link_transform(matches, source_index, middle_index)
        else {
            continue;
        };
        let Some((target_support, target_to_middle)) =
            focus_stack_link_transform(matches, target_index, middle_index)
        else {
            continue;
        };
        let Some(middle_to_target) = target_to_middle.try_inverse() else {
            continue;
        };
        let via_middle = middle_to_target * source_to_middle;
        let mut errors = samples
            .iter()
            .filter_map(|&point| {
                let direct_point = transformed_point(&direct, point)?;
                let indirect_point = transformed_point(&via_middle, point)?;
                let error = (direct_point - indirect_point).norm() / normalization;
                error.is_finite().then_some(error)
            })
            .collect::<Vec<_>>();
        let Some(error) = median_value(&mut errors) else {
            continue;
        };
        let weight = source_support.min(target_support).max(1) as f64;
        if error <= 0.025 {
            consistent_weight += weight.sqrt();
        } else if error >= 0.075 {
            conflicting_weight += weight.sqrt();
        }
    }
    if consistent_weight == 0.0 && conflicting_weight == 0.0 {
        return 1.0;
    }
    ((consistent_weight + 4.0) / (conflicting_weight + 4.0)).clamp(0.15, 3.0)
}

pub(super) fn focus_auto_order_edge_weight_between(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    left: usize,
    right: usize,
    motion_scale: f64,
) -> Option<f64> {
    if let Some(match_info) = matches.get(&(left, right)) {
        return Some(focus_auto_order_edge_weight(
            &images[left],
            &images[right],
            match_info,
            motion_scale,
        ));
    }
    matches.get(&(right, left)).map(|match_info| {
        focus_auto_order_edge_weight(&images[right], &images[left], match_info, motion_scale)
    })
}

pub(super) fn focus_auto_order_path_score(
    order: &[usize],
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    motion_scale: f64,
) -> (usize, f64) {
    let mut matched_edges = 0usize;
    let mut score = 0.0;
    for pair in order.windows(2) {
        if let Some(edge_weight) =
            focus_auto_order_edge_weight_between(images, matches, pair[0], pair[1], motion_scale)
        {
            matched_edges += 1;
            score += edge_weight;
        } else {
            score -= FOCUS_AUTO_ORDER_MISSING_EDGE_PENALTY;
        }
    }
    (matched_edges, score)
}

pub(super) fn focus_auto_order_path_motion_cost(
    order: &[usize],
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> f64 {
    order
        .windows(2)
        .filter_map(|pair| {
            if let Some(match_info) = matches.get(&(pair[0], pair[1])) {
                focus_match_center_motion_ratio(&images[pair[0]], &images[pair[1]], match_info)
            } else {
                matches.get(&(pair[1], pair[0])).and_then(|match_info| {
                    focus_match_center_motion_ratio(&images[pair[1]], &images[pair[0]], match_info)
                })
            }
        })
        .sum()
}

pub(super) fn focus_auto_order_path_max_motion(
    order: &[usize],
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> f64 {
    let mut maximum = 0.0f64;
    for pair in order.windows(2) {
        let Some(motion) = matches
            .get(&(pair[0], pair[1]))
            .and_then(|match_info| {
                focus_match_center_motion_ratio(&images[pair[0]], &images[pair[1]], match_info)
            })
            .or_else(|| {
                matches.get(&(pair[1], pair[0])).and_then(|match_info| {
                    focus_match_center_motion_ratio(&images[pair[1]], &images[pair[0]], match_info)
                })
            })
        else {
            return f64::INFINITY;
        };
        maximum = maximum.max(motion);
    }
    maximum
}

pub(super) fn canonical_focus_order_direction(order: &[usize], images: &[ImageInfo]) -> Vec<usize> {
    if order.len() < 2 {
        return order.to_vec();
    }
    let first = order[0];
    let last = *order
        .last()
        .expect("an order with two images has a last image");
    let direction = natural_path_cmp(&images[first].filename, &images[last].filename)
        .then_with(|| first.cmp(&last));
    if direction == Ordering::Greater {
        order.iter().rev().copied().collect()
    } else {
        order.to_vec()
    }
}

pub(super) fn focus_auto_order_adjacency(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    motion_scale: f64,
) -> Vec<Vec<(usize, f64)>> {
    let mut adjacency = vec![Vec::new(); images.len()];
    // Direct map iteration is order safe here: every neighbour list is sorted
    // below by `(weight desc, natural path, index)`, which is a total order, so the
    // insertion order cannot survive into the traversal (需求 14.6).
    for (&(left, right), match_info) in matches {
        if left >= images.len() || right >= images.len() || left == right {
            continue;
        }
        let weight =
            focus_auto_order_edge_weight(&images[left], &images[right], match_info, motion_scale);
        if !weight.is_finite() {
            continue;
        }
        adjacency[left].push((right, weight));
        adjacency[right].push((left, weight));
    }
    for neighbors in &mut adjacency {
        neighbors.sort_by(|(left_index, left_weight), (right_index, right_weight)| {
            right_weight
                .total_cmp(left_weight)
                .then_with(|| {
                    natural_path_cmp(
                        &images[*left_index].filename,
                        &images[*right_index].filename,
                    )
                })
                .then_with(|| left_index.cmp(right_index))
        });
    }
    adjacency
}

pub(super) fn focus_auto_order_greedy_path(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    motion_scale: f64,
) -> Vec<usize> {
    if images.len() < 2 {
        return (0..images.len()).collect();
    }
    let adjacency = focus_auto_order_adjacency(images, matches, motion_scale);
    let mut best_order = Vec::new();
    let mut best_matched_edges = 0usize;
    let mut best_score = f64::NEG_INFINITY;

    for start in 0..images.len() {
        let mut order = vec![start];
        let mut used = vec![false; images.len()];
        used[start] = true;
        while order.len() < images.len() {
            let current = *order
                .last()
                .expect("a greedy path always has a current image");
            let next = adjacency[current]
                .iter()
                .filter(|(neighbor, _)| !used[*neighbor])
                .max_by(|(left, left_weight), (right, right_weight)| {
                    let left_future = adjacency[*left]
                        .iter()
                        .filter(|(neighbor, _)| !used[*neighbor] && *neighbor != current)
                        .map(|(_, weight)| *weight)
                        .max_by(f64::total_cmp)
                        .unwrap_or(0.0);
                    let right_future = adjacency[*right]
                        .iter()
                        .filter(|(neighbor, _)| !used[*neighbor] && *neighbor != current)
                        .map(|(_, weight)| *weight)
                        .max_by(f64::total_cmp)
                        .unwrap_or(0.0);
                    (left_weight + left_future * 0.25)
                        .total_cmp(&(right_weight + right_future * 0.25))
                        .then_with(|| {
                            natural_path_cmp(&images[*left].filename, &images[*right].filename)
                                .reverse()
                        })
                })
                .map(|(neighbor, _)| *neighbor)
                .or_else(|| {
                    (0..images.len())
                        .filter(|neighbor| !used[*neighbor])
                        .min_by(|left, right| {
                            natural_path_cmp(&images[*left].filename, &images[*right].filename)
                                .then_with(|| left.cmp(right))
                        })
                });
            let Some(next) = next else {
                break;
            };
            used[next] = true;
            order.push(next);
        }

        let (matched_edges, score) =
            focus_auto_order_path_score(&order, images, matches, motion_scale);
        let motion_cost = focus_auto_order_path_motion_cost(&order, images, matches);
        let best_motion_cost = focus_auto_order_path_motion_cost(&best_order, images, matches);
        if best_order.is_empty()
            || matched_edges > best_matched_edges
            || (matched_edges == best_matched_edges
                && (motion_cost < best_motion_cost
                    || (motion_cost == best_motion_cost && score > best_score)))
        {
            best_order = order;
            best_matched_edges = matched_edges;
            best_score = score;
        }
    }
    best_order
}

pub(super) fn build_focus_sequence_homographies(
    order: &[usize],
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    motion_scale: f64,
) -> Option<HashMap<usize, Matrix3<f64>>> {
    let first = *order.first()?;
    let mut global_homographies = HashMap::new();
    global_homographies.insert(first, Matrix3::identity());

    for (position, &current) in order.iter().enumerate().skip(1) {
        let mut best_link: Option<(f64, f64, usize, f64, usize, Matrix3<f64>)> = None;
        // Preserve the capture path. Searching every earlier frame lets a
        // repeated character/seal beat the real neighbour and teleports the
        // next tile to a distant row. Only the immediate predecessor and one
        // frame of slack for a deleted/missing capture may supply the link.
        for previous_position in (position.saturating_sub(2)..position).rev() {
            let previous = order[previous_position];
            let Some((_, current_to_previous)) =
                focus_stack_link_transform(matches, current, previous)
            else {
                continue;
            };
            let Some(motion_ratio) = focus_sequence_link_motion_ratio(
                &images[current],
                &images[previous],
                &current_to_previous,
            ) else {
                continue;
            };
            let is_sequence_bridge = matches
                .get(&(current, previous))
                .or_else(|| matches.get(&(previous, current)))
                .is_some_and(|match_info| match_info.sequence_bridge);
            if !is_sequence_bridge && motion_ratio > FOCUS_SEQUENCE_MAX_LINK_MOTION_RATIO {
                continue;
            }
            let Some(edge_weight) = focus_auto_order_edge_weight_between(
                images,
                matches,
                current,
                previous,
                motion_scale,
            ) else {
                continue;
            };
            let gap = position - previous_position;
            let score = edge_weight - gap.saturating_sub(1) as f64 * FOCUS_AUTO_ORDER_GAP_PENALTY;
            let should_replace = best_link.as_ref().is_none_or(
                |(best_score, best_weight, best_gap, best_motion, _, _)| {
                    score > *best_score
                        || (score == *best_score
                            && (edge_weight > *best_weight
                                || (edge_weight == *best_weight
                                    && (gap < *best_gap
                                        || (gap == *best_gap && motion_ratio < *best_motion)))))
                },
            );
            if should_replace {
                best_link = Some((
                    score,
                    edge_weight,
                    gap,
                    motion_ratio,
                    previous,
                    current_to_previous,
                ));
            }
        }
        let (_, _, _, _, previous, current_to_previous) = best_link?;
        let previous_global = global_homographies.get(&previous).copied()?;
        global_homographies.insert(current, previous_global * current_to_previous);
    }
    Some(global_homographies)
}

pub(super) fn focus_sequence_link_motion_ratio(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
) -> Option<f64> {
    let source_center = Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5);
    let target_center = Point2::new(target.width as f64 * 0.5, target.height as f64 * 0.5);
    let mapped_center = transformed_point(source_to_target, source_center)?;
    let width = target.width.max(1) as f64;
    let height = target.height.max(1) as f64;
    Some(
        ((mapped_center.x - target_center.x) / width)
            .hypot((mapped_center.y - target_center.y) / height),
    )
}

pub(super) fn focus_match_graph_is_dense_continuous_scan(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> bool {
    if images.len() < 9 {
        return false;
    }
    let possible_edges = images.len().saturating_mul(images.len() - 1) / 2;
    if possible_edges == 0 {
        return false;
    }
    let mut degrees = vec![0usize; images.len()];
    let verified_edges = matches
        .iter()
        .filter(|((source, target), match_info)| {
            let verified = *source < images.len()
                && *target < images.len()
                && source != target
                && !match_info.sequence_bridge
                && !match_info.coarse_bridge
                && match_info.points.len() >= FOCUS_MODEL_MIN_INLIERS;
            if verified {
                degrees[*source] += 1;
                degrees[*target] += 1;
            }
            verified
        })
        .count();
    let minimum_degree = images.len().div_ceil(4);
    verified_edges.saturating_mul(100) >= possible_edges.saturating_mul(45)
        && degrees.iter().all(|degree| *degree >= minimum_degree)
}
