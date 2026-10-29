//! Capture geometry checks and compositor order for focus stacks.
//!
//! Per-capture and per-group geometry diagnostics, and the capture and
//! compositor orders derived from solved geometry or from the source order.

use crate::panorama_utils::stack_pipeline::determinism::sorted_keys;
use crate::panorama_utils::stack_pipeline::topology;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use super::{
    Dsu, FocusCaptureGroup, ImageInfo, MatchInfo, canonical_match_direction,
    focus_auto_order_motion_scale, focus_auto_order_path_motion_cost, focus_auto_order_path_score,
    focus_capture_groups, focus_compatible_focal_length, focus_image_direction_key,
    focus_overlap_quality, median_value, natural_path_cmp, panorama_model_linear_characteristics,
    panorama_transform_overlap_support, transformed_point,
};

pub(super) fn focus_capture_geometry_diagnostics(
    images: &[ImageInfo],
    order: &[usize],
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> (usize, usize, f64, f64) {
    let mut valid_edges = 0usize;
    let mut total_edges = 0usize;
    let mut overlap_values = Vec::new();
    let mut scales = Vec::new();
    for image in images {
        let Some(homography) = homographies.get(&image.id) else {
            continue;
        };
        let (scale, _, anisotropy) = panorama_model_linear_characteristics(homography);
        if scale.is_finite() && anisotropy.is_finite() {
            scales.push(scale);
        }
    }
    for pair in order.windows(2) {
        let source = pair[0];
        let target = pair[1];
        total_edges += 1;
        let Some(source_global) = homographies.get(&images[source].id) else {
            continue;
        };
        let Some(target_global) = homographies.get(&images[target].id) else {
            continue;
        };
        let Some(source_to_target) = target_global
            .try_inverse()
            .map(|inverse| inverse * source_global)
        else {
            continue;
        };
        let overlap = panorama_transform_overlap_support(
            &source_to_target,
            images[source].dimensions(),
            images[target].dimensions(),
        );
        let Some((intensity_ncc, edge_ncc, edge_orientation, samples)) =
            focus_overlap_quality(&images[source], &images[target], &source_to_target)
        else {
            continue;
        };
        let same_focal = focus_compatible_focal_length(&images[source], &images[target]);
        let scale = source_to_target[(0, 0)].abs();
        let accepted = same_focal
            && scale.is_finite()
            && (0.82..=1.18).contains(&scale)
            && overlap >= 0.10
            && samples >= 120
            && intensity_ncc >= 0.45
            && edge_ncc >= -0.10
            && edge_orientation >= -0.10;
        if accepted {
            valid_edges += 1;
            overlap_values.push(overlap);
        } else {
            println!(
                "  - Focus geometry retry candidate rejected edge '{}'<->'{}': scale {:.3}, overlap {:.1}%, NCC {:.3}, edges {:.3}/{:.3}, samples {}",
                Path::new(&images[source].filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                Path::new(&images[target].filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                scale,
                overlap * 100.0,
                intensity_ncc,
                edge_ncc,
                edge_orientation,
                samples,
            );
        }
    }
    let median_overlap = median_value(&mut overlap_values).unwrap_or(0.0);
    let median_scale = median_value(&mut scales).unwrap_or(0.0);
    (valid_edges, total_edges, median_overlap, median_scale)
}

pub(super) fn focus_capture_group_geometry_diagnostics(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    groups: &[FocusCaptureGroup],
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> (usize, usize, usize, usize, f64) {
    let mut group_for_image = vec![usize::MAX; images.len()];
    for (group_index, group) in groups.iter().enumerate() {
        for &image_index in &group.members {
            group_for_image[image_index] = group_index;
        }
    }
    // Both the station-pair order and the image-pair order inside a station pair
    // decide which candidate wins the `best_score` comparison that is reported in
    // the rejection diagnostics, so neither may follow the hash seed (需求 14.6).
    let mut relation_pairs = BTreeMap::<(usize, usize), Vec<(usize, usize)>>::new();
    for (source, target) in sorted_keys(matches) {
        let (Some(&source_group), Some(&target_group)) =
            (group_for_image.get(source), group_for_image.get(target))
        else {
            continue;
        };
        if source_group == usize::MAX || target_group == usize::MAX || source_group == target_group
        {
            continue;
        }
        relation_pairs
            .entry((
                source_group.min(target_group),
                source_group.max(target_group),
            ))
            .or_default()
            .push((source, target));
    }

    let candidate_relations = relation_pairs.len();
    let mut valid_relations = 0usize;
    let mut connectivity = Dsu::new(groups.len());
    let mut overlap_values = Vec::new();
    for ((left_group, right_group), image_pairs) in relation_pairs {
        let mut accepted_overlap = None::<f64>;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_summary = None;
        for (key_source, key_target) in image_pairs {
            // `relation_pairs` carries the `(min import index, max import index)`
            // orientation of the `matches` key, and everything below is
            // direction sensitive: `focus_overlap_quality` warps the source into
            // the target and correlates there, and the `scale` gate
            // `0.82..=1.18` is not symmetric around 1.0, so a relation measured
            // at 0.948 in one direction is measured at 1.055 in the other and a
            // near-threshold pair can be accepted one way and rejected the
            // other.  Measuring in the key direction therefore let a shuffled
            // import change which station relations pass this validation, which
            // in turn changed the accepted closure and every world pose.  Take
            // the same content derived direction the station solver takes
            // (需求 14.6).
            let (source, target, _) = canonical_match_direction(images, key_source, key_target);
            let (Some(source_global), Some(target_global)) = (
                homographies.get(&images[source].id),
                homographies.get(&images[target].id),
            ) else {
                continue;
            };
            let Some(source_to_target) = target_global
                .try_inverse()
                .map(|inverse| inverse * source_global)
            else {
                continue;
            };
            let overlap = panorama_transform_overlap_support(
                &source_to_target,
                images[source].dimensions(),
                images[target].dimensions(),
            );
            let Some((intensity_ncc, edge_ncc, edge_orientation, samples)) =
                focus_overlap_quality(&images[source], &images[target], &source_to_target)
            else {
                continue;
            };
            let scale = source_to_target[(0, 0)].abs();
            let accepted = focus_compatible_focal_length(&images[source], &images[target])
                && scale.is_finite()
                && (0.82..=1.18).contains(&scale)
                && overlap >= 0.10
                && samples >= 120
                && intensity_ncc >= 0.45
                && edge_ncc >= -0.10
                && edge_orientation >= -0.10;
            let score = intensity_ncc + edge_ncc.max(-0.10) + edge_orientation.max(-0.10);
            if score > best_score {
                best_score = score;
                best_summary = Some((
                    source,
                    target,
                    scale,
                    overlap,
                    intensity_ncc,
                    edge_ncc,
                    edge_orientation,
                    samples,
                ));
            }
            if accepted {
                accepted_overlap = Some(accepted_overlap.map_or(overlap, |best| best.max(overlap)));
            }
        }
        if let Some(overlap) = accepted_overlap {
            valid_relations += 1;
            connectivity.union(left_group, right_group);
            overlap_values.push(overlap);
        } else if let Some((
            source,
            target,
            scale,
            overlap,
            intensity_ncc,
            edge_ncc,
            edge_orientation,
            samples,
        )) = best_summary
        {
            println!(
                "  - Focus station relation rejected '{}'<->'{}': scale {:.3}, overlap {:.1}%, NCC {:.3}, edges {:.3}/{:.3}, samples {}",
                Path::new(&images[source].filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                Path::new(&images[target].filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                scale,
                overlap * 100.0,
                intensity_ncc,
                edge_ncc,
                edge_orientation,
                samples,
            );
        }
    }
    let mut component_sizes = HashMap::<usize, usize>::new();
    for group in 0..groups.len() {
        *component_sizes.entry(connectivity.find(group)).or_default() += 1;
    }
    let connected_groups = component_sizes.values().copied().max().unwrap_or(0);
    let median_overlap = median_value(&mut overlap_values).unwrap_or(0.0);
    (
        connected_groups,
        groups.len(),
        valid_relations,
        candidate_relations,
        median_overlap,
    )
}

pub(super) fn focus_capture_group_geometry_passes(
    connected_groups: usize,
    total_groups: usize,
    valid_relations: usize,
    median_overlap: f64,
) -> bool {
    total_groups > 1
        && connected_groups == total_groups
        && valid_relations + 1 >= total_groups
        && median_overlap >= 0.25
}

/// Build capture-order candidates from solved station centres.  This is used
/// for long scans when the import order and filenames are arbitrary.  The
/// station grid is inferred from the measured x/y centres; filenames only
/// break exact content ties inside one station.
pub(super) fn focus_geometry_capture_order(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<Vec<usize>> {
    let (groups, _) = focus_capture_groups(images, matches, homographies)?;
    if groups.len() < 2 {
        return None;
    }
    let centers = groups
        .iter()
        .map(|group| {
            let image = &images[group.anchor];
            let center = Point2::new(image.width as f64 * 0.5, image.height as f64 * 0.5);
            homographies
                .get(&image.id)
                .and_then(|pose| transformed_point(pose, center))
        })
        .collect::<Option<Vec<_>>>()?;

    let mut x_sorted = (0..groups.len()).collect::<Vec<_>>();
    x_sorted.sort_by(|left, right| {
        centers[*right]
            .x
            .total_cmp(&centers[*left].x)
            .then_with(|| centers[*left].y.total_cmp(&centers[*right].y))
    });
    let mut x_gaps = x_sorted
        .windows(2)
        .map(|pair| (centers[pair[0]].x - centers[pair[1]].x).abs())
        .filter(|gap| gap.is_finite() && *gap > 0.0)
        .collect::<Vec<_>>();
    x_gaps.sort_by(f64::total_cmp);
    let x_threshold = if x_gaps.len() >= 2 {
        let split = x_gaps
            .windows(2)
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                (right[1] / right[0].max(1.0)).total_cmp(&(left[1] / left[0].max(1.0)))
            })
            .map(|(index, _)| index);
        split
            .map(|index| (x_gaps[index] + x_gaps[index + 1]) * 0.5)
            .unwrap_or_else(|| x_gaps[x_gaps.len() / 2] * 2.0)
    } else {
        f64::INFINITY
    };
    let mut columns = Vec::<Vec<usize>>::new();
    for group_index in x_sorted.iter().copied() {
        let starts_new_column = columns.last().is_some_and(|column| {
            let previous = column.last().copied().unwrap_or(group_index);
            (centers[previous].x - centers[group_index].x).abs() > x_threshold
        });
        if starts_new_column {
            columns.push(Vec::new());
        }
        if columns.is_empty() {
            columns.push(Vec::new());
        }
        columns
            .last_mut()
            .expect("column was inserted")
            .push(group_index);
    }
    for column in &mut columns {
        column.sort_by(|left, right| {
            centers[*left]
                .y
                .total_cmp(&centers[*right].y)
                .then_with(|| centers[*right].x.total_cmp(&centers[*left].x))
        });
    }

    let mut group_orders = Vec::<Vec<usize>>::new();
    let mut append_columns = |reverse_columns: bool, reverse_rows: bool, alternate: bool| {
        let mut order = Vec::with_capacity(groups.len());
        let column_indices = if reverse_columns {
            (0..columns.len()).rev().collect::<Vec<_>>()
        } else {
            (0..columns.len()).collect::<Vec<_>>()
        };
        for (column_position, column_index) in column_indices.into_iter().enumerate() {
            let reverse = if alternate {
                reverse_rows ^ (column_position % 2 == 1)
            } else {
                reverse_rows
            };
            if reverse {
                order.extend(columns[column_index].iter().rev().copied());
            } else {
                order.extend(columns[column_index].iter().copied());
            }
        }
        group_orders.push(order);
    };
    append_columns(true, false, false);
    append_columns(true, true, false);
    append_columns(true, false, true);
    append_columns(true, true, true);
    append_columns(false, false, false);
    append_columns(false, true, false);

    // Keep the capture-order path as one candidate. It is useful when the
    // camera sequence is genuinely encoded in names, but it is scored by
    // measured station geometry below and therefore cannot fabricate a path
    // for renamed/shuffled imports.
    let mut filename_groups = (0..groups.len()).collect::<Vec<_>>();
    filename_groups.sort_by(|left, right| {
        let left_name = groups[*left]
            .members
            .iter()
            .map(|&index| images[index].filename.as_str())
            .min_by(|left, right| natural_path_cmp(left, right))
            .unwrap_or("");
        let right_name = groups[*right]
            .members
            .iter()
            .map(|&index| images[index].filename.as_str())
            .min_by(|left, right| natural_path_cmp(left, right))
            .unwrap_or("");
        natural_path_cmp(left_name, right_name)
    });
    group_orders.push(filename_groups);

    let mut nearest = vec![x_sorted.first().copied().unwrap_or(0)];
    let mut remaining = (0..groups.len()).collect::<HashSet<_>>();
    remaining.remove(&nearest[0]);
    while !remaining.is_empty() {
        let current = *nearest.last().expect("nearest path has a seed");
        let next = remaining
            .iter()
            .copied()
            .min_by(|left, right| {
                let left_distance = (centers[*left] - centers[current]).norm_squared();
                let right_distance = (centers[*right] - centers[current]).norm_squared();
                left_distance
                    .total_cmp(&right_distance)
                    .then_with(|| left.cmp(right))
            })
            .expect("remaining station exists");
        nearest.push(next);
        remaining.remove(&next);
    }
    group_orders.push(nearest);

    let mut best_order = None::<(f64, usize, f64, f64, Vec<usize>)>;
    for group_order in group_orders {
        let center_distance = group_order
            .windows(2)
            .map(|pair| (centers[pair[0]] - centers[pair[1]]).norm())
            .sum::<f64>();
        let mut expanded = Vec::with_capacity(images.len());
        for group_index in group_order {
            let mut members = groups[group_index].members.clone();
            members.sort_by(|left, right| {
                focus_image_direction_key(&images[*left])
                    .cmp(&focus_image_direction_key(&images[*right]))
            });
            expanded.extend(members);
        }
        if expanded.len() != images.len() {
            continue;
        }
        let (matched_edges, score) = focus_auto_order_path_score(
            &expanded,
            images,
            matches,
            focus_auto_order_motion_scale(images, matches),
        );
        let motion = focus_auto_order_path_motion_cost(&expanded, images, matches);
        if best_order.as_ref().is_none_or(
            |(best_distance, best_edges, best_score, best_motion, _)| {
                center_distance < *best_distance * 0.97
                    || (center_distance <= *best_distance * 1.03
                        && (score > *best_score
                            || (score.total_cmp(best_score).is_eq()
                                && (matched_edges > *best_edges
                                    || (matched_edges == *best_edges && motion < *best_motion)))))
            },
        ) {
            best_order = Some((center_distance, matched_edges, score, motion, expanded));
        }
    }
    best_order.map(|(_, _, _, _, order)| order)
}

pub(super) fn focus_geometry_group_compositor_order(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    homographies: &HashMap<usize, Matrix3<f64>>,
    groups: &[FocusCaptureGroup],
) -> Vec<usize> {
    let mut image_to_group = vec![usize::MAX; images.len()];
    for (group_index, group) in groups.iter().enumerate() {
        for &image_index in &group.members {
            if image_index < image_to_group.len() {
                image_to_group[image_index] = group_index;
            }
        }
    }
    let mut order = Vec::with_capacity(groups.len());
    let mut seen = vec![false; groups.len()];
    if let Some(source_order) = focus_geometry_capture_order(images, matches, homographies) {
        for image_index in source_order {
            let Some(&group_index) = image_to_group.get(image_index) else {
                continue;
            };
            if group_index < groups.len() && !seen[group_index] {
                seen[group_index] = true;
                order.push(group_index);
            }
        }
    }
    // The geometry path should cover every group. Keep a deterministic
    // geometric fallback for a partially solved component rather than
    // reintroducing filename order into the seam accumulator.
    if order.len() != groups.len() {
        let mut remaining = (0..groups.len())
            .filter(|&index| !seen[index])
            .collect::<Vec<_>>();
        remaining.sort_by(|left, right| {
            let left_anchor = &images[groups[*left].anchor];
            let right_anchor = &images[groups[*right].anchor];
            let center = |image: &ImageInfo| {
                homographies
                    .get(&image.id)
                    .and_then(|pose| {
                        transformed_point(
                            pose,
                            Point2::new(image.width as f64 * 0.5, image.height as f64 * 0.5),
                        )
                    })
                    .unwrap_or(Point2::new(f64::INFINITY, f64::INFINITY))
            };
            let left_center = center(left_anchor);
            let right_center = center(right_anchor);
            left_center
                .y
                .total_cmp(&right_center.y)
                .then_with(|| left_center.x.total_cmp(&right_center.x))
        });
        order.extend(remaining);
    }
    order
}

/// Decide the order in which Capture_Station virtual tiles enter the default
/// layered virtual-tile compositor.
///
/// The compositor places the first tile as the base and resolves every later
/// tile against the canvas built so far, so this order is part of the rendered
/// result: change it and the output pixels change.  It therefore has to be a
/// function of the measured geometry only (需求 14.6).
///
/// The previous implementation collapsed the *source* stitching path
/// (`ordered_indices`) to station ids.  That path is scored against image
/// evidence, but its orientation and several of its tie-breaks are filename
/// derived — `canonical_focus_order_direction` decides which end of a candidate
/// path comes first by comparing the first and last filename with
/// `natural_path_cmp`, and the greedy/graph builders break equal weights on the
/// import index.  Renaming the sources or importing them in reverse could
/// therefore reverse the whole accumulation even though every measured match was
/// identical, which is exactly what 需求 1.2/1.3/5.8 forbid: filename order may
/// only produce *candidate* orders and break exact ties, never stand in for
/// geometry.
///
/// Stations are ordered by the production `StationTopology`: row index
/// ascending (top to bottom), then column index ascending (right to left,
/// because topology inference numbers columns by descending world `x`). This
/// makes the scan convention explicit at the seam accumulator instead of
/// approximating it with continuous projected centres.
///
/// Requirement 5.1 makes `(row, column)` unique for every inferred station, so
/// filenames, source paths and import ranks never participate. The station
/// index is consulted only as the deterministic final key for an exact
/// topology-coordinate tie, which can occur only in malformed or synthetic
/// input; production topology remains rename/import-order invariant.
pub(super) fn focus_source_order_group_compositor_order(
    station_count: usize,
    station_topology: &topology::StationTopology,
) -> Vec<usize> {
    debug_assert_eq!(
        station_topology.row.len(),
        station_count,
        "topology must contain one row index per station"
    );
    debug_assert_eq!(
        station_topology.column.len(),
        station_count,
        "topology must contain one column index per station"
    );

    let topology_key = |station: usize| {
        (
            station_topology
                .row
                .get(station)
                .copied()
                .unwrap_or(u32::MAX),
            station_topology
                .column
                .get(station)
                .copied()
                .unwrap_or(u32::MAX),
        )
    };
    let mut order = (0..station_count).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        let left_topology = topology_key(left);
        let right_topology = topology_key(right);
        left_topology
            // Row indices increase from top to bottom. Column indices increase
            // from right to left, so their ascending lexicographic order is the
            // required scan convention inferred by Capture_Topology_Model.
            .cmp(&right_topology)
            // Requirement 5.1 makes topology coordinates unique in production.
            // Keep only this content-derived station index as a deterministic
            // final key for malformed/ambiguous exact topology ties; filenames,
            // import order and continuous world coordinates never participate.
            .then_with(|| left.cmp(&right))
    });
    order
}

pub(super) fn focus_capture_geometry_passes(
    images: &[ImageInfo],
    valid_edges: usize,
    total_edges: usize,
    median_overlap: f64,
    homographies: &HashMap<usize, Matrix3<f64>>,
) -> bool {
    if total_edges == 0
        // Filename neighbours include transitions between focus brackets and
        // camera positions. Keep this strict: a merely connected group graph
        // can still fold a long scroll into a visually invalid 2D collage.
        // Consecutive files can straddle capture stations and opposite focal
        // planes, so direct pixel correlation is not expected at every group
        // boundary. Group geometry has already required independent
        // multi-layer consensus and a connected robust closure solve.
        || valid_edges.saturating_mul(100) < total_edges.saturating_mul(85)
        || median_overlap < 0.25
    {
        return false;
    }
    let mut scales = images
        .iter()
        .filter_map(|image| {
            homographies
                .get(&image.id)
                .map(panorama_model_linear_characteristics)
                .map(|(scale, _, _)| scale)
                .filter(|scale| scale.is_finite())
        })
        .collect::<Vec<_>>();
    if scales.is_empty() {
        return false;
    }
    let min_scale = scales.iter().copied().fold(f64::INFINITY, f64::min);
    let max_scale = scales.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let Some(median_scale) = median_value(&mut scales) else {
        return false;
    };
    // Absolute scale depends on which source becomes the graph reference and
    // legitimately accumulates when a long scan is photographed from several
    // distances. Cross-station pixel agreement and graph connectivity are
    // validated separately. Keep this as an outlier guard, not as an
    // assumption that every camera station has nearly identical magnification.
    // A 25% scale outlier is large enough to indicate a folded or mixed-lens
    // chain rather than ordinary focus breathing. Long scans can accumulate
    // scale in both directions, so a low-side excursion is accepted when the
    // same scan also has corroborating high-side spread. A lone low-side
    // excursion remains rejected (the synthetic 0.78 drift case exercises
    // that guard).
    let minimum_allowed = median_scale * 0.75;
    let maximum_allowed = median_scale * 1.30;
    let one_sided_low_drift = min_scale < median_scale * 0.80 && max_scale < median_scale * 1.10;
    let scale_ok =
        min_scale >= minimum_allowed && max_scale <= maximum_allowed && !one_sided_low_drift;
    if !scale_ok {
        println!(
            "  - Focus geometry scale range rejected: {min_scale:.3}..{max_scale:.3} around median {median_scale:.3} (expected {minimum_allowed:.3}..{maximum_allowed:.3})"
        );
    }
    scale_ok
}
