//! Focus-stack capture grouping, geometry ordering, and disconnected-component rescue.
//!
//! This module owns the evidence that decides which frames share a camera
//! station, how capture groups are ordered, and how strict rescue bridges are
//! recovered when the initial focus graph is disconnected.

use super::*;

fn focus_capture_number_gap(source: &ImageInfo, target: &ImageInfo) -> Option<u64> {
    trailing_capture_number(&source.filename)
        .zip(trailing_capture_number(&target.filename))
        .map(|(left, right)| left.abs_diff(right))
}

pub(super) fn focus_compatible_focal_length(source: &ImageInfo, target: &ImageInfo) -> bool {
    match (source.focal_length_35mm, target.focal_length_35mm) {
        (Some(left), Some(right))
            if left.is_finite() && right.is_finite() && left > 0.0 && right > 0.0 =>
        {
            (left / right).max(right / left) <= 1.03
        }
        _ => true,
    }
}

pub(super) fn focus_adjacent_same_focal_pair(source: &ImageInfo, target: &ImageInfo) -> bool {
    focus_capture_number_gap(source, target).is_some_and(|gap| gap == 1)
        && focus_compatible_focal_length(source, target)
}

pub(super) fn recover_adjacent_focus_bracket_match(
    source: &ImageInfo,
    target: &ImageInfo,
    log_match: bool,
) -> Option<MatchInfo> {
    let adjacent_capture = trailing_capture_number(&source.filename)
        .zip(trailing_capture_number(&target.filename))
        .is_some_and(|(left, right)| left.abs_diff(right) == 1);
    if !adjacent_capture {
        return None;
    }
    let compatible_focal_length = source
        .focal_length_35mm
        .zip(target.focal_length_35mm)
        .map(|(left, right)| {
            left.is_finite()
                && right.is_finite()
                && left > 0.0
                && right > 0.0
                && (left / right).max(right / left) <= 1.03
        })
        // Missing EXIF must not disable focus-bracket recovery. The strict
        // same-pose checks below remain the authority.
        .unwrap_or(true);
    if !compatible_focal_length {
        return None;
    }
    let source_focal = source
        .focal_length_35mm
        .or(target.focal_length_35mm)
        .unwrap_or(50.0);
    let target_focal = target
        .focal_length_35mm
        .or(source.focal_length_35mm)
        .unwrap_or(source_focal);
    let (homography, points, score) = estimate_mixed_focal_coarse_registration(
        source,
        target,
        source_focal,
        target_focal,
        true,
        log_match,
    )?;
    // This fallback is only allowed to attach another focal plane at the same
    // camera pose. A spatial scan step or repeated motif must continue through
    // the ordinary feature/cycle-verified panorama graph.
    let scale = homography[(0, 0)].abs();
    let source_center = Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5);
    let target_center = Point2::new(target.width as f64 * 0.5, target.height as f64 * 0.5);
    let mapped_center = transformed_point(&homography, source_center)?;
    let normalization = source
        .width
        .max(source.height)
        .max(target.width.max(target.height))
        .max(1) as f64;
    let center_motion = (mapped_center - target_center).norm() / normalization;
    let overlap =
        panorama_transform_overlap_support(&homography, source.dimensions(), target.dimensions());
    // The overlap estimator deliberately ignores an outer validation margin;
    // identical full-frame images therefore report about 0.64, not 1.0.
    if !(0.88..=1.12).contains(&scale) || center_motion > 0.12 || overlap < 0.60 {
        if log_match {
            println!(
                "  - Rejecting adjacent focus-layer recovery: scale {scale:.3}, motion {:.3}%, overlap {:.1}%",
                center_motion * 100.0,
                overlap * 100.0,
            );
        }
        return None;
    }
    if log_match {
        println!(
            "  - Recovered same-pose adjacent focus layer: '{}' <-> '{}' (NCC {:.3}, motion {:.3}%, overlap {:.1}%)",
            Path::new(&source.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            Path::new(&target.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            score,
            center_motion * 100.0,
            overlap * 100.0,
        );
    }
    Some(MatchInfo {
        homography,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: true,
        points: points.clone(),
        candidate_points: points,
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
    })
}

pub(super) fn focus_alignment_support(image: &ImageInfo) -> usize {
    image.features.len().max(image.top_features.len())
}

pub(super) fn focus_position_recovery_accepts(
    capture_gap: u64,
    same_focal: bool,
    scale: f64,
    center_motion: f64,
    overlap: f64,
    samples: usize,
    intensity_ncc: f64,
    edge_ncc: f64,
    edge_orientation: f64,
    orphan_attach: bool,
) -> bool {
    let scale_ok = if same_focal {
        (0.86..=1.14).contains(&scale)
    } else {
        (MIXED_FOCAL_SCALE_MIN_RATIO..=MIXED_FOCAL_SCALE_MAX_RATIO).contains(&scale)
    };
    // A small NCC patch is useful for a same-pose focus layer, but cannot
    // establish that two scan positions are neighbours. Position bridges are
    // retried with feature/RANSAC matching instead of this relaxed path.
    if !scale_ok || center_motion <= 0.05 || center_motion > 0.92 || overlap < 0.10 || samples < 300
    {
        return false;
    }
    if orphan_attach && capture_gap <= 2 && intensity_ncc >= 0.74 && edge_ncc >= 0.12 {
        return true;
    }
    if capture_gap == 1 && intensity_ncc >= 0.62 && edge_ncc >= 0.35 && overlap >= 0.20 {
        return true;
    }
    if intensity_ncc < 0.70 || edge_ncc < 0.14 || overlap < 0.12 || samples < 400 {
        return false;
    }
    edge_orientation >= 0.16
        || (capture_gap <= 2 && intensity_ncc >= 0.85 && edge_ncc >= 0.20)
        || (capture_gap == 1 && intensity_ncc >= 0.78 && edge_ncc >= 0.16)
}

pub(super) fn focus_capture_stations(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Vec<Vec<usize>> {
    let mut dsu = Dsu::new(images.len());
    for (&(left, right), match_info) in sorted_pairs(matches) {
        if left >= images.len() || right >= images.len() {
            continue;
        }
        if focus_match_is_capture_station_link(images, left, right, match_info) {
            dsu.union(left, right);
        }
    }
    // Grouping by root in an ordered map makes the station order depend on the
    // representative image index instead of the hash seed (需求 14.6).
    let mut by_root = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..images.len() {
        by_root.entry(dsu.find(index)).or_default().push(index);
    }
    let mut stations = by_root.into_values().collect::<Vec<_>>();
    for station in &mut stations {
        station.sort_by(|&left, &right| {
            natural_path_cmp(&images[left].filename, &images[right].filename)
                .then_with(|| left.cmp(&right))
        });
    }
    stations.sort_by(|left, right| {
        let left_number = left
            .iter()
            .filter_map(|&index| trailing_capture_number(&images[index].filename))
            .min()
            .unwrap_or(u64::MAX);
        let right_number = right
            .iter()
            .filter_map(|&index| trailing_capture_number(&images[index].filename))
            .min()
            .unwrap_or(u64::MAX);
        // Sources without a trailing capture number all collapse to `u64::MAX`.
        // The members themselves are the stable secondary key (需求 14.6).
        left_number
            .cmp(&right_number)
            .then_with(|| left.first().cmp(&right.first()))
    });
    stations
}

pub(super) fn attach_focus_bracket_orphans_to_pose_graph(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    ordered_indices: &mut Vec<usize>,
    global_homographies: &mut HashMap<usize, Matrix3<f64>>,
) -> usize {
    let mut attached = 0usize;
    let mut pending: Vec<usize> = (0..images.len())
        .filter(|index| !global_homographies.contains_key(index))
        .collect();
    pending.sort_by(|&left, &right| {
        natural_path_cmp(&images[left].filename, &images[right].filename)
            .then_with(|| left.cmp(&right))
    });
    while !pending.is_empty() {
        let mut progress = false;
        let mut next_pending = Vec::new();
        for orphan in pending {
            let mut best: Option<(usize, Matrix3<f64>, usize)> = None;
            for anchor in 0..images.len() {
                if anchor == orphan || !global_homographies.contains_key(&anchor) {
                    continue;
                }
                let key = (anchor.min(orphan), anchor.max(orphan));
                let Some(match_info) = matches.get(&key) else {
                    continue;
                };
                let (source, target) = if anchor < orphan {
                    (&images[anchor], &images[orphan])
                } else {
                    (&images[orphan], &images[anchor])
                };
                if !focus_match_is_local_bracket(source, target, match_info) {
                    continue;
                }
                let Some((_, anchor_to_orphan)) =
                    focus_stack_link_transform(matches, anchor, orphan)
                else {
                    continue;
                };
                let homography = global_homographies[&anchor] * anchor_to_orphan;
                let anchor_support = focus_alignment_support(&images[anchor]);
                if best.as_ref().is_none_or(|(_, _, best_support)| {
                    anchor_support > *best_support
                        || (anchor_support == *best_support && anchor < orphan)
                }) {
                    best = Some((anchor, homography, anchor_support));
                }
            }
            if let Some((_, homography, _)) = best {
                global_homographies.insert(orphan, homography);
                if !ordered_indices.contains(&orphan) {
                    ordered_indices.push(orphan);
                }
                attached += 1;
                progress = true;
            } else {
                next_pending.push(orphan);
            }
        }
        if !progress {
            break;
        }
        pending = next_pending;
    }
    attached
}

pub(super) fn focus_station_representative_indices(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> HashSet<usize> {
    focus_capture_stations(images, matches)
        .into_iter()
        .map(|station| focus_station_representative(&station, images))
        .collect()
}

fn focus_stack_spine_edge(
    images: &[ImageInfo],
    left: usize,
    right: usize,
    match_info: &MatchInfo,
    representatives: &HashSet<usize>,
) -> bool {
    if match_info.sequence_bridge {
        return false;
    }
    if focus_match_is_local_bracket(&images[left], &images[right], match_info) {
        return true;
    }
    if !match_info.coarse_bridge {
        return match_info.inliers.max(match_info.points.len()) >= FOCUS_MODEL_MIN_INLIERS;
    }
    // Coarse NCC may recover another focus plane at one pose, but it is not
    // sufficient evidence for a scan-position edge. Filename-adjacent frames
    // can straddle a raster reset and share a repeated seal or figure. Keep
    // coarse position bridges representative-to-representative only; the
    // remaining position edges must come from feature/RANSAC evidence.
    // A multi-layer consensus bridge is still excluded from bundle
    // adjustment, but it is safe enough to seed the pose graph. The marker is
    // kept in `top_candidate_points`; ordinary coarse bridges leave that list
    // empty and remain representative-gated.
    let consensus_coarse = !match_info.top_candidate_points.is_empty();
    if !consensus_coarse && (!representatives.contains(&left) || !representatives.contains(&right))
    {
        return false;
    }
    let Some(motion) = focus_match_center_motion_ratio(&images[left], &images[right], match_info)
    else {
        return false;
    };
    if motion <= FOCUS_BRACKET_MAX_CENTER_MOTION_RATIO {
        return false;
    }
    let capture_gap = focus_capture_number_gap(&images[left], &images[right]).unwrap_or(u64::MAX);
    let same_focal = focus_compatible_focal_length(&images[left], &images[right]);
    let scale = match_info.homography[(0, 0)].abs();
    let overlap = panorama_transform_overlap_support(
        &match_info.homography,
        images[left].dimensions(),
        images[right].dimensions(),
    );
    if consensus_coarse {
        true
    } else {
        focus_overlap_quality(&images[left], &images[right], &match_info.homography).is_some_and(
            |(intensity_ncc, edge_ncc, edge_orientation, samples)| {
                focus_position_recovery_accepts(
                    capture_gap,
                    same_focal,
                    scale,
                    motion,
                    overlap,
                    samples,
                    intensity_ncc,
                    edge_ncc,
                    edge_orientation,
                    false,
                )
            },
        )
    }
}

pub(super) fn focus_stack_spine_matches(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> HashMap<(usize, usize), MatchInfo> {
    if images.len() < SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES {
        return matches.clone();
    }
    let representatives = focus_station_representative_indices(images, matches);
    matches
        .iter()
        .filter(|(key, match_info)| {
            focus_stack_spine_edge(images, key.0, key.1, match_info, &representatives)
        })
        .map(|(key, match_info)| (*key, match_info.clone()))
        .collect()
}

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

pub(super) fn focus_rescue_match_is_verified(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
) -> bool {
    if match_info.sequence_bridge
        || match_info.coarse_bridge
        || match_info.points.len() < FOCUS_MODEL_MIN_INLIERS
    {
        return false;
    }
    let scale = match_info.homography[(0, 0)].abs();
    let overlap = panorama_transform_overlap_support(
        &match_info.homography,
        source.dimensions(),
        target.dimensions(),
    );
    let Some((intensity_ncc, edge_ncc, edge_orientation, samples)) =
        focus_overlap_quality(source, target, &match_info.homography)
    else {
        return false;
    };
    scale.is_finite()
        && (0.65..=1.45).contains(&scale)
        && overlap >= 0.18
        && samples >= 180
        && intensity_ncc >= 0.45
        && edge_ncc >= -0.05
        && edge_orientation >= -0.05
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

fn focus_station_representative(station: &[usize], images: &[ImageInfo]) -> usize {
    station
        .iter()
        .copied()
        .max_by_key(|&index| {
            (
                focus_alignment_support(&images[index]),
                std::cmp::Reverse(trailing_capture_number(&images[index].filename).unwrap_or(0)),
            )
        })
        .unwrap_or(station[0])
}

fn focus_indices_share_component(left: usize, right: usize, components: &[Vec<usize>]) -> bool {
    components
        .iter()
        .any(|component| component.contains(&left) && component.contains(&right))
}

fn insert_recovered_focus_match(
    images: &[ImageInfo],
    left_index: usize,
    right_index: usize,
    log_match: bool,
    orphan_attach: bool,
) -> Option<((usize, usize), MatchInfo)> {
    let (source, target, invert_for_storage) =
        canonical_match_direction(images, left_index, right_index);
    let mut match_info = recover_adjacent_focus_position_match(
        &images[source],
        &images[target],
        log_match,
        orphan_attach,
    )?;
    // Record the fit before it is rotated into the key orientation; see
    // `MatchInfo::canonical_homography` (需求 14.6).
    match_info.canonical_homography = Some(match_info.homography);
    if invert_for_storage {
        match_info.homography = match_info.homography.try_inverse()?;
        match_info.points = match_info
            .points
            .into_iter()
            .map(|(source, target)| (target, source))
            .collect();
        match_info.candidate_points = match_info
            .candidate_points
            .into_iter()
            .map(|(source, target)| (target, source))
            .collect();
    }
    Some((
        (left_index.min(right_index), left_index.max(right_index)),
        match_info,
    ))
}

fn attach_focus_orphan_components(
    images: &[ImageInfo],
    matches: &mut HashMap<(usize, usize), MatchInfo>,
) -> usize {
    let components = focus_overlap_components(images, matches);
    if components.len() <= 1 {
        return 0;
    }
    let main_index = components
        .iter()
        .enumerate()
        .max_by_key(|(_, component)| component.len())
        .map(|(index, _)| index)
        .unwrap_or(0);
    let main_component = &components[main_index];
    let mut added = 0usize;
    for (orphan_index, orphan) in components.iter().enumerate() {
        if orphan_index == main_index {
            continue;
        }
        let mut candidates = Vec::new();
        for &left in main_component {
            for &right in orphan {
                let Some(gap) = trailing_capture_number(&images[left].filename)
                    .zip(trailing_capture_number(&images[right].filename))
                    .map(|(left, right)| left.abs_diff(right))
                else {
                    continue;
                };
                if gap == 0 || gap > 12 {
                    continue;
                }
                candidates.push((
                    gap,
                    std::cmp::Reverse(
                        focus_alignment_support(&images[left])
                            + focus_alignment_support(&images[right]),
                    ),
                    left,
                    right,
                ));
            }
        }
        candidates.sort_by_key(|candidate| (candidate.0, candidate.1, candidate.2, candidate.3));
        for (_, _, left, right) in candidates.into_iter().take(16) {
            let key = (left.min(right), left.max(right));
            if matches.contains_key(&key) {
                continue;
            }
            let Some(recovered) = insert_recovered_focus_match(images, left, right, false, true)
            else {
                continue;
            };
            matches.insert(recovered.0, recovered.1);
            added += 1;
            break;
        }
    }
    added
}

fn recover_adjacent_focus_position_match(
    source: &ImageInfo,
    target: &ImageInfo,
    log_match: bool,
    orphan_attach: bool,
) -> Option<MatchInfo> {
    let capture_gap = trailing_capture_number(&source.filename)
        .zip(trailing_capture_number(&target.filename))
        .map(|(left, right)| left.abs_diff(right))?;
    if capture_gap == 0 || capture_gap > FOCUS_BRACKET_MAX_CAPTURE_GAP {
        return None;
    }
    let source_focal = source
        .focal_length_35mm
        .or(target.focal_length_35mm)
        .unwrap_or(50.0);
    let target_focal = target
        .focal_length_35mm
        .or(source.focal_length_35mm)
        .unwrap_or(source_focal);
    let focal_ratio = (source_focal / target_focal).max(target_focal / source_focal);
    if !focal_ratio.is_finite() || focal_ratio > 1.7 {
        return None;
    }
    let same_focal = focal_ratio <= 1.03;
    let (homography, points, score) = estimate_mixed_focal_coarse_registration(
        source,
        target,
        source_focal,
        target_focal,
        // Spatial neighbours can include opposite ends of a focus bracket.
        // Let the coarse search retain a blurred but structurally consistent
        // candidate; the independent gates below are still stricter than the
        // same-pose recovery profile.
        true,
        log_match,
    )?;
    let scale = homography[(0, 0)].abs();
    let center_motion = focus_match_center_motion_ratio(
        source,
        target,
        &MatchInfo {
            homography,
            inliers: points.len(),
            sequence_bridge: false,
            coarse_bridge: true,
            points: points.clone(),
            candidate_points: Vec::new(),
            top_candidate_points: Vec::new(),
            dense_focus_points: Vec::new(),
            foreground_feature_points: Vec::new(),
            canonical_homography: None,
        },
    )?;
    let overlap =
        panorama_transform_overlap_support(&homography, source.dimensions(), target.dimensions());
    let (intensity_ncc, edge_ncc, edge_orientation, samples) =
        focus_overlap_quality(source, target, &homography)?;
    // Unlike the same-pose fallback, this edge may join two panorama
    // positions. Require strong independent luminance, edge-magnitude, and
    // edge-orientation agreement over a substantial real overlap. Filename
    // adjacency merely selects a bounded candidate; it is never sufficient
    // evidence by itself.
    if !focus_position_recovery_accepts(
        capture_gap,
        same_focal,
        scale,
        center_motion,
        overlap,
        samples,
        intensity_ncc,
        edge_ncc,
        edge_orientation,
        orphan_attach,
    ) {
        if log_match {
            println!(
                "  - Rejecting focus-position recovery '{}' <-> '{}': scale {scale:.3}, intensity {intensity_ncc:.3}, edges {edge_ncc:.3}/{edge_orientation:.3}, motion {:.1}%, overlap {:.1}%, samples {samples}",
                Path::new(&source.filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                Path::new(&target.filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                center_motion * 100.0,
                overlap * 100.0,
            );
        }
        return None;
    }
    if log_match {
        println!(
            "  - Recovered adjacent focus-position edge: '{}' <-> '{}' (NCC {:.3}, edges {:.3}/{:.3}, motion {:.1}%, overlap {:.1}%)",
            Path::new(&source.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            Path::new(&target.filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            score,
            edge_ncc,
            edge_orientation,
            center_motion * 100.0,
            overlap * 100.0,
        );
    }
    Some(MatchInfo {
        homography,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: true,
        points: points.clone(),
        candidate_points: points,
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
    })
}

pub(super) fn focus_overlap_components(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Vec<Vec<usize>> {
    let mut dsu = Dsu::new(images.len());
    for (left, right) in sorted_keys(matches) {
        if left < images.len() && right < images.len() {
            dsu.union(left, right);
        }
    }
    // Ordered by representative index so that the component order does not
    // depend on the hash seed (需求 14.6).
    let mut by_root = BTreeMap::<usize, Vec<usize>>::new();
    for index in 0..images.len() {
        let root = dsu.find(index);
        by_root.entry(root).or_default().push(index);
    }
    let mut components = by_root.into_values().collect::<Vec<_>>();
    for component in &mut components {
        component.sort_by(|&left, &right| {
            natural_path_cmp(&images[left].filename, &images[right].filename)
                .then_with(|| left.cmp(&right))
        });
    }
    components.sort_by(|left, right| {
        natural_path_cmp(
            &images[*left.first().unwrap_or(&0)].filename,
            &images[*right.first().unwrap_or(&0)].filename,
        )
        // `natural_path_cmp` can rate two distinct names equal; the member index
        // is the stable secondary key (需求 14.6).
        .then_with(|| left.first().cmp(&right.first()))
    });
    components
}

#[derive(Clone)]
struct FocusCoarseComponentCandidate {
    left: usize,
    right: usize,
    match_info: MatchInfo,
    quality: f64,
}

fn coarse_focus_component_candidate(
    images: &[ImageInfo],
    left_index: usize,
    right_index: usize,
) -> Option<FocusCoarseComponentCandidate> {
    let capture_gap = focus_capture_number_gap(&images[left_index], &images[right_index])?;
    if capture_gap == 0 || capture_gap > 12 {
        return None;
    }
    let (source_index, target_index, invert_for_storage) =
        canonical_match_direction(images, left_index, right_index);
    let source = &images[source_index];
    let target = &images[target_index];
    let source_focal = source
        .focal_length_35mm
        .or(target.focal_length_35mm)
        .unwrap_or(50.0);
    let target_focal = target
        .focal_length_35mm
        .or(source.focal_length_35mm)
        .unwrap_or(source_focal);
    let focal_ratio = (source_focal / target_focal).max(target_focal / source_focal);
    if !focal_ratio.is_finite() || focal_ratio > 1.7 {
        return None;
    }
    let (homography, points, coarse_score) = estimate_mixed_focal_coarse_registration(
        source,
        target,
        source_focal,
        target_focal,
        true,
        false,
    )?;
    let provisional = MatchInfo {
        homography,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: true,
        points: points.clone(),
        candidate_points: points.clone(),
        top_candidate_points: points.clone(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: Vec::new(),
        canonical_homography: None,
    };
    let scale = homography[(0, 0)].abs();
    let motion = focus_match_center_motion_ratio(source, target, &provisional)?;
    let overlap =
        panorama_transform_overlap_support(&homography, source.dimensions(), target.dimensions());
    let (intensity_ncc, edge_ncc, edge_orientation, samples) =
        focus_overlap_quality(source, target, &homography)?;
    let same_focal = focal_ratio <= 1.03;
    if !scale.is_finite()
        || (!same_focal
            && !(MIXED_FOCAL_SCALE_MIN_RATIO..=MIXED_FOCAL_SCALE_MAX_RATIO).contains(&scale))
        || (same_focal && !(0.86..=1.14).contains(&scale))
        || motion <= 0.025
        || motion > 0.92
        || overlap < 0.12
        || samples < 120
        || coarse_score < 0.55
        || intensity_ncc < 0.35
        || edge_ncc < -0.10
        || edge_orientation < -0.10
    {
        return None;
    }
    let quality = coarse_score * 0.40
        + intensity_ncc * 0.30
        + edge_ncc.max(0.0) * 0.20
        + edge_orientation.max(0.0) * 0.10;
    let mut match_info = provisional;
    // Record the fit before it is rotated into the key orientation; see
    // `MatchInfo::canonical_homography` (需求 14.6).
    match_info.canonical_homography = Some(match_info.homography);
    if invert_for_storage {
        match_info.homography = match_info.homography.try_inverse()?;
        match_info.points = match_info
            .points
            .into_iter()
            .map(|(source, target)| (target, source))
            .collect();
        match_info.candidate_points = match_info
            .candidate_points
            .into_iter()
            .map(|(source, target)| (target, source))
            .collect();
    }
    Some(FocusCoarseComponentCandidate {
        left: left_index.min(right_index),
        right: left_index.max(right_index),
        match_info,
        quality,
    })
}

pub(super) fn focus_component_bridge_candidates(
    component: &[usize],
    images: &[ImageInfo],
) -> Vec<usize> {
    let mut candidates = component.to_vec();
    candidates.sort_by_key(|&index| std::cmp::Reverse(focus_alignment_support(&images[index])));
    candidates.truncate(6);
    // Include both filename boundaries as candidate-order hints. Applying
    // `take(4)` after chaining the iterators only ever selected the first four
    // members, so the tail of a long scan component could never participate in
    // component recovery unless it also happened to be among the six sharpest
    // frames.
    for &index in component
        .iter()
        .take(2)
        .chain(component.iter().rev().take(2))
    {
        if !candidates.contains(&index) {
            candidates.push(index);
        }
    }
    candidates
}

pub(super) fn recover_focus_component_edges(
    images: &[ImageInfo],
    matches: &mut HashMap<(usize, usize), MatchInfo>,
) -> usize {
    let mut total_added = 0usize;
    for pass in 0..3 {
        let components = focus_overlap_components(images, matches);
        if components.len() <= 1 {
            break;
        }
        let mut pending = Vec::new();
        for left_component_index in 0..components.len() {
            for right_component_index in left_component_index + 1..components.len() {
                let left_candidates =
                    focus_component_bridge_candidates(&components[left_component_index], images);
                let right_candidates =
                    focus_component_bridge_candidates(&components[right_component_index], images);
                let mut candidates = Vec::new();
                for &left in &left_candidates {
                    for &right in &right_candidates {
                        let key = (left.min(right), left.max(right));
                        if matches.contains_key(&key) {
                            continue;
                        }
                        if let Some(candidate) =
                            coarse_focus_component_candidate(images, left, right)
                        {
                            candidates.push(candidate);
                        }
                    }
                }
                if candidates.len() < 2 {
                    continue;
                }
                let dimensions = images[components[left_component_index][0]].dimensions();
                let disagreement_limit = dimensions.0.max(dimensions.1) as f64 * 0.08;
                let mut best: Option<(usize, f64, usize)> = None;
                for (candidate_index, candidate) in candidates.iter().enumerate() {
                    let mut support = 0usize;
                    let mut score = 0.0;
                    let mut left_sources = HashSet::new();
                    let mut right_sources = HashSet::new();
                    for other in &candidates {
                        if focus_transform_disagreement_px(
                            &candidate.match_info.homography,
                            &other.match_info.homography,
                            dimensions,
                        )
                        .is_some_and(|error| error <= disagreement_limit)
                        {
                            support += 1;
                            score += other.quality;
                            left_sources.insert(other.left);
                            right_sources.insert(other.right);
                        }
                    }
                    if left_sources.len() < 2 || right_sources.len() < 2 {
                        continue;
                    }
                    let should_replace =
                        best.as_ref().is_none_or(|(best_support, best_score, _)| {
                            support > *best_support
                                || (support == *best_support && score > *best_score)
                        });
                    if should_replace {
                        best = Some((support, score, candidate_index));
                    }
                }
                let Some((support, score, candidate_index)) = best else {
                    continue;
                };
                if support < 2 {
                    continue;
                }
                let candidate = &candidates[candidate_index];
                println!(
                    "  - Accepted consensus coarse bridge '{}' <-> '{}' ({support} agreeing focus layers, score {score:.3})",
                    Path::new(&images[candidate.left].filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    Path::new(&images[candidate.right].filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                );
                pending.push((
                    (candidate.left, candidate.right),
                    candidate.match_info.clone(),
                ));
            }
        }
        if pending.is_empty() {
            break;
        }
        let added = pending.len();
        matches.extend(pending);
        total_added += added;
        println!(
            "  - Added {added} consensus coarse bridge(s) on retry pass {}",
            pass + 1
        );
    }
    total_added
}

pub(super) fn collect_dense_focus_region_points(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    if projection != Projection::Planar
        || source_image.alignment_image.width() < 64
        || source_image.alignment_image.height() < 64
        || target_image.alignment_image.width() < 64
        || target_image.alignment_image.height() < 64
    {
        return Vec::new();
    }

    let source_plane = LumaPlane::Gray(&source_image.alignment_image);
    let target_plane = LumaPlane::Gray(&target_image.alignment_image);
    let source_width = source_image.alignment_image.width() as f64;
    let source_height = source_image.alignment_image.height() as f64;
    let target_width = target_image.alignment_image.width() as i32;
    let target_height = target_image.alignment_image.height() as i32;
    let source_scale = source_image.scale_factor;
    let target_scale = target_image.scale_factor;
    let mut points = Vec::new();

    // Use a sparse grid over the complete source frame. This is the generic
    // residual-registration signal: ordinary paper texture and any foreground
    // object can contribute, while the later regional RANSAC decides whether a
    // consistent local model exists. The search radius is intentionally much
    // smaller than the previous foreground-only search so repeated strokes on
    // a long edge cannot jump to a different copy.
    // The grid is deliberately independent of colour, brightness, or a named
    // object. Extra samples near both frame edges let the generic regional
    // registration see structured layers near the frame boundary even when an
    // optional depth-mask detector does not fire.
    let mut source_y_fractions = vec![
        0.03, 0.08, 0.14, 0.20, 0.26, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88, 0.95,
    ];
    // A narrow depth layer is easy to miss when the fixed grid lands only on
    // its uniform interior. Add samples at both silhouettes and along the
    // interior whenever the capture contains a detected occlusion band. The
    // same path is used for every object; the detector only supplies optional
    // sampling locations and never changes the stitcher's geometry model.
    if let Some((minimum_y, maximum_y)) = source_image.foreground_range {
        let span = (maximum_y - minimum_y).max(0.0);
        source_y_fractions.extend([
            minimum_y,
            minimum_y + span * 0.25,
            minimum_y + span * 0.5,
            minimum_y + span * 0.75,
            maximum_y,
        ]);
        source_y_fractions.sort_unstable_by(f64::total_cmp);
        source_y_fractions.dedup_by(|left, right| (*left - *right).abs() < 0.005);
    }
    let mut sample_rows = source_y_fractions
        .into_iter()
        .map(|fraction| (fraction, 14))
        .collect::<Vec<_>>();
    for &edge_row in &source_image.horizontal_edge_rows {
        if edge_row.is_finite()
            && (0.02..=0.98).contains(&edge_row)
            && sample_rows
                .iter()
                .all(|(row, _)| (*row - edge_row).abs() >= 0.005)
        {
            sample_rows.push((edge_row, FOCUS_HORIZONTAL_EDGE_SEARCH_RADIUS));
        }
    }
    sample_rows.sort_unstable_by(|(left, _), (right, _)| left.total_cmp(right));
    let source_x_fractions = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
    let patch_radius = 8;
    for (source_y_fraction, search_radius) in sample_rows {
        for source_x_fraction in source_x_fractions {
            let source_x = (source_width * source_x_fraction).round() as i32;
            let source_y = (source_height * source_y_fraction).round() as i32;
            if gradient_patch_energy(&source_plane, source_x, source_y, patch_radius) < 1.5 {
                continue;
            }
            let source_full = Point2::new(
                source_x as f64 * source_scale,
                source_y as f64 * source_scale,
            );
            let predicted = *homography * Point3::new(source_full.x, source_full.y, 1.0);
            if predicted.z.abs() < 1e-8 {
                continue;
            }
            let predicted_target_full =
                Point2::new(predicted.x / predicted.z, predicted.y / predicted.z);
            let target_x = (predicted_target_full.x / target_scale).round() as i32;
            let target_y = (predicted_target_full.y / target_scale).round() as i32;
            if source_x < patch_radius
                || source_y < patch_radius
                || source_x + patch_radius >= source_width as i32
                || source_y + patch_radius >= source_height as i32
                || target_x < patch_radius + search_radius
                || target_y < patch_radius + search_radius
                || target_x + patch_radius + search_radius >= target_width
                || target_y + patch_radius + search_radius >= target_height
            {
                continue;
            }
            let Some((best_x, best_y, subpixel_x, subpixel_y)) = refine_foreground_patch_position(
                &source_plane,
                &target_plane,
                source_x,
                source_y,
                target_x,
                target_y,
                patch_radius,
                search_radius,
            ) else {
                continue;
            };
            let score = gradient_patch_ncc(
                &source_plane,
                &target_plane,
                source_x,
                source_y,
                best_x,
                best_y,
                patch_radius,
            );
            let luminance_score = patch_ncc(
                &source_plane,
                &target_plane,
                source_x,
                source_y,
                best_x,
                best_y,
                patch_radius,
            );
            if !score.is_finite()
                || score < FOCUS_DENSE_REGION_MIN_NCC
                || !luminance_score.is_finite()
                || luminance_score < 0.40
                || gradient_patch_energy(&target_plane, best_x, best_y, patch_radius) < 1.5
            {
                continue;
            }
            let target_full = Point2::new(
                (best_x as f64 + subpixel_x) * target_scale,
                (best_y as f64 + subpixel_y) * target_scale,
            );
            points.push((source_full, target_full));
        }
    }
    points
}

pub(super) fn collect_foreground_feature_points(
    source_image: &ImageInfo,
    target_image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
) -> Vec<(Point2<f64>, Point2<f64>)> {
    let Some((source_minimum_y, source_maximum_y)) = source_image.foreground_range else {
        return Vec::new();
    };
    let Some((target_minimum_y, target_maximum_y)) = target_image.foreground_range else {
        return Vec::new();
    };
    if projection != Projection::Planar {
        return Vec::new();
    }

    let source_plane = LumaPlane::Gray(&source_image.alignment_image);
    let target_plane = LumaPlane::Gray(&target_image.alignment_image);
    let source_width = source_image.alignment_image.width() as i32;
    let source_height = source_image.alignment_image.height() as i32;
    let target_width = target_image.alignment_image.width() as i32;
    let target_height = target_image.alignment_image.height() as i32;
    let patch_radius = FOCUS_FOREGROUND_PATCH_RADIUS;
    let search_radius = FOCUS_FOREGROUND_SEARCH_RADIUS;
    let mut points = Vec::new();

    for feature in &source_image.top_features {
        let source_x = feature.keypoint.x as i32;
        let source_y = feature.keypoint.y as i32;
        if source_x < patch_radius
            || source_y < patch_radius
            || source_x + patch_radius >= source_width - 1
            || source_y + patch_radius >= source_height - 1
        {
            continue;
        }
        let source_y_fraction = source_y as f64 / source_height.max(1) as f64;
        if !(source_minimum_y..=source_maximum_y).contains(&source_y_fraction) {
            continue;
        }
        if gradient_patch_corner_energy(&source_plane, source_x, source_y, patch_radius)
            < FOCUS_FOREGROUND_MIN_CORNER_ENERGY
        {
            continue;
        }
        let source_full = Point2::new(
            source_x as f64 * source_image.scale_factor,
            source_y as f64 * source_image.scale_factor,
        );
        let predicted = *homography * Point3::new(source_full.x, source_full.y, 1.0);
        if predicted.z.abs() < 1e-8 {
            continue;
        }
        let predicted_target_full =
            Point2::new(predicted.x / predicted.z, predicted.y / predicted.z);
        let target_x = (predicted_target_full.x / target_image.scale_factor).round() as i32;
        let target_y = (predicted_target_full.y / target_image.scale_factor).round() as i32;
        if target_x < patch_radius + search_radius
            || target_y < patch_radius + search_radius
            || target_x + patch_radius + search_radius >= target_width
            || target_y + patch_radius + search_radius >= target_height
        {
            continue;
        }
        let Some((best_x, best_y, subpixel_x, subpixel_y)) = refine_foreground_patch_position(
            &source_plane,
            &target_plane,
            source_x,
            source_y,
            target_x,
            target_y,
            patch_radius,
            search_radius,
        ) else {
            continue;
        };
        let score = gradient_patch_ncc(
            &source_plane,
            &target_plane,
            source_x,
            source_y,
            best_x,
            best_y,
            patch_radius,
        );
        let target_y_fraction = (best_y as f64 + subpixel_y) / target_height.max(1) as f64;
        if !score.is_finite()
            || score < FOCUS_DENSE_REGION_MIN_NCC
            || !(target_minimum_y..=target_maximum_y).contains(&target_y_fraction)
            || target_plane.luma_at(best_x, best_y).unwrap_or(0.0) < 110.0
            || gradient_patch_energy(&target_plane, best_x, best_y, patch_radius)
                < FOCUS_FOREGROUND_MIN_GRADIENT_ENERGY
            || gradient_patch_corner_energy(&target_plane, best_x, best_y, patch_radius)
                < FOCUS_FOREGROUND_MIN_CORNER_ENERGY
        {
            continue;
        }
        if points.iter().any(|(previous_source, previous_target)| {
            (previous_source - source_full).norm() < 10.0
                || (previous_target
                    - Point2::new(
                        (best_x as f64 + subpixel_x) * target_image.scale_factor,
                        (best_y as f64 + subpixel_y) * target_image.scale_factor,
                    ))
                .norm()
                    < 10.0
        }) {
            continue;
        }
        points.push((
            source_full,
            Point2::new(
                (best_x as f64 + subpixel_x) * target_image.scale_factor,
                (best_y as f64 + subpixel_y) * target_image.scale_factor,
            ),
        ));
    }
    points
}
