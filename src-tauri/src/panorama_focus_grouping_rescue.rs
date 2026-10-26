//! Focus-stack geometry and component-rescue helpers.
//!
//! This module is a sibling of the `focus_grouping` facade so its `pub(super)`
//! symbols remain available to the parent matching, execution, and test modules.

use super::*;

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

pub(super) fn focus_station_representative(station: &[usize], images: &[ImageInfo]) -> usize {
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
