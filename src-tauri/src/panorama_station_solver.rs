//! Capture-station grouping, station-relation validation, and closure solving.
//!
//! The parent module remains the integration facade; this child owns the
//! station solver decisions and exports the production helpers it consumes.

use super::*;

/// Frame-centre displacement of an explicit transform, in long sides.
pub(super) fn focus_transform_center_motion_ratio(
    source: &ImageInfo,
    target: &ImageInfo,
    source_to_target: &Matrix3<f64>,
) -> Option<f64> {
    let source_center = Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5);
    let target_center = Point2::new(target.width as f64 * 0.5, target.height as f64 * 0.5);
    let mapped_center = transformed_point(source_to_target, source_center)?;
    let scale = source
        .width
        .max(source.height)
        .max(target.width.max(target.height))
        .max(1) as f64;
    let displacement = (mapped_center - target_center).norm();
    (displacement.is_finite()).then_some(displacement / scale)
}

pub(super) fn focus_match_center_motion_ratio(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
) -> Option<f64> {
    focus_transform_center_motion_ratio(source, target, &match_info.homography)
}

pub(super) fn focus_match_is_local_bracket(
    source: &ImageInfo,
    target: &ImageInfo,
    match_info: &MatchInfo,
) -> bool {
    if match_info.sequence_bridge {
        return false;
    }
    let compatible_focal_length = source
        .focal_length_35mm
        .zip(target.focal_length_35mm)
        .map(|(left, right)| {
            let ratio = (left / right.max(1e-8)).max(right / left.max(1e-8));
            ratio.is_finite() && ratio <= 1.10
        })
        .unwrap_or(true);
    if match_info.coarse_bridge {
        let scale = match_info.homography[(0, 0)].abs();
        return compatible_focal_length
            && match_info.points.len() >= SCALE_ROBUST_MIN_INLIERS_FOR_CONNECTION
            && (0.88..=1.12).contains(&scale)
            && focus_match_center_motion_ratio(source, target, match_info)
                .is_some_and(|motion| motion <= 0.12)
            && panorama_transform_overlap_support(
                &match_info.homography,
                source.dimensions(),
                target.dimensions(),
            ) >= 0.60;
    }
    if match_info.points.len() < FOCUS_MODEL_MIN_INLIERS {
        return false;
    }
    compatible_focal_length
        && focus_match_center_motion_ratio(source, target, match_info)
            .is_some_and(|motion| motion <= FOCUS_BRACKET_MAX_CENTER_MOTION_RATIO)
        && panorama_transform_overlap_support(
            &match_info.homography,
            source.dimensions(),
            target.dimensions(),
        ) >= FOCUS_BRACKET_MIN_OVERLAP_SUPPORT
}

/// The four pieces of image evidence 需求 1.1 allows a Capture_Station
/// membership decision to rest on, together with the derived score 需求 1.2
/// compares when it needs a tie-break.
///
/// Every field is a measurement, never a threshold: the accept/reject decision
/// lives in [`StationEvidence::accepts`] so that the numbers can be reported
/// (需求 1.7) for accepted *and* rejected relations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StationEvidence {
    /// Verified overlap inlier correspondences (需求 1.1, ≥ 30).
    pub inliers: usize,
    /// Local normalised pixel correlation over the overlap (需求 1.1, ≥ 0.60).
    ///
    /// Meaningful only when `overlap_ncc_measured` is set.
    pub overlap_ncc: f64,
    /// Whether the overlap correlation could be measured at all.
    ///
    /// The correlation needs at least 120 usable samples out of a 32px-or-larger
    /// alignment image; the design keeps that as the *measurable precondition*
    /// of the threshold. Synthetic sources that carry a placeholder alignment
    /// image report no correlation evidence in either direction, exactly like
    /// the `alignment_is_measurable` gate the station relation scorer uses.
    pub overlap_ncc_measured: bool,
    /// Inlier spread inside the overlap (需求 1.1, ≥ 0.20).
    pub spatial_support: f64,
    /// `sqrt(σ1 · σ2)` of the linear part (需求 1.1, ∈ `[0.98, 1.02]`).
    pub scale_ratio: f64,
    /// Composite of the four measurements above. Ordering only — never a gate.
    pub score: f64,
}

impl Default for StationEvidence {
    fn default() -> Self {
        Self {
            inliers: 0,
            overlap_ncc: 0.0,
            overlap_ncc_measured: false,
            spatial_support: 0.0,
            scale_ratio: 1.0,
            score: 0.0,
        }
    }
}

impl StationEvidence {
    /// Stable identifier of the evidence kind, for the Stack_Report (需求 1.7).
    pub(crate) const KIND: &'static str = "verified_overlap";

    /// Whether all four measurements clear 需求 1.1 at once.
    pub(crate) fn accepts(&self) -> bool {
        self.inliers >= STATION_MIN_INLIER_MATCHES
            && (!self.overlap_ncc_measured || self.overlap_ncc >= STATION_MIN_OVERLAP_NCC)
            && self.spatial_support >= STATION_MIN_INLIER_SPATIAL_SUPPORT
            && (STATION_SCALE_RATIO_MIN..=STATION_SCALE_RATIO_MAX).contains(&self.scale_ratio)
    }

    /// Composite score used only where 需求 1.2 permits a comparison: to order
    /// candidates, and to decide whether an absolute-path tie-break is allowed
    /// at all (`|Δscore| ≤ 0.001`).
    ///
    /// Each factor is normalised into `[0, 1]` so the score stays a function of
    /// the measurements and of nothing else.
    fn from_measurements(
        inliers: usize,
        overlap_ncc: Option<f64>,
        spatial_support: f64,
        scale_ratio: f64,
    ) -> Self {
        let inlier_term =
            (inliers as f64 / STATION_MIN_INLIER_MATCHES as f64).clamp(0.0, 4.0) / 4.0;
        // An unmeasurable correlation contributes the bar itself, so a source
        // with a placeholder alignment image neither gains nor loses score.
        let correlation_term = overlap_ncc
            .unwrap_or(STATION_MIN_OVERLAP_NCC)
            .clamp(0.0, 1.0);
        let support_term = spatial_support.clamp(0.0, 1.0);
        let scale_term = if scale_ratio.is_finite() && scale_ratio > 0.0 {
            (1.0 - (scale_ratio - 1.0).abs() / (STATION_SCALE_RATIO_MAX - 1.0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            inliers,
            overlap_ncc: overlap_ncc.unwrap_or(0.0),
            overlap_ncc_measured: overlap_ncc.is_some(),
            spatial_support,
            scale_ratio,
            score: 0.25 * (inlier_term + correlation_term + support_term + scale_term),
        }
    }
}

/// Measure the 需求 1.1 evidence of one pairwise relation.
///
/// The measurement runs in the content-derived direction that
/// [`canonical_match_direction`] picks, not in the `(min, max)` orientation the
/// `matches` key happens to use: `focus_overlap_quality` warps the source into
/// the target before correlating and `panorama_spatial_support` weighs the
/// source spread first, so scoring in the key direction let a shuffled import
/// award the same physical relation different numbers (需求 1.2 / 14.6).
/// `canonical_homography` is the untouched fit in exactly that direction, which
/// avoids the lossy `inv(inv(M))` round trip.
pub(super) fn focus_station_link_evidence(
    images: &[ImageInfo],
    first: usize,
    second: usize,
    match_info: &MatchInfo,
) -> Option<StationEvidence> {
    // A coarse correlation peak or a filename-sequence bridge is an ordering
    // cue whose points are synthetic. It may connect two already established
    // stations, but it is not overlap evidence and can never merge two camera
    // positions into one bracket.
    if match_info.sequence_bridge || match_info.coarse_bridge {
        return None;
    }
    let (source_index, target_index, inverted) = canonical_match_direction(images, first, second);
    let source = images.get(source_index)?;
    let target = images.get(target_index)?;
    if !focus_compatible_focal_length(source, target) {
        return None;
    }
    let homography = match match_info.canonical_homography {
        Some(canonical) => canonical,
        None if inverted => match_info.homography.try_inverse()?,
        None => match_info.homography,
    };
    // A coarse precondition, subordinate to the accumulated-displacement split
    // of 需求 1.4: a relation whose own centre displacement is already several
    // times the 0.02 accumulated budget cannot be an intra-station relation, and
    // admitting it would only make the split undo it. The binding motion
    // criterion remains `STATION_MAX_ACCUMULATED_CENTER_MOTION_RATIO`.
    if !focus_transform_center_motion_ratio(source, target, &homography)
        .is_some_and(|motion| motion <= FOCUS_BRACKET_MAX_CENTER_MOTION_RATIO)
    {
        return None;
    }
    let points: Cow<'_, [(Point2<f64>, Point2<f64>)]> = if inverted {
        Cow::Owned(
            match_info
                .points
                .iter()
                .map(|&(source_point, target_point)| (target_point, source_point))
                .collect(),
        )
    } else {
        Cow::Borrowed(&match_info.points)
    };
    // `focus_overlap_quality` returns `None` below 120 usable samples, which is
    // the measurable precondition of the correlation threshold.  Distinguish
    // "the overlap correlates badly" (reject) from "this source pair carries no
    // alignment image large enough to correlate on" (no evidence either way),
    // using the same 32px test the station relation scorer applies.
    let alignment_is_measurable = source.alignment_image.width() >= 32
        && source.alignment_image.height() >= 32
        && target.alignment_image.width() >= 32
        && target.alignment_image.height() >= 32;
    let overlap_ncc = match focus_overlap_quality(source, target, &homography) {
        Some((intensity_ncc, _, _, _)) => Some(intensity_ncc),
        None if alignment_is_measurable => return None,
        None => None,
    };
    let spatial_support =
        panorama_spatial_support(&points, source.dimensions(), target.dimensions());
    let scale_ratio = homography_scale_ratio(&homography)?;
    Some(StationEvidence::from_measurements(
        points.len(),
        overlap_ncc,
        spatial_support,
        scale_ratio,
    ))
}

/// Whether two sources belong to the same Capture_Station (需求 1.1).
pub(super) fn focus_match_is_capture_station_link(
    images: &[ImageInfo],
    first: usize,
    second: usize,
    match_info: &MatchInfo,
) -> bool {
    focus_station_link_evidence(images, first, second, match_info)
        .is_some_and(|evidence| evidence.accepts())
}

/// One Capture_Station split, recorded for the Stack_Report (需求 1.4 / 1.10).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StationSplit {
    /// Verified-link component the split was applied to, in the component order
    /// the grouper walks. Station indices do not exist yet at split time.
    pub component: usize,
    /// Position inside the candidate order of the component at which the split
    /// falls: the candidate at this position starts the next Capture_Station.
    pub position: usize,
    /// Accumulated frame-centre displacement measured there, in long sides.
    pub motion_ratio: f64,
    /// [`degradation::GROUPING_ACCUMULATED_MOTION`] or
    /// [`degradation::GROUPING_MEMBER_LIMIT_SPLIT`].
    pub reason: &'static str,
}

/// Relative poses of one accepted station link, both derived from the untouched
/// canonical fit with exactly one inversion each.
///
/// Composing `homography` and `homography.try_inverse()` would make one of the
/// two directions an `inv(inv(M))` round trip whenever the `(min, max)` key
/// orientation disagrees with the content-derived measurement direction, and
/// that round trip is not bit exact (需求 14.6).
pub(super) fn station_link_relative_poses(
    images: &[ImageInfo],
    left: usize,
    right: usize,
    match_info: &MatchInfo,
) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
    let (source_index, _, inverted) = canonical_match_direction(images, left, right);
    let canonical = match match_info.canonical_homography {
        Some(canonical) => canonical,
        None if inverted => match_info.homography.try_inverse()?,
        None => match_info.homography,
    };
    let reverse = canonical.try_inverse()?;
    Some(if source_index == left {
        (canonical, reverse)
    } else {
        (reverse, canonical)
    })
}

/// Split one verified-link component into Capture_Stations at the candidates
/// where the accumulated frame-centre displacement leaves the bracket window
/// (需求 1.4), then enforce the 48-member ceiling (需求 1.5 / 1.10).
///
/// The candidate order is the relative world-centre order recovered from the
/// verified link graph; the absolute path only breaks an exact centre tie
/// (需求 1.2 / 1.3).
pub(super) fn split_focus_local_component_by_motion(
    component: Vec<usize>,
    component_index: usize,
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    splits: &mut Vec<StationSplit>,
) -> Vec<Vec<usize>> {
    if component.is_empty() {
        return Vec::new();
    }

    // Recover a relative pose for every member using the verified local-link
    // graph.  Sorting by filename and only testing consecutive pairs would let
    // a renamed or shuffled import split one focus bracket even when the same
    // geometric edges were present.
    let member_set = component.iter().copied().collect::<HashSet<_>>();
    let mut adjacency = HashMap::<usize, Vec<(usize, Matrix3<f64>, f64)>>::new();
    for &left in &component {
        for &right in &component {
            if left >= right || !member_set.contains(&right) {
                continue;
            }
            let Some(match_info) = matches.get(&(left, right)) else {
                continue;
            };
            let Some(evidence) = focus_station_link_evidence(images, left, right, match_info)
                .filter(StationEvidence::accepts)
            else {
                continue;
            };
            let Some((left_to_right, right_to_left)) =
                station_link_relative_poses(images, left, right, match_info)
            else {
                continue;
            };
            adjacency
                .entry(left)
                .or_default()
                .push((right, right_to_left, evidence.score));
            adjacency
                .entry(right)
                .or_default()
                .push((left, left_to_right, evidence.score));
        }
    }

    // 需求 1.2: the score may order candidates, and the absolute path may only
    // step in once two scores are within 0.001 of each other.
    let evidence_cmp = |left_score: f64, right_score: f64, left: usize, right: usize| {
        if (left_score - right_score).abs() <= STATION_EVIDENCE_SCORE_TIE_BREAK {
            focus_image_direction_key(&images[left])
                .cmp(&focus_image_direction_key(&images[right]))
                .then_with(|| natural_path_cmp(&images[left].filename, &images[right].filename))
                .then_with(|| left.cmp(&right))
        } else {
            right_score.total_cmp(&left_score)
        }
    };
    let link_support = |image_index: &usize| {
        adjacency
            .get(image_index)
            .into_iter()
            .flatten()
            .map(|(_, _, score)| *score)
            .sum::<f64>()
    };
    let anchor = component
        .iter()
        .copied()
        .min_by(|left, right| evidence_cmp(link_support(left), link_support(right), *left, *right))
        .unwrap_or(component[0]);
    let mut relative_to_anchor = HashMap::<usize, Matrix3<f64>>::new();
    relative_to_anchor.insert(anchor, Matrix3::identity());
    let mut pending = VecDeque::from([anchor]);
    while let Some(current) = pending.pop_front() {
        let current_to_anchor = relative_to_anchor[&current];
        let mut neighbours = adjacency.get(&current).cloned().unwrap_or_default();
        neighbours.sort_by(|left, right| evidence_cmp(left.2, right.2, left.0, right.0));
        for (neighbor, neighbor_to_current, _) in neighbours {
            if relative_to_anchor.contains_key(&neighbor) {
                continue;
            }
            relative_to_anchor.insert(neighbor, current_to_anchor * neighbor_to_current);
            pending.push_back(neighbor);
        }
    }

    // The candidate order is the recovered world-centre order.  Members whose
    // pose could not be recovered carry no centre and sort last; they cannot
    // take part in a displacement measurement either way.
    let member_center = |image_index: usize| {
        relative_to_anchor.get(&image_index).and_then(|relative| {
            transformed_point(
                relative,
                Point2::new(
                    images[image_index].width as f64 * 0.5,
                    images[image_index].height as f64 * 0.5,
                ),
            )
        })
    };
    let mut ordered = component;
    ordered.sort_by(|left, right| {
        let left_center = member_center(*left);
        let right_center = member_center(*right);
        match (left_center, right_center) {
            (Some(left_center), Some(right_center)) => left_center
                .y
                .total_cmp(&right_center.y)
                .then_with(|| left_center.x.total_cmp(&right_center.x)),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }
        // 需求 1.2: the absolute path enters only once the world centres compare
        // exactly equal, and the member index is the final stable key.
        .then_with(|| {
            focus_image_direction_key(&images[*left])
                .cmp(&focus_image_direction_key(&images[*right]))
        })
        .then_with(|| natural_path_cmp(&images[*left].filename, &images[*right].filename))
        .then_with(|| left.cmp(right))
    });

    let long_side = |image_index: usize| {
        images[image_index]
            .width
            .max(images[image_index].height)
            .max(1) as f64
    };
    // Step displacement between adjacent candidates, normalised by the longer of
    // the two long sides.  `None` means "not measurable", which is treated as a
    // boundary: two members without a common recovered pose have no evidence
    // that they share a camera position.
    let step_motion = |previous: usize, current: usize| -> Option<f64> {
        let previous_center = member_center(previous)?;
        let current_center = member_center(current)?;
        let scale = long_side(previous).max(long_side(current));
        let ratio = (current_center - previous_center).norm() / scale;
        ratio.is_finite().then_some(ratio)
    };

    // 需求 1.4: walk the candidate order and accumulate the centre displacement
    // since the first member of the current Capture_Station.  A single split
    // position is therefore a function of the measured poses alone.
    let mut groups = Vec::<Vec<usize>>::new();
    let mut current = Vec::<usize>::new();
    let mut accumulated = 0.0f64;
    for (position, image_index) in ordered.iter().copied().enumerate() {
        if current.is_empty() {
            current.push(image_index);
            accumulated = 0.0;
            continue;
        }
        let previous = *current.last().expect("non-empty above");
        match step_motion(previous, image_index) {
            Some(step) => {
                let candidate_accumulated = accumulated + step;
                if candidate_accumulated > STATION_MAX_ACCUMULATED_CENTER_MOTION_RATIO {
                    splits.push(StationSplit {
                        component: component_index,
                        position,
                        motion_ratio: candidate_accumulated,
                        reason: degradation::GROUPING_ACCUMULATED_MOTION,
                    });
                    groups.push(std::mem::take(&mut current));
                    current.push(image_index);
                    accumulated = 0.0;
                } else {
                    accumulated = candidate_accumulated;
                    current.push(image_index);
                }
            }
            None => {
                splits.push(StationSplit {
                    component: component_index,
                    position,
                    motion_ratio: f64::NAN,
                    reason: degradation::GROUPING_ACCUMULATED_MOTION,
                });
                groups.push(std::mem::take(&mut current));
                current.push(image_index);
                accumulated = 0.0;
            }
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }

    // 需求 1.5 / 1.10: a Capture_Station holds 1..=48 sources.  Split an
    // oversized one at the adjacent candidate pair carrying the largest
    // displacement, repeatedly, until every station fits.
    let mut bounded = Vec::<Vec<usize>>::with_capacity(groups.len());
    let mut oversized = VecDeque::from(groups);
    while let Some(group) = oversized.pop_front() {
        if station_member_count_is_supported(group.len()) {
            bounded.push(group);
            continue;
        }
        let mut best: Option<(f64, usize)> = None;
        for position in 1..group.len() {
            let step = step_motion(group[position - 1], group[position]).unwrap_or(f64::INFINITY);
            let balance = position.abs_diff(group.len() / 2);
            let better = match best {
                None => true,
                Some((best_step, best_position)) => {
                    if (step - best_step).abs() <= f64::EPSILON {
                        // Equal displacement — all-zero motion included — splits
                        // as evenly as possible so the recursion terminates
                        // quickly and deterministically.
                        balance < best_position.abs_diff(group.len() / 2)
                    } else {
                        step > best_step
                    }
                }
            };
            if better {
                best = Some((step, position));
            }
        }
        let (motion_ratio, position) = best.unwrap_or((f64::NAN, group.len() / 2));
        splits.push(StationSplit {
            component: component_index,
            position,
            motion_ratio,
            reason: degradation::GROUPING_MEMBER_LIMIT_SPLIT,
        });
        let tail = group[position..].to_vec();
        let head = group[..position].to_vec();
        // Push both halves back so a half that is still oversized is split again.
        oversized.push_front(tail);
        oversized.push_front(head);
    }
    bounded
}

/// Capture_Station membership of one selection, with everything 需求 1.6 / 1.7 /
/// 1.9 / 1.4 / 1.10 ask the Stack_Report to carry.
#[derive(Clone, Debug, Default)]
pub(crate) struct StationGroupingOutcome {
    /// Capture_Stations in station order, including connected single-frame scans.
    pub components: Vec<Vec<usize>>,
    /// 需求 1.6: sources with neither same-pose nor verified scan-overlap evidence.
    pub isolated: Vec<usize>,
    /// 需求 1.4 / 1.10: every split position, in the order it was applied.
    pub splits: Vec<StationSplit>,
}

/// Group the selection into Capture_Stations from image evidence only.
///
/// Membership is a DSU over the accepted 需求 1.1 relations — a set operation,
/// so it cannot see the insertion order — followed by the accumulated-motion
/// split above. A verified scan overlap marks a source as connected without
/// unioning the camera poses: an unbracketed scan remains a single-frame station.
pub(super) fn focus_station_grouping(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> StationGroupingOutcome {
    let mut dsu = Dsu::new(images.len());
    let mut linked = vec![false; images.len()];
    for (&(source_index, target_index), match_info) in sorted_pairs(matches) {
        if source_index >= images.len() || target_index >= images.len() {
            continue;
        }
        if focus_match_is_capture_station_link(images, source_index, target_index, match_info) {
            dsu.union(source_index, target_index);
            linked[source_index] = true;
            linked[target_index] = true;
        } else if (!linked[source_index] || !linked[target_index])
            && !match_info.coarse_bridge
            && !match_info.sequence_bridge
            && match_info.points.len() >= 30
            && median_symmetric_error(&match_info.homography, &match_info.points) <= 3.0
            && match_has_spatial_support(
                &match_info.points,
                images[source_index].dimensions(),
                images[target_index].dimensions(),
            )
            && (focus_overlap_quality(
                &images[source_index],
                &images[target_index],
                &match_info.homography,
            )
            .is_some_and(|quality| {
                focus_overlap_is_verified(
                    quality,
                    match_info.points.len(),
                    median_symmetric_error(&match_info.homography, &match_info.points),
                    false,
                )
            }) || shifted_focus_structure_is_verified(
                &images[source_index],
                &images[target_index],
                &match_info.homography,
                &match_info.points,
                false,
            ))
        {
            // A verified spatial scan edge proves this source is connected,
            // but does not prove the two frames share a camera pose. Preserve
            // each as its own station when it has no focus-bracket partner.
            linked[source_index] = true;
            linked[target_index] = true;
        }
    }
    // Ordered by representative index: the bracket split below is applied in this
    // order and its output feeds the station order (需求 14.6).
    let mut components = BTreeMap::<usize, Vec<usize>>::new();
    for image_index in 0..images.len() {
        let root = dsu.find(image_index);
        components.entry(root).or_default().push(image_index);
    }
    let mut splits = Vec::new();
    let mut components = components
        .into_values()
        .enumerate()
        .flat_map(|(component_index, component)| {
            split_focus_local_component_by_motion(
                component,
                component_index,
                images,
                matches,
                &mut splits,
            )
        })
        .collect::<Vec<_>>();
    // 需求 1.6: a source with no accepted overlap relation at all is isolated.
    // A single member left over by a *split* is not: it had verified evidence,
    // the accumulated motion simply put it at its own camera position.
    let isolated = (0..images.len())
        .filter(|&image_index| !linked[image_index])
        .collect::<Vec<_>>();
    components.retain(|component| component.iter().any(|&index| linked[index]));
    for component in &mut components {
        component.sort_by(|left, right| {
            natural_path_cmp(&images[*left].filename, &images[*right].filename)
                // `natural_path_cmp` can rate two distinct names equal; the
                // member index is the stable secondary key (需求 14.6).
                .then_with(|| left.cmp(right))
        });
    }
    components.sort_by(|left, right| {
        natural_path_cmp(&images[left[0]].filename, &images[right[0]].filename)
            .then_with(|| left[0].cmp(&right[0]))
    });
    StationGroupingOutcome {
        components,
        isolated,
        splits,
    }
}

pub(super) fn focus_local_bracket_components(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Vec<Vec<usize>> {
    focus_station_grouping(images, matches).components
}

/// The selected sources that produced no decoded image (需求 1.9).
///
/// Returned in selection order and by absolute path, which is what the
/// Stack_Report and the degradation ledger record.  Excluding them from every
/// Capture_Station is not a separate step: the grouper only ever sees the decoded
/// images, so a source named here cannot reach it.
pub(super) fn undecodable_source_paths(selection: &[String], decoded: &[ImageInfo]) -> Vec<String> {
    selection
        .iter()
        .filter(|path| !decoded.iter().any(|image| &image.filename == *path))
        .cloned()
        .collect()
}

/// The Stack_Report records of the sources one grouping excluded (需求 1.6 / 1.9).
///
/// Each record names the absolute path of the excluded Source_RAW and the stable
/// reason identifier it was excluded for, and nothing else: the exclusion itself
/// has already happened by the time this runs.
pub(super) fn excluded_source_records(
    images: &[ImageInfo],
    indices: &[usize],
    reason: &'static str,
) -> Vec<stack_report::ExcludedSourceRecord> {
    indices
        .iter()
        .filter_map(|&image_index| images.get(image_index))
        .map(|image| stack_report::ExcludedSourceRecord {
            path: image.filename.clone(),
            reason: reason.to_string(),
        })
        .collect()
}

/// Binding 需求 1.1 evidence of one Capture_Station: the weakest accepted
/// relation inside it.
///
/// Every member joined the station through at least this much evidence, so it is
/// the number 需求 1.7 should carry. Returns `None` for a single-member station,
/// which has no intra-station relation to report.
pub(super) fn focus_station_binding_evidence(
    members: &[usize],
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Option<StationEvidence> {
    let mut weakest: Option<StationEvidence> = None;
    for (offset, &left) in members.iter().enumerate() {
        for &right in &members[offset + 1..] {
            let key = (left.min(right), left.max(right));
            let Some(match_info) = matches.get(&key) else {
                continue;
            };
            let Some(evidence) = focus_station_link_evidence(images, key.0, key.1, match_info)
                .filter(StationEvidence::accepts)
            else {
                continue;
            };
            if weakest.is_none_or(|current| evidence.score < current.score) {
                weakest = Some(evidence);
            }
        }
    }
    weakest
}

pub(super) fn lock_focus_local_bracket_poses(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    global_homographies: &HashMap<usize, Matrix3<f64>>,
) -> HashMap<usize, Matrix3<f64>> {
    let components = focus_local_bracket_components(images, matches);
    if components.is_empty() {
        return global_homographies.clone();
    }
    let mut locked = global_homographies.clone();
    let mut locked_sources = 0usize;
    let mut maximum_center_shift = 0.0f64;

    for component in &components {
        let members = component.iter().copied().collect::<HashSet<_>>();
        let mut edges = matches
            .iter()
            .filter_map(|(&(source_index, target_index), match_info)| {
                if !members.contains(&source_index)
                    || !members.contains(&target_index)
                    || !focus_match_is_local_bracket(
                        &images[source_index],
                        &images[target_index],
                        match_info,
                    )
                {
                    return None;
                }
                let error = median_symmetric_error(&match_info.homography, &match_info.points);
                let precision = if error.is_finite() {
                    1.0 / (1.0 + error / FOCUS_MODEL_INLIER_THRESHOLD)
                } else {
                    0.0
                };
                Some((
                    match_info.inliers.max(match_info.points.len()) as f64 * precision,
                    source_index,
                    target_index,
                    match_info.homography,
                ))
            })
            .collect::<Vec<_>>();
        edges.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| natural_path_cmp(&images[left.1].filename, &images[right.1].filename))
                .then_with(|| natural_path_cmp(&images[left.2].filename, &images[right.2].filename))
        });

        let mut support = HashMap::<usize, f64>::new();
        for &(weight, source_index, target_index, _) in &edges {
            *support.entry(source_index).or_default() += weight;
            *support.entry(target_index).or_default() += weight;
        }
        let Some(anchor) = component.iter().copied().min_by(|left, right| {
            support
                .get(right)
                .copied()
                .unwrap_or(0.0)
                .total_cmp(&support.get(left).copied().unwrap_or(0.0))
                .then_with(|| natural_path_cmp(&images[*left].filename, &images[*right].filename))
        }) else {
            continue;
        };
        let Some(anchor_global) = global_homographies.get(&images[anchor].id).copied() else {
            continue;
        };

        let mut local_dsu = Dsu::new(images.len());
        let mut adjacency = HashMap::<usize, Vec<(usize, Matrix3<f64>)>>::new();
        for &(_, source_index, target_index, source_to_target) in &edges {
            if local_dsu.find(source_index) == local_dsu.find(target_index) {
                continue;
            }
            let Some(target_to_source) = source_to_target.try_inverse() else {
                continue;
            };
            local_dsu.union(source_index, target_index);
            adjacency
                .entry(source_index)
                .or_default()
                .push((target_index, target_to_source));
            adjacency
                .entry(target_index)
                .or_default()
                .push((source_index, source_to_target));
        }

        let mut relative_to_anchor = HashMap::new();
        relative_to_anchor.insert(anchor, Matrix3::identity());
        let mut pending = VecDeque::from([anchor]);
        while let Some(current) = pending.pop_front() {
            let current_to_anchor = relative_to_anchor[&current];
            for &(neighbor, neighbor_to_current) in adjacency.get(&current).into_iter().flatten() {
                if relative_to_anchor.contains_key(&neighbor) {
                    continue;
                }
                relative_to_anchor.insert(neighbor, current_to_anchor * neighbor_to_current);
                pending.push_back(neighbor);
            }
        }
        if relative_to_anchor.len() != component.len() {
            continue;
        }

        for &image_index in component {
            let replacement = anchor_global * relative_to_anchor[&image_index];
            if !transform_is_stable_for_focus_stack(&replacement, images[image_index].dimensions())
            {
                continue;
            }
            if let Some(previous) = global_homographies.get(&images[image_index].id) {
                let center = Point2::new(
                    images[image_index].width as f64 * 0.5,
                    images[image_index].height as f64 * 0.5,
                );
                if let (Some(before), Some(after)) = (
                    transformed_point(previous, center),
                    transformed_point(&replacement, center),
                ) {
                    maximum_center_shift = maximum_center_shift.max((after - before).norm());
                }
            }
            locked.insert(images[image_index].id, replacement);
            locked_sources += 1;
        }
    }
    println!(
        "  - Locked {} local focus-bracket component(s), {} source poses; maximum center correction {:.2}px",
        components.len(),
        locked_sources,
        maximum_center_shift
    );
    locked
}

pub(super) struct FocusCaptureGroup {
    pub(crate) members: Vec<usize>,
    pub(crate) anchor: usize,
    pub(crate) local_to_anchor: HashMap<usize, Matrix3<f64>>,
}

pub(super) struct FocusAnchorRenderFrames {
    pub(crate) homographies: HashMap<usize, Matrix3<f64>>,
    pub(crate) focus_warp: Option<FocusLayerWarp>,
    pub(crate) anchor_to_world: Vec<Matrix3<f64>>,
}

pub(super) fn focus_anchor_render_frames(
    groups: &[FocusCaptureGroup],
    sources: &[ImageInfo],
    global: &HashMap<usize, Matrix3<f64>>,
    warp: Option<&FocusLayerWarp>,
) -> Option<FocusAnchorRenderFrames> {
    let reference_anchor = sources.get(groups.first()?.anchor)?;
    let world_to_reference = global.get(&reference_anchor.id)?.try_inverse()?;
    let mut local = HashMap::new();
    let mut local_warp = warp.cloned();
    let mut placements = Vec::with_capacity(groups.len());
    for group in groups {
        let anchor = sources.get(group.anchor)?;
        let anchor_global = global.get(&anchor.id)?;
        let world_to_anchor = anchor_global.try_inverse()?;
        placements.push(normalized_homography(
            &(world_to_reference * anchor_global),
        )?);
        for &member in &group.members {
            let source = sources.get(member)?;
            local.insert(
                source.id,
                normalized_homography(&(world_to_anchor * global.get(&source.id)?))?,
            );
            if let Some(warp) = local_warp.as_mut() {
                for band in &mut warp.bands {
                    if let Some(pose) = band.homographies.get_mut(&source.id) {
                        *pose = normalized_homography(&(world_to_anchor * *pose))?;
                    }
                }
            }
        }
    }
    Some(FocusAnchorRenderFrames {
        homographies: local,
        focus_warp: local_warp,
        anchor_to_world: placements,
    })
}

#[derive(Clone, Copy, Debug)]
pub(super) struct StationRelationMeasurements {
    pub(crate) inliers: usize,
    pub(crate) median_error_px: f64,
    pub(crate) scale_ratio: f64,
    pub(crate) spatial_support: f64,
    pub(crate) low_frequency_mean_relative_difference: f64,
    pub(crate) edge_strength_ratio: f64,
    pub(crate) median_edge_orientation_difference_degrees: f64,
    pub(crate) preserves_convex_orientation: bool,
}

pub(super) fn station_relation_rejection_reasons(
    measurements: &StationRelationMeasurements,
) -> Vec<&'static str> {
    let mut reasons = Vec::with_capacity(8);
    if measurements.inliers < STATION_RELATION_MIN_INLIERS {
        reasons.push(degradation::STATION_RELATION_LOW_INLIERS);
    }
    if !measurements.median_error_px.is_finite()
        || measurements.median_error_px > STATION_RELATION_MAX_MEDIAN_ERROR_PX
    {
        reasons.push(degradation::STATION_RELATION_HIGH_REPROJECTION_ERROR);
    }
    if !measurements.scale_ratio.is_finite()
        || !(STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
            .contains(&measurements.scale_ratio)
    {
        reasons.push(degradation::STATION_RELATION_SCALE_OUT_OF_RANGE);
    }
    if !measurements.spatial_support.is_finite()
        || measurements.spatial_support < STATION_RELATION_MIN_SPATIAL_SUPPORT
    {
        reasons.push(degradation::STATION_RELATION_LOW_SPATIAL_SUPPORT);
    }
    if !measurements
        .low_frequency_mean_relative_difference
        .is_finite()
        || measurements.low_frequency_mean_relative_difference
            > STATION_RELATION_MAX_LOW_FREQUENCY_MEAN_RELATIVE_DIFFERENCE
    {
        reasons.push(degradation::STATION_RELATION_PHOTOMETRIC_MISMATCH);
    }
    if !measurements.edge_strength_ratio.is_finite()
        || !(STATION_RELATION_EDGE_STRENGTH_RATIO_MIN..=STATION_RELATION_EDGE_STRENGTH_RATIO_MAX)
            .contains(&measurements.edge_strength_ratio)
    {
        reasons.push(degradation::STATION_RELATION_EDGE_STRENGTH_MISMATCH);
    }
    if !measurements
        .median_edge_orientation_difference_degrees
        .is_finite()
        || measurements.median_edge_orientation_difference_degrees
            > STATION_RELATION_MAX_MEDIAN_EDGE_ORIENTATION_DIFFERENCE_DEGREES
    {
        reasons.push(degradation::STATION_RELATION_EDGE_ORIENTATION_MISMATCH);
    }
    if !measurements.preserves_convex_orientation {
        reasons.push(degradation::STATION_RELATION_NON_CONVEX_QUAD);
    }
    reasons
}

/// Apply the production Station_Relation rejection decision to its report.
///
/// One candidate increments `rejected_count` once, while every independently
/// failed criterion increments its stable reason bucket.  Keeping this at the
/// solver boundary makes combined defects visible without allowing a rejected
/// candidate to reach pose construction (requirements 6.5, 6.9, 6.10).
pub(super) fn station_relation_is_accepted_with_report(
    left: usize,
    right: usize,
    consensus_error_px: f64,
    measurements: &StationRelationMeasurements,
    report: &mut stack_report::StationRelationsReport,
) -> bool {
    let reasons = station_relation_rejection_reasons(measurements);
    if reasons.is_empty() {
        return true;
    }
    report.rejected_count += 1;
    for reason in reasons {
        *report
            .rejected_by_reason
            .entry(reason.to_string())
            .or_default() += 1;
        degradation::record_run_degradation(
            reason,
            serde_json::json!({
                "left": left,
                "right": right,
                "inliers": measurements.inliers,
                "median_error_px": measurements.median_error_px,
                "consensus_error_px": consensus_error_px,
                "scale_ratio": measurements.scale_ratio,
                "spatial_support": measurements.spatial_support,
                "low_frequency_mean_relative_difference": measurements.low_frequency_mean_relative_difference,
                "edge_strength_ratio": measurements.edge_strength_ratio,
                "median_edge_orientation_difference_degrees": measurements.median_edge_orientation_difference_degrees,
                "preserves_convex_orientation": measurements.preserves_convex_orientation,
            }),
        );
    }
    false
}

/// A topology candidate without a measured model was rejected, but no
/// Requirement 6 acceptance criterion was measured. Its matcher diagnostics
/// supply the separate unmeasured and failure-stage summaries at publication.
pub(super) fn record_station_relation_no_fit(
    _left: usize,
    _right: usize,
    report: &mut stack_report::StationRelationsReport,
) {
    report.rejected_count += 1;
}

pub(super) fn polygon_signed_double_area(points: &[Point2<f64>]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(left, right)| left.x * right.y - right.x * left.y)
        .sum()
}

pub(super) fn convex_hull_area(points: impl IntoIterator<Item = Point2<f64>>) -> f64 {
    polygon_signed_double_area(&convex_hull_vertices(points)).abs() * 0.5
}

/// Counter-clockwise convex hull vertices (Andrew's monotone chain); fewer
/// than three distinct finite points give an empty hull.
pub(super) fn convex_hull_vertices(
    points: impl IntoIterator<Item = Point2<f64>>,
) -> Vec<Point2<f64>> {
    let mut points = points
        .into_iter()
        .filter(|point| point.x.is_finite() && point.y.is_finite())
        .collect::<Vec<_>>();
    points.sort_by(|left, right| {
        left.x
            .total_cmp(&right.x)
            .then_with(|| left.y.total_cmp(&right.y))
    });
    points.dedup_by(|left, right| left.x == right.x && left.y == right.y);
    if points.len() < 3 {
        return Vec::new();
    }
    let cross = |origin: Point2<f64>, left: Point2<f64>, right: Point2<f64>| {
        (left.x - origin.x) * (right.y - origin.y) - (left.y - origin.y) * (right.x - origin.x)
    };
    let mut lower = Vec::with_capacity(points.len());
    for &point in &points {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], point) <= 0.0
        {
            lower.pop();
        }
        lower.push(point);
    }
    let mut upper = Vec::with_capacity(points.len());
    for &point in points.iter().rev() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], point) <= 0.0
        {
            upper.pop();
        }
        upper.push(point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

pub(super) fn clip_polygon_to_rectangle(
    polygon: &[Point2<f64>],
    dimensions: (u32, u32),
) -> Vec<Point2<f64>> {
    let mut clipped = polygon.to_vec();
    let boundaries: [(usize, f64, bool); 4] = [
        (0, 0.0, true),
        (0, dimensions.0 as f64, false),
        (1, 0.0, true),
        (1, dimensions.1 as f64, false),
    ];
    for (axis, limit, keep_greater) in boundaries {
        if clipped.is_empty() {
            break;
        }
        let input = std::mem::take(&mut clipped);
        let coordinate = |point: Point2<f64>| if axis == 0 { point.x } else { point.y };
        let inside = |point: Point2<f64>| {
            if keep_greater {
                coordinate(point) >= limit
            } else {
                coordinate(point) <= limit
            }
        };
        for index in 0..input.len() {
            let current = input[index];
            let previous = input[(index + input.len() - 1) % input.len()];
            let current_inside = inside(current);
            let previous_inside = inside(previous);
            if current_inside != previous_inside {
                let previous_coordinate = coordinate(previous);
                let current_coordinate = coordinate(current);
                let denominator = current_coordinate - previous_coordinate;
                if denominator.abs() > f64::EPSILON {
                    let t = (limit - previous_coordinate) / denominator;
                    clipped.push(previous + (current - previous) * t);
                }
            }
            if current_inside {
                clipped.push(current);
            }
        }
    }
    clipped
}

pub(super) fn station_relation_overlap_area(
    source_to_target: &Matrix3<f64>,
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    source_side: bool,
) -> Option<f64> {
    let target_to_source = source_to_target.try_inverse()?;
    let (footprint_dimensions, clip_dimensions, transform) = if source_side {
        (target_dimensions, source_dimensions, target_to_source)
    } else {
        (source_dimensions, target_dimensions, *source_to_target)
    };
    let footprint = [
        Point2::new(0.0, 0.0),
        Point2::new(footprint_dimensions.0 as f64, 0.0),
        Point2::new(footprint_dimensions.0 as f64, footprint_dimensions.1 as f64),
        Point2::new(0.0, footprint_dimensions.1 as f64),
    ]
    .into_iter()
    .map(|point| transformed_point(&transform, point))
    .collect::<Option<Vec<_>>>()?;
    let clipped = clip_polygon_to_rectangle(&footprint, clip_dimensions);
    let area = polygon_signed_double_area(&clipped).abs() * 0.5;
    (area.is_finite() && area > f64::EPSILON).then_some(area)
}

/// What requirement 6.4 counts as a relation's overlap area.
#[derive(Clone, Copy)]
pub(super) enum StationOverlapMeasure<'a> {
    /// Source-image evidence: the geometric intersection of the two frames,
    /// every pixel of which is image content.
    Rectangles,
    /// Virtual_Tile evidence: only pixels both Coverage_Masks cover. A tile's
    /// bounding rectangle also holds uncovered corners, which carry no content,
    /// can never hold an inlier and are never composited from both tiles
    /// (需求 6.1). Mask pixel `p` is the block `[f·p, f·p + f)` of the
    /// relation's pixels (`f = factor`, the [`VirtualTileMatchPlane`] layout).
    Covered {
        source: &'a stitching::CoverageMask,
        target: &'a stitching::CoverageMask,
        factor: u32,
    },
}

/// Area, in the relation's pixels, of the source-plane pixels covered in
/// `source_mask` whose image under `source_to_target` lands on a pixel
/// covered in `target_mask`. `None` when a mask does not tile its plane.
pub(super) fn virtual_tile_covered_overlap_area(
    source_mask: &stitching::CoverageMask,
    source_dimensions: (u32, u32),
    target_mask: &stitching::CoverageMask,
    target_dimensions: (u32, u32),
    factor: u32,
    source_to_target: &Matrix3<f64>,
) -> Option<f64> {
    let factor = factor.max(1);
    let tiles_plane = |mask: &stitching::CoverageMask, (width, height): (u32, u32)| {
        mask.dimensions() == (width / factor, height / factor)
    };
    if !tiles_plane(source_mask, source_dimensions) || !tiles_plane(target_mask, target_dimensions)
    {
        return None;
    }
    let scale = f64::from(factor);
    let offset = (scale - 1.0) * 0.5;
    let (width, height) = source_mask.dimensions();
    let (target_width, target_height) = target_mask.dimensions();
    // Rows are independent and integer counts add exactly, so the parallel
    // reduction does not depend on scheduling.
    let covered = (0..height)
        .into_par_iter()
        .map(|y| {
            (0..width)
                .filter(|&x| source_mask.is_covered(x, y))
                .filter(|&x| {
                    let centre =
                        Point2::new(scale * f64::from(x) + offset, scale * f64::from(y) + offset);
                    transformed_point(source_to_target, centre).is_some_and(|target| {
                        let column = ((target.x - offset) / scale).round();
                        let row = ((target.y - offset) / scale).round();
                        column >= 0.0
                            && row >= 0.0
                            && column < f64::from(target_width)
                            && row < f64::from(target_height)
                            && target_mask.is_covered(column as u32, row as u32)
                    })
                })
                .count() as u64
        })
        .sum::<u64>();
    Some(covered as f64 * scale * scale)
}

/// Requirement 6.4's literal support: inlier convex-hull area divided by the
/// overlap area, checked on both station planes. This is the source-image
/// measure (the frames' geometric intersection); Virtual_Tile evidence uses
/// [`station_relation_spatial_support_with`] over the doubly covered overlap.
pub(super) fn station_relation_spatial_support(
    points: &[(Point2<f64>, Point2<f64>)],
    source_to_target: &Matrix3<f64>,
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
) -> f64 {
    station_relation_spatial_support_with(
        points,
        source_to_target,
        source_dimensions,
        target_dimensions,
        StationOverlapMeasure::Rectangles,
    )
}

/// [`station_relation_spatial_support`] over the overlap `measure` defines.
pub(super) fn station_relation_spatial_support_with(
    points: &[(Point2<f64>, Point2<f64>)],
    source_to_target: &Matrix3<f64>,
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    measure: StationOverlapMeasure<'_>,
) -> f64 {
    let overlaps = match measure {
        StationOverlapMeasure::Rectangles => station_relation_overlap_area(
            source_to_target,
            source_dimensions,
            target_dimensions,
            true,
        )
        .zip(station_relation_overlap_area(
            source_to_target,
            source_dimensions,
            target_dimensions,
            false,
        )),
        StationOverlapMeasure::Covered {
            source,
            target,
            factor,
        } => source_to_target.try_inverse().and_then(|target_to_source| {
            virtual_tile_covered_overlap_area(
                source,
                source_dimensions,
                target,
                target_dimensions,
                factor,
                source_to_target,
            )
            .zip(virtual_tile_covered_overlap_area(
                target,
                target_dimensions,
                source,
                source_dimensions,
                factor,
                &target_to_source,
            ))
        }),
    };
    let Some((source_overlap, target_overlap)) =
        overlaps.filter(|&(source, target)| source > f64::EPSILON && target > f64::EPSILON)
    else {
        return 0.0;
    };
    let source_hull = convex_hull_area(points.iter().map(|point| point.0));
    let target_hull = convex_hull_area(points.iter().map(|point| point.1));
    (source_hull / source_overlap)
        .min(target_hull / target_overlap)
        .clamp(0.0, 1.0)
}

#[derive(Clone, Copy)]
pub(super) struct MeasuredStationRelationCandidate {
    pub(crate) measurements: StationRelationMeasurements,
    pub(crate) quality: f64,
}

/// Build the complete canonical Requirement 6 Station_Relation measurement.
/// Both provisional Virtual_Tile clustering and the final solver consume this
/// typed result and apply [`station_relation_rejection_reasons`] to it.
#[allow(clippy::too_many_arguments)]
pub(super) fn measure_station_relation_candidate(
    inliers: usize,
    source_to_target: &Matrix3<f64>,
    points: &[(Point2<f64>, Point2<f64>)],
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    overlap_measure: Option<StationOverlapMeasure<'_>>,
    dense_measurements: Option<FocusOverlapQuality>,
) -> Option<MeasuredStationRelationCandidate> {
    let median_error_px = median_symmetric_error(source_to_target, points);
    let precision = if median_error_px.is_finite() {
        1.0 / (1.0 + median_error_px / STATION_RELATION_MAX_MEDIAN_ERROR_PX)
    } else {
        0.0
    };
    let spatial = panorama_spatial_support(points, source_dimensions, target_dimensions);
    let overlap =
        panorama_transform_overlap_support(source_to_target, source_dimensions, target_dimensions);
    let dense_quality = dense_measurements.map(FocusOverlapQuality::legacy_tuple);
    let dense_weight = dense_quality.map_or(1.0, |(intensity, edge, orientation, _)| {
        0.25 + intensity.max(0.0) + 0.5 * edge.max(0.0) + 0.25 * orientation.max(0.0)
    });
    let quality = (inliers as f64).sqrt()
        * precision
        * (0.25 + 0.75 * spatial)
        * (0.25 + 0.75 * overlap)
        * dense_weight;
    if !quality.is_finite() || quality <= 0.0 {
        return None;
    }
    let virtual_tile_measurement =
        matches!(overlap_measure, Some(StationOverlapMeasure::Covered { .. }));
    let photometric = dense_measurements.map(|measurements| {
        if virtual_tile_measurement {
            virtual_tile_photometric_measurements(measurements)
        } else {
            (
                measurements.low_frequency_mean_relative_difference,
                measurements.edge_strength_ratio,
            )
        }
    });
    let edge_strength_ratio = photometric.map_or(f64::INFINITY, |(_, strength)| strength);
    Some(MeasuredStationRelationCandidate {
        measurements: StationRelationMeasurements {
            inliers,
            median_error_px,
            scale_ratio: if virtual_tile_measurement {
                station_relation_overlap_scale(source_to_target, points)
            } else {
                homography_scale_ratio(source_to_target)
            }
            .unwrap_or(f64::INFINITY),
            spatial_support: overlap_measure.map_or(0.0, |measure| {
                station_relation_spatial_support_with(
                    points,
                    source_to_target,
                    source_dimensions,
                    target_dimensions,
                    measure,
                )
            }),
            low_frequency_mean_relative_difference: photometric
                .map_or(f64::INFINITY, |(difference, _)| difference),
            edge_strength_ratio,
            median_edge_orientation_difference_degrees: dense_measurements
                .map_or(f64::INFINITY, |measurements| {
                    measurements.median_edge_orientation_difference_degrees
                }),
            preserves_convex_orientation: homography_preserves_focus_orientation(
                source_to_target,
                source_dimensions,
            ),
        },
        quality,
    })
}

#[derive(Clone, Copy)]
pub(super) struct FocusGroupEdgeCandidate {
    pub(crate) transform: Matrix3<f64>,
    pub(crate) quality: f64,
    pub(crate) independent_support: usize,
    pub(crate) inlier_count: usize,
    pub(crate) median_error_px: f64,
    pub(crate) scale_ratio: f64,
    pub(crate) spatial_support: f64,
    pub(crate) low_frequency_mean_relative_difference: f64,
    pub(crate) edge_strength_ratio: f64,
    pub(crate) median_edge_orientation_difference_degrees: f64,
    pub(crate) preserves_convex_orientation: bool,
}

/// Exact Closure_Optimizer M-estimator multiplier from requirement 7.3.
///
/// Residuals are measured in world-coordinate native pixels. Keeping this
/// independent of image dimensions makes the same geometric error receive the
/// same weight at every station and in the final report.
pub(super) fn closure_residual_weight(residual_px: f64) -> f64 {
    if residual_px.is_nan() {
        return 0.0;
    }
    let residual_px = residual_px.max(0.0);
    if residual_px <= 2.0 {
        1.0
    } else if residual_px <= 6.0 {
        1.0 - 0.9 * (residual_px - 2.0) / 4.0
    } else {
        0.1 * (6.0 / residual_px)
    }
}

pub(super) fn closure_weighted_total_residual(
    corrections: &[[f64; 2]],
    constraints: &[(usize, usize, [f64; 2], f64)],
) -> f64 {
    constraints
        .iter()
        .map(|&(left, right, observed, base_weight)| {
            let residual = (corrections[right][0] - corrections[left][0] - observed[0])
                .hypot(corrections[right][1] - corrections[left][1] - observed[1]);
            base_weight * closure_residual_weight(residual) * residual * residual
        })
        .sum()
}

pub(super) fn closure_relative_residual_decrease_converged(previous: f64, current: f64) -> bool {
    if !previous.is_finite() || !current.is_finite() || current > previous {
        return false;
    }
    if previous <= f64::EPSILON {
        return current <= f64::EPSILON;
    }
    (previous - current) / previous < CLOSURE_RELATIVE_RESIDUAL_DECREASE
}

#[derive(Debug)]
pub(super) struct ClosureTranslationSolve {
    pub(crate) corrections: Vec<[f64; 2]>,
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
}

pub(super) fn solve_robust_closure_translations(
    station_count: usize,
    reference_station: usize,
    constraints: &[(usize, usize, [f64; 2], f64)],
) -> Option<ClosureTranslationSolve> {
    if station_count < 2 || reference_station >= station_count || constraints.is_empty() {
        return None;
    }
    let variable_count = station_count.saturating_sub(1) * 2;
    let variable_index = |station: usize, axis: usize| {
        let compact = if station < reference_station {
            station
        } else {
            station - 1
        };
        compact * 2 + axis
    };
    let mut corrections = vec![[0.0f64; 2]; station_count];
    let mut previous_total = closure_weighted_total_residual(&corrections, constraints);
    let mut iterations = 0usize;
    let mut converged = false;

    for iteration in 1..=CLOSURE_MAX_ITERATIONS {
        let mut normal = nalgebra::DMatrix::<f64>::zeros(variable_count, variable_count);
        let mut rhs = nalgebra::DVector::<f64>::zeros(variable_count);
        for &(left, right, observed, base_weight) in constraints {
            let residual = [
                corrections[right][0] - corrections[left][0] - observed[0],
                corrections[right][1] - corrections[left][1] - observed[1],
            ];
            let magnitude = residual[0].hypot(residual[1]);
            let weight = base_weight * closure_residual_weight(magnitude);
            for axis in 0..2 {
                let left_variable = (left != reference_station).then(|| variable_index(left, axis));
                let right_variable =
                    (right != reference_station).then(|| variable_index(right, axis));
                if let Some(variable) = left_variable {
                    normal[(variable, variable)] += weight;
                    rhs[variable] -= weight * observed[axis];
                }
                if let Some(variable) = right_variable {
                    normal[(variable, variable)] += weight;
                    rhs[variable] += weight * observed[axis];
                }
                if let (Some(left_variable), Some(right_variable)) = (left_variable, right_variable)
                {
                    normal[(left_variable, right_variable)] -= weight;
                    normal[(right_variable, left_variable)] -= weight;
                }
            }
        }
        for variable in 0..variable_count {
            normal[(variable, variable)] += FOCUS_GLOBAL_DAMPING;
        }
        let solution = normal.lu().solve(&rhs)?;
        for station in 0..station_count {
            if station == reference_station {
                continue;
            }
            corrections[station] = [
                solution[variable_index(station, 0)],
                solution[variable_index(station, 1)],
            ];
        }
        iterations = iteration;
        let current_total = closure_weighted_total_residual(&corrections, constraints);
        if closure_relative_residual_decrease_converged(previous_total, current_total) {
            converged = true;
            break;
        }
        previous_total = current_total;
    }

    Some(ClosureTranslationSolve {
        corrections,
        iterations,
        converged,
    })
}

pub(super) fn station_relation_overlap_polygon(
    source_to_target: &Matrix3<f64>,
    source_dimensions: (u32, u32),
    target_dimensions: (u32, u32),
    source_side: bool,
) -> Option<Vec<Point2<f64>>> {
    let target_to_source = source_to_target.try_inverse()?;
    let (footprint_dimensions, clip_dimensions, transform) = if source_side {
        (target_dimensions, source_dimensions, target_to_source)
    } else {
        (source_dimensions, target_dimensions, *source_to_target)
    };
    let footprint = [
        Point2::new(0.0, 0.0),
        Point2::new(footprint_dimensions.0 as f64, 0.0),
        Point2::new(footprint_dimensions.0 as f64, footprint_dimensions.1 as f64),
        Point2::new(0.0, footprint_dimensions.1 as f64),
    ]
    .into_iter()
    .map(|point| transformed_point(&transform, point))
    .collect::<Option<Vec<_>>>()?;
    let clipped = clip_polygon_to_rectangle(&footprint, clip_dimensions);
    (!clipped.is_empty()).then_some(clipped)
}

pub(super) fn projected_polygon_short_side(
    polygon: &[Point2<f64>],
    local_to_world: &Matrix3<f64>,
) -> Option<f64> {
    let projected = polygon
        .iter()
        .copied()
        .map(|point| transformed_point(local_to_world, point))
        .collect::<Option<Vec<_>>>()?;
    let min_x = projected
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min);
    let max_x = projected
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = projected
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min);
    let max_y = projected
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let short_side = (max_x - min_x).min(max_y - min_y);
    (short_side.is_finite() && short_side > 0.0).then_some(short_side)
}

pub(super) fn maximum_station_corner_displacement(
    initial_pose: &Matrix3<f64>,
    corrected_pose: &Matrix3<f64>,
    dimensions: (u32, u32),
) -> Option<f64> {
    [
        Point2::new(0.0, 0.0),
        Point2::new(dimensions.0 as f64, 0.0),
        Point2::new(dimensions.0 as f64, dimensions.1 as f64),
        Point2::new(0.0, dimensions.1 as f64),
    ]
    .into_iter()
    .try_fold(0.0f64, |maximum, corner| {
        let initial = transformed_point(initial_pose, corner)?;
        let corrected = transformed_point(corrected_pose, corner)?;
        let displacement = (corrected - initial).norm();
        displacement
            .is_finite()
            .then_some(maximum.max(displacement))
    })
}

pub(super) fn clamp_station_translation_corrections(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    initial_poses: &[Matrix3<f64>],
    corrections: &mut [[f64; 2]],
) -> Option<(f64, usize)> {
    if groups.len() != initial_poses.len() || groups.len() != corrections.len() {
        return None;
    }
    let mut overlap_short_sides = vec![None::<f64>; groups.len()];
    for &(_, left, right, left_to_right, _, _) in group_edges {
        if left >= groups.len() || right >= groups.len() {
            continue;
        }
        let left_dimensions = images[groups[left].anchor].dimensions();
        let right_dimensions = images[groups[right].anchor].dimensions();
        for (station, source_side) in [(left, true), (right, false)] {
            let polygon = station_relation_overlap_polygon(
                &left_to_right,
                left_dimensions,
                right_dimensions,
                source_side,
            )?;
            let short_side = projected_polygon_short_side(&polygon, &initial_poses[station])?;
            overlap_short_sides[station] = Some(
                overlap_short_sides[station].map_or(short_side, |current| current.min(short_side)),
            );
        }
    }

    let mut clamped_stations = 0usize;
    let mut actual_maximum = 0.0f64;
    for station in 0..groups.len() {
        let correction = corrections[station];
        if !correction[0].is_finite() || !correction[1].is_finite() {
            return None;
        }
        let overlap_short_side = overlap_short_sides[station]?;
        let limit = (overlap_short_side * CLOSURE_OVERLAP_SHORT_SIDE_CORRECTION_RATIO)
            .min(CLOSURE_MAX_CORNER_CORRECTION_PX);
        let corrected_pose = Matrix3::new(
            1.0,
            0.0,
            correction[0],
            0.0,
            1.0,
            correction[1],
            0.0,
            0.0,
            1.0,
        ) * initial_poses[station];
        let displacement = maximum_station_corner_displacement(
            &initial_poses[station],
            &corrected_pose,
            images[groups[station].anchor].dimensions(),
        )?;
        if displacement > limit {
            let scale = if displacement > 0.0 {
                limit / displacement
            } else {
                0.0
            };
            corrections[station][0] *= scale;
            corrections[station][1] *= scale;
            clamped_stations += 1;
        }
        let clamped = corrections[station];
        let clamped_pose = Matrix3::new(1.0, 0.0, clamped[0], 0.0, 1.0, clamped[1], 0.0, 0.0, 1.0)
            * initial_poses[station];
        actual_maximum = actual_maximum.max(maximum_station_corner_displacement(
            &initial_poses[station],
            &clamped_pose,
            images[groups[station].anchor].dimensions(),
        )?);
    }
    Some((actual_maximum, clamped_stations))
}

pub(super) fn closure_residual_summary(residuals: &mut [f64]) -> Option<(f64, f64)> {
    if residuals.is_empty() || residuals.iter().any(|residual| !residual.is_finite()) {
        return None;
    }
    residuals.sort_by(f64::total_cmp);
    let median = residuals[residuals.len() / 2];
    let p95_index = residuals
        .len()
        .saturating_mul(95)
        .div_ceil(100)
        .saturating_sub(1);
    Some((median, residuals[p95_index]))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ClosureDirectPairP95 {
    pub(crate) left: usize,
    pub(crate) right: usize,
    pub(crate) p95_px: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ClosureAcceptance {
    Accepted,
    NoConstraints,
    UnreliableResidual,
    UnreliableIterations,
    UnreliablePairP95(ClosureDirectPairP95),
}

pub(super) fn closure_acceptance(
    station_count: usize,
    accepted_relation_count: usize,
    converged: bool,
    iterations: usize,
    residual_median_px: f64,
    direct_pairs: &[ClosureDirectPairP95],
) -> ClosureAcceptance {
    if accepted_relation_count <= station_count.saturating_sub(1) {
        return ClosureAcceptance::NoConstraints;
    }
    if !converged || iterations > CLOSURE_MAX_ITERATIONS {
        return ClosureAcceptance::UnreliableIterations;
    }
    if !residual_median_px.is_finite() || residual_median_px > CLOSURE_RESIDUAL_MEDIAN_LIMIT_PX {
        return ClosureAcceptance::UnreliableResidual;
    }
    direct_pairs
        .iter()
        .copied()
        .filter(|pair| !pair.p95_px.is_finite() || pair.p95_px > CLOSURE_DIRECT_PAIR_P95_LIMIT_PX)
        .max_by(|left, right| {
            left.p95_px
                .total_cmp(&right.p95_px)
                .then_with(|| (right.left, right.right).cmp(&(left.left, left.right)))
        })
        .map_or(
            ClosureAcceptance::Accepted,
            ClosureAcceptance::UnreliablePairP95,
        )
}

/// Measure overlap reprojection disagreement for every accepted direct
/// Station_Relation. The fixed 11×11 source-plane lattice is clipped by the
/// relation into the target tile, so every retained sample belongs to the
/// geometric overlap and repeated runs inspect identical world positions.
pub(super) fn closure_direct_pair_p95s(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    group_edges: &[(f64, usize, usize, Matrix3<f64>, usize, f64)],
    poses: &[Matrix3<f64>],
) -> Vec<ClosureDirectPairP95> {
    const SAMPLES_PER_AXIS: usize = 11;

    group_edges
        .iter()
        .map(|&(_, left, right, left_to_right, _, _)| {
            let mut errors = Vec::with_capacity(SAMPLES_PER_AXIS * SAMPLES_PER_AXIS);
            if let (Some(left_group), Some(right_group), Some(left_pose), Some(right_pose)) = (
                groups.get(left),
                groups.get(right),
                poses.get(left),
                poses.get(right),
            ) {
                let left_dimensions = images[left_group.anchor].dimensions();
                let right_dimensions = images[right_group.anchor].dimensions();
                for sample_y in 0..SAMPLES_PER_AXIS {
                    let y = (left_dimensions.1.saturating_sub(1) as f64) * sample_y as f64
                        / (SAMPLES_PER_AXIS - 1) as f64;
                    for sample_x in 0..SAMPLES_PER_AXIS {
                        let x = (left_dimensions.0.saturating_sub(1) as f64) * sample_x as f64
                            / (SAMPLES_PER_AXIS - 1) as f64;
                        let source = Point2::new(x, y);
                        let Some(target) = transformed_point(&left_to_right, source) else {
                            continue;
                        };
                        if target.x < 0.0
                            || target.y < 0.0
                            || target.x >= right_dimensions.0 as f64
                            || target.y >= right_dimensions.1 as f64
                        {
                            continue;
                        }
                        let (Some(left_world), Some(right_world)) = (
                            transformed_point(left_pose, source),
                            transformed_point(right_pose, target),
                        ) else {
                            continue;
                        };
                        let error = (left_world - right_world).norm();
                        if error.is_finite() {
                            errors.push(error);
                        }
                    }
                }
            }
            let p95_px = closure_residual_summary(&mut errors)
                .map(|(_, p95)| p95)
                .unwrap_or(f64::INFINITY);
            ClosureDirectPairP95 {
                left,
                right,
                p95_px,
            }
        })
        .collect()
}

pub(super) fn closure_offending_pair_record(
    pair: ClosureDirectPairP95,
    station_topology: &topology::StationTopology,
) -> stack_report::ClosureOffendingPairRecord {
    stack_report::ClosureOffendingPairRecord {
        left: pair.left,
        left_row: station_topology
            .row
            .get(pair.left)
            .copied()
            .unwrap_or(u32::MAX),
        left_column: station_topology
            .column
            .get(pair.left)
            .copied()
            .unwrap_or(u32::MAX),
        right: pair.right,
        right_row: station_topology
            .row
            .get(pair.right)
            .copied()
            .unwrap_or(u32::MAX),
        right_column: station_topology
            .column
            .get(pair.right)
            .copied()
            .unwrap_or(u32::MAX),
        p95_px: pair.p95_px,
    }
}

pub(super) fn station_tree_position_key(
    station_topology: &topology::StationTopology,
    station: usize,
) -> (u32, u32, usize) {
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
        station,
    )
}

pub(super) fn station_tree_edge_key(
    station_topology: &topology::StationTopology,
    left: usize,
    right: usize,
) -> ((u32, u32, usize), (u32, u32, usize)) {
    let left = station_tree_position_key(station_topology, left);
    let right = station_tree_position_key(station_topology, right);
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

/// Select a maximum-inlier spanning tree. The returned values are indices into
/// `relations`; relation score and input order never participate. Equal inlier
/// counts are ordered only by the stations' canonical row/column indices.
pub(super) fn maximum_inlier_spanning_tree_indices(
    station_count: usize,
    station_topology: &topology::StationTopology,
    relations: &[(usize, usize, usize)],
) -> Vec<usize> {
    let mut ordered = relations
        .iter()
        .enumerate()
        .filter(|(_, (left, right, _))| {
            *left < station_count && *right < station_count && left != right
        })
        .collect::<Vec<_>>();
    ordered.sort_by(|(_, left), (_, right)| {
        right.2.cmp(&left.2).then_with(|| {
            station_tree_edge_key(station_topology, left.0, left.1).cmp(&station_tree_edge_key(
                station_topology,
                right.0,
                right.1,
            ))
        })
    });

    let mut connectivity = Dsu::new(station_count);
    let mut selected = Vec::with_capacity(station_count.saturating_sub(1));
    for (index, &(left, right, _)) in ordered {
        if connectivity.find(left) == connectivity.find(right) {
            continue;
        }
        connectivity.union(left, right);
        selected.push(index);
        if selected.len() + 1 == station_count {
            break;
        }
    }
    selected
}
