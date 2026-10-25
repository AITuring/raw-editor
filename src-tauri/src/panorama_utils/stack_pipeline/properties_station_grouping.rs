// Stage 3 harness: Capture_Station membership, splits and the anchor frame
// ---------------------------------------------------------------------------
//
// Properties 1, 3, 4 and 5 all drive the Station_Grouper, so they share one
// fixture family and one set of helpers.  The fixture is
// `arb_artwork_focus_bracket()`: one focus bracket plus the neighbouring camera
// position, which is the smallest selection that carries both kinds of pair the
// membership evidence has to separate.
//
// The production functions reached here are the ones task 7.1 and task 7.2
// rewrote — `focus_station_link_evidence`, `focus_match_is_capture_station_link`,
// `focus_station_grouping`, `split_focus_local_component_by_motion` — plus the
// 需求 2.1 anchor selection of `focus_capture_groups` and
// `intra_station::select_anchor`.  They are reached through
// `grouping_test_access`, which only widens visibility.
//
// Every threshold below is written as the *requirement's* number rather than as
// the production constant, and the production constants are asserted to equal
// them.  A test that imported the constants would keep passing if a constant
// moved, which is the one failure these properties exist to catch.

/// 需求 1.1: verified overlap inlier correspondences of a station link.
const REQUIRED_MIN_INLIERS: usize = 30;

/// 需求 1.1: local normalised pixel correlation over the overlap.
const REQUIRED_MIN_OVERLAP_NCC: f64 = 0.6;

/// 需求 1.1: inlier spatial support inside the overlap.
const REQUIRED_MIN_SPATIAL_SUPPORT: f64 = 0.20;

/// 需求 1.1: scale-ratio window of two frames of one Capture_Station.
const REQUIRED_SCALE_RATIO: std::ops::RangeInclusive<f64> = 0.98..=1.02;

/// 需求 1.4: accumulated frame-centre displacement, in long sides, that splits a
/// Capture_Station.
const REQUIRED_MAX_ACCUMULATED_MOTION_RATIO: f64 = 0.02;

/// 需求 1.5: largest number of Source_RAWs one Capture_Station may hold.
const REQUIRED_MAX_STATION_MEMBERS: usize = 48;

/// 需求 2.1: median Sharpness_Score distance below which two anchor candidates
/// are a tie and the absolute path decides.
const REQUIRED_ANCHOR_SHARPNESS_TIE: f64 = 0.01;

/// The 需求 1.1 verdict, recomputed from the four measurements alone.
///
/// This is the independent oracle of Property 1: it applies the requirement's
/// own four thresholds to the numbers the production measurement returned, with
/// no access to the production decision.
fn requirement_accepts(evidence: &grouping_test_access::EvidenceView) -> bool {
    evidence.inliers >= REQUIRED_MIN_INLIERS
        // 需求 1.1 states a correlation threshold, which presupposes a measurable
        // correlation; `focus_overlap_quality` refuses to report one below 120
        // usable samples, and an unmeasurable correlation is evidence in neither
        // direction.
        && (!evidence.overlap_ncc_measured || evidence.overlap_ncc >= REQUIRED_MIN_OVERLAP_NCC)
        && evidence.spatial_support >= REQUIRED_MIN_SPATIAL_SUPPORT
        && REQUIRED_SCALE_RATIO.contains(&evidence.scale_ratio)
}

/// Connected components of the accepted-link graph, as a member set per
/// component, computed here so it is an independent answer to compare the
/// grouper's station partition against.
fn accepted_link_components(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
) -> Vec<BTreeSet<usize>> {
    let mut parent = (0..images.len()).collect::<Vec<_>>();
    fn root(parent: &mut [usize], mut index: usize) -> usize {
        while parent[index] != index {
            let grandparent = parent[parent[index]];
            parent[index] = grandparent;
            index = grandparent;
        }
        index
    }
    for key in sorted_keys(matches) {
        let match_info = &matches[&key];
        if !grouping_test_access::is_capture_station_link(images, key.0, key.1, match_info) {
            continue;
        }
        let (left, right) = (root(&mut parent, key.0), root(&mut parent, key.1));
        if left != right {
            parent[left] = right;
        }
    }
    let mut components = BTreeMap::<usize, BTreeSet<usize>>::new();
    for index in 0..images.len() {
        let representative = root(&mut parent, index);
        components.entry(representative).or_default().insert(index);
    }
    components.into_values().collect()
}

/// The `MatchInfo` of `source` with its point set and its fit replaced.
///
/// Used by the rejection halves of Property 1: the drawn relation is accepted, so
/// the only way to test that each of the four measurements is *necessary* is to
/// hand the measurement one deliberately broken input.
fn match_info_with(
    source: &MatchInfo,
    homography: Matrix3<f64>,
    points: Vec<(Point2<f64>, Point2<f64>)>,
) -> MatchInfo {
    MatchInfo {
        canonical_homography: Some(homography),
        homography,
        inliers: points.len(),
        sequence_bridge: false,
        coarse_bridge: false,
        points,
        candidate_points: Vec::new(),
        top_candidate_points: Vec::new(),
        dense_focus_points: Vec::new(),
        foreground_feature_points: source.foreground_feature_points.clone(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 1: 对于任意
    // Source_RAW 集合与任意两张之间的匹配证据，两张 Source_RAW 属于同一 Capture_Station
    // 当且仅当它们之间存在一条验证过的证据链，链上每条边同时满足内点 ≥30、重叠归一化相关
    // ≥0.6、内点空间支持 ≥20%、尺度比 ∈ `[0.98, 1.02]`。
    //
    // 三个方向都要断言，否则这条属性会在生产代码退化时保持绿色：
    //   1. 边判定的充要性：`focus_match_is_capture_station_link` 的结论等于把需求 1.1 的
    //      四个阈值直接套在实测量上的结论（阈值在测试里按需求原文写死，并断言生产常量等于
    //      它们）。
    //   2. 链的必要性：每个 Capture_Station 在被接受边构成的子图上连通；链的充分性：没有
    //      需求 1.4 拆分时，被接受边的连通分量恰好是一个 Capture_Station。
    //   3. 四个量各自的必要性：把一条已被接受的关系分别改成内点 29 对、尺度比 1.03、
    //      内点聚成一团、以及同一位置但不同内容（重叠相关塌到 0），每一种都必须由接受
    //      翻为拒绝。第四种用真实的第二张底图裁出，所以几何量全部保持合格，唯一失败的
    //      判据就是相关性。
    //
    // **Validates: Requirements 1.1**
    #[test]
    fn capture_station_membership_follows_the_image_evidence_only(
        scan in arb_artwork_focus_bracket(),
        probe in any::<usize>(),
    ) {
        // The production constants are the requirement's numbers.
        prop_assert_eq!(grouping_test_access::MIN_INLIER_MATCHES, REQUIRED_MIN_INLIERS);
        prop_assert_eq!(grouping_test_access::MIN_OVERLAP_NCC, REQUIRED_MIN_OVERLAP_NCC);
        prop_assert_eq!(
            grouping_test_access::MIN_INLIER_SPATIAL_SUPPORT,
            REQUIRED_MIN_SPATIAL_SUPPORT
        );
        prop_assert_eq!(grouping_test_access::SCALE_RATIO_MIN, *REQUIRED_SCALE_RATIO.start());
        prop_assert_eq!(grouping_test_access::SCALE_RATIO_MAX, *REQUIRED_SCALE_RATIO.end());

        // The decoy is part of the selection under test, not a separate fixture:
        // it is the only way to reach the correlation gate with every geometric
        // measurement still passing (see assertion 3d).
        let (scan, decoy) = scan.with_isolated_source(
            test_support::DecoySource::UnrelatedContent,
            0x0005_EED1_u64.wrapping_mul(probe as u64 | 1),
        );
        let images = scan.images();
        let (matches, _measured) = fit_synthetic_matches(&scan, &images);

        // 1. Edge decision: necessary and sufficient, on every measured pair.
        let mut accepted_bracket_pairs = Vec::new();
        let mut rejected_station_pairs = 0usize;
        for key in sorted_keys(&matches) {
            let match_info = &matches[&key];
            let decision =
                grouping_test_access::is_capture_station_link(&images, key.0, key.1, match_info);
            let evidence =
                grouping_test_access::station_link_evidence(&images, key.0, key.1, match_info);
            let expected = evidence.as_ref().is_some_and(requirement_accepts);
            prop_assert_eq!(
                decision,
                expected,
                "the membership decision of {:?} disagrees with 需求 1.1 applied to its own \
                 measurements {:?}",
                key,
                evidence
            );
            let same_station =
                scan.sources()[key.0].station == scan.sources()[key.1].station;
            if decision {
                prop_assert!(
                    same_station,
                    "{:?} joins two different camera positions on evidence {:?}",
                    key,
                    evidence
                );
                accepted_bracket_pairs.push(key);
            } else if !same_station {
                rejected_station_pairs += 1;
            }
        }
        // Coverage: without both kinds of pair the equivalence above would hold
        // for a fixture the grouper never had to separate.
        prop_assert!(
            !accepted_bracket_pairs.is_empty(),
            "the fixture must produce at least one accepted focus bracket relation"
        );
        prop_assert!(
            rejected_station_pairs >= 1,
            "the fixture must produce at least one rejected inter-station relation"
        );

        // 2. The station partition is the transitive closure of those decisions.
        let grouping = grouping_test_access::station_grouping(&images, &matches);
        let link_components = accepted_link_components(&images, &matches);
        let linked = link_components
            .iter()
            .filter(|component| component.len() >= 2)
            .count();
        let mut station_members = Vec::new();
        for station in &grouping.components {
            let members = station.iter().copied().collect::<BTreeSet<_>>();
            prop_assert_eq!(
                members.len(),
                station.len(),
                "a Capture_Station must not list a member twice"
            );
            // Necessity of the chain: the accepted-link subgraph over one
            // station's members is connected.
            let component = link_components
                .iter()
                .find(|component| component.contains(station.first().expect("non-empty")))
                .expect("every source belongs to a component");
            prop_assert!(
                members.is_subset(component),
                "station {:?} spans more than one accepted-link component",
                members
            );
            station_members.push(members);
        }
        for (first, left) in station_members.iter().enumerate() {
            for right in &station_members[first + 1..] {
                prop_assert!(
                    left.is_disjoint(right),
                    "two Capture_Stations share a member: {left:?} and {right:?}"
                );
            }
        }
        // Sufficiency of the chain: every station the grouper emitted plus every
        // single member a 需求 1.4 split left over accounts for exactly one
        // component, so with no split recorded a component *is* a station.
        let stations_from_components = station_members.len()
            + link_components
                .iter()
                .filter(|component| component.len() >= 2)
                .map(|component| {
                    component
                        .iter()
                        .filter(|member| {
                            !station_members.iter().any(|station| station.contains(member))
                        })
                        .count()
                })
                .sum::<usize>();
        prop_assert_eq!(
            stations_from_components,
            linked + grouping.splits.len(),
            "each accepted-link component must become exactly one Capture_Station per recorded \
             split (components {}, splits {:?})",
            linked,
            grouping.splits
        );
        // 需求 1.6 seen from this side: a source with no accepted relation at all
        // is in no component of size two and in no station.
        let unlinked = (0..images.len())
            .filter(|index| {
                !link_components
                    .iter()
                    .any(|component| component.len() >= 2 && component.contains(index))
            })
            .collect::<Vec<_>>();
        prop_assert_eq!(
            &grouping.isolated,
            &unlinked,
            "the isolated set must be exactly the sources without an accepted relation"
        );
        prop_assert!(
            grouping.isolated.contains(&decoy),
            "the unrelated-content decoy shares no Capture_Station with anything"
        );

        // 3. Each of the four measurements is necessary on its own.
        let key = accepted_bracket_pairs[probe % accepted_bracket_pairs.len()];
        let accepted = &matches[&key];
        let baseline = grouping_test_access::station_link_evidence(&images, key.0, key.1, accepted)
            .expect("an accepted relation carries measured evidence");
        prop_assert!(baseline.accepts);

        // 3a. 内点 ≥30: one correspondence short is not a station link.
        let mut short = accepted.points.clone();
        short.sort_by(|left, right| {
            left.0.x.total_cmp(&right.0.x).then(left.0.y.total_cmp(&right.0.y))
        });
        short.truncate(REQUIRED_MIN_INLIERS - 1);
        let short = match_info_with(accepted, accepted.homography, short);
        let short_evidence =
            grouping_test_access::station_link_evidence(&images, key.0, key.1, &short);
        prop_assert_eq!(
            short_evidence.map(|evidence| (evidence.inliers, evidence.accepts)),
            Some((REQUIRED_MIN_INLIERS - 1, false)),
            "{} verified inliers must not be a Capture_Station link",
            REQUIRED_MIN_INLIERS - 1
        );

        // 3b. 尺度比 ∈ [0.98, 1.02]: a 3% enlargement about the frame centre is a
        //     different shooting distance, not a focus bracket.
        let centre = Point2::new(
            f64::from(images[key.0].width) * 0.5,
            f64::from(images[key.0].height) * 0.5,
        );
        let mut enlargement = Matrix3::identity();
        enlargement[(0, 0)] = 1.03;
        enlargement[(1, 1)] = 1.03;
        enlargement[(0, 2)] = centre.x * (1.0 - 1.03);
        enlargement[(1, 2)] = centre.y * (1.0 - 1.03);
        let scaled_homography = enlargement * accepted.homography;
        let scaled_points = accepted
            .points
            .iter()
            .filter_map(|&(source, target)| {
                map_point(&enlargement, target).map(|target| (source, target))
            })
            .collect::<Vec<_>>();
        prop_assert_eq!(scaled_points.len(), accepted.points.len());
        let scaled = match_info_with(accepted, scaled_homography, scaled_points);
        let scaled_evidence =
            grouping_test_access::station_link_evidence(&images, key.0, key.1, &scaled);
        if let Some(evidence) = scaled_evidence {
            prop_assert!(
                evidence.scale_ratio > *REQUIRED_SCALE_RATIO.end(),
                "the enlargement must be visible in the measured scale ratio, got {evidence:?}"
            );
            prop_assert!(
                !evidence.accepts,
                "a scale ratio outside [0.98, 1.02] must not be a station link, got {evidence:?}"
            );
        }

        // 3c. 内点空间支持 ≥20%: 36 correspondences crowded into a few pixels
        //     support nothing, however many of them there are.
        let anchor_point = accepted.points[0];
        let clustered = (0..REQUIRED_MIN_INLIERS + 6)
            .filter_map(|index| {
                let offset = Point2::new(
                    anchor_point.0.x + (index % 3) as f64 * 0.5,
                    anchor_point.0.y + (index / 3) as f64 * 0.5,
                );
                map_point(&accepted.homography, offset).map(|target| (offset, target))
            })
            .collect::<Vec<_>>();
        let clustered = match_info_with(accepted, accepted.homography, clustered);
        let clustered_evidence =
            grouping_test_access::station_link_evidence(&images, key.0, key.1, &clustered);
        if let Some(evidence) = clustered_evidence {
            prop_assert!(
                evidence.inliers >= REQUIRED_MIN_INLIERS,
                "the clustered variant must still clear the inlier count, got {evidence:?}"
            );
            prop_assert!(
                evidence.spatial_support < REQUIRED_MIN_SPATIAL_SUPPORT,
                "correspondences inside a 2px box cannot support 20% of the overlap, got \
                 {evidence:?}"
            );
            prop_assert!(
                !evidence.accepts,
                "inlier spatial support below 20% must not be a station link, got {evidence:?}"
            );
        }

        // 3d. 重叠归一化相关 ≥0.6: the decoy sits at the *same* pose as a real
        //     frame and depicts another artwork, so the inlier count, the spatial
        //     support and the scale ratio all pass and the correlation is the only
        //     failing judgement.
        let reference = (0..scan.sources().len())
            .find(|&index| {
                scan.sources()[index].station == 0
                    && scan.sources()[index].layer == 0
                    && index != decoy
            })
            .expect("the fixture has a first layer of its first station");
        let decoy_key = (reference.min(decoy), reference.max(decoy));
        let decoy_match = matches
            .get(&decoy_key)
            .expect("the decoy overlaps the reference frame exactly, so a fit exists");
        let decoy_evidence =
            grouping_test_access::station_link_evidence(&images, decoy_key.0, decoy_key.1, decoy_match)
                .expect("the decoy pair carries measured evidence");
        prop_assert!(
            decoy_evidence.inliers >= REQUIRED_MIN_INLIERS
                && decoy_evidence.spatial_support >= REQUIRED_MIN_SPATIAL_SUPPORT
                && REQUIRED_SCALE_RATIO.contains(&decoy_evidence.scale_ratio),
            "the decoy must fail on the correlation alone, got {decoy_evidence:?}"
        );
        prop_assert!(
            decoy_evidence.overlap_ncc_measured
                && decoy_evidence.overlap_ncc < REQUIRED_MIN_OVERLAP_NCC,
            "two different artworks at the same place must not correlate, got {decoy_evidence:?}"
        );
        prop_assert!(
            !decoy_evidence.accepts,
            "an overlap correlation below 0.6 must not be a station link, got {decoy_evidence:?}"
        );
    }
}

/// Candidate order and per-step frame-centre displacement of one component,
/// computed from the *ground truth* poses of the fixture.
///
/// This is the independent oracle of Property 3.  `split_focus_local_component_by_motion`
/// recovers the same order from the verified link graph and measures the same
/// displacements from the recovered relative poses; the numbers here come from
/// the poses the generator cropped the tiles with, so agreement is a statement
/// about the production code rather than a restatement of it.
///
/// The recovered frame is the plane scaled by the inverse of the anchor frame's
/// focus breathing, so a measured ratio differs from the ratio computed here by
/// up to the breathing amplitude (0.15% in `arb_focus_bracket`).  Property 3
/// therefore skips candidates whose accumulated displacement sits within
/// [`ACCUMULATED_MOTION_DECISION_BAND`] of the 0.02 budget instead of pretending
/// the two frames are the same one.
fn ground_truth_candidate_order(scan: &SyntheticScan, component: &BTreeSet<usize>) -> Vec<usize> {
    let mut ordered = component.iter().copied().collect::<Vec<_>>();
    ordered.sort_by(|&left, &right| {
        let left_centre = scan.plane_centre(left);
        let right_centre = scan.plane_centre(right);
        left_centre
            .y
            .total_cmp(&right_centre.y)
            .then_with(|| left_centre.x.total_cmp(&right_centre.x))
    });
    ordered
}

/// Width of the band around 需求 1.4's 0.02 budget in which the ground truth
/// oracle and the production measurement are allowed to disagree, in long sides.
///
/// Five percent of the budget, i.e. 30 times the focus breathing amplitude the
/// two frames differ by.
const ACCUMULATED_MOTION_DECISION_BAND: f64 = 1e-3;

/// Accumulated frame-centre displacement per candidate position, with the
/// 需求 1.4 reset at every split, computed from ground truth.
///
/// Returns `(split positions, accumulated value at every position)`.
fn ground_truth_accumulated_motion(
    scan: &SyntheticScan,
    ordered: &[usize],
) -> (BTreeSet<usize>, Vec<f64>) {
    let long_side = f64::from(test_support::SYNTHETIC_TILE_SIDE);
    let mut splits = BTreeSet::new();
    let mut accumulated = vec![0.0f64; ordered.len()];
    let mut running = 0.0f64;
    for position in 1..ordered.len() {
        let step = (scan.plane_centre(ordered[position])
            - scan.plane_centre(ordered[position - 1]))
        .norm()
            / long_side;
        let candidate = running + step;
        accumulated[position] = candidate;
        if candidate > REQUIRED_MAX_ACCUMULATED_MOTION_RATIO {
            splits.insert(position);
            running = 0.0;
        } else {
            running = candidate;
        }
    }
    (splits, accumulated)
}

/// Whether Property 3 has seen a component that split and one that did not.
///
/// The accumulated displacement of a drawn bracket is data, not a rotation, so
/// this is a coverage witness rather than a schedule: bit 0 is "a component that
/// stayed whole", bit 1 is "a component that 需求 1.4 split".  Asserted once, on
/// the last case, so a fixture that degenerated into only one of the two outcomes
/// fails instead of quietly testing half the property.
static ACCUMULATED_MOTION_OUTCOMES: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 3 so far.
static ACCUMULATED_MOTION_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 3 is configured with, mirrored here so the coverage assertion
/// knows which case is the last one.
const ACCUMULATED_MOTION_CASES: usize = 100;

proptest! {
    #![proptest_config(ProptestConfig { cases: ACCUMULATED_MOTION_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 3: 对于任意候选序列与其
    // 累计画面中心位移序列，Station_Grouper 的拆分位置集合恰好等于累计位移超过画面长边
    // 0.02 倍的位置集合；且拆分后每个 Capture_Station 的成员数量不超过 48。
    //
    // 两个生成器分别覆盖两条判据：
    //   * `arb_artwork_focus_bracket()` 的机位内层序列驱动需求 1.4 的累计位移拆分。期望
    //     拆分位置由生成器的**真值位姿**独立算出（生产代码是从被接受的链接图里重新恢复
    //     相对位姿再测量的），所以这条断言比较的是两套独立的计算。
    //   * `arb_oversized_bracket()` 的 49..60 成员分量驱动需求 1.5 / 1.10 的成员上限：
    //     该分量的累计位移按构造低于 0.02，因此唯一能拆分它的就是 48 的上限，且首个拆分
    //     必须落在位移最大的相邻候选处。
    //
    // **Validates: Requirements 1.4, 1.5, 1.10**
    #[test]
    fn accumulated_motion_split_positions_are_uniquely_determined(
        scan in arb_artwork_focus_bracket(),
        oversized in arb_oversized_bracket(),
    ) {
        prop_assert_eq!(
            grouping_test_access::MAX_ACCUMULATED_CENTER_MOTION_RATIO,
            REQUIRED_MAX_ACCUMULATED_MOTION_RATIO
        );
        prop_assert_eq!(
            grouping_test_access::MAX_STATION_MEMBERS,
            REQUIRED_MAX_STATION_MEMBERS
        );

        // ---------------------------------------------------------------
        // 需求 1.4: the split positions of one focus bracket
        // ---------------------------------------------------------------
        let images = scan.images();
        let (matches, _measured) = fit_synthetic_matches(&scan, &images);
        let components = accepted_link_components(&images, &matches)
            .into_iter()
            .filter(|component| component.len() >= 2)
            .collect::<Vec<_>>();
        prop_assert!(
            !components.is_empty(),
            "the fixture must produce a verified-link component to split"
        );
        let mut split_components = 0usize;
        let mut whole_components = 0usize;
        for component in &components {
            let ordered = ground_truth_candidate_order(&scan, component);
            let (expected_splits, accumulated) = ground_truth_accumulated_motion(&scan, &ordered);
            // A candidate whose accumulated displacement sits on the budget is a
            // measurement the two frames cannot agree on to the last bit; skip
            // the case rather than assert a coin flip.
            for value in &accumulated {
                prop_assume!(
                    (value - REQUIRED_MAX_ACCUMULATED_MOTION_RATIO).abs()
                        > ACCUMULATED_MOTION_DECISION_BAND
                );
            }
            let (groups, splits) = grouping_test_access::split_component_by_motion(
                component.iter().copied().collect(),
                &images,
                &matches,
            );
            // Every member is measurable here: the layers of one camera position
            // all overlap each other, so the link graph reaches all of them and no
            // split can come from an unrecoverable pose.
            for split in &splits {
                prop_assert!(
                    split.motion_ratio.is_finite(),
                    "a fully linked bracket must not split on an unmeasurable pose: {split:?}"
                );
            }
            let measured_splits = splits
                .iter()
                .filter(|split| split.reason == degradation::GROUPING_ACCUMULATED_MOTION)
                .map(|split| split.position)
                .collect::<BTreeSet<_>>();
            prop_assert_eq!(
                &measured_splits,
                &expected_splits,
                "需求 1.4 split positions must be exactly the positions where the accumulated \
                 frame-centre displacement passes {} (accumulated {:?}, order {:?})",
                REQUIRED_MAX_ACCUMULATED_MOTION_RATIO,
                accumulated,
                ordered
            );
            for split in &splits {
                if split.reason == degradation::GROUPING_ACCUMULATED_MOTION {
                    prop_assert!(
                        (split.motion_ratio - accumulated[split.position]).abs()
                            <= ACCUMULATED_MOTION_DECISION_BAND,
                        "the recorded displacement {} of the split at {} disagrees with the \
                         ground truth {}",
                        split.motion_ratio,
                        split.position,
                        accumulated[split.position]
                    );
                }
            }
            // The stations are the runs of the candidate order between the splits,
            // so the split positions determine the membership completely.
            let mut expected_groups = Vec::<Vec<usize>>::new();
            for (position, &member) in ordered.iter().enumerate() {
                if position == 0 || expected_splits.contains(&position) {
                    expected_groups.push(Vec::new());
                }
                expected_groups
                    .last_mut()
                    .expect("a group was pushed for position 0")
                    .push(member);
            }
            prop_assert_eq!(
                &groups,
                &expected_groups,
                "the Capture_Stations of one component must be the runs of its candidate order"
            );
            for group in &groups {
                prop_assert!(
                    !group.is_empty() && group.len() <= REQUIRED_MAX_STATION_MEMBERS,
                    "a Capture_Station holds 1..={} sources, got {}",
                    REQUIRED_MAX_STATION_MEMBERS,
                    group.len()
                );
            }
            if measured_splits.is_empty() {
                whole_components += 1;
            } else {
                split_components += 1;
            }
        }

        // ---------------------------------------------------------------
        // 需求 1.5 / 1.10: the 48 member ceiling
        // ---------------------------------------------------------------
        prop_assert!(
            oversized.len() > REQUIRED_MAX_STATION_MEMBERS,
            "the ceiling is only reachable above {} members",
            REQUIRED_MAX_STATION_MEMBERS
        );
        prop_assert!(
            oversized.accumulated_motion_ratio < REQUIRED_MAX_ACCUMULATED_MOTION_RATIO,
            "the oversized fixture must not be splittable by 需求 1.4, got {}",
            oversized.accumulated_motion_ratio
        );
        let (bounded, ceiling_splits) = grouping_test_access::split_component_by_motion(
            (0..oversized.len()).collect(),
            &oversized.images,
            &oversized.matches,
        );
        for split in &ceiling_splits {
            prop_assert_eq!(
                split.reason,
                degradation::GROUPING_MEMBER_LIMIT_SPLIT,
                "an accumulated displacement below the budget must not split this component: \
                 {:?}",
                split
            );
        }
        for group in &bounded {
            prop_assert!(
                !group.is_empty() && group.len() <= REQUIRED_MAX_STATION_MEMBERS,
                "the member ceiling was not enforced: a station of {} sources",
                group.len()
            );
        }
        prop_assert_eq!(
            bounded.iter().flatten().copied().collect::<Vec<_>>(),
            (0..oversized.len()).collect::<Vec<_>>(),
            "the ceiling split must keep every member, in candidate order"
        );
        prop_assert_eq!(
            ceiling_splits.len(),
            bounded.len() - 1,
            "one recorded split per new Capture_Station (需求 1.10)"
        );
        // 需求 1.10 splits at the adjacent candidate pair carrying the largest
        // accumulated displacement, and the fixture puts a unique maximum there.
        prop_assert_eq!(
            ceiling_splits[0].position,
            oversized.largest_step_position,
            "the first ceiling split must fall at the largest frame-centre displacement"
        );
        let expected_largest_ratio = test_support::OVERSIZED_BRACKET_LARGEST_STEP_PX
            / f64::from(test_support::OVERSIZED_BRACKET_SIDE);
        prop_assert!(
            (ceiling_splits[0].motion_ratio - expected_largest_ratio).abs() < 1e-9,
            "the recorded displacement {} of the first ceiling split is not the largest step {}",
            ceiling_splits[0].motion_ratio,
            expected_largest_ratio
        );

        // Coverage of the 需求 1.4 dimension across the run.
        let outcomes = ACCUMULATED_MOTION_OUTCOMES.fetch_or(
            usize::from(whole_components > 0) | (usize::from(split_components > 0) << 1),
            Ordering::Relaxed,
        ) | usize::from(whole_components > 0)
            | (usize::from(split_components > 0) << 1);
        let case = ACCUMULATED_MOTION_CURSOR.fetch_add(1, Ordering::Relaxed);
        prop_assert!(
            case + 1 < ACCUMULATED_MOTION_CASES || outcomes == 0b11,
            "the run must have exercised both a bracket 需求 1.4 kept whole and one it split, \
             outcome mask {outcomes:#b}"
        );
    }
}

/// The Capture_Station partition of one grouping, as capture identities.
///
/// Members are compared by what they *are* — `(station, focus layer)` of the
/// fixture — never by import index, so a comparison survives a selection that
/// gained one more source.
///
/// `focus_station_grouping` reports only the components of two or more members,
/// so a source that a 需求 1.4 split left on its own is added back here as the
/// single-member Capture_Station it is (需求 1.5 allows one).  Without that the
/// partition of a heavily drifting bracket would look empty rather than like the
/// several one-frame stations it is.
fn station_identity_partition(
    scan: &SyntheticScan,
    grouping: &grouping_test_access::GroupingView,
) -> BTreeSet<BTreeSet<(usize, usize)>> {
    let identities = scan.capture_identities();
    let mut partition = grouping
        .components
        .iter()
        .map(|component| {
            component
                .iter()
                .filter_map(|&member| identities.get(member).copied())
                .collect::<BTreeSet<_>>()
        })
        .collect::<BTreeSet<_>>();
    for index in 0..identities.len() {
        let in_station = grouping
            .components
            .iter()
            .any(|component| component.contains(&index));
        if in_station || grouping.isolated.contains(&index) {
            continue;
        }
        partition.insert(BTreeSet::from([identities[index]]));
    }
    partition
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 4: 对于任意 Source_RAW
    // 集合，向其加入任意一张与所有其他源图都不满足重叠证据的源图或任意一张不可解码的源图，
    // 其余源图的 Capture_Station 划分与不加入该源图时完全相同，且新加入的源图被记录在孤立
    // 或解码失败列表中并附带文件路径与原因标识符。
    //
    // 孤立那一半是真正的对照实验：同一批源图分组两次，第二次多了一张无法与任何源图成立
    // 需求 1.1 证据的源图。两种失败方式都在输入空间内——`UnrelatedContent` 与某个机位
    // 完全重叠但画的是另一幅画（内点数、空间支持、尺度比全部合格，靠相关性被拒），
    // `NoOverlap` 则根本没有重叠可测——所以"不满足重叠证据"的两条路径都被走到。
    //
    // 解码失败那一半只能走到需求 1.9 在 `cargo test --lib` 里可达的部分：
    // `undecodable_source_paths` 是把该源图从每个 Capture_Station 中排除的那一步本身
    // （分组器只见得到已解码的图像），记录则由 `excluded_source_records` 与
    // `DegradationLedger` 给出。真正调用它们的 `stitch_images_with_options` 需要
    // `AppHandle` 与磁盘上的 RAW，由阶段 8 的 Acceptance_Harness 覆盖。
    //
    // **Validates: Requirements 1.6, 1.9**
    #[test]
    fn an_unusable_source_leaves_the_other_groupings_untouched(
        scan in arb_artwork_focus_bracket(),
        kind in prop::sample::select(test_support::DecoySource::ALL.to_vec()),
        seed in any::<u64>(),
    ) {
        // The selection without the unusable source: the control group.
        let reference_images = scan.images();
        let (reference_matches, _measured) =
            fit_synthetic_matches(&scan, &reference_images);
        let reference = grouping_test_access::station_grouping(&reference_images, &reference_matches);
        let reference_partition = station_identity_partition(&scan, &reference);
        prop_assert!(
            !reference_partition.is_empty(),
            "the control grouping must contain at least one Capture_Station"
        );

        // The same selection plus one source that belongs to no Capture_Station.
        let (with_decoy, decoy) = scan.with_isolated_source(kind, seed);
        let images = with_decoy.images();
        let (matches, _measured) = fit_synthetic_matches(&with_decoy, &images);
        let decoy_pairs = sorted_keys(&matches)
            .into_iter()
            .filter(|key| key.0 == decoy || key.1 == decoy)
            .collect::<Vec<_>>();
        // Coverage: the two failure modes have to be different failures.
        match kind {
            test_support::DecoySource::UnrelatedContent => {
                prop_assert_eq!(
                    decoy_pairs.len(),
                    1,
                    "the unrelated-content decoy must produce exactly one measurable relation"
                );
                let evidence = grouping_test_access::station_link_evidence(
                    &images,
                    decoy_pairs[0].0,
                    decoy_pairs[0].1,
                    &matches[&decoy_pairs[0]],
                )
                .expect("the decoy overlaps a real frame, so the evidence is measurable");
                prop_assert!(
                    !evidence.accepts && evidence.inliers >= REQUIRED_MIN_INLIERS,
                    "the decoy must be rejected on measured evidence, not on missing evidence: \
                     {evidence:?}"
                );
            }
            test_support::DecoySource::NoOverlap => {
                prop_assert!(
                    decoy_pairs.is_empty(),
                    "a frame of somewhere else must not produce a fit at all, got {decoy_pairs:?}"
                );
            }
        }

        let grouping = grouping_test_access::station_grouping(&images, &matches);

        // 1. 其余源图的划分完全相同.
        prop_assert_eq!(
            station_identity_partition(&with_decoy, &grouping),
            reference_partition.clone(),
            "adding a {:?} source changed the Capture_Station partition of the others",
            kind
        );
        // 2. 不并入任何 Capture_Station.
        prop_assert!(
            grouping
                .components
                .iter()
                .all(|component| !component.contains(&decoy)),
            "the unusable source joined a Capture_Station"
        );
        // 3. 被记录在孤立列表中并附带文件路径与原因标识符 (需求 1.6).  The other
        //    sources keep whatever isolation they had, which is the same property
        //    seen from the other side.
        let mut expected_isolated = reference.isolated.clone();
        expected_isolated.push(decoy);
        prop_assert_eq!(
            &grouping.isolated,
            &expected_isolated,
            "adding a {:?} source changed which other sources are isolated",
            kind
        );
        let isolated_records = grouping_test_access::excluded_sources(
            &images,
            &grouping.isolated,
            degradation::GROUPING_NO_OVERLAP_EVIDENCE,
        );
        prop_assert_eq!(isolated_records.len(), expected_isolated.len());
        let decoy_record = isolated_records
            .last()
            .expect("the decoy is the last isolated source");
        prop_assert_eq!(&decoy_record.path, &with_decoy.source_path(decoy));
        for record in &isolated_records {
            prop_assert!(
                Path::new(&record.path).is_absolute(),
                "需求 1.6 records the absolute path, got {}",
                record.path
            );
            prop_assert_eq!(&record.reason, degradation::GROUPING_NO_OVERLAP_EVIDENCE);
            prop_assert!(
                frozen_reason_identifiers().contains(record.reason.as_str()),
                "the isolation reason must be a frozen identifier"
            );
        }

        // 4. 不可解码的源图 (需求 1.9): the same selection, but this time the extra
        //    source is the one that did not decode, so the grouper is handed the
        //    other sources and nothing else.
        let selection = with_decoy.source_paths();
        let undecodable = grouping_test_access::undecodable_sources(&selection, &reference_images);
        prop_assert_eq!(
            &undecodable,
            &vec![with_decoy.source_path(decoy)],
            "the undecodable set must name exactly the source that produced no image"
        );
        prop_assert!(
            reference_images
                .iter()
                .all(|image| !undecodable.contains(&image.filename)),
            "an undecodable source must not reach the grouper at all"
        );
        prop_assert_eq!(
            station_identity_partition(&scan, &reference),
            reference_partition,
            "the surviving sources keep the grouping they would have had anyway"
        );
        // One `source_decode_failed` entry per excluded source, carrying its path.
        let mut ledger = DegradationLedger::new();
        for path in &undecodable {
            ledger.record(
                degradation::SOURCE_DECODE_FAILED,
                serde_json::json!({ "path": path }),
            );
        }
        let decoy_path = with_decoy.source_path(decoy);
        prop_assert_eq!(ledger.entries().len(), undecodable.len());
        prop_assert_eq!(ledger.entries()[0].reason, degradation::SOURCE_DECODE_FAILED);
        prop_assert_eq!(
            ledger.entries()[0].detail["path"].as_str(),
            Some(decoy_path.as_str())
        );
        let decode_records = grouping_test_access::excluded_sources(
            &images,
            &[decoy],
            degradation::SOURCE_DECODE_FAILED,
        );
        prop_assert_eq!(&decode_records[0].path, &with_decoy.source_path(decoy));
        prop_assert_eq!(&decode_records[0].reason, degradation::SOURCE_DECODE_FAILED);
    }
}

/// The 需求 2.1 anchor rule, applied to a candidate list from scratch.
///
/// The independent oracle of Property 5: highest median Sharpness_Score wins;
/// every candidate less than 0.01 below the highest is a tie and the lowest
/// absolute path among those wins; a candidate set with no measurable score at
/// all falls back to the lowest absolute path.
fn expected_anchor(candidates: &[intra_station::AnchorCandidate]) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }
    let best = candidates
        .iter()
        .map(|candidate| candidate.median_sharpness)
        .filter(|score| !score.is_nan())
        .fold(f64::NEG_INFINITY, f64::max);
    let lowest_path = |allowed: &dyn Fn(&intra_station::AnchorCandidate) -> bool| {
        candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| allowed(candidate))
            .min_by(|(left_index, left), (right_index, right)| {
                left.path
                    .cmp(&right.path)
                    .then_with(|| left_index.cmp(right_index))
            })
            .map(|(index, _)| index)
    };
    if !best.is_finite() {
        return lowest_path(&|_| true);
    }
    lowest_path(&|candidate| best - candidate.median_sharpness < REQUIRED_ANCHOR_SHARPNESS_TIE)
}

/// Which branch of 需求 2.1 decided the anchor of a drawn candidate list.
///
/// Bit 0: every candidate was inside the 0.01 tie window, so the absolute path
/// decided.  Bit 1: at least one candidate was more than the tie window below the
/// best, so the median Sharpness_Score decided.  Asserted on the last case, so a
/// generator that stopped producing one of the two fails rather than quietly
/// testing half the rule.
static ANCHOR_DECISION_BRANCHES: AtomicUsize = AtomicUsize::new(0);

/// Accepted cases of Property 5 so far, which also drives the rotation of the
/// station-level half.
static ANCHOR_DECISION_CURSOR: AtomicUsize = AtomicUsize::new(0);

/// Cases Property 5 is configured with.
const ANCHOR_DECISION_CASES: usize = 100;

/// Whether the station-level half of Property 5 has reached a Capture_Station of
/// several frames.
static ANCHOR_MULTI_MEMBER_SEEN: AtomicUsize = AtomicUsize::new(0);

/// Every how many cases the station-level half of Property 5 runs.
///
/// That half needs the station pose solve and `focus_capture_groups`, roughly
/// half a second per case, while the rule half costs nothing.  The wiring it
/// checks — which member is handed to `select_anchor` and whether the anchor's
/// transform is the identity — is not input dependent in the way the rule is, so
/// a quarter of the cases covers it at a quarter of the cost.  The whole run
/// still visits 25 independently drawn scans.
const ANCHOR_STATION_STRIDE: usize = 4;

proptest! {
    #![proptest_config(ProptestConfig { cases: ANCHOR_DECISION_CASES as u32, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 5: 对于任意
    // Capture_Station，被选为锚点帧的 Source_RAW 是有效像素 Sharpness_Score 中位数最高者；
    // 若最高值之差小于 0.01 则为其中绝对路径升序第一者；且锚点帧到机位坐标系的变换是
    // 恒等矩阵。
    //
    // 两层都要断言，因为它们会各自失效：
    //   1. 规则本身 —— `intra_station::select_anchor` 在任意候选集合上与独立重算的
    //      需求 2.1 规则一致，且结果不随候选顺序改变。分数落在 0.004 的网格上，所以
    //      "分数决定" 与 "并列后按路径决定" 两个分支都会被走到（运行结束时断言两者都出现
    //      过），不可测量的分数集合也在输入空间内。
    //   2. 接线 —— 真实 `focus_capture_groups` 为每个 Capture_Station 选出的锚点确实是
    //      `focus_frame_median_sharpness` 最高的成员（按需求 2.1 的规则重算），且
    //      `local_to_anchor[anchor]` 等于单位矩阵。任务 7.4 之前这里取的是世界中心排序的
    //      第一帧，那个实现能通过第 1 层的全部断言而在这里变红。
    //
    // 实测记录（合成素材，非生产缺陷断言）：`focus_frame_median_sharpness` 在 160px
    // 合成瓦片上的取值区间是 0.0010–0.0021（未归一化的 acutance 比值），因此同一机位内
    // 各层之差最大约 0.001，永远落在需求 2.1 的 0.01 并列窗口内——第 2 层因此只会走到
    // 路径 tie-break 分支。合成瓦片的高频内容只有颗粒（被 acutance 的二项式低通压掉）
    // 与少量笔画，真实 9504px 素材在 spacing≈4.6 下测得的是完全不同的量级，所以这是
    // 素材尺度的限制而不是判定规则的缺陷；规则的另一半由第 1 层在抽样分数上覆盖。
    //
    // **Validates: Requirements 2.1**
    #[test]
    fn the_anchor_frame_is_the_sharpest_member_and_maps_to_itself(
        candidates in test_support::arb_anchor_candidates(),
        scan in arb_artwork_focus_bracket(),
        probe in any::<usize>(),
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        prop_assert_eq!(
            intra_station::INTRA_STATION_ANCHOR_SHARPNESS_TIE,
            REQUIRED_ANCHOR_SHARPNESS_TIE
        );

        // 1. The rule.
        let chosen = intra_station::select_anchor(&candidates);
        prop_assert_eq!(
            chosen,
            expected_anchor(&candidates),
            "需求 2.1 picks another candidate from {:?}",
            candidates
        );
        let chosen = chosen.expect("a non-empty candidate list has an anchor");
        let best = candidates
            .iter()
            .map(|candidate| candidate.median_sharpness)
            .filter(|score| !score.is_nan())
            .fold(f64::NEG_INFINITY, f64::max);
        if best.is_finite() {
            // Necessity of the two halves of the rule: the winner is inside the
            // tie window, and nothing inside it sorts before its path.
            prop_assert!(
                best - candidates[chosen].median_sharpness < REQUIRED_ANCHOR_SHARPNESS_TIE,
                "the anchor must be within {} of the highest median Sharpness_Score",
                REQUIRED_ANCHOR_SHARPNESS_TIE
            );
            prop_assert!(
                candidates
                    .iter()
                    .filter(|candidate| {
                        best - candidate.median_sharpness < REQUIRED_ANCHOR_SHARPNESS_TIE
                    })
                    .all(|candidate| candidate.path >= candidates[chosen].path),
                "a tied candidate sorts before the chosen anchor {:?}",
                candidates[chosen]
            );
        }
        // The choice is a property of the photographs, not of the order they are
        // offered in.
        let mut rotated = candidates.clone();
        rotated.rotate_left(probe % candidates.len().max(1));
        let rotated_choice = intra_station::select_anchor(&rotated)
            .expect("a rotation keeps the list non-empty");
        prop_assert_eq!(
            &rotated[rotated_choice].path,
            &candidates[chosen].path,
            "the anchor changed when the candidates were offered in another order"
        );

        // Coverage of the two branches of the rule, on the drawn candidates.
        let branch = if best.is_finite()
            && candidates.iter().any(|candidate| {
                best - candidate.median_sharpness >= REQUIRED_ANCHOR_SHARPNESS_TIE
            })
        {
            0b10
        } else {
            0b01
        };
        let covered = ANCHOR_DECISION_BRANCHES.fetch_or(branch, Ordering::Relaxed) | branch;
        let case = ANCHOR_DECISION_CURSOR.fetch_add(1, Ordering::Relaxed);
        prop_assert!(
            case + 1 < ANCHOR_DECISION_CASES || covered == 0b11,
            "the run must have exercised both an anchor decided by its median Sharpness_Score \
             and one decided by the absolute path tie-break, branch mask {covered:#b}"
        );

        // 2. The wiring: the anchor of a real Capture_Station and its transform.
        //
        //    Rotated, see `ANCHOR_STATION_STRIDE`.  A `prop_assume!` would be
        //    wrong here: a rejected case does not count as a success, so
        //    `proptest` would simply draw until 100 cases *did* run this half.
        if !case.is_multiple_of(ANCHOR_STATION_STRIDE) {
            return Ok(());
        }
        let images = determinism_images(&scan);
        let locked = scan.locked_homographies();
        let (matches, _measured) = fit_synthetic_matches(&scan, &images);
        let solve = determinism_test_access::solve_station_poses_with_report(
            &images,
            &matches,
            &locked,
            scan.geometric_reference_index(),
        );
        let identities = scan.capture_identities();
        let geometry = synthetic_geometry_outcome(&images, &identities, &solve);
        let grouping_poses = match &geometry {
            SyntheticGeometryOutcome::Connected => &solve.poses,
            SyntheticGeometryOutcome::RejectedDisconnected { .. } => &locked,
        };
        let stations =
            determinism_test_access::capture_stations(&images, &matches, grouping_poses)
                .unwrap_or_default();
        if matches!(geometry, SyntheticGeometryOutcome::RejectedDisconnected { .. }) {
            let permuted = scan.with_permuted_import_order();
            let permuted_images = determinism_images(&permuted);
            let permuted_locked = permuted.locked_homographies();
            let (permuted_matches, _) = fit_synthetic_matches(&permuted, &permuted_images);
            let permuted_solve = determinism_test_access::solve_station_poses_with_report(
                &permuted_images,
                &permuted_matches,
                &permuted_locked,
                permuted.geometric_reference_index(),
            );
            let permuted_geometry = synthetic_geometry_outcome(
                &permuted_images,
                &permuted.capture_identities(),
                &permuted_solve,
            );
            prop_assert_eq!(
                &geometry,
                &permuted_geometry,
                "a disconnected anchor fixture must retain the same explicit rejection report under import permutation"
            );
        }
        prop_assert!(
            stations.iter().any(|station| station.members.len() >= 2),
            "the anchor decision is only under test on a station with several frames"
        );
        let mut multi_member_stations = 0usize;
        for station in &stations {
            let station_candidates = station
                .members
                .iter()
                .map(|&member| intra_station::AnchorCandidate {
                    path: images[member].filename.clone(),
                    median_sharpness: grouping_test_access::frame_median_sharpness(&images[member]),
                })
                .collect::<Vec<_>>();
            let expected = expected_anchor(&station_candidates)
                .map(|position| station.members[position])
                .expect("a station has at least one member");
            prop_assert_eq!(
                station.anchor,
                expected,
                "the Capture_Station anchor must be its highest median Sharpness_Score frame \
                 (candidates {:?})",
                station_candidates
            );
            // 锚点帧到机位坐标系的变换是恒等矩阵.
            let into_anchor = station
                .local_to_anchor
                .get(&station.anchor)
                .copied()
                .expect("the anchor carries a transform into its own station frame");
            // `anchor_global.try_inverse() * anchor_global` is a floating point
            // product, so the identity it produces is the identity to within a
            // few ulps rather than bit for bit — the measured worst coefficient
            // over the fixture is 1.4e-14 on the perspective row.  Both halves of
            // "is the identity" are therefore asserted: the normalised
            // coefficients, and the only thing a transform is for, namely that
            // every frame corner maps to itself to far below a pixel.
            let normalised = into_anchor / into_anchor[(2, 2)];
            for (index, (value, expected)) in normalised
                .iter()
                .zip(Matrix3::<f64>::identity().iter())
                .enumerate()
            {
                prop_assert!(
                    (value - expected).abs() <= 1e-12,
                    "coefficient {index} of the anchor transform is {value}, not {expected}"
                );
            }
            let width = f64::from(images[station.anchor].width);
            let height = f64::from(images[station.anchor].height);
            for corner in [
                Point2::new(0.0, 0.0),
                Point2::new(width, 0.0),
                Point2::new(0.0, height),
                Point2::new(width, height),
            ] {
                let mapped = map_point(&into_anchor, corner)
                    .expect("the anchor transform is invertible");
                prop_assert!(
                    (mapped - corner).norm() <= 1e-9,
                    "the anchor frame corner {corner:?} maps to {mapped:?}"
                );
            }
            multi_member_stations += usize::from(station.members.len() >= 2);
        }
        // A bracket whose accumulated drift split every frame into its own
        // Capture_Station is a legitimate 需求 1.4 outcome, and then there is no
        // anchor decision to make; the run as a whole still has to reach one.
        let stations_covered = ANCHOR_MULTI_MEMBER_SEEN
            .fetch_or(usize::from(multi_member_stations >= 1), Ordering::Relaxed)
            | usize::from(multi_member_stations >= 1);
        prop_assert!(
            case + ANCHOR_STATION_STRIDE < ANCHOR_DECISION_CASES || stations_covered == 1,
            "the run must have reached a Capture_Station of several frames, so that 需求 2.1 \
             really had an anchor to choose"
        );
    }
}

// ---------------------------------------------------------------------------
