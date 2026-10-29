//! Station_Grouper membership properties (1, 4).
//!
//! Membership follows the image evidence of 需求 1.1 only, and an unusable source
//! leaves every other grouping untouched. The requirement thresholds defined here
//! are shared with the split and anchor properties.

use super::super::degradation;
use super::super::degradation::DegradationLedger;
use super::super::determinism::sorted_keys;
use super::super::test_support;
use super::super::test_support::SyntheticScan;
use super::super::test_support::arb_artwork_focus_bracket;
use crate::panorama_stitching::grouping_test_access;
use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::shared::MatchInfo;
use nalgebra::Matrix3;
use nalgebra::Point2;
use proptest::prelude::*;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::Path;

use super::{fit_synthetic_matches, frozen_reason_identifiers, map_point};

// ---------------------------------------------------------------------------
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
pub(super) const REQUIRED_MIN_INLIERS: usize = 30;

/// 需求 1.1: local normalised pixel correlation over the overlap.
pub(super) const REQUIRED_MIN_OVERLAP_NCC: f64 = 0.6;

/// 需求 1.1: inlier spatial support inside the overlap.
pub(super) const REQUIRED_MIN_SPATIAL_SUPPORT: f64 = 0.20;

/// 需求 1.1: scale-ratio window of two frames of one Capture_Station.
pub(super) const REQUIRED_SCALE_RATIO: std::ops::RangeInclusive<f64> = 0.98..=1.02;

/// 需求 1.4: accumulated frame-centre displacement, in long sides, that splits a
/// Capture_Station.
pub(super) const REQUIRED_MAX_ACCUMULATED_MOTION_RATIO: f64 = 0.02;

/// 需求 1.5: largest number of Source_RAWs one Capture_Station may hold.
pub(super) const REQUIRED_MAX_STATION_MEMBERS: usize = 48;

/// 需求 2.1: median Sharpness_Score distance below which two anchor candidates
/// are a tie and the absolute path decides.
pub(super) const REQUIRED_ANCHOR_SHARPNESS_TIE: f64 = 0.01;

/// The 需求 1.1 verdict, recomputed from the four measurements alone.
///
/// This is the independent oracle of Property 1: it applies the requirement's
/// own four thresholds to the numbers the production measurement returned, with
/// no access to the production decision.
pub(super) fn requirement_accepts(evidence: &grouping_test_access::EvidenceView) -> bool {
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
pub(super) fn accepted_link_components(
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
pub(super) fn match_info_with(
    source: &MatchInfo,
    homography: Matrix3<f64>,
    points: Vec<(Point2<f64>, Point2<f64>)>,
) -> MatchInfo {
    MatchInfo {
        canonical_homography: Some(homography),
        native_verified_points: Vec::new(),
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
        let mut scan_linked=BTreeSet::new();
        for key in sorted_keys(&matches) {
            let record=&matches[&key];
            if grouping_test_access::has_verified_scan_overlap(&images,key.0,key.1,record) {
                prop_assert!(!record.coarse_bridge && !record.sequence_bridge);
                prop_assert!(record.points.len()>=30);
                scan_linked.insert(key.0);scan_linked.insert(key.1);
            }
        }
        let scan_singletons=link_components.iter().filter(|component|component.len()==1 && component.iter().any(|id|scan_linked.contains(id))).count();
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
            linked + grouping.splits.len() + scan_singletons,
            "each accepted-link component must become exactly one Capture_Station per recorded \
             split (components {}, splits {:?})",
            linked,
            grouping.splits
        );
        // 需求 1.6 seen from this side: a source with no accepted relation at all
        // is in no component of size two; a verified spatial scan overlap
        // independently preserves a source as a single-member station.
        let unlinked = (0..images.len())
            .filter(|index| {
                !link_components
                    .iter()
                    .any(|component| component.len() >= 2 && component.contains(index))
                    && !scan_linked.contains(index)
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
pub(super) fn station_identity_partition(
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
