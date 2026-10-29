//! Station_Pose_Solver properties (30-34).
//!
//! Relation defects map to exact reasons, poses keep eight degrees of freedom with
//! a bounded local scale, disconnected graphs reject publication, the spanning
//! tree has maximum support with stable ties, and every accepted relation takes
//! part in the closure.

use super::super::degradation;
use super::super::degradation::DegradationLedger;
use nalgebra::Matrix3;
use proptest::prelude::*;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use tempfile::TempDir;

use super::{
    above, accepted_relation_measurements, below, reason_identifier_pattern, splitmix64, unit_from,
};

// ---------------------------------------------------------------------------
// Properties 30–34: Station_Relation diagnostics, pose publication, graph
// connectivity, maximum-support tree, and closure participation
// ---------------------------------------------------------------------------

pub(super) const STATION_RELATION_DEFECT_REASONS: [&str; 8] = [
    degradation::STATION_RELATION_LOW_INLIERS,
    degradation::STATION_RELATION_HIGH_REPROJECTION_ERROR,
    degradation::STATION_RELATION_SCALE_OUT_OF_RANGE,
    degradation::STATION_RELATION_LOW_SPATIAL_SUPPORT,
    degradation::STATION_RELATION_PHOTOMETRIC_MISMATCH,
    degradation::STATION_RELATION_EDGE_STRENGTH_MISMATCH,
    degradation::STATION_RELATION_EDGE_ORIENTATION_MISMATCH,
    degradation::STATION_RELATION_NON_CONVEX_QUAD,
];

pub(super) fn relation_measurements_with_defects(
    defect_mask: u8,
) -> crate::panorama_stitching::station_relation_test_access::MeasurementsView {
    use crate::panorama_stitching::station_relation_test_access::MeasurementsView;

    let mut measurements = accepted_relation_measurements();
    if defect_mask & (1 << 0) != 0 {
        measurements.inliers = 23;
    }
    if defect_mask & (1 << 1) != 0 {
        measurements.median_error_px = above(3.0);
    }
    if defect_mask & (1 << 2) != 0 {
        measurements.scale_ratio = above(1.05);
    }
    if defect_mask & (1 << 3) != 0 {
        measurements.spatial_support = below(0.20);
    }
    if defect_mask & (1 << 4) != 0 {
        measurements.low_frequency_mean_relative_difference = above(0.20);
    }
    if defect_mask & (1 << 5) != 0 {
        measurements.edge_strength_ratio = above(1.40);
    }
    if defect_mask & (1 << 6) != 0 {
        measurements.median_edge_orientation_difference_degrees = above(10.0);
    }
    if defect_mask & (1 << 7) != 0 {
        measurements.preserves_convex_orientation = false;
    }
    MeasurementsView { ..measurements }
}

pub(super) fn station_relation_reason_oracle(defect_mask: u8) -> Vec<&'static str> {
    STATION_RELATION_DEFECT_REASONS
        .iter()
        .enumerate()
        .filter_map(|(defect, reason)| (defect_mask & (1 << defect) != 0).then_some(*reason))
        .collect()
}

pub(super) fn matrix_bits(matrix: &Matrix3<f64>) -> [u64; 9] {
    let mut bits = [0; 9];
    for (slot, value) in bits.iter_mut().zip(matrix.iter()) {
        *slot = value.to_bits();
    }
    bits
}

pub(super) fn translation_pose(x: f64, y: f64) -> Matrix3<f64> {
    Matrix3::new(1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0)
}

pub(super) fn graph_component_oracle(
    station_member_paths: &[Vec<String>],
    relation_pairs: &[(usize, usize)],
) -> Vec<Vec<String>> {
    let mut component = (0..station_member_paths.len()).collect::<Vec<_>>();
    for &(left, right) in relation_pairs {
        if left >= component.len() || right >= component.len() {
            continue;
        }
        let from = component[right];
        let to = component[left];
        for label in &mut component {
            if *label == from {
                *label = to;
            }
        }
    }
    let mut by_component = BTreeMap::<usize, Vec<String>>::new();
    for (station, members) in station_member_paths.iter().enumerate() {
        by_component
            .entry(component[station])
            .or_default()
            .extend(members.iter().cloned());
    }
    let mut components = by_component.into_values().collect::<Vec<_>>();
    for members in &mut components {
        members.sort();
    }
    components.sort();
    components
}

pub(super) fn assert_graph_outcome_and_publication(
    station_member_paths: &[Vec<String>],
    relation_pairs: &[(usize, usize)],
) -> Result<(), TestCaseError> {
    use crate::panorama_utils::stack_pipeline::degradation::{
        DegradationManager, OutcomeDecision, OutputPublication, StandardFinalOutputFileSystem,
    };

    let outcome = crate::panorama_stitching::station_relation_test_access::station_graph_outcome(
        station_member_paths,
        relation_pairs,
    );
    let expected = graph_component_oracle(station_member_paths, relation_pairs);
    let mut actual = outcome.report.connectivity.component_members.clone();
    for members in &mut actual {
        members.sort();
    }
    actual.sort();
    prop_assert_eq!(&actual, &expected);
    prop_assert_eq!(outcome.report.connectivity.components, expected.len());
    let expected_counts = expected.iter().map(Vec::len).collect::<Vec<_>>();
    let mut actual_counts = outcome.report.connectivity.component_member_counts.clone();
    actual_counts.sort_unstable();
    let mut expected_counts_sorted = expected_counts.clone();
    expected_counts_sorted.sort_unstable();
    prop_assert_eq!(actual_counts, expected_counts_sorted);

    let disconnected = expected.len() > 1;
    prop_assert_eq!(
        outcome.rejection_reason,
        disconnected.then_some(degradation::GEOMETRY_DISCONNECTED)
    );
    let mut ledger = DegradationLedger::new();
    if let (Some(reason), Some(detail)) =
        (outcome.rejection_reason, outcome.rejection_detail.clone())
    {
        ledger.record(reason, detail);
    }

    let directory = TempDir::new()
        .map_err(|error| TestCaseError::fail(format!("publication tempdir failed: {error}")))?;
    let staged = directory.path().join("result.tmp");
    let final_path = directory.path().join("result.tiff");
    fs::write(&staged, b"property-32-output")
        .map_err(|error| TestCaseError::fail(format!("staged output write failed: {error}")))?;
    let mut file_system = StandardFinalOutputFileSystem;
    let publication = DegradationManager::new(&ledger)
        .publish_staged_output(&mut file_system, &staged, &final_path)
        .map_err(TestCaseError::fail)?;

    if disconnected {
        prop_assert_eq!(publication, OutputPublication::Rejected);
        prop_assert!(!staged.exists());
        prop_assert!(!final_path.exists());
        prop_assert_eq!(
            DegradationManager::new(&ledger).decision(),
            OutcomeDecision::Rejected
        );
        let error = outcome.user_error.as_deref().unwrap_or_default();
        let expected_fragment = format!("{} 个不连通分量", expected.len());
        prop_assert!(
            error.contains(&expected_fragment),
            "error must name `{}`",
            expected_fragment
        );
        prop_assert!(error.contains("按连续场景重新分组"));
        prop_assert!(error.contains("补拍增加重叠"));
    } else {
        prop_assert_eq!(
            publication,
            OutputPublication::Published(OutcomeDecision::Success)
        );
        prop_assert!(!staged.exists());
        prop_assert!(final_path.exists());
        prop_assert_eq!(outcome.rejection_reason, None);
        prop_assert_eq!(outcome.user_error, None);
    }
    Ok(())
}

pub(super) fn station_position_key(
    rows: &[u32],
    columns: &[u32],
    station: usize,
) -> (u32, u32, usize) {
    (rows[station], columns[station], station)
}

pub(super) fn tree_edge_key(
    rows: &[u32],
    columns: &[u32],
    left: usize,
    right: usize,
) -> ((u32, u32, usize), (u32, u32, usize)) {
    let left = station_position_key(rows, columns, left);
    let right = station_position_key(rows, columns, right);
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

pub(super) fn exhaustive_maximum_inlier_tree_oracle(
    station_count: usize,
    rows: &[u32],
    columns: &[u32],
    relations: &[(usize, usize, usize, f64)],
) -> Vec<(usize, usize)> {
    let mut precedence = (0..relations.len()).collect::<Vec<_>>();
    precedence.sort_by(|&left_index, &right_index| {
        let left = relations[left_index];
        let right = relations[right_index];
        right.2.cmp(&left.2).then_with(|| {
            tree_edge_key(rows, columns, left.0, left.1)
                .cmp(&tree_edge_key(rows, columns, right.0, right.1))
        })
    });

    let mut best: Option<(usize, Vec<bool>, u64)> = None;
    for subset in 0u64..(1u64 << relations.len()) {
        if subset.count_ones() as usize != station_count.saturating_sub(1) {
            continue;
        }
        let mut reached = vec![false; station_count];
        reached[0] = true;
        loop {
            let mut changed = false;
            for (edge, &(left, right, _, _)) in relations.iter().enumerate() {
                if subset & (1 << edge) == 0 {
                    continue;
                }
                if reached[left] ^ reached[right] {
                    reached[left] = true;
                    reached[right] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if reached.iter().any(|reached| !reached) {
            continue;
        }
        let total = relations
            .iter()
            .enumerate()
            .filter(|(edge, _)| subset & (1 << edge) != 0)
            .map(|(_, relation)| relation.2)
            .sum::<usize>();
        let signature = precedence
            .iter()
            .map(|&edge| subset & (1 << edge) != 0)
            .collect::<Vec<_>>();
        if best.as_ref().is_none_or(|(best_total, best_signature, _)| {
            total > *best_total || (total == *best_total && signature > *best_signature)
        }) {
            best = Some((total, signature, subset));
        }
    }
    let subset = best.expect("a complete graph always has a spanning tree").2;
    precedence
        .into_iter()
        .filter(|edge| subset & (1 << edge) != 0)
        .map(|edge| {
            let (left, right, _, _) = relations[edge];
            (left.min(right), left.max(right))
        })
        .collect()
}

pub(super) fn relation_kind(
    rows: &[u32],
    columns: &[u32],
    left: usize,
    right: usize,
) -> crate::panorama_utils::stack_pipeline::report::AdjacencyKind {
    if columns[left] == columns[right] {
        crate::panorama_utils::stack_pipeline::report::AdjacencyKind::SameColumn
    } else if rows[left] == rows[right] {
        crate::panorama_utils::stack_pipeline::report::AdjacencyKind::SameRow
    } else {
        crate::panorama_utils::stack_pipeline::report::AdjacencyKind::CrossColumn
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 30: every
    // Station_Relation defect maps to its exact stable reason, combined defects
    // remain complete, report buckets aggregate defects, and rejected candidates
    // never cross the pose-write boundary.
    //
    // **Validates: Requirements 6.5, 6.9, 6.10**
    #[test]
    fn property_30_station_relation_defects_map_to_exact_reasons(seed in any::<u64>()) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use crate::panorama_stitching::station_relation_test_access;

        let mut masks = (0u16..=u8::MAX as u16).map(|mask| mask as u8).collect::<Vec<_>>();
        let rotation = seed as usize % masks.len();
        masks.rotate_left(rotation);
        if seed & 1 != 0 {
            masks.reverse();
        }
        let candidates = masks
            .iter()
            .enumerate()
            .map(|(candidate, &mask)| (candidate, candidate + 1_000, relation_measurements_with_defects(mask)))
            .collect::<Vec<_>>();
        let batch = station_relation_test_access::relation_decision_batch(&candidates);

        for &mask in &masks {
            let actual = station_relation_test_access::rejection_reasons(
                relation_measurements_with_defects(mask),
            );
            let expected = station_relation_reason_oracle(mask);
            prop_assert_eq!(actual, expected, "defect mask {:#010b}", mask);
        }
        prop_assert_eq!(batch.report.rejected_count, u8::MAX as usize);
        prop_assert_eq!(batch.pose_writes.len(), 1);
        let accepted_position = masks.iter().position(|&mask| mask == 0).unwrap();
        prop_assert_eq!(batch.pose_writes[0], (accepted_position, accepted_position + 1_000));
        for reason in STATION_RELATION_DEFECT_REASONS {
            prop_assert_eq!(batch.report.rejected_by_reason.get(reason), Some(&128));
            prop_assert!(reason_identifier_pattern().is_match(reason));
        }
        prop_assert_eq!(
            station_relation_test_access::rejection_reasons(relation_measurements_with_defects(u8::MAX)),
            STATION_RELATION_DEFECT_REASONS,
            "the all-defects witness must report every independent failure"
        );
    }

    // Feature: layered-camera-group-focus-stitching, Property 31: accepted
    // projective poses retain all eight solved parameters and bounded
    // Local_Scale; an excessive candidate records its reason and returns the
    // spanning-tree seed bit-for-bit.
    //
    // **Validates: Requirements 6.6**
    #[test]
    fn property_31_station_poses_keep_eight_dof_and_bounded_local_scale(
        seed in any::<u64>(),
        width in 128u32..768,
        height in 128u32..768,
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use crate::panorama_stitching::station_relation_test_access;

        let signed = |shift: u32, magnitude: f64| {
            let value = (((seed.rotate_left(shift) >> 11) & 0xffff) as f64 + 1.0) / 65_536.0;
            let sign = if seed.rotate_left(shift) & 1 == 0 { 1.0 } else { -1.0 };
            sign * magnitude * value
        };
        let angle = signed(3, 0.035);
        let (sin, cos) = angle.sin_cos();
        let scale_x = 1.0 + signed(7, 0.018);
        let scale_y = 1.0 + signed(11, 0.018);
        let shear = signed(17, 0.008);
        let extent = f64::from(width.max(height));
        let candidate = Matrix3::new(
            scale_x * cos,
            -scale_y * sin + shear,
            5.0 + signed(19, 80.0),
            scale_x * sin + shear * 0.5,
            scale_y * cos,
            -7.0 + signed(23, 80.0),
            signed(29, 0.002) / extent,
            signed(31, 0.002) / extent,
            1.0,
        );
        let tree_seed = translation_pose(11.25, -19.5);
        let accepted = station_relation_test_access::select_pose_candidate(
            candidate,
            tree_seed,
            (width, height),
        );
        let minimum = accepted.minimum_local_scale.expect("general pose has measurable Local_Scale");
        let maximum = accepted.maximum_local_scale.expect("general pose has measurable Local_Scale");
        prop_assert!(minimum > 0.0);
        prop_assert!(maximum / minimum <= 1.10);
        prop_assert_eq!(accepted.rejection_reason, None);
        prop_assert_eq!(matrix_bits(&accepted.selected), matrix_bits(&candidate));
        prop_assert_ne!(candidate[(0, 0)], 1.0, "scale remains free");
        prop_assert_ne!(candidate[(0, 1)], 0.0, "rotation/shear remains free");
        prop_assert_ne!(candidate[(1, 0)], 0.0, "rotation/shear remains free");
        prop_assert_ne!(candidate[(2, 0)], 0.0, "x perspective remains free");
        prop_assert_ne!(candidate[(2, 1)], 0.0, "y perspective remains free");

        let excessive = Matrix3::new(
            1.0,
            0.01,
            31.0,
            -0.02,
            1.0,
            -17.0,
            0.45 / f64::from(width),
            0.20 / f64::from(height),
            1.0,
        );
        let rejected = station_relation_test_access::select_pose_candidate(
            excessive,
            tree_seed,
            (width, height),
        );
        let rejected_ratio = rejected.maximum_local_scale.unwrap() / rejected.minimum_local_scale.unwrap();
        prop_assert!(rejected_ratio > 1.10);
        prop_assert_eq!(rejected.rejection_reason, Some(degradation::STATION_POSE_LOCAL_SCALE_EXCEEDED));
        prop_assert_eq!(matrix_bits(&rejected.selected), matrix_bits(&tree_seed));
    }

    // Feature: layered-camera-group-focus-stitching, Property 32: arbitrary
    // disconnected station graphs are rejected at the real publication
    // boundary with canonical components and actionable diagnostics; connected
    // graphs publish and never carry geometry_disconnected.
    //
    // **Validates: Requirements 6.7, 12.4**
    #[test]
    fn property_32_disconnected_station_graphs_reject_publication(
        seed in any::<u64>(),
        station_count in 2usize..7,
        edge_mask in any::<u64>(),
    ) {
        let station_member_paths = (0..station_count)
            .map(|station| {
                let member_count = 1 + ((seed.rotate_left(station as u32) as usize) % 3);
                (0..member_count)
                    .map(|member| format!("/synthetic/property-32/station-{station:02}/capture-{member:02}.raw"))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let all_pairs = (0..station_count)
            .flat_map(|left| ((left + 1)..station_count).map(move |right| (left, right)))
            .collect::<Vec<_>>();
        let random_relations = all_pairs
            .iter()
            .enumerate()
            .filter_map(|(edge, &pair)| (edge_mask & (1 << edge) != 0).then_some(pair))
            .collect::<Vec<_>>();
        assert_graph_outcome_and_publication(&station_member_paths, &random_relations)?;

        let connected_chain = (0..station_count - 1)
            .map(|station| (station, station + 1))
            .collect::<Vec<_>>();
        assert_graph_outcome_and_publication(&station_member_paths, &connected_chain)?;

        let split = 1 + seed as usize % (station_count - 1);
        let disconnected = all_pairs
            .into_iter()
            .filter(|&(left, right)| (left < split) == (right < split))
            .collect::<Vec<_>>();
        assert_graph_outcome_and_publication(&station_member_paths, &disconnected)?;
    }

    // Feature: layered-camera-group-focus-stitching, Property 33: production
    // tree selection equals an exhaustive maximum-inlier oracle; equal weights
    // use row/column order only and are independent of relation order and score.
    //
    // **Validates: Requirements 7.1**
    #[test]
    fn property_33_spanning_tree_is_maximum_support_with_stable_ties(
        seed in any::<u64>(),
        station_count in 2usize..6,
    ) {
        use crate::panorama_stitching::station_relation_test_access;

        let mut station_at_position = (0..station_count).collect::<Vec<_>>();
        let mut shuffle_state = seed ^ 0x33a5_9d91;
        for index in (1..station_at_position.len()).rev() {
            let other = splitmix64(&mut shuffle_state) as usize % (index + 1);
            station_at_position.swap(index, other);
        }
        let mut rows = vec![0; station_count];
        let mut columns = vec![0; station_count];
        for (position, &station) in station_at_position.iter().enumerate() {
            rows[station] = (position / 3) as u32;
            columns[station] = (position % 3) as u32;
        }
        let mut relations = (0..station_count)
            .flat_map(|left| ((left + 1)..station_count).map(move |right| (left, right)))
            .map(|(left, right)| {
                let inliers = 24 + (splitmix64(&mut shuffle_state) as usize % 5);
                let score = unit_from(&mut shuffle_state) * 1_000_000.0;
                if splitmix64(&mut shuffle_state) & 1 == 0 {
                    (left, right, inliers, score)
                } else {
                    (right, left, inliers, score)
                }
            })
            .collect::<Vec<_>>();
        let expected = exhaustive_maximum_inlier_tree_oracle(
            station_count,
            &rows,
            &columns,
            &relations,
        );
        let actual = station_relation_test_access::maximum_inlier_tree(
            station_count,
            &rows,
            &columns,
            &relations,
        );
        prop_assert_eq!(&actual, &expected);
        let actual_weight = actual
            .iter()
            .map(|&(left, right)| {
                relations.iter().find(|relation| {
                    (relation.0.min(relation.1), relation.0.max(relation.1)) == (left, right)
                }).unwrap().2
            })
            .sum::<usize>();
        let expected_weight = expected
            .iter()
            .map(|&(left, right)| relations.iter().find(|relation| {
                (relation.0.min(relation.1), relation.0.max(relation.1)) == (left, right)
            }).unwrap().2)
            .sum::<usize>();
        prop_assert_eq!(actual_weight, expected_weight);

        relations.reverse();
        for relation in &mut relations {
            relation.3 = 1_000_000_000.0 - relation.3;
        }
        let reordered = station_relation_test_access::maximum_inlier_tree(
            station_count,
            &rows,
            &columns,
            &relations,
        );
        prop_assert_eq!(reordered, expected);
    }

    // Feature: layered-camera-group-focus-stitching, Property 34: every
    // accepted horizontal, vertical, and cross-column relation reaches the
    // production closure report and final weight records, including an edge
    // beyond the removed large-residual pre-drop threshold.
    //
    // **Validates: Requirements 7.2**
    #[test]
    fn property_34_all_accepted_relations_participate_in_closure(seed in any::<u64>()) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        use crate::panorama_stitching::station_relation_test_access::{self, ClosureRelationView};

        let rows = vec![0, 0, 1, 1];
        let columns = vec![0, 1, 0, 1];
        let positions = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)];
        let base_poses = positions
            .iter()
            .map(|&(x, y)| translation_pose(x, y))
            .collect::<Vec<_>>();
        let pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)];
        let large_pair = (0, 3);
        let mut state = seed ^ 0x34c1_05e5;
        let relations = pairs
            .iter()
            .map(|&(left, right)| {
                let ideal_x = positions[left].0 - positions[right].0;
                let ideal_y = positions[left].1 - positions[right].1;
                let (residual_x, residual_y) = if (left, right) == large_pair {
                    (120.0 + 20.0 * unit_from(&mut state), 10.0)
                } else {
                    (unit_from(&mut state) - 0.5, unit_from(&mut state) - 0.5)
                };
                ClosureRelationView {
                    score: 0.5 + unit_from(&mut state),
                    left,
                    right,
                    left_to_right: translation_pose(ideal_x + residual_x, ideal_y + residual_y),
                    independent_support: 2,
                    median_error_px: 3.0 * unit_from(&mut state),
                }
            })
            .collect::<Vec<_>>();
        let closure = station_relation_test_access::closure_run(
            (400, 400),
            &rows,
            &columns,
            &base_poses,
            &relations,
        );
        let expected_pairs = pairs
            .iter()
            .map(|&(left, right)| (left.min(right), left.max(right)))
            .collect::<BTreeSet<_>>();
        let recorded_pairs = closure
            .report
            .constraint_weights
            .iter()
            .map(|record| (record.left.min(record.right), record.left.max(record.right)))
            .collect::<BTreeSet<_>>();
        prop_assert_eq!(closure.report.participating_constraints, relations.len());
        prop_assert_eq!(closure.report.constraint_weights.len(), relations.len());
        prop_assert_eq!(recorded_pairs, expected_pairs);
        prop_assert_eq!(closure.poses.len(), base_poses.len());

        let kinds = pairs
            .iter()
            .map(|&(left, right)| relation_kind(&rows, &columns, left, right))
            .collect::<Vec<_>>();
        prop_assert!(kinds.contains(&crate::panorama_utils::stack_pipeline::report::AdjacencyKind::SameRow));
        prop_assert!(kinds.contains(&crate::panorama_utils::stack_pipeline::report::AdjacencyKind::SameColumn));
        prop_assert!(kinds.contains(&crate::panorama_utils::stack_pipeline::report::AdjacencyKind::CrossColumn));
        let large = closure
            .report
            .constraint_weights
            .iter()
            .find(|record| (record.left.min(record.right), record.left.max(record.right)) == large_pair)
            .expect("large accepted relation must have a final weight record");
        prop_assert!(large.residual_px > 80.0, "large residual witness must exceed the removed pre-drop threshold");
        prop_assert!(large.weight.is_finite());
    }
}
