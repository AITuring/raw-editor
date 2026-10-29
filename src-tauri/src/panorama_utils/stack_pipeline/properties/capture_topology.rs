//! Capture_Topology_Model and Station_Relation properties (25-29).
//!
//! Injective row/column indices decided by measurements only, candidate classes
//! within the search budget, stable ambiguous station pairs, station evidence
//! from covered pixels only, and the all-gates relation acceptance.

use super::super::test_support;
use super::super::test_support::RenameScheme;
use super::super::test_support::SyntheticScan;
use crate::panorama_utils::processing;
use crate::panorama_utils::shared::Feature;
use crate::panorama_utils::shared::MatchInfo;
use crate::panorama_utils::stitching;
use nalgebra::Matrix3;
use nalgebra::Point2;
use proptest::prelude::*;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use super::{map_point, probe_texture, single_frame_info, splitmix64};

// ---------------------------------------------------------------------------
// Properties 25–29: production topology and Station_Relation decisions
// ---------------------------------------------------------------------------

pub(super) fn scan_topology_observations(
    scan: &SyntheticScan,
) -> Vec<crate::panorama_utils::stack_pipeline::topology::StationObservation> {
    let side = f64::from(test_support::SYNTHETIC_TILE_SIDE);
    let mut by_station = BTreeMap::<usize, Vec<(usize, Point2<f64>, f64, String)>>::new();
    for (source_index, source) in scan.sources().iter().enumerate() {
        let center = scan.plane_centre(source_index);
        let left = map_point(&source.tile_to_plane, Point2::new(0.0, side * 0.5))
            .expect("synthetic station geometry is finite");
        let right = map_point(&source.tile_to_plane, Point2::new(side, side * 0.5))
            .expect("synthetic station geometry is finite");
        by_station.entry(source.station).or_default().push((
            source.layer,
            center,
            (right - left).norm(),
            source.filename.clone(),
        ));
    }
    by_station
        .into_iter()
        .map(|(station_index, mut measurements)| {
            // Import order must not change floating-point reduction order.
            measurements.sort_by_key(|measurement| measurement.0);
            let count = measurements.len() as f64;
            let world_x = measurements.iter().map(|entry| entry.1.x).sum::<f64>() / count;
            let world_y = measurements.iter().map(|entry| entry.1.y).sum::<f64>() / count;
            let effective_width = measurements.iter().map(|entry| entry.2).sum::<f64>() / count;
            let filename = measurements
                .iter()
                .map(|entry| entry.3.as_str())
                .min()
                .unwrap_or_default()
                .to_string();
            crate::panorama_utils::stack_pipeline::topology::StationObservation {
                station_index,
                world_x: Some(world_x),
                world_y: Some(world_y),
                effective_width: Some(effective_width),
                filename,
            }
        })
        .collect()
}

pub(super) fn topology_score(seed: u64, left: usize, right: usize) -> f64 {
    let mut state = seed
        ^ (left as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (right as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let value = splitmix64(&mut state) >> 11;
    value as f64 / ((1u64 << 53) as f64)
}

pub(super) fn topology_scores(station_count: usize, seed: u64) -> BTreeMap<(usize, usize), f64> {
    let mut scores = BTreeMap::new();
    for left in 0..station_count {
        for right in (left + 1)..station_count {
            scores.insert((left, right), topology_score(seed, left, right));
        }
    }
    scores
}

pub(super) fn regular_topology_grid(
    rows: usize,
    columns: usize,
) -> Vec<crate::panorama_utils::stack_pipeline::topology::StationObservation> {
    (0..rows)
        .flat_map(|row| {
            (0..columns).map(move |column| {
                let station_index = row * columns + column;
                crate::panorama_utils::stack_pipeline::topology::StationObservation {
                    station_index,
                    world_x: Some((columns - column) as f64 * 70.0),
                    world_y: Some(row as f64 * 70.0),
                    effective_width: Some(100.0),
                    filename: format!("scan-{station_index:03}.raw"),
                }
            })
        })
        .collect()
}

pub(super) fn relation_gate_oracle(
    measurements: crate::panorama_stitching::station_relation_test_access::MeasurementsView,
) -> bool {
    measurements.inliers >= 24
        && measurements.median_error_px.is_finite()
        && measurements.median_error_px <= 3.0
        && measurements.scale_ratio.is_finite()
        && (0.95..=1.05).contains(&measurements.scale_ratio)
        && measurements.spatial_support.is_finite()
        && measurements.spatial_support >= 0.20
        && measurements
            .low_frequency_mean_relative_difference
            .is_finite()
        && measurements.low_frequency_mean_relative_difference <= 0.20
        && measurements.edge_strength_ratio.is_finite()
        && (0.70..=1.40).contains(&measurements.edge_strength_ratio)
        && measurements
            .median_edge_orientation_difference_degrees
            .is_finite()
        && measurements.median_edge_orientation_difference_degrees <= 10.0
        && measurements.preserves_convex_orientation
}

pub(super) fn below(value: f64) -> f64 {
    f64::from_bits(value.to_bits() - 1)
}

pub(super) fn above(value: f64) -> f64 {
    f64::from_bits(value.to_bits() + 1)
}

pub(super) fn accepted_relation_measurements()
-> crate::panorama_stitching::station_relation_test_access::MeasurementsView {
    crate::panorama_stitching::station_relation_test_access::MeasurementsView {
        inliers: 24,
        median_error_px: 3.0,
        scale_ratio: 1.0,
        spatial_support: 0.20,
        low_frequency_mean_relative_difference: 0.20,
        edge_strength_ratio: 1.0,
        median_edge_orientation_difference_degrees: 10.0,
        preserves_convex_orientation: true,
    }
}

pub(super) fn relation_boundary_cases()
-> Vec<crate::panorama_stitching::station_relation_test_access::MeasurementsView> {
    use crate::panorama_stitching::station_relation_test_access::MeasurementsView;

    let accepted = accepted_relation_measurements();
    let mut cases = vec![
        accepted,
        MeasurementsView {
            inliers: 23,
            ..accepted
        },
        MeasurementsView {
            median_error_px: below(3.0),
            ..accepted
        },
        MeasurementsView {
            median_error_px: above(3.0),
            ..accepted
        },
        MeasurementsView {
            scale_ratio: below(0.95),
            ..accepted
        },
        MeasurementsView {
            scale_ratio: 0.95,
            ..accepted
        },
        MeasurementsView {
            scale_ratio: above(0.95),
            ..accepted
        },
        MeasurementsView {
            scale_ratio: below(1.05),
            ..accepted
        },
        MeasurementsView {
            scale_ratio: 1.05,
            ..accepted
        },
        MeasurementsView {
            scale_ratio: above(1.05),
            ..accepted
        },
        MeasurementsView {
            spatial_support: below(0.20),
            ..accepted
        },
        MeasurementsView {
            spatial_support: above(0.20),
            ..accepted
        },
        MeasurementsView {
            low_frequency_mean_relative_difference: below(0.20),
            ..accepted
        },
        MeasurementsView {
            low_frequency_mean_relative_difference: above(0.20),
            ..accepted
        },
        MeasurementsView {
            edge_strength_ratio: below(0.70),
            ..accepted
        },
        MeasurementsView {
            edge_strength_ratio: 0.70,
            ..accepted
        },
        MeasurementsView {
            edge_strength_ratio: above(0.70),
            ..accepted
        },
        MeasurementsView {
            edge_strength_ratio: below(1.40),
            ..accepted
        },
        MeasurementsView {
            edge_strength_ratio: 1.40,
            ..accepted
        },
        MeasurementsView {
            edge_strength_ratio: above(1.40),
            ..accepted
        },
        MeasurementsView {
            median_edge_orientation_difference_degrees: below(10.0),
            ..accepted
        },
        MeasurementsView {
            median_edge_orientation_difference_degrees: above(10.0),
            ..accepted
        },
        MeasurementsView {
            preserves_convex_orientation: false,
            ..accepted
        },
    ];
    for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        cases.extend([
            MeasurementsView {
                median_error_px: nonfinite,
                ..accepted
            },
            MeasurementsView {
                scale_ratio: nonfinite,
                ..accepted
            },
            MeasurementsView {
                spatial_support: nonfinite,
                ..accepted
            },
            MeasurementsView {
                low_frequency_mean_relative_difference: nonfinite,
                ..accepted
            },
            MeasurementsView {
                edge_strength_ratio: nonfinite,
                ..accepted
            },
            MeasurementsView {
                median_edge_orientation_difference_degrees: nonfinite,
                ..accepted
            },
        ]);
    }
    cases
}

pub(super) const PROPERTY_28_CASES: usize = 100;

pub(super) const PROPERTY_28_RELATION_STRIDE: usize = 10;

pub(super) static PROPERTY_28_CURSOR: AtomicUsize = AtomicUsize::new(0);

pub(super) static PROPERTY_28_RELATION_RUNS: AtomicUsize = AtomicUsize::new(0);

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 25: non-ambiguous
    // row/column indices are injective and depend only on measured centre
    // displacements and effective widths.
    //
    // **Validates: Requirements 5.1, 5.2**
    #[test]
    fn property_25_topology_indices_are_injective_and_measurement_only(
        scan in test_support::arb_artwork_plane().prop_flat_map(test_support::arb_scan_grid),
        shift_x in -10_000.0f64..10_000.0,
        shift_y in -10_000.0f64..10_000.0,
        scale in 0.25f64..4.0,
        rename_slot in 0usize..RenameScheme::ALL.len(),
    ) {
        let observations = scan_topology_observations(&scan);
        let scores = topology_scores(observations.len(), 25);
        let topology = crate::panorama_utils::stack_pipeline::topology::infer_station_topology(&observations, &scores);

        prop_assert_eq!(topology.row.len(), observations.len());
        prop_assert_eq!(topology.column.len(), observations.len());
        let indices = topology
            .row
            .iter()
            .copied()
            .zip(topology.column.iter().copied())
            .collect::<BTreeSet<_>>();
        prop_assert_eq!(indices.len(), observations.len(), "(row, column) must be injective");

        let ambiguous = topology.ambiguous.iter().copied().collect::<BTreeSet<_>>();
        for left in 0..observations.len() {
            for right in (left + 1)..observations.len() {
                if ambiguous.contains(&left) || ambiguous.contains(&right) {
                    continue;
                }
                let left_x = observations[left].world_x.unwrap();
                let right_x = observations[right].world_x.unwrap();
                let width = observations[left]
                    .effective_width
                    .unwrap()
                    .min(observations[right].effective_width.unwrap());
                let oracle_different_column = (left_x - right_x).abs() > 0.5 * width;
                prop_assert_eq!(
                    topology.column[left] != topology.column[right],
                    oracle_different_column,
                    "non-ambiguous column decision must use the literal 0.5 effective-width boundary"
                );
            }
        }

        let mut transformed = observations.clone();
        for observation in &mut transformed {
            observation.world_x = observation.world_x.map(|value| value * scale + shift_x);
            observation.world_y = observation.world_y.map(|value| value * scale + shift_y);
            observation.effective_width = observation.effective_width.map(|value| value * scale);
            observation.filename = format!("nuisance-{:03}.raw", observations.len() - observation.station_index);
        }
        transformed.reverse();
        let transformed_topology = crate::panorama_utils::stack_pipeline::topology::infer_station_topology(&transformed, &scores);
        prop_assert_eq!(&transformed_topology, &topology, "translation, common scale, names, and import order are nuisance dimensions");

        let renamed_and_permuted = scan
            .with_renamed_sources(RenameScheme::ALL[rename_slot])
            .with_permuted_import_order_seeded(0x2519 ^ rename_slot as u64);
        let changed_observations = scan_topology_observations(&renamed_and_permuted);
        let changed_topology = crate::panorama_utils::stack_pipeline::topology::infer_station_topology(&changed_observations, &scores);
        prop_assert_eq!(changed_topology, topology, "renaming and shuffled import must preserve topology and candidates");
    }

    // Feature: layered-camera-group-focus-stitching, Property 26: candidate
    // classes/counts, exhaustive <=64 behavior, and the >64 shared 8*N budget.
    //
    // **Validates: Requirements 5.3, 5.5, 5.6**
    #[test]
    fn property_26_candidate_classes_and_budget_match_the_requirement(seed in any::<u64>()) {
        let small_count = 2 + seed as usize % 63;
        let small = regular_topology_grid(1, small_count);
        let small_scores = topology_scores(small_count, seed);
        let small_topology = crate::panorama_utils::stack_pipeline::topology::infer_station_topology(&small, &small_scores);
        prop_assert_eq!(small_topology.candidates.len(), small_count * (small_count - 1) / 2);
        prop_assert_eq!(small_topology.truncated_count, 0);
        prop_assert_eq!(small_topology.truncation_min_score, None);
        let small_pairs = small_topology
            .candidates
            .iter()
            .map(|candidate| (candidate.left, candidate.right))
            .collect::<BTreeSet<_>>();
        let all_small_pairs = (0..small_count)
            .flat_map(|left| ((left + 1)..small_count).map(move |right| (left, right)))
            .collect::<BTreeSet<_>>();
        prop_assert_eq!(small_pairs, all_small_pairs);

        let rows = 9;
        let columns = 8;
        let large_count = rows * columns;
        let large = regular_topology_grid(rows, columns);
        let large_scores = topology_scores(large_count, seed ^ 0x26);
        let large_topology = crate::panorama_utils::stack_pipeline::topology::infer_station_topology(&large, &large_scores);
        let mut expected = BTreeMap::new();
        for left in 0..large_count {
            for right in (left + 1)..large_count {
                let left_row = left / columns;
                let left_column = left % columns;
                let right_row = right / columns;
                let right_column = right % columns;
                let row_distance = left_row.abs_diff(right_row);
                let column_distance = left_column.abs_diff(right_column);
                let kind = if column_distance == 0 && row_distance == 1 {
                    Some(crate::panorama_utils::stack_pipeline::report::AdjacencyKind::SameColumn)
                } else if row_distance == 0 && column_distance == 1 {
                    Some(crate::panorama_utils::stack_pipeline::report::AdjacencyKind::SameRow)
                } else if row_distance == 1 && column_distance == 1 {
                    Some(crate::panorama_utils::stack_pipeline::report::AdjacencyKind::CrossColumn)
                } else {
                    None
                };
                if let Some(kind) = kind {
                    expected.insert((left, right), kind);
                }
            }
        }
        prop_assert_eq!(large_topology.candidates.len(), expected.len());
        for candidate in &large_topology.candidates {
            prop_assert_eq!(expected.get(&(candidate.left, candidate.right)), Some(&candidate.kind));
            prop_assert_eq!(candidate.score.to_bits(), large_scores[&(candidate.left, candidate.right)].to_bits());
        }
        for station in 0..large_count {
            let incident = large_topology.candidates.iter().filter(|candidate| candidate.left == station || candidate.right == station);
            let same_column = incident.clone().filter(|candidate| candidate.kind == crate::panorama_utils::stack_pipeline::report::AdjacencyKind::SameColumn).count();
            let same_row = incident.clone().filter(|candidate| candidate.kind == crate::panorama_utils::stack_pipeline::report::AdjacencyKind::SameRow).count();
            let cross_column = incident.filter(|candidate| candidate.kind == crate::panorama_utils::stack_pipeline::report::AdjacencyKind::CrossColumn).count();
            prop_assert!(same_column <= 2);
            prop_assert!(same_row <= 2);
            prop_assert!(cross_column <= 4);
        }

        let ambiguous_count = 65 + seed as usize % 8;
        let ambiguous = (0..ambiguous_count)
            .map(|station_index| crate::panorama_utils::stack_pipeline::topology::StationObservation {
                station_index,
                world_x: None,
                world_y: None,
                effective_width: None,
                filename: format!("ambiguous-{station_index:03}.raw"),
            })
            .collect::<Vec<_>>();
        let ambiguous_scores = topology_scores(ambiguous_count, seed ^ 0xa26);
        let ambiguous_topology = crate::panorama_utils::stack_pipeline::topology::infer_station_topology(&ambiguous, &ambiguous_scores);
        let budget = 8 * ambiguous_count;
        let total_pairs = ambiguous_count * (ambiguous_count - 1) / 2;
        let mut oracle = ambiguous_scores
            .iter()
            .map(|(&(left, right), &score)| (left, right, score))
            .collect::<Vec<_>>();
        oracle.sort_by(|left, right| right.2.total_cmp(&left.2).then_with(|| left.0.cmp(&right.0)).then_with(|| left.1.cmp(&right.1)));
        oracle.truncate(budget);
        prop_assert_eq!(ambiguous_topology.candidates.len(), budget);
        for (actual, expected) in ambiguous_topology.candidates.iter().zip(&oracle) {
            prop_assert_eq!((actual.left, actual.right), (expected.0, expected.1));
            prop_assert_eq!(actual.kind, crate::panorama_utils::stack_pipeline::report::AdjacencyKind::Ambiguous);
            prop_assert_eq!(actual.score.to_bits(), expected.2.to_bits());
        }
        prop_assert_eq!(ambiguous_topology.truncated_count, total_pairs - budget);
        prop_assert_eq!(
            ambiguous_topology.truncation_min_score.map(f64::to_bits),
            oracle.last().map(|candidate| candidate.2.to_bits())
        );
        let report = crate::panorama_utils::stack_pipeline::report::TopologyReport::from(&ambiguous_topology);
        prop_assert_eq!(report.truncated_count, total_pairs - budget);
        prop_assert_eq!(report.truncation_min_score.map(f64::to_bits), oracle.last().map(|candidate| candidate.2.to_bits()));
    }

    // Feature: layered-camera-group-focus-stitching, Property 27: every
    // displacement-ambiguous station enters all pairs before the shared budget.
    //
    // **Validates: Requirements 5.7**
    #[test]
    fn property_27_ambiguous_station_pairs_and_report_are_stable(seed in any::<u64>()) {
        let station_count = 65;
        let ambiguous_station = seed as usize % station_count;
        let mut observations = regular_topology_grid(5, 13);
        observations[ambiguous_station].world_x = None;
        observations[ambiguous_station].world_y = None;
        observations[ambiguous_station].effective_width = None;
        let scores = topology_scores(station_count, seed ^ 0x27);
        let topology = crate::panorama_utils::stack_pipeline::topology::infer_station_topology(&observations, &scores);
        prop_assert_eq!(topology.ambiguous.as_slice(), [ambiguous_station]);
        prop_assert!(topology.candidates.len() <= 8 * station_count);
        prop_assert_eq!(topology.truncated_count, 0, "one forced all-station fan-out must enter before, and fit within, the shared budget");
        for other in 0..station_count {
            if other == ambiguous_station {
                continue;
            }
            let pair = (ambiguous_station.min(other), ambiguous_station.max(other));
            let candidate = topology.candidates.iter().find(|candidate| (candidate.left, candidate.right) == pair);
            prop_assert!(candidate.is_some(), "ambiguous station must pair with station {other}");
            prop_assert_eq!(candidate.unwrap().kind, crate::panorama_utils::stack_pipeline::report::AdjacencyKind::Ambiguous);
        }

        let report = crate::panorama_utils::stack_pipeline::report::TopologyReport::from(&topology);
        prop_assert_eq!(report.ambiguous_stations.as_slice(), [ambiguous_station]);
        prop_assert_eq!(report.candidates.len(), topology.candidates.len());

        let mut renamed_and_shuffled = observations.clone();
        for observation in &mut renamed_and_shuffled {
            observation.filename = format!("renamed-{:03}.raw", station_count - observation.station_index);
        }
        renamed_and_shuffled.reverse();
        let repeated = crate::panorama_utils::stack_pipeline::topology::infer_station_topology(&renamed_and_shuffled, &scores);
        let repeated_report = crate::panorama_utils::stack_pipeline::report::TopologyReport::from(&repeated);
        prop_assert_eq!(repeated, topology);
        prop_assert_eq!(repeated_report, report);
    }

    // Feature: layered-camera-group-focus-stitching, Property 28: station
    // evidence comes only from fully covered Virtual_Tile neighborhoods and
    // single-source evidence is supplemental iff independent support >= 2.
    //
    // **Validates: Requirements 6.1, 6.2**
    #[test]
    fn property_28_station_evidence_uses_only_covered_consensus(
        seed in any::<u64>(),
        left_station in 0usize..32,
        right_delta in 1usize..32,
    ) {
        let case = PROPERTY_28_CURSOR.fetch_add(1, Ordering::Relaxed);
        let width = 96u32;
        let height = 96u32;
        let covered = (0..height)
            .flat_map(|y| (0..width).map(move |x| u8::from((8..88).contains(&x) && (8..88).contains(&y)) * u8::MAX))
            .collect::<Vec<_>>();
        let coverage = stitching::CoverageMask::from_bytes(width, height, covered).expect("coverage dimensions match");
        let covered_value = |x: u32, y: u32| {
            probe_texture(f64::from(x), f64::from(y), seed, 1.0)[0]
        };
        let original = image::Rgb32FImage::from_fn(width, height, |x, y| {
            let value = if coverage.is_covered(x, y) { covered_value(x, y) } else { 0.0 };
            image::Rgb([value, value, value])
        });
        let mutated = image::Rgb32FImage::from_fn(width, height, |x, y| {
            let value = if coverage.is_covered(x, y) {
                covered_value(x, y)
            } else {
                let mut state = seed.rotate_left(17) ^ (u64::from(y) << 32) ^ u64::from(x);
                0.2 + 0.8 * ((splitmix64(&mut state) >> 40) as f32 / ((1u32 << 24) as f32))
            };
            image::Rgb([value, value, value])
        });
        let brief_pairs = processing::generate_brief_pairs();
        let (original_gray, original_features) = crate::panorama_stitching::station_relation_test_access::covered_features(&original, &coverage, &brief_pairs);
        let (mutated_gray, mutated_features) = crate::panorama_stitching::station_relation_test_access::covered_features(&mutated, &coverage, &brief_pairs);
        let signature = |features: &[Feature]| features.iter().map(|feature| (feature.keypoint.x, feature.keypoint.y, feature.descriptor)).collect::<Vec<_>>();
        prop_assert_eq!(&original_gray, &mutated_gray);
        prop_assert!(!original_features.is_empty(), "covered procedural texture must exercise feature evidence");
        prop_assert_eq!(signature(&original_features), signature(&mutated_features));
        for feature in &original_features {
            let x = feature.keypoint.x;
            let y = feature.keypoint.y;
            prop_assert!(x >= 20 && y >= 20 && x + 20 < width && y + 20 < height);
            prop_assert!((y - 20..=y + 20).all(|sample_y| (x - 20..=x + 20).all(|sample_x| coverage.is_covered(sample_x, sample_y))));
        }

        for independent_support in 0usize..=3 {
            let discard = crate::panorama_stitching::station_relation_test_access::single_source_discard(
                independent_support,
                left_station,
                left_station + right_delta,
            );
            prop_assert_eq!(discard.is_none(), independent_support >= 2);
            if let Some(record) = discard {
                prop_assert_eq!(record["count"].as_u64(), Some(1));
                prop_assert_eq!(&record["stations"], &serde_json::json!([left_station, left_station + right_delta]));
                prop_assert_eq!(record["independent_support"].as_u64(), Some(independent_support as u64));
            }
        }

        if case % PROPERTY_28_RELATION_STRIDE == 0 {
            PROPERTY_28_RELATION_RUNS.fetch_add(1, Ordering::Relaxed);
            let original_relation_features = original_features
                .iter()
                .filter(|candidate| {
                    original_features
                        .iter()
                        .filter(|feature| feature.descriptor == candidate.descriptor)
                        .count()
                        == 1
                })
                .cloned()
                .collect::<Vec<_>>();
            let mutated_relation_features = mutated_features
                .iter()
                .filter(|candidate| {
                    mutated_features
                        .iter()
                        .filter(|feature| feature.descriptor == candidate.descriptor)
                        .count()
                        == 1
                })
                .cloned()
                .collect::<Vec<_>>();
            let mut original_left = single_frame_info(0, width);
            original_left.alignment_image = original_gray.clone();
            original_left.features = original_relation_features.clone();
            let mut original_right = single_frame_info(1, width);
            original_right.alignment_image = original_gray;
            original_right.features = original_relation_features;
            let mut mutated_left = single_frame_info(0, width);
            mutated_left.alignment_image = mutated_gray.clone();
            mutated_left.features = mutated_relation_features.clone();
            let mut mutated_right = single_frame_info(1, width);
            mutated_right.alignment_image = mutated_gray;
            mutated_right.features = mutated_relation_features;
            let topology = crate::panorama_utils::stack_pipeline::topology::StationTopology {
                row: vec![0, 0],
                column: vec![0, 1],
                candidates: vec![crate::panorama_utils::stack_pipeline::topology::CandidateAdjacency {
                    left: 0,
                    right: 1,
                    kind: crate::panorama_utils::stack_pipeline::report::AdjacencyKind::SameRow,
                    score: 1.0,
                }],
                ..crate::panorama_utils::stack_pipeline::topology::StationTopology::default()
            };
            let original_initial = HashMap::from([
                (original_left.id, Matrix3::identity()),
                (original_right.id, Matrix3::identity()),
            ]);
            let mutated_initial = HashMap::from([
                (mutated_left.id, Matrix3::identity()),
                (mutated_right.id, Matrix3::identity()),
            ]);
            let masks = vec![coverage.clone(), coverage.clone()];
            let original_relations = crate::panorama_stitching::station_relation_test_access::guided_station_matches(
                &[original_left, original_right],
                &masks,
                &original_initial,
                &topology,
            );
            let mutated_relations = crate::panorama_stitching::station_relation_test_access::guided_station_matches(
                &[mutated_left, mutated_right],
                &masks,
                &mutated_initial,
                &topology,
            );
            let relation_signature = |relations: &HashMap<(usize, usize), MatchInfo>| {
                let mut signatures = relations.iter().map(|(&(left, right), relation)| {
                    let mut transform = [0u64; 9];
                    for (slot, value) in transform.iter_mut().zip(relation.homography.iter()) {
                        *slot = value.to_bits();
                    }
                    let points = relation.points.iter().map(|(source, target)| ([source.x.to_bits(), source.y.to_bits()], [target.x.to_bits(), target.y.to_bits()])).collect::<Vec<_>>();
                    ((left, right), relation.inliers, transform, points)
                }).collect::<Vec<_>>();
                signatures.sort_by_key(|signature| signature.0);
                signatures
            };
            prop_assert_eq!(relation_signature(&original_relations), relation_signature(&mutated_relations), "uncovered payload mutations cannot change Station_Relation inputs or results");
        }
        let relation_runs = PROPERTY_28_RELATION_RUNS.load(Ordering::Relaxed);
        prop_assert!(case + 1 < PROPERTY_28_CASES || relation_runs >= PROPERTY_28_CASES / PROPERTY_28_RELATION_STRIDE);
    }

    // Feature: layered-camera-group-focus-stitching, Property 29: production
    // Station_Relation acceptance is exactly the conjunction of all eight gates.
    //
    // **Validates: Requirements 6.3, 6.4, 6.8**
    #[test]
    fn property_29_station_relation_acceptance_is_all_gates(
        inliers in 0usize..48,
        median_error_px in prop_oneof![0.0f64..6.0, Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
        scale_ratio in prop_oneof![0.6f64..1.4, Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
        spatial_support in prop_oneof![0.0f64..1.0, Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
        photometric_difference in prop_oneof![0.0f64..0.5, Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
        edge_strength_ratio in prop_oneof![0.4f64..1.8, Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
        orientation_difference in prop_oneof![0.0f64..20.0, Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
        convex in any::<bool>(),
    ) {
        use crate::panorama_stitching::station_relation_test_access::{self, MeasurementsView};
        let random = MeasurementsView {
            inliers,
            median_error_px,
            scale_ratio,
            spatial_support,
            low_frequency_mean_relative_difference: photometric_difference,
            edge_strength_ratio,
            median_edge_orientation_difference_degrees: orientation_difference,
            preserves_convex_orientation: convex,
        };
        prop_assert_eq!(station_relation_test_access::rejection_reasons(random).is_empty(), relation_gate_oracle(random));
        for boundary in relation_boundary_cases() {
            prop_assert_eq!(
                station_relation_test_access::rejection_reasons(boundary).is_empty(),
                relation_gate_oracle(boundary),
                "exact boundary/non-finite Station_Relation decision mismatch for {:?}",
                boundary
            );
        }
    }
}
