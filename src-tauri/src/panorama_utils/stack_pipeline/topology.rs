//! Capture_Topology_Model row/column inference and candidate adjacency
//! generation (requirements 5.1 through 5.9).
//!
//! This stage deliberately owns no pose or transform. Callers reduce their
//! image geometry to station centre displacements, effective widths, and
//! transform-derived overlap scores before entering this module.

use super::report::{AdjacencyKind, CandidateAdjacencyRecord, TopologyReport};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// Horizontal centre displacement above this fraction of either station's
/// effective width separates the stations into different columns.
pub(crate) const COLUMN_SEPARATION_WIDTH_RATIO: f64 = 0.5;

/// Small station sets are exhaustive at the station level (requirement 5.6).
pub(crate) const EXHAUSTIVE_STATION_LIMIT: usize = 64;

/// Large station sets retain at most eight candidates per station overall.
pub(crate) const CANDIDATE_BUDGET_PER_STATION: usize = 8;

/// Pose-free measurement supplied to the Capture_Topology_Model.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StationObservation {
    pub station_index: usize,
    pub world_x: Option<f64>,
    pub world_y: Option<f64>,
    pub effective_width: Option<f64>,
    /// Used only when both estimated centre coordinates compare exactly equal.
    pub filename: String,
}

/// A scored station pair that a later matcher may attempt.
///
/// No transform belongs here: this type controls only matching order and
/// matching quantity (requirement 5.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CandidateAdjacency {
    pub left: usize,
    pub right: usize,
    pub kind: AdjacencyKind,
    pub score: f64,
}

/// Pose-free output of row/column inference.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct StationTopology {
    pub row: Vec<u32>,
    pub column: Vec<u32>,
    pub ambiguous: Vec<usize>,
    pub candidates: Vec<CandidateAdjacency>,
    pub truncated_count: usize,
    pub truncation_min_score: Option<f64>,
}

impl StationTopology {
    /// Whether a measured station relation survived candidate generation and
    /// the shared topology budget.
    pub(crate) fn pair_is_eligible(&self, left: usize, right: usize) -> bool {
        let (left, right) = ordered_pair(left, right);
        self.candidates
            .iter()
            .any(|candidate| candidate.left == left && candidate.right == right)
    }
}

impl From<&StationTopology> for TopologyReport {
    fn from(topology: &StationTopology) -> Self {
        Self {
            candidates: topology
                .candidates
                .iter()
                .map(|candidate| CandidateAdjacencyRecord {
                    left: candidate.left,
                    right: candidate.right,
                    kind: candidate.kind,
                    score: candidate.score,
                })
                .collect(),
            truncated_count: topology.truncated_count,
            truncation_min_score: topology.truncation_min_score,
            ambiguous_stations: topology.ambiguous.clone(),
        }
    }
}

#[derive(Debug)]
struct ColumnCluster {
    members: Vec<usize>,
}

fn ordered_pair(left: usize, right: usize) -> (usize, usize) {
    (left.min(right), left.max(right))
}

fn valid_measurement(observation: &StationObservation) -> Option<(f64, f64, f64)> {
    let (Some(x), Some(y), Some(width)) = (
        observation.world_x,
        observation.world_y,
        observation.effective_width,
    ) else {
        return None;
    };
    (x.is_finite() && y.is_finite() && width.is_finite() && width > 0.0).then_some((x, y, width))
}

fn same_column(left: &StationObservation, right: &StationObservation) -> bool {
    let Some((left_x, _, left_width)) = valid_measurement(left) else {
        return false;
    };
    let Some((right_x, _, right_width)) = valid_measurement(right) else {
        return false;
    };
    let effective_width = left_width.min(right_width);
    (left_x - right_x).abs() <= COLUMN_SEPARATION_WIDTH_RATIO * effective_width
}

fn observation_order(left: &StationObservation, right: &StationObservation) -> Ordering {
    let left_valid = valid_measurement(left);
    let right_valid = valid_measurement(right);
    match (left_valid, right_valid) {
        (Some((left_x, left_y, _)), Some((right_x, right_y, _))) => right_x
            .total_cmp(&left_x)
            .then_with(|| left_y.total_cmp(&right_y))
            // Filenames are consulted only for an exact displacement tie.
            .then_with(|| {
                (left_x.to_bits() == right_x.to_bits() && left_y.to_bits() == right_y.to_bits())
                    .then(|| left.filename.cmp(&right.filename))
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.station_index.cmp(&right.station_index)),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left.station_index.cmp(&right.station_index),
    }
}

fn column_center_x(column: &ColumnCluster, observations: &[StationObservation]) -> f64 {
    column
        .members
        .iter()
        .filter_map(|&index| valid_measurement(&observations[index]).map(|value| value.0))
        .sum::<f64>()
        / column.members.len().max(1) as f64
}

fn overlap_score(overlap_scores: &BTreeMap<(usize, usize), f64>, left: usize, right: usize) -> f64 {
    let pair = ordered_pair(left, right);
    overlap_scores
        .get(&pair)
        .copied()
        .filter(|score| score.is_finite())
        .unwrap_or(0.0)
        .clamp(0.0, 1.0)
}

fn broad_adjacency_kind(
    row: &[u32],
    column: &[u32],
    ambiguous: &BTreeSet<usize>,
    left: usize,
    right: usize,
) -> AdjacencyKind {
    if ambiguous.contains(&left) || ambiguous.contains(&right) {
        AdjacencyKind::Ambiguous
    } else if column[left] == column[right] {
        AdjacencyKind::SameColumn
    } else if row[left] == row[right] {
        AdjacencyKind::SameRow
    } else {
        AdjacencyKind::CrossColumn
    }
}

fn candidate_order(left: &CandidateAdjacency, right: &CandidateAdjacency) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.left.cmp(&right.left))
        .then_with(|| left.right.cmp(&right.right))
}

/// Infer columns and rows, then generate and budget scored station pairs.
///
/// The input must contain one observation for every contiguous station index
/// `0..N`. Input slice order is intentionally irrelevant. `overlap_scores` is
/// computed by the production caller with `panorama_transform_overlap_support`
/// so this pose-free result stores only the resulting area ratio.
pub(crate) fn infer_station_topology(
    observations: &[StationObservation],
    overlap_scores: &BTreeMap<(usize, usize), f64>,
) -> StationTopology {
    if observations.is_empty() {
        return StationTopology::default();
    }

    let mut by_station = observations.to_vec();
    by_station.sort_by_key(|observation| observation.station_index);
    debug_assert!(
        by_station
            .iter()
            .enumerate()
            .all(|(index, observation)| observation.station_index == index),
        "station indices must be contiguous"
    );

    let mut sorted = (0..by_station.len()).collect::<Vec<_>>();
    sorted.sort_by(|&left, &right| observation_order(&by_station[left], &by_station[right]));

    // Complete-link clustering enforces the pairwise 0.5-width rule. A bridge
    // compatible with more than one resulting column is retained in a
    // deterministic nearest column and marked ambiguous below.
    let mut columns = Vec::<ColumnCluster>::new();
    let mut invalid = Vec::new();
    for station in sorted {
        if valid_measurement(&by_station[station]).is_none() {
            invalid.push(station);
            continue;
        }
        let compatible = columns
            .iter()
            .enumerate()
            .filter(|(_, column)| {
                column
                    .members
                    .iter()
                    .all(|&member| same_column(&by_station[station], &by_station[member]))
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let selected = compatible.into_iter().min_by(|&left, &right| {
            let x = by_station[station].world_x.unwrap_or_default();
            (x - column_center_x(&columns[left], &by_station))
                .abs()
                .total_cmp(&(x - column_center_x(&columns[right], &by_station)).abs())
                .then_with(|| left.cmp(&right))
        });
        if let Some(column) = selected {
            columns[column].members.push(station);
        } else {
            columns.push(ColumnCluster {
                members: vec![station],
            });
        }
    }

    // Discovery already follows descending x, but explicitly sorting the final
    // cluster centres makes that contract independent of clustering details.
    columns.sort_by(|left, right| {
        column_center_x(right, &by_station)
            .total_cmp(&column_center_x(left, &by_station))
            .then_with(|| left.members.cmp(&right.members))
    });

    let mut row = vec![0u32; by_station.len()];
    let mut column = vec![0u32; by_station.len()];
    let mut ambiguous = invalid.iter().copied().collect::<BTreeSet<_>>();

    for (column_index, cluster) in columns.iter_mut().enumerate() {
        cluster.members.sort_by(|&left, &right| {
            let (left_x, left_y, _) = valid_measurement(&by_station[left]).unwrap();
            let (right_x, right_y, _) = valid_measurement(&by_station[right]).unwrap();
            left_y
                .total_cmp(&right_y)
                .then_with(|| left_x.total_cmp(&right_x))
                // Filename is the last geometric tie-break and is reached only
                // when both estimated centre displacements are exactly equal.
                .then_with(|| {
                    (left_x.to_bits() == right_x.to_bits() && left_y.to_bits() == right_y.to_bits())
                        .then(|| by_station[left].filename.cmp(&by_station[right].filename))
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| left.cmp(&right))
        });

        for (row_index, &station) in cluster.members.iter().enumerate() {
            row[station] = row_index as u32;
            column[station] = column_index as u32;
        }

        // Equal vertical displacement cannot uniquely determine row indices.
        for pair in cluster.members.windows(2) {
            let left_y = by_station[pair[0]].world_y.unwrap();
            let right_y = by_station[pair[1]].world_y.unwrap();
            if left_y.to_bits() == right_y.to_bits() {
                ambiguous.insert(pair[0]);
                ambiguous.insert(pair[1]);
            }
        }
    }

    // A station compatible with another completed column has more than one
    // valid column assignment even though complete-link clustering selected a
    // deterministic one.
    for station in 0..by_station.len() {
        if valid_measurement(&by_station[station]).is_none() {
            continue;
        }
        let compatible_columns = columns
            .iter()
            .filter(|cluster| {
                cluster
                    .members
                    .iter()
                    .filter(|&&member| member != station)
                    .all(|&member| same_column(&by_station[station], &by_station[member]))
            })
            .count();
        if compatible_columns > 1 {
            ambiguous.insert(station);
        }
    }

    // Unmeasurable stations still receive unique non-negative indices, but the
    // report and candidate set make clear that those indices are placeholders.
    let first_invalid_column = columns.len() as u32;
    for (offset, station) in invalid.into_iter().enumerate() {
        row[station] = 0;
        column[station] = first_invalid_column + offset as u32;
    }

    let station_count = by_station.len();
    let ambiguous_stations = ambiguous.iter().copied().collect::<Vec<_>>();
    let mut candidate_kinds = BTreeMap::<(usize, usize), AdjacencyKind>::new();

    if station_count <= EXHAUSTIVE_STATION_LIMIT {
        // Requirement 5.6: every pair is attempted and no budget truncates it.
        for left in 0..station_count {
            for right in (left + 1)..station_count {
                candidate_kinds.insert(
                    (left, right),
                    broad_adjacency_kind(&row, &column, &ambiguous, left, right),
                );
            }
        }
    } else {
        // The immediate 8-neighbour grid gives each station at most two
        // same-column, two same-row, and four diagonal cross-column candidates.
        for left in 0..station_count {
            for right in (left + 1)..station_count {
                let row_distance = row[left].abs_diff(row[right]);
                let column_distance = column[left].abs_diff(column[right]);
                let kind = if column_distance == 0 && row_distance == 1 {
                    Some(AdjacencyKind::SameColumn)
                } else if row_distance == 0 && column_distance == 1 {
                    Some(AdjacencyKind::SameRow)
                } else if row_distance == 1 && column_distance == 1 {
                    Some(AdjacencyKind::CrossColumn)
                } else {
                    None
                };
                if let Some(kind) = kind {
                    candidate_kinds.insert((left, right), kind);
                }
            }
        }

        // Requirement 5.7: ambiguity-forced all-station pairs enter the same
        // scored budget as ordinary adjacency candidates.
        for &station in &ambiguous_stations {
            for other in 0..station_count {
                if station == other {
                    continue;
                }
                candidate_kinds.insert(ordered_pair(station, other), AdjacencyKind::Ambiguous);
            }
        }
    }

    let mut candidates = candidate_kinds
        .into_iter()
        .map(|((left, right), kind)| CandidateAdjacency {
            left,
            right,
            kind,
            score: overlap_score(overlap_scores, left, right),
        })
        .collect::<Vec<_>>();
    candidates.sort_by(candidate_order);

    let mut truncated_count = 0;
    let mut truncation_min_score = None;
    if station_count > EXHAUSTIVE_STATION_LIMIT {
        let budget = CANDIDATE_BUDGET_PER_STATION.saturating_mul(station_count);
        truncated_count = candidates.len().saturating_sub(budget);
        if truncated_count > 0 {
            candidates.truncate(budget);
            truncation_min_score = candidates.last().map(|candidate| candidate.score);
        }
    }

    StationTopology {
        row,
        column,
        ambiguous: ambiguous_stations,
        candidates,
        truncated_count,
        truncation_min_score,
    }
}

static RUN_TOPOLOGY: Mutex<Option<StationTopology>> = Mutex::new(None);

fn with_run_topology<T>(body: impl FnOnce(&mut Option<StationTopology>) -> T) -> T {
    let mut guard = RUN_TOPOLOGY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    body(&mut guard)
}

pub(crate) fn reset_run_topology() {
    with_run_topology(|topology| *topology = None);
}

pub(crate) fn record_run_topology(topology: StationTopology) {
    with_run_topology(|slot| *slot = Some(topology));
}

pub(crate) fn run_topology_snapshot() -> Option<StationTopology> {
    with_run_topology(|topology| topology.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn station(index: usize, x: f64, y: f64, filename: &str) -> StationObservation {
        StationObservation {
            station_index: index,
            world_x: Some(x),
            world_y: Some(y),
            effective_width: Some(1_000.0),
            filename: filename.to_string(),
        }
    }

    fn grid(rows: usize, columns: usize) -> Vec<StationObservation> {
        (0..rows)
            .flat_map(|row| {
                (0..columns).map(move |column| {
                    let index = row * columns + column;
                    station(
                        index,
                        (columns - column) as f64 * 700.0,
                        row as f64 * 700.0,
                        &format!("station-{index}.NEF"),
                    )
                })
            })
            .collect()
    }

    fn pair_scores(
        station_count: usize,
        score: impl Fn(usize, usize) -> f64,
    ) -> BTreeMap<(usize, usize), f64> {
        let mut scores = BTreeMap::new();
        for left in 0..station_count {
            for right in (left + 1)..station_count {
                scores.insert((left, right), score(left, right));
            }
        }
        scores
    }

    fn candidate(
        topology: &StationTopology,
        left: usize,
        right: usize,
    ) -> Option<&CandidateAdjacency> {
        let pair = ordered_pair(left, right);
        topology
            .candidates
            .iter()
            .find(|candidate| (candidate.left, candidate.right) == pair)
    }

    #[test]
    fn topology_is_invariant_to_rename_and_import_order() {
        let original = vec![
            station(0, 1_000.0, 0.0, "DSC_0001.NEF"),
            station(1, 990.0, 700.0, "DSC_0002.NEF"),
            station(2, 300.0, 5.0, "DSC_0003.NEF"),
            station(3, 290.0, 705.0, "DSC_0004.NEF"),
        ];
        let shuffled_and_renamed = vec![
            station(3, 290.0, 705.0, "renamed-z.NEF"),
            station(1, 990.0, 700.0, "renamed-y.NEF"),
            station(0, 1_000.0, 0.0, "renamed-x.NEF"),
            station(2, 300.0, 5.0, "renamed-w.NEF"),
        ];
        let scores = pair_scores(4, |left, right| (left + right) as f64 / 10.0);

        let expected = infer_station_topology(&original, &scores);
        let actual = infer_station_topology(&shuffled_and_renamed, &scores);

        assert_eq!(actual, expected);
        assert_eq!(actual.column, vec![0, 0, 1, 1]);
        assert_eq!(actual.row, vec![0, 1, 0, 1]);
        assert!(actual.ambiguous.is_empty());
    }

    #[test]
    fn exact_non_unique_indices_are_marked_and_forced_to_pair() {
        let topology = infer_station_topology(
            &[
                station(0, 1_000.0, 100.0, "a.NEF"),
                station(1, 1_000.0, 100.0, "b.NEF"),
                station(2, 300.0, 100.0, "c.NEF"),
            ],
            &pair_scores(3, |left, right| (left + right) as f64 / 4.0),
        );

        assert_eq!(topology.column, vec![0, 0, 1]);
        assert_eq!(topology.row, vec![0, 1, 0]);
        assert_eq!(topology.ambiguous, vec![0, 1]);
        assert!(topology.pair_is_eligible(0, 1));
        assert!(topology.pair_is_eligible(0, 2));
        assert!(topology.pair_is_eligible(1, 2));
        assert!(
            topology
                .candidates
                .iter()
                .all(|candidate| candidate.kind == AdjacencyKind::Ambiguous)
        );
    }

    #[test]
    fn station_count_boundary_is_exhaustive_only_through_64() {
        let stations_64 = grid(8, 8);
        let scores_64 = pair_scores(64, |left, right| (left * 64 + right) as f64 / 4096.0);
        let topology_64 = infer_station_topology(&stations_64, &scores_64);
        assert_eq!(topology_64.candidates.len(), 64 * 63 / 2);
        assert_eq!(topology_64.truncated_count, 0);
        assert_eq!(topology_64.truncation_min_score, None);

        let stations_65 = grid(5, 13);
        let scores_65 = pair_scores(65, |left, right| (left * 65 + right) as f64 / 4225.0);
        let topology_65 = infer_station_topology(&stations_65, &scores_65);
        assert!(topology_65.candidates.len() < 65 * 64 / 2);
        assert!(topology_65.candidates.len() <= CANDIDATE_BUDGET_PER_STATION * 65);
        assert!(candidate(&topology_65, 0, 64).is_none());
    }

    #[test]
    fn adjacency_classes_cover_the_immediate_eight_neighbours() {
        let rows = 8;
        let columns = 9;
        let topology = infer_station_topology(
            &grid(rows, columns),
            &pair_scores(rows * columns, |_, _| 0.5),
        );
        let center = 4 * columns + 4;
        let touching = topology
            .candidates
            .iter()
            .filter(|candidate| candidate.left == center || candidate.right == center)
            .collect::<Vec<_>>();

        assert_eq!(
            touching
                .iter()
                .filter(|candidate| candidate.kind == AdjacencyKind::SameColumn)
                .count(),
            2
        );
        assert_eq!(
            touching
                .iter()
                .filter(|candidate| candidate.kind == AdjacencyKind::SameRow)
                .count(),
            2
        );
        assert_eq!(
            touching
                .iter()
                .filter(|candidate| candidate.kind == AdjacencyKind::CrossColumn)
                .count(),
            4
        );
        assert!(candidate(&topology, center, center + columns).is_some());
        assert!(candidate(&topology, center, center + 1).is_some());
        assert!(candidate(&topology, center, center + columns + 1).is_some());
        assert!(candidate(&topology, center, center + 2 * columns).is_none());
        assert!(candidate(&topology, center, center + 2).is_none());
        assert!(candidate(&topology, center, center + 2 * columns + 1).is_none());
    }

    #[test]
    fn truncation_and_report_are_score_ordered_and_deterministic() {
        let station_count = 65;
        let observations = (0..station_count)
            .rev()
            .map(|station_index| StationObservation {
                station_index,
                world_x: None,
                world_y: None,
                effective_width: None,
                filename: format!("ambiguous-{station_index}.NEF"),
            })
            .collect::<Vec<_>>();
        let scores = pair_scores(station_count, |left, right| {
            ((left * 131 + right * 17) % 11) as f64 / 10.0
        });

        let topology = infer_station_topology(&observations, &scores);
        let repeated = infer_station_topology(&observations, &scores);
        let budget = CANDIDATE_BUDGET_PER_STATION * station_count;
        assert_eq!(topology, repeated);
        assert_eq!(topology.candidates.len(), budget);
        assert_eq!(topology.truncated_count, station_count * 64 / 2 - budget);
        assert_eq!(
            topology.truncation_min_score,
            topology.candidates.last().map(|candidate| candidate.score)
        );
        assert!(topology.candidates.windows(2).all(|pair| {
            pair[0].score > pair[1].score
                || (pair[0].score.to_bits() == pair[1].score.to_bits()
                    && (pair[0].left, pair[0].right) < (pair[1].left, pair[1].right))
        }));
        assert!(
            topology
                .candidates
                .iter()
                .all(|candidate| candidate.kind == AdjacencyKind::Ambiguous)
        );

        let report = TopologyReport::from(&topology);
        assert_eq!(report.truncated_count, topology.truncated_count);
        assert_eq!(report.truncation_min_score, topology.truncation_min_score);
        assert_eq!(report.candidates.len(), topology.candidates.len());
        assert!(
            report
                .candidates
                .iter()
                .zip(&topology.candidates)
                .all(|(record, candidate)| {
                    record.left == candidate.left
                        && record.right == candidate.right
                        && record.kind == candidate.kind
                        && record.score.to_bits() == candidate.score.to_bits()
                })
        );
    }

    #[test]
    fn topology_and_candidate_types_are_pose_free() {
        // Exhaustive destructuring is a compile guard: adding any field to
        // either production type requires this test to be consciously updated.
        let topology = StationTopology::default();
        let StationTopology {
            row,
            column,
            ambiguous,
            candidates,
            truncated_count,
            truncation_min_score,
        } = topology;
        let candidate = CandidateAdjacency {
            left: 0,
            right: 1,
            kind: AdjacencyKind::SameRow,
            score: 0.5,
        };
        let CandidateAdjacency {
            left,
            right,
            kind,
            score,
        } = candidate;

        assert!(row.is_empty());
        assert!(column.is_empty());
        assert!(ambiguous.is_empty());
        assert!(candidates.is_empty());
        assert_eq!(truncated_count, 0);
        assert_eq!(truncation_min_score, None);
        assert_eq!(
            (left, right, kind, score),
            (0, 1, AdjacencyKind::SameRow, 0.5)
        );

        // Source guard complements exhaustive destructuring: it rejects pose-
        // shaped names and matrix types even if somebody also updates the
        // destructuring above. Restrict the scan to the two struct bodies so
        // legitimate pose terminology in module documentation cannot weaken it.
        let source = include_str!("topology.rs");
        let struct_body = |name: &str| {
            let marker = format!("pub(crate) struct {name} {{");
            let start = source.find(&marker).expect("guarded struct exists") + marker.len();
            let end = source[start..]
                .find('}')
                .map(|offset| start + offset)
                .expect("guarded struct body closes");
            &source[start..end]
        };
        for name in ["StationTopology", "CandidateAdjacency"] {
            let body = struct_body(name);
            assert!(
                !body.contains("Matrix3"),
                "{name} must not own Matrix3 fields"
            );
            for field in body.lines().filter_map(|line| {
                let declaration = line.trim().strip_prefix("pub ")?;
                declaration.split_once(':').map(|(field, _)| field.trim())
            }) {
                assert!(
                    !field.to_ascii_lowercase().contains("pose")
                        && !field.to_ascii_lowercase().contains("transform")
                        && !field.to_ascii_lowercase().contains("homography"),
                    "{name} must not own pose field `{field}`"
                );
            }
        }
    }
}
