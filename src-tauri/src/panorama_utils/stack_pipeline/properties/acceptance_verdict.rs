//! Acceptance_Harness verdict (Property 94).
//!
//! The pure verdict over a Stack_Report, the baseline report and the violation
//! injection it is checked with, and the verdict tests on graph, membership and
//! fusion invariants and on undecoded sources.

use super::super::report::ConnectivityReport;
use super::super::report::FusionReport;
use super::super::report::FusionSolverStatus;
use super::super::report::QualityGateCriterionRecord;
use super::super::report::QualityGateVerdict;
use super::super::report::SourceRecord;
use super::super::report::StackReport;
use proptest::prelude::*;
use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// Property 94 / acceptance harness skeleton (17.5, 17.15)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct StackAcceptanceFailure {
    pub(super) identifier: &'static str,
    pub(super) measured: String,
    pub(super) threshold: String,
}

/// The one production acceptance decision.  The harness and P94 both call
/// this function; neither maintains a second copy of the acceptance rules.
pub(super) fn stack_acceptance_failures(report: &StackReport) -> Vec<StackAcceptanceFailure> {
    let mut failures = Vec::new();
    let mut add = |identifier: &'static str, measured: String, threshold: String| {
        failures.push(StackAcceptanceFailure {
            identifier,
            measured,
            threshold,
        });
    };
    let sources = report
        .input
        .sources
        .iter()
        .map(|source| source.path.as_str())
        .collect::<BTreeSet<_>>();
    let members = report
        .grouping
        .stations
        .iter()
        .flat_map(|station| station.members.iter().map(String::as_str))
        .collect::<Vec<_>>();
    let grouped = members.iter().copied().collect::<BTreeSet<_>>();
    if report.input.source_count != 84
        || sources.len() != 84
        || members.len() != 84
        || grouped != sources
    {
        add(
            "acceptance_source_count",
            format!(
                "input={}, grouped={}, unique={}",
                report.input.source_count,
                members.len(),
                grouped.len()
            ),
            "84".to_string(),
        );
    }
    let station_indices = report
        .grouping
        .stations
        .iter()
        .map(|station| station.index)
        .collect::<BTreeSet<_>>();
    if report.grouping.station_count != 28
        || report.grouping.stations.len() != 28
        || station_indices != (0..28).collect()
    {
        add(
            "acceptance_station_count",
            format!(
                "reported={}, records={}, unique_indices={}",
                report.grouping.station_count,
                report.grouping.stations.len(),
                station_indices.len()
            ),
            "28 stations indexed 0..28".to_string(),
        );
    }
    if !report.grouping.isolated.is_empty() {
        add(
            "acceptance_isolated_sources",
            report.grouping.isolated.len().to_string(),
            "0".to_string(),
        );
    }
    let undecoded_input_count = report
        .input
        .sources
        .iter()
        .filter(|source| !source.decoded)
        .count();
    if undecoded_input_count != 0 || !report.grouping.undecodable.is_empty() {
        add(
            "acceptance_undecodable_sources",
            format!(
                "input={}, grouping={}",
                undecoded_input_count,
                report.grouping.undecodable.len()
            ),
            "input=0, grouping=0".to_string(),
        );
    }

    let mut relation_components = (0..28).collect::<Vec<_>>();
    let mut relation_endpoints_valid = true;
    for relation in &report.station_relations.accepted {
        if relation.left >= 28 || relation.right >= 28 {
            relation_endpoints_valid = false;
            continue;
        }
        let from = relation_components[relation.right];
        let to = relation_components[relation.left];
        for component in &mut relation_components {
            if *component == from {
                *component = to;
            }
        }
    }
    let relation_component_count = relation_components
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len();
    if !relation_endpoints_valid || relation_component_count != 1 {
        add(
            "acceptance_relation_graph",
            format!(
                "endpoints_valid={relation_endpoints_valid}, components={relation_component_count}"
            ),
            "all endpoints in 0..28 and one component".to_string(),
        );
    }

    let connectivity_members = report
        .station_relations
        .connectivity
        .component_members
        .first()
        .map(|members| members.iter().map(String::as_str).collect::<BTreeSet<_>>());
    if report.station_relations.connectivity.components != 1
        || report
            .station_relations
            .connectivity
            .component_members
            .len()
            != 1
        || report
            .station_relations
            .connectivity
            .component_member_counts
            != [84]
        || connectivity_members.as_ref() != Some(&sources)
        || grouped != sources
    {
        add(
            "acceptance_connectivity_membership",
            format!(
                "components={}, member_lists={}, counts={:?}, unique_members={}",
                report.station_relations.connectivity.components,
                report
                    .station_relations
                    .connectivity
                    .component_members
                    .len(),
                report
                    .station_relations
                    .connectivity
                    .component_member_counts,
                connectivity_members.as_ref().map_or(0, BTreeSet::len)
            ),
            "one component containing all 84 grouped inputs".to_string(),
        );
    }
    if report.quality_gate.verdict != QualityGateVerdict::Pass {
        add(
            "acceptance_quality_gate_verdict",
            serde_json::to_string(&report.quality_gate.verdict).unwrap_or_default(),
            "pass".to_string(),
        );
    }
    let failed_criteria = report
        .quality_gate
        .criteria
        .iter()
        .filter(|criterion| {
            !criterion.diagnostic
                && (criterion.verdict == QualityGateVerdict::Fail || !criterion.failed.is_empty())
        })
        .count();
    if failed_criteria != 0 {
        add(
            "acceptance_failed_criteria",
            failed_criteria.to_string(),
            "0".to_string(),
        );
    }
    let union = report.composition.union_projected_pixels;
    let opaque_ratio = if union == 0 {
        0.0
    } else {
        report.composition.opaque_pixels as f64 / union as f64
    };
    if union == 0 || opaque_ratio < 0.98 {
        add(
            "acceptance_opaque_union_ratio",
            format!("{opaque_ratio:.9}"),
            "0.98".to_string(),
        );
    }
    if report.resources.peak_rss_bytes > report.resources.memory_threshold_bytes {
        add(
            "acceptance_peak_rss",
            report.resources.peak_rss_bytes.to_string(),
            report.resources.memory_threshold_bytes.to_string(),
        );
    }
    if report.resources.network_requests != 0 {
        add(
            "acceptance_network_requests",
            report.resources.network_requests.to_string(),
            "0".to_string(),
        );
    }
    let fusion_indices = report
        .fusion
        .iter()
        .map(|fusion| fusion.station_index)
        .collect::<BTreeSet<_>>();
    if report.fusion.len() != 28 || fusion_indices != (0..28).collect() {
        add(
            "acceptance_fusion_coverage",
            format!(
                "records={}, unique_indices={}",
                report.fusion.len(),
                fusion_indices.len()
            ),
            "28 stations indexed 0..28".to_string(),
        );
    }
    let non_graph_cut = report
        .fusion
        .iter()
        .filter(|fusion| fusion.solver_status != FusionSolverStatus::GraphCut)
        .count();
    if non_graph_cut != 0 {
        add(
            "acceptance_fusion_solver_status",
            non_graph_cut.to_string(),
            "0".to_string(),
        );
    }
    let changed_sources = report
        .input
        .sources
        .iter()
        .filter(|source| {
            source.sha256_before.is_empty()
                || source.sha256_after.is_empty()
                || source.sha256_before != source.sha256_after
        })
        .count();
    if report.input.sources.len() != 84 || changed_sources != 0 {
        add(
            "acceptance_source_sha256_stable",
            format!(
                "records={}, changed={changed_sources}",
                report.input.sources.len()
            ),
            "84 records, changed=0".to_string(),
        );
    }
    failures
}

pub(super) fn stack_acceptance_baseline_report() -> StackReport {
    let mut report = StackReport::default();
    let paths = (0..84)
        .map(|index| format!("/fixture/DSC_{:04}.NEF", 3680 + index))
        .collect::<Vec<_>>();
    report.input.source_count = 84;
    report.input.sources = paths
        .iter()
        .map(|path| SourceRecord {
            path: path.clone(),
            sha256_before: "a".repeat(64),
            sha256_after: "a".repeat(64),
            decoded: true,
        })
        .collect();
    report.grouping.station_count = 28;
    report.grouping.stations = paths
        .chunks_exact(3)
        .enumerate()
        .map(
            |(index, members)| crate::panorama_utils::stack_pipeline::report::StationRecord {
                index,
                members: members.to_vec(),
                ..crate::panorama_utils::stack_pipeline::report::StationRecord::default()
            },
        )
        .collect();
    report.station_relations.accepted = (0..27)
        .map(|left| crate::panorama_utils::stack_pipeline::report::AcceptedStationRelationRecord {
            left,
            right: left + 1,
            evidence_kind: crate::panorama_utils::stack_pipeline::report::StationRelationEvidenceKind::VirtualTile,
            inliers: 24,
            spatial_support: 0.2,
            scale_ratio: 1.0,
            median_error_px: 1.0,
            independent_support: 1,
        })
        .collect();
    report.station_relations.connectivity = ConnectivityReport {
        components: 1,
        component_members: vec![paths.clone()],
        component_member_counts: vec![84],
    };
    report.composition.union_projected_pixels = 10_000;
    report.composition.opaque_pixels = 9_800;
    report.resources.memory_threshold_bytes = 100;
    report.resources.peak_rss_bytes = 100;
    report.resources.network_requests = 0;
    report.fusion = (0..28)
        .map(|station_index| FusionReport {
            station_index,
            solver_status: FusionSolverStatus::GraphCut,
            ..FusionReport::default()
        })
        .collect();
    report.quality_gate.verdict = QualityGateVerdict::Pass;
    report.quality_gate.criteria = vec![QualityGateCriterionRecord {
        name: "baseline".to_string(),
        verdict: QualityGateVerdict::Pass,
        ..QualityGateCriterionRecord::default()
    }];
    report
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AcceptanceViolationKind {
    SourceCount,
    StationCount,
    Isolated,
    Undecodable,
    RelationGraph,
    ConnectivityMembership,
    FusionCoverage,
    QualityVerdict,
    FailedCriterion,
    OpaqueRatio,
    PeakRss,
    Network,
    Solver,
    SourceDigest,
}

pub(super) fn inject_acceptance_violation(
    report: &mut StackReport,
    kind: AcceptanceViolationKind,
) -> &'static str {
    match kind {
        AcceptanceViolationKind::SourceCount => {
            report.input.source_count = 83;
            "acceptance_source_count"
        }
        AcceptanceViolationKind::StationCount => {
            report.grouping.station_count = 27;
            "acceptance_station_count"
        }
        AcceptanceViolationKind::Isolated => {
            report.grouping.isolated.push(Default::default());
            "acceptance_isolated_sources"
        }
        AcceptanceViolationKind::Undecodable => {
            report.grouping.undecodable.push(Default::default());
            "acceptance_undecodable_sources"
        }
        AcceptanceViolationKind::RelationGraph => {
            report.station_relations.accepted.pop();
            "acceptance_relation_graph"
        }
        AcceptanceViolationKind::ConnectivityMembership => {
            report.station_relations.connectivity.components = 2;
            "acceptance_connectivity_membership"
        }
        AcceptanceViolationKind::FusionCoverage => {
            report.fusion.pop();
            "acceptance_fusion_coverage"
        }
        AcceptanceViolationKind::QualityVerdict => {
            report.quality_gate.verdict = QualityGateVerdict::Fail;
            "acceptance_quality_gate_verdict"
        }
        AcceptanceViolationKind::FailedCriterion => {
            report.quality_gate.criteria[0].verdict = QualityGateVerdict::Fail;
            report.quality_gate.criteria[0]
                .failed
                .push(Default::default());
            "acceptance_failed_criteria"
        }
        AcceptanceViolationKind::OpaqueRatio => {
            report.composition.opaque_pixels = 9_700;
            "acceptance_opaque_union_ratio"
        }
        AcceptanceViolationKind::PeakRss => {
            report.resources.peak_rss_bytes = 101;
            "acceptance_peak_rss"
        }
        AcceptanceViolationKind::Network => {
            report.resources.network_requests = 1;
            "acceptance_network_requests"
        }
        AcceptanceViolationKind::Solver => {
            report.fusion[0].solver_status = FusionSolverStatus::PerCellFallback;
            "acceptance_fusion_solver_status"
        }
        AcceptanceViolationKind::SourceDigest => {
            report.input.sources[0].sha256_after = "b".repeat(64);
            "acceptance_source_sha256_stable"
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]
    // Feature: layered-camera-group-focus-stitching, Property 94: 门禁 verdict 是纯函数。
    #[test]
    fn property_94_stack_acceptance_verdict_is_pure(
        violations in prop::collection::vec(any::<bool>(), 14),
    ) {
        let kinds = [
            AcceptanceViolationKind::SourceCount,
            AcceptanceViolationKind::StationCount,
            AcceptanceViolationKind::Isolated,
            AcceptanceViolationKind::Undecodable,
            AcceptanceViolationKind::RelationGraph,
            AcceptanceViolationKind::ConnectivityMembership,
            AcceptanceViolationKind::FusionCoverage,
            AcceptanceViolationKind::QualityVerdict,
            AcceptanceViolationKind::FailedCriterion,
            AcceptanceViolationKind::OpaqueRatio,
            AcceptanceViolationKind::PeakRss,
            AcceptanceViolationKind::Network,
            AcceptanceViolationKind::Solver,
            AcceptanceViolationKind::SourceDigest,
        ];
        let mut report = stack_acceptance_baseline_report();
        prop_assert!(stack_acceptance_failures(&report).is_empty());
        let expected = kinds
            .into_iter()
            .zip(violations)
            .filter(|&(_, inject)| inject)
            .map(|(kind, _)| inject_acceptance_violation(&mut report, kind))
            .collect::<BTreeSet<_>>();
        let failures = stack_acceptance_failures(&report);
        prop_assert_eq!(failures.iter().map(|failure| failure.identifier).collect::<BTreeSet<_>>(), expected);
    }
}

#[test]
fn acceptance_graph_membership_and_fusion_invariants_are_deterministic() {
    let baseline = stack_acceptance_baseline_report();
    assert!(stack_acceptance_failures(&baseline).is_empty());

    let mut out_of_bounds = baseline.clone();
    out_of_bounds.station_relations.accepted[0].left = 28;
    assert_eq!(
        stack_acceptance_failures(&out_of_bounds)
            .iter()
            .map(|failure| failure.identifier)
            .collect::<Vec<_>>(),
        ["acceptance_relation_graph"]
    );

    let mut duplicate_member = baseline.clone();
    duplicate_member
        .station_relations
        .connectivity
        .component_members[0][1] = duplicate_member
        .station_relations
        .connectivity
        .component_members[0][0]
        .clone();
    assert_eq!(
        stack_acceptance_failures(&duplicate_member)
            .iter()
            .map(|failure| failure.identifier)
            .collect::<Vec<_>>(),
        ["acceptance_connectivity_membership"]
    );

    for duplicate in [false, true] {
        let mut fusion = baseline.clone();
        if duplicate {
            fusion.fusion[27].station_index = 26;
        } else {
            fusion.fusion.pop();
        }
        assert_eq!(
            stack_acceptance_failures(&fusion)
                .iter()
                .map(|failure| failure.identifier)
                .collect::<Vec<_>>(),
            ["acceptance_fusion_coverage"]
        );
    }
}

#[test]
fn acceptance_verdict_rejects_an_input_source_record_marked_undecoded() {
    let mut report = stack_acceptance_baseline_report();
    report.input.sources[17].decoded = false;
    assert!(report.grouping.undecodable.is_empty());
    assert_eq!(
        stack_acceptance_failures(&report)
            .iter()
            .map(|failure| failure.identifier)
            .collect::<Vec<_>>(),
        ["acceptance_undecodable_sources"]
    );
}
