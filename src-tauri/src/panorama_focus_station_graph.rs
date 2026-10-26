//! Station candidate validation, graph connectivity, and pose-solver entry points.
//!
//! The parent `panorama_focus_pose_solver` module re-exports these symbols so
//! existing callers and test accessors retain their historical paths.

use super::*;

pub(crate) fn station_pose_candidate_has_bounded_local_scale(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    poses: &[Matrix3<f64>],
    stage: &str,
) -> bool {
    if poses.len() != groups.len() {
        return false;
    }
    let mut accepted = true;
    for (station_index, (group, pose)) in groups.iter().zip(poses).enumerate() {
        let anchor = &images[group.anchor];
        let range = station_pose_local_scale_range(pose, anchor.dimensions());
        let ratio = range.map(|(minimum, maximum)| maximum / minimum);
        if ratio.is_some_and(|ratio| ratio <= STATION_POSE_MAX_LOCAL_SCALE_RATIO) {
            continue;
        }
        accepted = false;
        degradation::record_run_degradation(
            degradation::STATION_POSE_LOCAL_SCALE_EXCEEDED,
            serde_json::json!({
                "station_index": station_index,
                "anchor_path": absolute_source_path(&anchor.filename),
                "stage": stage,
                "minimum_local_scale": range.map(|range| range.0),
                "maximum_local_scale": range.map(|range| range.1),
                "local_scale_ratio": ratio,
                "maximum_ratio": STATION_POSE_MAX_LOCAL_SCALE_RATIO,
            }),
        );
        println!(
            "  - Station pose candidate rejected at station {station_index}: Local_Scale ratio {} exceeds {:.2}; retaining spanning-tree seed",
            ratio
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "unmeasurable".to_string()),
            STATION_POSE_MAX_LOCAL_SCALE_RATIO,
        );
    }
    accepted
}

/// Keep a general projective candidate exactly as solved when its sampled
/// Local_Scale is bounded; otherwise preserve every bit of the spanning-tree
/// seed (requirement 6.6).
pub(crate) fn select_station_pose_candidate(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    candidate: &[Matrix3<f64>],
    spanning_tree_seed: &[Matrix3<f64>],
    stage: &str,
) -> Vec<Matrix3<f64>> {
    if station_pose_candidate_has_bounded_local_scale(images, groups, candidate, stage) {
        candidate.to_vec()
    } else {
        spanning_tree_seed.to_vec()
    }
}

pub(crate) fn absolute_source_path(path: &str) -> String {
    let path = Path::new(path);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(path)
    };
    absolute.to_string_lossy().into_owned()
}

pub(crate) fn station_graph_components_from_members(
    station_members: &[Vec<String>],
    relation_pairs: &[(usize, usize)],
) -> Vec<Vec<String>> {
    let mut connectivity = Dsu::new(station_members.len());
    for &(left, right) in relation_pairs {
        if left < station_members.len() && right < station_members.len() {
            connectivity.union(left, right);
        }
    }
    let mut stations_by_root = BTreeMap::<usize, Vec<usize>>::new();
    for station_index in 0..station_members.len() {
        let root = connectivity.find(station_index);
        stations_by_root
            .entry(root)
            .or_default()
            .push(station_index);
    }
    stations_by_root
        .into_values()
        .map(|station_indices| {
            let mut members = station_indices
                .into_iter()
                .flat_map(|station_index| station_members[station_index].iter().cloned())
                .collect::<Vec<_>>();
            members.sort();
            members
        })
        .collect()
}

pub(crate) fn station_graph_components(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    relation_pairs: &[(usize, usize)],
) -> Vec<Vec<String>> {
    let station_members = groups
        .iter()
        .map(|group| {
            group
                .members
                .iter()
                .map(|&image_index| absolute_source_path(&images[image_index].filename))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    station_graph_components_from_members(&station_members, relation_pairs)
}

/// Populate the canonical connectivity report and return the exact rejected
/// diagnostic payload when the accepted Station_Relation graph is disconnected.
pub(crate) fn station_graph_connectivity_rejection(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    relation_pairs: &[(usize, usize)],
    report: &mut stack_report::StationRelationsReport,
) -> Option<serde_json::Value> {
    let components = station_graph_components(images, groups, relation_pairs);
    report.connectivity.components = components.len();
    report.connectivity.component_member_counts = components.iter().map(Vec::len).collect();
    report.connectivity.component_members = components.clone();
    (components.len() > 1).then(|| {
        let component_details = components
            .iter()
            .map(|members| {
                serde_json::json!({
                    "member_count": members.len(),
                    "members": members,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "component_count": components.len(),
            "components": component_details,
            "actions": ["按连续场景重新分组", "补拍增加重叠"],
        })
    })
}

pub(crate) fn geometry_disconnected_user_error(component_count: usize) -> String {
    format!(
        "机位几何图包含 {component_count} 个不连通分量，无法导出完整拼图。可执行操作：“按连续场景重新分组”或“补拍增加重叠”。"
    )
}

#[cfg(test)]
pub(crate) fn solve_focus_capture_group_poses(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
    translation_geometry_verified: &mut bool,
) -> HashMap<usize, Matrix3<f64>> {
    let mut closure_report = stack_report::ClosureReport::default();
    let mut station_relations_report = stack_report::StationRelationsReport::default();
    solve_focus_capture_group_poses_with_report(
        images,
        matches,
        locked_homographies,
        reference_index,
        translation_geometry_verified,
        &mut closure_report,
        &mut station_relations_report,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StationRelationEvidenceKind {
    SourceRaw,
    VirtualTile,
}

impl StationRelationEvidenceKind {
    pub(crate) fn minimum_independent_support(self) -> usize {
        match self {
            Self::SourceRaw => 2,
            // One fused station tile is already the requirement-6.1 primary
            // observation.  This is not a fabricated second source: the
            // evidence kind explicitly exempts it from requirement 6.2.
            Self::VirtualTile => 1,
        }
    }
}

pub(crate) fn solve_focus_capture_group_poses_with_report(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
    translation_geometry_verified: &mut bool,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
) -> HashMap<usize, Matrix3<f64>> {
    solve_station_poses_for_evidence_with_report(
        images,
        matches,
        locked_homographies,
        reference_index,
        translation_geometry_verified,
        closure_report,
        station_relations_report,
        StationRelationEvidenceKind::SourceRaw,
        None,
        None,
    )
}

/// Coverage_Masks the station solver reads for Virtual_Tile evidence: every
/// re-check it makes on a tile pair looks only at pixels both tiles cover
/// (需求 6.1).
pub(crate) fn solve_virtual_tile_station_hypotheses_with_report(
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
    hypotheses: &VirtualTileRelationHypotheses,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
) -> HashMap<usize, Matrix3<f64>> {
    let selected = select_virtual_tile_relation_hypotheses(tiles, hypotheses, coverages);
    solve_virtual_tile_station_poses_with_report(
        tiles,
        coverages,
        &selected,
        initial_homographies,
        station_topology,
        closure_report,
        station_relations_report,
    )
}

pub(crate) fn solve_virtual_tile_station_poses_with_report(
    tiles: &[ImageInfo],
    coverages: VirtualTileSolverCoverages<'_>,
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial_homographies: &HashMap<usize, Matrix3<f64>>,
    station_topology: &topology::StationTopology,
    closure_report: &mut stack_report::ClosureReport,
    station_relations_report: &mut stack_report::StationRelationsReport,
) -> HashMap<usize, Matrix3<f64>> {
    let mut verified = false;
    solve_station_poses_for_evidence_with_report(
        tiles,
        matches,
        initial_homographies,
        0,
        &mut verified,
        closure_report,
        station_relations_report,
        StationRelationEvidenceKind::VirtualTile,
        Some(station_topology),
        Some(coverages),
    )
}
