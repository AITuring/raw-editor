//! Evidence-specific entry points into the shared station pose solver.
//!
//! Source_RAW frames and Virtual_Tile planes enter the same solver; these
//! wrappers only fix the evidence kind, the optional topology and the
//! Coverage_Masks a Virtual_Tile relation is re-measured on.

use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::shared::MatchInfo;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use crate::panorama_utils::stack_pipeline::topology;
use nalgebra::Matrix3;
use std::collections::HashMap;

use super::{
    StationRelationEvidenceKind, VirtualTileRelationHypotheses, VirtualTileSolverCoverages,
    select_virtual_tile_relation_hypotheses, solve_station_poses_for_evidence_with_report,
};

#[cfg(test)]
pub(super) fn solve_focus_capture_group_poses(
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

pub(super) fn solve_focus_capture_group_poses_with_report(
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

pub(super) fn solve_virtual_tile_station_hypotheses_with_report(
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

pub(super) fn solve_virtual_tile_station_poses_with_report(
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
