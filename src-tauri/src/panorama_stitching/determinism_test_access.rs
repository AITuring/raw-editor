use super::*;

/// One capture station in a shape a sibling module can hold.
///
/// [`FocusCaptureGroup`] itself stays private; this carries the same data
/// with the visibility the property test needs.
pub(crate) struct StationView {
    pub(crate) members: Vec<usize>,
    pub(crate) anchor: usize,
    pub(crate) local_to_anchor: HashMap<usize, Matrix3<f64>>,
}

fn into_views(groups: Vec<FocusCaptureGroup>) -> Vec<StationView> {
    groups
        .into_iter()
        .map(|group| StationView {
            members: group.members,
            anchor: group.anchor,
            local_to_anchor: group.local_to_anchor,
        })
        .collect()
}

fn from_views(stations: &[StationView]) -> Vec<FocusCaptureGroup> {
    stations
        .iter()
        .map(|station| FocusCaptureGroup {
            members: station.members.clone(),
            anchor: station.anchor,
            local_to_anchor: station.local_to_anchor.clone(),
        })
        .collect()
}

/// [`focus_capture_groups`] — station membership and station order.
pub(crate) fn capture_stations(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Option<Vec<StationView>> {
    focus_capture_groups(images, matches, locked_homographies).map(|(groups, _)| into_views(groups))
}

/// [`canonical_match_direction`] — pair orientation derived from feature
/// content whenever both synthetic images carry content signatures.
pub(crate) fn canonical_pair_direction(
    images: &[ImageInfo],
    first: usize,
    second: usize,
) -> (usize, usize, bool) {
    canonical_match_direction(images, first, second)
}

/// Full station-pose test outcome, including the connectivity report that
/// distinguishes a valid solved graph from the explicit requirements
/// 6.7/12.4 disconnected rejection.
pub(crate) struct StationPoseSolveView {
    pub(crate) poses: HashMap<usize, Matrix3<f64>>,
    pub(crate) translation_geometry_verified: bool,
    pub(crate) connectivity: stack_report::ConnectivityReport,
}

/// [`solve_focus_capture_group_poses_with_report`] — the station-level pose
/// solve together with its machine-readable connectivity outcome.
pub(crate) fn solve_station_poses_with_report(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
) -> StationPoseSolveView {
    let mut translation_geometry_verified = false;
    let mut closure_report = stack_report::ClosureReport::default();
    let mut station_relations_report = stack_report::StationRelationsReport::default();
    let poses = solve_focus_capture_group_poses_with_report(
        images,
        matches,
        locked_homographies,
        reference_index,
        &mut translation_geometry_verified,
        &mut closure_report,
        &mut station_relations_report,
    );
    StationPoseSolveView {
        poses,
        translation_geometry_verified,
        connectivity: station_relations_report.connectivity,
    }
}

/// [`solve_focus_capture_group_poses`] — the station level pose solve.
///
/// Returns the solved per-source world homographies together with the
/// `translation_geometry_verified` flag the caller receives out-of-band.
pub(crate) fn solve_station_poses(
    images: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    locked_homographies: &HashMap<usize, Matrix3<f64>>,
    reference_index: usize,
) -> (HashMap<usize, Matrix3<f64>>, bool) {
    let outcome =
        solve_station_poses_with_report(images, matches, locked_homographies, reference_index);
    (outcome.poses, outcome.translation_geometry_verified)
}

/// [`focus_source_order_group_compositor_order`] — the seam order derived
/// from the production Capture_Topology_Model row/column indices.
pub(crate) fn station_compositor_order(
    images: &[ImageInfo],
    stations: &[StationView],
    station_homographies: &HashMap<usize, Matrix3<f64>>,
) -> Vec<usize> {
    let mut groups = from_views(stations);
    let _ = canonicalize_station_geometry_anchors(images, &mut groups, station_homographies);
    let station_topology = focus_station_topology(images, &groups, station_homographies);
    focus_source_order_group_compositor_order(groups.len(), &station_topology)
}
