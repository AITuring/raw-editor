//! Station relation graph topology: spanning tree, layout observations and connectivity.
//!
//! The maximum-inlier spanning tree seeds station poses, topology observations
//! stay pose-free for the layout inference (requirements 5.4 and 5.5), and a
//! disconnected accepted graph is reported with its components instead of being
//! bridged.

use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use crate::panorama_utils::stack_pipeline::topology;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::BTreeMap;
use std::collections::HashMap;

use super::{
    Dsu, FocusCaptureGroup, absolute_source_path, panorama_transform_overlap_support,
    transformed_point,
};

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

pub(super) fn focus_station_topology(
    images: &[ImageInfo],
    groups: &[FocusCaptureGroup],
    station_homographies: &HashMap<usize, Matrix3<f64>>,
) -> topology::StationTopology {
    let observations = groups
        .iter()
        .enumerate()
        .map(|(station_index, group)| {
            let anchor = &images[group.anchor];
            let pose = station_homographies.get(&anchor.id);
            let center = pose.and_then(|pose| {
                transformed_point(
                    pose,
                    Point2::new(anchor.width as f64 * 0.5, anchor.height as f64 * 0.5),
                )
            });
            let effective_width = pose.and_then(|pose| {
                let corners = [
                    Point2::new(0.0, 0.0),
                    Point2::new(anchor.width as f64, 0.0),
                    Point2::new(0.0, anchor.height as f64),
                    Point2::new(anchor.width as f64, anchor.height as f64),
                ];
                let projected = corners
                    .into_iter()
                    .map(|corner| transformed_point(pose, corner))
                    .collect::<Option<Vec<_>>>()?;
                let min_x = projected
                    .iter()
                    .map(|point| point.x)
                    .fold(f64::INFINITY, f64::min);
                let max_x = projected
                    .iter()
                    .map(|point| point.x)
                    .fold(f64::NEG_INFINITY, f64::max);
                let width = max_x - min_x;
                (width.is_finite() && width > 0.0).then_some(width)
            });
            topology::StationObservation {
                station_index,
                world_x: center.map(|point| point.x),
                world_y: center.map(|point| point.y),
                effective_width,
                filename: anchor.filename.clone(),
            }
        })
        .collect::<Vec<_>>();

    // Candidate scores are the transform-estimated overlap area divided by one
    // station's effective area. Keep the matrix outside StationTopology so the
    // topology result remains pose-free (requirements 5.4 and 5.5).
    let mut overlap_scores = BTreeMap::new();
    for left in 0..groups.len() {
        let left_anchor = &images[groups[left].anchor];
        let Some(left_to_world) = station_homographies.get(&left_anchor.id) else {
            continue;
        };
        for right in (left + 1)..groups.len() {
            let right_anchor = &images[groups[right].anchor];
            let Some(world_to_right) = station_homographies
                .get(&right_anchor.id)
                .and_then(|pose| pose.try_inverse())
            else {
                continue;
            };
            let left_to_right = world_to_right * left_to_world;
            overlap_scores.insert(
                (left, right),
                panorama_transform_overlap_support(
                    &left_to_right,
                    left_anchor.dimensions(),
                    right_anchor.dimensions(),
                ),
            );
        }
    }

    topology::infer_station_topology(&observations, &overlap_scores)
}

pub(super) fn station_graph_components_from_members(
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

pub(super) fn station_graph_components(
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
pub(super) fn station_graph_connectivity_rejection(
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

pub(super) fn geometry_disconnected_user_error(component_count: usize) -> String {
    format!(
        "机位几何图包含 {component_count} 个不连通分量，无法导出完整拼图。可执行操作：“按连续场景重新分组”或“补拍增加重叠”。"
    )
}
