//! Test-only alignment cache writer.
//!
//! Persists the geometry of a real focus stack so review runs can inspect it
//! without repeating feature matching.

#[cfg(test)]
use crate::panorama_utils::stitching::Projection;
#[cfg(test)]
use nalgebra::Matrix3;
#[cfg(test)]
use nalgebra::Point2;
#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
use std::path::Path;

use super::{
    FocusCaptureGroup, FocusLayerWarp, ImageInfo, MatchInfo, focus_match_center_motion_ratio,
    focus_match_is_local_bracket, median_symmetric_error, median_value, panorama_spatial_support,
    panorama_transform_overlap_support, transformed_point,
};

/// Persist the geometry needed to inspect a real focus stack without paying
/// the feature matching cost again.  The cache intentionally contains source
/// paths and dimensions alongside every matrix so it cannot silently be
/// applied to a different selection.  This helper is test-only; normal builds
/// never read an environment variable or write this file.
#[cfg(test)]
pub(super) fn write_alignment_cache(
    path: &Path,
    images: &[ImageInfo],
    ordered_indices: &[usize],
    matches: &HashMap<(usize, usize), MatchInfo>,
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    focus_layer_warp: Option<&FocusLayerWarp>,
    capture_groups: Option<&[FocusCaptureGroup]>,
    projection: Projection,
    canvas_width: u32,
    canvas_height: u32,
) -> Result<(), String> {
    fn matrix_values(matrix: &Matrix3<f64>) -> [f64; 9] {
        [
            matrix[(0, 0)],
            matrix[(0, 1)],
            matrix[(0, 2)],
            matrix[(1, 0)],
            matrix[(1, 1)],
            matrix[(1, 2)],
            matrix[(2, 0)],
            matrix[(2, 1)],
            matrix[(2, 2)],
        ]
    }

    let source_records = images
        .iter()
        .map(|image| {
            serde_json::json!({
                "id": image.id,
                "filename": image.filename,
                "width": image.width,
                "height": image.height,
                "scale_factor": image.scale_factor,
                "focal_length_35mm": image.focal_length_35mm,
                "overview_reference": image.overview_reference,
            })
        })
        .collect::<Vec<_>>();
    let homographies = global_homographies
        .iter()
        .map(|(id, matrix)| (id.to_string(), serde_json::json!(matrix_values(matrix))))
        .collect::<serde_json::Map<_, _>>();
    let match_records = matches
        .iter()
        .filter_map(|(&(source_index, target_index), match_info)| {
            let source = images.get(source_index)?;
            let target = images.get(target_index)?;
            let direct = match_info.homography;
            let global_source = global_homographies.get(&source.id)?;
            let global_target = global_homographies.get(&target.id)?;
            let global_relative = global_target.try_inverse()? * global_source;
            let samples = [
                Point2::new(source.width as f64 * 0.2, source.height as f64 * 0.2),
                Point2::new(source.width as f64 * 0.8, source.height as f64 * 0.2),
                Point2::new(source.width as f64 * 0.5, source.height as f64 * 0.5),
                Point2::new(source.width as f64 * 0.2, source.height as f64 * 0.8),
                Point2::new(source.width as f64 * 0.8, source.height as f64 * 0.8),
            ];
            let mut disagreement = samples
                .into_iter()
                .filter_map(|point| {
                    let direct_point = transformed_point(&direct, point)?;
                    let global_point = transformed_point(&global_relative, point)?;
                    let distance = (direct_point - global_point).norm();
                    distance.is_finite().then_some(distance)
                })
                .collect::<Vec<_>>();
            let global_disagreement_px = median_value(&mut disagreement);
            Some(serde_json::json!({
                "source_index": source_index,
                "target_index": target_index,
                "source_id": source.id,
                "target_id": target.id,
                "inliers": match_info.inliers,
                "observations": match_info.points.len(),
                "sequence_bridge": match_info.sequence_bridge,
                "coarse_bridge": match_info.coarse_bridge,
                "local_bracket": focus_match_is_local_bracket(source, target, match_info),
                "center_motion_ratio": focus_match_center_motion_ratio(source, target, match_info),
                "overlap_support": panorama_transform_overlap_support(
                    &direct,
                    source.dimensions(),
                    target.dimensions(),
                ),
                "spatial_support": panorama_spatial_support(
                    &match_info.points,
                    source.dimensions(),
                    target.dimensions(),
                ),
                "direct_median_error_px": median_symmetric_error(&direct, &match_info.points),
                "global_disagreement_px": global_disagreement_px,
                "direct_homography": matrix_values(&direct),
                "global_relative_homography": matrix_values(&global_relative),
            }))
        })
        .collect::<Vec<_>>();
    let focus_bands = focus_layer_warp.map(|warp| {
        warp.bands
            .iter()
            .map(|band| {
                let homographies = band
                    .homographies
                    .iter()
                    .map(|(id, matrix)| (id.to_string(), serde_json::json!(matrix_values(matrix))))
                    .collect::<serde_json::Map<_, _>>();
                let source_ranges = band
                    .source_ranges
                    .iter()
                    .map(|(id, (start, end))| (id.to_string(), serde_json::json!([start, end])))
                    .collect::<serde_json::Map<_, _>>();
                let source_x_ranges = band
                    .source_x_ranges
                    .iter()
                    .map(|(id, (start, end))| (id.to_string(), serde_json::json!([start, end])))
                    .collect::<serde_json::Map<_, _>>();
                serde_json::json!({
                    "homographies": homographies,
                    "source_ranges": source_ranges,
                    "source_x_ranges": source_x_ranges,
                    "relax_foreground_seam": band.relax_foreground_seam,
                    "foreground_only": band.foreground_only,
                    "physical_edge": band.physical_edge,
                })
            })
            .collect::<Vec<_>>()
    });
    let capture_group_records = capture_groups.map(|groups| {
        groups
            .iter()
            .enumerate()
            .map(|(group_index, group)| {
                let anchor = &images[group.anchor];
                let anchor_center = Point2::new(
                    anchor.width as f64 * 0.5,
                    anchor.height as f64 * 0.5,
                );
                let anchor_world = global_homographies
                    .get(&anchor.id)
                    .and_then(|pose| transformed_point(pose, anchor_center));
                let mut drifts = group
                    .members
                    .iter()
                    .filter_map(|&member_index| {
                        let image = images.get(member_index)?;
                        let pose = global_homographies.get(&image.id)?;
                        let center = Point2::new(
                            image.width as f64 * 0.5,
                            image.height as f64 * 0.5,
                        );
                        let world = transformed_point(pose, center)?;
                        let anchor_world = anchor_world?;
                        let drift = (world - anchor_world).norm();
                        drift.is_finite().then_some(drift)
                    })
                    .collect::<Vec<_>>();
                drifts.sort_by(f64::total_cmp);
                let median_drift_px = median_value(&mut drifts);
                let maximum_drift_px = drifts.into_iter().fold(0.0, f64::max);
                serde_json::json!({
                    "group_index": group_index,
                    "anchor_id": anchor.id,
                    "member_ids": group.members.iter().filter_map(|&index| images.get(index).map(|image| image.id)).collect::<Vec<_>>(),
                    "member_count": group.members.len(),
                    "median_center_drift_px": median_drift_px,
                    "maximum_center_drift_px": maximum_drift_px,
                })
            })
            .collect::<Vec<_>>()
    });
    let projection_name = match projection {
        Projection::Planar => "planar",
        Projection::Cylindrical => "cylindrical",
        Projection::Spherical => "spherical",
    };
    let cache = serde_json::json!({
        "schema": 2,
        "projection": projection_name,
        "canvas": { "width": canvas_width, "height": canvas_height },
        "sources": source_records,
        "ordered_ids": ordered_indices.iter().map(|id| *id).collect::<Vec<_>>(),
        "global_homographies": homographies,
        "matches": match_records,
        "focus_warp": focus_bands,
        "capture_groups": capture_group_records,
    });
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("Failed to create alignment cache directory: {error}"))?;
    }
    let bytes = serde_json::to_vec_pretty(&cache)
        .map_err(|error| format!("Failed to encode alignment cache: {error}"))?;
    std::fs::write(path, bytes).map_err(|error| {
        format!(
            "Failed to write alignment cache {}: {error}",
            path.display()
        )
    })?;
    println!(
        "Alignment cache written: {} ({} source poses, canvas {}x{})",
        path.display(),
        images.len(),
        canvas_width,
        canvas_height
    );
    Ok(())
}
