//! Residual_Warp evidence from accepted Station_Relation correspondences.
//!
//! The residual field is fitted only to forward/reverse-consistent relation
//! points mapped through the final station poses. Without enough bidirectional
//! closure evidence the warp stays identity and the reason is recorded.

use crate::panorama_utils::shared::ImageInfo;
use crate::panorama_utils::shared::MatchInfo;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use crate::panorama_utils::stack_pipeline::residual_warp;
use nalgebra::Matrix3;
use std::collections::HashMap;

use super::transformed_point;

/// Build the run-scoped local residual field from the same accepted
/// Virtual_Tile correspondences that produced the station pose graph.
///
/// The relation matcher keeps only forward/reverse-consistent points in
/// `MatchInfo.points`.  Mapping both endpoints through the final station poses
/// turns those points into world-space displacement observations.  The model
/// remains empty unless the displacement P95 is above the residual threshold
/// and at least sixteen observations survive the geometric round-trip and
/// displacement bounds, so an accepted relation with a small residual remains
/// exactly on the global homography path.
pub(super) fn residual_model_from_station_relations(
    tiles: &[ImageInfo],
    tile_homographies: &HashMap<usize, Matrix3<f64>>,
    relations: &HashMap<(usize, usize), MatchInfo>,
    accepted: &[stack_report::AcceptedStationRelationRecord],
) -> residual_warp::ResidualWarp {
    let mut model = residual_warp::ResidualWarp::new();
    for relation in accepted.iter().copied() {
        if relation.evidence_kind != stack_report::StationRelationEvidenceKind::VirtualTile {
            continue;
        }
        let Some(left_tile) = tiles.get(relation.left) else {
            continue;
        };
        let Some(right_tile) = tiles.get(relation.right) else {
            continue;
        };
        let Some(left_to_world) = tile_homographies.get(&left_tile.id) else {
            continue;
        };
        let Some(right_to_world) = tile_homographies.get(&right_tile.id) else {
            continue;
        };

        // Station topology normally stores the key in this orientation.  The
        // reverse branch keeps the producer deterministic if a future caller
        // supplies a relation map keyed in the opposite direction.
        let (match_info, reverse_key) =
            if let Some(match_info) = relations.get(&(relation.left, relation.right)) {
                (match_info, false)
            } else if let Some(match_info) = relations.get(&(relation.right, relation.left)) {
                (match_info, true)
            } else {
                continue;
            };
        let relation_homography = if reverse_key {
            let Some(inverse) = match_info.homography.try_inverse() else {
                continue;
            };
            inverse
        } else {
            match_info.homography
        };

        let mut observations = Vec::with_capacity(match_info.points.len());
        let mut residual_magnitudes = Vec::with_capacity(match_info.points.len());
        let mut world_bounds = (
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        );
        for &(key_source, key_target) in &match_info.points {
            let (source, target) = if reverse_key {
                (key_target, key_source)
            } else {
                (key_source, key_target)
            };
            let Some(predicted_target) = transformed_point(&relation_homography, source) else {
                continue;
            };
            // These are the same bidirectional patch correspondences that
            // entered Station_Relation.  Retain their geometric round-trip
            // error as the local evidence gate rather than inventing a new
            // correspondence set after the global pose solve.
            let round_trip_error_px = (predicted_target - target).norm();
            if !round_trip_error_px.is_finite()
                || round_trip_error_px > residual_warp::RESIDUAL_WARP_MAX_ROUND_TRIP_ERROR_PX
            {
                continue;
            }
            let Some(source_world) = transformed_point(left_to_world, source) else {
                continue;
            };
            let Some(target_world) = transformed_point(right_to_world, target) else {
                continue;
            };
            let displacement = target_world - source_world;
            let displacement_px = displacement.norm();
            if !displacement_px.is_finite()
                || displacement_px > residual_warp::RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX
            {
                continue;
            }
            world_bounds.0 = world_bounds.0.min(target_world.x);
            world_bounds.1 = world_bounds.1.min(target_world.y);
            world_bounds.2 = world_bounds.2.max(target_world.x);
            world_bounds.3 = world_bounds.3.max(target_world.y);
            residual_magnitudes.push(displacement_px);
            observations.push(residual_warp::WarpObservation::with_errors(
                stack_report::WorldPoint {
                    x: target_world.x,
                    y: target_world.y,
                },
                [displacement.x, displacement.y],
                round_trip_error_px,
                displacement_px,
                f64::NAN,
            ));
        }

        if observations.len() < residual_warp::RESIDUAL_WARP_MIN_VERIFIED_POINTS
            || residual_magnitudes.is_empty()
        {
            continue;
        }
        residual_magnitudes.sort_by(f64::total_cmp);
        let rank = (residual_magnitudes.len() * 95)
            .div_ceil(100)
            .max(1)
            .saturating_sub(1)
            .min(residual_magnitudes.len() - 1);
        let p95_px = residual_magnitudes[rank];
        if !p95_px.is_finite() || p95_px <= residual_warp::RESIDUAL_WARP_ENABLE_P95_PX {
            continue;
        }
        let (min_x, min_y, max_x, max_y) = world_bounds;
        if !min_x.is_finite()
            || !min_y.is_finite()
            || !max_x.is_finite()
            || !max_y.is_finite()
            || max_x < min_x
            || max_y < min_y
        {
            continue;
        }
        let overlap = residual_warp::OverlapResidual::new(
            relation.left,
            relation.right,
            stack_report::WorldRect {
                left: min_x,
                top: min_y,
                width: (max_x - min_x).max(1.0),
                height: (max_y - min_y).max(1.0),
            },
            p95_px,
        );
        let _ = model.add_observations(overlap, &observations);
    }
    model
}

pub(super) fn record_residual_warp_for_station_relations(
    tiles: &[ImageInfo],
    tile_homographies: &HashMap<usize, Matrix3<f64>>,
    relations: &HashMap<(usize, usize), MatchInfo>,
    accepted: &[stack_report::AcceptedStationRelationRecord],
) {
    let model =
        residual_model_from_station_relations(tiles, tile_homographies, relations, accepted);
    residual_warp::record_run_model(&model);
}
