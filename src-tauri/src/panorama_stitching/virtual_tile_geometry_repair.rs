//! Rendering-prior repair and joint geometry refinement of Virtual_Tiles.
//!
//! Plans and applies bounded repairs of a cluster's rendering prior and refines
//! the native camera stations jointly.

use crate::panorama_utils::stack_pipeline::joint_warp;
use crate::panorama_utils::stack_pipeline::report as stack_report;
use crate::panorama_utils::stack_pipeline::residual_warp;
use crate::panorama_utils::stack_pipeline::topology;
use nalgebra::Matrix3;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
#[cfg(test)]
use std::path::Path;

use super::{
    CLOSURE_DIRECT_PAIR_P95_LIMIT_PX, CLOSURE_MAX_CORNER_CORRECTION_PX,
    CLOSURE_OVERLAP_SHORT_SIDE_CORRECTION_RATIO, Dsu, FocusCaptureGroup, FocusLayerWarp, ImageInfo,
    MatchInfo, STATION_RELATION_MIN_INLIERS, STATION_RELATION_SCALE_RATIO_MAX,
    STATION_RELATION_SCALE_RATIO_MIN, homography_preserves_focus_orientation,
    homography_scale_ratio, normalized_homography, projected_polygon_short_side,
    residual_distribution, station_pose_candidate_has_bounded_local_scale,
    station_relation_overlap_polygon, station_tree_edge_key, station_tree_position_key,
};

/// A densely verified prior-free relation between two stations whose rendering
/// priors disagree with it. It may only repair a prior; it is never a
/// Station_Relation.
#[derive(Clone, Copy, Debug)]
pub(super) struct VirtualTilePriorProposal {
    pub(super) left: usize,
    pub(super) right: usize,
    /// Measured `left tile -> right tile` relation in native tile pixels.
    pub(super) left_to_right: Matrix3<f64>,
    pub(super) inliers: usize,
}

/// One cluster member's rendering-prior repair: `world_correction` maps the
/// station's current world placement onto the placement the proposal measured
/// relative to an already consistent station.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct VirtualTilePriorRepair {
    pub(super) station: usize,
    pub(super) fixed_station: usize,
    pub(super) relation: (usize, usize),
    pub(super) relation_inliers: usize,
    pub(super) world_correction: Matrix3<f64>,
}

/// A rendering-prior repair may move a station only within the same bounded
/// geometry envelope used by closure corrections and the station scale gate.
/// Keep this decision pure so production and property tests share the exact
/// finite-value and inclusive-boundary semantics.
pub(super) fn virtual_tile_prior_repair_is_bounded(
    center_shift_px: f64,
    correction_scale_ratio: f64,
) -> bool {
    center_shift_px.is_finite()
        && center_shift_px >= 0.0
        && center_shift_px <= CLOSURE_MAX_CORNER_CORRECTION_PX
        && correction_scale_ratio.is_finite()
        && (STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX)
            .contains(&correction_scale_ratio)
}

/// Plan rendering-prior repairs from prior-free proposals.
///
/// Stations joined by authoritative (prior-seeded) relations form clusters
/// whose priors agree. The largest cluster is the reference (ties: smallest
/// canonical row/column key). Each proposal joining a fixed cluster to an
/// unfixed one — strongest first, ties by canonical pair key — moves the whole
/// unfixed cluster so that the proposal's measured relation holds; a cluster is
/// repaired at most once and a station therefore re-rendered at most once.
/// Stations no proposal reaches keep their prior (and stay disconnected).
pub(super) fn plan_virtual_tile_prior_repairs(
    station_count: usize,
    authoritative_pairs: &[(usize, usize)],
    proposals: &[VirtualTilePriorProposal],
    tile_to_world: &[Matrix3<f64>],
    station_topology: &topology::StationTopology,
) -> Vec<VirtualTilePriorRepair> {
    if station_count < 2 || tile_to_world.len() != station_count {
        return Vec::new();
    }
    let mut clusters = Dsu::new(station_count);
    for &(left, right) in authoritative_pairs {
        if left < station_count && right < station_count && left != right {
            clusters.union(left, right);
        }
    }
    let mut members = BTreeMap::<usize, Vec<usize>>::new();
    for station in 0..station_count {
        members
            .entry(clusters.find(station))
            .or_default()
            .push(station);
    }
    if members.len() < 2 {
        return Vec::new();
    }
    let cluster_key = |stations: &[usize]| {
        stations
            .iter()
            .map(|&station| station_tree_position_key(station_topology, station))
            .min()
    };
    let Some(reference) = members
        .iter()
        .max_by(|(_, left), (_, right)| {
            left.len()
                .cmp(&right.len())
                .then_with(|| cluster_key(right).cmp(&cluster_key(left)))
        })
        .map(|(&root, _)| root)
    else {
        return Vec::new();
    };
    let mut corrections = BTreeMap::from([(reference, Matrix3::identity())]);
    let mut remaining = proposals
        .iter()
        .copied()
        .filter(|proposal| {
            proposal.left < station_count
                && proposal.right < station_count
                && proposal.left != proposal.right
        })
        .collect::<Vec<_>>();
    let mut repairs = Vec::new();
    loop {
        let next = remaining
            .iter()
            .enumerate()
            .filter(|(_, proposal)| {
                corrections.contains_key(&clusters.find(proposal.left))
                    != corrections.contains_key(&clusters.find(proposal.right))
            })
            .max_by(|(_, left), (_, right)| {
                left.inliers.cmp(&right.inliers).then_with(|| {
                    station_tree_edge_key(station_topology, right.left, right.right).cmp(
                        &station_tree_edge_key(station_topology, left.left, left.right),
                    )
                })
            })
            .map(|(index, _)| index);
        let Some(index) = next else {
            break;
        };
        let proposal = remaining.remove(index);
        let left_root = clusters.find(proposal.left);
        let right_root = clusters.find(proposal.right);
        // tile pixel q of the unfixed station lands where the fixed station's
        // (already corrected) placement puts the measured corresponding pixel.
        let (fixed, unfixed, unfixed_root, fixed_to_unfixed_tile) =
            if corrections.contains_key(&left_root) {
                (
                    proposal.left,
                    proposal.right,
                    right_root,
                    proposal.left_to_right.try_inverse(),
                )
            } else {
                (
                    proposal.right,
                    proposal.left,
                    left_root,
                    Some(proposal.left_to_right),
                )
            };
        let Some(unfixed_to_fixed) = fixed_to_unfixed_tile else {
            continue;
        };
        let fixed_correction = corrections[&clusters.find(fixed)];
        let Some(unfixed_world_to_tile) = tile_to_world[unfixed].try_inverse() else {
            continue;
        };
        let Some(correction) = normalized_homography(
            &(fixed_correction * tile_to_world[fixed] * unfixed_to_fixed * unfixed_world_to_tile),
        ) else {
            continue;
        };
        if correction.try_inverse().is_none()
            || !virtual_tile_prior_repair_is_bounded(
                0.0,
                homography_scale_ratio(&correction).unwrap_or(f64::NAN),
            )
        {
            // An inadmissible strongest proposal must not mark this cluster as
            // repaired and suppress a weaker proposal that stays inside the
            // unchanged repair envelope. The render-time check below still
            // measures the actual centre shift before mutating source poses.
            continue;
        }
        corrections.insert(unfixed_root, correction);
        for &station in &members[&unfixed_root] {
            repairs.push(VirtualTilePriorRepair {
                station,
                fixed_station: fixed,
                relation: (proposal.left, proposal.right),
                relation_inliers: proposal.inliers,
                world_correction: correction,
            });
        }
    }
    repairs.sort_by_key(|repair| repair.station);
    repairs
}

/// Move one station's rendering prior: every member source pose (and its
/// focus-band poses) is composed with `world_correction`. Returns false, and
/// changes nothing, when a corrected pose would not be a valid planar pose.
pub(super) fn apply_virtual_tile_prior_repair(
    group: &[&ImageInfo],
    world_correction: &Matrix3<f64>,
    homographies: &mut HashMap<usize, Matrix3<f64>>,
    focus_warp: Option<&mut FocusLayerWarp>,
) -> bool {
    let mut corrected = Vec::with_capacity(group.len());
    for image in group {
        let Some(pose) = homographies.get(&image.id) else {
            return false;
        };
        let Some(pose) = normalized_homography(&(world_correction * pose)) else {
            return false;
        };
        if pose.try_inverse().is_none()
            || !homography_preserves_focus_orientation(&pose, image.dimensions())
        {
            return false;
        }
        corrected.push((image.id, pose));
    }
    for (id, pose) in corrected {
        homographies.insert(id, pose);
    }
    if let Some(warp) = focus_warp {
        for band in &mut warp.bands {
            for image in group {
                if let Some(pose) = band.homographies.get_mut(&image.id)
                    && let Some(corrected) = normalized_homography(&(world_correction * *pose))
                {
                    *pose = corrected;
                }
            }
        }
    }
    true
}

/// Refine native camera stations jointly, then validate the exact bounded
/// sampling field on points excluded from both global and local fitting.
/// Rejecting a relation never removes a source or disconnects the graph.
pub(super) fn refine_virtual_tile_joint_geometry(
    tiles: &[ImageInfo],
    matches: &HashMap<(usize, usize), MatchInfo>,
    initial: &HashMap<usize, Matrix3<f64>>,
    closure: &mut stack_report::ClosureReport,
    report: &mut stack_report::StationRelationsReport,
) -> Option<HashMap<usize, Matrix3<f64>>> {
    if tiles.len() < 3
        || report.accepted.len() < tiles.len()
        || closure.status == stack_report::ClosureStatus::Converged
    {
        return None;
    }
    let poses = tiles
        .iter()
        .map(|tile| initial.get(&tile.id).copied())
        .collect::<Option<Vec<_>>>()?;
    let dimensions = tiles.iter().map(ImageInfo::dimensions).collect::<Vec<_>>();
    let mut pairs = report
        .accepted
        .iter()
        .map(|r| (r.left.min(r.right), r.left.max(r.right)))
        .collect::<Vec<_>>();
    pairs.sort_unstable();
    pairs.dedup();
    // Coarse matches have no native reverse-check evidence. They retain the
    // established matrix-only path, including all small property fixtures.
    if pairs.iter().any(|pair| {
        matches
            .get(pair)
            .is_none_or(|m| m.native_verified_points.len() < STATION_RELATION_MIN_INLIERS)
    }) {
        return None;
    }
    let mut limits = vec![CLOSURE_MAX_CORNER_CORRECTION_PX; tiles.len()];
    for &(left, right) in &pairs {
        let relation = matches.get(&(left, right))?;
        for (station, source_side) in [(left, true), (right, false)] {
            let polygon = station_relation_overlap_polygon(
                &relation.homography,
                dimensions[left],
                dimensions[right],
                source_side,
            )?;
            let short = projected_polygon_short_side(&polygon, &poses[station])?;
            limits[station] =
                limits[station].min(short * CLOSURE_OVERLAP_SHORT_SIDE_CORRECTION_RATIO);
        }
    }
    let mut rejected = Vec::new();
    #[cfg(test)]
    if let Some(directory) = std::env::var_os("RAW_EDITOR_NATIVE_PROOF_DIAGNOSTICS_DIR")
        .or_else(|| std::env::var_os("RAW_EDITOR_VIRTUAL_TILE_DIAGNOSTICS_DIR"))
    {
        let records = report.accepted.iter().map(|r| {
            let key=(r.left.min(r.right),r.left.max(r.right));
            let relation=&matches[&key];
            serde_json::json!({"left":key.0,"right":key.1,"authoritative":true,
                "inliers":relation.inliers,"error":r.median_error_px,"scale":r.scale_ratio,
                "diagnostic":{"hull_support":r.spatial_support},
                "homography":relation.homography.as_slice(),
                "points":relation.points.iter().map(|(a,b)|[a.x,a.y,b.x,b.y]).collect::<Vec<_>>(),
                "native_verified_points":relation.native_verified_points.iter().map(|(a,b,e)|[a.x,a.y,b.x,b.y,*e]).collect::<Vec<_>>()})
        }).collect::<Vec<_>>();
        let snapshot = serde_json::json!({"poses":poses.iter().map(|h|(0..3).map(|r|(0..3).map(|c|h[(r,c)]).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>(),"dimensions":dimensions,"limits":limits,"records":records});
        let path = Path::new(&directory).join("joint-production-input.json");
        std::fs::create_dir_all(&directory).unwrap();
        if let Err(error) = std::fs::write(&path, serde_json::to_vec(&snapshot).unwrap()) {
            eprintln!(
                "Could not save native joint diagnostic {}: {error}",
                path.display()
            );
        }
    }
    // Use the complete native graph once to initialize the projective part.
    // Re-solving this gauge after every rejected local constraint can drift
    // through weakly observed corners. The final field is fitted and checked
    // only on retained relations; this proposal never establishes acceptance.
    let proposal_relations = pairs
        .iter()
        .map(|&(left, right)| joint_warp::Relation {
            left,
            right,
            points: matches[&(left, right)].points.clone(),
        })
        .collect::<Vec<_>>();
    let scale = dimensions.iter().map(|&(w, h)| w.max(h)).max()? as f64;
    let proposal = joint_warp::projective_proposal(&poses, &proposal_relations, 0, scale)?;
    for attempt in 0..8 {
        let relations = pairs
            .iter()
            .map(|&(left, right)| joint_warp::Relation {
                left,
                right,
                points: matches[&(left, right)].points.clone(),
            })
            .collect::<Vec<_>>();
        let fitted = joint_warp::fit(&proposal, &dimensions, &relations, 0)?;
        let mut verified = vec![Vec::new(); tiles.len()];
        for &(left, right) in &pairs {
            for &(a, b, error) in &matches[&(left, right)].native_verified_points {
                verified[left].push(joint_warp::VerifiedSource {
                    point: a,
                    round_trip_error_px: error,
                });
                verified[right].push(joint_warp::VerifiedSource {
                    point: b,
                    round_trip_error_px: error,
                });
            }
        }
        let mut bounded = joint_warp::materialize(
            &proposal,
            &fitted,
            &dimensions,
            &poses,
            &limits,
            &relations,
            &verified,
        )?;
        let record = |v: &joint_warp::PairValidation| stack_report::ClosurePairValidationRecord {
            left: v.left,
            right: v.right,
            count: v.count,
            median_px: v.median,
            p95_px: v.p95,
        };
        let failed = bounded
            .validation
            .iter()
            .filter(|v| !v.p95.is_finite() || v.p95 > CLOSURE_DIRECT_PAIR_P95_LIMIT_PX)
            .collect::<Vec<_>>();
        if failed.is_empty() {
            // A nonlinear correction does not waive the station-pose scale
            // or orientation gates used by the matrix-only solver.
            let station_groups = (0..tiles.len())
                .map(|station| FocusCaptureGroup {
                    members: vec![station],
                    anchor: station,
                    local_to_anchor: HashMap::from([(station, Matrix3::identity())]),
                })
                .collect::<Vec<_>>();
            if bounded
                .poses
                .iter()
                .zip(&dimensions)
                .any(|(pose, &size)| !homography_preserves_focus_orientation(pose, size))
                || !station_pose_candidate_has_bounded_local_scale(
                    tiles,
                    &station_groups,
                    &bounded.poses,
                    "native_joint_field",
                )
            {
                return None;
            }
            let distribution = residual_distribution(bounded.validation_errors.clone());
            let mut errors = bounded.validation_errors.clone();
            errors.sort_by(f64::total_cmp);
            closure.global_before_residual_median_px = Some(closure.residual_median_px);
            closure.global_before_residual_p95_px = Some(closure.residual_p95_px);
            closure.geometry_model = "bounded_station_field".to_string();
            closure.nonlinear_initialization_constraints = proposal_relations.len();
            closure.nonlinear_refit_passes = attempt + 1;
            closure.nonlinear_validation = bounded.validation.iter().map(record).collect();
            closure.rejected_nonlinear_constraints = rejected;
            closure.residual_median_px = distribution.median;
            closure.residual_p95_px = errors[(errors.len() * 95).div_ceil(100) - 1];
            closure.max_direct_pair_p95_px =
                bounded.validation.iter().map(|v| v.p95).fold(0.0, f64::max);
            closure.max_corner_correction_px = bounded.maximum_corner_motion;
            closure.status = stack_report::ClosureStatus::Converged;
            closure.fallback_reason = None;
            closure.offending_pair = None;
            closure.participating_constraints = pairs.len();
            closure.constraint_weights = bounded
                .validation
                .iter()
                .map(|v| stack_report::ClosureConstraintWeightRecord {
                    left: v.left,
                    right: v.right,
                    residual_px: v.median,
                    weight: 1.0,
                })
                .collect();
            report.accepted.retain(|r| {
                pairs
                    .binary_search(&(r.left.min(r.right), r.left.max(r.right)))
                    .is_ok()
            });
            report.rejected_count += closure.rejected_nonlinear_constraints.len();
            *report
                .rejected_by_reason
                .entry("native_joint_validation_p95".to_string())
                .or_default() += closure.rejected_nonlinear_constraints.len() as u64;
            for (station, tile) in tiles.iter().enumerate() {
                bounded.residual.register_station_alias(tile.id, station);
            }
            residual_warp::record_run_model(&bounded.residual);
            println!(
                "  - Bounded native station field: {} relations, independent P95 {:.3}px, maximum corner {:.3}px",
                pairs.len(),
                closure.max_direct_pair_p95_px,
                bounded.maximum_corner_motion
            );
            return Some(
                tiles
                    .iter()
                    .zip(bounded.poses)
                    .map(|(tile, pose)| (tile.id, pose))
                    .collect(),
            );
        }
        let failed_pairs = failed
            .iter()
            .map(|v| (v.left, v.right))
            .collect::<BTreeSet<_>>();
        let retained = pairs
            .iter()
            .copied()
            .filter(|pair| !failed_pairs.contains(pair))
            .collect::<Vec<_>>();
        let mut graph = Dsu::new(tiles.len());
        for &(left, right) in &retained {
            graph.union(left, right);
        }
        let root = graph.find(0);
        if retained.len() < tiles.len()
            || (0..tiles.len()).any(|station| graph.find(station) != root)
        {
            return None;
        }
        rejected.extend(failed.into_iter().map(record));
        pairs = retained;
    }
    None
}
