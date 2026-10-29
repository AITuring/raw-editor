//! Boundary stroke pairing and large-boundary measurement.
//!
//! Pairs edges on both sides of an owner or station boundary, samples sources
//! through local station relations and measures the large boundaries.

use super::super::degradation;
use super::super::quality_gate;
use super::super::report::UnmeasurableRecord;
use super::super::report::WorldPoint;
use super::super::residual_warp::ResidualWarp;
use super::super::tone;
use image::GrayImage;
use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use std::collections::BTreeMap;

use super::{Criterion, QualitySource, classify_unmeasurable_reason, project};

/// Return the two-sided subpixel edge displacement at one ownership boundary.
/// `normal` is either horizontal `(1,0)` or vertical `(0,1)` and is derived
/// directly from the neighbouring ownership pixels, so no full label image is
/// needed.
pub(super) fn boundary_point_error(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    x: u32,
    y: u32,
    normal: (f64, f64),
) -> Result<f64, &'static str> {
    quality_gate::paired_boundary_edge_error_with_coverage(image, Some(coverage), x, y, normal)
        .map(|value| value.0)
}

#[derive(Debug, Clone)]
pub(super) struct BoundaryEdge {
    pub(super) offset: f64,
    pub(super) strength: f64,
    pub(super) gradient: (f64, f64),
    pub(super) pixel: (u32, u32),
    pub(super) owner: u16,
}

#[derive(Debug, Clone)]
pub(super) struct BoundaryPairEvidence {
    pub(super) world: (f64, f64),
    pub(super) error: f64,
    pub(super) strength: f64,
    pub(super) owner_path: String,
}

#[derive(Debug, Clone)]
pub(super) enum BoundaryCellObservation {
    Measured(BoundaryPairEvidence),
    Unmeasurable {
        world: (f64, f64),
        reason: &'static str,
    },
}

pub(super) fn boundary_edge_on_side_matching<F>(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    seam: (f64, f64),
    normal: (f64, f64),
    side: f64,
    distance: f64,
    owner_matches: F,
) -> Result<BoundaryEdge, &'static str>
where
    F: Fn(u16) -> bool,
{
    const FIXED_CROSS_SECTION_RADIUS_PX: i32 = 2;
    let (width, height) = image.dimensions();
    let tangent = (-normal.1, normal.0);
    let luminance = |pixel: Rgb<f32>| {
        0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
    };
    let mut candidates = Vec::new();
    for offset in -FIXED_CROSS_SECTION_RADIUS_PX..=FIXED_CROSS_SECTION_RADIUS_PX {
        let offset = f64::from(offset);
        let sx = seam.0 + side * normal.0 * distance + tangent.0 * offset;
        let sy = seam.1 + side * normal.1 * distance + tangent.1 * offset;
        let ix = sx.round() as i32;
        let iy = sy.round() as i32;
        if ix < 1 || iy < 1 || ix + 1 >= width as i32 || iy + 1 >= height as i32 {
            continue;
        }
        let index = iy as usize * width as usize + ix as usize;
        let Some(owner) = ownership.get(index).copied() else {
            continue;
        };
        if coverage.as_raw().get(index).copied().unwrap_or(0) == 0 || !owner_matches(owner) {
            continue;
        }
        let gx = luminance(*image.get_pixel((ix + 1) as u32, iy as u32))
            - luminance(*image.get_pixel((ix - 1) as u32, iy as u32));
        let gy = luminance(*image.get_pixel(ix as u32, (iy + 1) as u32))
            - luminance(*image.get_pixel(ix as u32, (iy - 1) as u32));
        let strength = gx.hypot(gy) * 0.5;
        if strength < 0.15 {
            continue;
        }
        let tangent_alignment = (gx * tangent.0 + gy * tangent.1).abs() / gx.hypot(gy);
        if tangent_alignment < 30.0f64.to_radians().cos() {
            continue;
        }
        candidates.push(BoundaryEdge {
            offset,
            strength,
            gradient: (gx, gy),
            pixel: (ix as u32, iy as u32),
            owner,
        });
    }
    candidates.sort_by(|left, right| {
        right
            .strength
            .total_cmp(&left.strength)
            .then_with(|| left.offset.total_cmp(&right.offset))
    });
    let Some(best) = candidates.first().cloned() else {
        return Err(degradation::BOUNDARY_LOW_CONTRAST);
    };
    if candidates.iter().skip(1).any(|candidate| {
        (candidate.offset - best.offset).abs() >= 3.0 && candidate.strength > best.strength * 0.8
    }) {
        return Err(degradation::BOUNDARY_STROKE_NOT_UNIQUE);
    }
    Ok(best)
}

pub(super) fn boundary_edge_on_side(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    seam: (f64, f64),
    normal: (f64, f64),
    side: f64,
    distance: f64,
    expected_owner: u16,
) -> Result<BoundaryEdge, &'static str> {
    boundary_edge_on_side_matching(
        image,
        coverage,
        ownership,
        seam,
        normal,
        side,
        distance,
        |owner| owner == expected_owner,
    )
}

pub(super) fn boundary_pair_error(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    seam: (f64, f64),
    normal: (f64, f64),
    left_owner: u16,
    right_owner: u16,
) -> Result<(BoundaryEdge, BoundaryEdge), &'static str> {
    let distance = 4.0;
    let left = boundary_edge_on_side(
        image, coverage, ownership, seam, normal, -1.0, distance, left_owner,
    )?;
    let right = boundary_edge_on_side(
        image,
        coverage,
        ownership,
        seam,
        normal,
        1.0,
        distance,
        right_owner,
    )?;
    let left_length = left.gradient.0.hypot(left.gradient.1);
    let right_length = right.gradient.0.hypot(right.gradient.1);
    let cosine = (left.gradient.0 * right.gradient.0 + left.gradient.1 * right.gradient.1)
        / (left_length * right_length);
    let angle_error = cosine.clamp(-1.0, 1.0).acos().to_degrees();
    if angle_error > 10.0 {
        return Err(degradation::BOUNDARY_ORIENTATION_MISMATCH);
    }
    Ok((left, right))
}

pub(super) fn boundary_pair_error_for_stations(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    seam: (f64, f64),
    normal: (f64, f64),
    left_station: usize,
    right_station: usize,
    owner_stations: &BTreeMap<u16, usize>,
) -> Result<(BoundaryEdge, BoundaryEdge), &'static str> {
    let distance = 4.0;
    let left = boundary_edge_on_side_matching(
        image,
        coverage,
        ownership,
        seam,
        normal,
        -1.0,
        distance,
        |owner| owner_stations.get(&owner).copied() == Some(left_station),
    )?;
    let right = boundary_edge_on_side_matching(
        image,
        coverage,
        ownership,
        seam,
        normal,
        1.0,
        distance,
        |owner| owner_stations.get(&owner).copied() == Some(right_station),
    )?;
    let left_length = left.gradient.0.hypot(left.gradient.1);
    let right_length = right.gradient.0.hypot(right.gradient.1);
    let cosine = (left.gradient.0 * right.gradient.0 + left.gradient.1 * right.gradient.1)
        / (left_length * right_length);
    let angle_error = cosine.clamp(-1.0, 1.0).acos().to_degrees();
    if angle_error > 10.0 {
        return Err(degradation::BOUNDARY_ORIENTATION_MISMATCH);
    }
    Ok((left, right))
}

pub(super) fn parse_boundary_owner_pair(owner_path: &str) -> Option<(u16, u16)> {
    let owners = owner_path.split_once(";owner:")?.1;
    let (left, right) = owners.split_once("/owner:")?;
    Some((left.parse().ok()?, right.split(';').next()?.parse().ok()?))
}

pub(super) fn parse_boundary_station_pair(owner_path: &str) -> Option<(usize, usize)> {
    let stations = owner_path.split_once("station:")?.1;
    let (left, right) = stations.split_once("/station:")?;
    // Quality_Gate emits the internal zero-based station indices directly.
    // The compositor's explicit environment selector is one-based, but the
    // report path is already normalized and must not be decremented again.
    let left = left.parse::<usize>().ok()?;
    let right = right.split(';').next()?.parse::<usize>().ok()?;
    (left != right).then_some((left, right))
}

/// Return the run-scoped local relation in the same direction as the
/// station pair written by the boundary criterion.  The relation is kept
/// outside the authoritative pose graph; it is only used for the bounded
/// raw->station reference closure below.
pub(super) fn local_station_relation_for_pair(
    first_station: usize,
    second_station: usize,
) -> Option<Matrix3<f64>> {
    let topology = tone::run_topology_snapshot()?;
    let first_id = topology
        .stations
        .iter()
        .find(|(_, station)| *station == first_station)
        .map(|(id, _)| *id)?;
    let second_id = topology
        .stations
        .iter()
        .find(|(_, station)| *station == second_station)
        .map(|(id, _)| *id)?;
    let (left, right, bits) =
        topology
            .seam_candidate_transforms
            .iter()
            .find(|(left, right, _)| {
                (*left == first_id && *right == second_id)
                    || (*left == second_id && *right == first_id)
            })?;
    let relation = Matrix3::from_row_slice(&bits.map(f64::from_bits));
    if !relation.iter().all(|value| value.is_finite()) {
        return None;
    }
    if *left == first_id && *right == second_id {
        Some(relation)
    } else {
        relation.try_inverse()
    }
}

/// Reproduce a seam-only local relation while sampling a Quality_Gate raw
/// owner.  The authoritative global pose remains unchanged: the first
/// station is mapped normally, then its station-layer coordinate is sent
/// through the measured first->second relation before the second raw source
/// is sampled.  This mirrors `station_seam_pair_sample` in the compositor,
/// including the native displacement field, and closes the same geometric
/// loop that produced the boundary evidence.
pub(super) fn sample_quality_source_at_world_with_station_relation(
    source: &Rgb32FImage,
    world: Point2<f64>,
    source_info: &QualitySource,
    first_source_info: &QualitySource,
    relation: &Matrix3<f64>,
    residual: &ResidualWarp,
) -> Result<Rgb<f32>, &'static str> {
    let first_warp = first_source_info
        .native_warp
        .as_ref()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let second_warp = source_info
        .native_warp
        .as_ref()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let first_inverse = first_warp
        .station_to_world
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let unwarped = residual.warp_inverse(world, first_source_info.geometry.station_id);
    let first_tile =
        project(&first_inverse, unwarped).ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let mapped = project(relation, first_tile).ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let delta = second_warp
        .displacement
        .as_ref()
        .map(|displacement| displacement.at(mapped.x, mapped.y))
        .unwrap_or([0.0, 0.0]);
    // `relation` is fitted in the Virtual_Tile coordinate system.  The
    // native warp stores the raw->world pose, while the Quality_Gate source
    // geometry also carries the station crop that maps that pose into the
    // tile-local plane.  Inverting only `source_to_station` drops that crop
    // translation and sends the relation sample to the wrong raw coordinates.
    // Compose the same raw->tile transform used by the ordinary native
    // reference path before sampling the second Source_RAW.
    let source_to_tile = second_warp
        .station_to_world
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?
        * source_info.geometry.tile_to_world
        * source_info.geometry.member_to_anchor;
    let source_inverse = source_to_tile
        .try_inverse()
        .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let mapped_source = project(
        &source_inverse,
        Point2::new(mapped.x + delta[0], mapped.y + delta[1]),
    )
    .ok_or(degradation::OWNER_SOURCE_UNDECODABLE)?;
    let max_x = f64::from(source.width().saturating_sub(1));
    let max_y = f64::from(source.height().saturating_sub(1));
    if mapped_source.x < 0.0
        || mapped_source.y < 0.0
        || mapped_source.x > max_x
        || mapped_source.y > max_y
    {
        return Err(degradation::OWNER_SOURCE_UNDECODABLE);
    }
    Ok(
        crate::panorama_utils::stitching::get_high_quality_interpolated_pixel(
            source,
            mapped_source.x,
            mapped_source.y,
        ),
    )
}

pub(super) fn parse_boundary_sample_normal(owner_path: &str) -> Option<((f64, f64), (f64, f64))> {
    let samples = owner_path.split_once(";samples:")?.1;
    let (left, right) = samples.split_once('|')?;
    let parse_point = |value: &str| {
        let (x, y) = value.split_once(',')?;
        Some((x.parse::<f64>().ok()?, y.parse::<f64>().ok()?))
    };
    let left = parse_point(left)?;
    let right = parse_point(right)?;
    let dx = right.0 - left.0;
    let dy = right.1 - left.1;
    if dx.abs() < 1.0e-9 && dy.abs() < 1.0e-9 {
        return None;
    }
    let normal = if dy.abs() >= dx.abs() {
        (0.0, dy.signum())
    } else {
        (dx.signum(), 0.0)
    };
    Some((((left.0 + right.0) * 0.5, (left.1 + right.1) * 0.5), normal))
}

pub(super) fn source_contains_world(
    source: &QualitySource,
    world: (f64, f64),
    residual: &ResidualWarp,
) -> bool {
    quality_gate::map_world_to_source(Point2::new(world.0, world.1), &source.geometry, residual)
        .ok()
        .is_some_and(|point| {
            point.x >= 0.0
                && point.y >= 0.0
                && point.x < f64::from(source.dimensions.0)
                && point.y < f64::from(source.dimensions.1)
        })
}

pub(super) fn boundary_rejection_priority(reason: &str) -> u8 {
    match reason {
        degradation::BOUNDARY_PROVENANCE_UNAVAILABLE => 4,
        degradation::BOUNDARY_ORIENTATION_MISMATCH => 3,
        degradation::BOUNDARY_STROKE_NOT_UNIQUE => 2,
        degradation::BOUNDARY_LOW_CONTRAST => 1,
        _ => 0,
    }
}

/// Measure only accepted inter-station hard seams. Each 64px world cell can
/// contribute at most one independently detected two-sided stroke pair; this
/// keeps the sample population spatially deterministic and prevents jagged
/// one-pixel owner transitions from inflating the criterion.
pub(super) fn measure_large_boundaries(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    origin: (f64, f64),
    sources: &[QualitySource],
    accepted_station_relations: &[(usize, usize)],
    residual: &ResidualWarp,
    criterion: &mut Criterion,
    unmeasurable: &mut Vec<UnmeasurableRecord>,
) {
    const SPATIAL_DEDUP_PX: u32 = 64;
    let (width, height) = coverage.dimensions();
    let width_usize = width as usize;
    let source_by_owner = sources
        .iter()
        .map(|source| (source.owner, source))
        .collect::<BTreeMap<_, _>>();
    // A station can contribute several raw owners after native tiling and
    // overlap arbitration.  Boundary evidence is a station-level geometric
    // criterion, so all raw owners belonging to the same station must be
    // eligible for edge detection.  Keep the selected raw owner on each edge
    // for provenance and source containment checks below.
    let station_by_owner = sources
        .iter()
        .map(|source| (source.owner, source.geometry.station_id))
        .collect::<BTreeMap<_, _>>();
    let accepted = accepted_station_relations
        .iter()
        .filter_map(|&(left, right)| (left != right).then_some((left.min(right), left.max(right))))
        .collect::<std::collections::BTreeSet<_>>();
    let mut cells = BTreeMap::<(usize, usize, u32, u32), BoundaryCellObservation>::new();
    let mut provenance_unavailable = false;
    let mut visit = |x: u32, y: u32, normal: (f64, f64), left_owner: u16, right_owner: u16| {
        if left_owner == 0 || right_owner == 0 || left_owner == right_owner {
            return;
        }
        let (Some(left_source), Some(right_source)) = (
            source_by_owner.get(&left_owner).copied(),
            source_by_owner.get(&right_owner).copied(),
        ) else {
            provenance_unavailable = true;
            return;
        };
        let station_pair = (
            left_source
                .geometry
                .station_id
                .min(right_source.geometry.station_id),
            left_source
                .geometry
                .station_id
                .max(right_source.geometry.station_id),
        );
        if station_pair.0 == station_pair.1 || !accepted.contains(&station_pair) {
            return;
        }
        let seam = (f64::from(x) + normal.0 * 0.5, f64::from(y) + normal.1 * 0.5);
        let world = (origin.0 + seam.0, origin.1 + seam.1);
        let seam_key = (
            station_pair.0,
            station_pair.1,
            x / SPATIAL_DEDUP_PX,
            y / SPATIAL_DEDUP_PX,
        );
        let (key, observation) = match boundary_pair_error_for_stations(
            image,
            coverage,
            ownership,
            seam,
            normal,
            left_source.geometry.station_id,
            right_source.geometry.station_id,
            &station_by_owner,
        ) {
            Ok((left_edge, right_edge)) => {
                let left_world = (
                    origin.0 + f64::from(left_edge.pixel.0),
                    origin.1 + f64::from(left_edge.pixel.1),
                );
                let right_world = (
                    origin.0 + f64::from(right_edge.pixel.0),
                    origin.1 + f64::from(right_edge.pixel.1),
                );
                let stroke_key = (
                    station_pair.0,
                    station_pair.1,
                    (left_edge.pixel.0 / 2 + right_edge.pixel.0 / 2) / SPATIAL_DEDUP_PX,
                    (left_edge.pixel.1 / 2 + right_edge.pixel.1 / 2) / SPATIAL_DEDUP_PX,
                );
                let selected_left_source = source_by_owner
                    .get(&left_edge.owner)
                    .copied()
                    .unwrap_or(left_source);
                let selected_right_source = source_by_owner
                    .get(&right_edge.owner)
                    .copied()
                    .unwrap_or(right_source);
                if !source_contains_world(selected_left_source, left_world, residual)
                    || !source_contains_world(selected_right_source, right_world, residual)
                {
                    (
                        stroke_key,
                        BoundaryCellObservation::Unmeasurable {
                            world,
                            reason: degradation::BOUNDARY_PROVENANCE_UNAVAILABLE,
                        },
                    )
                } else {
                    (
                        stroke_key,
                        BoundaryCellObservation::Measured(BoundaryPairEvidence {
                            world,
                            error: (left_edge.offset - right_edge.offset).abs(),
                            strength: left_edge.strength.min(right_edge.strength),
                            owner_path: format!(
                                "station:{}/station:{};owner:{}/owner:{};samples:{:.1},{:.1}|{:.1},{:.1}",
                                station_pair.0,
                                station_pair.1,
                                left_edge.owner,
                                right_edge.owner,
                                left_world.0,
                                left_world.1,
                                right_world.0,
                                right_world.1,
                            ),
                        }),
                    )
                }
            }
            Err(reason) => (
                seam_key,
                BoundaryCellObservation::Unmeasurable { world, reason },
            ),
        };
        match (cells.get(&key), &observation) {
            (
                Some(BoundaryCellObservation::Measured(current)),
                BoundaryCellObservation::Measured(candidate),
            ) if candidate.strength > current.strength => {
                cells.insert(key, observation);
            }
            (Some(BoundaryCellObservation::Measured(_)), _) => {}
            (
                Some(BoundaryCellObservation::Unmeasurable {
                    reason: current, ..
                }),
                BoundaryCellObservation::Unmeasurable {
                    reason: candidate, ..
                },
            ) if boundary_rejection_priority(candidate) > boundary_rejection_priority(current) => {
                cells.insert(key, observation);
            }
            (_, BoundaryCellObservation::Measured(_)) | (None, _) => {
                cells.insert(key, observation);
            }
            _ => {}
        }
    };
    for y in 0..height {
        for x in 0..width.saturating_sub(1) {
            let index = (y * width + x) as usize;
            if coverage.as_raw()[index] != 0
                && coverage.as_raw()[index + 1] != 0
                && ownership[index] != ownership[index + 1]
            {
                visit(x, y, (1.0, 0.0), ownership[index], ownership[index + 1]);
            }
        }
    }
    for y in 0..height.saturating_sub(1) {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if coverage.as_raw()[index] != 0
                && coverage.as_raw()[index + width_usize] != 0
                && ownership[index] != ownership[index + width_usize]
            {
                visit(
                    x,
                    y,
                    (0.0, 1.0),
                    ownership[index],
                    ownership[index + width_usize],
                );
            }
        }
    }

    let mut errors = cells
        .values()
        .filter_map(|observation| match observation {
            BoundaryCellObservation::Measured(evidence) => Some(evidence.error),
            BoundaryCellObservation::Unmeasurable { .. } => None,
        })
        .collect::<Vec<_>>();
    errors.sort_by(f64::total_cmp);
    let p95 = (!errors.is_empty()).then(|| {
        errors[(errors.len() * 95)
            .div_ceil(100)
            .saturating_sub(1)
            .min(errors.len() - 1)]
    });
    for observation in cells.into_values() {
        match observation {
            BoundaryCellObservation::Measured(evidence) => {
                let p95_failed = p95.is_some_and(|value| {
                    value > quality_gate::QUALITY_BOUNDARY_P95_MAX && evidence.error >= value
                });
                criterion.push(
                    evidence.world,
                    evidence.error,
                    &evidence.owner_path,
                    evidence.error <= quality_gate::QUALITY_BOUNDARY_MAX && !p95_failed,
                );
            }
            BoundaryCellObservation::Unmeasurable { world, reason } => {
                criterion.miss(reason);
                unmeasurable.push(UnmeasurableRecord {
                    criterion: criterion.name.to_string(),
                    world: WorldPoint {
                        x: world.0,
                        y: world.1,
                    },
                    reason: reason.to_string(),
                    category: classify_unmeasurable_reason(reason),
                });
            }
        }
    }
    if criterion.observed_count == 0 && criterion.unmeasurable == 0 {
        let reason = if provenance_unavailable {
            degradation::BOUNDARY_PROVENANCE_UNAVAILABLE
        } else {
            degradation::BOUNDARY_NO_PAIRABLE_EDGE
        };
        criterion.miss(reason);
        unmeasurable.push(UnmeasurableRecord {
            criterion: criterion.name.to_string(),
            world: WorldPoint {
                x: origin.0,
                y: origin.1,
            },
            reason: reason.to_string(),
            category: classify_unmeasurable_reason(reason),
        });
    }
}
