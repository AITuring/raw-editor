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
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use super::{Criterion, QualitySource, classify_unmeasurable_reason, project};

const BOUNDARY_DEFAULT_MIN_GRADIENT_STRENGTH: f64 = 0.15;
const BOUNDARY_CANDIDATE_DIAGNOSTIC_LIMIT: usize = 50_000;

#[derive(Debug, Serialize)]
struct BoundaryCandidateRecord {
    seam: [f64; 2],
    normal: [f64; 2],
    side: f64,
    distance: f64,
    offset: f64,
    pixel: Option<[i32; 2]>,
    owner: Option<u16>,
    coverage_valid: bool,
    owner_valid: bool,
    gradient: Option<[f64; 2]>,
    strength: Option<f64>,
    tangent_alignment: Option<f64>,
    accepted: bool,
    rejection_reason: Option<String>,
    selected: bool,
    final_rejection_reason: Option<String>,
}

#[derive(Debug, Default)]
struct BoundaryCandidateDiagnostics {
    records: Vec<BoundaryCandidateRecord>,
    truncated: bool,
}

static BOUNDARY_CANDIDATE_DIAGNOSTICS: OnceLock<Mutex<BoundaryCandidateDiagnostics>> =
    OnceLock::new();

fn boundary_candidate_diagnostics_store() -> Option<&'static Mutex<BoundaryCandidateDiagnostics>> {
    // The environment check is deliberately at the call boundary rather than
    // in the production sampling loop.  With no explicit directory the
    // normal path does not allocate a store or acquire a lock.
    std::env::var_os("RAW_EDITOR_FOCUS_BOUNDARY_CANDIDATE_DIR")?;
    Some(BOUNDARY_CANDIDATE_DIAGNOSTICS.get_or_init(|| Mutex::new(Default::default())))
}

fn record_boundary_candidate(record: BoundaryCandidateRecord) -> Option<usize> {
    let store = boundary_candidate_diagnostics_store()?;
    let mut store = store.lock().ok()?;
    if store.records.len() >= BOUNDARY_CANDIDATE_DIAGNOSTIC_LIMIT {
        store.truncated = true;
        return None;
    }
    let index = store.records.len();
    store.records.push(record);
    Some(index)
}

fn update_boundary_candidate(
    index: Option<usize>,
    update: impl FnOnce(&mut BoundaryCandidateRecord),
) {
    let Some(index) = index else {
        return;
    };
    let Some(store) = boundary_candidate_diagnostics_store() else {
        return;
    };
    let Ok(mut store) = store.lock() else {
        return;
    };
    if let Some(record) = store.records.get_mut(index) {
        update(record);
    }
}

pub(super) fn flush_boundary_candidate_diagnostics() -> Result<(), String> {
    let Some(directory) = std::env::var_os("RAW_EDITOR_FOCUS_BOUNDARY_CANDIDATE_DIR") else {
        return Ok(());
    };
    let Some(store) = BOUNDARY_CANDIDATE_DIAGNOSTICS.get() else {
        return Ok(());
    };
    let mut store = store
        .lock()
        .map_err(|_| "boundary candidate diagnostics lock poisoned".to_string())?;
    let records = std::mem::take(&mut store.records);
    let truncated = store.truncated;
    store.truncated = false;
    drop(store);

    let directory = PathBuf::from(directory);
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("failed to create {}: {error}", directory.display()))?;
    let path = directory.join("boundary-candidates.json");
    let file = std::fs::File::create(&path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    serde_json::to_writer_pretty(
        std::io::BufWriter::new(file),
        &serde_json::json!({
            "schema_version": 1,
            "record_count": records.len(),
            "truncated": truncated,
            "records": records,
        }),
    )
    .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

#[derive(Debug, Clone)]
pub(super) struct BoundaryEdge {
    pub(super) offset: f64,
    pub(super) strength: f64,
    pub(super) gradient: (f64, f64),
    pub(super) pixel: (u32, u32),
    pub(super) owner: u16,
    diagnostic_index: Option<usize>,
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

/// Merge by the actual stroke-cell key, which can differ from the originating
/// seam-cell key. The first measurable result (pass or fail) and any technical
/// rejection are final; only content-ineligible candidates permit continued
/// search. Strength is never a reason to replace a measured failure.
pub(super) fn boundary_observation_can_replace(
    current: &BoundaryCellObservation,
    candidate: &BoundaryCellObservation,
) -> bool {
    let BoundaryCellObservation::Unmeasurable {
        reason: current, ..
    } = current
    else {
        return false;
    };
    if classify_unmeasurable_reason(current)
        == super::super::report::UnmeasurableCategory::Technical
    {
        return false;
    }
    match candidate {
        BoundaryCellObservation::Measured(_) => true,
        BoundaryCellObservation::Unmeasurable { reason, .. } => {
            classify_unmeasurable_reason(reason)
                == super::super::report::UnmeasurableCategory::Technical
                || boundary_rejection_priority(reason) > boundary_rejection_priority(current)
        }
    }
}

pub(super) fn boundary_edge_on_side_matching<F>(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    seam: (f64, f64),
    normal: (f64, f64),
    side: f64,
    distance: f64,
    minimum_strength: f64,
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
            let _ = record_boundary_candidate(BoundaryCandidateRecord {
                seam: [seam.0, seam.1],
                normal: [normal.0, normal.1],
                side,
                distance,
                offset,
                pixel: None,
                owner: None,
                coverage_valid: false,
                owner_valid: false,
                gradient: None,
                strength: None,
                tangent_alignment: None,
                accepted: false,
                rejection_reason: Some("boundary_candidate_out_of_bounds".to_string()),
                selected: false,
                final_rejection_reason: None,
            });
            continue;
        }
        let index = iy as usize * width as usize + ix as usize;
        let Some(owner) = ownership.get(index).copied() else {
            let _ = record_boundary_candidate(BoundaryCandidateRecord {
                seam: [seam.0, seam.1],
                normal: [normal.0, normal.1],
                side,
                distance,
                offset,
                pixel: Some([ix, iy]),
                owner: None,
                coverage_valid: false,
                owner_valid: false,
                gradient: None,
                strength: None,
                tangent_alignment: None,
                accepted: false,
                rejection_reason: Some("boundary_candidate_owner_missing".to_string()),
                selected: false,
                final_rejection_reason: None,
            });
            continue;
        };
        // The central-difference gradient reads the four neighbours as well,
        // so every stencil pixel must be covered and owned by the same
        // expected source; an opposite-owner neighbour would otherwise
        // contaminate the measured edge.
        let stencil = [
            (ix, iy),
            (ix - 1, iy),
            (ix + 1, iy),
            (ix, iy - 1),
            (ix, iy + 1),
        ];
        let coverage_valid = stencil.iter().all(|&(x, y)| {
            let index = y as usize * width as usize + x as usize;
            coverage.as_raw().get(index).copied().unwrap_or(0) != 0
        });
        let owner_valid = stencil.iter().all(|&(x, y)| {
            let index = y as usize * width as usize + x as usize;
            ownership
                .get(index)
                .copied()
                .is_some_and(|neighbour| neighbour == owner && owner_matches(neighbour))
        });
        if !coverage_valid || !owner_valid {
            let rejection_reason = match (coverage_valid, owner_valid) {
                (false, false) => "boundary_candidate_coverage_and_owner_rejected",
                (false, true) => "boundary_candidate_coverage_rejected",
                (true, false) => "boundary_candidate_owner_rejected",
                (true, true) => unreachable!(),
            };
            let _ = record_boundary_candidate(BoundaryCandidateRecord {
                seam: [seam.0, seam.1],
                normal: [normal.0, normal.1],
                side,
                distance,
                offset,
                pixel: Some([ix, iy]),
                owner: Some(owner),
                coverage_valid,
                owner_valid,
                gradient: None,
                strength: None,
                tangent_alignment: None,
                accepted: false,
                rejection_reason: Some(rejection_reason.to_string()),
                selected: false,
                final_rejection_reason: None,
            });
            continue;
        }
        let gx = luminance(*image.get_pixel((ix + 1) as u32, iy as u32))
            - luminance(*image.get_pixel((ix - 1) as u32, iy as u32));
        let gy = luminance(*image.get_pixel(ix as u32, (iy + 1) as u32))
            - luminance(*image.get_pixel(ix as u32, (iy - 1) as u32));
        if !gx.is_finite() || !gy.is_finite() {
            let _ = record_boundary_candidate(BoundaryCandidateRecord {
                seam: [seam.0, seam.1],
                normal: [normal.0, normal.1],
                side,
                distance,
                offset,
                pixel: Some([ix, iy]),
                owner: Some(owner),
                coverage_valid,
                owner_valid,
                gradient: Some([gx, gy]),
                strength: None,
                tangent_alignment: None,
                accepted: false,
                rejection_reason: Some(degradation::DIAGNOSTICS_ROI_INVALID.to_string()),
                selected: false,
                final_rejection_reason: None,
            });
            return Err(degradation::DIAGNOSTICS_ROI_INVALID);
        }
        let strength = gx.hypot(gy) * 0.5;
        if strength < minimum_strength {
            let _ = record_boundary_candidate(BoundaryCandidateRecord {
                seam: [seam.0, seam.1],
                normal: [normal.0, normal.1],
                side,
                distance,
                offset,
                pixel: Some([ix, iy]),
                owner: Some(owner),
                coverage_valid,
                owner_valid,
                gradient: Some([gx, gy]),
                strength: Some(strength),
                tangent_alignment: None,
                accepted: false,
                rejection_reason: Some("boundary_candidate_low_strength".to_string()),
                selected: false,
                final_rejection_reason: None,
            });
            continue;
        }
        let tangent_alignment = (gx * tangent.0 + gy * tangent.1).abs() / gx.hypot(gy);
        if tangent_alignment < 30.0f64.to_radians().cos() {
            let _ = record_boundary_candidate(BoundaryCandidateRecord {
                seam: [seam.0, seam.1],
                normal: [normal.0, normal.1],
                side,
                distance,
                offset,
                pixel: Some([ix, iy]),
                owner: Some(owner),
                coverage_valid,
                owner_valid,
                gradient: Some([gx, gy]),
                strength: Some(strength),
                tangent_alignment: Some(tangent_alignment),
                accepted: false,
                rejection_reason: Some("boundary_candidate_tangent_mismatch".to_string()),
                selected: false,
                final_rejection_reason: None,
            });
            continue;
        }
        let diagnostic_index = record_boundary_candidate(BoundaryCandidateRecord {
            seam: [seam.0, seam.1],
            normal: [normal.0, normal.1],
            side,
            distance,
            offset,
            pixel: Some([ix, iy]),
            owner: Some(owner),
            coverage_valid,
            owner_valid,
            gradient: Some([gx, gy]),
            strength: Some(strength),
            tangent_alignment: Some(tangent_alignment),
            accepted: true,
            rejection_reason: None,
            selected: false,
            final_rejection_reason: None,
        });
        candidates.push(BoundaryEdge {
            offset,
            strength,
            gradient: (gx, gy),
            pixel: (ix as u32, iy as u32),
            owner,
            diagnostic_index,
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
        for candidate in &candidates {
            update_boundary_candidate(candidate.diagnostic_index, |record| {
                record.final_rejection_reason =
                    Some(degradation::BOUNDARY_STROKE_NOT_UNIQUE.to_string());
            });
        }
        return Err(degradation::BOUNDARY_STROKE_NOT_UNIQUE);
    }
    update_boundary_candidate(best.diagnostic_index, |record| {
        record.selected = true;
    });
    for candidate in candidates.iter().skip(1) {
        update_boundary_candidate(candidate.diagnostic_index, |record| {
            record.final_rejection_reason = Some("boundary_candidate_not_selected".to_string());
        });
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
        BOUNDARY_DEFAULT_MIN_GRADIENT_STRENGTH,
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

fn boundary_pair_error_for_stations_with_threshold(
    image: &Rgb32FImage,
    coverage: &GrayImage,
    ownership: &[u16],
    seam: (f64, f64),
    normal: (f64, f64),
    left_station: usize,
    right_station: usize,
    owner_stations: &BTreeMap<u16, usize>,
    minimum_strength: f64,
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
        minimum_strength,
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
        minimum_strength,
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
    acceptance_render_scale: f64,
    criterion_specific_pool: bool,
    criterion: &mut Criterion,
    unmeasurable: &mut Vec<UnmeasurableRecord>,
) {
    const SPATIAL_DEDUP_PX: u32 = 64;
    const MAX_BOUNDARY_CELLS: usize = 2_048;
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
    // Raw focus ownership can switch thousands of times inside one
    // Capture_Station.  Those transitions are not cross-station seams and
    // cannot contribute evidence; reject them before the expensive edge
    // resampling path below.  Unknown owners still pass through so the gate
    // reports provenance_unavailable rather than silently dropping evidence.
    let pairable_transition = |left_owner: u16, right_owner: u16| {
        let (Some(left_source), Some(right_source)) = (
            source_by_owner.get(&left_owner),
            source_by_owner.get(&right_owner),
        ) else {
            return true;
        };
        let pair = (
            left_source
                .geometry
                .station_id
                .min(right_source.geometry.station_id),
            left_source
                .geometry
                .station_id
                .max(right_source.geometry.station_id),
        );
        pair.0 != pair.1 && accepted.contains(&pair)
    };
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
        // The spatial key is the observable unit of this criterion.  Avoid
        // resampling every raw-owner transition that lands in the same 64px
        // cell; the old code deduplicated only after the expensive source
        // edge reconstruction, which made fragmented focus ownership turn a
        // linear scan into minutes of repeated work.
        if matches!(
            cells.get(&seam_key),
            Some(BoundaryCellObservation::Measured(_))
        ) {
            return;
        }
        // A technical provenance error is sticky. Searching another pixel in
        // the same cell must not hide it behind a later measurable stroke.
        if matches!(cells.get(&seam_key), Some(BoundaryCellObservation::Unmeasurable { reason, .. })
            if classify_unmeasurable_reason(reason) == super::super::report::UnmeasurableCategory::Technical)
        {
            return;
        }
        if cells.len() >= MAX_BOUNDARY_CELLS && !cells.contains_key(&seam_key) {
            return;
        }
        let minimum_strength = if acceptance_render_scale.is_finite() {
            (BOUNDARY_DEFAULT_MIN_GRADIENT_STRENGTH * acceptance_render_scale.clamp(0.5, 1.0))
                .max(0.06)
        } else {
            BOUNDARY_DEFAULT_MIN_GRADIENT_STRENGTH
        };
        let (key, observation) = match boundary_pair_error_for_stations_with_threshold(
            image,
            coverage,
            ownership,
            seam,
            normal,
            left_source.geometry.station_id,
            right_source.geometry.station_id,
            &station_by_owner,
            minimum_strength,
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
        if cells
            .get(&key)
            .is_none_or(|current| boundary_observation_can_replace(current, &observation))
        {
            cells.insert(key, observation);
        }
    };
    for y in 0..height {
        for x in 0..width.saturating_sub(1) {
            let index = (y * width + x) as usize;
            if coverage.as_raw()[index] != 0
                && coverage.as_raw()[index + 1] != 0
                && ownership[index] != ownership[index + 1]
                && pairable_transition(ownership[index], ownership[index + 1])
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
                && pairable_transition(ownership[index], ownership[index + width_usize])
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
                if criterion_specific_pool
                    && classify_unmeasurable_reason(reason)
                        == super::super::report::UnmeasurableCategory::ContentNotApplicable
                {
                    // Search continued across every transition in this cell.
                    // Only the final cell disposition is a rejected candidate,
                    // not each of the thousands of tested seam pixels.
                    criterion.reject_content(reason, world, 1);
                    continue;
                }
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
    if provenance_unavailable
        || (!criterion_specific_pool
            && criterion.observed_count == 0
            && criterion.unmeasurable == 0)
    {
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
    criterion.set_candidate_pool(
        criterion.observed_count + criterion.unmeasurable + criterion.rejected_content,
        criterion.observed_count,
        "station_boundary_streaming_candidates",
    );
}

#[cfg(test)]
mod boundary_candidate_diagnostics_tests {
    use super::BoundaryCandidateRecord;

    #[test]
    fn candidate_record_serializes_required_contract_fields() {
        let record = BoundaryCandidateRecord {
            seam: [10.0, 20.0],
            normal: [1.0, 0.0],
            side: -1.0,
            distance: 4.0,
            offset: 0.0,
            pixel: Some([6, 20]),
            owner: Some(7),
            coverage_valid: true,
            owner_valid: true,
            gradient: Some([0.0, 0.8]),
            strength: Some(0.4),
            tangent_alignment: Some(1.0),
            accepted: true,
            rejection_reason: None,
            selected: true,
            final_rejection_reason: None,
        };
        let value = serde_json::to_value(record).expect("candidate record must be JSON");
        for field in [
            "seam",
            "normal",
            "side",
            "distance",
            "offset",
            "pixel",
            "owner",
            "coverage_valid",
            "owner_valid",
            "gradient",
            "strength",
            "tangent_alignment",
            "accepted",
            "rejection_reason",
            "selected",
            "final_rejection_reason",
        ] {
            assert!(value.get(field).is_some(), "missing field {field}");
        }
        assert_eq!(value["selected"], true);
        assert_eq!(value["rejection_reason"], serde_json::Value::Null);
    }
}
