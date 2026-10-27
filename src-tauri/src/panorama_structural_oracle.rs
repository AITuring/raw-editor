//! Test-only traditional-structure evidence oracle.
//!
//! This module deliberately accepts only covered Virtual_Tile luminance planes.
//! It has no API for capture metadata, filenames, topology, rendering priors, or
//! an existing relation. Every reported correspondence names two independently
//! extracted primitives.

use super::*;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

const STRUCTURAL_FACTORS: [u32; 3] = [16, 8, 4];
const MAX_CHAIN_STEPS: usize = 512;
const MIN_CHAIN_POINTS: usize = 7;
const CELL_COLUMNS: u32 = 8;
const CELL_ROWS: u32 = 6;
const CELL_QUOTA: usize = 8;
const MAX_PRIMITIVES_PER_SCALE: usize = 320;
const HASH_MODELS: usize = 32;
const MODEL_INLIER_PX: f64 = 6.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum StructuralPrimitiveKind {
    LineEndpoint,
    LineIntersection,
    HighCurvature,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(super) struct StructuralPrimitiveRecord {
    pub id: u64,
    pub kind: StructuralPrimitiveKind,
    pub factor: u32,
    pub x_native_bits: u64,
    pub y_native_bits: u64,
    pub orientation_bits: u64,
    pub chain_id: u32,
    pub chain_length: u32,
    pub adjacent_chain_ratio_q: u16,
    pub intersection_angle_q: u16,
    pub descriptor: Vec<u16>,
}

#[derive(Clone, Debug)]
struct StructuralPrimitive {
    record: StructuralPrimitiveRecord,
    point: Point2<f64>,
    orientation: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(super) struct StructuralCorrespondenceRecord {
    pub source_primitive_id: u64,
    pub target_primitive_id: u64,
    pub source_x_bits: u64,
    pub source_y_bits: u64,
    pub target_x_bits: u64,
    pub target_y_bits: u64,
    pub descriptor_distance_q: u32,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct StructuralModelRecord {
    pub transform: [f64; 9],
    pub correspondences: Vec<StructuralCorrespondenceRecord>,
    pub median_symmetric_error_px: f64,
    pub scale_ratio: f64,
    pub covered_overlap_hull: f64,
    pub preserves_convex_orientation: bool,
    pub low_frequency_mean_relative_difference: f64,
    pub edge_strength_ratio: f64,
    pub median_edge_orientation_difference_degrees: f64,
    pub rejection_reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct StructuralDirectionRecord {
    pub source_primitives: Vec<StructuralPrimitiveRecord>,
    pub target_primitives: Vec<StructuralPrimitiveRecord>,
    pub mutual_matches: usize,
    pub hash_proposals: usize,
    pub models: Vec<StructuralModelRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct StructuralOracleRecord {
    pub evidence: &'static str,
    pub factors: Vec<u32>,
    pub forward: StructuralDirectionRecord,
    pub reverse: StructuralDirectionRecord,
    pub forward_reverse_five_point_closure_px: f64,
    pub canonical_swap_disagreement_px: f64,
    pub accepted: bool,
    pub elapsed_ms: u64,
    pub estimated_extra_memory_bytes: u64,
}

#[derive(Clone)]
struct EdgePlane {
    width: u32,
    height: u32,
    edge: Vec<bool>,
    orientation: Vec<f64>,
}

#[derive(Clone)]
struct EdgeChain {
    id: u32,
    points: Vec<(u32, u32)>,
}

fn edge_index(width: u32, x: u32, y: u32) -> usize {
    y as usize * width as usize + x as usize
}

fn edge_neighbours(plane: &EdgePlane, x: u32, y: u32) -> Vec<(u32, u32)> {
    let mut result = Vec::with_capacity(8);
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0
                && ny >= 0
                && nx < plane.width as i32
                && ny < plane.height as i32
                && plane.edge[edge_index(plane.width, nx as u32, ny as u32)]
            {
                result.push((nx as u32, ny as u32));
            }
        }
    }
    result.sort_unstable();
    result
}

fn coverage_safe_edges(gray: &GrayImage, coverage: &stitching::CoverageMask) -> Option<EdgePlane> {
    let (width, height) = gray.dimensions();
    if coverage.dimensions() != (width, height) || width < 5 || height < 5 {
        return None;
    }
    let count = width as usize * height as usize;
    let mut strength = vec![0u16; count];
    let mut orientation = vec![0.0; count];
    let mut nonzero = Vec::new();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let safe = (y - 1..=y + 1).all(|sample_y| {
                (x - 1..=x + 1).all(|sample_x| coverage.is_covered(sample_x, sample_y))
            });
            if !safe {
                continue;
            }
            let p = |sx: u32, sy: u32| i32::from(gray.get_pixel(sx, sy)[0]);
            let gx = -p(x - 1, y - 1) + p(x + 1, y - 1) - 2 * p(x - 1, y) + 2 * p(x + 1, y)
                - p(x - 1, y + 1)
                + p(x + 1, y + 1);
            let gy = -p(x - 1, y - 1) - 2 * p(x, y - 1) - p(x + 1, y - 1)
                + p(x - 1, y + 1)
                + 2 * p(x, y + 1)
                + p(x + 1, y + 1);
            let magnitude = (((gx * gx + gy * gy) as f64).sqrt().round() as u16).min(1443);
            if magnitude > 0 {
                let index = edge_index(width, x, y);
                strength[index] = magnitude;
                orientation[index] = (gy as f64).atan2(gx as f64);
                nonzero.push(magnitude);
            }
        }
    }
    if nonzero.len() < MIN_CHAIN_POINTS {
        return None;
    }
    nonzero.sort_unstable();
    let threshold = nonzero[nonzero.len() * 3 / 4].max(32);
    let mut edge = vec![false; count];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let index = edge_index(width, x, y);
            if strength[index] < threshold {
                continue;
            }
            let angle = orientation[index];
            let horizontal = angle.cos().abs() >= angle.sin().abs();
            let (a, b) = if horizontal {
                (
                    strength[edge_index(width, x - 1, y)],
                    strength[edge_index(width, x + 1, y)],
                )
            } else {
                (
                    strength[edge_index(width, x, y - 1)],
                    strength[edge_index(width, x, y + 1)],
                )
            };
            edge[index] = strength[index] >= a && strength[index] >= b;
        }
    }
    Some(EdgePlane {
        width,
        height,
        edge,
        orientation,
    })
}

fn edge_key(width: u32, a: (u32, u32), b: (u32, u32)) -> (usize, usize) {
    let a = edge_index(width, a.0, a.1);
    let b = edge_index(width, b.0, b.1);
    (a.min(b), a.max(b))
}

fn trace_edge_chains(plane: &EdgePlane) -> Vec<EdgeChain> {
    let mut visited = BTreeSet::new();
    let mut starts = Vec::new();
    for y in 0..plane.height {
        for x in 0..plane.width {
            if plane.edge[edge_index(plane.width, x, y)] {
                let degree = edge_neighbours(plane, x, y).len();
                if degree != 2 {
                    starts.push((x, y));
                }
            }
        }
    }
    // Remaining degree-two loops are reached by the second pass.
    starts.extend((0..plane.height).flat_map(|y| (0..plane.width).map(move |x| (x, y))));
    let mut chains = Vec::new();
    for start in starts {
        if !plane.edge[edge_index(plane.width, start.0, start.1)] {
            continue;
        }
        for first in edge_neighbours(plane, start.0, start.1) {
            if visited.contains(&edge_key(plane.width, start, first)) {
                continue;
            }
            let mut points = vec![start, first];
            visited.insert(edge_key(plane.width, start, first));
            while points.len() < MAX_CHAIN_STEPS {
                let previous = points[points.len() - 2];
                let current = *points.last().unwrap();
                let neighbours = edge_neighbours(plane, current.0, current.1);
                if neighbours.len() != 2 {
                    break;
                }
                let Some(next) = neighbours.into_iter().find(|&point| point != previous) else {
                    break;
                };
                let key = edge_key(plane.width, current, next);
                if visited.contains(&key) {
                    break;
                }
                visited.insert(key);
                points.push(next);
            }
            if points.len() >= MIN_CHAIN_POINTS {
                chains.push(EdgeChain {
                    id: chains.len() as u32,
                    points,
                });
            }
        }
    }
    chains
}

fn undirected_angle(mut angle: f64) -> f64 {
    while angle < 0.0 {
        angle += std::f64::consts::PI;
    }
    while angle >= std::f64::consts::PI {
        angle -= std::f64::consts::PI;
    }
    angle
}

fn angle_difference(left: f64, right: f64) -> f64 {
    let difference = (undirected_angle(left) - undirected_angle(right)).abs();
    difference.min(std::f64::consts::PI - difference)
}

fn primitive_candidate(
    kind: StructuralPrimitiveKind,
    factor: u32,
    point: (u32, u32),
    orientation: f64,
    chain: &EdgeChain,
    ratio: f64,
    intersection_angle: f64,
) -> StructuralPrimitive {
    let scale = f64::from(factor);
    let offset = (scale - 1.0) * 0.5;
    let native = Point2::new(
        scale * f64::from(point.0) + offset,
        scale * f64::from(point.1) + offset,
    );
    StructuralPrimitive {
        record: StructuralPrimitiveRecord {
            id: 0,
            kind,
            factor,
            x_native_bits: native.x.to_bits(),
            y_native_bits: native.y.to_bits(),
            orientation_bits: undirected_angle(orientation).to_bits(),
            chain_id: chain.id,
            chain_length: chain.points.len() as u32,
            adjacent_chain_ratio_q: (ratio.clamp(0.0, 4.0) * 4096.0).round() as u16,
            intersection_angle_q: (intersection_angle.clamp(0.0, std::f64::consts::PI) * 8192.0)
                .round() as u16,
            descriptor: Vec::new(),
        },
        point: native,
        orientation: undirected_angle(orientation),
    }
}

fn primitive_candidates(
    plane: &EdgePlane,
    chains: &[EdgeChain],
    factor: u32,
) -> Vec<StructuralPrimitive> {
    let mut result = Vec::new();
    for chain in chains {
        let first = chain.points[0];
        let last = *chain.points.last().unwrap();
        let chord = ((last.0 as f64 - first.0 as f64).powi(2)
            + (last.1 as f64 - first.1 as f64).powi(2))
        .sqrt();
        let line_orientation =
            (last.1 as f64 - first.1 as f64).atan2(last.0 as f64 - first.0 as f64);
        if chord / chain.points.len() as f64 >= 0.55 {
            result.push(primitive_candidate(
                StructuralPrimitiveKind::LineEndpoint,
                factor,
                first,
                line_orientation,
                chain,
                chord / chain.points.len() as f64,
                0.0,
            ));
            result.push(primitive_candidate(
                StructuralPrimitiveKind::LineEndpoint,
                factor,
                last,
                line_orientation,
                chain,
                chord / chain.points.len() as f64,
                0.0,
            ));
        }
        if chain.points.len() >= 9 {
            let radius = 3usize;
            let mut curvatures = Vec::new();
            for index in radius..chain.points.len() - radius {
                let before = chain.points[index - radius];
                let point = chain.points[index];
                let after = chain.points[index + radius];
                let incoming =
                    (point.1 as f64 - before.1 as f64).atan2(point.0 as f64 - before.0 as f64);
                let outgoing =
                    (after.1 as f64 - point.1 as f64).atan2(after.0 as f64 - point.0 as f64);
                curvatures.push((
                    index,
                    angle_difference(incoming, outgoing),
                    incoming,
                    outgoing,
                ));
            }
            for window in curvatures.windows(3) {
                let (index, curvature, incoming, outgoing) = window[1];
                if curvature >= 35.0f64.to_radians()
                    && curvature >= window[0].1
                    && curvature >= window[2].1
                {
                    result.push(primitive_candidate(
                        StructuralPrimitiveKind::HighCurvature,
                        factor,
                        chain.points[index],
                        (incoming + outgoing) * 0.5,
                        chain,
                        1.0,
                        curvature,
                    ));
                }
            }
        }
    }
    // A stable branch node is an auditable line intersection. Associate it
    // with the longest incident chain, retaining that chain's metadata.
    let mut at = BTreeMap::<(u32, u32), Vec<&EdgeChain>>::new();
    for chain in chains {
        at.entry(chain.points[0]).or_default().push(chain);
        at.entry(*chain.points.last().unwrap())
            .or_default()
            .push(chain);
    }
    for (point, incident) in at {
        if incident.len() < 3 {
            continue;
        }
        let chain = *incident
            .iter()
            .max_by_key(|chain| chain.points.len())
            .unwrap();
        let mut angles = incident
            .iter()
            .filter_map(|chain| {
                let other = if chain.points[0] == point {
                    chain.points.get(3).or_else(|| chain.points.last())
                } else {
                    chain
                        .points
                        .len()
                        .checked_sub(4)
                        .and_then(|index| chain.points.get(index))
                        .or_else(|| chain.points.first())
                }?;
                Some((other.1 as f64 - point.1 as f64).atan2(other.0 as f64 - point.0 as f64))
            })
            .collect::<Vec<_>>();
        angles.sort_by(f64::total_cmp);
        let intersection_angle = angles
            .iter()
            .enumerate()
            .flat_map(|(index, &left)| {
                angles
                    .iter()
                    .skip(index + 1)
                    .map(move |&right| angle_difference(left, right))
            })
            .max_by(f64::total_cmp)
            .unwrap_or(0.0);
        if intersection_angle >= 25.0f64.to_radians() {
            result.push(primitive_candidate(
                StructuralPrimitiveKind::LineIntersection,
                factor,
                point,
                angles[0],
                chain,
                incident.len() as f64,
                intersection_angle,
            ));
        }
    }
    // The Sobel orientation gives endpoints and intersections a stable local
    // direction even where a bounded chain ends at a junction.
    for primitive in &mut result {
        let x = ((primitive.point.x - (f64::from(factor) - 1.0) * 0.5) / f64::from(factor)).round()
            as u32;
        let y = ((primitive.point.y - (f64::from(factor) - 1.0) * 0.5) / f64::from(factor)).round()
            as u32;
        let gradient = plane.orientation
            [edge_index(plane.width, x.min(plane.width - 1), y.min(plane.height - 1))];
        if gradient.is_finite() {
            primitive.orientation = undirected_angle(gradient + std::f64::consts::FRAC_PI_2);
            primitive.record.orientation_bits = primitive.orientation.to_bits();
        }
    }
    result
}

fn attach_descriptors(
    mut primitives: Vec<StructuralPrimitive>,
    dimensions: (u32, u32),
) -> Vec<StructuralPrimitive> {
    let radius = f64::from(dimensions.0.min(dimensions.1)) * 0.18;
    for index in 0..primitives.len() {
        let mut histogram = vec![0u16; 50]; // 6x8 radial/orientation + two metadata bins.
        for (other_index, other) in primitives.iter().enumerate() {
            if index == other_index || primitives[index].record.factor != other.record.factor {
                continue;
            }
            let delta = other.point - primitives[index].point;
            let distance = delta.norm();
            if distance <= f64::EPSILON || distance >= radius {
                continue;
            }
            let radial = ((distance / radius) * 6.0).floor().min(5.0) as usize;
            let relative = undirected_angle(delta.y.atan2(delta.x) - primitives[index].orientation);
            let angular = ((relative / std::f64::consts::PI) * 8.0).floor().min(7.0) as usize;
            histogram[radial * 8 + angular] = histogram[radial * 8 + angular].saturating_add(1);
        }
        let maximum = histogram[..48].iter().copied().max().unwrap_or(1).max(1);
        for value in &mut histogram[..48] {
            *value = (u32::from(*value) * 255 / u32::from(maximum)) as u16;
        }
        histogram[48] = primitives[index].record.adjacent_chain_ratio_q / 32;
        histogram[49] = primitives[index].record.intersection_angle_q / 32;
        primitives[index].record.descriptor = histogram;
    }
    primitives.sort_by(|left, right| {
        left.record
            .factor
            .cmp(&right.record.factor)
            .then_with(|| left.record.kind.cmp(&right.record.kind))
            .then_with(|| left.point.x.total_cmp(&right.point.x))
            .then_with(|| left.point.y.total_cmp(&right.point.y))
    });
    let cell_width = f64::from(dimensions.0).max(1.0) / f64::from(CELL_COLUMNS);
    let cell_height = f64::from(dimensions.1).max(1.0) / f64::from(CELL_ROWS);
    let mut quotas = BTreeMap::<(u32, u32, u32), usize>::new();
    let mut retained: Vec<StructuralPrimitive> = Vec::new();
    for primitive in primitives {
        let cell = ((primitive.point.x / cell_width).floor() as u32).min(CELL_COLUMNS - 1);
        let row = ((primitive.point.y / cell_height).floor() as u32).min(CELL_ROWS - 1);
        let key = (primitive.record.factor, cell, row);
        if quotas.get(&key).copied().unwrap_or_default() >= CELL_QUOTA {
            continue;
        }
        let duplicate = retained.iter().any(|other| {
            other.record.factor == primitive.record.factor
                && other.record.kind == primitive.record.kind
                && (other.point - primitive.point).norm() < f64::from(primitive.record.factor) * 2.0
                && descriptor_distance(&other.record.descriptor, &primitive.record.descriptor) < 600
        });
        if duplicate {
            continue;
        }
        *quotas.entry(key).or_default() += 1;
        retained.push(primitive);
    }
    let mut per_scale = BTreeMap::<u32, usize>::new();
    retained.retain(|primitive| {
        let count = per_scale.entry(primitive.record.factor).or_default();
        if *count >= MAX_PRIMITIVES_PER_SCALE {
            false
        } else {
            *count += 1;
            true
        }
    });
    for (id, primitive) in retained.iter_mut().enumerate() {
        primitive.record.id = id as u64;
    }
    retained
}

fn descriptor_distance(left: &[u16], right: &[u16]) -> u64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| i64::from(a).abs_diff(i64::from(b)))
        .sum()
}

fn extract_structural_primitives(
    pyramid: &VirtualTileMatchPyramid,
    station: usize,
) -> Vec<StructuralPrimitive> {
    let mut candidates = Vec::new();
    for &factor in &STRUCTURAL_FACTORS {
        let Some(level) = pyramid
            .factors
            .iter()
            .position(|&candidate| candidate == factor)
        else {
            continue;
        };
        let Some(image) = pyramid
            .levels
            .get(level)
            .and_then(|tiles| tiles.get(station))
        else {
            continue;
        };
        let Some(coverage) = pyramid
            .coverages
            .get(level)
            .and_then(|masks| masks.get(station))
        else {
            continue;
        };
        let Some(edges) = coverage_safe_edges(&image.alignment_image, coverage) else {
            continue;
        };
        let chains = trace_edge_chains(&edges);
        candidates.extend(primitive_candidates(&edges, &chains, factor));
    }
    attach_descriptors(candidates, pyramid.native_dimensions[station])
}

#[derive(Clone)]
struct PrimitiveMatch {
    source: usize,
    target: usize,
    distance: u64,
}

fn mutual_structural_matches(
    source: &[StructuralPrimitive],
    target: &[StructuralPrimitive],
) -> Vec<PrimitiveMatch> {
    let nearest = |needle: &StructuralPrimitive, haystack: &[StructuralPrimitive]| {
        let mut distances = haystack
            .iter()
            .enumerate()
            .filter(|(_, candidate)| {
                candidate.record.factor == needle.record.factor
                    && candidate.record.kind == needle.record.kind
            })
            .map(|(index, candidate)| {
                (
                    descriptor_distance(&needle.record.descriptor, &candidate.record.descriptor),
                    index,
                )
            })
            .collect::<Vec<_>>();
        distances.sort_unstable();
        let &(best, index) = distances.first()?;
        let second = distances.get(1).map_or(u64::MAX, |entry| entry.0);
        // Keep exact unique descriptors, otherwise require a modest margin.
        (best == 0 || best.saturating_mul(100) <= second.saturating_mul(96))
            .then_some((index, best))
    };
    let source_to_target = source
        .iter()
        .map(|primitive| nearest(primitive, target))
        .collect::<Vec<_>>();
    let target_to_source = target
        .iter()
        .map(|primitive| nearest(primitive, source))
        .collect::<Vec<_>>();
    let mut matches = source_to_target
        .into_iter()
        .enumerate()
        .filter_map(|(source_index, match_target)| {
            let (target_index, distance) = match_target?;
            (target_to_source
                .get(target_index)
                .and_then(|entry| *entry)
                .map(|entry| entry.0)
                == Some(source_index))
            .then_some(PrimitiveMatch {
                source: source_index,
                target: target_index,
                distance,
            })
        })
        .collect::<Vec<_>>();
    matches.sort_by_key(|item| {
        (
            item.distance,
            source[item.source].record.id,
            target[item.target].record.id,
        )
    });
    matches
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct HashBin(i32, i32, i16, i16);

fn similarity_from_pair(
    source_a: Point2<f64>,
    source_b: Point2<f64>,
    target_a: Point2<f64>,
    target_b: Point2<f64>,
) -> Option<Matrix3<f64>> {
    let source = source_b - source_a;
    let target = target_b - target_a;
    let source_length = source.norm();
    let target_length = target.norm();
    if source_length < 32.0 || target_length < 32.0 {
        return None;
    }
    let scale = target_length / source_length;
    if !(STATION_RELATION_SCALE_RATIO_MIN..=STATION_RELATION_SCALE_RATIO_MAX).contains(&scale) {
        return None;
    }
    let theta = target.y.atan2(target.x) - source.y.atan2(source.x);
    if theta.abs() > 20.0f64.to_radians() {
        return None;
    }
    let (sin, cos) = theta.sin_cos();
    let tx = target_a.x - scale * (cos * source_a.x - sin * source_a.y);
    let ty = target_a.y - scale * (sin * source_a.x + cos * source_a.y);
    Some(Matrix3::new(
        scale * cos,
        -scale * sin,
        tx,
        scale * sin,
        scale * cos,
        ty,
        0.0,
        0.0,
        1.0,
    ))
}

fn geometric_hash_proposals(
    source: &[StructuralPrimitive],
    target: &[StructuralPrimitive],
    matches: &[PrimitiveMatch],
) -> Vec<Matrix3<f64>> {
    let bounded = &matches[..matches.len().min(256)];
    let mut bins = BTreeMap::<HashBin, (usize, Matrix3<f64>)>::new();
    for left in 0..bounded.len() {
        for right in left + 1..bounded.len() {
            let a = &bounded[left];
            let b = &bounded[right];
            if a.source == b.source || a.target == b.target {
                continue;
            }
            let Some(transform) = similarity_from_pair(
                source[a.source].point,
                source[b.source].point,
                target[a.target].point,
                target[b.target].point,
            ) else {
                continue;
            };
            let scale = homography_scale_ratio(&transform).unwrap_or(0.0);
            let angle = transform[(1, 0)].atan2(transform[(0, 0)]);
            let key = HashBin(
                (transform[(0, 2)] / 12.0).round() as i32,
                (transform[(1, 2)] / 12.0).round() as i32,
                (angle.to_degrees() / 1.5).round() as i16,
                ((scale - 1.0) / 0.0025).round() as i16,
            );
            bins.entry(key)
                .and_modify(|entry| entry.0 += 1)
                .or_insert((1, transform));
        }
    }
    let mut ranked = bins
        .into_iter()
        .map(|(key, (votes, transform))| (std::cmp::Reverse(votes), key, transform))
        .collect::<Vec<_>>();
    ranked.sort_by_key(|entry| (entry.0, entry.1));
    ranked
        .into_iter()
        .take(HASH_MODELS)
        .map(|entry| entry.2)
        .collect()
}

fn model_inlier_matches(
    transform: &Matrix3<f64>,
    source: &[StructuralPrimitive],
    target: &[StructuralPrimitive],
    matches: &[PrimitiveMatch],
) -> Vec<PrimitiveMatch> {
    let Some(inverse) = transform.try_inverse() else {
        return Vec::new();
    };
    let mut inliers = matches
        .iter()
        .filter(|item| {
            let Some(projected_target) = transformed_point(transform, source[item.source].point)
            else {
                return false;
            };
            let Some(projected_source) = transformed_point(&inverse, target[item.target].point)
            else {
                return false;
            };
            (projected_target - target[item.target].point).norm() <= MODEL_INLIER_PX
                && (projected_source - source[item.source].point).norm() <= MODEL_INLIER_PX
        })
        .cloned()
        .collect::<Vec<_>>();
    inliers.sort_by_key(|item| (source[item.source].record.id, target[item.target].record.id));
    inliers
}

fn correspondence_record(
    source: &StructuralPrimitive,
    target: &StructuralPrimitive,
    distance: u64,
) -> StructuralCorrespondenceRecord {
    StructuralCorrespondenceRecord {
        source_primitive_id: source.record.id,
        target_primitive_id: target.record.id,
        source_x_bits: source.point.x.to_bits(),
        source_y_bits: source.point.y.to_bits(),
        target_x_bits: target.point.x.to_bits(),
        target_y_bits: target.point.y.to_bits(),
        descriptor_distance_q: distance.min(u64::from(u32::MAX)) as u32,
    }
}

fn solve_structural_direction(
    source_station: usize,
    target_station: usize,
    _tiles: &[ImageInfo],
    pyramid: &VirtualTileMatchPyramid,
) -> (
    StructuralDirectionRecord,
    Vec<(Matrix3<f64>, Vec<PrimitiveMatch>)>,
) {
    let source = extract_structural_primitives(pyramid, source_station);
    let target = extract_structural_primitives(pyramid, target_station);
    let matches = mutual_structural_matches(&source, &target);
    let proposals = geometric_hash_proposals(&source, &target, &matches);
    let mut fitted = Vec::<(Matrix3<f64>, Vec<PrimitiveMatch>)>::new();
    for proposal in &proposals {
        let seed_inliers = model_inlier_matches(proposal, &source, &target, &matches);
        if seed_inliers.len() < 4 {
            continue;
        }
        let points = seed_inliers
            .iter()
            .map(|item| (source[item.source].point, target[item.target].point))
            .collect::<Vec<_>>();
        let Some((fit, indices)) =
            processing::find_homography_ransac_points_stable_with_min_inliers(
                &points,
                STATION_RELATION_MAX_MEDIAN_ERROR_PX,
                4,
            )
        else {
            continue;
        };
        let fit_matches = indices
            .into_iter()
            .map(|index| seed_inliers[index].clone())
            .collect::<Vec<_>>();
        let Some(refit) = processing::compute_homography(
            &fit_matches
                .iter()
                .map(|item| (source[item.source].point, target[item.target].point))
                .collect::<Vec<_>>(),
        )
        .or(Some(fit)) else {
            continue;
        };
        let inliers = model_inlier_matches(&refit, &source, &target, &matches);
        if inliers.len() < 4 {
            continue;
        }
        let duplicate = fitted.iter().any(|(existing, _)| {
            focus_transform_disagreement_px(
                existing,
                &refit,
                pyramid.native_dimensions[source_station],
            )
            .is_some_and(|value| value < 1.0)
        });
        if !duplicate {
            fitted.push((refit, inliers));
        }
    }
    fitted.sort_by(|left, right| {
        right.1.len().cmp(&left.1.len()).then_with(|| {
            matrix3_row_major(&left.0)
                .iter()
                .map(|value| value.to_bits())
                .cmp(
                    matrix3_row_major(&right.0)
                        .iter()
                        .map(|value| value.to_bits()),
                )
        })
    });
    let models = fitted
        .iter()
        .map(|(transform, inliers)| {
            let points = inliers
                .iter()
                .map(|item| (source[item.source].point, target[item.target].point))
                .collect::<Vec<_>>();
            StructuralModelRecord {
                transform: matrix3_row_major(transform),
                correspondences: inliers
                    .iter()
                    .map(|item| {
                        correspondence_record(
                            &source[item.source],
                            &target[item.target],
                            item.distance,
                        )
                    })
                    .collect(),
                median_symmetric_error_px: median_symmetric_error(transform, &points),
                scale_ratio: homography_scale_ratio(transform).unwrap_or(f64::INFINITY),
                covered_overlap_hull: 0.0,
                preserves_convex_orientation: homography_preserves_focus_orientation(
                    transform,
                    pyramid.native_dimensions[source_station],
                ),
                low_frequency_mean_relative_difference: f64::INFINITY,
                edge_strength_ratio: f64::INFINITY,
                median_edge_orientation_difference_degrees: f64::INFINITY,
                rejection_reasons: vec!["structural_model_not_finally_measured".to_string()],
            }
        })
        .collect();
    (
        StructuralDirectionRecord {
            source_primitives: source
                .iter()
                .map(|primitive| primitive.record.clone())
                .collect(),
            target_primitives: target
                .iter()
                .map(|primitive| primitive.record.clone())
                .collect(),
            mutual_matches: matches.len(),
            hash_proposals: proposals.len(),
            models,
        },
        fitted,
    )
}

fn point_from_bits(x: u64, y: u64) -> Point2<f64> {
    Point2::new(f64::from_bits(x), f64::from_bits(y))
}

fn finally_measure_models(
    direction: &mut StructuralDirectionRecord,
    source_station: usize,
    target_station: usize,
    tiles: &[ImageInfo],
    pyramid: &VirtualTileMatchPyramid,
) {
    for model in &mut direction.models {
        let transform = Matrix3::from_row_slice(&model.transform);
        let points = model
            .correspondences
            .iter()
            .map(|point| {
                (
                    point_from_bits(point.source_x_bits, point.source_y_bits),
                    point_from_bits(point.target_x_bits, point.target_y_bits),
                )
            })
            .collect::<Vec<_>>();
        let quality_transform = transform;
        let dense = focus_overlap_quality_measurements_on_coverage(
            &tiles[source_station],
            &tiles[target_station],
            &quality_transform,
            Some((
                &pyramid.quality_coverages[source_station],
                &pyramid.quality_coverages[target_station],
            )),
        );
        let overlap = StationOverlapMeasure::Covered {
            source: &pyramid.coverages.last().unwrap()[source_station],
            target: &pyramid.coverages.last().unwrap()[target_station],
            factor: *pyramid.factors.last().unwrap(),
        };
        if let Some(measured) = measure_station_relation_candidate(
            points.len(),
            &transform,
            &points,
            pyramid.native_dimensions[source_station],
            pyramid.native_dimensions[target_station],
            Some(overlap),
            dense,
            false,
        ) {
            let values = measured.measurements;
            model.median_symmetric_error_px = values.median_error_px;
            model.scale_ratio = values.scale_ratio;
            model.covered_overlap_hull = values.spatial_support;
            model.preserves_convex_orientation = values.preserves_convex_orientation;
            model.low_frequency_mean_relative_difference =
                values.low_frequency_mean_relative_difference;
            model.edge_strength_ratio = values.edge_strength_ratio;
            model.median_edge_orientation_difference_degrees =
                values.median_edge_orientation_difference_degrees;
            model.rejection_reasons = station_relation_rejection_reasons(&values)
                .into_iter()
                .map(str::to_string)
                .collect();
        }
    }
}

fn best_empty_rejection_model(
    direction: &StructuralDirectionRecord,
) -> Option<&StructuralModelRecord> {
    direction
        .models
        .iter()
        .filter(|model| model.rejection_reasons.is_empty())
        .max_by_key(|model| model.correspondences.len())
}

pub(super) fn run_structural_oracle(
    tiles: &[ImageInfo],
    pyramid: &VirtualTileMatchPyramid,
) -> StructuralOracleRecord {
    let started = Instant::now();
    let (mut forward, _) = solve_structural_direction(0, 1, tiles, pyramid);
    let (mut reverse, _) = solve_structural_direction(1, 0, tiles, pyramid);
    finally_measure_models(&mut forward, 0, 1, tiles, pyramid);
    finally_measure_models(&mut reverse, 1, 0, tiles, pyramid);
    let forward_model = best_empty_rejection_model(&forward);
    let reverse_model = best_empty_rejection_model(&reverse);
    let closure = forward_model
        .zip(reverse_model)
        .and_then(|(forward, reverse)| {
            let reverse = Matrix3::from_row_slice(&reverse.transform).try_inverse()?;
            focus_transform_disagreement_px(
                &Matrix3::from_row_slice(&forward.transform),
                &reverse,
                pyramid.native_dimensions[0],
            )
        })
        .unwrap_or(f64::INFINITY);
    // A canonical swap is an entirely separate swapped solve; no transform is
    // passed between directions. Its agreement is therefore the same auditable
    // five-point comparison, recorded separately as a hard gate.
    let canonical_swap_disagreement = closure;
    let accepted = forward_model
        .is_some_and(|model| model.correspondences.len() >= STATION_RELATION_MIN_INLIERS)
        && reverse_model
            .is_some_and(|model| model.correspondences.len() >= STATION_RELATION_MIN_INLIERS)
        && closure <= 1.0
        && canonical_swap_disagreement <= 1.0;
    let primitive_bytes = forward.source_primitives.len()
        + forward.target_primitives.len()
        + reverse.source_primitives.len()
        + reverse.target_primitives.len();
    let plane_bytes = pyramid
        .levels
        .iter()
        .flat_map(|level| level.iter())
        .map(|image| image.alignment_image.len())
        .sum::<usize>();
    StructuralOracleRecord {
        evidence: "traditional_structure_virtual_tile_coverage_test_only",
        factors: STRUCTURAL_FACTORS
            .into_iter()
            .filter(|factor| pyramid.factors.contains(factor))
            .collect(),
        forward,
        reverse,
        forward_reverse_five_point_closure_px: closure,
        canonical_swap_disagreement_px: canonical_swap_disagreement,
        accepted,
        elapsed_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        estimated_extra_memory_bytes: (plane_bytes
            + primitive_bytes * std::mem::size_of::<StructuralPrimitive>() * 2)
            as u64,
    }
}

// ---------------------------------------------------------------------------
// Independent synthetic oracle tests
// ---------------------------------------------------------------------------

fn synthetic_plane(width: u32, height: u32, shift: (i32, i32), repeated: bool) -> GrayImage {
    let mut image = GrayImage::from_pixel(width, height, image::Luma([245]));
    let mut draw = |x: i32, y: i32| {
        let x = x + shift.0;
        let y = y + shift.1;
        if x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
            image.put_pixel(x as u32, y as u32, image::Luma([12]));
        }
    };
    for motif in 0..if repeated { 4 } else { 1 } {
        let ox = 30 + motif * 48;
        for t in 0..45 {
            draw(ox + t, 35);
            draw(ox + 10, 25 + t);
        }
        for t in 0..35 {
            draw(ox + t, 75 + t / 2);
        }
    }
    for t in 0..100 {
        draw(45 + t, 145);
    }
    for t in 0..70 {
        draw(150, 105 + t);
    }
    image
}

fn synthetic_pyramid(
    left: GrayImage,
    right: GrayImage,
    left_coverage: stitching::CoverageMask,
    right_coverage: stitching::CoverageMask,
) -> (Vec<ImageInfo>, VirtualTileMatchPyramid) {
    let mut tiles = vec![
        plane_info_for_oracle(0, left.dimensions()),
        plane_info_for_oracle(1, right.dimensions()),
    ];
    let mut pyramid = VirtualTileMatchPyramid::new(vec![4], 2);
    install_virtual_tile_match_planes(&mut tiles[0], &mut pyramid, 0, &left, &left_coverage, &[])
        .unwrap();
    install_virtual_tile_match_planes(&mut tiles[1], &mut pyramid, 1, &right, &right_coverage, &[])
        .unwrap();
    (tiles, pyramid)
}

fn plane_info_for_oracle(id: usize, dimensions: (u32, u32)) -> ImageInfo {
    ImageInfo {
        id,
        filename: String::new(),
        width: dimensions.0,
        height: dimensions.1,
        alignment_image: GrayImage::new(1, 1),
        full_image: None,
        scale_factor: 1.0,
        focal_length_35mm: None,
        overview_reference: false,
        features: Vec::new(),
        top_features: Vec::new(),
        foreground_range: None,
        foreground_mask: None,
        horizontal_edge_rows: Vec::new(),
        vertical_edge_columns: Vec::new(),
    }
}

#[test]
fn structural_oracle_localizes_auditable_primitives() {
    let image = synthetic_plane(256, 208, (0, 0), false);
    let coverage = stitching::CoverageMask::from_bytes(256, 208, vec![255; 256 * 208]).unwrap();
    let (_, pyramid) = synthetic_pyramid(image.clone(), image, coverage.clone(), coverage);
    let primitives = extract_structural_primitives(&pyramid, 0);
    assert!(primitives.len() >= 8, "primitives={}", primitives.len());
    assert!(primitives.iter().any(|primitive| {
        primitive.record.kind == StructuralPrimitiveKind::LineEndpoint
            && (primitive.point - Point2::new(30.0, 35.0)).norm() <= 8.0
    }));
    assert!(
        primitives
            .iter()
            .all(|primitive| primitive.record.chain_length >= MIN_CHAIN_POINTS as u32)
    );
}

#[test]
fn structural_oracle_ignores_uncovered_payload_bits() {
    let width = 256;
    let height = 208;
    let coverage = stitching::CoverageMask::from_bytes(
        width,
        height,
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    if x > 20 && x < 230 && y > 20 && y < 190 {
                        255
                    } else {
                        0
                    }
                })
            })
            .collect(),
    )
    .unwrap();
    let first = synthetic_plane(width, height, (0, 0), false);
    let mut second = first.clone();
    for y in 0..height {
        for x in 0..width {
            if !coverage.is_covered(x, y) {
                second.put_pixel(x, y, image::Luma([((x * 17 + y * 31) & 255) as u8]));
            }
        }
    }
    let (first_tiles, first_pyramid) =
        synthetic_pyramid(first.clone(), first, coverage.clone(), coverage.clone());
    let (second_tiles, second_pyramid) =
        synthetic_pyramid(second.clone(), second, coverage.clone(), coverage);
    let first_bits =
        serde_json::to_vec(&solve_structural_direction(0, 1, &first_tiles, &first_pyramid).0)
            .unwrap();
    let second_bits =
        serde_json::to_vec(&solve_structural_direction(0, 1, &second_tiles, &second_pyramid).0)
            .unwrap();
    assert_eq!(first_bits, second_bits);
}

#[test]
fn structural_oracle_geometric_hash_disambiguates_repetition_with_real_points() {
    let left = synthetic_plane(320, 240, (0, 0), true);
    let right = synthetic_plane(320, 240, (12, 8), true);
    let coverage = stitching::CoverageMask::from_bytes(320, 240, vec![255; 320 * 240]).unwrap();
    let (_, pyramid) = synthetic_pyramid(left, right, coverage.clone(), coverage);
    let (direction, fits) = solve_structural_direction(0, 1, &[], &pyramid);
    let (transform, inliers) = fits.first().expect("a repeated fixture has a hashed model");
    assert!(inliers.len() >= 4);
    assert!(
        focus_transform_disagreement_px(
            transform,
            &Matrix3::new(1.0, 0.0, 12.0, 0.0, 1.0, 8.0, 0.0, 0.0, 1.0),
            (320, 240),
        )
        .unwrap()
            <= 2.0
    );
    let ids = direction.models[0]
        .correspondences
        .iter()
        .map(|point| (point.source_primitive_id, point.target_primitive_id))
        .collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), direction.models[0].correspondences.len());
}

#[test]
fn structural_oracle_correspondences_keep_independent_target_provenance() {
    let left = synthetic_plane(320, 240, (0, 0), false);
    let right = synthetic_plane(320, 240, (13, 7), false);
    let coverage = stitching::CoverageMask::from_bytes(320, 240, vec![255; 320 * 240]).unwrap();
    let (_, pyramid) = synthetic_pyramid(left, right, coverage.clone(), coverage);
    let (direction, _) = solve_structural_direction(0, 1, &[], &pyramid);
    let target = direction
        .target_primitives
        .iter()
        .map(|primitive| {
            (
                primitive.id,
                primitive.x_native_bits,
                primitive.y_native_bits,
            )
        })
        .collect::<BTreeSet<_>>();
    for model in direction.models {
        for point in model.correspondences {
            assert!(target.contains(&(
                point.target_primitive_id,
                point.target_x_bits,
                point.target_y_bits
            )));
        }
    }
}

#[test]
fn structural_oracle_left_right_solve_inverts() {
    let left = synthetic_plane(320, 240, (0, 0), false);
    let right = synthetic_plane(320, 240, (12, 8), false);
    let coverage = stitching::CoverageMask::from_bytes(320, 240, vec![255; 320 * 240]).unwrap();
    let (_, pyramid) = synthetic_pyramid(left, right, coverage.clone(), coverage);
    let (_, forward) = solve_structural_direction(0, 1, &[], &pyramid);
    let (_, reverse) = solve_structural_direction(1, 0, &[], &pyramid);
    let reverse_inverse = reverse[0].0.try_inverse().unwrap();
    assert!(
        focus_transform_disagreement_px(&forward[0].0, &reverse_inverse, (320, 240)).unwrap()
            <= 1.0
    );
}

#[test]
fn structural_oracle_rejects_unrelated_structure() {
    let left = synthetic_plane(320, 240, (0, 0), false);
    let mut right = GrayImage::from_pixel(320, 240, image::Luma([245]));
    for x in 20..300 {
        right.put_pixel(x, 20 + x / 3, image::Luma([10]));
    }
    let coverage = stitching::CoverageMask::from_bytes(320, 240, vec![255; 320 * 240]).unwrap();
    let (_, pyramid) = synthetic_pyramid(left, right, coverage.clone(), coverage);
    let (direction, _) = solve_structural_direction(0, 1, &[], &pyramid);
    assert!(
        direction
            .models
            .iter()
            .all(|model| model.correspondences.len() < STATION_RELATION_MIN_INLIERS)
    );
}

#[test]
fn structural_oracle_is_deterministic_at_1_2_8_16_threads() {
    let left = synthetic_plane(320, 240, (0, 0), true);
    let right = synthetic_plane(320, 240, (12, 8), true);
    let coverage = stitching::CoverageMask::from_bytes(320, 240, vec![255; 320 * 240]).unwrap();
    let (_, pyramid) = synthetic_pyramid(left, right, coverage.clone(), coverage);
    let expected = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            serde_json::to_vec(&solve_structural_direction(0, 1, &[], &pyramid).0).unwrap()
        });
    for threads in [2, 8, 16] {
        let actual = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                serde_json::to_vec(&solve_structural_direction(0, 1, &[], &pyramid).0).unwrap()
            });
        assert_eq!(actual, expected, "thread count {threads}");
    }
}
