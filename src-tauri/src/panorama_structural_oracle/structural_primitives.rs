//! Structural primitives (line endpoints, line intersections and high-curvature points) and
//! their descriptors, extracted per pyramid factor from the edge chains.

use super::*;

pub(super) const CELL_COLUMNS: u32 = 8;
pub(super) const CELL_ROWS: u32 = 6;
pub(super) const CELL_QUOTA: usize = 8;
pub(super) const MAX_PRIMITIVES_PER_SCALE: usize = 320;

pub(super) fn undirected_angle(mut angle: f64) -> f64 {
    while angle < 0.0 {
        angle += std::f64::consts::PI;
    }
    while angle >= std::f64::consts::PI {
        angle -= std::f64::consts::PI;
    }
    angle
}

pub(super) fn angle_difference(left: f64, right: f64) -> f64 {
    let difference = (undirected_angle(left) - undirected_angle(right)).abs();
    difference.min(std::f64::consts::PI - difference)
}

pub(super) fn primitive_candidate(
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

pub(super) fn primitive_candidates(
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

pub(super) fn attach_descriptors(
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

pub(super) fn descriptor_distance(left: &[u16], right: &[u16]) -> u64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| i64::from(a).abs_diff(i64::from(b)))
        .sum()
}

pub(super) fn extract_structural_primitives(
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
