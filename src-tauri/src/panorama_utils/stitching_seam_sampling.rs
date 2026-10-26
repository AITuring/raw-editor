use super::*;

pub(crate) struct SeamContext<'a> {
    pub(crate) pano: &'a Rgb32FImage,
    pub(crate) pano_mask: &'a GrayImage,
    pub(crate) img_to_add_info: &'a ImageInfo,
    pub(crate) img_to_add: &'a Rgb32FImage,
    pub(crate) h_add: &'a Matrix3<f64>,
    pub(crate) projection: Projection,
    pub(crate) offset_x: f64,
    pub(crate) offset_y: f64,
    pub(crate) out_width: u32,
    pub(crate) out_height: u32,
    pub(crate) exposure: &'a ExposureCompensation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SeamOrientation {
    Vertical,
    Horizontal,
}

pub(crate) struct SeamInfo {
    pub(crate) orientation: SeamOrientation,
    pub(crate) coords: Vec<i32>,
    pub(crate) dx: f64,
    pub(crate) dy: f64,
    pub(crate) min_x: u32,
    pub(crate) max_x: u32,
    pub(crate) min_y: u32,
    pub(crate) max_y: u32,
}

/// One exact source-backed rewrite retained while both relation endpoints are
/// resident during planning. Samples are ordered by `(along, cross)` and are
/// bounded to the strip between the current and solved seam.
#[derive(Debug, Clone)]
pub(crate) struct StationSeamSample {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) expected_owner: u16,
    pub(crate) replacement_owner: u16,
    pub(crate) pixel: Rgb<f32>,
    pub(crate) source_image_id: usize,
    pub(crate) source_point: Point2<f64>,
    pub(crate) covered: bool,
}

/// A seam solved on an accepted Virtual_Tile overlap. The path and its exact
/// replacement samples are materialized while the endpoint tiles are loaded,
/// so application never renders either endpoint again.
#[derive(Debug, Clone)]
pub(crate) struct StationSeamPlan {
    pub(crate) first_owner: u16,
    pub(crate) second_owner: u16,
    pub(crate) min_x: u32,
    pub(crate) max_x: u32,
    pub(crate) min_y: u32,
    pub(crate) max_y: u32,
    pub(crate) along_min: u32,
    pub(crate) along_max: u32,
    pub(crate) orientation: SeamOrientation,
    pub(crate) cross_samples: Vec<f64>,
    pub(crate) replacement_samples: Vec<StationSeamSample>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StationOwnerTransition {
    pub(crate) cross: u32,
    pub(crate) low_owner: u16,
}

/// Reusable index of every provenance-bearing hard owner transition. Building
/// it reads the owner ledger exactly once; relation planning then visits only
/// transitions for its owner pair, orientation, and requested scanline.
#[derive(Debug, Default)]
pub(crate) struct StationOwnerTransitionIndex {
    pub(crate) by_scanline: BTreeMap<(u16, u16, SeamOrientation, u32), Vec<StationOwnerTransition>>,
    pub(crate) owner_reads: usize,
    pub(crate) indexed_transitions: usize,
}

impl StationOwnerTransitionIndex {
    pub(crate) fn build(owners: &[u16], width: u32, height: u32) -> Self {
        let mut index = Self::default();
        let stride = width as usize;
        for y in 0..height {
            for x in 0..width {
                let offset = y as usize * stride + x as usize;
                let owner = owners.get(offset).copied().unwrap_or(NO_OWNER);
                index.owner_reads += 1;
                if owner == NO_OWNER {
                    continue;
                }
                if x > 0 {
                    let low_owner = owners.get(offset - 1).copied().unwrap_or(NO_OWNER);
                    index.insert(SeamOrientation::Vertical, y, x, low_owner, owner);
                }
                if y > 0 {
                    let low_owner = owners
                        .get(offset.saturating_sub(stride))
                        .copied()
                        .unwrap_or(NO_OWNER);
                    index.insert(SeamOrientation::Horizontal, x, y, low_owner, owner);
                }
            }
        }
        index
    }

    pub(crate) fn insert(
        &mut self,
        orientation: SeamOrientation,
        along: u32,
        cross: u32,
        low_owner: u16,
        high_owner: u16,
    ) {
        if low_owner == NO_OWNER || high_owner == NO_OWNER || low_owner == high_owner {
            return;
        }
        let pair = (low_owner.min(high_owner), low_owner.max(high_owner));
        self.by_scanline
            .entry((pair.0, pair.1, orientation, along))
            .or_default()
            .push(StationOwnerTransition { cross, low_owner });
        self.indexed_transitions += 1;
    }

    pub(crate) fn query(
        &self,
        first_owner: u16,
        second_owner: u16,
        orientation: SeamOrientation,
        along: u32,
        cross_min: u32,
        cross_max: u32,
    ) -> (&[StationOwnerTransition], usize) {
        let pair = (first_owner.min(second_owner), first_owner.max(second_owner));
        let transitions = self
            .by_scanline
            .get(&(pair.0, pair.1, orientation, along))
            .map(Vec::as_slice)
            .unwrap_or_default();
        let start = transitions.partition_point(|transition| transition.cross < cross_min);
        let end = transitions.partition_point(|transition| transition.cross <= cross_max);
        (&transitions[start..end], end.saturating_sub(start))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StationSeamPlanningStats {
    pub(crate) index_owner_reads: usize,
    pub(crate) indexed_transitions: usize,
    pub(crate) relation_query_visits: usize,
    pub(crate) materialized_samples: usize,
    pub(crate) endpoint_tile_loads: usize,
    pub(crate) max_resident_endpoint_tiles: usize,
}

pub(crate) fn source_to_world_point(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    source_x: f64,
    source_y: f64,
) -> Option<Point2<f64>> {
    let projected = project_point(image, source_x, source_y, projection)?;
    let mapped = homography * Point3::new(projected.x, projected.y, 1.0);
    if mapped.z.abs() < 1e-8 {
        return None;
    }
    let point = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
    if point.x.is_finite() && point.y.is_finite() {
        Some(point)
    } else {
        None
    }
}

/// Approximate the distance from a source sample to its projected valid
/// coverage boundary in output-world pixels. The local Jacobian converts the
/// source-edge distance into world pixels, which is more faithful than using
/// a source-pixel threshold after a perspective transform.
pub(crate) fn projected_coverage_boundary_distance(
    image: &ImageInfo,
    homography: &Matrix3<f64>,
    projection: Projection,
    source: Point2<f64>,
) -> Option<f64> {
    let width = image.width().max(1) as f64;
    let height = image.height().max(1) as f64;
    let source_edge = source
        .x
        .min(source.y)
        .min((width - 1.0) - source.x)
        .min((height - 1.0) - source.y)
        .max(0.0);
    let center = source_to_world_point(image, homography, projection, source.x, source.y)?;
    let x_step = if source.x + 1.0 < width { 1.0 } else { -1.0 };
    let y_step = if source.y + 1.0 < height { 1.0 } else { -1.0 };
    let x_neighbour = source_to_world_point(
        image,
        homography,
        projection,
        (source.x + x_step).clamp(0.0, width - 1.0),
        source.y,
    )?;
    let y_neighbour = source_to_world_point(
        image,
        homography,
        projection,
        source.x,
        (source.y + y_step).clamp(0.0, height - 1.0),
    )?;
    let local_scale = (x_neighbour - center)
        .norm()
        .min((y_neighbour - center).norm())
        .max(1e-6);
    Some(source_edge * local_scale)
}

pub(crate) fn station_seam_sample(
    image: &ImageInfo,
    image_pixels: &Rgb32FImage,
    inverse: &Matrix3<f64>,
    projection: Projection,
    residual: &residual_warp::ResidualWarp,
    x: u32,
    y: u32,
    offset_x: f64,
    offset_y: f64,
) -> Option<(Rgb<f32>, Point2<f64>)> {
    let source = map_target_to_source_with_residual(
        inverse,
        Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0),
        image,
        projection,
        residual,
    )?;
    if source.x < 0.0
        || source.y < 0.0
        || source.x >= image_pixels.width() as f64
        || source.y >= image_pixels.height() as f64
        || !source_sample_has_support(image_pixels, source.x, source.y)
    {
        return None;
    }
    let pixel = get_high_quality_interpolated_pixel(image_pixels, source.x, source.y);
    (pixel[0].max(pixel[1]).max(pixel[2]) > 1e-6).then_some((pixel, source))
}

/// Sample a seam pair with an optional locally fitted relation.  The first
/// tile remains on the authoritative global pose; when a seam-only relation
/// is present, the second tile is read by projecting that same first-tile
/// sample through the local source-to-target fit.  This repairs the visible
/// relative geometry at a rejected closure edge without changing the global
/// pose graph or any accepted relation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn station_seam_pair_sample(
    first: &ImageInfo,
    first_pixels: &Rgb32FImage,
    first_inverse: &Matrix3<f64>,
    second: &ImageInfo,
    second_pixels: &Rgb32FImage,
    second_inverse: &Matrix3<f64>,
    relation: Option<&Matrix3<f64>>,
    projection: Projection,
    residual: &residual_warp::ResidualWarp,
    x: u32,
    y: u32,
    offset_x: f64,
    offset_y: f64,
) -> Option<((Rgb<f32>, Point2<f64>), (Rgb<f32>, Point2<f64>))> {
    let first_sample = station_seam_sample(
        first,
        first_pixels,
        first_inverse,
        projection,
        residual,
        x,
        y,
        offset_x,
        offset_y,
    )?;
    let second_sample = if let Some(relation) = relation {
        let target = Point3::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
        let first_source =
            map_target_to_source_with_residual(first_inverse, target, first, projection, residual)?;
        let first_projected = project_point(first, first_source.x, first_source.y, projection)?;
        let mapped = relation * Point3::new(first_projected.x, first_projected.y, 1.0);
        if !mapped.x.is_finite() || !mapped.y.is_finite() || mapped.z.abs() < 1e-8 {
            None
        } else {
            let projected_target = Point2::new(mapped.x / mapped.z, mapped.y / mapped.z);
            let source =
                unproject_point(second, projected_target.x, projected_target.y, projection)?;
            if source.x < 0.0
                || source.y < 0.0
                || source.x >= second_pixels.width() as f64
                || source.y >= second_pixels.height() as f64
                || !source_sample_has_support(second_pixels, source.x, source.y)
            {
                None
            } else {
                let pixel = get_high_quality_interpolated_pixel(second_pixels, source.x, source.y);
                (pixel[0].max(pixel[1]).max(pixel[2]) > 1e-6).then_some((pixel, source))
            }
        }
    } else {
        station_seam_sample(
            second,
            second_pixels,
            second_inverse,
            projection,
            residual,
            x,
            y,
            offset_x,
            offset_y,
        )
    }?;
    Some((first_sample, second_sample))
}

pub(crate) fn station_seam_edge_misalignment(left: (f64, f64), right: (f64, f64)) -> f64 {
    pub(crate) const EDGE_EVIDENCE_FLOOR: f64 = 0.005;
    let left_magnitude = left.0.hypot(left.1);
    let right_magnitude = right.0.hypot(right.1);
    if left_magnitude < EDGE_EVIDENCE_FLOOR && right_magnitude < EDGE_EVIDENCE_FLOOR {
        return 0.0;
    }
    if left_magnitude < EDGE_EVIDENCE_FLOOR || right_magnitude < EDGE_EVIDENCE_FLOOR {
        return 1.0;
    }
    let orientation = 1.0
        - ((left.0 * right.0 + left.1 * right.1).abs() / (left_magnitude * right_magnitude))
            .clamp(0.0, 1.0);
    let magnitude = (left_magnitude - right_magnitude).abs()
        / left_magnitude.max(right_magnitude).max(EDGE_EVIDENCE_FLOOR);
    (orientation + magnitude).clamp(0.0, 2.0)
}

pub(crate) fn station_seam_luma(pixel: Rgb<f32>) -> f64 {
    0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn station_seam_world_gradient(
    image: &ImageInfo,
    pixels: &Rgb32FImage,
    inverse: &Matrix3<f64>,
    projection: Projection,
    residual: &residual_warp::ResidualWarp,
    x: u32,
    y: u32,
    offset_x: f64,
    offset_y: f64,
) -> Option<(f64, f64)> {
    let left = station_seam_sample(
        image,
        pixels,
        inverse,
        projection,
        residual,
        x.checked_sub(1)?,
        y,
        offset_x,
        offset_y,
    )?;
    let right = station_seam_sample(
        image,
        pixels,
        inverse,
        projection,
        residual,
        x.checked_add(1)?,
        y,
        offset_x,
        offset_y,
    )?;
    let top = station_seam_sample(
        image,
        pixels,
        inverse,
        projection,
        residual,
        x,
        y.checked_sub(1)?,
        offset_x,
        offset_y,
    )?;
    let bottom = station_seam_sample(
        image,
        pixels,
        inverse,
        projection,
        residual,
        x,
        y.checked_add(1)?,
        offset_x,
        offset_y,
    )?;
    Some((
        (station_seam_luma(right.0) - station_seam_luma(left.0)) * 0.5,
        (station_seam_luma(bottom.0) - station_seam_luma(top.0)) * 0.5,
    ))
}

pub(crate) fn nearest_station_transition(
    transition_index: &StationOwnerTransitionIndex,
    plan: &StationSeamPlan,
    along: u32,
    target: f64,
    maximum_displacement: Option<f64>,
) -> (Option<(u32, bool)>, usize) {
    if !target.is_finite() {
        return (None, 0);
    }
    let (cross_min, cross_max) = match plan.orientation {
        SeamOrientation::Vertical => (plan.min_x, plan.max_x),
        SeamOrientation::Horizontal => (plan.min_y, plan.max_y),
    };
    let (query_min, query_max) =
        maximum_displacement.map_or((cross_min, cross_max), |maximum_displacement| {
            (
                (target - maximum_displacement)
                    .ceil()
                    .max(f64::from(cross_min)) as u32,
                (target + maximum_displacement)
                    .floor()
                    .min(f64::from(cross_max)) as u32,
            )
        });
    if query_min > query_max {
        return (None, 0);
    }
    let (transitions, visits) = transition_index.query(
        plan.first_owner,
        plan.second_owner,
        plan.orientation,
        along,
        query_min,
        query_max,
    );
    (
        transitions
            .iter()
            .min_by_key(|transition| transition.cross.abs_diff(target.round() as u32))
            .map(|transition| (transition.cross, transition.low_owner == plan.first_owner)),
        visits,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_station_seam_samples(
    plan: &mut StationSeamPlan,
    transition_index: &StationOwnerTransitionIndex,
    current_owners: &[u16],
    out_width: u32,
    first: &ImageInfo,
    first_pixels: &Rgb32FImage,
    first_inverse: &Matrix3<f64>,
    second: &ImageInfo,
    second_pixels: &Rgb32FImage,
    second_inverse: &Matrix3<f64>,
    projection: Projection,
    residual: &residual_warp::ResidualWarp,
    offset_x: f64,
    offset_y: f64,
) -> usize {
    let (cross_min, cross_max) = match plan.orientation {
        SeamOrientation::Vertical => (plan.min_x, plan.max_x),
        SeamOrientation::Horizontal => (plan.min_y, plan.max_y),
    };
    let mut query_visits = 0;
    for along in plan.along_min..=plan.along_max {
        let seam = station_seam_cross_at(plan, along);
        let (transition, visits) =
            nearest_station_transition(transition_index, plan, along, seam, Some(32.0));
        query_visits += visits;
        let Some((current_transition, first_on_low_side)) = transition else {
            continue;
        };
        let current_seam = f64::from(current_transition);
        if (seam - current_seam).abs() > 32.0 {
            continue;
        }
        let corridor_min = seam.min(current_seam).floor().max(f64::from(cross_min)) as u32;
        let corridor_max = seam.max(current_seam).ceil().min(f64::from(cross_max)) as u32;
        for cross in corridor_min..=corridor_max {
            let (x, y) = match plan.orientation {
                SeamOrientation::Vertical => (cross, along),
                SeamOrientation::Horizontal => (along, cross),
            };
            let offset = y as usize * out_width as usize + x as usize;
            let expected_owner = current_owners.get(offset).copied().unwrap_or(NO_OWNER);
            if expected_owner != plan.first_owner && expected_owner != plan.second_owner {
                continue;
            }
            let first_side = if first_on_low_side {
                f64::from(cross) <= seam
            } else {
                f64::from(cross) >= seam
            };
            let replacement_owner = if first_side {
                plan.first_owner
            } else {
                plan.second_owner
            };
            if replacement_owner == expected_owner {
                continue;
            }
            let (source_image, source_pixels, source_inverse) =
                if replacement_owner == plan.first_owner {
                    (first, first_pixels, first_inverse)
                } else {
                    (second, second_pixels, second_inverse)
                };
            if let Some((pixel, source_point)) = station_seam_sample(
                source_image,
                source_pixels,
                source_inverse,
                projection,
                residual,
                x,
                y,
                offset_x,
                offset_y,
            ) {
                plan.replacement_samples.push(StationSeamSample {
                    x,
                    y,
                    expected_owner,
                    replacement_owner,
                    pixel,
                    source_image_id: source_image.id,
                    source_point,
                    covered: true,
                });
            }
        }
    }
    query_visits
}
