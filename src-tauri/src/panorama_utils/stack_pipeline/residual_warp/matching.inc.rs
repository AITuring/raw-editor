// Residual_Warp module fragment; included at parent module scope.

/// Verify a forward match by matching the resulting point back into the source
/// image.  The existing patch matcher is used in both directions, so the
/// round-trip gate is exactly the same one used by production registration.
pub(crate) fn verify_round_trip_match(
    source: &GrayImage,
    target: &GrayImage,
    forward_transform: &Matrix3<f64>,
    backward_transform: &Matrix3<f64>,
    center: Point2<f64>,
) -> Option<RoundTripMatch> {
    verify_round_trip_match_with_template(
        source,
        target,
        forward_transform,
        backward_transform,
        center,
        9,
    )
}

pub(crate) fn verify_round_trip_match_with_template(
    source: &GrayImage,
    target: &GrayImage,
    forward_transform: &Matrix3<f64>,
    backward_transform: &Matrix3<f64>,
    center: Point2<f64>,
    half: i32,
) -> Option<RoundTripMatch> {
    let forward_match = refine_warped_patch(source, target, forward_transform, center, half, 7)?;
    let backward_match = refine_warped_patch(
        target,
        source,
        backward_transform,
        forward_match.target,
        half,
        7,
    )?;
    let round_trip_error_px = (backward_match.target - center).norm();
    (round_trip_error_px <= RESIDUAL_WARP_MAX_ROUND_TRIP_ERROR_PX).then_some(RoundTripMatch {
        forward: forward_match.target,
        backward: backward_match.target,
        round_trip_error_px,
    })
}

fn observation_is_valid(observation: WarpObservation) -> bool {
    observation.world.x.is_finite()
        && observation.world.y.is_finite()
        && observation
            .displacement
            .iter()
            .all(|value| value.is_finite())
        && observation.displacement[0].hypot(observation.displacement[1])
            <= RESIDUAL_WARP_MAX_NODE_DISPLACEMENT_PX
        && observation.round_trip_error_px.is_finite()
        && observation.round_trip_error_px <= RESIDUAL_WARP_MAX_ROUND_TRIP_ERROR_PX
}

fn sane_extent(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn contains(world: &WorldRect, x: f64, y: f64) -> bool {
    x >= world.left
        && y >= world.top
        && x <= world.left + sane_extent(world.width)
        && y <= world.top + sane_extent(world.height)
}

fn node_world(world: &WorldRect, column: usize, row: usize, step: f64) -> WorldPoint {
    WorldPoint {
        x: world.left + column as f64 * step,
        y: world.top + row as f64 * step,
    }
}

fn neighbour_indices(row: usize, column: usize, width: usize, height: usize) -> Vec<usize> {
    let mut result = Vec::with_capacity(4);
    if row > 0 {
        result.push((row - 1) * width + column);
    }
    if column > 0 {
        result.push(row * width + column - 1);
    }
    if row + 1 < height {
        result.push((row + 1) * width + column);
    }
    if column + 1 < width {
        result.push(row * width + column + 1);
    }
    result
}

fn displacement_delta(left: [f64; 2], right: [f64; 2]) -> f64 {
    (left[0] - right[0]).hypot(left[1] - right[1])
}

fn percentile95(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let rank = (values.len() * 95).div_ceil(100).max(1);
    let index = rank.saturating_sub(1).min(values.len() - 1);
    values[index]
}

fn edge_fade(world: &WorldRect, x: f64, y: f64) -> f64 {
    let width = sane_extent(world.width);
    let height = sane_extent(world.height);
    let distance = (x - world.left)
        .min(world.left + width - x)
        .min(y - world.top)
        .min(world.top + height - y);
    let t = (distance / RESIDUAL_WARP_EDGE_FADE_PX).clamp(0.0, 1.0);
    // Smoothstep gives a zero derivative at both ends and avoids a visible
    // crease where the local field meets the global homography.
    t * t * (3.0 - 2.0 * t)
}

