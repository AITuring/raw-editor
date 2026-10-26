// Quality_Gate module fragment; included at parent module scope.

// ==================== task 15.6: boundary stroke and confidence ====================
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BoundaryStrokeMeasurement {
    pub world: (f64, f64),
    pub normal_error_px: f64,
    pub orientation_error_deg: f64,
}
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundaryStrokeReport {
    pub measurements: Vec<BoundaryStrokeMeasurement>,
    pub pairable_count: usize,
    pub unpairable_count: usize,
    pub unmeasurable: Vec<((f64, f64), &'static str)>,
    pub p95_error_px: f64,
    pub max_error_px: f64,
}

pub(crate) fn trace_moore_boundary(
    labels: &OwnerRegions,
    left_label: u32,
    right_label: u32,
) -> Result<Vec<(u32, u32)>, &'static str> {
    if left_label == 0 || right_label == 0 || left_label == right_label {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    let w = labels.width as usize;
    let h = labels.height as usize;
    let touches = |x: usize, y: usize| -> bool {
        let i = y * w + x;
        if labels.labels[i] != left_label && labels.labels[i] != right_label {
            return false;
        }
        (-1i32..=1)
            .flat_map(|dy| (-1i32..=1).map(move |dx| (dx, dy)))
            .filter(|(dx, dy)| *dx != 0 || *dy != 0)
            .any(|(dx, dy)| {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                nx >= 0
                    && ny >= 0
                    && (nx as usize) < w
                    && (ny as usize) < h
                    && labels.labels[ny as usize * w + nx as usize]
                        == if labels.labels[i] == left_label {
                            right_label
                        } else {
                            left_label
                        }
            })
    };
    let start = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .find(|&(x, y)| touches(x, y))
        .ok_or(degradation::BOUNDARY_NO_PAIRABLE_EDGE)?;
    let dirs: [(i32, i32); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];
    let mut result = Vec::new();
    let mut current = start;
    let mut backtrack = 4usize;
    let mut states = BTreeSet::new();
    let max_steps = w.saturating_mul(h).saturating_mul(8).max(1);
    for _ in 0..max_steps {
        if !touches(current.0, current.1) {
            break;
        }
        if !states.insert((current, backtrack)) {
            break;
        }
        result.push((current.0 as u32, current.1 as u32));
        let mut found = None;
        for offset in 1..=8 {
            let index = (backtrack + offset) % 8;
            let nx = current.0 as i32 + dirs[index].0;
            let ny = current.1 as i32 + dirs[index].1;
            if nx >= 0
                && ny >= 0
                && (nx as usize) < w
                && (ny as usize) < h
                && touches(nx as usize, ny as usize)
            {
                found = Some(((nx as usize, ny as usize), (index + 4) % 8));
                break;
            }
        }
        let Some((next, next_backtrack)) = found else {
            break;
        };
        current = next;
        backtrack = next_backtrack;
        if current == start && result.len() > 2 {
            break;
        }
    }
    if result.len() < 2 {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    Ok(result)
}

fn luminance(pixel: image::Rgb<f32>) -> f64 {
    0.2126 * f64::from(pixel[0]) + 0.7152 * f64::from(pixel[1]) + 0.0722 * f64::from(pixel[2])
}

pub(crate) fn measure_boundary_strokes(
    image: &Rgb32FImage,
    boundary: &[(u32, u32)],
    world_origin: (f64, f64),
) -> Result<BoundaryStrokeReport, &'static str> {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 || boundary.len() < 2 {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    let mut measurements = Vec::new();
    let mut unpairable_count = 0;
    let mut unmeasurable = Vec::new();
    for (index, &(x, y)) in boundary.iter().enumerate().step_by(256) {
        let prev = boundary[(index + boundary.len() - 1) % boundary.len()];
        let next = boundary[(index + 1) % boundary.len()];
        let tangent = (
            f64::from(next.0) - f64::from(prev.0),
            f64::from(next.1) - f64::from(prev.1),
        );
        let length = tangent.0.hypot(tangent.1);
        if length <= f64::EPSILON {
            unpairable_count += 1;
            continue;
        }
        let normal = (-tangent.1 / length, tangent.0 / length);
        let mut best_left = (0.0f64, 0.0f64, 0.0f64);
        let mut best_right = (0.0f64, 0.0f64, 0.0f64);
        for distance in 1..=16 {
            let d = f64::from(distance);
            for side in [-1.0, 1.0] {
                let sx = f64::from(x) + side * normal.0 * d;
                let sy = f64::from(y) + side * normal.1 * d;
                let ix = sx.round() as i32;
                let iy = sy.round() as i32;
                if ix < 1 || iy < 1 || ix + 1 >= width as i32 || iy + 1 >= height as i32 {
                    continue;
                }
                let gx = luminance(*image.get_pixel((ix + 1) as u32, iy as u32))
                    - luminance(*image.get_pixel((ix - 1) as u32, iy as u32));
                let gy = luminance(*image.get_pixel(ix as u32, (iy + 1) as u32))
                    - luminance(*image.get_pixel(ix as u32, (iy - 1) as u32));
                let contrast = gx.hypot(gy) * 0.5;
                if contrast < 0.15 {
                    continue;
                }
                let orientation = gy.atan2(gx).to_degrees();
                if side < 0.0 {
                    best_left.0 += contrast;
                    best_left.1 += side * d * contrast;
                    best_left.2 += orientation * contrast;
                } else {
                    best_right.0 += contrast;
                    best_right.1 += side * d * contrast;
                    best_right.2 += orientation * contrast;
                }
            }
        }
        if best_left.0 <= 0.0 || best_right.0 <= 0.0 {
            unpairable_count += 1;
            unmeasurable.push((
                (world_origin.0 + f64::from(x), world_origin.1 + f64::from(y)),
                degradation::BOUNDARY_LOW_CONTRAST,
            ));
            continue;
        }
        let left_angle = best_left.2 / best_left.0;
        let right_angle = best_right.2 / best_right.0;
        let mut angle_error = (left_angle - right_angle).abs() % 180.0;
        if angle_error > 90.0 {
            angle_error = 180.0 - angle_error;
        }
        if angle_error > 10.0 {
            unpairable_count += 1;
            unmeasurable.push((
                (world_origin.0 + f64::from(x), world_origin.1 + f64::from(y)),
                degradation::BOUNDARY_ORIENTATION_MISMATCH,
            ));
            continue;
        }
        let left_position = best_left.1 / best_left.0;
        let right_position = best_right.1 / best_right.0;
        let normal_error = (left_position + right_position).abs();
        measurements.push(BoundaryStrokeMeasurement {
            world: (world_origin.0 + f64::from(x), world_origin.1 + f64::from(y)),
            normal_error_px: normal_error,
            orientation_error_deg: angle_error,
        });
    }
    if measurements.is_empty() {
        return Err(degradation::BOUNDARY_NO_PAIRABLE_EDGE);
    }
    let mut errors = measurements
        .iter()
        .map(|m| m.normal_error_px)
        .collect::<Vec<_>>();
    errors.sort_by(f64::total_cmp);
    let p95 = errors[((errors.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(errors.len() - 1)];
    Ok(BoundaryStrokeReport {
        pairable_count: measurements.len(),
        unpairable_count,
        unmeasurable,
        p95_error_px: p95,
        max_error_px: *errors.last().unwrap_or(&0.0),
        measurements,
    })
}

