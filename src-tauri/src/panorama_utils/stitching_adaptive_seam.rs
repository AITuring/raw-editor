use super::*;

/// Restore a small amount of native edge contrast after focus ownership.  The
/// focus score intentionally uses a local blur and the ownership mask is
/// upsampled from an analysis canvas, both of which are stable but slightly
/// conservative around thin brush strokes.  This bounded luma-only unsharp
/// pass runs per virtual tile, never across a projected invalid corner, and
/// therefore cannot blend two camera positions together.
pub(crate) fn sharpen_focus_tile_detail(image: &mut Rgb32FImage, mask: &GrayImage, amount: f32) {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 || mask.dimensions() != (width, height) || amount <= 0.0 {
        return;
    }
    let source = image.clone();
    let source_pixels = source.as_raw();
    let source_mask = mask.as_raw();
    let width_usize = width as usize;
    let height_usize = height as usize;
    image
        .as_mut()
        .par_chunks_mut(width_usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            if y == 0 || y + 1 >= height_usize {
                return;
            }
            for x in 1..width_usize - 1 {
                let center_index = y * width_usize + x;
                if source_mask[center_index] == 0 {
                    continue;
                }
                let center_start = center_index * 3;
                let center_luma = source_pixels[center_start] * 0.299
                    + source_pixels[center_start + 1] * 0.587
                    + source_pixels[center_start + 2] * 0.114;
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx) as usize;
                        let ny = (y as i32 + dy) as usize;
                        let index = ny * width_usize + nx;
                        if source_mask[index] == 0 {
                            continue;
                        }
                        let start = index * 3;
                        sum += source_pixels[start] * 0.299
                            + source_pixels[start + 1] * 0.587
                            + source_pixels[start + 2] * 0.114;
                        count += 1;
                    }
                }
                if count < 5 {
                    continue;
                }
                let detail = center_luma - sum / count as f32;
                let boost = (detail * amount).clamp(-0.12, 0.12);
                let start = x * 3;
                row[start] = (row[start] + boost).clamp(0.0, 1.0);
                row[start + 1] = (row[start + 1] + boost).clamp(0.0, 1.0);
                row[start + 2] = (row[start + 2] + boost).clamp(0.0, 1.0);
            }
        });
}

pub(crate) fn find_adaptive_seam(ctx: &SeamContext) -> Option<SeamInfo> {
    let h_add_inv = ctx.h_add.try_inverse()?;
    let (w_add, h_add_img) = ctx.img_to_add.dimensions();

    let mut min_ox = u32::MAX;
    let mut max_ox = 0;
    let mut min_oy = u32::MAX;
    let mut max_oy = 0;
    let mut has_overlap = false;

    let (candidate_left, candidate_right, candidate_top, candidate_bottom) =
        transformed_image_region(
            ctx.img_to_add_info,
            ctx.h_add,
            ctx.projection,
            ctx.offset_x,
            ctx.offset_y,
            ctx.out_width,
            ctx.out_height,
        )?;
    for y in candidate_top..=candidate_bottom {
        for x in candidate_left..=candidate_right {
            if ctx.pano_mask.get_pixel(x, y)[0] > 0 {
                let target_p = Point3::new(x as f64 - ctx.offset_x, y as f64 - ctx.offset_y, 1.0);
                let Some(source) =
                    map_target_to_source(&h_add_inv, target_p, ctx.img_to_add_info, ctx.projection)
                else {
                    continue;
                };
                let sx = source.x;
                let sy = source.y;
                if sx >= 0.0
                    && sx < w_add as f64
                    && sy >= 0.0
                    && sy < h_add_img as f64
                    && source_sample_has_support(ctx.img_to_add, sx, sy)
                {
                    has_overlap = true;
                    min_ox = min_ox.min(x);
                    max_ox = max_ox.max(x);
                    min_oy = min_oy.min(y);
                    max_oy = max_oy.max(y);
                }
            }
        }
    }

    if !has_overlap {
        return None;
    }

    println!(
        "    - Overlap bounds: x={}..{}, y={}..{}",
        min_ox, max_ox, min_oy, max_oy
    );

    let center_source = project_point(
        ctx.img_to_add_info,
        w_add as f64 / 2.0,
        h_add_img as f64 / 2.0,
        ctx.projection,
    )?;
    let center_p_source = Point3::new(center_source.x, center_source.y, 1.0);
    let center_p_target = ctx.h_add * center_p_source;
    let center_add_x = (center_p_target.x / center_p_target.z) + ctx.offset_x;
    let center_add_y = (center_p_target.y / center_p_target.z) + ctx.offset_y;

    let center_overlap_x = (min_ox + max_ox) as f64 / 2.0;
    let center_overlap_y = (min_oy + max_oy) as f64 / 2.0;

    let dx = center_add_x - center_overlap_x;
    let dy = center_add_y - center_overlap_y;

    if dx.abs() > dy.abs() {
        println!("    - Overlap is vertical. Finding vertical seam...");
        let seam = find_pairwise_seam_dp(
            ctx,
            SeamOrientation::Vertical,
            min_ox,
            max_ox,
            min_oy,
            max_oy,
        );
        if let Some(active) = seam.get(min_oy as usize..=max_oy as usize) {
            let minimum = active.iter().copied().min().unwrap_or_default();
            let maximum = active.iter().copied().max().unwrap_or_default();
            println!("    - Vertical seam range: x={minimum}..{maximum}");
        }
        Some(SeamInfo {
            orientation: SeamOrientation::Vertical,
            coords: seam,
            dx,
            dy,
            min_x: min_ox,
            max_x: max_ox,
            min_y: min_oy,
            max_y: max_oy,
        })
    } else {
        println!("    - Overlap is horizontal. Finding horizontal seam...");
        let seam = find_pairwise_seam_dp(
            ctx,
            SeamOrientation::Horizontal,
            min_ox,
            max_ox,
            min_oy,
            max_oy,
        );
        if let Some(active) = seam.get(min_ox as usize..=max_ox as usize) {
            let minimum = active.iter().copied().min().unwrap_or_default();
            let maximum = active.iter().copied().max().unwrap_or_default();
            println!("    - Horizontal seam range: y={minimum}..{maximum}");
        }
        Some(SeamInfo {
            orientation: SeamOrientation::Horizontal,
            coords: seam,
            dx,
            dy,
            min_x: min_ox,
            max_x: max_ox,
            min_y: min_oy,
            max_y: max_oy,
        })
    }
}

pub(crate) fn seam_energy_at(
    ctx: &SeamContext,
    h_add_inv: &Matrix3<f64>,
    x: u32,
    y: u32,
) -> Option<f64> {
    if ctx.pano_mask.get_pixel(x, y)[0] == 0 {
        return None;
    }
    let target = Point3::new(x as f64 - ctx.offset_x, y as f64 - ctx.offset_y, 1.0);
    let source = map_target_to_source(h_add_inv, target, ctx.img_to_add_info, ctx.projection)?;
    if source.x < 0.0
        || source.y < 0.0
        || source.x >= ctx.img_to_add.width() as f64 - 1.0
        || source.y >= ctx.img_to_add.height() as f64 - 1.0
        || !source_sample_has_support(ctx.img_to_add, source.x, source.y)
    {
        return None;
    }
    let base = ctx.pano.get_pixel(x, y);
    let candidate = apply_exposure_compensation(
        get_interpolated_pixel(ctx.img_to_add, source.x, source.y),
        ctx.exposure,
        x,
        y,
    );
    Some(
        ((base[0] as f64 - candidate[0] as f64).powi(2)
            + (base[1] as f64 - candidate[1] as f64).powi(2)
            + (base[2] as f64 - candidate[2] as f64).powi(2))
        .sqrt(),
    )
}

pub(crate) fn find_pairwise_seam_dp(
    ctx: &SeamContext,
    orientation: SeamOrientation,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
) -> Vec<i32> {
    pub(crate) const MAX_GRID_DIMENSION: u32 = 2_400;
    let (along_min, along_max, cross_min, cross_max, output_length) = match orientation {
        SeamOrientation::Vertical => (min_y, max_y, min_x, max_x, ctx.out_height),
        SeamOrientation::Horizontal => (min_x, max_x, min_y, max_y, ctx.out_width),
    };
    if along_min > along_max || cross_min > cross_max {
        return Vec::new();
    }

    let along_span = along_max - along_min;
    let cross_span = cross_max - cross_min;
    let step = along_span
        .max(cross_span)
        .div_ceil(MAX_GRID_DIMENSION)
        .max(1);
    let along_count = along_span.div_ceil(step) as usize + 1;
    let cross_count = cross_span.div_ceil(step) as usize + 1;
    if along_count < 2 || cross_count < 2 {
        return Vec::new();
    }
    println!(
        "    - Seam search grid: {}x{} ({}px sampling)",
        cross_count, along_count, step
    );

    let coordinate = |minimum: u32, maximum: u32, index: usize| {
        minimum
            .saturating_add((index as u32).saturating_mul(step))
            .min(maximum)
    };
    let Some(h_add_inv) = ctx.h_add.try_inverse() else {
        return Vec::new();
    };
    let mut previous = vec![f64::INFINITY; cross_count];
    let mut current = vec![f64::INFINITY; cross_count];
    let mut predecessors = vec![i8::MAX; along_count * cross_count];
    let mut last_active_index = None;
    let mut last_active_costs = Vec::new();

    for along_index in 0..along_count {
        current.fill(f64::INFINITY);
        let has_previous_path = previous.iter().any(|cost| cost.is_finite());
        let along = coordinate(along_min, along_max, along_index);
        for cross_index in 0..cross_count {
            let cross = coordinate(cross_min, cross_max, cross_index);
            let (x, y) = match orientation {
                SeamOrientation::Vertical => (cross, along),
                SeamOrientation::Horizontal => (along, cross),
            };
            let Some(mut energy) = seam_energy_at(ctx, &h_add_inv, x, y) else {
                continue;
            };

            // Discourage paths that merely trace a warped image border. Such paths make
            // rectangular exposure changes visible even when the geometry is correct.
            let edge_distance = cross_index.min(cross_count - 1 - cross_index) as f64;
            energy += (6.0 - edge_distance).max(0.0) * 0.01;

            if !has_previous_path {
                current[cross_index] = energy;
                predecessors[along_index * cross_count + cross_index] = 2;
                continue;
            }
            let first_neighbor = cross_index.saturating_sub(1);
            let last_neighbor = (cross_index + 1).min(cross_count - 1);
            let mut best_previous = f64::INFINITY;
            let mut best_index = cross_index;
            for (previous_index, &previous_cost) in previous
                .iter()
                .enumerate()
                .take(last_neighbor + 1)
                .skip(first_neighbor)
            {
                if previous_cost < best_previous {
                    best_previous = previous_cost;
                    best_index = previous_index;
                }
            }
            if best_previous.is_finite() {
                current[cross_index] = best_previous + energy;
                predecessors[along_index * cross_count + cross_index] =
                    (best_index as i32 - cross_index as i32) as i8;
            }
        }
        std::mem::swap(&mut previous, &mut current);
        if previous.iter().any(|cost| cost.is_finite()) {
            last_active_index = Some(along_index);
            last_active_costs.clone_from(&previous);
        }
    }

    let Some(last_active_index) = last_active_index else {
        return Vec::new();
    };
    let Some((mut current_cross, _)) = last_active_costs
        .iter()
        .enumerate()
        .filter(|(_, cost)| cost.is_finite())
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return Vec::new();
    };
    let mut sampled_path = vec![0usize; along_count];
    sampled_path[last_active_index] = current_cross;
    let mut first_active_index = last_active_index;
    for along_index in (1..=last_active_index).rev() {
        let predecessor = predecessors[along_index * cross_count + current_cross];
        if predecessor == 2 {
            first_active_index = along_index;
            break;
        }
        if predecessor == i8::MAX {
            return Vec::new();
        }
        current_cross =
            (current_cross as i32 + predecessor as i32).clamp(0, cross_count as i32 - 1) as usize;
        sampled_path[along_index - 1] = current_cross;
        first_active_index = along_index - 1;
    }
    for index in 0..first_active_index {
        sampled_path[index] = sampled_path[first_active_index];
    }
    for index in (last_active_index + 1)..along_count {
        sampled_path[index] = sampled_path[last_active_index];
    }

    let sampled_cross: Vec<f64> = sampled_path
        .iter()
        .map(|&index| coordinate(cross_min, cross_max, index) as f64)
        .collect();
    let mut seam = vec![sampled_cross[0].round() as i32; output_length as usize];
    for along in along_min..=along_max {
        let relative = along - along_min;
        let lower_index = ((relative / step) as usize).min(along_count - 2);
        let lower_along = coordinate(along_min, along_max, lower_index);
        let upper_along = coordinate(along_min, along_max, lower_index + 1);
        let interpolation = if upper_along == lower_along {
            0.0
        } else {
            (along - lower_along) as f64 / (upper_along - lower_along) as f64
        };
        let cross = sampled_cross[lower_index] * (1.0 - interpolation)
            + sampled_cross[lower_index + 1] * interpolation;
        seam[along as usize] = cross.round() as i32;
    }
    let last_cross = sampled_cross.last().copied().unwrap_or(sampled_cross[0]);
    for value in seam.iter_mut().skip(along_max as usize + 1) {
        *value = last_cross.round() as i32;
    }
    seam
}

pub fn warp_image_homography(
    source: &Rgb32FImage,
    homography: &Matrix3<f64>,
    width: u32,
    height: u32,
) -> Rgb32FImage {
    assert!(width > 0 && height > 0, "warp output must be non-empty");
    let mut buffer = vec![0.0f32; (width as usize) * (height as usize) * 3];
    buffer
        .par_chunks_mut(width as usize * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let mapped = homography * Point3::new(x as f64, y as f64, 1.0);
                let pixel = if mapped.z.abs() < 1e-8 {
                    Rgb([0.0, 0.0, 0.0])
                } else {
                    get_high_quality_interpolated_pixel(
                        source,
                        mapped.x / mapped.z,
                        mapped.y / mapped.z,
                    )
                };
                let base = x as usize * 3;
                row[base] = pixel[0];
                row[base + 1] = pixel[1];
                row[base + 2] = pixel[2];
            }
        });
    Rgb32FImage::from_raw(width, height, buffer)
        .expect("warp buffer dimensions must match output image")
}

pub(crate) fn get_interpolated_pixel(img: &Rgb32FImage, x: f64, y: f64) -> Rgb<f32> {
    let (width, height) = img.dimensions();
    let x_floor = x.floor() as u32;
    let y_floor = y.floor() as u32;
    if x_floor + 1 >= width || y_floor + 1 >= height || x < 0.0 || y < 0.0 {
        return *img.get_pixel(
            x.max(0.0).min(width as f64 - 1.0) as u32,
            y.max(0.0).min(height as f64 - 1.0) as u32,
        );
    }
    let dx = x - x_floor as f64;
    let dy = y - y_floor as f64;
    let p00 = img.get_pixel(x_floor, y_floor);
    let p10 = img.get_pixel(x_floor + 1, y_floor);
    let p01 = img.get_pixel(x_floor, y_floor + 1);
    let p11 = img.get_pixel(x_floor + 1, y_floor + 1);
    let mut final_pixel = [0.0; 3];
    for i in 0..3 {
        let c00 = p00[i] as f64;
        let c10 = p10[i] as f64;
        let c01 = p01[i] as f64;
        let c11 = p11[i] as f64;
        let top = c00 * (1.0 - dx) + c10 * dx;
        let bottom = c01 * (1.0 - dx) + c11 * dx;
        final_pixel[i] = top * (1.0 - dy) + bottom * dy;
    }
    Rgb([
        final_pixel[0] as f32,
        final_pixel[1] as f32,
        final_pixel[2] as f32,
    ])
}

pub(crate) fn cubic_sample(p0: f64, p1: f64, p2: f64, p3: f64, amount: f64) -> f64 {
    p1 + 0.5
        * amount
        * (p2 - p0
            + amount * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + amount * (3.0 * (p1 - p2) + p3 - p0)))
}

/// Return whether a source coordinate has actual image support.
///
/// Warped virtual tiles are stored in a rectangular RGB buffer even when the
/// projective footprint is triangular or otherwise clipped.  Those uncovered
/// corners are represented by exact black pixels, so a rectangle-only bounds
/// check would incorrectly feed them into the seam and its low-frequency
/// pyramid.  Real artwork may contain very dark strokes, but a decoded image
/// sample still has non-zero energy in at least one channel; requiring a small
/// positive floor keeps those samples while rejecting the transparent zero
/// support used by the renderer.
#[inline]
pub(crate) fn source_sample_has_support(image: &Rgb32FImage, x: f64, y: f64) -> bool {
    pub(crate) const SUPPORT_EPSILON: f32 = 1e-6;
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x >= width as f64
        || y >= height as f64
    {
        return false;
    }
    // Use the nearest source sample rather than the interpolated value.  A
    // cubic sample near a transparent corner can be non-zero solely because
    // its support straddles the valid footprint; the nearest source pixel is
    // the stable ownership signal and avoids re-introducing that corner.
    let source_x = (x.floor() as u32).min(width - 1);
    let source_y = (y.floor() as u32).min(height - 1);
    let pixel = image.get_pixel(source_x, source_y);
    pixel.0.iter().all(|value| value.is_finite())
        && pixel.0.iter().any(|value| *value > SUPPORT_EPSILON)
}

pub(crate) fn get_high_quality_interpolated_pixel(img: &Rgb32FImage, x: f64, y: f64) -> Rgb<f32> {
    let (width, height) = img.dimensions();
    if width < 4
        || height < 4
        || x < 1.0
        || y < 1.0
        || x >= width as f64 - 2.0
        || y >= height as f64 - 2.0
    {
        return get_interpolated_pixel(img, x, y);
    }

    let x_floor = x.floor() as i32;
    let y_floor = y.floor() as i32;
    let amount_x = x - x_floor as f64;
    let amount_y = y - y_floor as f64;
    if amount_x.abs() < 1e-9 && amount_y.abs() < 1e-9 {
        return *img.get_pixel(x_floor as u32, y_floor as u32);
    }

    let mut output = [0.0f32; 3];
    for (channel, output_channel) in output.iter_mut().enumerate() {
        let mut rows = [0.0f64; 4];
        let mut local_min = f64::INFINITY;
        let mut local_max = f64::NEG_INFINITY;
        for (row_index, sample_y) in ((y_floor - 1)..=(y_floor + 2)).enumerate() {
            let mut samples = [0.0f64; 4];
            for (column_index, sample_x) in ((x_floor - 1)..=(x_floor + 2)).enumerate() {
                let value = img.get_pixel(sample_x as u32, sample_y as u32)[channel] as f64;
                samples[column_index] = value;
                local_min = local_min.min(value);
                local_max = local_max.max(value);
            }
            rows[row_index] =
                cubic_sample(samples[0], samples[1], samples[2], samples[3], amount_x);
        }
        *output_channel = cubic_sample(rows[0], rows[1], rows[2], rows[3], amount_y)
            .clamp(local_min, local_max) as f32;
    }
    Rgb(output)
}
