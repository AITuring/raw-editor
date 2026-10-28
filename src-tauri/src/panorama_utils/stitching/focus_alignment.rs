//! Full-resolution focus-layer alignment.
//!
//! Patch scores, translation candidates, native station-frame registration and
//! foreground alignment of a rendered layer to the existing canvas.

use super::super::coordinates::Projection;
use crate::panorama_utils::shared::FocusLayerWarp;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Rgb;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point2;
use nalgebra::Point3;
use rayon::prelude::*;

use super::{
    FOCUS_FOREGROUND_REFINEMENT_MAX_SHIFT, FOCUS_FULL_RES_ALIGNMENT_GRID_SIZE,
    FOCUS_FULL_RES_ALIGNMENT_MAX_FOREGROUND_FRACTION, FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT,
    FOCUS_FULL_RES_ALIGNMENT_MIN_ENERGY, FOCUS_FULL_RES_ALIGNMENT_MIN_MARGIN,
    FOCUS_FULL_RES_ALIGNMENT_MIN_NCC, FOCUS_FULL_RES_ALIGNMENT_MIN_PATCHES,
    FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS, FocusForegroundMove, FocusSourceSampling,
    RenderedFocusLayer,
};

pub(super) fn focus_rgb_luma_at(image: &Rgb32FImage, x: i32, y: i32) -> Option<f64> {
    let (width, height) = image.dimensions();
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        return None;
    }
    let pixel = image.get_pixel(x as u32, y as u32);
    pixel
        .0
        .iter()
        .all(|value| value.is_finite())
        .then_some(pixel[0] as f64 * 0.299 + pixel[1] as f64 * 0.587 + pixel[2] as f64 * 0.114)
}

pub(super) fn focus_alignment_gradient_at(
    image: &Rgb32FImage,
    mask: &GrayImage,
    x: i32,
    y: i32,
) -> Option<(f64, f64)> {
    let (width, height) = image.dimensions();
    if x < 1 || y < 1 || x + 1 >= width as i32 || y + 1 >= height as i32 {
        return None;
    }
    let valid =
        |sample_x: i32, sample_y: i32| mask.get_pixel(sample_x as u32, sample_y as u32)[0] > 0;
    if !valid(x, y) || !valid(x - 1, y) || !valid(x + 1, y) || !valid(x, y - 1) || !valid(x, y + 1)
    {
        return None;
    }
    let horizontal = focus_rgb_luma_at(image, x + 1, y)? - focus_rgb_luma_at(image, x - 1, y)?;
    let vertical = focus_rgb_luma_at(image, x, y + 1)? - focus_rgb_luma_at(image, x, y - 1)?;
    Some((horizontal, vertical))
}

pub(super) fn focus_full_resolution_patch_score(
    candidate: &RenderedFocusLayer,
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
    center_x: i32,
    center_y: i32,
    delta_x: i32,
    delta_y: i32,
    radius: i32,
) -> Option<(f64, f64, f64)> {
    let (candidate_width, candidate_height) = candidate.image.dimensions();
    let mut candidate_horizontal_sum = 0.0;
    let mut candidate_vertical_sum = 0.0;
    let mut merged_horizontal_sum = 0.0;
    let mut merged_vertical_sum = 0.0;
    let mut candidate_squared_sum = 0.0;
    let mut merged_squared_sum = 0.0;
    let mut product_sum = 0.0;
    let mut sample_count = 0usize;
    let mut foreground_count = 0usize;
    let mut patch_count = 0usize;
    for offset_y in -radius..=radius {
        for offset_x in -radius..=radius {
            let local_x = center_x + offset_x;
            let local_y = center_y + offset_y;
            if local_x >= 0
                && local_y >= 0
                && local_x < candidate_width as i32
                && local_y < candidate_height as i32
            {
                patch_count += 1;
                if candidate
                    .foreground_mask
                    .get_pixel(local_x as u32, local_y as u32)[0]
                    > 0
                {
                    foreground_count += 1;
                }
            }
            let Some((candidate_horizontal, candidate_vertical)) =
                focus_alignment_gradient_at(&candidate.image, &candidate.mask, local_x, local_y)
            else {
                continue;
            };
            let merged_x = candidate.left as i32 + local_x + delta_x;
            let merged_y = candidate.top as i32 + local_y + delta_y;
            let Some((merged_horizontal, merged_vertical)) =
                focus_alignment_gradient_at(merged, merged_mask, merged_x, merged_y)
            else {
                continue;
            };
            candidate_horizontal_sum += candidate_horizontal;
            candidate_vertical_sum += candidate_vertical;
            merged_horizontal_sum += merged_horizontal;
            merged_vertical_sum += merged_vertical;
            candidate_squared_sum += candidate_horizontal * candidate_horizontal
                + candidate_vertical * candidate_vertical;
            merged_squared_sum +=
                merged_horizontal * merged_horizontal + merged_vertical * merged_vertical;
            product_sum +=
                candidate_horizontal * merged_horizontal + candidate_vertical * merged_vertical;
            sample_count += 1;
        }
    }
    let minimum_samples = (((radius * 2 + 1) * (radius * 2 + 1)) as f64 * 0.45)
        .round()
        .max(48.0) as usize;
    if sample_count < minimum_samples || patch_count == 0 {
        return None;
    }
    let sample_count_f64 = sample_count as f64;
    let candidate_variance = candidate_squared_sum
        - (candidate_horizontal_sum * candidate_horizontal_sum
            + candidate_vertical_sum * candidate_vertical_sum)
            / sample_count_f64;
    let merged_variance = merged_squared_sum
        - (merged_horizontal_sum * merged_horizontal_sum
            + merged_vertical_sum * merged_vertical_sum)
            / sample_count_f64;
    if candidate_variance <= f64::EPSILON || merged_variance <= f64::EPSILON {
        return None;
    }
    let covariance = product_sum
        - (candidate_horizontal_sum * merged_horizontal_sum
            + candidate_vertical_sum * merged_vertical_sum)
            / sample_count_f64;
    let score = covariance / (candidate_variance * merged_variance).sqrt();
    let candidate_energy = (candidate_variance / sample_count_f64).sqrt();
    let merged_energy = (merged_variance / sample_count_f64).sqrt();
    let foreground_fraction = foreground_count as f64 / patch_count as f64;
    (score.is_finite() && candidate_energy.is_finite() && merged_energy.is_finite()).then_some((
        score,
        candidate_energy.min(merged_energy),
        foreground_fraction,
    ))
}

// Build only the two bounded patches used by one correspondence, rather than
// a second full-resolution copy of both 60MP layers. Gaussian decimation samples
// even pixel centers, so multiplying a displacement by two is exact between
// levels. A stride over native pixels is not a pyramid: it can skip an entire
// narrow correlation peak.
pub(super) fn focus_alignment_patch_pyramid(
    image: &Rgb32FImage,
    mask: &GrayImage,
    foreground: Option<&GrayImage>,
    center_x: i32,
    center_y: i32,
    radius: i32,
    levels: usize,
) -> Vec<RenderedFocusLayer> {
    let size = (2 * radius + 1) as u32;
    let mut first = RenderedFocusLayer {
        image: Rgb32FImage::new(size, size),
        mask: GrayImage::new(size, size),
        foreground_mask: GrayImage::new(size, size),
        relaxed_foreground_mask: GrayImage::new(0, 0),
        left: 0,
        top: 0,
    };
    for y in 0..size {
        for x in 0..size {
            let sx = center_x + x as i32 - radius;
            let sy = center_y + y as i32 - radius;
            if sx < 0 || sy < 0 || sx >= image.width() as i32 || sy >= image.height() as i32 {
                continue;
            }
            first
                .image
                .put_pixel(x, y, *image.get_pixel(sx as u32, sy as u32));
            first
                .mask
                .put_pixel(x, y, *mask.get_pixel(sx as u32, sy as u32));
            if let Some(foreground) = foreground {
                first
                    .foreground_mask
                    .put_pixel(x, y, *foreground.get_pixel(sx as u32, sy as u32));
            }
        }
    }
    let mut pyramid = vec![first];
    for _ in 0..levels {
        let previous = pyramid.last().unwrap();
        let width = (previous.image.width() + 1) / 2;
        let height = (previous.image.height() + 1) / 2;
        let mut next = RenderedFocusLayer {
            image: Rgb32FImage::new(width, height),
            mask: GrayImage::new(width, height),
            foreground_mask: GrayImage::new(width, height),
            relaxed_foreground_mask: GrayImage::new(0, 0),
            left: 0,
            top: 0,
        };
        for y in 0..height {
            for x in 0..width {
                let mut pixel = [0.0f32; 3];
                let mut valid = true;
                let mut foreground = 0;
                for oy in -1i32..=1 {
                    for ox in -1i32..=1 {
                        let sx =
                            (2 * x as i32 + ox).clamp(0, previous.image.width() as i32 - 1) as u32;
                        let sy =
                            (2 * y as i32 + oy).clamp(0, previous.image.height() as i32 - 1) as u32;
                        let weight = (if ox == 0 { 2.0 } else { 1.0 })
                            * (if oy == 0 { 2.0 } else { 1.0 })
                            / 16.0;
                        let sample = previous.image.get_pixel(sx, sy);
                        for channel in 0..3 {
                            pixel[channel] += sample[channel] * weight;
                        }
                        valid &= previous.mask.get_pixel(sx, sy)[0] > 0;
                        foreground |= previous.foreground_mask.get_pixel(sx, sy)[0];
                    }
                }
                next.image.put_pixel(x, y, Rgb(pixel));
                next.mask
                    .put_pixel(x, y, image::Luma([u8::from(valid) * 255]));
                next.foreground_mask
                    .put_pixel(x, y, image::Luma([foreground]));
            }
        }
        pyramid.push(next);
    }
    pyramid
}

pub(super) fn focus_patch_translation_candidates(
    candidate: &RenderedFocusLayer,
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
    center_x: i32,
    center_y: i32,
    max_shift: i32,
) -> Vec<(f64, f64, i32, i32)> {
    let mut levels = 0;
    while (max_shift >> levels) > 24 {
        levels += 1;
    }
    let scale = 1 << levels;
    let patch_radius = FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS;
    let candidate_radius = (patch_radius + 2) * scale;
    let merged_radius = candidate_radius + (max_shift + scale - 1) / scale * scale;
    let mut candidate_pyramid = focus_alignment_patch_pyramid(
        &candidate.image,
        &candidate.mask,
        Some(&candidate.foreground_mask),
        center_x,
        center_y,
        candidate_radius,
        levels,
    );
    let merged_pyramid = focus_alignment_patch_pyramid(
        merged,
        merged_mask,
        None,
        candidate.left as i32 + center_x,
        candidate.top as i32 + center_y,
        merged_radius,
        levels,
    );
    let mut beam: Vec<(f64, f64, i32, i32)> = Vec::new();
    for level in (0..=levels).rev() {
        let scale = 1 << level;
        let limit = (max_shift + scale - 1) / scale;
        let candidate_patch = &mut candidate_pyramid[level];
        candidate_patch.left = ((merged_radius - candidate_radius) / scale) as u32;
        candidate_patch.top = candidate_patch.left;
        let merged_patch = &merged_pyramid[level];
        let mut offsets = Vec::new();
        if beam.is_empty() && level == levels {
            for dy in -limit..=limit {
                for dx in -limit..=limit {
                    offsets.push((dx, dy));
                }
            }
        } else {
            for &(_, _, coarse_x, coarse_y) in &beam {
                // Coarse focus texture can shift by several pixels after
                // Gaussian decimation. Keep a broad native refinement window
                // around each of the few coarse seeds; a tiny ±3 window can
                // permanently exclude the true subpixel peak.
                for dy in -16..=16 {
                    for dx in -16..=16 {
                        let x = coarse_x * 2 + dx;
                        let y = coarse_y * 2 + dy;
                        if x.abs() <= limit && y.abs() <= limit {
                            offsets.push((x, y));
                        }
                    }
                }
            }
        }
        // The baseline must always be measured, including when the search
        // bounds are not an exact multiple of a sampling stride.
        offsets.push((0, 0));
        offsets.sort_unstable();
        offsets.dedup();
        let mut scored = offsets
            .into_iter()
            .filter_map(|(dx, dy)| {
                let (score, energy, foreground_fraction) = focus_full_resolution_patch_score(
                    candidate_patch,
                    &merged_patch.image,
                    &merged_patch.mask,
                    candidate_radius / scale,
                    candidate_radius / scale,
                    dx,
                    dy,
                    patch_radius,
                )?;
                // Coarse levels only propose locations. Native pixels supply the
                // final quality gate, allowing a low-contrast coarse feature to
                // lead to a sharply localized, well-supported native peak.
                (foreground_fraction <= FOCUS_FULL_RES_ALIGNMENT_MAX_FOREGROUND_FRACTION
                    && energy > 1e-6)
                    .then_some((score, energy, dx, dy))
            })
            .collect::<Vec<_>>();
        scored.sort_unstable_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| (a.2.abs() + a.3.abs()).cmp(&(b.2.abs() + b.3.abs())))
        });
        if level == 0 {
            return scored;
        }
        beam.clear();
        for entry in scored {
            if beam.iter().all(|previous| {
                (previous.2 - entry.2).abs() > 2 || (previous.3 - entry.3).abs() > 2
            }) {
                beam.push(entry);
                if beam.len() == 4 {
                    break;
                }
            }
        }
        if beam.is_empty() {
            return Vec::new();
        }
    }
    Vec::new()
}

/// Run the Intra_Station_Registrar over one already rendered layer (需求 2.2–2.9).
///
/// The measurement itself is `mosaic::register_station_frame_native`, the same
/// code the mosaic comparison path uses; this only reconstructs the pose the
/// caller actually rendered with.  `applied_translation` is the integer shift
/// `translate_rendered_focus_layer` has already put on those pixels, and a canvas
/// shift is a translation in the projection plane, so pre-multiplying the
/// station homography with it reproduces the layer exactly.
#[allow(clippy::too_many_arguments)]
pub(super) fn register_station_frame_native_layer(
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
    image: &ImageInfo,
    source_image: &Rgb32FImage,
    homography: &Matrix3<f64>,
    projection: Projection,
    offset: (f64, f64),
    applied_translation: (i32, i32),
    candidate: &RenderedFocusLayer,
    anchor_long_side: u32,
    anchor_to_station_inverse: Option<Matrix3<f64>>,
    focus_warp: Option<&FocusLayerWarp>,
) -> Option<crate::panorama_utils::mosaic::StationFrameRegistration> {
    let (candidate_width, candidate_height) = candidate.image.dimensions();
    if candidate_width < 2 || candidate_height < 2 {
        return None;
    }
    let mut translation = Matrix3::identity();
    translation[(0, 2)] = f64::from(applied_translation.0);
    translation[(1, 2)] = f64::from(applied_translation.1);
    let rendered_pose = translation * homography;
    // 需求 2.1: every record is expressed in the anchor frame's coordinates.
    let into_anchor = anchor_to_station_inverse
        .map(|inverse| inverse * rendered_pose)
        .unwrap_or_else(Matrix3::identity);
    let sampling = FocusSourceSampling::new(image, homography, projection, focus_warp);
    let native_mapping = |target: Point3<f64>| {
        sampling.source_point(Point3::new(
            target.x - f64::from(applied_translation.0),
            target.y - f64::from(applied_translation.1),
            target.z,
        ))
    };
    crate::panorama_utils::mosaic::register_station_frame_native_with_mapping(
        merged,
        merged_mask,
        image,
        source_image,
        &rendered_pose,
        projection,
        offset,
        (
            candidate.left,
            candidate.left + candidate_width - 1,
            candidate.top,
            candidate.top + candidate_height - 1,
        ),
        anchor_long_side,
        into_anchor,
        focus_warp
            .filter(|warp| !warp.bands.is_empty())
            .map(|_| &native_mapping as &(dyn Fn(Point3<f64>) -> Option<Point2<f64>> + Sync)),
    )
}

/// Apply a measured native displacement field to an already rendered layer
/// (需求 2.3–2.7).
///
/// The field is what [`crate::panorama_utils::mosaic::register_station_frame_native`] accepted,
/// expressed as a canvas-pixel offset to add before sampling, so resampling the
/// layer at that offset moves its pixels by exactly the accepted correction. A
/// pixel whose bilinear support is not fully covered loses coverage instead of
/// borrowing an invented neighbour.
pub(super) fn warp_rendered_focus_layer(
    candidate: RenderedFocusLayer,
    displacement: &crate::panorama_utils::mosaic::NativeResidualDisplacement,
) -> RenderedFocusLayer {
    let (width, height) = candidate.image.dimensions();
    if width == 0 || height == 0 {
        return candidate;
    }
    let row_pixels = width as usize * 3;
    let mut pixels = vec![0.0f32; row_pixels * height as usize];
    let mut mask = vec![0u8; width as usize * height as usize];
    let mut foreground_mask = vec![0u8; width as usize * height as usize];
    let mut relaxed_foreground_mask = vec![0u8; width as usize * height as usize];
    pixels
        .par_chunks_mut(row_pixels)
        .zip(mask.par_chunks_mut(width as usize))
        .zip(foreground_mask.par_chunks_mut(width as usize))
        .zip(relaxed_foreground_mask.par_chunks_mut(width as usize))
        .enumerate()
        .for_each(|(y, (((row, mask_row), foreground_row), relaxed_row))| {
            for x in 0..width as usize {
                let [delta_x, delta_y] = displacement.at(
                    f64::from(candidate.left) + x as f64,
                    f64::from(candidate.top) + y as f64,
                );
                let sample_x = x as f64 + delta_x;
                let sample_y = y as f64 + delta_y;
                if !sample_x.is_finite() || !sample_y.is_finite() {
                    continue;
                }
                let x0 = sample_x.floor();
                let y0 = sample_y.floor();
                if x0 < 0.0
                    || y0 < 0.0
                    || x0 + 1.0 >= f64::from(width)
                    || y0 + 1.0 >= f64::from(height)
                {
                    continue;
                }
                let (x0, y0) = (x0 as u32, y0 as u32);
                if [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)]
                    .iter()
                    .any(|&(px, py)| candidate.mask.get_pixel(px, py)[0] == 0)
                {
                    continue;
                }
                let fx = (sample_x - f64::from(x0)) as f32;
                let fy = (sample_y - f64::from(y0)) as f32;
                for channel in 0..3 {
                    let at = |px: u32, py: u32| candidate.image.get_pixel(px, py)[channel];
                    let top = at(x0, y0) * (1.0 - fx) + at(x0 + 1, y0) * fx;
                    let bottom = at(x0, y0 + 1) * (1.0 - fx) + at(x0 + 1, y0 + 1) * fx;
                    row[x * 3 + channel] = top * (1.0 - fy) + bottom * fy;
                }
                mask_row[x] = 255;
                // The detected layer masks are categorical, so they follow
                // the nearest sample rather than an average of two classes.
                let nearest_x = (sample_x.round() as u32).min(width - 1);
                let nearest_y = (sample_y.round() as u32).min(height - 1);
                foreground_row[x] = candidate.foreground_mask.get_pixel(nearest_x, nearest_y)[0];
                relaxed_row[x] = candidate
                    .relaxed_foreground_mask
                    .get_pixel(nearest_x, nearest_y)[0];
            }
        });
    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(width, height, pixels).expect("warped layer dimensions"),
        mask: GrayImage::from_raw(width, height, mask).expect("warped mask dimensions"),
        foreground_mask: GrayImage::from_raw(width, height, foreground_mask)
            .expect("warped foreground dimensions"),
        relaxed_foreground_mask: GrayImage::from_raw(width, height, relaxed_foreground_mask)
            .expect("warped relaxed foreground dimensions"),
        left: candidate.left,
        top: candidate.top,
    }
}

pub(super) fn estimate_focus_layer_translation(
    candidate: &RenderedFocusLayer,
    merged: &Rgb32FImage,
    merged_mask: &GrayImage,
) -> Option<(i32, i32, f64, usize, usize)> {
    let (candidate_width, candidate_height) = candidate.image.dimensions();
    let (merged_width, merged_height) = merged.dimensions();
    if candidate_width < 2 * (FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS + 2) as u32
        || candidate_height < 2 * (FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS + 2) as u32
        || merged_width < 2
        || merged_height < 2
        || merged_mask.dimensions() != merged.dimensions()
    {
        return None;
    }

    let patch_radius = FOCUS_FULL_RES_ALIGNMENT_PATCH_RADIUS;
    let overlap_left = candidate.left.max(0).min(merged_width.saturating_sub(1)) as i32;
    let overlap_top = candidate.top.max(0).min(merged_height.saturating_sub(1)) as i32;
    let overlap_right = (candidate
        .left
        .saturating_add(candidate_width)
        .saturating_sub(1))
    .min(merged_width.saturating_sub(1)) as i32;
    let overlap_bottom = (candidate
        .top
        .saturating_add(candidate_height)
        .saturating_sub(1))
    .min(merged_height.saturating_sub(1)) as i32;
    if overlap_right - overlap_left < patch_radius * 2 + 2
        || overlap_bottom - overlap_top < patch_radius * 2 + 2
    {
        return None;
    }

    let grid_size = FOCUS_FULL_RES_ALIGNMENT_GRID_SIZE;
    let mut patch_centers = Vec::with_capacity((grid_size * grid_size) as usize);
    for grid_y in 0..grid_size {
        let global_y =
            overlap_top + ((overlap_bottom - overlap_top) * (grid_y + 1) / (grid_size + 1));
        for grid_x in 0..grid_size {
            let global_x =
                overlap_left + ((overlap_right - overlap_left) * (grid_x + 1) / (grid_size + 1));
            let local_x = global_x - candidate.left as i32;
            let local_y = global_y - candidate.top as i32;
            patch_centers.push((local_x, local_y));
        }
    }

    let mut patch_deltas = Vec::new();
    let image_short_side = candidate_width.min(candidate_height) as i32;
    let max_shift = FOCUS_FULL_RES_ALIGNMENT_MAX_SHIFT
        .min((image_short_side as f32 * 0.08).round() as i32)
        .max(1);
    for (center_x, center_y) in patch_centers.iter().copied() {
        // Record zero-shift NCC before applying the acceptance threshold. A
        // genuinely displaced patch commonly has low baseline correlation;
        // rejecting that baseline would make large corrections impossible.
        let current_score = focus_full_resolution_patch_score(
            candidate,
            merged,
            merged_mask,
            center_x,
            center_y,
            0,
            0,
            patch_radius,
        )
        .map(|value| value.0);
        let scores = focus_patch_translation_candidates(
            candidate,
            merged,
            merged_mask,
            center_x,
            center_y,
            max_shift,
        );
        let Some(&(best_score, energy, best_x, best_y)) = scores.first() else {
            continue;
        };
        let second_best = scores
            .iter()
            .find(|entry| (entry.2 - best_x).abs() > 2 || (entry.3 - best_y).abs() > 2)
            .map(|entry| entry.0)
            .unwrap_or(f64::NEG_INFINITY);
        if best_score < FOCUS_FULL_RES_ALIGNMENT_MIN_NCC
            || energy < FOCUS_FULL_RES_ALIGNMENT_MIN_ENERGY
            || best_score - second_best < FOCUS_FULL_RES_ALIGNMENT_MIN_MARGIN
            || (best_x != 0 || best_y != 0)
                && current_score
                    .is_some_and(|score| best_score - score < FOCUS_FULL_RES_ALIGNMENT_MIN_MARGIN)
        {
            continue;
        }
        patch_deltas.push((best_x, best_y, best_score));
    }
    if std::env::var_os("RAW_EDITOR_FOCUS_TRANSLATION_DIAGNOSTICS").is_some() {
        println!(
            "  - Focus translation diagnostics: candidate={}x{} patches={} accepted={}",
            candidate_width,
            candidate_height,
            patch_centers.len(),
            patch_deltas.len()
        );
    }
    if patch_deltas.is_empty() {
        return None;
    }

    let mut x_values = patch_deltas.iter().map(|(x, _, _)| *x).collect::<Vec<_>>();
    let mut y_values = patch_deltas.iter().map(|(_, y, _)| *y).collect::<Vec<_>>();
    x_values.sort_unstable();
    y_values.sort_unstable();
    let median_x = x_values[x_values.len() / 2];
    let median_y = y_values[y_values.len() / 2];
    let support = patch_deltas
        .iter()
        .filter(|(x, y, _)| (x - median_x).abs() <= 1 && (y - median_y).abs() <= 1)
        .count();
    let minimum_support = FOCUS_FULL_RES_ALIGNMENT_MIN_PATCHES;
    if support < minimum_support
        || (support as f64) < patch_deltas.len() as f64 * 0.5
        || (median_x == 0 && median_y == 0)
    {
        return None;
    }
    Some((
        median_x,
        median_y,
        patch_deltas
            .iter()
            .filter(|(x, y, _)| (x - median_x).abs() <= 1 && (y - median_y).abs() <= 1)
            .map(|(_, _, score)| score)
            .sum::<f64>()
            / support as f64,
        support,
        patch_deltas.len(),
    ))
}

pub(super) fn translate_rendered_focus_layer(
    candidate: RenderedFocusLayer,
    delta_x: i32,
    delta_y: i32,
    out_width: u32,
    out_height: u32,
) -> RenderedFocusLayer {
    if delta_x == 0 && delta_y == 0 {
        return candidate;
    }
    let (width, height) = candidate.image.dimensions();
    if width == 0 || height == 0 || out_width == 0 || out_height == 0 {
        return candidate;
    }
    let old_left = candidate.left as i64;
    let old_top = candidate.top as i64;
    let old_right = old_left + width as i64 - 1;
    let old_bottom = old_top + height as i64 - 1;
    let new_left = (old_left + i64::from(delta_x)).clamp(0, i64::from(out_width - 1));
    let new_top = (old_top + i64::from(delta_y)).clamp(0, i64::from(out_height - 1));
    let new_right = (old_right + i64::from(delta_x)).clamp(0, i64::from(out_width - 1));
    let new_bottom = (old_bottom + i64::from(delta_y)).clamp(0, i64::from(out_height - 1));
    if new_left > new_right || new_top > new_bottom {
        return RenderedFocusLayer {
            image: Rgb32FImage::new(0, 0),
            mask: GrayImage::new(0, 0),
            foreground_mask: GrayImage::new(0, 0),
            relaxed_foreground_mask: GrayImage::new(0, 0),
            left: 0,
            top: 0,
        };
    }
    let new_width = (new_right - new_left + 1) as u32;
    let new_height = (new_bottom - new_top + 1) as u32;
    let mut image_pixels = vec![0.0f32; new_width as usize * new_height as usize * 3];
    let mut mask_pixels = vec![0u8; new_width as usize * new_height as usize];
    let mut foreground_pixels = vec![0u8; new_width as usize * new_height as usize];
    let mut relaxed_pixels = vec![0u8; new_width as usize * new_height as usize];
    let source_image = candidate.image.as_raw();
    let source_mask = candidate.mask.as_raw();
    let source_foreground = candidate.foreground_mask.as_raw();
    let source_relaxed = candidate.relaxed_foreground_mask.as_raw();
    for source_y in 0..height as i64 {
        for source_x in 0..width as i64 {
            let destination_x = old_left + source_x + i64::from(delta_x);
            let destination_y = old_top + source_y + i64::from(delta_y);
            if destination_x < new_left
                || destination_x > new_right
                || destination_y < new_top
                || destination_y > new_bottom
            {
                continue;
            }
            let source_index = source_y as usize * width as usize + source_x as usize;
            let destination_index = (destination_y - new_top) as usize * new_width as usize
                + (destination_x - new_left) as usize;
            let source_start = source_index * 3;
            let destination_start = destination_index * 3;
            image_pixels[destination_start..destination_start + 3]
                .copy_from_slice(&source_image[source_start..source_start + 3]);
            mask_pixels[destination_index] = source_mask[source_index];
            foreground_pixels[destination_index] = source_foreground[source_index];
            relaxed_pixels[destination_index] = source_relaxed[source_index];
        }
    }
    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(new_width, new_height, image_pixels)
            .expect("translated focus layer image dimensions must match"),
        mask: GrayImage::from_raw(new_width, new_height, mask_pixels)
            .expect("translated focus layer mask dimensions must match"),
        foreground_mask: GrayImage::from_raw(new_width, new_height, foreground_pixels)
            .expect("translated focus foreground mask dimensions must match"),
        relaxed_foreground_mask: GrayImage::from_raw(new_width, new_height, relaxed_pixels)
            .expect("translated focus relaxed foreground mask dimensions must match"),
        left: new_left as u32,
        top: new_top as u32,
    }
}

pub(super) fn align_focus_foreground_layer_to_existing(
    candidate: RenderedFocusLayer,
    merged_foreground_mask: &GrayImage,
    refinement: &mut Option<FocusForegroundMove>,
) -> RenderedFocusLayer {
    let (layer_width, layer_height) = candidate.foreground_mask.dimensions();
    if layer_width == 0 || layer_height == 0 || merged_foreground_mask.width() == 0 {
        return candidate;
    }

    let foreground = candidate.foreground_mask.as_raw();
    let foreground_count = foreground.iter().filter(|value| **value > 0).count();
    let existing_count = merged_foreground_mask
        .as_raw()
        .iter()
        .filter(|value| **value > 0)
        .count();
    if foreground_count < 32 || existing_count < 32 {
        return candidate;
    }

    // Use a bounded, deterministic set of mask anchors. This makes the
    // refinement cheap even for a full 60MP source while retaining enough
    // support to distinguish a real overlap from an accidental nearby edge.
    let anchor_stride = (foreground_count / 2_048).max(1);
    let mut anchors = Vec::with_capacity(foreground_count.min(2_048));
    let mut foreground_index = 0usize;
    for (index, value) in foreground.iter().enumerate() {
        if *value == 0 {
            continue;
        }
        if foreground_index.is_multiple_of(anchor_stride) {
            let local_x = (index % layer_width as usize) as u32;
            let local_y = (index / layer_width as usize) as u32;
            anchors.push((candidate.left + local_x, candidate.top + local_y));
        }
        foreground_index += 1;
    }
    if anchors.len() < 32 {
        return candidate;
    }

    let base_width = merged_foreground_mask.width() as i32;
    let base_height = merged_foreground_mask.height() as i32;
    let base_pixels = merged_foreground_mask.as_raw();
    let search_radius = ((layer_width.max(layer_height) as f32) * 0.01)
        .round()
        .clamp(4.0, FOCUS_FOREGROUND_REFINEMENT_MAX_SHIFT as f32) as i32;
    let coarse_step = (search_radius / 24).max(1);
    let hit_count = |delta_x: i32, delta_y: i32| -> usize {
        anchors
            .iter()
            .filter(|&&(x, y)| {
                let x = x as i32 + delta_x;
                let y = y as i32 + delta_y;
                x >= 0
                    && y >= 0
                    && x < base_width
                    && y < base_height
                    && base_pixels[y as usize * base_width as usize + x as usize] > 0
            })
            .count()
    };

    let current_hits = hit_count(0, 0);
    let mut best_delta = (0i32, 0i32);
    let mut best_hits = current_hits;
    for delta_y in (-search_radius..=search_radius).step_by(coarse_step as usize) {
        for delta_x in (-search_radius..=search_radius).step_by(coarse_step as usize) {
            let hits = hit_count(delta_x, delta_y);
            if hits > best_hits {
                best_hits = hits;
                best_delta = (delta_x, delta_y);
            }
        }
    }
    // Refine around the best coarse translation at pixel precision. The
    // foreground mask is only a registration cue; the final source pixels are
    // still selected by the focus score below.
    if coarse_step > 1 {
        let (coarse_x, coarse_y) = best_delta;
        let fine_min_y = (coarse_y - coarse_step + 1).max(-search_radius);
        let fine_max_y = (coarse_y + coarse_step - 1).min(search_radius);
        let fine_min_x = (coarse_x - coarse_step + 1).max(-search_radius);
        let fine_max_x = (coarse_x + coarse_step - 1).min(search_radius);
        for delta_y in fine_min_y..=fine_max_y {
            for delta_x in fine_min_x..=fine_max_x {
                let hits = hit_count(delta_x, delta_y);
                if hits > best_hits {
                    best_hits = hits;
                    best_delta = (delta_x, delta_y);
                }
            }
        }
    }

    let minimum_gain = (anchors.len() / 20).max(12);
    if best_delta == (0, 0)
        || best_hits < current_hits.saturating_add(minimum_gain)
        || best_hits < anchors.len() / 20
    {
        return candidate;
    }

    let (delta_x, delta_y) = best_delta;
    let mut image_pixels = candidate.image.into_raw();
    let mut candidate_mask = candidate.mask.into_raw();
    let mut foreground_mask = vec![0u8; foreground.len()];
    let relaxed_foreground = candidate.relaxed_foreground_mask.as_raw();
    let mut relaxed_foreground_mask = vec![0u8; foreground.len()];
    let mut moved_pixels = Vec::with_capacity(foreground_count);
    for (index, value) in foreground.iter().enumerate() {
        if *value == 0 {
            continue;
        }
        let local_x = (index % layer_width as usize) as i32;
        let local_y = (index / layer_width as usize) as i32;
        let destination_x = local_x + delta_x;
        let destination_y = local_y + delta_y;
        if destination_x < 0
            || destination_y < 0
            || destination_x >= layer_width as i32
            || destination_y >= layer_height as i32
        {
            continue;
        }
        let destination_index =
            destination_y as usize * layer_width as usize + destination_x as usize;
        let source_start = index * 3;
        moved_pixels.push((
            destination_index,
            [
                image_pixels[source_start],
                image_pixels[source_start + 1],
                image_pixels[source_start + 2],
            ],
        ));
        candidate_mask[index] = 0;
        foreground_mask[destination_index] = 255;
        if relaxed_foreground[index] > 0 {
            relaxed_foreground_mask[destination_index] = 255;
        }
    }
    *refinement = Some(FocusForegroundMove {
        delta: best_delta,
        left: candidate.left,
        top: candidate.top,
        destinations: GrayImage::from_raw(layer_width, layer_height, foreground_mask.clone())
            .expect("foreground refinement dimensions"),
    });
    for (destination_index, pixel) in moved_pixels {
        let destination_start = destination_index * 3;
        image_pixels[destination_start..destination_start + 3].copy_from_slice(&pixel);
        candidate_mask[destination_index] = 255;
    }

    println!(
        "  - Focus foreground mask refinement: delta=({delta_x},{delta_y}), overlap={current_hits}->{best_hits} anchors={}",
        anchors.len()
    );
    RenderedFocusLayer {
        image: Rgb32FImage::from_raw(layer_width, layer_height, image_pixels)
            .expect("aligned focus layer image dimensions must match"),
        mask: GrayImage::from_raw(layer_width, layer_height, candidate_mask)
            .expect("aligned focus layer mask dimensions must match"),
        foreground_mask: GrayImage::from_raw(layer_width, layer_height, foreground_mask)
            .expect("aligned focus foreground mask dimensions must match"),
        relaxed_foreground_mask: GrayImage::from_raw(
            layer_width,
            layer_height,
            relaxed_foreground_mask,
        )
        .expect("aligned focus relaxed foreground mask dimensions must match"),
        left: candidate.left,
        top: candidate.top,
    }
}
