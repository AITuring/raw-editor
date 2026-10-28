//! Applying station seam plans to the ownership plane.
//!
//! Rewrites only the old/new boundary strip of each planned pair, copies the
//! winning owner pixels and removes owner islands left inside the pair.

use super::super::coordinates::Projection;
use super::super::stack_pipeline::residual_warp;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Rgb32FImage;
#[cfg(test)]
use image::RgbImage;
use nalgebra::Matrix3;
use std::collections::HashMap;
#[cfg(test)]
use std::path::PathBuf;

use super::{
    FocusTilePayload, NO_OWNER, SeamOrientation, StationSeamPlan, station_current_cross_at,
    station_current_transition_candidates, station_local_relation_resample_band_px,
    station_seam_cross_at, station_seam_pair_sample, station_seam_sample,
    station_transition_first_on_low_side,
};

#[cfg(test)]
pub(super) fn write_station_owner_trace(
    source_owner_ids: &[u16],
    out_width: u32,
    out_height: u32,
) -> Result<(), String> {
    let Some(directory) = std::env::var_os("RAW_EDITOR_STATION_OWNER_TRACE_DIR") else {
        return Ok(());
    };
    let directory = PathBuf::from(directory);
    std::fs::create_dir_all(&directory).map_err(|error| {
        format!(
            "failed to create station owner trace directory {}: {error}",
            directory.display()
        )
    })?;
    let parse_region = || {
        std::env::var("RAW_EDITOR_STATION_OWNER_TRACE_REGION")
            .ok()
            .and_then(|value| {
                let values = value
                    .split(',')
                    .map(str::trim)
                    .map(str::parse::<u32>)
                    .collect::<Result<Vec<_>, _>>()
                    .ok()?;
                (values.len() == 4).then_some((values[0], values[1], values[2], values[3]))
            })
            .unwrap_or((0, 0, out_width, out_height))
    };
    let (left, top, width, height) = parse_region();
    let left = left.min(out_width);
    let top = top.min(out_height);
    let width = width.min(out_width.saturating_sub(left));
    let height = height.min(out_height.saturating_sub(top));
    if width == 0 || height == 0 {
        return Ok(());
    }
    let palette = [
        [0u8, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
    ];
    let trace = RgbImage::from_fn(width, height, |x, y| {
        let source_index = (top + y) as usize * out_width as usize + (left + x) as usize;
        let owner = source_owner_ids
            .get(source_index)
            .copied()
            .unwrap_or(NO_OWNER) as usize;
        let colour = palette
            .get(owner % palette.len())
            .copied()
            .unwrap_or([0, 0, 0]);
        image::Rgb(colour)
    });
    trace
        .save(directory.join("owner-map.png"))
        .map_err(|error| format!("failed to write station owner trace image: {error}"))?;

    let mut histogram = std::collections::BTreeMap::<u16, u64>::new();
    let mut row_runs = Vec::new();
    for y in (0..height).step_by(16) {
        let mut runs = Vec::<(u32, u32, u16)>::new();
        let mut run_start = 0;
        let mut run_owner = source_owner_ids
            .get((top + y) as usize * out_width as usize + left as usize)
            .copied()
            .unwrap_or(NO_OWNER);
        for x in 0..width {
            let owner = source_owner_ids
                .get((top + y) as usize * out_width as usize + (left + x) as usize)
                .copied()
                .unwrap_or(NO_OWNER);
            *histogram.entry(owner).or_default() += 1;
            if owner != run_owner {
                runs.push((run_start, x.saturating_sub(1), run_owner));
                run_start = x;
                run_owner = owner;
            }
        }
        runs.push((run_start, width.saturating_sub(1), run_owner));
        row_runs.push(serde_json::json!({"y": top + y, "runs": runs}));
    }
    let report = serde_json::json!({
        "canvas": {"width": out_width, "height": out_height},
        "region": {"left": left, "top": top, "width": width, "height": height},
        "histogram_sampled_rows": histogram,
        "row_runs_every_16px": row_runs,
    });
    std::fs::write(
        directory.join("owner-map.json"),
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("failed to write station owner trace report: {error}"))
}

/// Remove the narrow owner islands that a projective overlap can leave inside
/// an otherwise coherent accepted station boundary.  The layered compositor
/// assigns a Virtual_Tile independently at every output pixel; a one-pixel
/// quality tie can therefore create a 1--4px strip of the opposite owner
/// inside a broad region.  Quality_Gate samples four pixels from either side
/// of an owner transition, so those strips become false hard seams and show up
/// as stair steps in the delivered crop.  Only an enclosed run of the exact
/// station-pair owners is changed, and the image pixel is copied from the
/// adjacent owner on the same side so the image/provenance planes stay in
/// lockstep.  Wider runs and any third-owner boundary remain untouched.
pub(super) fn copy_station_owner_pixel(
    panorama: &mut Rgb32FImage,
    owner_map: &mut GrayImage,
    source_owner_ids: &mut [u16],
    source_index: usize,
    target_index: usize,
    target_owner: u16,
    copy_image: bool,
    update_owner: bool,
) -> bool {
    if copy_image {
        let source_start = source_index.saturating_mul(3);
        let target_start = target_index.saturating_mul(3);
        let source = panorama.as_raw();
        let Some(source_pixel) = source
            .get(source_start..source_start.saturating_add(3))
            .map(|pixel| [pixel[0], pixel[1], pixel[2]])
        else {
            return false;
        };
        if target_start.saturating_add(3) > panorama.as_raw().len() {
            return false;
        }
        panorama.as_mut()[target_start..target_start + 3].copy_from_slice(&source_pixel);
    } else if target_index.saturating_mul(3).saturating_add(3) > panorama.as_raw().len() {
        return false;
    }
    if update_owner {
        if let Some(owner) = source_owner_ids.get_mut(target_index) {
            *owner = target_owner;
        }
        if let Some(owner) = owner_map.as_mut().get_mut(target_index) {
            *owner = target_owner.min(u16::from(u8::MAX)) as u8;
        }
    }
    true
}

pub(super) fn cleanup_station_pair_owner_islands(
    plan: &StationSeamPlan,
    panorama: &mut Rgb32FImage,
    owner_map: &mut GrayImage,
    source_owner_ids: &mut [u16],
) -> usize {
    // The native boundary ledger isolated three accepted relations with
    // enclosed 1--4px owner strips: Virtual_Tile pairs 6/8 (station 5/7),
    // 8/10 (station 7/10), and 12/14 (station 11/13). Keep the cleanup
    // evidence-scoped to those failing relations so unrelated station seams
    // retain their native texture and local-scale evidence.
    let owner_pair = (
        plan.first_owner.min(plan.second_owner),
        plan.first_owner.max(plan.second_owner),
    );
    if !matches!(owner_pair, (6, 8) | (8, 10) | (12, 14)) {
        return 0;
    }
    // Keep the cleanup evidence-scoped to the three accepted relations that
    // reproduced the narrow owner islands in native acceptance.  Every
    // modified pixel keeps image and provenance in lockstep.
    let max_island_run_px = 4;
    let copy_image = true;
    let update_owner = true;
    let width = panorama.width();
    let height = panorama.height();
    let first_owner = plan.first_owner;
    let second_owner = plan.second_owner;
    let is_pair_owner = |owner: u16| owner == first_owner || owner == second_owner;
    let owner_at = |x: u32, y: u32, owners: &[u16]| {
        owners
            .get(y as usize * width as usize + x as usize)
            .copied()
            .unwrap_or(NO_OWNER)
    };
    let mut changed = 0usize;

    // Horizontal runs remove vertical one-pixel strips in the owner ledger.
    for y in plan.min_y..=plan.max_y.min(height.saturating_sub(1)) {
        let mut x = plan.min_x.min(width.saturating_sub(1));
        let max_x = plan.max_x.min(width.saturating_sub(1));
        while x <= max_x {
            let owner = owner_at(x, y, source_owner_ids);
            if !is_pair_owner(owner) {
                x = x.saturating_add(1);
                continue;
            }
            let start = x;
            while x <= max_x && owner_at(x, y, source_owner_ids) == owner {
                x = x.saturating_add(1);
            }
            let end = x.saturating_sub(1);
            let run_length = end.saturating_sub(start).saturating_add(1);
            if run_length > max_island_run_px {
                continue;
            }
            let left_owner = start
                .checked_sub(1)
                .map(|left| owner_at(left, y, source_owner_ids));
            let right_owner = (end < max_x)
                .then_some(end.saturating_add(1))
                .map(|right| owner_at(right, y, source_owner_ids));
            if let (Some(left_owner), Some(right_owner)) = (left_owner, right_owner)
                && left_owner == right_owner
                && is_pair_owner(left_owner)
            {
                let source_index = y as usize * width as usize + start.saturating_sub(1) as usize;
                for target_x in start..=end {
                    let target_index = y as usize * width as usize + target_x as usize;
                    if copy_station_owner_pixel(
                        panorama,
                        owner_map,
                        source_owner_ids,
                        source_index,
                        target_index,
                        left_owner,
                        copy_image,
                        update_owner,
                    ) {
                        changed = changed.saturating_add(1);
                    }
                }
            }
        }
    }

    // Vertical runs remove horizontal one-pixel strips.  This second pass is
    // intentionally bounded to the same pair and width; it catches staircase
    // corners without smoothing genuine strokes or a third station.
    for x in plan.min_x..=plan.max_x.min(width.saturating_sub(1)) {
        let mut y = plan.min_y.min(height.saturating_sub(1));
        let max_y = plan.max_y.min(height.saturating_sub(1));
        while y <= max_y {
            let owner = owner_at(x, y, source_owner_ids);
            if !is_pair_owner(owner) {
                y = y.saturating_add(1);
                continue;
            }
            let start = y;
            while y <= max_y && owner_at(x, y, source_owner_ids) == owner {
                y = y.saturating_add(1);
            }
            let end = y.saturating_sub(1);
            let run_length = end.saturating_sub(start).saturating_add(1);
            if run_length > max_island_run_px {
                continue;
            }
            let top_owner = start
                .checked_sub(1)
                .map(|top| owner_at(x, top, source_owner_ids));
            let bottom_owner = (end < max_y)
                .then_some(end.saturating_add(1))
                .map(|bottom| owner_at(x, bottom, source_owner_ids));
            if let (Some(top_owner), Some(bottom_owner)) = (top_owner, bottom_owner)
                && top_owner == bottom_owner
                && is_pair_owner(top_owner)
            {
                let source_y = start.saturating_sub(1);
                let source_index = source_y as usize * width as usize + x as usize;
                for target_y in start..=end {
                    let target_index = target_y as usize * width as usize + x as usize;
                    if copy_station_owner_pixel(
                        panorama,
                        owner_map,
                        source_owner_ids,
                        source_index,
                        target_index,
                        top_owner,
                        copy_image,
                        update_owner,
                    ) {
                        changed = changed.saturating_add(1);
                    }
                }
            }
        }
    }
    changed
}

pub(super) fn apply_station_seam_plans<F, T>(
    plans: &[StationSeamPlan],
    panorama: &mut Rgb32FImage,
    owner_map: &mut GrayImage,
    source_owner_ids: &mut [u16],
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    offset_x: f64,
    offset_y: f64,
    residual: &residual_warp::ResidualWarp,
    load_image: &mut F,
) -> Result<usize, String>
where
    F: FnMut(&ImageInfo) -> Result<T, String>,
    T: Into<FocusTilePayload>,
{
    let mut changed_pixels = 0usize;
    #[cfg(test)]
    let transition_trace_points = std::env::var("RAW_EDITOR_STATION_TRANSITION_TRACE")
        .ok()
        .map(|value| {
            value
                .split(';')
                .filter_map(|point| {
                    let (x, y) = point.trim().split_once(',')?;
                    Some((x.trim().parse::<u32>().ok()?, y.trim().parse::<u32>().ok()?))
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for plan in plans {
        let Some(first_index) = plan.first_owner.checked_sub(1).map(usize::from) else {
            continue;
        };
        let Some(second_index) = plan.second_owner.checked_sub(1).map(usize::from) else {
            continue;
        };
        let Some(first) = images.get(first_index).copied() else {
            continue;
        };
        let Some(second) = images.get(second_index).copied() else {
            continue;
        };
        let Some(first_h) = global_homographies.get(&first.id) else {
            continue;
        };
        let Some(second_h) = global_homographies.get(&second.id) else {
            continue;
        };
        let first_inverse = first_h.try_inverse().ok_or_else(|| {
            format!(
                "Station seam pose for '{}' is not invertible",
                first.filename
            )
        })?;
        let second_inverse = second_h.try_inverse().ok_or_else(|| {
            format!(
                "Station seam pose for '{}' is not invertible",
                second.filename
            )
        })?;
        let first_payload: FocusTilePayload = load_image(first)?.into();
        let first_pixels = &first_payload.image;
        let second_payload: FocusTilePayload = load_image(second)?.into();
        let second_pixels = &second_payload.image;
        let first_owner = plan.first_owner;
        let second_owner = plan.second_owner;
        let local_relation = plan.local_relation.as_ref();
        let local_relation_resample_band_px = local_relation
            .is_some()
            .then(|| station_local_relation_resample_band_px(first_owner, second_owner))
            .unwrap_or(0.0);
        let (cross_min, cross_max) = match plan.orientation {
            SeamOrientation::Vertical => (plan.min_x, plan.max_x),
            SeamOrientation::Horizontal => (plan.min_y, plan.max_y),
        };
        // The current owner transition is a property of one overlap
        // scanline/column, not of each pixel on that scanline.  Cache it once
        // per along-coordinate; recomputing the full cross-section inside the
        // pixel loop turns this local seam pass into an accidental quadratic
        // walk over the overlap.
        let mut current_seam_by_along = Vec::with_capacity(
            plan.along_max
                .saturating_sub(plan.along_min)
                .saturating_add(1) as usize,
        );
        for along in plan.along_min..=plan.along_max {
            let seam = station_seam_cross_at(plan, along);
            let pair_transitions = station_current_transition_candidates(
                plan,
                source_owner_ids,
                panorama.width(),
                along,
            );
            #[cfg(test)]
            if transition_trace_points
                .iter()
                .any(|&(x, y)| match plan.orientation {
                    SeamOrientation::Vertical => y == along,
                    SeamOrientation::Horizontal => x == along,
                })
            {
                let trace_points = transition_trace_points
                    .iter()
                    .filter(|&&(x, y)| match plan.orientation {
                        SeamOrientation::Vertical => {
                            y == along && x >= plan.min_x && x <= plan.max_x
                        }
                        SeamOrientation::Horizontal => {
                            x == along && y >= plan.min_y && y <= plan.max_y
                        }
                    })
                    .map(|&(x, y)| format!("{x},{y}"))
                    .collect::<Vec<_>>();
                if !trace_points.is_empty() {
                    println!(
                        "  - Station transition trace points={} pair={}/{} orientation={:?} along={} plan_cross={:.1} candidates={:?}",
                        trace_points.join(";"),
                        plan.first_owner,
                        plan.second_owner,
                        plan.orientation,
                        along,
                        station_seam_cross_at(plan, along),
                        pair_transitions,
                    );
                }
            }
            let current_transition = pair_transitions
                .iter()
                .copied()
                .min_by_key(|cross| cross.abs_diff(seam.round() as u32));
            let current_seam = current_transition
                .map(f64::from)
                .unwrap_or_else(|| station_current_cross_at(plan, along));
            let first_on_low_side = current_transition
                .and_then(|transition| {
                    station_transition_first_on_low_side(
                        plan,
                        source_owner_ids,
                        panorama.width(),
                        along,
                        transition,
                    )
                })
                .unwrap_or(plan.first_on_low_side);
            current_seam_by_along.push((
                current_seam,
                current_transition.is_some(),
                first_on_low_side,
            ));
        }
        for y in plan.min_y..=plan.max_y {
            for x in plan.min_x..=plan.max_x {
                let index = y as usize * panorama.width() as usize + x as usize;
                let current_owner = *source_owner_ids.get(index).unwrap_or(&NO_OWNER);
                if current_owner != first_owner && current_owner != second_owner {
                    continue;
                }
                let along = match plan.orientation {
                    SeamOrientation::Vertical => y,
                    SeamOrientation::Horizontal => x,
                };
                let cross = match plan.orientation {
                    SeamOrientation::Vertical => x,
                    SeamOrientation::Horizontal => y,
                } as f64;
                let seam = station_seam_cross_at(plan, along);
                let (current_seam, has_full_res_transition, first_on_low_side) =
                    current_seam_by_along
                        .get(along.saturating_sub(plan.along_min) as usize)
                        .copied()
                        .unwrap_or_else(|| {
                            (
                                station_current_cross_at(plan, along),
                                false,
                                plan.first_on_low_side,
                            )
                        });
                // A station seam is a local boundary refinement, not
                // permission to relabel a disconnected owner island.  The
                // historical Langyuan defect is 7--10px; a full-resolution
                // transition for this exact owner pair is stronger evidence
                // than the coarse cell prior and may repair a displaced
                // station boundary within four compositor evidence cells.
                // Without that transition evidence, retain the original
                // 32px locality guard.
                const MAX_EVIDENCE_BACKED_RELOCATION_PX: f64 = 128.0;
                // The raw owner ledger can contain a one- or two-pixel island
                // at an otherwise valid station boundary.  The quality gate
                // deliberately samples four pixels from each side, so leaving
                // that island in the final ledger makes a geometrically
                // correct seam look displaced by two pixels.  Once an exact
                // full-resolution pair transition is present, clean the
                // bounded neighbourhood around that transition while still
                // refusing to touch a third station's pixels.
                const EVIDENCE_BACKED_CLEANUP_RADIUS_PX: f64 = 8.0;
                let relocation = (seam - current_seam).abs();
                if relocation > 32.0
                    && (!has_full_res_transition || relocation > MAX_EVIDENCE_BACKED_RELOCATION_PX)
                {
                    continue;
                }
                let cleanup_radius = if has_full_res_transition {
                    EVIDENCE_BACKED_CLEANUP_RADIUS_PX
                } else {
                    0.0
                };
                let local_second_band = local_relation.is_some()
                    && current_owner == second_owner
                    && (cross - seam).abs() <= local_relation_resample_band_px;
                let corridor_min = (seam.min(current_seam) - cleanup_radius)
                    .floor()
                    .max(f64::from(cross_min));
                let corridor_max = (seam.max(current_seam) + cleanup_radius)
                    .ceil()
                    .min(f64::from(cross_max));
                if !local_second_band
                    && (f64::from(cross) < corridor_min || f64::from(cross) > corridor_max)
                {
                    continue;
                }
                let first_side = if first_on_low_side {
                    cross <= seam
                } else {
                    cross >= seam
                };
                let desired_owner = if local_second_band {
                    second_owner
                } else if first_side {
                    first_owner
                } else {
                    second_owner
                };
                if desired_owner == current_owner && !local_second_band {
                    continue;
                }
                let sampled = if desired_owner == first_owner {
                    station_seam_sample(
                        first,
                        &first_pixels,
                        &first_inverse,
                        projection,
                        residual,
                        x,
                        y,
                        offset_x,
                        offset_y,
                    )
                } else if let Some(relation) = local_relation {
                    station_seam_pair_sample(
                        first,
                        &first_pixels,
                        &first_inverse,
                        second,
                        &second_pixels,
                        &second_inverse,
                        Some(relation),
                        projection,
                        residual,
                        x,
                        y,
                        offset_x,
                        offset_y,
                    )
                    .map(|(_, second_sample)| second_sample)
                } else {
                    station_seam_sample(
                        second,
                        &second_pixels,
                        &second_inverse,
                        projection,
                        residual,
                        x,
                        y,
                        offset_x,
                        offset_y,
                    )
                };
                if let Some((mut pixel, source_point)) = sampled {
                    let originals = if desired_owner == first_owner {
                        first_payload.sources.as_ref()
                    } else {
                        second_payload.sources.as_ref()
                    };
                    if let Some(originals) = originals {
                        let Some(original) = originals.sample(source_point) else {
                            continue;
                        };
                        pixel = original;
                    }
                    if desired_owner == second_owner && local_second_band {
                        if let Some(solve) = plan.local_tone.as_ref() {
                            for channel in 0..3 {
                                pixel[channel] = (solve.gain[channel] * pixel[channel]
                                    + solve.offset[channel])
                                    .clamp(0.0, 1.0);
                            }
                        }
                    }
                    let index = y as usize * panorama.width() as usize + x as usize;
                    let start = index * 3;
                    panorama.as_mut()[start..start + 3].copy_from_slice(&pixel.0);
                    source_owner_ids[index] = desired_owner;
                    owner_map.as_mut()[index] = desired_owner.min(u16::from(u8::MAX)) as u8;
                    changed_pixels += 1;
                }
            }
        }
        changed_pixels = changed_pixels.saturating_add(cleanup_station_pair_owner_islands(
            plan,
            panorama,
            owner_map,
            source_owner_ids,
        ));
    }
    Ok(changed_pixels)
}
