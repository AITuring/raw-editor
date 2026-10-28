//! In-memory detail-preserving focus mosaic.
//!
//! The public entry point: refines every layer, chooses source ownership and
//! renders only owned cells into one canvas; a large canvas is handed to the
//! streaming compositor.

use super::super::stack_pipeline::intra_station;
use super::super::stack_pipeline::report::IntraStationFrameRecord;
use super::super::stack_pipeline::report::IntraStationFrameStatus;
use super::super::stack_pipeline::station_degradation::GroupJoinEvidence;
use super::super::stack_pipeline::station_degradation::StationMember;
use super::super::stack_pipeline::station_degradation::group_join_rejection;
use super::super::stack_pipeline::station_degradation::plan_station_fusion;
use super::super::stack_pipeline::station_degradation::record_run_entries;
use super::super::stack_pipeline::station_degradation::record_run_group_join_rejection;
use super::super::stack_pipeline::station_degradation::station_plan_entries;
use super::super::stitching::Projection;
use super::super::stitching::downsample_rgb_half;
use super::super::stitching::output_bounds;
use super::super::stitching::pixel_aligned_canvas;
use super::super::stitching::transformed_image_region;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point3;
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;
use tauri::AppHandle;
use tauri::Emitter;
use tauri::Runtime;

#[cfg(test)]
use super::diagnostics;
use super::{
    ANALYSIS_LONG_SIDE, Field, GRID_STEP, LONG_SEQUENCE_SELECTION_LONG_SIDE, LayerSampler,
    NativeRefineMode, STREAMING_CANVAS_MIN_PIXELS, adjusted, detail_preserving_mosaic_streaming,
    intra_station_frame_record, ownership, ownership_with_long_side, refine_layer,
    refine_native_layer, station_member_median_sharpness, tone_field,
};

pub(super) fn mask_covered_bounds(mask: &GrayImage) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = mask.width();
    let mut min_y = mask.height();
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;
    for (x, y, pixel) in mask.enumerate_pixels() {
        if pixel[0] == 0 {
            continue;
        }
        found = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    found.then_some((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

pub(crate) fn detail_preserving_mosaic<R: Runtime, F>(
    images: &[&ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    capture_group_ids: Option<&HashMap<usize, u8>>,
    sequence_gap_aware: bool,
    app: AppHandle<R>,
    event: &str,
    load: &mut F,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let (min_x, max_x, min_y, max_y) = output_bounds(images, homographies, projection);
    if ![min_x, max_x, min_y, max_y].iter().all(|x| x.is_finite()) {
        return Err("The stack does not have a finite aligned canvas.".into());
    }
    let (offset_x, width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, height) = pixel_aligned_canvas(min_y, max_y);
    let canvas_pixels = u64::from(width).saturating_mul(u64::from(height));
    #[cfg(test)]
    let force_streaming = std::env::var_os("RAW_EDITOR_FORCE_STREAMING_MOSAIC").is_some();
    #[cfg(not(test))]
    let force_streaming = false;
    let capture_group_count = capture_group_ids
        .map(|ids| ids.values().copied().collect::<HashSet<_>>().len())
        .unwrap_or(0);
    if force_streaming
        || capture_group_count > 1
        || (sequence_gap_aware && images.len() > 8)
        || canvas_pixels > STREAMING_CANVAS_MIN_PIXELS
    {
        return detail_preserving_mosaic_streaming(
            images,
            homographies,
            projection,
            capture_group_ids,
            sequence_gap_aware,
            app,
            event,
            load,
            offset_x,
            offset_y,
            width,
            height,
        );
    }
    let mut result = Rgb32FImage::new(width, height);
    let mut mask = GrayImage::new(width, height);
    let mut covered_bounds: Option<(u32, u32, u32, u32)> = None;
    let local_refinement_enabled = !sequence_gap_aware || images.len() <= 8;
    // Everything past the streaming branch above has `capture_group_count <= 1`,
    // so this loop composites one Capture_Station: layer 0 is the frame every
    // later layer registers against, which makes this the intra-station path of
    // 需求 2.2–2.9.  A `sequence_gap_aware` run is different in kind — a filename
    // bridge joined captures that share no measured overlap — and stays on the
    // legacy comparison behaviour, early exit included.
    let intra_station_path = !sequence_gap_aware;
    let anchor_long_side = images
        .first()
        .map(|info| info.width.max(info.height))
        .unwrap_or(0);
    let native_refine_mode = if intra_station_path {
        NativeRefineMode::IntraStation { anchor_long_side }
    } else {
        NativeRefineMode::LegacyMosaic
    };
    let station_index = images
        .first()
        .and_then(|info| capture_group_ids.and_then(|ids| ids.get(&info.id)).copied())
        .map(usize::from)
        .unwrap_or(0);
    let anchor_path = images
        .first()
        .map(|info| info.filename.clone())
        .unwrap_or_default();
    // 需求 2.1: the anchor frame carries the identity transform, so its record
    // is the reference every other frame's transform is expressed against.
    let anchor_to_world_inverse = images
        .first()
        .and_then(|info| homographies.get(&info.id))
        .and_then(|transform| transform.try_inverse());
    // 需求 12.1 / 12.2: the members this station fuses, each with the verdict
    // the registrar reaches below.  The anchor carries the identity transform,
    // so it is registered by definition.
    let mut station_members = Vec::<StationMember>::new();
    if intra_station_path {
        if let Some(info) = images.first() {
            intra_station::record_run_frame(
                station_index,
                &anchor_path,
                IntraStationFrameRecord {
                    path: info.filename.clone(),
                    status: IntraStationFrameStatus::Anchor,
                    inlier_area_coverage: 1.0,
                    ..Default::default()
                },
            );
            station_members.push(StationMember {
                path: info.filename.clone(),
                median_sharpness: station_member_median_sharpness(info),
                status: IntraStationFrameStatus::Anchor,
            });
        }
    }
    println!("  - Detail-preserving mosaic canvas: {width}x{height}");
    for (index, &info) in images.iter().enumerate() {
        let _ = app.emit(
            event,
            format!(
                "Refining and selecting detail {} of {}",
                index + 1,
                images.len()
            ),
        );
        let mut source = load(info)?;
        let transform = &homographies[&info.id];
        let mut source_divisor = 1.0;
        if projection == Projection::Planar {
            let center = Point3::new(info.width as f64 * 0.5, info.height as f64 * 0.5, 1.0);
            let c = transform * center;
            let dx = transform * (center + nalgebra::Vector3::new(1.0, 0.0, 0.0));
            let dy = transform * (center + nalgebra::Vector3::new(0.0, 1.0, 0.0));
            let sx = (dx.x / dx.z - c.x / c.z).hypot(dx.y / dx.z - c.y / c.z);
            let sy = (dy.x / dy.z - c.x / c.z).hypot(dy.y / dy.z - c.y / c.z);
            let scale = sx.min(sy);
            // Prefilter a telephoto tile before minification. Otherwise weave
            // aliases into false sharpness and defeats native focus selection.
            while scale.is_finite()
                && scale > 0.0
                && scale * source_divisor < 0.67
                && source.width().min(source.height()) > 8
            {
                source = downsample_rgb_half(&source);
                source_divisor *= 2.0;
            }
        }
        let inverse = transform
            .try_inverse()
            .ok_or("The stack contains a non-invertible alignment.")?;
        let Some((left, right, top, bottom)) = transformed_image_region(
            info, transform, projection, offset_x, offset_y, width, height,
        ) else {
            continue;
        };
        let layer_width = right - left + 1;
        let layer_height = bottom - top + 1;
        let scale = (layer_width.max(layer_height) as f64 / ANALYSIS_LONG_SIDE as f64).max(1.0);
        let aw = (layer_width as f64 / scale).ceil() as u32;
        let ah = (layer_height as f64 / scale).ceil() as u32;
        let mut sampler = LayerSampler {
            info,
            source: &source,
            source_divisor,
            inverse,
            native_mapping: None,
            projection,
            offset: (offset_x, offset_y),
            left,
            top,
            scale,
            residual: Field::new(aw, ah, GRID_STEP as f64),
        };
        println!("  - Selecting '{}'", info.filename);

        let layer_overlaps_existing = covered_bounds.is_some_and(
            |(covered_left, covered_right, covered_top, covered_bottom)| {
                let overlap_width = right
                    .min(covered_right)
                    .saturating_sub(left.max(covered_left));
                let overlap_height = bottom
                    .min(covered_bottom)
                    .saturating_sub(top.max(covered_top));
                overlap_width >= 128 && overlap_height >= 128
            },
        );
        if sequence_gap_aware && !layer_overlaps_existing {
            // A filename bridge explicitly means that these captures are
            // adjacent but have no measured overlap. Do not spend the costly
            // residual/graph-cut pass trying to register empty canvas against
            // an unrelated block; copy the source tile at its verified global
            // pose, preserving every native detail pixel.
            println!("    - Sequence gap: copying native detail without overlap refinement");
            copy_sequence_gap_layer(&mut result, &mut mask, &sampler, left, right, top, bottom);
            covered_bounds = Some(match covered_bounds {
                Some((covered_left, covered_right, covered_top, covered_bottom)) => (
                    covered_left.min(left),
                    covered_right.max(right),
                    covered_top.min(top),
                    covered_bottom.max(bottom),
                ),
                None => (left, right, top, bottom),
            });
            continue;
        }
        if index > 0 && local_refinement_enabled {
            // 需求 2.2: two global + local rounds at the analysis resolution,
            // then the native patch refinement.
            for _ in 0..intra_station::INTRA_STATION_ANALYSIS_ROUNDS {
                refine_layer(&result, &mask, &mut sampler, aw, ah);
            }
            let outcome =
                refine_native_layer(&result, &mask, &mut sampler, aw, ah, native_refine_mode);
            if intra_station_path {
                let into_anchor = anchor_to_world_inverse
                    .map(|inverse| inverse * transform)
                    .unwrap_or_else(Matrix3::identity);
                let record = intra_station_frame_record(&info.filename, into_anchor, &outcome);
                intra_station::record_run_frame(station_index, &anchor_path, record.clone());
                // 需求 12.5: inlier spatial support below 20%, or an inlier
                // median symmetric reprojection error above 0.01 x the long
                // side, keeps this Source_RAW out of every Capture_Station.
                // 需求 12.2: it is therefore excluded from this station's
                // fusion, with its absolute path and reason identifier
                // reported one by one.  A frame whose evidence was never
                // measured (no control point was even evaluated) has nothing
                // to judge and keeps the behaviour it had before.
                let rejection = (outcome.evaluated_control_points > 0)
                    .then(|| {
                        group_join_rejection(&GroupJoinEvidence {
                            path: info.filename.clone(),
                            inlier_area_coverage: record.inlier_area_coverage,
                            median_symmetric_error_px: record.median_symmetric_error_px,
                            anchor_long_side_px: anchor_long_side,
                        })
                    })
                    .flatten();
                station_members.push(StationMember {
                    path: info.filename.clone(),
                    median_sharpness: station_member_median_sharpness(info),
                    // The member status is what this station actually did with
                    // the frame, so the plan below can never claim a frame was
                    // excluded while its pixels are on the canvas.  With no
                    // control point evaluated there is no local evidence to
                    // judge and the frame is placed by its global model, which
                    // is exactly `GlobalFallback`.
                    status: match (rejection.is_some(), record.status) {
                        (true, _) => IntraStationFrameStatus::Failed,
                        (false, IntraStationFrameStatus::Failed) => {
                            IntraStationFrameStatus::GlobalFallback
                        }
                        (false, status) => status,
                    },
                });
                if let Some(rejection) = rejection {
                    println!(
                        "    - Rejected from every Capture_Station: inlier support {:.3} (minimum {:.2}), median symmetric error {:.2}px (limit {:.2}px)",
                        rejection.inlier_area_coverage,
                        rejection.minimum_inlier_area_coverage,
                        rejection.median_symmetric_error_px,
                        rejection.symmetric_error_limit_px
                    );
                    record_run_group_join_rejection(station_index, &rejection);
                    continue;
                }
            }
        } else if index > 0 && sequence_gap_aware {
            println!("    - Long sequence: using verified global alignment for focus ownership");
        }
        let tone = if index > 0 {
            tone_field(&result, &mask, &sampler, aw, ah)
        } else {
            Field::new(aw, ah, 56.0)
        };
        let decision = if sequence_gap_aware && images.len() > 8 {
            ownership_with_long_side(
                &result,
                &mask,
                &sampler,
                &tone,
                aw,
                ah,
                LONG_SEQUENCE_SELECTION_LONG_SIDE,
            )
        } else {
            ownership(&result, &mask, &sampler, &tone, aw, ah)
        };
        #[cfg(test)]
        diagnostics::capture_layer(index, &sampler, &tone, &decision, aw, ah);
        let stride = width as usize * 3;
        // Render only ownership cells assigned to this layer. The previous
        // scan visited every pixel in the projected rectangle even when the
        // graph cut had already rejected most of those cells. On large
        // 31k×11k stacks that redundant decision check dominated runtime.
        let decision_width = decision.width();
        let decision_height = decision.height();
        result
            .as_mut()
            .par_chunks_mut(stride)
            .zip(mask.as_mut().par_chunks_mut(width as usize))
            .enumerate()
            .skip(top as usize)
            .take(layer_height as usize)
            .for_each(|(y, (row, covered))| {
                let ly = (y - top as usize) as f64;
                let ay = ly / scale;
                let dy =
                    ((ay / ah as f64 * decision_height as f64) as u32).min(decision_height - 1);
                for dx in 0..decision_width {
                    if decision.get_pixel(dx, dy)[0] == 0 {
                        continue;
                    }
                    // These bounds are the inverse of the decision lookup
                    // above. Adjacent cells meet without leaving a pixel
                    // gap, while rejected cells are never sampled.
                    let cell_start =
                        (dx as f64 * aw as f64 / decision_width as f64 * scale).ceil() as u32;
                    let cell_end = (((dx + 1) as f64 * aw as f64 / decision_width as f64 * scale)
                        .ceil() as u32)
                        .min(layer_width);
                    let x_start = left + cell_start.min(layer_width);
                    let x_end = left + cell_end;
                    for x in x_start..x_end {
                        let lx = (x - left) as f64;
                        let ax = lx / scale;
                        let Some(pixel) = sampler.sample(lx, ly) else {
                            continue;
                        };
                        let pixel = adjusted(pixel, tone.at(ax, ay));
                        row[x as usize * 3..x as usize * 3 + 3].copy_from_slice(&pixel.0);
                        covered[x as usize] = 255;
                    }
                }
            });
        covered_bounds = Some(match covered_bounds {
            Some((covered_left, covered_right, covered_top, covered_bottom)) => (
                covered_left.min(left),
                covered_right.max(right),
                covered_top.min(top),
                covered_bottom.max(bottom),
            ),
            None => (left, right, top, bottom),
        });
    }
    // 需求 12.1 / 12.2: the station's fusion path, with every excluded
    // Source_RAW listed individually.  `station_members` carries what the loop
    // above really fused, so the recorded plan and the pixels agree.
    if intra_station_path {
        let plan = plan_station_fusion(&station_members);
        let entries = station_plan_entries(station_index, &station_members, &plan);
        if !entries.is_empty() {
            println!(
                "  - Capture_Station {station_index} fused {} of {} frame(s) ({:?})",
                plan.fused.len(),
                station_members.len(),
                plan.mode
            );
            for excluded in &plan.excluded {
                println!("    - Excluded '{}': {}", excluded.path, excluded.reason);
            }
        }
        record_run_entries(&entries);
    }
    let covered = mask.as_raw().iter().filter(|&&v| v != 0).count();
    if covered == 0 {
        return Err("The aligned stack has no covered pixels.".into());
    }
    // Keep the complete union of real source coverage. Cropping to the largest
    // hole-free rectangle silently removes photographed frame edges whenever
    // the scan is slightly rotated or the coverage is non-rectangular.
    #[cfg(test)]
    diagnostics::capture_crop(&mask);
    let (crop_x, crop_y, crop_width, crop_height) =
        mask_covered_bounds(&mask).ok_or("The aligned stack has no covered pixels.")?;
    let output =
        image::imageops::crop_imm(&result, crop_x, crop_y, crop_width, crop_height).to_image();
    println!(
        "  - Verified covered output: {}x{}; every pixel has source ownership",
        output.width(),
        output.height()
    );
    Ok(output)
}

pub(super) fn copy_sequence_gap_layer(
    result: &mut Rgb32FImage,
    mask: &mut GrayImage,
    sampler: &LayerSampler<'_>,
    left: u32,
    right: u32,
    top: u32,
    bottom: u32,
) {
    let width = result.width();
    let stride = width as usize * 3;
    result
        .as_mut()
        .par_chunks_mut(stride)
        .zip(mask.as_mut().par_chunks_mut(width as usize))
        .enumerate()
        .skip(top as usize)
        .take((bottom - top + 1) as usize)
        .for_each(|(y, (row, covered))| {
            let local_y = (y as u32 - top) as f64;
            for x in left..=right {
                let local_x = (x - left) as f64;
                let Some(pixel) = sampler.sample(local_x, local_y) else {
                    continue;
                };
                let start = x as usize * 3;
                row[start..start + 3].copy_from_slice(&pixel.0);
                covered[x as usize] = 255;
            }
        });
}
