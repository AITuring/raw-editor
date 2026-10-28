//! Streaming (tiled) detail-preserving focus mosaic.
//!
//! One function by design: layer refinement, ownership, tile writes and the
//! low-frequency harmonization run in a fixed order over the on-disk canvas.

use super::super::stack_pipeline::intra_station;
use super::super::stack_pipeline::report::IntraStationFrameRecord;
use super::super::stack_pipeline::report::IntraStationFrameStatus;
use super::super::stitching::Projection;
use super::super::stitching::downsample_rgb_half;
use super::super::stitching::transformed_image_region;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Luma;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use nalgebra::Point3;
use std::collections::HashMap;
use std::collections::HashSet;
use tauri::AppHandle;
use tauri::Emitter;
use tauri::Runtime;

use super::{
    Field, GRID_STEP, LayerSampler, NativeRefineMode, STREAMING_ANALYSIS_LONG_SIDE,
    STREAMING_FLAT_TONE_SIGMA_CELLS, STREAMING_OWNERSHIP_FEATHER, STREAMING_TILE_SIZE,
    StreamingLowFrequencyConsensus, StreamingMosaicStore, apply_streaming_seam_harmonization,
    apply_streaming_source_owner_low_frequency_correction, copy_streaming_owned_tile_region,
    copy_streaming_tile_region, fill_streaming_uncovered_tile_region, intra_station_frame_record,
    materialize_streaming_focus_output, refine_layer_from_analysis, refine_native_layer_streaming,
    smooth_streaming_low_frequency_illumination, streaming_candidate_analysis,
    streaming_ownership_from_analysis, streaming_refinement_pair, streaming_seam_harmonization,
    streaming_tone_field_from_analysis,
};

pub(super) fn detail_preserving_mosaic_streaming<R: Runtime, F>(
    images: &[&ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    capture_group_ids: Option<&HashMap<usize, u8>>,
    panorama_transition: bool,
    app: AppHandle<R>,
    event: &str,
    load: &mut F,
    offset_x: f64,
    offset_y: f64,
    width: u32,
    height: u32,
) -> Result<Rgb32FImage, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    let store = StreamingMosaicStore::new(width, height)?;
    let mut low_frequency_consensus = StreamingLowFrequencyConsensus::new(width, height);
    let mut seen_capture_groups = HashSet::new();
    // 需求 2.1 / 2.9: the frame every later frame of a Capture_Station registers
    // against is that station's first layer in render order.  Its absolute path,
    // inverse world pose and native long side are what the other frames' records
    // are expressed against.
    let mut station_anchors = HashMap::<u8, (String, Option<Matrix3<f64>>, u32)>::new();
    println!(
        "  - Streaming detail-preserving mosaic canvas: {width}x{height}, tiles {}x{} of {}px",
        store.tile_columns, store.tile_rows, STREAMING_TILE_SIZE
    );
    for (index, &info) in images.iter().enumerate() {
        let capture_group_id = capture_group_ids
            .and_then(|groups| groups.get(&info.id))
            .copied()
            .unwrap_or_else(|| {
                u8::try_from(index + 1).expect("focus stack source limit keeps group ids in u8")
            });
        // Capture_Station ownership is the right key for a focus bracket, but
        // a flat-artwork panorama needs one label per source tile so the final
        // low-frequency gain solve can distinguish adjacent exposures inside
        // one connected grouping component. Keep the station key for all
        // registration decisions and use this separate source label only in
        // the materialized owner plane.
        let source_owner_id = panorama_transition
            .then(|| u8::try_from(info.id.saturating_add(1)).ok())
            .flatten()
            .filter(|&id| id != 0)
            .unwrap_or(capture_group_id);
        let first_capture_group_layer = seen_capture_groups.insert(capture_group_id);
        let _ = app.emit(
            event,
            format!(
                "Streaming detail and selecting sharp source {} of {}",
                index + 1,
                images.len()
            ),
        );
        let mut source = load(info)?;
        let transform = &homographies[&info.id];
        if first_capture_group_layer {
            station_anchors.insert(
                capture_group_id,
                (
                    info.filename.clone(),
                    transform.try_inverse(),
                    info.width.max(info.height),
                ),
            );
            intra_station::record_run_frame(
                usize::from(capture_group_id),
                &info.filename,
                IntraStationFrameRecord {
                    path: info.filename.clone(),
                    status: IntraStationFrameStatus::Anchor,
                    inlier_area_coverage: 1.0,
                    ..Default::default()
                },
            );
        }
        let mut source_divisor = 1.0;
        if projection == Projection::Planar {
            let center = Point3::new(info.width as f64 * 0.5, info.height as f64 * 0.5, 1.0);
            let c = transform * center;
            let dx = transform * (center + nalgebra::Vector3::new(1.0, 0.0, 0.0));
            let dy = transform * (center + nalgebra::Vector3::new(0.0, 1.0, 0.0));
            let sx = (dx.x / dx.z - c.x / c.z).hypot(dx.y / dx.z - c.y / c.z);
            let sy = (dy.x / dy.z - c.x / c.z).hypot(dy.y / dy.z - c.y / c.z);
            let scale = sx.min(sy);
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
        let layer_scale =
            (layer_width.max(layer_height) as f64 / STREAMING_ANALYSIS_LONG_SIDE as f64).max(1.0);
        let analysis_width = (layer_width as f64 / layer_scale).ceil() as u32;
        let analysis_height = (layer_height as f64 / layer_scale).ceil() as u32;
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
            scale: layer_scale,
            residual: Field::new(analysis_width, analysis_height, GRID_STEP as f64),
        };
        let (base_analysis, base_analysis_mask, base_analysis_owner) =
            store.analysis_region(left, top, layer_scale, analysis_width, analysis_height)?;
        let has_existing_coverage = base_analysis_mask.as_raw().iter().any(|&value| value != 0);
        let mut overlap_candidate = None;
        let (tone, decision) = if has_existing_coverage {
            let group_refinement_mask =
                (!first_capture_group_layer && !panorama_transition).then(|| {
                    // A later focal plane must register against its own camera
                    // position.  Using pixels already owned by another panorama
                    // tile pulls repeated calligraphy toward the wrong column.
                    // Keep those older positions available for colour matching,
                    // but exclude them from residual motion estimation.
                    GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
                        Luma([(base_analysis_mask.get_pixel(x, y)[0] != 0
                            && base_analysis_owner.get_pixel(x, y)[0] == capture_group_id)
                            .then_some(255)
                            .unwrap_or(0)])
                    })
                });
            // FlatArtwork owner labels identify source tiles rather than
            // Capture_Stations. Every prior tile is therefore valid overlap
            // for a panorama registration/tone solve; restricting the mask to
            // the station id would leave only the first tile and reintroduce
            // frame-sized exposure blocks.
            let refinement_mask = group_refinement_mask
                .as_ref()
                .unwrap_or(&base_analysis_mask);
            // 需求 2.2: two global + local rounds at the analysis resolution,
            // then the native patch refinement below.
            for _ in 0..intra_station::INTRA_STATION_ANALYSIS_ROUNDS {
                let (candidate, candidate_mask) =
                    streaming_candidate_analysis(&sampler, analysis_width, analysis_height);
                let (a, b, valid) = streaming_refinement_pair(
                    &base_analysis,
                    refinement_mask,
                    &candidate,
                    &candidate_mask,
                );
                refine_layer_from_analysis(&a, &b, &valid, &mut sampler);
            }
            // 需求 2.2–2.9: the native patch refinement that follows the two
            // analysis rounds.  Only a later layer of a station already on the
            // canvas is an intra-station registration; the first layer of a
            // station has no same-station pixels to register against, so it is
            // the anchor and keeps its global pose.
            if !first_capture_group_layer {
                let anchor = station_anchors.get(&capture_group_id);
                let anchor_long_side = anchor
                    .map(|&(_, _, long_side)| long_side)
                    .unwrap_or_else(|| info.width.max(info.height));
                let outcome = refine_native_layer_streaming(
                    &store,
                    Some(capture_group_id),
                    &mut sampler,
                    analysis_width,
                    analysis_height,
                    (left, right, top, bottom),
                    NativeRefineMode::IntraStation { anchor_long_side },
                )?;
                let into_anchor = anchor
                    .and_then(|&(_, inverse, _)| inverse)
                    .map(|inverse| inverse * transform)
                    .unwrap_or_else(Matrix3::identity);
                intra_station::record_run_frame(
                    usize::from(capture_group_id),
                    anchor.map(|(path, _, _)| path.as_str()).unwrap_or_default(),
                    intra_station_frame_record(&info.filename, into_anchor, &outcome),
                );
            }
            let (candidate, candidate_mask) =
                streaming_candidate_analysis(&sampler, analysis_width, analysis_height);
            // A focus bracket may contain several camera positions.  Once a
            // position has entered the mosaic, estimate its colour correction
            // only from overlap owned by that same position.  Comparing a
            // later position against a neighbouring group's pixels compounds
            // exposure drift into broad vertical bands.
            let tone = if first_capture_group_layer {
                // A new camera position has no same-station reference, but it
                // does overlap the existing flat-artwork mosaic. Estimate a
                // low-frequency field from that overlap now so the source
                // owner does not carry a whole-frame exposure block into the
                // final canvas. The field is sampled at analysis resolution;
                // brush/wood detail remains source-owned and untouched.
                streaming_tone_field_from_analysis(
                    &base_analysis,
                    &base_analysis_mask,
                    &candidate,
                    &candidate_mask,
                    if panorama_transition {
                        STREAMING_FLAT_TONE_SIGMA_CELLS
                    } else {
                        1.0
                    },
                )
            } else {
                streaming_tone_field_from_analysis(
                    &base_analysis,
                    refinement_mask,
                    &candidate,
                    &candidate_mask,
                    1.0,
                )
            };
            let decision = streaming_ownership_from_analysis(
                &base_analysis,
                &base_analysis_mask,
                &candidate,
                &candidate_mask,
                &tone,
                &base_analysis_owner,
                capture_group_id,
                panorama_transition || first_capture_group_layer,
            );
            overlap_candidate = Some((candidate, candidate_mask));
            (tone, decision)
        } else {
            (
                Field::new(analysis_width, analysis_height, 112.0),
                GrayImage::from_pixel(1, 1, Luma([255])),
            )
        };
        // Keep the per-layer tone field on FlatArtwork writes so seams remain
        // soft and source detail does not become a hard rectangular cut. The
        // flat path deliberately has no second owner-relation solve: raw
        // overlap ratios mix unlike brush/wood content and can become a
        // source-sized gain block.
        let write_tone = if panorama_transition
            && std::env::var_os("RAW_EDITOR_SKIP_FLAT_TONE_CORRECTION").is_some()
        {
            Field::new(analysis_width, analysis_height, 112.0)
        } else {
            tone.clone()
        };
        // Accumulate only overlap-supported, low-frequency candidate
        // pixels.  This is deliberately before ownership writes: the
        // final correction compares the consensus target with the
        // materialised source-owned output, so no detail is averaged.
        if !panorama_transition
            || std::env::var_os("RAW_EDITOR_ENABLE_FLAT_ARTWORK_CONSENSUS").is_some()
        {
            if let Some((candidate, candidate_mask)) = overlap_candidate.as_ref() {
                low_frequency_consensus.observe_overlap(
                    left,
                    top,
                    layer_scale,
                    &base_analysis,
                    &base_analysis_mask,
                    candidate,
                    candidate_mask,
                    &write_tone,
                );
            }
        }
        println!(
            "    - Streaming '{}': projected region {}..{} x {}..{}, shared analysis {}x{}",
            info.filename, left, right, top, bottom, analysis_width, analysis_height
        );
        // Keep focus ownership binary away from a seam, but cross-fade one
        // decision cell on either side of the cut. Rendering the raw graph
        // labels directly reproduced their staircase at native resolution.
        // This narrow transition hides the grid without averaging detail over
        // the rest of either source layer.
        let ownership_feather = if panorama_transition {
            std::env::var("RAW_EDITOR_FLAT_OWNERSHIP_FEATHER")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite() && *value >= 0.0)
                .unwrap_or(STREAMING_OWNERSHIP_FEATHER)
        } else {
            STREAMING_OWNERSHIP_FEATHER
        };
        let soft_decision = image::imageops::blur(&decision, ownership_feather);
        let first_column = left / STREAMING_TILE_SIZE;
        let last_column = right / STREAMING_TILE_SIZE;
        let first_row = top / STREAMING_TILE_SIZE;
        let last_row = bottom / STREAMING_TILE_SIZE;
        for tile_row in first_row..=last_row {
            for tile_column in first_column..=last_column {
                let (tile_left, tile_top, tile_width, tile_height) =
                    store.tile_extent(tile_column, tile_row);
                let x_start = left.max(tile_left);
                let x_end = right.min(tile_left + tile_width - 1);
                let y_start = top.max(tile_top);
                let y_end = bottom.min(tile_top + tile_height - 1);
                if x_start > x_end || y_start > y_end {
                    continue;
                }
                let (mut tile, mut tile_mask) = store.load_tile(tile_column, tile_row)?;
                let mut tile_owner = store.load_owner_tile(tile_column, tile_row);
                if !has_existing_coverage {
                    copy_streaming_tile_region(
                        &mut tile,
                        &mut tile_mask,
                        &mut tile_owner,
                        source_owner_id,
                        &sampler,
                        tile_left,
                        tile_top,
                        x_start,
                        x_end,
                        y_start,
                        y_end,
                    );
                } else {
                    copy_streaming_owned_tile_region(
                        &mut tile,
                        &mut tile_mask,
                        &mut tile_owner,
                        source_owner_id,
                        &sampler,
                        &write_tone,
                        &soft_decision,
                        tile_left,
                        tile_top,
                        x_start,
                        x_end,
                        y_start,
                        y_end,
                        layer_scale,
                        analysis_width,
                        analysis_height,
                        panorama_transition
                            && std::env::var_os("RAW_EDITOR_FLAT_BLEND_OVERLAPS").is_some(),
                    );
                    // Cell-centre ownership decisions can miss a thin valid
                    // sliver inside an otherwise rejected cell. Preserve the
                    // sharp owner everywhere it already exists, but always
                    // admit real source pixels into still-uncovered holes so
                    // the final crop represents the source union rather than
                    // the largest accidental hole-free island.
                    fill_streaming_uncovered_tile_region(
                        &mut tile,
                        &mut tile_mask,
                        &mut tile_owner,
                        source_owner_id,
                        &sampler,
                        &write_tone,
                        tile_left,
                        tile_top,
                        x_start,
                        x_end,
                        y_start,
                        y_end,
                        layer_scale,
                    );
                }
                store.save_tile(tile_column, tile_row, &tile, &tile_mask)?;
                store.save_owner_tile(tile_column, tile_row, &tile_owner)?;
            }
        }
    }
    let crop = store
        .covered_bounds()?
        .ok_or("The aligned stack has no covered pixels.")?;
    println!(
        "  - Streaming complete source-union bounds: {}x{} at ({}, {})",
        crop.2, crop.3, crop.0, crop.1
    );
    let mut output = materialize_streaming_focus_output(&store, crop)?;
    // Every layer receives a low-frequency tone field from its valid overlap;
    // the first layer of a new camera position uses the existing mosaic as its
    // reference, while later focal layers use only their station owner. Once
    // ownership is final, estimate any residual group constants from blurred,
    // non-black paper samples across the owner boundaries. This bounded gain
    // pass never averages the high-frequency painting detail.
    // FlatArtwork does not run the relation-based owner gain solve below: its
    // boundary samples frequently compare unlike brush content. The source
    // owner low-frequency pass is the bounded residual correction. Focus-stack
    // output keeps the historical relation-gain ordering.
    let seam_gains = if panorama_transition {
        None
    } else {
        streaming_seam_harmonization(&store, crop, &[])?
    };
    if std::env::var_os("RAW_EDITOR_SKIP_STREAMING_ILLUMINATION").is_none() {
        smooth_streaming_low_frequency_illumination(&mut output);
    }
    if !panorama_transition
        && std::env::var_os("RAW_EDITOR_SKIP_FLAT_SEAM_HARMONIZATION").is_none()
        && let Some((gains, scale)) = seam_gains.as_ref()
    {
        apply_streaming_seam_harmonization(&mut output, gains, *scale);
    }
    if panorama_transition {
        if std::env::var_os("RAW_EDITOR_SKIP_SOURCE_OWNER_CORRECTION").is_none() {
            apply_streaming_source_owner_low_frequency_correction(&store, crop, &mut output)?;
        }
        if std::env::var_os("RAW_EDITOR_ENABLE_FLAT_ARTWORK_CONSENSUS").is_some() {
            low_frequency_consensus.apply(&mut output, crop);
        }
    } else {
        low_frequency_consensus.apply(&mut output, crop);
    }
    println!(
        "  - Verified covered output: {}x{}; every pixel has source ownership",
        output.width(),
        output.height()
    );
    Ok(output)
}
