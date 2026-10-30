//! Streaming (tiled) detail-preserving focus mosaic.
//!
//! One function by design: layer refinement, ownership, tile writes and the
//! low-frequency harmonization run in a fixed order over the on-disk canvas.

use super::super::stack_pipeline::intra_station;
use super::super::stack_pipeline::report::IntraStationFrameRecord;
use super::super::stack_pipeline::report::IntraStationFrameStatus;
use super::super::stitching::Projection;
use super::super::stitching::transformed_image_region;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Luma;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use std::collections::HashMap;
use std::collections::HashSet;
use tauri::AppHandle;
use tauri::Emitter;
use tauri::Runtime;

use super::{
    DetailGrid, Field, FrameDetail, GRID_STEP, LayerSampler, MosaicToneFinish, NativeRefineMode,
    OwnerStations, STREAMING_ANALYSIS_LONG_SIDE, STREAMING_FLAT_TONE_SIGMA_CELLS,
    STREAMING_OWNERSHIP_FEATHER, STREAMING_SCAN_EDGE_MARGIN_FRACTION,
    STREAMING_SCAN_TRANSITION_FOCUS_WEIGHT, STREAMING_SEAM_TONE_SIGMA_CELLS, STREAMING_TILE_SIZE,
    StreamingLowFrequencyConsensus, StreamingMosaicStore, analysis_density_source,
    apply_streaming_seam_harmonization, apply_streaming_source_owner_low_frequency_correction,
    copy_streaming_owned_tile_region, copy_streaming_tile_region,
    fill_streaming_uncovered_tile_region, intra_station_frame_record,
    materialize_streaming_focus_output, measure_frame_detail, measure_output_focus,
    planar_canvas_scale, prefilter_minified_source, refine_layer_from_analysis,
    refine_native_layer_streaming, smooth_streaming_low_frequency_illumination, soft_frames,
    streaming_candidate_analysis, streaming_ownership_from_analysis, streaming_ownership_grid,
    streaming_refinement_pair, streaming_seam_harmonization, streaming_tone_field_from_analysis,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn detail_preserving_mosaic_streaming<R: Runtime, F>(
    images: &[&ImageInfo],
    homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    capture_group_ids: Option<&HashMap<usize, u8>>,
    panorama_transition: bool,
    tone_finish: MosaicToneFinish,
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
    let mut owner_stations = OwnerStations::identity();
    // Runs other than a focus scan keep comparing owner labels with station
    // ids directly, as they always have.
    let identity_stations = OwnerStations::identity();
    #[cfg(test)]
    let mut owner_names = HashMap::<u8, String>::new();
    // A focus scan: camera positions along the artwork, each with one or
    // more focus brackets, toned only by overlap evidence.
    let focus_scan = panorama_transition && tone_finish == MosaicToneFinish::OverlapMeasured;
    println!(
        "  - Streaming detail-preserving mosaic canvas: {width}x{height}, tiles {}x{} of {}px",
        store.tile_columns, store.tile_rows, STREAMING_TILE_SIZE
    );
    // A focus scan leaves its mostly soft frames out of the composition; see
    // `soft_frames` for the rule and why grouping still sees them.
    let skipped_soft_frames = if focus_scan && projection == Projection::Planar {
        let grid = DetailGrid::new(width, height);
        let mut details = Vec::with_capacity(images.len());
        for (index, &info) in images.iter().enumerate() {
            let _ = app.emit(
                event,
                format!(
                    "Measuring the detail of source {} of {}",
                    index + 1,
                    images.len()
                ),
            );
            let transform = &homographies[&info.id];
            let Some((left, right, top, bottom)) = transformed_image_region(
                info, transform, projection, offset_x, offset_y, width, height,
            ) else {
                details.push(FrameDetail::default());
                continue;
            };
            let (source, source_divisor) =
                prefilter_minified_source(load(info)?, planar_canvas_scale(info, transform));
            let layer_width = right - left + 1;
            let layer_height = bottom - top + 1;
            let layer_scale = (layer_width.max(layer_height) as f64
                / STREAMING_ANALYSIS_LONG_SIDE as f64)
                .max(1.0);
            let sampler = LayerSampler {
                info,
                source: &source,
                source_divisor,
                inverse: transform
                    .try_inverse()
                    .ok_or("The stack contains a non-invertible alignment.")?,
                native_mapping: None,
                projection,
                offset: (offset_x, offset_y),
                left,
                top,
                scale: layer_scale,
                residual: Field::new(
                    (layer_width as f64 / layer_scale).ceil() as u32,
                    (layer_height as f64 / layer_scale).ceil() as u32,
                    GRID_STEP as f64,
                ),
            };
            details.push(measure_frame_detail(
                &sampler,
                (left, right, top, bottom),
                &grid,
            ));
        }
        let skipped = soft_frames(&details, &grid);
        if !skipped.is_empty() {
            println!(
                "  - Focus scan: leaving out {} mostly soft frame(s): {}",
                skipped.len(),
                skipped
                    .iter()
                    .map(|&(index, share)| format!(
                        "{} ({:.0}% of its cells soft)",
                        std::path::Path::new(&images[index].filename)
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                        share * 100.0
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        skipped
            .into_iter()
            .map(|(index, _)| images[index].id)
            .collect::<HashSet<_>>()
    } else {
        HashSet::new()
    };
    for (index, &info) in images.iter().enumerate() {
        if skipped_soft_frames.contains(&info.id) {
            continue;
        }
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
        owner_stations.record(source_owner_id, capture_group_id);
        #[cfg(test)]
        owner_names.insert(source_owner_id, info.filename.clone());
        let _ = app.emit(
            event,
            format!(
                "Streaming detail and selecting sharp source {} of {}",
                index + 1,
                images.len()
            ),
        );
        let source = load(info)?;
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
        let (source, source_divisor) = if projection == Projection::Planar {
            prefilter_minified_source(source, planar_canvas_scale(info, transform))
        } else {
            (source, 1.0)
        };
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
        // A focus scan compares analysis images of silk weave; sample both
        // sides from anti-aliased copies (see `analysis_density_source`).
        let analysis_source = (focus_scan && projection == Projection::Planar)
            .then(|| {
                analysis_density_source(
                    &source,
                    source_divisor,
                    layer_scale,
                    planar_canvas_scale(info, transform),
                )
            })
            .flatten();
        let candidate_analysis = |sampler: &LayerSampler<'_>| match analysis_source.as_ref() {
            Some((analysis, divisor)) => streaming_candidate_analysis(
                &sampler.with_source(analysis, *divisor),
                analysis_width,
                analysis_height,
            ),
            None => streaming_candidate_analysis(sampler, analysis_width, analysis_height),
        };
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
        let (base_analysis, base_analysis_mask, base_analysis_owner) = if focus_scan {
            store.analysis_region_area_averaged(
                left,
                top,
                layer_scale,
                analysis_width,
                analysis_height,
            )?
        } else {
            store.analysis_region(left, top, layer_scale, analysis_width, analysis_height)?
        };
        // Station of every analysed base pixel. A focus scan labels pixels by
        // source; every decision below is about camera positions.
        let station_lookup = if focus_scan {
            &owner_stations
        } else {
            &identity_stations
        };
        let base_analysis_station = GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
            Luma([station_lookup.station(base_analysis_owner.get_pixel(x, y)[0])])
        });
        let has_existing_coverage = base_analysis_mask.as_raw().iter().any(|&value| value != 0);
        let mut overlap_candidate = None;
        let (tone, decision) = if has_existing_coverage {
            // A later focal plane must register against its own camera
            // position.  Using pixels already owned by another panorama tile
            // pulls repeated calligraphy toward the wrong column. Keep those
            // older positions available for colour matching, but exclude them
            // from residual motion estimation. The first layer of a position
            // has no same-station pixels and registers against the whole
            // overlap with the existing mosaic.
            let group_refinement_mask =
                (!first_capture_group_layer && (!panorama_transition || focus_scan)).then(|| {
                    GrayImage::from_fn(analysis_width, analysis_height, |x, y| {
                        Luma([
                            if base_analysis_mask.get_pixel(x, y)[0] != 0
                                && base_analysis_station.get_pixel(x, y)[0] == capture_group_id
                            {
                                255
                            } else {
                                0
                            },
                        ])
                    })
                });
            let refinement_mask = group_refinement_mask
                .as_ref()
                .unwrap_or(&base_analysis_mask);
            // 需求 2.2: two global + local rounds at the analysis resolution,
            // then the native patch refinement below.
            for _ in 0..intra_station::INTRA_STATION_ANALYSIS_ROUNDS {
                let (candidate, candidate_mask) = candidate_analysis(&sampler);
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
            let mut bracket_registered = false;
            if !first_capture_group_layer {
                let anchor = station_anchors.get(&capture_group_id);
                let anchor_long_side = anchor
                    .map(|&(_, _, long_side)| long_side)
                    .unwrap_or_else(|| info.width.max(info.height));
                let outcome = refine_native_layer_streaming(
                    &store,
                    Some((capture_group_id, station_lookup)),
                    &mut sampler,
                    analysis_width,
                    analysis_height,
                    (left, right, top, bottom),
                    NativeRefineMode::IntraStation { anchor_long_side },
                )?;
                bracket_registered = outcome.inlier_area_coverage
                    >= intra_station::INTRA_STATION_MIN_INLIER_AREA_COVERAGE;
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
            let (candidate, candidate_mask) = candidate_analysis(&sampler);
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
                // A focus scan also matches the seam itself: with only the
                // overlap evidence left to set tone, a whole-overlap average
                // leaves the vignetting step of two frames at the seam.
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
                    focus_scan.then_some(STREAMING_SEAM_TONE_SIGMA_CELLS),
                )
            } else {
                // A bracket of a focus scan covers its whole station and may
                // take any part of it, so it matches the canvas it replaces:
                // all covered pixels, not only those its own station kept.
                streaming_tone_field_from_analysis(
                    &base_analysis,
                    if focus_scan {
                        &base_analysis_mask
                    } else {
                        refinement_mask
                    },
                    &candidate,
                    &candidate_mask,
                    1.0,
                    None,
                )
            };
            // The first layer of a camera position joins the mosaic through a
            // seam. In a focus scan a later, registered focus bracket of the
            // same position competes on focus inside the pixels its own
            // station owns; the scan used to treat every bracket as a new
            // position, so the sharper bracket was never chosen where it
            // overlapped. A bracket whose registration failed joins through a
            // seam like a new position, and in a focus scan with the same
            // gated focus vote: it may be displaced by a few pixels, so its
            // focus counts only where it agrees with the canvas after the
            // transition low-pass. A purely spatial seam let a defocused
            // bracket (祁序1 DSC_3861 at 1:1) own part of the canvas.
            let seam_transition = first_capture_group_layer
                || (panorama_transition && (!focus_scan || !bracket_registered));
            // Focus is measured on output pixels: the analysis images cannot
            // resolve the silk weave or the blur of a defocused bracket.
            let output_focus = if focus_scan {
                Some(measure_output_focus(
                    &store,
                    &sampler,
                    (left, right, top, bottom),
                    streaming_ownership_grid(analysis_width, analysis_height),
                )?)
            } else {
                None
            };
            let decision = streaming_ownership_from_analysis(
                &base_analysis,
                &base_analysis_mask,
                &candidate,
                &candidate_mask,
                &tone,
                &base_analysis_station,
                capture_group_id,
                seam_transition,
                if focus_scan && seam_transition {
                    STREAMING_SCAN_TRANSITION_FOCUS_WEIGHT
                } else {
                    0.0
                },
                if focus_scan {
                    STREAMING_SCAN_EDGE_MARGIN_FRACTION
                        * f64::from(analysis_width.max(analysis_height))
                } else {
                    0.0
                },
                output_focus.as_ref(),
            );
            println!(
                "    - Ownership by {}: {} of {} cells{}",
                if seam_transition {
                    "seam"
                } else {
                    "focus contest"
                },
                decision
                    .as_raw()
                    .iter()
                    .filter(|&&value| value > 127)
                    .count(),
                decision.as_raw().len(),
                output_focus
                    .as_ref()
                    .map(|focus| format!(
                        ", output-pixel focus on {} cells",
                        focus.base.iter().filter(|value| value.is_finite()).count()
                    ))
                    .unwrap_or_default()
            );
            #[cfg(test)]
            super::diagnostics::write_ownership_trace(
                index,
                &info.filename,
                left,
                top,
                layer_scale,
                &base_analysis_owner,
                &owner_names,
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
    #[cfg(test)]
    if let Some(path) = std::env::var_os("RAW_EDITOR_STREAMING_OWNER_MAP") {
        super::diagnostics::write_streaming_owner_map(
            &store,
            crop,
            &owner_names,
            std::path::Path::new(&path),
        )?;
    }
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
    // A focus scan keeps the tone each layer was written with: that field was
    // measured on overlap pixels both sources saw. The two levelling passes
    // below compare unlike regions instead, which is valid only when every
    // region holds the same kind of paper.
    let content_levelling = tone_finish == MosaicToneFinish::ContentLevelling;
    if !content_levelling {
        println!(
            "  - Focus scan tone: keeping the overlap-measured layer tone; content levelling is off"
        );
    }
    if content_levelling && std::env::var_os("RAW_EDITOR_SKIP_STREAMING_ILLUMINATION").is_none() {
        smooth_streaming_low_frequency_illumination(&mut output);
    }
    if !panorama_transition
        && std::env::var_os("RAW_EDITOR_SKIP_FLAT_SEAM_HARMONIZATION").is_none()
        && let Some((gains, scale)) = seam_gains.as_ref()
    {
        apply_streaming_seam_harmonization(&mut output, gains, *scale);
    }
    if panorama_transition {
        if content_levelling
            && std::env::var_os("RAW_EDITOR_SKIP_SOURCE_OWNER_CORRECTION").is_none()
        {
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
