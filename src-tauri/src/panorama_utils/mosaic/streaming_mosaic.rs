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
use serde_json::json;
use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;
use tauri::AppHandle;
use tauri::Emitter;
use tauri::Runtime;

use super::{
    Field, GRID_STEP, LayerSampler, MosaicCoverageObservation, NativeRefineMode,
    STREAMING_ANALYSIS_LONG_SIDE, STREAMING_FLAT_TONE_SIGMA_CELLS, STREAMING_OWNERSHIP_FEATHER,
    STREAMING_TILE_SIZE, StreamingLowFrequencyConsensus, StreamingMosaicStore,
    apply_streaming_seam_harmonization, apply_streaming_source_owner_low_frequency_correction,
    copy_streaming_owned_tile_region, copy_streaming_tile_region,
    fill_streaming_uncovered_tile_region, intra_station_frame_record,
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
    coverage_observation: &mut Option<MosaicCoverageObservation>,
    retain_quality_evidence: bool,
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
    let diagnostics_dir =
        std::env::var_os("RAW_EDITOR_STREAMING_MOSAIC_DIAGNOSTICS_DIR").map(PathBuf::from);
    let mut overlap_diagnostics = Vec::new();
    let mut tone_diagnostics = Vec::new();
    let mut tone_replay = super::artwork_tone::ArtworkToneReplay::default();
    let mut quality_capture =
        (retain_quality_evidence && panorama_transition && projection == Projection::Planar)
            .then(super::legacy_quality::LegacyQualityCapture::default);
    let mut owner_overlap_gain = super::owner_overlap_gain::OwnerOverlapGain::default();
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
            if panorama_transition && diagnostics_dir.is_some() {
                overlap_diagnostics.extend(measure_source_overlap(
                    source_owner_id,
                    &base_analysis,
                    &base_analysis_mask,
                    &base_analysis_owner,
                    &candidate,
                    &candidate_mask,
                ));
            }
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
        if panorama_transition
            && std::env::var_os("RAW_EDITOR_MATCHED_ARTWORK_TONE").is_some()
            && std::env::var_os("RAW_EDITOR_LEGACY_ARTWORK_TONE").is_none()
            && std::env::var_os("RAW_EDITOR_SKIP_FLAT_TONE_CORRECTION").is_none()
            && let Some((candidate, candidate_mask)) = overlap_candidate.as_ref()
        {
            let (matched, summary) = super::artwork_tone::artwork_tone(
                &base_analysis,
                &base_analysis_mask,
                &base_analysis_owner,
                candidate,
                candidate_mask,
                &write_tone,
            );
            tone_replay.record(
                source_owner_id,
                left,
                top,
                layer_scale,
                &write_tone,
                matched,
            );
            println!("    - Frozen artwork tone '{}': {summary}", info.filename);
            if diagnostics_dir.is_some() {
                tone_diagnostics.push(serde_json::json!({"source":info.filename,"owner":source_owner_id,"evidence":summary}));
            }
        }
        if panorama_transition {
            if let Some((candidate, candidate_mask)) = overlap_candidate.as_ref() {
                owner_overlap_gain.observe(
                    source_owner_id,
                    &base_analysis,
                    &base_analysis_mask,
                    &base_analysis_owner,
                    candidate,
                    candidate_mask,
                    &write_tone,
                    left,
                    top,
                    layer_scale,
                );
            }
        }
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
        if let Some(capture) = quality_capture.as_mut() {
            capture.observe(
                source_owner_id,
                capture_group_id,
                &sampler,
                &write_tone,
                right,
                bottom,
            );
        }
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
    let (source_covered_pixels, source_owned_pixels) = store.covered_and_owned_pixel_counts();
    if source_owned_pixels != source_covered_pixels {
        return Err(
            "The streamed source coverage contains pixels without source ownership.".into(),
        );
    }
    let coverage = store.materialize_coverage(crop);
    if coverage
        .as_raw()
        .iter()
        .filter(|&&pixel| pixel != 0)
        .count() as u64
        != source_covered_pixels
    {
        return Err("The streamed source coverage changed during materialization.".into());
    }
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
    let tone_replay_summary = if panorama_transition {
        Some(tone_replay.apply(&store, crop, &mut output)?)
    } else {
        None
    };
    if let Some(summary) = tone_replay_summary.as_ref() {
        println!("  - Frozen tone replay: {summary}");
    }
    if std::env::var_os("RAW_EDITOR_SKIP_STREAMING_ILLUMINATION").is_none() {
        if panorama_transition {
            if std::env::var_os("RAW_EDITOR_ENABLE_ARTWORK_ILLUMINATION").is_some() {
                super::streaming_illumination::smooth_artwork_low_frequency_illumination(
                    &mut output,
                );
            } else {
                println!(
                    "  - FlatArtwork illumination: source-overlap-only; no canvas material normalization"
                );
            }
        } else {
            smooth_streaming_low_frequency_illumination(&mut output);
        }
    }
    if !panorama_transition
        && std::env::var_os("RAW_EDITOR_SKIP_FLAT_SEAM_HARMONIZATION").is_none()
        && let Some((gains, scale)) = seam_gains.as_ref()
    {
        apply_streaming_seam_harmonization(&mut output, gains, *scale);
    }
    let mut owner_gain_summary = None;
    let mut owner_gain_reference = None;
    if panorama_transition {
        if std::env::var_os("RAW_EDITOR_SKIP_SOURCE_OWNER_CORRECTION").is_none() {
            if std::env::var_os("RAW_EDITOR_LEGACY_SOURCE_OWNER_CORRECTION").is_some() {
                apply_streaming_source_owner_low_frequency_correction(&store, crop, &mut output)?;
            } else {
                let (summary, reference) = owner_overlap_gain.apply(&store, crop, &mut output)?;
                owner_gain_reference = Some(reference);
                println!("  - Evidence constrained owner gain: {summary}");
                owner_gain_summary = Some(summary);
            }
        }
        if std::env::var_os("RAW_EDITOR_ENABLE_FLAT_ARTWORK_CONSENSUS").is_some() {
            low_frequency_consensus.apply(&mut output, crop);
        }
    } else {
        low_frequency_consensus.apply(&mut output, crop);
    }
    if let Some(directory) = diagnostics_dir {
        std::fs::create_dir_all(&directory)
            .map_err(|error| format!("Could not create streaming mosaic diagnostics: {error}"))?;
        let scale = (f64::from(crop.2.max(crop.3)) / 2048.0).max(1.0);
        let width = (f64::from(crop.2) / scale).ceil() as u32;
        let height = (f64::from(crop.3) / scale).ceil() as u32;
        let (_, diagnostic_coverage, diagnostic_owner) =
            store.analysis_region(crop.0, crop.1, scale, width, height)?;
        diagnostic_coverage
            .save(directory.join("coverage.png"))
            .map_err(|error| format!("Could not write streaming coverage diagnostic: {error}"))?;
        diagnostic_owner
            .save(directory.join("owner.png"))
            .map_err(|error| format!("Could not write streaming owner diagnostic: {error}"))?;
        let metadata = serde_json::json!({
            "source_union_crop": [crop.0, crop.1, crop.2, crop.3],
            "analysis_width": width,
            "analysis_height": height,
            "analysis_scale": scale,
            "source_covered_pixels": source_covered_pixels,
            "source_owned_pixels": source_owned_pixels,
            "owner_gain": owner_gain_summary,
            "tone_observations": tone_diagnostics,
            "tone_replay": tone_replay_summary,
            "matched_tone_enabled": std::env::var_os("RAW_EDITOR_MATCHED_ARTWORK_TONE").is_some(),
            "canvas_material_normalization_enabled": std::env::var_os("RAW_EDITOR_ENABLE_ARTWORK_ILLUMINATION").is_some(),
        });
        std::fs::write(
            directory.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)
                .map_err(|error| format!("Could not encode streaming diagnostics: {error}"))?,
        )
        .map_err(|error| format!("Could not write streaming diagnostics metadata: {error}"))?;
        std::fs::write(
            directory.join("source-overlap.json"),
            serde_json::to_vec_pretty(&overlap_diagnostics)
                .map_err(|error| format!("Could not encode source overlap diagnostics: {error}"))?,
        )
        .map_err(|error| format!("Could not write source overlap diagnostics: {error}"))?;
    }
    let mut coverage = coverage;
    let mut publication_crop = None;
    if panorama_transition && std::env::var_os("RAW_EDITOR_ARTWORK_FRAMED_OUTPUT").is_some() {
        let frame = StreamingMosaicStore::largest_fully_covered_rect(&coverage)
            .ok_or("The artwork has no fully covered publication frame.")?;
        let frame_pixels = u64::from(frame.2) * u64::from(frame.3);
        let source_canvas_pixels = u64::from(crop.2) * u64::from(crop.3);
        if frame_pixels * 4 < source_covered_pixels * 3
            || frame_pixels * 20 < source_canvas_pixels * 13
        {
            return Err(
                "The complete artwork frame would discard too much source coverage.".into(),
            );
        }
        output = image::imageops::crop_imm(&output, frame.0, frame.1, frame.2, frame.3).to_image();
        coverage =
            image::imageops::crop_imm(&coverage, frame.0, frame.1, frame.2, frame.3).to_image();
        publication_crop = Some(frame);
        println!(
            "  - Framed artwork publication: {}x{} at ({}, {}), retained {} of {} covered source pixels",
            frame.2, frame.3, frame.0, frame.1, frame_pixels, source_covered_pixels
        );
    }
    let delivered_crop = publication_crop
        .map(|frame| (crop.0 + frame.0, crop.1 + frame.1, frame.2, frame.3))
        .unwrap_or(crop);
    let quality = quality_capture
        .map(|capture| {
            capture.finish(
                &store,
                delivered_crop,
                &coverage,
                (offset_x, offset_y),
                owner_gain_reference,
                &mut |source| {
                    let info = images
                        .iter()
                        .find(|info| info.filename == source.path)
                        .ok_or("legacy evidence source missing from render manifest")?;
                    load(info)
                },
            )
        })
        .transpose()?;
    *coverage_observation = Some(MosaicCoverageObservation {
        source_covered_pixels,
        source_owned_pixels: Some(source_owned_pixels),
        output_pixels: u64::from(crop.2) * u64::from(crop.3),
        coverage: Some(coverage),
        publication_crop,
        quality,
    });
    println!(
        "  - Verified source-union output: {}x{}; {} pixels covered and {} owned",
        output.width(),
        output.height(),
        source_covered_pixels,
        source_owned_pixels
    );
    Ok(output)
}

fn measure_source_overlap(
    candidate_owner: u8,
    base: &Rgb32FImage,
    base_mask: &GrayImage,
    owners: &GrayImage,
    candidate: &Rgb32FImage,
    candidate_mask: &GrayImage,
) -> Vec<serde_json::Value> {
    #[derive(Default)]
    struct Count {
        overlap: u64,
        textured: u64,
        matched: u64,
        delta_sum: f64,
        delta_abs_sum: f64,
        quadrants: [u64; 4],
    }
    let mut counts = HashMap::<u8, Count>::new();
    let (width, height) = base.dimensions();
    if width < 7 || height < 7 {
        return Vec::new();
    }
    for y in (3..height - 3).step_by(8) {
        for x in (3..width - 3).step_by(8) {
            if base_mask.get_pixel(x, y)[0] == 0 || candidate_mask.get_pixel(x, y)[0] == 0 {
                continue;
            }
            let owner = owners.get_pixel(x, y)[0];
            if owner == 0 || owner == candidate_owner {
                continue;
            }
            let count = counts.entry(owner).or_default();
            count.overlap += 1;
            let mut a = [0.0_f64; 25];
            let mut b = [0.0_f64; 25];
            let mut valid = true;
            let mut n = 0;
            for dy in -2_i32..=2 {
                for dx in -2_i32..=2 {
                    let px = (x as i32 + dx) as u32;
                    let py = (y as i32 + dy) as u32;
                    if base_mask.get_pixel(px, py)[0] == 0
                        || candidate_mask.get_pixel(px, py)[0] == 0
                    {
                        valid = false;
                        break;
                    }
                    let luma = |pixel: &image::Rgb<f32>| {
                        f64::from(pixel[0]) * 0.2126
                            + f64::from(pixel[1]) * 0.7152
                            + f64::from(pixel[2]) * 0.0722
                    };
                    a[n] = luma(base.get_pixel(px, py));
                    b[n] = luma(candidate.get_pixel(px, py));
                    n += 1;
                }
                if !valid {
                    break;
                }
            }
            if !valid || n != 25 {
                continue;
            }
            let mean_a = a.iter().sum::<f64>() / 25.0;
            let mean_b = b.iter().sum::<f64>() / 25.0;
            if !(0.06..=0.93).contains(&mean_a) || !(0.06..=0.93).contains(&mean_b) {
                continue;
            }
            let mut var_a = 0.0;
            let mut var_b = 0.0;
            let mut cross = 0.0;
            for i in 0..25 {
                let da = a[i] - mean_a;
                let db = b[i] - mean_b;
                var_a += da * da;
                var_b += db * db;
                cross += da * db;
            }
            if var_a < 0.0004 || var_b < 0.0004 {
                continue;
            }
            count.textured += 1;
            if cross / (var_a * var_b).sqrt() < 0.8 {
                continue;
            }
            count.matched += 1;
            let delta = mean_b - mean_a;
            count.delta_sum += delta;
            count.delta_abs_sum += delta.abs();
            count.quadrants[usize::from(x >= width / 2) + 2 * usize::from(y >= height / 2)] += 1;
        }
    }
    let mut result: Vec<_> = counts
        .into_iter()
        .map(|(base_owner, count)| {
            json!({
                "base_owner": base_owner,
                "candidate_owner": candidate_owner,
                "overlap_samples": count.overlap,
                "textured_samples": count.textured,
                "same_content_samples": count.matched,
                "same_content_fraction_of_textured": if count.textured > 0 { count.matched as f64 / count.textured as f64 } else { 0.0 },
                "mean_candidate_minus_base_luma": if count.matched > 0 { count.delta_sum / count.matched as f64 } else { 0.0 },
                "mean_absolute_luma_difference": if count.matched > 0 { count.delta_abs_sum / count.matched as f64 } else { 0.0 },
                "same_content_quadrants": count.quadrants,
            })
        })
        .collect();
    result.sort_by_key(|value| value["base_owner"].as_u64().unwrap_or(0));
    result
}
