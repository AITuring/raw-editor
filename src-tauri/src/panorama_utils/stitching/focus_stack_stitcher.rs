//! The global focus-stack compositor body.
//!
//! Renders layers, decides ownership, harmonises tone and fills margins under
//! one margin policy; the optional final tile sharpen stays off by default.

use super::super::coordinates::Projection;
use super::super::coordinates::pixel_aligned_canvas;
use super::super::focus_geometry::focus_stack_is_shifted_mosaic;
use super::super::render_geometry::focus_output_bounds;
use super::super::stack_pipeline::degradation;
use super::super::stack_pipeline::focus_fuser;
use super::super::stack_pipeline::focus_fuser::CandidateCells;
use super::super::stack_pipeline::focus_fuser::StationFusion;
use super::super::stack_pipeline::intra_station;
use super::super::stack_pipeline::report::FusionReport;
use super::super::stack_pipeline::report::IntraStationFrameRecord;
use super::super::stack_pipeline::report::IntraStationFrameStatus;
use super::super::stack_pipeline::report::OwnershipGridSize;
use super::super::stack_pipeline::station_degradation::GroupJoinEvidence;
use super::super::stack_pipeline::station_degradation::group_join_rejection;
use super::super::stack_pipeline::station_degradation::plan_station_fusion;
use super::super::stack_pipeline::station_degradation::record_run_entries;
use super::super::stack_pipeline::station_degradation::record_run_group_join_rejection;
use super::super::stack_pipeline::station_degradation::station_plan_entries;
use crate::panorama_utils::shared::FocusLayerWarp;
use crate::panorama_utils::shared::ImageInfo;
use image::GrayImage;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use tauri::AppHandle;
use tauri::Emitter;
use tauri::Runtime;

use super::{
    ConfidenceMap, CoverageMask, FOCUS_ALLOW_LOW_FREQUENCY_SEAM_BLEND,
    FOCUS_NATIVE_REFINEMENT_MIN_OFFSET_PX, FocusSourceRefinement, FocusSourceSampling,
    FocusStackTileMasks, FocusStackTileRender, NO_OWNER, OwnershipMap,
    align_focus_foreground_layer_to_existing, blend_focus_seam_band, commit_focus_analysis_scores,
    estimate_focus_layer_translation, fill_focus_canvas_margins, focus_analysis_dimensions,
    focus_cell_decision_for_layer, focus_cell_plane_to_pixels, focus_cell_texture_to_pixels,
    focus_decision_mask, focus_score_map, hard_select_focus_layer, mark_focus_foreground_pixels,
    mark_focus_owner_pixels, measure_focus_cell_disagreement, measure_focus_cells,
    native_station_geometry, order_focus_station_frames, place_focus_foreground_mask,
    place_focus_layer, record_focus_layer_ownership, register_station_frame_native_layer,
    render_focus_analysis_layer, render_focus_layer, render_focus_layer_with_refinement,
    resize_binary_mask, resize_rgb, station_members_from_records, suppress_focus_canvas_switches,
    suppress_focus_foreground_ownership_switches,
    suppress_focus_foreground_ownership_switches_on_canvas, suppress_focus_foreground_switches,
    suppress_focus_foreground_switches_on_canvas, translate_rendered_focus_layer,
};
#[cfg(test)]
use super::{
    ScoreNoiseDiagnostics, ScoreNoiseRegistrationCell, owner_quality_forced_labeling,
    score_noise_registration_cells,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn focus_stack_stitcher_with_margin_policy<R: Runtime, F>(
    images: &[&ImageInfo],
    global_homographies: &HashMap<usize, Matrix3<f64>>,
    projection: Projection,
    focus_warp: Option<&FocusLayerWarp>,
    capture_group_ids: Option<&HashMap<usize, u8>>,
    station_index: usize,
    sequence_gap_aware: bool,
    fill_margins: bool,
    app_handle: AppHandle<R>,
    progress_event: &str,
    load_image: &mut F,
) -> Result<FocusStackTileRender, String>
where
    F: FnMut(&ImageInfo) -> Result<Rgb32FImage, String>,
{
    // A sequence-gap render preserves its spatial order because its members
    // are separate camera positions. A compact Capture_Station, however,
    // must use the sharpest valid frame as the anchor before native registration
    // and focus ownership are measured. Reordering only this local slice leaves
    // the global station topology and all source identities unchanged.
    let ordered_images = if !sequence_gap_aware {
        order_focus_station_frames(images, station_index)
    } else {
        images.to_vec()
    };
    let images = ordered_images.as_slice();
    if images.is_empty() {
        return Ok(FocusStackTileRender::unattributed(Rgb32FImage::new(0, 0)));
    }
    let capture_group_count = capture_group_ids
        .map(|ids| {
            ids.values()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len()
        })
        .unwrap_or(0);
    // Focus breathing and small tripod movement can trip the framing/scale
    // heuristic even when every source belongs to one focus bracket.  The
    // hard, position-owned mosaic is only appropriate when registration has
    // independently recovered multiple camera positions (or an explicit
    // sequence gap).  A single capture group must retain the mature focus
    // fusion path; otherwise its cell-wise local warp visibly fragments
    // strokes that are intentionally present at different focal depths.
    let shifted_mosaic = sequence_gap_aware
        || (capture_group_count > 1
            && focus_stack_is_shifted_mosaic(images, global_homographies, projection));
    if !shifted_mosaic {
        println!(
            "  - Compact focus capture detected ({} inferred group(s)); preserving standard focus fusion",
            capture_group_count.max(1)
        );
    }
    if shifted_mosaic {
        println!("  - Framing/lens shift detected; using local registration and detail ownership");
        let _ = app_handle.emit(
            progress_event,
            "Framing or lens change detected; refining registration and selecting sharp detail...",
        );
        return crate::panorama_utils::mosaic::detail_preserving_mosaic(
            images,
            global_homographies,
            projection,
            capture_group_ids,
            sequence_gap_aware,
            app_handle,
            progress_event,
            load_image,
        )
        .map(FocusStackTileRender::unattributed);
    }
    // An attributed station measures its own native residual against its
    // anchor. Global scan depth bands can carry a different camera position's
    // mask fit and are not evidence for this bracket. Retain independently
    // observed physical-edge consensus; measure other local motion here.
    let native_station_warp = (!fill_margins).then(|| {
        focus_warp
            .map(native_station_geometry)
            .unwrap_or(FocusLayerWarp { bands: Vec::new() })
    });
    let focus_warp = native_station_warp
        .as_ref()
        .map_or(focus_warp, |warp| (!warp.bands.is_empty()).then_some(warp));
    let (min_x, max_x, min_y, max_y) =
        focus_output_bounds(images, global_homographies, projection, focus_warp);
    if !min_x.is_finite() || !max_x.is_finite() || !min_y.is_finite() || !max_y.is_finite() {
        return Ok(FocusStackTileRender::unattributed(Rgb32FImage::new(0, 0)));
    }
    let (offset_x, out_width) = pixel_aligned_canvas(min_x, max_x);
    let (offset_y, out_height) = pixel_aligned_canvas(min_y, max_y);
    // 需求 12.1 / 12.2 / 12.5: which members this Capture_Station can fuse, as
    // the records of earlier runs of this station already describe them.  The
    // loop below measures this run's evidence itself (需求 2.2–2.9) and updates
    // `station_members` frame by frame, so the plan that is finally *recorded* is
    // the one the pixels followed.  The canvas bounds above deliberately stay
    // over the whole group: the Virtual_Tile geometry the caller already
    // published is computed from the same set, and a degraded station must not
    // silently resize its tile.
    let (mut station_members, group_join_rejections) =
        station_members_from_records(images, station_index);
    for rejection in &group_join_rejections {
        println!(
            "  - Rejected from every Capture_Station: '{}' inlier support {:.3} (minimum {:.2}), median symmetric error {:.2}px (limit {:.2}px)",
            rejection.path,
            rejection.inlier_area_coverage,
            rejection.minimum_inlier_area_coverage,
            rejection.median_symmetric_error_px,
            rejection.symmetric_error_limit_px
        );
        record_run_group_join_rejection(station_index, rejection);
    }
    let station_plan = plan_station_fusion(&station_members);
    // 需求 3.10 and 12.1 both retain source-exact pixel writes; the former
    // physically contains one successful frame, while the latter selects one
    // survivor from a larger station.
    // 需求 12.1: a degraded station keeps one Source_RAW, so `fused` holds that
    // frame alone and every covered pixel below is copied from it.
    let seed_index = station_plan.fused.first().copied().unwrap_or(0);
    // 需求 2.1 / 2.9: the seed frame is the one every later frame of this station
    // is composited against, so it is this station's anchor; it carries the
    // identity transform and its record is the reference the other frames'
    // transforms are expressed against.
    let anchor_path = images[seed_index].filename.clone();
    let anchor_long_side = images[seed_index].width.max(images[seed_index].height);
    let anchor_to_station_inverse = global_homographies
        .get(&images[seed_index].id)
        .and_then(|transform| transform.try_inverse());
    intra_station::record_run_frame(
        station_index,
        &anchor_path,
        IntraStationFrameRecord {
            path: anchor_path.clone(),
            status: IntraStationFrameStatus::Anchor,
            inlier_area_coverage: 1.0,
            ..Default::default()
        },
    );
    if let Some(member) = station_members.get_mut(seed_index) {
        member.status = IntraStationFrameStatus::Anchor;
    }
    let mut source_refinements = HashMap::new();
    source_refinements.insert(
        images[seed_index].id,
        FocusSourceRefinement {
            source_sampling: Some(FocusSourceSampling::new(
                images[seed_index],
                &global_homographies[&images[seed_index].id],
                projection,
                focus_warp,
            )),
            ..Default::default()
        },
    );
    let first_source = load_image(images[seed_index])?;
    let first_layer = render_focus_layer(
        images[seed_index],
        &first_source,
        &global_homographies[&images[seed_index].id],
        projection,
        focus_warp,
        offset_x,
        offset_y,
        out_width,
        out_height,
    );
    drop(first_source);
    // The merged image occupies the final canvas, but each subsequent source
    // layer is kept local to its transformed bounds. This is important for a
    // long scan: allocating a full-canvas candidate for every 60MP source can
    // multiply memory use by the number of frames.
    let mut merged = Rgb32FImage::new(out_width, out_height);
    let mut merged_mask = GrayImage::new(out_width, out_height);
    let mut merged_foreground_mask = GrayImage::new(out_width, out_height);
    let mut merged_owner = GrayImage::new(out_width, out_height);
    place_focus_layer(&mut merged, &mut merged_mask, &first_layer);
    place_focus_foreground_mask(&mut merged_foreground_mask, &first_layer);
    // Published Ownership_Map (需求 3.11 / 4.1).  `merged_owner` above stays
    // exactly as it was: it is an 8-bit tone-harmonisation grouping key with
    // its own identifier scheme, while this map is the per-pixel Source_RAW
    // attribution the Virtual_Tile publishes.  It is only tracked for the
    // unfilled tile form, where every covered pixel is a hard selection from
    // one source; the margin-filling output invents pixels for the projected
    // trapezoid corners and therefore cannot attest ownership.
    let seed_owner = (seed_index + 1).min(u16::MAX as usize) as u16;
    let mut ownership_ids = (!fill_margins).then(|| {
        let mut owners = vec![NO_OWNER; out_width as usize * out_height as usize];
        owners
            .par_iter_mut()
            .zip(merged_mask.as_raw().par_iter())
            .for_each(|(owner, &covered)| {
                if covered > 0 {
                    *owner = seed_owner;
                }
            });
        owners
    });
    merged_owner
        .as_mut()
        .par_chunks_mut(out_width as usize)
        .zip(merged_mask.as_raw().par_chunks(out_width as usize))
        .for_each(|(owner_row, mask_row)| {
            for (owner, mask) in owner_row.iter_mut().zip(mask_row) {
                if *mask > 0 {
                    *owner = 1;
                }
            }
        });
    // Focus_Fuser (需求 3.1–3.11).  The station's ownership grid decides which
    // source owns each cell; the loop below copies that source's pixels and
    // nothing else, so an owner boundary is an ownership cell boundary.
    // 需求 3.10 / 12.1: the frame count is the number of frames the plan fuses,
    // so a single-frame degraded station reports `SingleFrame` and skips the cut
    // it has nothing to cut against.
    let mut fusion = StationFusion::new(
        out_width,
        out_height,
        station_plan.fused.len().min(images.len()).max(1),
    );
    let mut native_displacements = vec![None; images.len()];
    #[cfg(test)]
    fusion.force_test_owners(owner_quality_forced_labeling(station_index));
    let fusion_geometry = fusion.geometry();
    let fusion_plan = fusion.sampling_plan();
    #[cfg(test)]
    let mut score_noise_diagnostics = {
        let write_score_noise =
            std::env::var_os("RAW_EDITOR_FOCUS_SCORE_NOISE_DIAGNOSTICS_DIR").is_some();
        let final_world =
            std::env::var_os("RAW_EDITOR_FOCUS_FINAL_WORLD_OWNER_ORACLE_DIR").is_some();
        // Full native runs can contain fourteen stations and hundreds of
        // thousands of ownership cells.  Keep the existing opt-in diagnostic
        // semantics, but allow a caller to select one station when tracing a
        // concrete Quality_Gate ROI; this avoids producing gigabytes of
        // unrelated score tables while leaving production untouched.
        let requested_station = std::env::var("RAW_EDITOR_FOCUS_SCORE_NOISE_STATION")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let station_selected = requested_station.is_none_or(|requested| requested == station_index);
        let legacy_directory = std::env::var_os("RAW_EDITOR_FOCUS_OWNER_QUALITY_ORACLE_DIR")
            .or_else(|| std::env::var_os("RAW_EDITOR_FOCUS_SCORE_NOISE_DIAGNOSTICS_DIR"))
            .map(PathBuf::from);
        (station_selected && (final_world || legacy_directory.is_some())).then(|| {
            ScoreNoiseDiagnostics::new(
                station_index,
                (out_width, out_height),
                fusion_geometry,
                fusion_plan,
                write_score_noise,
            )
        })
    };
    {
        // The composite currently holds exactly the seed frame, so measuring it
        // here is measuring that frame — with the same plan every candidate
        // will use (需求 3.2).
        let (sharpness, covered) =
            measure_focus_cells(&fusion_geometry, &fusion_plan, &merged, &merged_mask, 0, 0);
        fusion.seed(seed_index, &sharpness, &covered);
        #[cfg(test)]
        if let Some(diagnostics) = score_noise_diagnostics.as_mut() {
            diagnostics.record_candidate(
                seed_index,
                &images[seed_index].filename,
                Matrix3::new(1.0, 0.0, offset_x, 0.0, 1.0, offset_y, 0.0, 0.0, 1.0)
                    * global_homographies[&images[seed_index].id],
                &merged,
                &merged_mask,
                0,
                0,
                &sharpness,
                &covered,
                &vec![0.0; fusion_geometry.cell_count()],
                &vec![ScoreNoiseRegistrationCell::default(); fusion_geometry.cell_count()],
                None,
            );
        }
    }
    println!(
        "  - Focus ownership grid: {}x{} cells of {}px over a {}x{} station plane",
        fusion_geometry.columns,
        fusion_geometry.rows,
        fusion_geometry.cell_size_px,
        out_width,
        out_height
    );
    let source_longest_dimension = images
        .iter()
        .map(|image| image.width().max(image.height()))
        .max()
        .unwrap_or(1);
    let (analysis_width, analysis_height) =
        focus_analysis_dimensions(out_width, out_height, source_longest_dimension);
    let mut merged_analysis = resize_rgb(&merged, analysis_width, analysis_height);
    let mut merged_analysis_mask =
        resize_binary_mask(&merged_mask, analysis_width, analysis_height);
    let mut merged_analysis_foreground_mask =
        resize_binary_mask(&merged_foreground_mask, analysis_width, analysis_height);
    let mut merged_focus = focus_score_map(&merged_analysis, &merged_analysis_mask);

    // 需求 12.2: only the frames the plan fused are loaded, rendered and offered
    // to the cut.  An excluded Source_RAW is never sampled, so no pixel of it can
    // reach the Virtual_Tile while the report calls it excluded.
    for &index in station_plan.fused.iter().skip(1) {
        let image = images[index];
        let _ = app_handle.emit(
            progress_event,
            format!("Focus-stacking image {} of {}", index + 1, images.len()),
        );
        let source_image = load_image(image)?;
        let mut candidate = render_focus_layer(
            image,
            &source_image,
            &global_homographies[&image.id],
            projection,
            focus_warp,
            offset_x,
            offset_y,
            out_width,
            out_height,
        );
        let mut applied_translation = (0i32, 0i32);
        let mut applied_displacement = None;
        if let Some((delta_x, delta_y, score, support, patch_count)) =
            estimate_focus_layer_translation(&candidate, &merged, &merged_mask)
        {
            println!(
                "  - Full-resolution focus refinement: delta=({delta_x},{delta_y}), score={score:.3}, consensus={support}/{patch_count} patches"
            );
            candidate =
                translate_rendered_focus_layer(candidate, delta_x, delta_y, out_width, out_height);
            applied_translation = (delta_x, delta_y);
        }
        // Intra_Station_Registrar (需求 2.2–2.9).  The integer translation above
        // is this frame's *global* model on this path; the registrar now measures
        // and corrects what is left of it at native resolution, against the
        // station canvas the anchor seeded.
        #[cfg(test)]
        let mut score_noise_registration =
            vec![ScoreNoiseRegistrationCell::default(); fusion_geometry.cell_count()];
        let registration = register_station_frame_native_layer(
            &merged,
            &merged_mask,
            image,
            &source_image,
            &global_homographies[&image.id],
            projection,
            (offset_x, offset_y),
            applied_translation,
            &candidate,
            anchor_long_side,
            anchor_to_station_inverse,
            focus_warp,
        );
        if let Some(registration) = registration {
            let record = registration.record.clone();
            #[cfg(test)]
            {
                score_noise_registration = score_noise_registration_cells(
                    &fusion_geometry,
                    &record,
                    registration.displacement.as_ref(),
                );
            }
            println!(
                "    - Intra-station registration '{}': {:?}, {} inliers of {} control points, coverage {:.3}, rejected {:.3}, median symmetric error {:.2}px, control point spacing {:.1}px",
                image.filename,
                record.status,
                record.inliers,
                registration.evaluated_control_points,
                record.inlier_area_coverage,
                record.rejected_control_point_ratio,
                record.median_symmetric_error_px,
                record.control_point_spacing_px
            );
            intra_station::record_run_frame(station_index, &anchor_path, record.clone());
            // 需求 12.5: inlier spatial support below 20%, or an inlier median
            // symmetric reprojection error above 0.01 × the anchor's long side,
            // keeps this Source_RAW out of every Capture_Station — so this
            // station cannot fuse it either (需求 12.2).  A frame whose evidence
            // was never measured has nothing to judge and keeps the behaviour it
            // had before.
            let rejection = (registration.evaluated_control_points > 0 && record.inliers > 0)
                .then(|| {
                    group_join_rejection(&GroupJoinEvidence {
                        path: image.filename.clone(),
                        inlier_area_coverage: record.inlier_area_coverage,
                        median_symmetric_error_px: record.median_symmetric_error_px,
                        anchor_long_side_px: anchor_long_side,
                    })
                })
                .flatten();
            if let Some(member) = station_members.get_mut(index) {
                member.status = match (rejection.is_some(), record.status) {
                    (true, _) => IntraStationFrameStatus::Failed,
                    (false, IntraStationFrameStatus::Failed) => {
                        IntraStationFrameStatus::GlobalFallback
                    }
                    (false, status) => status,
                };
            }
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
            // Apply exactly what was measured: the same displacement field the
            // gates above accepted, resampled onto the layer that produced it.
            if let Some(displacement) = registration.displacement {
                let max_offset = displacement.max_offset_px();
                if max_offset >= FOCUS_NATIVE_REFINEMENT_MIN_OFFSET_PX {
                    println!("    - Applying native refinement field: up to {max_offset:.2}px");
                    native_displacements[index] = Some(displacement.clone());
                    applied_displacement = Some(displacement.clone());
                }
            }
        }
        // Compose the measured inverse field with the member's inverse geometry.
        // A second warp of the developed layer blurs non-anchor frames and makes
        // the focus comparison asymmetric. Final pixels sample the source once.
        if applied_displacement.is_some() {
            let inverse_refinement = FocusSourceRefinement {
                translation: applied_translation,
                displacement: applied_displacement.clone(),
                ..Default::default()
            };
            let (width, height) = candidate.image.dimensions();
            if width > 0 && height > 0 {
                let region = (
                    candidate.left,
                    candidate.left + width - 1,
                    candidate.top,
                    candidate.top + height - 1,
                );
                candidate = render_focus_layer_with_refinement(
                    image,
                    &source_image,
                    &global_homographies[&image.id],
                    projection,
                    focus_warp,
                    offset_x,
                    offset_y,
                    out_width,
                    out_height,
                    Some(&inverse_refinement),
                    Some(region),
                );
            }
        }
        drop(source_image);
        // A regional warp can still leave a small residual translation on a
        // depth-discontinuous layer. Refine only the detected foreground
        // ownership mask against the already selected foreground when the
        // stack is a normal, mostly-overlapping focus capture. In a shifted
        // mosaic the existing mask can contain a different physical instance
        // of a repeated rail/seal, so mask-only matching is not image-backed
        // registration and can move the whole foreground onto the wrong copy.
        // The shifted-mosaic path has already used the bounded edge-consensus
        // geometry above; leave its residuals to that model and the focus
        // ownership pass.
        // A measured native inverse field also supersedes this mask-only
        // fallback: moving just the bright silhouette afterward introduces a
        // discontinuity between that object and adjacent pixels of one RAW.
        let mut foreground_move = None;
        if !shifted_mosaic && applied_displacement.is_none() {
            candidate = align_focus_foreground_layer_to_existing(
                candidate,
                &merged_foreground_mask,
                &mut foreground_move,
            );
        }
        source_refinements.insert(
            image.id,
            FocusSourceRefinement {
                translation: applied_translation,
                displacement: applied_displacement,
                foreground_move,
                source_sampling: Some(FocusSourceSampling::new(
                    image,
                    &global_homographies[&image.id],
                    projection,
                    focus_warp,
                )),
            },
        );

        // Requirement 3.6: registration chooses the corresponding sample and
        // ownership chooses the frame; neither step may tone-modify the
        // candidate while continuing to identify that Source_RAW as owner.
        // Inter-station low-frequency harmonisation is a later pipeline stage
        // and must not alter this Virtual_Tile's source-level provenance.
        let candidate_analysis = render_focus_analysis_layer(
            &candidate,
            out_width,
            out_height,
            analysis_width,
            analysis_height,
        );
        let candidate_focus = focus_score_map(&candidate_analysis.image, &candidate_analysis.mask);
        let mut analysis_decision = focus_decision_mask(
            &merged_focus,
            &candidate_focus,
            &merged_analysis_mask,
            &candidate_analysis.mask,
        );
        // Keep the low-resolution ownership model in lockstep with the final
        // image. If focus scores are allowed to switch a near-field object at
        // analysis resolution and the full-resolution pass vetoes that switch,
        // the next source can still see the stale analysis winner and add a
        // second copy of the object. The mask is deliberately generic and
        // applies to any detected near-field/occlusion layer.
        if shifted_mosaic {
            suppress_focus_foreground_switches_on_canvas(
                &mut analysis_decision,
                &merged_analysis_foreground_mask,
                &candidate_analysis.foreground_mask,
                &candidate_analysis.relaxed_foreground_mask,
            );
            suppress_focus_foreground_ownership_switches_on_canvas(
                &mut analysis_decision,
                &merged_analysis_foreground_mask,
                &candidate_analysis.mask,
                &candidate_analysis.relaxed_foreground_mask,
            );
        }
        // In a compact focus bracket, the paper/canvas plane is exactly what
        // the stack is supposed to sharpen, so later focused layers must be
        // allowed to replace an already covered canvas pixel.  The guard is
        // only needed for a shifted scan, where a canvas-like overlap can be
        // a different physical camera position and switching it creates broad
        // rectangular exposure blocks.
        if shifted_mosaic {
            suppress_focus_canvas_switches(
                &mut analysis_decision,
                &merged_analysis,
                &candidate_analysis.image,
                &merged_analysis_mask,
                &candidate_analysis.mask,
            );
        }
        // Focus_Fuser (需求 3.2–3.6): on the standard focus path the ownership
        // grid, not the analysis-resolution focus map, decides this candidate.
        // `analysis_decision` above keeps feeding the shifted-mosaic guards and
        // the analysis score table; the pixel write follows the cut alone, at
        // ownership cell resolution and with no transition band.  The cut runs
        // on the *photometrically corrected* candidate, so the inconsistency of
        // 需求 3.3 measures geometry rather than exposure.
        let mut full_resolution_decision = {
            let mut cells = CandidateCells::new(fusion_geometry.cell_count());
            let (sharpness, covered) = measure_focus_cells(
                &fusion_geometry,
                &fusion_plan,
                &candidate.image,
                &candidate.mask,
                candidate.left,
                candidate.top,
            );
            cells.sharpness = sharpness;
            cells.covered = covered;
            (cells.disagreement, cells.seam_disagreement) = measure_focus_cell_disagreement(
                &fusion_geometry,
                &candidate,
                &merged,
                &merged_mask,
            );
            #[cfg(test)]
            {
                if let Some(diagnostics) = score_noise_diagnostics.as_mut() {
                    let base_evidence = fusion.sharpness_evidence();
                    diagnostics.record_candidate(
                        index,
                        &image.filename,
                        Matrix3::new(1.0, 0.0, offset_x, 0.0, 1.0, offset_y, 0.0, 0.0, 1.0)
                            * global_homographies[&image.id],
                        &candidate.image,
                        &candidate.mask,
                        candidate.left,
                        candidate.top,
                        &cells.sharpness,
                        &cells.covered,
                        &cells.disagreement,
                        &score_noise_registration,
                        Some(&base_evidence),
                    );
                }
            }
            let took = fusion.fold(index, &cells);
            focus_cell_decision_for_layer(&fusion_geometry, &took, &candidate)
        };
        // A detected near-field layer is geometrically separate from the main
        // image plane. Do not let focus sharpness make later frames replace an
        // already placed instance of that layer; doing so creates block-shaped
        // brightness and edge jumps in a moving scan. New pixels can still be
        // added where the mosaic has no ownership yet.
        if shifted_mosaic {
            suppress_focus_foreground_switches(
                &mut full_resolution_decision,
                &merged_foreground_mask,
                &candidate,
            );
            suppress_focus_foreground_ownership_switches(
                &mut full_resolution_decision,
                &merged_foreground_mask,
                &candidate.mask,
                candidate.left,
                candidate.top,
                &candidate.relaxed_foreground_mask,
            );
        }
        // A shifted scan can put the ownership boundary through a face, sleeve,
        // or painted contour. Averaging the full-resolution detail there turns
        // two slightly displaced sharp samples into a soft double exposure.
        // The seam path therefore blends only low-frequency canvas tone while
        // keeping detail-band source ownership hard and deterministic.
        let seam_blend_enabled = shifted_mosaic && FOCUS_ALLOW_LOW_FREQUENCY_SEAM_BLEND;
        if seam_blend_enabled {
            // A blended seam band mixes two sources in one pixel, so no single
            // Source_RAW owns it.  Drop the published map instead of claiming
            // an owner that is not the only contributor.
            ownership_ids = None;
            blend_focus_seam_band(
                &mut merged,
                &mut merged_mask,
                &merged_foreground_mask,
                &candidate,
                &full_resolution_decision,
            );
        } else {
            if let Some(ownership_ids) = ownership_ids.as_mut() {
                // Recorded from the coverage state *before* the selection, so
                // it marks exactly the pixels the call below copies.
                record_focus_layer_ownership(
                    ownership_ids,
                    &merged_mask,
                    &candidate,
                    &full_resolution_decision,
                    (index + 1).min(u16::MAX as usize) as u16,
                );
            }
            hard_select_focus_layer(
                &mut merged,
                &mut merged_mask,
                &candidate,
                &full_resolution_decision,
            );
        }
        mark_focus_foreground_pixels(
            &mut merged_foreground_mask,
            &merged_mask,
            &candidate,
            &full_resolution_decision,
        );
        mark_focus_owner_pixels(
            &mut merged_owner,
            &candidate,
            &full_resolution_decision,
            (index + 2).min(u8::MAX as usize) as u8,
        );
        commit_focus_analysis_scores(
            &mut merged_focus,
            &candidate_focus,
            &candidate_analysis.mask,
            &candidate,
            &full_resolution_decision,
            out_width,
            out_height,
            analysis_width,
            analysis_height,
        );
        // Rebuild the analysis image from the authoritative full-resolution
        // result for the canvas/foreground guards. Focus scores intentionally
        // stay in the separate best-score table above: recomputing them from
        // this image would make a selected seam, a sub-pixel warp, or a cubic
        // resample look like a new focus layer and would reintroduce order-
        // dependent blur.
        merged_analysis = resize_rgb(&merged, analysis_width, analysis_height);
        merged_analysis_mask = resize_binary_mask(&merged_mask, analysis_width, analysis_height);
        merged_analysis_foreground_mask =
            resize_binary_mask(&merged_foreground_mask, analysis_width, analysis_height);
    }
    // 需求 12.1 / 12.2: the station's fusion path, with every excluded
    // Source_RAW listed individually.  `station_members` now carries what the
    // loop really fused — the registrar's own verdict for every frame it
    // measured — so the recorded plan and the pixels agree.
    //
    // A station of one Source_RAW has no non-anchor frame, so neither 需求 12.1
    // ("all non-anchor frames failed") nor 需求 12.2 ("at least one failed") has
    // a premise: nothing degraded and nothing is reported.  The fusion above
    // still saw exactly one frame, which is 需求 3.10's single-frame short
    // circuit and not a degradation.
    if station_members.len() >= 2 {
        let measured_plan = plan_station_fusion(&station_members);
        let entries = station_plan_entries(station_index, &station_members, &measured_plan);
        if !entries.is_empty() {
            println!(
                "  - Capture_Station {station_index} fused {} of {} frame(s) ({:?})",
                measured_plan.fused.len(),
                station_members.len(),
                measured_plan.mode
            );
            for excluded in &measured_plan.excluded {
                println!("    - Excluded '{}': {}", excluded.path, excluded.reason);
            }
        }
        record_run_entries(&entries);
    }
    let merged_dimensions = merged.dimensions();
    if fill_margins {
        let (filled_pixels, remaining_invalid_pixels) =
            fill_focus_canvas_margins(&mut merged, &mut merged_mask);
        println!(
            "  - Kept full focus-stack canvas {}x{}; filled {} invalid margin pixels{}",
            merged_dimensions.0,
            merged_dimensions.1,
            filled_pixels,
            if remaining_invalid_pixels > 0 {
                format!(" ({} remain)", remaining_invalid_pixels)
            } else {
                String::new()
            }
        );
    } else {
        println!(
            "  - Kept unfilled focus-tile canvas {}x{}; retaining projected invalid margins",
            merged_dimensions.0, merged_dimensions.1
        );
        // Requirement 3.6: an attributed Virtual_Tile is the selected source
        // sample itself. Sharpening would create a value that no owner supplied.
        println!("  - Source-owned Capture_Station tile; skipping tile sharpening");
    }
    if std::env::var_os("RAW_EDITOR_FOCUS_OWNERSHIP_DIAGNOSTICS").is_some() {
        let mut counts = vec![0usize; images.len() + 2];
        for (&owner, &valid) in merged_owner.as_raw().iter().zip(merged_mask.as_raw()) {
            if valid == 0 {
                continue;
            }
            let index = owner as usize;
            if index >= counts.len() {
                counts.resize(index + 1, 0);
            }
            counts[index] += 1;
        }
        let total: usize = counts.iter().sum();
        let summary = counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(index, count)| {
                format!(
                    "{}:{:.1}%",
                    index,
                    *count as f64 * 100.0 / total.max(1) as f64
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "  - Focus ownership diagnostics: valid={} owners=[{}]",
            total, summary
        );
    }
    #[cfg(test)]
    if let Some(diagnostics) = score_noise_diagnostics {
        diagnostics.finish_and_write(fusion.owners(), &merged_mask)?;
    }
    // Requirement 3.6: source-level ownership and tone modification are
    // mutually exclusive. Tone harmonisation belongs after inter-station seam
    // ownership and cannot run inside an attributed Capture_Station tile.
    println!("  - Source-owned Capture_Station tile; skipping owner tone harmonization");
    // 需求 3.8 / 3.9 / 4.6: publish the fuser's per-cell confidence at pixel
    // resolution and the low sharpness regions in world coordinates.  The tile
    // plane is a pure translation of the world plane, so the tile's top-left
    // pixel sits at `(-offset_x, -offset_y)`.
    let confidence_cells = fusion.confidence();
    let textured_cells = fusion.textured_cells();
    let sharpness_evidence = fusion.sharpness_evidence();
    let low_sharpness_regions = fusion.low_sharpness_region_records((-offset_x, -offset_y));
    if !low_sharpness_regions.is_empty() {
        println!(
            "  - Focus fusion found {} low-sharpness region(s) below the station P10",
            low_sharpness_regions.len()
        );
        degradation::record_run_degradation(
            degradation::FUSION_LOW_SHARPNESS_REGION,
            serde_json::json!({ "regions": low_sharpness_regions.len() }),
        );
    }
    let confidence_values =
        focus_cell_plane_to_pixels(&fusion_geometry, &confidence_cells, out_width, out_height);
    let textured_values =
        focus_cell_texture_to_pixels(&fusion_geometry, &textured_cells, out_width, out_height);
    let mut fusion_report = FusionReport {
        cell_size_px: fusion_geometry.cell_size_px,
        grid: OwnershipGridSize {
            columns: fusion_geometry.columns,
            rows: fusion_geometry.rows,
        },
        solver_status: fusion.solver_status(),
        graph_cut_seconds: fusion.graph_cut_seconds(),
        confidence: focus_fuser::confidence_summary(&confidence_values, merged_mask.as_raw()),
        low_sharpness_regions,
        ..FusionReport::default()
    };
    println!(
        "  - Focus ownership solved in {:.2}s ({:?}); mean confidence {:.3}, below 0.05 {:.2}%",
        fusion_report.graph_cut_seconds,
        fusion_report.solver_status,
        fusion_report.confidence.mean,
        fusion_report.confidence.below_0_05_ratio * 100.0
    );
    let masks = ownership_ids
        .map(|owners| -> Result<FocusStackTileMasks, String> {
            let legend = images
                .iter()
                .map(|image| PathBuf::from(&image.filename))
                .collect::<Vec<_>>();
            let confidence = ConfidenceMap::per_pixel(out_width, out_height, confidence_values)?;
            let ownership = OwnershipMap::new(out_width, out_height, owners, legend)?;
            fusion_report.owned_pixels = ownership.assigned_pixels();
            fusion_report.uncovered_pixels = u64::from(out_width) * u64::from(out_height)
                - merged_mask
                    .as_raw()
                    .par_iter()
                    .filter(|&&covered| covered > 0)
                    .count() as u64;
            Ok(FocusStackTileMasks {
                source_refinements,
                coverage: CoverageMask::from_gray(merged_mask),
                ownership,
                confidence,
                textured: textured_values,
                sharpness_evidence,
                native_displacements,
                fusion: fusion_report,
            })
        })
        .transpose()?;
    Ok(FocusStackTileRender {
        image: merged,
        masks,
        sampling_origin: (-offset_x, -offset_y),
    })
}

/// Restore a small amount of native edge contrast after focus ownership.  The
/// focus score intentionally uses a local blur and the ownership mask is
/// upsampled from an analysis canvas, both of which are stable but slightly
/// conservative around thin brush strokes.  This bounded luma-only unsharp
/// pass runs per virtual tile, never across a projected invalid corner, and
/// therefore cannot blend two camera positions together.
pub(super) fn sharpen_focus_tile_detail(image: &mut Rgb32FImage, mask: &GrayImage, amount: f32) {
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
