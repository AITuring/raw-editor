macro_rules! __panorama_execution_virtual_render {
    ($_progress:ident, $_run_scope:ident, $accepted_station_relations:ident, $added:ident, $alignment_image:ident, $alignment_mode:ident, $anchor_frames:ident, $anchor_layout:ident, $app_handle:ident, $authoritative_pairs:ident, $blend_mode:ident, $brief_pairs:ident, $brightness_ratio_count:ident, $brightness_ratio_sum:ident, $candidate_focus_warp:ident, $candidate_homographies:ident, $candidates:ident, $center:ident, $center_luma:ident, $closure:ident, $clusters:ident, $color_full_u8:ident, $completed:ident, $component_count:ident, $components:ident, $compositor_choice:ident, $compositor_owner:ident, $confidence:ident, $confidence_scores:ident, $consensus_bridges:ident, $coverage:ident, $coverage_plane:ident, $covered_pixel:ident, $decoded_paths:ident, $defer_authoritative_station_solve:ident, $diagnostics_recorder:ident, $diagnostics_world_origin:ident, $digests:ident, $disconnected_component_count:ident, $distance:ident, $dynamic_image:ident, $evidence:ident, $exhaustive_search:ident, $expected:ident, $features:ident, $floor_x:ident, $floor_y:ident, $focal_length_35mm:ident, $focus_capture_group_ids:ident, $focus_capture_groups:ident, $focus_layer_warp:ident, $focus_stack:ident, $focus_station_grouping_outcome:ident, $footprint_has_uncovered:ident, $footprint_uncovered:ident, $foreground_mask:ident, $foreground_range:ident, $full_canvas_height:ident, $full_canvas_width:ident, $full_image:ident, $fusion:ident, $geometries:ident, $geometry:ident, $global_homographies:ident, $gray_full:ident, $group_id:ident, $group_images:ident, $group_index:ident, $group_order:ident, $group_slices:ident, $groups:ident, $groups_complete:ident, $horizontal_edge_rows:ident, $ids:ident, $image:ident, $image_data:ident, $image_data_results:ident, $image_paths:ident, $inferred:ident, $isolated:ident, $key:ident, $left:ident, $left_candidates:ident, $load_render_image:ident, $load_tile:ident, $luma:ident, $match_info:ident, $match_results:ident, $matched_pair_count:ident, $matching_strategy:ident, $members:ident, $message:ident, $mixed_focal_stack:ident, $nearest_distance:ident, $neighbour_count:ident, $neighbour_index:ident, $neighbour_sum:ident, $next_owner:ident, $nx:ident, $ny:ident, $ordered_filenames:ident, $ordered_indices:ident, $ordered_paths:ident, $output_coverage:ident, $overall_percentage:ident, $owner:ident, $owner_disagreement:ident, $owner_reverse_lookup_failures:ident, $owner_shortfall:ident, $ownership:ident, $pair_progress_step:ident, $pairs:ident, $pairs_to_check:ident, $pairwise_matches:ident, $panorama:ident, $path:ident, $path_selection:ident, $physical_memory_bytes:ident, $pid:ident, $pixel:ident, $placement:ident, $pool:ident, $preparation_error:ident, $preparation_workers:ident, $prepare_images:ident, $prepared_count:ident, $prepared_source:ident, $prior_repair_records:ident, $progress_event:ident, $projection:ident, $proposals:ident, $px:ident, $py:ident, $pyramid:ident, $quality_input:ident, $quality_loader:ident, $quality_planes:ident, $quality_sources:ident, $raw_display_highs:ident, $raw_exposure_reference:ident, $raw_local:ident, $raw_owner_ids:ident, $raw_sources:ident, $recorder:ident, $redundant_focus_sources:ident, $rejected:ident, $render_by_id:ident, $render_homographies:ident, $render_image_data:ident, $render_images_info:ident, $render_percentage:ident, $render_scale:ident, $rendered:ident, $repair_clusters:ident, $repaired:ident, $repairs:ident, $report:ident, $rescue_cache:ident, $rescue_pairs:ident, $residual:ident, $retained_full_images:ident, $reverse_residual:ident, $right:ident, $right_candidates:ident, $rounded_has_owner:ident, $rounded_x:ident, $rounded_y:ident, $rss_sampler:ident, $run_generation:ident, $sampled:ident, $sampler:ident, $scalable_stack:ident, $scale_robust_alignment:ident, $scaled_global_homographies:ident, $search_description:ident, $seen:ident, $selected_tile_relations:ident, $selection:ident, $sequence_gap_aware:ident, $sequence_only:ident, $settings:ident, $shared_raw_display_gain:ident, $sharpen:ident, $shortfall:ident, $solved:ident, $source:ident, $source_ids:ident, $source_image:ident, $source_sha256_before:ident, $sources:ident, $splits:ident, $stack_report:ident, $stack_report_path:ident, $start_time:ident, $station:ident, $station_count:ident, $station_crop:ident, $station_focus_warp:ident, $station_masks:ident, $station_origin:ident, $station_poses_replaced:ident, $station_relations:ident, $station_render_homographies:ident, $station_sampling_origins:ident, $station_tile_to_world:ident, $station_to_world:ident, $station_topology:ident, $station_world_centres:ident, $stations:ident, $stitched_images_info:ident, $stitched_indices:ident, $stitching_order:ident, $system:ident, $target:ident, $target_image:ident, $textured:ident, $threshold:ident, $tile:ident, $tile_focus_warp:ident, $tile_group_indices:ident, $tile_groups:ident, $tile_homographies:ident, $tile_id:ident, $tile_index:ident, $tile_infos:ident, $tile_refs:ident, $tile_relation_candidates:ident, $tile_relations:ident, $tile_result:ident, $tiles:ident, $top_features:ident, $topology_ambiguous:ident, $topology_report:ident, $transforms:ident, $undecodable:ident, $undecodable_paths:ident, $unresolved_pixel_count:ident, $unstitched_count:ident, $unstitched_filenames:ident, $vertical_edge_columns:ident, $warning_msg:ident, $world_origin:ident, $x:ident, $y:ident) => {
                    let $tile_refs = $group_order
                        .iter()
                        .map(|&$group_index| &$tile_infos[$group_index])
                        .collect::<Vec<_>>();
                    let mut $station_masks: Vec<Option<stitching::FocusStackTileMasks>> =
                        (0..$tile_infos.len()).map(|_| None).collect();
                    let mut $station_sampling_origins = vec![(0.0f64, 0.0f64); $tile_infos.len()];
                    let mut $load_tile = |$tile: &ImageInfo| {
                        ensure_focus_memory_available($rss_sampler.as_ref())?;
                        let $group_index =
                            $tile_group_indices.get(&$tile.id).copied().ok_or_else(|| {
                                format!("Missing virtual tile provenance {}", $tile.id)
                            })?;
                        let $rendered = stitching::focus_stack_stitcher_unfilled(
                            $group_slices[$group_index],
                            &$station_render_homographies,
                            $projection,
                            $station_focus_warp.as_ref(),
                            None,
                            $group_index,
                            false,
                            $app_handle.clone(),
                            $progress_event,
                            &mut $load_render_image,
                        )?;
                        let $expected = $geometries[$group_index];
                        let (rendered_width, rendered_height) = $rendered.$image.dimensions();
                        if (rendered_width, rendered_height) != ($expected.width, $expected.height) {
                            return Err(format!(
                                "Virtual tile {} dimensions changed from {}x{} to {}x{}",
                                $group_index,
                                $expected.width,
                                $expected.height,
                                rendered_width,
                                rendered_height
                            ));
                        }
                        // The render now also publishes the Coverage_Mask, the
                        // Ownership_Map and the Sharpness_Confidence it built at
                        // the focus selection's pixel write sites (需求 3.7 /
                        // 3.9 / 3.11 / 4.6).  The Virtual_Tile_Store (task 5.3)
                        // is what consumes the masks; the tile compositors below
                        // still take pixels only, so hand them the same pixel
                        // buffer as before.  The Focus_Fuser observations are
                        // attached to their Capture_Station here, because this is
                        // the first caller that knows which station was rendered
                        // (需求 3.5 / 3.8 / 3.9).
                        if let Some(masks) = $rendered.masks.as_ref() {
                            $station_masks[$group_index] = Some(masks.clone());
                            $station_sampling_origins[$group_index] = $rendered.sampling_origin;
                            let mut $fusion = masks.$fusion.clone();
                            $fusion.station_index = $group_index;
                            focus_fuser::record_run_station($fusion);

                            // Diagnostics are produced from the same masks and
                            // source metadata that drive the compositor.  The
                            // recorder is disabled by default, so these
                            // closures do not allocate or run during ordinary
                            // renders.
                            let $members = $group_slices[$group_index]
                                .iter()
                                .map(|$source| $source.filename.clone())
                                .collect::<Vec<_>>();
                            $diagnostics_recorder.record(
                                $group_index,
                                DiagnosticItem::Members,
                                "json",
                                || serde_json::to_vec(&$members).unwrap_or_default(),
                            );
                            let $transforms = $group_slices[$group_index]
                                .iter()
                                .map(|$source| {
                                    serde_json::json!({
                                        "path": $source.filename,
                                        "width": $source.width,
                                        "height": $source.height,
                                    })
                                })
                                .collect::<Vec<_>>();
                            $diagnostics_recorder.record(
                                $group_index,
                                DiagnosticItem::FrameTransforms,
                                "json",
                                || serde_json::to_vec(&$transforms).unwrap_or_default(),
                            );
                            $diagnostics_recorder.record(
                                $group_index,
                                DiagnosticItem::ResidualField,
                                "json",
                                || {
                                    serde_json::to_vec(&residual_warp::run_report_snapshot())
                                        .unwrap_or_default()
                                },
                            );
                            let $ownership = masks.$ownership.owners().to_vec();
                            $diagnostics_recorder.record(
                                $group_index,
                                DiagnosticItem::OwnershipMap,
                                "json",
                                || serde_json::to_vec(&$ownership).unwrap_or_default(),
                            );
                            let $confidence = masks.$confidence.to_row_major();
                            $diagnostics_recorder.record(
                                $group_index,
                                DiagnosticItem::SharpnessConfidence,
                                "json",
                                || serde_json::to_vec(&$confidence).unwrap_or_default(),
                            );
                            let $coverage = masks.$coverage.covered().to_vec();
                            $diagnostics_recorder.record(
                                $group_index,
                                DiagnosticItem::CoverageMask,
                                "json",
                                || serde_json::to_vec(&$coverage).unwrap_or_default(),
                            );
                            $diagnostics_recorder.record(
                                $group_index,
                                DiagnosticItem::ToneField,
                                "json",
                                || {
                                    serde_json::to_vec(&tone::run_report_snapshot())
                                        .unwrap_or_default()
                                },
                            );
                        }
                        Ok($rendered.into_image())
                    };
                    if let Some($recorder) = $stack_report.as_ref() {
                        let $tiles = $tile_infos
                            .iter()
                            .enumerate()
                            .map(|($group_index, $tile)| stack_report::VirtualTileReport {
                                station_index: $group_index,
                                width: $tile.width,
                                height: $tile.height,
                                provenance: $group_slices[$group_index]
                                    .iter()
                                    .map(|$image| stack_report::ProvenanceRecord {
                                        $path: $image.filename.clone(),
                                        ..Default::default()
                                    })
                                    .collect(),
                                ..Default::default()
                            })
                            .collect::<Vec<_>>();
                        $recorder.update(|$report| $report.virtual_tiles = $tiles);
                    }
                    if let Some($recorder) = $stack_report.as_ref() {
                        // 任务 13.4 / 需求 11.6: 0.0 unless a diagnostic session
                        // overrode it.
                        let $sharpen = f64::from(stitching::layered_final_sharpen_amount());
                        $recorder.update(|$report| $report.composition.final_sharpen_amount = $sharpen);
                    }
                    let $tile_result = match $compositor_choice {
                        // Comparison: the streaming detail-preserving mosaic.
                        StackCompositorChoice::StreamingMosaic => {
                            let $tile_groups = $tile_infos
                                .iter()
                                .enumerate()
                                .map(|(index, $tile)| {
                                    ($tile.id, (index + 1).min(u8::MAX as usize) as u8)
                                })
                                .collect::<HashMap<_, _>>();
                            crate::panorama_utils::mosaic::detail_preserving_mosaic(
                                &$tile_refs,
                                &$tile_homographies,
                                Projection::Planar,
                                Some(&$tile_groups),
                                false,
                                $app_handle.clone(),
                                $progress_event,
                                &mut $load_tile,
                            )
                        }
                        // Comparison: the image-space minimum-cost seam with a
                        // low-frequency multiband transition.  It was the
                        // default until 任务 13.1; it is order dependent and it
                        // crops and sharpens the delivered canvas, which
                        // 需求 10.2 / 10.3 / 11.6 no longer allow.
                        StackCompositorChoice::ProgressiveSeamTile => {
                            stitching::progressive_seam_stitcher(
                                &$tile_refs,
                                &$tile_homographies,
                                Projection::Planar,
                                $app_handle.clone(),
                                $progress_event,
                                &mut $load_tile,
                            )
                        }
                        // Default (需求 15.1): hard source-interior ownership,
                        // the covered union as the canvas and no second-pass
                        // sharpen.
                        //
                        // `LegacySingleLayerMosaic` never reaches this match —
                        // it clears `use_virtual_tiles` above — so it is folded
                        // in here instead of panicking on an unreachable arm.
                        StackCompositorChoice::LayeredVirtualTile
                        | StackCompositorChoice::LegacySingleLayerMosaic => {
                            ensure_focus_memory_available($rss_sampler.as_ref())?;
                            // The ownership planes are retained for the Quality_Gate. A
                            // compositor error still reaches the fallback renderer below.
                            stitching::layered_virtual_tile_compositor_with_ownership(
                                &$tile_refs,
                                &$tile_homographies,
                                Projection::Planar,
                                $app_handle.clone(),
                                $progress_event,
                                &mut $load_tile,
                            )
                            .map(|$rendered| {
                                if let Some($recorder) = $stack_report.as_ref() {
                                    // The layered compositor decodes one full-size
                                    // Virtual_Tile at a time; focus masks are retained
                                    // separately and contain no pixel buffer.
                                    $recorder.record_max_resident_virtual_tiles(1);
                                }
                                let (width, height) = $rendered.$coverage.dimensions();
                                // The compositor names the owning Virtual_Tile.  Resolve that
                                // name back through the station's focus Ownership_Map so the
                                // quality gate samples the actual Source_RAW that supplied the
                                // pixel (output -> station -> virtual-tile coordinate -> raw
                                // ownership cell), rather than a synthetic virtual path.
                                let mut $raw_sources = Vec::new();
                                let mut $raw_owner_ids = vec![Vec::new(); $group_slices.len()];
                                let mut $next_owner = 1u16;
                                for $station in 0..$group_slices.len() {
                                    let $station_to_world = $tile_homographies
                                        .get(&$tile_infos[$station].id)
                                        .copied()
                                        .unwrap_or_else(Matrix3::identity);
                                    let $station_origin = $station_sampling_origins
                                        .get($station)
                                        .copied()
                                        .unwrap_or((0.0, 0.0));
                                    let $station_crop = Matrix3::new(
                                        1.0,
                                        0.0,
                                        -$station_origin.0,
                                        0.0,
                                        1.0,
                                        -$station_origin.1,
                                        0.0,
                                        0.0,
                                        1.0,
                                    );
                                    for $source in $group_slices[$station] {
                                        let $owner = $next_owner;
                                        $raw_owner_ids[$station].push($owner);
                                        $raw_sources.push(quality_gate_runner::QualitySource {
                                            $owner,
                                            $path: $source.filename.clone(),
                                            $geometry: quality_gate::SourceGeometry {
                                                member_to_anchor: Matrix3::identity(),
                                                tile_to_world: $station_to_world
                                                    * $station_crop
                                                    * $station_render_homographies
                                                        .get(&$source.id)
                                                        .copied()
                                                        .unwrap_or_else(Matrix3::identity),
                                                station_id: $station,
                                            },
                                            dimensions: ($source.width, $source.height),
                                            native_registration: $station_masks
                                                .get($station)
                                                .and_then(Option::as_ref)
                                                .and_then(|masks| {
                                                    masks.source_refinements.get(&$source.id)
                                                })
                                                .and_then(|refinement| {
                                                    Some(quality_gate_runner::NativeReferenceMap {
                                                        world_to_station: $station_to_world
                                                            .try_inverse()?,
                                                        $station_to_world,
                                                        $station_origin,
                                                        refinement: refinement.clone(),
                                                    })
                                                }),
                                        });
                                        $next_owner = $next_owner.saturating_add(1);
                                    }
                                }
                                let mut $ownership = $rendered.$ownership.owners().to_vec();
                                let mut $confidence = vec![f32::NAN; $ownership.len()];
                                let mut $textured = vec![0u8; $ownership.len()];
                                // 0..=100 stores owner shortfall in hundredths;
                                // 255 marks a covered pixel whose reverse lookup
                                // could not establish a Source_RAW owner.
                                let mut $owner_shortfall = vec![u8::MAX; $ownership.len()];
                                let mut $owner_disagreement = vec![u8::MAX; $ownership.len()];
                                // Report-only diagnostic plane: a non-transparent
                                // output sample whose exact bicubic footprint
                                // touched an uncovered station pixel.
                                let mut $footprint_uncovered = vec![0u8; $ownership.len()];
                                let mut $unresolved_pixel_count = 0u64;
                                let $reverse_residual = residual_warp::run_model_snapshot();
                                let $covered_pixel = |index: usize| {
                                    $rendered.$coverage.covered().get(index).copied().unwrap_or(0)
                                        != 0
                                };
                                for (index, $owner) in $ownership.iter_mut().enumerate() {
                                    let $compositor_owner = *$owner;
                                    let $tile_index = usize::from($compositor_owner).checked_sub(1);
                                    let Some($tile_index) = $tile_index else {
                                        *$owner = 0;
                                        if $covered_pixel(index) {
                                            $unresolved_pixel_count += 1;
                                            $owner_reverse_lookup_failures.compositor_owner_zero +=
                                                1;
                                        }
                                        continue;
                                    };
                                    let Some(&$station) = $group_order.get($tile_index) else {
                                        *$owner = 0;
                                        if $covered_pixel(index) {
                                            $unresolved_pixel_count += 1;
                                            $owner_reverse_lookup_failures.station_missing += 1;
                                        }
                                        continue;
                                    };
                                    let Some(masks) =
                                        $station_masks.get($station).and_then(Option::as_ref)
                                    else {
                                        *$owner = 0;
                                        if $covered_pixel(index) {
                                            $unresolved_pixel_count += 1;
                                            $owner_reverse_lookup_failures.station_missing += 1;
                                        }
                                        continue;
                                    };
                                    let $station_to_world = $tile_homographies
                                        .get(&$tile_infos[$station].id)
                                        .copied()
                                        .unwrap_or_else(Matrix3::identity);
                                    let Some(world_to_station) = $station_to_world.try_inverse()
                                    else {
                                        *$owner = 0;
                                        if $covered_pixel(index) {
                                            $unresolved_pixel_count += 1;
                                            $owner_reverse_lookup_failures.transform_failed += 1;
                                        }
                                        continue;
                                    };
                                    let $x = (index % width as usize) as f64;
                                    let $y = (index / width as usize) as f64;
                                    let Some(local) = stitching::map_target_to_source_with_residual(
                                        &world_to_station,
                                        Point3::new(
                                            $rendered.sampling_origin.0 + $x,
                                            $rendered.sampling_origin.1 + $y,
                                            1.0,
                                        ),
                                        &$tile_infos[$station],
                                        $projection,
                                        &$reverse_residual,
                                    ) else {
                                        *$owner = 0;
                                        if $covered_pixel(index) {
                                            $unresolved_pixel_count += 1;
                                            $owner_reverse_lookup_failures.transform_failed += 1;
                                        }
                                        continue;
                                    };
                                    let (mask_width, mask_height) = masks.$ownership.dimensions();
                                    let $rounded_x = local.$x.round();
                                    let $rounded_y = local.$y.round();
                                    let mut $sampled = None;
                                    let mut $rounded_has_owner = false;
                                    let mut $footprint_has_uncovered = false;
                                    if $rounded_x.is_finite()
                                        && $rounded_y.is_finite()
                                        && mask_width > 0
                                        && mask_height > 0
                                    {
                                        // The compositor's bicubic sampler uses the
                                        // source coordinate's floor as its anchor;
                                        // rounding is only a diagnostic for whether
                                        // that nearest pixel carried an owner.
                                        let $floor_x = local.$x.floor() as i64;
                                        let $floor_y = local.$y.floor() as i64;
                                        let mut $nearest_distance = f64::INFINITY;
                                        for dy in -1i64..=2 {
                                            for dx in -1i64..=2 {
                                                let Some($px) = $floor_x.checked_add(dx) else {
                                                    continue;
                                                };
                                                let Some($py) = $floor_y.checked_add(dy) else {
                                                    continue;
                                                };
                                                if $px < 0
                                                    || $py < 0
                                                    || $px >= i64::from(mask_width)
                                                    || $py >= i64::from(mask_height)
                                                {
                                                    continue;
                                                }
                                                let $px = $px as u32;
                                                let $py = $py as u32;
                                                if masks
                                                    .$coverage
                                                    .covered()
                                                    .get(
                                                        $py as usize * mask_width as usize
                                                            + $px as usize,
                                                    )
                                                    .copied()
                                                    .unwrap_or(0)
                                                    == 0
                                                {
                                                    $footprint_has_uncovered = true;
                                                }
                                                let $raw_local = masks.$ownership.owner_at($px, $py);
                                                let Some(raw_owner) = $raw_local
                                                    .checked_sub(1)
                                                    .and_then(|raw_index| {
                                                        $raw_owner_ids.get($station).and_then(
                                                            |owners| {
                                                                owners
                                                                    .get(usize::from(raw_index))
                                                                    .copied()
                                                            },
                                                        )
                                                    })
                                                else {
                                                    continue;
                                                };
                                                if f64::from($px) == $rounded_x
                                                    && f64::from($py) == $rounded_y
                                                {
                                                    $rounded_has_owner = true;
                                                }
                                                let $distance = (f64::from($px) - local.$x).mul_add(
                                                    f64::from($px) - local.$x,
                                                    (f64::from($py) - local.$y)
                                                        * (f64::from($py) - local.$y),
                                                );
                                                if $distance < $nearest_distance {
                                                    $nearest_distance = $distance;
                                                    $sampled = Some(($px, $py, raw_owner));
                                                }
                                            }
                                        }
                                    }
                                    if $footprint_has_uncovered
                                        && $rendered
                                            .$image
                                            .get_pixel($x as u32, $y as u32)
                                            .0
                                            .iter()
                                            .any(|value| *value > 1e-6)
                                    {
                                        $footprint_uncovered[index] = 1;
                                    }
                                    let Some((sx, sy, raw_owner)) = $sampled else {
                                        *$owner = 0;
                                        if $covered_pixel(index) {
                                            $unresolved_pixel_count += 1;
                                            if !$rounded_x.is_finite() || !$rounded_y.is_finite() {
                                                $owner_reverse_lookup_failures.transform_failed += 1;
                                            } else if local.$x < 0.0
                                                || local.$y < 0.0
                                                || local.$x >= f64::from(mask_width)
                                                || local.$y >= f64::from(mask_height)
                                            {
                                                $owner_reverse_lookup_failures.out_of_bounds += 1;
                                            } else {
                                                $owner_reverse_lookup_failures
                                                    .raw_owner_unresolved += 1;
                                            }
                                        }
                                        continue;
                                    };
                                    if !$rounded_has_owner {
                                        $owner_reverse_lookup_failures.snapped_to_footprint += 1;
                                    }
                                    *$owner = raw_owner;
                                    $confidence[index] = masks.$confidence.value_at(sx, sy);
                                    $textured[index] = masks
                                        .$textured
                                        .get(sy as usize * mask_width as usize + sx as usize)
                                        .copied()
                                        .unwrap_or(0);
                                    let $evidence = masks.sharpness_evidence.at_pixel(sx, sy);
                                    let $shortfall = $evidence.map_or(
                                        quality_gate_runner::OWNER_EVIDENCE_UNKNOWN,
                                        |$evidence| {
                                            quality_gate_runner::quantize_hundredths(
                                                $evidence.$owner_shortfall(),
                                            )
                                        },
                                    );
                                    // An owned pixel without usable cell evidence
                                    // is an evidence gap, not a flat pixel (需求 11.12).
                                    if $shortfall == quality_gate_runner::OWNER_EVIDENCE_UNKNOWN
                                        && $covered_pixel(index)
                                    {
                                        $unresolved_pixel_count += 1;
                                        $owner_reverse_lookup_failures.evidence_missing += 1;
                                    }
                                    if let Some($evidence) = $evidence {
                                        $owner_shortfall[index] = $shortfall;
                                        $owner_disagreement[index] =
                                            quality_gate_runner::quantize_hundredths(
                                                $evidence.disagreement,
                                            );
                                        quality_gate_runner::accumulate_confidence_score(
                                            &mut $confidence_scores,
                                            $rendered.$coverage.covered()[index],
                                            $confidence[index],
                                            $textured[index],
                                            $evidence,
                                        );
                                    }
                                }
                                let mut $brightness_ratio_sum = 0.0f64;
                                let mut $brightness_ratio_count = 0u64;
                                for (index, &incomplete) in $footprint_uncovered.iter().enumerate() {
                                    if incomplete == 0 {
                                        continue;
                                    }
                                    let $x = index % width as usize;
                                    let $y = index / width as usize;
                                    let $center = $rendered.$image.get_pixel($x as u32, $y as u32).0;
                                    let $center_luma = f64::from($center[0]) * 0.299
                                        + f64::from($center[1]) * 0.587
                                        + f64::from($center[2]) * 0.114;
                                    if !$center_luma.is_finite() || $center_luma <= 1e-6 {
                                        continue;
                                    }
                                    let mut $neighbour_sum = 0.0f64;
                                    let mut $neighbour_count = 0u32;
                                    for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                                        let $nx = $x as i32 + dx;
                                        let $ny = $y as i32 + dy;
                                        if $nx < 0
                                            || $ny < 0
                                            || $nx >= width as i32
                                            || $ny >= height as i32
                                        {
                                            continue;
                                        }
                                        let $neighbour_index =
                                            $ny as usize * width as usize + $nx as usize;
                                        if $footprint_uncovered[$neighbour_index] != 0
                                            || $rendered.$coverage.covered()[$neighbour_index] == 0
                                        {
                                            continue;
                                        }
                                        let $pixel =
                                            $rendered.$image.get_pixel($nx as u32, $ny as u32).0;
                                        let $luma = f64::from($pixel[0]) * 0.299
                                            + f64::from($pixel[1]) * 0.587
                                            + f64::from($pixel[2]) * 0.114;
                                        if $luma.is_finite() && $luma > 1e-6 {
                                            $neighbour_sum += $luma;
                                            $neighbour_count += 1;
                                        }
                                    }
                                    if $neighbour_count > 0 {
                                        $brightness_ratio_sum += $center_luma
                                            / ($neighbour_sum / f64::from($neighbour_count));
                                        $brightness_ratio_count += 1;
                                    }
                                }
                                $owner_reverse_lookup_failures.footprint_uncovered_nontransparent =
                                    $footprint_uncovered
                                        .iter()
                                        .filter(|&&value| value != 0)
                                        .count() as u64;
                                $owner_reverse_lookup_failures.footprint_brightness_ratio =
                                    ($brightness_ratio_count > 0).then_some(
                                        $brightness_ratio_sum / $brightness_ratio_count as f64,
                                    );
                                $quality_sources = Some($raw_sources);
                                let $coverage_plane = GrayImage::from_raw(
                                    width,
                                    height,
                                    $rendered.$coverage.covered().to_vec(),
                                )
                                .unwrap_or_else(|| GrayImage::new(width, height));
                                $output_coverage = Some($coverage_plane.clone());
                                $quality_planes = Some((
                                    $coverage_plane,
                                    $ownership,
                                    $confidence,
                                    $textured,
                                    $owner_shortfall,
                                    $owner_disagreement,
                                    $unresolved_pixel_count,
                                    $rendered.sampling_origin,
                                ));
                                $rendered.$image
                            })
                        }
                    };
    };
}
