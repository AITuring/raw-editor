macro_rules! __panorama_execution_focus {
    ($_progress:ident, $_run_scope:ident, $accepted_station_relations:ident, $added:ident, $alignment_image:ident, $alignment_mode:ident, $anchor_frames:ident, $anchor_layout:ident, $app_handle:ident, $authoritative_pairs:ident, $blend_mode:ident, $brief_pairs:ident, $brightness_ratio_count:ident, $brightness_ratio_sum:ident, $candidate_focus_warp:ident, $candidate_homographies:ident, $candidates:ident, $center:ident, $center_luma:ident, $closure:ident, $clusters:ident, $color_full_u8:ident, $completed:ident, $component_count:ident, $components:ident, $compositor_choice:ident, $compositor_owner:ident, $confidence:ident, $confidence_scores:ident, $consensus_bridges:ident, $coverage:ident, $coverage_plane:ident, $covered_pixel:ident, $decoded_paths:ident, $defer_authoritative_station_solve:ident, $diagnostics_recorder:ident, $diagnostics_world_origin:ident, $digests:ident, $disconnected_component_count:ident, $distance:ident, $dynamic_image:ident, $evidence:ident, $exhaustive_search:ident, $expected:ident, $features:ident, $floor_x:ident, $floor_y:ident, $focal_length_35mm:ident, $focus_capture_group_ids:ident, $focus_capture_groups:ident, $focus_layer_warp:ident, $focus_stack:ident, $focus_station_grouping_outcome:ident, $footprint_has_uncovered:ident, $footprint_uncovered:ident, $foreground_mask:ident, $foreground_range:ident, $full_canvas_height:ident, $full_canvas_width:ident, $full_image:ident, $fusion:ident, $geometries:ident, $geometry:ident, $global_homographies:ident, $gray_full:ident, $group_id:ident, $group_images:ident, $group_index:ident, $group_order:ident, $group_slices:ident, $groups:ident, $groups_complete:ident, $horizontal_edge_rows:ident, $ids:ident, $image:ident, $image_data:ident, $image_data_results:ident, $image_paths:ident, $inferred:ident, $isolated:ident, $key:ident, $left:ident, $left_candidates:ident, $load_render_image:ident, $load_tile:ident, $luma:ident, $match_info:ident, $match_results:ident, $matched_pair_count:ident, $matching_strategy:ident, $members:ident, $message:ident, $mixed_focal_stack:ident, $nearest_distance:ident, $neighbour_count:ident, $neighbour_index:ident, $neighbour_sum:ident, $next_owner:ident, $nx:ident, $ny:ident, $ordered_filenames:ident, $ordered_indices:ident, $ordered_paths:ident, $output_coverage:ident, $overall_percentage:ident, $owner:ident, $owner_disagreement:ident, $owner_reverse_lookup_failures:ident, $owner_shortfall:ident, $ownership:ident, $pair_progress_step:ident, $pairs:ident, $pairs_to_check:ident, $pairwise_matches:ident, $panorama:ident, $path:ident, $path_selection:ident, $physical_memory_bytes:ident, $pid:ident, $pixel:ident, $placement:ident, $pool:ident, $preparation_error:ident, $preparation_workers:ident, $prepare_images:ident, $prepared_count:ident, $prepared_source:ident, $prior_repair_records:ident, $progress_event:ident, $projection:ident, $proposals:ident, $px:ident, $py:ident, $pyramid:ident, $quality_input:ident, $quality_loader:ident, $quality_planes:ident, $quality_sources:ident, $raw_display_highs:ident, $raw_exposure_reference:ident, $raw_local:ident, $raw_owner_ids:ident, $raw_sources:ident, $recorder:ident, $redundant_focus_sources:ident, $rejected:ident, $render_by_id:ident, $render_homographies:ident, $render_image_data:ident, $render_images_info:ident, $render_percentage:ident, $render_scale:ident, $rendered:ident, $repair_clusters:ident, $repaired:ident, $repairs:ident, $report:ident, $rescue_cache:ident, $rescue_pairs:ident, $residual:ident, $retained_full_images:ident, $reverse_residual:ident, $right:ident, $right_candidates:ident, $rounded_has_owner:ident, $rounded_x:ident, $rounded_y:ident, $rss_sampler:ident, $run_generation:ident, $sampled:ident, $sampler:ident, $scalable_stack:ident, $scale_robust_alignment:ident, $scaled_global_homographies:ident, $search_description:ident, $seen:ident, $selected_tile_relations:ident, $selection:ident, $sequence_gap_aware:ident, $sequence_only:ident, $settings:ident, $shared_raw_display_gain:ident, $sharpen:ident, $shortfall:ident, $solved:ident, $source:ident, $source_ids:ident, $source_image:ident, $source_sha256_before:ident, $sources:ident, $splits:ident, $stack_report:ident, $stack_report_path:ident, $start_time:ident, $station:ident, $station_count:ident, $station_crop:ident, $station_focus_warp:ident, $station_masks:ident, $station_origin:ident, $station_poses_replaced:ident, $station_relations:ident, $station_render_homographies:ident, $station_sampling_origins:ident, $station_tile_to_world:ident, $station_to_world:ident, $station_topology:ident, $station_world_centres:ident, $stations:ident, $stitched_images_info:ident, $stitched_indices:ident, $stitching_order:ident, $system:ident, $target:ident, $target_image:ident, $textured:ident, $threshold:ident, $tile:ident, $tile_focus_warp:ident, $tile_group_indices:ident, $tile_groups:ident, $tile_homographies:ident, $tile_id:ident, $tile_index:ident, $tile_infos:ident, $tile_refs:ident, $tile_relation_candidates:ident, $tile_relations:ident, $tile_result:ident, $tiles:ident, $top_features:ident, $topology_ambiguous:ident, $topology_report:ident, $transforms:ident, $undecodable:ident, $undecodable_paths:ident, $unresolved_pixel_count:ident, $unstitched_count:ident, $unstitched_filenames:ident, $vertical_edge_columns:ident, $warning_msg:ident, $world_origin:ident, $x:ident, $y:ident) => {
{
            // Collapse every evidence-backed focus bracket before composing
            // stations.  The virtual-tile path is deliberately selected by
            // the inferred groups, never by filename ranges or a fixture's
            // expected station count.
            //
            // 需求 15.2: the compositor comes from a setting, not from an
            // environment variable, and an unset setting is the default layered
            // path.  需求 15.10: every comparison path is therefore off unless a
            // developer opts in.
            let $compositor_choice =
                $compositor_choice.expect("focus-stack compositor choice resolved above");
            if $compositor_choice.is_comparison_path() {
                println!(
                    "  - Focus compositor comparison path selected: {}",
                    $compositor_choice.as_identifier()
                );
            }
            let $station_count = $focus_capture_groups.as_ref().map(|$groups| $groups.len());
            let $path_selection = select_and_record_run_path(
                $station_count,
                $compositor_choice,
                $stack_report.as_deref(),
            );
            if !$path_selection.use_virtual_tiles {
                stitching::focus_stack_stitcher(
                    &$render_images_info,
                    $render_homographies,
                    $projection,
                    if $render_scale < 1.0 {
                        // Test-only reduced renders validate global layout. Local
                        // focus bands are expressed in full-resolution coordinates
                        // and are exercised by the normal 1:1 acceptance render.
                        None
                    } else {
                        $focus_layer_warp.as_ref()
                    },
                    $focus_capture_group_ids.as_ref(),
                    $sequence_gap_aware,
                    $app_handle.clone(),
                    $progress_event,
                    &mut $load_render_image,
                )
            } else {
                let $groups = $focus_capture_groups.as_ref().expect("checked above");
                // The focus grouping indices refer to the source list, while
                // the compositor must use the render-scaled metadata.  Using
                // `image_data` here silently mixed full-resolution dimensions
                // with scaled homographies and produced oversized, distorted
                // virtual tiles in reduced renders.
                let $render_by_id = $render_images_info
                    .iter()
                    .map(|$image| ($image.id, *$image))
                    .collect::<HashMap<_, _>>();
                let $group_images = $groups
                    .iter()
                    .map(|group| {
                        group
                            .$members
                            .iter()
                            .filter_map(|&index| {
                                let $source = $image_data.get(index)?;
                                let $image = $render_by_id.get(&$source.id).copied()?;
                                $render_homographies.contains_key(&$image.id).then_some($image)
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let $group_slices = $group_images.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let $groups_complete = $group_images
                    .iter()
                    .zip($groups.iter())
                    .all(|($members, group)| $members.len() == group.$members.len());
                let $tile_focus_warp = if $render_scale < 1.0 {
                    None
                } else {
                    $focus_layer_warp.as_ref()
                };
                // Fuse in a station's own anchor grid. Baking uncertain global
                // placement into these pixels made the subsequent station
                // solve try to undo a distorted image through a bounded repair.
                let $anchor_frames = if $compositor_choice
                    == StackCompositorChoice::LayeredVirtualTile
                    && $projection == Projection::Planar
                {
                    Some(
                        focus_anchor_render_frames(
                            $groups,
                            &$image_data,
                            $render_homographies,
                            $tile_focus_warp,
                        )
                        .ok_or("cannot establish camera-station anchor coordinates")?,
                    )
                } else {
                    None
                };
                let $anchor_layout = $anchor_frames
                    .as_ref()
                    .map(|frames| frames.anchor_to_world.clone());
                let mut $station_render_homographies = $anchor_frames.as_ref().map_or_else(
                    || $render_homographies.clone(),
                    |frames| frames.homographies.clone(),
                );
                let mut $station_focus_warp = $anchor_frames.as_ref().map_or_else(
                    || $tile_focus_warp.cloned(),
                    |frames| frames.focus_warp.clone(),
                );
                let $geometries = if $groups_complete {
                    $group_slices
                        .iter()
                        .map(|group| {
                            stitching::focus_stack_virtual_tile_geometry(
                                group,
                                &$station_render_homographies,
                                $projection,
                                $station_focus_warp.as_ref(),
                            )
                        })
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| {
                            "one or more inferred groups have invalid bounds".to_string()
                        })
                } else {
                    Err("one or more inferred groups lack a rendered source".to_string())
                };
                if let Ok(mut $geometries) = $geometries {
                    __panorama_execution_virtual_tile!($_progress, $_run_scope, $accepted_station_relations, $added, $alignment_image, $alignment_mode, $anchor_frames, $anchor_layout, $app_handle, $authoritative_pairs, $blend_mode, $brief_pairs, $brightness_ratio_count, $brightness_ratio_sum, $candidate_focus_warp, $candidate_homographies, $candidates, $center, $center_luma, $closure, $clusters, $color_full_u8, $completed, $component_count, $components, $compositor_choice, $compositor_owner, $confidence, $confidence_scores, $consensus_bridges, $coverage, $coverage_plane, $covered_pixel, $decoded_paths, $defer_authoritative_station_solve, $diagnostics_recorder, $diagnostics_world_origin, $digests, $disconnected_component_count, $distance, $dynamic_image, $evidence, $exhaustive_search, $expected, $features, $floor_x, $floor_y, $focal_length_35mm, $focus_capture_group_ids, $focus_capture_groups, $focus_layer_warp, $focus_stack, $focus_station_grouping_outcome, $footprint_has_uncovered, $footprint_uncovered, $foreground_mask, $foreground_range, $full_canvas_height, $full_canvas_width, $full_image, $fusion, $geometries, $geometry, $global_homographies, $gray_full, $group_id, $group_images, $group_index, $group_order, $group_slices, $groups, $groups_complete, $horizontal_edge_rows, $ids, $image, $image_data, $image_data_results, $image_paths, $inferred, $isolated, $key, $left, $left_candidates, $load_render_image, $load_tile, $luma, $match_info, $match_results, $matched_pair_count, $matching_strategy, $members, $message, $mixed_focal_stack, $nearest_distance, $neighbour_count, $neighbour_index, $neighbour_sum, $next_owner, $nx, $ny, $ordered_filenames, $ordered_indices, $ordered_paths, $output_coverage, $overall_percentage, $owner, $owner_disagreement, $owner_reverse_lookup_failures, $owner_shortfall, $ownership, $pair_progress_step, $pairs, $pairs_to_check, $pairwise_matches, $panorama, $path, $path_selection, $physical_memory_bytes, $pid, $pixel, $placement, $pool, $preparation_error, $preparation_workers, $prepare_images, $prepared_count, $prepared_source, $prior_repair_records, $progress_event, $projection, $proposals, $px, $py, $pyramid, $quality_input, $quality_loader, $quality_planes, $quality_sources, $raw_display_highs, $raw_exposure_reference, $raw_local, $raw_owner_ids, $raw_sources, $recorder, $redundant_focus_sources, $rejected, $render_by_id, $render_homographies, $render_image_data, $render_images_info, $render_percentage, $render_scale, $rendered, $repair_clusters, $repaired, $repairs, $report, $rescue_cache, $rescue_pairs, $residual, $retained_full_images, $reverse_residual, $right, $right_candidates, $rounded_has_owner, $rounded_x, $rounded_y, $rss_sampler, $run_generation, $sampled, $sampler, $scalable_stack, $scale_robust_alignment, $scaled_global_homographies, $search_description, $seen, $selected_tile_relations, $selection, $sequence_gap_aware, $sequence_only, $settings, $shared_raw_display_gain, $sharpen, $shortfall, $solved, $source, $source_ids, $source_image, $source_sha256_before, $sources, $splits, $stack_report, $stack_report_path, $start_time, $station, $station_count, $station_crop, $station_focus_warp, $station_masks, $station_origin, $station_poses_replaced, $station_relations, $station_render_homographies, $station_sampling_origins, $station_tile_to_world, $station_to_world, $station_topology, $station_world_centres, $stations, $stitched_images_info, $stitched_indices, $stitching_order, $system, $target, $target_image, $textured, $threshold, $tile, $tile_focus_warp, $tile_group_indices, $tile_groups, $tile_homographies, $tile_id, $tile_index, $tile_infos, $tile_refs, $tile_relation_candidates, $tile_relations, $tile_result, $tiles, $top_features, $topology_ambiguous, $topology_report, $transforms, $undecodable, $undecodable_paths, $unresolved_pixel_count, $unstitched_count, $unstitched_filenames, $vertical_edge_columns, $warning_msg, $world_origin, $x, $y)
                } else {
                    let $message = format!(
                        "Virtual-tile geometry unavailable ({}); falling back to the existing focus renderer.",
                        $geometries.unwrap_err()
                    );
                    println!("  - {}", $message);
                    let _ = $app_handle.emit($progress_event, &$message);
                    if let Some($recorder) = $stack_report.as_ref() {
                        $recorder
                            .set_selected_path(stack_report::SelectedPath::LegacySingleLayerMosaic);
                    }
                    stitching::focus_stack_stitcher(
                        &$render_images_info,
                        $render_homographies,
                        $projection,
                        $focus_layer_warp.as_ref(),
                        $focus_capture_group_ids.as_ref(),
                        $sequence_gap_aware,
                        $app_handle.clone(),
                        $progress_event,
                        &mut $load_render_image,
                    )
                }
                }
            }
    };
}
