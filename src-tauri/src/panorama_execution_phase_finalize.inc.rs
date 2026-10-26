macro_rules! __panorama_execution_finalize {
    ($_progress:ident, $_run_scope:ident, $accepted_station_relations:ident, $added:ident, $alignment_image:ident, $alignment_mode:ident, $anchor_frames:ident, $anchor_layout:ident, $app_handle:ident, $authoritative_pairs:ident, $blend_mode:ident, $brief_pairs:ident, $brightness_ratio_count:ident, $brightness_ratio_sum:ident, $candidate_focus_warp:ident, $candidate_homographies:ident, $candidates:ident, $center:ident, $center_luma:ident, $closure:ident, $clusters:ident, $color_full_u8:ident, $completed:ident, $component_count:ident, $components:ident, $compositor_choice:ident, $compositor_owner:ident, $confidence:ident, $confidence_scores:ident, $consensus_bridges:ident, $coverage:ident, $coverage_plane:ident, $covered_pixel:ident, $decoded_paths:ident, $defer_authoritative_station_solve:ident, $diagnostics_recorder:ident, $diagnostics_world_origin:ident, $digests:ident, $disconnected_component_count:ident, $distance:ident, $dynamic_image:ident, $evidence:ident, $exhaustive_search:ident, $expected:ident, $features:ident, $floor_x:ident, $floor_y:ident, $focal_length_35mm:ident, $focus_capture_group_ids:ident, $focus_capture_groups:ident, $focus_layer_warp:ident, $focus_stack:ident, $focus_station_grouping_outcome:ident, $footprint_has_uncovered:ident, $footprint_uncovered:ident, $foreground_mask:ident, $foreground_range:ident, $full_canvas_height:ident, $full_canvas_width:ident, $full_image:ident, $fusion:ident, $geometries:ident, $geometry:ident, $global_homographies:ident, $gray_full:ident, $group_id:ident, $group_images:ident, $group_index:ident, $group_order:ident, $group_slices:ident, $groups:ident, $groups_complete:ident, $horizontal_edge_rows:ident, $ids:ident, $image:ident, $image_data:ident, $image_data_results:ident, $image_paths:ident, $inferred:ident, $isolated:ident, $key:ident, $left:ident, $left_candidates:ident, $load_render_image:ident, $load_tile:ident, $luma:ident, $match_info:ident, $match_results:ident, $matched_pair_count:ident, $matching_strategy:ident, $members:ident, $message:ident, $mixed_focal_stack:ident, $nearest_distance:ident, $neighbour_count:ident, $neighbour_index:ident, $neighbour_sum:ident, $next_owner:ident, $nx:ident, $ny:ident, $ordered_filenames:ident, $ordered_indices:ident, $ordered_paths:ident, $output_coverage:ident, $overall_percentage:ident, $owner:ident, $owner_disagreement:ident, $owner_reverse_lookup_failures:ident, $owner_shortfall:ident, $ownership:ident, $pair_progress_step:ident, $pairs:ident, $pairs_to_check:ident, $pairwise_matches:ident, $panorama:ident, $path:ident, $path_selection:ident, $physical_memory_bytes:ident, $pid:ident, $pixel:ident, $placement:ident, $pool:ident, $preparation_error:ident, $preparation_workers:ident, $prepare_images:ident, $prepared_count:ident, $prepared_source:ident, $prior_repair_records:ident, $progress_event:ident, $projection:ident, $proposals:ident, $px:ident, $py:ident, $pyramid:ident, $quality_input:ident, $quality_loader:ident, $quality_planes:ident, $quality_sources:ident, $raw_display_highs:ident, $raw_exposure_reference:ident, $raw_local:ident, $raw_owner_ids:ident, $raw_sources:ident, $recorder:ident, $redundant_focus_sources:ident, $rejected:ident, $render_by_id:ident, $render_homographies:ident, $render_image_data:ident, $render_images_info:ident, $render_percentage:ident, $render_scale:ident, $rendered:ident, $repair_clusters:ident, $repaired:ident, $repairs:ident, $report:ident, $rescue_cache:ident, $rescue_pairs:ident, $residual:ident, $retained_full_images:ident, $reverse_residual:ident, $right:ident, $right_candidates:ident, $rounded_has_owner:ident, $rounded_x:ident, $rounded_y:ident, $rss_sampler:ident, $run_generation:ident, $sampled:ident, $sampler:ident, $scalable_stack:ident, $scale_robust_alignment:ident, $scaled_global_homographies:ident, $search_description:ident, $seen:ident, $selected_tile_relations:ident, $selection:ident, $sequence_gap_aware:ident, $sequence_only:ident, $settings:ident, $shared_raw_display_gain:ident, $sharpen:ident, $shortfall:ident, $solved:ident, $source:ident, $source_ids:ident, $source_image:ident, $source_sha256_before:ident, $sources:ident, $splits:ident, $stack_report:ident, $stack_report_path:ident, $start_time:ident, $station:ident, $station_count:ident, $station_crop:ident, $station_focus_warp:ident, $station_masks:ident, $station_origin:ident, $station_poses_replaced:ident, $station_relations:ident, $station_render_homographies:ident, $station_sampling_origins:ident, $station_tile_to_world:ident, $station_to_world:ident, $station_topology:ident, $station_world_centres:ident, $stations:ident, $stitched_images_info:ident, $stitched_indices:ident, $stitching_order:ident, $system:ident, $target:ident, $target_image:ident, $textured:ident, $threshold:ident, $tile:ident, $tile_focus_warp:ident, $tile_group_indices:ident, $tile_groups:ident, $tile_homographies:ident, $tile_id:ident, $tile_index:ident, $tile_infos:ident, $tile_refs:ident, $tile_relation_candidates:ident, $tile_relations:ident, $tile_result:ident, $tiles:ident, $top_features:ident, $topology_ambiguous:ident, $topology_report:ident, $transforms:ident, $undecodable:ident, $undecodable_paths:ident, $unresolved_pixel_count:ident, $unstitched_count:ident, $unstitched_filenames:ident, $vertical_edge_columns:ident, $warning_msg:ident, $world_origin:ident, $x:ident, $y:ident) => {{
    ensure_focus_memory_available($rss_sampler.as_ref())?;
    if let Some((
        $coverage,
        $ownership,
        $confidence,
        $textured,
        $owner_shortfall,
        $owner_disagreement,
        $unresolved_pixel_count,
        $world_origin,
    )) = $quality_planes.take()
    {
        $diagnostics_world_origin = Some($world_origin);
        let $sources = $quality_sources.take().unwrap_or_default();
        let mut $quality_loader =
            |$source: &quality_gate_runner::QualitySource| {
                load_prepared_stack_source(&$source.$path, &$settings, $raw_exposure_reference).map(
                    |prepared| {
                        let mut $rendered = source_to_render_rgb32f(
                            prepared.$image,
                            $source.dimensions.0,
                            $source.dimensions.1,
                        );
                        if is_raw_file(&$source.$path) {
                            if let Some(gain) = std::env::var("RAW_EDITOR_STACK_RAW_FIXED_GAIN")
                                .ok()
                                .and_then(|value| value.parse::<f32>().ok())
                                .filter(|value| value.is_finite() && *value > 0.0)
                            {
                                $rendered.as_mut().par_iter_mut().for_each(|channel| {
                                    *channel = (*channel * gain).clamp(0.0, 1.0)
                                });
                            } else if let Some(gain) = $shared_raw_display_gain {
                                $rendered.as_mut().par_iter_mut().for_each(|channel| {
                                    *channel = (*channel * gain).clamp(0.0, 1.0)
                                });
                            } else if std::env::var_os(
                                "RAW_EDITOR_SKIP_STACK_RAW_EXPOSURE_NORMALIZATION",
                            )
                            .is_none()
                            {
                                normalize_stack_raw_display_exposure(&mut $rendered);
                            }
                        }
                        $rendered
                    },
                )
            };
        let $residual = residual_warp::run_model_snapshot();
        let $accepted_station_relations = tone::run_topology_snapshot()
            .map(|topology| {
                topology
                    .accepted
                    .iter()
                    .filter_map(|&(left_id, right_id)| {
                        let $left = topology.station_of(left_id)?;
                        let $right = topology.station_of(right_id)?;
                        ($left != $right).then_some(($left.min($right), $left.max($right)))
                    })
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let $quality_input = quality_gate_runner::QualityGateInput {
            output: &$panorama,
            $coverage: &$coverage,
            $ownership: &$ownership,
            $confidence: &$confidence,
            $textured: Some(&$textured),
            $owner_shortfall: Some(&$owner_shortfall),
            $owner_disagreement: Some(&$owner_disagreement),
            $unresolved_pixel_count,
            $world_origin,
            $sources: &$sources,
            $accepted_station_relations: &$accepted_station_relations,
            $residual: &$residual,
            acceptance_render_scale: $render_scale,
            final_sharpen_amount: stitching::layered_final_sharpen_amount() as f64,
        };
        let $report = quality_gate_runner::run_quality_gate(&$quality_input, &mut $quality_loader);
        if let Some($recorder) = $stack_report.as_ref() {
            $recorder.update(|stack| {
                stack.quality_gate = $report;
                stack.quality_gate.$confidence_scores = $confidence_scores.clone();
                stack.quality_gate.$owner_reverse_lookup_failures =
                    $owner_reverse_lookup_failures.clone();
            });
        }
    } else if let Some($recorder) = $stack_report.as_ref() {
        $recorder.update(|stack| {
            stack.quality_gate.verdict = stack_report::QualityGateVerdict::InsufficientEvidence;
        });
    }
    if $diagnostics_recorder.is_enabled() {
        let (width, height) = $panorama.dimensions();
        let (origin_x, origin_y) = $diagnostics_world_origin.unwrap_or((0.0, 0.0));
        $diagnostics_recorder.record_crop(
            origin_x.round() as i64,
            origin_y.round() as i64,
            width,
            height,
        );
        if let Err(error) = $diagnostics_recorder.finish() {
            eprintln!("{error}");
        }
    }
    if $rss_sampler.as_mut().is_some_and(|guard| {
        guard
            .finish()
            .is_some_and(|sample| sample.threshold_exceeded)
    }) {
        return Err("memory threshold exceeded during focus-stack run".to_string());
    }
    // The source-stage canvas places every station at its source-stage pose.
    // Once the Virtual_Tile stage replaced those poses the composed canvas is
    // the full-resolution one (langyuan-10: a repaired station moved 3,606px
    // and the composed canvas, 8788x10949, outgrew it; wenyuan-10: the
    // authoritative solve alone, without a repair, composed 8816 columns).
    let ($full_canvas_width, $full_canvas_height) = if $station_poses_replaced {
        let (width, height) = $panorama.dimensions();
        (
            (f64::from(width) / $render_scale).ceil() as u32,
            (f64::from(height) / $render_scale).ceil() as u32,
        )
    } else {
        ($full_canvas_width, $full_canvas_height)
    };

    println!("Stitching completed in {:.2?}\n", $start_time.elapsed());

    let _ = $app_handle.emit($progress_event, "Finalizing image result...");

    // Only a mask of the delivered size can become the result alpha.
    let $coverage = $output_coverage.filter(|mask| mask.dimensions() == $panorama.dimensions());

    if let Some($recorder) = $stack_report.as_ref() {
        let (width, height) = $panorama.dimensions();
        $recorder.update(|$report| {
            $report.composition.canvas.width = u64::from(width);
            $report.composition.canvas.height = u64::from(height);
            $report.degradation.result = stack_report::StackRunResult::Success;
        });
        // Flush eagerly so the report exists before the result travels back to
        // the caller; the drop guard then becomes a no-op.
        let $stack_report_path = $recorder.write_once();

        return Ok(StitchOutcome {
            $image: DynamicImage::ImageRgb32F($panorama),
            $coverage,
            $full_canvas_width,
            $full_canvas_height,
            $render_scale,
            $ordered_paths,
            $stack_report_path,
        });
    }

    Ok(StitchOutcome {
        $image: DynamicImage::ImageRgb32F($panorama),
        $coverage,
        $full_canvas_width,
        $full_canvas_height,
        $render_scale,
        $ordered_paths,
        $stack_report_path: None,
    })
    }};
}
