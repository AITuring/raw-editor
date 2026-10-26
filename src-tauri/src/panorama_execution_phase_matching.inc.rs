macro_rules! __panorama_execution_matching {
    ($_progress:ident, $_run_scope:ident, $accepted_station_relations:ident, $added:ident, $alignment_image:ident, $alignment_mode:ident, $anchor_frames:ident, $anchor_layout:ident, $app_handle:ident, $authoritative_pairs:ident, $blend_mode:ident, $brief_pairs:ident, $brightness_ratio_count:ident, $brightness_ratio_sum:ident, $candidate_focus_warp:ident, $candidate_homographies:ident, $candidates:ident, $center:ident, $center_luma:ident, $closure:ident, $clusters:ident, $color_full_u8:ident, $completed:ident, $component_count:ident, $components:ident, $compositor_choice:ident, $compositor_owner:ident, $confidence:ident, $confidence_scores:ident, $consensus_bridges:ident, $coverage:ident, $coverage_plane:ident, $covered_pixel:ident, $decoded_paths:ident, $defer_authoritative_station_solve:ident, $diagnostics_recorder:ident, $diagnostics_world_origin:ident, $digests:ident, $disconnected_component_count:ident, $distance:ident, $dynamic_image:ident, $evidence:ident, $exhaustive_search:ident, $expected:ident, $features:ident, $floor_x:ident, $floor_y:ident, $focal_length_35mm:ident, $focus_capture_group_ids:ident, $focus_capture_groups:ident, $focus_layer_warp:ident, $focus_stack:ident, $focus_station_grouping_outcome:ident, $footprint_has_uncovered:ident, $footprint_uncovered:ident, $foreground_mask:ident, $foreground_range:ident, $full_canvas_height:ident, $full_canvas_width:ident, $full_image:ident, $fusion:ident, $geometries:ident, $geometry:ident, $global_homographies:ident, $gray_full:ident, $group_id:ident, $group_images:ident, $group_index:ident, $group_order:ident, $group_slices:ident, $groups:ident, $groups_complete:ident, $horizontal_edge_rows:ident, $ids:ident, $image:ident, $image_data:ident, $image_data_results:ident, $image_paths:ident, $inferred:ident, $isolated:ident, $key:ident, $left:ident, $left_candidates:ident, $load_render_image:ident, $load_tile:ident, $luma:ident, $match_info:ident, $match_results:ident, $matched_pair_count:ident, $matching_strategy:ident, $members:ident, $message:ident, $mixed_focal_stack:ident, $nearest_distance:ident, $neighbour_count:ident, $neighbour_index:ident, $neighbour_sum:ident, $next_owner:ident, $nx:ident, $ny:ident, $ordered_filenames:ident, $ordered_indices:ident, $ordered_paths:ident, $output_coverage:ident, $overall_percentage:ident, $owner:ident, $owner_disagreement:ident, $owner_reverse_lookup_failures:ident, $owner_shortfall:ident, $ownership:ident, $pair_progress_step:ident, $pairs:ident, $pairs_to_check:ident, $pairwise_matches:ident, $panorama:ident, $path:ident, $path_selection:ident, $physical_memory_bytes:ident, $pid:ident, $pixel:ident, $placement:ident, $pool:ident, $preparation_error:ident, $preparation_workers:ident, $prepare_images:ident, $prepared_count:ident, $prepared_source:ident, $prior_repair_records:ident, $progress_event:ident, $projection:ident, $proposals:ident, $px:ident, $py:ident, $pyramid:ident, $quality_input:ident, $quality_loader:ident, $quality_planes:ident, $quality_sources:ident, $raw_display_highs:ident, $raw_exposure_reference:ident, $raw_local:ident, $raw_owner_ids:ident, $raw_sources:ident, $recorder:ident, $redundant_focus_sources:ident, $rejected:ident, $render_by_id:ident, $render_homographies:ident, $render_image_data:ident, $render_images_info:ident, $render_percentage:ident, $render_scale:ident, $rendered:ident, $repair_clusters:ident, $repaired:ident, $repairs:ident, $report:ident, $rescue_cache:ident, $rescue_pairs:ident, $residual:ident, $retained_full_images:ident, $reverse_residual:ident, $right:ident, $right_candidates:ident, $rounded_has_owner:ident, $rounded_x:ident, $rounded_y:ident, $rss_sampler:ident, $run_generation:ident, $sampled:ident, $sampler:ident, $scalable_stack:ident, $scale_robust_alignment:ident, $scaled_global_homographies:ident, $search_description:ident, $seen:ident, $selected_tile_relations:ident, $selection:ident, $sequence_gap_aware:ident, $sequence_only:ident, $settings:ident, $shared_raw_display_gain:ident, $sharpen:ident, $shortfall:ident, $solved:ident, $source:ident, $source_ids:ident, $source_image:ident, $source_sha256_before:ident, $sources:ident, $splits:ident, $stack_report:ident, $stack_report_path:ident, $start_time:ident, $station:ident, $station_count:ident, $station_crop:ident, $station_focus_warp:ident, $station_masks:ident, $station_origin:ident, $station_poses_replaced:ident, $station_relations:ident, $station_render_homographies:ident, $station_sampling_origins:ident, $station_tile_to_world:ident, $station_to_world:ident, $station_topology:ident, $station_world_centres:ident, $stations:ident, $stitched_images_info:ident, $stitched_indices:ident, $stitching_order:ident, $system:ident, $target:ident, $target_image:ident, $textured:ident, $threshold:ident, $tile:ident, $tile_focus_warp:ident, $tile_group_indices:ident, $tile_groups:ident, $tile_homographies:ident, $tile_id:ident, $tile_index:ident, $tile_infos:ident, $tile_refs:ident, $tile_relation_candidates:ident, $tile_relations:ident, $tile_result:ident, $tiles:ident, $top_features:ident, $topology_ambiguous:ident, $topology_report:ident, $transforms:ident, $undecodable:ident, $undecodable_paths:ident, $unresolved_pixel_count:ident, $unstitched_count:ident, $unstitched_filenames:ident, $vertical_edge_columns:ident, $warning_msg:ident, $world_origin:ident, $x:ident, $y:ident) => {
    let $start_time = Instant::now();
    let $exhaustive_search = !$scalable_stack
        || ($blend_mode == BlendMode::FocusStack
            && $image_data.len() <= FOCUS_AUTO_ORDER_EXHAUSTIVE_MAX_SOURCES)
        || (SCALE_ROBUST_EXHAUSTIVE_MIN_SOURCES..=SCALE_ROBUST_EXHAUSTIVE_MAX_SOURCES)
            .contains(&$image_data.len());
    let $matching_strategy = if $exhaustive_search {
        "all pairwise"
    } else {
        "ordered-neighbor + visual retrieval"
    };
    let _ = $app_handle.emit(
        $progress_event,
        format!("Finding image matches ({})...", $matching_strategy),
    );
    println!("Finding {} matches (in parallel)...", $matching_strategy);
    let $projection = $alignment_mode.projection_for($blend_mode);
    let mut $pairwise_matches: HashMap<(usize, usize), MatchInfo> = HashMap::new();

    #[cfg(test)]
    let $sequence_only = std::env::var_os("RAW_EDITOR_STACK_SEQUENCE_ONLY").is_some();
    #[cfg(not(test))]
    let $sequence_only = false;

    let $pairs_to_check = if $sequence_only {
        Vec::new()
    } else {
        pairs_to_match_for_images(&$image_data, $blend_mode)
    };
    let $matched_pair_count = Mutex::new(0);
    let $pair_progress_step = ($pairs_to_check.len() / 100).max(1);

    let $match_results: Vec<Option<((usize, usize), MatchInfo)>> = $pairs_to_check
        .par_iter()
        .map(|&(i, j)| {
            if degradation::run_generation_changed_for($run_generation) {
                return None;
            }
            let $_progress = PairMatchProgress {
                $completed: &$matched_pair_count,
                total: $pairs_to_check.len(),
                progress_step: $pair_progress_step,
                $app_handle: &$app_handle,
                $progress_event,
            };
            // Focus-stack matching is not perfectly symmetric: the ratio test and
            // local NCC search are evaluated from source to target. Use a stable
            // path-based direction, then convert back to the (i, j) key orientation.
            // Otherwise changing import order can produce a different transform graph
            // and expose defocus halos at object silhouettes. Keep panorama matching's
            // established direction unchanged.
            let (source_index, target_index, invert_for_storage) =
                if $blend_mode == BlendMode::FocusStack {
                    canonical_match_direction(&$image_data, i, j)
                } else {
                    (i, j, false)
                };
            let $source_image = &$image_data[source_index];
            let $target_image = &$image_data[target_index];
            let mut $match_info = match_image_pair(
                $source_image,
                $target_image,
                $projection,
                $blend_mode,
                $alignment_mode,
                $scale_robust_alignment,
                true,
                false,
            )?;
            if $blend_mode == BlendMode::FocusStack {
                // Record the fit before it is rotated into the key orientation;
                // see `MatchInfo::canonical_homography` (需求 14.6).
                $match_info.canonical_homography = Some($match_info.homography);
            }
            if invert_for_storage {
                $match_info.homography = $match_info.homography.try_inverse()?;
                $match_info.points = $match_info
                    .points
                    .into_iter()
                    .map(|($source, $target)| ($target, $source))
                    .collect();
                $match_info.candidate_points = $match_info
                    .candidate_points
                    .into_iter()
                    .map(|($source, $target)| ($target, $source))
                    .collect();
                $match_info.top_candidate_points = $match_info
                    .top_candidate_points
                    .into_iter()
                    .map(|($source, $target)| ($target, $source))
                    .collect();
                $match_info.dense_focus_points = $match_info
                    .dense_focus_points
                    .into_iter()
                    .map(|($source, $target)| ($target, $source))
                    .collect();
                $match_info.foreground_feature_points = $match_info
                    .foreground_feature_points
                    .into_iter()
                    .map(|($source, $target)| ($target, $source))
                    .collect();
            }
            Some(((i, j), $match_info))
        })
        .collect();
    ensure_focus_memory_available($rss_sampler.as_ref())?;

    for result in $match_results.into_iter().flatten() {
        $pairwise_matches.insert(result.0, result.1);
    }
    if $sequence_only {
        add_capture_sequence_bridges(&$image_data, &mut $pairwise_matches);
    }
    // Filename adjacency never creates or clears a focus-stack edge. The old
    // filename-window recovery pass (clear + retry on natural-sort neighbours)
    // was removed in favour of the geometry-driven retry below: a capture
    // sequence may reset to another raster row, and repeated seals/figures can
    // make such a false edge look convincing. Scan-position edges are therefore
    // never synthesized from filename order or a coarse NCC peak; if the full
    // feature/RANSAC matcher finds no evidence the stack stays disconnected and
    // is rejected instead of rendered as a duplicated collage.
    if $scalable_stack && $focus_stack && !$sequence_only {
        // A disconnected focus stack gets a bounded, evidence-only retry. The
        // old retry examined only filename-window boundaries, which is exactly
        // where a raster reset can masquerade as a neighbouring tile. Search
        // the strongest frames in every component pair instead, and accept a
        // bridge only when the normal feature/RANSAC matcher returns a real
        // geometric match (never a synthetic/coarse bridge).
        let mut $rescue_cache = HashMap::<usize, ImageInfo>::new();
        for pass in 0..3 {
            let $components = focus_overlap_components(&$image_data, &$pairwise_matches);
            if $components.len() <= 1 {
                break;
            }
            let mut $rescue_pairs = Vec::<(usize, usize, usize)>::new();
            for left_component_index in 0..$components.len() {
                for right_component_index in left_component_index + 1..$components.len() {
                    let mut $left_candidates = $components[left_component_index].clone();
                    let mut $right_candidates = $components[right_component_index].clone();
                    $left_candidates.sort_by_key(|&index| {
                        std::cmp::Reverse(focus_alignment_support(&$image_data[index]))
                    });
                    $right_candidates.sort_by_key(|&index| {
                        std::cmp::Reverse(focus_alignment_support(&$image_data[index]))
                    });
                    $left_candidates.truncate(6);
                    $right_candidates.truncate(6);
                    for &index in $components[left_component_index]
                        .iter()
                        .chain($components[left_component_index].iter().rev())
                        .take(4)
                    {
                        if !$left_candidates.contains(&index) {
                            $left_candidates.push(index);
                        }
                    }
                    for &index in $components[right_component_index]
                        .iter()
                        .chain($components[right_component_index].iter().rev())
                        .take(4)
                    {
                        if !$right_candidates.contains(&index) {
                            $right_candidates.push(index);
                        }
                    }
                    for &$left in &$left_candidates {
                        for &$right in &$right_candidates {
                            let $key = ($left.min($right), $left.max($right));
                            if !$pairwise_matches.contains_key(&$key) {
                                $rescue_pairs.push((
                                    focus_alignment_support(&$image_data[$left])
                                        + focus_alignment_support(&$image_data[$right]),
                                    $left,
                                    $right,
                                ));
                            }
                        }
                    }
                }
            }
            $rescue_pairs
                .sort_by_key(|(support, $left, $right)| (std::cmp::Reverse(*support), *$left, *$right));
            $rescue_pairs.truncate(128);
            if $rescue_pairs.is_empty() {
                break;
            }
            println!(
                "Retrying {} disconnected focus component pair(s) with strict feature geometry (pass {})...",
                $rescue_pairs.len(),
                pass + 1,
            );
            let mut $added = 0usize;
            for (_, first, second) in $rescue_pairs {
                let (source_index, target_index, invert_for_storage) =
                    canonical_match_direction(&$image_data, first, second);
                if !$rescue_cache.contains_key(&source_index) {
                    $rescue_cache.insert(
                        source_index,
                        prepare_focus_rescue_image(
                            &$image_data[source_index],
                            &$settings,
                            $raw_exposure_reference,
                            &$brief_pairs,
                        )?,
                    );
                }
                if !$rescue_cache.contains_key(&target_index) {
                    $rescue_cache.insert(
                        target_index,
                        prepare_focus_rescue_image(
                            &$image_data[target_index],
                            &$settings,
                            $raw_exposure_reference,
                            &$brief_pairs,
                        )?,
                    );
                }
                let $source = $rescue_cache
                    .get(&source_index)
                    .expect("rescue source cached");
                let $target = $rescue_cache
                    .get(&target_index)
                    .expect("rescue target cached");
                let Some(mut $match_info) = match_image_pair(
                    $source,
                    $target,
                    $projection,
                    $blend_mode,
                    $alignment_mode,
                    true,
                    true,
                    true,
                ) else {
                    continue;
                };
                if !focus_rescue_match_is_verified($source, $target, &$match_info) {
                    println!(
                        "  - Rejected rescue candidate '{}' <-> '{}' after geometry validation",
                        Path::new(&$image_data[first].filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy(),
                        Path::new(&$image_data[second].filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy(),
                    );
                    continue;
                }
                // Record the fit before it is rotated into the key orientation;
                // see `MatchInfo::canonical_homography` (需求 14.6).
                $match_info.canonical_homography = Some($match_info.homography);
                if invert_for_storage {
                    $match_info.homography = $match_info
                        .homography
                        .try_inverse()
                        .ok_or_else(|| "Failed to invert a focus rescue transform".to_string())?;
                    $match_info.points = $match_info
                        .points
                        .into_iter()
                        .map(|($source, $target)| ($target, $source))
                        .collect();
                    $match_info.candidate_points = $match_info
                        .candidate_points
                        .into_iter()
                        .map(|($source, $target)| ($target, $source))
                        .collect();
                    $match_info.top_candidate_points = $match_info
                        .top_candidate_points
                        .into_iter()
                        .map(|($source, $target)| ($target, $source))
                        .collect();
                    $match_info.dense_focus_points = $match_info
                        .dense_focus_points
                        .into_iter()
                        .map(|($source, $target)| ($target, $source))
                        .collect();
                    $match_info.foreground_feature_points = $match_info
                        .foreground_feature_points
                        .into_iter()
                        .map(|($source, $target)| ($target, $source))
                        .collect();
                }
                let $key = (first.min(second), first.max(second));
                println!(
                    "  - Accepted strict focus rescue: '{}' <-> '{}' ({} inliers)",
                    Path::new(&$image_data[first].filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    Path::new(&$image_data[second].filename)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    $match_info.points.len()
                );
                $pairwise_matches.insert($key, $match_info);
                $added += 1;
            }
            if $added == 0 {
                break;
            }
        }
        let $consensus_bridges = recover_focus_component_edges(&$image_data, &mut $pairwise_matches);
        if $consensus_bridges > 0 {
            println!(
                "  - Recovered {} component bridge(s) from multi-layer coarse consensus",
                $consensus_bridges,
            );
        }
    }
    // Never invent a focus-stack edge from capture order. A large artwork is
    // commonly photographed in several raster passes and focus brackets, so a
    // filename-adjacent pair can be a scan reset rather than a neighbouring
    // tile. Every global pose must be connected by verified image evidence.
    println!(
        "Pairwise matching completed in {:.2?}\n",
        $start_time.elapsed()
    );

    if $pairwise_matches.is_empty() {
        return Err(if $scalable_stack && $focus_stack {
            if $exhaustive_search {
                "No overlap was found among the selected focus-stack images. Make sure the selection is one contiguous scene with sufficient overlap."
                    .to_string()
            } else {
                "No overlap was found among the automatic focus-stack candidate pairs. Large stacks use a bounded search; use frames with sufficient overlap or reduce the number of layers."
                    .to_string()
            }
        } else if $scalable_stack {
            if $exhaustive_search {
                "No overlap was found among the selected panorama images. Make sure the selection is one contiguous scene with sufficient overlap."
                    .to_string()
            } else {
                "No overlap was found between nearby images. For large panoramas, make sure consecutive images overlap."
                    .to_string()
            }
        } else {
            "No suitable matches found between any pair of images. Cannot create a panorama."
                .to_string()
        });
    }

    };
}
