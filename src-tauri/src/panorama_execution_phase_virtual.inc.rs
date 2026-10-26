macro_rules! __panorama_execution_virtual_tile {
    ($_progress:ident, $_run_scope:ident, $accepted_station_relations:ident, $added:ident, $alignment_image:ident, $alignment_mode:ident, $anchor_frames:ident, $anchor_layout:ident, $app_handle:ident, $authoritative_pairs:ident, $blend_mode:ident, $brief_pairs:ident, $brightness_ratio_count:ident, $brightness_ratio_sum:ident, $candidate_focus_warp:ident, $candidate_homographies:ident, $candidates:ident, $center:ident, $center_luma:ident, $closure:ident, $clusters:ident, $color_full_u8:ident, $completed:ident, $component_count:ident, $components:ident, $compositor_choice:ident, $compositor_owner:ident, $confidence:ident, $confidence_scores:ident, $consensus_bridges:ident, $coverage:ident, $coverage_plane:ident, $covered_pixel:ident, $decoded_paths:ident, $defer_authoritative_station_solve:ident, $diagnostics_recorder:ident, $diagnostics_world_origin:ident, $digests:ident, $disconnected_component_count:ident, $distance:ident, $dynamic_image:ident, $evidence:ident, $exhaustive_search:ident, $expected:ident, $features:ident, $floor_x:ident, $floor_y:ident, $focal_length_35mm:ident, $focus_capture_group_ids:ident, $focus_capture_groups:ident, $focus_layer_warp:ident, $focus_stack:ident, $focus_station_grouping_outcome:ident, $footprint_has_uncovered:ident, $footprint_uncovered:ident, $foreground_mask:ident, $foreground_range:ident, $full_canvas_height:ident, $full_canvas_width:ident, $full_image:ident, $fusion:ident, $geometries:ident, $geometry:ident, $global_homographies:ident, $gray_full:ident, $group_id:ident, $group_images:ident, $group_index:ident, $group_order:ident, $group_slices:ident, $groups:ident, $groups_complete:ident, $horizontal_edge_rows:ident, $ids:ident, $image:ident, $image_data:ident, $image_data_results:ident, $image_paths:ident, $inferred:ident, $isolated:ident, $key:ident, $left:ident, $left_candidates:ident, $load_render_image:ident, $load_tile:ident, $luma:ident, $match_info:ident, $match_results:ident, $matched_pair_count:ident, $matching_strategy:ident, $members:ident, $message:ident, $mixed_focal_stack:ident, $nearest_distance:ident, $neighbour_count:ident, $neighbour_index:ident, $neighbour_sum:ident, $next_owner:ident, $nx:ident, $ny:ident, $ordered_filenames:ident, $ordered_indices:ident, $ordered_paths:ident, $output_coverage:ident, $overall_percentage:ident, $owner:ident, $owner_disagreement:ident, $owner_reverse_lookup_failures:ident, $owner_shortfall:ident, $ownership:ident, $pair_progress_step:ident, $pairs:ident, $pairs_to_check:ident, $pairwise_matches:ident, $panorama:ident, $path:ident, $path_selection:ident, $physical_memory_bytes:ident, $pid:ident, $pixel:ident, $placement:ident, $pool:ident, $preparation_error:ident, $preparation_workers:ident, $prepare_images:ident, $prepared_count:ident, $prepared_source:ident, $prior_repair_records:ident, $progress_event:ident, $projection:ident, $proposals:ident, $px:ident, $py:ident, $pyramid:ident, $quality_input:ident, $quality_loader:ident, $quality_planes:ident, $quality_sources:ident, $raw_display_highs:ident, $raw_exposure_reference:ident, $raw_local:ident, $raw_owner_ids:ident, $raw_sources:ident, $recorder:ident, $redundant_focus_sources:ident, $rejected:ident, $render_by_id:ident, $render_homographies:ident, $render_image_data:ident, $render_images_info:ident, $render_percentage:ident, $render_scale:ident, $rendered:ident, $repair_clusters:ident, $repaired:ident, $repairs:ident, $report:ident, $rescue_cache:ident, $rescue_pairs:ident, $residual:ident, $retained_full_images:ident, $reverse_residual:ident, $right:ident, $right_candidates:ident, $rounded_has_owner:ident, $rounded_x:ident, $rounded_y:ident, $rss_sampler:ident, $run_generation:ident, $sampled:ident, $sampler:ident, $scalable_stack:ident, $scale_robust_alignment:ident, $scaled_global_homographies:ident, $search_description:ident, $seen:ident, $selected_tile_relations:ident, $selection:ident, $sequence_gap_aware:ident, $sequence_only:ident, $settings:ident, $shared_raw_display_gain:ident, $sharpen:ident, $shortfall:ident, $solved:ident, $source:ident, $source_ids:ident, $source_image:ident, $source_sha256_before:ident, $sources:ident, $splits:ident, $stack_report:ident, $stack_report_path:ident, $start_time:ident, $station:ident, $station_count:ident, $station_crop:ident, $station_focus_warp:ident, $station_masks:ident, $station_origin:ident, $station_poses_replaced:ident, $station_relations:ident, $station_render_homographies:ident, $station_sampling_origins:ident, $station_tile_to_world:ident, $station_to_world:ident, $station_topology:ident, $station_world_centres:ident, $stations:ident, $stitched_images_info:ident, $stitched_indices:ident, $stitching_order:ident, $system:ident, $target:ident, $target_image:ident, $textured:ident, $threshold:ident, $tile:ident, $tile_focus_warp:ident, $tile_group_indices:ident, $tile_groups:ident, $tile_homographies:ident, $tile_id:ident, $tile_index:ident, $tile_infos:ident, $tile_refs:ident, $tile_relation_candidates:ident, $tile_relations:ident, $tile_result:ident, $tiles:ident, $top_features:ident, $topology_ambiguous:ident, $topology_report:ident, $transforms:ident, $undecodable:ident, $undecodable_paths:ident, $unresolved_pixel_count:ident, $unstitched_count:ident, $unstitched_filenames:ident, $vertical_edge_columns:ident, $warning_msg:ident, $world_origin:ident, $x:ident, $y:ident) => {{
                    // Feed the default layered compositor in the canonical
                    // row/column order inferred by Capture_Topology_Model.
                    // Continuous projected centres, source names and the
                    // candidate source path must not break topology ties or
                    // change seam accumulation (任务 9.10, 需求 14.6).
                    let $station_topology = $station_topology
                        .as_ref()
                        .expect("multi-station layered composition requires inferred topology");
                    let $group_order =
                        focus_source_order_group_compositor_order($groups.len(), $station_topology);
                    println!(
                        "  - Focus virtual-tile compositor order: {}",
                        $group_order
                            .iter()
                            .map(|$group_index| ($group_index + 1).to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    println!(
                        "  - Composing {} focus virtual tile(s) from inferred capture groups",
                        $geometries.len()
                    );

                    let mut $tile_infos = Vec::with_capacity($group_order.len());
                    let mut $tile_homographies = HashMap::with_capacity($group_order.len());
                    let mut $tile_group_indices = HashMap::with_capacity($group_order.len());
                    for $group_index in 0..$groups.len() {
                        let $geometry = $geometries[$group_index];
                        let $tile_id = usize::MAX.saturating_sub($group_index);
                        let $source_ids = $group_slices[$group_index]
                            .iter()
                            .map(|$image| $image.id.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        let $placement = $anchor_layout
                            .as_ref()
                            .map_or($geometry.tile_to_world, |anchors| {
                                anchors[$group_index] * $geometry.tile_to_world
                            });
                        $tile_homographies.insert($tile_id, $placement);
                        $tile_group_indices.insert($tile_id, $group_index);
                        $tile_infos.push(ImageInfo {
                            id: $tile_id,
                            filename: format!(
                                "virtual://focus-group-{}-{}",
                                $group_index + 1,
                                $source_ids
                            ),
                            width: $geometry.width,
                            height: $geometry.height,
                            // Replaced by the finest probe plane once rendered.
                            $alignment_image: GrayImage::new(1, 1),
                            $full_image: None,
                            scale_factor: 1.0,
                            $focal_length_35mm: None,
                            overview_reference: false,
                            $features: Vec::new(),
                            $top_features: Vec::new(),
                            $foreground_range: None,
                            $foreground_mask: None,
                            $horizontal_edge_rows: Vec::new(),
                            $vertical_edge_columns: Vec::new(),
                        });
                    }

                    // The production station-relation pass works on the actual
                    // focus-fused Virtual_Tiles, not on rectangular source
                    // buffers. Render one station at a time, keep only its
                    // area-averaged Coverage_Mask planes, and discard the
                    // fused pixels before moving to the next station. This
                    // preserves source ownership and does not increase the
                    // compositor's resident-tile set.
                    //
                    // Rendering priors may be repaired below, so every later
                    // render (including the compositor's) reads these owned
                    // copies rather than the source-stage poses.
                    let mut $pyramid = VirtualTileMatchPyramid::new(
                        virtual_tile_pyramid_factors(
                            $geometries
                                .iter()
                                .map(|$geometry| $geometry.width.max($geometry.height))
                                .min()
                                .unwrap_or(0),
                        ),
                        $tile_infos.len(),
                    );
                    for ($group_index, $tile) in $tile_infos.iter_mut().enumerate() {
                        let (gray, $coverage) = render_virtual_tile_match_plane(
                            $group_slices[$group_index],
                            &$station_render_homographies,
                            $projection,
                            $station_focus_warp.as_ref(),
                            $group_index,
                            $geometries[$group_index],
                            $app_handle.clone(),
                            $progress_event,
                            &mut $load_render_image,
                        )?;
                        install_virtual_tile_match_planes(
                            $tile,
                            &mut $pyramid,
                            $group_index,
                            &gray,
                            &$coverage,
                            &$brief_pairs,
                        )?;
                    }
                    let (mut $tile_relations, mut $tile_relation_candidates) =
                        virtual_tile_measure_prior_relations(
                            &$tile_infos,
                            &$pyramid,
                            &$tile_homographies,
                            $station_topology,
                            None,
                            false,
                        );

                    // A station whose rendering prior is grossly wrong cannot be
                    // reached by any bounded probe. Only candidates that join
                    // two different prior-consistent clusters get a prior-free
                    // covered-feature proposal. The bounded pixel probe fits
                    // an independent relation before any solver admission.
                    // Anchor-grid tiles can use that measured relation directly;
                    // globally warped comparison tiles instead repair their
                    // prior and render again. Feature proposals never enter the
                    // solver themselves.
                    let mut $prior_repair_records = Vec::new();
                    if $tile_infos.len() >= 2 {
                        let $authoritative_pairs = virtual_tile_authoritative_pairs(
                            &$tile_relations,
                            &$tile_infos,
                            $pyramid.solver_coverages(),
                        );
                        let mut $clusters = Dsu::new($tile_infos.len());
                        for &($left, $right) in &$authoritative_pairs {
                            $clusters.union($left, $right);
                        }
                        let mut $proposals = Vec::new();
                        for candidate in &$station_topology.$candidates {
                            let ($left, $right) = (candidate.$left, candidate.$right);
                            if $left >= $tile_infos.len()
                                || $right >= $tile_infos.len()
                                || $clusters.find($left) == $clusters.find($right)
                            {
                                continue;
                            }
                            let Some((seed, feature_inliers)) = virtual_tile_prior_free_relation(
                                $pyramid.feature_factors[$left],
                                &$tile_infos[$left].$features,
                                $pyramid.feature_factors[$right],
                                &$tile_infos[$right].$features,
                                $tile_infos[$left].dimensions(),
                            ) else {
                                continue;
                            };
                            let (verified, mut diagnostic) = virtual_tile_pyramid_station_match(
                                $left,
                                $right,
                                &$pyramid,
                                &seed,
                                VIRTUAL_TILE_SEED_PRIOR_FREE,
                            );
                            diagnostic.prior_free_inliers = feature_inliers;
                            let $selection = select_virtual_tile_relation_hypothesis(
                                &$tile_infos,
                                $left,
                                $right,
                                &verified,
                                $pyramid.solver_coverages(),
                            );
                            if let Some($selection) = $selection.as_ref() {
                                diagnostic.hypotheses = $selection.outcomes.clone();
                            }
                            populate_station_relation_candidate_geometry(
                                &mut diagnostic,
                                &$tile_infos,
                                $pyramid.solver_coverages(),
                                &$tile_homographies,
                            );
                            log_virtual_tile_candidate(&diagnostic, false);
                            if let Some($selection) = $selection.filter(|$selection| $selection.passes)
                            {
                                if $anchor_layout.is_some() {
                                    $clusters.union($left, $right);
                                    $tile_relations
                                        .entry(($left, $right))
                                        .or_default()
                                        .extend(verified);
                                } else {
                                    $proposals.push(VirtualTilePriorProposal {
                                        $left,
                                        $right,
                                        left_to_right: $selection.relation.homography,
                                        inliers: $selection.relation.inliers,
                                    });
                                }
                            } else if $anchor_layout.is_none()
                                && virtual_tile_prior_free_seed_fallback_allowed(&diagnostic)
                            {
                                // Keep this relation out of the authoritative graph. It only
                                // moves the unfixed cluster close enough for a fresh Virtual_Tile
                                // render, after which the regular measurement pass verifies it.
                                $proposals.push(VirtualTilePriorProposal {
                                    $left,
                                    $right,
                                    left_to_right: seed,
                                    inliers: feature_inliers,
                                });
                            }
                            replace_virtual_tile_candidate_record(
                                &mut $tile_relation_candidates,
                                diagnostic,
                            );
                        }
                        let $authoritative_pairs = virtual_tile_authoritative_pairs(
                            &$tile_relations,
                            &$tile_infos,
                            $pyramid.solver_coverages(),
                        );
                        let $station_tile_to_world = $tile_infos
                            .iter()
                            .map(|$tile| $tile_homographies[&$tile.id])
                            .collect::<Vec<_>>();
                        let $station_world_centres = $geometries
                            .iter()
                            .zip(&$station_tile_to_world)
                            .filter_map(|($geometry, tile_to_world)| {
                                transformed_point(
                                    tile_to_world,
                                    Point2::new(
                                        f64::from($geometry.width) * 0.5,
                                        f64::from($geometry.height) * 0.5,
                                    ),
                                )
                            })
                            .collect::<Vec<_>>();
                        let $repairs = plan_virtual_tile_prior_repairs_with_centres(
                            $tile_infos.len(),
                            &$authoritative_pairs,
                            &$proposals,
                            &$station_tile_to_world,
                            &$station_world_centres,
                            $station_topology,
                        );
                        let mut $repaired = BTreeSet::new();
                        let mut $repair_clusters =
                            BTreeMap::<usize, Vec<&VirtualTilePriorRepair>>::new();
                        for repair in &$repairs {
                            $repair_clusters
                                .entry(repair.cluster_root)
                                .or_default()
                                .push(repair);
                        }
                        for cluster_repairs in $repair_clusters.values() {
                            // Preflight every member on one copy. No rendering
                            // prior is committed unless the entire cluster has
                            // valid poses, geometry, centre shift and scale.
                            let mut $candidate_homographies = $station_render_homographies.clone();
                            let mut $candidate_focus_warp = $station_focus_warp.clone();
                            let mut $candidates = Vec::with_capacity(cluster_repairs.len());
                            let mut $rejected = None;
                            for &repair in cluster_repairs {
                                let $station = repair.$station;
                                if !apply_virtual_tile_prior_repair(
                                    $group_slices[$station],
                                    &repair.world_correction,
                                    &mut $candidate_homographies,
                                    $candidate_focus_warp.as_mut(),
                                ) {
                                    $rejected = Some((
                                        $station,
                                        "a corrected source pose is not a valid planar pose",
                                    ));
                                    break;
                                }
                                let Some(candidate_geometry) =
                                    stitching::focus_stack_virtual_tile_geometry(
                                        $group_slices[$station],
                                        &$candidate_homographies,
                                        $projection,
                                        $candidate_focus_warp.as_ref(),
                                    )
                                else {
                                    $rejected = Some(($station, "corrected geometry is invalid"));
                                    break;
                                };
                                let Some((center_shift_px, correction_scale_ratio)) =
                                    virtual_tile_prior_repair_measurements(
                                        &repair.world_correction,
                                        $station_world_centres[$station],
                                    )
                                else {
                                    $rejected =
                                        Some(($station, "correction measurements are invalid"));
                                    break;
                                };
                                if !virtual_tile_prior_repair_is_bounded(
                                    center_shift_px,
                                    correction_scale_ratio,
                                ) {
                                    $rejected = Some(($station, "correction is outside bounds"));
                                    break;
                                }
                                $candidates.push((
                                    repair,
                                    candidate_geometry,
                                    center_shift_px,
                                    correction_scale_ratio,
                                ));
                            }
                            if let Some(($station, reason)) = $rejected {
                                println!(
                                    "  - Virtual_Tile prior repair cluster containing station {} rejected atomically: {}",
                                    $station,
                                    reason,
                                );
                                continue;
                            }
                            $station_render_homographies = $candidate_homographies;
                            $station_focus_warp = $candidate_focus_warp;
                            for (repair, $geometry, center_shift_px, correction_scale_ratio) in
                                $candidates
                            {
                                let $station = repair.$station;
                                println!(
                                    "  - Virtual_Tile prior repair: station {} moved {:.1}px (scale {:.4}) onto station {} via prior-free relation {}->{} ({} verified inliers); rendering it once more",
                                    $station,
                                    center_shift_px,
                                    correction_scale_ratio,
                                    repair.fixed_station,
                                    repair.relation.0,
                                    repair.relation.1,
                                    repair.relation_inliers,
                                );
                                $geometries[$station] = $geometry;
                                let $tile = &mut $tile_infos[$station];
                                $tile.width = $geometry.width;
                                $tile.height = $geometry.height;
                                $tile_homographies.insert($tile.id, $geometry.tile_to_world);
                                let (gray, $coverage) = render_virtual_tile_match_plane(
                                    $group_slices[$station],
                                    &$station_render_homographies,
                                    $projection,
                                    $station_focus_warp.as_ref(),
                                    $station,
                                    $geometry,
                                    $app_handle.clone(),
                                    $progress_event,
                                    &mut $load_render_image,
                                )?;
                                install_virtual_tile_match_planes(
                                    $tile,
                                    &mut $pyramid,
                                    $station,
                                    &gray,
                                    &$coverage,
                                    &$brief_pairs,
                                )?;
                                $repaired.insert($station);
                                $prior_repair_records.push(stack_report::StationPriorRepairRecord {
                                    $station,
                                    fixed_station: repair.fixed_station,
                                    relation_left: repair.relation.0,
                                    relation_right: repair.relation.1,
                                    relation_inliers: repair.relation_inliers,
                                    center_shift_px,
                                    correction_scale_ratio,
                                });
                            }
                        }
                        if !$repaired.is_empty() {
                            $station_poses_replaced = true;
                            let $pairs = $station_topology
                                .$candidates
                                .iter()
                                .filter(|candidate| {
                                    $repaired.contains(&candidate.$left)
                                        || $repaired.contains(&candidate.$right)
                                })
                                .map(|candidate| {
                                    (
                                        candidate.$left.min(candidate.$right),
                                        candidate.$left.max(candidate.$right),
                                    )
                                })
                                .collect::<BTreeSet<_>>();
                            $tile_relations.retain(|&($left, $right), _| {
                                !$pairs.contains(&($left.min($right), $left.max($right)))
                            });
                            let (remeasured, remeasured_candidates) =
                                virtual_tile_measure_prior_relations(
                                    &$tile_infos,
                                    &$pyramid,
                                    &$tile_homographies,
                                    $station_topology,
                                    Some(&$pairs),
                                    true,
                                );
                            $tile_relations.extend(remeasured);
                            for record in remeasured_candidates {
                                replace_virtual_tile_candidate_record(
                                    &mut $tile_relation_candidates,
                                    record,
                                );
                            }
                        }
                    }
                    if $compositor_choice == StackCompositorChoice::LayeredVirtualTile {
                        // Authoritative stage 4 starts only now: each ImageInfo
                        // is one focus-fused Virtual_Tile and every feature was
                        // extracted through its Coverage_Mask.  The shared
                        // relation/tree/closure core knows this evidence kind,
                        // so requirement 6.2's single-Source_RAW discard does
                        // not apply and no support value is fabricated.
                        let mut $closure = stack_report::ClosureReport::default();
                        let mut $station_relations = stack_report::StationRelationsReport::default();
                        if $tile_infos.len() >= 2 {
                            let $solved = solve_virtual_tile_station_hypotheses_with_report(
                                &$tile_infos,
                                $pyramid.solver_coverages(),
                                &$tile_relations,
                                &$tile_homographies,
                                $station_topology,
                                &mut $closure,
                                &mut $station_relations,
                            );
                            let $component_count = $station_relations.connectivity.$components;
                            // 需求 9.1: the Tone_Harmonizer samples only the
                            // overlaps of these accepted relations.
                            tone::record_run_topology(tone::ToneStationTopology {
                                $stations: $tile_infos
                                    .iter()
                                    .enumerate()
                                    .map(|($station, $tile)| ($tile.id, $station))
                                    .collect(),
                                accepted: $station_relations
                                    .accepted
                                    .iter()
                                    .filter_map(|relation| {
                                        Some((
                                            $tile_infos.get(relation.$left)?.id,
                                            $tile_infos.get(relation.$right)?.id,
                                        ))
                                    })
                                    .collect(),
                            });
                            $station_relations.$candidates = $tile_relation_candidates.clone();
                            $station_relations.summarize_candidate_measurements();
                            $station_relations.prior_repairs = $prior_repair_records.clone();
                            if let Some($recorder) = $stack_report.as_ref() {
                                $recorder.update(|$report| {
                                    $report.$closure = $closure.clone();
                                    $report.$station_relations = $station_relations.clone();
                                });
                            }
                            if $component_count > 1 || $solved.len() != $tile_infos.len() {
                                return Err(geometry_disconnected_user_error(
                                    $component_count.max(2),
                                ));
                            }
                            $tile_homographies = $solved;
                            $station_poses_replaced = true;
                            println!(
                                "  - Authoritative Virtual_Tile station solve accepted {} relation(s)",
                                $station_relations.accepted.len()
                            );
                        }
                    } else if !$tile_relations.is_empty() {
                        // Diagnostic comparison compositors retain their old
                        // image-space optimiser. Their authoritative source
                        // solve already ran before rendering.
                        let $selected_tile_relations = select_virtual_tile_relation_hypotheses(
                            &$tile_infos,
                            &$tile_relations,
                            $pyramid.solver_coverages(),
                        );
                        $tile_homographies = optimize_focus_stack_global_homographies_with_reference(
                            &$tile_infos,
                            &$selected_tile_relations,
                            &$tile_homographies,
                            0,
                        );
                    }

                    __panorama_execution_virtual_render!($_progress, $_run_scope, $accepted_station_relations, $added, $alignment_image, $alignment_mode, $anchor_frames, $anchor_layout, $app_handle, $authoritative_pairs, $blend_mode, $brief_pairs, $brightness_ratio_count, $brightness_ratio_sum, $candidate_focus_warp, $candidate_homographies, $candidates, $center, $center_luma, $closure, $clusters, $color_full_u8, $completed, $component_count, $components, $compositor_choice, $compositor_owner, $confidence, $confidence_scores, $consensus_bridges, $coverage, $coverage_plane, $covered_pixel, $decoded_paths, $defer_authoritative_station_solve, $diagnostics_recorder, $diagnostics_world_origin, $digests, $disconnected_component_count, $distance, $dynamic_image, $evidence, $exhaustive_search, $expected, $features, $floor_x, $floor_y, $focal_length_35mm, $focus_capture_group_ids, $focus_capture_groups, $focus_layer_warp, $focus_stack, $focus_station_grouping_outcome, $footprint_has_uncovered, $footprint_uncovered, $foreground_mask, $foreground_range, $full_canvas_height, $full_canvas_width, $full_image, $fusion, $geometries, $geometry, $global_homographies, $gray_full, $group_id, $group_images, $group_index, $group_order, $group_slices, $groups, $groups_complete, $horizontal_edge_rows, $ids, $image, $image_data, $image_data_results, $image_paths, $inferred, $isolated, $key, $left, $left_candidates, $load_render_image, $load_tile, $luma, $match_info, $match_results, $matched_pair_count, $matching_strategy, $members, $message, $mixed_focal_stack, $nearest_distance, $neighbour_count, $neighbour_index, $neighbour_sum, $next_owner, $nx, $ny, $ordered_filenames, $ordered_indices, $ordered_paths, $output_coverage, $overall_percentage, $owner, $owner_disagreement, $owner_reverse_lookup_failures, $owner_shortfall, $ownership, $pair_progress_step, $pairs, $pairs_to_check, $pairwise_matches, $panorama, $path, $path_selection, $physical_memory_bytes, $pid, $pixel, $placement, $pool, $preparation_error, $preparation_workers, $prepare_images, $prepared_count, $prepared_source, $prior_repair_records, $progress_event, $projection, $proposals, $px, $py, $pyramid, $quality_input, $quality_loader, $quality_planes, $quality_sources, $raw_display_highs, $raw_exposure_reference, $raw_local, $raw_owner_ids, $raw_sources, $recorder, $redundant_focus_sources, $rejected, $render_by_id, $render_homographies, $render_image_data, $render_images_info, $render_percentage, $render_scale, $rendered, $repair_clusters, $repaired, $repairs, $report, $rescue_cache, $rescue_pairs, $residual, $retained_full_images, $reverse_residual, $right, $right_candidates, $rounded_has_owner, $rounded_x, $rounded_y, $rss_sampler, $run_generation, $sampled, $sampler, $scalable_stack, $scale_robust_alignment, $scaled_global_homographies, $search_description, $seen, $selected_tile_relations, $selection, $sequence_gap_aware, $sequence_only, $settings, $shared_raw_display_gain, $sharpen, $shortfall, $solved, $source, $source_ids, $source_image, $source_sha256_before, $sources, $splits, $stack_report, $stack_report_path, $start_time, $station, $station_count, $station_crop, $station_focus_warp, $station_masks, $station_origin, $station_poses_replaced, $station_relations, $station_render_homographies, $station_sampling_origins, $station_tile_to_world, $station_to_world, $station_topology, $station_world_centres, $stations, $stitched_images_info, $stitched_indices, $stitching_order, $system, $target, $target_image, $textured, $threshold, $tile, $tile_focus_warp, $tile_group_indices, $tile_groups, $tile_homographies, $tile_id, $tile_index, $tile_infos, $tile_refs, $tile_relation_candidates, $tile_relations, $tile_result, $tiles, $top_features, $topology_ambiguous, $topology_report, $transforms, $undecodable, $undecodable_paths, $unresolved_pixel_count, $unstitched_count, $unstitched_filenames, $vertical_edge_columns, $warning_msg, $world_origin, $x, $y);
                    match $tile_result {
                        Ok($image) => Ok($image),
                        // 需求 10.11: an oversized canvas is a rejection, not a
                        // failure a fallback renderer may retry.
                        Err(error)
                            if crate::panorama_utils::stack_pipeline::compositor::is_canvas_rejection(
                                &error,
                            ) =>
                        {
                            Err(error)
                        }
                        Err(error) => {
                            let $message = format!(
                                "Virtual-tile panorama failed ({error}); falling back to the existing focus renderer."
                            );
                            println!("  - {}", $message);
                            let _ = $app_handle.emit($progress_event, &$message);
                            if let Some($recorder) = $stack_report.as_ref() {
                                $recorder.set_selected_path(
                                    stack_report::SelectedPath::LegacySingleLayerMosaic,
                                );
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
    }};
}
