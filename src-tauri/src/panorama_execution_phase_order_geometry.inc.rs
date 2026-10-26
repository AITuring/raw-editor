macro_rules! __panorama_execution_order_geometry {
    ($_progress:ident, $_run_scope:ident, $accepted_station_relations:ident, $added:ident, $alignment_image:ident, $alignment_mode:ident, $anchor_frames:ident, $anchor_layout:ident, $app_handle:ident, $authoritative_pairs:ident, $blend_mode:ident, $brief_pairs:ident, $brightness_ratio_count:ident, $brightness_ratio_sum:ident, $candidate_focus_warp:ident, $candidate_homographies:ident, $candidates:ident, $center:ident, $center_luma:ident, $closure:ident, $clusters:ident, $color_full_u8:ident, $completed:ident, $component_count:ident, $components:ident, $compositor_choice:ident, $compositor_owner:ident, $confidence:ident, $confidence_scores:ident, $consensus_bridges:ident, $coverage:ident, $coverage_plane:ident, $covered_pixel:ident, $decoded_paths:ident, $defer_authoritative_station_solve:ident, $diagnostics_recorder:ident, $diagnostics_world_origin:ident, $digests:ident, $disconnected_component_count:ident, $distance:ident, $dynamic_image:ident, $evidence:ident, $exhaustive_search:ident, $expected:ident, $features:ident, $floor_x:ident, $floor_y:ident, $focal_length_35mm:ident, $focus_capture_group_ids:ident, $focus_capture_groups:ident, $focus_layer_warp:ident, $focus_stack:ident, $focus_station_grouping_outcome:ident, $footprint_has_uncovered:ident, $footprint_uncovered:ident, $foreground_mask:ident, $foreground_range:ident, $full_canvas_height:ident, $full_canvas_width:ident, $full_image:ident, $fusion:ident, $geometries:ident, $geometry:ident, $global_homographies:ident, $gray_full:ident, $group_id:ident, $group_images:ident, $group_index:ident, $group_order:ident, $group_slices:ident, $groups:ident, $groups_complete:ident, $horizontal_edge_rows:ident, $ids:ident, $image:ident, $image_data:ident, $image_data_results:ident, $image_paths:ident, $inferred:ident, $isolated:ident, $key:ident, $left:ident, $left_candidates:ident, $load_render_image:ident, $load_tile:ident, $luma:ident, $match_info:ident, $match_results:ident, $matched_pair_count:ident, $matching_strategy:ident, $members:ident, $message:ident, $mixed_focal_stack:ident, $nearest_distance:ident, $neighbour_count:ident, $neighbour_index:ident, $neighbour_sum:ident, $next_owner:ident, $nx:ident, $ny:ident, $ordered_filenames:ident, $ordered_indices:ident, $ordered_paths:ident, $output_coverage:ident, $overall_percentage:ident, $owner:ident, $owner_disagreement:ident, $owner_reverse_lookup_failures:ident, $owner_shortfall:ident, $ownership:ident, $pair_progress_step:ident, $pairs:ident, $pairs_to_check:ident, $pairwise_matches:ident, $panorama:ident, $path:ident, $path_selection:ident, $physical_memory_bytes:ident, $pid:ident, $pixel:ident, $placement:ident, $pool:ident, $preparation_error:ident, $preparation_workers:ident, $prepare_images:ident, $prepared_count:ident, $prepared_source:ident, $prior_repair_records:ident, $progress_event:ident, $projection:ident, $proposals:ident, $px:ident, $py:ident, $pyramid:ident, $quality_input:ident, $quality_loader:ident, $quality_planes:ident, $quality_sources:ident, $raw_display_highs:ident, $raw_exposure_reference:ident, $raw_local:ident, $raw_owner_ids:ident, $raw_sources:ident, $recorder:ident, $redundant_focus_sources:ident, $rejected:ident, $render_by_id:ident, $render_homographies:ident, $render_image_data:ident, $render_images_info:ident, $render_percentage:ident, $render_scale:ident, $rendered:ident, $repair_clusters:ident, $repaired:ident, $repairs:ident, $report:ident, $rescue_cache:ident, $rescue_pairs:ident, $residual:ident, $retained_full_images:ident, $reverse_residual:ident, $right:ident, $right_candidates:ident, $rounded_has_owner:ident, $rounded_x:ident, $rounded_y:ident, $rss_sampler:ident, $run_generation:ident, $sampled:ident, $sampler:ident, $scalable_stack:ident, $scale_robust_alignment:ident, $scaled_global_homographies:ident, $search_description:ident, $seen:ident, $selected_tile_relations:ident, $selection:ident, $sequence_gap_aware:ident, $sequence_only:ident, $settings:ident, $shared_raw_display_gain:ident, $sharpen:ident, $shortfall:ident, $solved:ident, $source:ident, $source_ids:ident, $source_image:ident, $source_sha256_before:ident, $sources:ident, $splits:ident, $stack_report:ident, $stack_report_path:ident, $start_time:ident, $station:ident, $station_count:ident, $station_crop:ident, $station_focus_warp:ident, $station_masks:ident, $station_origin:ident, $station_poses_replaced:ident, $station_relations:ident, $station_render_homographies:ident, $station_sampling_origins:ident, $station_tile_to_world:ident, $station_to_world:ident, $station_topology:ident, $station_world_centres:ident, $stations:ident, $stitched_images_info:ident, $stitched_indices:ident, $stitching_order:ident, $system:ident, $target:ident, $target_image:ident, $textured:ident, $threshold:ident, $tile:ident, $tile_focus_warp:ident, $tile_group_indices:ident, $tile_groups:ident, $tile_homographies:ident, $tile_id:ident, $tile_index:ident, $tile_infos:ident, $tile_refs:ident, $tile_relation_candidates:ident, $tile_relations:ident, $tile_result:ident, $tiles:ident, $top_features:ident, $topology_ambiguous:ident, $topology_report:ident, $transforms:ident, $undecodable:ident, $undecodable_paths:ident, $unresolved_pixel_count:ident, $unstitched_count:ident, $unstitched_filenames:ident, $vertical_edge_columns:ident, $warning_msg:ident, $world_origin:ident, $x:ident, $y:ident) => {
    let $start_time = Instant::now();
    let _ = $app_handle.emit($progress_event, "Determining stitching order...");
    println!("Determining stitching order...");
    let $compositor_choice = ($blend_mode == BlendMode::FocusStack)
        .then(|| StackCompositorChoice::resolve($settings.stack_compositor.as_deref()));
    let $defer_authoritative_station_solve =
        $compositor_choice == Some(StackCompositorChoice::LayeredVirtualTile);
    let ($ordered_indices, $global_homographies) = if $blend_mode == BlendMode::FocusStack {
        // The default two-stage path uses source images only to group stations
        // and establish each station's local tile geometry.  Its authoritative
        // requirement-6/7 report is produced later from the fused tiles, so an
        // early source-only support=1 graph must neither reject nor publish a
        // report that could overwrite the Virtual_Tile result.
        let mut $closure = stack_report::ClosureReport::default();
        let mut $station_relations = stack_report::StationRelationsReport::default();
        let $stitching_order = build_focus_stack_stitching_order_with_report(
            &$image_data,
            &$pairwise_matches,
            &mut $closure,
            &mut $station_relations,
            !$defer_authoritative_station_solve,
        );
        let $disconnected_component_count = $station_relations.connectivity.$components;
        if !$defer_authoritative_station_solve {
            if let Some($recorder) = $stack_report.as_ref() {
                $recorder.update(|$report| {
                    $report.$closure = $closure;
                    $report.$station_relations = $station_relations;
                });
            }
            if $disconnected_component_count > 1 {
                return Err(geometry_disconnected_user_error(
                    $disconnected_component_count,
                ));
            }
        }
        $stitching_order
    } else {
        build_stitching_order(&$image_data, &$pairwise_matches)
    };
    let $focus_layer_warp = ($blend_mode == BlendMode::FocusStack).then(|| {
        build_focus_layer_warp(
            &$image_data,
            &$pairwise_matches,
            &$global_homographies,
            $projection,
        )
    });
    let $focus_station_grouping_outcome = if $blend_mode == BlendMode::FocusStack {
        focus_capture_groups_with_grouping(&$image_data, &$pairwise_matches, &$global_homographies)
    } else {
        None
    };
    let $station_topology = $focus_station_grouping_outcome
        .as_ref()
        .and_then(|($groups, _, _)| {
            ($groups.len() >= 2).then(|| {
                // For long scans this is the exact pre-pose topology already
                // used by `solve_focus_capture_group_poses`. Compact
                // multi-station runs do not enter that solver yet, so infer
                // here from their image-level world centres. This value is
                // shared by reporting and the default layered compositor.
                topology::run_topology_snapshot()
                    .filter(|topology| topology.row.len() == $groups.len())
                    .unwrap_or_else(|| {
                        let $inferred =
                            focus_station_topology(&$image_data, $groups, &$global_homographies);
                        topology::record_run_topology($inferred.clone());
                        $inferred
                    })
            })
        });
    if let (Some($recorder), Some(($groups, _, grouping))) = (
        $stack_report.as_ref(),
        $focus_station_grouping_outcome.as_ref(),
    ) {
        // 需求 1.7: member paths, evidence kind, score, inlier count and inlier
        // spatial support of every Capture_Station.  Task 9.1 augments these
        // same production records with displacement-derived topology indices.
        let $stations = $groups
            .iter()
            .enumerate()
            .map(|(index, group)| {
                let $evidence =
                    focus_station_binding_evidence(&group.$members, &$image_data, &$pairwise_matches);
                let $topology_ambiguous = $station_topology
                    .as_ref()
                    .is_some_and(|topology| topology.ambiguous.binary_search(&index).is_ok());
                stack_report::StationRecord {
                    index,
                    row: $station_topology
                        .as_ref()
                        .and_then(|topology| topology.row.get(index))
                        .map(|&row| row as usize),
                    column: $station_topology
                        .as_ref()
                        .and_then(|topology| topology.column.get(index))
                        .map(|&column| column as usize),
                    topology_status: $topology_ambiguous
                        .then(|| degradation::TOPOLOGY_INDEX_AMBIGUOUS.to_string()),
                    $members: group
                        .$members
                        .iter()
                        .filter_map(|&member| $image_data.get(member))
                        .map(|$image| $image.filename.clone())
                        .collect(),
                    anchor: $image_data
                        .get(group.anchor)
                        .map(|$image| $image.filename.clone())
                        .unwrap_or_default(),
                    $evidence: $evidence
                        .map(|$evidence| stack_report::StationEvidenceRecord {
                            kind: StationEvidence::KIND.to_string(),
                            score: $evidence.score,
                            inliers: $evidence.inliers,
                            overlap_ncc: $evidence.overlap_ncc,
                            spatial_support: $evidence.spatial_support,
                            scale_ratio: $evidence.scale_ratio,
                        })
                        .unwrap_or_default(),
                }
            })
            .collect::<Vec<_>>();
        // 需求 1.6: a source that satisfies the 需求 1.1 evidence with no other
        // source is reported by absolute path with its reason identifier, and the
        // remaining sources keep the grouping they would have had anyway.
        let $isolated = excluded_source_records(
            &$image_data,
            &grouping.$isolated,
            degradation::GROUPING_NO_OVERLAP_EVIDENCE,
        );
        // 需求 1.4 / 1.10: every split position.
        let $splits = grouping
            .$splits
            .iter()
            .map(|split| stack_report::StationSplitRecord {
                station_index: split.component,
                position: split.position,
                reason: split.reason.to_string(),
                motion_ratio: split.motion_ratio,
            })
            .collect::<Vec<_>>();
        for split in &grouping.$splits {
            degradation::record_run_degradation(
                split.reason,
                serde_json::json!({
                    "component": split.component,
                    "position": split.position,
                    "motion_ratio": split.motion_ratio,
                }),
            );
        }
        for record in &$isolated {
            degradation::record_run_degradation(
                degradation::GROUPING_NO_OVERLAP_EVIDENCE,
                serde_json::json!({ "path": record.$path }),
            );
        }
        let $topology_report = $station_topology.as_ref().map(|topology| {
            for &station_index in &topology.ambiguous {
                degradation::record_run_degradation(
                    degradation::TOPOLOGY_INDEX_AMBIGUOUS,
                    serde_json::json!({ "station_index": station_index }),
                );
            }
            if topology.truncated_count > 0 {
                degradation::record_run_degradation(
                    degradation::TOPOLOGY_CANDIDATES_TRUNCATED,
                    serde_json::json!({
                        "truncated_count": topology.truncated_count,
                        "truncation_min_score": topology.truncation_min_score,
                    }),
                );
            }
            stack_report::TopologyReport::from(topology)
        });
        $recorder.update(|$report| {
            $report.grouping.$station_count = $stations.len();
            $report.grouping.$stations = $stations;
            $report.grouping.$isolated = $isolated;
            $report.grouping.$splits = $splits;
            if let Some(topology) = $topology_report {
                $report.topology = topology;
            }
        });
    }
    let $focus_capture_groups = $focus_station_grouping_outcome.map(|($groups, _, _)| $groups);
    let $focus_capture_group_ids = $focus_capture_groups.as_ref().map(|$groups| {
        let mut $ids = HashMap::new();
        for ($group_index, group) in $groups.iter().enumerate() {
            let $group_id = u8::try_from($group_index + 1)
                .expect("focus stack source limit keeps group ids in u8");
            for &image_index in &group.$members {
                $ids.insert($image_data[image_index].id, $group_id);
            }
        }
        $ids
    });

    if $ordered_indices.len() < 2 {
        return Err("Could not find a connected sequence of at least two images.".to_string());
    }

    let $ordered_filenames: Vec<_> = $ordered_indices
        .iter()
        .map(|&i| {
            Path::new(&$image_data[i].filename)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string()
        })
        .collect();
    println!("Stitching order determined: {:?}", $ordered_filenames);
    let _ = $app_handle.emit(
        $progress_event,
        format!("Stitching order: {}", $ordered_filenames.join(" -> ")),
    );
    let $ordered_paths = $ordered_indices
        .iter()
        .map(|&index| $image_data[index].filename.clone())
        .collect::<Vec<_>>();

    let mut $retained_full_images = HashMap::new();
    for $image in &mut $image_data {
        if let Some($full_image) = $image.$full_image.take() {
            $retained_full_images.insert($image.id, $full_image);
        }
    }
    let mut $stitched_images_info: Vec<&ImageInfo> =
        $ordered_indices.iter().map(|&i| &$image_data[i]).collect();
    if let Some(group_ids) = $focus_capture_group_ids.as_ref() {
        order_focus_render_sources_by_capture_group(&mut $stitched_images_info, group_ids);
    }
    let $unstitched_count = $image_data.len() - $stitched_images_info.len();
    if $unstitched_count > 0 {
        let $stitched_indices = $ordered_indices.iter().copied().collect::<HashSet<_>>();
        let $unstitched_filenames = $image_data
            .iter()
            .enumerate()
            .filter(|(index, _)| !$stitched_indices.contains(index))
            .map(|(_, $image)| {
                Path::new(&$image.filename)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();
        let $warning_msg = format!(
            "{} image(s) could not be aligned with the selected set: {}.",
            $unstitched_count,
            $unstitched_filenames.join(", ")
        );
        println!("{}", $warning_msg);
        let _ = $app_handle.emit($progress_event, $warning_msg);
        let $redundant_focus_sources = $scalable_stack
            && $focus_stack
            && focus_unstitched_sources_are_redundant(
                &$image_data,
                &$pairwise_matches,
                &$stitched_indices,
            );
        if $redundant_focus_sources {
            let $message = "Continuing without weak interior focus source(s); verified neighbouring captures preserve their coverage.";
            println!("{}", $message);
            let _ = $app_handle.emit($progress_event, $message);
        }
        let $search_description = if $exhaustive_search {
            "the full pairwise search"
        } else {
            "the bounded candidate search"
        };
        if !$redundant_focus_sources {
            return Err(if $scalable_stack && $focus_stack {
                format!(
                    "Focus-stack geometry was rejected after strict feature/RANSAC retries: {} image(s) have no verified overlap in {}. No unsafe collage was rendered; select one contiguous scene or ensure each layer has enough visual overlap.",
                    $unstitched_count,
                    $search_description,
                )
            } else if $scalable_stack {
                format!(
                    "Could not align all selected panorama images; {} image(s) have no verified overlap in {}. The selection may contain multiple independent scenes. Select one contiguous scene or ensure consecutive images overlap.",
                    $unstitched_count,
                    $search_description,
                )
            } else {
                format!(
                    "Could not align all selected images; {} image(s) have no verified overlap. The selection may contain multiple independent scenes; select one contiguous scene and retry.",
                    $unstitched_count,
                )
            });
        }
    }
    println!(
        "Global homography calculation completed in {:.2?}\n",
        $start_time.elapsed()
    );

    let ($full_canvas_width, $full_canvas_height) = if $blend_mode == BlendMode::FocusStack {
        stitching::output_canvas_dimensions_with_focus_warp(
            &$stitched_images_info,
            &$global_homographies,
            $projection,
            $focus_layer_warp.as_ref(),
        )
    } else {
        stitching::output_canvas_dimensions(&$stitched_images_info, &$global_homographies, $projection)
    };
    if $full_canvas_width == 0 || $full_canvas_height == 0 {
        return Err("The aligned panorama canvas is empty or invalid.".to_string());
    }
    // The acceptance harness can persist the expensive alignment result for
    // offline ROI diagnostics.  This is compiled only into test binaries so
    // the production path has no environment-controlled behavior.
    #[cfg(test)]
    if let Some(cache_path) = std::env::var_os("RAW_EDITOR_STACK_ALIGNMENT_CACHE") {
        write_alignment_cache(
            Path::new(&cache_path),
            &$image_data,
            &$ordered_indices,
            &$pairwise_matches,
            &$global_homographies,
            $focus_layer_warp.as_ref(),
            $focus_capture_groups.as_deref(),
            $projection,
            $full_canvas_width,
            $full_canvas_height,
        )?;
    }
    #[cfg(test)]
    if std::env::var_os("RAW_EDITOR_STACK_ALIGNMENT_ONLY").is_some() {
        println!(
            "Alignment-only acceptance: skipping source rendering after {}x{} geometry validation",
            $full_canvas_width, $full_canvas_height
        );
        return Ok(StitchOutcome {
            $image: DynamicImage::ImageRgb32F(Rgb32FImage::new(1, 1)),
            $coverage: None,
            $full_canvas_width,
            $full_canvas_height,
            $render_scale: 1.0,
            $ordered_paths,
            $stack_report_path: None,
        });
    }
    let $render_scale = if $blend_mode == BlendMode::Panorama {
        memory_safe_panorama_render_scale($full_canvas_width, $full_canvas_height)
    } else {
        1.0
    };
    #[cfg(test)]
    let $render_scale = std::env::var("RAW_EDITOR_STACK_ACCEPTANCE_RENDER_SCALE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && (0.02..=1.0).contains(value))
        .unwrap_or($render_scale);
    let $render_image_data = ($render_scale < 1.0).then(|| {
        $stitched_images_info
            .iter()
            .map(|$image| scaled_render_image_info($image, $render_scale))
            .collect::<Vec<_>>()
    });
    let $render_images_info = $render_image_data.as_ref().map_or_else(
        || $stitched_images_info.clone(),
        |images| images.iter().collect::<Vec<_>>(),
    );
    #[cfg(test)]
    let mut $render_images_info = $render_images_info;
    #[cfg(test)]
    if std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_GROUP_REPRESENTATIVES").is_some()
        && let Some(group_ids) = $focus_capture_group_ids.as_ref()
    {
        let mut $seen = HashSet::new();
        $render_images_info.retain(|$image| {
            group_ids
                .get(&$image.id)
                .is_some_and(|group| $seen.insert(*group))
        });
        println!(
            "Acceptance geometry/illumination preview: rendering {} capture-group representative(s)",
            $render_images_info.len()
        );
    }
    let $scaled_global_homographies;
    let $render_homographies = if $render_scale < 1.0 {
        $scaled_global_homographies =
            scaled_source_render_homographies(&$global_homographies, $render_scale);
        let (render_width, render_height) = stitching::output_canvas_dimensions(
            &$render_images_info,
            &$scaled_global_homographies,
            $projection,
        );
        let $render_percentage = $render_scale * 100.0;
        let $message = format!(
            "Large canvas {}x{}; pre-scaling sources and rendering a memory-safe {}x{} result ({:.0}%).",
            $full_canvas_width,
            $full_canvas_height,
            render_width,
            render_height,
            $render_percentage,
        );
        println!("{}", $message);
        let _ = $app_handle.emit($progress_event, &$message);
        &$scaled_global_homographies
    } else {
        &$global_homographies
    };

    let $start_time = Instant::now();
    let _ = $app_handle.emit($progress_event, "Warping and blending images...");
    println!("Warping and blending images with progressive optimal seams...");

    let mut $load_render_image = |$image: &ImageInfo| {
        ensure_focus_memory_available($rss_sampler.as_ref())?;
        let mut $rendered = if let Some($full_image) = $retained_full_images.remove(&$image.id) {
            if $full_image.dimensions() == $image.dimensions() {
                Ok($full_image)
            } else {
                Ok(image::imageops::resize(
                    &$full_image,
                    $image.width,
                    $image.height,
                    image::imageops::FilterType::Triangle,
                ))
            }
        } else {
            load_prepared_stack_source(&$image.filename, &$settings, $raw_exposure_reference)
                .map(|$source| source_to_render_rgb32f($source.$image, $image.width, $image.height))
        }?;
        if is_raw_file(&$image.filename) {
            if let Some(gain) = std::env::var("RAW_EDITOR_STACK_RAW_FIXED_GAIN")
                .ok()
                .and_then(|value| value.parse::<f32>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)
            {
                $rendered
                    .as_mut()
                    .par_iter_mut()
                    .for_each(|channel| *channel = (*channel * gain).clamp(0.0, 1.0));
            } else if let Some(gain) = $shared_raw_display_gain {
                $rendered
                    .as_mut()
                    .par_iter_mut()
                    .for_each(|channel| *channel = (*channel * gain).clamp(0.0, 1.0));
            } else if std::env::var_os("RAW_EDITOR_SKIP_STACK_RAW_EXPOSURE_NORMALIZATION").is_none()
            {
                normalize_stack_raw_display_exposure(&mut $rendered);
            }
        }
        Ok($rendered)
    };
    let $sequence_gap_aware = $blend_mode == BlendMode::FocusStack
        && $pairwise_matches
            .values()
            .any(|$match_info| $match_info.sequence_bridge);
    // Set once the Virtual_Tile stage replaces the source-stage station poses
    // (a rendering-prior repair or the authoritative station solve): the
    // stations then no longer sit where the source-stage canvas above was
    // measured.
    let mut $station_poses_replaced = false;
    #[allow(clippy::type_complexity)]
    let mut $quality_planes: Option<(
        GrayImage,
        Vec<u16>,
        Vec<f32>,
        Vec<u8>,
        Vec<u8>,
        Vec<u8>,
        u64,
        (f64, f64),
    )> = None;
    let mut $owner_reverse_lookup_failures = stack_report::OwnerReverseLookupFailures::default();
    let mut $confidence_scores = stack_report::ConfidenceScoreDistribution {
        winner_histogram: vec![0; 21],
        runner_up_histogram: vec![0; 21],
        ..Default::default()
    };
    };
}
