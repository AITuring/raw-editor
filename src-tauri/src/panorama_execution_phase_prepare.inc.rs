macro_rules! __panorama_execution_prepare {
    ($_progress:ident, $_run_scope:ident, $accepted_station_relations:ident, $added:ident, $alignment_image:ident, $alignment_mode:ident, $anchor_frames:ident, $anchor_layout:ident, $app_handle:ident, $authoritative_pairs:ident, $blend_mode:ident, $brief_pairs:ident, $brightness_ratio_count:ident, $brightness_ratio_sum:ident, $candidate_focus_warp:ident, $candidate_homographies:ident, $candidates:ident, $center:ident, $center_luma:ident, $closure:ident, $clusters:ident, $color_full_u8:ident, $completed:ident, $component_count:ident, $components:ident, $compositor_choice:ident, $compositor_owner:ident, $confidence:ident, $confidence_scores:ident, $consensus_bridges:ident, $coverage:ident, $coverage_plane:ident, $covered_pixel:ident, $decoded_paths:ident, $defer_authoritative_station_solve:ident, $diagnostics_recorder:ident, $diagnostics_world_origin:ident, $digests:ident, $disconnected_component_count:ident, $distance:ident, $dynamic_image:ident, $evidence:ident, $exhaustive_search:ident, $expected:ident, $features:ident, $floor_x:ident, $floor_y:ident, $focal_length_35mm:ident, $focus_capture_group_ids:ident, $focus_capture_groups:ident, $focus_layer_warp:ident, $focus_stack:ident, $focus_station_grouping_outcome:ident, $footprint_has_uncovered:ident, $footprint_uncovered:ident, $foreground_mask:ident, $foreground_range:ident, $full_canvas_height:ident, $full_canvas_width:ident, $full_image:ident, $fusion:ident, $geometries:ident, $geometry:ident, $global_homographies:ident, $gray_full:ident, $group_id:ident, $group_images:ident, $group_index:ident, $group_order:ident, $group_slices:ident, $groups:ident, $groups_complete:ident, $horizontal_edge_rows:ident, $ids:ident, $image:ident, $image_data:ident, $image_data_results:ident, $image_paths:ident, $inferred:ident, $isolated:ident, $key:ident, $left:ident, $left_candidates:ident, $load_render_image:ident, $load_tile:ident, $luma:ident, $match_info:ident, $match_results:ident, $matched_pair_count:ident, $matching_strategy:ident, $members:ident, $message:ident, $mixed_focal_stack:ident, $nearest_distance:ident, $neighbour_count:ident, $neighbour_index:ident, $neighbour_sum:ident, $next_owner:ident, $nx:ident, $ny:ident, $ordered_filenames:ident, $ordered_indices:ident, $ordered_paths:ident, $output_coverage:ident, $overall_percentage:ident, $owner:ident, $owner_disagreement:ident, $owner_reverse_lookup_failures:ident, $owner_shortfall:ident, $ownership:ident, $pair_progress_step:ident, $pairs:ident, $pairs_to_check:ident, $pairwise_matches:ident, $panorama:ident, $path:ident, $path_selection:ident, $physical_memory_bytes:ident, $pid:ident, $pixel:ident, $placement:ident, $pool:ident, $preparation_error:ident, $preparation_workers:ident, $prepare_images:ident, $prepared_count:ident, $prepared_source:ident, $prior_repair_records:ident, $progress_event:ident, $projection:ident, $proposals:ident, $px:ident, $py:ident, $pyramid:ident, $quality_input:ident, $quality_loader:ident, $quality_planes:ident, $quality_sources:ident, $raw_display_highs:ident, $raw_exposure_reference:ident, $raw_local:ident, $raw_owner_ids:ident, $raw_sources:ident, $recorder:ident, $redundant_focus_sources:ident, $rejected:ident, $render_by_id:ident, $render_homographies:ident, $render_image_data:ident, $render_images_info:ident, $render_percentage:ident, $render_scale:ident, $rendered:ident, $repair_clusters:ident, $repaired:ident, $repairs:ident, $report:ident, $rescue_cache:ident, $rescue_pairs:ident, $residual:ident, $retained_full_images:ident, $reverse_residual:ident, $right:ident, $right_candidates:ident, $rounded_has_owner:ident, $rounded_x:ident, $rounded_y:ident, $rss_sampler:ident, $run_generation:ident, $sampled:ident, $sampler:ident, $scalable_stack:ident, $scale_robust_alignment:ident, $scaled_global_homographies:ident, $search_description:ident, $seen:ident, $selected_tile_relations:ident, $selection:ident, $sequence_gap_aware:ident, $sequence_only:ident, $settings:ident, $shared_raw_display_gain:ident, $sharpen:ident, $shortfall:ident, $solved:ident, $source:ident, $source_ids:ident, $source_image:ident, $source_sha256_before:ident, $sources:ident, $splits:ident, $stack_report:ident, $stack_report_path:ident, $start_time:ident, $station:ident, $station_count:ident, $station_crop:ident, $station_focus_warp:ident, $station_masks:ident, $station_origin:ident, $station_poses_replaced:ident, $station_relations:ident, $station_render_homographies:ident, $station_sampling_origins:ident, $station_tile_to_world:ident, $station_to_world:ident, $station_topology:ident, $station_world_centres:ident, $stations:ident, $stitched_images_info:ident, $stitched_indices:ident, $stitching_order:ident, $system:ident, $target:ident, $target_image:ident, $textured:ident, $threshold:ident, $tile:ident, $tile_focus_warp:ident, $tile_group_indices:ident, $tile_groups:ident, $tile_homographies:ident, $tile_id:ident, $tile_index:ident, $tile_infos:ident, $tile_refs:ident, $tile_relation_candidates:ident, $tile_relations:ident, $tile_result:ident, $tiles:ident, $top_features:ident, $topology_ambiguous:ident, $topology_report:ident, $transforms:ident, $undecodable:ident, $undecodable_paths:ident, $unresolved_pixel_count:ident, $unstitched_count:ident, $unstitched_filenames:ident, $vertical_edge_columns:ident, $warning_msg:ident, $world_origin:ident, $x:ident, $y:ident) => {
    let $run_generation = degradation::active_run_generation();
    let $_run_scope = degradation::begin_run_scope_for_generation($run_generation).map_err(|_| {
        "image-stack run cancelled by a newer request before it started".to_string()
    })?;
    let $image_paths = $image_paths
        .into_iter()
        .filter(|$path| !is_generated_stitch_output($path) && !is_auxiliary_stitch_file($path))
        .collect::<Vec<_>>();
    // Open a fresh degradation observation window for this run. Recording is
    // additive: no fallback decision below consults the ledger.
    degradation::reset_run_ledger();
    // The Intra_Station_Registrar records one entry per registered frame
    // (需求 2.9); open its window alongside the ledger so a second run in the
    // same process cannot inherit the previous run's frames.
    intra_station::reset_run_records();
    // The Focus_Fuser records one entry per Capture_Station (需求 3.5 / 3.8 / 3.9)
    // through the same run scoped sink.
    focus_fuser::reset_run_records();
    // Residual_Warp_Model and Tone_Harmonizer report through run scoped sinks
    // as well, and the Tile_Compositor samples the recorded residual model, so
    // a later run must never see an earlier run's model (需求 8.1, 9.11).
    residual_warp::reset_run_records();
    tone::reset_run_records();
    // The accepted Station_Relations the Tone_Harmonizer samples from belong to
    // this run's station solve only (需求 9.1).
    tone::reset_run_topology();
    // Capture_Topology_Model observations are likewise scoped to this run so
    // the report receives exactly the topology used by the production station
    // relation candidate gate (需求 5.1 / 5.7).
    topology::reset_run_topology();
    // Install the RANSAC seed of this run before any matching starts (需求 14.6).
    // The focus stack path seeds from the Virtual_Tile `cache_key`, which binds
    // the pipeline version, the sorted source paths and the sorted per-source
    // content digests (需求 4.5), so the sampling sequence follows the input
    // content rather than the wall clock or the thread count.  The panorama path
    // keeps its site constants, and clearing here stops it from inheriting the
    // seed of a preceding focus stack run in the same process.
    // The same read-only pass also produces the `sha256_before` column of the
    // Stack_Report (需求 4.8 / 12.8), so the sources are hashed once per run.
    let mut $source_sha256_before = Vec::<String>::new();
    if $blend_mode == BlendMode::FocusStack {
        let $digests = focus_stack_cache_key_inputs(&$image_paths);
        determinism::set_run_random_seed(determinism::derive_run_seed_from_cache_key(
            &$digests.inputs.digest(),
        ));
        $source_sha256_before = $digests.sha256_before;
    } else {
        determinism::clear_run_random_seed();
    }
    // Observation only: the Stack_Report records what a focus-stack run did and
    // is flushed on every terminating path (success, degraded output,
    // rejection, cancellation) by its drop guard, well inside the 30 s budget
    // of requirement 10.8.  No field written here feeds a decision yet.
    let $stack_report = ($blend_mode == BlendMode::FocusStack).then(|| {
        let $recorder = stack_report::StackReportRecorder::new(
            crate::image_stack::STACK_PIPELINE_VERSION,
            stack_report::resolve_stack_report_dir(&$app_handle),
        )
        // Requirement 4.8 / 12.8: whichever way this run ends — success,
        // failure or cancellation — the recorder's drop guard re-reads every
        // Source_RAW and records the digest next to the one taken above, so a
        // source modified behind the pipeline's back stays visible.  The
        // pipeline itself only ever opens a source read only, so the two
        // columns are expected to match.
        .with_source_digest_recheck(Box::new(|paths| {
            focus_stack_source_digests(paths)
                .into_iter()
                .map(source_digest_hex)
                .collect()
        }));
        $recorder.update(|$report| {
            $report.input.source_count = $image_paths.len();
            $report.input.max_sources = MAX_STITCH_SOURCE_IMAGES;
            $report.input.$sources = $image_paths
                .iter()
                .enumerate()
                .map(|(index, $path)| stack_report::SourceRecord {
                    $path: $path.clone(),
                    sha256_before: $source_sha256_before.get(index).cloned().unwrap_or_default(),
                    ..Default::default()
                })
                .collect();
            // Any return before the compositor finishes is a run that produced
            // no result; a successful path overwrites this below.
            $report.degradation.result = stack_report::StackRunResult::Rejected;
            $report.resources.worker_threads = rayon::current_num_threads();
            $report.resources.random_seed_source = determinism::run_random_seed_source().to_string();
        });
        Arc::new($recorder)
    });
    // Sample process RSS for the complete focus-stack run.  No settings
    // source is exposed by the current runtime entry point, so the documented
    // default threshold is used and reported as such.
    let mut $rss_sampler = if $blend_mode == BlendMode::FocusStack {
        let mut $system = sysinfo::System::new();
        $system.refresh_memory();
        let $physical_memory_bytes = $system.total_memory();
        let $threshold = resources::resolve_memory_threshold($physical_memory_bytes, None, false);
        let $pid = sysinfo::get_current_pid().ok();
        let $sampler = resources::RssSampler::start($threshold.bytes, move || {
            let Some($pid) = $pid else {
                return 0;
            };
            let mut $system = sysinfo::System::new();
            $system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[$pid]), true);
            $system.process($pid).map_or(0, |process| process.memory())
        });
        if let Some($recorder) = $stack_report.as_ref() {
            $recorder.update(|$report| {
                $report.resources.memory_threshold_bytes = $threshold.bytes;
                $report.resources.memory_threshold_source = $threshold.$source.as_str().to_string();
                $report.resources.$physical_memory_bytes = $physical_memory_bytes;
            });
        }
        Some(FocusRssGuard {
            $sampler: Some($sampler),
            threshold_bytes: $threshold.bytes,
            $recorder: $stack_report.clone(),
        })
    } else {
        None
    };
    ensure_focus_memory_available($rss_sampler.as_ref())?;
    if $image_paths.len() < 2 {
        degradation::record_run_degradation(
            degradation::INPUT_SOURCE_COUNT_OUT_OF_RANGE,
            serde_json::json!({
                "count": $image_paths.len(),
                "minimum": 2,
                "maximum": MAX_STITCH_SOURCE_IMAGES,
            }),
        );
        return Err("At least two images are required for a panorama.".to_string());
    }
    if $image_paths.len() > MAX_STITCH_SOURCE_IMAGES {
        degradation::record_run_degradation(
            degradation::INPUT_SOURCE_COUNT_OUT_OF_RANGE,
            serde_json::json!({
                "count": $image_paths.len(),
                "minimum": 2,
                "maximum": MAX_STITCH_SOURCE_IMAGES,
            }),
        );
        return Err(format!(
            "Image stitching is limited to {MAX_STITCH_SOURCE_IMAGES} source images."
        ));
    }

    let _ = $app_handle.emit($progress_event, "Starting image alignment process...");
    println!(
        "Starting panorama stitching process for {} images...",
        $image_paths.len()
    );

    let $focus_stack = $blend_mode == BlendMode::FocusStack;
    let $scalable_stack = stack_requires_bounded_memory(&$image_paths);
    let $mixed_focal_stack = !$scalable_stack && selection_has_mixed_focal_lengths(&$image_paths);
    let $scale_robust_alignment = $scalable_stack || $mixed_focal_stack;
    let $settings = load_settings_for_runtime(&$app_handle).unwrap_or_default();
    // Diagnostics are opt-in through the persisted settings.  Creating the
    // directory up front keeps the recorder's per-station writes atomic while
    // leaving the default run allocation-free and pixel-identical.
    let mut $diagnostics_recorder = if $focus_stack {
        $settings
            .stack_diagnostics
            .output_dir
            .as_deref()
            .map(str::trim)
            .filter(|directory| !directory.is_empty())
            .map(|directory| {
                let $path = PathBuf::from(directory);
                match std::fs::create_dir_all(&$path) {
                    Ok(()) => DiagnosticsRecorder::new(Some($path)),
                    Err(error) => {
                        degradation::record_run_degradation(
                            degradation::DIAGNOSTICS_WRITE_FAILED,
                            serde_json::json!({
                                "directory": directory,
                                "error": error.to_string(),
                            }),
                        );
                        DiagnosticsRecorder::new(None)
                    }
                }
            })
            .unwrap_or_else(|| DiagnosticsRecorder::new(None))
    } else {
        DiagnosticsRecorder::new(None)
    };
    // A focus stack must retain one shared display scale across its RAW
    // sources.  Per-frame percentile normalization makes the 97th percentile
    // of every frame identical even when that percentile belongs to artwork,
    // a holder, or a different amount of canvas.  The overlap solver then
    // sees a pre-distorted sequence and can only repair it with tile-sized
    // gains.  Collect the inexpensive low-resolution percentile while the
    // preparation workers already have each developed frame in memory; the
    // rendering closure below applies one robust gain to every RAW.
    let $raw_exposure_reference = $focus_stack
        .then(|| stack_raw_exposure_reference(&$image_paths))
        .flatten();
    if let Some(reference) = $raw_exposure_reference {
        println!("Shared RAW physical exposure reference: {reference:.6} (time × ISO / aperture²)");
    }
    let $raw_display_highs = Mutex::new(Vec::<f32>::new());

    let $start_time = Instant::now();
    let _ = $app_handle.emit($progress_event, "Loading and preparing images...");
    let (alignment_max_dimension, alignment_max_features) =
        scalable_alignment_budget($image_paths.len());
    println!(
        "Loading and preparing images ({} mode)...",
        if $scalable_stack {
            "bounded-memory"
        } else if $mixed_focal_stack {
            "full-resolution, mixed-focal"
        } else {
            "full-resolution"
        }
    );
    let $brief_pairs = processing::generate_brief_pairs();
    let $prepared_count = Mutex::new(0usize);
    let $prepare_images = || {
        $image_paths
            .par_iter()
            .enumerate()
            .map(|(i, filename)| {
                if degradation::run_generation_changed_for($run_generation) {
                    return Err("image-stack run cancelled by a newer request".to_string());
                }
                println!("  - Processing '{}'", filename);
                let $prepared_source =
                    load_prepared_stack_source(filename, &$settings, $raw_exposure_reference)?;
                if degradation::run_generation_changed_for($run_generation) {
                    return Err("image-stack run cancelled by a newer request".to_string());
                }
                let $dynamic_image = $prepared_source.$image;
                let $focal_length_35mm = $prepared_source.$focal_length_35mm;
                if $focus_stack
                    && is_raw_file(filename)
                    && std::env::var_os("RAW_EDITOR_SKIP_STACK_RAW_EXPOSURE_NORMALIZATION")
                        .is_none()
                    && let Some(high) = raw_display_channel_percentile(&$dynamic_image)
                {
                    $raw_display_highs.lock().unwrap().push(high);
                }
                let (width, height) = $dynamic_image.dimensions();
                let (new_width, new_height, scale_factor) = if $scalable_stack {
                    processing::calculate_downscale_dimensions_capped(
                        width,
                        height,
                        alignment_max_dimension,
                    )
                } else {
                    processing::calculate_downscale_dimensions(width, height)
                };

                let $alignment_image = if $scalable_stack {
                    if (new_width, new_height) == (width, height) {
                        $dynamic_image.to_luma8()
                    } else {
                        $dynamic_image
                            .resize_exact(
                                new_width,
                                new_height,
                                image::imageops::FilterType::Triangle,
                            )
                            .to_luma8()
                    }
                } else {
                    let $color_full_u8 = $dynamic_image.to_rgb8();
                    let $gray_full = image::imageops::colorops::grayscale(&$color_full_u8);
                    image::imageops::resize(
                        &$gray_full,
                        new_width,
                        new_height,
                        image::imageops::FilterType::Triangle,
                    )
                };

                let $features = find_alignment_features(
                    &$alignment_image,
                    &$brief_pairs,
                    alignment_max_features,
                    $scale_robust_alignment,
                    $focal_length_35mm,
                );
                let $foreground_range = if $focus_stack {
                    detect_foreground_range(&$alignment_image)
                } else {
                    None
                };
                let $foreground_mask = if $focus_stack {
                    build_foreground_mask(&$alignment_image, $foreground_range)
                } else {
                    None
                };
                let $horizontal_edge_rows = if $focus_stack {
                    horizontal_edge_row_candidates(&$alignment_image, $foreground_range)
                } else {
                    Vec::new()
                };
                let $vertical_edge_columns = if $focus_stack {
                    detect_vertical_edge_columns(&$alignment_image, $foreground_range)
                } else {
                    Vec::new()
                };
                let $top_features = if $focus_stack {
                    find_top_alignment_features(&$alignment_image, &$brief_pairs, $foreground_range)
                } else {
                    Vec::new()
                };
                let $full_image = (!$scalable_stack).then(|| $dynamic_image.to_rgb32f());
                println!("    Found {} features in '{}'", $features.len(), filename);

                let mut $prepared_count = $prepared_count
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                *$prepared_count += 1;
                let $completed = *$prepared_count;
                let $overall_percentage = 5.0 + ($completed as f64 / $image_paths.len() as f64) * 24.0;
                let _ = $app_handle.emit(
                    $progress_event,
                    format!(
                        "Analyzing image {} of {} ({:.0}%): {}",
                        $completed,
                        $image_paths.len(),
                        $overall_percentage,
                        Path::new(filename)
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                    ),
                );
                drop($prepared_count);

                Ok(ImageInfo {
                    id: i,
                    filename: filename.to_string(),
                    width,
                    height,
                    $alignment_image,
                    $full_image,
                    scale_factor,
                    $focal_length_35mm,
                    overview_reference: false,
                    $features,
                    $top_features,
                    $foreground_range,
                    $foreground_mask,
                    $horizontal_edge_rows,
                    $vertical_edge_columns,
                })
            })
            .collect::<Vec<Result<ImageInfo, String>>>()
    };

    let $image_data_results = if $scalable_stack {
        let $preparation_workers = scalable_preparation_worker_count($image_paths.len());
        println!("Preparing images with {} bounded worker(s)...", $preparation_workers);
        let $pool = ThreadPoolBuilder::new()
            .num_threads($preparation_workers)
            .thread_name(|index| format!("image-stack-analysis-{index}"))
            .build()
            .map_err(|error| format!("Failed to start image-stack workers: {error}"))?;
        $pool.install($prepare_images)
    } else {
        $prepare_images()
    };
    ensure_focus_memory_available($rss_sampler.as_ref())?;

    // Preparation already finished for every source, so collect the whole batch
    // before propagating.  The caller still observes the first error exactly as
    // a `?` in the loop would, but the Stack_Report can now name which sources
    // decoded instead of reporting the entire selection as undecodable.
    let mut $image_data = Vec::new();
    let mut $preparation_error = None;
    for result in $image_data_results {
        match result {
            Ok($image) => $image_data.push($image),
            Err(error) => {
                if $preparation_error.is_none() {
                    $preparation_error = Some(error);
                }
            }
        }
    }
    // 需求 1.9: a Source_RAW that does not decode is excluded from every
    // Capture_Station and reported by absolute path with `source_decode_failed`,
    // while the remaining sources keep the grouping they would have had anyway.
    let $undecodable_paths = undecodable_source_paths(&$image_paths, &$image_data);
    if let Some($recorder) = $stack_report.as_ref() {
        let $decoded_paths = $image_data
            .iter()
            .map(|$image| $image.filename.as_str())
            .collect::<HashSet<_>>();
        let $undecodable = $undecodable_paths
            .iter()
            .map(|$path| stack_report::ExcludedSourceRecord {
                $path: $path.clone(),
                reason: degradation::SOURCE_DECODE_FAILED.to_string(),
            })
            .collect::<Vec<_>>();
        $recorder.update(|$report| {
            for $source in &mut $report.input.$sources {
                $source.decoded = $decoded_paths.contains($source.$path.as_str());
            }
            $report.grouping.$undecodable = $undecodable;
        });
    }
    // The ledger entry is independent of the report recorder: 需求 1.9 asks for
    // one `source_decode_failed` record per excluded source, by absolute path.
    for $path in &$undecodable_paths {
        degradation::record_run_degradation(
            degradation::SOURCE_DECODE_FAILED,
            serde_json::json!({ "path": $path }),
        );
    }
    if let Some(error) = $preparation_error {
        // 需求 1.9: on the FocusStack path an undecodable Source_RAW is excluded
        // from every Capture_Station and the remaining sources keep the grouping
        // they would have had anyway, so the run continues on the decodable
        // subset.  This entry point is shared with the panorama path, which has
        // no Capture_Station layer to exclude a source from and no Stack_Report
        // to name it in: a panorama still aborts on the first failure.
        if !$focus_stack {
            return Err(error);
        }
        // 需求 1.8: fewer than two decodable sources leaves nothing to group, so
        // the run is rejected with the input-range identifier rather than
        // degraded.
        if $image_data.len() < 2 {
            degradation::record_run_degradation(
                degradation::INPUT_SOURCE_COUNT_OUT_OF_RANGE,
                serde_json::json!({
                    "count": $image_data.len(),
                    "minimum": 2,
                    "maximum": MAX_STITCH_SOURCE_IMAGES,
                    "undecodable": $undecodable_paths.len(),
                }),
            );
            return Err(format!(
                "Only {} of {} selected images could be decoded; at least two are required. First failure: {error}",
                $image_data.len(),
                $image_paths.len(),
            ));
        }
        println!(
            "  - Excluded {} undecodable source(s) from every capture station; continuing with {} of {}. First failure: {error}",
            $undecodable_paths.len(),
            $image_data.len(),
            $image_paths.len(),
        );
        let _ = $app_handle.emit(
            $progress_event,
            format!(
                "Skipped {} undecodable image(s); continuing with {}.",
                $undecodable_paths.len(),
                $image_data.len(),
            ),
        );
    }

    let $shared_raw_display_gain = if $focus_stack
        && std::env::var_os("RAW_EDITOR_SKIP_STACK_RAW_EXPOSURE_NORMALIZATION").is_none()
        && std::env::var_os("RAW_EDITOR_STACK_RAW_FIXED_GAIN").is_none()
        && std::env::var_os("RAW_EDITOR_DISABLE_SHARED_RAW_NORMALIZATION").is_none()
    {
        shared_stack_raw_display_gain(&$raw_display_highs.into_inner().unwrap_or_default())
    } else {
        None
    };
    if let Some(gain) = $shared_raw_display_gain {
        println!(
            "Shared RAW display normalization: gain={gain:.3} from {} frame percentiles",
            $image_paths.len()
        );
    }

    println!(
        "Image loading and feature detection completed in {:.2?}\n",
        $start_time.elapsed()
    );

    };
}
